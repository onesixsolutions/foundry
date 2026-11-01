import numpy as np
import pandas as pd
import pytest
import torch

from foundry.hlm import Hlm


def _simulate_visits_and_spend(num_players: int, corr: float = .6, seed: int = 0) -> pd.DataFrame:
    """
    Each row is a period of ``days`` in which a player visits on some days (binomial); when they visit, we observe
    their spend (log-normal). Players have correlated random-intercepts for visits and spend; ``x1`` has fixed-effects
    on both.
    """
    rs = np.random.RandomState(seed)
    stds = np.array([.8, .5])
    cov = np.diag(stds) @ np.array([[1, corr], [corr, 1]]) @ np.diag(stds)
    res = rs.multivariate_normal(np.zeros(2), cov, size=num_players)
    player = np.repeat(np.arange(num_players), rs.randint(2, 12, size=num_players))
    df = pd.DataFrame({'player': player, 'x1': rs.randn(len(player)), 'days': rs.randint(1, 15, size=len(player))})
    logits = -1.5 + .5 * df['x1'] + res[player, 0]
    df['visits'] = rs.binomial(df['days'], 1 / (1 + np.exp(-logits)))
    spend = np.exp(4. - .3 * df['x1'] + res[player, 1] + .7 * rs.randn(len(df)))
    df['spend'] = np.where(df['visits'] > 0, spend, np.nan)
    df.attrs['cov'] = cov
    df.attrs['res'] = res
    return df


def _Xy(df: pd.DataFrame) -> tuple:
    return df[['player', 'x1']], {'value': df[['visits', 'spend']], 'visits__total_count': df['days']}


def test_multivariate_recovery():
    df = _simulate_visits_and_spend(num_players=1500)
    X, y = _Xy(df)
    hlm = Hlm({'visits': 'binomial', 'spend': 'lognormal'}, raneff_design={'player': []})
    hlm.fit(X, y, verbose=False)

    cov = hlm.raneff_cov_
    assert list(cov.index) == ['visits:Intercept', 'spend:Intercept']
    stds = np.sqrt(np.diag(cov.values))
    np.testing.assert_allclose(stds, np.sqrt(np.diag(df.attrs['cov'])), rtol=.15)
    assert abs(cov.values[0, 1] / stds.prod() - .6) < .1

    coefs = hlm.coef_dataframe_.set_index('name')['estimate']
    np.testing.assert_allclose(
        coefs[['visits__probs_bias__bias', 'visits__probs_weight__x1', 'spend__loc_bias__bias',
               'spend__loc_weight__x1']].values,
        [-1.5, .5, 4., -.3],
        atol=.1
    )
    assert hlm.predict(X).shape == (len(X), 2)


def test_single_outcome_dict_matches_univariate():
    df = _simulate_visits_and_spend(num_players=100).dropna()
    X, y = df[['player', 'x1']], df['spend']
    torch.manual_seed(0)
    hlm1 = Hlm('lognormal', raneff_design={'player': []}).fit(X, y, verbose=False)
    torch.manual_seed(0)
    hlm2 = Hlm({'spend': 'lognormal'}, raneff_design={'player': []}).fit(X, y.to_frame(), verbose=False)
    np.testing.assert_allclose(hlm1.raneff_cov_.values, hlm2.raneff_cov_.values, rtol=1e-3)
    np.testing.assert_allclose(hlm1.predict(X), hlm2.predict(X), rtol=1e-3)


def test_per_outcome_raneff_design():
    df = _simulate_visits_and_spend(num_players=60)
    X, y = _Xy(df)
    hlm = Hlm({'visits': 'binomial', 'spend': 'lognormal'}, raneff_design={'player': {'visits': [], 'spend': ['x1']}})
    hlm.fit(X, y, verbose=False, estimate_laplace_coefs=False)
    assert list(hlm.raneff_cov_.index) == ['visits:Intercept', 'spend:Intercept', 'spend:x1']
    assert hlm.raneff_dataframe_.shape == (60, 3)


def test_visit_history_informs_spend_for_new_players():
    """
    For new players whose history has visits but no spend (e.g. spend not yet recorded), conditioning on that history
    should still move their spend predictions, via the correlated random-effects.
    """
    df = _simulate_visits_and_spend(num_players=2000, seed=1)
    is_train = df['player'] < 1000
    X, y = _Xy(df[is_train])
    hlm = Hlm({'visits': 'binomial', 'spend': 'lognormal'}, raneff_design={'player': []}).fit(X, y, verbose=False)

    new = df[~is_train]
    X_new, y_new = _Xy(new)
    y_new['value'] = y_new['value'].assign(spend=np.nan)  # history has visits only
    levels, res = hlm.get_raneff_table(group_data=(X_new, y_new))
    new_players = np.unique(new['player'])
    spend_res = pd.Series(res[:, 1].numpy(), index=levels)[new_players]
    assert np.corrcoef(spend_res, df.attrs['res'][new_players, 1])[0, 1] > .3

    pred_cond = hlm.predict(X_new, group_data=(X_new, y_new))[:, 1]
    pred_zero = hlm.predict(X_new)[:, 1]
    assert not np.allclose(pred_cond, pred_zero)
