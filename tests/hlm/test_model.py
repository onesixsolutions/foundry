import numpy as np
import pandas as pd
import pytest
import torch
from sklearn import clone

from foundry.data import simulate_grouped_data
from foundry.hlm import Hlm
from foundry.util import SliceDict


def _sim(family: str, **kwargs) -> tuple:
    kwargs = {**dict(n_groups=200, n_samples_per_group=(3, 15), n_raneff_features=1, n_fixeff_features=1,
                     random_state=2), **kwargs}
    if family == 'binomial':
        kwargs['total_count'] = 8
    df = simulate_grouped_data(family, **kwargs)
    X = df.drop(columns=['target', 'total_count'], errors='ignore')
    y = {'value': df['target'], 'total_count': df['total_count']} if family == 'binomial' else df['target']
    return df, X, y


def test_gaussian_matches_statsmodels():
    smf = pytest.importorskip('statsmodels.formula.api')
    df, X, y = _sim('gaussian', bias=1., scale=-.5, n_samples_per_group=(2, 20), random_state=1)
    hlm = Hlm('gaussian', raneff_design={'group_id': ['x1']}).fit(X, y, verbose=False, stopping=(1e-6,))
    sm = smf.mixedlm('target ~ x1 + f1', df, groups=df['group_id'], re_formula='~x1').fit(reml=False)

    coefs = hlm.coef_dataframe_.set_index('name')['estimate']
    np.testing.assert_allclose(
        coefs[['loc_bias__bias', 'loc_weight__x1', 'loc_weight__f1']].values, sm.fe_params.values, atol=1e-3
    )
    np.testing.assert_allclose(hlm.raneff_cov_.values, sm.cov_re.values, atol=1e-3)


@pytest.mark.parametrize('family', ['binomial', 'poisson'])
def test_parameter_recovery(family: str):
    df, X, y = _sim(family, n_groups=300, bias=.5)
    hlm = Hlm(family, raneff_design={'group_id': ['x1']}).fit(X, y, verbose=False)
    coefs = hlm.coef_dataframe_.set_index('name')
    param0 = hlm.family.params[0]
    truth = df.attrs['ground_truth_coefs']
    for nm, truth_nm in [(f'{param0}_bias__bias', 'bias'), (f'{param0}_weight__x1', 'x1'),
                         (f'{param0}_weight__f1', 'f1')]:
        assert abs(coefs.loc[nm, 'estimate'] - truth[truth_nm]) < 3 * coefs.loc[nm, 'se'], nm
    np.testing.assert_allclose(hlm.raneff_cov_.values, df.attrs['ground_truth_raneff_cov'].values, atol=.05)
    # random-effects should correlate strongly with the ground truth:
    corr = np.corrcoef(hlm.raneff_dataframe_['Intercept'], df.attrs['ground_truth_raneffs']['Intercept'])[0, 1]
    assert corr > .8


def test_slicedict_input_matches_dataframe():
    df, X, y = _sim('gaussian', n_groups=50)
    torch.manual_seed(0)
    hlm1 = Hlm('gaussian', raneff_design={'group_id': ['x1']}).fit(X, y, verbose=False)
    torch.manual_seed(0)
    X2 = SliceDict(loc=X.drop(columns='group_id'), group_id=X['group_id'])
    hlm2 = Hlm('gaussian', raneff_design={'group_id': ['x1']}).fit(X2, y, verbose=False)
    np.testing.assert_allclose(hlm1.raneff_cov_.values, hlm2.raneff_cov_.values, rtol=1e-4)
    np.testing.assert_allclose(hlm1.predict(X), hlm2.predict(X2), rtol=1e-4)


def test_init_params_not_mutated():
    df, X, y = _sim('gaussian', n_groups=30)
    selector = lambda d: [c for c in d.columns if c.startswith('x')]
    hlm = Hlm('gaussian', raneff_design={'group_id': selector})
    params_before = hlm.get_params()
    hlm.fit(X, y, verbose=False)
    assert hlm.raneff_design == params_before['raneff_design']
    assert hlm.raneff_design_ == {'group_id': ['x1']}
    clone(hlm)


def test_missing_grouping_factor_hint():
    df, X, y = _sim('gaussian', n_groups=30)
    with pytest.raises(ValueError, match="did you mean"):
        Hlm('gaussian', raneff_design={'group_id': []}).fit(X.rename(columns={'group_id': 'remainder__group_id'}), y)


def _holdout_split(df: pd.DataFrame, X: pd.DataFrame, y, frac_train_groups: float = .5):
    is_train = X['group_id'] < X['group_id'].max() * frac_train_groups
    take = lambda obj, m: {k: v[m] for k, v in obj.items()} if isinstance(obj, dict) else obj[m]
    return (X[is_train], take(y, is_train)), (X[~is_train], take(y, ~is_train))


@pytest.mark.parametrize('family', ['gaussian', 'binomial'])
def test_group_data_conditioning(family: str):
    from foundry.hlm import ConditionalScorer
    df, X, y = _sim(family, n_groups=300, n_samples_per_group=(10, 20), random_state=3)
    (X_train, y_train), (X_val, y_val) = _holdout_split(df, X, y)
    hlm = Hlm(family, raneff_design={'group_id': ['x1']}).fit(X_train, y_train, verbose=False)

    # history/target split within each validation group:
    is_hist = ConditionalScorer(history_frac=.5).split(X_val, 'group_id')
    take = lambda obj, m: {k: v[m] for k, v in obj.items()} if isinstance(obj, dict) else obj[m]
    X_hist, y_hist, X_targ, y_targ = X_val[is_hist], take(y_val, is_hist), X_val[~is_hist], take(y_val, ~is_hist)

    # predictions:
    truth = df.attrs['ground_truth_raneffs']
    # (for binomial, total_count isn't in X, so predict proportions)
    type_ = 'probs' if family == 'binomial' else 'mean'
    pred_zero = hlm.predict(X_targ, type=type_)
    pred_cond = hlm.predict(X_targ, type=type_, group_data=(X_hist, y_hist))
    y_true = (y_targ['value'] / y_targ['total_count'] if isinstance(y_targ, dict) else y_targ).values
    mse_zero = ((pred_zero.squeeze() - y_true) ** 2).mean()
    mse_cond = ((pred_cond.squeeze() - y_true) ** 2).mean()
    assert mse_cond < .9 * mse_zero

    # random-effects from group_data should track the ground-truth:
    levels, res = hlm.get_raneff_table(group_data=(X_hist, y_hist))
    val_levels = np.unique(X_val['group_id'])
    est = pd.Series(res[:, 0].numpy(), index=levels)[val_levels]
    assert np.corrcoef(est, truth['Intercept'][val_levels])[0, 1] > .7

    # scores: conditioning beats marginal; scorer agrees with manual
    score_cond = hlm.score(X_targ, y_targ, group_data=(X_hist, y_hist))
    assert score_cond > hlm.score(X_targ, y_targ)
    assert ConditionalScorer(history_frac=.5)(hlm, X_val, y_val) == pytest.approx(score_cond)


def test_unknown_groups_raise():
    df, X, y = _sim('gaussian', n_groups=40)
    (X_train, y_train), (X_val, y_val) = _holdout_split(df, X, y)
    hlm = Hlm('gaussian', raneff_design={'group_id': ['x1']}, unknown_groups='raise').fit(X_train, y_train,
                                                                                           verbose=False)
    hlm.predict(X_train)
    with pytest.raises(ValueError, match="unknown random-effects"):
        hlm.predict(X_val)
    hlm.predict(X_val, group_data=(X_val, y_val))


def test_bernoulli_predict_proba_with_group_data():
    df = simulate_grouped_data('bernoulli', n_groups=60, n_samples_per_group=(5, 10), random_state=4)
    X, y = df.drop(columns='target'), df['target']
    hlm = Hlm('bernoulli', raneff_design={'group_id': []}).fit(X, y, verbose=False)
    probs = hlm.predict_proba(X, group_data=(X, y))
    assert probs.shape == (len(X), 2)
    assert set(hlm.predict(X)) <= {0., 1.}


def test_pipeline_and_grid_search():
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from foundry.preprocessing import DataFrameTransformer
    from foundry.hlm import ConditionalScorer, conditional_predict

    df, X, y = _sim('gaussian', n_groups=60, n_samples_per_group=(4, 10))
    pipeline = make_pipeline(
        DataFrameTransformer(
            [('scale', StandardScaler(), ['x1', 'f1'])], remainder='passthrough', verbose_feature_names_out=False
        ),
        Hlm('gaussian', raneff_design={'group_id': ['x1']})
    )
    gcv = GridSearchCV(
        pipeline,
        param_grid={'hlm__penalty': [0., 10.]},
        cv=GroupKFold(n_splits=3),
        scoring=ConditionalScorer()
    )
    gcv.fit(X, y, groups=X['group_id'], hlm__verbose=False)
    assert np.isfinite(gcv.cv_results_['mean_test_score']).all()

    # conditional prediction through the pipeline:
    pred = conditional_predict(gcv.best_estimator_, X.iloc[:10], group_data=(X.iloc[10:], y.iloc[10:]))
    assert pred.shape[0] == 10


def test_penalty_search_defaults_to_group_kfold():
    df, X, y = _sim('gaussian', n_groups=40)
    hlm = Hlm('gaussian', raneff_design={'group_id': ['x1']}, penalty=[0., 1., 100.])
    hlm.fit(X, y, verbose=False)
    assert hlm.penalty in (0., 1., 100.)
