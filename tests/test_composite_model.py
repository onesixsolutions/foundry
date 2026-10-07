import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression

from foundry.composite_model import HurdleModel
from foundry.glm import Glm, LogSmearingRegressor


def _sim_lognormal(seed: int, n: int = 5000) -> tuple:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({'x': rng.normal(size=n)})
    y = np.exp(1 + .5 * X['x'] + np.exp(-.5 + .3 * X['x']) * rng.normal(size=n))
    return X, y.to_numpy()


def test_hurdle_model():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({'x': rng.normal(size=1000)})
    y = np.where(X['x'] > 0, 2. + rng.random(len(X)), 0.)

    model = HurdleModel(event_est=LogisticRegression(), value_given_event_est=DummyRegressor())
    model.fit(X, y)
    assert model.value_given_event_est_.constant_.item() == pytest.approx(y[y > 0].mean())

    components = model.predict_components(X)
    np.testing.assert_allclose(model.predict(X), components['event_est'] * components['value_given_event_est'])
    assert set(model.score(X, y, score_sub_models=True)) == {'combined', 'event_est', 'value_given_event_est'}

    clone(model)
    with pytest.raises(ValueError, match='non-negative'):
        model.fit(X, -y)


def test_log_smearing_matches_lognormal():
    """When residuals really are normal, smearing should approx. match the parametric mean."""
    X, y = _sim_lognormal(seed=0)
    glm = Glm('lognormal', col_mapping=['loc', 'scale'])
    smeared = LogSmearingRegressor(glm).fit(X, y, verbose=False, estimate_laplace_coefs=False)
    parametric = smeared.estimator_.predict(X).ravel()
    np.testing.assert_allclose(smeared.predict(X), parametric, rtol=.06)  # high-sigma rows depend on few tail resids

    # quantiles should have the right coverage on new data:
    X_new, y_new = _sim_lognormal(seed=1)
    for q in (.1, .5, .9):
        assert np.mean(y_new < smeared.predict_quantile(X_new, q)) == pytest.approx(q, abs=.015)
