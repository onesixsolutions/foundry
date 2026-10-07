from typing import Optional, Tuple

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, RegressorMixin, clone

from foundry.util import to_1d


class LogSmearingRegressor(RegressorMixin, BaseEstimator):
    r"""
    Wraps a location-scale model of ``log(y)`` -- typically ``Glm('lognormal', col_mapping=['loc', 'scale'])`` --
    and replaces its parametric back-transform with a 'smearing' estimate (Duan, 1983; heteroscedastic version).

    The wrapped model gives :math:`\mu(x)` and :math:`\sigma(x)`. Instead of assuming
    :math:`\log(y) \sim N(\mu, \sigma^2)`, we assume only that :math:`z = (\log(y) - \mu) / \sigma` has the same
    distribution for every row, and estimate that distribution with the training residuals :math:`z_i`:

    .. math::

        E[y | x] = \exp(\mu(x)) \cdot \frac{1}{n} \sum_i \exp(\sigma(x) z_i)

    The residuals are in-sample; with many parameters relative to rows, these will be slightly too small.

    :param estimator: An estimator whose ``predict(X, type='loc')`` and ``predict(X, type='scale')`` return the
     location and scale of ``log(y)``. This includes a ``Pipeline`` ending in such an estimator.
    :param n_grid: The smearing factor depends only on :math:`\sigma(x)`, so at predict-time it's computed on a grid of
     this many sigmas and interpolated.
    """

    def __init__(self, estimator: BaseEstimator, n_grid: int = 256):
        self.estimator = estimator
        self.n_grid = n_grid

    def fit(self, X, y, sample_weight: Optional[np.ndarray] = None, **fit_params) -> 'LogSmearingRegressor':
        """
        :param X: Predictors.
        :param y: A strictly-positive target (not ``log(y)``).
        :param sample_weight: Weights used for the residual distribution; also passed to ``estimator.fit`` if given.
        :param fit_params: Passed to ``estimator.fit``.
        :return: This instance.
        """
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight
        self.estimator_ = clone(self.estimator).fit(X, y, **fit_params)

        loc, scale = self.predict_loc_scale(X)
        resid = (np.log(to_1d(np.asarray(y, dtype='float'))) - loc) / scale
        weights = np.ones_like(resid) if sample_weight is None else to_1d(np.asarray(sample_weight, dtype='float'))
        order = np.argsort(resid)
        self.residuals_ = resid[order]
        self.residual_weights_ = weights[order] / weights.sum()
        return self

    def predict_loc_scale(self, X) -> Tuple[np.ndarray, np.ndarray]:
        loc = to_1d(self.estimator_.predict(X, type='loc'))
        scale = to_1d(self.estimator_.predict(X, type='scale'))
        return loc, scale

    def predict(self, X) -> np.ndarray:
        """
        :return: The smearing estimate of ``E[y | X]``.
        """
        loc, scale = self.predict_loc_scale(X)
        grid = np.linspace(scale.min(), scale.max(), self.n_grid)
        # log(mean_i(exp(s * z_i))) for each s in grid:
        log_factor = np.array([logsumexp(s * self.residuals_, b=self.residual_weights_) for s in grid])
        return np.exp(loc + np.interp(scale, grid, log_factor))

    def predict_quantile(self, X, q: float) -> np.ndarray:
        """
        :return: The ``q``-th quantile of ``y | X``, using the empirical quantile of the residuals.
        """
        loc, scale = self.predict_loc_scale(X)
        idx = np.searchsorted(np.cumsum(self.residual_weights_), q)
        z_q = self.residuals_[min(idx, len(self.residuals_) - 1)]
        return np.exp(loc + scale * z_q)
