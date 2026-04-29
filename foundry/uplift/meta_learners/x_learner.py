import warnings
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from foundry.util import SliceDict, to_1d, safe_predict
from ..util import get_qini_curve, get_cumulative_gain_score


class XLearner(BaseEstimator):
    """
    An X-learner. Please note that current implementation assumes randomized treatment/control!
    Adapted from https://matheusfacure.github.io/python-causality-handbook/21-Meta-Learners.html 
    and the original paper by Kunzel et al. (2019): https://arxiv.org/abs/1706.03461

    :param first_stage_estimator: Any instance that supports the sklearn API (fit/predict and can call ``clone()`` on it).
    :param second_stage_estimator: Any instance that supports the sklearn API (fit/predict and can call ``clone()`` on it).
    :param propensity_estimator: Any instance that supports the sklearn API (fit/predict and can call ``clone()`` on it).
    :param first_stage_fit_params: Optional dict of kwargs passed to ``fit()`` for the first-stage models.
    :param second_stage_fit_params: Optional dict of kwargs passed to ``fit()`` for the second-stage models.
    :param propensity_fit_params: Optional dict of kwargs passed to ``fit()`` for the propensity model.
    """
    first_treatment_est_ = None
    first_control_est_ = None
    second_treatment_est_ = None
    second_control_est_ = None

    def __init__(
        self,
        first_stage_estimator: BaseEstimator,
        second_stage_estimator: BaseEstimator,
        propensity_estimator: BaseEstimator,
        first_stage_fit_params: dict = None,
        second_stage_fit_params: dict = None,
        propensity_fit_params: dict = None,
    ):
        self.first_stage_estimator = first_stage_estimator
        self.second_stage_estimator = second_stage_estimator
        self.propensity_estimator = propensity_estimator
        self.first_stage_fit_params = first_stage_fit_params
        self.second_stage_fit_params = second_stage_fit_params
        self.propensity_fit_params = propensity_fit_params

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: SliceDict) -> "XLearner":
        y, treatment_ind = self._normalize_y(y)

        _first_stage_fit_params = self.first_stage_fit_params or {}
        _second_stage_fit_params = self.second_stage_fit_params or {}
        _propensity_fit_params = self.propensity_fit_params or {}

        self.first_control_est_ = clone(self.first_stage_estimator).fit(
            X[~treatment_ind], y[~treatment_ind], **_first_stage_fit_params
        )
        self.first_treatment_est_ = clone(self.first_stage_estimator).fit(
            X[treatment_ind], y[treatment_ind], **_first_stage_fit_params
        )

        self.propensity_est_ = clone(self.propensity_estimator).fit(
            X, treatment_ind, **_propensity_fit_params
        )

        imputed_te = np.where(
            treatment_ind,
            y - safe_predict(self.first_control_est_, X),
            safe_predict(self.first_treatment_est_, X) - y,
        )

        self.second_control_est_ = clone(self.second_stage_estimator).fit(
            X[~treatment_ind], imputed_te[~treatment_ind], **_second_stage_fit_params
        )
        self.second_treatment_est_ = clone(self.second_stage_estimator).fit(
            X[treatment_ind], imputed_te[treatment_ind], **_second_stage_fit_params
        )

        return self

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        return_components: bool = False,
        **predict_kwargs,
    ) -> np.ndarray:
        ps = self.propensity_est_.predict_proba(X)
        p_control   = ps[:, 0]
        p_treatment = ps[:, 1]

        tau0 = safe_predict(self.second_control_est_,   X, **predict_kwargs)
        tau1 = safe_predict(self.second_treatment_est_, X, **predict_kwargs)

        if return_components:
            return tau0, tau1
        return p_treatment * tau0 + p_control * tau1

    def score(self, X, y, sample_weight=None, method='qini', normalize=True, **kwargs) -> float:
        y, treatment_ind = self._normalize_y(y)
        if sample_weight is not None:
            raise NotImplementedError
        pred = self.predict(X=X)
        if method == 'cumulative_gain':
            return get_cumulative_gain_score(
                y_true=y,
                treatment=treatment_ind,
                score=pred,
                **kwargs,
            )
        qini = get_qini_curve(
            y_true=y,
            treatment=treatment_ind,
            score=pred,
            normalize=normalize,
            **kwargs,
        )
        random_area = np.linspace(0, qini[-1], qini.shape[0]).sum()
        return (np.nansum(qini) - random_area) / qini.shape[0]

    @staticmethod
    def _normalize_y(y: SliceDict) -> tuple[np.ndarray, np.ndarray]:
        y = y.copy()
        y_arr = to_1d(np.asanyarray(y.pop('value')))
        treatment_ind = to_1d(np.asanyarray(y.pop('is_treatment')).astype(bool))
        if len(y.keys()):
            warnings.warn(f"Unused keys in ``y``: {set(y)}")
        return y_arr, treatment_ind