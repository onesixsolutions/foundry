from typing import Any, Dict, Optional, Tuple, Union, overload, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from foundry.util import SliceDict, safe_predict
from ..util import get_qini_curve, get_cumulative_gain_score
from .base import MetaLearner


class XLearner(MetaLearner):
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
    first_treatment_est_: Optional[BaseEstimator] = None
    first_control_est_: Optional[BaseEstimator] = None
    second_treatment_est_: Optional[BaseEstimator] = None
    second_control_est_: Optional[BaseEstimator] = None
    propensity_est_: Optional[BaseEstimator] = None

    def __init__(
        self,
        first_stage_estimator: BaseEstimator,
        second_stage_estimator: BaseEstimator,
        propensity_estimator: BaseEstimator,
        first_stage_fit_params: Optional[Dict[str, Any]] = None,
        second_stage_fit_params: Optional[Dict[str, Any]] = None,
        propensity_fit_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.first_stage_estimator = first_stage_estimator
        self.second_stage_estimator = second_stage_estimator
        self.propensity_estimator = propensity_estimator
        self.first_stage_fit_params = first_stage_fit_params
        self.second_stage_fit_params = second_stage_fit_params
        self.propensity_fit_params = propensity_fit_params

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: SliceDict) -> "XLearner":
        y_arr, treatment_ind = self.normalize_y(y)

        _first_stage_fit_params = self.first_stage_fit_params or {}
        _second_stage_fit_params = self.second_stage_fit_params or {}
        _propensity_fit_params = self.propensity_fit_params or {}

        self.first_control_est_ = clone(self.first_stage_estimator).fit(
            X[~treatment_ind], y_arr[~treatment_ind], **_first_stage_fit_params
        )
        self.first_treatment_est_ = clone(self.first_stage_estimator).fit(
            X[treatment_ind], y_arr[treatment_ind], **_first_stage_fit_params
        )

        self.propensity_est_ = clone(self.propensity_estimator).fit(
            X, treatment_ind, **_propensity_fit_params
        )

        imputed_te = np.where(
            treatment_ind,
            y_arr - safe_predict(self.first_control_est_, X),
            safe_predict(self.first_treatment_est_, X) - y_arr,
        )

        self.second_control_est_ = clone(self.second_stage_estimator).fit(
            X[~treatment_ind], imputed_te[~treatment_ind], **_second_stage_fit_params
        )
        self.second_treatment_est_ = clone(self.second_stage_estimator).fit(
            X[treatment_ind], imputed_te[treatment_ind], **_second_stage_fit_params
        )

        return self

    @overload
    def predict(self, X: Union[pd.DataFrame, np.ndarray], return_components: Literal[False] = ..., **predict_kwargs) -> np.ndarray: ...
    @overload
    def predict(self, X: Union[pd.DataFrame, np.ndarray], return_components: Literal[True], **predict_kwargs) -> Tuple[np.ndarray, np.ndarray]: ...

    def predict(self, X, return_components=False, **predict_kwargs):
        p_treatment = safe_predict(self.propensity_est_, X)
        p_control = 1 - p_treatment

        tau0 = safe_predict(self.second_control_est_, X, **predict_kwargs)
        tau1 = safe_predict(self.second_treatment_est_, X, **predict_kwargs)

        if return_components:
            return tau0, tau1
        return p_treatment * tau0 + p_control * tau1

    def score(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: SliceDict,
        sample_weight: Optional[np.ndarray] = None,
        method: str = 'qini',
        normalize: bool = True,
        **kwargs,
    ) -> float:
        y_arr, treatment_ind = self.normalize_y(y)
        if sample_weight is not None:
            raise NotImplementedError
        pred = self.predict(X=X)
        if method == 'cumulative_gain':
            return get_cumulative_gain_score(
                y_true=y_arr,
                treatment=treatment_ind,
                score=pred,
                **kwargs,
            )
        qini = get_qini_curve(
            y_true=y_arr,
            treatment=treatment_ind,
            score=pred,
            normalize=normalize,
            **kwargs,
        )
        random_area = np.linspace(0, qini[-1], qini.shape[0]).sum()
        return (np.nansum(qini) - random_area) / qini.shape[0]
