import warnings
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from foundry.util import SliceDict, to_1d, safe_predict
from ..util import get_qini_curve, get_cumulative_gain_score


class TLearner(BaseEstimator):
    """
    A T-learner. Please note that current implementation assumes randomized treatment/control!

    :param estimator: Any instance that supports the sklearn API (fit/predict and can call ``clone()`` on it).
    """
    treatment_est_ = None
    control_est_ = None

    def __init__(self, estimator: BaseEstimator):
        self.estimator = estimator

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: SliceDict, **fit_kwargs) -> "TLearner":
        y, treatment_ind = self._normalize_y(y)
        self.treatment_est_ = clone(self.estimator).fit(X=X[treatment_ind], y=y[treatment_ind], **fit_kwargs)
        self.control_est_ = clone(self.estimator).fit(X[~treatment_ind], y[~treatment_ind], **fit_kwargs)

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray], return_components: bool = False,
                **predict_kwargs) -> np.ndarray:

        yhat_t = safe_predict(self.treatment_est_, X=X, **predict_kwargs)
        yhat_c = safe_predict(self.control_est_, X=X, **predict_kwargs)
        if return_components:
            return yhat_t, yhat_c
        return yhat_t - yhat_c

    def score(self, X, y, sample_weight=None, method='qini', **kwargs) -> float:
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
            normalize=True, 
            **kwargs
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
