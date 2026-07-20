from typing import Optional, Tuple, Union, overload, Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from foundry.util import SliceDict, safe_predict
from ..util import get_qini_curve, get_cumulative_gain_score
from .base import MetaLearner


class TLearner(MetaLearner):
    """
    A T-learner. Please note that current implementation assumes randomized treatment/control!

    :param estimator: Any instance that supports the sklearn API (fit/predict and can call ``clone()`` on it).
    """
    treatment_est_: Optional[BaseEstimator] = None
    control_est_: Optional[BaseEstimator] = None

    def __init__(self, estimator: BaseEstimator) -> None:
        self.estimator = estimator

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: SliceDict, **fit_kwargs) -> "TLearner":
        y_arr, treatment_ind = self.normalize_y(y)
        self.treatment_est_ = clone(self.estimator).fit(X=X[treatment_ind], y=y_arr[treatment_ind], **fit_kwargs)
        self.control_est_ = clone(self.estimator).fit(X[~treatment_ind], y_arr[~treatment_ind], **fit_kwargs)

        return self

    @overload
    def predict(self, X: Union[pd.DataFrame, np.ndarray], return_components: Literal[False] = ..., **predict_kwargs) -> np.ndarray: ...
    @overload
    def predict(self, X: Union[pd.DataFrame, np.ndarray], return_components: Literal[True], **predict_kwargs) -> Tuple[np.ndarray, np.ndarray]: ...

    def predict(self, X, return_components=False, **predict_kwargs):
        yhat_t = safe_predict(self.treatment_est_, X=X, **predict_kwargs)
        yhat_c = safe_predict(self.control_est_, X=X, **predict_kwargs)
        if return_components:
            return yhat_t, yhat_c
        return yhat_t - yhat_c

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
            **kwargs
        )
        random_area = np.linspace(0, qini[-1], qini.shape[0]).sum()
        return (np.nansum(qini) - random_area) / qini.shape[0]
