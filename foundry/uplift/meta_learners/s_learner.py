from typing import Tuple, Union, overload, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from foundry.util import SliceDict, safe_predict
from ..util import get_qini_curve, get_cumulative_gain_score
from .base import MetaLearner


class SLearner(MetaLearner):
    """
    An S-learner. Please note that current implementation assumes randomized treatment/control!

    :param estimator: Any instance that supports the sklearn API (fit/predict and can call ``clone()`` on it).
    :param include_interaction: Whether to include X * treatment interaction terms.
    """
    estimator_: Optional[BaseEstimator] = None

    def __init__(self, estimator: BaseEstimator, include_interaction: bool = True) -> None:
        self.estimator = estimator
        self.include_interaction = include_interaction

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: SliceDict, **fit_kwargs) -> "SLearner":
        y_arr, treatment_ind = self.normalize_y(y)

        X_aug = self._augment_with_treatment(X, treatment_ind)

        self.estimator_ = clone(self.estimator).fit(X=X_aug, y=y_arr, **fit_kwargs)
        return self

    @overload
    def predict(self, X: Union[pd.DataFrame, np.ndarray], return_components: Literal[False] = ..., **predict_kwargs) -> np.ndarray: ...
    @overload
    def predict(self, X: Union[pd.DataFrame, np.ndarray], return_components: Literal[True], **predict_kwargs) -> Tuple[np.ndarray, np.ndarray]: ...

    def predict(self, X, return_components=False, **predict_kwargs):
        X_t = self._augment_with_treatment(X, np.ones(len(X), dtype=bool))
        X_c = self._augment_with_treatment(X, np.zeros(len(X), dtype=bool))

        yhat_t = safe_predict(self.estimator_, X=X_t, **predict_kwargs)
        yhat_c = safe_predict(self.estimator_, X=X_c, **predict_kwargs)

        if return_components:
            return yhat_t, yhat_c

        return yhat_t - yhat_c

    def score(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: SliceDict,
        sample_weight: Union[np.ndarray, None] = None,
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

    def _augment_with_treatment(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        treatment_ind: np.ndarray,
    ) -> Union[pd.DataFrame, np.ndarray]:
        treatment_col = treatment_ind.astype(int)

        if isinstance(X, pd.DataFrame):
            X_aug = X.copy()
            X_aug["treatment"] = treatment_col

            if self.include_interaction:
                for col in X.columns:
                    X_aug[f"{col}_x_treatment"] = X[col] * treatment_col

            return X_aug

        else:
            treatment_col = treatment_col.reshape(-1, 1)

            if self.include_interaction:
                interaction_terms = X * treatment_col
                return np.hstack([X, treatment_col, interaction_terms])

            return np.hstack([X, treatment_col])
