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

    Supports both single-component and multi-component estimators.

    For single-component models, pass y as a SliceDict with keys 'value' and
    'is_treatment'. For multi-component models (e.g. ThreeComponentAgent), pass
    sub_model_keys and include those keys directly in the SliceDict alongside
    'is_treatment'.

    :param estimator: Any instance that supports the sklearn API (fit/predict and can call ``clone()`` on it).
    :param sub_model_keys: If provided, y is treated as a multi-component target
        where each key in sub_model_keys is a separate sub-model target. Leave
        empty for single-component usage.
    
    Examples
    --------
    Single-component usage::

        learner = TLearner(estimator=LGBMRegressor())
        learner.fit(
            X=df,
            y=SliceDict(
                value=df['net_theo_90d'],
                is_treatment=(df['treatment'] == 1),
            )
        )
        uplift_scores = learner.predict(X=df)

    Multi-component usage with ThreeComponentAgent::

        net_theo_model = ThreeComponentAgent(
            sub_models={
                'visit_rate_model':  visit_rate_model,
                'gross_theo_model':  gross_theo_model,
                'redemptions_model': redemptions_model,
            },
            visit_rate_model_is_per_day=True,
        )
        learner = TLearner(
            estimator=net_theo_model,
            sub_model_keys=('visit_rate_model', 'gross_theo_model', 'redemptions_model'),
        )
        learner.fit(
            X=df,
            y=SliceDict(
                visit_rate_model=df['prop_visits'],
                gross_theo_model=df['avg_gross_theo'],
                redemptions_model=df['avg_redemptions'],
                is_treatment=(df['treatment'] == 1),
            )
        )
        uplift_scores = learner.predict(X=df)
    """
    treatment_est_ = None
    control_est_ = None

    def __init__(self, estimator: BaseEstimator, sub_model_keys: tuple = ()):
        self.estimator = estimator
        self.sub_model_keys = sub_model_keys

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: SliceDict, **fit_kwargs) -> "TLearner":
        y, treatment_ind = self._normalize_y(y, self.sub_model_keys)
        self.treatment_est_ = clone(self.estimator).fit(X=X[treatment_ind], y=y[treatment_ind], **fit_kwargs)
        self.control_est_ = clone(self.estimator).fit(X=X[~treatment_ind], y=y[~treatment_ind], **fit_kwargs)

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray], return_components: bool = False,
                **predict_kwargs) -> np.ndarray:

        yhat_t = safe_predict(self.treatment_est_, X=X, **predict_kwargs)
        yhat_c = safe_predict(self.control_est_, X=X, **predict_kwargs)
        if return_components:
            return yhat_t, yhat_c
        return yhat_t - yhat_c

    def score(self, X, y, sample_weight=None, method='qini', normalize=True, **kwargs) -> float:
        y, treatment_ind = self._normalize_y(y, self.sub_model_keys)
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
            **kwargs
        )
        random_area = np.linspace(0, qini[-1], qini.shape[0]).sum()
        return (np.nansum(qini) - random_area) / qini.shape[0]

    @staticmethod
    def _normalize_y(y: SliceDict, sub_model_keys: tuple) -> tuple[np.ndarray, np.ndarray]:
        y = y.copy()
        treatment_ind = to_1d(np.asanyarray(y.pop('is_treatment')).astype(bool))
        
        if sub_model_keys:
            y_arr = SliceDict(**{k: y.pop(k) for k in sub_model_keys if k in y})
        else:
            y_arr = to_1d(np.asanyarray(y.pop('value')))
        
        if len(y.keys()):
            warnings.warn(f"Unused keys in ``y``: {set(y)}")
        
        return y_arr, treatment_ind
