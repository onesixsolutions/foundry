import warnings
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from foundry.util import SliceDict, to_1d, safe_predict
from ..util import get_qini_curve, get_cumulative_gain_score


class SLearner(BaseEstimator):
    """
    An S-learner. Please note that current implementation assumes randomized treatment/control!

    Supports both single-component and multi-component estimators.

    For single-component models, pass y as a SliceDict with keys 'value' and
    'is_treatment'. For multi-component models (e.g. ThreeComponentAgent), pass
    sub_model_keys and include those keys directly in the SliceDict alongside
    'is_treatment'.

    :param estimator: Any instance that supports the sklearn API (fit/predict and can call ``clone()`` on it).
    :param include_interaction: Whether to include X * treatment interaction terms.
    :param sub_model_keys: If provided, y is treated as a multi-component target
        where each key in sub_model_keys is a separate sub-model target. Leave
        empty for single-component usage.
    
    Examples
    --------
    Single-component usage::

        learner = SLearner(estimator=LGBMRegressor())
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
        learner = SLearner(
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
    estimator_ = None

    def __init__(self, estimator: BaseEstimator, include_interaction: bool = False, sub_model_keys: tuple = ()):
        self.estimator = estimator
        self.include_interaction = include_interaction
        self.sub_model_keys = sub_model_keys

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: SliceDict, **fit_kwargs) -> "SLearner":
        y, treatment_ind = self._normalize_y(y, self.sub_model_keys)

        X_aug = self._augment_with_treatment(X, treatment_ind)

        self.estimator_ = clone(self.estimator).fit(X=X_aug, y=y, **fit_kwargs)
        # verify treatment signal is preserved — if CATE std is zero treatment
        # was likely stripped by a downstream transformer e.g. ColumnSelector
        cate = self.predict(X)
        assert cate.std() > 0, (  ## TODO: numpy is close instead of gt
            "SLearner CATE std is zero — treatment column is likely being stripped "
            "by a downstream transformer (e.g. ColumnSelector missing 'treatment')."
        )
        return self

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        return_components: bool = False,
        **predict_kwargs,
    ) -> np.ndarray:

        X_t = self._augment_with_treatment(X, np.ones(len(X), dtype=bool))
        X_c = self._augment_with_treatment(X, np.zeros(len(X), dtype=bool))

        yhat_t = safe_predict(self.estimator_, X=X_t, **predict_kwargs)
        yhat_c = safe_predict(self.estimator_, X=X_c, **predict_kwargs)

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
            **kwargs,
        )
        random_area = np.linspace(0, qini[-1], qini.shape[0]).sum()
        return (np.nansum(qini) - random_area) / qini.shape[0]

    def _augment_with_treatment(self, X, treatment_ind):
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

    @staticmethod
    def _normalize_y(y: SliceDict, sub_model_keys: tuple) -> tuple[np.ndarray, np.ndarray]:
        y = y.copy()
        treatment_ind = to_1d(np.asanyarray(y.pop("is_treatment")).astype(bool))
        
        if sub_model_keys:
            y_arr = SliceDict(**{k: y.pop(k) for k in sub_model_keys if k in y})
        else:
            y_arr = to_1d(np.asanyarray(y.pop("value")))
        
        if len(y.keys()):
            warnings.warn(f"Unused keys in ``y``: {set(y)}")
        
        return y_arr, treatment_ind
