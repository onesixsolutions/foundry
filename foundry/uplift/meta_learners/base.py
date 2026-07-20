import warnings

import numpy as np
from foundry.util import SliceDict, to_1d
from sklearn.base import BaseEstimator


class MetaLearner(BaseEstimator):
    @classmethod
    def normalize_y(cls, y: SliceDict) -> tuple[np.ndarray, np.ndarray]:
        y = y.copy()
        y_arr = to_1d(np.asanyarray(y.pop("value")))
        treatment_ind = to_1d(np.asanyarray(y.pop("is_treatment")).astype(bool))
        # TODO: validate treatment is binary?
        if len(y):
            warnings.warn(f"Unused keys in ``y``: {set(y)}")
        return y_arr, treatment_ind
