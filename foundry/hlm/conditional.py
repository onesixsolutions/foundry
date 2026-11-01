from typing import Optional, Callable, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from foundry.util import ModelMatrix, SliceDict


def conditional_predict(estimator: BaseEstimator,
                        X: ModelMatrix,
                        group_data: Optional[tuple] = None,
                        method: str = 'predict',
                        **kwargs) -> Union[np.ndarray, float]:
    """
    Call ``predict`` (or another method like ``predict_proba``, ``score``) on an :class:`foundry.hlm.Hlm`, or a
    ``Pipeline`` ending in one, passing ``group_data``. For a pipeline, both ``X`` and the ``X`` in ``group_data``
    are transformed by the preceding steps.

    :param estimator: A fitted ``Hlm`` or pipeline ending in one.
    :param X: Data to predict.
    :param group_data: Tuple of ``(X, y)`` or ``(X, y, sample_weight)`` with data for the groups in ``X``.
    :param method: The method to call, e.g. 'predict', 'predict_proba', 'score'.
    :param kwargs: Other keyword-arguments to the method, e.g. ``y`` for 'score'.
    """
    if isinstance(estimator, Pipeline):
        preprocessor, estimator = estimator[:-1], estimator[-1]
        X = preprocessor.transform(X)
        if group_data is not None:
            group_data = (preprocessor.transform(group_data[0]),) + tuple(group_data[1:])
    return getattr(estimator, method)(X, group_data=group_data, **kwargs)


class ConditionalScorer:
    """
    A scorer (for ``GridSearchCV``, ``cross_val_score``, etc.) for the use-case where predictions for a group are made
    conditional on that group's data available at prediction-time. Within each group in the validation data, rows are
    split into "history" (passed as ``group_data``) and "target" (scored). Use with a group-wise splitter (e.g.
    ``GroupKFold``) so that validation groups are unseen in training.

    :param grouping_factor: The column (or dict-key) in the *raw* ``X`` identifying groups; defaults to the
     estimator's grouping factor.
    :param history_frac: Fraction of each group's rows used as history (rounded down, but leaving at least one target
     row). Groups with a single row have no history.
    :param order_by: Optional column used to order rows within each group (e.g. a date), so that history precedes
     targets. Default is row-order.
    :param metric: Optional function of ``(y_true, y_pred)``; default is the mean log-prob (``Hlm.score``).
    :param greater_is_better: If ``metric`` is a loss, set to False (the score is then negated, as sklearn expects).
    :param predict_kwargs: Keyword-arguments to ``predict`` when ``metric`` is provided (e.g. ``type='probs'``).
    """

    def __init__(self,
                 grouping_factor: Optional[str] = None,
                 history_frac: float = .5,
                 order_by: Optional[str] = None,
                 metric: Optional[Callable] = None,
                 greater_is_better: bool = True,
                 predict_kwargs: Optional[dict] = None):
        self.grouping_factor = grouping_factor
        self.history_frac = history_frac
        self.order_by = order_by
        self.metric = metric
        self.greater_is_better = greater_is_better
        self.predict_kwargs = predict_kwargs

    def split(self, X: ModelMatrix, grouping_factor: str) -> np.ndarray:
        """
        :return: A boolean array, True for history rows.
        """
        groups = pd.Series(np.asarray(X[grouping_factor]).reshape(-1))
        order = pd.Series(np.arange(len(groups)) if self.order_by is None else np.asarray(X[self.order_by]))
        rank = order.groupby(groups.values).rank(method='first').values - 1
        size = groups.map(groups.value_counts()).values
        n_history = np.minimum(np.floor(size * self.history_frac), size - 1)
        return rank < n_history

    def __call__(self, estimator: BaseEstimator, X: ModelMatrix, y: ModelMatrix, sample_weight=None) -> float:
        grouping_factor = self.grouping_factor
        if grouping_factor is None:
            final = estimator[-1] if isinstance(estimator, Pipeline) else estimator
            grouping_factor = final.grouping_factor
        is_history = self.split(X, grouping_factor)
        group_data = (_take(X, is_history), _take(y, is_history))
        if sample_weight is not None:
            group_data += (_take(sample_weight, is_history),)
            sample_weight = _take(sample_weight, ~is_history)
        X, y = _take(X, ~is_history), _take(y, ~is_history)

        if self.metric is None:
            return conditional_predict(
                estimator, X, group_data=group_data, method='score', y=y, sample_weight=sample_weight
            )
        y_pred = conditional_predict(estimator, X, group_data=group_data, **(self.predict_kwargs or {}))
        y_true = y['value'] if isinstance(y, dict) else y
        score = self.metric(y_true, y_pred)
        return score if self.greater_is_better else -score


def _take(obj, mask: np.ndarray):
    if isinstance(obj, dict):
        out = {k: _take(v, mask) for k, v in obj.items()}
        return SliceDict(**out) if isinstance(obj, SliceDict) else out
    if not hasattr(obj, '__len__') or isinstance(obj, str) or np.ndim(obj) == 0:
        return obj  # scalars pass through
    if hasattr(obj, 'iloc'):
        return obj.iloc[np.flatnonzero(mask)]
    return np.asarray(obj)[mask]
