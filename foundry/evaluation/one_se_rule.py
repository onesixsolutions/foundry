from typing import Callable, Union, Optional, Sequence

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    import lightgbm as lgb

    _LGBM_DEFAULTS = lgb.LGBMRegressor().get_params()
except ImportError:
    lgb = None
    _LGBM_DEFAULTS = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": -1, "num_leaves": 31}

_DT_DEFAULTS = {**DecisionTreeClassifier().get_params(), **DecisionTreeRegressor().get_params()}
_RF_DEFAULTS = {**RandomForestClassifier().get_params(), **RandomForestRegressor().get_params()}


class OneSeRule:
    """
    Pass this to the `refit` parameter of GridSearchCV so that, instead of ``best_estimator_`` returning the
    best-scoring configuration, it returns the *simplest* configuration within one SE of the best score.

    :param estimator: The estimator that will get tuned (bare, or wrapped in a Pipeline -- the last step is
     used automatically). If unsupported, pass a callable instead that takes param-names as keyword args and
     returns a >0 float for complexity.
    :param sub_estimators: Only needed when `estimator` is a meta-estimator (or a pipeline holding one) with more than
     one independently tunable sub-estimator inside (e.g. a hurdle model). Two supported forms:
       - A single attribute name (str) holding a list of ``(name, sub_estimator, ...)`` tuples on `estimator`
         -- e.g. VotingRegressor/StackingRegressor's ``estimators`` attribute. Extra tuple elements are ignored.
       - A sequence of attribute names, each directly holding one sub-estimator -- e.g.
         ``sub_estimators=("classifier", "regressor")`` for a hurdle model storing its branches under
         `self.classifier` / `self.regressor`.
       These are read off the *unfitted* estimator (OneSeRule is built before GridSearchCV calls `.fit()`), so
       use constructor-parameter attribute names, not attributes only set during fitting.
    :param complexity_reduce_fun: Required whenever there's more than one sub-estimator. Takes a sequence of
     per-sub-estimator complexity floats and reduces them to one float (or tuple) for the combined model --
     e.g. `math.prod` for a hurdle model, where the branches compose serially (partition refinement), which
     multiplication captures better than summing independent contributors.
    """

    def __init__(self,
                 estimator: Union[BaseEstimator, Callable],
                 sub_estimators: Union[str, Sequence[str]] = (),
                 complexity_reduce_fun: Optional[Callable] = None,
                 verbose: bool = True):

        self.verbose = verbose

        prefix, estimator = self._peel_pipeline(estimator)

        estimators = {}
        if sub_estimators:
            if isinstance(sub_estimators, str):
                for nm, sest, *_ in getattr(estimator, sub_estimators):
                    sub_prefix, sest = self._peel_pipeline(sest)
                    estimators[f"{prefix}{nm}__{sub_prefix}"] = sest
            else:
                _missing = object()
                for attr in sub_estimators:
                    sest = getattr(estimator, attr, _missing)
                    if sest is _missing:
                        raise ValueError(
                            f"{estimator!r} has no attribute {attr!r}. `sub_estimators` names must match "
                            "constructor-parameter attributes on the *unfitted* estimator (no trailing underscore)."
                        )
                    sub_prefix, sest = self._peel_pipeline(sest)
                    estimators[f"{prefix}{attr}__{sub_prefix}"] = sest
        else:
            estimators = {prefix: estimator}

        self.complexity_reduce_fun = complexity_reduce_fun

        self.params_to_complexity_funs = {}
        for pfx, est_or_callable in estimators.items():
            if callable(est_or_callable):
                self.params_to_complexity_funs[pfx] = est_or_callable
            else:
                self.params_to_complexity_funs[pfx] = self._estimator_to_complexity_func(est_or_callable)

    @staticmethod
    def _peel_pipeline(estimator) -> tuple[str, BaseEstimator]:
        """Unwrap one level of Pipeline, returning (stepname__ or '', leaf_estimator)."""
        steps = getattr(estimator, "steps", None)
        if not steps:
            return "", estimator
        step_name, leaf = steps[-1]
        return f"{step_name}__", leaf

    def __call__(self, cv_results: dict[str, np.ndarray]) -> int:
        mean_scores = np.array(cv_results["mean_test_score"])
        std_scores = np.array(cv_results["std_test_score"])

        n_splits = sum(1 for k in cv_results if k.startswith("split") and k.endswith("_test_score"))
        se = std_scores / np.sqrt(n_splits)

        best_idx = np.argmax(mean_scores)
        threshold = mean_scores[best_idx] - se[best_idx]
        candidates = [i for i, s in enumerate(mean_scores) if s >= threshold]

        any_row_keys = set(cv_results["params"][0])
        active_prefixes = {}
        for prefix, func in self.params_to_complexity_funs.items():
            if any(k.startswith(prefix) for k in any_row_keys):
                active_prefixes[prefix] = func
            elif self.verbose:
                print(f"`{prefix}` does not appear to be getting tuned so will not contribute to complexity calcs")

        if not active_prefixes:
            raise ValueError(
                "None of the registered sub-estimator prefixes "
                f"{list(self.params_to_complexity_funs)} matched any params in this grid "
                f"(got keys {any_row_keys}). Nothing would be measured for complexity."
            )
        if self.complexity_reduce_fun is None:
            if len(active_prefixes) == 1:
                self.complexity_reduce_fun = _default_complexity_reduce_fun
            else:
                raise ValueError("If multiple estimators are passed, must supply `complexity_reduce_fun`.")

        def complexity(i):
            params_this_row = cv_results["params"][i]
            complexities = []
            for prefix, params_to_complexity in active_prefixes.items():
                _params = {
                    k.removeprefix(prefix): v for k, v in params_this_row.items() if k.startswith(prefix)
                }
                try:
                    complexities.append(params_to_complexity(**_params))
                except TypeError as e:
                    raise RuntimeError(
                        f"Failed to compute complexity when passing {set(_params)} for prefix {prefix!r}."
                    ) from e
            return self.complexity_reduce_fun(complexities), -mean_scores[i]

        return min(candidates, key=complexity)

    @classmethod
    def _estimator_to_complexity_func(cls, estimator: BaseEstimator) -> Callable:
        if isinstance(estimator,
                      (DecisionTreeClassifier, DecisionTreeRegressor)):
            return cls._decision_tree_complexity

        if isinstance(estimator,
                      (RandomForestClassifier, RandomForestRegressor)):
            return cls._random_forest_complexity

        if lgb is not None and isinstance(estimator,
                                          (lgb.LGBMRegressor, lgb.LGBMClassifier, lgb.LGBMModel, lgb.LGBMRanker)):
            return cls._lgbm_complexity

        raise ValueError(
            "Do not know how to define complexity for {}, please provide a callable that takes params and "
            "returns a float".format(type(estimator).__name__)
        )

    @staticmethod
    def _decision_tree_complexity(max_depth=_DT_DEFAULTS["max_depth"],
                                  max_leaf_nodes=_DT_DEFAULTS["max_leaf_nodes"],
                                  **kwargs) -> float:
        unexpected = set(kwargs) - set(_DT_DEFAULTS)
        if unexpected:
            raise ValueError(
                f"Got params not recognized by the DecisionTree sklearn API defaults: {unexpected}. "
            )
        return _treelike_complexity(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, **kwargs)

    @staticmethod
    def _random_forest_complexity(max_depth=_RF_DEFAULTS["max_depth"],
                                  max_leaf_nodes=_RF_DEFAULTS["max_leaf_nodes"],
                                  **kwargs) -> float:
        unexpected = set(kwargs) - set(_RF_DEFAULTS)
        if unexpected:
            raise ValueError(
                f"Got params not recognized by the RandomForest sklearn API defaults: {unexpected}. "
            )
        return _treelike_complexity(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, **kwargs)

    @staticmethod
    def _lgbm_complexity(n_estimators=_LGBM_DEFAULTS["n_estimators"],
                         learning_rate=_LGBM_DEFAULTS["learning_rate"],
                         max_depth=_LGBM_DEFAULTS["max_depth"],
                         num_leaves=_LGBM_DEFAULTS["num_leaves"],
                         **kwargs) -> float:
        unexpected = set(kwargs) - set(_LGBM_DEFAULTS)
        if unexpected:
            raise ValueError(
                f"Got params not recognized by the LGBM sklearn API defaults: {unexpected}. "
                "Are you using a native LightGBM alias (e.g. `min_data_in_leaf`, `feature_fraction`, `lambda_l1`) "
                "instead of the sklearn wrapper's canonical param name?"
            )
        cap = np.inf if (max_depth is None or max_depth <= 0) else 2 ** max_depth
        effective_iterations = n_estimators * learning_rate
        return effective_iterations * min(cap, num_leaves)


def _treelike_complexity(**kwargs) -> float:
    max_depth = kwargs.pop("max_depth", None)
    max_leaf_nodes = kwargs.pop("max_leaf_nodes", None)
    depth_cap = np.inf if (max_depth is None or max_depth <= 0) else 2 ** max_depth
    leaf_cap = np.inf if max_leaf_nodes is None else max_leaf_nodes
    return min(depth_cap, leaf_cap)


def _default_complexity_reduce_fun(complexities: Sequence[float]) -> float:
    assert len(complexities) == 1
    return complexities[0]
