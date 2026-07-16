import itertools
from typing import Union, Sequence, Tuple, Callable
from warnings import warn

import pandas as pd
from pandas.core.arrays import SparseArray
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError


class InteractionFeatures(TransformerMixin, BaseEstimator):
    """
    Take a model-matrix (dataframe) and return a copy with added interaction-terms.

    :param interactions: A list of tuples. Each contains either (1) a column-name, (2) a 'column-selector' like
     ``sklearn.compose.make_column_selector()`` that takes a model-matrix and returns a list of column-names.
    :param sep: The separator between colnames in the outputted columns.
    :param no_selection_handling: During fit, if an element of ``interactions`` is not in the input (or is a
     column-selector and it returns no columns), how should this be handled? The default 'raise' will throw an
     exception, 'warn' will emit a warning, and 'ignore' will ignore.
    """

    def __init__(
            self,
            interactions: Sequence[Tuple[Union[str, Callable], ...]] = (),
            sep: str = ":",
            no_selection_handling: str = 'raise'
    ):
        self.interactions = interactions
        self.sep = sep
        self.unrolled_interactions_ = None
        self.no_selection_handling = no_selection_handling
        self.feature_names_in_ = None

    def fit(self, X: pd.DataFrame, y=None) -> 'InteractionFeatures':
        assert not X.empty
        self.feature_names_in_ = X.columns
        self.no_selection_handling = self.no_selection_handling.lower()
        assert self.no_selection_handling in {'raise', 'warn', 'ignore'}

        # TODO: change this implementation to be more efficient for dense*onehot.
        #       for example, imagine modeling a coupon_amount * customer_id interaction for many customers, and
        #       we've one-hot encoded customer. current implementation would require looping over each customer to
        #       multiply their sparse indicator by the `coupon_amount` column. more efficient would be selecting all
        #       customer-indicator columns and multiplying by the `coupon_amount` column in one operation.
        unrolled_interactions = []
        unrolled_interaction_sets = set()
        for int_idx, cols in enumerate(self.interactions):
            any_callables = False
            to_unroll = {}
            for i, col in enumerate(cols):
                if callable(col):
                    any_callables = True
                    to_unroll[i] = col(X)
                    if not to_unroll[i]:
                        _repr = f'({col}) ' if '<' not in repr(col) else ''
                        msg = f"`interactions[{int_idx}][{i}]` {_repr}is a callable that returned no columns"
                        if self.no_selection_handling == 'raise':
                            raise RuntimeError(msg)
                        if self.no_selection_handling == 'warn':
                            warn(msg)
                elif isinstance(col, str):
                    if col not in X.columns:
                        msg = f"{col} not in X.columns"
                        if self.no_selection_handling == 'raise':
                            raise RuntimeError(msg)
                        if self.no_selection_handling == 'warn':
                            warn(msg)
                    to_unroll[i] = [col]
                else:
                    raise ValueError(f"Expected {col} to be str or callable, got {type(col)}")

            for interaction in itertools.product(*to_unroll.values()):
                # e.g. interactions=[('my_col',make_column_selector('*'))]
                # will include a my_col*my_col interaction that should be dropped
                if len(interaction) != len(set(interaction)):
                    if len(interaction) > 3:
                        # for three-way interactions or less, any duplicate means the interaction can be dropped
                        # for >3, unclear whether we should drop the interaction or just drop the duplicates
                        raise NotImplementedError(f"n>3-way interaction with duplicates: {interaction}")
                    if not any_callables:
                        # if callable, no warning needed
                        warn(f"Dropping {interaction} because of duplicates.")
                    continue

                frozen_interaction = frozenset(interaction)
                if frozen_interaction in unrolled_interaction_sets:
                    warn(f"{interaction} specified more than once, only keeping the first.")
                else:
                    unrolled_interactions.append(interaction)
                    unrolled_interaction_sets.add(frozen_interaction)

        self.unrolled_interactions_ = unrolled_interactions

        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        if not X.index.is_unique:
            raise ValueError(f"{type(self).__name__} only works when X has a unique index; try resetting the index.")

        orig_cols = set(X.columns)

        new_cols = {}
        for interaction_column_names in self.unrolled_interactions_:
            if len(interaction_column_names) < 2:
                continue
            if not set(interaction_column_names).issubset(orig_cols):
                raise RuntimeError(
                    f"Columns {interaction_column_names} in `interactions`, but not in columns:\n{orig_cols}"
                )

            new_colname = self.sep.join(interaction_column_names)
            if new_colname in orig_cols:
                warn(f"{new_colname} is already in X, overwriting.")

            new_cols[new_colname] = None
            for col in interaction_column_names:
                new_cols[new_colname] = _sparse_safe_multiply(new_cols[new_colname], X[col].values)

        df_new_cols = pd.DataFrame(new_cols, index=X.index)

        return X.drop(columns=X.columns[X.columns.isin(new_cols)]).join(df_new_cols)

    def get_feature_names_out(self, feature_names_in=None):
        if self.feature_names_in_ is None:
            raise NotFittedError(f"This {type(self).__name__} is not fitted. Cannot get_feature_names_out.")

        feature_names_out = list(self.feature_names_in_)
        for interaction_column_names in self.unrolled_interactions_:
            if len(interaction_column_names) < 2:
                continue
            new_col = self.sep.join(interaction_column_names)
            if new_col in feature_names_out:
                # this follows the behavior above, where interactions are always placed in the same location
                # regardless of where they were in the original df
                feature_names_out.remove(new_col)
            feature_names_out.append(new_col)

        return feature_names_out


def _sparse_safe_multiply(old_vals: pd.Series, new_vals: pd.Series) -> Union[SparseArray, pd.Series]:
    if old_vals is None:
        return new_vals.copy()

    # because sparse-arrays allow for any fill-value, they don't leverage the fact that, if fill_value=0,
    # sparse*sparse only needs to capture the intersection of the two; instead, they fill the union.
    old_is_sparse = isinstance(old_vals.dtype, pd.SparseDtype)
    new_is_sparse = isinstance(new_vals.dtype, pd.SparseDtype)
    if new_is_sparse and old_is_sparse:
        index_intersection = old_vals.sp_index.intersect(new_vals.sp_index)
        assert new_vals.fill_value == old_vals.fill_value == 0
    elif new_is_sparse:
        index_intersection = new_vals.sp_index
        assert new_vals.fill_value == 0
    elif old_is_sparse:
        index_intersection = old_vals.sp_index
        assert old_vals.fill_value == 0
    else:
        old_vals = old_vals * new_vals
        return old_vals
    product = (
            old_vals[index_intersection.to_int_index().indices] *
            new_vals[index_intersection.to_int_index().indices]
    )
    return SparseArray(data=product, sparse_index=index_intersection, fill_value=0.)
