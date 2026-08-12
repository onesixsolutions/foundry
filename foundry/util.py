from typing import Union, Dict

import numpy as np
import pandas as pd
import torch
from pandas.core.dtypes.cast import find_common_type
from scipy.sparse import coo_matrix
from sklearn.base import BaseEstimator, TransformerMixin

ArrayType = Union[np.ndarray, torch.Tensor]
ModelMatrix = Union[np.ndarray, pd.DataFrame, Dict[str, Union[np.ndarray, pd.DataFrame]]]


def safe_predict(estimator, *args, **kwargs) -> np.ndarray:
    if hasattr(estimator, 'predict_proba'):
        try:
            out = estimator.predict_proba(*args, **kwargs)
        except NotImplementedError:
            out = None

        if out is not None:
            if len(out.shape) == 2:
                if out.shape[1] == 2:
                    out = out[:, 1]
                elif out.shape[1] > 2:
                    raise NotImplementedError("Multi-class ``predict_proba`` not supported.")
            elif len(out.shape) > 2:
                raise RuntimeError(f"Expected 1d or 2d but {estimator:} returned shape {out.shape}.")
            return out
    return estimator.predict(*args, **kwargs)


def transpose_last_dims(x: torch.Tensor) -> torch.Tensor:
    args = list(range(len(x.shape)))
    args[-2], args[-1] = args[-1], args[-2]
    return x.permute(*args)


def is_array(x) -> bool:
    """
    Check if an object is an array, by checking if it has an `__array__` method. This is usually to distinguish between
    arguments that are meant to be converted to tensors, and those that are meant to be left as-is, e.g., when
    preparing arbitrary arguments for an arbitrary pytorch method in ``Glm.predict``.
    """
    return hasattr(x, '__array__')


def get_df_density(x: pd.DataFrame) -> float:
    densities = pd.Series(np.ones(x.shape[1]), index=x.columns)
    sparse_cols = x.columns[x.apply(pd.api.types.is_sparse).astype('bool').values]
    if len(sparse_cols):
        densities[sparse_cols] = x[sparse_cols].apply(lambda x: x.sparse.density)
    return densities.mean()


def dataframe_to_coo(x: pd.DataFrame) -> coo_matrix:
    sparse_cols = x.columns[x.apply(pd.api.types.is_sparse).astype('bool').values]
    dense_cols = x.columns[~x.columns.isin(sparse_cols)]
    sparse_dtypes = x[sparse_cols].dtypes

    # get fill-value:
    _fill_values = [dtype.fill_value for dtype in sparse_dtypes]
    if len(set(_fill_values)) != 1:
        raise ValueError(
            f"sparse columns have different `fill_values` ({_fill_values}) so cannot convert to a single sparse tensor. "
            f"Use a single fill_value, or pass `sparse_threshold=0`."
        )
    fill_value = _fill_values[0]

    # get dtype:
    common_dtype = find_common_type(
        [dtype.subtype for dtype in sparse_dtypes] + x[dense_cols].dtypes.tolist()
    )

    # convert dense cols to sparse:
    x = x.copy(deep=False)
    for col in dense_cols:
        x[col] = x[col].astype(pd.SparseDtype(common_dtype, fill_value))

    # convert to sparse array:
    return x.sparse.to_coo()


def to_tensor(x: Union[np.ndarray, torch.Tensor, pd.DataFrame],
              sparse_threshold: float = 0.,
              **kwargs) -> torch.Tensor:
    """
    Convert anything that ``np.asarray`` can handle into a tensor; additionally handle dataframes with sparse arrays.

    :param x: An array, tensor, dataframe, or anything that ``np.asarray`` can handle.
    :param sparse_threshold: If the input is a dataframe, then this determines whether the output tensor will be sparse
     (similar to the same argument in ``ColumnTransformer``). If the density is less than this value, then the output
     will be a sparse tensor.
    :param kwargs: Arguments to ``torch.as_tensor``.
    :return: A tensor.
    """
    # need special handling for sparsity:
    if sparse_threshold and isinstance(x, pd.DataFrame) and get_df_density(x) < sparse_threshold:
        coo = dataframe_to_coo(x)
        sparse_tensor = torch.sparse_coo_tensor(
            torch.as_tensor(np.vstack([coo.row, coo.col])),
            torch.as_tensor(coo.data, **kwargs),
            torch.Size(coo.shape),
            device=kwargs.get('device', 'cpu')
        )
        # https://github.com/pytorch/pytorch/issues/16187#issuecomment-457791600
        return sparse_tensor.coalesce()

    # otherwise, handling is simple.
    if not isinstance(x, torch.Tensor):
        x = np.asarray(x)
    return torch.as_tensor(x, **kwargs)


def get_to_kwargs(x) -> dict:
    """
    Get kwargs for ``Tensor.to(dtype=?, device=?)``
    :param x: A tensor or Module.
    :return: Dictionary of kwargs.
    """
    if isinstance(x, torch.nn.Module):
        return get_to_kwargs(next(iter(x.parameters())))
    return {'dtype': x.dtype, 'device': x.device}


def is_invalid(x: torch.Tensor, reduce: bool = True) -> bool:
    """ Checks for NaNs or infs in a tensor. Set reduce=False to return element-wise """
    if reduce:
        return torch.isinf(x).any() or torch.isnan(x).any()
    else:
        return torch.isinf(x) | torch.isnan(x)


class FitFailedException(RuntimeError):
    pass


class SliceDict(dict):
    """
    Adapted from https://github.com/skorch-dev/skorch/blob/baf0580/skorch/helper.py#L20
    """

    def __init__(self, **kwargs):
        kwargs = {k: self._standardize_val(v) for k, v in kwargs.items()}
        lengths = [value.shape[0] for value in kwargs.values() if hasattr(value, 'shape')]
        lengths_set = set(lengths)
        if lengths_set and (len(lengths_set) != 1):
            raise ValueError(
                "Initialized with items of different lengths: {}".format(', '.join(map(str, sorted(lengths_set))))
            )

        if not lengths:
            self._len = 0  # will be set later when we get our first item, see __setitem__
        else:
            self._len = lengths[0]

        super().__init__(**kwargs)

        # sklearn checks if it should use pandas indexing by checking if there's an iloc attribute
        if self.is_pandas:
            self.iloc = True

    @property
    def is_pandas(self) -> bool:
        is_pandas = [hasattr(v, 'iloc') for v in self.values() if hasattr(v, 'shape')]
        any_pandas = any(is_pandas)
        all_pandas = all(is_pandas)
        if any_pandas and not all_pandas:
            raise ValueError("Currenlty SliceDict does not support a mix of pandas and non-pandas")
        return any_pandas

    @staticmethod
    def _standardize_val(val):
        return np.asarray(val) if is_array(val) and not hasattr(val, 'shape') else val

    def __len__(self) -> int:
        return self._len

    def __array__(self) -> np.ndarray:
        array = self.get('__array__', None)
        if array is None:
            raise RuntimeError(
                f"Tried to convert a {type(self).__name__} to an array, which is ambiguous. If you'd like this to work,"
                f" you should add a key '__array__', which can have its value passed to np.asarray."
            )
        return np.asarray(array)

    def __getitem__(self, sl: Union[int, str, slice]) -> Union['SliceDict', ArrayType]:
        if isinstance(sl, int):
            raise ValueError(
                f"`{type(self).__name__}` can't be indexed with a single int; did you mean to wrap in a list?"
            )
        if isinstance(sl, str):
            return super(SliceDict, self).__getitem__(sl)
        cls = type(self)
        return cls(**{k: (v[sl] if hasattr(v, 'shape') else v) for k, v in self.items()})

    def __setitem__(self, key: str, value: ArrayType):
        value = self._standardize_val(value)
        if hasattr(value, 'shape'):
            val_len = value.shape[0]
            try:
                current_length = self._len
            except AttributeError:
                current_length = None  # special handling: unserializing
            if not current_length:
                self._len = val_len
            elif len(self) != val_len:
                raise ValueError(f"value must have .shape[0] of {len(self):,}, got {val_len:,}")

            if not isinstance(key, str):
                raise TypeError("Key must be str, not {}.".format(type(key)))

        elif hasattr(value, '__len__'):
            raise ValueError(
                f"Values inserted into `{type(self).__name__}` must be arrays (i.e. have a `shape`) or must be "
                f"dimensionless (e.g. `bool`). This value is not an array but has a length."
            )

        super().__setitem__(key, value)

        # sklearn checks if it should use pandas indexing by checking if there's an iloc attribute
        if self.is_pandas:
            self.__dict__['iloc'] = True
        else:
            self.__dict__.pop('iloc', None)

    def take(self, indices, axis: int = 0, **kwargs) -> 'SliceDict':
        if axis:
            raise ValueError("Only axis=0 is supported")
        cls = type(self)
        return cls(**{k: (v.take(indices, axis, **kwargs) if hasattr(v, 'shape') else v) for k, v in self.items()})

    def update(self, kwargs: dict):
        for key, value in kwargs.items():
            self.__setitem__(key, value)

    def __repr__(self):
        out = super().__repr__()
        return "SliceDict(**{})".format(out)

    @property
    def shape(self):
        return (self._len,)

    def copy(self) -> 'SliceDict':
        return type(self)(**self)

    def fromkeys(self, *args, **kwargs):
        """fromkeys method makes no sense with SliceDict and is thus not supported."""
        raise TypeError("SliceDict does not support fromkeys.")

    def __eq__(self, other) -> bool:
        raise NotImplementedError


class ToSliceDict(TransformerMixin, BaseEstimator):
    """
    Many distributions have multiple parameters: for example the normal distribution has a location and scale
    parameter. We configure which predictors should handle which dist-params in :class:`foundry.glm.Glm` by passing
    a dictionary for X, with keys being arguments to the distribution, and values being model-matrices. This class
    enables us to configure this behavior within a pipeline, for example:

    >>> make_pipeline(ToSliceDict(['loc','scale']), Glm(family='gaussian'))

    :class:`foundry.glm.Glm` also supports this specification via the ``col_mapping`` argument.

    :param mapping: This can be either a list or a dictionary. For example, ``mapping=['loc','scale']`` will create a
     dict with the full model-matrix assigned to both params. A dictionary allows finer-grained control, e.g.:
     ``mapping={'loc':[col1,col2,col3],'scale':[col1]}``. Finally, instead of the dict-values being lists of columns,
     these can be functions that takes the data and return the relevant columns: e.g.
     ``mapping={'loc':sklearn.compose.make_column_selector('^col+.'), 'scale':[col1]}``.
    """

    def __init__(self, mapping: Union[list, dict, None]):
        self.mapping = mapping

    def transform(self, X: ModelMatrix, extras_ok: bool = True) -> SliceDict:
        if not isinstance(self.mapping, dict):
            raise RuntimeError("This instance is not fitted yet.")

        if isinstance(X, dict):
            X = X.copy()  # consistent behavior

            # if it's already a dict, not much transforming to do, but need to validate
            keys_ok = set(self.mapping).issubset(X) if extras_ok else (set(X) == set(self.mapping))
            if not keys_ok:
                raise RuntimeError(
                    f"`mapping` from ``ToSliceDict(mapping=)`` is `{self.mapping}`, but ``X.keys()`` is `{set(X)}`"
                )

            for k, selection in self.mapping.items():
                actual = _get_col_names_or_idx(X[k])
                expected = _get_col_names_or_idx(_select_col_names_or_idx(X[k], selection))
                if actual != expected:
                    raise RuntimeError(f"For {k}, expected cols {expected} but got {actual}")

        else:
            X = {k: _select_col_names_or_idx(X, selection) for k, selection in self.mapping.items()}

        if not isinstance(X, SliceDict):
            X = SliceDict(**X)
        return X

    def fit(self, X: ModelMatrix, y: np.ndarray = None) -> 'ToSliceDict':
        if self.mapping is None:
            if isinstance(X, dict):
                self.mapping = list(X)
            else:
                raise RuntimeError(f"Can only set mapping to None if X is a dict, but got {type(X)}")

        if isinstance(X, dict) and set(self.mapping) != set(X):
            raise RuntimeError(
                f"`mapping` from ``ToSliceDict(mapping=)`` is `{self.mapping}`, but ``X.keys()`` is `{set(X)}`"
            )

        if isinstance(self.mapping, (list, tuple)):
            if isinstance(X, dict):
                self.mapping = {k: _get_col_names_or_idx(thisX) for k, thisX in X.items()}
            else:
                self.mapping = {k: _get_col_names_or_idx(X) for k in self.mapping}

        if not isinstance(self.mapping, dict):
            raise ValueError("`mapping` must be a list, tuple or dict.")

        remainder_name = None
        for k in list(self.mapping):
            thisX = X[k] if isinstance(X, dict) else X
            if callable(self.mapping[k]):
                self.mapping[k] = list(self.mapping[k](thisX))
                if not self.mapping[k]:
                    raise RuntimeError(f"self.mapping['{k}'] is a callable that returned no columns.")
            elif isinstance(self.mapping[k], str):
                if self.mapping[k] == 'remainder':
                    if remainder_name is not None:
                        raise RuntimeError("Only one element of mapping can be 'remainder'.")
                    remainder_name = k
                else:
                    raise ValueError(f"mapping (for {k}) is a string not a list-- did you mean to wrap in []?")
            else:
                # will raise if missing
                _select_col_names_or_idx(thisX, self.mapping[k])

        if remainder_name:
            if isinstance(X, dict):
                raise RuntimeError("Cannot set any values of `mapping` to 'remainder' if X is a dict.")
            # one and only one of the selections can be the literal 'remainder': everything not selected by the others
            remainder = set(X.columns)
            for k, selection in self.mapping.items():
                if k == remainder_name:
                    continue
                remainder = remainder - set(selection)
            self.mapping[remainder_name] = list(remainder)

        return self


def _get_col_names_or_idx(x: Union[pd.DataFrame, np.ndarray]) -> list:
    if len(x.shape) != 2:
        raise ValueError(f"Expected 2d array, got shape=`{x.shape}`.")
    if hasattr(x, 'columns'):
        return list(x.columns)
    else:
        return list(range(x.shape[1]))


def _select_col_names_or_idx(x: Union[pd.DataFrame, np.ndarray], selection: list) -> Union[pd.DataFrame, np.ndarray]:
    if hasattr(x, 'columns'):
        all_cols = set(x.columns)
        extra = set(selection) - all_cols
        if extra:
            msg = f"The following columns are not present:{extra}."
            if len(all_cols) < 25:
                msg += f"\nAvailable cols:{all_cols}"
            raise ValueError(msg)
        return x.loc[:, selection]
    else:
        return x[:, selection]


def to_1d(arr: ArrayType) -> ArrayType:
    """
    - If 1d, return unchanged.
    - If 2d-nd, raise unless all but first dim is singleton.

    :param arr: Numpy array or tensor.
    :return: 1d of same type
    """
    if len(arr.shape) == 1:
        return arr
    arr = arr.squeeze()
    if len(arr.shape) > 1:
        # unable to squeeze
        raise ValueError("Unable to squeeze array to 1d.")
    if not arr.shape:
        # squeezed to scalar
        arr = arr[None, ...]
    return arr


def to_2d(arr: ArrayType) -> ArrayType:
    """
    - If 1d, unsqueeze 2nd dim.
    - If 2d, return unchanged
    - If 3d, raise unless all but first two dims are singleton.

    :param arr: Numpy array or tensor.
    :return: 2d of same type
    """
    # TODO what about pandas? Slicing arr[None, ...] not officially supported.

    ndim = len(arr.shape)
    if ndim == 2:
        return arr
    if ndim > 2:
        for _ in range(2, ndim + 1):
            arr = arr.squeeze(-1)  # this will fail if size > 1
        if len(arr.shape) > 2:
            raise ValueError("Unable to squeeze array to 2d.")
    else:
        if len(arr.shape) == 0:
            arr = arr[None, ...]
        if len(arr.shape) == 1:
            arr = arr[..., None]
    return arr


def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """
    Implement a numerically stable ``log(1 - exp(x))``, as described in
    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    out_dtype = x.dtype if torch.is_tensor(x) and torch.is_floating_point(x) else torch.get_default_dtype()
    x = torch.as_tensor(x, dtype=torch.double)
    out = torch.empty_like(x)
    mask = -x < 0.693
    out[mask] = torch.log(-torch.expm1(x[mask]))
    out[~mask] = torch.log1p(-torch.exp(x[~mask]))
    return out.to(dtype=out_dtype)
