from typing import Union, Sequence

from sklearn.base import TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline

from .backports import FunctionTransformer


def as_transformer(
        x: Union[TransformerMixin, Sequence[TransformerMixin], callable, None],
        **kwargs
) -> Union[TransformerMixin, Pipeline]:
    """
    Standardize a transformer, function, or list of these into a transformer. Lists are converted to sklearn.pipelines.
    """
    if callable(x):
        return FunctionTransformer(x, kw_args=kwargs)

    if kwargs:
        raise ValueError("`kwargs` can only be passed if x is a function.")

    if x is None:
        return FunctionTransformer()
    if hasattr(x, '__iter__') and not isinstance(x, str):
        return make_pipeline(*[as_transformer(xi) for xi in x])
    if hasattr(x, 'transform'):
        return x
    else:
        raise TypeError(f"{type(x).__name__} does not have a `transform()` method.")
