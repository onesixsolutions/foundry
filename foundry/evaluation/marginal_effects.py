from typing import Collection, Dict, Iterable, List, Optional, Tuple
from warnings import warn

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from typing import Callable, Sequence, Union

from sklearn.utils import _safe_indexing
from tqdm.auto import tqdm

from foundry.util import to_1d


class Binned:
    def __init__(self, col: str, bins: Union[int, Sequence] = 20, **kwargs):
        """
        This class creates an object which can bin a pandas.Series.
        ```
        binner = Binned("my_feature", bins=20)
        binned: pd.Series = binner(model_matrix)
        ```
        leaves `binned` as a pd.Series that's been cut into 20 categoricals.

        Other options:
        If `bins` is iterable, then use `pd.cut`, and the elements in `bins` are the
        edges of the cut
        If `bins` is anything else, then return the column, uncut.
        """
        self.orig_name = col
        self.bins = bins
        self.kwargs = kwargs

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        if isinstance(df[self.orig_name].dtype, pd.CategoricalDtype):
            return df[self.orig_name]

        if hasattr(self.bins, '__iter__'):
            return pd.cut(df[self.orig_name], self.bins, **self.kwargs)
        elif self.bins:
            return pd.qcut(df[self.orig_name], self.bins, duplicates='drop', **self.kwargs)
        else:
            return df[self.orig_name]


def binned(col: str, bins: Union[int, Sequence] = 20, **kwargs) -> Binned:
    """
    A concise way of indicating to ``MarginalEffects`` cls: "bin the result by 'col'".

    :param col: The column to bin
    :param bins: Either (a) number of bins, passed to ``pandas.qcut``, or (b) the cuts, passed to ``pandas.cut``
    :param kwargs: More arguments to the ``{q}cut`` function
    :return: A function that takes a dataframe and returns the binned version of the column.
    """

    return Binned(col=col, bins=bins, **kwargs)


def raw(col: str) -> Callable:
    """
    If you pass a raw column-name to ``MarginalEffects``, it'll by default get wrapped in ``binned``. Sometimes you
    really want the raw value, so wrap it in this.

    :param col: The column-name.
    :return: A function that takes a dataframe and returns the column as-is.
    """

    return Binned(col=col, bins=False)


class MarginalEffects:
    def __init__(self, pipeline: Pipeline, predict_method: Optional[str] = None, quiet: bool = False):
        self.pipeline = pipeline
        if predict_method is not None:
            warn(
                "Passing `predict_method` to `MarginalEffects` is deprecated, pass to `__call__` instead.",
                DeprecationWarning
            )
        self.predict_method = predict_method
        self._dataframe = None
        self.config = None
        self.quiet = quiet

    @property
    def feature_names_in(self) -> Sequence[str]:
        """
        The actual features passed to the ColumnTransformer.
        """
        col_transformer: ColumnTransformer = self.pipeline[0]
        if not hasattr(col_transformer, '_columns'):
            raise TypeError("``self.pipeline[0]`` does not have ``_columns``. Is it a ``ColumnTransformer``?")
        if col_transformer.remainder != 'drop':
            raise NotImplementedError("Not currently implemented if ``col_transformer.remainder != 'drop'``")
        feature_names_in = set()
        for cols in col_transformer._columns:
            if isinstance(cols, str):
                cols = [cols]
            for col in cols:
                feature_names_in.add(col)
        return list(feature_names_in)

    @property
    def all_column_names_in(self) -> Sequence[str]:
        """
        The original columns passed to the ColumnTransformer.
        """
        col_transformer: ColumnTransformer = self.pipeline[0]
        if not hasattr(col_transformer, 'feature_names_in_'):
            raise TypeError("``self.pipeline[0]`` does not have ``feature_names_in_``. Is it a ``ColumnTransformer``?")
        return col_transformer.feature_names_in_

    def __call__(self,
                 X: pd.DataFrame,
                 y: Optional[np.ndarray],
                 vary_features: Sequence[Union[str, Binned]],
                 groupby_features: Collection[Union[str, Binned]] = (),
                 vary_features_aggfun: Union[str, dict, Callable] = 'mean',
                 marginalize_aggfun: Union[str, dict, Callable, None] = 'downsample100000',
                 y_aggfun: Union[str, Callable] = 'mean',
                 predict_method: Optional[str] = None,
                 **predict_kwargs) -> 'MarginalEffects':
        """
        Prepare a dataframe/plot showing how predictions vary when one or more features are varied, holding
        all others at a constant value.

        :param X: The data with predictors to be passed to the pipeline.
        :param y: The target. Pass `None` if you don't want to include "actuals", only predictions.
        :param vary_features: Strings indicating the feature(s) to vary, so as to observe their independent effect in
         the model. By default will be binned by passing to :function:`foundry.evaluation.binned`. You can pass to this
         function yourself to manually control/remove binning.
        :param groupby_features: Strings indicating the feature(s) to group/segment on, so as to observe different
         effects per segment. By default will be binned by passing to :function:`foundry.evaluation.binned`. You can
         pass to this function yourself to manually control/remove binning.
        :param vary_features_aggfun: Numeric varying feature(s) will be binned, then within each bin we need to convert
         back to numeric before plugging into the model. This string indicates how to do so (default: mean). Either an
         aggregation that will be applied, or 'mid' to use the midpoint of the bin. The latter will be used regardless
         when no actual data exists in that bin. This argument can also be a dictionary with keys being feature-names.
        :param marginalize_aggfun: How to marginalize across features that are not in the vary/groupby. This is a
         string/callable that will be applied to each feature. Can also be 'downsample{int}' or false-y, in which case
         will *not* collapse these, but instead call ``predict`` for each row then summarize the *predictions* with
         `y_aggfun`. If 'downsample{int}' then downsampling will be performed before this (slow) procedure.
        :param y_aggfun: The function to use when aggregating predictions (and actuals, if supplied). Only has an
         effect if ``marginalize_aggfun` is False/None.
        :param predict_kwargs: Keyword-arguments to pass to the pipeline's ``predict`` method.
        """
        X = X.copy(deep=False)
        if isinstance(marginalize_aggfun, str) and marginalize_aggfun.startswith('downsample'):
            downsample_int = int(marginalize_aggfun.replace('downsample', '').rstrip('_'))
            if X.shape[0] > downsample_int:
                idx = np.random.choice(X.shape[0], size=downsample_int, replace=False)
                X = _safe_indexing(X, idx)
                if y is not None:
                    y = _safe_indexing(y, idx)
            marginalize_aggfun = False

        # validate/standardize args:
        vary_features = self._standardize_maybe_binned(X, vary_features)
        check_set_membership(vary_features, self.feature_names_in)
        groupby_features = self._standardize_maybe_binned(X, groupby_features)
        assert set(vary_features).isdisjoint(set(groupby_features))

        # no-vary features ----
        if groupby_features:
            groupby_colnames = []
            for old_fname, new_fname, values in self._get_maybe_binned_features(X, groupby_features):
                groupby_colnames.append(new_fname)
                if old_fname == new_fname:
                    continue
                X[new_fname] = values
        else:
            X['_dummy'] = ''
            groupby_colnames = ['_dummy']

        df_no_vary = self._get_df_novary(
            X=X,
            marginalize_aggfun=marginalize_aggfun,
            groupby_colnames=groupby_colnames,
            colnames_to_agg=list(set(self.feature_names_in) - set(vary_features))
        )

        # vary features ----
        default = vary_features_aggfun.pop('_default', 'mean') if isinstance(vary_features_aggfun, dict) else 'mean'
        vary_features_aggfuns = self._standardize_maybe_dict(
            maybe_dict=vary_features_aggfun,
            keys=vary_features,
            default=default
        )
        # get mappings of binned(vary_feature) -> agg(vary_feature)
        # the former is needed so we don't drop bins with no actual observations
        vary_features_mappings = {}
        for old_fname, new_fname, values in self._get_maybe_binned_features(X, vary_features):
            if old_fname == new_fname:
                vary_features_mappings[new_fname] = None
            else:
                X[new_fname] = values
                vary_features_mappings[new_fname] = self._get_binned_feature_map(
                    X=X,
                    binned_fname=new_fname,
                    fname=old_fname,
                    aggfun=vary_features_aggfuns[old_fname]
                )

        # get a grid of vary feature-bins, including data on 'actual' if available
        agg_kwargs = {'n': (groupby_colnames[0], 'count')}
        if y is not None:
            X['_actual'] = y
            agg_kwargs['actual'] = ('_actual', y_aggfun)
            agg_kwargs['actual_lower'] = ('_actual', lambda s: s.quantile(.25, interpolation='lower'))
            agg_kwargs['actual_upper'] = ('_actual', lambda s: s.quantile(.75, interpolation='higher'))
        df_vary_grid = (X
                        .groupby(groupby_colnames + list(vary_features_mappings.keys()), observed=False)
                        .agg(**agg_kwargs)
                        .reset_index())
        if 'actual' in df_vary_grid.columns:  # handle binary targets with low-n
            for col in ('actual', 'actual_lower', 'actual_upper'):
                df_vary_grid[col] = df_vary_grid[col].astype('float')

        # map the binned version of vary features back to their aggregated values:
        for binned_fname, df_mapping in vary_features_mappings.items():
            if df_mapping is None:  # binned_fname==fname, i.e. feature was already categorical
                continue
            df_vary_grid = df_vary_grid.merge(df_mapping, on=binned_fname)

        predict_method = predict_method or self.predict_method

        self.config = {'pred_colnames': []}
        if marginalize_aggfun:
            df_me_base = df_vary_grid.merge(df_no_vary, how='left', on=groupby_colnames)
            df_me = self.add_predictions(df_me_base, predict_method=predict_method, **predict_kwargs)
            self.config['pred_colnames'].extend(set(df_me.columns) - set(df_me_base.columns))
        else:
            pred_colnames = set()
            if df_no_vary.shape[0] > 100_000 and not self.quiet:
                print("Consider setting `marginalize_aggfun='downsample100000'`.")

            chunks = [df for _, df in df_vary_grid.groupby(list(vary_features), sort=False, observed=False)]
            if not self.quiet:
                chunks = tqdm(chunks, delay=10)

            df_me = []
            for _df_vary_chunk in chunks:
                _df_merged_base = _df_vary_chunk.merge(df_no_vary, how='left', on=groupby_colnames)
                _df_merged = self.add_predictions(_df_merged_base, predict_method=predict_method, **predict_kwargs)
                pred_colnames.update(set(_df_merged.columns) - set(_df_merged_base.columns))

                _df_collapsed = (_df_merged
                                 .groupby(groupby_colnames + list(vary_features), observed=False)
                                 .agg(**{k: (k, y_aggfun) for k in pred_colnames})
                                 .reset_index())

                # pandas silently drops nan groupbys, so even though one-row-per-actual, can't groupby actual
                df_me.append(_df_collapsed.merge(_df_vary_chunk, on=groupby_colnames + list(vary_features)))

            df_me = pd.concat(df_me).reset_index(drop=True)
            self.config['pred_colnames'] = list(pred_colnames)

        df_me['n'] = df_me['n'].fillna(0)
        if '_dummy' in df_me.columns:
            del df_me['_dummy']

        self._dataframe = df_me
        self.config.update({
            'vary_features': vary_features,
            'groupby_features': groupby_features
        })
        return self

    fit = __call__

    def to_dataframe(self) -> pd.DataFrame:
        if self._dataframe is None:
            raise RuntimeError("This `MarginalEffects()` needs to be called on a dataset first.")
        return self._dataframe.copy()

    def plot(self,
             x: Optional[str] = None,
             color: Optional[str] = None,
             facets: Optional[Sequence[str]] = None,
             include_actuals: Optional[bool] = None,
             line_alpha: Optional[float] = None) -> 'ggplot':

        try:
            from plotnine import (
                ggplot, aes, geom_line, geom_hline, facet_wrap, theme, theme_bw, geom_point, ylab, geom_col
            )
        except ImportError as e:
            raise RuntimeError("plotting requires `plotnine` package") from e
        if isinstance(facets, str):
            facets = [facets]
        facets = list(facets or [])

        data = self.to_dataframe()

        # available default features:
        available_default_features = list(self.config['vary_features']) + list(self.config['groupby_features'])

        group_by_prediction_col = False
        if len(self.config['pred_colnames']) > 1:
            assert 'prediction_col' not in data.columns
            data = data.melt(
                id_vars=[c for c in data if c not in self.config['pred_colnames']],
                var_name='prediction_col',
                value_name='predicted',
                value_vars=self.config['pred_colnames']
            )
            if (data['prediction_col'].nunique() > 9) and ('class' not in self.config['pred_colnames'][0]):
                # a common use-case for multiple prediction columns is predict_posterior, where we have 100s of
                # predictions and it doesn't make sense to map them to color/facets
                group_by_prediction_col = True
                warn(
                    "Many prediction columns, will map to group but not color/facets; you can override this by setting "
                    "color='prediction_col' or facets='prediction_col'"
                )
            else:
                available_default_features.append('prediction_col')

        # if x is set, that's not an available default feature:
        if x:
            for colname in [x, x.replace('_binned', '')]:
                if colname in available_default_features:
                    available_default_features.remove(colname)
        # if color (or its binned version) is set, that's not an available default feature:
        if color:
            for colname in [color, color.replace('_binned', '')]:
                if colname in available_default_features:
                    available_default_features.remove(colname)
        # facets (or their binned versions) that are set are not available default features:
        for facet in facets:
            for colname in [facet, facet.replace('_binned', '')]:
                if colname in available_default_features:
                    available_default_features.remove(colname)

        aes_kwargs = {'y': 'predicted'}

        # x:
        if not x:
            x = available_default_features.pop(0)
        aes_kwargs['x'] = x

        # for color/facets use binned versions if available:
        for i, feat in enumerate(list(available_default_features)):
            if feat + '_binned' in self._dataframe.columns:
                available_default_features[i] += '_binned'

        # color:
        if not color and available_default_features:
            color = available_default_features.pop(0)
        if color:
            aes_kwargs['group'] = aes_kwargs['color'] = color
        else:
            aes_kwargs['group'] = '1'

        # remaining go to facets:
        if available_default_features:
            if not facets:
                facets = list(available_default_features)
            else:
                warn(f"Adding {available_default_features} to `facets`")
                facets.extend(available_default_features)

        if group_by_prediction_col:
            if aes_kwargs['group'] == '1':
                aes_kwargs['group'] = 'prediction_col'
            else:
                data['_group'] = data[aes_kwargs['group']].astype('str') + data['prediction_col'].astype('str')
                aes_kwargs['group'] = '_group'

        plot = (
                ggplot(data, aes(**aes_kwargs)) +
                geom_hline(yintercept=0) +
                theme_bw() +
                theme(figure_size=(8, 6), subplots_adjust={'wspace': 0.10})
        )
        if isinstance(data[x.replace('_binned', '')].dtype, pd.CategoricalDtype):
            if group_by_prediction_col:
                raise RuntimeError("Please explicitly map 'prediction_col' to color or facet.")
            plot += geom_col()
        else:
            if line_alpha is None:
                line_alpha = max(
                    1 / (data[aes_kwargs['group']].nunique() if aes_kwargs['group'] in data.columns else 1),
                    .01
                )
            plot += geom_line(alpha=line_alpha)

        if include_actuals is None:
            include_actuals = 'actual' in data.columns
        if include_actuals:
            if 'actual' in data.columns:
                plot += geom_point(aes(y='actual', size='n'))
                plot += ylab("predicted (line) & actual (dots)")
            else:
                raise RuntimeError("`y` was not passed so cannot `include_actuals`")
        if facets:
            plot += facet_wrap(facets, scales='free_y', labeller='label_both')
        return plot

    @staticmethod
    def _get_maybe_binned_features(X: pd.DataFrame,
                                   features: Dict[str, Binned]) -> Iterable[Tuple[str, str, Optional[pd.Series]]]:
        for fname, bin_fun in features.items():
            binned_feature = bin_fun(X)
            if binned_feature is X[fname]:
                # `bin_fun` will be a no-op if the user passed ``raw(feature)`` or if they passed
                # the feature and it's categorical.
                # TODO: test this works!
                if not isinstance(binned_feature.dtype, pd.CategoricalDtype):
                    warn(f"{fname} is not categorical, values not present in the data will be dropped.")
                binned_fname = fname
            else:
                binned_fname = f'{fname}_binned'
            yield fname, binned_fname, binned_feature

    def add_predictions(self, X: pd.DataFrame, predict_method: Optional[str], **kwargs) -> pd.DataFrame:
        Xorig = X
        # avoid warning from ColumnTransformer:
        X = X.reindex(columns=self.all_column_names_in)

        # pipeline doesn't let us forward any methods aside from predict{_proba}, so we split:
        prep_steps_, estimator_ = self.pipeline[0:-1], self.pipeline[-1]

        # default behavior: use proba if it's available
        if predict_method is None:
            if hasattr(estimator_, 'predict_proba'):
                predict_method = 'predict_proba'
            else:
                predict_method = 'predict'
        predfun = getattr(estimator_, predict_method)
        Xt = prep_steps_.transform(X)
        predictions = predfun(Xt, **kwargs)

        # validate output:
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            assert len(predictions.shape) == 2
            if predict_method == 'predict_proba' and predictions.shape[1] == 2:
                # handle common case of 2-class prediction, only plot p(positive-class)
                predictions = predictions[:, 1]
            elif not isinstance(predictions, pd.DataFrame):
                predictions = pd.DataFrame(predictions)
                if predict_method == 'predict_proba':
                    predictions.columns = [f'class{i}' for i in range(predictions.shape[1])]
                else:
                    predictions.columns = [f'pred{i}' for i in range(predictions.shape[1])]

        # handle single or multi-output:
        if not isinstance(predictions, pd.DataFrame):
            predictions = pd.DataFrame({'predicted': to_1d(predictions)})

        predictions.index = Xorig.index
        return Xorig.join(predictions)

    @staticmethod
    def _standardize_maybe_binned(data: pd.DataFrame, features: Collection[Union[str, Binned]]) -> Dict[str, Binned]:
        """
        Take a collection of items that may either strings or ``Binned`` instances. Convert any strings to ``Binned``
        instances, return a dictionary keyed by the original feature-names.
        """
        if isinstance(features, str):
            features = [features]
        out = {}
        for maybe_binned in features:
            if hasattr(maybe_binned, 'orig_name'):
                deffy_binned = maybe_binned
            elif not isinstance(maybe_binned, str):
                raise ValueError(f"Expected {maybe_binned} to be a string or wrapped in ``binned``.")
            elif isinstance(data[maybe_binned].dtype, pd.CategoricalDtype):
                deffy_binned = Binned(maybe_binned, bins=False)
            elif pd.api.types.is_numeric_dtype(data[maybe_binned]):
                deffy_binned = Binned(maybe_binned)
            else:
                raise ValueError(f"{maybe_binned} is neither categorical nor numeric; unable bin or keep as-is.")
            out[deffy_binned.orig_name] = deffy_binned
        return out

    @staticmethod
    def _standardize_maybe_dict(maybe_dict: any, keys: Collection[str], default: any) -> dict:
        """
        Sometimes we want to allow either of (a) ``arg='median'`` applies median to each feature,
        (b) ``{'f':'median'}`` applies median to feature 'f'. This function standardizes into (b).

        :param maybe_dict: Maybe a dictionary, or maybe the value for all dictionary items.
        :param keys: The keys that should be in the dictionary. Any not present will be added with value `default`.
        :param default: See `keys`.
        :return: A dictionary.
        """
        if isinstance(maybe_dict, dict):
            deffy_dict = maybe_dict.copy()
            for key in keys:
                if key not in deffy_dict:
                    deffy_dict[key] = default
            extra = set(deffy_dict) - set(keys)
            if extra:
                warn(f"Unexpected keys: `{extra}`")
        else:
            deffy_dict = {key: maybe_dict for key in keys}
        return deffy_dict

    @staticmethod
    def _get_binned_feature_map(X: pd.DataFrame,
                                binned_fname: str,
                                fname: str,
                                aggfun: Union[str, Callable]) -> pd.DataFrame:
        """
        Get a dataframe that maps the binned version of a feature to the aggregates of its original values.
        """
        assert binned_fname != fname

        if aggfun == 'mid':
            # creates a df with unique values of `binned_fname` and `nans` for `fname`.
            # this will then get filled with the midpoint below:
            # todo: less hacky way to do this
            df_mapping = X.groupby(binned_fname, observed=False)[fname].agg('count').reset_index()
            df_mapping[fname] = float('nan')
        else:
            df_mapping = X.groupby(binned_fname, observed=False)[fname].agg(aggfun).reset_index()

        # for any bins that aren't actually observed, use the midpoint:
        midpoints = pd.Series([x.mid for x in df_mapping[binned_fname]])
        if np.isinf(midpoints).any() and df_mapping[fname].isnull().any():
            raise ValueError(
                f"[{fname}] `inf` bin cuts cannot be used when no data present in the bin:"
                f"{df_mapping[binned_fname][np.isinf(midpoints)]}"
            )
        df_mapping[fname] = df_mapping[fname].fillna(midpoints)
        return df_mapping

    def _get_df_novary(self,
                       X: pd.DataFrame,
                       marginalize_aggfun: Union[str, dict, Callable, None],
                       groupby_colnames: Sequence[str],
                       colnames_to_agg: Sequence[str]) -> pd.DataFrame:
        """
        Get a dataframe with all the features except for `vary_features`. These are collapsed to the groups defined by
        groupby_colnames.
        """
        if marginalize_aggfun:
            # TODO: default needs to depend on dtype
            default = marginalize_aggfun.pop('_default', 'median') if isinstance(marginalize_aggfun, dict) else 'median'
            marginalize_aggfuns = self._standardize_maybe_dict(
                maybe_dict=marginalize_aggfun,
                keys=colnames_to_agg,
                default=default
            )
            single_cols = []
            agg_kwargs = {}
            for feature in colnames_to_agg:
                # TODO: if `feature` is single or groupby, we need to watch out for nulls. how best to do this?
                if feature in groupby_colnames:
                    # groupby colnames are usually the binned version of a subset of marginalize_features.
                    # but sometimes (e.g. was already categorical so no binning needed) the groupby is all we need
                    continue
                if X[feature].nunique() <= 1:
                    single_cols.append(feature)
                else:
                    agg_kwargs[feature] = (feature, marginalize_aggfuns[feature])
            if not agg_kwargs:
                raise NotImplementedError("TODO")

            # collapse:
            df_no_vary = X.groupby(list(groupby_colnames) + single_cols, observed=False).agg(**agg_kwargs).reset_index()
            # validate:
            if (df_no_vary.groupby(groupby_colnames, observed=False).size() > 1).any():
                for col in colnames_to_agg:
                    if (df_no_vary.groupby(groupby_colnames, observed=False)[col].nunique() > 1).any():
                        raise ValueError(f"The aggfun {agg_kwargs[col]} did not result in 1 value per group for {col}")
                # shouldn't generally get here:
                raise ValueError(f"The ``marginalize_aggfun`` did not result in one row per group:\n{df_no_vary}")
        else:
            df_no_vary = X.loc[:, list(set(list(groupby_colnames) + list(colnames_to_agg)))]
        return df_no_vary


def check_set_membership(maybe_members: Collection, full_set: Collection):
    extras = set(maybe_members) - set(full_set)
    if extras:
        raise ValueError(f"Got unexpected values: {extras}. Expected one/more of: {full_set}")
