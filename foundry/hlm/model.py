from typing import Union, Sequence, Optional, Dict, Tuple, Callable
from warnings import warn

import numpy as np
import pandas as pd
import torch

from foundry.covariance import log_chol_to_chol
from foundry.glm import Glm
from foundry.glm.family import Family
from foundry.hlm.solver import ReSolver, get_solver_cls
from foundry.util import ModelMatrix, SliceDict, get_to_kwargs, to_tensor

# (group-levels, random-effects with one row per level)
ReTable = Tuple[pd.Index, torch.Tensor]


class Hlm(Glm):
    """
    A hierarchical (aka mixed-effects) GLM: a :class:`foundry.glm.Glm` whose first distribution-parameter (e.g.
    ``loc``, ``probs``) additionally gets random-effects for each level of a grouping factor. The fixed-effects and the
    random-effects covariance are estimated by maximizing the (laplace-approximated) marginal likelihood.

    :param family: A family name; see ``Hlm.family_names``.
    :param raneff_design: A dictionary whose key is the name of the grouping factor, and whose value is a list of
     predictors with random-slopes (a random-intercept is always included, so ``[]`` means intercept-only). Instead of
     a list of column-names, can pass a callable that takes the dataframe and returns column-names (e.g.
     ``sklearn.compose.make_column_selector``). The grouping factor can be a column in ``X``, or (if ``X`` is a dict)
     an entry in ``X``. Currently only one grouping factor is supported.
    :param penalty: Penalty on the fixed-effects, see :class:`foundry.glm.Glm`. Random-effects are regularized by their
     estimated covariance, so are not penalized.
    :param col_mapping: See :class:`foundry.glm.Glm`. The default is that all columns (except the grouping factor)
     are fixed-effects for the first distribution-parameter.
    :param unknown_groups: How to predict for groups whose random-effects are unknown (not in the training data or in
     ``group_data``). 'zero' sets random-effects to zero; 'raise' raises an error.
    :param solver_kwargs: Keyword-arguments for the :class:`foundry.hlm.solver.ReSolver`.
    :param sparse_mm_threshold: See :class:`foundry.glm.Glm`.
    """

    def __init__(self,
                 family: Union[str, Family],
                 raneff_design: Dict[str, Union[Sequence[str], Callable]],
                 penalty: Union[float, Sequence[float], Dict[str, float]] = 0.,
                 col_mapping: Union[list, dict, None] = None,
                 unknown_groups: str = 'zero',
                 solver_kwargs: Optional[dict] = None,
                 sparse_mm_threshold: float = 0.0,
                 _warm_start: Optional[dict] = None):
        self.raneff_design = raneff_design
        self.unknown_groups = unknown_groups
        self.solver_kwargs = solver_kwargs
        super().__init__(
            family=family,
            penalty=penalty,
            col_mapping=col_mapping,
            sparse_mm_threshold=sparse_mm_threshold,
            _warm_start=_warm_start
        )
        # set in _init_module:
        self.raneff_design_ = None
        self.raneff_cov_module_ = None
        self.solver_ = None
        # set at the end of fit:
        self.raneff_table_ = None

    @property
    def grouping_factor(self) -> str:
        if len(self.raneff_design) != 1:
            raise NotImplementedError("Currently only one grouping factor is supported.")
        return list(self.raneff_design)[0]

    # fitting ---------
    def fit(self,
            X: ModelMatrix,
            y: ModelMatrix,
            sample_weight: Optional[np.ndarray] = None,
            cv_kwargs: Optional[dict] = None,
            **kwargs) -> 'Hlm':
        """
        See :func:`foundry.glm.Glm.fit`. If ``penalty`` is a list, the default cross-validation splits by group (i.e.
        ``GroupKFold``).
        """
        if isinstance(self.penalty, (list, tuple)):
            cv_kwargs = dict(cv_kwargs or {})
            if 'cv' not in cv_kwargs:
                from sklearn.model_selection import GroupKFold
                _, groups, _ = self._split_X(X)
                cv_kwargs['cv'] = list(GroupKFold(n_splits=5).split(groups, groups=groups))
        return super().fit(X=X, y=y, sample_weight=sample_weight, cv_kwargs=cv_kwargs, **kwargs)

    def _fit(self, X: ModelMatrix, y: ModelMatrix, sample_weight: Optional[np.ndarray] = None, **kwargs) -> 'Hlm':
        self._res_warm_start = {}
        try:
            super()._fit(X=X, y=y, sample_weight=sample_weight, **kwargs)
        finally:
            self._res_warm_start = {}
        self.raneff_table_ = self._solve_raneffs(X=X, y=y, sample_weight=sample_weight)
        return self

    def _init_module(self, X: ModelMatrix, y: ModelMatrix, verbose: bool = False):
        self.grouping_factor  # validate
        self.raneff_design_ = self._standardize_design(X)
        self.solver_ = get_solver_cls(self.family)(**(self.solver_kwargs or {}))
        self.raneff_cov_module_ = torch.nn.ModuleDict({
            gf: _CholeskyCov(rank=len(cols) + 1) for gf, cols in self.raneff_design_.items()
        })
        X_glm, *_ = self._split_X(X)
        super()._init_module(X_glm, y, verbose=verbose)

    def _init_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.LBFGS(
            list(self.module_.parameters()) + list(self.raneff_cov_module_.parameters()),
            max_iter=10, line_search_fn='strong_wolfe', lr=.25
        )

    def get_log_prob(self,
                     x_dict: dict,
                     lp_dict: dict,
                     mean: bool = True,
                     include_penalty: bool = True) -> torch.Tensor:
        """
        Get the penalized, laplace-approximated, marginal log prob.
        """
        x_dict = dict(x_dict)
        Z, labels = x_dict.pop('_re_Z'), x_dict.pop('_re_labels')
        levels, group_idx = _factorize(labels)
        family_kwargs = self._get_family_kwargs(**x_dict)

        cache_key = id(lp_dict)
        res_init = getattr(self, '_res_warm_start', {}).get(cache_key)
        if res_init is not None and res_init.shape[0] != len(levels):
            res_init = None

        log_probs, res = self.solver_.laplace_log_prob(
            loglik=self._make_loglik(family_kwargs, lp_dict),
            eta_fixed=self._get_eta_fixed(family_kwargs),
            Z=Z,
            group_idx=group_idx,
            raneff_cov_chol=self.raneff_cov_module_[self.grouping_factor](),
            res_init=res_init,
            num_groups=len(levels)
        )
        if hasattr(self, '_res_warm_start'):
            self._res_warm_start[cache_key] = res

        penalty = self._get_penalty() if include_penalty else 0.
        log_prob = log_probs.sum() - penalty
        if mean:
            log_prob = log_prob / lp_dict['weight'].sum()
        return log_prob

    @torch.no_grad()
    def _solve_raneffs(self, X: ModelMatrix, y: ModelMatrix, sample_weight: Optional[np.ndarray] = None) -> ReTable:
        x_dict, lp_dict = self._build_model_mats(X, y, sample_weight, include_y=True)
        Z, labels = x_dict.pop('_re_Z'), x_dict.pop('_re_labels')
        levels, group_idx = _factorize(labels)
        family_kwargs = self._get_family_kwargs(**x_dict)
        res = self.solver_.solve(
            loglik=self._make_loglik(family_kwargs, lp_dict),
            eta_fixed=self._get_eta_fixed(family_kwargs),
            Z=Z,
            group_idx=group_idx,
            precision=torch.cholesky_inverse(self.raneff_cov_module_[self.grouping_factor]()),
            num_groups=len(levels)
        )
        return levels, res

    def _make_loglik(self, family_kwargs: dict, lp_dict: dict) -> Callable[[torch.Tensor], torch.Tensor]:
        param0 = self.family.params[0]

        def loglik(eta: torch.Tensor) -> torch.Tensor:
            return self.family.log_prob(self.family(**{**family_kwargs, param0: eta}), **lp_dict)

        return loglik

    def _get_eta_fixed(self, family_kwargs: dict) -> torch.Tensor:
        eta_fixed = family_kwargs[self.family.params[0]]
        if eta_fixed.shape[-1] != 1:
            raise NotImplementedError("Random-effects currently only supported for distribution-params with 1 output.")
        return eta_fixed

    # model-mats ---------
    def _standardize_design(self, X: ModelMatrix) -> Dict[str, Sequence[str]]:
        _, _, Z_df = self._split_X(X, design=self.raneff_design)
        out = {}
        for gf, cols in self.raneff_design.items():
            if callable(cols):
                cols = [c for c in cols(Z_df) if c not in self.raneff_design]
            elif isinstance(cols, str):
                raise TypeError(f"`raneff_design['{gf}']` should be a list of column-names, not a string.")
            missing = set(cols) - set(Z_df.columns)
            if missing:
                raise ValueError(f"Random-slope columns not found: {missing}")
            out[gf] = list(cols)
        return out

    def _split_X(self,
                 X: ModelMatrix,
                 design: Optional[dict] = None) -> Tuple[ModelMatrix, np.ndarray, pd.DataFrame]:
        """
        Split X into (1) the part that goes to ``Glm`` (i.e. the fixed-effects), (2) the group-labels, (3) the
        dataframe which contains the random-slope predictors.
        """
        gf = self.grouping_factor
        if design is None:
            design = self.raneff_design_
        if isinstance(X, dict):
            if gf not in X:
                raise ValueError(f"Grouping factor '{gf}' not in X; keys are {set(X)}")
            groups = X[gf]
            X_glm = SliceDict(**{k: v for k, v in X.items() if k != gf})
            Z_df = X_glm[self.family.params[0]] if self.family.params[0] in X_glm else pd.DataFrame(index=range(len(X)))
        elif isinstance(X, pd.DataFrame):
            if gf not in X.columns:
                hint = [c for c in X.columns if str(c).endswith(gf)]
                raise ValueError(
                    f"Grouping factor '{gf}' not in X.columns" + (f"; did you mean {hint}?" if hint else ".")
                )
            groups = X[gf]
            X_glm = X.drop(columns=[gf])
            Z_df = X_glm
        else:
            raise TypeError(f"{type(self).__name__} requires X to be a DataFrame or a dict, got {type(X)}")

        groups = np.asarray(groups)
        if groups.ndim == 2 and groups.shape[1] == 1:
            groups = groups[:, 0]
        if not isinstance(Z_df, pd.DataFrame):
            if any(design[gf]) and not callable(design[gf]):
                raise TypeError("Random-slopes require a DataFrame (for looking up column-names).")
            Z_df = pd.DataFrame(index=range(len(groups)))
        return X_glm, groups, Z_df

    def _build_model_mats(self,
                          X: ModelMatrix,
                          y: Optional[ModelMatrix],
                          sample_weight: Optional[np.ndarray] = None,
                          include_y: bool = False,
                          sparse_threshold: Optional[float] = None) -> Tuple[dict, Optional[dict]]:
        X_glm, groups, Z_df = self._split_X(X)
        x_dict, lp_dict = super()._build_model_mats(
            X=X_glm, y=y, sample_weight=sample_weight, include_y=include_y, sparse_threshold=sparse_threshold
        )
        _to_kwargs = get_to_kwargs(self.module_)
        cols = self.raneff_design_[self.grouping_factor]
        Z = torch.ones((len(groups), len(cols) + 1), **_to_kwargs)
        if cols:
            Z[:, 1:] = to_tensor(Z_df[cols].values.astype('float64'), **_to_kwargs)
        x_dict['_re_Z'] = Z
        x_dict['_re_labels'] = groups
        return x_dict, lp_dict

    def _get_family_kwargs(self, **kwargs) -> dict:
        """
        Adds random-effects to the first distribution-param, if a table of random-effects is provided.
        """
        Z = kwargs.pop('_re_Z', None)
        labels = kwargs.pop('_re_labels', None)
        re_table = kwargs.pop('_re_table', None)
        out = super()._get_family_kwargs(**kwargs)
        if re_table is not None:
            levels, res = re_table
            idx = levels.get_indexer(labels)
            unknown = idx < 0
            if unknown.any() and self.unknown_groups != 'zero':
                if self.unknown_groups == 'raise':
                    raise ValueError(f"{len(set(labels[unknown]))} groups have unknown random-effects.")
                raise NotImplementedError(f"`unknown_groups={self.unknown_groups}`")
            res_padded = torch.cat([res, torch.zeros_like(res[:1])])  # unknown groups -> last row (zeros)
            idx = torch.as_tensor(np.where(unknown, len(res), idx), device=res.device)
            param0 = self.family.params[0]
            out[param0] = ReSolver.get_eta(self._get_eta_fixed(out), Z, idx, res_padded)
        return out

    # prediction ---------
    def predict(self,
                X: ModelMatrix,
                type: Optional[str] = None,
                kwargs_as_is: Union[bool, dict] = False,
                group_data: Optional[tuple] = None,
                **kwargs) -> np.ndarray:
        """
        See :func:`foundry.glm.Glm.predict`.

        :param group_data: A tuple of ``(X, y)`` or ``(X, y, sample_weight)`` with data for the groups in ``X``; the
         random-effects will be solved from this data (holding fixed-effects and the random-effects covariance fixed).
         Groups not in ``group_data`` use the random-effects from training, if available, else see ``unknown_groups``.
        """
        if '_re_table' not in kwargs:
            kwargs['_re_table'] = self.get_raneff_table(group_data)
        return super().predict(X=X, type=type, kwargs_as_is=kwargs_as_is, **kwargs)

    def _prepare_predict(self, X: ModelMatrix, kwargs_as_is: Union[bool, dict] = False, **kwargs) -> Tuple[dict, dict]:
        re_table = kwargs.pop('_re_table', None)
        if re_table is None:
            re_table = self.raneff_table_
        x_dict, kwargs = super()._prepare_predict(X=X, kwargs_as_is=kwargs_as_is, **kwargs)
        x_dict = dict(x_dict, _re_table=re_table)  # plain dict: SliceDict only holds row-aligned values
        return x_dict, kwargs

    def get_raneff_table(self, group_data: Optional[tuple] = None) -> ReTable:
        """
        Get the random-effects for each group: from training, updated with any groups in ``group_data``.
        """
        if self.raneff_table_ is None:
            raise RuntimeError("Must call ``fit()`` first.")
        if group_data is None:
            return self.raneff_table_
        new_levels, new_res = self._solve_raneffs(*group_data)
        train_levels, train_res = self.raneff_table_
        keep = ~train_levels.isin(new_levels)
        return new_levels.append(train_levels[keep]), torch.cat([new_res, train_res[torch.as_tensor(keep)]])

    @torch.no_grad()
    def score(self,
              X: ModelMatrix,
              y: ModelMatrix,
              sample_weight: Optional[np.ndarray] = None,
              group_data: Optional[tuple] = None) -> float:
        """
        Mean log-prob (without penalty). For groups with known random-effects (from training or ``group_data``) this is
        the log-prob given those random-effects; for other groups, it's the (laplace-approximated) marginal log-prob.
        """
        levels, res = self.get_raneff_table(group_data)
        x_dict, lp_dict = self._build_model_mats(X, y, sample_weight, include_y=True)
        known = levels.get_indexer(x_dict['_re_labels']) >= 0

        total = 0.
        if known.any():
            xk, lpk = _subset(x_dict, known), _subset(lp_dict, known)
            family_kwargs = self._get_family_kwargs(**xk, _re_table=(levels, res))
            total += self.family.log_prob(self.family(**family_kwargs), **lpk).sum()
        if (~known).any():
            total += self.get_log_prob(
                _subset(x_dict, ~known), _subset(lp_dict, ~known), mean=False, include_penalty=False
            )
        return (total / lp_dict['weight'].sum()).item()

    # fitted attributes ---------
    @property
    def raneff_cov_(self) -> pd.DataFrame:
        """
        The estimated covariance-matrix of the random-effects.
        """
        names = self._raneff_names
        with torch.no_grad():
            L = self.raneff_cov_module_[self.grouping_factor]()
            return pd.DataFrame((L @ L.T).cpu().numpy(), index=names, columns=names)

    @property
    def raneff_dataframe_(self) -> pd.DataFrame:
        """
        The random-effects for each group in the training data.
        """
        levels, res = self.raneff_table_
        return pd.DataFrame(res.cpu().numpy(), index=levels.rename(self.grouping_factor), columns=self._raneff_names)

    @property
    def _raneff_names(self) -> Sequence[str]:
        return ['Intercept'] + list(self.raneff_design_[self.grouping_factor])

    # laplace coefs ---------
    def _get_all_params_and_names(self):
        prefixes, all_param_names, all_params = super()._get_all_params_and_names()
        for gf, cov_module in self.raneff_cov_module_.items():
            prefixes.append(f"raneff_cov_{gf}")
            all_param_names.append(cov_module.param_names(self._raneff_names))
            all_params.append(cov_module.params)
        return prefixes, all_param_names, all_params

    @torch.no_grad()
    def _set_all_params(self, unrolled: torch.Tensor):
        *_, all_params = self._get_all_params_and_names()
        expected = sum(p.numel() for p in all_params)
        if expected != len(unrolled):
            raise ValueError(f"Expected unrolled to have length {expected:,}, but was {len(unrolled):,}.")
        start_ = 0
        for param_values in all_params:
            end_ = start_ + param_values.numel()
            param_values[:] = unrolled[start_:end_].reshape(param_values.shape)
            start_ = end_


class _CholeskyCov(torch.nn.Module):
    """
    Covariance-matrix parameterized by the log-cholesky; ``forward()`` returns the cholesky factor.
    """

    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank
        self.params = torch.nn.Parameter(torch.zeros(rank + rank * (rank - 1) // 2))

    def forward(self) -> torch.Tensor:
        return log_chol_to_chol(self.params[:self.rank], self.params[self.rank:])

    def param_names(self, names: Sequence[str]) -> np.ndarray:
        assert len(names) == self.rank
        off_diag = torch.tril_indices(self.rank, self.rank, offset=-1)
        return np.asarray(
            [f'log_chol_diag__{n}' for n in names] +
            [f'chol_off_diag__{names[i]}:{names[j]}' for i, j in zip(*off_diag.tolist())]
        )


def _factorize(labels: np.ndarray) -> Tuple[pd.Index, torch.Tensor]:
    codes, levels = pd.factorize(labels, sort=True)
    if (codes < 0).any():
        raise ValueError("Grouping factor has missing values.")
    return pd.Index(levels), torch.as_tensor(codes)


def _subset(d: dict, mask: np.ndarray) -> dict:
    out = {}
    for k, v in d.items():
        if hasattr(v, 'shape') and len(v.shape) and v.shape[0] == len(mask):
            out[k] = v[torch.as_tensor(mask)] if isinstance(v, torch.Tensor) else v[mask]
        else:
            out[k] = v
    return out
