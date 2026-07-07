import copy
import os
from time import sleep
from typing import Union, Sequence, Optional, Callable, Tuple, Dict
from warnings import warn

import numpy as np
import pandas as pd

import torch
import torch.nn
from torch import distributions
from torch.distributions import transforms

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError

from tenacity import retry, retry_if_exception_type, stop_after_attempt
from tqdm import tqdm

from foundry.glm.distributions import (
    NegativeBinomial,
    Multinomial,
    Exponential,
    Weibull,
    CeilingWeibull
)
from foundry.glm.family import Family, SurvivalFamily, FamilyArgs
from foundry.glm.util import NoWeightModule, Stopping, SigmoidTransformForClassification, SoftmaxKp1

from foundry.covariance import Covariance
from foundry.hessian import hessian
from foundry.penalty import L2
from foundry.util import (
    FitFailedException,
    is_invalid,
    get_to_kwargs,
    to_tensor,
    to_1d,
    to_2d,
    is_array,
    ModelMatrix,
    ToSliceDict,
    SliceDict
)

N_FIT_RETRIES = int(os.getenv('FOUNDRY_N_FIT_RETRIES', 10))

family_names = {
    'bernoulli': FamilyArgs(
        distributions.Bernoulli,
        {'probs': SigmoidTransformForClassification()},
    ),
    'binomial': FamilyArgs(
        distributions.Binomial,
        {'probs': transforms.SigmoidTransform()},
        from_y=['total_count']
    ),
    'categorical': FamilyArgs(
        distributions.Categorical,
        {'probs': SoftmaxKp1()}
    ),
    'multinomial': FamilyArgs(
        Multinomial,
        {'probs': SoftmaxKp1()},
        from_y=['total_count']
    ),
    'poisson': FamilyArgs(
        distributions.Poisson,
        {'rate': transforms.ExpTransform()}
    ),
    'negative_binomial': FamilyArgs(
        NegativeBinomial,
        {'loc': transforms.ExpTransform(), 'dispersion': transforms.ExpTransform()},
    ),
    'exponential': FamilyArgs(
        Exponential,
        {'rate': transforms.ExpTransform(), }
    ),
    'weibull': FamilyArgs(
        Weibull,
        {
            'scale': transforms.ExpTransform(),
            'concentration': transforms.ExpTransform()
        }
    ),
    'normal': FamilyArgs(
        torch.distributions.Normal,
        {
            'loc': transforms.identity_transform,
            'scale': transforms.ExpTransform()
        }
    ),
    'multivariate_normal': FamilyArgs(
        torch.distributions.MultivariateNormal,
        {
            'loc': transforms.identity_transform,
            'covariance_matrix': Covariance()
        }
    ),
    'lognormal': FamilyArgs(
        torch.distributions.LogNormal,
        {
            'loc': transforms.identity_transform,
            'scale': transforms.ExpTransform()
        }
    ),
    'ceiling_weibull': FamilyArgs(
        CeilingWeibull,
        {
            'scale': transforms.ExpTransform(),
            'concentration': transforms.ExpTransform(),
            'ceiling': transforms.SigmoidTransform()
        }
    )
}
family_names['gaussian'] = family_names['normal']
family_names['mvnorm'] = family_names['multivariate_normal']

_posterior_predictive_cache = {}


class _glm_pp:

    def __enter__(self):
        _posterior_predictive_cache['_enabled'] = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        _posterior_predictive_cache.clear()


class Glm(BaseEstimator):
    """
    :param family: A family name; you can see available names with ``Glm.family_names``. (Advanced: you can also pass
     the :class:`foundry.glm.family.Family` instead of a name). Some names can be prefixed with ``survival_`` for
     support for censored data.
    :param penalty: Specify a penalty on coefficients. Can be a single value, or a dictionary of these to support
     different penalties per distribution-parameter. Values can either be floats, which will be interpreted as L2
     penalty magnitudes, or can be functions that take a ``torch.nn.Module`` as the first argument and that module's
     param-names as second argument, and returns a scalar penalty that will be applied to the log-prob. Finally,
     ``penalty`` can be a list/tuple of any of the above, in which case ``GridSearchCV`` will be performed to find the
     best penalty.
    :param col_mapping: Many distributions have multiple parameters: for example the normal distribution has a
     location and scale parameter. If a single dataframe/matrix is passed to  :class:`foundry.glm.Glm.fit`, the default
     behavior is to use this to predict the *first* parameter (e.g. loc) while other parameters (e.g. scale) have no
     predictors (i.e. are "intercept-only"). Sometimes this is not desired: for example, we want to use predictors to
     predict both. This can be either a list or a dictionary. For example, ``col_mapping=['loc','scale']`` will create a
      dict with the full model-matrix assigned to both params. A dictionary allows finer-grained control, e.g.:
     ``col_mapping={'loc':[col1,col2,col3],'scale':[col1]}``. Finally, instead of the dict-values being lists of
     columns, these can be functions that takes the data and return the relevant columns: e.g.
     ``col_mapping={'loc':sklearn.compose.make_column_selector('^col+.'), 'scale':[col1]}``.
    :param sparse_mm_threshold: Density threshold for creating a sparse model-matrix. If X has density less than this,
     the model-matrix will be sparse; otherwise it will be dense. Default 0, meaning never use sparse tensors.
    """
    family_names = family_names

    fixed_posterior = _glm_pp()

    def __init__(self,
                 family: Union[str, Family],
                 penalty: Union[float, Sequence[float], Dict[str, float]] = 0.,
                 col_mapping: Union[list, dict, None] = None,
                 sparse_mm_threshold: float = 0.0,
                 _warm_start: Optional[dict] = None):

        self.family = family
        self.penalty = penalty
        self.col_mapping = col_mapping
        self.sparse_mm_threshold = sparse_mm_threshold
        self._warm_start = _warm_start

        # set in _init_module:
        self._module_ = None
        self._module_param_names_ = None
        self.label_encoder_ = None
        self.to_slice_dict_ = None

        # laplace params:
        self._coef_mvnorm_ = None
        self.converged_ = None
        self._laplace_coefs_names_ = None

        # updated on each fit():
        self._fit_failed = 0

    @property
    def module_(self) -> torch.nn.ModuleDict:
        if self._module_ is None:
            raise NotFittedError("Tried to access `module_` prior to fitting.")
        return self._module_

    @module_.setter
    def module_(self, value: torch.nn.ModuleDict):
        self._module_ = value

    def _predict_proba(self,
                       X: ModelMatrix,
                       kwargs_as_is: Union[bool, dict] = False,
                       **kwargs) -> np.ndarray:
        """
        Return the probability of each class.
        """
        assert self.family.supports_predict_proba
        probs = self.predict(X=X, type='probs', kwargs_as_is=kwargs_as_is, **kwargs)
        assert len(probs.shape) == 2

        # pytorch distributions can vary in behavior: e.g. bernoulli outputs a single prediction for 2-classes,
        # while categorical/multinomial output a final dim whose extent is equal to the number of classes
        # so bernoulli/binomial breaks sklearn convention and we need to add a 2nd col
        if probs.shape[-1] == 1:
            if self.label_encoder_ is not None:
                expected_num_classes = len(self.label_encoder_.classes_)
                if expected_num_classes != 2:
                    raise RuntimeError(
                        f"There are {expected_num_classes} ``self.label_encoder_.classes_``, but distribution "
                        f"``probs.shape[-1]`` is {probs.shape[-1]}"
                    )
            # currently no way to get `expected_num_classes` if label_encoder_ is not present (afaik, only binomial)
            assert np.all((probs >= 0) & (probs <= 1))
            probs = np.concatenate([1 - probs, probs], axis=1)
            probs /= probs.sum(axis=1, keepdims=True)

        return probs

    def _init_family(self, family: Union[Family, str], y: Optional[dict] = None) -> Family:
        """ if family is a string, turns family to a Family. else return unchanged. """
        if isinstance(family, str):
            family = family_from_string(family, y=y)
        return family

    def fit(self,
            X: ModelMatrix,
            y: ModelMatrix,
            sample_weight: Optional[np.ndarray] = None,
            cv_kwargs: Optional[dict] = None,
            **kwargs) -> 'Glm':
        """
        :param X: A array/dataframe of predictors, or a dictionary of these. If a dict, then keys should correspond to
         ``self.family.params``.
        :param y: An array of targets. This can instead be a dictionary, with the target-value in the "value" entry,
         and additional auxiliary information (e.g. sample-weights, upper/lower censoring for survival modeling) in
         other entries.
        :param sample_weight: The weight for each row. If performing cross-validation, this argument should not be used,
         as sklearn does not support it; instead, pass a :class:`foundry.util.SliceDict` for the ``y`` argument, with a
         'sample_weight' entry.
        :param cv_kwargs: If ``self.penalty`` is a list/tuple, then the optimal penalty will be chosen with
         ``GridSearchCV``; these are keyword-arguments to that cls.
        :param reset: If calling ``fit()`` more than once, should the module/weights be reinitialized. Default True.
        :param callbacks: A list of callbacks: functions that take the ``Glm`` instance as a first argument and the
         train-loss as a second argument.
        :param stopping: Controls stopping based on converging loss/parameters. This argument is passed to
         :class:`foundry.glm.util.Stopping` (e.g. ``(.01,)`` would use abstol of .01).
        :param max_iter: The max. number of iterations before stopping training regardless of convergence. Default 200.
        :param max_loss: If training stops and loss is higher than this, a class:`foundry.util.FitFailedException` will
         be raised and fitting will be retried with a different set of inits.
        :param verbose: Whether to allow print statements and a progress bar during training. Default True.
        :param estimate_laplace_coefs: If true, then after fitting, the hessian of the optimized parameters will be
         estimated; this can then be used for confidence-intervals and statistical inference (see ``coef_dataframe_``).
         Can set to False if you want to save time and skip this step.
        :return: This ``Glm`` instance.
        """
        self.family = self._init_family(self.family, y=y)

        if self.family.supports_predict_proba:
            # sklearn uses `hasattr` in some cases to check for `predict_proba`, so can't just
            # have it error out for some distributions -- need to add it dynamically.
            # noinspection PyAttributeOutsideInit
            self.predict_proba = self._predict_proba

        if isinstance(self.penalty, (list, tuple)):
            from sklearn.model_selection import GridSearchCV
            penalties = list(self.penalty)
            fit_kwargs = kwargs.copy()
            fit_kwargs['estimate_laplace_coefs'] = False

            # move sample weight to y:
            y = self._standardize_y(y=y, sample_weight=sample_weight)

            # warm-start:
            if kwargs.get('verbose', True):
                print("Initializing warm start...")
            self.set_params(penalty=penalties[len(penalties) // 2])
            self._fit(X=X, y=y, **fit_kwargs)
            self.set_params(_warm_start=self.module_.state_dict())
            self.module_ = None
            fit_kwargs['verbose'] = False

            # search:
            if kwargs.get('verbose', True):
                print("GridSearchCV...")
            cv_kwargs = cv_kwargs or {}
            gcv = GridSearchCV(
                estimator=self,
                param_grid={'penalty': penalties},
                refit=False,
                **cv_kwargs
            )
            gcv.fit(X=X, y=y, **fit_kwargs)

            # set to best penalty, refit:
            best_penalty = gcv.best_params_['penalty']
            if kwargs.get('verbose', True):
                print(f"Fitting with best_penalty={best_penalty}...")
            self.set_params(penalty=best_penalty, _warm_start=None)
            return self._fit(X=X, y=y, **kwargs)
        else:
            if cv_kwargs:
                warn("Ignoring `cv_kwargs`, penalty is scalar.")
            return self._fit(X=X, y=y, sample_weight=sample_weight, **kwargs)

    @retry(retry=retry_if_exception_type(FitFailedException), reraise=True, stop=stop_after_attempt(N_FIT_RETRIES + 1))
    def _fit(self,
             X: ModelMatrix,
             y: ModelMatrix,
             sample_weight: Optional[np.ndarray] = None,
             reset: bool = True,
             callbacks: Sequence[Callable] = (),
             stopping: Union['Stopping', tuple, dict] = (.001,),
             max_iter: int = 200,
             max_loss: float = float('inf'),
             verbose: bool = True,
             estimate_laplace_coefs: bool = True) -> 'Glm':

        # tallying fit-failurs:
        if self._fit_failed:
            if verbose:
                print(f"Retry attempt {self._fit_failed}/{N_FIT_RETRIES}")
            reset = True

        # build model:
        if self._module_ is None or reset:
            if self._module_ is not None and verbose:
                warn("Resetting module with reset=True")
            # initialize module:
            self._init_module(X, y, verbose=verbose)

        # optimizer:
        self.optimizer_ = self._init_optimizer()

        # stopping:
        if isinstance(stopping, dict):
            stopping = Stopping(**stopping, optimizer=self.optimizer_)
        elif isinstance(stopping, (list, tuple)):
            stopping = Stopping(*stopping, optimizer=self.optimizer_)
        else:
            assert isinstance(stopping, Stopping)
            stopping.optimizer = self.optimizer_

        # build model-mats:
        x_dict, lp_dict = self._build_model_mats(X, y, sample_weight, include_y=True)

        # progress bar:
        prog = None
        if verbose:
            prog = self._init_pb()

        epoch = 0

        # 'closure' for torch.optimizer.Optimizer.step method:
        def closure():
            self.optimizer_.zero_grad()
            # TODO: if distribution fails validation due to nans/infs, will not currently trigger FitFailedException
            loss = -self.get_log_prob(x_dict, lp_dict)
            if is_invalid(loss):
                self._fit_failed += 1
                raise FitFailedException("`nan`/`inf` loss")
            loss.backward()
            if prog:
                prog.update()
                prog.set_description(
                    f"Epoch {epoch:,}; Loss {loss.item():.4}; Convergence {stopping.get_info()}"
                )
            return loss

        # fit-loop:
        assert max_iter > 0
        for epoch in range(max_iter):
            try:
                if prog:
                    prog.reset()
                    prog.set_description(f"Epoch {epoch:,}; Loss -; Convergence {stopping.get_info()}")
                train_loss = self.optimizer_.step(closure).item()
                for callback in callbacks:
                    callback(self, train_loss)

                if stopping(train_loss):
                    break

            except KeyboardInterrupt:
                sleep(1)
                break
            finally:
                self.optimizer_.zero_grad(set_to_none=True)

        if train_loss > max_loss:
            self._fit_failed += 1
            raise FitFailedException(f"train_loss ({train_loss}) too high, consider increasing max_loss ({max_loss})")

        self._fit_failed = 0

        if estimate_laplace_coefs:
            if verbose:
                print("Estimating laplace coefs... (you can safely keyboard-interrupt to cancel)")
            try:
                self.estimate_laplace_coefs(X=X, y=y, sample_weight=sample_weight)
            except KeyboardInterrupt:
                pass

        return self

    def get_log_prob(self,
                     x_dict: dict,
                     lp_dict: dict,
                     mean: bool = True,
                     include_penalty: bool = True) -> torch.Tensor:
        """
        Get the penalized log prob, applying weights as needed.
        """
        log_probs = self.family.log_prob(
            self.family(**self._get_family_kwargs(**x_dict)),  # distribution(ilink(X \theta))
            **lp_dict
        )
        penalty = self._get_penalty() if include_penalty else 0.
        log_prob = log_probs.sum() - penalty
        if mean:
            log_prob = log_prob / lp_dict['weight'].sum()
        return log_prob

    def _get_family_kwargs(self, **kwargs) -> dict:
        """
        Call each entry in the module_ dict to get predicted family params. Any leftover keywords are passed as-is.
        """
        out = {}
        for dp, dp_module in self.module_.items():
            mm = kwargs.pop(dp, None)
            if isinstance(dp_module, NoWeightModule):
                if mm is not None and mm.numel():
                    raise RuntimeError(f"When fitted, no predictors were passed for {dp}, so can't pass them now.")
                out[dp] = dp_module(mm)
            else:
                out[dp] = dp_module(mm)
        out.update(kwargs)  # remaining are passthru
        return out

    def _get_xdict(self, X: ModelMatrix, sparse_threshold: float) -> Dict[str, torch.Tensor]:
        _to_kwargs = get_to_kwargs(self.module_)

        Xdict = self.to_slice_dict_.transform(X)

        # convert to tensors:
        for nm in list(Xdict):
            if is_array(Xdict[nm]):
                Xdict[nm] = to_tensor(Xdict[nm], sparse_threshold=sparse_threshold, **_to_kwargs)

            if nm in self.family.params:
                # model-mat params
                assert len(Xdict[nm].shape) == 2, f"len(X['{nm}'].shape)!=2"
                if is_invalid(Xdict[nm]):
                    raise ValueError(f"nans/infs in X ({nm})")

        return Xdict

    @staticmethod
    def _standardize_y(y: ModelMatrix, sample_weight: Optional[np.ndarray]) -> SliceDict:
        if isinstance(y, dict):
            assert 'value' in y
            ydict = y.copy()
        else:
            ydict = {'value': y}

        if 'weight' in ydict:
            if sample_weight is not None:
                raise ValueError("Please pass either `sample_weight` or y=dict(weight=), but not both.")
        elif sample_weight is not None:
            ydict['weight'] = sample_weight

        if ydict.get('weight', None) is None:
            ydict['weight'] = torch.ones(ydict['value'].shape[0])
        else:
            assert (ydict['weight'] >= 0).all()
        return SliceDict(**{k: to_2d(np.asarray(v)) if is_array(v) else v for k, v in ydict.items()})

    def _get_ydict(self, y: ModelMatrix, sample_weight: Optional[np.ndarray]) -> Dict[str, torch.Tensor]:
        ydict = self._standardize_y(y=y, sample_weight=sample_weight)

        _to_kwargs = get_to_kwargs(self.module_)

        for k in list(ydict):
            # special handling for classification:
            if k == 'value' and self.label_encoder_ is not None:
                ydict['value'] = self.label_encoder_.transform(to_1d(ydict['value']))
                ydict['value'] = to_tensor(ydict['value'], **_to_kwargs)
                continue

            if not hasattr(ydict[k], '__iter__'):
                if isinstance(ydict[k], (int, float)):
                    continue  # pass as-is
                else:
                    # wait until we have a use-case
                    raise NotImplementedError(
                        f"Unclear how to handle {k}, please report this error to the package maintainer"
                    )

            # standard handling:
            ydict[k] = to_2d(to_tensor(ydict[k], **_to_kwargs))
            # note: no `is_invalid` check, some families might support missing values or infs
            if ydict[k].shape[0] != ydict['value'].shape[0]:
                raise ValueError(f"{k}.shape[0] does not match y.shape[0]")

        return ydict

    def _build_model_mats(self,
                          X: ModelMatrix,
                          y: Optional[ModelMatrix],
                          sample_weight: Optional[np.ndarray] = None,
                          include_y: bool = False,
                          sparse_threshold: Optional[float] = None) -> Tuple[dict, Optional[dict]]:
        """
        :param X: A dataframe/ndarray/tensor, or dictionary of these.
        :param y: An optional dataframe/ndarray/tensor, or dictionary of these.
        :param sample_weight: Optional sample-weights
        :param include_y: If True, then will raise if y is not present; if False, will return None in place of y.
        :param sparse_threshold: Allows overriding ``self.sparse_threshold``.
        :return: Two dictionaries: one of model-mat kwargs (for prediction) and one of target-related kwargs
         (for evaluation i.e. log-prob).
        """
        _to_kwargs = get_to_kwargs(self.module_)

        if sparse_threshold is None:
            sparse_threshold = self.sparse_mm_threshold
        Xdict = self._get_xdict(X, sparse_threshold=sparse_threshold)

        if not include_y:
            return Xdict, None

        if y is None:
            raise ValueError("Must pass `y` when fitting.")

        ydict = self._get_ydict(y=y, sample_weight=sample_weight)

        if Xdict and list(Xdict.values())[0].shape[0] != ydict['value'].shape[0]:
            raise ValueError("Expected X.shape[0] to match y.shape[0]")

        for key in self.family.from_y:
            if key in ydict:
                if key in Xdict:
                    warn(f"'{key}' in both X and y, ignoring y")
                Xdict[key] = ydict.pop(key)

        return Xdict, ydict

    def _init_module(self, X: ModelMatrix, y: ModelMatrix, verbose: bool = False):
        # memorize the col -> dist-param mapping
        self._init_col_mapping(X)

        # standardize X and y:
        X = self.to_slice_dict_.transform(X)
        if not isinstance(y, dict):
            y = {'value': y}
        y_arr = y['value']
        y_arr = to_2d(np.asanyarray(y_arr))

        # if classification, handle label encoding:
        self.label_encoder_ = None
        if self.family.supports_predict_proba and not self.family.has_total_count:
            if y_arr.shape[-1] != 1:
                raise ValueError(
                    f"GLM w/family {self.family} expects a 1D ``y`` whose values are the class-labels. However,"
                    f"y.shape is {y_arr.shape}."
                )
            self.label_encoder_ = LabelEncoder()
            self.label_encoder_.fit(to_1d(y_arr))
            assert len(self.label_encoder_.classes_) > 1

        # create modules that predict params:
        self._module_param_names_ = {}
        self.module_ = torch.nn.ModuleDict()
        for dp in self.family.params:
            Xp = X.get(dp, None)
            module_num_outputs = self._get_module_num_outputs(y_arr, dp)
            if Xp is None or not Xp.shape[1]:
                # always use the base approach when no input features:
                module, nms = Glm.module_factory(X=None, output_dim=module_num_outputs)
            else:
                module, nms = self.module_factory(X=Xp, output_dim=module_num_outputs)
            self.module_[dp] = module
            self._module_param_names_[dp] = {k: np.asarray(v) for k, v in nms.items()}

        if self._warm_start is not None:
            if verbose:
                print("Initializing module with warm-start")
            try:
                self.module_.load_state_dict(self._warm_start)
            except Exception as e:  # TODO
                warn(f"Unable to load warm start: {str(e)}")

    def _init_col_mapping(self, X: ModelMatrix):
        col_mapping = self.col_mapping
        if col_mapping is None and not isinstance(X, dict):
            # if col_mapping is None and they didn't pass a dict for X, default is use first family param
            # otherwise use col_mapping as-is (if X is a dict, will be used by ToSliceDict)
            col_mapping = [self.family.params[0]]
        self.to_slice_dict_ = ToSliceDict(mapping=col_mapping)
        self.to_slice_dict_.fit(X)

    def _get_module_num_outputs(self, y: np.ndarray, dist_param_name: str) -> int:
        assert len(y.shape) == 2
        if self.label_encoder_ is not None:
            # classification
            y_dim = len(self.label_encoder_.classes_)
        else:
            # regression:
            y_dim = y.shape[1]

        # for most distribution-params, the number of outputs we need to predict is dictated by the structure of `y`:
        # we need to predict one value if `y` is (n, 1), two values if `y` is (n, 2), etc.
        # however, there are exceptions:
        # - parameters of a covariance-matrix
        # - parameters of a categorical distribution (k-1 classes)
        # the way this is implemented is by having that inverse-link-function also have a `get_param_dim` method
        # which takes the shape of y and returns the number of parameters needed
        ilink = self.family.params_and_links[dist_param_name]
        get_param_dim = getattr(ilink, 'get_param_dim', None)
        if get_param_dim:
            return get_param_dim(y_dim)
        else:
            return y_dim

    @classmethod
    def module_factory(cls,
                       X: Union[None, pd.DataFrame, np.ndarray],
                       output_dim: int) -> Tuple[torch.nn.Module, dict]:
        """
        Given a model-matrix and output-dim, produce a ``torch.nn.Module`` that predicts a distribution-parameter. The
        default produces a ``torch.nn.Linear`` layer. Additionally, this function returns a dictionary whose keys are
        param-names and whose values are the names of the individual param-elements. For the default case, each weight
        is named according to the column-names in X (or "x{i}" if X is not a dataframe).

        :param X: A dataframe or ndarray.
        :param output_dim: The number of output dimensions.
        :return: A tuple of (module, dict). The dictionary should correspond to ``module.named_parameters()``: the dict
         keys should be module param-names, and the dict values should be arrays (or coercible to arrays) of strings
         whose shape matches the parameter shapes.
        """
        if X is None or not X.shape[1]:
            module = NoWeightModule(dim=output_dim)
            columns = []
        else:
            module = torch.nn.Linear(in_features=X.shape[1], out_features=output_dim, bias=True)
            with torch.no_grad():
                module.weight.normal_(std=.1)
            columns = list(X.columns) if hasattr(X, 'columns') else [f'x{i}' for i in range(X.shape[1])]

        module_param_names = {'bias': [], 'weight': []}
        if output_dim == 1:
            # for most common case of 1d output, param names are just feature-names
            module_param_names['bias'].append('bias')
            module_param_names['weight'].append(columns)
        else:
            # if multi-output, we prefix with output idx:
            for i in range(output_dim):
                module_param_names['bias'].append(f'y{i}__bias')
                module_param_names['weight'].append([f'y{i}__{c}' for c in columns])

        return module, module_param_names

    def _init_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.LBFGS(
            self.module_.parameters(), max_iter=10, line_search_fn='strong_wolfe', lr=.25
        )

    def _init_pb(self) -> Optional[tqdm]:
        max_eval = self.optimizer_.param_groups[0].get('max_eval', False)
        if max_eval:
            return tqdm(total=max_eval)
        return None

    @torch.inference_mode()
    def score(self, X: ModelMatrix, y: ModelMatrix, sample_weight: Optional[np.ndarray] = None) -> float:
        """
        Uses log_prob (without penalty) for scoring.
        """
        x_dict, lp_dict = self._build_model_mats(X, y, sample_weight, include_y=True)
        return self.get_log_prob(x_dict, lp_dict, mean=True, include_penalty=False).item()

    @torch.inference_mode()
    def predict(self,
                X: ModelMatrix,
                type: Optional[str] = None,
                kwargs_as_is: Union[bool, dict] = False,
                **kwargs) -> np.ndarray:
        """
        Get the predicted distribution, then extract an attribute from that distribution and return as an ndarray.

        :param X: An array or dictionary of arrays.
        :param type: The type of the prediction -- i.e. the attribute to be extracted from the resulting
         ``torch.Distribution``. The default depends on the family. If ``self.family.supports_predict_proba``, and the
         distribution doesn't have a ``total_count`` parameter (e.g. multinomial), then this method will predict the
         class. Otherwise, this method will predict the mean of the distribution.
        :param kwargs_as_is: If the ``type`` is callable, then kwargs that are arrays will be converted to tensors,
         and if 1D then will be unsqueezed to 2D (unsqueezing is to avoid accidental broadcasting, wherein (e.g.) a
         distribution with batch_shape of N*1 receives a ``value`` w/shape (N,1), resulting in a (N,N) tensor,
         and usually an OOM error). Can be ``True``, or can pass a dict to set true/false on per-kwarg basis.
        :param kwargs: Keyword arguments to pass if ``type`` is callable. See ``kwargs_as_is``.
        :return: A ndarray of predictions.
        """
        if type is None:
            if self.label_encoder_ is not None:
                probs = self.predict_proba(X, kwargs_as_is=kwargs_as_is, **kwargs)
                class_idxs = probs.argmax(axis=1)
                return self.label_encoder_.inverse_transform(to_1d(class_idxs))
            else:
                type = 'mean'

        x_dict, kwargs = self._prepare_predict(X=X, kwargs_as_is=kwargs_as_is, **kwargs)

        result = self._predict_from_x_dict(x_dict=x_dict, type_=type, **kwargs)

        return result

    @torch.inference_mode()
    def _prepare_predict(self, X: ModelMatrix, kwargs_as_is: Union[bool, dict] = False, **kwargs) -> Tuple[dict, dict]:
        x_dict, *_ = self._build_model_mats(X=X, y=None)
        if 'validate_args' in kwargs:
            x_dict['validate_args'] = kwargs.pop('validate_args')

        if not isinstance(kwargs_as_is, dict):
            kwargs_as_is = {k: kwargs_as_is for k in kwargs}
        for k in list(kwargs):
            if not kwargs_as_is.get(k, False) and is_array(kwargs[k]):
                kwargs[k] = to_2d(to_tensor(kwargs[k], **get_to_kwargs(self.module_)))

        return x_dict, kwargs

    @torch.inference_mode()
    def _predict_from_x_dict(self, x_dict: dict, type_: str, **kwargs) -> np.ndarray:
        dist_kwargs = self._get_family_kwargs(**x_dict)
        dist = self.family(**dist_kwargs)
        result = getattr(dist, type_)
        if callable(result):
            result = result(**kwargs)
        elif kwargs:
            warn(f"Ignoring {set(kwargs)}, `{type(dist).__name__}.{type_}` not callable.")
        return result.numpy()

    def _sample_coef_mvnorm(self, n_iters: int, use_cache: bool = None) -> torch.Tensor:
        if use_cache is None:
            use_cache = _posterior_predictive_cache.get('_enabled', False)

        if not use_cache:
            return self._coef_mvnorm_.sample((n_iters,))

        key = (id(self._coef_mvnorm_), n_iters)
        if key not in _posterior_predictive_cache:
            _posterior_predictive_cache[key] = self._sample_coef_mvnorm(n_iters=n_iters, use_cache=False)
        return _posterior_predictive_cache[key]

    @torch.no_grad()
    def posterior_predict(self,
                          X: ModelMatrix,
                          type: str = 'mean',
                          kwargs_as_is: Union[bool, dict] = False,
                          n_iters: int = 500,
                          collate_fn: Optional[callable] = None,
                          **kwargs) -> np.ndarray:
        """
        Generate predictions from the model's (MAP) posterior.

        :param X: An array or dictionary of arrays.
        :param type: The type of the prediction -- i.e. the attribute to be extracted from the resulting
         ``torch.Distribution``. Default is to predict the mean of the distribution (note this differs from the default
         in ``predict`` since that method follows sklearn behavior for classifiers and predicts the argmax).
        :param kwargs_as_is: See documentation in ``predict()``
        :param n_iters: How many samples to draw, default 500.
        :param collate_fn: A function that will take the sampled output as a list of length ``n_iters`` and
         stack/concatenate into an array. Default is to create an array that is ``orig_shape + (n_iters,)``, unless
         the output from ``predict`` has shape ``(n_records, 1)``, in which case will create an array with
         ``(n_records, n_iters)``.
        :param kwargs: Keyword arguments to pass if ``type`` is callable. See also ``kwargs_as_is``.
        :return: A ndarray of predictions, see ``collate_fn``.
        """
        if self._coef_mvnorm_ is None:
            raise RuntimeError("Must call ``estimate_laplace_coefs()`` first.")
        x_dict, kwargs = self._prepare_predict(X=X, kwargs_as_is=kwargs_as_is, **kwargs)

        samples = self._sample_coef_mvnorm(n_iters=n_iters)
        orig = copy.deepcopy(self.module_)
        try:
            out = []
            for sample in tqdm(samples, delay=10):
                self._set_all_params(sample)
                out.append(self._predict_from_x_dict(x_dict, type_=type, **kwargs))
        finally:
            self.module_ = orig

        if collate_fn is None:
            if len(out[0].shape) != 2 or out[0].shape[-1] > 1:
                collate_fn = lambda x: np.stack(x, -1)
            else:
                collate_fn = lambda x: np.concatenate(x, 1)

        return collate_fn(out)

    def _get_penalty(self) -> torch.Tensor:
        """
        Get penalty on sum(log_prob) based on the module-weights and self.penalty.
        """
        if not self.penalty:
            return torch.zeros(1, **get_to_kwargs(self.module_))

        # standardize to dictionary with param-names:
        if isinstance(self.penalty, dict):
            if set(self.penalty) != set(self.module_.keys()):
                raise ValueError(
                    f"``self.penalty.keys()`` is {set(self.penalty)}, but expected {set(self.module_.keys())}"
                )
        else:
            self.penalty = {k: self.penalty for k in self.module_.keys()}

        # standardize to values = callables:
        for param_name in list(self.penalty):
            maybe_callable = self.penalty[param_name]
            if not callable(maybe_callable):
                # if not callable, it's a multiplier --i.e. L2's 'precision'
                self.penalty[param_name] = L2(precision=maybe_callable)

        # call each:
        to_sum = []
        for param_name, penalty_fun in self.penalty.items():
            to_sum.append(
                penalty_fun(self.module_[param_name], self._module_param_names_[param_name])
            )
        return torch.stack(to_sum).sum()

    @property
    def coef_dataframe_(self) -> pd.DataFrame:
        if self._coef_mvnorm_ is None:
            raise RuntimeError("Must call ``estimate_laplace_coefs()`` first.")
        out = []
        with torch.inference_mode():
            ses = self._coef_mvnorm_.covariance_matrix.diag().sqrt()
            for name, estimate, se in zip(self._laplace_coefs_names_, self._coef_mvnorm_.mean, ses):
                estimate = estimate.item()
                se = se.item()
                if not self.converged_:
                    se = float('nan')
                out.append({'name': name, 'estimate': estimate, 'se': se})
        return pd.DataFrame(out)

    def estimate_laplace_coefs(self, X: ModelMatrix, y: ModelMatrix, sample_weight: Optional[np.ndarray] = None):
        self._laplace_coefs_names_, means, hess = self._estimate_laplace_coefs(X=X, y=y, sample_weight=sample_weight)

        # create mvnorm for laplace approx:
        self.converged_ = False
        with torch.inference_mode():
            try:
                self._coef_mvnorm_ = torch.distributions.MultivariateNormal(
                    means, precision_matrix=hess, validate_args=True
                )
                self.converged_ = True
            except (RuntimeError, ValueError) as e:
                warn(f"Second order estimation of parameter distribution failed.\n{str(e)}")
                fake_cov = torch.diag(torch.diag(hess).pow(-1).clip(min=1E-5))
                self._coef_mvnorm_ = torch.distributions.MultivariateNormal(means, covariance_matrix=fake_cov)

    def _get_all_params_and_names(self) -> Tuple[Sequence[str], Sequence[np.ndarray], Sequence[torch.Tensor]]:
        """
        Get all params as a list, and their names as a matching list of ndarrays w/same-shape
        """
        prefixes = []
        all_param_names = []
        all_params = []
        for dp in self.family.params:
            for nm, param_values in self.module_[dp].named_parameters():
                prefixes.append(f"{dp}_{nm}")
                param_names = self._module_param_names_[dp][nm]
                assert param_names.shape == param_values.shape, f"param_names.shape!=param_values.shape for {dp}.{nm}"
                all_param_names.append(param_names)
                all_params.append(param_values)

        return prefixes, all_param_names, all_params

    @torch.no_grad()
    def _set_all_params(self, unrolled: torch.Tensor):
        """
        ``_estimate_laplace_coefs()`` provides names, means, hess that are 'unrolled' from the nested structure in
        ``self.module_``. This function takes a 1d tensor with that same unrolled structure, and puts it back into
        ``self.module_``.
        """
        orig = copy.deepcopy(self.module_)
        start_ = 0
        for dp in self.family.params:
            for nm, param_values in self.module_[dp].named_parameters():
                end_ = start_ + param_values.numel()
                param_values[:] = unrolled[start_:end_]
                start_ = end_
        if start_ != len(unrolled):
            self.module_ = orig
            raise ValueError(
                f"Expected unrolled to have length {start_:,}, but was {len(unrolled):,}. Restored original params."
            )

    def _estimate_laplace_coefs(self,
                                X: ModelMatrix,
                                y: ModelMatrix,
                                sample_weight: Optional[np.ndarray] = None
                                ) -> Tuple[Sequence[str], torch.Tensor, torch.Tensor]:
        # get all params as a list, and their names as a matching list:
        prefixes, all_param_names, all_params = self._get_all_params_and_names()

        # get the log-prob -> hessian:
        x_dict, lp_dict = self._build_model_mats(X, y, sample_weight, include_y=True)
        log_prob = self.get_log_prob(x_dict=x_dict, lp_dict=lp_dict, mean=False)
        hess = hessian(output=-log_prob.squeeze(), inputs=all_params, allow_unused=True, progress=False)

        # flatten out the params and names so that names, means, and hess all match:
        # TODO: any way to assert reshape(-1) matches internals of `hessian()`?
        means = torch.cat([p.reshape(-1) for p in all_params])

        all_param_names_flat = []
        for prefix, pns in zip(prefixes, all_param_names):
            all_param_names_flat.extend(f"{prefix}__{pnm}" for pnm in pns.reshape(-1))

        return all_param_names_flat, means, hess


def family_from_string(family: str, y: Optional[dict] = None) -> Family:
    # survival analysis can be indicated by passing a target that involves censoring
    is_survival = isinstance(y, dict) and any('cens' in k for k in y)
    if family.startswith('survival'):
        family = family.replace('survival', '').lstrip('_')
        warn(
            "Prefixing family-alias with 'survival' is deprecated; censoring in ``y`` is sufficient.",
            DeprecationWarning
        )
        if y is not None:
            assert is_survival

    if is_survival:
        return SurvivalFamily(**family_names[family].to_dict())
    else:
        return Family(**family_names[family].to_dict())
