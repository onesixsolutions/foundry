from typing import Callable, Optional, Tuple, Type, List
from warnings import warn

import torch
from torch import distributions
from torch.distributions import transforms

from foundry.glm.family import Family

# a function that takes the linear-predictor (for the first distribution-param) and returns per-row log-probs:
LogLikFun = Callable[[torch.Tensor], torch.Tensor]


class ReSolver:
    """
    Finds the posterior mode of the random-effects for each group, and computes the Laplace-approximation of the
    marginal log-likelihood.

    There can be multiple linear-predictors per row (e.g. one per outcome in a multivariate model, possibly with
    different families). Random-effects ``b`` (one vector per group, stacked across linear-predictors) enter as
    ``eta[i, j] = eta_fixed[i, j] + Z[i, j] @ b[group_idx[i]]``. The log-likelihood must be separable across rows and
    linear-predictors (i.e. outcomes are conditionally independent given the random-effects); unobserved outcomes
    should simply contribute zero.

    The default implementation is family-agnostic: it uses batched (per-group) Newton steps, with first and second
    derivatives of the log-prob wrt the linear-predictor obtained by autograd. Subclasses can override when a better
    solver exists for a family (see ``register_solver``).

    :param max_iter: Max. number of Newton iterations.
    :param tol: Convergence tolerance: a group is converged when its Newton decrement (the predicted increase in its
     log-posterior from the next step) is less than ``tol * (1 + abs(log_posterior))``. Floored by the precision of the
     dtype, since smaller improvements can't be resolved.
    :param max_halvings: Max. number of step-halvings in the per-group backtracking line-search.
    """
    line_search: bool = True

    def __init__(self, max_iter: int = 100, tol: float = 1e-8, max_halvings: int = 20):
        self.max_iter = max_iter
        self.tol = tol
        self.max_halvings = max_halvings

    @staticmethod
    def derivatives(loglik: LogLikFun,
                    eta: torch.Tensor,
                    create_graph: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns per-row log-probs and their first and second derivatives wrt ``eta`` (each with the same shape as
        ``eta``). Since the log-prob is separable across the elements of ``eta``, the hessian is diagonal, and we can
        get it from the grad of the sum of the grads.
        """
        with torch.enable_grad():
            if not create_graph or not eta.requires_grad:
                eta = eta.detach().requires_grad_()
            log_probs = loglik(eta)
            d1, = torch.autograd.grad(log_probs.sum(), eta, create_graph=True)
            d2, = torch.autograd.grad(d1.sum(), eta, create_graph=create_graph, allow_unused=True)
        if d2 is None:
            d2 = torch.zeros_like(d1)
        if not create_graph:
            log_probs, d1, d2 = log_probs.detach(), d1.detach(), d2.detach()
        return log_probs.reshape(-1), d1, d2

    @staticmethod
    def get_eta(eta_fixed: torch.Tensor, Z: torch.Tensor, group_idx: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
        return eta_fixed + torch.einsum('njk,nk->nj', _as_3d(Z), res[group_idx])

    @staticmethod
    def _group_objective(log_probs: torch.Tensor,
                         res: torch.Tensor,
                         group_idx: torch.Tensor,
                         precision: torch.Tensor) -> torch.Tensor:
        """
        Unnormalized log-posterior of the random-effects, per group.
        """
        ll = torch.zeros(len(res), dtype=log_probs.dtype, device=log_probs.device).index_add(0, group_idx, log_probs)
        return ll - .5 * ((res @ precision) * res).sum(1)

    @staticmethod
    def _group_grad_and_hess(Z: torch.Tensor,
                             group_idx: torch.Tensor,
                             d1: torch.Tensor,
                             d2: torch.Tensor,
                             res: torch.Tensor,
                             precision: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gradient and (negative) hessian of the log-posterior of the random-effects, per group.
        """
        num_groups, num_res = res.shape
        Z = _as_3d(Z)
        grad = torch.zeros_like(res).index_add(0, group_idx, torch.einsum('nj,njk->nk', d1, Z)) - res @ precision
        # clamp in case log-prob isn't concave in eta; this keeps the hessian positive-definite (cf. fisher-scoring)
        w = (-d2).clamp(min=0)
        hess = torch.zeros((num_groups, num_res, num_res), dtype=Z.dtype, device=Z.device).index_add(
            0, group_idx, torch.einsum('nj,njk,njl->nkl', w, Z, Z)
        )
        return grad, hess + precision

    @torch.no_grad()
    def solve(self,
              loglik: LogLikFun,
              eta_fixed: torch.Tensor,
              Z: torch.Tensor,
              group_idx: torch.Tensor,
              precision: torch.Tensor,
              res_init: Optional[torch.Tensor] = None,
              num_groups: Optional[int] = None) -> torch.Tensor:
        """
        Find the posterior mode of the random-effects.

        :param loglik: A function that takes the linear-predictor and returns per-row log-probs.
        :param eta_fixed: The linear-predictor(s) from the fixed-effects, shape (num_obs, num_eta).
        :param Z: The random-effects model-matrix, shape (num_obs, num_eta, num_res): ``Z[i, j]`` maps the stacked
         random-effects to the j-th linear-predictor of row i. If ``num_eta`` is 1, can be (num_obs, num_res).
        :param group_idx: Integer group-index for each row, in ``range(num_groups)``.
        :param precision: The prior precision of the random-effects, shape (num_res, num_res).
        :param res_init: Optional initial values (i.e. warm-start), shape (num_groups, num_res).
        :param num_groups: Number of groups, defaults to ``group_idx.max() + 1``.
        :return: The random-effects, shape (num_groups, num_res).
        """
        eta_fixed, precision = eta_fixed.detach(), precision.detach()
        if num_groups is None:
            num_groups = int(group_idx.max()) + 1
        if res_init is None:
            res = torch.zeros((num_groups, Z.shape[-1]), dtype=Z.dtype, device=Z.device)
        else:
            res = res_init.detach().clone()

        eps = torch.finfo(res.dtype).eps
        for _ in range(self.max_iter):
            log_probs, d1, d2 = self.derivatives(loglik, self.get_eta(eta_fixed, Z, group_idx, res))
            grad, hess = self._group_grad_and_hess(Z, group_idx, d1, d2, res, precision)
            step = torch.cholesky_solve(grad.unsqueeze(-1), _safe_cholesky(hess)).squeeze(-1)
            if not self.line_search:
                res = res + step
                continue

            # freeze converged groups:
            obj = self._group_objective(log_probs, res, group_idx, precision)
            noise = 10 * eps * (1 + obj.abs())
            decrement = .5 * (grad * step).sum(1)
            converged = decrement <= torch.maximum(self.tol * (1 + obj.abs()), noise)
            if converged.all():
                break
            step[converged] = 0

            # backtracking line-search (per-group), tolerating changes too small to resolve:
            step_size = torch.ones((num_groups, 1), dtype=res.dtype, device=res.device)
            for _ in range(self.max_halvings):
                new_res = res + step_size * step
                new_obj = self._group_objective(
                    loglik(self.get_eta(eta_fixed, Z, group_idx, new_res)).reshape(-1),
                    new_res,
                    group_idx,
                    precision
                )
                worse = ~(new_obj >= obj - noise)  # also catches nans
                if not worse.any():
                    break
                step_size[worse] /= 2
            res = res + step_size * step
        else:
            if self.max_iter > 1:
                warn(f"{type(self).__name__} did not converge in {self.max_iter} iterations.")
        return res

    def laplace_log_prob(self,
                         loglik: LogLikFun,
                         eta_fixed: torch.Tensor,
                         Z: torch.Tensor,
                         group_idx: torch.Tensor,
                         raneff_cov_chol: torch.Tensor,
                         res_init: Optional[torch.Tensor] = None,
                         num_groups: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Laplace approximation of the marginal log-likelihood, per group:

        ``log p(y_g|b_g) + log N(b_g; 0, cov) - .5 * log|H_g| + (k/2) * log(2 * pi)``, evaluated at the mode ``b_g``.

        This is differentiable wrt the fixed-effects (via ``loglik``/``eta_fixed``) and the covariance. The mode is
        found without gradients, then a single differentiable Newton-step is taken from it; at the mode this step is
        zero, but its derivative is the implicit derivative of the mode wrt the parameters.

        :param raneff_cov_chol: Cholesky factor of the random-effects covariance.
        :return: Tuple of log-probs (one per group), and the (detached) random-effects (num_groups, num_res).
        """
        precision = torch.cholesky_inverse(raneff_cov_chol)
        cov_logdet = 2 * raneff_cov_chol.diagonal().log().sum()

        res_mode = self.solve(
            loglik=loglik,
            eta_fixed=eta_fixed,
            Z=Z,
            group_idx=group_idx,
            precision=precision,
            res_init=res_init,
            num_groups=num_groups
        )

        # differentiable newton-step from the mode:
        _, d1, d2 = self.derivatives(loglik, self.get_eta(eta_fixed, Z, group_idx, res_mode), create_graph=True)
        grad, hess = self._group_grad_and_hess(Z, group_idx, d1, d2, res_mode, precision)
        res = res_mode + torch.cholesky_solve(grad.unsqueeze(-1), _safe_cholesky(hess)).squeeze(-1)

        # laplace approx:
        log_probs, _, d2 = self.derivatives(loglik, self.get_eta(eta_fixed, Z, group_idx, res), create_graph=True)
        _, hess = self._group_grad_and_hess(Z, group_idx, d1, d2, res, precision)
        hess_logdet = 2 * _safe_cholesky(hess).diagonal(dim1=-2, dim2=-1).log().sum(-1)
        log_probs = self._group_objective(log_probs.reshape(-1), res, group_idx, precision)
        log_probs = log_probs - .5 * cov_logdet - .5 * hess_logdet
        return log_probs, res_mode


class GaussianReSolver(ReSolver):
    """
    With a gaussian log-likelihood and identity-link, the log-posterior of the random-effects is quadratic, so a single
    Newton step from any starting point is the closed-form solution.
    """
    line_search = False

    def __init__(self, **kwargs):
        kwargs['max_iter'] = 1
        super().__init__(**kwargs)


def _is_identity(link: Callable) -> bool:
    return link is transforms.identity_transform or (
            isinstance(link, transforms.ComposeTransform) and not link.parts
    )


_solver_registry: List[Tuple[Callable[[Family], bool], Type[ReSolver]]] = []


def register_solver(predicate: Callable[[Family], bool], solver_cls: Type[ReSolver]):
    """
    Register a ``ReSolver`` subclass to be used for families where ``predicate(family)`` is true. Later registrations
    take precedence.
    """
    _solver_registry.append((predicate, solver_cls))


def get_solver_cls(family: Family) -> Type[ReSolver]:
    for predicate, solver_cls in reversed(_solver_registry):
        if predicate(family):
            return solver_cls
    return ReSolver


register_solver(
    lambda family: (
            type(family) is Family  # e.g. not SurvivalFamily, where censoring makes the log-lik non-quadratic
            and issubclass(family.distribution_cls, (distributions.Normal, distributions.LogNormal))
            and _is_identity(family.params_and_links[family.params[0]])
    ),
    GaussianReSolver
)


def _as_3d(Z: torch.Tensor) -> torch.Tensor:
    return Z.unsqueeze(1) if Z.dim() == 2 else Z


def _safe_cholesky(mat: torch.Tensor, max_tries: int = 5) -> torch.Tensor:
    eps = 1e-6 * mat.diagonal(dim1=-2, dim2=-1).abs().mean()
    eye = torch.eye(mat.shape[-1], dtype=mat.dtype, device=mat.device)
    for i in range(max_tries + 1):
        L, info = torch.linalg.cholesky_ex(mat + (eps * 10 ** i if i else 0) * eye)
        if not info.any():
            return L
    raise RuntimeError("Matrix not positive-definite even after adding jitter.")
