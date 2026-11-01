import numpy as np
import pytest
import torch

from foundry.covariance import log_chol_to_chol
from foundry.glm.glm import family_from_string
from foundry.hlm.solver import ReSolver, GaussianReSolver, get_solver_cls


def _make_problem(family: str, num_groups: int = 7, num_res: int = 2, seed: int = 0):
    torch.manual_seed(seed)
    family = family_from_string(family)
    group_sizes = torch.randint(1, 8, (num_groups,))
    group_idx = torch.repeat_interleave(torch.arange(num_groups), group_sizes)
    num_obs = len(group_idx)
    Z = torch.cat([torch.ones(num_obs, 1), torch.randn(num_obs, num_res - 1)], 1).double()
    X = torch.randn(num_obs, 3).double()

    # parameters we'll differentiate wrt:
    beta = (.5 * torch.randn(3, 1)).double().requires_grad_()
    log_scale = torch.tensor(-.5, dtype=torch.float64, requires_grad=True)
    cov_params = (.3 * torch.randn(num_res + num_res * (num_res - 1) // 2)).double().requires_grad_()

    if family.distribution_cls is torch.distributions.Normal:
        y = torch.randn(num_obs, 1).double() + X @ beta.detach()
        extra = {'scale': lambda: log_scale.expand(num_obs, 1)}
        lp_kwargs = {'value': y}
    elif family.distribution_cls is torch.distributions.Binomial:
        y = torch.randint(0, 6, (num_obs, 1)).double()
        extra = {'total_count': lambda: torch.full((num_obs, 1), 5.).double()}
        lp_kwargs = {'value': y}
    else:
        y = torch.poisson(torch.ones(num_obs, 1) * 2).double()
        extra = {}
        lp_kwargs = {'value': y}
    param0 = family.params[0]

    def loglik(eta):
        return family.log_prob(family(**{param0: eta}, **{k: f() for k, f in extra.items()}), **lp_kwargs)

    def chol():
        return log_chol_to_chol(cov_params[:num_res], cov_params[num_res:])

    return dict(
        loglik=loglik, X=X, Z=Z, y=y, group_idx=group_idx, beta=beta, log_scale=log_scale, cov_params=cov_params,
        chol=chol, num_groups=num_groups
    )


def _exact_gaussian_log_prob(p: dict) -> torch.Tensor:
    cov = p['chol']() @ p['chol']().T
    out = []
    for g in range(p['num_groups']):
        mask = p['group_idx'] == g
        Zg = p['Z'][mask]
        mvnorm = torch.distributions.MultivariateNormal(
            loc=(p['X'][mask] @ p['beta']).squeeze(-1),
            covariance_matrix=Zg @ cov @ Zg.T + torch.eye(len(Zg)).double() * p['log_scale'].exp() ** 2
        )
        out.append(mvnorm.log_prob(p['y'][mask].squeeze(-1)))
    return torch.stack(out)


@pytest.mark.parametrize('solver_cls', [GaussianReSolver, ReSolver])
def test_gaussian_laplace_is_exact(solver_cls):
    p = _make_problem('gaussian')
    laplace, _ = solver_cls().laplace_log_prob(
        loglik=p['loglik'], eta_fixed=p['X'] @ p['beta'], Z=p['Z'], group_idx=p['group_idx'],
        raneff_cov_chol=p['chol']()
    )
    exact = _exact_gaussian_log_prob(p)
    np.testing.assert_allclose(laplace.detach().numpy(), exact.detach().numpy(), rtol=1e-6)

    # gradients wrt fixed-effects, residual-scale, and covariance should also match:
    params = [p['beta'], p['log_scale'], p['cov_params']]
    grads_laplace = torch.autograd.grad(laplace.sum(), params)
    grads_exact = torch.autograd.grad(exact.sum(), params)
    for gl, ge in zip(grads_laplace, grads_exact):
        np.testing.assert_allclose(gl.numpy(), ge.numpy(), rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize('family', ['binomial', 'poisson'])
def test_laplace_gradient_matches_finite_differences(family: str):
    """
    The one-differentiable-newton-step trick should give the exact gradient of the laplace-approximation (where the
    mode is re-solved for each parameter value).
    """
    p = _make_problem(family)
    solver = ReSolver(tol=0)

    def laplace():
        return solver.laplace_log_prob(
            loglik=p['loglik'], eta_fixed=p['X'] @ p['beta'], Z=p['Z'], group_idx=p['group_idx'],
            raneff_cov_chol=p['chol']()
        )[0].sum()

    params = [p['beta'], p['cov_params']]
    grads = torch.autograd.grad(laplace(), params)
    eps = 1e-6
    for param, grad in zip(params, grads):
        for i in range(param.numel()):
            with torch.no_grad():
                param.view(-1)[i] += eps
                up = laplace()
                param.view(-1)[i] -= 2 * eps
                down = laplace()
                param.view(-1)[i] += eps
            np.testing.assert_allclose(grad.view(-1)[i].item(), ((up - down) / (2 * eps)).item(), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize('family', ['binomial', 'poisson'])
def test_solve_finds_mode(family: str):
    p = _make_problem(family)
    precision = torch.cholesky_inverse(p['chol']()).detach()
    eta_fixed = (p['X'] @ p['beta']).detach()
    res = ReSolver(tol=0).solve(  # i.e. converge to the precision of the dtype
        loglik=p['loglik'], eta_fixed=eta_fixed, Z=p['Z'], group_idx=p['group_idx'], precision=precision
    )
    _, d1, d2 = ReSolver.derivatives(p['loglik'], ReSolver.get_eta(eta_fixed, p['Z'], p['group_idx'], res))
    grad, _ = ReSolver._group_grad_and_hess(p['Z'], p['group_idx'], d1, d2, res, precision)
    assert grad.abs().max() < 1e-5


def test_solver_registry():
    assert get_solver_cls(family_from_string('gaussian')) is GaussianReSolver
    assert get_solver_cls(family_from_string('lognormal')) is GaussianReSolver
    assert get_solver_cls(family_from_string('binomial')) is ReSolver
    assert get_solver_cls(family_from_string('weibull', y={'value': None, 'right_censoring': None})) is ReSolver
