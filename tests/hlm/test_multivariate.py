"""
Multiple linear-predictors per row (e.g. a multivariate model with different families), with random-effects that
are correlated across them. These test the solver directly; there's no ``Hlm`` API for this yet.
"""
import numpy as np
import pytest
import torch

from foundry.covariance import log_chol_to_chol
from foundry.hlm.solver import ReSolver, GaussianReSolver


def _block_design(Zs: list) -> torch.Tensor:
    """
    Given one random-effects model-matrix per outcome, create the (num_obs, num_outcomes, num_res) design where each
    outcome uses its own block of the stacked random-effects.
    """
    num_obs = Zs[0].shape[0]
    num_res = sum(Z.shape[1] for Z in Zs)
    out = torch.zeros((num_obs, len(Zs), num_res), dtype=Zs[0].dtype)
    start = 0
    for j, Z in enumerate(Zs):
        out[:, j, start:start + Z.shape[1]] = Z
        start += Z.shape[1]
    return out


def _chol(cov_params: torch.Tensor, rank: int) -> torch.Tensor:
    return log_chol_to_chol(cov_params[:rank], cov_params[rank:])


@pytest.mark.parametrize('solver_cls', [GaussianReSolver, ReSolver])
def test_multivariate_gaussian_with_missing_is_exact(solver_cls):
    torch.manual_seed(1)
    num_groups, num_outcomes = 6, 2
    group_idx = torch.repeat_interleave(torch.arange(num_groups), torch.randint(1, 6, (num_groups,)))
    num_obs = len(group_idx)
    Zs = [torch.cat([torch.ones(num_obs, 1), torch.randn(num_obs, 1)], 1).double() for _ in range(num_outcomes)]
    Z = _block_design(Zs)
    num_res = Z.shape[-1]
    X = torch.randn(num_obs, 2).double()
    y = torch.randn(num_obs, num_outcomes).double()
    observed = torch.rand(num_obs, num_outcomes) > .3
    observed[:, 0] = True  # every row has at least one outcome

    beta = (.5 * torch.randn(2, num_outcomes)).double().requires_grad_()
    log_scales = (.3 * torch.randn(num_outcomes)).double().requires_grad_()
    cov_params = (.3 * torch.randn(num_res + num_res * (num_res - 1) // 2)).double().requires_grad_()

    def loglik(eta):
        lp = torch.distributions.Normal(eta, log_scales.exp()).log_prob(y)
        return torch.where(observed, lp, torch.zeros_like(lp)).sum(1)

    laplace, _ = solver_cls().laplace_log_prob(
        loglik=loglik, eta_fixed=X @ beta, Z=Z, group_idx=group_idx, raneff_cov_chol=_chol(cov_params, num_res)
    )

    # exact: stack each group's observed values into one mvnorm:
    L = _chol(cov_params, num_res)
    exact = []
    for g in range(num_groups):
        rows = group_idx == g
        obs = observed[rows]
        Zg = Z[rows][obs]  # (num_observed, num_res)
        mvnorm = torch.distributions.MultivariateNormal(
            loc=(X[rows] @ beta)[obs],
            covariance_matrix=Zg @ L @ L.T @ Zg.T + torch.diag(log_scales.exp().expand(obs.shape[0], -1)[obs] ** 2)
        )
        exact.append(mvnorm.log_prob(y[rows][obs]))
    exact = torch.stack(exact)
    np.testing.assert_allclose(laplace.detach().numpy(), exact.detach().numpy(), rtol=1e-6)

    params = [beta, log_scales, cov_params]
    for gl, ge in zip(torch.autograd.grad(laplace.sum(), params), torch.autograd.grad(exact.sum(), params)):
        np.testing.assert_allclose(gl.numpy(), ge.numpy(), rtol=1e-5, atol=1e-8)


def _simulate_visits_and_spend(num_groups: int, corr: float, seed: int = 0) -> dict:
    """
    A hurdle-like structure: each row is a period of ``num_days`` in which a player visits on some days (binomial);
    when they visit at least once, we observe their (log-normal) spend. Each player has a random-intercept for each,
    and these are correlated.
    """
    rs = np.random.RandomState(seed)
    stds = np.array([.8, .5])
    cov = np.diag(stds) @ np.array([[1, corr], [corr, 1]]) @ np.diag(stds)
    res = rs.multivariate_normal(np.zeros(2), cov, size=num_groups)
    group_idx = np.repeat(np.arange(num_groups), rs.randint(2, 12, size=num_groups))
    num_obs = len(group_idx)
    num_days = rs.randint(1, 15, size=num_obs).astype('float64')
    visits = rs.binomial(num_days.astype(int), 1 / (1 + np.exp(-(-1.5 + res[group_idx, 0]))))
    spend = np.exp(4. + res[group_idx, 1] + .7 * rs.randn(num_obs))
    return dict(
        group_idx=torch.as_tensor(group_idx),
        num_days=torch.as_tensor(num_days),
        visits=torch.as_tensor(visits, dtype=torch.float64),
        spend=torch.as_tensor(np.where(visits > 0, spend, 1.)),  # placeholder where unobserved
        observed_spend=torch.as_tensor(visits > 0),
        cov=cov,
        res=res,
    )


class _VisitSpendModel(torch.nn.Module):
    def __init__(self, data: dict, num_res: int = 2):
        super().__init__()
        self.data = data
        self.num_res = num_res
        self.bias = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        self.log_scale = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.cov_params = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
        num_obs = len(data['group_idx'])
        self.Z = _block_design([torch.ones(num_obs, 1, dtype=torch.float64)] * 2)

    def loglik(self, eta: torch.Tensor) -> torch.Tensor:
        d = self.data
        lp_visits = torch.distributions.Binomial(d['num_days'], logits=eta[:, 0]).log_prob(d['visits'])
        lp_spend = torch.distributions.LogNormal(eta[:, 1], self.log_scale.exp()).log_prob(d['spend'])
        return lp_visits + torch.where(d['observed_spend'], lp_spend, torch.zeros_like(lp_spend))

    def chol(self) -> torch.Tensor:
        return _chol(self.cov_params, self.num_res)

    def laplace(self, solver: ReSolver) -> torch.Tensor:
        eta_fixed = self.bias.expand(len(self.Z), -1)
        log_probs, _ = solver.laplace_log_prob(
            loglik=self.loglik, eta_fixed=eta_fixed, Z=self.Z, group_idx=self.data['group_idx'],
            raneff_cov_chol=self.chol()
        )
        return log_probs.sum()


def test_mixed_family_gradient_matches_finite_differences():
    data = _simulate_visits_and_spend(num_groups=30, corr=.5)
    model = _VisitSpendModel(data)
    with torch.no_grad():
        model.bias[:] = torch.tensor([-1., 3.5])
        model.cov_params[:] = torch.tensor([-.3, -.5, .2])
    solver = ReSolver(tol=0)
    params = list(model.parameters())
    grads = torch.autograd.grad(model.laplace(solver), params)
    eps = 1e-6
    for param, grad in zip(params, grads):
        for i in range(param.numel()):
            with torch.no_grad():
                param.view(-1)[i] += eps
                up = model.laplace(solver)
                param.view(-1)[i] -= 2 * eps
                down = model.laplace(solver)
                param.view(-1)[i] += eps
            np.testing.assert_allclose(grad.view(-1)[i].item(), ((up - down) / (2 * eps)).item(), rtol=1e-4, atol=1e-6)


def test_mixed_family_recovers_cross_covariance():
    data = _simulate_visits_and_spend(num_groups=1500, corr=.6)
    model = _VisitSpendModel(data)
    solver = ReSolver()
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=200, line_search_fn='strong_wolfe')
    num_obs = len(data['group_idx'])

    def closure():
        optimizer.zero_grad()
        loss = -model.laplace(solver) / num_obs
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        L = model.chol()
        cov = (L @ L.T).numpy()
    stds = np.sqrt(np.diag(cov))
    np.testing.assert_allclose(stds, np.sqrt(np.diag(data['cov'])), rtol=.15)
    assert abs(cov[0, 1] / stds.prod() - .6) < .1
    np.testing.assert_allclose(model.bias.detach().numpy(), [-1.5, 4.], atol=.1)
    assert abs(model.log_scale.exp().item() - .7) < .05
