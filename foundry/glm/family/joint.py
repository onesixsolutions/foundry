from typing import Dict, Optional, Sequence

import torch

from foundry.glm.family.family import Family
from foundry.util import to_1d


class JointDistribution:
    """
    Independent distributions for multiple outcomes (each with batch-shape ``(num_obs, 1)``). Attributes like ``mean``
    are concatenated across outcomes, giving ``(num_obs, num_outcomes)``.
    """

    def __init__(self, components: Dict[str, torch.distributions.Distribution]):
        self.components = components

    def __getattr__(self, item: str) -> torch.Tensor:
        if item == 'components':
            raise AttributeError(item)
        values = [getattr(dist, item) for dist in self.components.values()]
        if any(callable(v) for v in values):
            raise AttributeError(f"{type(self).__name__} only supports non-callable attributes like `mean`, not {item}")
        return torch.cat(values, -1)


class JointFamily(Family):
    """
    A family for multiple outcomes, each with its own family, which are independent given the predicted
    distribution-parameters. Params are named ``{outcome}__{param}``; values are a ``(num_obs, num_outcomes)`` tensor
    (in the order of ``families``), where NaN indicates an unobserved outcome. Outcome-specific values from ``y``
    (e.g. a binomial's ``total_count``) are named ``{outcome}__total_count``.

    :param families: A dictionary whose keys are outcome-names and whose values are a :class:`Family`.
    """

    def __init__(self, families: Dict[str, Family]):
        for outcome, family in families.items():
            if type(family) is not Family:
                raise NotImplementedError(f"Outcome '{outcome}': only plain ``Family`` instances supported.")
        self.families = families
        super().__init__(
            distribution_cls=JointDistribution,
            params_and_links={
                f'{outcome}__{param}': link
                for outcome, family in families.items() for param, link in family.params_and_links.items()
            },
            supports_predict_proba=False,
            from_y=[f'{outcome}__{key}' for outcome, family in families.items() for key in family.from_y]
        )
        self.has_total_count = False

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(f'{o}={f}' for o, f in self.families.items())})"

    @property
    def outcomes(self) -> Sequence[str]:
        return list(self.families)

    @property
    def first_params(self) -> Sequence[str]:
        """
        The first distribution-param of each outcome (e.g. ``loc``, ``probs``).
        """
        return [f'{outcome}__{family.params[0]}' for outcome, family in self.families.items()]

    def __call__(self, **kwargs) -> JointDistribution:
        validate_args = kwargs.pop('validate_args', None)
        components = {}
        for outcome, family in self.families.items():
            family_kwargs = {
                key: kwargs.pop(f'{outcome}__{key}')
                for key in list(family.params) + list(family.from_y) if f'{outcome}__{key}' in kwargs
            }
            if validate_args is not None:
                family_kwargs['validate_args'] = validate_args
            components[outcome] = family(**family_kwargs)
        if kwargs:
            raise TypeError(f"Unexpected arguments: {set(kwargs)}")
        return JointDistribution(components)

    def log_prob(self,
                 distribution: JointDistribution,
                 value: torch.Tensor,
                 weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Return a 1d array of log-probs, one for each row in ``value``, summed across observed outcomes.
        """
        if value.shape[-1] != len(self.families):
            raise ValueError(f"Expected value.shape[-1] to be {len(self.families)} (one per outcome)")
        observed = ~torch.isnan(value)
        log_probs = torch.zeros(value.shape[0], dtype=value.dtype, device=value.device)
        for j, dist in enumerate(distribution.components.values()):
            obs = observed[:, j:j + 1]
            if not obs.any():
                continue
            # unobserved values get a placeholder that's in the support (an observed value), so log-probs are finite:
            filled = torch.where(obs, value[:, j:j + 1], value[:, j:j + 1][obs][0])
            lp = dist.log_prob(filled)
            log_probs = log_probs + to_1d(torch.where(obs, lp, torch.zeros_like(lp)))
        if weight is not None:
            log_probs = to_1d(weight) * log_probs
        return log_probs
