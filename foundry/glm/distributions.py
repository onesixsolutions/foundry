from typing import Optional, Union

import torch
from torch import distributions
from torch.distributions import constraints, Exponential, Weibull, Poisson, Uniform
from torch.distributions.utils import (
  broadcast_all,
  lazy_property
)

from foundry.util import log1mexp


class NegativeBinomial(distributions.NegativeBinomial):
    def __init__(self,
                 loc: torch.Tensor,
                 dispersion: torch.Tensor,
                 validate_args: Optional[bool] = None):
        self.loc, self.dispersion = broadcast_all(loc, dispersion)
        super().__init__(
            total_count=self.dispersion,
            logits=self.loc.log() - self.dispersion.log(),
            validate_args=validate_args
        )
        
    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(NegativeBinomial, _instance)
        batch_shape = torch.Size(batch_shape)
        loc = self.loc.expand(batch_shape)
        dispersion = self.dispersion.expand(batch_shape)
        NegativeBinomial.__init__(new, loc=loc, dispersion=dispersion, validate_args=False)
        new._validate_args = self._validate_args
        return new


class _MultinomialStrict(constraints.Constraint):
    is_discrete = True
    event_dim = 1

    def __init__(self, upper_bound: Optional[torch.Tensor]):
        self.upper_bound = upper_bound

    def check(self, x: torch.Tensor) -> bool:
        check = (x >= 0).all(dim=-1)
        if self.upper_bound is not None:
            check &= (x.sum(dim=-1) == self.upper_bound).all()
        return check


class Multinomial(distributions.Multinomial):
    """
    In torch's Multinomial, ``total_count`` is used:

    1) Only for validation in log_prob (which unnecessarily enforces a limitation that total_count be a single value).
    2) Critically in ``sample``, where sampling for a tensor of total-counts is limited by
     https://github.com/pytorch/pytorch/issues/42407.
    3) In mean, variance, etc. where total_count can be a tensor without issue.

    In this implementation, ``total_count`` is instead allowed to be a tensor, except when using ``sample``.
    """

    def __init__(self,
                 total_count: Optional[torch.Tensor] = None,
                 probs: Optional[torch.Tensor] = None,
                 logits: Optional[torch.Tensor] = None,
                 validate_args: Optional[bool] = None):
        self._categorical = torch.distributions.Categorical(probs=probs, logits=logits)
        batch_shape = self._categorical.batch_shape
        event_shape = self._categorical.param_shape[-1:]
        if isinstance(total_count, torch.Tensor) and total_count.shape != batch_shape:
            raise ValueError(
                f"If ``total_count`` is a tensor, it must have shape==probs/logits.shape[:-1] ({batch_shape})"
            )
        self.total_count: Optional[Union[torch.Tensor, int]] = total_count
        super(distributions.Multinomial, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=True, event_dim=1)
    def support(self):
        return _MultinomialStrict(self.total_count)

    def sample(self, sample_shape=torch.Size(), total_count: Optional[int] = None):
        old_total_count = self.total_count
        try:
            if total_count is not None:
                self.total_count = total_count
            if not isinstance(self.total_count, int):
                raise NotImplementedError('inhomogeneous total_count is not supported')
            out = super().sample(sample_shape=sample_shape)
        finally:
            self.total_count = old_total_count
        return out


def exp_log_surv(self: Exponential, value: torch.Tensor) -> torch.Tensor:
    value, rate = broadcast_all(value, self.rate)
    log_surv = torch.zeros_like(value)
    nonzero = ~torch.isclose(value, torch.zeros_like(value))
    log_surv[nonzero] = -value[nonzero] * rate[nonzero]
    return log_surv


Exponential.log_surv = exp_log_surv


def weibull_log_surv(self: Weibull, value: torch.Tensor) -> torch.Tensor:
    value, scale, concentration = broadcast_all(value, self.scale, self.concentration)
    log_surv = torch.zeros_like(value)
    nonzero = ~torch.isclose(value, torch.zeros_like(value))
    log_surv[nonzero] = -((value[nonzero] / scale[nonzero]) ** concentration[nonzero])
    return log_surv


Weibull.log_surv = weibull_log_surv


class CeilingWeibull(Weibull):
    arg_constraints = {
        'scale': constraints.positive,
        'concentration': constraints.positive,
        'ceiling': constraints.unit_interval
    }

    def __init__(self,
                 scale: torch.Tensor,
                 concentration: torch.Tensor,
                 ceiling: Union[torch.Tensor, float] = 1.,
                 validate_args: Optional[bool] = None):
        scale, concentration, self.ceiling = broadcast_all(scale, concentration, ceiling)
        super().__init__(
            scale=scale,
            concentration=concentration,
            validate_args=validate_args
        )

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        _, ceiling = broadcast_all(value, self.ceiling)
        return super().log_prob(value) + ceiling.log()

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        _, ceiling = broadcast_all(value, self.ceiling)
        return ceiling * super().cdf(value)

    def log_cdf(self, value: torch.Tensor) -> torch.Tensor:
        _, ceiling = broadcast_all(value, self.ceiling)
        return log1mexp(super().log_surv(value)) + ceiling.log()

    def log_surv(self, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def expand(self, batch_shape, _instance=None):
        raise NotImplementedError

    @property
    def mean(self):
        if (self.ceiling < 1).any():
            raise NotImplementedError("Mean not implemented when ceiling < 1")
        return super().mean

    @property
    def variance(self):
        if (self.ceiling < 1).any():
            raise NotImplementedError("Variance not implemented when ceiling < 1")
        return super().variance
      
def _broadcast_shape(*shapes, **kwargs):
    """
    Helper borrowed from pyro source code under Apache License 2.0.
    
    Similar to ``np.broadcast()`` but for shapes.
    Equivalent to ``np.broadcast(*map(np.empty, shapes)).shape``.
    :param tuple shapes: shapes of tensors.
    :param bool strict: whether to use extend-but-not-resize broadcasting.
    :returns: broadcasted shape
    :rtype: tuple
    :raises: ValueError
    """
    strict = kwargs.pop("strict", False)
    reversed_shape = []
    for shape in shapes:
        for i, size in enumerate(reversed(shape)):
            if i >= len(reversed_shape):
                reversed_shape.append(size)
            elif reversed_shape[i] == 1 and not strict:
                reversed_shape[i] = size
            elif reversed_shape[i] != size and (size != 1 or strict):
                raise ValueError(
                    "shape mismatch: objects cannot be broadcast to a single shape: {}".format(
                        " vs ".join(map(str, shapes))
                    )
                )
    return tuple(reversed(reversed_shape))
  
class ZeroInflatedDistribution(distributions.Distribution):
    """
    Generic zero-inflated distribution, adapted/modified from the pyro package source code (under Apache License 2.0).

    Within foundry, this is intended as a base class for
    :class:`ZeroInflatedPoisson` and :class:`ZeroInflatedNegativeBinomial`.
    
    :param torch.Tensor prob_zero: probability of extra zeros given via a Bernoulli distribution.
    :param TorchDistribution base_dist: the base distribution.
    """

    arg_constraints = {
        "prob_zero": constraints.unit_interval,
    }

    def __init__(self, prob_zero, base_dist, validate_args=None):

        batch_shape = _broadcast_shape(base_dist.batch_shape, prob_zero.shape)
        self.prob_zero = prob_zero.expand(batch_shape)

        if base_dist.event_shape:
            raise ValueError(
                "ZeroInflatedDistribution expected empty "
                "base_dist.event_shape but got {}".format(base_dist.event_shape)
            )

        self.base_dist = base_dist.expand(batch_shape)
        event_shape = torch.Size()

        super().__init__(batch_shape, event_shape, validate_args)

    @constraints.dependent_property
    def support(self):
        return self.base_dist.support

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        prob_zero, value = broadcast_all(self.prob_zero, value)
        log_prob = (-prob_zero).log1p() + self.base_dist.log_prob(value)
        log_prob = torch.where(value == 0, (prob_zero + log_prob.exp()).log(), log_prob)
        return log_prob

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            mask = torch.bernoulli(self.prob_zero.expand(shape)).bool()
            samples = self.base_dist.expand(shape).sample()
            samples = torch.where(mask, samples.new_zeros(()), samples)
        return samples


    @lazy_property
    def mean(self):
        return (1 - self.prob_zero) * self.base_dist.mean

    @lazy_property
    def variance(self):
        return (1 - self.prob_zero) * (
            self.base_dist.mean**2 + self.base_dist.variance
        ) - (self.mean) ** 2

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(type(self), _instance)
        batch_shape = torch.Size(batch_shape)
        prob_zero = self.gate.expand(batch_shape) if "prob_zero" in self.__dict__ else None
        base_dist = self.base_dist.expand(batch_shape)
        ZeroInflatedDistribution.__init__(
            new, prob_zero=prob_zero, base_dist=base_dist, validate_args=False
        )
        new._validate_args = self._validate_args
        return new



class ZeroInflatedPoisson(ZeroInflatedDistribution):
    """
    A Zero Inflated Poisson distribution.
    
    :param torch.Tensor prob_zero: probability of extra zeros.
    :param torch.Tensor rate: rate of poisson distribution.
    """

    arg_constraints = {
        "prob_zero": constraints.unit_interval,
        "rate": constraints.positive,
    }
    support = constraints.nonnegative_integer

    def __init__(self, prob_zero, rate, validate_args=None):
        
        prob_zero, = broadcast_all(prob_zero)
            
        base_dist = Poisson(rate=rate, validate_args=False)
        base_dist._validate_args = validate_args

        super().__init__(
            prob_zero=prob_zero, base_dist=base_dist, validate_args=validate_args
        )

    @property
    def rate(self):
        return self.base_dist.rate
      
class ZeroInflatedNegativeBinomial(ZeroInflatedDistribution):
    """
    A Zero Inflated Negative Binomial distribution, based on pyro implementation
    with foundry parameterization of negative binomial.
    
    :param torch.Tensor prob_zero: probability of extra zeros.
    :param torch.Tensor loc: mean of negative binomial distribution.
    :param torch.Tensor dispersion: overdispersion of negative binomial distribution
    """

    arg_constraints = {
        "prob_zero": constraints.unit_interval,
        "loc": constraints.positive,
        "dispersion": constraints.positive,
    }
    support = constraints.nonnegative_integer

    def __init__(self, prob_zero, loc, dispersion, validate_args=None):
        
        prob_zero, = broadcast_all(prob_zero)
            
        base_dist = NegativeBinomial(loc=loc, dispersion=dispersion, validate_args=False)
        base_dist._validate_args = validate_args

        super().__init__(
            prob_zero=prob_zero, base_dist=base_dist, validate_args=validate_args
        )

    @property
    def loc(self):
        return self.base_dist.loc
      
    @property
    def dispersion(self):
        return self.base_dist.dispersion

      
class HurdleDistribution(distributions.Distribution):
    """
    Hurdle base distribution, adapted/modified from the pyro package source code for 
    zero-inflated distributions (under Apache License 2.0) and formulae from Stan.
    
    :param torch.Tensor prob_zero: probability of extra zeros given via a Bernoulli distribution.
    """

    arg_constraints = {
        "prob_zero": constraints.unit_interval,
    }

    def __init__(self, prob_zero, base_dist, validate_args=None):

        batch_shape = _broadcast_shape(base_dist.batch_shape, prob_zero.shape)
        self.prob_zero = prob_zero.expand(batch_shape)

        if base_dist.event_shape:
            raise ValueError(
                "HurdleDistribution expected empty "
                "base_dist.event_shape but got {}".format(base_dist.event_shape)
            )

        self.base_dist = base_dist.expand(batch_shape)
        event_shape = torch.Size()

        super().__init__(batch_shape, event_shape, validate_args)

    @constraints.dependent_property
    def support(self):
        return self.base_dist.support
      
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        
        zeros, _ = broadcast_all(torch.Tensor([0]), value)

        prob_zero, value = broadcast_all(self.prob_zero, value)
        log_prob = (-prob_zero).log1p() + self.base_dist.log_prob(value) - (1 - torch.exp(self.base_dist.log_prob(zeros))).log()
        log_prob = torch.where(value == 0, prob_zero.log(), log_prob)
        return log_prob

    @lazy_property
    def mean(self):
        truncated_mean = self.base_dist.mean/(1 - torch.exp(self.base_dist.log_prob(0)))
        return (1 - self.prob_zero) * truncated_mean

    @lazy_property
    def variance(self):
        truncated_mean_sq = (self.base_dist.mean/(
          1 - torch.exp(self.base_dist.log_prob(0))
        ))**2
        
        truncated_variance = self.base_dist.variance/(
          1 - torch.exp(self.base_dist.log_prob(0))
        ) - torch.exp(self.base_dist.log_prob(0))*truncated_mean_sq
        
        return (1 - self.prob_zero)*(
          truncated_variance
        ) + self.prob_zero*(
          1-self.prob_zero
        )*truncated_mean_sq

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(type(self), _instance)
        batch_shape = torch.Size(batch_shape)
        prob_zero = self.gate.expand(batch_shape) if "prob_zero" in self.__dict__ else None
        base_dist = self.base_dist.expand(batch_shape)
        HurdleDistribution.__init__(
            new, prob_zero=prob_zero, base_dist=base_dist, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

class HurdlePoisson(HurdleDistribution):
    """
    A Hurdle Poisson distribution.
    
    :param torch.Tensor prob_zero: probability of zeros.
    :param torch.Tensor rate: rate of poisson distribution.
    """

    arg_constraints = {
        "prob_zero": constraints.unit_interval,
        "rate": constraints.positive,
    }
    support = constraints.nonnegative_integer

    def __init__(self, prob_zero, rate, validate_args=None):
        
        prob_zero, = broadcast_all(prob_zero)
            
        base_dist = Poisson(rate=rate, validate_args=False)
        base_dist._validate_args = validate_args

        super().__init__(
            prob_zero=prob_zero, base_dist=base_dist, validate_args=validate_args
        )

    @property
    def rate(self):
        return self.base_dist.rate

class HurdleNegativeBinomial(HurdleDistribution):
    """
    A Hurdle Negative Binomial distribution.
    
    :param torch.Tensor prob_zero: probability of extra zeros.
    :param torch.Tensor loc: mean of negative binomial distribution.
    :param torch.Tensor dispersion: overdispersion of negative binomial distribution
    """

    arg_constraints = {
        "prob_zero": constraints.unit_interval,
        "loc": constraints.positive,
        "dispersion": constraints.positive,
    }
    support = constraints.nonnegative_integer

    def __init__(self, prob_zero, loc, dispersion, validate_args=None):
        
        prob_zero, = broadcast_all(prob_zero)
            
        base_dist = NegativeBinomial(loc=loc, dispersion=dispersion, validate_args=False)
        base_dist._validate_args = validate_args

        super().__init__(
            prob_zero=prob_zero, base_dist=base_dist, validate_args=validate_args
        )

    @property
    def loc(self):
        return self.base_dist.loc
      
    @property
    def dispersion(self):
        return self.base_dist.dispersion
      
