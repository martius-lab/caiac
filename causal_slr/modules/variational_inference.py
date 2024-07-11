import torch
import math
import numpy as np

from torch.distributions.multivariate_normal import MultivariateNormal
from causal_slr.cmi.kl_torch import kl_div
from causal_slr.utils.pytorch_utils import ten2ar


class Gaussian:
    """ Represents a gaussian distribution """

    def __init__(self, mu, log_sigma=None):
        """

        :param mu:
        :param log_sigma: If none, mu is divided into two chunks, mu and log_sigma
        """
        if log_sigma is None:
            if not isinstance(mu, torch.Tensor):
                import pdb
                pdb.set_trace()
            mu, log_sigma = torch.chunk(mu, 2, -1)

        self.mu = mu
        self.log_sigma = torch.clamp(log_sigma, min=-10, max=2) if isinstance(log_sigma, torch.Tensor) else \
            np.clip(log_sigma, a_min=-10, a_max=2)
        self._sigma = None

    def sample(self):
        return self.mu + self.sigma * torch.randn_like(self.sigma)

    def kl_divergence(self, other):
        """Here self=q and other=p and we compute KL(q, p).
        Assumes both distributions are diagonal gaussians.
        Check https://statproofbook.github.io/P/mvn-kl.html for derivation."""

        v1 = self.sigma
        v2 = other.sigma
        m1 = self.mu
        m2 = other.mu
        kl_div_ = kl_div(m1, v1, m2, v2)
        return kl_div_

    def nll(self, x):
        # Negative log likelihood (probability)
        return -1 * self.log_prob(x)

    def log_prob(self, val):
        """Computes the log-probability of a value under the Gaussian distribution."""
        return -1 * ((val - self.mu) ** 2) / (2 * self.sigma**2) - self.log_sigma - math.log(math.sqrt(2*math.pi))

    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + 0.5 * torch.log(self.sigma**2)

    @property
    def sigma(self):
        if self._sigma is None:
            self._sigma = self.log_sigma.exp()
        return self._sigma

    @property
    def shape(self):
        return self.mu.shape

    def detach(self):
        """Detaches internal variables. Returns detached Gaussian."""
        return type(self)(self.mu.detach(), self.log_sigma.detach())

    def to_numpy(self):
        """Convert internal variables to numpy arrays."""
        return type(self)(ten2ar(self.mu), ten2ar(self.log_sigma))


class UnitGaussian(Gaussian):
    def __init__(self, size, device):
        mu = torch.zeros(size, device=device)
        log_sigma = torch.zeros(size, device=device)
        super().__init__(mu, log_sigma)


class MultivariateGaussian(Gaussian):
    def log_prob(self, val):
        return super().log_prob(val).sum(-1)

    @property
    def dim(self):
        return self.sigma.shape[-1]

    def entropy(self):
        entropy = 0.5 * self.dim * \
            (1 + math.log(2 * math.pi)) + 0.5*torch.log(self.det)
        return entropy

    @property
    def det(self):
        return torch.prod(self.sigma, 1)  # sigma is batch_size * dim_size


class MultivariateDiagNormal(MultivariateNormal):
    def __init__(self, loc, scale, *args, **kwargs):
        cov = torch.diag_embed(scale.pow(2))
        super().__init__(loc, cov, *args, **kwargs)


def mc_kl_divergence(p, q, n_samples=1):
    """Computes monte-carlo estimate of KL divergence. n_samples: how many samples are used for the estimate."""
    samples = [p.sample() for _ in range(n_samples)]
    return torch.stack([p.log_prob(x) - q.log_prob(x) for x in samples], dim=1).mean(dim=1)
