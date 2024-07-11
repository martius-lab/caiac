import torch
from torch import nn
import numpy as np
from torch.nn import functional as F


class GaussianLikelihoodHead(nn.Module):
    def __init__(self, in_features, out_features,
                 initial_var=1, min_var=1e-8, max_var=100,
                 mean_scale=1, var_scale=1,
                 use_spectral_norm_mean=False,
                 use_spectral_norm_var=False):
        super().__init__()
        assert min_var <= initial_var <= max_var

        self.min_var = min_var
        self.max_var = max_var
        self.init_var_offset = np.log(np.exp(initial_var - min_var) - 1)

        self.mean_scale = mean_scale
        self.var_scale = var_scale

        if use_spectral_norm_mean:
            self.mean = nn.utils.spectral_norm(
                nn.Linear(in_features, out_features))
        else:
            self.mean = nn.Linear(in_features, out_features)

        if use_spectral_norm_var:
            self.var = nn.utils.spectral_norm(
                nn.Linear(in_features, out_features))
        else:
            self.var = nn.Linear(in_features, out_features)

    def forward(self, inp):
        mean = self.mean(inp) * self.mean_scale
        var = self.var(inp) * self.var_scale

        var = F.softplus(var + self.init_var_offset) + self.min_var
        var = torch.clamp(var, self.min_var, self.max_var)

        return mean, var
