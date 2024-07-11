import enum
from typing import Dict, Set
import numpy as np
import torch

from .kl_torch import (kl_div, kl_div_mixture_app)


class _KLType(enum.Enum):
    MEAN_APPROX = 1
    VAR_PROD_APPROX = 2
    VAR_PROD_APPROX_SANDWICH = 3


class CMIScorer:
    "Code adapted from: Causal Influence Detection for Improving Efficiency in Reinforcement Learning https://github.com/martius-lab/cid-in-rl "
    def __init__(self,
                 full_model: torch.nn.Module,
                 n_expectation_samples=64,
                 n_mixture_samples=64,
                 reuse_action_samples=False,
                 kl_type='var_prod_approx',
                 threshold_zero=False,
                 ):
        # np.random.seed(345)
        self._model = full_model.eval()
        self._n_expectation_samples = n_expectation_samples
        self._n_mixture_samples = n_mixture_samples
        self._reuse_action_samples = reuse_action_samples
        if reuse_action_samples:
            assert n_expectation_samples == n_mixture_samples
        self._threshold_zero = threshold_zero

        if kl_type == 'mean_approx':
            self._kl_type = _KLType.MEAN_APPROX
        elif kl_type == 'var_prod_approx':
            self._kl_type = _KLType.VAR_PROD_APPROX
        elif kl_type == 'var_prod_approx_sandwich':
            self._kl_type = _KLType.VAR_PROD_APPROX_SANDWICH
        else:
            raise ValueError(f'Unknown KL type {kl_type}')

    def _eval_model(self, states, actions):
        bs = len(states)
        n_actions = actions.shape[1]  # n_expectation_samples
        states = (states.unsqueeze(1)
                        .repeat(1, n_actions, 1)
                        .view(-1, states.shape[-1])).to(self._model.device)  # (bs * n_expectatin_samples) x dim_state
        actions = actions.view(bs * n_actions, -1).to(self._model.device)

        input = self._model.input_factorizer(states)
        input['action'] = actions

        res = self._model(input)
        means, variances = {}, {}
        for k in res[0].keys():
            means[k] = res[0][k].view(bs, n_actions, -1)
            variances[k] = res[1][k].view(bs, n_actions, -1)

        return means, variances

    @staticmethod
    # uniform distribution!! [-1,1] (range should be the same as the one used for training model)
    def _sample_actions(bs, n_actions, dim_actions):
        actions = np.random.rand(bs, n_actions, dim_actions) * 2 - 1
        return actions.astype(np.float32)

    @torch.no_grad()
    def action_scores(self, states, actions) -> dict:
        """Compute the CMI scores for each action.
        Shape `states`: Batch x DimState
        Shape `actions`: Batch x n_expectation_samples x DimAction
        Returns a dictionary with n keys (where n is the number of factorized states) 
        containing the scores for each factorized state. Scores are of shape: Batch x n_expectation_samples.
        """
        if self._model.training:
            self._model.eval()

        states = torch.from_numpy(states) if not isinstance(states, torch.Tensor) else states
        actions = torch.from_numpy(actions) if not isinstance(actions, torch.Tensor) else actions

        # Compute p(s'_j | s,a )

        # [batch_size x n_expectation_samples x dim_s'_j] for mean and var.
        means_full, vars_full = self._eval_model(states, actions)

        # Compute the p(s'|s),
        if self._reuse_action_samples:  # reuse samples to compute the p(s'|s)
            means_capped = means_full
            vars_capped = vars_full  # batch_size x n_mixture_samples x dim_s'_j
        else:
            actions = self._sample_actions(len(states),
                                           self._n_mixture_samples,
                                           actions.shape[-1])
            actions = torch.from_numpy(actions)
            means_capped, vars_capped = self._eval_model(states, actions)

        means_full_, vars_full_, means_capped_, vars_capped_ = means_full, vars_full, means_capped, vars_capped

        total_scores = {}
        # for each factorized state s'_j : dimension [batch_size x n_expectation_samples x dim_s'_j]
        for k in means_capped.keys():
            means_capped = means_capped_[k]
            vars_capped = vars_capped_[k]
            means_full = means_full_[k]
            vars_full = vars_full_[k]

            if self._kl_type == _KLType.MEAN_APPROX:
                means_capped = torch.mean(means_capped, dim=1)
                vars_capped = torch.mean(vars_capped, dim=1)
                kls = kl_div(means_full, vars_full,
                             means_capped[:, None], vars_capped[:, None])
            elif self._kl_type == _KLType.VAR_PROD_APPROX:
                kls = kl_div_mixture_app(means_full,
                                         vars_full,
                                         means_capped[:, None],
                                         vars_capped[:, None])

            elif self._kl_type == _KLType.VAR_PROD_APPROX_SANDWICH:
                kls, kls_upper = kl_div_mixture_app(means_full,
                                                    vars_full,
                                                    means_capped[:, None],
                                                    vars_capped[:, None],
                                                    return_upper_bound=True)
                kls = 0.5 * (kls + kls_upper)
            # KL [ p(s'|s,a) || p(s'|s) ] where p(s'|s) is a mix of gaussians. [batch_size x n_expectation_samples] i.e. for each state we compute KL for n_expectation_samples actions
            scores = kls

            if self._threshold_zero:
                scores = np.clip(scores, a_min=0, a_max=None)
            total_scores[f'score_{k}'] = scores.cpu().numpy()

        return total_scores

    def __call__(self, observations, actions) -> np.ndarray:
        actions = self._sample_actions(len(observations),
                                       self._n_expectation_samples,
                                       actions.shape[-1])

        # [batch_size x n_expectation_samples). KL between p(s'|s) and p(s'|s,a), for n_expectation_samples (montecarlo sampling)
        kls = self.action_scores(observations, actions)

        # Compute real CMI measure by taking expectation over n_expectation_samples actions is E_[a \in pi] [KLs]
        if isinstance(kls, dict):
            cmi = {}
            for k, v in kls.items():
                cmi[k] = np.mean(v, axis=1)

        else:
            cmi = np.mean(kls, axis=1)
        return cmi
