import torch
import torch.nn as nn
from contextlib import contextmanager

from causal_slr.utils.general_utils import AttrDict
from causal_slr.modules.recurrent_modules import LSTMClass
from causal_slr.modules.mlp import MLP
from causal_slr.modules.variational_inference import MultivariateGaussian
from causal_slr.components.base_model import BaseModel

"Code in this file adapted from: Accelerating Reinforcement Learning with Learned Skill Priors: https://github.com/clvrai/spirl/tree/master"

class SkillMdl(BaseModel):
    """Skill embedding + prior model for skill algorithm."""

    def __init__(self, params, logger=None):
        BaseModel.__init__(self, logger)
        # ProbabilisticModel.__init__(self)
        self._sample_prior = False
        self.__device = None
        self._hp = params

        self.build_network()

    def build_network(self):
        # need to decode based on state for closed-loop low-level
        assert self._hp.cond_decode
        # q(z|s_hist, a_hist) Gaussian skill inference network (conditioned on states too)
        self.q = self._build_encoder_net()

        # pi(a |s.z)
        self.decoder = self._build_decoder_net()

        # p(z|s, (g)) Gaussian prior. Returns mu, sigma (nz.vae dimensional each)
        self.learnt_prior = self._build_prior_net()

        # self.log_sigma = get_constant_parameter(0., learnable=False)

    def decode(self, z, steps, inputs):

        if not self.full_decode:
            seq_states = inputs.states
            decode_inputs = torch.cat(
                (seq_states, z[:, None].repeat(1, steps, 1)), dim=-1)  # bs x len_skill x (nz_vae + states_dim)

            # reshape so that we can apply decode
            # (bs x len_skill) x (nz_vae + states_dim)
            inp_reshape = decode_inputs.reshape(-1, decode_inputs.shape[-1])

            # reshape back to batch_size x len_skill x dim_action
            out = self.decoder(inp_reshape).reshape(
                decode_inputs.shape[0], decode_inputs.shape[1], -1)

        else:
            out = self.decoder(z)  # bs x (len_skill*dim_action)
        return out

    def _build_decoder_net(self):

        # pi(a |s,z) (closed loop since it is condicitioned on current state)
        if not self.full_decode:
            input_size = self.state_dim + self.latent_dim
            output_size = self._hp.action_dim
        else:  # pi(\hat s, \hat a |z) (decode full state and action trajectory)

            input_size = self.latent_dim

            if self.state_action_decode:
                output_size = (self._hp.action_dim +
                               self.state_dim) * self._hp.len_skill

            else:
                output_size = (self._hp.action_dim) * self._hp.len_skill

        decoder = MLP(input_size,
                      output_size,
                      hidden_dims=[self._hp.ndim_mid_dec] *
                      self._hp.n_decoder_layers,
                      weight_init=self._hp.weight_init,
                      bias_init=self._hp.bias_init,
                      outp_activation=nn.Tanh)

        return decoder

    def _build_encoder_net(self):
        # condition inference on states since decoder is conditioned on states too (q(z|s,a))
        input_size = self._hp.action_dim + self.state_dim
        # LSTM outputs mu_z, sigma_z (each of dim nz_vae) for LAST step in traj: ie. batch_size x 1 x 2 (nz_vae)
        return LSTMClass(self._hp, input_size, self.latent_dim * 2)

    def _build_prior_net(self):
        """Gaussian skill prior. p(z|s,g)
        outputs mu_z, sigma_z (nz.vae_dims for each)
        """
        prior = MLP(input_size=self.prior_input_size, output_size=self.latent_dim * 2,
                    hidden_dims=[self._hp.ndim_mid_prior] *
                    self._hp.n_prior_net_layers,
                    weight_init=self._hp.weight_init,
                    bias_init=self._hp.bias_init)
        return prior

    def _run_inference(self, inputs):
        # run inference with state sequence conditioning
        inf_input = torch.cat(
            (inputs.actions, inputs.states), dim=-1)
        # size batch_size x [nz_vae x 2] (returns mu_z, sigma_z for LAST step (after encoding all trajectory via LSTM)
        return MultivariateGaussian(self.q(inf_input))  # mu_z, sigma_z

    def forward(self, inputs, use_learned_prior=False):
        """Forward pass of the skill model.
        :arg inputs: dict with 'states', 'actions', 'images' keys from data loader
        :arg use_learned_prior: if True, decodes samples from learned prior instead of posterior, used for RL
        """
        output = AttrDict()

        # POSTERIOR: run inference. Returns multivariate Gaussian representing q(z | s_traj, a_traj)
        output.q = self._run_inference(inputs)

        # FIXED PRIOR: compute (fixed) prior p(z) = N(0,1).
        # output.p = Gaussian(torch.zeros_like(
        #     output.q.mu), torch.zeros_like(output.q.log_sigma))

        # LEARNT PRIOR: infer learned skill prior p(z | s_0, (g))
        output.learnt_prior = self.compute_learned_prior(
            self._get_input_learned_prior(inputs))

        if use_learned_prior:
            # use output of learned skill prior for sampling latent z. Done for RL
            output.p = output.learnt_prior

        # Sample z for decoding and reconstruction. While training skills, sample from the posterior q(z|s_traj,a_traj) with reparameritization and from fixed prior at validation time. For RL sample from learnt prior.
        # sample prior is False while training, True for  validation
        # sample from Gaussian using reparamerization trick!
        output.z = output.q.sample() if not self._sample_prior else output.learnt_prior.sample()

        # decode
        assert inputs.actions.shape[1] == self._hp.len_skill
        output.reconstruction = self.decode(output.z,
                                            steps=self._hp.len_skill,
                                            inputs=inputs)  # bs x len_skill x dim_action

        return output

    def loss(self, model_output, inputs):
        """Loss computation of the TACO-RL model.
        :arg model_output: output of TACO-RL model forward pass
        :arg inputs: dict with 'states', 'actions'
        """
        losses = AttrDict()
        if not self.full_decode:
            rec_mse = torch.nn.MSELoss(reduction='mean')(
                model_output.reconstruction, inputs.actions)
        else:
            if self.state_action_decode:
                groundtruth = torch.cat((torch.flatten(inputs.states, start_dim=1), torch.flatten(
                    inputs.actions, start_dim=1)), 1)  # bs x (len_skill)*dim_state + len_skill*dim_action
            elif self.action_decode:
                # bs x (len_skill)*action_state
                groundtruth = torch.flatten(inputs.actions, start_dim=1)

            rec_mse = torch.nn.MSELoss(reduction='mean')(
                model_output.reconstruction, groundtruth)

        losses.rec_mse = AttrDict(
            value=rec_mse, weight=self._hp.reconstruction_mse_weight)

        # learned skill prior net loss. KL ( q(z | s_traj, a_traj) || learnt_prior(z|s,g)).
        kl_div = self.compute_kl_loss(
            model_output.q, model_output.learnt_prior)

        losses.kl_q_learntprior = AttrDict(
            value=kl_div, weight=self._hp.kl_div_weight_lp)

        # Optionally update beta
        if self.training and self._hp.target_kl is not None:
            self._update_beta(losses.kl_loss.value)

        losses.total = self._compute_total_loss(losses)
        return losses

    def compute_kl_loss(self, posterior, prior) -> torch.Tensor:
        """
        Compute the KL divergence loss between the distributions of the plan recognition (posterior) and plan proposal (prior) network.
        We use KL balancing similar to "MASTERING ATARI WITH DISCRETE WORLD MODELS" by Hafner et al.
        (https://arxiv.org/pdf/2010.02193.pdf)

        Args:
            posterior: Distribution produced by plan recognition network.
            prior: Distribution produced by plan prorposal network.

        Returns:
            Balanced KL loss.
        """

        kl_lhs = posterior.detach().kl_divergence(
            prior).mean(-1)  # average over batches

        kl_rhs = posterior.kl_divergence(
            prior.detach()).mean(-1)  # average over batches

        alpha = self._hp.kl_balancing_mix
        kl_loss = alpha * kl_lhs + (1 - alpha) * kl_rhs
        return kl_loss

    def compute_learned_prior(self, inputs):
        """Compute learned prior p(z|s_0, (g))"""
        # inputs: batch_size x (state_dim + goal_dim)
        # outputs: Gaussian(mu_z, sigma_z)
        out = self.learnt_prior(inputs)
        return MultivariateGaussian(out)

    def _get_input_learned_prior(self, inputs):
        if not self.goal_conditioned:
            return inputs.states[:, 0]
        elif self._hp.n_objects == 0:  # non object centric approach
            return torch.cat((inputs.states[:, 0], inputs.goals), 1)

        else:  # return start and end i.e. p(z|s_0,s_end)
            return torch.cat((inputs.states[:, 0], inputs.goals), 1)

    def enc_obs(self, obs):
        """Optionally encode observation for decoder."""
        return obs

    @property
    def beta(self):
        return self._log_beta().exp()[0].detach() if self._hp.target_kl is not None else self._hp.kl_div_weight_q

    def switch_to_prior(self):
        self._sample_prior = True

    def switch_to_inference(self):
        self._sample_prior = False

    @contextmanager
    def val_mode(self):
        self.switch_to_prior()
        yield
        self.switch_to_inference()

    @property
    def latent_dim(self):
        return self._hp.nz_vae

    @property
    def state_dim(self):
        return self._hp.state_dim

    @property
    def goal_dim(self):
        return self._hp.goal_dim

    @property
    def prior_input_size(self):
        # + self._hp.n_objects
        return self.state_dim if not self._hp.goal_cond else self.state_dim + self.goal_dim

    @property
    def full_decode(self):
        return self._hp.full_decode

    @property
    def state_action_decode(self):
        return self._hp.state_action_decode

    @property
    def action_decode(self):
        return self._hp.full_decode and not self._hp.state_action_decode

    @property
    def goal_conditioned(self):
        return self._hp.goal_cond

    @property
    def device(self):
        return self.__device

    @device.setter
    def device(self, device):
        self.to(device)
        self.__device = device
