import os
from contextlib import contextmanager
import torch
import torch.nn as nn

from causal_slr.utils.general_utils import AttrDict

"Code in this file adapted from: Accelerating Reinforcement Learning with Learned Skill Priors: https://github.com/clvrai/spirl/tree/master"

class BaseModel(nn.Module):
    def __init__(self, logger):
        super().__init__()
        self._logger = logger

    @contextmanager
    def val_mode(self):
        """Sets validation parameters. To be used like: with model.val_mode(): ...<do something>..."""
        raise NotImplementedError(
            "Need to implement val_mode context manager in subclass!")

    def call_children(self, fn, cls):
        def conditional_fn(module):
            if isinstance(module, cls):
                getattr(module, fn).__call__()

        self.apply(conditional_fn)

    def apply_to(self, fn, cls):
        def conditional_fn(module):
            if isinstance(module, cls):
                fn(module)

        self.apply(conditional_fn)

    def step(self):
        """Provides interface for any function that should be called in every training step."""
        pass

    def build_network(self):
        raise NotImplementedError(
            "Need to implement this function in the subclass!")

    def forward(self, inputs):
        raise NotImplementedError(
            "Need to implement this function in the subclass!")

    def loss(self, model_output, inputs):
        raise NotImplementedError(
            "Need to implement this function in the subclass!")

    def log_outputs(self, model_output, inputs, losses, step, log_images, phase, **logging_kwargs):
        # Log generally useful outputs
        self._log_losses(losses, step, log_images, phase)

        if phase == 'train':
            self.log_gradients(step, phase)

    def _log_losses(self, losses, step, log_images, phase):
        for name, loss in losses.items():
            if isinstance(loss.value, torch.Tensor):
                loss.value = loss.value.detach().cpu().numpy()
            self._logger.log_scalar(loss.value, name + '_loss', step, phase)

    def _load_weights(self, weight_loading_info):
        """
        Loads weights of submodels from defined checkpoints + scopes.
        :param weight_loading_info: list of tuples: [(model_handle, scope, checkpoint_path)]
        """

        def get_filtered_weight_dict(checkpoint_path, scope):
            if os.path.isfile(checkpoint_path):
                checkpoint = torch.load(
                    checkpoint_path, map_location=self._hp.device)
                filtered_state_dict = {}
                # need to remove scope from checkpoint key
                remove_key_length = len(scope) + 1
                for key, item in checkpoint['state_dict'].items():
                    if key.startswith(scope):
                        filtered_state_dict[key[remove_key_length:]] = item
                if not filtered_state_dict:
                    raise ValueError("No variable with scope '{}' found in checkpoint '{}'!".format(
                        scope, checkpoint_path))
                return filtered_state_dict
            else:
                raise ValueError("Cannot find checkpoint file '{}' for loading '{}'.".format(
                    checkpoint_path, scope))

        print("")
        for loading_op in weight_loading_info:
            print(("=> loading '{}' from checkpoint '{}'".format(
                loading_op[1], loading_op[2])))
            filtered_weight_dict = get_filtered_weight_dict(checkpoint_path=loading_op[2],
                                                            scope=loading_op[1])
            loading_op[0].load_state_dict(filtered_weight_dict)
            print(("=> loaded '{}' from checkpoint '{}'".format(
                loading_op[1], loading_op[2])))
        print("")

    def log_gradients(self, step, phase):
        grad_norms = list([torch.norm(p.grad.data)
                          for p in self.parameters() if p.grad is not None])
        if len(grad_norms) == 0:
            return
        grad_norms = torch.stack(grad_norms)

        self._logger.log_scalar(
            grad_norms.mean().detach().cpu().numpy(), 'gradients/mean_norm', step, phase)
        self._logger.log_scalar(
            grad_norms.max().detach().cpu().numpy(), 'gradients/max_norm', step, phase)

    @staticmethod
    def _compute_total_loss(losses):
        total_loss = torch.stack([loss[1].value * loss[1].weight for loss in
                                  filter(lambda x: x[1].weight > 0, losses.items())]).sum()
        return AttrDict(value=total_loss)

    def reset(self):
        """Optional reset of internal variables, e.g. when running model as policy."""
        pass
