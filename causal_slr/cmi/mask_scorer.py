import enum
from typing import Dict, Set
import numpy as np
import torch


class CodaScorer():
    def __init__(self,
                 full_model: torch.nn.Module,
                 key_source: str = 'action',
                 key_target: str = 'o',
                 **kwargs):
        assert hasattr(full_model, 'get_mask'), \
            'MaskScorer needs a model supporting getting masks with `get_mask`'
        assert hasattr(full_model, 'get_input_index'), \
            'MaskScorer needs a model with `get_input_index`'
        assert hasattr(full_model, 'get_output_index'), \
            'MaskScorer needs a model with `get_output_index`'
        self._model = full_model.eval()
        self._idx_source = full_model.get_input_index(key_source)

    def __call__(self,
                 observations: np.ndarray,
                 actions: np.ndarray):
        if self._model.training:
            self._model.eval()

        states = torch.from_numpy(observations) if not isinstance(
            observations, torch.Tensor) else observations
        actions = torch.from_numpy(actions) if not isinstance(
            actions, torch.Tensor) else actions
        states = states.to(self._model.device)
        actions = actions.to(self._model.device)
        input = self._model.input_factorizer(states)
        input['action'] = actions

        with torch.no_grad():
            # B x Num_Inp x Num_Out For example for 2 object task: [512 x 4 x 3] (inp = agent, object1, object2, action)
            mask, layers = self._model.get_mask(input)
            mask = mask.cpu().numpy()
        total_mask = {'mask': mask}
        for idx, layer in enumerate(layers):
            total_mask[f'layer_{idx}'] = layer.cpu().numpy()
        return total_mask
        

class MaskScorer(CodaScorer):
    def __call__(self,
                 observations: np.ndarray,
                 actions: np.ndarray):
        total_scores = {}
        mask = super().__call__(observations, actions)['mask']
        for key_target in self._model.outp_dims.keys():
            self._idx_target = self._model.get_output_index(key_target)

            score = mask[:, self._idx_source, self._idx_target]
            total_scores[f'score_{key_target}'] = score
        return total_scores
