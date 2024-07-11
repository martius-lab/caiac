import numpy as np
import tqdm


class CMIBuffer():
    def __init__(self,  data, scorer_cls, model, **kwargs) -> None:
        self.key_map = {'obs': 'observations',
                        'a': 'actions', 'ag': 'ag', 'g': 'g'}
        self.key_remap = {}
        self._buffer = data.copy()
        shape = next(iter(self._buffer.values())).shape
        if len(shape) > 2:
            print(f'Reshaping buffer from shape {shape} ....')
            self.buffer = {}
            for k, v in self._buffer.items():
                if k not in self.key_map.values() and k in self.key_map.keys():
                    print(f'Mapping key {k} to: {self.key_map[k]}')
                    self.key_remap[self.key_map[k]] = k
                    k = self.key_map[k]
                dim_shape = v.shape[-1]
                self.buffer[k] = v.reshape(-1, dim_shape)
            self._buffer = self.buffer
            del self.buffer
        shape = next(iter(self._buffer.values())).shape
        print(f'Final buffer shape is: {shape}')
        self._size = shape[0]
        self._next_idx = 0

        assert all([x in self._buffer.keys() for x in ['observations', 'actions']
                    ]), f'Buffer must contain observation and actions keys, but contains {self._buffer.keys()}'
        self.scorer = scorer_cls(
            model, **kwargs)

    def compute_scores(self, batch_size):
        """Recompute scores for transitions in buffer

        :param batch_size: Number of episodes to run through scorer at once

        :param verbose: Print a progress bar
        """

        indices = tqdm.trange(0,  self._size, batch_size,
                              desc='Rescoring transitions')

        for idx in indices:
            # batchsize, except if exceeding total buffer
            bs = min(idx + batch_size, self._size) - idx

            transitions = {k: self._buffer[k][idx:idx + bs]
                           for k in ['actions', 'observations']
                           }

            # transitions should be a dict with states and actions and length: (batch_size x dimension of state/action
            scores = self.scorer(**transitions)
            for k, v in scores.items():
                if k not in self._buffer.keys():
                    self._buffer[k] = np.empty((self._size, *v.shape[1:]))
                self._buffer[k][idx:idx + bs] = v

        # Remap back to original keys
        for k in self.key_remap.keys():
            self._buffer[self.key_remap[k]] = self._buffer[k]
            self._buffer.pop(k)

    def compute_single_transition_score(self, states, actions):
        scores = self.scorer(states, actions)
        return scores
