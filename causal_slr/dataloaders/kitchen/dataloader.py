import random
import os
import d4rl
import gym
import numpy as np

from torch.utils import data
from causal_slr.utils.general_utils import AttrDict
from causal_slr.cmi import CMIScorer, CMIBuffer, MaskScorer, CodaScorer
from causal_slr.utils.data_utils import smooth_fct
from causal_slr.envs.kitchen_env import OBJ_2_IDX
from typing import Callable, Dict, Set
import pickle


class BasicDataLoader:
    def get_data_loader(self, batch_size):
        assert self.device in ['cuda', 'cpu']
        return data.DataLoader(self, batch_size=batch_size, shuffle=self.shuffle, num_workers=self.n_worker,
                               drop_last=True, pin_memory=self.device == 'cuda',
                               worker_init_fn=lambda x: np.random.seed(np.random.randint(65536) + x))


def _get_split_indices(phase, shuffle, indices, split_ratios):
    split_idx = 0 if phase == 'train' else 1
    if shuffle:  # both train and val should be shuffled!
        rng = np.random.RandomState(59942)
        rng.shuffle(indices)
    else:
        raise ValueError('Shuffle should be True')
    cum_split_ratios = np.cumsum(np.array([0] + split_ratios))
    assert cum_split_ratios[-1] == 1, 'Split ratios have to sum to 1'
    split_ends = np.floor(cum_split_ratios *
                          np.array([len(indices)] * len(cum_split_ratios)))
    split_ends = split_ends.astype(np.int64)
    split_indices = indices[split_ends[split_idx]:split_ends[split_idx + 1]]
    split_indices = np.sort(split_indices)
    return split_indices


def find_sublists_with_difference(sequence, min_length=1):
    """Given a list of ordered integer numbers, returns a list of tuples with the start and end index of sublists with difference > 1 and length >= min_length"""
    sublists = []
    start = 0
    if len(sequence) == 0:
        return sublists
    for end in range(1, len(sequence)):
        if sequence[end] - sequence[end - 1] > 1:
            if ((end - 1) - start) >= min_length - 1:
                sublists.append((sequence[start], sequence[end-1]))
            start = end
    if sequence[-1] - sequence[start] > min_length-1:
        sublists.append((sequence[start], sequence[-1]))
    return sublists


class D4RLSequenceSplitDataset(BasicDataLoader):

    def __init__(self, loader_params, data_params, phase, shuffle=True, model_cai=None):
        self.phase = phase
        self.skill_len = data_params.dataset_spec.len_skill
        self.spec = data_params.dataset_spec
        self.device = data_params.device
        self.data_conf = loader_params
        self.n_worker = 4
        self.shuffle = shuffle
        self.SPLIT = AttrDict(train=self.spec.train_split,
                              val=self.spec.val_split, test=0.0)
        dataset = self.load_dataset(data_dir=data_params.data_dir)
        self.model_cai = model_cai
        # For now even if we don't do cf cais are computed, but we don't use them, so better compute with mask since faster
        scorer_cls = MaskScorer if self.spec.scorer_cls == 'mask' else CMIScorer if self.spec.scorer_cls == 'cai' else CodaScorer if self.spec.scorer_cls == 'coda' else 'mask'
        if not scorer_cls:
            raise NotImplementedError(
                f'Couldn\'t find scorer for model_class {self.scorer_cls}')

        self.cai_computer = CMIBuffer(
            dataset, scorer_cls, self.model_cai, **self.data_conf.cai_computer_params)

        self.thr_cai = loader_params.thr_cai

        self.compute_cais()

        # split dataset into sequences
        seq_end_idxs = np.where(dataset['terminals'])[0]
        start = 0
        seqs = []
        for end_idx in seq_end_idxs:
            uncontr_seqs = self.compute_uncontr_seqs(
                start, end_idx) if self.spec.scorer_cls != 'coda' else None
            coda_cc = self.compute_coda_cc(
                start, end_idx) if self.spec.scorer_cls == 'coda' else None
            influence = self.dataset['influence'][start:end_idx +
                                                  1] if self.spec.scorer_cls != 'coda' else None

            seqs.append(AttrDict(
                states=self.dataset['observations'][start:end_idx+1],
                actions=self.dataset['actions'][start:end_idx+1],
                influence=influence,
                uncontr_seqs=uncontr_seqs,
                coda_cc=coda_cc
            ))
            start = end_idx+1

        # Ensure same shuffle at train and val
        random.Random(7839).shuffle(seqs)
        self.n_seqs = len(seqs)

        if self.phase == "train":
            self.start = 0
            self.end = int(self.SPLIT.train * self.n_seqs)
        elif self.phase == "val":
            self.start = int(self.SPLIT.train * self.n_seqs)
            self.end = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
            self.data_conf.augment_counterfactual = False

        self.len_dataloader = int(
            self.SPLIT[self.phase] * self.dataset['observations'].shape[0] - (self.end-self.start)*(self.skill_len)+1)

        self.seqs = seqs[self.start:self.end]
        self.num_accepted = self.num_rejected = 0
        print(
            f'Total number of demos {self.n_seqs}, Number of demos in {self.phase} phase: {self.end-self.start}, Num of samples {len(self)}')

    def __getitem__(self, index):
        # Sample one full demo
        demo = self._sample_demo()
        start_idx = np.random.randint(
            0, demo.states.shape[0] - self.skill_len)

        # Sample a goal idx from a geometric distribution
        geom_idx = (np.random.geometric(
            self.data_conf.p_geom, 1)-1)[0]

        # All possible goals
        goal_idxs = np.arange(start_idx+self.skill_len, demo.states.shape[0])
        goal_idx = goal_idxs[geom_idx] if geom_idx < len(
            goal_idxs) else goal_idxs[-1]

        seq = AttrDict(
            states=demo.states[start_idx:start_idx+self.skill_len],
            actions=demo.actions[start_idx:start_idx+self.skill_len],
            goals=demo.states[goal_idx][self.spec.goal_idxs]
        )

        if self.data_conf.augment_counterfactual:
            if self.spec.scorer_cls != 'coda':
                controlled_objs = set.union(
                    *demo.influence[start_idx:start_idx+self.skill_len])
                seq = self.compute_counterfactual(seq, controlled_objs)
            else:
                seq.coda_cc = demo.coda_cc[start_idx:start_idx+self.skill_len]
                seq = self.compute_counterfactual_coda(seq)

        return seq

    def _sample_demo(self):
        idx = np.random.randint(len(self.seqs))
        return self.seqs[idx]

    def __len__(self):
        return self.len_dataloader

    def compute_cais(self):
        print('\n\n Computing scores\n\n*****')
        self.cai_computer.compute_scores(batch_size=1000)
        self.dataset = self.cai_computer._buffer
        if self.spec.scorer_cls == 'coda':
            return
        self.dataset['influence'] = [set()
                                     for x in range(len(self.dataset['actions']))]
        for id, obj in enumerate(OBJ_2_IDX.keys()):
            cai_obj = smooth_fct(
                self.dataset[f'score_{obj}'], kernel_size=self.data_conf.smooth_cai_kernel)
            self.dataset[f'cai_{obj}'] = cai_obj
            for i in np.where(cai_obj > self.thr_cai)[0]:
                self.dataset['influence'][i].add(obj)

    def compute_uncontr_seqs(self, start_idx, end_idx):
        # Given a demo (starting at start_idx and end_idx) returns a dict with keys the objects, and values: a list of tuples with start and end indexes of the demo,
        # for which the object is not under control.
        uncontr_seqs = {}
        for id, obj in enumerate(OBJ_2_IDX.keys()):
            uncontr_idxs = np.where(
                self.dataset[f'cai_{obj}'][start_idx: end_idx + 1] < (self.thr_cai-0.05))[0]
            uncontr_seqs[obj] = find_sublists_with_difference(
                uncontr_idxs, min_length=self.skill_len+1)  # add the +1 so we don't have issues later when sampling counterfactual
        return uncontr_seqs

    def compute_counterfactual(self, seq, controlled_objs):
        """ Compute a counterfactual for sequence seq.
        """
        uncontrolled_objs = set(OBJ_2_IDX) - controlled_objs

        counterfactual_states = seq.states.copy()  # array skill_len x dim state
        counterfactual_goal = seq.goals.copy()
        for obj in uncontrolled_objs:
            if np.random.rand() < self.data_conf.prob_counterfactual:
                demo = self._sample_demo()
                start_idx, end_idx = demo.uncontr_seqs[obj][np.random.randint(
                    len(demo.uncontr_seqs[obj]))]
                idx = np.random.randint(start_idx, end_idx - self.skill_len+1)
                obj_counterfactual = demo.states[idx:idx +
                                                 self.skill_len, OBJ_2_IDX[obj]]
                counterfactual_states[:, OBJ_2_IDX[obj]] = obj_counterfactual
                counterfactual_goal[OBJ_2_IDX[obj] -
                                    self.spec.goal_idxs[0]] = obj_counterfactual[-1]

        counterfactual = AttrDict(
            states=counterfactual_states,
            actions=seq.actions.copy(),
            goals=counterfactual_goal.copy(),
        )
        return counterfactual


    def compute_counterfactual_coda(self, seq):
        """ Compute a counterfactual for sequence seq using CODA.
        Finds connected components within a sequence. Samples a new sequence, finds connected components.
        Computes intersection of connected components and swaps the objects in the intersection.
        """

        # action component, not to swap if action is the entity since we would need to swap the action too
        action_cc = self.model_cai.get_input_index('action')
        counterfactual_states = seq.states.copy()  # array skill_len x dim state
        counterfactual_goal = seq.goals.copy()
        # list set of tuples: [ {(0,1), (2,3), (4,5)}, {(0,1), (2,3), (4,), (5,)}, ...]
        cc_seq = seq.coda_cc
        # find common connected components in seq1 {(0,1), (2,3)}
        cc_seq = set.intersection(*list(cc_seq))
        # remove the ccs that contain the action component
        cc_seq = [i for i in list(cc_seq) if action_cc not in i]
        swapped = False
        for subgraph in cc_seq:
            demo = self._sample_demo()
            start_idx = np.random.randint(
                0, demo.states.shape[0] - self.skill_len)
            seq2 = AttrDict(
                states=demo.states[start_idx:start_idx+self.skill_len],
                coda_cc=demo.coda_cc[start_idx:start_idx+self.skill_len]
            )
            cc_seq2 = seq2.coda_cc
            # find common connected components in seq2
            cc_seq2 = set.intersection(*list(cc_seq2))
            if not subgraph in cc_seq2:  # no common subgraph structure
                continue
            for transf_idx in subgraph:  # iterate over all nodes in the subgraph
                # transformer idx to object name
                k = list(self.model_cai.inp_projs)[transf_idx]
                counterfactual_states[:, OBJ_2_IDX[k]
                                      ] = seq2.states[:,  OBJ_2_IDX[k]]
                counterfactual_goal[OBJ_2_IDX[k] - self.spec.goal_idxs[0]
                                    ] = seq2.states[-1,  OBJ_2_IDX[k]][-1]
                swapped = True

        if not swapped:
            # No common subraphs, return original seq
            seq.pop('coda_cc')
            return seq
        counterfactual = AttrDict(
            states=counterfactual_states,
            actions=seq.actions.copy(),
            goals=counterfactual_goal.copy(),
        )

        return counterfactual

    def compute_coda_cc(self, start_idx, end_idx):
        from scipy.sparse.csgraph import connected_components
        coda_masks = (
            self.dataset['mask'][start_idx:end_idx] > self.thr_cai).astype(int)
        # Add dummy components for computing connected components via scipy --> (11,10): (11,11)
        coda_masks = np.concatenate(
            [coda_masks, np.zeros((*coda_masks.shape[:2], 1))], axis=-1)

        # Compute connected components for each mask:
        # samples = tqdm.trange(0, len(coda_masks) ,
        #                 desc='Computing connected components ...')
        coda_cc = []
        for sample in range(len(coda_masks)):
            # connected_components converts a mask into a list of CC indices tuples.
            # e.g., if mask is [[1,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,1,0]] --> cc = {(0,), (1,), (2,3)}
            m1 = coda_masks[sample]
            num_ccs, cc_idxs = connected_components(m1)
            cc = [np.where(cc_idxs == i)[0] for i in range(num_ccs)]
            # get set of cc: {(0,), (1,), (2,3))}
            cc = set(tuple(i) for i in cc)
            coda_cc.append(cc)
        coda_cc = np.array(coda_cc)
        # array demo_len x set of tuples: [{(0,), (1,), (2,3))}, ... , {...}]
        return coda_cc

    def load_dataset(self, data_dir):
        print('Loading dataset from {}'.format(
            os.path.join(data_dir, self.spec.dataset_name)))
        dataset = np.load(os.path.join(
            data_dir, self.spec.dataset_name), allow_pickle=True).reshape(-1)[0]  # dict
        assert set(('terminals', 'observations', 'actions')
                   ).issubset(set(dataset.keys()))

        if self.spec.remove_goal:
            print('check dimensionality of data, maybe goal is already removed')
            dataset['observations'] = dataset['observations'][...,
                                                              :int(self.spec.state_dim)]
        return dataset


class FactorizedForwardDataset(BasicDataLoader):
    """Dataset with factorized grouping of the state

    Uses the full next state as target.
    """

    def __init__(self,
                 path: str,
                 factorizer: Callable[[np.ndarray], Dict[str, np.ndarray]],
                 target_factorizer: Callable[[np.ndarray],
                                             Dict[str, np.ndarray]] = None,
                 state_dim=None,
                 use_state_diff_as_target=True,
                 target_keys_postfix='',
                 unwrap_target=False,
                 target_scale=None,
                 phase: str = 'train',
                 split_ratios: list = None,
                 normalize_data=False,
                 device=None,
                 shuffle=True,
                 expert_percent=1.,
                 random_percent=1.,
                 shrink_dataset=1.,):
        """
        :param path: Path to dataset
        :param factorizer: Callable that returns dictionary of named
            state groups
        :param target_factorizer: Callable that returns dictionary of named
            state groups for target variable. If `None`, use `factorizer`.
        :param use_state_diff_as_target: If `True`, return difference
            between state and next state as target

        :param target_keys_postfix: String to append to keys of the target
        :param target_scale: Scalar value that target gets multiplied with
        :param unwrap_target: Turn target dictionary into vector
        :param filtered_idxs: List. List of all idxs from data that are valid
        :param split_ratios: list. List accounting for percentage of data to be used (sum must be 1) 
        :param split_idx: int. Default 0. Index for the split_ratios list. If zero used split_ratios[0] percentage of data.

        """
        print('Creating Dataloader')
        super().__init__()
        original_memory = get_data(path, expert_percent=expert_percent,
                                   random_percent=random_percent, shrink_dataset=shrink_dataset)
        self.state_dim = state_dim
        self.original_memory, filtered_idxs = preprocess_data(
            original_memory, exclude_last_idx=True)
        self.phase = phase
        self._factorizer = factorizer
        self.device = device
        self._target_factorizer = target_factorizer if target_factorizer is not None else factorizer

        self._use_state_diff_as_target = use_state_diff_as_target
        self._target_keys_postfix = target_keys_postfix
        self._unwrap_target = unwrap_target
        self._target_scale = target_scale if target_scale is not None else 1.0
        self.keys_current_state = {}
        self.keys_next_state = {}
        self.shuffle = shuffle
        self.n_worker = 4
        self.key_map = {'s0': 'observations',
                        's1': 'observations', 'a': 'actions'}
        for required_key, given_key in self.key_map.items():
            if required_key.endswith('0'):
                self.keys_current_state[required_key] = given_key
            elif required_key.endswith('1'):
                self.keys_next_state[required_key] = given_key
            else:
                self.keys_current_state[required_key] = given_key

        # Indxs of data to be used accounting for: filtered_idxs and split_ratio.
        self.valid_idxs = _get_split_indices(phase=phase, shuffle=self.shuffle,
                                             indices=filtered_idxs, split_ratios=split_ratios)

        # dict with keys: s0, s1, a) and values np.array of shape (valid_idxs, dim)
        self._memory = {key: np.stack([self.get_filtered_data(idx)[key]
                                       for idx in self.valid_idxs])
                        for key in self.required_keys
                        }

        if normalize_data:

            inp = self._memory['s0']
            self.norm_params_input = {'mean': np.mean(
                inp, axis=0), 'std': np.std(inp, axis=0)}
            assert len(
                self.norm_params_input) == 2, 'Couldnt compute mean or std for normalization'
            assert all([self.norm_params_input[k].shape[0] == inp.shape[-1]
                       for k in self.norm_params_input.keys()])

            if self._use_state_diff_as_target:
                target = self._memory['s1']-self._memory['s0']
                self.norm_params_target = {'mean': np.mean(
                    target, axis=0), 'std': np.std(target, axis=0)}
                assert len(
                    self.norm_params_target) == 2, 'Couldnt compute mean or std for normalization'
            else:
                self.norm_params_target = self.norm_params_input
        else:
            shape = self._memory['s0'].shape[-1]
            self.norm_params_input = {
                'mean': np.zeros((1, shape)), 'std': np.ones((1, shape))}
            self.norm_params_target = {
                'mean': np.zeros((1, shape)), 'std': np.ones((1, shape))}

        self.norm_params_input_fact = {}
        self.norm_params_input_fact['mean'] = self._factorizer(
            self.norm_params_input['mean'])
        self.norm_params_input_fact['std'] = self._factorizer(
            self.norm_params_input['std'])

        self.norm_params_target_fact = {}
        self.norm_params_target_fact['mean'] = self._target_factorizer(
            self.norm_params_target['mean'])
        self.norm_params_target_fact['std'] = self._target_factorizer(
            self.norm_params_target['std'])

        # Setting to original name
        self.norm_params_target = self.norm_params_target_fact
        self.norm_params_input = self.norm_params_input_fact
        del self.norm_params_target_fact
        del self.norm_params_input_fact

        self.dataset = self.get_final_dataset()
        print('Done creating dataloader.')

    def get_final_dataset(self):
        """Returns the dataset with inputs and outputs"""
        dataset = AttrDict()

        input = self._memory['s0']

        if self._use_state_diff_as_target:
            output = self._memory['s1']-self._memory['s0']
        else:
            output = self._memory['s1']

        dataset = AttrDict(
            states=input,
            outputs=output
        )
        actions = self._memory['a']
        dataset.actions = actions

        return dataset

    def get_filtered_data(self, idx: int, step_size=1) -> Dict[str, np.ndarray]:
        filt_data = {}
        for data_key, source_key in self.keys_current_state.items():
            filt_data[data_key] = self.original_memory[source_key][idx]
        for data_key, source_key in self.keys_next_state.items():
            filt_data[data_key] = self.original_memory[source_key][idx+step_size]

        return filt_data

    @property
    def shapes(self):
        example = self[0]
        inp_shapes = {name: list(val.shape)
                      for name, val in example[0].items()}

        if self._unwrap_target:
            target_shapes = example[1].shape
        else:
            target_shapes = {name: list(val.shape)
                             for name, val in example[1].items()}

        return AttrDict({'input': inp_shapes, 'output': target_shapes})

    @property
    def required_keys(self) -> Set[str]:
        keys = {'s0', 's1', 'a'}
        return keys

    def __len__(self):
        return len(self._memory['s0'])

    def __getitem__(self, idx: int):
        states = self.dataset.states[idx]
        flat_output = {'states': states}
        actions = self.dataset.actions[idx]

        fact_input = self._factorizer(states)
        fact_input['action'] = actions
        flat_output['actions'] = actions
        output = self._target_factorizer(self.dataset.outputs[idx])

        return fact_input, output, flat_output


def get_data(path, **kwargs):
    data_path = os.environ['DATA_DIR']
    if '.npy' not in path:
        data = load_mixture(data_path, path[:-3], **kwargs)
    else:
        path = os.path.join(os.environ['DATA_DIR'], path)
        print(f'Loading dataset for world model training from {path}')
        data = np.load(path, allow_pickle=True).reshape(-1,)[0]
    return data


def preprocess_data(data, exclude_last_idx=True):
    assert set(('terminals', 'observations', 'actions')
               ).issubset(set(data.keys())),  f'Missing keys observations, actions or terminals in data.keys() : {data.keys()}'

    def filter_fn(x): return np.where(~x['terminals'])[0]
    if exclude_last_idx:  # in this 1-step scenario all indexes are valid except the ones for terminal state, so when we do indx + 1, we are still in end of episode and not next one!
        entries_to_include = filter_fn(data)
    else:
        entries_to_include = np.arange(len(data['observations']) - 1)

    return data, entries_to_include


def load_mixture(data_path, env_name, expert_percent, random_percent, shrink_dataset):
    expert_data, random_data = None, None
    path_expert = f'{data_path}/expert/{env_name}/buffer.pkl'
    path_random = f'{data_path}/random/{env_name}/buffer.pkl'
    print(
        f'Loading dataset for world model training from {path_expert} and {path_random}')
    with open(path_expert, "rb") as fp:
        expert_data = pickle.load(fp)

    with open(path_random, "rb") as fp:
        random_data = pickle.load(fp)

    size_expert = expert_data['o'].shape[0]
    size_random = random_data['o'].shape[0]
    size_expert = int(size_expert*expert_percent)
    size_random = int(size_random*random_percent)
    if shrink_dataset < 1.0:
        size_expert = int(size_expert*shrink_dataset)
        size_random = int(size_random*shrink_dataset)

    T = expert_data['u'].shape[1]
    current_size = size_expert + size_random

    # Join expert and random data in one dict
    list_datasets = [[expert_data, size_expert], [random_data,
                                                  size_random]] if random_data is not None else [expert_data]
    all_data = dict()
    for k in expert_data.keys():
        all_data[k] = np.concatenate(
            list(d[k][:size, :T].astype(np.float32) for d, size in list_datasets))
    all_data['terminals'] = np.zeros(
        (current_size, T, 1), dtype=np.float32).astype(bool)
    all_data['terminals'][:, -1] = True

    final_size = current_size * T

    for k, v in all_data.items():
        all_data[k] = v.reshape(final_size, -1)
    key_map = {'o': 'observations', 'u': 'actions', 'terminals': 'terminals'}

    for old_k, new_k in key_map.items():
        all_data[new_k] = all_data.pop(old_k)
    print('Done loading data')
    return all_data
