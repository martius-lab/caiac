import hashlib
import pickle
import numpy as np
from causal_slr.cmi import CMIScorer, CMIBuffer, MaskScorer, CodaScorer
from causal_slr.utils.data_utils import smooth_fct
from causal_slr.utils.general_utils import get_data_dir
import os
import torch
from causal_slr.utils.general_utils import rec_map_dict


class ReplayBuffer:

    def __init__(self, args, buffer_size, sample_func, model_cai=None, reward_func=None, env_obj_2_idx=None, env_obs_2_ag=None, env_input_factorizer=None):
        self.args = args
        self.T = args.env_max_timesteps
        self.size = buffer_size // self.T
        self.sample_func = sample_func
        self.current_size = 0
        assert self.args.relabel_goal_coda != self.args.relabel_goal_ours, 'Cannot relabel cf with both "ours" and "coda"!'

        self.buffers = {
            'obs': np.empty([self.size, self.T, self.args.env_obs_size], dtype=np.float32),
            'ag': np.empty([self.size, self.T, self.args.env_goal_size], dtype=np.float32),
            'g': np.empty([self.size, self.T-1, self.args.env_goal_size], dtype=np.float32),
            'actions': np.empty([self.size, self.T-1, self.args.env_action_size], dtype=np.float32),
        }
        self.key_map = {'o': 'obs', 'ag': 'ag', 'g': 'g', 'u': 'actions'}
        self.model_cai = model_cai
        if self.model_cai:
            self.scorer_cls = MaskScorer if args.scorer_cls == 'mask' else CMIScorer if args.scorer_cls == 'cai' else CodaScorer if args.scorer_cls == 'coda' else None
        self.reward_func = reward_func
        self.env_obj_2_idx = env_obj_2_idx
        self.env_obs_2_ag = env_obs_2_ag
        self.env_input_factorizer = env_input_factorizer
        self.sinb_buffer = None  # seeing is not believing baseline
        assert min(self.args.ratio_cf,
                   self.args.ratio_dyna) == 0, f'Ratio cf {self.args.ratio_cf} or ratio_dyna {self.args.ratio_dyna} are incompatible'

    def sample(self, batch_size, future_p=None, agent=None):

        original_batch_size = batch_size - \
            int(self.args.ratio_cf*batch_size) - \
            int(self.args.ratio_dyna*batch_size)

        transitions = self.sample_func(
            self.buffers.copy(), original_batch_size, future_p=future_p)['transitions']

        if self.args.ratio_cf > 0:
            cf_transitions = self.sample_cf(batch_size, future_p=future_p)

            # Merge original and counterfactual transitions
            for key in cf_transitions.keys():
                transitions[key] = np.concatenate(
                    [transitions[key], cf_transitions[key]], axis=0)

        elif self.args.ratio_dyna > 0:
            dyna_transitions = self.sample_dyna(
                self.buffers.copy(), batch_size, future_p=future_p, agent=agent)
            # Merge original and counterfactual transitions
            for key in dyna_transitions.keys():
                transitions[key] = np.concatenate(
                    [transitions[key], dyna_transitions[key]], axis=0)

        else:
            # Relabel with random goals
            num_coda_relabel = int(future_p * 0.5 * batch_size)
            rand_idxs = np.random.randint(
                self.buffers['ag'].shape[0] * self.buffers['ag'].shape[1], size=num_coda_relabel)
            transitions['g'][:num_coda_relabel] = self.buffers['ag'].reshape(
                -1, self.args.env_goal_size)[rand_idxs]
            transitions['r'] = np.expand_dims(self.reward_func(
                transitions['ag'], transitions['g'], None), 1)

        assert len(transitions[key]) == batch_size
        transitions['transitions'] = transitions

        # HER CASE
        return transitions

    def sample_cf(self, batch_size, future_p=0):
        # Counterfactual samples
        cf_batch_size = int(self.args.ratio_cf*batch_size)
        if self.cf_buffer_size < cf_batch_size:
            del self.cf_buffer_size
            # Create new counterfactuals if we already sampled all of them (approx)
            self.compute_counterfactuals(
                future_p) if self.args.scorer_cls != 'coda' else self.compute_counterfactuals_coda()
        cf_idxs = np.random.randint(
            self.cf_buffer['obs'].shape[0], size=cf_batch_size)
        cf_transitions = {}
        for key in self.cf_buffer.keys():
            cf_transitions[key] = self.cf_buffer[key][cf_idxs].copy()

        cf_transitions['r'] = np.expand_dims(self.reward_func(
            cf_transitions['ag'], cf_transitions['g'], None), 1)
        self.cf_buffer_size -= cf_batch_size

        return cf_transitions

    def sample_dyna(self, episode_batch, batch_size, agent, future_p=0.):
        if self.args.sinb:
            if self.sinb_buffer is None:
                self.sinb_buffer = self.compute_sinb_buffer(episode_batch)

        dyna_batch_size = int(self.args.ratio_dyna*batch_size)
        episode_bs, episode_len, _ = episode_batch['actions'].shape
        episode_idxs = np.random.randint(episode_bs, size=dyna_batch_size)
        # substract one so we can always compute next step
        t_idxs = np.random.randint(
            episode_len-self.args.dyna_rollouts - 1, size=dyna_batch_size)
        start_obs = episode_batch['obs'][episode_idxs, t_idxs].copy()
        actions = None

        if self.args.sinb:
            object_to_perturb = np.random.choice(
                self.sinb_buffer['obs'].shape[0], size=start_obs.shape[0])
            start_obs = self.sinb_buffer['obs'][object_to_perturb,
                                                episode_idxs, t_idxs].copy()
            # sample actions offpolicy from the buffer
            actions = episode_batch['actions'][episode_idxs, t_idxs].copy()

        goals = episode_batch['g'][episode_idxs, t_idxs].copy()

        dyna_obs, dyna_obs_next, dyna_action = self.rollout_model(
            agent, start_obs, goals, actions)
        transitions = {}
        transitions['obs'] = dyna_obs
        transitions['actions'] = dyna_action
        transitions['obs_next'] = dyna_obs_next
        # Original goal in the data
        transitions['g'] = goals  # original

        # Do random relabeling
        idxs_relabel = np.where(np.random.uniform(
            size=dyna_batch_size) < future_p)[0]
        # RANDOM
        idxs_random_relabel = idxs_relabel[0: int(
            self.args.dyna_random_relabel*len(idxs_relabel))]
        rand_idxs = np.random.randint(
            episode_batch['ag'].shape[0] * episode_batch['ag'].shape[1], size=len(idxs_random_relabel))
        if not self.args.sinb:
            transitions['g'][idxs_random_relabel] = episode_batch['ag'].reshape(
                -1, self.args.env_goal_size)[rand_idxs]
        else:
            # Relabel by taking future achievd goals corresponding to the perturbed states
            ag_dim = self.sinb_buffer['ag'].reshape(
                self.sinb_buffer['obs'].shape[0], -1, self.args.env_goal_size)  # num_ nodes x N x goal_size
            transitions['g'][idxs_random_relabel] = ag_dim[object_to_perturb[idxs_random_relabel], rand_idxs]

        # Future.
        her_idxs = idxs_relabel[int(
            self.args.dyna_random_relabel*len(idxs_relabel))::]
        future_offset = np.random.uniform(
            size=dyna_batch_size) * (episode_len - t_idxs - self.args.dyna_rollouts - 1)
        future_offset = future_offset.astype(int)
        future_t = (t_idxs + self.args.dyna_rollouts + future_offset)
        if not self.args.sinb:
            future_ag = episode_batch['ag'][episode_idxs[her_idxs],
                                            future_t[her_idxs]]
        else:
            # Relabel by taking future achievd goals corresponding to the perturbed states!
            future_ag = self.sinb_buffer['ag'][object_to_perturb[her_idxs],
                                               episode_idxs[her_idxs], future_t[her_idxs]]

        transitions['g'][her_idxs] = future_ag

        transitions['ag'] = self.env_obs_2_ag(dyna_obs)

        transitions['r'] = np.expand_dims(self.reward_func(
            transitions['ag'], transitions['g'], None), 1)
        return transitions

    def rollout_model(self, agent, obs, g, actions):
        for _ in range(self.args.dyna_rollouts):
            with torch.no_grad():
                self.model_cai.eval()
                # Normalize data for policy network
                input_tensor = agent._preprocess_inputs(obs, g)
                if actions is None:  # default dyna use on-policy actions
                    actions = agent._deterministic_action(input_tensor)
                else:  # sinb uses offpolicy action from the buffer
                    actions = torch.tensor(actions).clone().detach()
                # Factorized observation for the world model
                fact_obs = self.env_input_factorizer(obs)
                fact_obs = rec_map_dict(lambda x: torch.tensor(
                    x), fact_obs)

                fact_obs['action'] = actions.squeeze()
                next_fact_obs = self.model_cai(fact_obs)[0]  # get mean only

                prev_fact_obs = fact_obs
                fact_obs = next_fact_obs
                obs = np.concatenate(
                    [fact_obs[key] for key in self.model_cai.inp_dims.keys() if key != 'action'], axis=-1)

        obs = np.concatenate([prev_fact_obs[key] for key in self.model_cai.inp_dims.keys(
        ) if key != 'action'], axis=-1)
        obs_next = np.concatenate(
            [fact_obs[key] for key in self.model_cai.inp_dims.keys() if key != 'action'], axis=-1)

        action = prev_fact_obs['action'].numpy()
        return obs, obs_next, action

    def compute_sinb_buffer(self, episode_batch):
        # Input original buffer num_episode x len_episode x dim_state
        # Output buffer with additional dimension # num_nodes where each dimension has the indexes for that object swapped --> num_nodes x num_episode x len_episode x dim_state
        n_samples = self.args.sinb_nsamples
        episode_bs, episode_len, dim_state = episode_batch['obs'].shape
        # episode_batch = episode_batch.reshape(episode_bs * episode_len, -1)
        # dict with obj_keys, each_key, (episode_len x num_episodes)x dim_object
        data_fact = self.env_input_factorizer(
            episode_batch['obs'].reshape(-1, dim_state))
        data_fact = rec_map_dict(
            lambda x: torch.tensor(x), data_fact)  # map to tensor

        # idxs of the samples to use to compare
        sample_idxs = np.random.choice(
            episode_bs*episode_len, size=(n_samples,))
        samples = {k: v[sample_idxs] for k, v in data_fact.items()}
        # compute all pairwise distances, by broadcasting (reshape here just adds dimensionalities for broadcasting)
        distances = {k: ((data_fact[k].reshape(episode_bs*episode_len, 1, -1) - samples[k].reshape(
            1, n_samples, -1))**2).sum(-1) for k in data_fact.keys()}  # each key = N x (num_samples)
        # compute all scores
        obj_scores = {k: v / torch.stack([distances[nk] for nk in distances.keys(
        ) if nk != k], 0).sum(0) for k, v in distances.items()}
        # compute optimal matching
        # for each key, N x 1 (with the best matching sample for each object)
        matching_obs = {k: torch.argmax(v, -1) for k, v in obj_scores.items()}
        # build buffer with samples ready to swap!
        sinb_buffer = {k: torch.tensor(
            episode_batch['obs'].reshape(-1, dim_state)) for k in matching_obs.keys()}
        for k in matching_obs.keys():
            sinb_buffer[k][:, self.env_obj_2_idx[k]
                           ] = samples[k][matching_obs[k]]
        # reshape in 3D tensor, might not be necessary
        sinb_buffer = {k: v.reshape(episode_bs, episode_len, dim_state)
                       for k, v in sinb_buffer.items()}
        # n_objects x episode_bs x episode_len x dim_state (for each sample, I have all the 3 swapp)
        sinb_buffer = torch.stack([v for k, v in sinb_buffer.items()], 0)
        sinb_buffer = sinb_buffer.detach().cpu().numpy()

        buffer = {}
        buffer['obs'] = sinb_buffer
        buffer['ag'] = self.env_obs_2_ag(sinb_buffer)

        return buffer

    def load_mixture(self, path_expert, path_random, expert_percent, random_percent, shrink_dataset, **kwargs):

        expert_data, random_data = None, None
        if expert_percent != 0.:
            with open(path_expert, "rb") as fp:
                expert_data = pickle.load(fp)
        if random_percent != 0.:
            with open(path_random, "rb") as fp:
                random_data = pickle.load(fp)
        print(
            f'Loading dataset for world model training from {path_expert} and {path_random}')
        size_expert = 0 if expert_data is None else expert_data['o'].shape[0]
        size_random = 0 if random_data is None else random_data['o'].shape[0]
        size_expert = int(size_expert*expert_percent)
        size_random = int(size_random*random_percent)
        print('Size expert data (episodes):', size_expert,
              'Size random data:', size_random)
        if shrink_dataset < 1.0:
            size_expert = int(size_expert*shrink_dataset)
            size_random = int(size_random*shrink_dataset)
        self.current_size = size_expert + size_random
        assert self.current_size <= self.size, f'Size of the buffer cannot feed size of data {self.current_size} > {self.size}'
        if expert_data:
            for key in expert_data.keys():
                self.buffers[self.key_map[key]
                             ][:size_expert] = expert_data[key][:size_expert]
        if random_data:
            for key in random_data.keys():
                self.buffers[self.key_map[key]
                             ][size_expert:self.current_size] = random_data[key][:size_random]

        for key in self.buffers.keys():
            self.buffers[key] = self.buffers[key][:self.current_size]

        self.buffers['obs_next'] = self.buffers['obs'][:, 1:, :].copy()
        self.buffers['ag_next'] = self.buffers['ag'][:, 1:, :].copy()

        # #Reshape so that obs, ag, ag_next,g and actions have the same shape
        self.buffers['obs'] = self.buffers['obs'][:, :-1, :].copy()
        self.buffers['ag'] = self.buffers['ag'][:, :-1, :].copy()

        self.data_mix_name = f'{self.args.env}-exp{self.args.expert_percent}-rnd{self.args.random_percent}--shr{shrink_dataset}'

    def compute_cais(self, future_p):
        num_episodes, len_ep, _ = self.buffers['obs'].shape
        # Copy of self.buffers is already done inside CMIBuffer
        self.cai_computer = CMIBuffer(
            self.buffers, self.scorer_cls, self.model_cai, **self.args.cai_computer_params)
        with open(self.model_cai.weights_file, 'rb') as inputfile:
            hash = hashlib.md5(inputfile.read()).hexdigest()

        if self.args.scorer_cls != 'coda':
            cai_dataset_path = os.path.join(
                get_data_dir(), 'cai_scores', f'{self.data_mix_name}_{self.args.scorer_cls}_{hash}.npy')
            if not os.path.exists(cai_dataset_path):
                print(
                    f'Dataset {cai_dataset_path} doesnt exist. Computing ...')
                self.cai_computer.compute_scores(batch_size=1000)
                dataset = self.cai_computer._buffer
                print('Saving dataset with cai scores in ...', cai_dataset_path)
                os.makedirs(os.path.dirname(cai_dataset_path), exist_ok=True)
                np.save(cai_dataset_path, dataset)
            else:
                print('Loading precomputed dataset with cai scores from...',
                      cai_dataset_path)
                dataset = np.load(cai_dataset_path,
                                  allow_pickle=True).reshape(-1)[0]  # dict

            # Reshape dataset back to 3D
            for k, v in dataset.items():
                if k in self.buffers.keys():
                    dataset[k] = v.reshape(self.buffers[k].shape)
                else:
                    dataset[k] = v.reshape(
                        self.current_size, self.buffers['obs'].shape[1], 1)

            for obj in self.model_cai.outp_dims.keys():
                dataset[f'cai_{obj}'] = np.zeros_like(dataset[f'score_{obj}'])
            num_objs = len(self.model_cai.outp_dims.keys())
            M_curr = np.zeros((num_episodes, len_ep-1, num_objs))
            M_next = np.zeros((num_episodes, len_ep-1, num_objs))
            for ep in range(num_episodes):
                for i, k in enumerate(self.model_cai.outp_dims.keys()):
                    dataset[f'cai_{k}'][ep, :, :] = smooth_fct(
                        dataset[f'score_{k}'][ep, :, :], kernel_size=self.args.smooth_cai_kernel).reshape(-1, 1)
                    M_curr[ep, :, i] = (
                        dataset[f'cai_{k}'][ep, :-1, :] < self.args.thr_cai).squeeze()
                    M_next[ep, :, i] = (
                        dataset[f'cai_{k}'][ep, 1:, :] < self.args.thr_cai).squeeze()

            # M is True if object is NOT under control in current and next state, so we can swap it
            M = M_curr * M_next
            M = M.reshape(-1, num_objs)  # num_samples x num_objs
            # List (of length num objs) of lists of indices of objects that are NOT under control
            self.L = [np.where(M[:, col])[0] for col in range(num_objs)]

            # Reshape M to obs_size:
            self.M = np.zeros((M.shape[0], self.args.env_obs_size))
            for i, k in enumerate(self.model_cai.outp_dims.keys()):
                self.M[:, self.env_obj_2_idx[k]] = M[:, i].reshape(-1, 1)

            self.compute_counterfactuals(future_p)

        else:
            # THR_CAI needs to be here already!
            cai_dataset_path = os.path.join(
                get_data_dir(), 'cai_scores', f'{self.data_mix_name}_{self.args.scorer_cls}_{self.args.thr_cai}_{hash}.npy')
            print(
                f'Dataset {cai_dataset_path} doesnt exist. Computing ...')
            self.cai_computer.compute_scores(batch_size=1000)
            dataset = self.cai_computer._buffer
            from scipy.sparse.csgraph import connected_components
            import tqdm
            coda_masks = (dataset['mask'] > self.args.thr_cai).astype(int)
            # Add dummy components for computing connected components via scipy --> (4,3): (4,4)
            coda_masks = np.concatenate(
                [coda_masks,  np.zeros((*coda_masks.shape[:2], 1))], axis=-1)

            # Compute connected components for each mask:
            samples = tqdm.trange(0, len(coda_masks),
                                  desc='Computing connected components ...')
            self.coda_cc = []
            for sample in samples:
                # connected_components converts a mask into a list of CC indices tuples.
                # e.g., if mask is [[1,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,1,1]],
                # this will return [array([0]), array([1]), array([2, 3])]
                m1 = coda_masks[sample]
                num_ccs, cc_idxs = connected_components(m1)
                cc = [np.where(cc_idxs == i)[0] for i in range(num_ccs)]
                # get set of cc: {(0,), (1,), (2,3))}
                cc = set(tuple(i) for i in cc)
                self.coda_cc.append(cc)
            self.coda_cc = np.array(self.coda_cc)
            dataset['coda_cc'] = self.coda_cc
            print('Done. ')

            self.compute_counterfactuals_coda()

    def compute_counterfactuals(self, future_p):
        print('Computing new counterfactuals ... ')
        # Output: counterfactual_buffer. Counterfactuals are done only per step, so the buffer
        # will be of size N x dim (no episode dim)
        # S = N x dim_s, A= N x dim_A, S'= N x dim_s, R = N x 1, done = N x 1

        # For each object index, sample only if not under control:)

        # Since we now have one less sample per episode, we need to remove one sample from each key!
        s = self.buffers['obs'][:, :-1,
                                :].reshape(-1, self.args.env_obs_size).copy()
        s_next = self.buffers['obs_next'][:, :-1,
                                          :].reshape(-1, self.args.env_obs_size).copy()
        a = self.buffers['actions'][:, :-1, :].reshape(-1,
                                                       self.args.env_action_size).copy()

        len_ep = self.buffers['obs'][:, :-1, :].shape[1]
        # Offset for computing goals: sample uniformly from current state to end of the episode
        offset = np.random.uniform(low=np.zeros(
            s.shape[0]), high=len_ep-np.arange(s.shape[0]) % len_ep).astype(int)
        fs_cf_idx = np.arange(s.shape[0]) + offset  #  sample future states
        fs_cf_idx = fs_cf_idx % s.shape[0]  # wrap around the ones at the end
        fs = s.copy()[fs_cf_idx]  # future states that we will use as goals
        fs_cf = s.copy()[fs_cf_idx]
        s_cf = s.copy()
        s_cf_next = s_next.copy()
        # Swap indexes with objects that are not under control
        for i, k in enumerate(self.model_cai.outp_dims.keys()):
            # no index to be swapped
            if len(self.L[i]) == 0:
                continue
            idx = np.random.choice(self.L[i], size=s.shape[0])
            s_cf[:, self.env_obj_2_idx[k]] = s[idx][:, self.env_obj_2_idx[k]]
            fs_cf[:, self.env_obj_2_idx[k]] = s[idx][:, self.env_obj_2_idx[k]]
            s_cf_next[:, self.env_obj_2_idx[k]
                      ] = s_next[idx][:, self.env_obj_2_idx[k]]

        # Bring back to original the indexes for which object is under control in current, next or FUTURE step or next step
        # recompute M such that we do not swap objects if they under control in current, next or future step
        M = self.M * self.M[fs_cf_idx]
        s_cf = s_cf * M + s * (1-M)
        s_cf_next = s_cf_next * M + s_next * (1-M)
        fs_cf = fs_cf * M + fs * (1-M)
        g_cf = self.env_obs_2_ag(fs_cf)

        if self.args.relabel_goal_coda:
            original_idxs = np.where(np.random.uniform(size=len(s)) < (1))[0]
            g_cf[original_idxs] = self.buffers['g'][:, :-1,
                                                    :].reshape(-1, self.args.env_goal_size).copy()[original_idxs]

            rand_idxs = np.where(np.random.uniform(
                size=len(s)) < (0.5*future_p))[0]
            g_cf[rand_idxs] = self.buffers['ag'][:, :-1,
                                                 :].reshape(-1, self.args.env_goal_size).copy()[rand_idxs]

        ag = self.env_obs_2_ag(s_cf)
        ag_next = self.env_obs_2_ag(s_cf_next)
        assert ag_next.shape[-1] == self.args.env_goal_size
        self.cf_buffer = {'obs': s_cf,
                          'obs_next': s_cf_next,
                          'actions': a,
                          'g': g_cf,
                          'ag_next': ag_next,
                          'ag': ag,
                          }
        self.cf_buffer_size = s_cf.shape[0]
        print('Done computing new counterfactuals.')

    def compute_counterfactuals_coda(self):
        print('Computing new counterfactuals ... ')
        # Output: counterfactual_buffer. Counterfactuals are done only per step, so the buffer
        # will be of size N x dim (no episode dim)
        # S = N x dim_s, A= N x dim_A, S'= N x dim_s, R = N x 1, done = N x 1

        # For each object index, sample only if not under control:)

        # Since we now have one less sample per episode, we need to remove one sample from each key!
        s = self.buffers['obs'][:, :-1,
                                :].reshape(-1, self.args.env_obs_size).copy()
        s_next = self.buffers['obs_next'][:, :-1,
                                          :].reshape(-1, self.args.env_obs_size).copy()
        a = self.buffers['actions'][:, :-1, :].reshape(-1,
                                                       self.args.env_action_size).copy()

        # Reshape to initial shape of obs, remove last step, and reshape to 2D
        coda_cc = self.coda_cc.reshape(
            self.buffers['obs'].shape[0], self.buffers['obs'].shape[1], -1)[:, :-1]
        coda_cc = coda_cc.reshape(-1, 1)

        offset = np.random.uniform(low=np.zeros(
            s.shape[0]), high=48-np.arange(s.shape[0]) % 48).astype(int)
        fs_cf_idx = np.arange(s.shape[0]) + offset  #  sample future states
        fs_cf_idx = fs_cf_idx % s.shape[0]  # wrap around the ones at the end
        fs_cf = s.copy()[fs_cf_idx]  # future states that we will use as goals
        s_cf = s.copy()
        s_cf_next = s_next.copy()
        # action component, not to swap if action is the entity since we would need to swap the action too
        action_cc = self.model_cai.get_input_index('action')

        for sample_idx in range(s.shape[0]):
            # get set of cc: {(0,), (1,), (2,3))}
            cc_curr = coda_cc[sample_idx][0]
            # Compute intersection of cc_curr with cc of future state
            cc_future = coda_cc[fs_cf_idx[sample_idx]][0]
            cc = cc_curr.intersection(cc_future)
            # remove the ccs that contain the action component (switch the other which is equivalent!) {(0,), (1,), (2,3))} --> {(0,), (1,)}
            cc = [i for i in list(cc) if action_cc not in i]

            for subgraph in cc:
                random_idx = np.random.randint(s.shape[0])
                cc2 = coda_cc[random_idx][0]
                if not subgraph in cc2:  # no common subgraphs
                    continue
                for object_idx in subgraph:
                    k = list(self.model_cai.inp_projs)[
                        object_idx]  # object idx to key
                    s_cf[sample_idx, self.env_obj_2_idx[k]
                         ] = s[random_idx, self.env_obj_2_idx[k]]
                    s_cf_next[sample_idx, self.env_obj_2_idx[k]
                              ] = s_next[random_idx, self.env_obj_2_idx[k]]
                    fs_cf[sample_idx, self.env_obj_2_idx[k]
                          ] = s[random_idx, self.env_obj_2_idx[k]]

        g_cf = self.env_obs_2_ag(fs_cf)
        if self.args.relabel_goal_coda:
            g_cf = self.buffers['g'][:, :-1,
                                     :].reshape(-1, self.args.env_goal_size).copy()
        ag = self.env_obs_2_ag(s_cf)
        ag_next = self.env_obs_2_ag(s_cf_next)
        assert ag_next.shape[-1] == self.args.env_goal_size
        self.cf_buffer = {'obs': s_cf,
                          'obs_next': s_cf_next,
                          'actions': a,
                          'g': g_cf,
                          'ag_next': ag_next,
                          'ag': ag,
                          }
        self.cf_buffer_size = s_cf.shape[0]
        print('Done computing new counterfactuals.')
