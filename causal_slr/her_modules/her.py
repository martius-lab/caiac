import numpy as np


class HERSampler:
    "Adapted from https://github.com/TianhongDai/hindsight-experience-replay/tree/master"

    def __init__(self, replay_strategy, relabel_percent, reward_func=None):
        self.replay_strategy = replay_strategy
        self.relabel_percent = relabel_percent
        self.relabel_percent_initial = relabel_percent
        if self.replay_strategy in ['future', 'clearning', 'mbold']:
            self.future_p = relabel_percent
        else:
            self.future_p = 0
        self.reward_func = reward_func

    def sample_her_transitions(self, episode_batch, batch_size, future_p=0.):

        # select which episode and which timesteps to be used
        episode_bs, episode_len, _ = episode_batch['actions'].shape
        episode_idxs = np.random.randint(episode_bs, size=batch_size)
        # substract one so we can always compute next step
        t_idxs = np.random.randint(episode_len-1, size=batch_size)

        transitions = {}
        for key in episode_batch.keys():
            transitions[key] = episode_batch[key][episode_idxs, t_idxs].copy()
        if self.replay_strategy == 'future':

            her_idxs = np.where(np.random.uniform(size=batch_size) < future_p)
            future_offset = np.random.uniform(
                size=batch_size) * (episode_len - t_idxs)
            future_offset = future_offset.astype(int)
            future_t = (t_idxs + future_offset)[her_idxs]
            future_ag = episode_batch['ag'][episode_idxs[her_idxs], future_t]
            transitions['g'][her_idxs] = future_ag

        elif self.replay_strategy == 'clearning':

            # label the first half batch with next goals
            relabel_next_num = int(future_p * batch_size)
            future_offset = np.zeros(shape=batch_size).astype(int)
            future_t = (t_idxs + 1)[:relabel_next_num]
            future_ag = episode_batch['ag'][episode_idxs[:relabel_next_num], future_t]
            transitions['g'][:relabel_next_num] = future_ag

            # label the next half batch with random goals
            random_goals = future_ag.copy()
            np.random.shuffle(random_goals)
            transitions['g'][relabel_next_num:] = random_goals

        elif self.replay_strategy == 'mbold':
            if future_p == 0:
                print('Relabeling everything with random goals!')
            # label with prob future_p with achieved goals in same episode sampled from geom distribution
            relabel_next_num = int(future_p * batch_size)
            future_offset = np.random.geometric(
                p=self.relabel_geom, size=batch_size).astype(int)
            future_offset = np.minimum(future_offset, (episode_len - t_idxs-1))
            future_t = (t_idxs + future_offset)[:relabel_next_num]
            transitions['g'][:relabel_next_num] = episode_batch['ag'][episode_idxs[:relabel_next_num], future_t]
            # label the rest with achieved goals from random episodes
            rand_episode_idxs = np.random.randint(
                episode_bs, size=batch_size-relabel_next_num)
            rand_t_idxs = np.random.randint(
                episode_len, size=batch_size-relabel_next_num)
            future_offset[relabel_next_num:] = episode_len
            transitions['g'][relabel_next_num:] = episode_batch['ag'][rand_episode_idxs, rand_t_idxs]

        transitions['r'] = np.expand_dims(self.reward_func(
            transitions['ag'], transitions['g'], None), 1)

        return {'transitions': transitions, 'future_offset': future_offset}
