import torch
from rl_modules.models import actor
from arguments import get_args
import gym
import numpy as np
import causal_slr.envs
import pickle
from collections import defaultdict

# process the inputs


def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -
                     args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -
                     args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs


if __name__ == '__main__':
    args = get_args()

    size = args.demo_length = 40000

    # load the model param
    model_path = args.save_dir + args.env_name + args.extra + '/model.pt'
    o_mean, o_std, g_mean, g_std, model = torch.load(
        model_path, map_location=lambda storage, loc: storage)
    # create the environment
    print(args.env_name)

    env = gym.make(args.env_name)
    # get the env param
    observation = env.reset()
    # get the environment params
    env_params = {'obs': observation['observation'].shape[0],
                  'goal': observation['desired_goal'].shape[0],
                  'action': env.action_space.shape[0],
                  'action_max': env.action_space.high[0],
                  }

    T = env._max_episode_steps
    args.env_obs_size = env.observation_space['observation'].shape[0]
    args.env_goal_size = env.observation_space['desired_goal'].shape[0]
    args.env_action_size = env.action_space.shape[0]
    args.env_action_max = float(env.action_space.high[0])
    args.env_max_timesteps = env._max_episode_steps

    buffers = {
        'o': np.empty([size, T, args.env_obs_size], dtype=np.float32),
        'ag': np.empty([size, T, args.env_goal_size], dtype=np.float32),
        'g': np.empty([size, T-1, args.env_goal_size], dtype=np.float32),
        'u': np.empty([size, T-1, args.env_action_size], dtype=np.float32),
    }
    # create the actor network
    actor_network = actor(env_params)
    actor_network.load_state_dict(model)
    actor_network.eval()
    for i in range(args.demo_length):
        observation = env.reset()
        # start to do the demo

        obs, ag, g = (observation[k] for k in [
                      'observation', 'achieved_goal', 'desired_goal'])
        episode = defaultdict(list, {k: [] for k in buffers.keys()})
        for t in range(env._max_episode_steps-1):
            # env.render()
            if not args.random:
                inputs = process_inputs(
                    obs, g, o_mean, o_std, g_mean, g_std, args)
                with torch.no_grad():
                    pi = actor_network(inputs)
                action = pi.detach().numpy().squeeze()
            else:
                action = env.action_space.sample()
            [episode[k].append(v) for k, v in zip(
                ['o', 'ag', 'g', 'u'], [obs, ag, g, action])]

            # put actions into the environment, and update obs_new --> obs
            observation_new, reward, _, info = env.step(action)
            obs, ag, g = (observation_new[k] for k in [
                      'observation', 'achieved_goal', 'desired_goal'])

        # Add last obs and ag
        [episode[k].append(v) for k, v in zip(['o', 'ag'], [obs, ag])]
        episode = {k: np.array(v) for k, v in episode.items()}
        print('the episode is: {}, is success: {}'.format(
            i, info['is_success']))

        for k in episode.keys():
            buffers[k][i] = episode[k]

    # Save buffers
    extra2 = '_random' if args.random else ''
    data_path = f'data_collection/collected_data/{args.env_name}{args.extra}{extra2}.pkl'

    with open(data_path, 'wb') as handle:
        pickle.dump(buffers, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'Saved data to {data_path}')
