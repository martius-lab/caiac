import torch
from rl_modules.models import actor, critic
from arguments import get_args
import gym
import numpy as np
import causal_slr.envs
import os

# process the inputs

"Code adapted from https://github.com/TianhongDai/hindsight-experience-replay"

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

def _preprocess_states(o, g):
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
    # load the model param
    # model_path = os.path.join(args.save_dir, args.env_name,args.extra) + '/model.pt'
    if not len(args.model_path):
        model_path = args.save_dir + args.env_name + args.extra + '/model.pt'
    else:
        model_path = args.model_path
    
    print(f'Loading model from {model_path}')
    try:
        o_mean, o_std, g_mean, g_std, model = torch.load(
            model_path, map_location=lambda storage, loc: storage)
    except ValueError:
        o_mean, o_std, g_mean, g_std, model, critic_model = torch.load(
            model_path, map_location=lambda storage, loc: storage)
    # create the environment
    print('Env name', args.env_name)
    env = gym.make(args.env_name)
    # get the env param
    observation = env.reset()
    # get the environment params
    env_params = {'obs': observation['observation'].shape[0],
                  'goal': observation['desired_goal'].shape[0],
                  'action': env.action_space.shape[0],
                  'action_max': env.action_space.high[0],
                  }
    # create the actor network
    actor_network = actor(env_params)
    try:
        actor_network.load_state_dict(model.state_dict())
    except AttributeError: # for old saved models
        actor_network.load_state_dict(model)


    #create critic network

    critic_network = critic(env_params)
    try:
        critic_network.load_state_dict(critic_model.state_dict())
    except:
        pass # old saved models don't save the critic

    actor_network.eval()
    for i in range(args.demo_length):
        observation = env.reset()
        # start to do the demo
        obs = observation['observation']
        g = observation['desired_goal']
        for t in range(env._max_episode_steps):
            env.render()
            inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
            with torch.no_grad():
                pi = actor_network(inputs)
            action = pi.detach()
            # qvalue = critic_network(_preprocess_states(obs, g).unsqueeze(dim=0), action.unsqueeze(dim=0))
            action = action.numpy().squeeze()
            # put actions into the environment
            observation_new, reward, _, info = env.step(action)
            obs = observation_new['observation']
        print('the episode is: {}, is success: {}'.format(
            i, info['is_success']))
