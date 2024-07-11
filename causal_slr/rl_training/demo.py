import torch
from causal_slr.rl_training.models import Actor
import gym
import numpy as np
import causal_slr.envs
import argparse
import glob
import os
import yaml
from smart_settings.param_classes import recursive_objectify


# process the inputs

def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--env-name', type=str,
                        default='FetchReach-v1', help='the environment name')
    parser.add_argument('--extra', type=str, default='',
                        help='extra information for the model path')
    parser.add_argument('--save-dir', type=str,
                        default=f'{os.getcwd()}/experiments/rl_learning/', help='the path to save the models')
    args = parser.parse_args()

    return args


def update_dynamically(args, env):
    args.env_obs_size = env.observation_space['observation'].shape[0]
    args.env_goal_size = env.observation_space['desired_goal'].shape[0]
    args.env_action_size = env.action_space.shape[0]
    args.env_action_max = float(env.action_space.high[0])
    args.env_max_timesteps = env._max_episode_steps
    args.demo_length = env._max_episode_steps
    return args


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


def get_last_model_epochs(model_path):
    checkpoint_names = glob.glob(os.path.abspath(model_path) + "/*.pt")
    processed_names = [file.split('/')[-1].replace('e', '').replace('.pt', '')
                       for file in checkpoint_names]
    epochs = list(filter(lambda x: x is not None, [
        int(name) for name in processed_names]))

    last_epoch = np.max(epochs)
    last_model_path = os.path.join(model_path, f'e{last_epoch}.pt')
    return last_model_path


if __name__ == '__main__':
    args = get_args()
    # load the model param
    model_path_ = args.save_dir + args.env_name + '/' + args.extra + '/checkpoints'
    model_path = get_last_model_epochs(model_path_)

    config_path = args.save_dir + args.env_name + '/' + args.extra + '/config.yaml'
    with open(config_path, 'r') as file:
        config_args = yaml.safe_load(file)
    # Joining the config args with the args
    for k, v in config_args.items():
        if isinstance(v, dict):
            v = v['value']
        args.__setattr__(k, v)

    # create the environment
    print('Env name', args.env_name)
    env = gym.make(args.env_name)
    observation = env.reset()
    # get env params
    args = update_dynamically(args, env)

    o_mean, o_std, g_mean, g_std, model = torch.load(
        model_path, map_location=lambda storage, loc: storage)

    # create the actor network
    actor_network = Actor(args)
    actor_network.load_state_dict(model.state_dict())
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
            action = pi.detach().numpy().squeeze()
            # put actions into the environment
            observation_new, reward, _, info = env.step(action)
            obs = observation_new['observation']
        print('the episode is: {}, is success: {}'.format(
            i, info['is_success']))
