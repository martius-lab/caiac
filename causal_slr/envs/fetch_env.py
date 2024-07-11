from gym.utils import EzPickle
import torch
import os
from causal_slr.utils.general_utils import AttrDict
from collections import OrderedDict
from gym.envs.robotics import rotations, utils
import numpy as np
from gym.envs.robotics.fetch.push import FetchPushEnv
from gym.envs.robotics.fetch_env import goal_distance
from gym.envs.robotics import fetch_env
from gym import spaces
import gym
import causal_slr.envs

"Code adapted from Counterfactual Data Augmentation using Locally Factored Dynamics https://github.com/spitis/mrl/blob/97b3b638d338473ee5e2c12ed5b6a58a0ae9e095/envs/customfetch/custom_fetch.py "


class FetchPickPlaceEnv():
    def __init__(self, hyper_params):
        pass


dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = f'{os.getcwd()}/causal_slr/envs'

PUSH_N_XML = os.path.join(dir_path, 'xmls', 'FetchPush#.xml')

INIT_Q_POSES = [
    [1.3, 0.6, 0.41, 1., 0., 0., 0.],
    [1.3, 0.9, 0.41, 1., 0., 0., 0.],
    [1.2, 0.68, 0.41, 1., 0., 0., 0.],
    [1.4, 0.82, 0.41, 1., 0., 0., 0.],
    [1.4, 0.68, 0.41, 1., 0., 0., 0.],
    [1.2, 0.82, 0.41, 1., 0., 0., 0.],
]
INIT_Q_POSES_SLIDE = [
    [1.3, 0.7, 0.42, 1., 0., 0., 0.],
    [1.3, 0.9, 0.42, 1., 0., 0., 0.],
    [1.25, 0.8, 0.42, 1., 0., 0., 0.],
    [1.35, 0.8, 0.42, 1., 0., 0., 0.],
    [1.35, 0.7, 0.42, 1., 0., 0., 0.],
    [1.25, 0.9, 0.42, 1., 0., 0., 0.],
]

ELEMENT_LIMITS = {
    'object': [0, np.array([-1, 1])],
    'agent': [7, np.array([-1, 1])],
}


class FetchPickPlaceEnv():
    def __init__(self, hyper_params):
        pass


class DisentangledFetchPushNEnv(fetch_env.FetchEnv, EzPickle):
    def __init__(self,
                 n=1,
                 distance_threshold=0.05,
                 max_episode_steps=50,
                 full_state_pred=False):
        print(f'Using DisentangledFetchPushNEnv with {n} objects')
        self.n = n
        initial_qpos = {
            'robot0:slide0': 0.05,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        for i in range(self.n):
            k = 'object{}:joint'.format(i)
            initial_qpos[k] = INIT_Q_POSES_SLIDE[i]

        fetch_env.FetchEnv.__init__(self,
                                    PUSH_N_XML.replace('#', '{}'.format(n)),
                                    has_object=True,
                                    block_gripper=True,
                                    n_substeps=20,
                                    gripper_extra_height=0.,
                                    target_in_the_air=False,
                                    target_offset=np.array([-0.075, 0.0, 0.0]),
                                    obj_range=0.15,
                                    target_range=0.25,
                                    distance_threshold=distance_threshold,
                                    initial_qpos=initial_qpos,
                                    reward_type='sparse')
        EzPickle.__init__(self)

        self.max_step = max_episode_steps
        self.num_step = 0
        self.ELEMENT_LIMITS = {k: ELEMENT_LIMITS[k] if k in ELEMENT_LIMITS.keys(
        ) else ELEMENT_LIMITS[k[:-1]] for k in self.OBJ_2_IDX.keys()}
        obs = self._get_obs()
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf,
                                    shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf,
                                     shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf,
                                   shape=obs['observation'].shape, dtype='float32'),
        ))
        self.full_state_pred = full_state_pred

    def set_goal(self, goal):
        self.goal = goal

    def reset(self):
        obs = super().reset()
        self.num_step = 0
        return obs

    def set_state_from_observation(self, observation):
        # This only sets the position of the object not the robot!
        object_position = observation[10:13]
        object_qpos = self.sim.data.get_joint_qpos(
            'object0:joint')
        assert object_qpos.shape == (7,)
        object_qpos[:3] = object_position
        self.sim.data.set_joint_qpos(
            'object0:joint', object_qpos)

        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)


        bad_poses = [self.initial_gripper_xpos[:2]]
        # Randomize start positions of pucks.
        for i in range(self.n):
            object_xpos = self.initial_gripper_xpos[:2]
            while min([np.linalg.norm(object_xpos - p) for p in bad_poses]) < 0.08:
                object_xpos = self.initial_gripper_xpos[:2] + \
                    self.np_random.uniform(-self.obj_range,
                                           self.obj_range, size=2)
            bad_poses.append(object_xpos)

            object_qpos = self.sim.data.get_joint_qpos(
                'object{}:joint'.format(i))
            object_qvel = self.sim.data.get_joint_qvel(
                'object{}:joint'.format(i))
            object_qpos[:2] = object_xpos
            object_qpos[2:] = np.array([0.42, 1., 0., 0., 0.])
            self.sim.data.set_joint_qpos(
                'object{}:joint'.format(i), object_qpos)
            self.sim.data.set_joint_qvel(
                'object{}:joint'.format(i), np.zeros_like(object_qvel))

        self.sim.forward()
        return True

    def _sample_goal(self):
        first_puck = self.initial_gripper_xpos[:2] + \
            self.np_random.uniform(-self.target_range,
                                   self.target_range, size=2)

        goal_xys = [first_puck[:2]]
        for i in range(self.n - 1):
            object_xpos = self.initial_gripper_xpos[:2] + \
                self.np_random.uniform(-self.target_range,
                                       self.target_range, size=2)
            while min([np.linalg.norm(object_xpos - p) for p in goal_xys]) < 0.08:
                object_xpos = self.initial_gripper_xpos[:2] + \
                    self.np_random.uniform(-self.target_range,
                                           self.target_range, size=2)
            goal_xys.append(object_xpos)

        goals = [np.concatenate((goal, [self.height_offset]))
                 for goal in goal_xys]

        return np.concatenate(goals)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array(
            [-0.548, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos(
            'robot0:grip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def step(self, action):
        obs, reward, _, info = super().step(action)
        self.num_step += 1
        done = True if self.num_step >= self.max_step else False
        if done:
            info['TimeLimit.truncated'] = True

        info['is_success'] = np.allclose(reward, 0.)
        return obs, reward, done, info

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos)
        goals = np.split(self.goal, self.n)

        for i in range(self.n):
            site_id = self.sim.model.site_name2id('target{}'.format(i))
            self.sim.model.site_pos[site_id] = goals[i] - sites_offset[i]
        self.sim.forward()

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.

        if len(achieved_goal.shape) == 1:
            actual_goals = np.split(goal, self.n)
            achieved_goals = np.split(achieved_goal, self.n)
            success = 1.
        else:
            actual_goals = np.split(goal, self.n, axis=1)
            achieved_goals = np.split(achieved_goal, self.n, axis=1)
            success = np.ones(achieved_goal.shape[0], dtype=np.float32)

        for b, g in zip(achieved_goals, actual_goals):
            d = goal_distance(b, g)
            success *= (d <= self.distance_threshold).astype(np.float32)

        return success - 1.

    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        obj_feats = []
        obj_poses = []

        for i in range(self.n):
            obj_labl = 'object{}'.format(i)
            object_pos = self.sim.data.get_site_xpos(obj_labl).ravel()
            object_pos[2] = max(object_pos[2], self.height_offset)
            # rotations
            object_rot = rotations.mat2euler(
                self.sim.data.get_site_xmat(obj_labl)).ravel()
            # velocities
            object_velp = (self.sim.data.get_site_xvelp(obj_labl) * dt).ravel()
            object_velr = (self.sim.data.get_site_xvelr(obj_labl) * dt).ravel()
            # Make pos and velocity relative to gripper
            # object_rel_pos = object_pos - grip_pos
            # object_velp -= grip_velp

            obj_feats.append([
                object_pos.ravel(),
                object_rot.ravel(),
                object_velp.ravel(),
                object_velr.ravel(),
            ])
            obj_poses.append(object_pos)

        gripper_state = robot_qpos[-2:]
        # change to a scalar if the gripper is made symmetric
        gripper_vel = robot_qvel[-2:] * dt

        achieved_goal = np.concatenate(obj_poses)

        grip_obs = np.concatenate([
            grip_pos,
            gripper_state,
            grip_velp,
            gripper_vel,
        ])

        obs = np.concatenate(
            [grip_pos, gripper_state, grip_velp, gripper_vel] + sum(obj_feats, []))

        return {
            'observation': obs,
            'achieved_goal': achieved_goal,
            'desired_goal': self.goal.copy(),
        }

    def get_input_factorizer(self, x):
        d = OrderedDict()
        for k, v in self.OBJ_2_IDX.items():
            d[k] = x[..., v]
        return d

    def get_output_factorizer(self, x):
        d = OrderedDict()
        if self.full_state_pred:
            for k, v in self.OBJ_2_IDX.items():
                d[k] = x[..., v]
        else:
            for k, v in self.OBJ_2_IDX.items():
                # only use the gripper pos (3) and gripper state (2)
                if k == 'agent':
                    v = v[:5]
                else:
                    v = v[:3]  # only use the obj pos (3)
                d[k] = x[..., v]
        return d

    @property
    def shapes(self):
        example = [self.get_input_factorizer(torch.randn(
            (100))), self.get_output_factorizer(torch.randn((100)))]
        inp_shapes = {name: list(val.shape)
                      for name, val in example[0].items()}
        inp_shapes['action'] = [self.action_space.shape[0]]
        target_shapes = {name: list(val.shape)
                         for name, val in example[1].items()}

        return AttrDict({'input': inp_shapes, 'output': target_shapes})

    @property
    def OBJ_2_IDX(self):
        d = OrderedDict()
        d['agent'] = np.arange(10)
        for i in range(self.n):
            d['object{}'.format(i)] = np.arange(10 + 12*i, 10 + 12*(i+1))
        return d

    def obs_2_ag(self, obs):
        obs_pos_idxs = np.concatenate(
            [np.arange(10 + 12*i, 10 + 12*i + 3) for i in range(self.n)])
        return obs[..., obs_pos_idxs]
    
    @property
    def OBJ_2_GOALIDX(self):
        d = OrderedDict()
        for i in range(self.n):
            d['object{}'.format(i)] = np.arange(3*i, 3*(i+1))
        return d


if __name__ == '__main__':
    from causal_slr.utils.general_utils import set_seeds
    seed = 10
    set_seeds(seed)
    env = gym.make('DisentangledFpp_4Blocks-v1')
    env.seed(seed)
    s = env.reset()
    while True:
        # if mode is 'human' need to set LD_PRELOAD
        env.render(mode='human')
        env.step(env.action_space.sample())  # take a random action
    env.close()
