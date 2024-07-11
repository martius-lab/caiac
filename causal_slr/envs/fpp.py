import os
from gym import utils as gym_utils
from causal_slr.envs.construction import fetch_env
from gym.envs.robotics import rotations, utils
import numpy as np
import mujoco_py
from causal_slr.envs.construction.xml import generate_xml
import tempfile
import gym
import causal_slr.envs
from collections import OrderedDict
import torch
from causal_slr.utils.general_utils import AttrDict

ELEMENT_LIMITS = {
    'object': [0, np.array([-1, 1])],
    'agent': [7, np.array([-1, 1])],
}

dir_path = f'{os.getcwd()}/causal_slr/envs'
class FetchBlockConstructionEnv(fetch_env.FetchEnv, gym_utils.EzPickle):
    def __init__(self, initial_qpos,
                 num_blocks=1,
                 reward_type='incremental',
                 render_size=42,
                 full_state_pred=False):
        self.num_blocks = num_blocks
        self.object_names = ['object{}'.format(i) for i in range(self.num_blocks)]

        with tempfile.NamedTemporaryFile(mode='wt', dir=f"{dir_path}/construction/assets/fetch", delete=False, suffix=".xml") as fp:
            fp.write(generate_xml(self.num_blocks))
            MODEL_XML_PATH = fp.name

        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, render_size=render_size)

        os.remove(MODEL_XML_PATH)

        gym_utils.EzPickle.__init__(self, initial_qpos, num_blocks, reward_type, render_size)
        self.render_image_obs = False
        self.ELEMENT_LIMITS = {k: ELEMENT_LIMITS[k] if k in ELEMENT_LIMITS.keys(
        ) else ELEMENT_LIMITS[k[:-1]] for k in self.OBJ_2_IDX.keys()}

        self.full_state_pred = full_state_pred

    def gripper_pos_far_from_goals(self, achieved_goal, goal):
        gripper_pos = achieved_goal[..., -3:] # Get the grip position only

        block_goals = goal[..., :-3] # Get all the goals EXCEPT the zero'd out grip position

        distances = [
            np.linalg.norm(gripper_pos - block_goals[..., i*3:(i+1)*3], axis=-1) for i in range(self.num_blocks)
        ]
        return np.all([d > self.distance_threshold * 2 for d in distances], axis=0)

    def subgoal_distances(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        for i in range(self.num_blocks - 1):
            assert goal_a[..., i * 3:(i + 1) * 3].shape == goal_a[..., (i + 1) * 3:(i + 2) * 3].shape
        return [
            np.linalg.norm(goal_a[..., i * 3:(i + 1) * 3] - goal_b[..., i * 3:(i + 1) * 3], axis=-1) for i in
            range(self.num_blocks)
        ]

    def compute_reward(self, achieved_goal, goal, info):
        """
        Computes reward, perhaps in an off-policy way during training. Doesn't make sense to use any of the simulator state besides that provided in achieved_goal, goal.
        :param achieved_goal:
        :param goal:
        :param info:
        :return:
        """
        subgoal_distances = self.subgoal_distances(achieved_goal, goal)
        if self.reward_type == 'incremental':
            # Using incremental reward for each block in correct position
            reward = -np.sum([(d > self.distance_threshold).astype(np.float32) for d in subgoal_distances], axis=0)
            reward = np.asarray(reward)

            # If blocks are successfully aligned with goals, add a bonus for the gripper being away from the goals
            np.putmask(reward, reward == 0, self.gripper_pos_far_from_goals(achieved_goal, goal))
            return reward
        elif self.reward_type == "sparse":
            reward = np.min([-(d > self.distance_threshold).astype(np.float32) for d in subgoal_distances], axis=0)
            reward = np.asarray(reward)
            return reward
        elif self.reward_type == "dense":
            # Dense incremental
            stacked_reward = -np.sum([(d > self.distance_threshold).astype(np.float32) for d in subgoal_distances], axis=0)
            stacked_reward = np.asarray(stacked_reward)

            reward = stacked_reward.copy()
            np.putmask(reward, reward == 0, self.gripper_pos_far_from_goals(achieved_goal, goal))

            if stacked_reward != 0:
                next_block_id = int(self.num_blocks - np.abs(stacked_reward))
                assert 0 <= next_block_id < self.num_blocks
                gripper_pos = achieved_goal[..., -3:]
                block_goals = goal[..., :-3]
                reward -= .01 * np.linalg.norm(gripper_pos - block_goals[next_block_id*3: (next_block_id+1)*3])
            return reward
        else:
            raise ("Reward not defined!")
    
    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        obs = np.concatenate([
            grip_pos,
            gripper_state,
            grip_velp,
            gripper_vel,
        ])

        achieved_goal = []
        for i in range(self.num_blocks):
            object_i_pos = self.sim.data.get_site_xpos(self.object_names[i])
            # rotations
            object_i_rot = rotations.mat2euler(self.sim.data.get_site_xmat(self.object_names[i]))
            # velocities
            object_i_velp = self.sim.data.get_site_xvelp(self.object_names[i]) * dt
            object_i_velr = self.sim.data.get_site_xvelr(self.object_names[i]) * dt
            # Make pos and velocity relative to gripper
            # object_i_rel_pos = object_i_pos - grip_pos
            # object_i_velp -= grip_velp

            obs = np.concatenate([
                obs,
                object_i_pos.ravel(),
                # object_i_rel_pos.ravel(),
                object_i_rot.ravel(),
                object_i_velp.ravel(),
                object_i_velr.ravel()
            ])

            achieved_goal = np.concatenate([
                achieved_goal, object_i_pos.copy()
            ])

        # # Append the grip
        # achieved_goal = np.concatenate([achieved_goal, grip_pos.copy()])

        achieved_goal = np.squeeze(achieved_goal)

        return_dict = {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }
        return return_dict


    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()

        for i in range(self.num_blocks):
            site_id = self.sim.model.site_name2id('target{}'.format(i))
            self.sim.model.site_pos[site_id] = self.goal[i * 3:(i + 1) * 3] - sites_offset[i]

        self.sim.forward()

    def _reset_sim(self):
        assert self.num_blocks <= 17 # Cannot sample collision free block inits with > 17 blocks
        self.sim.set_state(self.initial_state)

        # Initialize objects in a column.
        rows, cols = self.initial_gripper_xpos[0] + np.linspace(-self.obj_range-0.02, self.obj_range-0.03,
                                                                self.num_blocks), self.initial_gripper_xpos[1] + np.linspace(-self.obj_range, self.obj_range, self.num_blocks)

        col = np.random.randint(0, len(cols))
        for idx_obj, obj_name in enumerate(self.object_names):
            object_xypos = rows[idx_obj], cols[col]

            object_qpos = self.sim.data.get_joint_qpos(F"{obj_name}:joint")
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xypos
            object_qpos[2] = self.height_offset
            self.sim.data.set_joint_qpos(F"{obj_name}:joint", object_qpos)
            self.sim.forward()
        return True

    def _sample_goal(self):
        goals = []
        self.obj2lift = np.random.randint(self.num_blocks)
        for i in range(self.num_blocks):
            goal_object = self.sim.data.get_joint_qpos(F"object{i}:joint")[:3].copy()
            if i == self.obj2lift:  # Lift this block, rest kept same!
                goal_object[2] += 0.2 # fix height for now -->self.np_random.uniform(0, 0.45)
                # goal_object[1] += 0.1
            goals.append(goal_object)

        return np.concatenate(goals, axis=0).copy()

    def _is_success(self, achieved_goal, desired_goal):
        subgoal_distances = self.subgoal_distances(achieved_goal, desired_goal)
        if np.sum([-(d > self.distance_threshold).astype(np.float32) for d in subgoal_distances]) == 0:
            return True
        else:
            return False

    def _set_action(self, action):
        assert action.shape == (4,), action.shape
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl, gripper_ctrl = action[:3], action[3]

        pos_ctrl *= 0.05  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        try:
            self.sim.step()
        except mujoco_py.builder.MujocoException as e:
            print(e)
            print(F"action {action}")
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

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
        for i in range(self.num_blocks):
            d['object{}'.format(i)] = np.arange(10 + 12*i, 10 + 12*(i+1))
        return d
    
    @property
    def OBJ_2_GOALIDX(self):
        d = OrderedDict()
        for i in range(self.num_blocks):
            d['object{}'.format(i)] = np.arange(3*i, 3*(i+1))
        return d

    def obs_2_ag(self, obs):
        obs_pos_idxs = np.concatenate(
            [np.arange(10 + 12*i, 10 + 12*i + 3) for i in range(self.num_blocks)])
        return obs[..., obs_pos_idxs]


class UnstructuredFetchBlockConstruction(FetchBlockConstructionEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _reset_sim(self):
        assert self.num_blocks <= 17 # Cannot sample collision free block inits with > 17 blocks
        self.sim.set_state(self.initial_state)

        # Initialize objects in a column.
        rows, cols = self.initial_gripper_xpos[0] + np.linspace(-self.obj_range-0.02, self.obj_range-0.03,
                                                                self.num_blocks), self.initial_gripper_xpos[1] + np.linspace(-self.obj_range, self.obj_range, self.num_blocks)

        for idx_obj, obj_name in enumerate(self.object_names):
            col = np.random.randint(0, len(cols))
            object_xypos = rows[idx_obj], cols[col]

            object_qpos = self.sim.data.get_joint_qpos(F"{obj_name}:joint")
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xypos
            object_qpos[2] = self.height_offset
            self.sim.data.set_joint_qpos(F"{obj_name}:joint", object_qpos)
            self.sim.forward()
        return True
    

if __name__ == '__main__':
    from causal_slr.utils.general_utils import set_seeds
    seed = 2
    set_seeds(seed)
    env = gym.make('DisentangledFpp_4Blocks-v1')
    env.seed(seed)
    s = env.reset()
    while True:
        # if mode is 'human' need to set LD_PRELOAD
        env.render(mode='human')
        env.step(env.action_space.sample())  # take a random action
    env.close()

