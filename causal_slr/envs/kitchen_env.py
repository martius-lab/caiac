import numpy as np
from contextlib import contextmanager
from dm_control.mujoco import engine
import random
from collections import OrderedDict
import torch
from causal_slr.utils.general_utils import AttrDict


OBJ_2_IDX = {
    'agent': np.arange(0, 9),
    'br_burner': np.array([9, 10]),
    'bl_burner': np.array([11, 12]),
    'tr_burner': np.array([13, 14]),
    'tl_burner': np.array([15, 16]),
    'light': np.array([17, 18]),
    'slide': np.array([19]),
    'hinge': np.array([20, 21]),
    'mw': np.array([22]),
    'kettle': np.array([23, 24, 25, 26, 27, 28, 29])
}


# [9 - 29] in OBS. [9-10: bottom right burner] [ 13-14 top right burner]
# OBS_ELEMENT_INDICES = {
#     'bottom burner': [0, np.array([11, 12])],
#     'top (left) burner': [1, np.array([15, 16])],
#     'light switch': [2, np.array([17, 18])],
#     'slide cabinet': [3, np.array([19])],
#     'hinge cabinet': [4, np.array([20, 21])],
#     'microwave': [5, np.array([22])],
#     'kettle': [6, np.array([23, 24, 25, 26, 27, 28, 29])],
#     'robot': [7, np.arange(0, 9)],
#     'bottom (right) burner': [8, np.array([9, 10])],
#     'top (right) burner': [9, np.array([13, 14])],
# }


OBS_ELEMENT_INDICES = {
    'bottom burner':  [0, np.array([11, 12])],
    'top burner': [1, np.array([15, 16])],
    'light switch': [2, np.array([17, 18])],
    'slide cabinet': [3, np.array([19])],
    'hinge cabinet': [4, np.array([20, 21])],
    'microwave':  [5, np.array([22])],
    'kettle': [6, np.array([23, 24, 25, 26, 27, 28, 29])],
    'robot': [7, np.arange(0, 9)],
}
OBS_ELEMENT_GOALS = {
    # rotation of the knob [rad], joint opening [m]
    'bottom burner': [0, np.array([-0.88, -0.01])],
    'top burner': [1, np.array([-0.92, -0.01])],
    'light switch': [2, np.array([-0.69, -0.05])],
    'slide cabinet': [3, np.array([0.37])],
    'hinge cabinet': [4, np.array([0., 1.45])],
    'microwave': [5, np.array([-0.75])],
    'kettle': [6, np.array([-0.23, 0.75, 1.62, 0.99, 0., 0., -0.06])],
    'robot': [7, np.zeros((1, 9))],
    'bottom (right) burner': [8, np.array([-0.88, -0.01])],
    'top (right) burner': [9, np.array([-0.92, -0.01])],
}

INTEREST_GOAL_IDX = {
    # rotation of the knob [rad], joint opening [m]
    'bottom burner': [0],
    'top burner': [0],
    'light switch': [0],
    'slide cabinet': [0],
    'hinge cabinet': [1],
    'microwave': [0],
    'kettle': [1],
    'robot': [0],
    'bottom (right) burner': [0],
    'top (right) burner': [0]

}

ELEMENT_LIMITS = {
    # rotation of the knob [rad], joint opening [m]
    'bl_burner': [0, np.array([-1, 0.5])],
    'tl_burner': [1, np.array([-1, 0.5])],
    'light': [2, np.array([-0.8, 0.5])],
    'slide': [3, np.array([-0.3, 0.6])],
    'hinge': [4, np.array([-0.3, 1.6])],
    'mw': [5, np.array([-1, 0.1])],
    'kettle': [6, np.array([-1, 1])],
    'agent': [7, np.array([-2.5, 2.5])],
    'br_burner': [8, np.array([-1., 0.5])],
    'tr_burner': [9, np.array([-1, 0.5])],
}


def create_task_generator(lst):
    index = 0
    while True:
        yield lst[index]
        if index == len(lst) - 1:
            random.shuffle(lst)
        index = (index + 1) % len(lst)


class MyKitchenEnv():
    def __init__(self, env, n_objects=8, postprocess_obs=True, tasks=[], random_init_objs=False, **kwargs):
        self._env = env
        self.screen_height = self.screen_width = 200
        ALL_TASKS = ['microwave', 'kettle', 'light switch',
                     'slide cabinet', 'bottom burner', 'hinge cabinet']
        assert all(t in ALL_TASKS for t in tasks)
        self.TASK_ELEMENTS = tasks
        print('Evaluating on tasks: ', self.TASK_ELEMENTS)
        self.task_generator = create_task_generator(self.TASK_ELEMENTS)
        self.robot_obs_dims = len(self.obs_dict['qp'])
        self.obj_obs_dims = len(self.obs_dict['obj_qp'])
        self.verbose = False
        self.num_objects = n_objects
        self.postprocess_obs_ = postprocess_obs
        self.random_init_objs = random_init_objs
        assert len(OBS_ELEMENT_INDICES.keys()) == self.num_objects
        self.ELEMENT_LIMITS = ELEMENT_LIMITS

    def step(self, a, b=None):
        obs, reward, done, info = self._env.step(a)
        obs = self.postprocess_obs(obs)
        add_info = {}
        if self.REMOVE_TASKS_WHEN_COMPLETE:  # my success signal relies on this
            add_info['tasks_to_complete'] = list(
                self.tasks_to_complete)
            add_info['success'] = not add_info['tasks_to_complete']
        else:
            raise NotImplementedError

        return obs, reward, done, [info, add_info]

    def __getattr__(
            self, attr: str):
        """Gets attribute from wrapped environment.

        Args:
            attr (str): attribute name

        Returns:
            Any: requested attribute
        """
        return getattr(self._env, attr)

    def render(self, mode='rgb_array'):
        if mode == 'human':  # KitchenV0(robot_env.RobotEnv) modify azimuth
            self.sim_robot.renderer.render_to_window()
        if mode == 'rgb_array':
            camera = engine.MovableCamera(
                self.sim, self.screen_height, self.screen_width)
            camera.set_pose(
                distance=2.2, lookat=[-0.2, .5, 2.], azimuth=86, elevation=-35)
            img = camera.render()
            return img

    def postprocess_obs(self, obs):
        """Postprocesses observation to include [ state + object_goal]  i.e  [robot_state, object_state, object_goal] """

        if not self.postprocess_obs_:
            return obs
        obs = np.concatenate(
            (obs[:self.robot_obs_dims+self.obj_obs_dims], self.obj_goal), axis=-1)
        return obs

    def reset(self):
        # Very important! (setattr(env, 'TASK_ELEMENTS', TASK_ELEMENTS) doesnt work
        self.get_new_task()
        if self.verbose:
            print('New task', self.task)
        self._env.set_tasks(self.task)

        obs = self._env.reset()

        if self.random_init_objs:
            obs = self.random_initialize_objects(obs)
            self.set_pos(obs[0:30])
        new_goal = self._get_task_goal(obs.copy())
        self.set_goal(new_goal)
        assert (new_goal == self.goal).all()

        obs = self.postprocess_obs(obs)
        return obs

    def random_initialize_objects(self, obs):
        """Randomly initializes objects in the scene.

        Args:
            obs (np.ndarray): observation

        Returns:
            np.ndarray: modified observation
        """
        obs_ = obs.copy()
        for obj, idx in OBS_ELEMENT_INDICES.items():
            if obj != 'robot' and obj not in self.task:
                if np.random.rand() < 0.5:
                    obs_[idx[1]] = OBS_ELEMENT_GOALS[obj][1]

        # Close hinge cabinet unless task is [kettle or microwave] to avoid collisions
        if self.task not in ['microwave', 'kettle']:
            obs_[OBS_ELEMENT_INDICES['hinge cabinet'][1]] = obs[OBS_ELEMENT_INDICES['hinge cabinet'][1]]

        return obs_

    def get_new_task(self):
        self.task = [next(self.task_generator)]

    def set_pos(
        self, pos: np.ndarray
    ):
        """Sets agent position; useful for precise evaluation.

        Args:
            pos (np.ndarray): agent and env state (i.e. 9+21 dim vector)
        """
        if pos is not None:
            assert len(pos) == self.robot_obs_dims + self.obj_obs_dims
            # reset_pos = self.init_qpos[:].copy()
            # reset_pos[23:26] = [0.69397440e-01, 5.50383255e-01,  1.61944683e+00] # modifying kettle
            reset_pos = pos
            reset_vel = self.init_qvel[:].copy()
            self.robot.reset(self, reset_pos, reset_vel)
            self.sim.forward()
            # self.goal = self._get_task_goal()  #sample a new goal on reset
            return self.postprocess_obs(self.env._get_obs())
        
    def set_pos_robot(
        self, pos: np.ndarray, obs: np.ndarray
    ):
        """Sets agent position ONLY!; useful for precise evaluation, rest kept hte same

        Args:
            pos (np.ndarray): agent and env state (i.e. 9+21 dim vector)
        """
        if pos is not None:
            assert len(pos) == self.robot_obs_dims + self.obj_obs_dims
            # reset_pos = self.init_qpos[:].copy()
            # reset_pos[23:26] = [0.69397440e-01, 5.50383255e-01,  1.61944683e+00] # modifying kettle
            reset_pos = obs
            reset_pos[:9] = pos[:9]
            reset_vel = self.init_qvel[:].copy()
            self.robot.reset(self, reset_pos, reset_vel)
            self.sim.forward()
            # self.goal = self._get_task_goal()  #sample a new goal on reset
            return self.postprocess_obs(self.env._get_obs())

    def _get_task_goal(self, obs):
        # Set the goal as the initial observation for all objects, except the desired task that is modified
        new_goal = obs.copy()
        for element in self.task:
            element_idx = OBS_ELEMENT_INDICES[element][1]
            element_goal = OBS_ELEMENT_GOALS[element][1]
            new_goal[element_idx] = element_goal

        new_goal = new_goal[0:self.robot_obs_dims+self.obj_obs_dims]
        return new_goal

    @property
    def obj_goal(self):
        """Returns only the indexes of the goal accounting for object properties (exclude robot part)"""
        return self.goal[self.robot_obs_dims::]

    @contextmanager
    def val_mode(self):
        """Sets validation parameters if desired. To be used like: with env.val_mode(): ...<do something>..."""
        pass
        yield
        pass

    def get_input_factorizer(self, x):

        return OrderedDict(agent=x[..., np.arange(0, 9)],
                           br_burner=x[..., np.array([9, 10])],
                           bl_burner=x[..., np.array([11, 12])],
                           tr_burner=x[..., np.array([13, 14])],
                           tl_burner=x[..., np.array([15, 16])],
                           light=x[..., np.array([17, 18])],
                           slide=x[..., np.array([19])],
                           hinge=x[..., np.array([20, 21])],
                           mw=x[..., np.array([22])],
                           kettle=x[..., np.array(
                               [23, 24, 25, 26, 27, 28, 29])],
                           )

    def get_output_factorizer(self, x):

        return OrderedDict(agent=x[..., np.arange(0, 9)],
                           br_burner=x[..., np.array([9, 10])],
                           bl_burner=x[..., np.array([11, 12])],
                           tr_burner=x[..., np.array([13, 14])],
                           tl_burner=x[..., np.array([15, 16])],
                           light=x[..., np.array([17, 18])],
                           slide=x[..., np.array([19])],
                           hinge=x[..., np.array([20, 21])],
                           mw=x[..., np.array([22])],
                           kettle=x[..., np.array(
                               [23, 24, 25, 26, 27, 28, 29])],
                           )

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
        return OBJ_2_IDX