from causal_slr.utils.general_utils import listdict2dictlist, AttrDict
import time


class RolloutGenerator():
    def __init__(self, agent, env, max_episode_len) -> None:
        self.agent = agent
        self.env = env
        self._max_episode_len = max_episode_len

    def reset(self):
        obs = self.env.reset()

        self.agent.reset_hl_step_counter()
        self.success_steps = 0
        self._episode_step, self._episode_reward = 0, 0.

        return obs

    def sample_episode(self, render=False, render_mode='human'):
        obs = self.reset()
        task = self.env.task[0]
        episode, done = [], False
        while not done and self._episode_step < self._max_episode_len:
            agent_output = self.agent.act(obs, task)
            # info is a list: [defaualt_env info, dditional info]
            next_obs, reward, done, info = self.env.step(
                agent_output.ll_action)
            self._episode_step += 1
            render_obs = self._render(render, render_mode)
            episode.append(AttrDict(
                reward=reward,
                done=done,
                info=info[0],
                add_info=info[1],
                image=render_obs))
            obs = next_obs
            if self._episode_step == 1:
                time.sleep(1.5)
        # make sure episode is marked as done at final time step
        episode[-1].done = True
        ep_dict = listdict2dictlist(episode)
        ep_dict['ep_success'] = info[1]['success']
        ep_dict['ep_len'] = self._episode_step
        ep_dict['task'] = task
        return ep_dict

    def _render(self, render, render_mode):
        if not render:
            return None
        if render:
            return self.env.render(mode=render_mode)
