from gym.envs.registration import register

# register(
#     id='DisentangledFetchPush-v0',
#     entry_point='causal_slr.envs.fetch_env:DisentangledFetchPushEnv',
#     max_episode_steps=50
# )

register(
    id='DisentangledFetchPush-2B-v0',
    entry_point='causal_slr.envs.fetch_env:DisentangledFetchPushNEnv',
    max_episode_steps=100,
    kwargs={'n': 2}
)

register(
    id='FullStatePredDisentangledFetchPush-2B-v0',
    entry_point='causal_slr.envs.fetch_env:DisentangledFetchPushNEnv',
    max_episode_steps=100,
    kwargs={'n': 2, 'full_state_pred': True}
)

register(
    id='DisentangledFetchPush-1B-v0',
    entry_point='causal_slr.envs.fetch_env:DisentangledFetchPushNEnv',
    max_episode_steps=50,
    kwargs={'n': 1}
)


for num_blocks in range(1, 5):
    initial_qpos = {
        'robot0:slide0': 0.405,
        'robot0:slide1': 0.48,
        'robot0:slide2': 0.0,
        'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
    }

    for i in range(num_blocks):
        initial_qpos[F"object{i}:joint"] = [
            1.25, 0.53, .4 + i*.06, 1., 0., 0., 0.]

    register(
        id=f'DisentangledFpp_{num_blocks}Blocks-v1',
        entry_point='causal_slr.envs.fpp:FetchBlockConstructionEnv',

        kwargs={
            'reward_type': 'sparse',
            'initial_qpos': initial_qpos,
            'num_blocks': num_blocks,
            'render_size': 64,
        },
        max_episode_steps=50
    )

    register(
        id=f'UnstructuredDisentangledFpp_{num_blocks}Blocks-v1',
        entry_point='causal_slr.envs.fpp:UnstructuredFetchBlockConstruction',

        kwargs={
            'reward_type': 'sparse',
            'initial_qpos': initial_qpos,
            'num_blocks': num_blocks,
            'render_size': 64,
        },
        max_episode_steps=50
    )

    register(
        id=f'FullStatePredDisentangledFpp_{num_blocks}Blocks-v1',
        entry_point='causal_slr.envs.fpp:FetchBlockConstructionEnv',

        kwargs={
            'reward_type': 'sparse',
            'initial_qpos': initial_qpos,
            'num_blocks': num_blocks,
            'render_size': 64,
            'full_state_pred': True
        },
        max_episode_steps=50
    )

    register(
        id=f'FullStatePredUnstructuredDisentangledFpp_{num_blocks}Blocks-v1',
        entry_point='causal_slr.envs.fpp:UnstructuredFetchBlockConstruction',

        kwargs={
            'reward_type': 'sparse',
            'initial_qpos': initial_qpos,
            'num_blocks': num_blocks,
            'render_size': 64,
            'full_state_pred': True
        },
        max_episode_steps=50
    )



