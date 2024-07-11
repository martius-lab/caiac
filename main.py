import yaml
import sys
import json
import datetime
import random
import causal_slr.rl_training as rl_modules
import numpy as np
import gym
from causal_slr.model_training.model_eval import WorldModelEvaluator
from causal_slr.model_training.train_model import WorldModelTrainer
from causal_slr.utils.dict_utils import recursive_objectify
from causal_slr.components.evaluator import InferenceEvaluator
from causal_slr.skill_training.skill_train import SkillTrainer
import os
import torch
os.environ["MUJOCO_GL"] = "egl" #glfw


def datetime_str():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def update_dynamically(args, env):
    args.use_disc = True if args.reward_type == 'disc' else False
    if args.relabel == False:
        args.relabel_percent = 0.

    args.env_obs_size = env.observation_space['observation'].shape[0]
    args.env_goal_size = env.observation_space['desired_goal'].shape[0]
    args.env_action_size = env.action_space.shape[0]
    args.env_action_max = float(env.action_space.high[0])
    args.env_max_timesteps = env._max_episode_steps
    args.ratio_dyna = args.get('ratio_dyna', 0)
    args.run_name = f'{args.env}-{args.expert_percent}-{args.random_percent}-{args.method}-{args.seed}'
    args.working_dir = os.path.join(
        args.working_dir, args.env, datetime_str())
    return args


def main():
    kwargs = yaml.safe_load(open(sys.argv[1]))
    # Make args an AttributeDict of Attributedicts (smart settings class) and mutable
    args = recursive_objectify(kwargs, make_immutable=False)
    args.working_dir = args.working_dir.replace(
        '{__timestamp__}', '{}'.format(datetime_str()))
    print('Starting the main loop!')

    metrics = dict()
    if args.get('learn_mdp', False):
        if not args.world_model.validate_only:
            print('\n\n***** Start model learning ******')
            # MDP modl training
            model_trainer = WorldModelTrainer(args)
            if model_trainer is not None:  # case for resuming and model learning has fiinished
                model_trainer.run()
                # # Do a last evaluation
                results_model = model_trainer.val()
                # Finish wandb otw its saved in same logger!
                model_trainer.logger.wandb_finish()
                for k, v in results_model.items():
                    metrics[k] = v
                print('\n\n***** End model learning ******')
        else:
            evaluator = WorldModelEvaluator(
                args.world_model, model=None, train_loader=None, mdp_path=args.preset_mdp_config_path)

    if args.get('skill_learn', False):
        print('\n\n***** Start skill discovery ******')
        args.mdp_config_path = model_trainer.exp_path if args.get(
            'learn_mdp', False) else args.get('preset_mdp_config_path', None)
        assert args.mdp_config_path is not None, 'No mdp_config_path provided and cf required!'
        if not args.evaluate:
            # SKILL TRAINING
            skill_trainer = SkillTrainer(args)
            args.skill_config_path = skill_trainer.exp_path
            print('Config path is', args.skill_config_path)
            if not args.dont_save:
                evaluator = InferenceEvaluator(
                    args, logger=skill_trainer.logger)
                skill_trainer.evaluator = evaluator
            skill_trainer.run()

            # # Do a last evaluation
            print('Doing last evaluation for cluster metrics... \n')
            results_skill = skill_trainer.val()
            num_vals = len(args.env_config.tasks) * 20
            results_task = skill_trainer.do_task_evaluation(
                render=False, num_vals=num_vals)
            # Render some tasks
            skill_trainer.do_task_evaluation(render=True)
            # Finish wandb otw its saved in same logger!
            skill_trainer.logger.wandb_finish()
            metrics['total_val_skill_loss'] = results_skill
            for k, v in results_task.items():
                metrics[k] = v
            print('\n\n***** End skill discovery ******')

        else:
            evaluator = InferenceEvaluator(args, logger=None)
            evaluator.task_evaluation(0, render=True, render_mode='human')

    # RL
    elif args.get('train_rl', False):
        print('\n\n***** Train RL agent ******')
        args.mdp_config_path = model_trainer.exp_path if args.get(
            'learn_mdp', False) else args.preset_mdp_config_path
        env = gym.make(args.env)
        args = update_dynamically(args, env)
        args = recursive_objectify(args, make_immutable=True)
        env.seed(args.seed)
        [f(args.seed) for f in [random.seed, np.random.seed,
                                torch.manual_seed, torch.cuda.manual_seed]]
        trainer = getattr(rl_modules, args.method)(args, env)
        metrics = trainer.learn()
        print('\n\n***** End RL agent ******')

    print('Final metrics:', metrics)
    print(
        f'Saving metrics in json file {args.working_dir}/final_metrics.csv')
    os.makedirs(args.working_dir, exist_ok=True)
    out_file = open(f"{args.working_dir}/metrics.csv", "w")
    json.dump(metrics, out_file, indent=6)
    out_file.close()
    return metrics


if __name__ == '__main__':
    main()
