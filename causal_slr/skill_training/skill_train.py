import torch
import hashlib
import copy
import glob
import time
import datetime
from torch import autograd
from cluster import exit_for_resume
from causal_slr.utils import conf_utils
from causal_slr.utils.general_utils import RecursiveAverageMeter, map_dict, save_config, set_seeds
from causal_slr.components.checkpointer import CheckpointHandler, save_cmd, save_git
from causal_slr.utils.general_utils import AttrDict, get_optimizer_class, \
    AverageMeter, recursive_objectify, get_exp_dir, get_data_dir
from causal_slr.utils.wandb import WandBLogger
import yaml
import os
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


WANDB_PROJECT_NAME = 'causal_slr'
WANDB_ENTITY_NAME = 'nuria'
WANDB_MODE = "default"  # "disabled"

os.environ['EXP_DIR'] = './experiments'
PREFIX = os.getcwd()
DATA_PATH = PREFIX + '/data'


START_TIME = time.time()
EXIT_FOR_RESUME_SECONDS = 60*60*3


def should_exit():
    return (time.time() - START_TIME) > EXIT_FOR_RESUME_SECONDS


class SkillTrainer():
    def __init__(self, args):
        os.environ['DATA_DIR'] = DATA_PATH
        self.args = args
        self.setup_device()

        os.environ['DATA_DIR'] = 'data'
        # set up params
        self.conf = conf = self.postprocess_conf(self.get_config())
        self.with_cai = self.conf.skill_general.with_cai
        self.exp_path = os.path.join(args.working_dir, self.conf.env_config.name,
                                     args.sweep_name, datetime_str())
        set_seeds(self.conf.seed)

        # set up logging + training monitoring
        self.logger = self.setup_logging(conf)

        # buld dataset, model. logger, etc.
        train_params = AttrDict(skill_model=self.conf.skill_model_config,
                                loader=self.conf.data_config.data_loader,
                                data_params=self.conf.data_config,
                                )
        self.model, self.train_loader = self.build_phase(
            train_params, 'train')

        test_params = AttrDict(skill_model=self.conf.skill_model_config,
                               loader=self.conf.data_config.data_loader,
                               data_params=self.conf.data_config,
                               )
        self.model_test, self.val_loader = self.build_phase(
            test_params, phase='val')

        # set up optimizer + evaluator
        self.optimizer = get_optimizer_class(self.conf.skill_general)(filter(lambda p: p.requires_grad,
                                                                             self.model.parameters()), lr=self.conf.skill_general.lr)
        print('len train and val loader', len(
            self.train_loader), len(self.val_loader))

    def run(self):
        # global step is number of optimizations steps in total
        self.global_step, start_epoch = 0, 0
        # load model params from checkpoint
        start_epoch = self.resume()

        self.train(start_epoch)

    def train(self, start_epoch):
        self.val()
        self.log_outputs_interval = self.args.skill_general.log_interval
        epoch = start_epoch

        while self.global_step < self.conf.skill_general.total_grad_steps:

            self.train_epoch(epoch)
            epoch += 1

    def train_epoch(self, epoch):
        self.model.train()
        end = time.time()
        time_epoch = time.time()
        # timers
        batch_time = AverageMeter()
        upto_log_time = AverageMeter()
        data_load_time = AverageMeter()

        for self.batch_idx, sample_batched in enumerate(self.train_loader):
            data_load_time.update(time.time() - end)
            inputs = AttrDict(
                map_dict(lambda x: x.to(self.device), sample_batched))
            self.optimizer.zero_grad()
            output = self.model(inputs)
            losses = self.model.loss(output, inputs)
            losses.total.value.backward()

            if self.global_step < self.conf.skill_general.init_grad_clip_step:
                # clip gradients in initial steps to avoid NaN gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.conf.skill_general.init_grad_clip)
            self.optimizer.step()
            self.model.step()

            upto_log_time.update(time.time() - end)
            if self.log_outputs_now and not self.args.dont_save:
                self.model.log_outputs(output, inputs, losses, self.global_step,
                                       log_images=False, phase='train', **self._logging_kwargs)
            batch_time.update(time.time() - end)
            end = time.time()
            self.global_step = self.global_step + 1

            if self.log_outputs_now:
                # print('GPU {}: {}'.format(os.environ["CUDA_VISIBLE_DEVICES"] if self.use_cuda else 'none',
                #                           self.exp_path))
                print(('Gradient step: {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.global_step, epoch, self.batch_idx, len(
                        self.train_loader),
                    100. * self.batch_idx / len(self.train_loader), losses.total.value.item())))

                print('avg time for loading: {:.2f}s, logs: {:.2f}s, compute: {:.2f}s, total: {:.2f}s'
                      .format(data_load_time.avg,
                              batch_time.avg - upto_log_time.avg,
                              upto_log_time.avg - data_load_time.avg,
                              batch_time.avg))

                togo_train_time = batch_time.avg * \
                    (self.conf.skill_general.total_grad_steps - self.global_step) / \
                    3600.
                print('ETA: {:.2f}h \n'.format(togo_train_time))

                if not self.args.dont_save:
                    self.logger.log_scalar_dict({'Time/data_load': data_load_time.avg, 'Time/log': batch_time.avg - upto_log_time.avg, 'Time/compute': upto_log_time.avg - data_load_time.avg,
                                                'Time/total': batch_time.avg}, step=self.global_step)

            del output, losses

            # Save checkpoint
            if not self.args.dont_save and (self.global_step % self.args.skill_general.val_interval == 0 or self.global_step >= (self.conf.skill_general.total_grad_steps) or should_exit()):

                # Save checkpoint
                save_checkpoint({
                    'epoch': epoch,
                    'global_step': self.global_step,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                },  os.path.join(self.exp_path, 'weights'), CheckpointHandler.get_ckpt_name(self.global_step))

                # Validate
                self.val()
                # Evaluate success rate on environment specific tasks
                render = should_exit()
                self.do_task_evaluation(render=render)

            if should_exit():
                print('Time limit reached. Exiting.')
                # then at next run, call resume() instead of train()
                exit_for_resume()

            if self.global_step >= (self.conf.skill_general.total_grad_steps):
                print('\nTotal number of gradient steps reached. Exiting.')
                break

        if not self.args.dont_save and not epoch % 100:
            print('Time for full epoch: Epoch {} : {:.2f}s \n\n'.format(
                epoch, time.time() - time_epoch))
            self.logger.log_scalar_dict(
                {'Time/Full_epoch': time.time() - time_epoch}, step=self.global_step)

    def do_task_evaluation(self, render=False, num_vals=None):
        results = self.evaluator.task_evaluation(
            step=self.global_step, render=render, num_vals=num_vals)
        return results

    def val(self):
        print('\nEvaluating on validation set...')
        start = time.time()
        self.model_test.load_state_dict(self.model.state_dict())
        losses_meter = RecursiveAverageMeter()
        batch_idx = -1  # hack to ensure val loader is not empty
        with autograd.no_grad():
            for batch_idx, sample_batched in enumerate(self.val_loader):
                inputs = AttrDict(
                    map_dict(lambda x: x.to(self.device), sample_batched))

                with self.model_test.val_mode():
                    output = self.model_test(inputs)
                    losses = self.model_test.loss(output, inputs)
                losses_meter.update(losses)
                del losses
            if not batch_idx > -1:
                raise ValueError(
                    f'No validation data. Please check your train and val dataloaders sizes: ({len(self.train_loader)} and {len(self.val_loader)})')

            if not self.args.dont_save:
                self.model_test.log_outputs(output, inputs, losses_meter.avg, self.global_step,
                                            log_images=False, phase='val', **self._logging_kwargs)
                print(('Validation set: Average loss: {:.4f} in {:.2f}s\n'
                       .format(losses_meter.avg.total.value.item(), time.time() - start)))

        del output
        return losses_meter.avg.total.value.item()

    def setup_device(self):
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device(
            'cuda') if self.use_cuda else torch.device('cpu')

    def get_config(self):
        conf = AttrDict()
        conf.skill_general = self.args.skill_general

        conf.skill_model_config = self.args.skill_model_config
        conf.seed = self.args.seed
        conf.env_config = self.args.env_config
        # add device to env config as it directly returns tensors
        conf.env_config.device = self.device.type
        conf.data_config = self.args.data_config
        conf.mdp_model_config_path = self.args.mdp_config_path
        return conf

    def postprocess_conf(self, conf):

        conf.skill_model_config.update(conf.data_config.dataset_spec)
        conf.skill_model_config.n_objects = conf.env_config.n_objects if conf.skill_general.with_cai else 0
        # Needef for initialising hidden state in LSTM
        conf.skill_model_config.batch_size = conf.skill_general.batch_size
        conf.skill_model_config.device = conf.data_config.device = self.device.type
        # flat last action from seq gets cropped
        conf.data_config.dataset_spec.len_skill = conf.skill_model_config.len_skill

        return conf

    def setup_logging(self, conf):
        if not self.args.dont_save:
            print('Writing to the experiment directory: {}'.format(self.exp_path))

            os.makedirs(self.exp_path, exist_ok=True)
            if not os.path.exists(os.path.join(self.exp_path, 'cmd.txt')):
                save_cmd(self.exp_path)
                save_git(self.exp_path)
                conf_name = os.path.join(
                    self.exp_path, "conf.yaml")
                save_config(conf, conf_name)
                id_wandb = None
            else:
                with open(os.path.join(self.exp_path, 'id_wandb'), 'r') as file:
                    id_wandb = yaml.safe_load(file)
            sweep_name = self.args.working_dir
            group = self.args.data_config.dataset_spec.dataset_name.split(
                '/')[-1][:-4] + '_' + self.args.data_config.dataset_spec.scorer_cls + '_' + sweep_name
            logger = WandBLogger(self.exp_path, WANDB_PROJECT_NAME, entity=WANDB_ENTITY_NAME,
                                 path=self.exp_path, conf=conf, wandb_mode=WANDB_MODE, id=id_wandb, group=group)
        else:
            logger = None

        # set up additional logging args
        self._logging_kwargs = AttrDict(
        )
        return logger

    def build_phase(self, params, phase):

        model = conf_utils.get_skill_model(
            params.skill_model.model_class)(params=params.skill_model, logger=self.logger)
        model.device = self.device
        model_cai = self.load_model_cai()
        loader = self.get_dataloader(
            params.loader, params.data_params, phase, model_cai=model_cai)

        return model, loader

    def get_dataloader(self, loader_params, data_params, phase, **kwargs):
        from causal_slr.dataloaders.kitchen.dataloader import D4RLSequenceSplitDataset
        data_params.data_dir = get_data_dir()
        # Important pass as a copy! Otw parameters we change inside the class will be changed for all objects
        loader_params_ = AttrDict(copy.deepcopy(loader_params))
        data_params_ = AttrDict(copy.deepcopy(data_params))
        loader = D4RLSequenceSplitDataset(loader_params_, data_params_,
                                          phase=phase, shuffle=True, **kwargs). \
            get_data_loader(self.conf.skill_general.batch_size)

        return loader

    def load_model_cai(self):
        from causal_slr.model_training.train_model import load_weights, build_model
        filelist = glob.glob(f'{self.args.mdp_config_path}/*.yaml')
        if len(filelist) == 1:
            config_path = filelist[0]
        else:
            raise FileNotFoundError(
                f'Couldnt find world model file {self.args.mdp_config_path}')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        config = recursive_objectify(AttrDict(config))
        config.model_params.config_path = config_path
        print('\nBuilding dynamics model cai ...')
        self.env = conf_utils.get_env(
            config.env_config.name, config.env_config)
        model = build_model(config, env=self.env)

        # this does the model.to(self.device) already
        model.device = self.device
        model = load_weights(model, config, type='best')
        print('Dynamics model loaded!\n')

        if not self.args.dont_save:
            # Saving hash of model file to working_dir
            with open(model.weights_file, 'rb') as inputfile:
                hash = hashlib.md5(inputfile.read()).hexdigest()
            with open(os.path.join(self.exp_path, 'model_hash.txt'), 'w') as outputfile:
                print(f'Saving hash for model in ', self.exp_path)
                outputfile.write(hash)

        return model

    def resume(self, resume_epoch='latest'):
        path = os.path.join(
            self.exp_path, 'weights')
        if not os.path.exists(path):
            print('\nStarting training from scratch\n')
            start_epoch = 0
            return start_epoch

        print('\nResuming from weights found in {}'.format(path))
        weights_file = CheckpointHandler.get_resume_ckpt_file(
            resume_epoch, path)
        self.global_step, start_epoch, _, _ = \
            CheckpointHandler.load_weights(weights_file, self.model,
                                           load_step=True, load_opt=True, optimizer=self.optimizer,
                                           strict=True)
        self.model.device = self.device
        return start_epoch

    @property
    def log_outputs_now(self):
        return self.global_step % self.log_outputs_interval == 0


def save_checkpoint(state, folder, filename='checkpoint.pth'):
    os.makedirs(folder, exist_ok=True)
    torch.save(state, os.path.join(folder, filename))
    print(f"Saved checkpoint to {os.path.join(folder, filename)}!")


def datetime_str():
    return datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")


def make_path(ours: bool, sweep_name: str,  working_dir: str, dir_name: str = None):
    exp_dir = get_exp_dir()
    # TODO: adapt with env name!
    is_ours = 'ours' if ours else 'baseline'
    path = os.path.join(dir_name, is_ours)
    return os.path.join(exp_dir, path, sweep_name, working_dir)


if __name__ == '__main__':
    SkillTrainer(args=get_args())
