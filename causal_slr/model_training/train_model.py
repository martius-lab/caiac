from torch import autograd
from causal_slr.utils.general_utils import AttrDict, get_optimizer_class, \
    AverageMeter, map_dict, RecursiveAverageMeter, get_exp_dir, set_seeds, save_config, rec_map_dict
import time
import yaml
from causal_slr.utils.wandb import WandBLogger
from causal_slr.components.checkpointer import CheckpointHandler, save_cmd, save_git
from causal_slr.utils import conf_utils
from causal_slr.cmi import CMIScorer, MaskScorer

from causal_slr.dataloaders.kitchen.dataloader import FactorizedForwardDataset
import torch
import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import datetime

os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'

os.environ['EXP_DIR'] = './experiments'

os.environ['DATA_DIR'] = 'data'

WANDB_PROJECT_NAME = 'world_model'
WANDB_ENTITY_NAME = 'nuria' # 'your_username'
WANDB_MODE = "default"  # "disabled"


def make_path(sweep_name: str,  working_dir: str, dir_name: str = None):
    exp_dir = get_exp_dir()
    return os.path.join(exp_dir, dir_name, sweep_name, working_dir)



class WorldModelTrainer():  # BaseTrainer):
    def __init__(self, args):

        self.conf = args.world_model

        self.args = args

        if self.conf.validate_only:
            self.args.dont_save = True

        self.args.working_dir = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.exp_path = make_path(sweep_name=self.args.sweep_name,
                                  working_dir=self.args.working_dir, dir_name=f'model_learning/{self.conf.env_config.name}/')

        counter = get_model_training_counter(self.exp_path)
        if counter >= self.conf.train_params.total_grad_steps:
            print(
                f'Model training already finished, skipping training. Counter: {counter}')
            return None
        set_seeds(self.args.seed)

        # set up logging + training monitoring
        self.logger = self.setup_logging(self.conf)
        self.setup_device()
        self.loss_fn = conf_utils.get_loss(self.conf.train_params.loss_fn.name,
                                           hyper_params=self.conf.train_params.loss_fn.get('params', None))
        self.loss_name = self.conf.train_params.loss_fn.name if self.conf.train_params.loss_fn.name != 'nll_loss' else 'gauss_ll'
        self.model, self.train_loader = self.build_phase(
            phase='train')
        self.model_test, self.val_loader = self.build_phase(
            phase='val')

        self.optimizer = get_optimizer_class(self.conf.train_params)(filter(
            lambda p: p.requires_grad, self.model.parameters()), lr=self.conf.train_params.lr)

        # Save config here since we updated params in build_phase
        conf_name = os.path.join(
            self.exp_path, "conf_model.yaml")
        if not self.args.dont_save:
            save_config(self.conf, conf_name)

    def run(self):
        # global step is number of optimizations steps in total
        self.global_step, start_epoch = 0, 0
        # load model params from checkpoint
        start_epoch = self.resume()
        self.train(start_epoch)
        update_model_training_counter(
            path=self.exp_path, counter=self.conf.train_params.total_grad_steps)

    def train(self, start_epoch):
        print(f'Training model on {len(self.train_loader.dataset)} samples'
              f'for {self.conf.train_params.total_grad_steps} grad steps with batch_size {self.batch_size}')
        print(
            f'Validating model on {len(self.val_loader.dataset)} transitions')
        self.val()
        self.log_outputs_interval = self.conf.train_params.log_interval
        epoch = start_epoch

        while self.global_step < self.conf.train_params.total_grad_steps:
            self.train_epoch(epoch)
            epoch += 1

    def train_epoch(self, epoch):
        self.model.train()
        end = time.time()
        # timers
        batch_time = AverageMeter()
        upto_log_time = AverageMeter()
        data_load_time = AverageMeter()

        for inputs, targets, _ in self.train_loader:
            data_load_time.update(time.time() - end)
            inputs = map_dict(lambda x: x.to(self.device), inputs)
            targets = map_dict(lambda x: x.to(self.device), targets)
            self.optimizer.zero_grad()
            # tuple(mean, var) both of size batch_size x target.dim
            loss = self.model.compute_loss(inputs, targets)
            loss.backward()
            if self.conf.train_params.gradient_clip > 0.:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.conf.train_params.gradient_clip)
            # if self.global_step < self.conf.train_params.init_grad_clip_step:
            #     torch.nn.utils.clip_grad_norm_(
            #         self.model.parameters(), self.conf.train_params.init_grad_clip)
            self.optimizer.step()
            upto_log_time.update(time.time() - end)
            if self.log_outputs_now and not self.args.dont_save:
                d = {self.loss_name: AttrDict(value=loss)}
                losses = AttrDict(d)
                self.model.log_outputs(model_output=None, inputs=None, losses=losses, step=self.global_step,
                                       log_images=False, phase='train')

            batch_time.update(time.time() - end)
            end = time.time()
            self.global_step = self.global_step + 1

            if self.log_outputs_now:
                print(('\nGradient step: {} Train Epoch: {} \tLoss: {:.3f}'.format(
                    self.global_step, epoch, loss)))

                print('avg time for loading: {:.2f}s, logs: {:.2f}s, compute: {:.2f}s, total: {:.2f}s'
                      .format(data_load_time.avg,
                              batch_time.avg - upto_log_time.avg,
                              upto_log_time.avg - data_load_time.avg,
                              batch_time.avg))

                togo_train_time = batch_time.avg * \
                    (self.conf.train_params.total_grad_steps - self.global_step) / \
                    3600.
                print('ETA: {:.2f}h'.format(togo_train_time))

                if not self.args.dont_save:
                    self.logger.log_scalar_dict({'Time/data_load': data_load_time.avg, 'Time/log': batch_time.avg - upto_log_time.avg, 'Time/compute': upto_log_time.avg - data_load_time.avg,
                                                'Time/total': batch_time.avg}, step=self.global_step)

            # Save checkpoint
            if not self.args.dont_save and (self.global_step % self.conf.train_params.val_interval == 0 or self.global_step >= (self.conf.train_params.total_grad_steps)):

                save_checkpoint(self.get_model_dict(epoch),  os.path.join(
                    self.exp_path, 'weights'), CheckpointHandler.get_ckpt_name(self.global_step))
                val_losses = self.val()
                if self.best_loss > val_losses[self.loss_name]:
                    self.best_loss = val_losses[self.loss_name]

                    save_checkpoint(self.get_model_dict(epoch),  os.path.join(self.exp_path, 'weights'), filename=f'model_val_{self.loss_name}_best.pth',
                                    step=self.global_step, metric=self.loss_name, val=self.best_loss)

            if self.global_step >= (self.conf.train_params.total_grad_steps):
                print('\nTotal number of gradient steps reached. Exiting.')
                break

    def get_model_dict(self, epoch):
        return {
            'epoch': epoch,
            'global_step': self.global_step,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'norm_params_input': self.model.norm_params_input,
            'norm_params_target': self.model.norm_params_target,
            'best_metric': self.best_loss,
            'inp_dims': self.model.inp_dims,
            'outp_dims': self.model.outp_dims
        }

    def val(self):
        self.model_test.load_state_dict(self.model.state_dict())
        losses_meter = RecursiveAverageMeter()
        if self.model_test.training:
            self.model_test.eval()
        with autograd.no_grad():
            for inputs, targets, flat_inputs in self.val_loader:
                inputs = map_dict(lambda x: x.to(self.device), inputs)
                targets = map_dict(lambda x: x.to(self.device), targets)
                # tuple(mean, var) both of size batch_size x target.dim
                cai_precision = self.compute_model_infl_precision(
                    **flat_inputs, factorized_states=inputs)
                loss = self.model_test.compute_loss(inputs, targets).cpu()
                losses = {self.loss_name: AttrDict(value=loss)}
                for k, v in cai_precision.items():
                    if v is not np.nan:
                        losses[k] = AttrDict(value=v)
                losses_meter.update(losses)
                del losses, cai_precision
            results = losses_meter.avg
            self.model_test.log_outputs(model_output=None, inputs=None, losses=results, step=self.global_step,
                                        log_images=False, phase='val')
            print(('\nValidation set:'))
            for k, v in results.items():
                results[k] = v.value.item()
                print(f'{k}: {v}')
        return results

    @property
    def log_outputs_now(self):
        return self.global_step % self.log_outputs_interval == 0

    def setup_logging(self, conf):
        if not self.args.dont_save:
            print('Writing to the experiment directory: {}'.format(self.exp_path))
            if not os.path.exists(self.exp_path):
                os.makedirs(self.exp_path)
            if not os.path.exists(os.path.join(self.exp_path, 'cmd.txt')):
                save_cmd(self.exp_path)
                save_git(self.exp_path)
                id_wandb = None
            else:
                with open(os.path.join(self.exp_path, 'id_wandb'), 'r') as file:
                    id_wandb = yaml.safe_load(file)

            exp_name = self.exp_path.split('/')[-1]
            env_name = self.conf.env_config.name
            if 'kitchen' in env_name:
                env_name += '_' + self.conf.path_data.split('/')[-1][:-4]
            sweep_name = self.args.sweep_name
            group = env_name + '_' + self.conf.model_params.model_type + '_' + sweep_name
            logger = WandBLogger(exp_name, WANDB_PROJECT_NAME, entity=WANDB_ENTITY_NAME,
                                 path=self.exp_path, conf=conf, wandb_mode=WANDB_MODE, id=id_wandb, group=group)
        else:
            logger = None

        return logger

    def setup_device(self):
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device(
            'cuda') if self.use_cuda else torch.device('cpu')

    def build_phase(self, phase):

        self.batch_size = self.conf.train_params.batch_size
        self.env = conf_utils.get_env(
            self.conf.env_config.name, self.conf.env_config)
        loader = self.get_dataloader(phase)

        model = build_model(self.conf, logger=self.logger,
                            device=self.device, loss_fn=self.loss_fn, env=self.env)

        if phase != 'train':  # load norm params from train
            model.norm_params_input = self.model.norm_params_input
            model.norm_params_target = self.model.norm_params_target
        else:
            model.norm_params_input = rec_map_dict(lambda x: torch.Tensor(
                x).to(self.device), loader.dataset.norm_params_input)
            model.norm_params_target = rec_map_dict(lambda x: torch.Tensor(
                x).to(self.device), loader.dataset.norm_params_target)

        return model, loader

    def get_dataloader(self, phase):
        factorizer = self.env.get_input_factorizer
        target_factorizer = self.env.get_output_factorizer
        path = self.conf.path_data if len(
            self.conf.path_data) > 0 else self.conf.env_config.name
        try:
            data_percentages = {'expert_percent': self.conf.data_params.expert_percent,
                                'random_percent': self.conf.data_params.random_percent, 'shrink_dataset': self.conf.data_params.shrink_dataset}
        except:
            import collections
            data_percentages = collections.defaultdict()
        # TODO make nicer

        loader = FactorizedForwardDataset(path=path, factorizer=factorizer, target_factorizer=target_factorizer, phase=phase,
                                            use_state_diff_as_target=self.conf.model_params.use_state_diff,
                                            normalize_data=self.conf.train_params.normalize_data,
                                            split_ratios=self.conf.train_params.split_ratios,
                                            device=self.device.type,
                                            **data_percentages). \
            get_data_loader(self.conf.train_params.batch_size)

        return loader

    def resume(self, resume_epoch='latest'):
        path = os.path.join(
            self.exp_path, 'weights')
        if not os.path.exists(path):
            print('\nStarting training from scratch\n')
            start_epoch = 0
            self.best_loss = np.Inf
            return start_epoch

        print('\nResuming from weights found in {}'.format(path))
        weights_file = CheckpointHandler.get_resume_ckpt_file(
            resume_epoch, path)
        self.global_step, start_epoch, _, self.best_loss = \
            CheckpointHandler.load_weights(weights_file, self.model,
                                           load_step=True, load_opt=True,
                                           optimizer=self.optimizer,
                                           strict=True)
        self.model.device = self.device
        return start_epoch

    def compute_model_infl_precision(self, states, actions, factorized_states):
        # if 'kitchen' in self.conf.env_config.name:
        return {}
        thresholds = [0.1, 0.3, 0.5, 1., 2.]
        cais = self.compute_cais(states, actions)
        precision = {}
        for k, v in factorized_states.items():
            if k not in ['agent', 'action']:
                influence = (np.linalg.norm(
                    factorized_states['agent'][:, 0:3].cpu().numpy() - v[:, 0:3].cpu().numpy(), axis=1) < 0.07).astype(int)
                zero_division = np.nan if np.sum(influence) == 0 else 0.
                for thr in thresholds:
                    pred_influence = (cais[f'score_{k}'] > thr).astype(int)
                    # precision = TP/(TP+FP) ability of the classifier not to label a negative sample as positive.
                    prec = precision_recall_fscore_support(
                        influence, pred_influence, average='binary', zero_division=zero_division)[0]
                    precision[f'prec_{k}_{thr}'] = prec
        return precision

    def compute_cais(self, states, actions):
        if self.model_test.training:
            self.model_test.eval()
        self.cai_computer = CMIScorer(
            self.model_test) if not self.conf.model_params.model_type == 'Transformer' else MaskScorer(self.model_test)
        cais = self.cai_computer(states, actions)
        return cais


def build_model(args, logger=None, device=None, loss_fn=None, env=None):
    dims = env.shapes
    model_params = get_model_params(args)
    try:
        a = args.model_params.model_type
    except AttributeError:
        args.model_params.model_type = 'FactorizedMLP'
    model = conf_utils.get_model(class_name=args.model_params.model_type,
                                 inp_dim=dims.input, outp_dim=dims.output, device=device, logger=logger, **model_params, loss_fn=loss_fn, input_factorizer=env.get_input_factorizer)
    return model


def get_model_params(args):
    dict_params = {}
    model_params = args.model_params
    if model_params.get('outp_layer', False):
        dict_params['outp_layer'] = model_params.outp_layer
        dict_params['outp_layer_params'] = model_params.outp_layer_params

    dict_params['use_state_diff_as_target'] = model_params.use_state_diff
    for k, v in model_params.items():
        dict_params[k] = v

    return dict_params


def save_checkpoint(state, folder, filename='checkpoint.pth', step: int = 0, metric: str = None, val: int = 0):
    os.makedirs(folder, exist_ok=True)
    torch.save(state, os.path.join(folder, filename))
    if metric is not None:
        print(
            f"Saved best model so far to {os.path.join(folder, filename)}! at {step} as {metric} improved to {val:.3f}")
    else:
        print(f"Saved checkpoint to {os.path.join(folder, filename)}!")


def load_weights(model, args, type: str = 'best'):
    weights_file = (
        '/'.join(args.model_params.config_path.split('/')[:-1])) + '/weights'
    if type == 'best':
        weights_file = CheckpointHandler.get_best_ckpt(weights_file)
    elif type == 'last':
        weights_file = CheckpointHandler.get_last_ckpt(weights_file)

    CheckpointHandler.load_weights(weights_file, model)

    model.weights_file = weights_file
    return model


def update_model_training_counter(path, counter):
    with open(os.path.join(path, 'counter'), 'w') as outfile:
        print(f'Saving current gradient step counter {counter} in ', path)
        yaml.dump(counter, outfile, default_flow_style=False)


def get_model_training_counter(path):
    filename = os.path.join(path, 'counter')
    if os.path.exists(filename):
        with open(filename, 'r') as outfile:
            counter = int(yaml.safe_load(outfile))
    else:
        counter = 0
    return counter
