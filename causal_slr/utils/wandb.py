import wandb
import inspect
import numpy as np
import torch
import os
from causal_slr.utils.general_utils import flatten_dict, prefix_dict
import csv


"Code in this file adapted from: Accelerating Reinforcement Learning with Learned Skill Priors: https://github.com/clvrai/spirl/tree/master"


class WandBLogger:
    """Logs to WandB."""
    N_LOGGED_SAMPLES = 3    # how many examples should be logged in each logging step

    def __init__(self, exp_name, project_name, entity, path, conf, exclude=None, wandb_mode='default', id = None, group: str = None):
        """
        :param exp_name: full name of experiment in WandB
        :param project_name: name of overall project
        :param entity: name of head entity in WandB that hosts the project
        :param path: path to which WandB log-files will be written
        :param conf: hyperparam config that will get logged to WandB
        :param exclude: (optional) list of (flattened) hyperparam names that should not get logged
        """
        if exclude is None:
            exclude = []
        flat_config = flatten_dict(conf)
        filtered_config = {k: v for k, v in flat_config.items() if (
            k not in exclude and not inspect.isclass(v))}
        if wandb_mode != 'default':
            print('***************WAND DISABLED*********')
            wandb.init(mode='disabled')
        else:
            print("INIT WANDB")
            if id is None:
                id = wandb.util.generate_id()
                with open(os.path.join(path, 'id_wandb'), 'w') as outfile:
                    print(f'Saving id {id} in ', path)
                    outfile.write(id)

            wandb.init(
                name=exp_name,
                project=project_name,
                config=filtered_config,
                dir=path,
                entity=entity,
                notes=conf.notes if 'notes' in conf else '',
                resume='allow',
                id=id,
                group=group
            )
        file_path = os.path.join(path, 'metrics.csv')
        self.keys = True if os.path.isfile(file_path) else None
        self.file = open(file_path, 'a', newline='')
        self.csv_writer = csv.writer(self.file)

    def log_csv_file(self, d):
        if not self.keys:
            self.keys = list(d.keys())
            self.csv_writer.writerow(self.keys)
        self.csv_writer.writerow(list(d.values()))
        self.file.flush()

    def close_csv_file(self):
        self.file.close()

    def log_scalar_dict(self, d, prefix='', step=None):
        """Logs all entries from a dict of scalars. Optionally can prefix all keys in dict before logging."""
        if prefix:
            d = prefix_dict(d, prefix + '_')
        wandb.log(d) if step is None else wandb.log(d, step=step)

    def log_scalar(self, v, k, step=None, phase=''):
        if phase:
            k = phase + '/' + k
        self.log_scalar_dict({k: v}, step=step)

    def log_histogram(self, array, name, step=None, phase=''):
        if phase:
            name = phase + '/' + name
        if isinstance(array, torch.Tensor):
            array = array.cpu().detach().numpy()
        wandb.log({name: wandb.Histogram(array)}, step=step)

    def log_videos(self, vids, name, step=None, fps=20):
        """Logs videos to WandB in mp4 format.
        Assumes list of numpy arrays as input with [time, channels, height, width]."""
        assert len(vids[0].shape) == 4 and vids[0].shape[1] == 3
        assert isinstance(vids[0], np.ndarray)
        if vids[0].max() <= 1.0:
            vids = [np.asarray(vid * 255.0, dtype=np.uint8) for vid in vids]
        log_dict = {
            name: [wandb.Video(vid, fps=fps, format="mp4") for vid in vids]}
        wandb.log(log_dict) if step is None else wandb.log(log_dict, step=step)

    def log_gif(self, v, k, step=None, phase='', fps=20):
        if phase:
            k = phase + '/' + k
        if len(v[0].shape) != 4:
            v = v.unsqueeze(0)
        if isinstance(v, torch.Tensor):
            v = v.cpu().detach().numpy()
        self.log_videos(v, k, step=step, fps=fps)

    def log_images(self, images, name, step=None, phase=''):
        if phase:
            name = phase + '/' + name
        if len(images.shape) == 4:
            for img in images:
                wandb.log({name: [wandb.Image(img)]})
        else:
            wandb.log({name: [wandb.Image(images)]})

    def log_plot(self, fig, name, step=None):
        """Logs matplotlib graph to WandB.
        fig is a matplotlib figure handle."""
        img = wandb.Image(fig)
        wandb.log({name: img}) if step is None else wandb.log(
            {name: img}, step=step)

    @property
    def n_logged_samples(self):
        return self.N_LOGGED_SAMPLES

    def visualize(self, *args, **kwargs):
        """Subclasses can implement this method to visualize training results."""
        pass

    def wandb_finish(self):
        wandb.finish()
