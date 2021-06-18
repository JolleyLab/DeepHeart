import datetime
import json
import logging
import math
import os
import shutil
import random
import string
import numpy as np
import torch

from utils.util import ensure_dir
from utils.visualization import WriterTensorboardX


class BaseTrainer:
    """
    Base class for all trainers
    """

    @staticmethod
    def prepare_device(n_gpu_use, logger=None):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        log_method = print if not logger else logger.warning
        if n_gpu_use > 0 and n_gpu == 0:
            log_method("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            log_method("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this "
                       "machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    @staticmethod
    def eval_metrics(output, target, metrics, writer=None):
        metric_results = dict()
        for i, metric in enumerate(metrics):
            metric_result = metric(output, target)

            main_metric_name = metric.__class__.__name__
            if type(metric_result) is np.ndarray:
                scalars = dict()
                size = metric_result.size
                if size > 1:
                    for resIdx in range(size):
                        metric_name = '{}_{}'.format(main_metric_name, resIdx)
                        scalars[metric_name] = metric_result[resIdx]
                    scalars['{}_avg'.format(main_metric_name)] = metric_result.mean(axis=0)
                else:
                    scalars['{}'.format(main_metric_name)] = metric_result
                if writer:
                    for scalar, value in scalars.items():
                        writer.add_scalar(scalar, value)
                metric_results.update(scalars)
            else:
                if writer:
                    writer.add_scalar(main_metric_name, metric_result)
                metric_results[main_metric_name] = metric_result
        return metric_results

    def __init__(self, model, loss, metrics, optimizer, resume, config, train_logger=None):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self.prepare_device(config['n_gpu'], self.logger)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()

        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.train_logger = train_logger

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.verbosity = cfg_trainer['verbosity']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = math.inf if self.mnt_mode == 'min' else -math.inf
            self.early_stop = cfg_trainer.get('early_stop', math.inf)

        self.start_epoch = 1

        # setup directory for checkpoint saving
        start_date = datetime.datetime.now().strftime('%m%d%Y')
        start_time = datetime.datetime.now().strftime('%H%M%S')
        random_hash = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
        directory = "{}/{}_{}".format(start_date, start_time, random_hash)
        self.checkpoint_dir = os.path.join(cfg_trainer['save_dir'], config['name'], directory)
        print("Tensorboard directory: %s" % self.checkpoint_dir)
        # setup visualization writer instance
        train_writer_dir = os.path.join(cfg_trainer['log_dir'], config['name'], directory+"_training")
        val_writer_dir = os.path.join(cfg_trainer['log_dir'], config['name'], directory+"_validation")

        self.train_writer = WriterTensorboardX(train_writer_dir, self.logger, cfg_trainer['tensorboardX'])
        self.val_writer = WriterTensorboardX(val_writer_dir, self.logger, cfg_trainer['tensorboardX'])

        # Save configuration file into checkpoint directory:
        ensure_dir(self.checkpoint_dir)
        config_save_path = os.path.join(self.checkpoint_dir, 'config.json')
        copied_config = config.copy()
        with open(config_save_path, 'w') as handle:
            json.dump(copied_config, handle, indent=4, sort_keys=False)

        # save entire source code into checkpoint directory
        source_code_dir = os.path.join(config["source_code_directory"])
        ensure_dir(source_code_dir)
        shutil.copytree(source_code_dir, os.path.join(self.checkpoint_dir, "source_code"))

        if resume:
            self._resume_checkpoint(resume)

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs+1):
            result = self._train_epoch(epoch)

            # save logged information into log dict
            log = dict()
            for key, value in result.items():
                if key == 'metrics':
                    log.update(value)
                elif key == 'val_metrics':
                    log.update({'val_' + mtr: val for mtr, val in value.items()})
                else:
                    log[key] = value

            log['epoch'] = epoch

            # print logged information to the screen
            if self.train_logger is not None:
                self.train_logger.add_entry(log)
                if self.verbosity >= 1:
                    for key, value in log.items():
                        self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] < self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] > self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False
                    not_improved_count = 0

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0 or best is True:
                if epoch % self.save_period == 0:
                    self._save_checkpoint(epoch)
                if best is True:
                    self._save_best_checkpoint(epoch)


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _save_checkpoint(self, epoch):
        """
        Saving checkpoints

        :param epoch: current epoch number
        """
        state = self._get_state(epoch)
        filename = os.path.join(self.checkpoint_dir, 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _get_state(self, epoch):
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'logger': self.train_logger,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        return state

    def _save_best_checkpoint(self, epoch):
        state = self._get_state(epoch)
        best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
        torch.save(state, best_path)
        self.logger.info("Saving current best: {} ...".format('model_best.pth'))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['loss']['type'] != self.config['loss']['type']:
            self.logger.warning("Info: Loss type given in config file is different from that of checkpoint.")

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.train_logger = checkpoint['logger']
        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))

    def addVideos(self, data, target, output, writer):
        permutations = {"axial": (0, 4, 1, 2, 3), "coronal": (0, 2, 1, 4, 3)} #, "sagittal": (0, 3, 1, 4, 2)}

        mode = writer.mode

        for orientation, permutation in permutations.items():
            writer.mode = "{}_images_{}".format(mode, orientation)
            for cGtdx in range(data.shape[1]):
                img = data[:, cGtdx, ...][:, np.newaxis, ...]
                img = self.__prepare_video(img.permute(*permutation).cpu())
                writer.add_video('image_{}'.format(cGtdx), img, fps=25)

            writer.mode = "{}_gt_labels_{}".format(mode, orientation)
            for cGtdx in range(target.shape[1]):
                gt = target[:, cGtdx, ...][:, np.newaxis, ...]
                gt = self.__prepare_video(gt.permute(*permutation).cpu())
                writer.add_video('label_{}'.format(cGtdx), gt, fps=25)

            writer.mode = "{}_pred_labels_{}".format(mode, orientation)
            for cPredIdx in range(output.shape[1]):
                pred = output[:, cPredIdx, ...][:, np.newaxis, ...]
                pred = self.__prepare_video(pred.permute(*permutation).cpu())
                writer.add_video('label_{}'.format(cPredIdx), pred, fps=25)

        writer.mode = mode

    def __prepare_video(self, data):

        b, t, c, h, w = data.shape

        def is_power2(num):
            return num != 0 and ((num & (num - 1)) == 0)

        # pad to nearest power of 2, all at once
        if not is_power2(data.shape[0]):
            len_addition = int(2 ** data.shape[0].bit_length() - data.shape[0])
            data = np.concatenate(
                (data, np.zeros(shape=(len_addition, t, c, h, w))), axis=0)
        return data
