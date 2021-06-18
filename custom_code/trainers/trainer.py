import os
import numpy as np
import torch
from base import BaseTrainer

from pdb import set_trace


class Trainer(BaseTrainer):

    @staticmethod
    def run_model_and_calculate_loss(dataset, loss, model, device, epoch=1):
        """ This method has to be implemented by subclasses

        :return: loss, output
        """
        raise NotImplementedError

    def __init__(self, model, loss, metrics, optimizer, resume, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, resume, config, train_logger)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.mixed_precision = config["mixed_precision"]
        self._write_data_split_files()

    def _write_data_split_files(self):
        if not hasattr(self.data_loader.dataset, "training_subset") \
           or not hasattr(self.data_loader.dataset, "testing_subset"):
            return
        import csv

        def write_data_split_file(file_list, csv_file):
            with open(csv_file, mode='w') as csv_stream:
                writer = csv.writer(csv_stream, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['Filename'])
                for f in file_list:
                    writer.writerow([f])

        write_data_split_file(self.data_loader.dataset.training_subset,
                              os.path.join(self.checkpoint_dir, 'training_data.csv'))

        write_data_split_file(self.data_loader.dataset.testing_subset,
                              os.path.join(self.checkpoint_dir, 'testing_data.csv'))

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        step = 0
        total_loss = 0
        total_metrics = None

        # data = target = output = None

        for batch_idx, dataset in enumerate(self.data_loader):

            self.optimizer.zero_grad()

            loss, output, data, target = self.run_model_and_calculate_loss(dataset, self.loss, self.model,
                                                                           self.device, epoch)

            if self.mixed_precision:
                from apex import amp
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            self.optimizer.step()

            step = (epoch - 1) * len(self.data_loader) + batch_idx

            total_loss += loss.item()

            self.train_writer.set_step(step)
            self.train_writer.add_scalar('loss', loss)

            eval_metrics = self.eval_metrics(output, target, self.metrics, self.train_writer)
            if total_metrics is None:
                total_metrics = eval_metrics
            else:
                total_metrics = dict((k, v+total_metrics[k]) for k, v in eval_metrics.items())

            if batch_idx % self.log_step == 0:
                if self.verbosity >= 2:
                    self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                        epoch,
                        batch_idx * self.data_loader.batch_size,
                        self.data_loader.n_samples,
                        100.0 * batch_idx / len(self.data_loader),
                        loss))
                # self.addVideos(data, target, self.get_visual_output(output), self.train_writer)

            del loss
            del output
            del data
            del target
            torch.cuda.empty_cache()

        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': dict((k, v/len(self.data_loader)) for k, v in total_metrics.items())
        }

        val_log = None
        if self.do_validation:
            val_log = self._valid_epoch(epoch, step)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(val_log["val_loss"])

        return log

    def _valid_epoch(self, epoch, currentStep):
        """ Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_loss = 0
        total_metrics = None

        with torch.no_grad():

            data = target = output = None

            for dataset in self.valid_data_loader:
                loss, output, data, target = self.run_model_and_calculate_loss(dataset, self.loss, self.model,
                                                                               self.device, epoch)
                self.val_writer.set_step(currentStep, mode='train')

                total_loss += loss.item()

                eval_metrics = self.eval_metrics(output, target,
                                                 self.metrics,
                                                 self.val_writer)
                if total_metrics is None:
                    total_metrics = eval_metrics
                else:
                    total_metrics = dict((k, v + total_metrics[k]) for k, v in eval_metrics.items())

                del loss
                torch.cuda.empty_cache()

            self.val_writer.set_step(currentStep, mode='validation')

            if epoch % 5 == 0:
                self.addVideos(data, target, self.get_visual_output(output), self.val_writer)

            del output
            del data
            del target

        self.val_writer.set_step(currentStep, mode='train')
        self.val_writer.add_scalar('loss', total_loss / len(self.valid_data_loader))

        return {
            'val_loss': total_loss / len(self.valid_data_loader),
            'val_metrics': dict((k, v / len(self.valid_data_loader)) for k, v in total_metrics.items())
        }

    def get_visual_output(self, output):
        return output

    def _run_model_and_calculate_loss(self, data, target):
        return self.run_model_and_calculate_loss(data, target, self.loss, self.model)
