import os
import json
import argparse
import torch

from data_loader import CommonDataLoader
import model.loss as module_loss
import model.metric as module_metric
import models as module_arch
import optimizer.optimizer as optim

from utils import Logger
from utils.util import get_instance, get_instances, get_class, get_transform


def main(config):
    train_logger = Logger()
    resume = config.get("resume")

    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    # print(models)
    use_cuda = config['n_gpu'] > 0 and torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()

    # setup data_loader instances
    data_loader = CommonDataLoader(**config["data_loader"],
                                   input_field=config["trainer"]["input_field"],
                                   output_field=config["trainer"]["output_field"],
                                   data_augmentation=config["data_augmentation"],
                                   training=True,
                                   pin_memory=use_cuda)
    valid_data_loader = data_loader.split_validation()

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = getOptimizer(config, trainable_params)

    if config["mixed_precision"] is True and config['n_gpu'] > 0:
        from apex import amp
        amp.register_float_function(torch, 'sigmoid')
        model, optimizer = amp.initialize(model, optimizer)

    lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer)

    # get function handles of loss and metrics
    loss = getattr(module_loss, config['loss']['type'])(**config['loss']['args'])
    metrics = get_instances(module_metric, 'metrics', config)
    metrics = list(filter(lambda m: m.training is True, metrics))

    trainer_class = get_class(config["trainer"]["module"], "Trainer")

    trainer = trainer_class(model, loss, metrics, optimizer, resume=resume, config=config, data_loader=data_loader,
                            valid_data_loader=valid_data_loader, lr_scheduler=lr_scheduler, train_logger=train_logger)

    trainer.train()


def getOptimizer(config, trainable_params):
    try:
        return get_instance(torch.optim, 'optimizer', config, trainable_params)
    except AttributeError:
        return get_instance(optim, 'optimizer', config, trainable_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    config = None

    if args.config:
        # load config file
        with open(args.config) as handle:
            config = json.load(handle)
        # setting path to save trained model and log files
        path = os.path.join(config['trainer']['save_dir'], config['name'])
        import pathlib
        config["source_code_directory"] = str(pathlib.Path(__file__).parent.absolute())
    if args.resume:
        # load config from checkpoint if new config file is not given.
        # Use '--config' and '--resume' together to fine-tune trained model with changed configurations.
        if not config:
            config = torch.load(args.resume)['config']
        config["resume"] = args.resume

    if not config:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config)
