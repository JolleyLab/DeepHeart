import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import nibabel as nib

import model.loss as module_loss
import model.metric as module_metric
import models as module_arch
from data_loader import CommonDataLoader
from utils.util import *


# NB: apply only to tricuspid
LEAFLET_LABEL_VALUE = {0: "anterior",
                       1: "posterior",
                       2: "septal"}


def main(config, resume):
    # setup data_loader instances
    data_loader = CommonDataLoader(
        config['data_loader']['data_dir'],
        batch_size=config['data_loader']['batch_size'],
        input_field=config["trainer"]["input_field"],
        output_field=config["trainer"]["output_field"],
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=config['data_loader']['num_workers'],
        inputs=config['data_loader']['inputs'],
        data_augmentation=config["data_augmentation"]
    )

    # build model architecture
    model = get_instance(module_arch, 'arch', config)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss']['type'])(**config['loss']['args'])
    loss_fn.training = False

    metrics = get_instances(module_metric, 'metrics', config)
    metrics = list(filter(lambda m: m.training is True, metrics))

    # load state dict
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    trainer_class = get_class(config["trainer"]["module"], "Trainer")

    total_loss = 0.0
    total_metrics = None

    cases_root_dir = os.path.join(os.path.dirname(resume),
                                  "Testing_{}".format(os.path.splitext(os.path.basename(resume))[0]))

    with torch.no_grad():

        for batch_idx, dataset in enumerate(tqdm(data_loader)):

            loss, output, data, target = trainer_class.predict(dataset, loss_fn, model, device)

            # max_idx = torch.argmax(output, dim=1, keepdim=True)
            # one_hot = torch.FloatTensor(output.shape).zero_()
            # output = one_hot.scatter_(1, max_idx, 1)1

            output = output[:, 1:, ...]
            target = target[:, 1:, ...]

            output = (output > 0.5).type(torch.FloatTensor)

            batch_size = data.shape[0]

            for idx in range(batch_size):
                case = dataset["cases"][idx]

                path = os.path.join(cases_root_dir, "Case_{}".format(case))
                if not os.path.exists(path):
                    os.makedirs(path)

                affine = dataset['affines'][idx]
                save_img(data[idx].cpu(), os.path.join(path, "InputImage.nii.gz"), affine)

                num_channels = target[idx].shape[0]
                for ch in range(num_channels):
                    save_img(target[idx].cpu(),
                             os.path.join(path, "InputLabelGT_{}.nii.gz".format(ch if num_channels == 1
                                                                                else LEAFLET_LABEL_VALUE[ch])),
                             affine, ch)

                    # NB: following line is for binary label only!
                    img = nib.Nifti1Image(output[idx][ch].cpu().numpy().astype(np.float32), affine)

                    # img = nib.Nifti1Image(output[idx][0].cpu().numpy(), affine[idx])
                    nib.save(img,
                             os.path.join(path, "OutputLabelDL_{}.nii.gz".format(ch if num_channels == 1
                                                                                 else LEAFLET_LABEL_VALUE[ch])))

            total_loss += loss.item()
            eval_metrics = trainer_class.eval_metrics(output, target, metrics)

            if total_metrics is None:
                total_metrics = eval_metrics
            else:
                total_metrics = dict((k, v + total_metrics[k]) for k, v in eval_metrics.items())

    log = {
        'loss': total_loss / len(data_loader),
        'metrics': dict((k, np.array(v).mean() if type(v) is list else v / len(data_loader)) for k, v in total_metrics.items())
    }

    with open(os.path.join(cases_root_dir, "loss.log"), "w+") as f:
        f.write(log.__str__())

    print(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    print("Processing {}".format(os.path.basename(args.resume)))

    main(config, args.resume)
