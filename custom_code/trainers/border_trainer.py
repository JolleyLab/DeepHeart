import torch
from .trainer import Trainer as BaseTrainer


class Trainer(BaseTrainer):

    @staticmethod
    def run_model_and_calculate_loss(dataset, loss, model, device, epoch=1):
        image = dataset["images"]
        label = dataset["labels"]
        if device.type == "cuda":
            image, label = image.cuda(), label.cuda()

        output = model(image)

        output = torch.nn.functional.softmax(output.reshape(output.size(0), output.size(1), -1),
                                             dim=1).view_as(output)

        loss = loss(output, label, epoch)

        output = output[:, 1:, ...]
        label = label[:, 1:, ...]

        output = output.detach().cpu()
        image = image.detach().cpu()
        label = label.detach().cpu()

        return loss, output, image, label

    @staticmethod
    def predict(dataset, loss, model, device):
        image = dataset["images"]
        label = dataset["labels"]
        if device.type == "cuda":
            image, label = image.cuda(), label.cuda()

        output = model(image)

        output = torch.nn.functional.softmax(output.reshape(output.size(0), output.size(1), -1),
                                             dim=1).view_as(output)

        loss = loss(output, label)

        output = output.detach().cpu()
        image = image.detach().cpu()
        label = label.detach().cpu()

        return loss, output, image, label

    def __init__(self, model, loss, metrics, optimizer, resume, config, data_loader, valid_data_loader=None,
                 lr_scheduler=None, train_logger=None):
        super(Trainer, self).__init__(model, loss, metrics, optimizer, resume, config, data_loader,
                                      valid_data_loader, lr_scheduler, train_logger)
