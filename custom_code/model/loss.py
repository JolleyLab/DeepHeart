from torch.nn.modules.loss import *
from torch import nn, einsum
import torch
from torch.autograd import Variable
from pdb import set_trace
import torch.nn.functional as F
import numpy as np
from utils.util import flatten, simplex, one_hot


EPS = 0.00001


class DiceLoss(nn.Module):
    """ formula: DICE=2 * (Gt * Pred) / (Gt^2 + Pred^2) """

    def forward(self, output, target):
        num = (output * target).sum(dim=4, keepdim=True).sum(dim=3, keepdim=True).sum(dim=2, keepdim=True)
        den1 = output.pow(2).sum(dim=4, keepdim=True).sum(dim=3, keepdim=True).sum(dim=2, keepdim=True)
        den2 = target.pow(2).sum(dim=4, keepdim=True).sum(dim=3, keepdim=True).sum(dim=2, keepdim=True)

        dice = (2.0 * num / (den1 + den2 + EPS))

        return (1.0 - dice).mean()


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.loss_fn(output, target.squeeze().long())


class CrossEntropy(nn.Module):
    """ source: https://github.com/LIVIAETS/surface-loss/blob/master/losses.py """

    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, probs, target):
        assert simplex(probs) and simplex(target)

        eps = 1e-10

        log_p = (probs + eps).log()
        mask = target.type(torch.float32)

        loss = - einsum("bcwhd,bcwhd->", mask, log_p)
        loss /= mask.sum() + eps

        return loss


class DicePlusXEntropyLoss(nn.Module):

    def __init__(self):
        super(DicePlusXEntropyLoss, self).__init__()
        self.diceLoss = DiceLoss()
        self.crossentropy = CrossEntropy()

    def forward(self, pred, label):
        loss = self.diceLoss(pred, label)
        crossentropy = self.crossentropy(pred, label)
        loss = loss + crossentropy
        return loss


class CategoricalCrossentropyLoss(nn.Module):
    """
    Categorical crossentropy with optional categorical weights and spatial prior

    Adapted from weighted categorical crossentropy via wassname:
    https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        loss = CategoricalCrossentropy()(x)
        loss = CategoricalCrossentropy(weights=weights)(x)
    """

    def __init__(self, weights=None, scale=True, boundaries_weight_factor=None,
                 boundaries_pool=None, epsilon=1e-07, multiply_boundaries_over_all_channels=True):
        super(CategoricalCrossentropyLoss, self).__init__()

        self.weights = weights if (weights is not None) else None
        if self.weights:
            self.weights = torch.tensor(self.weights).cuda()
        self.scale = scale
        self.epsilon = epsilon

        self.boundaries_weight_factor = boundaries_weight_factor
        self.boundaries_pool = boundaries_pool
        self.multiply_boundaries_over_all_channels = multiply_boundaries_over_all_channels

    def forward(self, y_pred, y_true, epoch=None):
        """ categorical crossentropy loss """

        if self.scale:
            y_pred = y_pred / torch.sum(y_pred, dim=1, keepdim=True)
        y_pred = torch.clamp(y_pred, self.epsilon, 1)

        # compute log probability
        log_post = y_pred.log()  # likelihood

        # loss
        cross_entropy = -y_true * log_post

        # we will add to this for each term in our loss function
        loss = cross_entropy

        if self.weights is not None:
            weights = self.weights.to(y_pred.device)
            loss = (loss.permute((0, 2, 3, 4, 1)) * weights).permute((0, 4, 1, 2, 3))

        weighted_boundaries = None
        with torch.no_grad():
            if self.boundaries_weight_factor is not None:
                # find the segmentation borders via average pooling:
                # the borders will be in the blurry region that is neither 0 nor 1

                padding = max((y_true.shape[2] - 1) + self.boundaries_pool - y_true.shape[2], 0) // 2
                y_true_avg = torch.nn.AvgPool3d(self.boundaries_pool, stride=1, padding=padding)(y_true)

                reshaped_boundaries = (y_true_avg >= self.epsilon).short() & (y_true_avg <= 1.0 - self.epsilon).short()

                boundaries = reshaped_boundaries.max(dim=1, keepdim=True)

                weighted_boundaries = boundaries[0].float() * self.boundaries_weight_factor

                del reshaped_boundaries
                del boundaries
                del y_true_avg

        if weighted_boundaries is not None:
            loss = loss + (cross_entropy * weighted_boundaries)
            del weighted_boundaries

        mloss = torch.mean(torch.sum(loss.float(), 1))
        assert torch.isfinite(mloss), 'Loss not finite'
        return mloss


class DicePlusCatCrossEntropyLoss(nn.Module):

    def __init__(self, boundaries_weight_factor, boundaries_pool, weights=None, scale=False, sigma=0.02,
                 multiply_boundaries_over_all_channels=True):
        super(DicePlusCatCrossEntropyLoss, self).__init__()
        self.diceLoss = DiceLoss()
        self.boundaryLoss = CategoricalCrossentropyLoss(boundaries_weight_factor=boundaries_weight_factor,
                                                        boundaries_pool=boundaries_pool,
                                                        scale=scale,
                                                        weights=weights,
                                                        multiply_boundaries_over_all_channels=multiply_boundaries_over_all_channels)
        self.primaryLossFn = self.diceLoss
        self.secondaryLossFn = self.boundaryLoss

        self.sigma = sigma
        self.factor = (1.0 - (1.0 - max(0.01, self.sigma)))

    def forward(self, pred, label, epoch=1):
        primaryLoss = self.primaryLossFn(pred, label)
        secondaryLoss = 0

        primaryFactor = 1
        secondayFactor = (1.0 - (1.0 - max(0.01, epoch * self.sigma)))
        secondayFactor = np.clip(secondayFactor, 0, 1)

        if self.training:
            primaryFactor = 1 - secondayFactor
            primaryFactor = np.clip(primaryFactor, 0, 1)
            secondaryLoss = self.secondaryLossFn(pred, label)

            if secondayFactor != self.factor:
                # logging.info("boundary loss factor changed to {}".format(secondayFactor))
                self.factor = secondayFactor
        loss = primaryFactor * primaryLoss + secondayFactor * secondaryLoss
        return loss


class DicePlusConstantCatCrossEntropyLoss(nn.Module):

    def __init__(self, boundaries_weight_factor, boundaries_pool, weights=None, scale=False, sigma=0.02,
                 multiply_boundaries_over_all_channels=True):
        super(DicePlusConstantCatCrossEntropyLoss, self).__init__()
        self.diceLoss = DiceLoss()
        self.boundaryLoss = CategoricalCrossentropyLoss(boundaries_weight_factor=boundaries_weight_factor,
                                                        boundaries_pool=boundaries_pool,
                                                        scale=scale,
                                                        weights=weights,
                                                        multiply_boundaries_over_all_channels=multiply_boundaries_over_all_channels)
        self.sigma = sigma

    def forward(self, pred, label, epoch=1):
        diceLoss = self.diceLoss(pred, label)
        boundaryLoss = self.boundaryLoss(pred, label)
        loss = diceLoss + self.sigma * boundaryLoss
        return loss


class CatCrossEntropyLossPlusDice(DicePlusCatCrossEntropyLoss):

    def __init__(self, boundaries_weight_factor, boundaries_pool, weights=None, scale=False, sigma=0.02,
                 multiply_boundaries_over_all_channels=True):
        super(CatCrossEntropyLossPlusDice, self).__init__(boundaries_weight_factor, boundaries_pool,
                                                          weights, scale, sigma, multiply_boundaries_over_all_channels)
        self.primaryLossFn = self.boundaryLoss
        self.secondaryLossFn = self.diceLoss


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.loss_fn = nn.BCELoss()

    def forward(self, output, target):
        return self.loss_fn(output, target)


class DicePlusBCELoss(nn.Module):
    def __init__(self):
        super(DicePlusBCELoss, self).__init__()
        self.diceLoss = DiceLoss()
        self.bceLoss = BCELoss()

    def forward(self, output, target):
        return self.diceLoss(output, target) + self.bceLoss(output, target)


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(BCEWithLogitsLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, output, target):
        return self.loss_fn(output, target)


class EuclideanLoss(nn.Module):
    def __init__(self, reduction="mean"):
        self.reduction = reduction
        super(EuclideanLoss, self).__init__()

    def forward(self, output, target):
        loss = torch.sqrt((output - target)**2)
        loss = loss.sum() if self.reduction == "sum" else loss.mean()
        return loss


class WeightedMSELoss(nn.Module):
    def __init__(self, sigma2=25, reduction="mean"):
        self.sigma2 = sigma2  # smaller numbers put more weight on segmentation boundary
        self.reduction = reduction
        super(WeightedMSELoss, self).__init__()

    def forward(self, output, target):
        weights = torch.exp(-(target * target) / self.sigma2)
        out = (output - target) ** 2
        out = out * weights.expand_as(out)
        return out.sum() if self.reduction == "sum" else out.mean()


class ACNNLoss(nn.Module):
    def __init__(self, aeModelPath, lambda_1=0.01, sigma=0.01, training=True):
        super(ACNNLoss, self).__init__()
        self.training = training
        self.autoencoder = None
        self.state_dict = torch.load(aeModelPath)['state_dict']
        self.lambda_1 = lambda_1  # TODO: has no effect right now
        self.sigma = sigma

        self.mainLoss = BCELoss()
        self.euclideanLoss = EuclideanLoss()

    def forward(self, output, target, epoch, model=None):

        losses = [self.mainLoss(output, target)]

        if self.training:
            if not self.autoencoder:
                self.autoencoder = model
                self.autoencoder.load_state_dict(self.state_dict)

            # threshold
            #t = Variable(torch.Tensor([0.5])).cuda()
            #output = (output > t).float() * 1

            aeOutputCodes = self.autoencoder.module.encoder(output)
            aeTargetCodes = self.autoencoder.module.encoder(target)

            loss = (1 - (1.0 - max(0.01, (epoch - 1) * self.sigma))) * self.euclideanLoss(aeOutputCodes, aeTargetCodes)
            losses.append(loss)

        print("Loss: " + "+".join(["{:.6f}".format(l) for l in losses]))
        return sum(losses)


class DiceBorderLoss(nn.Module):
    def __init__(self, boundaries_weight_factor, boundaries_pool, scale, lambda_1=0.01):
        super(DiceBorderLoss, self).__init__()
        self.lambda_1 = lambda_1

        self.diceLoss = DiceLoss()
        self.catXEntLoss = CategoricalCrossentropyLoss(boundaries_weight_factor=boundaries_weight_factor,
                                                       boundaries_pool=boundaries_pool,
                                                       scale=scale)

    def forward(self, output, target, model=None):
        dice = self.diceLoss(output, target)
        catXEnt = self.catXEntLoss(output, target)
        return sum([dice, self.lambda_1*catXEnt])


class DicePlusMSELoss(nn.Module):
    def __init__(self, lambda_1=0.2, reduction="mean"):
        super(DicePlusMSELoss, self).__init__()
        self.lambda_1 = lambda_1

        self.diceLoss = DiceLoss()
        self.mseLoss = MSELoss(reduction=reduction)

    def forward(self, output, target):
        mse = self.mseLoss(output, target)
        dice = self.diceLoss((output < 0.5).float(), (target < 0.5).float())
        return sum([mse * abs(1-self.lambda_1), self.lambda_1 * dice])


class SurfaceLoss(nn.Module):

    def __init__(self):
        super(SurfaceLoss, self).__init__()
        pass

    def forward(self, probs, dist_maps):
        assert simplex(probs, axis=1)
        assert not one_hot(dist_maps)

        pc = probs.type(torch.float32)[:, 1:, ...]
        dc = dist_maps.type(torch.float32)[:, 1:, ...]

        multipled = einsum("bcwhd,bcwhd->bcwhd", pc, dc)

        return multipled.mean()


# class DicePlusSurfaceLoss(nn.Module):
#
#     def __init__(self, sigma=0.02):
#         super(DicePlusSurfaceLoss, self).__init__()
#         self.diceLoss = DiceLoss()
#         self.surfaceLoss = SurfaceLoss()
#
#         self.sigma = sigma
#         self.factor = (1.0 - (1.0 - max(0.01, self.sigma)))
#
#     def forward(self, pred, distmap, label, epoch=1):
#         primaryLoss = self.diceLoss(pred, label)
#         secondaryLoss = 0
#
#         primaryFactor = 1
#         secondayFactor = (1.0 - (1.0 - max(0.01, epoch * self.sigma)))
#         secondayFactor = np.clip(secondayFactor, 0, 1)
#
#         if self.training:
#             primaryFactor = 1 - secondayFactor
#             primaryFactor = np.clip(primaryFactor, 0, 1)
#             secondaryLoss = self.surfaceLoss(pred, distmap)
#
#             if secondayFactor != self.factor:
#                 logging.info("surface loss factor changed to {}".format(secondayFactor))
#                 self.factor = secondayFactor
#         loss = primaryFactor * primaryLoss + secondayFactor * secondaryLoss
#         return loss


class DicePlusSurfaceLoss(nn.Module):

    def __init__(self, sigma=0.02):
        super(DicePlusSurfaceLoss, self).__init__()
        self.diceLoss = DiceLoss()
        self.surfaceLoss = SurfaceLoss()
        self.sigma = sigma

    def forward(self, pred, label, distmap=None, epoch=1):
        loss = self.diceLoss(pred, label)

        if self.training and distmap is not None:
            surfaceLoss = self.surfaceLoss(pred, distmap)
            loss = loss + self.sigma * surfaceLoss
        return loss


class BCEPlusSurfaceLoss(nn.Module):

    def __init__(self, sigma=0.02):
        super(BCEPlusSurfaceLoss, self).__init__()
        self.bceLoss = BCELoss()
        self.surfaceLoss = SurfaceLoss()
        self.sigma = sigma

    def forward(self, pred, distmap, label, alpha, epoch=1):
        loss = self.bceLoss(pred[:, 1:, ...], label)
        if self.training:
            boundaryLoss = self.surfaceLoss(pred, distmap)
            loss = loss + (1 - (1.0 - max(0.01, epoch * self.sigma))) * boundaryLoss
        return loss


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int, long)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, pred, target):
        if pred.dim() > 2:
            pred = pred.view(pred.size(0), pred.size(1), -1)  # N,C,H,W => N,C,H*W
            pred = pred.transpose(1, 2)    # N,C,H*W => N,H*W,C
            pred = pred.contiguous().view(-1, pred.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(pred)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != pred.data.type():
                self.alpha = self.alpha.type_as(pred.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

