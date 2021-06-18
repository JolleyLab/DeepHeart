from torch import nn
import torch


EPS = 0.00001


class DiceLoss(nn.Module):
    """ formula: DICE=2 * (Gt * Pred) / (Gt^2 + Pred^2) """

    def forward(self, output, target):
        num = (output * target).sum(dim=4, keepdim=True).sum(dim=3, keepdim=True).sum(dim=2, keepdim=True)
        den1 = output.pow(2).sum(dim=4, keepdim=True).sum(dim=3, keepdim=True).sum(dim=2, keepdim=True)
        den2 = target.pow(2).sum(dim=4, keepdim=True).sum(dim=3, keepdim=True).sum(dim=2, keepdim=True)

        dice = (2.0 * num / (den1 + den2 + EPS))

        return (1.0 - dice).mean()


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

    def forward(self, y_pred, y_true):
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


class DicePlusConstantCatCrossEntropyLoss(nn.Module):

    def __init__(self, boundaries_weight_factor, boundaries_pool, weights=None, scale=False, sigma=0.02,
                 multiply_boundaries_over_all_channels=True):
        super(DicePlusConstantCatCrossEntropyLoss, self).__init__()
        self.diceLoss = DiceLoss() # TODO: use MONAI implementation
        self.boundaryLoss = CategoricalCrossentropyLoss(boundaries_weight_factor=boundaries_weight_factor,
                                                        boundaries_pool=boundaries_pool,
                                                        scale=scale,
                                                        weights=weights,
                                                        multiply_boundaries_over_all_channels=multiply_boundaries_over_all_channels)
        self.sigma = sigma

    def forward(self, pred, label):
        diceLoss = self.diceLoss(pred, label)
        boundaryLoss = self.boundaryLoss(pred, label)
        loss = diceLoss + self.sigma * boundaryLoss
        return loss