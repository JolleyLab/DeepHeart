import numpy as np
from scipy import ndimage
from sklearn import neighbors

import torch
import torch.utils.data
from ignite.metrics.accuracy import Accuracy
from ignite.metrics.mean_squared_error import MeanSquaredError
from medpy.metric.binary import hd as max_hausdorff_distance
from medpy.metric.binary import obj_asd as avg_hausdorff_distance


class BaseMetric(object):

    @property
    def training(self):
        return self._training

    @training.setter
    def training(self, value):
        assert type(value) is bool
        self._training = value

    def __init__(self, training=True):
        self.training = training


class Dice(BaseMetric):

    def __init__(self, training=True, per_class=True, eps=1e-5):
        self.per_class = per_class
        self.eps = eps
        super(Dice, self).__init__(training)

    def _dice(self, num, den1, den2):
        return 2.0 * num / (den1 + den2 + self.eps)

    def __call__(self, output, target):
        assert target.shape[1] == output.shape[1]
        num = (output * target).sum(dim=4, keepdim=True).sum(dim=3, keepdim=True).sum(dim=2, keepdim=True)
        den1 = output.pow(2).sum(dim=4, keepdim=True).sum(dim=3, keepdim=True).sum(dim=2, keepdim=True)
        den2 = target.pow(2).sum(dim=4, keepdim=True).sum(dim=3, keepdim=True).sum(dim=2, keepdim=True)
        dice = self._dice(num, den1, den2)
        if self.per_class:
            return dice.mean(dim=0).squeeze().numpy()
        else:
            return dice.mean().numpy()


class DistanceMapDice(Dice):

    def __call__(self, output, target):
        return super(DistanceMapDice, self).__call__((torch.nn.Sigmoid()(output) < 0.5).float(), (target < 0.5).float())


class DistanceACC(BaseMetric):

    def __init__(self, training=True, is_multilabel=False):
        self.accuracy = Accuracy(is_multilabel=is_multilabel)
        super(DistanceACC, self).__init__(training)

    def __call__(self, output, target):
        self.accuracy.update(((output < 0.5).long(), (target < 0.5).long()))
        return self.accuracy.compute()


class MBD(BaseMetric):

    def __init__(self, training=True, per_class=True):
        self.per_class = per_class
        super(MBD, self).__init__(training)

    def _mbd(self, output, target):
        # 1. Find boundary points
        # Convolve with all 1
        kernel = np.ones(np.repeat(3, output.ndim))
        b1 = ndimage.convolve(output, kernel)
        b2 = ndimage.convolve(target, kernel)
        nMax = 3 ** output.ndim
        b1[b1 > (nMax - 0.5)] = 0
        b2[b2 > (nMax - 0.5)] = 0
        x1 = np.argwhere(b1 > 0.5)
        x2 = np.argwhere(b2 > 0.5)

        # 2. Compute nearest neighbors distances
        nbrs1 = neighbors.NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(x1)
        nbrs2 = neighbors.NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(x2)
        d1, i1 = nbrs1.kneighbors(x2)
        d2, i2 = nbrs2.kneighbors(x1)
        return np.mean(d2), np.mean(d1)

    def __call__(self, output, target):
        assert target.shape[1] == output.shape[1]

        output = output.squeeze().numpy()
        target = target.squeeze().numpy()

        if len(output.shape) == 5:
            raise NotImplementedError("Currently no support for MBD with batch size >1")

        mbds = list()
        for cls in range(output.shape[0]):
            mbds.append(self._mbd(output[cls], target[cls]))

        if self.per_class:
            return np.array(mbds)
        else:
            np.array(mbds).mean(axis=0)


class Hausdorff(object):

    def __init__(self, training=True, per_class=False, reduction="avg"):
        self.per_class = per_class
        self.hausdorff_distance = avg_hausdorff_distance if reduction == "avg" else max_hausdorff_distance
        super(Hausdorff, self).__init__(training)

    def __call__(self, output, target, pix_dims=None):
        # TODO: need to split out per class and overall average
        assert target.shape[1] == output.shape[1]

        output = output.numpy()
        target = target.numpy()

        distances = list()

        for ch in range(output.shape[1]):
            distances.append(np.array([self.hausdorff_distance(target[bIdx, ch], output[bIdx, ch], pix_dims)
                                       for bIdx in range(output.shape[0])]).mean())
        return np.array(distances)


class MSE(BaseMetric):

    def __call__(self, output, target):
        mse = MeanSquaredError()
        mse.update((output, target))
        return mse.compute()


class ACC(BaseMetric):

    def __init__(self, training=True, is_multilabel=False):
        self.accuracy = Accuracy(is_multilabel=is_multilabel)
        super(ACC, self).__init__(training)

    def __call__(self, output, target):
        self.accuracy.update((output.long(), target.long()))
        return self.accuracy.compute()
