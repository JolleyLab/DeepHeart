import numpy as np


class MinMaxScaling:
    """
    Normalizes a given input tensor's data to lie within range 0 and 1
    """

    def __init__(self, field):
        self.field = field

    def _normalize(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def __call__(self, data):
        d = data[self.field]

        if len(d.shape) == 3:
            data[self.field] = self._normalize(d)
        elif len(d.shape) == 4:
            data[self.field] = \
                np.concatenate([self._normalize(d[idx])[np.newaxis] for idx in range(d.shape[0])], axis=0)
        return data


class NormalizeDistanceMap:
    """
    Normalizes a given input tensor's data to lie within range 0 and 1

    To get the original segmentation
    threshold to 0.5 with img < 0.5

    """

    def __init__(self, field):
        self.field = field

    def _normalize(self, data):
        data -= 0.5
        return (data + np.max(abs(data))) / (2 * np.max(abs(data)))

    def __call__(self, data):
        d = data[self.field]

        if len(d.shape) == 3:
            data[self.field] = self._normalize(d)
        elif len(d.shape) == 4:
            data[self.field] = \
                np.concatenate([self._normalize(d[idx])[np.newaxis] for idx in range(d.shape[0])], axis=0)
        else:
            raise NotImplementedError
        return data
