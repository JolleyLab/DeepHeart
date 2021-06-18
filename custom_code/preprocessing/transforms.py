import nibabel as nib
import torch
import os
import numpy as np
from utils.util import np_to_one_hot, one_hot_to_dist


class OneHotTransform(object):

    def __init__(self, in_field, out_field=None):
        self.in_field = in_field
        self.out_field = out_field if out_field else in_field

    def __call__(self, data):
        d = data[self.in_field]
        data[self.out_field] = np_to_one_hot(d)
        assert np.isfinite(data[self.out_field]).all()
        return data


class DistanceTransform(object):
    """ Create distance map on the fly for labels
    """

    def __init__(self, in_field, out_field=None):
        self.in_field = in_field
        self.out_field = out_field if out_field else in_field

    def __call__(self, data):
        d = data[self.in_field]
        data[self.out_field] = one_hot_to_dist(d).astype(np.float32)[1:, ...]
        assert np.isfinite(data[self.out_field]).all()
        return data


class NumpyFromFilename(object):

    def __init__(self, field):
        self.field = field

    def __call__(self, data):
        entry = nib.load(data[self.field])
        data[self.field] = entry.get_data().astype(np.float32)
        return data


class AffineFromFilename(object):

    def __init__(self, in_field, out_field):
        self.in_field = in_field
        self.out_field = out_field

    def __call__(self, data):
        assert os.path.exists(data[self.in_field])
        entry = nib.load(data[self.in_field])
        data[self.out_field] = entry.affine
        return data


class LabelFromFilename(object):

    def __init__(self, field, out_field=None, merged=True):
        self.field = field
        self.out_field = field if not out_field else out_field
        self.merged = merged

    def __call__(self, data):
        entry = nib.load(data[self.field])
        img_t = entry.get_data().astype(np.float32)
        if self.merged:
            img_t = (img_t > 0.5).astype(np.float32)
        data[self.out_field] = img_t.astype(np.float32)
        return data


class ToTensor(object):

    def __init__(self, field, dtype=np.float32):
        self.field = field
        self.dtype = dtype

    def __call__(self, data):
        entry_t = data[self.field]
        if len(entry_t.shape) == 3:
            entry_t = entry_t[np.newaxis]
        data[self.field] = torch.from_numpy(entry_t.astype(self.dtype))
        return data


class Concatenate(object):
    """ Concatenate numpy arrays
    """

    def __init__(self, in_fields, out_field):
        self.in_fields = in_fields
        self.out_field = out_field

    def __call__(self, data):
        to_concatenate = list()
        for field in self.in_fields:
            d = data[field]
            if len(d.shape) == 3:
                d = d[np.newaxis]
            to_concatenate.append(d)
        data[self.out_field] = np.concatenate(to_concatenate, axis=0)
        return data
    
    
class BorderTransform(object):

    def __init__(self, in_field, out_field=None, boundaries_pool=3, epsilon=1e-07):
        self.in_field = in_field
        self.out_field = out_field if out_field else in_field
        self.boundaries_pool = boundaries_pool
        self.epsilon = epsilon

    def __call__(self, data):
        d = data[self.in_field]
        padding = max((d.shape[1] - 1) + self.boundaries_pool - d.shape[1], 0) // 2
        d_avg = torch.nn.AvgPool3d(self.boundaries_pool, stride=1, padding=padding)(torch.tensor(d))
        reshaped_boundaries = (d_avg >= self.epsilon).short() & (d_avg <= 1.0 - self.epsilon).short()
        data[self.out_field] = reshaped_boundaries.numpy()
        assert np.isfinite(data[self.out_field]).all()
        return data

