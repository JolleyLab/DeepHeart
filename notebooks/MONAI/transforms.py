import SimpleITK as sitk
import numpy as np
import torch
import math
from monai.transforms import MapTransform, RandomizableTransform
from utils import is_one_hot
from scipy.ndimage import affine_transform
from monai.utils import ensure_tuple_rep


class OneHotTransform(object):

    @classmethod
    def run(cls, data):
        if len(data.shape) == 4:
            assert data.shape[0] == 1
            data = data[0]

        n_classes = (len(np.unique(data)))
        assert n_classes > 1, f"{cls.__name__}: Not enough unique pixel values found in data."
        assert n_classes < 10, f"{cls.__name__}: Too many unique pixel values found in data."

        w, h, d = data.shape
        res = np.stack([data == c for c in range(n_classes)], axis=0).astype(np.int32)
        assert res.shape == (n_classes, w, h, d)
        assert np.all(res.sum(axis=0) == 1)
        return res

    def __init__(self, fields):
        self.fields = fields

    def __call__(self, data):
        for field in self.fields:
            data[field] = self.run(data[field])
            assert np.isfinite(data[field]).all()
        return data


class OneHotTransformd(MapTransform):

    def __init__(self, keys):
        super(OneHotTransformd, self).__init__(keys)

    def __call__(self, data):
        for key in self.keys:
            one_hot = OneHotTransform.run(data[key])
            assert np.isfinite(one_hot).all()
            assert np.any(one_hot)

            data[key] = one_hot.astype(np.float32)
        return data


class DistanceTransform(object):
    """ Create distance map on the fly for labels
    """

    METHODS = {
        "SDM": sitk.SignedMaurerDistanceMapImageFilter,
        "EDM": sitk.DanielssonDistanceMapImageFilter
    }
    DEFAULT_METHOD = "SDM"

    @classmethod
    def get_distance_map(cls, data, method=DEFAULT_METHOD):
        image = sitk.GetImageFromArray(data.astype(np.int16))
        distanceMapFilter = cls.METHODS[method]()
        distanceMapFilter.SetUseImageSpacing(True)
        distanceMapFilter.SetSquaredDistance(False)
        out = distanceMapFilter.Execute(image)
        return sitk.GetArrayFromImage(out)

    def __init__(self, fields, method=DEFAULT_METHOD):
        self.fields = fields
        self.computationMethod = method

    def __call__(self, data):
        for field in self.fields:
            d = data[field]
            assert is_one_hot(torch.Tensor(d), axis=0)
            # NB: skipping computation of background distance map
            d = d[1:, ...]
            assert d.shape[0] > 0
            data[field] = np.stack([
                self.get_distance_map(d[ch].astype(np.float32), self.computationMethod) for ch in range(d.shape[0])],
                axis=0)
            assert np.isfinite(data[field]).all()
        return data


class DistanceTransformd(MapTransform):

    def one_hot_to_dist(self, input_array):
        assert is_one_hot(torch.Tensor(input_array), axis=0)
        out = np.stack(
            [DistanceTransform.get_distance_map(input_array[ch].astype(np.float32),
                                                method=self.method) for ch in range(input_array.shape[0])], axis=0)
        return out

    def __init__(self, keys, method=DistanceTransform.DEFAULT_METHOD):
        super(DistanceTransformd, self).__init__(keys)
        self.method = method

    def __call__(self, data):
        for key in self.keys:
            one_hot = OneHotTransform.run(data[key])
            assert np.isfinite(one_hot).all()
            assert np.any(one_hot)

            result_np = self.one_hot_to_dist(one_hot).astype(np.float32)
            data[key] = result_np[1:, ...]
        return data


class AffineTransform(object):

    def __init__(self, rx, ry, rz, tx, ty, tz, zoom):
        self.rx = rx
        self.ry = ry
        self.rz = rz
        self.tx = tx
        self.ty = ty
        self.tz = tz
        self.zoom = zoom

    @staticmethod
    def get_zoom_transform(zoom):
        zx = np.dot(zoom, [[1., 0., 0., 0.]])
        zy = np.dot(zoom, [[0., 1., 0., 0.]])
        zz = np.dot(zoom, [[0., 0., 1., 0.]])

        return np.concatenate([zx, zy, zz, [[0., 0., 0., 1.]]], axis=0)

    @staticmethod
    def get_shift_transform(tx, ty, tz):
        raz = np.zeros((1, 4))
        return np.concatenate([tx,ty,tz,raz], axis=0) + np.eye(4)

    @staticmethod
    def get_rotation_matrix(rx, ry, rz):
        o = np.ones((1, 1))
        z = np.zeros((1, 1))

        Rx = np.reshape(
            np.concatenate([o, z, z, z, z, np.cos(rx), -np.sin(rx), z, z, np.sin(rx), np.cos(rx), z, z, z, z, o]),
            (4, 4))
        Ry = np.reshape(
            np.concatenate([np.cos(ry), z, np.sin(ry), z, z, o, z, z, -np.sin(ry), z, np.cos(ry), z, z, z, z, o]),
            (4, 4))
        Rz = np.reshape(
            np.concatenate([np.cos(rz), -np.sin(rz), z, z, np.sin(rz), np.cos(rz), z, z, z, z, o, z, z, z, z, o]),
            (4, 4))
        return np.dot(Rx, np.dot(Ry, Rz))


    @staticmethod
    def get_affine_transform(grid_size, rx, ry, rz, tx, ty, tz, zoom):

        cx, cy, cz = [p / 2.0 for p in grid_size]
        ux, uy, uz = [-p / 2.0 for p in grid_size]

        centering_matrix = np.array([[1, 0, 0, cx], [0, 1, 0, cy], [0, 0, 1, cz], [0, 0, 0, 1]])
        uncentering_matrix = np.array([[1, 0, 0, ux], [0, 1, 0, uy], [0, 0, 1, uz], [0, 0, 0, 1]])

        transform_matrix = centering_matrix

        rotation_matrix = AffineTransform.get_rotation_matrix(rx, ry, rz)
        transform_matrix = np.dot(transform_matrix, rotation_matrix)

        shift_matrix = AffineTransform.get_shift_transform(tx, ty, tz)
        transform_matrix = np.dot(transform_matrix, shift_matrix)

        zoom_matrix = AffineTransform.get_zoom_transform(zoom)
        transform_matrix = np.dot(transform_matrix, zoom_matrix)

        # un-shift from center
        transform_matrix = np.dot(transform_matrix, uncentering_matrix)

        return transform_matrix

    def __call__(self, data, order):
        grid_size = data.shape[-3:]
        transform = self.get_affine_transform(grid_size, self.rx, self.ry, self.rz, self.tx, self.ty, self.tz, self.zoom)

        if data.ndim == 3:
            data = affine_transform(data, transform, order=order, mode='constant')
        else:
            raise NotImplementedError
        return data


class RandAffineTransformd(RandomizableTransform, MapTransform):

    def __init__(self, keys, rotation_range, shift_range, zoom_range, interpolation_order, prob=0.1, allow_missing_keys=False):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.rotation_range = rotation_range
        self.shift_range = shift_range
        self.zoom_range = zoom_range
        self.interpolation_order = ensure_tuple_rep(interpolation_order, len(self.keys))

    def randomize(self, data=None):
        super().randomize(None)
        self.rx = self.R.uniform(-self.rotation_range, self.rotation_range, (1, 1)) * math.pi / 180.0
        self.ry = self.R.uniform(-self.rotation_range, self.rotation_range, (1, 1)) * math.pi / 180.0
        self.rz = self.R.uniform(-self.rotation_range, self.rotation_range, (1, 1)) * math.pi / 180.0

        r3 = [[0., 0., 0., 1.]]
        self.tx = np.dot(np.array(self.R.uniform(-self.shift_range, self.shift_range)).reshape(1, 1), r3)
        self.ty = np.dot(np.array(self.R.uniform(-self.shift_range, self.shift_range)).reshape(1, 1), r3)
        self.tz = np.dot(np.array(self.R.uniform(-self.shift_range, self.shift_range)).reshape(1, 1), r3)

        self.zoom = np.array(self.R.uniform(1.0 - self.zoom_range, 1.0 + self.zoom_range)).reshape(1, 1)

    def __call__(self, data):
        d = dict(data)
        self.randomize()
        if not self._do_transform:
            return d
        affine = AffineTransform(self.rx, self.ry, self.rz, self.tx, self.ty, self.tz, self.zoom)
        for idx, key in enumerate(self.key_iterator(d)):
            d[key] = affine(d[key], self.interpolation_order[idx])
        return d


class RandomContrastd(RandomizableTransform, MapTransform):

    def __init__(self, keys, prob=0.1, factor=0.5, allow_missing_keys=False):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.factor = factor

    def randomize(self, data=None):
        super().randomize(None)
        self.contrast_factor = self.factor + self.R.uniform()

    def _normalize(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def __call__(self, data):
        d = dict(data)
        self.randomize()
        if not self._do_transform:
            return d
        for key in self.key_iterator(d):
            m = d[key]
            if len(m.shape) == 3:
                m = self._normalize(m)
            elif len(m.shape) == 4:
                m = np.concatenate([self._normalize(m[idx])[np.newaxis] for idx in range(m.shape[0])], axis=0)
            d[key] = np.clip(m * self.contrast_factor, 0, 1.0)
        return d


class HistogramClippingd(RandomizableTransform, MapTransform):
    """
        Based on: https://github.com/perone/medicaltorch/blob/master/medicaltorch/transforms.py
    """

    def __init__(self, keys, min_percentile=5.0, max_percentile=95.0, prob=0.5, allow_missing_keys=False):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            m = d[key]
            percentile1 = np.percentile(m, self.min_percentile)
            percentile2 = np.percentile(m, self.max_percentile)
            m[m <= percentile1] = percentile1
            m[m >= percentile2] = percentile2
            d[key] = m
        return d


class NormalizeDistanceMapd(MapTransform):
    """
    Normalizes a given input tensor's data to lie within range 0 and 1

    To get the original segmentation
    threshold to 0.5 with img < 0.5

    """

    def __init__(self, keys, allow_missing_keys=False):
        MapTransform.__init__(self, keys, allow_missing_keys)

    def _normalize(self, data):
        data -= 0.5
        return (data + np.max(abs(data))) / (2 * np.max(abs(data)))

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            m = d[key]
            if len(m.shape) == 3:
                m = self._normalize(m)
            elif len(m.shape) == 4:
                m = np.concatenate([self._normalize(m[idx])[np.newaxis] for idx in range(m.shape[0])], axis=0)
            d[key] = m
        return d