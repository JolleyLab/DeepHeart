import itk
import numpy as np
import math
from scipy.ndimage import affine_transform
from skimage import exposure
from .normalization import MinMaxScaling


class AffineTransform(object):

    def __init__(self, fields, rotation_range, shift_range, zoom_range, execution_probability):
        self.fields = fields
        self.rotation_range = rotation_range
        self.shift_range = shift_range
        self.zoom_range = zoom_range
        self.random_state = np.random.RandomState()
        self.execution_probability = execution_probability

    @staticmethod
    def random_zoom_transform(rs, zoom_range):
        zoom = np.array(rs.uniform(1.0 - zoom_range, 1.0 + zoom_range)).reshape(1,1)

        zx = np.dot(zoom, [[1., 0., 0., 0.]])
        zy = np.dot(zoom, [[0., 1., 0., 0.]])
        zz = np.dot(zoom, [[0., 0., 1., 0.]])

        return np.concatenate([zx, zy, zz, [[0., 0., 0., 1.]]], axis=0)

    @staticmethod
    def random_shift_transform(rs, shift_range):
        raz = np.zeros((1, 4))
        r3 = [[0., 0., 0., 1.]]

        tx = np.dot(np.array(rs.uniform(-shift_range, shift_range)).reshape(1,1), r3)
        ty = np.dot(np.array(rs.uniform(-shift_range, shift_range)).reshape(1,1), r3)
        tz = np.dot(np.array(rs.uniform(-shift_range, shift_range)).reshape(1,1), r3)
        return np.concatenate([tx,ty,tz,raz], axis=0) + np.eye(4)

    @staticmethod
    def random_rotation_matrix(rs, rotation_range):
        o = np.ones((1, 1))
        z = np.zeros((1, 1))

        rx = rs.uniform(-rotation_range, rotation_range, (1, 1)) * math.pi / 180.0
        ry = rs.uniform(-rotation_range, rotation_range, (1, 1)) * math.pi / 180.0
        rz = rs.uniform(-rotation_range, rotation_range, (1, 1)) * math.pi / 180.0

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
    def random_affine_transform(rs, grid_size, rotation_range=0.0, shift_range=0.0, zoom_range=0.0):

        cx, cy, cz = [p / 2.0 for p in grid_size]
        ux, uy, uz = [-p / 2.0 for p in grid_size]

        centering_matrix = np.array([[1, 0, 0, cx], [0, 1, 0, cy], [0, 0, 1, cz], [0, 0, 0, 1]])
        uncentering_matrix = np.array([[1, 0, 0, ux], [0, 1, 0, uy], [0, 0, 1, uz], [0, 0, 0, 1]])

        transform_matrix = centering_matrix

        if rotation_range:
            rotation_matrix = AffineTransform.random_rotation_matrix(rs, rotation_range)
            transform_matrix = np.dot(transform_matrix, rotation_matrix)

        if shift_range:
            shift_matrix = AffineTransform.random_shift_transform(rs, shift_range)
            transform_matrix = np.dot(transform_matrix, shift_matrix)

        if zoom_range:
            zoom_matrix = AffineTransform.random_zoom_transform(rs, zoom_range)
            transform_matrix = np.dot(transform_matrix, zoom_matrix)

        # un-shift from center
        transform_matrix = np.dot(transform_matrix, uncentering_matrix)

        return transform_matrix

    def __call__(self, data):
        if self.random_state.uniform() < self.execution_probability:
            grid_size = data[self.fields[0]].shape
            transform = self.random_affine_transform(self.random_state, grid_size, self.rotation_range,
                                                     self.shift_range, self.zoom_range)

            for field in self.fields:
                order = 0 if any(field == f for f in ["labels", "landmarks", "annuli", "distmaps"]) else 3
                m = data[field]
                if m.ndim == 3:
                    m = m.transpose(2,1,0)
                    transformed = affine_transform(m, transform, order=order, mode='constant')
                    data[field] = transformed.transpose(2,1,0)
                else:
                    raise NotImplementedError
        return data


class RandomContrast(object):
    """
        Adjust the brightness of an image by a random factor.
        Based on: https://github.com/wolny/pytorch-3dunet/blob/master/augment/transforms.py#L119-L149
    """

    def __init__(self, fields, factor=0.5, execution_probability=0.1):
        self.fields = fields
        self.random_state = np.random.RandomState()
        self.factor = factor
        self.execution_probability = execution_probability

    def __call__(self, data):
        if self.random_state.uniform() < self.execution_probability:
            brightness_factor = self.factor + self.random_state.uniform()
            # NB: running min/max normalization to ensure that input data is in range 0 and 1.0
            for field in self.fields:
                data = MinMaxScaling(field)(data)
                d = data[field]
                data[field] = np.clip(d * brightness_factor, 0, 1.0)
        return data


class HistogramClipping(object):
    """
        Based on: https://github.com/perone/medicaltorch/blob/master/medicaltorch/transforms.py
    """

    def __init__(self, fields, min_percentile=5.0, max_percentile=95.0, execution_probability=0.5):
        self.fields = fields
        self.random_state = np.random.RandomState()
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        self.execution_probability = execution_probability

    def __call__(self, data):
        if self.random_state.uniform() < self.execution_probability:
            for field in self.fields:
                d = data[field]
                percentile1 = np.percentile(d, self.min_percentile)
                percentile2 = np.percentile(d, self.max_percentile)
                d[d <= percentile1] = percentile1
                d[d >= percentile2] = percentile2
                data[field] = d
        return data