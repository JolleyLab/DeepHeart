import logging
from monai.transforms import MapTransform
from monai.utils.enums import TransformBackends

import SimpleITK as sitk
import numpy as np
import torch

logger = logging.getLogger(__name__)


def simplex(t, axis: int = 1) -> bool:
  """ source: https://github.com/LIVIAETS/boundary-loss/blob/master/utils.py """
  _sum = t.sum(axis).type(torch.float32)
  _ones = torch.ones_like(_sum, dtype=torch.float32)
  return torch.allclose(_sum, _ones)


def is_one_hot(t, axis=1) -> bool:
  """ source: https://github.com/LIVIAETS/boundary-loss/blob/master/utils.py """
  return simplex(t, axis) and sset(t, [0, 1])


def sset(a, sub) -> bool:
  """ source: https://github.com/LIVIAETS/boundary-loss/blob/master/utils.py """
  return uniq(a).issubset(sub)


def uniq(a) -> set:
  """ source: https://github.com/LIVIAETS/boundary-loss/blob/master/utils.py """
  return set(torch.unique(a.cpu()).numpy())


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



class MergeLabelsd(MapTransform):
    """
    Merge items from data dictionary into single label.
    Expect all the items are numpy array.
    """

    backend = [TransformBackends.NUMPY]

    def __init__(self, keys, name, allow_missing_keys=False):
        """
        Args:
            keys: keys of the corresponding items to be concatenated together.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            name: the name corresponding to the key to store the reduced data.
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.name = name

    def __call__(self, data):
        d = dict(data)
        output = []
        data_type = None
        for key in self.key_iterator(d):
            if data_type is None:
                data_type = type(d[key])
            elif not isinstance(d[key], data_type):
                raise TypeError("All items in data must have the same type.")
            output.append(d[key])
        import functools
        out = functools.reduce(lambda a, b: np.logical_or(a,b), output).astype(int)
        # print(f"shape: {out.shape}, min: {out.min()}, max: {out.max()}, unique: {np.unique(out)}")
        d[self.name] = out
        return d