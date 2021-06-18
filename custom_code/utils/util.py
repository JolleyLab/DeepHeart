import os
from importlib import import_module
import nibabel as nib
import numpy as np
import torch
import SimpleITK as sitk


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def get_instances(module, name, config, *args):
    return [getattr(module, entry['type'])(*args, **entry['args']) for entry in config[name]]


def get_class(module_name, class_name):
    module = import_module(module_name)
    return getattr(module, class_name)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_transform(config, transform_type):
    for transform in config["data_loader"]["args"]["input_transforms"]:
        if transform["type"] == transform_type:
            return transform
    return None


def to_one_hot(x, n_classes):
    """
    Converts NxDxHxW label image to NxCxDxHxW, where each label gets converted to its corresponding one-hot vector
    :param x: 5D input image (NxCxDxHxW)
    :param n_classes: number of channels/labels
    :return: 5D output image (NxCxDxHxW)
    """

    shape = list(x.size())
    shape[1] = n_classes
    return torch.zeros(tuple(shape)).to(x.device).scatter_(1, x.long(), 1)


def np_to_one_hot(input_array):
    n_classes = (len(np.unique(input_array)))

    if len(input_array.shape) == 4:
        assert input_array.shape[0] == 1
        input_array = input_array[0]

    w, h, d = input_array.shape

    res = np.stack([input_array == c for c in range(n_classes)], axis=0).astype(np.int32)
    assert res.shape == (n_classes, w, h, d)
    assert np.all(res.sum(axis=0) == 1)
    return res


def get_distance_map_sitk(np_input):
    image = sitk.GetImageFromArray(np_input.astype(np.int).transpose(2,1,0))
    distanceMapFilter = sitk.SignedMaurerDistanceMapImageFilter()
    distanceMapFilter.SetUseImageSpacing(True)
    distanceMapFilter.SetSquaredDistance(False)
    out = distanceMapFilter.Execute(image)
    return sitk.GetArrayFromImage(out).transpose(2,1,0)


def one_hot_to_dist(input_array):
    assert one_hot(torch.Tensor(input_array), axis=0)
    out = np.stack([get_distance_map_sitk(input_array[ch].astype(np.float32)) for ch in range(input_array.shape[0])], axis=0)
    return out


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)

       source: https://raw.githubusercontent.com/wolny/pytorch-3dunet/master/unet3d/losses.py
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.view(C, -1)


# Assert utils
def uniq(a):
    """ source: https://github.com/LIVIAETS/boundary-loss/blob/749602d822fcb097c79a7fb708805b8982c37030/utils.py """
    return set(torch.unique(a.cpu()).numpy())


def sset(a, sub):
    """ source: https://github.com/LIVIAETS/boundary-loss/blob/749602d822fcb097c79a7fb708805b8982c37030/utils.py """
    return uniq(a).issubset(sub)


def simplex(t, axis=1):
    """ source: https://github.com/LIVIAETS/boundary-loss/blob/749602d822fcb097c79a7fb708805b8982c37030/utils.py """
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t, axis=1):
    """ source: https://github.com/LIVIAETS/boundary-loss/blob/749602d822fcb097c79a7fb708805b8982c37030/utils.py """
    return simplex(t, axis) and sset(t, [0, 1])


def save_img(data, path, affine=np.eye(4), ch=0):
    img = nib.Nifti1Image(data[ch].numpy() if not type(data) is np.ndarray else data[ch], affine)
    nib.save(img, path)


def load_images(data, field):
    from multiprocessing.pool import ThreadPool
    from functools import partial

    image_list = list()

    def load_img(idx):
        image_list.append(nib.load(data[idx][field]).get_data().astype(np.float32))

    load_img = partial(load_img)

    with ThreadPool(32) as p:
        p.map(load_img, range(len(data)))

    return image_list


def mean_img(data, field):
    image_list = load_images(data, field)
    return np.mean(np.array(image_list), axis=0)


def std_img(data, field):
    image_list = load_images(data, field)
    return np.std(np.array(image_list), axis=0)
