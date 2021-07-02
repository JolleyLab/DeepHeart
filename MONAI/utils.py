import os
from pathlib import Path


def generate_directory_name() -> str:
    import random
    import string
    import datetime
    start_date = datetime.datetime.now().strftime('%m%d%Y')
    start_time = datetime.datetime.now().strftime('%H%M%S')
    random_hash = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
    return f"{start_date}/{start_time}_{random_hash}"


def get_list_of_file_names(directory: Path, ext: str = 'nii.gz', absolute_path: bool = False):
    if absolute_path:
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(ext)]
    else:
        return [f for f in os.listdir(directory) if f.endswith(ext)]


def is_one_hot(t, axis=1) -> bool:
    """ source: https://github.com/LIVIAETS/boundary-loss/blob/master/utils.py """
    return simplex(t, axis) and sset(t, [0, 1])


def sset(a, sub) -> bool:
    """ source: https://github.com/LIVIAETS/boundary-loss/blob/master/utils.py """
    return uniq(a).issubset(sub)


def uniq(a) -> set:
    """ source: https://github.com/LIVIAETS/boundary-loss/blob/master/utils.py """
    import torch
    return set(torch.unique(a.cpu()).numpy())


def simplex(t, axis: int = 1) -> bool:
    """ source: https://github.com/LIVIAETS/boundary-loss/blob/master/utils.py """
    import torch
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)