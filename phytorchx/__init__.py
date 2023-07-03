from itertools import starmap
from operator import mul
from typing import Iterable

import torch
from more_itertools import last
from torch import Tensor, Size

from ._version import __version__, version, __version_tuple__, version_tuple
from .broadcast import *


def _mid_one_unnormed(a: Tensor, dim: int) -> Tensor:
    return torch.narrow(a, dim, 0, a.shape[dim]-1) + torch.narrow(a, dim, 1, a.shape[dim]-1)


def mid_one(a: Tensor, dim: int) -> Tensor:
    return _mid_one_unnormed(a, dim) / 2


def mid_many(a: Tensor, axes: Iterable[int]) -> Tensor:
    axes = [ax % a.ndim for ax in axes]
    return last(
        _a for _a in [a] for ax in axes
        for _a in [_mid_one_unnormed(_a, ax)]
    ) / 2**len(axes) if axes else a


def ravel_multi_index(indices: Iterable[Tensor], shape: Size):
    return sum(starmap(mul, zip(indices, [p for p in [1] for s in shape[:0:-1] for p in [s*p]][::-1] + [1])))


def polyval(p: Iterable[Tensor], x: Tensor) -> Tensor:
    result = 0
    for _p in p:
        result = _p + x * result
    return result
