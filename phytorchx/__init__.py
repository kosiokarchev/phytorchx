from functools import wraps
from itertools import starmap
from operator import mul
from typing import Iterable

import torch
from more_itertools import last
from optree import tree_map_
from torch import Tensor, Size, tensor

from ._version import __version__, version, __version_tuple__, version_tuple
from .broadcast import *


def get_default_device():
    """Try really hard to return the "default" |PyTorch| device.

    This is defined as the device on which a tensor would be allocated by default
    **at the point this function is called**.
    """
    return torch._C._get_default_device()


@wraps(torch.load, ('__annotations__',), ())
def load(*args, **kwargs):
    """`torch.load` onto the default device and with the default `~torch.dtype`.

    Parameters
    ----------
    map_location
        set by default to :py:func:`get_default_device` but can still be
        overridden.
    """
    return tree_map_(
        lambda arg: arg.to(dtype=torch.get_default_dtype()) if torch.is_tensor(arg) and torch.is_floating_point(arg) else arg,
        torch.load(*args, **{'map_location': get_default_device(), **kwargs})
    )


def _mid_one_unnormed(a: Tensor, dim: int) -> Tensor:
    return torch.narrow(a, dim, 0, a.shape[dim]-1) + torch.narrow(a, dim, 1, a.shape[dim]-1)


def mid_one(a: Tensor, axis: int) -> Tensor:
    r"""Create a new tensor which, along the :arg:`dim`-th dimension, contains the
    average of adjacent elements of the input:

    .. math::
       \mathrm{out}[..., i, ...] = (\mathrm{a}[..., i, ...] + \mathrm{a}[..., i+1, ...]) / 2.

    Parameters
    ----------
    a
        input tensor
    axis
        axis along which to average

    Returns
    -------
    :
        ``(*a.shape[:axis], a.shape[axis]-1, *a.shape[axis+1:])``
    """
    return _mid_one_unnormed(a, axis) / 2


def mid_many(a: Tensor, axes: Iterable[int]) -> Tensor:
    r"""Perform `mid_one` along  many (not necessarily dajacent) :arg:`axes` at the same time."""
    axes = [ax % a.ndim for ax in axes]
    return last(
        _a for _a in [a] for ax in axes
        for _a in [_mid_one_unnormed(_a, ax)]
    ) / 2**len(axes) if axes else a


def ravel_multi_index(indices: Iterable[Tensor], shape: Size) -> Tensor:
    """Transform a multi-dimensional index into a one-dimensional index into the ravelled tensor.

    Parameters
    ----------
    indices
        iterable of indices, which will be broadcast against each other.
    shape
        of the tensor, giving sizes along the respective dimensions that :arg:`indices` index.
        Must be at least as long (from the right) as :arg:`indices`.

    Examples
    --------
    >>> ravel_multi_index(tensor([3, 2, 1]), Size((4, 5, 6)))
    tensor(103)  # = 3 * (5*6) + 2 * (6) + 1

    (Note that in this example ``indices <- tensor(3), tensor(2), tensor(1)``.)
    """
    return sum(starmap(mul, zip(indices, [p for p in [1] for s in shape[:0:-1] for p in [s*p]][::-1] + [1])))


def polyval(p: Iterable[Tensor], x: Tensor) -> Tensor:
    r"""Evaluate a polynomial using Horner's scheme.

    Parameters
    ----------
    p
        iterable of the polynomial coefficients, **starting from the highest power**
    x
        independent variable

    Returns
    -------
    :
        .. math::
            \sum_{i=0}^{\mathtt{len(p)}-1} \mathtt{p[i]} \times \mathtt{x}^{\mathtt{len(p)-1-i}}
    """
    result = 0
    for _p in p:
        result = _p + x * result
    return result
