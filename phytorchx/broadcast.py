from itertools import chain
from typing import Iterable

from more_itertools import split_into, lstrip
from torch import broadcast_shapes, Size, Tensor, cat, gather


__all__ = (
    'broadcast_except', 'broadcast_left', 'broadcast_gather', 'broadcast_cat',
    'insert_dims', 'pad_dims', 'align_dims',
    'aligned_expand', 'fancy_align'
)


def broadcast_except(*tensors: Tensor, dim=-1):
    shape = broadcast_shapes(*(t.select(dim, 0).shape for t in tensors))
    return [t.expand(*shape[:t.ndim + dim], t.shape[dim], *shape[t.ndim + dim:])
            for t in pad_dims(*tensors, ndim=len(shape)+1)]


def broadcast_left(*tensors, ndim):
    shape = broadcast_shapes(*(t.shape[:ndim] for t in tensors))
    return (t.expand(*shape, *t.shape[ndim:]) for t in tensors)


def broadcast_gather(input, dim, index, sparse_grad=False, index_ndim=1):
    """
    input: Size(batch_shape..., N, event_shape...)
    index: Size(batch_shape..., index_shape...)
       ->: Size(batch_shape..., index_shape..., event_shape...)
    """
    index_shape = index.shape[-index_ndim:]
    index = index.flatten(-index_ndim)
    batch_shape = broadcast_shapes(input.shape[:dim], index.shape[:-1])
    input = input.expand(*batch_shape, *input.shape[dim:])
    index = index.expand(*batch_shape, index.shape[-1])
    return gather(input, dim, index.reshape(
        *index.shape, *(input.ndim - index.ndim)*(1,)).expand(
        *index.shape, *input.shape[index.ndim:]
    ) if input.ndim > index.ndim else index, sparse_grad=sparse_grad).reshape(*index.shape[:-1], *index_shape, *input.shape[dim % input.ndim + 1:])


def broadcast_cat(ts: Iterable[Tensor], dim=-1):
    return cat(broadcast_except(*ts, dim=dim), dim)


def insert_dims(t: Tensor, loc: int, shape: Size):
    loc = loc % (t.ndim + 1) if loc < 0 else (loc % t.ndim) + 1
    return t.reshape(t.shape[:loc] + len(shape)*(1,) + t.shape[loc:]).expand(
        t.shape[:loc] + shape + t.shape[loc:]
    )


# TODO: improve so that nbatch=-1 means "auto-derive nbatch from number of
#  matching dimensions on the left"
def pad_dims(*tensors: Tensor, ndim: int = None, nbatch: int = 0) -> list[Tensor]:
    """Pad shapes with ones on the left until at least `ndim` dimensions."""
    if ndim is None:
        ndim = max([t.ndim for t in tensors])
    return [t.reshape(t.shape[:nbatch] + (1,)*(ndim-t.ndim) + t.shape[nbatch:]) for t in tensors]


def align_dims(t: Tensor, ndims: Iterable[int], target_ndims: Iterable[int]):
    assert sum(ndims) == t.ndim
    return t.reshape(*chain.from_iterable(
        (target_ndim - len(s)) * [1] + s for s, target_ndim
        in zip(split_into(t.shape, ndims), target_ndims)
    ))


def aligned_expand(t: Tensor, ndims: Iterable[int], shapes: Iterable[Size]):
    return align_dims(t, ndims, map(len, shapes)).expand(*chain.from_iterable(shapes))


def fancy_align(*tensors: Tensor):
    shapes = dict(enumerate(
        Size(lstrip(t.shape, lambda s: s == 1))
        for t in tensors
    ))

    shape = Size()
    js = {}
    for i, tshape in sorted(shapes.items(), key=lambda arg: len(arg[1])):
        j = 0
        for j in range(len(tshape)+1):
            try:
                shape = broadcast_shapes(shape, tshape[:len(tshape)-j])
                break
            except RuntimeError:
                pass
        js[i] = j
    maxj = max(js.values())
    return tuple(
        t.reshape(t.shape + (maxj - js[i])*(1,))
        for i, t in enumerate(tensors)
    )
