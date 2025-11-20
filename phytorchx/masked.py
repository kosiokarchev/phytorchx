from functools import partial
from typing import Iterable
from warnings import warn

from torch import BoolTensor, Tensor
from torch.utils.data._utils.collate import default_collate_fn_map, default_collate


class MaskedTensor(Tensor):
    _mask: BoolTensor

    def mask(self, mask: BoolTensor):
        self._mask = mask
        return self

    def valid(self):
        return self.as_subclass(Tensor)[self._mask]

    def __getitem__(self, item):
        return self.as_subclass(Tensor)[item].as_subclass(type(self)).mask(self._mask[item])

    def unbind(self, dim=0):
        return tuple(d.as_subclass(type(self)).mask(m) for d, m in zip(
            self.as_subclass(Tensor).unbind(dim), self._mask.unbind(dim)))

    def unsqueeze(self, dim):
        return self.as_subclass(Tensor).unsqueeze(dim).as_subclass(type(self)).mask(self._mask.unsqueeze(dim))

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        res = super().__torch_function__(func, types, args, kwargs)

        if isinstance(res, MaskedTensor) and isinstance(args[0], MaskedTensor) and res.shape == args[0].shape:
            res._mask = args[0]._mask
        else:
            warn(f'Shady operation on a {cls.__name__}: {func}', UserWarning)
        return res



@partial(default_collate_fn_map.__setitem__, MaskedTensor)
def collate_maskedtensor(batch: Iterable[MaskedTensor], *args, **kwargs):
    data, mask = map(default_collate, zip(*((b.as_subclass(Tensor), b.mask) for b in batch)))
    return data.as_subclass(MaskedTensor).mask(mask)
