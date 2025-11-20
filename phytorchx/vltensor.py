from functools import partial
from typing import Iterable

from torch import Tensor
from torch.utils.data._utils.collate import default_collate_fn_map


class VLTensor(Tensor):
    pass


class FancyList(list):
    def __getitem__(self, item):
        if isinstance(item, Iterable):
            return type(self)(super(type(self), self).__getitem__(i) for i in item)
        return super().__getitem__(item)


@partial(default_collate_fn_map.__setitem__, VLTensor)
def collate_vltensor(batch: Iterable[VLTensor], *args, **kwargs):
    return FancyList(el.as_subclass(Tensor) for el in batch)
