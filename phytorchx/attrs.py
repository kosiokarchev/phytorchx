from __future__ import annotations

from typing import Union

import attr
import torch
from torch.nn import Module


@attr.s(eq=False)
class AttrsModule(Module):
    def __attrs_pre_init__(self):
        super().__init__()


@attr.s(eq=False)
class ParametrizedAttrsModule(AttrsModule):
    device: Union[str, torch.device] = attr.field(default=None, kw_only=True, repr=False)
    dtype: torch.dtype = attr.field(default=None, kw_only=True, repr=False)

    @property
    def factory_kwargs(self):
        return dict(device=self.device, dtype=self.dtype)
