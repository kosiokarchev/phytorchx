from __future__ import annotations

from typing import Union, get_type_hints, get_origin, Annotated, ClassVar, get_args

import attr
import torch
from torch.nn import Module, Parameter


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

    def __attrs_post_init__(self):
        for key, hint in get_type_hints(self, include_extras=True).items():
            if get_origin(hint) is ClassVar:
                hint = get_args(hint)[0]
            if get_origin(hint) is Annotated:
                func, shape_spec = hint.__metadata__
                self.register_parameter(key, Parameter(func(*(
                    getattr(self, k) if isinstance(k, str) else k
                    for k in shape_spec
                ), **self.factory_kwargs)))
