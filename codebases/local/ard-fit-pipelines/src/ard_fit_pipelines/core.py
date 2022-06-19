from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
import logging
from typing import Any, Optional, TypeVar
T = TypeVar('T', bound='Transform')

from attrs import define, field, fields_dict, mutable
import pandas as pd

LOG = logging.getLogger(__name__)

@define(slots=False)
class Transform(ABC):
    """
    Two operations: fit and apply. Fit-time and apply-time.

    Params and hyperparams.
    """
    bound_params = None

    def fit(
        self: T,
        X_fit: pd.DataFrame,
        hp_bindings: Optional[dict[str, Any]]=None
    ) -> T:
        LOG.debug('Fitting Transform on %d rows of data: %r', len(X_fit), self)
        self.bind_params(hp_bindings)
        return self

    def bind_params(self, hp_bindings: dict[str, Any]):
        param_names = fields_dict(self.__class__).keys()
        unbound_params = dict(zip(
            param_names, map(self.__getattribute__, param_names)
        ))
        self.bound_params = {
            name: (
                unbound_val.resolve_value(hp_bindings) 
                    if isinstance(unbound_val, hp) else unbound_val
            ) for name, unbound_val in unbound_params.items()
        }

    @abstractmethod
    def apply(self, X_apply: pd.DataFrame) -> pd.DataFrame:
        LOG.debug('Applying Transform to %d rows of data: %r', len(X_apply),
            self)
        return X_apply

@define
class hp:
    """
    Hyperparameterization of a Transform argument.
    """
    name: str

    def resolve_value(self, hp_bindings: dict[str, Any]) -> Any:
        # by default name is treated as the "name" of the hp, i.e., the
        # key of the desired value in hp_bindings
        return hp_bindings[self.name]

class hp_fmtstr(hp):
    def resolve_value(self, hp_bindings: dict[str, Any]) -> Any:
        return hp_bindings[self.name]

def not_empty(instance, attribute, value):
    """
    attrs field validator that throws a ValueError if the field value is empty.
    """
    if len(value) < 1:
        raise ValueError(f'{attribute.name} must not be empty but got {value}')

def columns_tuple_converter(x: str | hp | Iterable[str | hp]) -> tuple[str | hp]:
    if isinstance(x, str) or isinstance(x, hp):
        return (x,)
    return tuple(x)

@define
class ColumnsTransform(Transform):
    """
    Abstract base clase of all Transforms that operate on a parameterized list
    of columns.
    """
    cols: tuple[str | hp] = field(validator=not_empty,
                                  converter=columns_tuple_converter)
