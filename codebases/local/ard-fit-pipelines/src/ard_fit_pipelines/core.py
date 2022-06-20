from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from functools import wraps, partial, update_wrapper
import logging
from typing import Any, Optional, TypeVar
T = TypeVar('T', bound='Transform')

from attrs import define, field, fields_dict, mutable
import pandas as pd

LOG = logging.getLogger(__name__)

# a proper way to do this with metaclass... or __init_subclass__
# DAG of transforms should be lightweight, immutable, unbound.
# Fitting process then binds parameters, concretizes nodes into "fit nodes" with
# bound params and heavyweight state
# It's these fit nodes that can be applied.
# Kind of like dcat's StreamingTableFactory vs StreamingTable
# Writing a new transform should be a breeze, little boilerplate.
# ZScore.fit(...) -> ZScore.FitZScore
# ZScore.FitZScore.apply(df) -> df
# ZScore.FitZscore.refit(...) -> ZScore.FitZScore
#
# Transform has fit() but no apply()
# Subclasser defines _fit(df) -> state and _apply(df, state) -> df
# Transform.__init_subclass__:
#   Makes SubClass an attrs class
#   Creates Subclass.FitSubClass with:
#       mirror of attrs attributes from SubClass (meant to be given bound values)
#       plus a state attribute
#       apply() method that routes to original _apply
#   SubClass.fit(df) -> SubClass.FitSubClass: binds params, gets state from
#       original _fit, constructs new FitSubClass with bound params and state
# StatelessTransform changes behavior to allow apply() directly on transform
# StatelessPipeline 

@define
class hp:
    name: str
    
    @staticmethod
    def resolve(v, bindings):
        if isinstance(v, hp):
            return bindings[v.name]
        return v

@define(slots=False)
class Transform(ABC):
    "The abstract base class of all (unfit) Transforms"

    @abstractmethod
    def _fit(self, X_fit: pd.DataFrame) -> object:
        raise NotImplementedError
        
    @abstractmethod
    def _apply(self, X_apply: pd.DataFrame, state: object=None):
        raise NotImplementedError
        
    def fit(
        self,
        X_fit: pd.DataFrame,
        bindings: Optional[dict[str, object]]=None
    ) -> FitTransform:
        if X_fit is None:
            X_fit = pd.DataFrame()
        LOG.debug('Fitting %s on %d rows: %r', self.__class__.__name__,
            len(X_fit), self)
        fit_class = getattr(self, self._fit_class_name)
        return fit_class(self, X_fit, bindings)

    def __init_subclass__(cls, /, no_magic=False, **kwargs):
        """
        Implements black magic to help with writing Transform subclasses.
        """
        super().__init_subclass__(**kwargs)
        if no_magic:
            return
        class DerivedFitTransform(FitTransform, transform_class=cls):
            pass
            
        fit_class = DerivedFitTransform
        fit_class_name = fit_class.__name__
        fit_class.__qualname__ = '.'.join((cls.__qualname__, fit_class_name))
        cls.fit.__annotations__['return'] = fit_class.__qualname__
        setattr(cls, fit_class_name, fit_class)
        cls._fit_class_name = fit_class_name

class FitTransform(ABC):
    "The result of fitting a {transform_class_name} Transform. Blah blah."
    
    def __init__(self, transform: Transform, X_fit: pd.DataFrame, bindings=None):
        "Docstr for FitTransform.__init__"
        bindings = bindings or {}
        for name in self._field_names:
            unbound_val = getattr(transform, name)
            bound_val = hp.resolve(unbound_val, bindings)
            print("%s: Bound %r -> %r" % (name, unbound_val, bound_val))
            setattr(self, name, bound_val)
        self.__nrows = len(X_fit)
        self.__state = transform._fit.__func__(self, X_fit)

    def __repr__(self):
        fields_str = ', '.join([
            '%s=%r' % (name, getattr(self, name)) 
            for name in self._field_names
        ])
        data_str = f'<{self.__nrows} rows of fitting data>'
        if fields_str:
            return (
                f'{self.__class__.__name__}({", ".join([fields_str, data_str])})'
            )
        else:
            return (
                f'{self.__class__.__name__}({data_str})'
            )
    
    @abstractmethod
    def _apply(self, X_apply, state=None):
        raise NotImplementedError
        
    def apply(self, X_apply: pd.DataFrame) -> pd.DataFrame:
        "Docstr for FitTransform.apply"
        LOG.debug('Applying %s to %d rows: %r', self.__class__.__qualname__,
            len(X_apply), self)
        # TODO: raise exception here if any fields are unbound hyperparameters
        return self._apply(X_apply, state=self.__state)
    
    # TODO: refit()
    
    def state(self):
        # what if there's a field named 'state'?
        return self.__state

    def __init_subclass__(cls, /, transform_class=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if transform_class is None:
            return
        cls._apply = transform_class._apply
        cls.__name__ = f'Fit{transform_class.__name__}'
        cls.__doc__ = FitTransform.__doc__.format(
            transform_class_name=transform_class.__name__)
        cls.__init__.__annotations__['transform'] = transform_class.__name__

        field_names = list(fields_dict(transform_class).keys())
        cls._field_names = field_names

class StatelessTransform(Transform):
    def _fit(self, X_fit: pd.DataFrame):
        return None

    def apply(self, X_apply: pd.DataFrame) -> pd.DataFrame:
        """
        Convenience function to directly apply a StatelessTransform without a
        preceding call to fit, as long as the StatelessTransform has no
        hyperparameters that need to be bound.
        """
        return self.fit(None, bindings=None).apply(X_apply)

        
def not_empty(instance, attribute, value):
    """
    attrs field validator that throws a ValueError if the field value is empty.
    """
    if len(value) < 1:
        raise ValueError(f'{attribute.name} must not be empty but got {value}')

def columns_list_converter(x: str | hp | Iterable[str | hp]) -> list[str | hp]:
    if isinstance(x, str) or isinstance(x, hp):
        return (x,)
    return list(x)

@define
class ColumnsTransform(Transform):
    """
    Abstract base clase of all Transforms that operate on a parameterized list
    of columns.
    """
    cols: list[str | hp] = field(validator=None, # TODO
                                 converter=None)
        



class hp_fmtstr(hp):
    def resolve_value(self, hp_bindings: dict[str, Any]) -> Any:
        return hp_bindings[self.name]
