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

# DAG of transforms should be lightweight, immutable, unbound.
# Fitting process then binds parameters, concretizes nodes into "fit nodes" with
# bound params and heavyweight state
# It's these fit nodes that can be applied.
# Kind of like dcat's StreamingTableFactory vs StreamingTable
# Writing a new transform should be a breeze, no boilerplate.

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

@define(slots=False)
class Transform(ABC):
    """
    The abstract base class of all (unfit) Transforms. Subclasses must implement
    the `_fit()` and `_apply()` methods (but see StatelessTransform, which removes
    the requirement to implement `_fit()`).

    Subclasses should use `attrs` field variables to hold parameters (but not
    fit state) of the transformation being implemented, with the expectation
    that these parameters will be provided by the user of the subclass as
    constructor arguments. Thanks to `attrs`, in most cases no constructor needs
    to be written explicitly by the subclass author, and in any case only field
    variables will be treated as potential hyperparameters at fit-time (i.e., to
    potentially get their values from the `bindings=` kwarg to `fit()`).

    The implementations of `_fit()` and `_apply()` may refer freely to any
    `attrs` fields (generally understood as parameters of the transformation) as
    instance variables on `self`. If any fields were given as hyperparameters at
    construction time, they are resolved to concrete bindings before `_fit()`
    and `_apply()` are invoked.

    _fit() should accept some training data and return an arbitrary object
    representing fit state, and which will be passed to `_apply()` at
    apply-time. Generally speaking, `_fit()` should *not* mutate anything about
    `self`.

    `_apply()` should then accept a state object as returned by `_fit()` and
    return the result of applying the transformation to some given apply-time
    data.

    Once implemented, the subclass is used like any Transform, which is to say
    by constructing an instance with some parameters (which may be
    hypeparameters), and then calling its `fit()` and `apply()` methods (note no
    leading underscores).

    A subclass `C` will automatically find itself in possession of an inner
    class `FitC`, which derives from `FitTransform`.  `C.fit()` will then return
    `C.FitC` instances (encapsulating the state returned by the subclasser's
    `_fit()` implementation), whose `apply()` methods (i.e., `C.FitC.apply()`)
    employ the subclasser's `_apply()` implementation.

    Subclasses must not keep parameters in fields named `fit`, `apply`, `state`,
    or `params`, as these would break functionality by overriding expected
    method names.

    Examples of writing Transforms:

    ```
    # A simple stateful transform from scratch, subclassing Transform directly.
    @define
    class DeMean(Transform):
        "De-mean some columns."

        cols: list[str] # Or, get this for free by subclassing ColumnsTransform
        
        def _fit(self, X_fit: pd.DataFrame) -> object:
            return X_fit[self.cols].mean()
        
        def _apply(self, X_apply: pd.DataFrame, state: object):
            means = state
            return X_apply.assign(**{
                c: X_apply[c] - means[c]
                for c in self.cols
            })

    # A stateless transform whose only parameter is a list of columns; the
    # implementation is simplified by subclassing two "convenience base
    # classes": StatelessTransform for the common case of a transform with no
    # state to fit, and ColumnsTransform, for the common case of operating on a
    # parameterized list of columns, which is made available as an attrs-managed
    # field `self.cols`.
    # (@define is not necessary because we are not introducing any fields in our
    # sublcass.)
    class KeepColumns(StatelessTransform, ColumnsTransform):
        def _apply(self, X_apply: pd.DataFrame, state: object=None) -> pd.DataFrame:
            return X_apply[self.cols]
    ```

    """

    @abstractmethod
    def _fit(self, X_fit: pd.DataFrame) -> object:
        raise NotImplementedError
        
    @abstractmethod
    def _apply(self, X_apply: pd.DataFrame, state: object=None) -> pd.DataFrame:
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

    def params(self) -> list[str]:
        field_names = list(fields_dict(self.__class__).keys())
        return field_names

    def __init_subclass__(cls, /, no_magic=False, **kwargs):
        """
        Implements black magic to help with writing Transform subclasses.
        """
        super().__init_subclass__(**kwargs)
        if no_magic:
            return
        class DerivedFitTransform(FitTransform, transform_class=cls):
            pass

        # we should freak out if the subclass has any attribute named
        # 'state', because that will collide at apply-time with
        # fit_class.state().
        if hasattr(cls, 'state'):
            raise AttributeError(
                'Subclasses of Transform are not allowed to have an attribute '
                'named "state". Deal with it.')
            
        fit_class = DerivedFitTransform
        fit_class_name = fit_class.__name__
        fit_class.__qualname__ = '.'.join((cls.__qualname__, fit_class_name))
        cls.fit.__annotations__['return'] = fit_class.__qualname__
        setattr(cls, fit_class_name, fit_class)
        cls._fit_class_name = fit_class_name

class FitTransform(ABC):
    """
    The result of fitting a {transform_class_name} Transform. Call this
    object's `apply()` method on some data to get the result of applying the
    now-fit transformation.

    All parameters of the fit {transform_class_name} are available as instance
    variables, with any hyperparameters fully resolved against whatever bindings
    were provided at fit-time.

    The fit state of the transformation, as returned by {transform_class_name}'s
    `_fit()` method at fit-time, is available from `state()`, and this is the
    state that will be used at apply-time.
    """
    
    def __init__(self, transform: Transform, X_fit: pd.DataFrame, bindings=None):
        "Docstr for FitTransform.__init__"
        bindings = bindings or {}
        for name in self._field_names:
            unbound_val = getattr(transform, name)
            bound_val = hp.resolve_maybe(unbound_val, bindings)
            print("%s: Bound %r -> %r" % (name, unbound_val, bound_val))
            setattr(self, name, bound_val)
        # TODO: freak out if any hyperparameters failed to bind
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
        """
        Return the result of applying this fit Transform to the given DataFrame.
        """
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

@define
class hp:
    name: str

    def resolve(self, bindings):
        # default: treat hp name as key into bindings
        return bindings.get(self.name, self)
    
    @staticmethod
    def resolve_maybe(v, bindings):
        if isinstance(v, hp):
            return v.resolve(bindings)
        return v

class hp_fmtstr(hp):
    def resolve(self, bindings):
        # treate name as format string to be formatted against bindings
        return self.name.format(**bindings)

class StatelessTransform(Transform):
    def _fit(self, X_fit: pd.DataFrame):
        return None

    def apply(self, X_apply: pd.DataFrame) -> pd.DataFrame:
        """
        Convenience function allowing one to apply a StatelessTransform without
        a preceding call to fit, as long as the StatelessTransform has no
        hyperparameters that need to be bound.
        """
        return self.fit(None, bindings=None).apply(X_apply)

        
# Valid column list specs (routed by field converter):
# hp('which_cols') -> plain old hp
# ['x', 'y', 'z'] -> hp_cols (plain old list?)
# ['x', hp('some_col'), 'z'] -> hp_cols
# ['x', '{som_col}', 'z'] -> hp_cols
# Scalars rewritten to lists of one:
#   'z' -> ['z'] -> hp_cols
#   '{some_col}' - ['{som_col}'] -> hp_cols

@define
class hp_cols(hp):
    cols: list[str|hp]
    name: str = None

    @classmethod
    def maybe_from_value(cls, x: str | hp | Iterable[str | hp]):
        if isinstance(x, hp):
            return x
        if isinstance(x, str):
            return cls([x])
        return cls(list(x))

    def resolve(self, bindings):
        return [
            c.resolve(bindings) if isinstance(c, hp) else
            c.format(**bindings) if isinstance(c, str) else
            c
            for c in self.cols
        ]

    def __repr__(self):
        return repr(self.cols)

    def __len__(self):
        return len(self.cols)

    def __iter__(self):
        return iter(self.cols)

def not_empty(instance, attribute, value):
    """
    attrs field validator that throws a ValueError if the field value is empty.
    """
    if hasattr(value, '__len__'):
        if len(value) < 1:
            raise ValueError(f'{attribute.name} must not be empty but got {value}')
    elif isinstance(value, hp):
        return
    else:
        raise TypeError(f'{attribute.name} value has no length: {value}')


@define
class ColumnsTransform(Transform):
    """
    Abstract base clase of all Transforms that operate on a parameterized list
    of columns. Subclasses acquire a mandatory `cols` argument to their
    constructors, which ...
    """
    cols: list[str | hp] = field(validator=not_empty,
                                 converter=hp_cols.maybe_from_value)
        

@define
class WeightedTransform(Transform):
    w_col: str = None