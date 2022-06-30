from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
import logging
from typing import Callable, Optional, Union

from attrs import define, field, fields_dict
import pandas as pd

_LOG = logging.getLogger(__name__)

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


class Dataset(ABC):
    @abstractmethod
    def to_dataframe(self):
        raise NotImplementedError

    @staticmethod
    def from_pandas(df: pd.DataFrame) -> Dataset:
        return PandasDataset(df)


@define
class PandasDataset(Dataset):
    df: pd.DataFrame

    def to_dataframe(self):
        return self.df


class UnknownDatasetError(KeyError):
    """
    Thrown by DatasetCollection.get_dataset() if the requested dataset name is not
    found.
    """


def _dataset_collection_converter(map: dict[str, pd.DataFrame | Dataset]):
    map = {
        name: PandasDataset(value) if isinstance(value, pd.DataFrame) else value
        for name, value in map.items()
    }
    # let's say that every dsc implicitly has a __pass__ dataset, which is an empty
    # dataframe if no dataset was provided with that name
    if "__pass__" not in map:
        map["__pass__"] = PandasDataset(pd.DataFrame())
    return map


@define
class DatasetCollection:

    map: dict[str, Dataset] = field(converter=_dataset_collection_converter)

    def get_dataset(self, name):
        if name not in self.map:
            if name == "__pass__":
                return PandasDataset(pd.DataFrame())
            raise UnknownDatasetError(
                f"Asked for a Dataset named {name!r} but this DatasetCollection only "
                f"has: {list(self.map.keys())}"
            )
        return self.map[name]

    def to_dataframe(self, name="__pass__"):
        return self.get_dataset(name).to_dataframe()

    @classmethod
    def from_data(cls, data: Optional[Data]):
        if data is None:
            dsc = cls({"__pass__": PandasDataset(pd.DataFrame())})
        elif isinstance(data, pd.DataFrame):
            dsc = cls({"__pass__": PandasDataset(data)})
        elif isinstance(data, Dataset):
            dsc = cls({"__pass__": data})
        elif isinstance(data, cls):
            dsc = data
        else:
            raise TypeError(
                f"Expected {Data} but got {data} which has type {type(data)}"
            )
        return dsc

    def __getitem__(self, dataset_name):
        return self.get_dataset(dataset_name)

    def __or__(self, other):
        if isinstance(other, dict):
            return DatasetCollection(self.map | other)
        if isinstance(other, DatasetCollection):
            return DatasetCollection(self.map | other.map)
        raise TypeError(
            f"Don't know how to union a DatasetCollection with a {type(other)}"
        )


# Type alias for the primary argument to Transform.fit() and .apply()
Data = Union[pd.DataFrame, Dataset, DatasetCollection]


# TODO: remove need for Transform subclasses to write @define
@define(slots=False)
class Transform(ABC):
    """
    The abstract base class of all (unfit) Transforms. Subclasses must implement the
    `_fit()` and `_apply()` methods (but see StatelessTransform, which removes the
    requirement to implement `_fit()`).

    Subclasses should use `attrs` field variables to hold parameters (but not fit state)
    of the transformation being implemented, with the expectation that these parameters
    will be provided by the user of the subclass as constructor arguments. Thanks to
    `attrs`, in most cases no constructor needs to be written explicitly by the subclass
    author, and in any case only field variables will be treated as potential
    hyperparameters at fit-time (i.e., to potentially get their values from the
    `bindings=` kwarg to `fit()`).

    The implementations of `_fit()` and `_apply()` may refer freely to any `attrs`
    fields (generally understood as parameters of the transformation) as instance
    variables on `self`. If any fields were given as hyperparameters at construction
    time, they are resolved to concrete bindings before `_fit()` and `_apply()` are
    invoked.

    _fit() should accept some training data and return an arbitrary object representing
    fit state, and which will be passed to `_apply()` at apply-time. Generally speaking,
    `_fit()` should *not* mutate anything about `self`.

    `_apply()` should then accept a state object as returned by `_fit()` and return the
    result of applying the transformation to some given apply-time data.

    Once implemented, the subclass is used like any Transform, which is to say by
    constructing an instance with some parameters (which may be hypeparameters), and
    then calling its `fit()` and `apply()` methods (note no leading underscores).

    A subclass `C` will automatically find itself in possession of an inner class
    `FitC`, which derives from `FitTransform`.  `C.fit()` will then return `C.FitC`
    instances (encapsulating the state returned by the subclasser's `_fit()`
    implementation), whose `apply()` methods (i.e., `C.FitC.apply()`) employ the
    subclasser's `_apply()` implementation.

    Subclasses must not keep parameters in fields named `fit`, `apply`, `state`, or
    `params`, as these would break functionality by overriding expected method names.

    Examples of writing Transforms:

    ```
    # A simple stateful transform from scratch, subclassing Transform directly.
    @define
    class DeMean(Transform):
        "De-mean some columns."

        cols: list[str] # Or, get this for free by subclassing ColumnsTransform

        def _fit(self, df_fit: pd.DataFrame) -> object:
            return df_fit[self.cols].mean()

        def _apply(self, df_apply: pd.DataFrame, state: object):
            means = state
            return df_apply.assign(**{
                c: df_apply[c] - means[c]
                for c in self.cols
            })

    # A stateless transform whose only parameter is a list of columns; the
    # implementation is simplified by subclassing two "convenience base classes":
    # StatelessTransform for the common case of a transform with no state to fit, and
    # ColumnsTransform, for the common case of operating on a parameterized list of
    # columns, which is made available as an attrs-managed field `self.cols`.  (@define
    # is not necessary because we are not introducing any fields in our sublcass.)
    class KeepColumns(StatelessTransform, ColumnsTransform):
        def _apply(self, df_apply: pd.DataFrame, state: object=None) -> pd.DataFrame:
            return df_apply[self.cols]
    ```
    """

    # Note the following are regular attributes, NOT managed by attrs

    # Special (and default) value "__pass__" means give me give me the output of the
    # preceding Transform in a Pipeline, or the user's unnamed DataFrame/Dataset arg to
    # fit()/apply()
    dataset_name = "__pass__"  # TODO: docs

    @abstractmethod
    def _fit(self, df_fit: pd.DataFrame) -> object:
        raise NotImplementedError

    @abstractmethod
    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        raise NotImplementedError

    def fit(
        self, data_fit: Data, bindings: Optional[dict[str, object]] = None
    ) -> FitTransform:
        dsc = DatasetCollection.from_data(data_fit)
        fit_class: FitTransform = getattr(self, self._fit_class_name)
        return fit_class(self, dsc, bindings)

    def params(self) -> list[str]:
        field_names = list(fields_dict(self.__class__).keys())
        return field_names

    def hyperparams(self) -> dict[str, HP]:
        return {
            name: hp_obj
            for name in self.params()
            if isinstance(hp_obj := getattr(self, name), HP)
        }

    def __init_subclass__(cls, /, no_magic=False, **kwargs):
        """
        Implements black magic to help with writing Transform subclasses.
        """
        super().__init_subclass__(**kwargs)
        if no_magic:
            return

        class DerivedFitTransform(FitTransform, transform_class=cls):
            pass

        # we should freak out if the subclass has any attribute named 'state' or
        # 'bindings', because those will collide at apply-time with fit_class method
        # names.
        if hasattr(cls, "state"):
            raise AttributeError(
                "Subclasses of Transform are not allowed to have an attribute "
                'named "state". Deal with it.'
            )
        if hasattr(cls, "bindings"):
            raise AttributeError(
                "Subclasses of Transform are not allowed to have an attribute "
                'named "bindings". Deal with it.'
            )

        fit_class = DerivedFitTransform
        fit_class_name = fit_class.__name__
        fit_class.__qualname__ = ".".join((cls.__qualname__, fit_class_name))
        cls.fit.__annotations__["return"] = fit_class.__qualname__
        setattr(cls, fit_class_name, fit_class)
        cls._fit_class_name = fit_class_name


class UnresolvedHyperparameterError(NameError):
    """Exception thrown when a Transform is not able to resolve all of its
    hyperparameters at fit-time."""


class FitTransform(ABC):
    """
    The result of fitting a {transform_class_name} Transform. Call this object's
    `apply()` method on some data to get the result of applying the now-fit
    transformation.

    All parameters of the fit {transform_class_name} are available as instance
    variables, with any hyperparameters fully resolved against whatever bindings were
    provided at fit-time.

    The fit state of the transformation, as returned by {transform_class_name}'s
    `_fit()` method at fit-time, is available from `state()`, and this is the state that
    will be used at apply-time.
    """

    dataset_collection: DatasetCollection = None

    def __init__(self, transform: Transform, dsc_fit: DatasetCollection, bindings=None):
        "Docstr for FitTransform.__init__"
        bindings = bindings or {}
        self._field_names = transform.params()
        for name in self._field_names:
            unbound_val = getattr(transform, name)
            bound_val = HP.resolve_maybe(unbound_val, bindings)
            # print("%s: Bound %r -> %r" % (name, unbound_val, bound_val))
            setattr(self, name, bound_val)
        self.__bindings = bindings
        self.dataset_name: str = transform.dataset_name
        # freak out if any hyperparameters failed to bind
        self._check_hyperparams()

        # materialize data for user _fit function.
        df_fit = dsc_fit.to_dataframe(self.dataset_name)
        self.__nrows = len(df_fit)
        # but also keep the original collection around (temporarily) in case the user
        # _fit function wants it
        self.dataset_collection = dsc_fit
        _LOG.debug(
            "Fitting %s on %d rows: %r", self.__class__.__name__, len(df_fit), self
        )
        # run user _fit function
        self.__state = transform._fit.__func__(self, df_fit)
        # now that fitting is done, we don't want to carry around a reference to all the
        # data
        self.dataset_collection = None

    def _check_hyperparams(self):
        unresolved = []
        for name in self._field_names:
            val = getattr(self, name)
            if isinstance(val, HP):
                unresolved.append(val)
        if unresolved:
            raise UnresolvedHyperparameterError(
                f"One or more hyperparameters of {self.__class__.__qualname__} were "
                f"not resolved at fit-time: {unresolved}. Bindings were: "
                f"{self.__bindings}"
            )

    def __repr__(self):
        fields_str = ", ".join(
            ["%s=%r" % (name, getattr(self, name)) for name in self._field_names]
        )
        data_str = f"<{self.__nrows} rows of fitting data>"
        if fields_str:
            return f'{self.__class__.__name__}({", ".join([fields_str, data_str])})'
        return f"{self.__class__.__name__}({data_str})"

    @abstractmethod
    def _apply(self, df_apply: pd.DataFrame, state=None) -> pd.DataFrame:
        raise NotImplementedError

    def apply(self, data_apply: Data) -> pd.DataFrame:
        """
        Return the result of applying this fit Transform to the given DataFrame.
        """
        # materialize data for user _apply function.
        dsc_apply = DatasetCollection.from_data(data_apply)
        df_apply = dsc_apply.to_dataframe(self.dataset_name)
        # but also keep the original collection around (temporarily) in case the user
        # _apply function wants it
        self.dataset_collection = dsc_apply
        _LOG.debug(
            "Applying %s to %d rows: %r",
            self.__class__.__qualname__,
            len(df_apply),
            self,
        )
        result = self._apply(df_apply, state=self.__state)
        # now that application is done, we don't want to carry around a reference to all
        # the data
        self.dataset_collection = None
        return result

    # TODO: refit()

    def bindings(self) -> dict[str, object]:
        """
        Return the bindings dict according to which the transformation's hyperparameters
        were resolved.
        """
        return self.__bindings

    def state(self) -> object:
        """
        Return the fit state of the transformation, which is an arbitrary object
        determined by the implementation of {transform_class_name}._fit().
        """
        return self.__state

    def __init_subclass__(cls, /, transform_class: type = None, **kwargs):
        # TODO: futz with base classes so that super() works like normal in the user's
        # _fit() and _apply() methods when subclassing another Transform.
        super().__init_subclass__(**kwargs)
        if transform_class is None:
            return
        cls._apply = transform_class._apply
        cls.__name__ = f"Fit{transform_class.__name__}"
        cls.__doc__ = FitTransform.__doc__.format(
            transform_class_name=transform_class.__name__
        )
        cls.state.__doc__ = FitTransform.state.__doc__.format(
            transform_class_name=transform_class.__name__
        )
        cls.__init__.__annotations__["transform"] = transform_class.__name__

        field_names = list(fields_dict(transform_class).keys())
        cls._field_names = field_names


class StatelessTransform(Transform):
    def _fit(self, df_fit: pd.DataFrame):
        return None

    def apply(
        self, data_apply: Data, bindings: dict[str, object] = None
    ) -> pd.DataFrame:
        """
        Convenience function allowing one to apply a StatelessTransform without an
        explicit preceding call to fit. Implemented by calling fit() on no data (but
        with optional hyperparameter bindings as provided) and then returning the result
        of applying the resulting FitTransform to the given DataFrame.
        """
        return self.fit(None, bindings=bindings).apply(data_apply)


@define
class HP:
    """
    A transformation parameter whose concrete value is deferred until fit-time, at which
    point its value is "resolved" by a dict of "bindings" provided to the fit() call.
    ...
    """

    name: str

    def resolve(self, bindings: dict[str, object]) -> object:
        # default: treat hp name as key into bindings
        return bindings.get(self.name, self)

    @staticmethod
    def resolve_maybe(v, bindings: dict[str, object]) -> object:
        if isinstance(v, HP):
            return v.resolve(bindings)
        return v


class HPFmtStr(HP):
    def resolve(self, bindings: dict[str, object]) -> object:
        # treate name as format string to be formatted against bindings
        return self.name.format(**bindings)

    @classmethod
    def maybe_from_value(cls, x: str | HP):
        if isinstance(x, HP):
            return x
        if isinstance(x, str):
            if x != "":
                return HPFmtStr(x)
            return x
        raise TypeError(
            f"Unable to create a HPFmtStr from {x!r} which has type {type(x)}"
        )


def fmt_str_field(**kwargs):
    return field(converter=HPFmtStr.maybe_from_value, **kwargs)


# Valid column list specs (routed by field converter):
# hp('which_cols') -> plain old hp
# ['x', 'y', 'z'] -> hp_cols (plain old list?)
# ['x', hp('some_col'), 'z'] -> hp_cols
# ['x', '{som_col}', 'z'] -> hp_cols
# Scalars rewritten to lists of one:
#   'z' -> ['z'] -> hp_cols
#   '{some_col}' - ['{som_col}'] -> hp_cols


@define
class HPLambda(HP):
    resolve_fun: Callable
    name: str = None

    def resolve(self, bindings: dict[str, object]) -> object:
        return self.resolve_fun(bindings)


@define
class HPCols(HP):
    cols: list[str | HP]
    name: str = None

    @classmethod
    def maybe_from_value(cls, x: str | HP | Iterable[str | HP]) -> HPCols | HP:
        if isinstance(x, HP):
            return x
        if isinstance(x, str):
            return cls([x])
        return cls(list(x))

    def resolve(self, bindings):
        return [
            c.resolve(bindings)
            if isinstance(c, HP)
            else c.format(**bindings)
            if isinstance(c, str)
            else c
            for c in self.cols
        ]

    def __repr__(self):
        return repr(self.cols)

    def __len__(self):
        return len(self.cols)

    def __iter__(self):
        return iter(self.cols)


def _validate_not_empty(instance, attribute, value):
    """
    attrs field validator that throws a ValueError if the field value is empty.
    """
    if hasattr(value, "__len__"):
        if len(value) < 1:
            raise ValueError(f"{attribute.name} must not be empty but got {value}")
    elif isinstance(value, HP):
        return
    else:
        raise TypeError(f"{attribute.name} value has no length: {value}")


def columns_field(**kwargs):
    return field(
        validator=_validate_not_empty, converter=HPCols.maybe_from_value, **kwargs
    )


@define
class HPDict(HP):
    mapping: dict
    name: str = None

    def resolve(self, bindings: dict[str, object]) -> object:
        return {
            (k.resolve(bindings) if isinstance(k, HP) else k): v.resolve(bindings)
            if isinstance(v, HP)
            else v
            for k, v in self.mapping.items()
        }

    @classmethod
    def maybe_from_value(cls, x: dict | HP):
        if isinstance(x, HP):
            return x
        if not isinstance(x, dict):
            raise TypeError(
                f"HPDict.maybe_from_value requires an HP or a dict, but got {x} which "
                f"has type {type(x)}"
            )
        # it's a dict
        if all(map(lambda k: not isinstance(k, HP), x.keys())) and all(
            map(lambda v: not isinstance(v, HP), x.values())
        ):
            return x
        return cls(x)


def dict_field(**kwargs):
    return field(converter=HPDict.maybe_from_value, **kwargs)
