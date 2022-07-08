"""
Provides a library of basic Transform subclasses.
"""
from __future__ import annotations
import inspect
import logging
from logging import Logger

from typing import Callable, Optional, TextIO
from attrs import define
import pandas as pd

from .core import (
    Transform,
    columns_field,
    StatelessTransform,
    HP,
    dict_field,
    fmt_str_field,
)

_LOG = logging.getLogger(__name__)


class Identity(StatelessTransform):
    """
    The stateless Transform that, at apply-time, simply returns the input data
    unaltered.
    """

    def _apply(self, df_apply: pd.DataFrame, state: object = None):
        return df_apply


@define(slots=False)
class ColumnsTransform(Transform):
    """
    Abstract base clase of all Transforms that require a list of columns as a parameter
    (cols).  Subclasses acquire a mandatory `cols` argument to their constructors, which
    can be supplied as a list of any combination of:

    - string column names,
    - hyperparameters exepcted to resolve to column names,
    - format strings that will be evaluated as hyperparameters formatted on the bindings
      dict.

    A scalar value for cols (e.g. a single string) will be converted automatically to a
    list of one element at construction time.

    Subclasses may define additional parameters with the same behavior as cols by using
    columns_field().
    """

    cols: list[str | HP] = columns_field()


@define(slots=False)
class WeightedTransform(Transform):
    """
    Abstract base class of Transforms that accept an optional weight column as a
    parameter (w_col).
    """

    w_col: Optional[str] = None


@define
class Copy(StatelessTransform, ColumnsTransform):
    """
    A stateless Transform that copies values from one or more source columns into
    corresponding destination columns, either creating them or overwriting their
    contents.
    """

    dest_cols: list[str | HP] = columns_field()

    # FIXME: we actually may not be able to validate this invariant until after
    # hyperparams are bound
    @dest_cols.validator
    def _check_dest_cols(self, attribute, value):
        lc = len(self.cols)
        lv = len(value)
        if lc == 1 and lv > 0:
            return

        if lv != lc:
            raise ValueError(
                "When copying more than one source column, "
                f"cols (len {lc}) and dest_cols (len {lv}) must have the same "
                "length."
            )

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        if len(self.cols) == 1:
            src_col = self.cols[0]
            return df_apply.assign(
                **{dest_col: df_apply[src_col] for dest_col in self.dest_cols}
            )

        return df_apply.assign(
            **{
                dest_col: df_apply[src_col]
                for src_col, dest_col in zip(self.cols, self.dest_cols)
            }
        )


class Select(StatelessTransform, ColumnsTransform):
    """
    Select the given columns from the data.

    .. TIP::
        As syntactic sugar, :class:`Pipeline` overrides the index operator (via a custom
        ``__getitem__`` implementatino) as a synonym for appending a ``Select``
        transform to the pipeline. For example, the following three pipelines are
        equivalent::

            ff.Pipeline([..., Select(["col1", "col2"]), ...])

            (
                ff.Pipeline()
                ...
                .select(["col1", "col2"]
                ...
            )

            (
                ff.Pipeline()
                ...
                ["col1", "col2"]
                ...
            )
    """

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        return df_apply[self.cols]


class Drop(StatelessTransform, ColumnsTransform):
    """
    Drop the given columns from the data.
    """

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        return df_apply.drop(columns=self.cols)


@define
class Rename(StatelessTransform):
    """
    Rename columns.

    :param how: Either a function that, given a column name, returns what it should be
        renamed do, or a dict from old column names to corresponding new names.
    """

    how: Callable | dict[str, str]

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        return df_apply.rename(columns=self.how)


@define
class StatelessLambda(StatelessTransform):
    apply_fun: Callable  # df[, bindings] -> df

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        sig = inspect.signature(self.apply_fun).parameters
        if len(sig) == 1:
            return self.apply_fun(df_apply)
        elif len(sig) == 2:
            return self.apply_fun(df_apply, self.bindings())
        else:
            # TODO: raise this earlier in field validator
            raise TypeError(f"Expected lambda with 1 or 2 parameters, found {len(sig)}")


@define
class StatefulLambda(Transform):
    fit_fun: Callable  # df[, bindings] -> state
    apply_fun: Callable  # df, state[, bindings] -> df

    def _fit(self, df_fit: pd.DataFrame) -> object:
        sig = inspect.signature(self.fit_fun).parameters
        if len(sig) == 1:
            return self.fit_fun(df_fit)
        elif len(sig) == 2:
            return self.fit_fun(df_fit, self.bindings())
        else:
            # TODO: raise this earlier in field validator
            raise TypeError(f"Expected lambda with 1 or 2 parameters, found {len(sig)}")

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        sig = inspect.signature(self.apply_fun).parameters
        if len(sig) == 2:
            return self.apply_fun(df_apply, state)
        elif len(sig) == 3:
            return self.apply_fun(df_apply, state, self.bindings())
        else:
            # TODO: raise this earlier in field validator
            raise TypeError(f"Expected lambda with 2 or 3 parameters, found {len(sig)}")


@define
class Pipe(StatelessTransform, ColumnsTransform):
    apply_fun: Callable  # df[, bindings] -> df

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        result = self.apply_fun(df_apply[self.cols])
        return df_apply.assign(**{c: result[c] for c in self.cols})


# TODO Rank, MapQuantiles


@define
class Clip(StatelessTransform, ColumnsTransform):
    upper: Optional[float] = None
    lower: Optional[float] = None

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        return df_apply.assign(
            **{
                col: df_apply[col].clip(upper=self.upper, lower=self.lower)
                for col in self.cols
            }
        )


@define
class Winsorize(ColumnsTransform):
    # assume symmetric, i.e. trim the upper and lower `limit` percent of observations
    limit: float

    def _fit(self, df_fit: pd.DataFrame) -> object:
        if not isinstance(self.limit, float):
            raise TypeError(
                f"Winsorize.limit must be a float between 0 and 1. Got: {self.limit}"
            )
        if self.limit < 0 or self.limit > 1:
            raise ValueError(
                f"Winsorize.limit must be a float between 0 and 1. Got: {self.limit}"
            )

        return {
            "lower": df_fit[self.cols].quantile(self.limit, interpolation="nearest"),
            "upper": df_fit[self.cols].quantile(
                1.0 - self.limit, interpolation="nearest"
            ),
        }

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        return df_apply.assign(
            **{
                col: df_apply[col].clip(
                    upper=state["upper"][col], lower=state["lower"][col]
                )
                for col in self.cols
            }
        )


@define
class ImputeConstant(StatelessTransform, ColumnsTransform):
    value: object

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        return df_apply.assign(
            **{col: df_apply[col].fillna(self.value) for col in self.cols}
        )


def _weighted_means(df, cols, w_col):
    return df[cols].multiply(df[w_col], axis="index").sum() / df[w_col].sum()


@define
class DeMean(WeightedTransform, ColumnsTransform):
    """
    De-mean some columns.
    """

    def _fit(self, df_fit: pd.DataFrame) -> object:
        if self.w_col is not None:
            return _weighted_means(df_fit, self.cols, self.w_col)
        return df_fit[self.cols].mean()

    def _apply(self, df_apply: pd.DataFrame, state: pd.DataFrame):
        means = state
        return df_apply.assign(**{c: df_apply[c] - means[c] for c in self.cols})


@define
class ImputeMean(WeightedTransform, ColumnsTransform):
    def _fit(self, df_fit: pd.DataFrame) -> object:
        if self.w_col is not None:
            return _weighted_means(df_fit, self.cols, self.w_col)
        return df_fit[self.cols].mean()

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        means = state
        return df_apply.assign(**{c: df_apply[c].fillna(means[c]) for c in self.cols})


@define
class ZScore(WeightedTransform, ColumnsTransform):
    def _fit(self, df_fit: pd.DataFrame) -> object:
        if self.w_col is not None:
            means = _weighted_means(df_fit, self.cols, self.w_col)
        else:
            means = df_fit[self.cols].mean()
        return {"means": means, "stddevs": df_fit[self.cols].std()}

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        means, stddevs = state["means"], state["stddevs"]
        return df_apply.assign(
            **{c: (df_apply[c] - means[c]) / stddevs[c] for c in self.cols}
        )


@define
class Print(Identity):
    """
    An identity transform that has the side-effect of printing a message at fit- and/or
    apply-time.

    :param fit_msg: Message to print at fit-time.
    :param apply_msg: Message to print at apply-time.
    :param dest: File object to which to print, or the name of a file to open in append
        mode. If ``None`` (default), print to stdout.
    """

    fit_msg: Optional[str] = None
    apply_msg: Optional[str] = None
    dest: Optional[TextIO | str] = None  # if str, will be opened in append mode

    def _fit(self, df_fit: pd.DataFrame):
        if self.fit_msg is None:
            return
        if isinstance(self.dest, str):
            with open(self.dest, "a") as dest:
                print(self.fit_msg, file=dest)
        else:
            print(self.fit_msg, file=self.dest)

        # Idiomatic super() doesn't work because at call time self is a FitPrint
        # instance, which inherits directly from FitTransform, and not from
        # Print/Identity. Could maybe FIXME by futzing with base classes in the
        # metaprogramming that goes on in core.py
        # return super()._fit(df_fit)

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        if self.apply_msg is None:
            return df_apply
        if isinstance(self.dest, str):
            with open(self.dest, "a") as dest:
                print(self.apply_msg, file=dest)
        else:
            print(self.apply_msg, file=self.dest)
        return df_apply
        # Same issue with super() as in _fit().
        # return super()._apply(df_apply)


@define
class LogMessage(Identity):
    """
    An identity transform that has the side-effect of logging a message at fit- and/or
    apply-time. The message string(s) must be fully known at construction-time.

    :param fit_msg: Message to log at fit-time.
    :param apply_msg: Message to log at apply-time.
    :param logger: Logger instance to which to log. If None (default), use
        ``logging.getLogger("frankenfit.transforms")``
    :param level: Level at which to log, default ``INFO``.
    """

    fit_msg: Optional[str] = None
    apply_msg: Optional[str] = None
    logger: Optional[Logger] = None
    level: int = logging.INFO

    def _fit(self, df_fit: pd.DataFrame):
        if self.fit_msg is not None:
            logger = self.logger or _LOG
            logger.log(self.level, self.fit_msg)
        return Identity._fit(self, df_fit)

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        if self.apply_msg is not None:
            logger = self.logger or _LOG
            logger.log(self.level, self.apply_msg)
        return Identity._apply(self, df_apply, state=state)


@define(slots=False)
class SKLearn(Transform):
    """
    Wrap a scikit-learn ("sklearn") model. At fit-time, the given sklearn model class
    is instantiated (with arguments from ``class_params``) and trained on the fitting
    data by calling its ``fit()`` method. At apply-time, the now-fit sklearn model
    object is used to generated predictions by calling its `predict()` method, which are
    assigned to the apply-time data as a new column, ``hat_col``.

    :param sklearn_class: The sklearn class to wrap.
    :param x_cols: The predictor columns. These are selected from the fit/apply-data to
        create the ``X`` argument to the sklearn model's ``fit()`` and ``predict()``
        methods.
    :param response_col: The response column. At fit-time, this is selected from the
        fitting data to create the ``y`` argument to the sklearn model's ``fit()``
        method.
    :param hat_col: The name of the new column to create at apply-time containing
        predictions from the sklearn model.
    :param class_params: Parameters to pass as kwargs to the ``sklearn_class``
        constructor when instantiating it.
    :param w_col: The sample weight column. If specified, this is selected at fit-time
        from the fitting data to create the ``sample_weight`` keyword argument to the
        sklearn model's ``fit()`` method.

        .. WARNING:: Not every sklearn model accepts a ``sample_weight`` keyword
            argument to its ``fit()`` method. Consult the documentation of whichever
            sklearn model you are using.
    """

    sklearn_class: type | HP
    x_cols: list[str] = columns_field()
    response_col: str = fmt_str_field()
    hat_col: str = fmt_str_field()
    class_params: dict[str, object] = dict_field(factory=dict)
    w_col: Optional[str] = fmt_str_field(factory=str)

    def _fit(self, df_fit: pd.DataFrame) -> object:
        model = self.sklearn_class(**self.class_params)
        X = df_fit[self.x_cols]
        y = df_fit[self.response_col]
        if self.w_col:
            w = df_fit[self.w_col]
            # TODO: raise exception if model.fit signature has no sample_weight arg
            model = model.fit(X, y, sample_weight=w)
        else:
            model = model.fit(X, y)

        return model

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        model = state
        return df_apply.assign(**{self.hat_col: model.predict(df_apply[self.x_cols])})


@define(slots=False)
class Statsmodels(Transform):
    """
    Wrap a statsmodels model.
    """

    sm_class: type | HP
    x_cols: list[str] = columns_field()
    response_col: str = fmt_str_field()
    hat_col: str = fmt_str_field()
    class_params: dict[str, object] = dict_field(factory=dict)

    def _fit(self, df_fit: pd.DataFrame) -> object:
        X = df_fit[self.x_cols]
        y = df_fit[self.response_col]
        model = self.sm_class(y, X, **self.class_params)
        return model.fit()

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        model = state
        return df_apply.assign(**{self.hat_col: model.predict(df_apply[self.x_cols])})


@define
class Correlation(StatelessTransform):
    """
    Compute the correlation between each pair of columns in the cross-product of
    ``left_cols`` and ``right_cols``.

    :param left_cols: List of "left" correlands. Result will have one row per element of
        ``left_cols``.
    :param right_cols: List of "right" correlands. Result will have one column per
        element of ``right_cols``.
    :param method: One of ``"pearson"``, ``"spearman"``, or ``"kendall"``, specifying
        which type of correlation coefficient to compute.
    :param min_obs: The minimum number of non-missing values for each pair of columns.
        If a pair has fewer than this many non-missing observations, then the
        correlation for that pair will be missing in the result.

    Example::

        from pydataset import data
        df = data("diamonds")
        ff.Correlation(["price"], ["carat"]).apply(df)
        # -->           carat
        # --> price  0.921591

        ff.Correlation(["table", "depth"], ["x", "y", "z"]).apply(df)
        # -->               x         y         z
        # --> table  0.195344  0.183760  0.150929
        # --> depth -0.025289 -0.029341  0.094924

    .. SEEALSO::
        The parameters of :meth:`pandas.DataFrame.corr()`.
    """

    left_cols: list[str] = columns_field()
    right_cols: list[str] = columns_field()
    method: Optional[str | HP] = "pearson"
    min_obs: Optional[int] = 1

    def _apply(self, df_apply: pd.DataFrame, state: object = None) -> pd.DataFrame:
        cm = df_apply[self.left_cols + self.right_cols].corr(
            method=self.method, min_periods=self.min_obs
        )
        return cm.loc[self.left_cols, self.right_cols]
