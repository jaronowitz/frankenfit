# Design notes

## Options considered for `join()` method on `Pipeline` objects.

The question is: do we want `Pipeline.join()` to always use `self` as the `left`
argument of the `Join`, or should we also be able to use the `join()` method to append a
`Join` of two Piplines unrelated to `self`? We decided that's kind of weird and possibly
confusing, plus pandas `DataFrame.merge()` treats `self` as `left`, so for clarity and
consistency with pandas we do the same. If you really want to append a `Join` of two
other Pipelines, use `.then(Join(...))`. But this is still probably weird because aren't
you effacing all of the data up to that point in the enclosing Pipeline?

## Options considered for `groupby()` method on `Pipeline` objects.
Decision: Option 1, for its succinctness of style and similarity to Pandas.

**Option 1**, like pandas:
```python
(
  ff.Pipeline()
  ...
  .groupby("cut")
    .z_score(cols)
) # -> Pipeline([..., GroupBy("cut", ZScore(cols))])

(
  ff.Pipeline()
  ...
  .groupby("cut")
    .then(bake_features(cols))
) # -> Pipeline([..., GroupBy("cut", bake_features(cols))])
```
This requires `groupby()` to return some kind of intermediate "grouper" object (like
pandas), which is not a Pipeline, but has the same call-chain methods as a Pipeline, and
consumes the next call to finally create the GroupBy Transform and return the result of
appending that to the matrix Pipeline.

**Option 2**, as a sort of postfix operator on the Pipeline (or on the last Transform in the
Pipeline):
```python
(
  ff.Pipeline()
  ...
  .z_score(cols)
  .grouped_by("cut")
) # -> GroupBy("cut", Pipeline([..., ZScore(cols)]))
```
Some nuance here... does it group the whole Pipeline that it's called on (that's what's
annotated above) or just the *last* Transform in the Pipeline? E.g., the above would
come out as `Pipeline([..., GroupBy("cut", ZScore(cols))])`.

The latter semantics is quite appealing because it's in the same spirit as Option 1
(just reverse ordering), and much simpler to implement, because `grouped_by()` just
needs to look at the last element of `self.transforms` for the thing to be grouped, no
intermeidate grouper object needed. This may interact with some of the Pipeline
coalescing logic, though. It would no longer be the case that pipelines `[t1, t2, t3]`
and `[t1, [t2, t3]]` are equivalent, because `.grouped_by()` would have different
meaning on them.

**Option 3**, as a bivalent function:
```python
(
  ff.Pipeline()
  ...
  .groupby(
    "cut",
    ff.Pipeline().z_score(cols)
  )
)
```
Feels like the most unambiguous/clear, but also the most verbose, especially if you just
want to group a single Transform, which forces you either to break out of call-chaining
style, or create a length-1 Pipeline.

The choice will influence syntax for other groupers like Sequentially.

## Options for CrossValidate graph Transform

```python
(
  ff.Pipeline()
  ...
  .cross_validated(
    score_transform,
    k=5,
    split_on="time",  # optional folds are defined by distinct values of "time" rather
                      # than by rows?
  )  # -> "score" column, row per fold
  .stateless_lambda(lambda df: df[["score"]].mean()) #  mean of per-fold scores
)

# Behavior is analogous to sequential fitting, just with folds rather than periods.
# This is a pipline that always generates OOS predictions even on its own training set.
pip = (
  ff.Pipeline()
  ...
  # randomly divide rows into 5 partitions and add a "__fold__" column indicating which
  # partition each row is in
  .partition('__fold__', k=5, shuffle=True, units="<row>")
  # for each distinct value of "__fold__", fit the pipeline on data from all *other*
  # folds and apply it out-of-sample on that fold. State is a map from distinct value of
  # __fold__ to a FitTransform trained on all the other folds.
  .cross_validate(by='__fold__')
)

# CV evaluation is just
pip.correlation(...)
# or
pip.groupby("__fold__").correlation(...).mean(...)

# apply to new test data, as long as it comes with a __fold__ column
pip.apply(...)
```
... Maybe a better name would be like `k_folded()`?

Key insight is that cross-validated and sequential fits are both instances of
resampling, which is a generalization of groupby. GroupBy and Resample both have fit
state per chunk of data, which is used to apply on new chunks with matching key...
Resample is just GroupBy with an added layer of indirection for determining the training
data of the model that goes with each apply-time chunk.

```python
(
  ff.Pipeline()
  .group_by('foo').then(...)
)
(
  ff.Pipeline()
  .resample_by('foo', training_schedule).then(...)
)
```
... we could even just add a `training_schedule=` arg to GroupBy instead of a new
ResampleBy transform, with the default training schedule being in-sample.

## Interesting potential equivalencies

```python
transform.fit(df) == Pipeline().read_data_frame(df).then(transform).fit()
transform.fit(df) == (ReadDataFrame(df) + transform).fit()
transform.fit(df) == ReadDataFrame(df).then(transform).fit()
```
Same for ``StatelessTransform.apply()`` and ``Pipeline.apply()``.

Take-aways:

* ``__add__`` and ``then()`` could become methods of ``Transform`` rather than
  of ``Pipeline``; they would return ``Pipeline``, which still handles the data
  passing logic. ``Pipeline`` would have to become a core class. ``Done.``

  One could even imagine the call-chain API hanging on ``Transform`` rather
  than on ``Pipeline``... you'd no longer need to begin a de-novo call-chain
  with an empty ``ff.Pipeline()``... but by having the call-chain API on the
  super class of all Transforms, we'd severely interfere with the namespaces of
  all Transforms' parameters.

* ``Transform.fit(df)``, ``StatelessTransform.apply(df)``,
  ``Pipeline.apply(df)`` could be implemented by prepending a ``ReadDataFrame`` to
  ``self`` in a new ``Pipeline`` and calling its ``fit()``/``apply()`` method
  with no arguments. ``ReadDataFrame`` would have to become a core class.

## Constant transforms

``Done.``

``DataReader``s/constant transforms should be subclasses of some
``ConstantTransform``, which is a ``StatelessTransform`` with special
``fit()``/``apply()`` implementations that warn about non-empty df args before
calling super.

## A tale of two APIs

We have the class-based API:

```python
ff.Pipeline([
  ff.ReadDataset(...),
  ff.GroupBy(
    "x",
    ff.ZScore("y")
  ),
  ff.Select(["x", "y"]),
])

# Or:
(
  ff.ReadDataSet(...)
  + ff.GroupBy(
    "x",
    ff.ZScore("y")
  )
  + ff.Select(["x", "y"])
)
```

And we have the Pandas-inspired call-chain API:

```python
(
  ff.Pipeline()
  .read_dataset(...)
  .group_by("x")
    .z_score("y")
  [["x", "y"]]
)
```

Does it make sense to have both around?

The Transforms and FitTransforms really do need to be objects in a class
hierarchy, so we need the classes.

Incorporation of user-defined Transforms: happens naturally in the class-based
API:
```python
ff.Pipeline([
  ff.ReadDataset(...),
  ff.GroupBy(
    "x",
    ff.ZScore("y")
  ),
  ff.Select(["x", "y"]),
  MyTransform(42),
])
```

For the call-chain API, there are some options:

* Interrupt the call-chaining to manually append the user-defined Transform:
  ```python
  (
    (
      ff.Pipeline()
      .read_dataset(...)
      .group_by("x")
        .z_score("y")
      [["x", "y"]]
    )
    + MyTransform(42)
    + (
      ff.Pipeline()
      ...
    )
  )
  ```

* Use a special call-chain method that appends an arbitrary Transform to the Pipeline:
  ```python
  (
    ff.Pipeline()
    .read_dataset(...)
    .group_by("x")
      .z_score("y")
    [["x", "y"]]
    .then(
      MyTransform()
    )
  )
  ```

  This ``then()`` method is useful to have around anyway for grouping
  sub-pipelines even without user-defined Transforms, e.g.:

  ```python
  (
    ff.Pipeline()
    .group_by("x")
      .then(
        ff.Pipeline()
        .winsorize("y")
        .z_score("y")
      )
  )
  ```

* Provide a mechanism for authors of new Transforms to register new
  call-chainable methods on Pipeline objects. This has a bad code smell; it will
  get confusing and frustrating when the public API of a frequently used class
  like Pipeline is effectively mutable and dependent on what sequence of user
  code has been run.
  ```python
  @Pipeline.register_transform("my_transform")
  @define
  class MyTransform(Transform):
    foo: int
    # ...

  (
    ff.Pipeline()
    .read_dataset(...)
    .group_by("x")
      .z_score("y")
    [["x", "y"]]
    .my_transform(42)
  )
  ```
* Provide an easy way for the user to create his own derived Pipeline subclasses that
  have additional call-chain methods, and build all of his data pipelines from
  those.
  ```python
  class MyPipeline(Pipeline):
    pass

  MyPipeline.register_transform("my_transform", MyTransform)
  # or even
  @MyPipeline.register_transform("my_transform")
  @define
  class MyTransform(Transform):
    foo: int

  (
    MyPipeline()
    .read_dataset(...)
    .group_by("x")
      .z_score("y")
    [["x", "y"]]
    .my_transform(42)
  )

  # or with a class method that creates new derived classes:
  MyPipeline = ff.Pipeline.with_methods(my_transform=MyTransform)
  (
    MyPipeline()
    ...
  )

  # or inlined
  (
    ff.Pipeline(
      my_transform=MyTransform
    )
    .read_dataset(...)
    .group_by("x")
      .z_score("y")
    [["x", "y"]]
    .my_transform(42)
  )

  # We could even use this ourselves to define various specializations of Pipeline:
  TimeseriesPipeline = ff.Pipeline.with_methods(
    cross_sectionally=CrossSectionally,
    longitudinally=Longitudinally,
    add_windows=AddWindows,
    read_dataset=ReadTimeseriesDataset,
  )
  # Even with new behaviors:
  class TimeseriesPipeline(ff.Pipeline.with_methods(
    cross_sectionally=CrossSectionally,
    longitudinally=Longitudinally,
    add_windows=AddWindows,
    read_dataset=ReadTimeseriesDataset,
  )):
    def asof_join(self, other, ...):
      ...
      return TimeseriesPipeline([AsOfJoin(self, other, ...)])
  ```

  In general this could be useful to establish conventions about dataset shape
  for some problem domain (e.g. an ordered "time" index) and to work with
  a set of transformations that only make sense on datasets with those
  conventions.

Would it make sense to reduce the duplicity of class names and method names by
changing all of the class names to lower_snake_case, making them the same as the
call-chain methods?

```python
ff.pipeline().then(ff.z_score())
```

Or what if we still had the CamelCase classes, but each was accompanied by a
global method that returns a pipeline containing it?
```
ff.z_score("y") -> ff.Pipeline([ff.ZScore("y")])
```

We could then commence pipelines without the Pipeline() heading:
```
(
  ff
  .read_dataset()
  .group_by("x")
    .z_score("y")
  [["x", "y"]]
)
```

Presumably it would be up to the author of a user-defined Transform to provide
his own such global method, and there's still the question of how to fit into
call-chaining. And how would this play with specializations of Pipeline?

Pipeline specialization is a cool idea. We could even reframe our current
functionality as ``DataFramePipeline``, which is a specialization of the root
``ObjectPipeline``.

* ``ObjectPipeline``: the arguments to fit and apply are arbitrary objects, with
  no convention. Call-chain methods are exposed only for the core set of
  Transforms that don't assume anything about the type of the data: ``then()``,
  ``stateless_lambda()``, ``stateful_lambda()``, ``print()``, ``log_message()``,
  ``group_by_bindings()``(but actually how to combine results?),
  ``if_hyperparam_is_true()``, ``if_hyperparam_lambda()``,
  ``if_fitting_data_lambda()``....

* ``DataFramePipeline``: the arguments to fit and apply are Pandas dataframes,
  but with no assumptions about index or column schema. Maybe just that they are
  2-d? This inherits the methods of ``ObjectPipeline`` and adds ``select()``,
  ``rename()``, ``drop()``, ``assign()``, ``group_by()``, etc.
  ``TimeSeriesPipeline`` is a subclass.

* Could actually have Mixins then for various other thematic sets of Transforms
  (normalizers, smoothers, windowers).  Have to think about how they'd
  coordinate compatible/incompatible data conventions.

* Could imagine specializations for other problem domains: ``ImagePipeline``,
  ``TextPipeline``.


Rename ``group_by_bindings()`` to ``for_each_binding()``?
