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
