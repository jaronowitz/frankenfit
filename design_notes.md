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
