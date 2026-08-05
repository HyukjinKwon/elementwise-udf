# elementwise-udf

**A PySpark Python UDF that works inside Spark's native higher-order functions**
-- `transform`, `filter`, `exists`, `aggregate`, `array_sort` and the rest.

Spark normally refuses this. A Python UDF called inside a higher-order function's
lambda fails at analysis
([SPARK-27052](https://issues.apache.org/jira/browse/SPARK-27052)):

```python
from pyspark.sql import functions as F

@F.udf("long")
def plus_one(x):
    return x + 1

df.select(F.transform("values", lambda x: plus_one(x)).alias("result"))
# AnalysisException: [UNSUPPORTED_FEATURE.LAMBDA_FUNCTION_WITH_PYTHON_UDF]
```

That is the only problem this package solves. Declare the UDF with its `udf`
instead of PySpark's, import `functions` from here instead of from `pyspark.sql`,
and the *same* call works -- same `select`, same `F.transform`, same lambda:

```python
# `udf` is aliased to `eudf` here so it is never confused with
# pyspark.sql.functions.udf; either name works.
from elementwise_udf import udf as eudf, functions as F

@eudf("long")
def plus_one(x):
    return x + 1

df.select(F.transform("values", lambda x: plus_one(x)).alias("result")).show()
# +---------+
# |   result|
# +---------+
# |[2, 3, 4]|
# +---------+
```

The UDF still works as an ordinary UDF everywhere else -- `plus_one(F.lit(1))`,
`plus_one("id")`, `spark.udf.register(...)` -- so it can replace
`pyspark.sql.functions.udf` outright rather than sitting beside it.

The name: "element-wise" describes how the UDF runs, one call per array *element*,
which is exactly what a higher-order function's lambda expresses. Nothing about
the package is specific to any one function; every higher-order function Spark
has is covered.

## How it works

A lambda's body must be evaluable inside the JVM, so the UDF is lifted *out* of
it. `F.transform(col, lambda x: plus_one(x) * 2)` is rewritten to roughly:

```python
zipped = F.arrays_zip(col.alias("c0"), plus_one_over_array(col).alias("u0"))
F.transform(zipped, lambda s: s["u0"] * 2)
```

`plus_one_over_array` is the same Python function rebuilt to take a whole array
and loop over it in Python, so it runs once per row rather than once per element.
Its results ride alongside the original elements, and the lambda is re-run with
each UDF call replaced by a reference to the precomputed field.

The native higher-order function still does the iterating; only the UDF moved.
There is no `explode` and no shuffle -- one row in, one row out. The generated
plan is identical to the hand-written version:

```
Project [transform(arrays_zip(values, pythonUDF0, c0, u0), lambdafunction(x.u0))]
+- BatchEvalPython [plus_one_over_array(values)]
   +- Scan ExistingRDD
```

Because the substitution happens before Spark sees the lambda, the UDF result is
just another column there, so expressions around it are ordinary JVM work:

```python
F.transform("values", lambda x: plus_one(x) * 2)            # arithmetic
F.transform("values", lambda x, i: plus_one(x) + i)         # with the index
F.transform("values", lambda x: plus_one(x * 10))           # expression as arg
F.transform("values", lambda x: plus_one(times_ten(x)))     # nested UDFs
F.filter("values", lambda x: is_odd(x))                     # as a predicate
F.transform("values", lambda x: F.when(plus_one(x) > 2, 1).otherwise(0))
```

Each higher-order call is rewritten independently, so any number of them can
appear in one `select` alongside ordinary columns.

## Supported higher-order functions

| Function | How the lambda is handled |
|---|---|
| `transform`, `filter`, `exists`, `forall` | UDF precomputed over the array; index supported |
| `zip_with` | The two arrays are zipped, collapsing to the single-array case |
| `aggregate`, `reduce` | UDF precomputed over the elements; `finish` lambda supported |
| `array_sort`, `sort_array` | UDF becomes a per-element sort *key*; the JVM compares keys |
| `transform_keys`, `transform_values`, `map_filter` | Map split into key/value arrays, then rebuilt |
| `map_zip_with` | Union of both key sets, each map looked up per key |

`functions` is a transparent proxy: every attribute is forwarded to
`pyspark.sql.functions` untouched, nothing in PySpark is patched, and lambdas
that use no element-wise UDF are passed straight through.

## A drop-in for `pyspark.sql.functions.udf`

`udf` accepts every form PySpark's own does, so `from elementwise_udf import udf`
can replace `from pyspark.sql.functions import udf` wholesale:

```python
@eudf("long")                             # decorator with a return type
def plus_one(x):
    return x + 1

@eudf                                     # bare decorator, returnType="string"
def stringify(x):
    return str(x)

plus_one = eudf(lambda x: x + 1, "long")               # called directly
arrow_udf = eudf(lambda x: x + 1, "long", useArrow=True)
```

Outside a higher-order function these behave exactly like a plain PySpark UDF --
`plus_one("id")`, `plus_one(F.lit(1))`, `asNondeterministic()`, `.returnType`.
For SQL registration, hand Spark the real UDF underneath via `.scalar`:

```python
spark.udf.register("plus_one", plus_one.scalar)
```

`pandas_udf` is not covered: it receives a Series rather than single values, so
the element-wise rewrite does not apply to it.

## Slow paths (they work, but warn)

Two shapes cannot be precomputed, because the value the UDF needs does not exist
until the higher-order function is already running. They still work, by moving
the *whole* operation into one Python call per row, and each raises a
`RuntimeWarning` because the cost profile is much worse:

```python
# UDF on aggregate's accumulator: the whole fold runs in Python,
# calling the UDF once per element sequentially.
F.aggregate("v", F.lit(0).cast("long"), lambda acc, x: plus_one(acc) + x)

# Pairwise comparator: the whole sort runs in Python to produce per-element
# ranks, calling the UDF O(n log n) times; array_sort then orders by rank.
F.array_sort("v", lambda a, b: my_compare(a, b))
```

Applying the UDF to the element instead keeps it on the fast path.

## Not supported

**A Python `if` on a UDF result.**

```python
F.transform("values", lambda x: 1 if plus_one(x) else 0)   # CANNOT_CONVERT_COLUMN_INTO_BOOL
```

`if` needs a real boolean while the lambda is being traced, and a Column cannot
provide one. This is the same limitation plain PySpark has for `if col > 1`. Use
`F.when(...)` instead.

**A UDF on `aggregate`'s accumulator when the merge step *also* uses Spark
expressions.**

```python
# supported: the whole fold replays in Python
F.aggregate("v", F.lit(0).cast("long"), lambda acc, x: plus_one(acc) + x)

# not supported: F.when cannot be replayed in Python, and the accumulator
# cannot be precomputed -> TypeError explaining both options
F.aggregate("v", F.lit(0).cast("long"),
            lambda acc, x: F.when(plus_one(acc) > 2, 1).otherwise(0) + x)
```

Writing such a fold out longhand -- one explicit step per element -- looks
tempting, and it does produce the right answer, but each step references the
previous accumulator twice (once per `when` branch), so the expression tree
*doubles* per step. Spark has no let-binding to share those subtrees. Measured on
a single 3-element array with a bound of 8 steps: **13.3s** versus **0.13s** for
the Python replay, roughly 100x for an identical result. Spark Connect is worse
still -- it serializes the tree to protobuf with no node sharing, and even 2
steps never finished serializing. That approach was implemented, measured, and
removed; the clear `TypeError` is deliberate. Keep the merge step in plain
Python, or apply the UDF to the element.

## Performance

Measured on 4 cores, 2M elements, identical on classic and Connect:

| | 200k rows x 10 | 2k rows x 1000 |
|---|---|---|
| `F.transform` + element-wise UDF | 0.34s | 0.27s |
| native `transform`, no UDF (floor) | 0.03s | 0.03s |

The floor is the Python UDF boundary itself, not this rewrite: if your logic is
expressible in native Spark functions, that remains far faster. On the fast path
the UDF runs once per row over a whole array, so work parallelizes across rows
but not within a single row.

The two warning paths above are far slower again. `aggregate` over the
accumulator replays the fold in Python once per row (0.13s for the case above,
versus 0.06s to build the fast path); a pairwise comparator calls the UDF
O(n log n) times per row. Both are correctness escape hatches, not something to
build a pipeline on.

## Requirements

PySpark 4.0+, on classic PySpark or Spark Connect / Databricks Connect
(including serverless). CI covers Spark 4.0, 4.1 and 4.2 in both session modes.
Plain and Arrow-optimized Python UDFs are both supported.

## Development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
SPARK_MODE=classic pytest   # classic session
SPARK_MODE=connect pytest   # Spark Connect session
black --line-length 100 .
```

## License

Apache License 2.0.
