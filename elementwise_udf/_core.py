"""
Use a Python UDF inside Spark's native higher-order functions.

Spark rejects a Python UDF in a higher-order function's lambda
(`SPARK-27052 <https://issues.apache.org/jira/browse/SPARK-27052>`__)::

    F.transform("values", lambda x: plus_one(x))
    # [UNSUPPORTED_FEATURE.LAMBDA_FUNCTION_WITH_PYTHON_UDF]

Declare the UDF with :func:`elementwise_udf`, and import ``functions`` from here
instead of from ``pyspark.sql``. The higher-order call then works as written::

    from elementwise_udf import elementwise_udf, functions as F

    @elementwise_udf("long")
    def plus_one(x):
        return x + 1

    df.select(F.transform("values", lambda x: plus_one(x)).alias("result"))

The rejection is on the plan shape -- any Python UDF inside a ``lambdafunction``
node is refused -- so the UDF has to be lifted out of the lambda. That is done
by rewriting the call from outside::

    zipped = F.arrays_zip(col.alias("c0"), plus_one_over_array(col).alias("u0"))
    F.transform(zipped, lambda s: s["u0"])

``plus_one_over_array`` is the same Python function rebuilt to take a whole array
and loop over it in Python, so it runs once per row instead of once per element.
Its results ride alongside the original elements, and the lambda is re-run with
each UDF call replaced by a reference to the precomputed field. The native
higher-order function still does the iterating; only the UDF moved.

Because the substitution happens before Spark sees the lambda, the UDF result is
just another column there, so expressions around it are ordinary JVM work::

    F.transform("values", lambda x: plus_one(x) * 2)          # arithmetic
    F.transform("values", lambda x, i: plus_one(x) + i)       # with the index
    F.transform("values", lambda x: plus_one(x * 10))         # expression as arg
    F.transform("values", lambda x: plus_one(times_ten(x)))   # nested UDFs
    F.filter("values", lambda x: is_odd(x))                   # as a predicate
    F.transform("values", lambda x: F.when(plus_one(x) > 2, 1).otherwise(0))

Every higher-order function is covered. Those whose lambda is not a plain element
mapping are reduced to that case:

``zip_with``
    The two arrays are zipped, collapsing it to the single-array case.
``aggregate`` / ``reduce``
    The UDF is precomputed over the element array and the fold then runs over the
    zipped elements. A ``finish`` lambda is supported.
``array_sort`` / ``sort_array``
    The UDF becomes a sort *key*, computed once per element; the JVM comparator
    then compares precomputed keys.
``transform_keys`` / ``transform_values`` / ``map_filter``
    The map is split into its key and value arrays, rewritten as arrays, and
    rebuilt into a map.
``map_zip_with``
    The union of both maps' keys is built explicitly and each map looked up per
    key, giving three aligned arrays.

Two shapes cannot be precomputed, because the value the UDF needs does not exist
until the higher-order function is already running. They still work, by moving
the *whole* operation into one Python call per row, and each warns
(``RuntimeWarning``) because the cost profile is much worse:

* A UDF applied to ``aggregate``'s *accumulator*, which only exists between fold
  steps. The entire fold runs in Python, calling the UDF once per element
  sequentially.
* A comparator passing *both* elements to one UDF call -- a genuinely pairwise
  decision. The whole sort runs in Python to produce per-element ranks, calling
  the UDF O(n log n) times, and native ``array_sort`` then orders by rank.

In both cases, applying the UDF to the element instead keeps it on the fast path.

One thing genuinely cannot work: a Python ``if`` on a UDF result
(``lambda x: 1 if plus_one(x) else 0``). ``if`` needs a real boolean while the
lambda is being traced, and a Column cannot provide one -- the same limitation
plain PySpark has for ``if col > 1``. Use ``F.when(...)`` instead.

``functions`` is a transparent proxy: every attribute is forwarded to
``pyspark.sql.functions`` untouched, nothing in PySpark is patched, and lambdas
that use no ``elementwise_udf`` are passed straight through.

Works on classic PySpark and on Spark Connect / Databricks Connect (serverless).

Cost: on the fast path the UDF runs once per row over a whole array, so work
parallelizes across rows but not within a single row.
"""

import functools
import inspect
import itertools
import types
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from pyspark.sql import Column
from pyspark.sql import functions as _F
from pyspark.sql.types import ArrayType, DataType, IntegerType

__all__ = ["elementwise_udf", "functions"]

# ``filter`` returns the input elements; every other array higher-order function
# returns the lambda's values. Its lambda is only a predicate, so the elements
# have to be recovered after filtering.
_RETURNS_INPUT_ELEMENTS = frozenset(["filter"])

# A comparator's second parameter is another element, not the element's position:
# ``array_sort(col, lambda a, b)`` is indistinguishable from
# ``transform(col, lambda x, i)`` from the call site. Reading ``b`` as an index
# would silently mis-sort, and a comparator has no element-wise reading anyway.
_COMPARATORS = frozenset(["array_sort", "sort_array"])

# Folds: the lambda takes (accumulator, element). Only the element can be
# precomputed, since the accumulator is produced by the fold itself.
_FOLDS = frozenset(["aggregate", "reduce"])

# Map-valued higher-order functions, handled by splitting the map into its key
# and value arrays and rebuilding a map afterwards.
_MAP_HOFS = frozenset(["transform_keys", "transform_values", "map_filter"])

# Takes two maps and a three-parameter lambda (key, value1, value2).
_MAP_ZIP = frozenset(["map_zip_with"])


class _Recorder:
    """Tracks the UDF calls a lambda makes while a rewrite is being built.

    The lambda is run twice: once to discover its UDF calls, and once more to
    build the expression Spark sees, with each call replaced by its precomputed
    column. ``mode`` says which run is in progress.
    """

    def __init__(self) -> None:
        self.mode = "record"
        self.calls: List[Tuple["_ElementwiseUDF", Tuple[Any, ...]]] = []
        self.index = 0
        self.substitutions: List[Column] = []
        # placeholder id -> index of the call that produced it
        self.placeholders: Dict[int, int] = {}
        # Placeholders are identified by ``id``, which is only unique while the
        # object lives, so keep a reference to every one handed out.
        self.alive: List[Column] = []

    def on_call(self, udf: "_ElementwiseUDF", args: Tuple[Any, ...]) -> Column:  # noqa: D401
        i = self.index
        self.index += 1
        if self.mode == "substitute":
            if i >= len(self.substitutions):
                raise TypeError(
                    "the lambda called element-wise UDFs a different number of "
                    "times on separate runs, so its shape cannot be determined; "
                    "avoid making UDF calls conditional on Python state"
                )
            return self.substitutions[i]
        self.calls.append((udf, args))
        # A typed null keeps any surrounding expression type-correct while this
        # exploratory run finishes; it is never evaluated. Recorded by id and
        # kept alive, so a UDF call that receives another's result can be
        # recognized as nested and pointed back at the producing call.
        placeholder = _F.lit(None).cast(udf.returnType)
        self.alive.append(placeholder)
        self.placeholders[id(placeholder)] = i
        return placeholder


# Set only while the proxy is running a lambda; otherwise UDF calls are normal.
_active: Optional[_Recorder] = None

# Set while a lambda is being replayed on real Python values inside a UDF (the
# sequential fold), where a UDF call should compute rather than build a Column.
_direct = False


class _ElementwiseUDF:
    """A Python UDF usable both normally and inside a higher-order function."""

    def __init__(self, func: Callable[..., Any], returnType: Union[str, DataType]):
        self.func = func
        self._returnType = returnType
        self._scalar: Optional[Any] = None
        functools.update_wrapper(self, func)

    @property
    def scalar(self) -> Any:
        # Built on first use: a DDL return type is parsed against the active
        # session, which need not exist when the decorator runs.
        if self._scalar is None:
            self._scalar = _F.udf(self.func, self._returnType)
        return self._scalar

    @property
    def returnType(self) -> DataType:
        return self.scalar.returnType

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if _direct:
            # Sequential-fold replay: operate on plain Python values directly.
            return self.func(*args, **kwargs)
        if _active is not None:
            # Keyword arguments are bound to their positions first, so that the
            # rewrite only ever deals with a positional argument list.
            return _active.on_call(self, self._bind(args, kwargs))
        return self.scalar(*args, **kwargs)

    def _bind(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Any, ...]:
        if not kwargs:
            return args
        bound = inspect.signature(self.func).bind(*args, **kwargs)
        bound.apply_defaults()
        return tuple(bound.arguments[p] for p in bound.arguments)

    def over_array(self, arrays: Sequence[Column], consts: Sequence[Any], plan: Sequence[int]):
        """Rebuild as an ``array<returnType>`` UDF looping over whole arrays.

        ``plan`` gives, per argument, which array it comes from, or ``-1`` for a
        constant column that does not vary across the loop.
        """
        n = len(arrays)
        plan = list(plan)
        # Bind the plain function: closing over ``self`` would pull the
        # JVM-bound UDF into the closure, which cloudpickle cannot serialize.
        func = self.func

        def mapper(*args: Any) -> Any:
            got, const_vals = args[:n], args[n:]
            if any(a is None for a in got):
                return None
            out = []
            for elems in itertools.zip_longest(*got):
                # A negative plan entry indexes the constants from the end:
                # -1 -> const_vals[-1] is wrong, so map it explicitly.
                out.append(func(*[elems[p] if p >= 0 else const_vals[-p - 1] for p in plan]))
            return out

        mapper.__name__ = f"{self.func.__name__}_over_array"
        base = self.scalar._unwrapped
        array_udf = type(base)(
            mapper,
            returnType=ArrayType(self.returnType),
            evalType=base.evalType,
            deterministic=base.deterministic,
        )._wrapped()
        return array_udf(*arrays, *consts)


def elementwise_udf(returnType: Union[str, DataType] = "string"):
    """Create a Python UDF that also works inside a higher-order function.

    A drop-in for ``pyspark.sql.functions.udf``; ``returnType`` is the *element*
    return type::

        @elementwise_udf("long")
        def plus_one(x):
            return x + 1

    Needs the :data:`functions` proxy from this module for the higher-order
    rewrite. Ordinary use (``plus_one(F.lit(1))``) is unaffected.
    """
    if callable(returnType) and not isinstance(returnType, (str, DataType)):
        raise TypeError(
            "elementwise_udf must be called with a return type, as in "
            "@elementwise_udf('long'), not applied directly to a function"
        )
    return lambda func: _ElementwiseUDF(func, returnType)


def _run(f: Callable, rec: _Recorder, mode: str, slots: Sequence[Column]) -> Any:
    """Run the user's lambda on ``slots``, with UDF calls in the given mode."""
    global _active
    rec.mode, rec.index = mode, 0
    if mode == "record":
        rec.calls, rec.placeholders, rec.alive = [], {}, []
    prev, _active = _active, rec
    try:
        return f(*slots)
    finally:
        _active = prev


def _rewrite(name: str, args: Sequence[Any], fun_pos: int, f: Callable) -> Optional[Column]:
    """Rewrite one higher-order call, or return None to leave it untouched."""
    col_pos = [i for i, a in enumerate(args) if i != fun_pos and isinstance(a, (str, Column))]
    if not col_pos:
        return None
    n_params = len(inspect.signature(f).parameters)

    # A map-valued higher-order function is reduced to the array case: split the
    # map into its key and value arrays, rewrite over those, and rebuild a map.
    if name in _MAP_ZIP:
        return _rewrite_map_zip(name, args, fun_pos, f)
    if name in _MAP_HOFS:
        return _rewrite_map(name, args, fun_pos, f)
    # A comparator receives two elements of the same array, so both parameters
    # are fed the same array rather than one array each.
    if name in _COMPARATORS:
        return _rewrite_comparator(name, args, fun_pos, f)
    # ``aggregate``'s lambda takes (accumulator, element): only the element can
    # be precomputed, so the accumulator slot is excluded from the arrays.
    if name in _FOLDS:
        return _rewrite_fold(name, args, fun_pos, f)

    n_arrays = min(len(col_pos), n_params)
    col_pos = col_pos[:n_arrays]
    cols = [_F.col(args[i]) if isinstance(args[i], str) else args[i] for i in col_pos]
    has_index = n_params > n_arrays

    # Probes stand in for the lambda's parameters during the exploratory run.
    probes = [_F.col(f"__ew_probe_{i}__") for i in range(n_params)]
    rec = _Recorder()
    try:
        _run(f, rec, "record", probes)
    except TypeError:
        raise
    except Exception:
        return None  # Not a lambda this can rewrite.
    if not rec.calls:
        return None  # No element-wise UDF: leave the native call alone.

    index_array = _F.transform(cols[0], lambda _x, i: i)

    def elements_of(call_i: int, pos: int) -> Column:
        """Precompute a UDF argument that is an expression over the element.

        ``plus_one(x * 10)`` needs ``x * 10`` for every element before Python
        runs; a native ``transform`` evaluates it in the JVM. Re-running the
        lambda over the plain elements is what re-derives the expression.
        """
        base_fields = [c.alias(f"c{i}") for i, c in enumerate(cols)]
        if has_index:
            base_fields.append(index_array.alias("ix"))
        base = _F.arrays_zip(*base_fields)

        def one(s: Column) -> Column:
            slots = [s[f"c{i}"] for i in range(n_arrays)]
            if has_index:
                slots.append(s["ix"])
            _run(f, rec, "record", slots)
            return rec.calls[call_i][1][pos]

        return _F.transform(base, one)

    # Apply each UDF to whole arrays, outside the lambda. Calls are handled in
    # recording order, which is inside-out, so a nested call's array result is
    # already available when the call that consumes it is reached.
    recorded = list(rec.calls)
    mapped: List[Column] = []
    for call_i, (udf, call_args) in enumerate(recorded):
        arrays: List[Column] = []
        consts: List[Any] = []
        plan: List[int] = []
        for pos, arg in enumerate(call_args):
            inner = rec.placeholders.get(id(arg)) if isinstance(arg, Column) else None
            slot = next((j for j, probe in enumerate(probes) if arg is probe), None)
            if inner is not None:
                # Another element-wise UDF's result, e.g. plus_one(times_ten(x)).
                # Both are already array UDFs here, so they simply compose: the
                # inner array becomes the outer one's input.
                plan.append(len(arrays))
                arrays.append(mapped[inner])
            elif slot is not None:
                # The lambda variable itself, passed straight through.
                plan.append(len(arrays))
                arrays.append(cols[slot] if slot < n_arrays else index_array)
            elif isinstance(arg, Column) and _over_probe(arg, probes):
                # An expression over the lambda variable, e.g. plus_one(x * 10):
                # evaluate it per element in the JVM first.
                plan.append(len(arrays))
                arrays.append(elements_of(call_i, pos))
            else:
                # A constant with respect to the loop, e.g. a captured column.
                plan.append(-1 - len(consts))
                consts.append(arg)
        mapped.append(udf.over_array(arrays, consts, plan))

    # Carry the elements, the index and every precomputed UDF result together.
    fields = [c.alias(f"c{i}") for i, c in enumerate(cols)]
    if has_index:
        fields.append(index_array.alias("ix"))
    fields += [c.alias(f"u{i}") for i, c in enumerate(mapped)]
    carrier = _F.arrays_zip(*fields)

    def rewritten(s: Column) -> Column:
        slots = [s[f"c{i}"] for i in range(n_arrays)]
        if has_index:
            slots.append(s["ix"])
        rec.substitutions = [s[f"u{i}"] for i in range(len(mapped))]
        return _run(f, rec, "substitute", slots)

    # Rebuild the call with the carrier in place of the iterated arrays. The
    # carrier already zips them into one, so a multi-array function collapses to
    # its single-array form (zip_with(l, r, g) -> transform(zip(l, r), g')).
    if n_arrays == 1:
        new_args = list(args)
        new_args[fun_pos] = rewritten
        new_args[col_pos[0]] = carrier
    else:
        keep = [a for i, a in enumerate(args) if i != fun_pos and i not in col_pos]
        new_args = [carrier, *keep, rewritten]
        name = "transform"
    out = getattr(_F, name)(*new_args)
    if name in _RETURNS_INPUT_ELEMENTS:
        # The result is made of input elements, so recover them afterwards.
        return _F.transform(out, lambda s: s["c0"])
    return out


def _udf_arrays(
    f: Callable, probes: Sequence[Column], sources: Sequence[Column]
) -> Optional[Tuple[_Recorder, List[Column]]]:
    """Run ``f`` on ``probes`` and apply each UDF it calls to whole arrays.

    ``sources[i]`` is the array supplying probe ``i``. Returns the recorder and
    one ``array<T>`` column per recorded call, or None if there is nothing to
    rewrite. This is the shared core of every rewrite below.
    """
    rec = _Recorder()
    try:
        _run(f, rec, "record", probes)
    except TypeError:
        raise
    except Exception:
        return None
    if not rec.calls:
        return None

    mapped: List[Column] = []
    for udf, call_args in list(rec.calls):
        arrays: List[Column] = []
        consts: List[Any] = []
        plan: List[int] = []
        for arg in call_args:
            inner = rec.placeholders.get(id(arg)) if isinstance(arg, Column) else None
            slot = next((j for j, p in enumerate(probes) if arg is p), None)
            if inner is not None:
                plan.append(len(arrays))
                arrays.append(mapped[inner])  # Nested: compose the array UDFs.
            elif slot is not None:
                plan.append(len(arrays))
                arrays.append(sources[slot])
            else:
                plan.append(-1 - len(consts))
                consts.append(arg)
        mapped.append(udf.over_array(arrays, consts, plan))
    return rec, mapped


def _rewrite_comparator(
    name: str, args: Sequence[Any], fun_pos: int, f: Callable
) -> Optional[Column]:
    """Rewrite ``array_sort(col, lambda a, b: ...)`` using a Python UDF.

    A comparator sees two elements at once, so its Python part cannot be run
    pairwise -- there are O(n log n) comparisons and no array to precompute them
    over. What can be precomputed is the UDF applied to *each element*, giving a
    sort key. Both comparator parameters are therefore fed the same array, and
    each element travels with its keys, so the JVM does the comparing::

        array_sort(zip(col, key(col)), lambda a, b: cmp(a.u0, b.u0))

    A comparator that passes both of its parameters to one UDF call -- a genuinely
    pairwise decision -- has no such key and is refused.
    """
    col_pos = [i for i, a in enumerate(args) if i != fun_pos and isinstance(a, (str, Column))]
    col = _F.col(args[col_pos[0]]) if isinstance(args[col_pos[0]], str) else args[col_pos[0]]
    # Both parameters range over the same array.
    probes = [_F.col(f"__ew_cmp_{i}__") for i in range(2)]
    got = _udf_arrays(f, probes, [col, col])
    if got is None:
        return None
    rec, mapped = got
    # Which comparator parameter each call read: 0 for the left element, 1 for
    # the right. A call reading both is a genuinely pairwise decision.
    sides: List[int] = []
    for udf, call_args in rec.calls:
        seen = {j for arg in call_args for j, p in enumerate(probes) if arg is p}
        seen |= {
            j
            for arg in call_args
            if isinstance(arg, Column)
            for j, p in enumerate(probes)
            if _over_probe(arg, [p])
        }
        if len(seen) > 1:
            # A genuinely pairwise comparison: no per-element key exists, so fall
            # back to doing the whole sort inside Python (see _rewrite_pairwise).
            return _rewrite_pairwise(name, col, f, udf)
        sides.append(next(iter(seen)) if seen else 0)
    # Each element carries its precomputed keys, so a and b both see them.
    fields = [col.alias("c0")] + [c.alias(f"u{i}") for i, c in enumerate(mapped)]
    carrier = _F.arrays_zip(*fields)

    def cmp(a: Column, b: Column) -> Column:
        # Each key is read from the struct of the side whose element produced it,
        # so the comparator the JVM runs compares precomputed keys and never
        # calls Python.
        rec.substitutions = [(a if sides[i] == 0 else b)[f"u{i}"] for i in range(len(mapped))]
        return _run(f, rec, "substitute", [a["c0"], b["c0"]])

    sorted_ = getattr(_F, name)(carrier, cmp)
    return _F.transform(sorted_, lambda s: s["c0"])


def _rewrite_pairwise(name: str, col: Column, f: Callable, udf: "_ElementwiseUDF") -> Column:
    """Sort by a genuinely pairwise Python comparator.

    When the comparator hands both elements to one UDF call there is no
    per-element key to precompute, so the comparison cannot be lifted out. What
    can be lifted is the *whole sort*: one UDF call per row receives the entire
    array, sorts it in Python with the user's comparator, and returns each
    element's rank. Native ``array_sort`` then orders by that rank.

    The comparator therefore runs O(n log n) times per row inside a single Python
    call, rather than once per element -- the cost the warning below reports.
    """
    warnings.warn(
        f"{name}: the comparator passes both elements to {udf.func.__name__!r}, "
        "so it is a pairwise comparison. The whole sort is performed inside one "
        "Python call per row, calling the UDF O(n log n) times for an array of "
        "length n, which is much slower than a per-element sort key. Consider "
        f"rewriting it as a key, e.g. lambda a, b: "
        f"when({udf.func.__name__}(a) < {udf.func.__name__}(b), -1)...",
        RuntimeWarning,
        stacklevel=4,
    )
    compare = udf.func

    def rank(values: Any) -> Any:
        if values is None:
            return None
        order = sorted(
            range(len(values)),
            key=functools.cmp_to_key(lambda i, j: compare(values[i], values[j])),
        )
        ranks = [0] * len(values)
        for position, i in enumerate(order):
            ranks[i] = position
        return ranks

    rank.__name__ = f"{compare.__name__}_ranks"
    base = udf.scalar._unwrapped
    rank_udf = type(base)(
        rank,
        returnType=ArrayType(IntegerType()),
        evalType=base.evalType,
        deterministic=base.deterministic,
    )._wrapped()
    carrier = _F.arrays_zip(col.alias("c0"), rank_udf(col).alias("rk"))
    ordered = getattr(_F, name)(
        carrier,
        lambda a, b: _F.when(a["rk"] < b["rk"], -1).when(a["rk"] > b["rk"], 1).otherwise(0),
    )
    return _F.transform(ordered, lambda s: s["c0"])


def _rewrite_fold(name: str, args: Sequence[Any], fun_pos: int, f: Callable) -> Optional[Column]:
    """Rewrite ``aggregate(col, init, lambda acc, x: ...)`` using a Python UDF.

    The accumulator is produced by the fold itself, so a UDF applied to it cannot
    be precomputed and is refused. A UDF applied to the *element* is fine: it is
    precomputed over the whole array, and the fold then runs over the zipped
    elements.
    """
    col_pos = [i for i, a in enumerate(args) if i != fun_pos and isinstance(a, (str, Column))]
    col = _F.col(args[col_pos[0]]) if isinstance(args[col_pos[0]], str) else args[col_pos[0]]
    acc_probe = _F.col("__ew_acc__")
    elem_probe = _F.col("__ew_elem__")
    # Probe order must match the lambda's own (accumulator, element). The
    # accumulator has no array behind it, so its source is never used: any UDF
    # call touching it is rejected below.
    got = _udf_arrays(f, [acc_probe, elem_probe], [col, col])
    if got is None:
        return None
    rec, mapped = got
    for udf, call_args in rec.calls:
        if any(
            arg is acc_probe or (isinstance(arg, Column) and _over_probe(arg, [acc_probe]))
            for arg in call_args
        ):
            # The accumulator only exists mid-fold, so it cannot be precomputed
            # as an array. Run the whole fold in Python instead.
            return _rewrite_fold_in_python(name, args, fun_pos, f, col, col_pos[0], udf)
    fields = [col.alias("c0")] + [c.alias(f"u{i}") for i, c in enumerate(mapped)]
    carrier = _F.arrays_zip(*fields)

    def merge(acc: Column, s: Column) -> Column:
        rec.substitutions = [s[f"u{i}"] for i in range(len(mapped))]
        return _run(f, rec, "substitute", [acc, s["c0"]])

    new_args = list(args)
    new_args[col_pos[0]] = carrier
    new_args[fun_pos] = lambda acc, s: merge(acc, s)
    return getattr(_F, name)(*new_args)


def _rewrite_fold_in_python(
    name: str,
    args: Sequence[Any],
    fun_pos: int,
    f: Callable,
    col: Column,
    col_arg: int,
    udf: "_ElementwiseUDF",
) -> Column:
    """Fold entirely in Python, for a UDF applied to the accumulator.

    ``aggregate(v, 0, lambda acc, x: plus_one(acc) + x)`` cannot precompute
    ``plus_one(acc)``: the accumulator only exists between steps. Instead the
    whole fold moves into one UDF call per row, which walks the array and applies
    the merge step in Python, so the UDF runs once per element *sequentially*::

        acc = init
        for x in values:
            acc = merge(acc, x)     # plus_one(acc) + x, in Python

    The lambda is called on plain Python values, with UDF calls executing
    immediately rather than recording themselves. Only merge steps Python can
    evaluate on its own work this way: a lambda needing Spark expressions
    (``F.when``, column methods) raises inside the UDF rather than being silently
    mis-evaluated. A ``finish`` lambda, if given, is applied to the final
    accumulator in the same call.
    """
    warnings.warn(
        f"{name}: {udf.func.__name__!r} is applied to the accumulator, whose "
        "value only exists between fold steps, so the entire fold runs inside "
        "one Python call per row. The UDF is invoked once per element "
        "sequentially and the JVM does no folding at all. Consider applying the "
        "UDF to the element instead, which is precomputed for the whole array.",
        RuntimeWarning,
        stacklevel=4,
    )
    init = args[1] if len(args) > 1 and not callable(args[1]) else _F.lit(None)
    finish = next(
        (a for i, a in enumerate(args) if i != fun_pos and callable(a) and i > fun_pos), None
    )

    # The merge step is reduced to plain Python functions here on the driver, so
    # the closure shipped to the worker holds no Spark objects (a UDF wrapper
    # carries a JVM handle that cloudpickle cannot serialize).
    plain = _plain_lambda(f)
    plain_finish = _plain_lambda(finish) if finish is not None else None

    def merge_in_python(values: Any, initial: Any) -> Any:
        if values is None:
            return None
        acc = initial
        for x in values:
            acc = plain(acc, x)
        return plain_finish(acc) if plain_finish is not None else acc

    merge_in_python.__name__ = f"{name}_in_python"
    base = udf.scalar._unwrapped
    fold_udf = type(base)(
        merge_in_python,
        # The accumulator, and so the result, has the initial value's type.
        returnType=udf.returnType,
        evalType=base.evalType,
        deterministic=base.deterministic,
    )._wrapped()
    return fold_udf(col, init)


def _plain_lambda(f: Callable) -> Callable:
    """Turn a lambda over element-wise UDFs into one over plain Python values.

    Each ``_ElementwiseUDF`` the lambda closes over is swapped for its underlying
    function, so the result can be pickled to a worker and called on real values.
    A lambda whose body needs Spark expressions will raise there, which surfaces
    as a Python worker error rather than a wrong answer.
    """
    closure = f.__closure__ or ()
    names = f.__code__.co_freevars
    cells = []
    for name, cell in zip(names, closure):
        try:
            value = cell.cell_contents
        except ValueError:
            value = None
        if isinstance(value, _ElementwiseUDF):
            value = value.func
        cells.append(types.CellType(value))
    globs = {k: (v.func if isinstance(v, _ElementwiseUDF) else v) for k, v in f.__globals__.items()}
    rebuilt = types.FunctionType(
        f.__code__, globs, f.__name__, f.__defaults__, tuple(cells) or None
    )
    return rebuilt


def _rewrite_map_zip(name: str, args: Sequence[Any], fun_pos: int, f: Callable) -> Optional[Column]:
    """Rewrite ``map_zip_with(m1, m2, lambda k, v1, v2: ...)`` using a Python UDF.

    ``map_zip_with`` visits the union of both maps' keys, so that union is built
    explicitly and each map is looked up per key -- giving three aligned arrays
    that the array machinery then handles. Keys missing from one map look up as
    null, matching ``map_zip_with``'s own behaviour.
    """
    col_pos = [i for i, a in enumerate(args) if i != fun_pos and isinstance(a, (str, Column))]
    if len(col_pos) < 2:
        return None
    m1, m2 = [_F.col(args[i]) if isinstance(args[i], str) else args[i] for i in col_pos[:2]]
    keys = _F.array_union(_F.map_keys(m1), _F.map_keys(m2))
    v1 = _F.transform(keys, lambda k: _F.element_at(m1, k))
    v2 = _F.transform(keys, lambda k: _F.element_at(m2, k))
    probes = [_F.col("__ew_k__"), _F.col("__ew_v1__"), _F.col("__ew_v2__")]
    got = _udf_arrays(f, probes, [keys, v1, v2])
    if got is None:
        return None
    rec, mapped = got
    fields = [keys.alias("k"), v1.alias("v1"), v2.alias("v2")]
    fields += [c.alias(f"u{i}") for i, c in enumerate(mapped)]
    carrier = _F.arrays_zip(*fields)

    def body(s: Column) -> Column:
        rec.substitutions = [s[f"u{i}"] for i in range(len(mapped))]
        return _run(f, rec, "substitute", [s["k"], s["v1"], s["v2"]])

    return _F.map_from_arrays(keys, _F.transform(carrier, body))


def _rewrite_map(name: str, args: Sequence[Any], fun_pos: int, f: Callable) -> Optional[Column]:
    """Rewrite a map-valued higher-order function using a Python UDF.

    Maps are turned into their key and value arrays, the UDF is applied to those,
    and a map is rebuilt -- so the array machinery covers ``transform_keys``,
    ``transform_values`` and ``map_filter`` without special cases beyond which
    part of the result to keep.
    """
    col_pos = [i for i, a in enumerate(args) if i != fun_pos and isinstance(a, (str, Column))]
    src = _F.col(args[col_pos[0]]) if isinstance(args[col_pos[0]], str) else args[col_pos[0]]
    keys, vals = _F.map_keys(src), _F.map_values(src)
    kp, vp = _F.col("__ew_key__"), _F.col("__ew_val__")
    got = _udf_arrays(f, [kp, vp], [keys, vals])
    if got is None:
        return None
    rec, mapped = got
    fields = [keys.alias("k"), vals.alias("v")]
    fields += [c.alias(f"u{i}") for i, c in enumerate(mapped)]
    carrier = _F.arrays_zip(*fields)

    def body(s: Column) -> Column:
        rec.substitutions = [s[f"u{i}"] for i in range(len(mapped))]
        return _run(f, rec, "substitute", [s["k"], s["v"]])

    if name == "map_filter":
        kept = _F.filter(carrier, body)
        return _F.map_from_arrays(
            _F.transform(kept, lambda s: s["k"]), _F.transform(kept, lambda s: s["v"])
        )
    computed = _F.transform(carrier, body)
    if name == "transform_keys":
        return _F.map_from_arrays(computed, vals)
    return _F.map_from_arrays(keys, computed)  # transform_values


def _over_probe(col: Column, probes: Sequence[Column]) -> bool:
    """Whether ``col`` is an expression built over a probe rather than being one.

    Compares rendered expressions, which both the classic and the Connect Column
    provide. Only decides whether an argument needs precomputing, so
    over-reporting is harmless.
    """
    text = repr(col)
    return any(repr(p)[8:-2] in text for p in probes)


def _uses_elementwise(f: Any) -> bool:
    """Cheap check for whether ``f`` could reach an element-wise UDF at all.

    Lets an ordinary native lambda skip the exploratory run. Over-reporting is
    safe: the run simply happens and may still decline to rewrite.
    """
    if isinstance(f, _ElementwiseUDF):
        return True
    code = getattr(f, "__code__", None)
    if code is None:
        return True  # A callable object: let the exploratory run decide.
    for cell in f.__closure__ or ():
        try:
            if isinstance(cell.cell_contents, _ElementwiseUDF):
                return True
        except ValueError:
            return True  # Empty cell, e.g. a recursive definition.
    return _reaches(code, getattr(f, "__globals__", {}))


def _reaches(code: Any, globs: Dict[str, Any]) -> bool:
    if any(isinstance(globs.get(n), _ElementwiseUDF) for n in code.co_names):
        return True
    # Nested lambdas and comprehensions carry their own code objects.
    return any(isinstance(c, type(code)) and _reaches(c, globs) for c in code.co_consts)


def _as_lambda(f: Any) -> Callable:
    """A bare UDF argument acts as ``lambda x: udf(x)``."""
    if isinstance(f, _ElementwiseUDF):
        n = len(inspect.signature(f.func).parameters)
        return lambda *xs: f(*xs[:n])
    return f


class _Functions:
    """Drop-in proxy for ``pyspark.sql.functions``."""

    def __getattr__(self, name: str) -> Any:
        attr = getattr(_F, name)
        if not callable(attr) or name.startswith("_"):
            return attr

        @functools.wraps(attr)
        def maybe_hof(*args: Any, **kwargs: Any) -> Any:
            lambdas = [
                i
                for i, a in enumerate(args)
                if isinstance(a, _ElementwiseUDF)
                or (callable(a) and not isinstance(a, (Column, str)))
            ]
            using = [i for i in lambdas if _uses_elementwise(args[i])]
            if using and not kwargs:
                # A fold legitimately takes a second (finish) lambda, which the
                # fold rewrites handle; anywhere else two lambdas have no
                # element-wise reading.
                if len(lambdas) > 1 and name not in _FOLDS:
                    raise TypeError(
                        f"{name} takes more than one lambda; only one can use an "
                        f"element-wise UDF. Apply the UDF to the array first, "
                        f"then call {name} on the result."
                    )
                out = _rewrite(name, args, using[0], _as_lambda(args[using[0]]))
                if out is not None:
                    return out
            return attr(*args, **kwargs)

        setattr(self, name, maybe_hof)  # Cached: later lookups reuse it.
        return maybe_hof

    def __dir__(self) -> List[str]:
        return dir(_F)


functions = _Functions()
