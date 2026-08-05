"""Complex queries: element-wise UDFs mixed with everything else.

Patterned after Apache Spark's own ``python/pyspark/sql/tests/test_udf.py``,
which exercises UDFs in join conditions, aggregates, generators, subqueries and
window functions rather than in isolation. The same treatment matters more here,
because a higher-order call is *rewritten* into a different expression tree: it
has to keep behaving like an ordinary column wherever one is allowed.
"""

import pyspark.sql.functions as real
import pytest
from pyspark.sql import Window

from elementwise_udf import udf as eudf, functions as sf


@eudf("long")
def plus_one(x):
    return x + 1


@eudf("long")
def times_ten(x):
    return x * 10


@eudf("boolean")
def is_odd(x):
    return x % 2 == 1


@eudf("string")
def name_of(x):
    return f"n{x}"


def plain_udf(column, fn=lambda x: x * 2, ret="long"):
    """An ordinary PySpark UDF, built lazily (a DDL type needs a session)."""
    return real.udf(fn, ret)(column)


def rows(df):
    return [tuple(r) for r in df.collect()]


def col(df, column):
    return [r[0] for r in df.select(column).collect()]


# --------------------------------------------------------------------------
# Mixed with Spark's built-in functions.
# --------------------------------------------------------------------------


def test_builtin_wrapping_the_rewritten_array(spark):
    df = spark.createDataFrame([([1, 2, 3],)], "v array<int>")
    mapped = sf.transform("v", lambda x: plus_one(x))
    out = df.select(
        real.array_max(mapped).alias("largest"),
        real.array_min(mapped).alias("smallest"),
        real.size(mapped).alias("n"),
        real.array_contains(mapped, 3).alias("has_three"),
        real.sort_array(mapped, asc=False).alias("descending"),
    )
    assert rows(out) == [(4, 2, 3, True, [4, 3, 2])]


def test_builtin_feeding_the_higher_order_call(spark):
    df = spark.createDataFrame([([3, 1, 2],)], "v array<int>")
    # A built-in produces the array the rewrite then iterates.
    out = df.select(sf.transform(real.sort_array("v"), lambda x: plus_one(x)).alias("m"))
    assert rows(out) == [([2, 3, 4],)]


def test_builtin_between_two_rewrites(spark):
    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    first = sf.transform("v", lambda x: plus_one(x))
    reversed_ = real.reverse(first)
    second = sf.transform(reversed_, lambda x: times_ten(x))
    assert col(df, second) == [[30, 20]]


def test_concat_of_two_rewritten_arrays(spark):
    df = spark.createDataFrame([([1],)], "v array<int>")
    out = df.select(
        real.concat(
            sf.transform("v", lambda x: plus_one(x)),
            sf.transform("v", lambda x: times_ten(x)),
        ).alias("both")
    )
    assert rows(out) == [([2, 10],)]


def test_explode_of_a_rewritten_array(spark):
    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    out = df.select(real.explode(sf.transform("v", lambda x: plus_one(x))).alias("e"))
    assert rows(out) == [(2,), (3,)]


def test_rewrite_inside_a_struct_and_back_out(spark):
    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    packed = real.struct(sf.transform("v", lambda x: plus_one(x)).alias("m"))
    assert col(df, packed["m"]) == [[2, 3]]


def test_aggregate_over_a_rewritten_array_with_a_builtin(spark):
    df = spark.createDataFrame([([1, 2, 3],)], "v array<int>")
    # Native aggregate folding the array a Python UDF produced.
    mapped = sf.transform("v", lambda x: plus_one(x))
    out = df.select(sf.aggregate(mapped, real.lit(0).cast("long"), lambda a, x: a + x).alias("s"))
    assert rows(out) == [(9,)]


# --------------------------------------------------------------------------
# Mixed with ordinary Python UDFs (both kinds in one plan).
# --------------------------------------------------------------------------


def test_plain_udf_and_rewrite_in_one_projection(spark):
    df = spark.createDataFrame([(3, [1, 2])], "n int, v array<int>")
    out = df.select(
        plain_udf("n").alias("plain"),
        sf.transform("v", lambda x: plus_one(x)).alias("rewritten"),
    )
    assert rows(out) == [(6, [2, 3])]


def test_plain_udf_wrapping_a_rewrite(spark):
    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    mapped = sf.transform("v", lambda x: plus_one(x))
    out = df.select(plain_udf(real.array_max(mapped)).alias("doubled_max"))
    assert rows(out) == [(6,)]


def test_rewrite_over_an_array_a_plain_udf_produced(spark):
    df = spark.createDataFrame([(2,)], "n int")
    produced = real.udf(lambda n: list(range(n)), "array<int>")("n")
    assert col(df, sf.transform(produced, lambda x: plus_one(x))) == [[1, 2]]


def test_element_wise_udf_used_plainly_and_in_a_lambda_together(spark):
    df = spark.createDataFrame([(5, [1, 2])], "n int, v array<int>")
    out = df.select(
        plus_one("n").alias("scalar"),
        sf.transform("v", lambda x: plus_one(x)).alias("array"),
    )
    assert rows(out) == [(6, [2, 3])]


def test_chained_element_wise_udfs_outside_any_lambda(spark):
    df = spark.createDataFrame([(1,)], "n int")
    # Chaining is ordinary UDF composition when no higher-order function is
    # involved; the recorder must stay out of the way entirely.
    assert col(df, plus_one(times_ten(plus_one("n")))) == [21]


def test_many_udfs_of_both_kinds(spark):
    df = spark.createDataFrame([(1, [1, 2])], "n int, v array<int>")
    out = df.select(
        plain_udf("n").alias("a"),
        plus_one("n").alias("b"),
        sf.transform("v", lambda x: plus_one(x)).alias("c"),
        sf.transform("v", lambda x: times_ten(x)).alias("d"),
        sf.filter("v", lambda x: is_odd(x)).alias("e"),
        sf.transform("v", lambda x: name_of(x)).alias("f"),
    )
    assert rows(out) == [(2, 2, [2, 3], [10, 20], [1], ["n1", "n2"])]


# --------------------------------------------------------------------------
# Joins, following test_udf.py's join coverage.
# --------------------------------------------------------------------------


def test_rewrite_in_filter_on_top_of_a_join(spark):
    left = spark.createDataFrame([(1, [1, 2]), (2, [8])], "id int, v array<int>")
    right = spark.createDataFrame([(1,), (2,)], "id int")
    joined = left.join(right, "id")
    out = joined.where(real.array_max(sf.transform("v", lambda x: plus_one(x))) > 5)
    assert col(out.select("id"), "id") == [2]


def test_rewrite_in_filter_on_top_of_an_outer_join(spark):
    left = spark.createDataFrame([(1, [1]), (2, [8])], "id int, v array<int>")
    right = spark.createDataFrame([(1,)], "id int")
    joined = left.join(right, "id", "left")
    out = joined.where(real.array_max(sf.transform("v", lambda x: plus_one(x))) > 5)
    assert col(out.select("id"), "id") == [2]


def test_rewrite_as_the_whole_join_condition(spark):
    left = spark.createDataFrame([([1, 2],)], "v array<int>")
    right = spark.createDataFrame([([2, 3],)], "w array<int>")
    condition = sf.transform("v", lambda x: plus_one(x)) == real.col("w")
    out = left.join(right, condition)
    assert len(rows(out)) == 1


def test_rewrite_and_a_common_filter_in_a_join_condition(spark):
    left = spark.createDataFrame([(1, [1]), (2, [5])], "id int, v array<int>")
    right = spark.createDataFrame([(1, 2), (2, 99)], "id int, k int")
    mapped = sf.transform("v", lambda x: plus_one(x))
    out = left.join(right, (left["id"] == right["id"]) & (real.element_at(mapped, 1) == right["k"]))
    assert col(out.select(left["id"]), left["id"]) == [1]


def test_self_join_with_rewrites_on_both_sides(spark):
    df = spark.createDataFrame([(1, [1]), (2, [2])], "id int, v array<int>")
    left = df.select("id", sf.transform("v", lambda x: plus_one(x)).alias("m"))
    right = df.select(
        real.col("id").alias("rid"), sf.transform("v", lambda x: times_ten(x)).alias("t")
    )
    out = left.join(right, left["id"] == right["rid"]).orderBy("id")
    assert rows(out) == [(1, [2], 1, [10]), (2, [3], 2, [20])]


def test_cross_join_with_a_rewrite(spark):
    left = spark.createDataFrame([([1],), ([2],)], "v array<int>")
    right = spark.createDataFrame([("x",), ("y",)], "t string")
    out = left.crossJoin(right).select(sf.transform("v", lambda x: plus_one(x)).alias("m"), "t")
    assert sorted(rows(out)) == [([2], "x"), ([2], "y"), ([3], "x"), ([3], "y")]


# --------------------------------------------------------------------------
# Aggregates, windows, subqueries, generators.
# --------------------------------------------------------------------------


def test_rewrite_with_an_aggregate_function(spark):
    df = spark.createDataFrame([("a", [1]), ("a", [2]), ("b", [3])], "k string, v array<int>")
    mapped = sf.transform("v", lambda x: plus_one(x))
    out = (
        df.groupBy("k")
        .agg(
            real.sum(real.element_at(mapped, 1)).alias("total"),
            real.collect_list(real.element_at(mapped, 1)).alias("all"),
        )
        .orderBy("k")
    )
    assert rows(out) == [("a", 5, [2, 3]), ("b", 4, [4])]


def test_rewrite_in_a_having_style_filter_after_grouping(spark):
    df = spark.createDataFrame([("a", [1]), ("b", [9])], "k string, v array<int>")
    mapped = sf.transform("v", lambda x: plus_one(x))
    out = df.groupBy("k").agg(real.max(real.element_at(mapped, 1)).alias("m")).where("m > 5")
    assert rows(out) == [("b", 10)]


def test_rewrite_in_a_window_frame(spark):
    df = spark.createDataFrame([("a", [1]), ("a", [2]), ("a", [3])], "k string, v array<int>")
    mapped = sf.transform("v", lambda x: plus_one(x))
    window = Window.partitionBy("k").orderBy(real.element_at(mapped, 1))
    out = df.select(
        real.element_at(mapped, 1).alias("m"),
        real.sum(real.element_at(mapped, 1)).over(window).alias("running"),
    ).orderBy("m")
    assert rows(out) == [(2, 2), (3, 5), (4, 9)]


def test_rewrite_in_a_subquery(spark):
    df = spark.createDataFrame([([1, 2],), ([9],)], "v array<int>")
    inner = df.select(sf.transform("v", lambda x: plus_one(x)).alias("m"))
    inner.createOrReplaceTempView("elementwise_inner")
    try:
        out = spark.sql(
            "SELECT m FROM elementwise_inner WHERE element_at(m, 1) > 5 ORDER BY element_at(m, 1)"
        )
        assert rows(out) == [([10],)]
    finally:
        spark.catalog.dropTempView("elementwise_inner")


def test_rewrite_in_generate_posexplode(spark):
    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    mapped = sf.transform("v", lambda x: plus_one(x))
    out = df.select(real.posexplode(mapped).alias("pos", "value"))
    assert rows(out) == [(0, 2), (1, 3)]


def test_rewrite_with_order_by_and_limit(spark):
    df = spark.createDataFrame([([3],), ([1],), ([2],)], "v array<int>")
    mapped = sf.transform("v", lambda x: plus_one(x))
    out = df.orderBy(real.element_at(mapped, 1).desc()).limit(2).select(mapped.alias("m"))
    assert rows(out) == [([4],), ([3],)]


def test_rewrite_survives_a_union_then_group_by(spark):
    left = spark.createDataFrame([("a", [1])], "k string, v array<int>")
    right = spark.createDataFrame([("a", [2])], "k string, v array<int>")
    mapped = sf.transform("v", lambda x: plus_one(x))
    both = left.select("k", mapped.alias("m")).union(right.select("k", mapped.alias("m")))
    out = both.groupBy("k").agg(real.count("*").alias("n"))
    assert rows(out) == [("a", 2)]


def test_repeated_argument_to_one_udf(spark):
    @eudf("long")
    def add(a, b):
        return a + b

    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    # The same lambda variable in both positions.
    assert col(df, sf.transform("v", lambda x: add(x, x))) == [[2, 4]]


def test_udf_without_arguments_inside_a_lambda(spark):
    @eudf("long")
    def constant():
        return 7

    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    assert col(df, sf.transform("v", lambda x: constant())) == [[7, 7]]


def test_many_arguments(spark):
    @eudf("long")
    def total(a, b, c, d, e):
        return a + b + c + d + e

    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    got = col(
        df,
        sf.transform("v", lambda x: total(x, x, real.lit(1), real.lit(2), real.lit(3))),
    )
    assert got == [[8, 10]]


def test_stop_iteration_inside_a_udf_surfaces(spark):
    @eudf("long")
    def raises_stop(x):
        raise StopIteration("stop")

    df = spark.createDataFrame([([1],)], "v array<int>")
    with pytest.raises(Exception):
        df.select(sf.transform("v", lambda x: raises_stop(x))).collect()
