"""Element-wise UDFs in the middle of real queries.

The rewrite replaces one column expression, so it has to survive everything a
query does around that column: joins, grouping, window functions, caching,
writing, and mixing with ordinary PySpark UDFs.
"""

import pyspark.sql.functions as real
from pyspark.sql import Window

from elementwise_udf import udf, functions as F


@udf("long")
def plus_one(x):
    return x + 1


@udf("string")
def label(x):
    return f"n{x}"


def regular_double(column):
    """A plain PySpark UDF, to be mixed with the element-wise ones.

    Built on call rather than at import: ``pyspark.sql.functions.udf`` resolves
    its DDL return type against the active session, which does not exist while
    this module is being collected.
    """
    return real.udf(lambda x: x * 2, "long")(column)


def rows(df):
    return [tuple(r) for r in df.collect()]


# --------------------------------------------------------------------------
# Mixed with ordinary PySpark UDFs.
# --------------------------------------------------------------------------


def test_regular_udf_on_a_scalar_column_beside_a_higher_order_call(spark):
    df = spark.createDataFrame([(2, [1, 2])], "n int, v array<int>")
    out = df.select(
        regular_double("n").alias("doubled"),
        F.transform("v", lambda x: plus_one(x)).alias("mapped"),
    )
    assert rows(out) == [(4, [2, 3])]


def test_regular_udf_applied_to_the_rewritten_array(spark):
    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    mapped = F.transform("v", lambda x: plus_one(x))
    # A plain UDF over the *result* array: two Python UDFs in one expression.
    out = df.select(regular_double(real.element_at(mapped, 1)).alias("first_doubled"))
    assert rows(out) == [(4,)]


def test_element_wise_udf_used_as_a_plain_udf_alongside_a_rewrite(spark):
    df = spark.createDataFrame([(3, [1, 2])], "n int, v array<int>")
    out = df.select(
        plus_one("n").alias("scalar"),
        F.transform("v", lambda x: plus_one(x)).alias("array"),
    )
    assert rows(out) == [(4, [2, 3])]


def test_three_python_udfs_in_one_projection(spark):
    df = spark.createDataFrame([(1, [1, 2])], "n int, v array<int>")
    out = df.select(
        regular_double("n").alias("a"),
        F.transform("v", lambda x: plus_one(x)).alias("b"),
        F.transform("v", lambda x: label(x)).alias("c"),
    )
    assert rows(out) == [(2, [2, 3], ["n1", "n2"])]


# --------------------------------------------------------------------------
# Joins.
# --------------------------------------------------------------------------


def test_rewrite_survives_an_inner_join(spark):
    left = spark.createDataFrame([(1, [1, 2]), (2, [3])], "id int, v array<int>")
    right = spark.createDataFrame([(1, "x"), (2, "y")], "id int, tag string")
    out = (
        left.join(right, "id")
        .select("id", "tag", F.transform("v", lambda x: plus_one(x)).alias("mapped"))
        .orderBy("id")
    )
    assert rows(out) == [(1, "x", [2, 3]), (2, "y", [4])]


def test_rewrite_on_both_sides_of_a_join(spark):
    left = spark.createDataFrame([(1, [1])], "id int, v array<int>")
    right = spark.createDataFrame([(1, [10])], "id int, w array<int>")
    out = left.join(right, "id").select(
        F.transform("v", lambda x: plus_one(x)).alias("a"),
        F.transform("w", lambda x: plus_one(x)).alias("b"),
    )
    assert rows(out) == [([2], [11])]


def test_rewrite_in_a_join_condition(spark):
    left = spark.createDataFrame([(1, [1, 2])], "id int, v array<int>")
    right = spark.createDataFrame([(2,), (9,)], "k int")
    # The join key is the first element of the rewritten array.
    mapped = F.transform("v", lambda x: plus_one(x))
    out = left.join(right, real.element_at(mapped, 1) == right["k"]).select("id", "k")
    assert rows(out) == [(1, 2)]


def test_left_outer_join_keeps_null_arrays_null(spark):
    left = spark.createDataFrame([(1, [1]), (2, None)], "id int, v array<int>")
    right = spark.createDataFrame([(1, "x")], "id int, tag string")
    out = (
        left.join(right, "id", "left")
        .select("id", F.transform("v", lambda x: plus_one(x)).alias("mapped"))
        .orderBy("id")
    )
    assert rows(out) == [(1, [2]), (2, None)]


# --------------------------------------------------------------------------
# Grouping, windows, ordering, set operations.
# --------------------------------------------------------------------------


def test_rewrite_inside_a_group_by_aggregation(spark):
    df = spark.createDataFrame([("a", [1, 2]), ("a", [3]), ("b", [4])], "k string, v array<int>")
    mapped = F.transform("v", lambda x: plus_one(x))
    out = df.groupBy("k").agg(real.sum(real.element_at(mapped, 1)).alias("total")).orderBy("k")
    assert rows(out) == [("a", 6), ("b", 5)]


def test_rewrite_as_a_group_by_key(spark):
    df = spark.createDataFrame([([1],), ([1],), ([2],)], "v array<int>")
    mapped = F.transform("v", lambda x: plus_one(x))
    out = df.groupBy(mapped.alias("k")).count().orderBy("k")
    assert rows(out) == [([2], 2), ([3], 1)]


def test_rewrite_with_a_window_function(spark):
    df = spark.createDataFrame([("a", [1]), ("a", [2]), ("b", [3])], "k string, v array<int>")
    mapped = F.transform("v", lambda x: plus_one(x))
    window = Window.partitionBy("k").orderBy(real.element_at(mapped, 1))
    out = df.select("k", real.row_number().over(window).alias("rn"), mapped.alias("m")).orderBy(
        "k", "rn"
    )
    assert rows(out) == [("a", 1, [2]), ("a", 2, [3]), ("b", 1, [4])]


def test_rewrite_in_order_by(spark):
    df = spark.createDataFrame([([3],), ([1],), ([2],)], "v array<int>")
    mapped = F.transform("v", lambda x: plus_one(x))
    out = df.orderBy(real.element_at(mapped, 1)).select(mapped.alias("m"))
    assert rows(out) == [([2],), ([3],), ([4],)]


def test_rewrite_in_a_filter_predicate(spark):
    df = spark.createDataFrame([([1],), ([5],)], "v array<int>")
    mapped = F.transform("v", lambda x: plus_one(x))
    out = df.where(real.element_at(mapped, 1) > 3).select(mapped.alias("m"))
    assert rows(out) == [([6],)]


def test_union_of_two_rewrites(spark):
    left = spark.createDataFrame([([1],)], "v array<int>")
    right = spark.createDataFrame([([9],)], "v array<int>")
    out = left.select(F.transform("v", lambda x: plus_one(x)).alias("m")).union(
        right.select(F.transform("v", lambda x: plus_one(x)).alias("m"))
    )
    assert sorted(rows(out)) == [([2],), ([10],)]


def test_distinct_over_a_rewrite(spark):
    df = spark.createDataFrame([([1],), ([1],), ([2],)], "v array<int>")
    out = df.select(F.transform("v", lambda x: plus_one(x)).alias("m")).distinct()
    assert sorted(rows(out)) == [([2],), ([3],)]


# --------------------------------------------------------------------------
# Reuse across the query lifecycle.
# --------------------------------------------------------------------------


def test_the_same_column_object_reused_in_two_selects(spark):
    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    mapped = F.transform("v", lambda x: plus_one(x))
    first = df.select(mapped.alias("m"))
    second = df.select(mapped.alias("m"))
    assert rows(first) == rows(second) == [([2, 3],)]


def test_chained_withcolumn_calls(spark):
    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    out = (
        df.withColumn("a", F.transform("v", lambda x: plus_one(x)))
        .withColumn("b", F.transform("a", lambda x: plus_one(x)))
        .select("a", "b")
    )
    assert rows(out) == [([2, 3], [3, 4])]


def test_cached_dataframe(spark):
    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    out = df.select(F.transform("v", lambda x: plus_one(x)).alias("m")).cache()
    try:
        assert rows(out) == [([2, 3],)]
        assert rows(out) == [([2, 3],)]  # second read hits the cache
    finally:
        out.unpersist()


def test_written_and_read_back(spark, tmp_path):
    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    target = str(tmp_path / "mapped")
    df.select(F.transform("v", lambda x: plus_one(x)).alias("m")).write.parquet(target)
    assert rows(spark.read.parquet(target)) == [([2, 3],)]


def test_temp_view_and_repeated_collect(spark):
    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    df.select(F.transform("v", lambda x: plus_one(x)).alias("m")).createOrReplaceTempView(
        "elementwise_view"
    )
    try:
        assert rows(spark.sql("SELECT * FROM elementwise_view")) == [([2, 3],)]
    finally:
        spark.catalog.dropTempView("elementwise_view")


def test_explain_does_not_raise(spark):
    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    df.select(F.transform("v", lambda x: plus_one(x))).explain(extended=False)


def test_schema_is_array_of_the_declared_return_type(spark):
    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    out = df.select(F.transform("v", lambda x: plus_one(x)).alias("m"))
    assert out.schema["m"].dataType.simpleString() == "array<bigint>"


def test_repartition_then_rewrite(spark):
    df = spark.createDataFrame([([1],), ([2],), ([3],)], "v array<int>").repartition(3)
    out = df.select(F.transform("v", lambda x: plus_one(x)).alias("m"))
    assert sorted(rows(out)) == [([2],), ([3],), ([4],)]
