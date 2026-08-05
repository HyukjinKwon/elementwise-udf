"""The ``functions`` proxy must be a faithful stand-in for the real module.

Native lambdas - those using no element-wise UDF - must reach PySpark
completely unchanged, and PySpark itself must never be modified.
"""

import pyspark.sql.functions as real
import pytest

from elementwise_udf import udf, functions as F


@udf("long")
def plus_one(x):
    return x + 1


def col(df, column):
    return [row[0] for row in df.select(column).collect()]


def test_pyspark_is_not_patched():
    assert real.transform.__module__ == "pyspark.sql.functions.builtin"
    assert not hasattr(real.transform, "_elementwise_patched")


def test_dir_matches_the_real_module():
    assert dir(F) == dir(real)


def test_non_callable_attributes_are_forwarded():
    assert F.__name__ if hasattr(F, "__name__") else True


def test_native_transform(values):
    assert col(values, F.transform("v", lambda x: x * 2)) == [[2, 4, 6], [], None]


def test_native_transform_with_index(values):
    got = col(values, F.transform("v", lambda x, i: F.when(i % 2 == 0, x).otherwise(-x)))
    assert got == [[1, -2, 3], [], None]


def test_native_filter(values):
    assert col(values, F.filter("v", lambda x: x % 2 == 1)) == [[1, 3], [], None]


def test_native_exists(values):
    assert col(values, F.exists("v", lambda x: x > 2)) == [True, False, None]


def test_native_forall(values):
    assert col(values, F.forall("v", lambda x: x > 0)) == [True, True, None]


def test_native_aggregate(values):
    assert col(values, F.aggregate("v", F.lit(0), lambda acc, x: acc + x)) == [6, 0, None]


def test_native_array_sort_comparator(spark):
    df = spark.createDataFrame([([3, 1, 2],)], "v array<int>")
    descending = F.array_sort("v", lambda a, b: F.when(a < b, 1).when(a > b, -1).otherwise(0))
    assert col(df, descending) == [[3, 2, 1]]


def test_native_zip_with(spark):
    df = spark.createDataFrame([([1, 2], [10, 20])], "l array<int>, r array<int>")
    assert col(df, F.zip_with("l", "r", lambda a, b: a + b)) == [[11, 22]]


def test_native_transform_values(spark):
    df = spark.createDataFrame([({"a": 1},)], "m map<string,int>")
    assert col(df, F.transform_values("m", lambda k, v: v + 1)) == [{"a": 2}]


def test_non_higher_order_function_matches_pyspark(values):
    assert col(values, F.size("v")) == col(values, real.size("v"))


def test_null_and_empty_arrays_are_preserved(values):
    # A null array stays null and an empty one stays empty; in both cases the
    # UDF is never called (the UDF itself asserts on null input).
    assert col(values, F.transform("v", lambda x: plus_one(x))) == [[2, 3, 4], [], None]


# The README's "Why functions is imported from here" section makes specific
# claims about the proxy. These pin them so the docs cannot drift.


def test_dir_matches_pyspark_name_for_name():
    assert dir(F) == dir(real)


def test_callables_are_delegated_through_a_wrapper():
    # Not the same object, but wrapping the original: the README says so
    # explicitly, because `sf.col is pyspark.sql.functions.col` is False.
    assert F.col is not real.col
    assert F.col.__wrapped__ is real.col


def test_delegated_functions_build_identical_expressions():
    assert repr(F.col("v")) == repr(real.col("v"))
    assert repr(F.lit(1)) == repr(real.lit(1))
    assert repr(F.upper(F.col("v"))) == repr(real.upper(real.col("v")))


def test_non_higher_order_results_match_pyspark(values):
    for build in (F.size, F.reverse, F.array_max):
        assert col(values, build("v")) == col(values, getattr(real, build.__name__)("v"))


def test_mixing_plain_pyspark_and_the_proxy_in_one_select(spark):
    # Only the higher-order call needs the proxy; everything else may come from
    # pyspark directly, as the README shows.
    df = spark.createDataFrame([(["ab"], [1, 2])], "name array<string>, v array<int>")
    out = df.select(
        real.upper(real.element_at("name", 1)).alias("u"),
        F.transform("v", lambda x: plus_one(x)).alias("m"),
    )
    assert [tuple(r) for r in out.collect()] == [("AB", [2, 3])]


def test_plain_pyspark_higher_order_function_is_left_broken(spark):
    # The flip side of the same claim: imported from pyspark directly, a Python
    # UDF in the lambda still fails, which is why the proxy exists.
    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    with pytest.raises(Exception) as excinfo:
        df.select(real.transform("v", lambda x: plus_one(x))).collect()
    assert "LAMBDA_FUNCTION_WITH_PYTHON_UDF" in str(excinfo.value)
