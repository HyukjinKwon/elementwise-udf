"""Every higher-order function, with and without an element-wise UDF."""

import warnings

import pytest

import elementwise_udf.functions as esf


@esf.udf("long")
def plus_one(x):
    assert x is not None, "a null element must never reach the UDF"
    return x + 1


@esf.udf("long")
def times_ten(x):
    return x * 10


@esf.udf("long")
def add(a, b):
    return (a or 0) + (b or 0)


@esf.udf("boolean")
def is_odd(x):
    return x % 2 == 1


@esf.udf("string")
def upper(s):
    return str(s).upper()


@esf.udf("int")
def compare(a, b):
    return -1 if a < b else (1 if a > b else 0)


def col(df, column):
    return [row[0] for row in df.select(column).collect()]


# --------------------------------------------------------------------------
# The headline case: a native higher-order function with a Python UDF.
# --------------------------------------------------------------------------


def test_transform(values):
    assert col(values, esf.transform("v", lambda x: plus_one(x))) == [[2, 3, 4], [], None]


def test_transform_keeps_alias(values):
    out = values.select(esf.transform("v", lambda x: plus_one(x)).alias("result"))
    assert out.columns == ["result"]


def test_plain_udf_call_still_works(spark):
    assert col(spark.range(1), plus_one(esf.lit(1))) == [2]


def test_bare_udf_instead_of_lambda(values):
    assert col(values, esf.transform("v", plus_one)) == [[2, 3, 4], [], None]


def test_column_argument(values):
    assert col(values, esf.transform(esf.col("v"), lambda x: plus_one(x))) == [[2, 3, 4], [], None]


# --------------------------------------------------------------------------
# Expressions around the UDF result: it is a plain column by then.
# --------------------------------------------------------------------------


def test_arithmetic_on_udf_result(values):
    assert col(values, esf.transform("v", lambda x: plus_one(x) * 2)) == [[4, 6, 8], [], None]


def test_udf_result_mixed_with_element_and_index(values):
    got = col(values, esf.transform("v", lambda x, i: plus_one(x) * 100 + x + i))
    assert got == [[201, 303, 405], [], None]


def test_when_on_udf_result(values):
    got = col(values, esf.transform("v", lambda x: esf.when(plus_one(x) > 2, 1).otherwise(0)))
    assert got == [[0, 1, 1], [], None]


def test_cast_on_udf_result(values):
    got = col(values, esf.transform("v", lambda x: plus_one(x).cast("string")))
    assert got == [["2", "3", "4"], [], None]


def test_two_udfs_in_one_lambda(values):
    got = col(values, esf.transform("v", lambda x: plus_one(x) + times_ten(x)))
    assert got == [[12, 23, 34], [], None]


def test_udf_argument_is_an_expression(values):
    assert col(values, esf.transform("v", lambda x: plus_one(x * 10))) == [[11, 21, 31], [], None]


def test_nested_udfs(values):
    assert col(values, esf.transform("v", lambda x: plus_one(times_ten(x)))) == [
        [11, 21, 31],
        [],
        None,
    ]


def test_udf_called_with_keyword_argument(values):
    assert col(values, esf.transform("v", lambda x: plus_one(x=x))) == [[2, 3, 4], [], None]


def test_captured_column_argument(spark):
    df = spark.createDataFrame([([1, 2], 10), ([3], 20)], "v array<int>, k int")
    got = col(df, esf.transform("v", lambda x: add(x, esf.col("k"))))
    assert got == [[11, 12], [23]]


# --------------------------------------------------------------------------
# One select holding several higher-order calls and ordinary columns.
# --------------------------------------------------------------------------


def test_many_higher_order_calls_with_plain_columns(spark):
    df = spark.createDataFrame(
        [(1, "a", [1, 2, 3]), (2, "b", [4])], "id int, tag string, v array<int>"
    )
    out = df.select(
        "id",
        "tag",
        esf.transform("v", lambda x: plus_one(x) + 1).alias("a"),
        esf.transform("v", lambda x: plus_one(x) + 2).alias("b"),
        esf.filter("v", lambda x: times_ten(x) > 15).alias("c"),
        esf.exists("v", lambda x: plus_one(x) > 4).alias("d"),
    )
    assert [tuple(r) for r in out.orderBy("id").collect()] == [
        (1, "a", [3, 4, 5], [4, 5, 6], [2, 3], False),
        (2, "b", [6], [7], [4], True),
    ]


def test_nested_higher_order_calls(values):
    got = col(values, esf.transform(esf.filter("v", lambda x: is_odd(x)), lambda y: times_ten(y)))
    assert got == [[10, 30], [], None]


# --------------------------------------------------------------------------
# Each higher-order function.
# --------------------------------------------------------------------------


def test_filter(values):
    assert col(values, esf.filter("v", lambda x: is_odd(x))) == [[1, 3], [], None]


def test_filter_on_an_expression(values):
    assert col(values, esf.filter("v", lambda x: plus_one(x) > 2)) == [[2, 3], [], None]


def test_exists(values):
    assert col(values, esf.exists("v", lambda x: plus_one(x) > 3)) == [True, False, None]


def test_forall(values):
    assert col(values, esf.forall("v", lambda x: plus_one(x) > 1)) == [True, True, None]


def test_zip_with(spark):
    df = spark.createDataFrame(
        [([1, 2, 3], [10, 20, 30]), ([1], [9])], "l array<int>, r array<int>"
    )
    assert col(df, esf.zip_with("l", "r", lambda a, b: add(a, b))) == [[11, 22, 33], [10]]


def test_zip_with_expression(spark):
    df = spark.createDataFrame([([1, 2], [10, 20])], "l array<int>, r array<int>")
    assert col(df, esf.zip_with("l", "r", lambda a, b: add(a, b) * 2)) == [[22, 44]]


def test_aggregate_on_element(values):
    got = col(values, esf.aggregate("v", esf.lit(0).cast("long"), lambda acc, x: acc + plus_one(x)))
    assert got == [9, 0, None]


def test_reduce_on_element(values):
    got = col(values, esf.reduce("v", esf.lit(0).cast("long"), lambda acc, x: acc + plus_one(x)))
    assert got == [9, 0, None]


def test_aggregate_with_finish_lambda(values):
    got = col(
        values,
        esf.aggregate(
            "v", esf.lit(0).cast("long"), lambda acc, x: acc + plus_one(x), lambda a: a * 10
        ),
    )
    assert got == [90, 0, None]


def test_array_sort_by_udf_key(spark):
    df = spark.createDataFrame([([3, 1, 2],)], "v array<int>")
    ascending = esf.array_sort(
        "v",
        lambda a, b: esf.when(plus_one(a) < plus_one(b), -1)
        .when(plus_one(a) > plus_one(b), 1)
        .otherwise(0),
    )
    assert col(df, ascending) == [[1, 2, 3]]


def test_array_sort_by_udf_key_descending(spark):
    df = spark.createDataFrame([([3, 1, 2],)], "v array<int>")
    descending = esf.array_sort(
        "v",
        lambda a, b: esf.when(plus_one(a) < plus_one(b), 1)
        .when(plus_one(a) > plus_one(b), -1)
        .otherwise(0),
    )
    assert col(df, descending) == [[3, 2, 1]]


def test_transform_values(spark):
    df = spark.createDataFrame([({"a": 1, "b": 2},)], "m map<string,int>")
    assert col(df, esf.transform_values("m", lambda k, v: plus_one(v))) == [{"a": 2, "b": 3}]


def test_transform_keys(spark):
    df = spark.createDataFrame([({"a": 1, "b": 2},)], "m map<string,int>")
    assert col(df, esf.transform_keys("m", lambda k, v: upper(k))) == [{"A": 1, "B": 2}]


def test_map_filter(spark):
    df = spark.createDataFrame([({"a": 1, "b": 2},)], "m map<string,int>")
    assert col(df, esf.map_filter("m", lambda k, v: plus_one(v) > 2)) == [{"b": 2}]


def test_map_zip_with(spark):
    df = spark.createDataFrame(
        [
            (
                {"a": 1, "b": 2},
                {"a": 5, "c": 7},
            )
        ],
        "m1 map<string,int>, m2 map<string,int>",
    )
    got = col(df, esf.map_zip_with("m1", "m2", lambda k, v1, v2: add(v1, v2)))
    assert got == [{"a": 6, "b": 2, "c": 7}]


# --------------------------------------------------------------------------
# The slow paths: they work, and they warn.
# --------------------------------------------------------------------------


def test_udf_on_accumulator_works_and_warns(spark):
    df = spark.createDataFrame([([3, 1, 2],)], "v array<int>")
    expression = esf.aggregate("v", esf.lit(0).cast("long"), lambda acc, x: plus_one(acc) + x)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        expression = esf.aggregate("v", esf.lit(0).cast("long"), lambda acc, x: plus_one(acc) + x)
        assert any(w.category is RuntimeWarning for w in caught)
    # acc=0; x=3 -> (0+1)+3=4; x=1 -> (4+1)+1=6; x=2 -> (6+1)+2=9
    assert col(df, expression) == [9]


def test_pairwise_comparator_works_and_warns(spark):
    df = spark.createDataFrame([([3, 1, 2],)], "v array<int>")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        expression = esf.array_sort("v", lambda a, b: compare(a, b))
        assert any(w.category is RuntimeWarning for w in caught)
    assert col(df, expression) == [[1, 2, 3]]


def test_fast_paths_do_not_warn(values):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        esf.transform("v", lambda x: plus_one(x))
        esf.aggregate("v", esf.lit(0).cast("long"), lambda acc, x: acc + plus_one(x))
        assert [w for w in caught if w.category is RuntimeWarning] == []


# --------------------------------------------------------------------------
# The one genuine limitation.
# --------------------------------------------------------------------------


def test_python_if_on_udf_result_is_rejected(values):
    from pyspark.errors import PySparkValueError

    with pytest.raises((PySparkValueError, ValueError, TypeError)):
        values.select(esf.transform("v", lambda x: 1 if plus_one(x) else 0)).collect()
