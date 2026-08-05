"""Corner cases: unusual types, unusual lambdas, unusual data."""

import pytest

from elementwise_udf import udf, functions as F


@udf("long")
def plus_one(x):
    return x + 1


def col(df, column):
    return [row[0] for row in df.select(column).collect()]


# --------------------------------------------------------------------------
# Data shapes.
# --------------------------------------------------------------------------


def test_empty_dataframe(spark):
    df = spark.createDataFrame([], "v array<int>")
    assert col(df, F.transform("v", lambda x: plus_one(x))) == []


def test_single_element_array(spark):
    df = spark.createDataFrame([([7],)], "v array<int>")
    assert col(df, F.transform("v", lambda x: plus_one(x))) == [[8]]


def test_null_elements_inside_the_array(spark):
    @udf("long")
    def null_safe(x):
        return -1 if x is None else x + 1

    df = spark.createDataFrame([([1, None, 3],)], "v array<int>")
    assert col(df, F.transform("v", lambda x: null_safe(x))) == [[2, -1, 4]]


def test_udf_returning_null(spark):
    @udf("long")
    def only_even(x):
        return x if x % 2 == 0 else None

    df = spark.createDataFrame([([1, 2, 3, 4],)], "v array<int>")
    assert col(df, F.transform("v", lambda x: only_even(x))) == [[None, 2, None, 4]]


def test_long_array(spark):
    df = spark.createDataFrame([(list(range(1000)),)], "v array<int>")
    got = col(df, F.transform("v", lambda x: plus_one(x)))
    assert got == [list(range(1, 1001))]


def test_many_rows(spark):
    df = spark.range(500).select(F.array(F.col("id"), F.col("id") + 1).alias("v"))
    got = col(df, F.transform("v", lambda x: plus_one(x)))
    assert got[0] == [1, 2]
    assert got[-1] == [500, 501]


def test_all_rows_null(spark):
    df = spark.createDataFrame([(None,), (None,)], "v array<int>")
    assert col(df, F.transform("v", lambda x: plus_one(x))) == [None, None]


def test_arrays_of_differing_length_in_zip_with(spark):
    @udf("long")
    def add(a, b):
        return (a or 0) + (b or 0)

    df = spark.createDataFrame([([1, 2, 3], [10])], "l array<int>, r array<int>")
    # zip_with pads the shorter side with nulls, and so does the rewrite.
    assert col(df, F.zip_with("l", "r", lambda a, b: add(a, b))) == [[11, 2, 3]]


# --------------------------------------------------------------------------
# Element and return types.
# --------------------------------------------------------------------------


def test_string_elements(spark):
    @udf("string")
    def shout(s):
        return s.upper() + "!"

    df = spark.createDataFrame([(["a", "b"],)], "v array<string>")
    assert col(df, F.transform("v", lambda x: shout(x))) == [["A!", "B!"]]


def test_boolean_return_type_used_as_a_predicate(spark):
    @udf("boolean")
    def keep(x):
        return x != 2

    df = spark.createDataFrame([([1, 2, 3],)], "v array<int>")
    assert col(df, F.filter("v", lambda x: keep(x))) == [[1, 3]]


def test_double_return_type(spark):
    @udf("double")
    def half(x):
        return x / 2

    df = spark.createDataFrame([([1, 3],)], "v array<int>")
    assert col(df, F.transform("v", lambda x: half(x))) == [[0.5, 1.5]]


def test_array_return_type_gives_nested_arrays(spark):
    @udf("array<long>")
    def repeat(x):
        return [x, x]

    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    assert col(df, F.transform("v", lambda x: repeat(x))) == [[[1, 1], [2, 2]]]


def test_map_return_type(spark):
    @udf("map<string,long>")
    def as_map(x):
        return {"n": x}

    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    assert col(df, F.transform("v", lambda x: as_map(x))) == [[{"n": 1}, {"n": 2}]]


def test_struct_elements(spark):
    @udf("long")
    def take_field(row):
        return row["n"] + 1

    df = spark.createDataFrame([([{"n": 1}, {"n": 2}],)], "v array<struct<n:int>>")
    assert col(df, F.transform("v", lambda x: take_field(x))) == [[2, 3]]


def test_timestamp_elements(spark):
    import datetime

    @udf("long")
    def year_of(ts):
        return ts.year

    moment = datetime.datetime(2026, 1, 1, 12, 0)
    df = spark.createDataFrame([([moment],)], "v array<timestamp>")
    assert col(df, F.transform("v", lambda x: year_of(x))) == [[2026]]


def test_decimal_elements(spark):
    from decimal import Decimal

    @udf("string")
    def render(d):
        return str(d)

    df = spark.createDataFrame([([Decimal("1.50")],)], "v array<decimal(5,2)>")
    assert col(df, F.transform("v", lambda x: render(x))) == [["1.50"]]


# --------------------------------------------------------------------------
# Lambda shapes.
# --------------------------------------------------------------------------


def test_named_function_instead_of_a_lambda(spark):
    def body(x):
        return plus_one(x)

    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    assert col(df, F.transform("v", body)) == [[2, 3]]


def test_lambda_ignoring_its_argument(spark):
    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    assert col(df, F.transform("v", lambda x: F.lit(0))) == [[0, 0]]


def test_udf_on_a_captured_variable_only(spark):
    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    # The UDF argument does not depend on the element at all.
    assert col(df, F.transform("v", lambda x: plus_one(F.lit(10)))) == [[11, 11]]


def test_same_udf_four_times_in_one_lambda(spark):
    df = spark.createDataFrame([([1],)], "v array<int>")
    expression = F.transform("v", lambda x: plus_one(x) + plus_one(x) + plus_one(x) + plus_one(x))
    assert col(df, expression) == [[8]]


def test_deeply_nested_udf_calls(spark):
    df = spark.createDataFrame([([1],)], "v array<int>")
    assert col(df, F.transform("v", lambda x: plus_one(plus_one(plus_one(x))))) == [[4]]


def test_udf_inside_a_coalesce(spark):
    @udf("long")
    def nullify(x):
        return None

    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    got = col(df, F.transform("v", lambda x: F.coalesce(nullify(x), F.lit(-1))))
    assert got == [[-1, -1]]


def test_udf_result_compared_to_another_column(spark):
    df = spark.createDataFrame([(2, [1, 2, 3])], "n int, v array<int>")
    got = col(df, F.transform("v", lambda x: (plus_one(x) > F.col("n")).cast("int")))
    assert got == [[0, 1, 1]]


def test_index_only_lambda(spark):
    df = spark.createDataFrame([([9, 9, 9],)], "v array<int>")
    assert col(df, F.transform("v", lambda x, i: plus_one(i))) == [[1, 2, 3]]


def test_nested_higher_order_functions_three_deep(spark):
    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    inner = F.transform("v", lambda x: plus_one(x))
    middle = F.filter(inner, lambda x: plus_one(x) > 3)
    outer = F.transform(middle, lambda x: plus_one(x))
    assert col(df, outer) == [[4]]


def test_two_different_arrays_in_one_lambda_via_zip_with(spark):
    @udf("string")
    def pair(a, b):
        return f"{a}-{b}"

    df = spark.createDataFrame([([1, 2], [3, 4])], "l array<int>, r array<int>")
    assert col(df, F.zip_with("l", "r", lambda a, b: pair(a, b))) == [["1-3", "2-4"]]


# --------------------------------------------------------------------------
# Errors surface clearly rather than becoming wrong answers.
# --------------------------------------------------------------------------


def test_exception_inside_the_udf_propagates(spark):
    @udf("long")
    def explode_on_two(x):
        if x == 2:
            raise ValueError("boom")
        return x

    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    with pytest.raises(Exception, match="boom"):
        df.select(F.transform("v", lambda x: explode_on_two(x))).collect()


def test_python_if_on_a_udf_result_is_rejected(spark):
    df = spark.createDataFrame([([1],)], "v array<int>")
    with pytest.raises(Exception):
        df.select(F.transform("v", lambda x: 1 if plus_one(x) else 0)).collect()


def test_udf_on_the_accumulator_with_spark_expressions_is_reported(spark):
    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    with pytest.raises(TypeError, match="applied to the accumulator"):
        F.aggregate(
            "v",
            F.lit(0).cast("long"),
            lambda acc, x: F.when(plus_one(acc) > 2, plus_one(acc)).otherwise(0) + x,
        )


def test_two_lambdas_where_only_one_is_meaningful_is_reported(spark):
    @udf("long")
    def key(x):
        return x

    with pytest.raises(TypeError, match="more than one lambda"):
        F.map_zip_with("m1", "m2", lambda k, v1, v2: key(v1), lambda k: k)
