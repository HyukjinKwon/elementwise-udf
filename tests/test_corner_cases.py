"""Corner cases: unusual types, unusual lambdas, unusual data."""

import pytest

import elementwise_udf.functions as esf


@esf.udf("long")
def plus_one(x):
    return x + 1


def col(df, column):
    return [row[0] for row in df.select(column).collect()]


# --------------------------------------------------------------------------
# Data shapes.
# --------------------------------------------------------------------------


def test_empty_dataframe(spark):
    df = spark.createDataFrame([], "v array<int>")
    assert col(df, esf.transform("v", lambda x: plus_one(x))) == []


def test_single_element_array(spark):
    df = spark.createDataFrame([([7],)], "v array<int>")
    assert col(df, esf.transform("v", lambda x: plus_one(x))) == [[8]]


def test_null_elements_inside_the_array(spark):
    @esf.udf("long")
    def null_safe(x):
        return -1 if x is None else x + 1

    df = spark.createDataFrame([([1, None, 3],)], "v array<int>")
    assert col(df, esf.transform("v", lambda x: null_safe(x))) == [[2, -1, 4]]


def test_udf_returning_null(spark):
    @esf.udf("long")
    def only_even(x):
        return x if x % 2 == 0 else None

    df = spark.createDataFrame([([1, 2, 3, 4],)], "v array<int>")
    assert col(df, esf.transform("v", lambda x: only_even(x))) == [[None, 2, None, 4]]


def test_long_array(spark):
    df = spark.createDataFrame([(list(range(1000)),)], "v array<int>")
    got = col(df, esf.transform("v", lambda x: plus_one(x)))
    assert got == [list(range(1, 1001))]


def test_many_rows(spark):
    df = spark.range(500).select(esf.array(esf.col("id"), esf.col("id") + 1).alias("v"))
    got = col(df, esf.transform("v", lambda x: plus_one(x)))
    assert got[0] == [1, 2]
    assert got[-1] == [500, 501]


def test_all_rows_null(spark):
    df = spark.createDataFrame([(None,), (None,)], "v array<int>")
    assert col(df, esf.transform("v", lambda x: plus_one(x))) == [None, None]


def test_arrays_of_differing_length_in_zip_with(spark):
    @esf.udf("long")
    def add(a, b):
        return (a or 0) + (b or 0)

    df = spark.createDataFrame([([1, 2, 3], [10])], "l array<int>, r array<int>")
    # zip_with pads the shorter side with nulls, and so does the rewrite.
    assert col(df, esf.zip_with("l", "r", lambda a, b: add(a, b))) == [[11, 2, 3]]


# --------------------------------------------------------------------------
# Element and return types.
# --------------------------------------------------------------------------


def test_string_elements(spark):
    @esf.udf("string")
    def shout(s):
        return s.upper() + "!"

    df = spark.createDataFrame([(["a", "b"],)], "v array<string>")
    assert col(df, esf.transform("v", lambda x: shout(x))) == [["A!", "B!"]]


def test_boolean_return_type_used_as_a_predicate(spark):
    @esf.udf("boolean")
    def keep(x):
        return x != 2

    df = spark.createDataFrame([([1, 2, 3],)], "v array<int>")
    assert col(df, esf.filter("v", lambda x: keep(x))) == [[1, 3]]


def test_double_return_type(spark):
    @esf.udf("double")
    def half(x):
        return x / 2

    df = spark.createDataFrame([([1, 3],)], "v array<int>")
    assert col(df, esf.transform("v", lambda x: half(x))) == [[0.5, 1.5]]


def test_array_return_type_gives_nested_arrays(spark):
    @esf.udf("array<long>")
    def repeat(x):
        return [x, x]

    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    assert col(df, esf.transform("v", lambda x: repeat(x))) == [[[1, 1], [2, 2]]]


def test_map_return_type(spark):
    @esf.udf("map<string,long>")
    def as_map(x):
        return {"n": x}

    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    assert col(df, esf.transform("v", lambda x: as_map(x))) == [[{"n": 1}, {"n": 2}]]


def test_struct_elements(spark):
    @esf.udf("long")
    def take_field(row):
        return row["n"] + 1

    df = spark.createDataFrame([([{"n": 1}, {"n": 2}],)], "v array<struct<n:int>>")
    assert col(df, esf.transform("v", lambda x: take_field(x))) == [[2, 3]]


def test_timestamp_elements(spark):
    import datetime

    @esf.udf("long")
    def year_of(ts):
        return ts.year

    moment = datetime.datetime(2026, 1, 1, 12, 0)
    df = spark.createDataFrame([([moment],)], "v array<timestamp>")
    assert col(df, esf.transform("v", lambda x: year_of(x))) == [[2026]]


def test_decimal_elements(spark):
    from decimal import Decimal

    @esf.udf("string")
    def render(d):
        return str(d)

    df = spark.createDataFrame([([Decimal("1.50")],)], "v array<decimal(5,2)>")
    assert col(df, esf.transform("v", lambda x: render(x))) == [["1.50"]]


# --------------------------------------------------------------------------
# Lambda shapes.
# --------------------------------------------------------------------------


def test_named_function_instead_of_a_lambda(spark):
    def body(x):
        return plus_one(x)

    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    assert col(df, esf.transform("v", body)) == [[2, 3]]


def test_lambda_ignoring_its_argument(spark):
    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    assert col(df, esf.transform("v", lambda x: esf.lit(0))) == [[0, 0]]


def test_udf_on_a_captured_variable_only(spark):
    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    # The UDF argument does not depend on the element at all.
    assert col(df, esf.transform("v", lambda x: plus_one(esf.lit(10)))) == [[11, 11]]


def test_same_udf_four_times_in_one_lambda(spark):
    df = spark.createDataFrame([([1],)], "v array<int>")
    expression = esf.transform("v", lambda x: plus_one(x) + plus_one(x) + plus_one(x) + plus_one(x))
    assert col(df, expression) == [[8]]


def test_deeply_nested_udf_calls(spark):
    df = spark.createDataFrame([([1],)], "v array<int>")
    assert col(df, esf.transform("v", lambda x: plus_one(plus_one(plus_one(x))))) == [[4]]


def test_udf_inside_a_coalesce(spark):
    @esf.udf("long")
    def nullify(x):
        return None

    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    got = col(df, esf.transform("v", lambda x: esf.coalesce(nullify(x), esf.lit(-1))))
    assert got == [[-1, -1]]


def test_udf_result_compared_to_another_column(spark):
    df = spark.createDataFrame([(2, [1, 2, 3])], "n int, v array<int>")
    got = col(df, esf.transform("v", lambda x: (plus_one(x) > esf.col("n")).cast("int")))
    assert got == [[0, 1, 1]]


def test_index_only_lambda(spark):
    df = spark.createDataFrame([([9, 9, 9],)], "v array<int>")
    assert col(df, esf.transform("v", lambda x, i: plus_one(i))) == [[1, 2, 3]]


def test_nested_higher_order_functions_three_deep(spark):
    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    inner = esf.transform("v", lambda x: plus_one(x))
    middle = esf.filter(inner, lambda x: plus_one(x) > 3)
    outer = esf.transform(middle, lambda x: plus_one(x))
    assert col(df, outer) == [[4]]


def test_two_different_arrays_in_one_lambda_via_zip_with(spark):
    @esf.udf("string")
    def pair(a, b):
        return f"{a}-{b}"

    df = spark.createDataFrame([([1, 2], [3, 4])], "l array<int>, r array<int>")
    assert col(df, esf.zip_with("l", "r", lambda a, b: pair(a, b))) == [["1-3", "2-4"]]


# --------------------------------------------------------------------------
# Errors surface clearly rather than becoming wrong answers.
# --------------------------------------------------------------------------


def test_exception_inside_the_udf_propagates(spark):
    @esf.udf("long")
    def explode_on_two(x):
        if x == 2:
            raise ValueError("boom")
        return x

    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    with pytest.raises(Exception, match="boom"):
        df.select(esf.transform("v", lambda x: explode_on_two(x))).collect()


def test_python_if_on_a_udf_result_is_rejected(spark):
    df = spark.createDataFrame([([1],)], "v array<int>")
    with pytest.raises(Exception):
        df.select(esf.transform("v", lambda x: 1 if plus_one(x) else 0)).collect()


def test_udf_on_the_accumulator_with_spark_expressions_is_reported(spark):
    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    with pytest.raises(TypeError, match="applied to the accumulator"):
        esf.aggregate(
            "v",
            esf.lit(0).cast("long"),
            lambda acc, x: esf.when(plus_one(acc) > 2, plus_one(acc)).otherwise(0) + x,
        )


def test_udf_in_a_folds_finish_lambda(spark):
    # aggregate/reduce are the only functions taking two lambdas. The finish one
    # runs once on the final accumulator, so a UDF there is applied to the fold's
    # result rather than rewritten.
    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    got = col(
        df,
        esf.aggregate("v", esf.lit(0).cast("long"), lambda a, x: a + x, lambda a: plus_one(a)),
    )
    assert got == [4]


def test_udf_in_both_a_folds_merge_and_finish_lambdas(spark):
    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    got = col(
        df,
        esf.aggregate(
            "v", esf.lit(0).cast("long"), lambda a, x: a + plus_one(x), lambda a: plus_one(a)
        ),
    )
    # merge: 0 + 2 + 3 = 5; finish: plus_one(5) = 6
    assert got == [6]


def test_fold_finish_udf_is_not_called_for_a_null_array(spark):
    # A fold over a null array is null, and native Spark does not evaluate the
    # finish lambda for it. Found on a real DBR cluster: the UDF was applied to
    # that null and a null-unaware body raised, where plain PySpark returns null.
    # Wrapping in `when` does not help - Spark evaluates both branches - so the
    # guard lives inside the rebuilt UDF.
    df = spark.createDataFrame([([1, 2, 3],), ([],), (None,)], "v array<int>")
    got = col(
        df,
        esf.aggregate("v", esf.lit(0).cast("long"), lambda a, x: a + x, lambda a: plus_one(a)),
    )
    assert got == [7, 1, None]


def test_fold_finish_matches_native_spark_for_nulls(spark):
    import pyspark.sql.functions as sf

    df = spark.createDataFrame([([1, 2, 3],), ([],), (None,)], "v array<int>")
    rewritten = col(
        df,
        esf.aggregate("v", esf.lit(0).cast("long"), lambda a, x: a + x, lambda a: plus_one(a)),
    )
    native = col(df, sf.aggregate("v", sf.lit(0).cast("long"), lambda a, x: a + x, lambda a: a + 1))
    assert rewritten == native
