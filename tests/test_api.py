"""The ``udf`` decorator's own surface."""

import pytest

from elementwise_udf import udf, functions as F


def test_bare_decorator_defaults_to_string(spark):
    @udf
    def stringify(x):
        return str(x)

    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    assert [r[0] for r in df.select(F.transform("v", lambda x: stringify(x))).collect()] == [
        ["1", "2"]
    ]


def test_called_directly_with_function_and_type(spark):
    plus_one = udf(lambda x: x + 1, "long")
    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    assert [r[0] for r in df.select(F.transform("v", lambda x: plus_one(x))).collect()] == [[2, 3]]


def test_use_arrow(spark):
    plus_one = udf(lambda x: x + 1, "long", useArrow=True)
    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    assert [r[0] for r in df.select(F.transform("v", lambda x: plus_one(x))).collect()] == [[2, 3]]


def test_scalar_exposes_a_real_pyspark_udf_for_sql_registration(spark):
    plus_one = udf(lambda x: x + 1, "long")
    spark.udf.register("elementwise_plus_one", plus_one.scalar)
    assert spark.sql("SELECT elementwise_plus_one(41) AS r").collect()[0]["r"] == 42


def test_as_nondeterministic(spark):
    plus_one = udf(lambda x: x + 1, "long").asNondeterministic()
    assert plus_one.deterministic is False
    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    assert [r[0] for r in df.select(F.transform("v", lambda x: plus_one(x))).collect()] == [[2, 3]]


def test_ddl_and_datatype_return_types_agree(spark):
    from pyspark.sql.types import LongType

    @udf("long")
    def from_ddl(x):
        return x + 1

    @udf(LongType())
    def from_datatype(x):
        return x + 1

    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    ddl = df.select(F.transform("v", lambda x: from_ddl(x))).collect()
    typed = df.select(F.transform("v", lambda x: from_datatype(x))).collect()
    assert ddl == typed


def test_metadata_is_preserved():
    @udf("long")
    def documented(x):
        """A docstring worth keeping."""
        return x

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "A docstring worth keeping."


def test_struct_return_type(spark):
    @udf("struct<doubled:long,text:string>")
    def to_struct(x):
        return (x * 2, str(x))

    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    got = df.select(F.transform("v", lambda x: to_struct(x)["doubled"])).collect()
    assert [r[0] for r in got] == [[2, 4]]


def test_nested_array_elements(spark):
    @udf("long")
    def total(xs):
        return sum(xs)

    df = spark.createDataFrame([([[1, 2], [3]],)], "v array<array<int>>")
    assert [r[0] for r in df.select(F.transform("v", lambda x: total(x))).collect()] == [[3, 3]]


def test_used_outside_any_higher_order_function(spark):
    @udf("long")
    def plus_one(x):
        return x + 1

    assert [r[0] for r in spark.range(3).select(plus_one("id")).collect()] == [1, 2, 3]
