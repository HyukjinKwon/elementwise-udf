"""The ``elementwise_udf`` decorator's own surface."""

import pytest

from elementwise_udf import elementwise_udf, functions as F


def test_return_type_is_required():
    with pytest.raises(TypeError, match="must be called with a return type"):

        @elementwise_udf
        def missing_return_type(x):
            return x


def test_ddl_and_datatype_return_types_agree(spark):
    from pyspark.sql.types import LongType

    @elementwise_udf("long")
    def from_ddl(x):
        return x + 1

    @elementwise_udf(LongType())
    def from_datatype(x):
        return x + 1

    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    ddl = df.select(F.transform("v", lambda x: from_ddl(x))).collect()
    typed = df.select(F.transform("v", lambda x: from_datatype(x))).collect()
    assert ddl == typed


def test_metadata_is_preserved():
    @elementwise_udf("long")
    def documented(x):
        """A docstring worth keeping."""
        return x

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "A docstring worth keeping."


def test_struct_return_type(spark):
    @elementwise_udf("struct<doubled:long,text:string>")
    def to_struct(x):
        return (x * 2, str(x))

    df = spark.createDataFrame([([1, 2],)], "v array<int>")
    got = df.select(F.transform("v", lambda x: to_struct(x)["doubled"])).collect()
    assert [r[0] for r in got] == [[2, 4]]


def test_nested_array_elements(spark):
    @elementwise_udf("long")
    def total(xs):
        return sum(xs)

    df = spark.createDataFrame([([[1, 2], [3]],)], "v array<array<int>>")
    assert [r[0] for r in df.select(F.transform("v", lambda x: total(x))).collect()] == [[3, 3]]


def test_used_outside_any_higher_order_function(spark):
    @elementwise_udf("long")
    def plus_one(x):
        return x + 1

    assert [r[0] for r in spark.range(3).select(plus_one("id")).collect()] == [1, 2, 3]
