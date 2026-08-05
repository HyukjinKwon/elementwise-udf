"""Use Python UDFs inside Spark's native higher-order functions.

Import :mod:`elementwise_udf.functions` in place of ``pyspark.sql.functions``::

    import elementwise_udf.functions as esf

    @esf.udf("long")
    def plus_one(x):
        return x + 1

    df.select(esf.transform("values", lambda x: plus_one(x)).alias("result"))

Every name is delegated to ``pyspark.sql.functions`` untouched; only ``esf.udf``
and the higher-order functions behave differently. See
:mod:`elementwise_udf._core` for how the rewrite works and which lambda shapes
are supported.
"""

from elementwise_udf._core import udf

__all__ = ["udf"]

__version__ = "0.1.0"
