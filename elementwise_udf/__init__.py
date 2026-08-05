"""Use Python UDFs inside Spark's native higher-order functions.

Declare the UDF with :func:`udf` from this package instead of
``pyspark.sql.functions.udf``, and import :data:`functions` from here instead of
from ``pyspark.sql``::

    from elementwise_udf import udf, functions as F

    @udf("long")
    def plus_one(x):
        return x + 1

    df.select(F.transform("values", lambda x: plus_one(x)).alias("result"))

See :mod:`elementwise_udf._core` for how the rewrite works and which lambda
shapes are supported.
"""

from elementwise_udf._core import functions, udf

__all__ = ["udf", "functions"]

__version__ = "0.1.0"
