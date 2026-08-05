"""Use Python UDFs inside Spark's native higher-order functions.

See :mod:`elementwise_udf._core` for the full explanation of how the rewrite
works and which lambda shapes are supported.
"""

from elementwise_udf._core import elementwise_udf, functions

__all__ = ["elementwise_udf", "functions"]

__version__ = "0.1.0"
