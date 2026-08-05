"""Drop-in replacement for :mod:`pyspark.sql.functions`.

Import it in place of PySpark's own module::

    import elementwise_udf.functions as esf

Every name is delegated to :mod:`pyspark.sql.functions` and behaves identically,
so ``esf.col``, ``esf.lit``, ``esf.when`` and the rest are PySpark's. Two things
differ, and they are the whole point of the package:

* ``esf.udf`` builds a UDF that may be called inside a higher-order function's
  lambda, which PySpark's ``udf`` cannot (`SPARK-27052
  <https://issues.apache.org/jira/browse/SPARK-27052>`__).
* A higher-order call whose lambda uses such a UDF is rewritten so the UDF sits
  outside the lambda, which is the only place Spark allows it.

Everything else is forwarded untouched, and nothing in PySpark is modified::

    import elementwise_udf.functions as esf

    @esf.udf("long")
    def plus_one(x):
        return x + 1

    df.select(esf.transform("values", lambda x: plus_one(x)).alias("result"))
"""

from typing import Any, List

from pyspark.sql import functions as _F

from elementwise_udf import _core

# ``udf`` shadows PySpark's deliberately: this is the element-wise one.
udf = _core.udf


def __getattr__(name: str) -> Any:
    """Delegate every other name to ``pyspark.sql.functions``.

    A module-level ``__getattr__`` (PEP 562) is consulted only for names not
    already defined here, so ``udf`` above wins while everything else falls
    through to PySpark. Callables are wrapped so a higher-order call can be
    rewritten; the wrapper is cached in this module's namespace, which means the
    lookup happens once per name.
    """
    attr = _core.wrap_function(name)
    globals()[name] = attr
    return attr


def __dir__() -> List[str]:
    return dir(_F)
