"""Native higher-order functions with a Python UDF.

Run against a local session::

    python examples/quickstart.py

or against Databricks serverless by replacing the session setup with::

    from databricks.connect import DatabricksSession
    spark = DatabricksSession.builder.serverless().getOrCreate()

Note ``functions`` is imported from this package rather than from ``pyspark.sql``.
It is a transparent proxy: every attribute is forwarded to
``pyspark.sql.functions`` untouched, so it is a drop-in for
``import pyspark.sql.functions as sf``. The one difference is that it rewrites a
higher-order call whose lambda uses an element-wise UDF, lifting the UDF out of
the lambda - which is the only place Spark allows it. Importing
``pyspark.sql.functions`` directly leaves the call unrewritten and Spark rejects
it. See the README section "Why functions is imported from here".
"""

from pyspark.sql import SparkSession

from elementwise_udf import functions as sf
from elementwise_udf import udf as eudf

spark = SparkSession.builder.master("local[2]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = spark.createDataFrame([([1, 2, 3],)], ["values"])

# A native higher-order function, no UDF involved: unchanged.
df.select(sf.transform("values", lambda x: x + 1).alias("result")).show()


@eudf("long")
def plus_one(x):
    return x + 1


# An ordinary UDF call, exactly as with pyspark.sql.functions.udf("long").
spark.range(1).select(plus_one(sf.lit(1))).show()

# The same native higher-order function, now with the Python UDF inside it.
df.select(sf.transform("values", lambda x: plus_one(x)).alias("result")).show()

# The UDF result is a plain column by the time Spark sees the lambda, so
# expressions around it are ordinary JVM work.
df.select(sf.transform("values", lambda x: plus_one(x) * 2).alias("doubled")).show()
