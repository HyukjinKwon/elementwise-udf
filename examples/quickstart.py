"""Native higher-order functions with a Python UDF.

Run against a local session::

    python examples/quickstart.py

or against Databricks serverless by replacing the session setup with::

    from databricks.connect import DatabricksSession
    spark = DatabricksSession.builder.serverless().getOrCreate()

``elementwise_udf.functions`` stands in for ``pyspark.sql.functions``: every name
is delegated to the real module, so only the import line differs from ordinary
PySpark. ``esf.udf`` builds a UDF that may be called inside a higher-order
function's lambda, and such a call is rewritten so the UDF sits outside the
lambda, which is the only place Spark allows it. Importing
``pyspark.sql.functions`` directly leaves the call unrewritten and Spark rejects
it.
"""

from pyspark.sql import SparkSession

import elementwise_udf.functions as esf

spark = SparkSession.builder.master("local[2]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = spark.createDataFrame([([1, 2, 3],)], ["values"])

# A native higher-order function, no UDF involved: unchanged.
df.select(esf.transform("values", lambda x: x + 1).alias("result")).show()


@esf.udf("long")
def plus_one(x):
    return x + 1


# An ordinary UDF call, exactly as with pyspark.sql.functions.udf("long").
spark.range(1).select(plus_one(esf.lit(1))).show()

# The same native higher-order function, now with the Python UDF inside it.
df.select(esf.transform("values", lambda x: plus_one(x)).alias("result")).show()

# The UDF result is a plain column by the time Spark sees the lambda, so
# expressions around it are ordinary JVM work.
df.select(esf.transform("values", lambda x: plus_one(x) * 2).alias("doubled")).show()
