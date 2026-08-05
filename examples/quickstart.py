"""Native higher-order functions with a Python UDF.

Run against a local session::

    python examples/quickstart.py

or against Databricks serverless by replacing the session setup with::

    from databricks.connect import DatabricksSession
    spark = DatabricksSession.builder.serverless().getOrCreate()
"""

from pyspark.sql import SparkSession

# Aliased to eudf so it is never confused with pyspark.sql.functions.udf; this
# one additionally works inside a higher-order function's lambda.
from elementwise_udf import udf as eudf, functions as F

spark = SparkSession.builder.master("local[2]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = spark.createDataFrame([([1, 2, 3],)], ["values"])

# A native higher-order function, no UDF involved: unchanged.
df.select(F.transform("values", lambda x: x + 1).alias("result")).show()


@eudf("long")
def plus_one(x):
    return x + 1


# An ordinary UDF call, exactly as with pyspark.sql.functions.udf("long").
spark.range(1).select(plus_one(F.lit(1))).show()

# The same native higher-order function, now with the Python UDF inside it.
df.select(F.transform("values", lambda x: plus_one(x)).alias("result")).show()

# The UDF result is a plain column by the time Spark sees the lambda, so
# expressions around it are ordinary JVM work.
df.select(F.transform("values", lambda x: plus_one(x) * 2).alias("doubled")).show()
