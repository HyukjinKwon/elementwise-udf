"""Session fixture.

The rewrite has to hold on both classic PySpark and Spark Connect, but the two
session kinds cannot coexist in one process (``SESSION_ALREADY_EXIST``), so the
mode is chosen per run via ``SPARK_MODE``:

    SPARK_MODE=classic pytest
    SPARK_MODE=connect pytest

``tox``/CI runs both. Unset, it defaults to classic.
"""

import os

import pytest


@pytest.fixture(scope="session")
def spark():
    from pyspark.sql import SparkSession

    mode = os.environ.get("SPARK_MODE", "classic")
    builder = SparkSession.builder
    if mode == "connect":
        try:
            session = builder.remote("local[2]").getOrCreate()
        except Exception as exc:  # pragma: no cover - depends on extras
            pytest.skip(f"Spark Connect unavailable: {exc}")
    else:
        session = builder.master("local[2]").getOrCreate()
    try:
        session.sparkContext.setLogLevel("ERROR")
    except Exception:
        pass  # Not available on a Connect session.
    yield session


@pytest.fixture
def values(spark):
    """One array column covering the populated, empty and null cases."""
    return spark.createDataFrame([([1, 2, 3],), ([],), (None,)], "v array<int>")
