# Changelog

All notable changes to **elementwise-udf** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> Unofficial personal project. Not affiliated with, sponsored by, or endorsed by
> the Apache Software Foundation. "Apache Spark", "Spark", and "PySpark" are
> trademarks of the ASF, used here only to describe interoperability.

Distribution name on PyPI: `elementwise-udf`. Import name: `elementwise_udf`.

<!--
Release runbook:

  1. Green on `main`: the `ci` workflow, all Spark x mode combinations.
  2. Move the "Unreleased" entries into a new "## [X.Y.Z] - YYYY-MM-DD" section.
     Keep the heading shape EXACT - release.yml's awk extractor matches
     `^## \[X.Y.Z\]` and copies until the next `## [`.
  3. Bump `version` in pyproject.toml AND `__version__` in
     elementwise_udf/__init__.py to X.Y.Z.
  4. Commit, then tag: `git tag vX.Y.Z && git push origin vX.Y.Z`.
     release.yml enforces that the tag matches the packaged version.
  5. The workflow builds, re-runs the suite on the oldest and newest supported
     Spark, publishes to PyPI with the `PYPI_TOKEN` secret, and cuts a GitHub
     Release with this section as the notes.

  Dry run without touching PyPI: run the `release` workflow manually with
  dry_run=build-only, or dry_run=testpypi to upload to TestPyPI via OIDC.
-->

## [Unreleased]

### Added
- _Nothing yet._

## [0.2.0] - 2026-08-05

### Fixed
- A UDF in a fold's `finish` lambda was applied to the fold's result even when
  that result was null, so a null-unaware UDF raised where plain PySpark returns
  null. Native Spark does not evaluate `finish` for a null array, and the rewrite
  now matches: `aggregate(v, 0, merge, lambda a: my_udf(a))` over a null array
  returns null instead of failing. Wrapping the call in `when` does not fix this,
  because Spark evaluates both branches, so the guard is built into the UDF.

  Found by running the package from PyPI on a real Databricks Shared cluster
  (`USER_ISOLATION`) across DBR 17.3, 18.3 and 19 - 23 of 24 checks passed, and
  this was the one that did not. Local tests had covered the `finish` lambda and
  null arrays, but never both at once.

### Changed
- `pyspark` is no longer a hard dependency, so `pip install elementwise-udf`
  leaves the environment's own pyspark alone. Use the `spark` or `connect` extra
  to install one.

## [0.1.0] - 2026-08-05

### Added
- `elementwise_udf.functions`, a drop-in for `pyspark.sql.functions`. Every name
  is delegated to the real module; `esf.udf` builds a UDF that may be called
  inside a native higher-order function's lambda, which Spark otherwise rejects
  ([SPARK-27052](https://issues.apache.org/jira/browse/SPARK-27052)), and such a
  call is rewritten so the UDF sits outside the lambda. Nothing in PySpark is
  patched, and lambdas using no element-wise UDF are forwarded untouched.
- Support for every higher-order function PySpark has, meaning every function
  taking a lambda: `transform`, `filter`, `exists`, `forall`, `zip_with`,
  `aggregate`, `reduce`, `array_sort`, `transform_keys`, `transform_values`,
  `map_filter` and `map_zip_with`. For a fold, a UDF may be in the `merge`
  lambda, the `finish` lambda, or both.
- Arbitrary expressions around a UDF result - arithmetic, `esf.when`, casts,
  comparisons with other columns, the element index, nested UDF calls, several
  UDFs in one lambda, and UDF arguments that are themselves expressions over the
  element.
- `pyspark` is not a hard dependency: the package is installed into an existing
  Spark environment that provides its own. The `spark` and `connect` extras pull
  one in for local use.

### Notes
- Two shapes cannot be precomputed, because the value the UDF needs does not
  exist until the higher-order function is already running. Both work by moving
  the whole operation into one Python call per row, and both warn
  (`RuntimeWarning`): a UDF on `aggregate`'s accumulator, and a genuinely
  pairwise comparator. Applying the UDF to the element keeps it on the fast path.
- Verified on classic PySpark and Spark Connect (including Databricks Connect
  serverless), across Spark 4.0, 4.1 and 4.2: 271 tests per session mode, 99%
  statement coverage.

