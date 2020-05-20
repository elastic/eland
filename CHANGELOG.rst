Changelog
=========

7.7.0a1 (2020-05-20)
--------------------

* Added the package to Conda Forge, install via
  ``conda install -c conda-forge eland`` (`#209`_)
* Added ``DataFrame.sample()`` and ``Series.sample()`` for querying
  a random sample of data from the index (`#196`_, contributed by `@mesejo`_)
* Added ``Series.isna()`` and ``Series.notna()`` for filtering out
  missing, ``NaN`` or null values from a column (`#210`_, contributed by `@mesejo`_)
* Added ``DataFrame.filter()`` and ``Series.filter()`` for reducing an axis
  using a sequence of items or a pattern (`#212`_)
* Added ``DataFrame.to_pandas()`` and ``Series.to_pandas()`` for converting
  an Eland dataframe or series into a Pandas dataframe or series inline (`#208`_)
* Added support for XGBoost v1.0.0 (`#200`_)
* Deprecated ``info_es()`` in favor of ``es_info()`` (`#208`_)
* Deprecated ``eland.read_csv()`` in favor of ``eland.csv_to_eland()`` (`#208`_)
* Deprecated ``eland.read_es()`` in favor of ``eland.DataFrame()`` (`#208`_)
* Fixed ``DeprecationWarning`` raised from ``pandas.Series`` when an
  an empty series was created without specifying ``dtype`` (`#188`_, contributed by `@mesejo`_)
* Fixed a bug when filtering columns on complex combinations of and and or (`#204`_)
* Fixed an issue where ``DataFrame.shape`` would return a larger value than
  in the index if a sized operation like ``.head(X)`` was applied to the data
  frame (`#205`_, contributed by `@mesejo`_)
* Fixed issue where both ``scikit-learn`` and ``xgboost`` libraries were
  required to use ``eland.ml.ImportedMLModel``, now only one library is
  required to use this feature (`#206`_)
* Changed ``var`` and ``std`` aggregations to use sample instead of
  population in line with Pandas (`#185`_)
* Changed painless scripts to use ``source`` rather than ``inline`` to improve
  script caching performance (`#191`_, contributed by `@mesejo`_)
* Changed minimum ``elasticsearch`` Python library version to v7.7.0 (`#207`_)
* Changed name of ``Index.field_name`` to ``Index.es_field_name`` (`#208`_)

 .. _#200: https://github.com/elastic/eland/pull/200
 .. _#201: https://github.com/elastic/eland/pull/201
 .. _#204: https://github.com/elastic/eland/pull/204
 .. _#205: https://github.com/elastic/eland/pull/205
 .. _#206: https://github.com/elastic/eland/pull/206
 .. _#207: https://github.com/elastic/eland/pull/207
 .. _#191: https://github.com/elastic/eland/pull/191
 .. _#210: https://github.com/elastic/eland/pull/210
 .. _#185: https://github.com/elastic/eland/pull/185
 .. _#188: https://github.com/elastic/eland/pull/188
 .. _#196: https://github.com/elastic/eland/pull/196
 .. _#208: https://github.com/elastic/eland/pull/208
 .. _#209: https://github.com/elastic/eland/pull/209
 .. _#212: https://github.com/elastic/eland/pull/212

7.6.0a5 (2020-04-14)
--------------------

* Added support for Pandas v1.0.0 (`#141`_, contributed by `@mesejo`_)
* Added ``use_pandas_index_for_es_ids`` parameter to ``pandas_to_eland()`` (`#154`_)
* Added ``es_type_overrides`` parameter to ``pandas_to_eland()`` (`#181`_)
* Added ``NDFrame.var()``, ``.std()`` and ``.median()`` aggregations (`#175`_, `#176`_, contributed by `@mesejo`_)
* Added ``DataFrame.es_query()`` to allow modifying ES queries directly (`#156`_)
* Added ``eland.__version__`` (`#153`_, contributed by `@mesejo`_)
* Removed support for Python 3.5 (`#150`_)
* Removed ``eland.Client()`` interface, use
  ``elasticsearch.Elasticsearch()`` client instead (`#166`_)
* Removed all private objects from top-level ``eland`` namespace (`#170`_)
* Removed ``geo_points`` from ``pandas_to_eland()`` in favor of ``es_type_overrides`` (`#181`_)
* Fixed ``inference_config`` being required on ML models for ES >=7.8 (`#174`_)
* Fixed unpacking for ``DataFrame.aggregate("median")`` (`#161`_)
* Changed ML model serialization to be slightly smaller (`#159`_)
* Changed minimum ``elasticsearch`` Python library version to v7.6.0 (`#181`_)

 .. _@mesejo: https://github.com/mesejo
 .. _#141: https://github.com/elastic/eland/pull/141
 .. _#150: https://github.com/elastic/eland/pull/150
 .. _#153: https://github.com/elastic/eland/pull/153
 .. _#154: https://github.com/elastic/eland/pull/154
 .. _#156: https://github.com/elastic/eland/pull/156
 .. _#159: https://github.com/elastic/eland/pull/159
 .. _#161: https://github.com/elastic/eland/pull/161
 .. _#166: https://github.com/elastic/eland/pull/166
 .. _#170: https://github.com/elastic/eland/pull/170
 .. _#174: https://github.com/elastic/eland/pull/174
 .. _#175: https://github.com/elastic/eland/pull/175
 .. _#176: https://github.com/elastic/eland/pull/176
 .. _#181: https://github.com/elastic/eland/pull/181

7.6.0a4 (2020-03-23)
--------------------

* Fixed issue in ``DataFrame.info()`` when called on an empty frame (`#135`_)
* Fixed issues where many ``_source`` fields would generate
  a ``too_long_frame`` error (`#135`_, `#137`_)
* Changed requirement for ``xgboost`` from ``>=0.90`` to ``==0.90``

 .. _#135: https://github.com/elastic/eland/pull/135
 .. _#137: https://github.com/elastic/eland/pull/137
