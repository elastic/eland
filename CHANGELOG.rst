=========
Changelog
=========

8.0.0 (2022-02-10)
------------------

Added
^^^^^

* Added support for Natural Language Processing (NLP) models using PyTorch (`#394`_)
* Added new extra ``eland[pytorch]`` for installing all dependencies needed for PyTorch (`#394`_)
* Added a CLI script ``eland_import_hub_model`` for uploading HuggingFace models to Elasticsearch (`#403`_)
* Added support for v8.0 of the Python Elasticsearch client (`#415`_)
* Added a warning if Eland detects it's communicating with an incompatible Elasticsearch version (`#419`_)
* Added support for ``number_samples`` to LightGBM and Scikit-Learn models (`#397`_, contributed by `@V1NAY8`_)
* Added ability to use datetime types for filtering dataframes (`284`_, contributed by `@Fju`_)
* Added pandas ``datetime64`` type to use the Elasticsearch ``date`` type (`#425`_, contributed by `@Ashton-Sidhu`_)
* Added ``es_verify_mapping_compatibility`` parameter to disable schema enforcement with ``pandas_to_eland`` (`#423`_, contributed by `@Ashton-Sidhu`_)

Changed
^^^^^^^

* Changed ``to_pandas()`` to only use Point-in-Time and ``search_after`` instead of using Scroll APIs
  for pagination.

.. _@Fju: https://github.com/Fju
.. _@Ashton-Sidhu: https://github.com/Ashton-Sidhu
.. _#419: https://github.com/elastic/eland/pull/419
.. _#415: https://github.com/elastic/eland/pull/415
.. _#397: https://github.com/elastic/eland/pull/397
.. _#394: https://github.com/elastic/eland/pull/394
.. _#403: https://github.com/elastic/eland/pull/403
.. _#284: https://github.com/elastic/eland/pull/284
.. _#424: https://github.com/elastic/eland/pull/425
.. _#423: https://github.com/elastic/eland/pull/423


7.14.1b1 (2021-08-30)
---------------------

Added
^^^^^

* Added support for ``DataFrame.iterrows()`` and ``DataFrame.itertuples()`` (`#380`_, contributed by `@kxbin`_)

Performance
^^^^^^^^^^^

* Simplified result collectors to increase performance transforming Elasticsearch results to pandas (`#378`_, contributed by `@V1NAY8`_)
* Changed search pagination function to yield batches of hits (`#379`_)

.. _@kxbin: https://github.com/kxbin
.. _#378: https://github.com/elastic/eland/pull/378
.. _#379: https://github.com/elastic/eland/pull/379
.. _#380: https://github.com/elastic/eland/pull/380


7.14.0b1 (2021-08-09)
---------------------

Added
^^^^^

* Added support for Pandas 1.3.x (`#362`_, contributed by `@V1NAY8`_)
* Added support for LightGBM 3.x (`#362`_, contributed by `@V1NAY8`_)
* Added ``DataFrame.idxmax()`` and ``DataFrame.idxmin()`` methods (`#353`_, contributed by `@V1NAY8`_)
* Added type hints to ``eland.ndframe`` and ``eland.operations`` (`#366`_, contributed by `@V1NAY8`_)

Removed
^^^^^^^

* Removed support for Pandas <1.2 (`#364`_)
* Removed support for Python 3.6 to match Pandas (`#364`_)

Changed
^^^^^^^

* Changed paginated search function to use `Point-in-Time`_ and `Search After`_ features
  instead of Scroll when connected to Elasticsearch 7.12+ (`#370`_ and `#376`_, contributed by `@V1NAY8`_)
* Optimized the ``FieldMappings.aggregate_field_name()`` method (`#373`_, contributed by `@V1NAY8`_)

 .. _Point-in-Time: https://www.elastic.co/guide/en/elasticsearch/reference/current/point-in-time-api.html
 .. _Search After: https://www.elastic.co/guide/en/elasticsearch/reference/7.14/paginate-search-results.html#search-after
 .. _#353: https://github.com/elastic/eland/pull/353 
 .. _#362: https://github.com/elastic/eland/pull/362
 .. _#364: https://github.com/elastic/eland/pull/364
 .. _#366: https://github.com/elastic/eland/pull/366
 .. _#370: https://github.com/elastic/eland/pull/370
 .. _#373: https://github.com/elastic/eland/pull/373
 .. _#376: https://github.com/elastic/eland/pull/376


7.13.0b1 (2021-06-22)
---------------------

Added
^^^^^

* Added ``DataFrame.quantile()``, ``Series.quantile()``, and
  ``DataFrameGroupBy.quantile()`` aggregations (`#318`_ and `#356`_, contributed by `@V1NAY8`_)

Changed
^^^^^^^

* Changed the error raised when ``es_index_pattern`` doesn't point to any indices
  to be more user-friendly (`#346`_)

Fixed
^^^^^

* Fixed a warning about conflicting field types when wildcards are used
  in ``es_index_pattern`` (`#346`_)

* Fixed sorting when using ``DataFrame.groupby()`` with ``dropna``
  (`#322`_, contributed by `@V1NAY8`_)

* Fixed deprecated usage ``numpy.int`` in favor of ``numpy.int_`` (`#354`_, contributed by `@V1NAY8`_)

 .. _#318: https://github.com/elastic/eland/pull/318
 .. _#322: https://github.com/elastic/eland/pull/322
 .. _#346: https://github.com/elastic/eland/pull/346
 .. _#354: https://github.com/elastic/eland/pull/354
 .. _#356: https://github.com/elastic/eland/pull/356


7.10.1b1 (2021-01-12)
---------------------

Added
^^^^^

* Added support for Pandas 1.2.0 (`#336`_)

* Added ``DataFrame.mode()`` and ``Series.mode()`` aggregation (`#323`_, contributed by `@V1NAY8`_)

* Added support for ``pd.set_option("display.max_rows", None)``
  (`#308`_, contributed by `@V1NAY8`_)

* Added Elasticsearch storage usage to ``df.info()`` (`#321`_, contributed by `@V1NAY8`_)

Removed
^^^^^^^

* Removed deprecated aliases ``read_es``, ``read_csv``, ``DataFrame.info_es``,
  and ``MLModel(overwrite=True)`` (`#331`_, contributed by `@V1NAY8`_)

 .. _#336: https://github.com/elastic/eland/pull/336
 .. _#331: https://github.com/elastic/eland/pull/331
 .. _#323: https://github.com/elastic/eland/pull/323
 .. _#321: https://github.com/elastic/eland/pull/321
 .. _#308: https://github.com/elastic/eland/pull/308


7.10.0b1 (2020-10-29)
---------------------

Added
^^^^^

* Added ``DataFrame.groupby()`` method with all aggregations
  (`#278`_, `#291`_, `#292`_, `#300`_ contributed by `@V1NAY8`_)

* Added ``es_match()`` method to ``DataFrame`` and ``Series`` for
  filtering rows with full-text search (`#301`_)

* Added support for type hints of the ``elasticsearch-py`` package (`#295`_)

* Added support for passing dictionaries to ``es_type_overrides`` parameter
  in the ``pandas_to_eland()`` function to directly control the field mapping
  generated in Elasticsearch (`#310`_)

* Added ``es_dtypes`` property to ``DataFrame`` and ``Series`` (`#285`_) 

Changed
^^^^^^^

* Changed ``pandas_to_eland()`` to use the ``parallel_bulk()``
  helper instead of single-threaded ``bulk()`` helper to improve
  performance (`#279`_, contributed by `@V1NAY8`_)

* Changed the ``es_type_overrides`` parameter in ``pandas_to_eland()``
  to raise ``ValueError`` if an unknown column is given (`#302`_)

* Changed ``DataFrame.filter()`` to preserve the order of items
  (`#283`_, contributed by `@V1NAY8`_)

* Changed when setting ``es_type_overrides={"column": "text"}`` in
  ``pandas_to_eland()`` will automatically add the ``column.keyword``
  sub-field so that aggregations are available for the field as well (`#310`_)

Fixed
^^^^^

* Fixed ``Series.__repr__`` when the series is empty (`#306`_)

 .. _#278: https://github.com/elastic/eland/pull/278
 .. _#279: https://github.com/elastic/eland/pull/279
 .. _#283: https://github.com/elastic/eland/pull/283
 .. _#285: https://github.com/elastic/eland/pull/285
 .. _#291: https://github.com/elastic/eland/pull/291
 .. _#292: https://github.com/elastic/eland/pull/292
 .. _#295: https://github.com/elastic/eland/pull/295
 .. _#300: https://github.com/elastic/eland/pull/300
 .. _#301: https://github.com/elastic/eland/pull/301
 .. _#302: https://github.com/elastic/eland/pull/302
 .. _#306: https://github.com/elastic/eland/pull/306
 .. _#310: https://github.com/elastic/eland/pull/310


7.9.1a1 (2020-09-29)
--------------------

Added
^^^^^

* Added the ``predict()`` method and ``model_type``,
  ``feature_names``, and ``results_field`` properties
  to ``MLModel``  (`#266`_)


Deprecated
^^^^^^^^^^

* Deprecated ``ImportedMLModel`` in favor of
  ``MLModel.import_model(...)`` (`#266`_)


Changed
^^^^^^^

* Changed DataFrame aggregations to use ``numeric_only=None``
  instead of ``numeric_only=True`` by default. This is the same
  behavior as Pandas (`#270`_, contributed by `@V1NAY8`_)

Fixed
^^^^^

* Fixed ``DataFrame.agg()`` when given a string instead of a list of
  aggregations will now properly return a ``Series`` instead of
  a ``DataFrame`` (`#263`_, contributed by `@V1NAY8`_)


 .. _#263: https://github.com/elastic/eland/pull/263
 .. _#266: https://github.com/elastic/eland/pull/266
 .. _#270: https://github.com/elastic/eland/pull/270


7.9.0a1 (2020-08-18)
--------------------

Added
^^^^^

* Added support for Pandas v1.1 (`#253`_)
* Added support for LightGBM ``LGBMRegressor`` and ``LGBMClassifier`` to ``ImportedMLModel`` (`#247`_, `#252`_)
* Added support for ``multi:softmax`` and ``multi:softprob`` XGBoost operators to ``ImportedMLModel`` (`#246`_)
* Added column names to ``DataFrame.__dir__()`` for better auto-completion support (`#223`_, contributed by `@leonardbinet`_)
* Added support for ``es_if_exists='append'`` to ``pandas_to_eland()`` (`#217`_)
* Added support for aggregating datetimes with ``nunique`` and ``mean`` (`#253`_)
* Added ``es_compress_model_definition`` parameter to ``ImportedMLModel`` constructor (`#220`_)
* Added ``.size`` and ``.ndim`` properties to ``DataFrame`` and ``Series`` (`#231`_ and `#233`_)
* Added ``.dtype`` property to ``Series`` (`#258`_)
* Added support for using ``pandas.Series`` with ``Series.isin()`` (`#231`_)
* Added type hints to many APIs in ``DataFrame`` and ``Series`` (`#231`_)

Deprecated
^^^^^^^^^^

* Deprecated  the ``overwrite`` parameter in favor of ``es_if_exists`` in ``ImportedMLModel`` constructor (`#249`_, contributed by `@V1NAY8`_)

Changed
^^^^^^^

* Changed aggregations for datetimes to be higher precision when available (`#253`_)

Fixed
^^^^^

* Fixed ``ImportedMLModel.predict()`` to fail when ``errors`` are present in the ``ingest.simulate`` response (`#220`_)
* Fixed ``Series.median()`` aggregation to return a scalar instead of ``pandas.Series`` (`#253`_)
* Fixed ``Series.describe()`` to return a ``pandas.Series`` instead of ``pandas.DataFrame`` (`#258`_)
* Fixed ``DataFrame.mean()`` and ``Series.mean()`` dtype (`#258`_)
* Fixed ``DataFrame.agg()`` aggregations when using ``extended_stats`` Elasticsearch aggregation (`#253`_)

 .. _@leonardbinet: https://github.com/leonardbinet
 .. _@V1NAY8: https://github.com/V1NAY8
 .. _#217: https://github.com/elastic/eland/pull/217
 .. _#220: https://github.com/elastic/eland/pull/220
 .. _#223: https://github.com/elastic/eland/pull/223
 .. _#231: https://github.com/elastic/eland/pull/231
 .. _#233: https://github.com/elastic/eland/pull/233
 .. _#246: https://github.com/elastic/eland/pull/246
 .. _#247: https://github.com/elastic/eland/pull/247
 .. _#249: https://github.com/elastic/eland/pull/249
 .. _#252: https://github.com/elastic/eland/pull/252
 .. _#253: https://github.com/elastic/eland/pull/253
 .. _#258: https://github.com/elastic/eland/pull/258


7.7.0a1 (2020-05-20)
--------------------

Added
^^^^^

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

Deprecated
^^^^^^^^^^

* Deprecated ``info_es()`` in favor of ``es_info()`` (`#208`_)
* Deprecated ``eland.read_csv()`` in favor of ``eland.csv_to_eland()`` (`#208`_)
* Deprecated ``eland.read_es()`` in favor of ``eland.DataFrame()`` (`#208`_)

Changed
^^^^^^^

* Changed ``var`` and ``std`` aggregations to use sample instead of
  population in line with Pandas (`#185`_)
* Changed painless scripts to use ``source`` rather than ``inline`` to improve
  script caching performance (`#191`_, contributed by `@mesejo`_)
* Changed minimum ``elasticsearch`` Python library version to v7.7.0 (`#207`_)
* Changed name of ``Index.field_name`` to ``Index.es_field_name`` (`#208`_)

Fixed
^^^^^

* Fixed ``DeprecationWarning`` raised from ``pandas.Series`` when an
  an empty series was created without specifying ``dtype`` (`#188`_, contributed by `@mesejo`_)
* Fixed a bug when filtering columns on complex combinations of and and or (`#204`_)
* Fixed an issue where ``DataFrame.shape`` would return a larger value than
  in the index if a sized operation like ``.head(X)`` was applied to the data
  frame (`#205`_, contributed by `@mesejo`_)
* Fixed issue where both ``scikit-learn`` and ``xgboost`` libraries were
  required to use ``eland.ml.ImportedMLModel``, now only one library is
  required to use this feature (`#206`_)

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

Added
^^^^^

* Added support for Pandas v1.0.0 (`#141`_, contributed by `@mesejo`_)
* Added ``use_pandas_index_for_es_ids`` parameter to ``pandas_to_eland()`` (`#154`_)
* Added ``es_type_overrides`` parameter to ``pandas_to_eland()`` (`#181`_)
* Added ``NDFrame.var()``, ``.std()`` and ``.median()`` aggregations (`#175`_, `#176`_, contributed by `@mesejo`_)
* Added ``DataFrame.es_query()`` to allow modifying ES queries directly (`#156`_)
* Added ``eland.__version__`` (`#153`_, contributed by `@mesejo`_)

Removed
^^^^^^^

* Removed support for Python 3.5 (`#150`_)
* Removed ``eland.Client()`` interface, use
  ``elasticsearch.Elasticsearch()`` client instead (`#166`_)
* Removed all private objects from top-level ``eland`` namespace (`#170`_)
* Removed ``geo_points`` from ``pandas_to_eland()`` in favor of ``es_type_overrides`` (`#181`_)

Changed
^^^^^^^

* Changed ML model serialization to be slightly smaller (`#159`_)
* Changed minimum ``elasticsearch`` Python library version to v7.6.0 (`#181`_)

Fixed
^^^^^

* Fixed ``inference_config`` being required on ML models for ES >=7.8 (`#174`_)
* Fixed unpacking for ``DataFrame.aggregate("median")`` (`#161`_)

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

Changed
^^^^^^^

* Changed requirement for ``xgboost`` from ``>=0.90`` to ``==0.90``

Fixed
^^^^^

* Fixed issue in ``DataFrame.info()`` when called on an empty frame (`#135`_)
* Fixed issues where many ``_source`` fields would generate
  a ``too_long_frame`` error (`#135`_, `#137`_)

 .. _#135: https://github.com/elastic/eland/pull/135
 .. _#137: https://github.com/elastic/eland/pull/137
