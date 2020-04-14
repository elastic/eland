Changelog
=========

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
