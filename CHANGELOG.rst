Changelog
=========

7.6.0a4 (2020-03-23)
--------------------

* Fixed issue in ``DataFrame.info()`` when called on an empty frame (`#135`_)
* Fixed issues where many ``_source`` fields would generate
  a ``too_long_frame`` error (`#135`_, `#137`_)
* Changed requirement for ``xgboost`` from ``>=0.90`` to ``==0.90``

 .. _#135: https://github.com/elastic/eland/pull/135
 .. _#137: https://github.com/elastic/eland/pull/137
