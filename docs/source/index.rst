.. eland documentation master file, created by

.. module:: eland

****************************************************************
eland: pandas-like data analysis toolkit backed by Elasticsearch
****************************************************************

**Date**: |today| **Version**: |version|

**Useful links**:
`Source Repository <https://github.com/elastic/eland>`__ |
`Issues & Ideas <https://github.com/elastic/eland/issues>`__ |
`Q&A Support <https://discuss.elastic.co>`__ |

:mod:`eland` is an open source, Apache2-licensed elasticsearch Python client to analyse, explore and manipulate data that resides in elasticsearch.
Where possible the package uses existing Python APIs and data structures to make it easy to switch between Numpy, Pandas, Scikit-learn to their elasticsearch powered equivalents.
In general, the data resides in elasticsearch and not in memory, which allows eland to access large datasets stored in elasticsearch.


.. toctree::
   :maxdepth: 2
   :hidden:

   reference/index
   implementation/index

* :doc:`reference/index`

  * :doc:`reference/io`
  * :doc:`reference/general_utility_functions`
  * :doc:`reference/dataframe`
  * :doc:`reference/indexing`

* :doc:`implementation/index`

  * :doc:`implementation/details`
  * :doc:`implementation/dataframe_supported`

