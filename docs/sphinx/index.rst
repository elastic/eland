.. module:: eland

**************************************************************
Eland: DataFrames and Machine Learning backed by Elasticsearch
**************************************************************

**Date**: |today| **Version**: |version|

**Useful links**:
`Source Repository <https://github.com/elastic/eland>`__ |
`Issues & Ideas <https://github.com/elastic/eland/issues>`__ |
`Q&A Support <https://discuss.elastic.co>`__

Eland is a Python Elasticsearch client for exploring and analyzing data
in Elasticsearch with a familiar Pandas-compatible API.

Where possible the package uses existing Python APIs and data structures to make it easy to switch between numpy,
pandas, scikit-learn to their Elasticsearch powered equivalents. In general, the data resides in Elasticsearch and
not in memory, which allows Eland to access large datasets stored in Elasticsearch.

Installing Eland
~~~~~~~~~~~~~~~~

Eland can be installed from `PyPI <https://pypi.org/project/eland>`_ via pip:

 .. code-block:: bash

    $ python -m pip install eland

Eland can also be installed from `Conda Forge <https://anaconda.org/conda-forge/eland>`_ with Conda:

 .. code-block:: bash

    $ conda install -c conda-forge eland

Getting Started
~~~~~~~~~~~~~~~

If it's your first time using Eland we recommend looking through the
:doc:`examples/index` documentation for ideas on what Eland is capable of.

If you're new to Elasticsearch we recommend `reading the documentation <https://www.elastic.co/elasticsearch>`_.

.. toctree::
   :maxdepth: 2
   :hidden:

   reference/index
   reference/ml
   examples/index
   development/index

* :doc:`reference/index`

  * :doc:`reference/supported_apis`
  * :doc:`reference/dataframe`
  * :doc:`reference/series`
  * :doc:`reference/ml`
  * :doc:`reference/indexing`
  * :doc:`reference/general_utility_functions`
  * :doc:`reference/io`

* :doc:`development/index`

  * :doc:`development/contributing`
  * :doc:`development/implementation`

* :doc:`examples/index`

  * :doc:`examples/demo_notebook`
  * :doc:`examples/introduction_to_eland_webinar`
  * :doc:`examples/online_retail_analysis`
