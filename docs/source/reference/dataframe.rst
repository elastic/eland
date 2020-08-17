.. _api.dataframe:

=========
DataFrame
=========
.. currentmodule:: eland

Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame

Attributes and underlying data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.index
   DataFrame.columns
   DataFrame.dtypes
   DataFrame.select_dtypes
   DataFrame.values
   DataFrame.empty
   DataFrame.shape
   DataFrame.ndim
   DataFrame.size

Indexing, iteration
~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.head
   DataFrame.keys
   DataFrame.tail
   DataFrame.get
   DataFrame.query
   DataFrame.sample

Function application, GroupBy & window
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.agg
   DataFrame.aggregate

.. _api.dataframe.stats:

Computations / descriptive stats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.count
   DataFrame.describe
   DataFrame.info
   DataFrame.max
   DataFrame.mean
   DataFrame.min
   DataFrame.median
   DataFrame.mad
   DataFrame.std
   DataFrame.var
   DataFrame.sum
   DataFrame.nunique

Reindexing / selection / label manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.drop
   DataFrame.filter

Plotting
~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.hist

Elasticsearch Functions
~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.es_info
   DataFrame.es_query

Serialization / IO / conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.info
   DataFrame.to_numpy
   DataFrame.to_csv
   DataFrame.to_html
   DataFrame.to_string
   DataFrame.to_pandas
