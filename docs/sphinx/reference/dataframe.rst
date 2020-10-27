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

Attributes and Underlying Data
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

Indexing, Iteration
~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.head
   DataFrame.keys
   DataFrame.tail
   DataFrame.get
   DataFrame.query
   DataFrame.sample

Function Application, GroupBy & Window
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.agg
   DataFrame.aggregate
   DataFrame.groupby

.. currentmodule:: eland.groupby

.. autosummary::
   :toctree: api/

   DataFrameGroupBy
   DataFrameGroupBy.agg
   DataFrameGroupBy.aggregate
   DataFrameGroupBy.count
   DataFrameGroupBy.mad
   DataFrameGroupBy.max
   DataFrameGroupBy.mean
   DataFrameGroupBy.median
   DataFrameGroupBy.min
   DataFrameGroupBy.nunique
   DataFrameGroupBy.std
   DataFrameGroupBy.sum
   DataFrameGroupBy.var
   GroupBy

.. currentmodule:: eland

.. _api.dataframe.stats:

Computations / Descriptive Stats
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

Reindexing / Selection / Label Manipulation
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
   DataFrame.es_dtypes

Serialization / IO / Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.info
   DataFrame.to_numpy
   DataFrame.to_csv
   DataFrame.to_html
   DataFrame.to_string
   DataFrame.to_pandas
