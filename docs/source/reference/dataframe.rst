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
**Axes**

.. autosummary::
   :toctree: api/

   DataFrame.index
   DataFrame.columns
   DataFrame.dtypes   
   DataFrame.select_dtypes   
   DataFrame.empty   

Indexing, iteration
~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.head
   DataFrame.tail
   DataFrame.get

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

Reindexing / selection / label manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.drop

Plotting
~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.hist

Serialization / IO / conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.info

