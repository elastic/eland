.. _api.series:

=========
Series
=========
.. currentmodule:: eland

Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Series

Attributes and underlying data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Axes**

.. autosummary::
   :toctree: api/

   Series.index
   Series.shape
   Series.name   
   Series.empty   

Indexing, iteration
~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Series.head
   Series.tail

Binary operator functions
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Series.add
   Series.sub
   Series.mul
   Series.div
   Series.truediv
   Series.floordiv
   Series.mod
   Series.pow

Computations / descriptive stats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Series.describe
   Series.max
   Series.mean
   Series.min
   Series.sum
   Series.nunique
   Series.value_counts

Reindexing / selection / label manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Series.rename

Serialization / IO / conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Series.to_string
   Series.to_numpy

Elasticsearch utilities
~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Series.info_es
