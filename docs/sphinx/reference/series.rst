.. _api.series:

======
Series
======
.. currentmodule:: eland

Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Series

Attributes and Underlying Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Series.index
   Series.dtype
   Series.dtypes
   Series.shape
   Series.name
   Series.empty
   Series.ndim
   Series.size

Indexing, Iteration
~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Series.head
   Series.tail
   Series.sample

Binary Operator Functions
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Series.add
   Series.sub
   Series.subtract
   Series.mul
   Series.multiply
   Series.div
   Series.divide
   Series.truediv
   Series.floordiv
   Series.mod
   Series.pow
   Series.radd
   Series.rsub
   Series.rsubtract
   Series.rmul
   Series.rmultiply
   Series.rdiv
   Series.rdivide
   Series.rtruediv
   Series.rfloordiv
   Series.rmod
   Series.rpow

Computations / Descriptive Stats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Series.describe
   Series.max
   Series.mean
   Series.min
   Series.sum
   Series.median
   Series.mad
   Series.std
   Series.var
   Series.nunique
   Series.value_counts
   Series.mode

Reindexing / Selection / Label Manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Series.rename
   Series.isna
   Series.notna
   Series.isnull
   Series.notnull
   Series.isin
   Series.filter

Plotting
~~~~~~~~
.. autosummary::
   :toctree: api/

   Series.hist

Serialization / IO / Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Series.to_string
   Series.to_numpy
   Series.to_pandas

Elasticsearch Functions
~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Series.es_info
   Series.es_match
   Series.es_dtype
   Series.es_dtypes
