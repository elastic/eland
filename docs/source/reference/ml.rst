.. _api.ml:

================
Machine Learning
================
.. currentmodule:: eland.ml

Machine learning is built into the Elastic Stack and enables users to gain insights into their Elasticsearch data. 
There are a wide range of capabilities from identifying in
anomalies in your data, to training and deploying regression or classification models based on Elasticsearch data.

To use the Elastic Stack machine learning features, you must have the appropriate license and at least one machine 
learning node in your Elasticsearch cluster. If Elastic Stack security features are enabled, you must also ensure 
your users have the necessary privileges.

The fastest way to get started with machine learning features is to
`start a free 14-day trial of Elastic Cloud <https://www.elastic.co/cloud/elasticsearch-service/signup>`_.

See `Elasticsearch Machine Learning documentation <https://www.elastic.co/guide/en/machine-learning/current/setup.html>`_ more details.

ImportedMLModel
~~~~~~~~~~~~~~~
.. currentmodule:: eland.ml

Constructor
^^^^^^^^^^^
.. autosummary::
   :toctree: api/

   ImportedMLModel

Predictions
^^^^^^^^^^^
.. autosummary::
   :toctree: api/

   ImportedMLModel.predict

Manage Models
^^^^^^^^^^^^^
.. autosummary::
   :toctree: api/

   MLModel.exists_model
   MLModel.delete_model
