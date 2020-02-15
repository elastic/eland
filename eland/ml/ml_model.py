#  Copyright 2019 Elasticsearch BV
#
#      Licensed under the Apache License, Version 2.0 (the "License");
#      you may not use this file except in compliance with the License.
#      You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#      Unless required by applicable law or agreed to in writing, software
#      distributed under the License is distributed on an "AS IS" BASIS,
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#      See the License for the specific language governing permissions and
#      limitations under the License.

import elasticsearch

from eland import Client

class MLModel:
    """
    A machine learning model managed by Elasticsearch.
    (See https://www.elastic.co/guide/en/elasticsearch/reference/master/put-inference.html)

    These models can be created by Elastic ML, or transformed from supported python formats such as scikit-learn or
    xgboost and imported into Elasticsearch.

    The methods for this class attempt to mirror standard python classes.
    """
    TYPE_CLASSIFICATION = "classification"
    TYPE_REGRESSION = "regression"

    def __init__(self,
                 es_client,
                 model_id: str):
        """
        Parameters
        ----------
        es_client: Elasticsearch client argument(s)
            - elasticsearch-py parameters or
            - elasticsearch-py instance or
            - eland.Client instance

        model_id: str
            The unique identifier of the trained inference model in Elasticsearch.
        """
        self._client = Client(es_client)
        self._model_id = model_id

    def delete_model(self):
        """
        Delete an inference model saved in Elasticsearch

        If model doesn't exist, ignore failure.
        """
        try:
            self._client.perform_request("DELETE", "/_ml/inference/" + self._model_id)
        except elasticsearch.exceptions.NotFoundError:
            pass
