#  Licensed to Elasticsearch B.V. under one or more contributor
#  license agreements. See the NOTICE file distributed with
#  this work for additional information regarding copyright
#  ownership. Elasticsearch B.V. licenses this file to you under
#  the Apache License, Version 2.0 (the "License"); you may
#  not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.

import elasticsearch
from eland.common import ensure_es_client


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

    def __init__(self, es_client, model_id: str):
        """
        Parameters
        ----------
        es_client: Elasticsearch client argument(s)
            - elasticsearch-py parameters or
            - elasticsearch-py instance

        model_id: str
            The unique identifier of the trained inference model in Elasticsearch.
        """
        self._client = ensure_es_client(es_client)
        self._model_id = model_id

    def delete_model(self) -> None:
        """
        Delete an inference model saved in Elasticsearch

        If model doesn't exist, ignore failure.
        """
        try:
            self._client.ml.delete_trained_model(model_id=self._model_id, ignore=(404,))
        except elasticsearch.NotFoundError:
            pass

    def check_existing_model(self) -> bool:
        """
        Check If model exists in Elastic
        """
        try:
            self._client.ml.get_trained_models(
                model_id=self._model_id, include_model_definition=False
            )
        except elasticsearch.NotFoundError:
            return False
        return True
