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
from typing import Union, List

from eland import Client
from eland.ml._model_transformers import SKLearnDecisionTreeTransformer, SKLearnForestRegressorTransformer, \
    SKLearnForestClassifierTransformer, XGBoostRegressorTransformer, XGBoostClassifierTransformer
from eland.ml._optional import import_optional_dependency

sklearn = import_optional_dependency("sklearn")
xgboost = import_optional_dependency("xgboost")

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBRegressor, XGBClassifier




class Model:
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

        Raises elasticsearch.NotFoundError if model_id does not exist

        Parameters
        ----------
        es_client: Elasticsearch client argument(s)
            - elasticsearch-py parameters or
            - elasticsearch-py instance or
            - eland.Client instance

        model_id: str
            The unique identifier of the trained inference model in Elasticsearch.
        """
        self._client.perform_request("DELETE", "/_ml/inference/" + self._model_id)

class ExternalModel(Model):
    """
    An external model that is transformed and added to Elasticsearch.
    """

    def __init__(self,
                 es_client,
                 model_id: str,
                 model: Union[DecisionTreeClassifier,
                          DecisionTreeRegressor,
                          RandomForestRegressor,
                          RandomForestClassifier,
                          XGBClassifier,
                          XGBRegressor],
                 feature_names: List[str],
                 classification_labels: List[str] = None,
                 classification_weights: List[float] = None):
        """
        Put a trained inference model in Elasticsearch based on an external model.

        Parameters
        ----------
        es_client: Elasticsearch client argument(s)
            - elasticsearch-py parameters or
            - elasticsearch-py instance or
            - eland.Client instance

        model_id: str
            The unique identifier of the trained inference model in Elasticsearch.

        model: An instance of a supported python model. We support the following model types:
            - sklearn.tree.DecisionTreeClassifier
            - sklearn.tree.DecisionTreeRegressor
            - sklearn.ensemble.RandomForestRegressor
            - sklearn.ensemble.RandomForestClassifier
            - xgboost.XGBClassifier
            - xgboost.XGBRegressor

        feature_names: List[str]
            Names of the features (required)

        classification_labels: List[str]
            Labels of the classification targets

        classification_weights: List[str]
            Weights of the classification targets
        """
        super().__init__(
            es_client,
            model_id
        )

        self._feature_names = feature_names
        self._model_type = None

        # Transform model
        if isinstance(model, DecisionTreeRegressor):
            serializer = SKLearnDecisionTreeTransformer(model, feature_names).transform()
            self._model_type = Model.TYPE_REGRESSION
        elif isinstance(model, DecisionTreeClassifier):
            serializer = SKLearnDecisionTreeTransformer(model, feature_names, classification_labels).transform()
            self._model_type = Model.TYPE_CLASSIFICATION
        elif isinstance(model, RandomForestRegressor):
            serializer = SKLearnForestRegressorTransformer(model, feature_names).transform()
            self._model_type = Model.TYPE_REGRESSION
        elif isinstance(model, RandomForestClassifier):
            serializer = SKLearnForestClassifierTransformer(model, feature_names, classification_labels).transform()
            self._model_type = Model.TYPE_CLASSIFICATION
        elif isinstance(model, XGBRegressor):
            serializer = XGBoostRegressorTransformer(model, feature_names).transform()
            self._model_type = Model.TYPE_REGRESSION
        elif isinstance(model, XGBClassifier):
            serializer = XGBoostClassifierTransformer(model, feature_names, classification_labels).transform()
            self._model_type = Model.TYPE_CLASSIFICATION
        else:
            raise NotImplementedError("ML model of type {}, not currently implemented".format(type(model)))

        serialized_model = str(serializer.serialize_and_compress_model())[2:-1]  # remove `b` and str quotes
        self._client.perform_request(
            "PUT", "/_ml/inference/" + self._model_id,
            body={
                "input": {
                    "field_names": feature_names
                },
                "compressed_definition": serialized_model
            }
        )

    def predict(self, X):
        """
        Make a prediction using a trained inference model in Elasticsearch.

        Parameters for this method are not fully compatible with standard sklearn.predict.

        Parameters
        ----------
        X: list or list of lists -
            Input feature vector - TODO support DataFrame and other formats

        Returns
        -------
        y: list or list of lists

        """
        docs = []
        if isinstance(X, list):
            # Is it a list of lists?
            if all(isinstance(i, list) for i in X):
                for i in X:
                    doc = dict()
                    doc['_source'] = dict(zip(self._feature_names, i))
                    docs.append(doc)

            else: # single feature vector1
                doc = dict()
                doc['_source'] = dict(zip(self._feature_names, i))
                docs.append(doc)
        else:
            raise NotImplementedError("Prediction for type {}, not supported".format(type(X)))

        results = self._client.perform_request(
            "POST",
            "/_ingest/pipeline/_simulate",
            body={
                "pipeline": {
                    "processors": [
                        {"inference": {
                            "model_id": self._model_id,
                            "inference_config": { self._model_type: {} },
                            "field_mappings": {}
                        }}
                    ]
                },
                "docs": docs
            })

        y = []

        # TODO return results

