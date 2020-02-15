#  Copyright 2020 Elasticsearch BV
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

import numpy as np

from eland.ml._model_transformers import SKLearnDecisionTreeTransformer, SKLearnForestRegressorTransformer, \
    SKLearnForestClassifierTransformer, XGBoostRegressorTransformer, XGBoostClassifierTransformer
from eland.ml._optional import import_optional_dependency
from eland.ml.ml_model import MLModel

sklearn = import_optional_dependency("sklearn")
xgboost = import_optional_dependency("xgboost")

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBRegressor, XGBClassifier


class ImportedMLModel(MLModel):
    """
    Transform and serialize a trained 3rd party model into Elasticsearch.
    This model can then be used for inference in the Elastic Stack.

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

    overwrite: bool
        Delete and overwrite existing model (if exists)

    Examples
    --------
    >>> from sklearn import datasets
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from eland.ml import ImportedMLModel

    >>> # Train model
    >>> training_data = datasets.make_classification(n_features=5, random_state=0)
    >>> test_data = [[-50.1, 0.2, 0.3, -0.5, 1.0], [1.6, 2.1, -10, 50, -1.0]]
    >>> classifier = DecisionTreeClassifier()
    >>> classifier = classifier.fit(training_data[0], training_data[1])

    >>> # Get some test results
    >>> classifier.predict(test_data)
    array([0, 1])

    >>> # Serialise the model to Elasticsearch
    >>> feature_names = ["f0", "f1", "f2", "f3", "f4"]
    >>> model_id = "test_decision_tree_classifier"
    >>> es_model = ImportedMLModel('localhost', model_id, classifier, feature_names, overwrite=True)

    >>> # Get some test results from Elasticsearch model
    >>> es_model.predict(test_data)
    array([0, 1])

    >>> # Delete model from Elasticsearch
    >>> es_model.delete_model()

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
                 classification_weights: List[float] = None,
                 overwrite=False):
        super().__init__(
            es_client,
            model_id
        )

        self._feature_names = feature_names
        self._model_type = None

        # Transform model
        if isinstance(model, DecisionTreeRegressor):
            serializer = SKLearnDecisionTreeTransformer(model, feature_names).transform()
            self._model_type = MLModel.TYPE_REGRESSION
        elif isinstance(model, DecisionTreeClassifier):
            serializer = SKLearnDecisionTreeTransformer(model, feature_names, classification_labels).transform()
            self._model_type = MLModel.TYPE_CLASSIFICATION
        elif isinstance(model, RandomForestRegressor):
            serializer = SKLearnForestRegressorTransformer(model, feature_names).transform()
            self._model_type = MLModel.TYPE_REGRESSION
        elif isinstance(model, RandomForestClassifier):
            serializer = SKLearnForestClassifierTransformer(model, feature_names, classification_labels).transform()
            self._model_type = MLModel.TYPE_CLASSIFICATION
        elif isinstance(model, XGBRegressor):
            serializer = XGBoostRegressorTransformer(model, feature_names).transform()
            self._model_type = MLModel.TYPE_REGRESSION
        elif isinstance(model, XGBClassifier):
            serializer = XGBoostClassifierTransformer(model, feature_names, classification_labels).transform()
            self._model_type = MLModel.TYPE_CLASSIFICATION
        else:
            raise NotImplementedError("ML model of type {}, not currently implemented".format(type(model)))

        if overwrite:
            self.delete_model()

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
        Make a prediction using a trained model stored in Elasticsearch.

        Parameters for this method are not yet fully compatible with standard sklearn.predict.

        Parameters
        ----------
        X: list or list of lists of type float
            Input feature vector - TODO support DataFrame and other formats

        Returns
        -------
        y: np.ndarray of dtype float for regressors or int for classifiers

        Examples
        --------
        >>> from sklearn import datasets
        >>> from xgboost import XGBRegressor
        >>> from eland.ml import ImportedMLModel

        >>> # Train model
        >>> training_data = datasets.make_classification(n_features=6, random_state=0)
        >>> test_data = [[-1, -2, -3, -4, -5, -6], [10, 20, 30, 40, 50, 60]]
        >>> regressor = XGBRegressor(objective='reg:squarederror')
        >>> regressor = regressor.fit(training_data[0], training_data[1])

        >>> # Get some test results
        >>> regressor.predict(np.array(test_data))
        array([0.23733574, 1.1897984 ], dtype=float32)

        >>> # Serialise the model to Elasticsearch
        >>> feature_names = ["f0", "f1", "f2", "f3", "f4", "f5"]
        >>> model_id = "test_xgb_regressor"
        >>> es_model = ImportedMLModel('localhost', model_id, regressor, feature_names, overwrite=True)

        >>> # Get some test results from Elasticsearch model
        >>> es_model.predict(test_data)
        array([0.2373357, 1.1897984], dtype=float32)

        >>> # Delete model from Elasticsearch
        >>> es_model.delete_model()

        """
        docs = []
        if isinstance(X, list):
            # Is it a list of lists?
            if all(isinstance(i, list) for i in X):
                for i in X:
                    doc = dict()
                    doc['_source'] = dict(zip(self._feature_names, i))
                    docs.append(doc)

            else:  # single feature vector1
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
                            "inference_config": {self._model_type: {}},
                            "field_mappings": {}
                        }}
                    ]
                },
                "docs": docs
            })

        y = [
            doc['doc']['_source']['ml']['inference']['predicted_value'] for doc in results['docs']
        ]

        # Return results as np.ndarray of float32 or int (consistent with sklearn/xgboost)
        if self._model_type == MLModel.TYPE_CLASSIFICATION:
            dt = np.int
        else:
            dt = np.float32
        return np.asarray(y, dtype=dt)
