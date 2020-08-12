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

from typing import Union, List, Optional, Tuple, TYPE_CHECKING, cast, Dict, Any

import numpy as np  # type: ignore

from .ml_model import MLModel
from .transformers import get_model_transformer
from ..common import es_version


if TYPE_CHECKING:
    from elasticsearch import Elasticsearch  # type: ignore # noqa: F401

    # Try importing each ML lib separately so mypy users don't have to
    # have both installed to use type-checking.
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # type: ignore # noqa: F401
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # type: ignore # noqa: F401
    except ImportError:
        pass
    try:
        from xgboost import XGBRegressor, XGBClassifier  # type: ignore # noqa: F401
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor, LGBMClassifier  # type: ignore # noqa: F401
    except ImportError:
        pass


class ImportedMLModel(MLModel):
    """
    Transform and serialize a trained 3rd party model into Elasticsearch.
    This model can then be used for inference in the Elastic Stack.

    Parameters
    ----------
    es_client: Elasticsearch client argument(s)
        - elasticsearch-py parameters or
        - elasticsearch-py instance

    model_id: str
        The unique identifier of the trained inference model in Elasticsearch.

    model: An instance of a supported python model. We support the following model types:
        - sklearn.tree.DecisionTreeClassifier
        - sklearn.tree.DecisionTreeRegressor
        - sklearn.ensemble.RandomForestRegressor
        - sklearn.ensemble.RandomForestClassifier
        - lightgbm.LGBMRegressor
            - Categorical fields are expected to already be processed
            - Only the following objectives are supported
                - "regression"
                - "regression_l1"
                - "huber"
                - "fair"
                - "quantile"
                - "mape"
        - lightgbm.LGBMClassifier
            - Categorical fields are expected to already be processed
            - Only the following objectives are supported
                - "binary"
                - "multiclass"
                - "multiclassova"
        - xgboost.XGBClassifier
            - only the following objectives are supported:
                - "binary:logistic"
                - "multi:softmax"
                - "multi:softprob"
        - xgboost.XGBRegressor
            - only the following objectives are supported:
                - "reg:squarederror"
                - "reg:linear"
                - "reg:squaredlogerror"
                - "reg:logistic"
                - "reg:pseudohubererror"

    feature_names: List[str]
        Names of the features (required)

    classification_labels: List[str]
        Labels of the classification targets

    classification_weights: List[str]
        Weights of the classification targets

    overwrite: bool
        Delete and overwrite existing model (if exists)

    es_compress_model_definition: bool
        If True will use 'compressed_definition' which uses gzipped
        JSON instead of raw JSON to reduce the amount of data sent
        over the wire in HTTP requests. Defaults to 'True'.

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

    def __init__(
        self,
        es_client: Union[str, List[str], Tuple[str, ...], "Elasticsearch"],
        model_id: str,
        model: Union[
            "DecisionTreeClassifier",
            "DecisionTreeRegressor",
            "RandomForestRegressor",
            "RandomForestClassifier",
            "XGBClassifier",
            "XGBRegressor",
            "LGBMRegressor",
            "LGBMClassifier",
        ],
        feature_names: List[str],
        classification_labels: Optional[List[str]] = None,
        classification_weights: Optional[List[float]] = None,
        overwrite: bool = False,
        es_compress_model_definition: bool = True,
    ):
        super().__init__(es_client, model_id)

        self._feature_names = feature_names

        transformer = get_model_transformer(
            model,
            feature_names=feature_names,
            classification_labels=classification_labels,
            classification_weights=classification_weights,
        )
        self._model_type = transformer.model_type
        serializer = transformer.transform()

        if overwrite:
            self.delete_model()

        body: Dict[str, Any] = {
            "input": {"field_names": feature_names},
        }
        # 'inference_config' is required in 7.8+ but isn't available in <=7.7
        if es_version(self._client) >= (7, 8):
            body["inference_config"] = {self._model_type: {}}

        if es_compress_model_definition:
            body["compressed_definition"] = serializer.serialize_and_compress_model()
        else:
            body["definition"] = serializer.serialize_model()

        self._client.ml.put_trained_model(
            model_id=self._model_id, body=body,
        )

    def predict(self, X: Union[List[float], List[List[float]]]) -> np.ndarray:
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
        array([0.06062475, 0.9990102 ], dtype=float32)

        >>> # Serialise the model to Elasticsearch
        >>> feature_names = ["f0", "f1", "f2", "f3", "f4", "f5"]
        >>> model_id = "test_xgb_regressor"
        >>> es_model = ImportedMLModel('localhost', model_id, regressor, feature_names, overwrite=True)

        >>> # Get some test results from Elasticsearch model
        >>> es_model.predict(test_data)
        array([0.0606248 , 0.99901026], dtype=float32)

        >>> # Delete model from Elasticsearch
        >>> es_model.delete_model()

        """
        docs = []
        if isinstance(X, list):
            # Is it a list of floats?
            if all(isinstance(i, float) for i in X):
                features = cast(List[List[float]], [X])
            else:
                features = cast(List[List[float]], X)
            for i in features:
                doc = {"_source": dict(zip(self._feature_names, i))}
                docs.append(doc)
        else:
            raise NotImplementedError(f"Prediction for type {type(X)}, not supported")

        # field_mappings -> field_map in ES 7.7
        field_map_name = (
            "field_map" if es_version(self._client) >= (7, 7) else "field_mappings"
        )

        results = self._client.ingest.simulate(
            body={
                "pipeline": {
                    "processors": [
                        {
                            "inference": {
                                "model_id": self._model_id,
                                "inference_config": {self._model_type: {}},
                                field_map_name: {},
                            }
                        }
                    ]
                },
                "docs": docs,
            }
        )

        # Unpack results into an array. Errors can be present
        # within the response without a non-2XX HTTP status code.
        y = []
        for res in results["docs"]:
            if "error" in res:
                raise RuntimeError(
                    f"Failed to run prediction for model ID {self._model_id!r}",
                    res["error"],
                )

            y.append(res["doc"]["_source"]["ml"]["inference"]["predicted_value"])

        # Return results as np.ndarray of float32 or int (consistent with sklearn/xgboost)
        if self._model_type == MLModel.TYPE_CLASSIFICATION:
            dt = np.int
        else:
            dt = np.float32
        return np.asarray(y, dtype=dt)
