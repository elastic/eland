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

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

import elasticsearch
import numpy as np  # type: ignore

from eland.common import ensure_es_client, es_version
from eland.utils import deprecated_api

from .common import TYPE_CLASSIFICATION, TYPE_REGRESSION
from .transformers import get_model_transformer

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch  # noqa: F401

    # Try importing each ML lib separately so mypy users don't have to
    # have both installed to use type-checking.
    try:
        from sklearn.ensemble import (  # type: ignore # noqa: F401
            RandomForestClassifier,
            RandomForestRegressor,
        )
        from sklearn.tree import (  # type: ignore # noqa: F401
            DecisionTreeClassifier,
            DecisionTreeRegressor,
        )
    except ImportError:
        pass
    try:
        from xgboost import XGBClassifier, XGBRegressor  # type: ignore # noqa: F401
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor  # type: ignore # noqa: F401
    except ImportError:
        pass


class MLModel:
    """
    A machine learning model managed by Elasticsearch.
    (See https://www.elastic.co/guide/en/elasticsearch/reference/master/put-inference.html)

    These models can be created by Elastic ML, or transformed from supported Python formats
    such as scikit-learn or xgboost and imported into Elasticsearch.

    The methods for this class attempt to mirror standard Python classes.
    """

    def __init__(
        self,
        es_client: Union[str, List[str], Tuple[str, ...], "Elasticsearch"],
        model_id: str,
    ):
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
        self._trained_model_config_cache: Optional[Dict[str, Any]] = None

    def predict(
        self, X: Union[np.ndarray, List[float], List[List[float]]]
    ) -> np.ndarray:
        """
        Make a prediction using a trained model stored in Elasticsearch.

        Parameters for this method are not yet fully compatible with standard sklearn.predict.

        Parameters
        ----------
        X: Input feature vector.
           Must be either a numpy ndarray or a list or list of lists
           of type float. TODO: support DataFrame and other formats

        Returns
        -------
        y: np.ndarray of dtype float for regressors or int for classifiers

        Examples
        --------
        >>> from sklearn import datasets
        >>> from xgboost import XGBRegressor
        >>> from eland.ml import MLModel

        >>> # Train model
        >>> training_data = datasets.make_classification(n_features=6, random_state=0)
        >>> test_data = [[-1, -2, -3, -4, -5, -6], [10, 20, 30, 40, 50, 60]]
        >>> regressor = XGBRegressor(objective='reg:squarederror')
        >>> regressor = regressor.fit(training_data[0], training_data[1])

        >>> # Get some test results
        >>> regressor.predict(np.array(test_data))  # doctest: +SKIP
        array([0.06062475, 0.9990102 ], dtype=float32)

        >>> # Serialise the model to Elasticsearch
        >>> feature_names = ["f0", "f1", "f2", "f3", "f4", "f5"]
        >>> model_id = "test_xgb_regressor"
        >>> es_model = MLModel.import_model('localhost', model_id, regressor, feature_names, es_if_exists='replace')

        >>> # Get some test results from Elasticsearch model
        >>> es_model.predict(test_data)  # doctest: +SKIP
        array([0.0606248 , 0.99901026], dtype=float32)

        >>> # Delete model from Elasticsearch
        >>> es_model.delete_model()
        """
        docs = []
        if isinstance(X, np.ndarray):

            def to_list_or_float(x: Any) -> Union[List[Any], float]:
                if isinstance(x, np.ndarray):
                    return [to_list_or_float(i) for i in x.tolist()]
                elif isinstance(x, list):
                    return [to_list_or_float(i) for i in x]
                return float(x)

            X = to_list_or_float(X)

        # Is it a list of floats?
        if isinstance(X, list) and all(isinstance(i, (float, int)) for i in X):
            features = cast(List[List[float]], [X])
        # If not a list of lists of floats then we error out.
        elif isinstance(X, list) and all(
            [
                isinstance(i, list) and all([isinstance(ix, (float, int)) for ix in i])
                for i in X
            ]
        ):
            features = cast(List[List[float]], X)
        else:
            raise NotImplementedError(
                f"Prediction for type {type(X)}, not supported: {X!r}"
            )

        for i in features:
            doc = {"_source": dict(zip(self.feature_names, i))}
            docs.append(doc)

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
                                "inference_config": {self.model_type: {}},
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

            y.append(res["doc"]["_source"]["ml"]["inference"][self.results_field])

        # Return results as np.ndarray of float32 or int (consistent with sklearn/xgboost)
        if self.model_type == TYPE_CLASSIFICATION:
            dt = np.int
        else:
            dt = np.float32
        return np.asarray(y, dtype=dt)

    @property
    def model_type(self) -> str:
        # Legacy way of finding model_type from the model definition.
        if "inference_config" not in self._trained_model_config:
            trained_model = self._trained_model_config["definition"]["trained_model"]
            if "tree" in trained_model:
                target_type = trained_model["tree"]["target_type"]
            else:
                target_type = trained_model["ensemble"]["target_type"]
            return cast(str, target_type)

        inference_config = self._trained_model_config["inference_config"]
        if "classification" in inference_config:
            return TYPE_CLASSIFICATION
        elif "regression" in inference_config:
            return TYPE_REGRESSION
        raise ValueError("Unable to determine 'model_type' for MLModel")

    @property
    def feature_names(self) -> List[str]:
        return list(self._trained_model_config["input"]["field_names"])

    @property
    def results_field(self) -> str:
        if "inference_config" not in self._trained_model_config:
            return "predicted_value"
        return cast(
            str,
            self._trained_model_config["inference_config"][self.model_type][
                "results_field"
            ],
        )

    @classmethod
    def import_model(
        cls,
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
        es_if_exists: Optional[str] = None,
        es_compress_model_definition: bool = True,
    ) -> "MLModel":
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

        es_if_exists: {'fail', 'replace'} default 'fail'
            How to behave if model already exists

            - fail: Raise a Value Error
            - replace: Overwrite existing model

        es_compress_model_definition: bool
            If True will use 'compressed_definition' which uses gzipped
            JSON instead of raw JSON to reduce the amount of data sent
            over the wire in HTTP requests. Defaults to 'True'.

        Examples
        --------
        >>> from sklearn import datasets
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> from eland.ml import MLModel

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
        >>> es_model = MLModel.import_model(
        ...   'localhost',
        ...   model_id=model_id,
        ...   model=classifier,
        ...   feature_names=feature_names,
        ...   es_if_exists='replace'
        ... )

        >>> # Get some test results from Elasticsearch model
        >>> es_model.predict(test_data)
        array([0, 1])

        >>> # Delete model from Elasticsearch
        >>> es_model.delete_model()
        """
        es_client = ensure_es_client(es_client)
        transformer = get_model_transformer(
            model,
            feature_names=feature_names,
            classification_labels=classification_labels,
            classification_weights=classification_weights,
        )
        serializer = transformer.transform()
        model_type = transformer.model_type

        if es_if_exists is None:
            es_if_exists = "fail"

        ml_model = MLModel(
            es_client=es_client,
            model_id=model_id,
        )
        if es_if_exists not in ("fail", "replace"):
            raise ValueError("'es_if_exists' must be either 'fail' or 'replace'")
        elif es_if_exists == "fail":
            if ml_model.exists_model():
                raise ValueError(
                    f"Trained machine learning model {model_id} already exists"
                )
        elif es_if_exists == "replace":
            ml_model.delete_model()

        body: Dict[str, Any] = {
            "input": {"field_names": feature_names},
        }
        # 'inference_config' is required in 7.8+ but isn't available in <=7.7
        if es_version(es_client) >= (7, 8):
            body["inference_config"] = {model_type: {}}

        if es_compress_model_definition:
            body["compressed_definition"] = serializer.serialize_and_compress_model()
        else:
            body["definition"] = serializer.serialize_model()

        ml_model._client.ml.put_trained_model(
            model_id=model_id,
            body=body,
        )
        return ml_model

    def delete_model(self) -> None:
        """
        Delete an inference model saved in Elasticsearch

        If model doesn't exist, ignore failure.
        """
        try:
            self._client.ml.delete_trained_model(model_id=self._model_id, ignore=(404,))
        except elasticsearch.NotFoundError:
            pass

    def exists_model(self) -> bool:
        """
        Check if the model already exists in Elasticsearch
        """
        try:
            self._client.ml.get_trained_models(model_id=self._model_id)
        except elasticsearch.NotFoundError:
            return False
        return True

    @property
    def _trained_model_config(self) -> Dict[str, Any]:
        """Lazily loads an ML models 'trained_model_config' information"""
        if self._trained_model_config_cache is None:

            # In Elasticsearch 7.7 and earlier you can't get
            # target type without pulling the model definition
            # so we check the version first.
            if es_version(self._client) < (7, 8):
                resp = self._client.ml.get_trained_models(
                    model_id=self._model_id, include_model_definition=True
                )
            else:
                resp = self._client.ml.get_trained_models(model_id=self._model_id)

            if resp["count"] > 1:
                raise ValueError(f"Model ID {self._model_id!r} wasn't unambiguous")
            elif resp["count"] == 0:
                raise ValueError(f"Model with Model ID {self._model_id!r} wasn't found")
            self._trained_model_config_cache = resp["trained_model_configs"][0]
        return self._trained_model_config_cache


ImportedMLModel = deprecated_api("MLModel.import_model()")(MLModel.import_model)
