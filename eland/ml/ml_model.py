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

from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple, Union, cast

import elasticsearch
import numpy as np

from eland.common import ensure_es_client, es_version, is_serverless_es
from eland.utils import deprecated_api

from .common import TYPE_CLASSIFICATION, TYPE_LEARNING_TO_RANK, TYPE_REGRESSION
from .ltr import LTRModelConfig
from .transformers import get_model_transformer

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch
    from numpy.typing import ArrayLike, DTypeLike

    # Try importing each ML lib separately so mypy users don't have to
    # have both installed to use type-checking.
    try:
        from sklearn.ensemble import (  # type: ignore # noqa: F401
            RandomForestClassifier,
            RandomForestRegressor,
        )
        from sklearn.pipeline import Pipeline  # type: ignore # noqa: F401
        from sklearn.tree import (  # type: ignore # noqa: F401
            DecisionTreeClassifier,
            DecisionTreeRegressor,
        )
    except ImportError:
        pass
    try:
        from xgboost import (  # type: ignore # noqa: F401
            XGBClassifier,
            XGBRanker,
            XGBRegressor,
        )
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor  # type: ignore # noqa: F401
    except ImportError:
        pass


class MLModel:
    """
    A machine learning model managed by Elasticsearch.
    (See https://www.elastic.co/guide/en/elasticsearch/reference/current/put-inference.html)

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
        self._client: Elasticsearch = ensure_es_client(es_client)
        self._model_id = model_id
        self._trained_model_config_cache: Optional[Dict[str, Any]] = None

    def predict(
        self, X: Union["ArrayLike", List[float], List[List[float]]]
    ) -> "ArrayLike":
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
        >>> es_model = MLModel.import_model('http://localhost:9200', model_id, regressor, feature_names, es_if_exists='replace')

        >>> # Get some test results from Elasticsearch model
        >>> es_model.predict(test_data)  # doctest: +SKIP
        array([0.0606248 , 0.99901026], dtype=float32)

        >>> # Delete model from Elasticsearch
        >>> es_model.delete_model()
        """
        if self.model_type not in (TYPE_CLASSIFICATION, TYPE_REGRESSION):
            raise NotImplementedError(
                f"Prediction for type {self.model_type} is not supported."
            )

        docs: List[Mapping[str, Any]] = []
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
            pipeline={
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
            docs=docs,
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
            dt: "DTypeLike" = np.int_
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
        elif "learning_to_rank" in inference_config:
            return TYPE_LEARNING_TO_RANK
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
            "XGBRanker",
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
                - NOTE: When calculating the probabilities of a given classification label, Elasticsearch utilizes
                        softMax. SKLearn instead normalizes the results. We try to account for this during model
                        serialization, but probabilities may be slightly different in the predictions.
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
            - xgboost.XGBRanker
                - only the following objectives are supported:
                    - "rank:map"
                    - "rank:ndcg"
                    - "rank:pairwise"
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
        ...   'http://localhost:9200',
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

        return cls._import_model(
            es_client=es_client,
            model_id=model_id,
            model=model,
            feature_names=feature_names,
            classification_labels=classification_labels,
            classification_weights=classification_weights,
            es_if_exists=es_if_exists,
            es_compress_model_definition=es_compress_model_definition,
        )

    @classmethod
    def import_ltr_model(
        cls,
        es_client: Union[str, List[str], Tuple[str, ...], "Elasticsearch"],
        model_id: str,
        model: Union[
            "DecisionTreeRegressor",
            "RandomForestRegressor",
            "XGBRanker",
            "XGBRegressor",
            "LGBMRegressor",
        ],
        ltr_model_config: LTRModelConfig,
        es_if_exists: Optional[str] = None,
        es_compress_model_definition: bool = True,
    ) -> "MLModel":
        """
        Transform and serialize a trained 3rd party model into Elasticsearch.
        This model can then be used as a learning_to_rank rescorer in the Elastic Stack.

        Parameters
        ----------
        es_client: Elasticsearch client argument(s)
            - elasticsearch-py parameters or
            - elasticsearch-py instance

        model_id: str
            The unique identifier of the trained inference model in Elasticsearch.

        model: An instance of a supported python model. We support the following model types for LTR prediction:
            - sklearn.tree.DecisionTreeRegressor
            - sklearn.ensemble.RandomForestRegressor
            - xgboost.XGBRanker
                - only the following objectives are supported:
                    - "rank:map"
                    - "rank:ndcg"
                    - "rank:pairwise"
            - xgboost.XGBRegressor
                - only the following objectives are supported:
                    - "reg:squarederror"
                    - "reg:linear"
                    - "reg:squaredlogerror"
                    - "reg:logistic"
                    - "reg:pseudohubererror"
            - lightgbm.LGBMRegressor
                - Categorical fields are expected to already be processed
                - Only the following objectives are supported
                    - "regression"
                    - "regression_l1"
                    - "huber"
                    - "fair"
                    - "quantile"
                    - "mape"

        ltr_model_config: LTRModelConfig
            The LTR model configuration is used to configure feature extractors for the LTR model.
            Feature names are automatically inferred from the feature extractors.

        es_if_exists: {'fail', 'replace'} default 'fail'
            How to behave if model already exists

            - fail: Raise a Value Error
            - replace: Overwrite existing model

        es_compress_model_definition: bool
            If True will use 'compressed_definition' which uses gzipped
            JSON instead of raw JSON to reduce the amount of data sent
            over the wire in HTTP requests. Defaults to 'True'.
        """

        return cls._import_model(
            es_client=es_client,
            model_id=model_id,
            model=model,
            feature_names=ltr_model_config.feature_names,
            inference_config=ltr_model_config.to_dict(),
            es_if_exists=es_if_exists,
            es_compress_model_definition=es_compress_model_definition,
        )

    @classmethod
    def _import_model(
        cls,
        es_client: Union[str, List[str], Tuple[str, ...], "Elasticsearch"],
        model_id: str,
        model: Union[
            "DecisionTreeClassifier",
            "DecisionTreeRegressor",
            "RandomForestRegressor",
            "RandomForestClassifier",
            "XGBClassifier",
            "XGBRanker",
            "XGBRegressor",
            "LGBMRegressor",
            "LGBMClassifier",
        ],
        feature_names: List[str],
        classification_labels: Optional[List[str]] = None,
        classification_weights: Optional[List[float]] = None,
        es_if_exists: Optional[str] = None,
        es_compress_model_definition: bool = True,
        inference_config: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> "MLModel":
        """
        Actual implementation of model import used by public API methods.
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

        if inference_config is None:
            inference_config = {model_type: {}}

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

        trained_model_input = None
        is_ltr = next(iter(inference_config)) is TYPE_LEARNING_TO_RANK
        if not is_ltr or (
            es_version(es_client) < (8, 15) and not is_serverless_es(es_client)
        ):
            trained_model_input = {"field_names": feature_names}

        if es_compress_model_definition:
            ml_model._client.ml.put_trained_model(
                model_id=model_id,
                inference_config=inference_config,
                input=trained_model_input,
                compressed_definition=serializer.serialize_and_compress_model(),
            )
        else:
            ml_model._client.ml.put_trained_model(
                model_id=model_id,
                inference_config=inference_config,
                input=trained_model_input,
                definition=serializer.serialize_model(),
            )

        return ml_model

    def delete_model(self) -> None:
        """
        Delete an inference model saved in Elasticsearch

        If model doesn't exist, ignore failure.
        """
        try:
            self._client.options(ignore_status=404).ml.delete_trained_model(
                model_id=self._model_id
            )
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

    def export_model(self) -> "Pipeline":
        """Export Elastic ML model as sklearn Pipeline.

        Returns
        -------
        sklearn.pipeline.Pipeline
            _description_

        Raises
        ------
        AssertionError
            If preprocessors JSON definition has unexpected schema.
        ValueError
            The model is expected to be trained in Elastic Stack. Models initially imported
            from xgboost, lgbm, or sklearn are not supported.
        ValueError
            If unexpected categorical encoding is found in the list of preprocessors.
        NotImplementedError
            Only regression and binary classification models are supported currently.
        """
        from sklearn.compose import ColumnTransformer  # type: ignore # noqa: F401
        from sklearn.pipeline import Pipeline

        from .exporters._sklearn_deserializers import (
            FrequencyEncoder,
            OneHotEncoder,
            TargetMeanEncoder,
        )
        from .exporters.es_gb_models import (
            ESGradientBoostingClassifier,
            ESGradientBoostingRegressor,
        )

        if self.model_type == TYPE_CLASSIFICATION:
            model = ESGradientBoostingClassifier(
                es_client=self._client, model_id=self._model_id
            )
        elif self.model_type == TYPE_REGRESSION:
            model = ESGradientBoostingRegressor(
                es_client=self._client, model_id=self._model_id
            )
        else:
            raise NotImplementedError(
                "Only regression and binary classification models are supported currently."
            )

        transformers = []
        for p in model.preprocessors:
            assert (
                len(p) == 1
            ), f"Unexpected preprocessor data structure: {p}. One-key mapping expected."
            encoding_type = list(p.keys())[0]
            field = p[encoding_type]["field"]
            if encoding_type == "frequency_encoding":
                transform = FrequencyEncoder(p)
                transformers.append((f"{field}_{encoding_type}", transform, field))
            elif encoding_type == "target_mean_encoding":
                transform = TargetMeanEncoder(p)
                transformers.append((f"{field}_{encoding_type}", transform, field))
            elif encoding_type == "one_hot_encoding":
                transform = OneHotEncoder(p)
                transformers.append((f"{field}_{encoding_type}", transform, [field]))
            else:
                raise ValueError(
                    f"Unexpected categorical encoding type {encoding_type} found. "
                    + "Expected encodings: frequency_encoding, target_mean_encoding, one_hot_encoding."
                )
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="passthrough",
            verbose_feature_names_out=False,
        )

        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("es_model", model)])

        return pipeline

    @property
    def _trained_model_config(self) -> Dict[str, Any]:
        """Lazily loads an ML models 'trained_model_config' information"""
        if self._trained_model_config_cache is None:
            resp = self._client.ml.get_trained_models(model_id=self._model_id)

            if resp["count"] > 1:
                raise ValueError(f"Model ID {self._model_id!r} wasn't unambiguous")
            elif resp["count"] == 0:
                raise ValueError(f"Model with Model ID {self._model_id!r} wasn't found")
            self._trained_model_config_cache = resp["trained_model_configs"][0]
        return self._trained_model_config_cache


ImportedMLModel = deprecated_api("MLModel.import_model()")(MLModel.import_model)
