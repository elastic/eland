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

from abc import ABC
from typing import Any, List, Literal, Mapping, Optional, Set, Tuple, Union

import numpy as np
from elasticsearch import Elasticsearch
from numpy.typing import ArrayLike

from .._optional import import_optional_dependency

import_optional_dependency("sklearn", on_version="warn")

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble._gb_losses import (
    BinomialDeviance,
    HuberLossFunction,
    LeastSquaresError,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_array

from eland.common import ensure_es_client
from eland.ml.common import TYPE_CLASSIFICATION, TYPE_REGRESSION

from ._sklearn_deserializers import Tree
from .common import ModelDefinitionKeyError


class ESGradientBoostingModel(ABC):
    """
    Abstract class for converting Elastic ML model into sklearn Pipeline.
    """

    def __init__(
        self,
        es_client: Union[str, List[str], Tuple[str, ...], "Elasticsearch"],
        model_id: str,
    ) -> None:
        """
        Parameters
        ----------
        es_client : Elasticsearch client argument(s)
            - elasticsearch-py parameters or
            - elasticsearch-py instance
        model_id : str
            The unique identifier of the trained inference model in Elasticsearch.

        Raises
        ------
        RuntimeError
            On failure to retrieve trained model information to the specified model ID.
        ValueError
            The model is expected to be trained in Elastic Stack. Models initially imported
            from xgboost, lgbm, or sklearn are not supported.
        """
        self.es_client: Elasticsearch = ensure_es_client(es_client)
        self.model_id = model_id

        self._trained_model_result = self.es_client.ml.get_trained_models(
            model_id=self.model_id,
            decompress_definition=True,
            include=["hyperparameters", "definition"],
        )

        if (
            "trained_model_configs" not in self._trained_model_result
            or len(self._trained_model_result["trained_model_configs"]) == 0
        ):
            raise RuntimeError(
                f"Failed to retrieve the trained model for model ID {self.model_id!r}"
            )

        if "metadata" not in self._trained_model_result["trained_model_configs"][0]:
            raise ValueError(
                "Error initializing sklearn classifier. Incorrect prior class probability. "
                + "Note: only export of models trained in the Elastic Stack is supported."
            )
        preprocessors = []
        if "preprocessors" in self._definition:
            preprocessors = self._definition["preprocessors"]
        (
            self.feature_names_in_,
            self.input_field_names,
        ) = ESGradientBoostingModel._get_feature_names_in_(
            preprocessors,
            self._definition["trained_model"]["ensemble"]["feature_names"],
            self._trained_model_result["trained_model_configs"][0]["input"][
                "field_names"
            ],
        )

        feature_names_map = {name: i for i, name in enumerate(self.feature_names_in_)}

        trained_models = self._definition["trained_model"]["ensemble"]["trained_models"]
        self._trees = []
        for trained_model in trained_models:
            self._trees.append(Tree(trained_model["tree"], feature_names_map))

        # 0's tree is the constant estimator
        self.n_estimators = len(trained_models) - 1

    def _initialize_estimators(self, decision_tree_type) -> None:
        self.estimators_ = np.ndarray(
            (len(self._trees) - 1, 1), dtype=decision_tree_type
        )
        self.n_estimators_ = self.estimators_.shape[0]

        for i in range(self.n_estimators_):
            estimator = decision_tree_type()
            estimator.tree_ = self._trees[i + 1].tree
            estimator.n_features_in_ = self.n_features_in_
            estimator.max_depth = self._max_depth
            estimator.max_features_ = self.max_features_
            self.estimators_[i, 0] = estimator

    def _extract_common_parameters(self) -> None:
        self.n_features_in_ = len(self.feature_names_in_)
        self.max_features_ = self.n_features_in_

    @property
    def _max_depth(self) -> int:
        return max(map(lambda x: x.max_depth, self._trees))

    @property
    def _n_outputs(self) -> int:
        return self._trees[0].n_outputs

    @property
    def _definition(self) -> Mapping[Union[str, int], Any]:
        return self._trained_model_result["trained_model_configs"][0]["definition"]

    @staticmethod
    def _get_feature_names_in_(
        preprocessors, feature_names, field_names
    ) -> Tuple[List[str], Set[str]]:
        input_field_names = set()

        def add_input_field_name(preprocessor_type: str, feature_name: str) -> None:
            if feature_name in feature_names:
                input_field_names.add(preprocessor[preprocessor_type]["field"])

        for preprocessor in preprocessors:
            if "target_mean_encoding" in preprocessor:
                add_input_field_name(
                    "target_mean_encoding",
                    preprocessor["target_mean_encoding"]["feature_name"],
                )
            elif "frequency_encoding" in preprocessor:
                add_input_field_name(
                    "frequency_encoding",
                    preprocessor["frequency_encoding"]["feature_name"],
                )
            elif "one_hot_encoding" in preprocessor:
                for feature_name in preprocessor["one_hot_encoding"][
                    "hot_map"
                ].values():
                    add_input_field_name("one_hot_encoding", feature_name)

        for field_name in field_names:
            if field_name in feature_names and field_name not in input_field_names:
                input_field_names.add(field_name)

        return feature_names, input_field_names

    @property
    def preprocessors(self) -> List[Any]:
        """
        Returns the list of preprocessor JSON definitions.

        Returns
        -------
        List[Any]
            List of preprocessors definitions or [].
        """
        if "preprocessors" in self._definition:
            return self._definition["preprocessors"]
        return []

    def fit(self, X, y, sample_weight=None, monitor=None) -> None:
        """
        Override of the sklearn fit() method. It does nothing since Elastic ML models are
        trained in the Elastic Stack or imported.
        """
        # Do nothing, model if fitted using Elasticsearch API
        pass


class ESGradientBoostingClassifier(ESGradientBoostingModel, GradientBoostingClassifier):
    """
    Elastic ML model wrapper compatible with sklearn GradientBoostingClassifier.
    """

    def __init__(
        self,
        es_client: Union[str, List[str], Tuple[str, ...], "Elasticsearch"],
        model_id: str,
    ) -> None:
        """
        Parameters
        ----------
        es_client : Elasticsearch client argument(s)
            - elasticsearch-py parameters or
            - elasticsearch-py instance
        model_id : str
            The unique identifier of the trained inference model in Elasticsearch.

        Raises
        ------
        NotImplementedError
            Multi-class classification is not supported at the moment.
        ValueError
            The classifier should be defined for at least 2 classes.
        ModelDefinitionKeyError
            If required data cannot be extracted from the model definition due to a schema change.
        """

        try:
            ESGradientBoostingModel.__init__(self, es_client, model_id)
            self._extract_common_parameters()
            GradientBoostingClassifier.__init__(
                self,
                learning_rate=1.0,
                n_estimators=self.n_estimators,
                max_depth=self._max_depth,
            )

            if "classification_labels" in self._definition["trained_model"]["ensemble"]:
                self.classes_ = np.array(
                    self._definition["trained_model"]["ensemble"][
                        "classification_labels"
                    ]
                )
            else:
                self.classes_ = None

            self.n_outputs = self._n_outputs
            if self.classes_ is not None:
                self.n_classes_ = len(self.classes_)
            elif self.n_outputs <= 2:
                self.n_classes_ = 2
            else:
                self.n_classes_ = self.n_outputs

            if self.n_classes_ == 2:
                self._loss = BinomialDeviance(self.n_classes_)
                # self.n_outputs = 1
            elif self.n_classes_ > 2:
                raise NotImplementedError("Only binary classification is implemented.")
            else:
                raise ValueError(f"At least 2 classes required. got {self.n_classes_}.")

            self.init_ = self._initialize_init_()
            self._initialize_estimators(DecisionTreeClassifier)
        except KeyError as ex:
            raise ModelDefinitionKeyError(ex) from ex

    @property
    def analysis_type(self) -> Literal["classification"]:
        return TYPE_CLASSIFICATION

    def _initialize_init_(self) -> DummyClassifier:
        estimator = DummyClassifier(strategy="prior")

        estimator.n_classes_ = self.n_classes_
        estimator.n_outputs_ = self.n_outputs
        estimator.classes_ = np.arange(self.n_classes_)
        estimator._strategy = estimator.strategy

        if self.n_classes_ == 2:
            log_odds = self._trees[0].tree.value.flatten()[0]
            if np.isnan(log_odds):
                raise ValueError(
                    "Error initializing sklearn classifier. Incorrect prior class probability. "
                    + "Note: only export of models trained in the Elastic Stack is supported."
                )
            class_prior = 1 / (1 + np.exp(-log_odds))
            estimator.class_prior_ = np.array([1 - class_prior, class_prior])
        else:
            raise NotImplementedError("Only binary classification is implemented.")

        return estimator

    def predict_proba(
        self, X, feature_names_in: Optional[Union["ArrayLike", List[str]]] = None
    ) -> "ArrayLike":
        """Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        feature_names_in : {array of string, list of string} of length n_features.
            Feature names of the corresponding columns in X. Important, since the column list
            can be extended by ColumnTransformer through the pipeline. By default None.

        Returns
        -------
        ArrayLike of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        if feature_names_in is not None:
            if X.shape[1] != len(feature_names_in):
                raise ValueError(
                    f"Dimension mismatch: X with {X.shape[1]} columns has to be the same size as feature_names_in with {len(feature_names_in)}."
                )
            if isinstance(feature_names_in, np.ndarray):
                feature_names_in = feature_names_in.tolist()
            # select columns used by the model in the correct order
            X = X[:, [feature_names_in.index(fn) for fn in self.feature_names_in_]]

        X = check_array(X)
        return GradientBoostingClassifier.predict_proba(self, X)

    def predict(
        self,
        X: "ArrayLike",
        feature_names_in: Optional[Union["ArrayLike", List[str]]] = None,
    ) -> "ArrayLike":
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        feature_names_in : {array of string, list of string} of length n_features.
            Feature names of the corresponding columns in X. Important, since the column list
            can be extended by ColumnTransformer through the pipeline. By default None.

        Returns
        -------
        ArrayLike of shape (n_samples,)
            The predicted values.
        """
        if feature_names_in is not None:
            if X.shape[1] != len(feature_names_in):
                raise ValueError(
                    f"Dimension mismatch: X with {X.shape[1]} columns has to be the same size as feature_names_in with {len(feature_names_in)}."
                )
            if isinstance(feature_names_in, np.ndarray):
                feature_names_in = feature_names_in.tolist()
            # select columns used by the model in the correct order
            X = X[:, [feature_names_in.index(fn) for fn in self.feature_names_in_]]

        X = check_array(X)
        return GradientBoostingClassifier.predict(self, X)


class ESGradientBoostingRegressor(ESGradientBoostingModel, GradientBoostingRegressor):
    """
    Elastic ML model wrapper compatible with sklearn GradientBoostingRegressor.
    """

    def __init__(
        self,
        es_client: Union[str, List[str], Tuple[str, ...], "Elasticsearch"],
        model_id: str,
    ) -> None:
        """
        Parameters
        ----------
        es_client : Elasticsearch client argument(s)
            - elasticsearch-py parameters or
            - elasticsearch-py instance
        model_id : str
            The unique identifier of the trained inference model in Elasticsearch.

        Raises
        ------
        NotImplementedError
            Only MSE, MSLE, and Huber loss functions are supported.
        ModelDefinitionKeyError
            If required data cannot be extracted from the model definition due to a schema change.
        """
        try:
            ESGradientBoostingModel.__init__(self, es_client, model_id)
            self._extract_common_parameters()
            GradientBoostingRegressor.__init__(
                self,
                learning_rate=1.0,
                n_estimators=self.n_estimators,
                max_depth=self._max_depth,
            )

            self.n_outputs = 1
            loss_function = self._trained_model_result["trained_model_configs"][0][
                "metadata"
            ]["analytics_config"]["analysis"][self.analysis_type]["loss_function"]
            if loss_function == "mse" or loss_function == "msle":
                self.criterion = "squared_error"
                self._loss = LeastSquaresError()
            elif loss_function == "huber":
                loss_parameter = loss_function = self._trained_model_result[
                    "trained_model_configs"
                ][0]["metadata"]["analytics_config"]["analysis"][self.analysis_type][
                    "loss_function_parameter"
                ]
                self.criterion = "huber"
                self._loss = HuberLossFunction(loss_parameter)
            else:
                raise NotImplementedError(
                    "Only MSE, MSLE and Huber loss functions are supported."
                )

            self.init_ = self._initialize_init_()
            self._initialize_estimators(DecisionTreeRegressor)
        except KeyError as ex:
            raise ModelDefinitionKeyError(ex) from ex

    @property
    def analysis_type(self) -> Literal["regression"]:
        return TYPE_REGRESSION

    def _initialize_init_(self) -> DummyRegressor:
        constant = self._trees[0].tree.value[0]
        estimator = DummyRegressor(
            strategy="constant",
            constant=constant,
        )
        estimator.constant_ = np.array([constant])
        estimator.n_outputs_ = 1
        return estimator

    def predict(
        self,
        X: "ArrayLike",
        feature_names_in: Optional[Union["ArrayLike", List[str]]] = None,
    ) -> "ArrayLike":
        """Predict targets for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        feature_names_in : {array of string, list of string} of length n_features.
            Feature names of the corresponding columns in X. Important, since the column list
            can be extended by ColumnTransformer through the pipeline. By default None.

        Returns
        -------
        ArrayLike of shape (n_samples,)
            The predicted values.
        """
        if feature_names_in is not None:
            if X.shape[1] != len(feature_names_in):
                raise ValueError(
                    f"Dimension mismatch: X with {X.shape[1]} columns has to be the same size as feature_names_in with {len(feature_names_in)}."
                )
            if isinstance(X, np.ndarray):
                feature_names_in = feature_names_in.tolist()
            # select columns used by the model in the correct order
            X = X[:, [feature_names_in.index(fn) for fn in self.feature_names_in_]]

        X = check_array(X)
        return GradientBoostingRegressor.predict(self, X)
