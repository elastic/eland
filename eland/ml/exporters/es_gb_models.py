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

from abc import ABC, abstractmethod
from typing import Any, List, Literal, Mapping, Set, Tuple, Union, Optional

import numpy as np
import scipy as sp
from elasticsearch import Elasticsearch
from numpy.typing import ArrayLike
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
        """
        self.es_client: Elasticsearch = ensure_es_client(es_client)
        self.model_id = model_id

        self._trained_model_result = self.es_client.ml.get_trained_models(
            model_id=self.model_id,
            decompress_definition=True,
            include=["hyperparameters", "definition"],
        )

        self.feature_names_in_, self.input_field_names = self._get_feature_names_in_(
            self._definition["preprocessors"],
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

        self.n_estimators = (
            len(trained_models) - 1
        )  # 0's tree is the constant estimator

    def _initialize_estimators(self, DecisionTreeType) -> None:
        self.estimators_ = np.ndarray((len(self._trees) - 1, 1), dtype=DecisionTreeType)
        self.n_estimators_ = self.estimators_.shape[0]

        for i in range(self.n_estimators_):
            estimator = DecisionTreeRegressor()
            estimator.tree_ = self._trees[i + 1].tree_
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
    def _definition(self) -> Mapping[Union[str, int], Any]:
        return self._trained_model_result["trained_model_configs"][0]["definition"]

    def _get_feature_names_in_(
        self, preprocessors, feature_names, field_names
    ) -> Tuple[List[str], Set[str]]:
        input_field_names = set()
        for preprocessor in preprocessors:
            if "target_mean_encoding" in preprocessor:
                feature_name = preprocessor["target_mean_encoding"]["feature_name"]
                if feature_name in feature_names:
                    input_field_names.add(preprocessor["target_mean_encoding"]["field"])
            elif "frequency_encoding" in preprocessor:
                feature_name = preprocessor["frequency_encoding"]["feature_name"]
                if feature_name in feature_names:
                    input_field_names.add(preprocessor["frequency_encoding"]["field"])

            elif "one_hot_encoding" in preprocessor:
                for feature_name in preprocessor["one_hot_encoding"][
                    "hot_map"
                ].values():
                    if feature_name in feature_names:
                        input_field_names.add(preprocessor["one_hot_encoding"]["field"])

        for field_name in field_names:
            if field_name in feature_names and field_name not in input_field_names:
                input_field_names.add(field_name)

        return feature_names, input_field_names

    def fit(self, X, y, sample_weight=None, monitor=None) -> None:
        """
        Override of the sklearn fit() method. It does nothing since Elastic ML models are
        trained in the Elastic Stack or imported.
        """
        # Do nothing, model if fitted using Elasticsearch API
        pass

    @property
    @abstractmethod
    def analysis_type(self):
        """
        Type of the data frame analysis. It can be classification or regression.
        """
        pass

    @abstractmethod
    def _initialize_init_(self) -> Union[DummyClassifier, DummyRegressor]:
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
        """
        ESGradientBoostingModel.__init__(self, es_client, model_id)
        self._extract_common_parameters()
        GradientBoostingClassifier.__init__(
            self,
            learning_rate=1.0,
            n_estimators=self.n_estimators,
            max_depth=self._max_depth,
        )

        self.classes_ = np.array(
            self._definition["trained_model"]["ensemble"]["classification_labels"]
        )
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ == 2:
            self._loss = BinomialDeviance(self.n_classes_)
            self.n_outputs = 1
        elif self.n_classes_ > 2:
            raise NotImplementedError("Only binary classification is implemented.")
            # TODO: implement business logic for multiclass classification
            # self._loss = MultinomialDeviance(self.n_classes_)
            # self.n_outputs = self.n_classes_
        else:
            raise ValueError(f"At least 2 classes required. got {self.n_classes_}.")

        self.init_ = self._initialize_init_()
        self._initialize_estimators(DecisionTreeClassifier)

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
            log_odds = self._trees[0].tree_.value.flatten()[0]
            class_prior = sp.special.expit(log_odds)
            estimator.class_prior_ = np.array([1 - class_prior, class_prior])
        else:
            # TODO: implement business logic for multiclass classification
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
        if feature_names_in:
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
        if feature_names_in:
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
        """
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

    @property
    def analysis_type(self) -> Literal["regression"]:
        return TYPE_REGRESSION

    def _initialize_init_(self) -> DummyRegressor:
        constant = self._trees[0].tree_.value[0]
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
        if feature_names_in:
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