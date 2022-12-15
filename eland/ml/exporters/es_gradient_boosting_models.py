import pprint
import random
import string
from abc import ABC

import eland as ed
from elasticsearch.client import IngestClient
import numpy as np
import scipy as sp
import pandas as pd
import sklearn
import sklearn.ensemble
from sklearn.ensemble._gb_losses import (
    BinomialDeviance,
    MultinomialDeviance,
    LeastSquaresError,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor

from .trees import Forest
from ... import DataFrame


class ESGradientBoostingModel(ABC):
    def __init__(self, es_client, model_id):
        self.es_client = es_client
        self.model_id = model_id
        suffix = "".join(random.choices(string.ascii_lowercase, k=5))
        self.pipeline_id = model_id + suffix
        self.get_model_definition()
        self.n_estimators = (
            len(self.definition["trained_model"]["ensemble"]["trained_models"]) - 1
        )  # 0's tree is the constant estimator

    def get_model_definition(self) -> None:
        self.res = self.es_client.ml.get_trained_models(
            model_id=self.model_id,
            decompress_definition=True,
            include=["hyperparameters", "definition"],
        )
        self.definition = self.res["trained_model_configs"][0]["definition"]
        trained_models = self.definition["trained_model"]["ensemble"]["trained_models"]
        self.forest = Forest(trained_models)

    def get_data(self, is_training=True) -> ed.DataFrame:
        ed_df = ed.DataFrame(
            es_client=self.es_client, es_index_pattern=self.destination_index
        )
        query = {
            "bool": {
                "filter": [
                    {
                        "bool": {
                            "should": [{"match": {"ml.is_training": is_training}}],
                            "minimum_should_match": 1,
                        }
                    }
                ]
            }
        }
        return ed_df.es_query(query)

    def get_test_data(self) -> DataFrame:
        return self.get_data(is_training=False)

    def get_training_data(self) -> DataFrame:
        return self.get_data(is_training=True)

    def get_feature_names_in_(self, preprocessors, feature_names):
        field_names = set()
        for preprocessor in preprocessors:
            if "target_mean_encoding" in preprocessor:
                feature_name = preprocessor["target_mean_encoding"]["feature_name"]
                if feature_name in feature_names:
                    field_names.add(preprocessor["target_mean_encoding"]["field"])
            elif "frequency_encoding" in preprocessor:
                feature_name = preprocessor["frequency_encoding"]["feature_name"]
                if feature_name in feature_names:
                    field_names.add(preprocessor["frequency_encoding"]["field"])

            elif "one_hot_encoding" in preprocessor:
                for feature_name in preprocessor["one_hot_encoding"][
                    "hot_map"
                ].values():
                    if feature_name in feature_names:
                        field_names.add(preprocessor["one_hot_encoding"]["field"])

        return feature_names, field_names

    def fit(self, X, y, sample_weight=None, monitor=None) -> None:
        # do nothing, model if fitted using elasticsearch API
        pass

    def extract_common_parameters(self) -> None:
        self.included_field_names = self.res["trained_model_configs"][0]["metadata"][
            "analytics_config"
        ]["analyzed_fields"]["includes"]
        self.dependent_variable = self.res["trained_model_configs"][0]["metadata"][
            "analytics_config"
        ]["analysis"][self.analysis_type]["dependent_variable"]

        self.feature_names_in_, self.input_field_names = self.get_feature_names_in_(
            self.definition["preprocessors"],
            self.definition["trained_model"]["ensemble"]["feature_names"],
        )
        for field_name in self.res["trained_model_configs"][0]["input"]["field_names"]:
            if (
                field_name in self.feature_names_in_
                and field_name not in self.input_field_names
            ):
                self.input_field_names.add(field_name)

        self.feature_names_map = {
            name: i for i, name in enumerate(self.feature_names_in_)
        }
        self.n_features_in_ = len(self.feature_names_in_)
        self.max_features_ = self.n_features_
        self.destination_index = self.res["trained_model_configs"][0]["metadata"][
            "analytics_config"
        ]["dest"]["index"]
        for _, tree in enumerate(self.forest.trees):
            for _, node in tree.nodes.items():
                if node.is_leaf is False:
                    node.split_feature = self.feature_names_map[node.split_feature_name]

    def _initialize_estimator(self, i) -> DecisionTreeRegressor:
        estimator = DecisionTreeRegressor()
        estimator.tree_ = self.forest.trees[i + 1].to_sklearn()
        estimator.n_features_in_ = self.n_features_in_
        # estimator.n_features_in = self.n_features_in_
        estimator.max_depth = self.max_depth
        estimator.max_features_ = self.max_features_
        self.n_outputs = self.n_outputs
        return estimator

    def initialize_estimators(self, DecisionTreeType) -> None:
        self.init_ = self._initialize_init_()

        self.estimators_ = np.ndarray(
            (self.forest.num_trees() - 1, 1), dtype=DecisionTreeType
        )
        self.n_estimators_ = self.estimators_.shape[0]

        for i in range(self.n_estimators_):
            self.estimators_[i, 0] = self._initialize_estimator(i)


class ESGradientBoostingClassifier(
    ESGradientBoostingModel, sklearn.ensemble.GradientBoostingClassifier
):
    def __init__(self, es_client, model_id) -> None:
        ESGradientBoostingModel.__init__(self, es_client, model_id)

        sklearn.ensemble.GradientBoostingClassifier.__init__(
            self,
            learning_rate=1.0,
            n_estimators=self.n_estimators,
            max_depth=self.forest.max_depth(),
        )
        self.analysis_type = "classification"
        self._complete()

    def _complete(self) -> None:

        self.extract_common_parameters()

        self.classes_ = self.definition["trained_model"]["ensemble"][
            "classification_labels"
        ]
        self.classes_ = np.array(self.classes_)

        self.n_classes_ = len(self.classes_)
        if self.n_classes_ == 2:
            self._loss = BinomialDeviance(self.n_classes_)
            self.n_outputs = 1
        elif self.n_classes_ > 2:
            self._loss = MultinomialDeviance(self.n_classes_)
            self.n_outputs = self.n_classes_
        else:
            raise ValueError(f"At least 2 classes required. got {self.n_classes_}.")

        self.initialize_estimators(DecisionTreeClassifier)

    def _initialize_init_(self) -> DummyClassifier:

        estimator = DummyClassifier(strategy="prior")

        estimator.n_classes_ = self.n_classes_
        estimator.n_outputs_ = self.n_outputs
        estimator.classes_ = np.arange(self.n_classes_)
        estimator._strategy = estimator.strategy
        estimator.estimator_type = "classifier"

        if self.n_classes_ == 2:
            log_odds = self.forest.trees[0].nodes[0].value
            class_prior = sp.special.expit(log_odds)
            estimator.class_prior_ = np.array([1 - class_prior, class_prior])
        else:
            # TODO: implement business logic for multiclass classification
            raise NotImplementedError("Only binary classification is implemented.")

        return estimator

    def predict_proba(self, X, **kwargs):
        feature_names_in = kwargs.get("feature_names_in", None)
        X = pd.DataFrame(X, columns=feature_names_in)
        return sklearn.ensemble.GradientBoostingClassifier.predict_proba(
            self, X[self.feature_names_in_].to_numpy()
        )

    def predict(self, X, **kwargs):
        feature_names_in = kwargs.get("feature_names_in", None)
        X = pd.DataFrame(X, columns=feature_names_in)
        return sklearn.ensemble.GradientBoostingClassifier.predict(
            self, X[self.feature_names_in_].to_numpy()
        )


class ESGradientBoostingRegressor(
    ESGradientBoostingModel, sklearn.ensemble.GradientBoostingRegressor
):
    def __init__(self, es_client, model_id) -> None:
        ESGradientBoostingModel.__init__(self, es_client, model_id)
        sklearn.ensemble.GradientBoostingRegressor.__init__(
            self,
            learning_rate=1.0,
            n_estimators=self.n_estimators,
            max_depth=self.forest.max_depth(),
        )
        self.analysis_type = "regression"
        self._complete()

    def _complete(self) -> None:
        self.extract_common_parameters()

        self.constant = self.forest.trees[0].nodes[0].value
        self.n_outputs = 1

        self.initialize_estimators(DecisionTreeRegressor)

        self.learning_rate = 1.0
        # TODO works only for a single case
        loss_function = self.res["trained_model_configs"][0]["metadata"][
            "analytics_config"
        ]["analysis"][self.analysis_type]["loss_function"]
        if loss_function == "mse":
            self.criterion = "squared_error"
            self._loss = LeastSquaresError()
        else:
            raise NotImplementedError(
                "Only MSE loss function is implemented at the moment"
            )

    def _initialize_init_(self) -> DummyRegressor:
        estimator = DummyRegressor(
            strategy="constant",
            constant=self.constant,
        )
        estimator.constant_ = np.array([self.constant])
        estimator.n_outputs_ = 1
        return estimator

    def predict(self, X, **kwargs):
        feature_names_in = kwargs.get("feature_names_in", None)
        X = pd.DataFrame(X, columns=feature_names_in)
        return sklearn.ensemble.GradientBoostingRegressor.predict(
            self, X[self.feature_names_in_].to_numpy()
        )
