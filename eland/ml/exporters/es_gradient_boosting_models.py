
import pprint
import random
import string

import eland as ed
import eland.ml
from elasticsearch.client import IngestClient
import numpy as np
import scipy as sp
import pandas as pd
import sklearn
import sklearn.ensemble
from sklearn.ensemble._gb_losses import BinomialDeviance, MultinomialDeviance, LeastSquaresError
from zmq import PROTOCOL_ERROR_ZMTP_MALFORMED_COMMAND_MESSAGE

from .trees import Forest
from .. import MLModel
from ... import DataFrame

class ESGradientBoostingClassifier(sklearn.ensemble.GradientBoostingClassifier):
    def __init__(self, es_client, model_id):
        self.es_client = es_client
        self.model_id = model_id
        suffix = "".join(random.choices(string.ascii_lowercase, k=5))
        self.pipeline_id = model_id + suffix
        self.analysis_type = "classification"
        self.es_model = MLModel(es_client=es_client, model_id=model_id)
        self.get_model_definition()
        n_estimators = (
            len(self.definition["trained_model"]["ensemble"]["trained_models"]) - 1
        )  # 0's tree is the constant estimator
        super().__init__(
            learning_rate=1.0,
            n_estimators=n_estimators,
            max_depth=self.forest.max_depth(),
        )
        self._complete()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.delete_ingestion_pipeline()

    def create_ingestion_pipeline(self):

        processors = [
            {
                "inference": {
                    "model_id": self.model_id,
                    "target_field": "prediction",
                    "inference_config": {self.analysis_type: {}},
                    "field_map": {},
                }
            }
        ]
        ingestClient = IngestClient(self.es_client)
        response = ingestClient.put_pipeline(id=self.pipeline_id, processors=processors)
        pprint.pprint(response)

    def delete_ingestion_pipeline(self):
        ingestClient = IngestClient(self.es_client)
        response = ingestClient.delete_pipeline(id=self.pipeline_id)
        pprint.pprint(response)

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

    def get_model_definition(self) -> None:
        self.res = self.es_client.ml.get_trained_models(
            model_id=self.model_id,
            decompress_definition=True,
            include=["hyperparameters", "definition"],
        )
        self.definition = self.res["trained_model_configs"][0]["definition"]
        trained_models = self.definition["trained_model"]["ensemble"]["trained_models"]
        self.forest = Forest(trained_models)

    def get_feature_names_in_(self, preprocessors, feature_names):
        # feature_names = []
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
                for feature_name in preprocessor["one_hot_encoding"]["hot_map"].values():
                    if feature_name in feature_names:
                        field_names.add(preprocessor["one_hot_encoding"]["field"])

        return feature_names, field_names

    def get_test_data(self) -> DataFrame:
        return self.get_data(is_training=False)

    def get_training_data(self) -> DataFrame:
        return self.get_data(is_training=True)

    def fit(self, X, y, sample_weight=None, monitor=None):
        # do nothing, model if fitted using elasticsearch API
        pass

    # def fit(self, dataset):
    #     self.model = train(dataset_name=self.dataset_name, dataset=dataset)
    #     self.model.wait_to_complete()

    def _complete(self) -> None:

        # definition = res["trained_model_configs"][0]["definition"]
        self.classes_ = self.definition["trained_model"]["ensemble"][
            "classification_labels"
        ]
        self.classes_ = np.array(self.classes_)

        self.included_field_names = self.res["trained_model_configs"][0]["metadata"][
            "analytics_config"
        ]["analyzed_fields"]["includes"]
        self.dependent_variable = self.res["trained_model_configs"][0]["metadata"][
            "analytics_config"
        ]["analysis"]["classification"]["dependent_variable"]

        self.feature_names_in_, self.input_field_names = self.get_feature_names_in_(
            self.definition["preprocessors"], self.definition['trained_model']['ensemble']['feature_names']
        )
        for field_name in self.res["trained_model_configs"][0]["input"]["field_names"]:
            if field_name in self.feature_names_in_ and field_name not in self.input_field_names:
                self.input_field_names.add(field_name)

        # for input_field in self.input_field_names:
            # if input_field not in preprocessor_field_names:
            #     self.feature_names_in_.append(input_field)
        self.feature_names_map = {
            name: i for i, name in enumerate(self.feature_names_in_)
        }
        self.n_features_in_ = len(self.feature_names_in_)
        print(f"n_features_in_ is {self.n_features_in_}")
        self.max_features_ = self.n_features_
        self.destination_index = self.res["trained_model_configs"][0]["metadata"][
            "analytics_config"
        ]["dest"]["index"]
        for _, tree in enumerate(self.forest.trees):
            for _, node in tree.nodes.items():
                if node.is_leaf is False:
                    node.split_feature = self.feature_names_map[node.split_feature_name]

        self.n_classes_ = len(self.classes_)
        if self.n_classes_ == 2:
            self._loss = BinomialDeviance(self.n_classes_)
            self.n_outputs = 1
        elif self.n_classes_ > 2:
            self._loss = MultinomialDeviance(self.n_classes_)
            self.n_outputs = self.n_classes_
        else:
            raise ValueError(f"At least 2 classes required. got {self.n_classes_}.")

        self.init_ = self._initialize_init_()

        self.estimators_ = np.ndarray(
            (self.forest.num_trees() - 1, 1), dtype=sklearn.tree.DecisionTreeClassifier
        )
        self.n_estimators_ = self.estimators_.shape[0]

        for i in range(self.n_estimators_):
            self.estimators_[i, 0] = self._initialize_estimator(i)

    def _initialize_estimator(self, i) -> sklearn.tree.DecisionTreeRegressor:
        estimator = sklearn.tree.DecisionTreeRegressor()
        estimator.tree_ = self.forest.trees[i + 1].to_sklearn()
        estimator.n_features_in_ = self.n_features_in_
        # estimator.n_features_in = self.n_features_in_
        estimator.max_depth = self.max_depth
        estimator.max_features_ = self.max_features_
        self.n_outputs = self.n_outputs
        return estimator

    def _initialize_init_(self):

        estimator = sklearn.dummy.DummyClassifier(strategy="prior")

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

    def predict_proba(self, X,  **kwargs):
        feature_names_in = kwargs.get('feature_names_in', None)
        X = pd.DataFrame(X, columns=feature_names_in)
        return super().predict_proba(X[self.feature_names_in_].to_numpy())

    def predict(self, X,  **kwargs):
        feature_names_in = kwargs.get('feature_names_in', None)
        X = pd.DataFrame(X, columns=feature_names_in)
        return super().predict(X[self.feature_names_in_].to_numpy())
        # if isinstance(X, np.ndarray) and X.shape[1] == self.n_features_in_:
        #     return super().predict(X)
        # if isinstance(X, DataFrame):
        #     X = X[self.input_field_names].to_pandas()
        # elif isinstance(X, pd.core.frame.DataFrame):
        #     X = X[self.input_field_names]
        # else:
        #     raise NotImplementedError("Only pandas and eland DataFrame are supported.")
        # predictions = self.es_model.predict(X.to_numpy())
        # return predictions

class ESGradientBoostingRegressor(sklearn.ensemble.GradientBoostingRegressor):
    def __init__(self, es_client, model_id):
        self.es_client = es_client
        self.model_id = model_id
        suffix = "".join(random.choices(string.ascii_lowercase, k=5))
        self.pipeline_id = model_id + suffix
        self.analysis_type = "regression"
        self.es_model = MLModel(es_client=es_client, model_id=model_id)
        self.get_model_definition()
        n_estimators = (
            len(self.definition["trained_model"]["ensemble"]["trained_models"]) - 1
        )  # 0's tree is the constant estimator
        super().__init__(
            learning_rate=1.0,
            n_estimators=n_estimators,
            max_depth=self.forest.max_depth(),
        )
        self._complete()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.delete_ingestion_pipeline()

    def create_ingestion_pipeline(self):

        processors = [
            {
                "inference": {
                    "model_id": self.model_id,
                    "target_field": "prediction",
                    "inference_config": {self.analysis_type: {}},
                    "field_map": {},
                }
            }
        ]
        ingestClient = IngestClient(self.es_client)
        response = ingestClient.put_pipeline(id=self.pipeline_id, processors=processors)
        pprint.pprint(response)

    def delete_ingestion_pipeline(self):
        ingestClient = IngestClient(self.es_client)
        response = ingestClient.delete_pipeline(id=self.pipeline_id)
        pprint.pprint(response)

    def get_data(self, is_training=True) -> DataFrame:
        ed_df = DataFrame(
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

    def get_model_definition(self) -> None:
        self.res = self.es_client.ml.get_trained_models(
            model_id=self.model_id,
            decompress_definition=True,
            include=["hyperparameters", "definition"],
        )
        self.definition = self.res["trained_model_configs"][0]["definition"]
        trained_models = self.definition["trained_model"]["ensemble"]["trained_models"]
        self.forest = Forest(trained_models)

    def get_feature_names_in_(self, preprocessors):
        feature_names = []
        field_names = set()
        for preprocessor in preprocessors:
            if "target_mean_encoding" in preprocessor:
                feature_names.append(
                    preprocessor["target_mean_encoding"]["feature_name"]
                )
                field_names.add(preprocessor["target_mean_encoding"]["field"])
            elif "frequency_encoding" in preprocessor:
                feature_names.append(preprocessor["frequency_encoding"]["feature_name"])
                field_names.add(preprocessor["frequency_encoding"]["field"])

            elif "one_hot_encoding" in preprocessor:
                feature_names += list(
                    preprocessor["one_hot_encoding"]["hot_map"].values()
                )
                field_names.add(preprocessor["one_hot_encoding"]["field"])

        return feature_names, field_names

    def get_test_data(self) -> DataFrame:
        return self.get_data(is_training=False)

    def get_training_data(self) -> DataFrame:
        return self.get_data(is_training=True)

    def fit(self, X, y, sample_weight=None, monitor=None):
        # raise NotImplementedError("Fit the model using elasticsearch API.")
        pass

    # def fit(self, dataset):
    #     self.model = train(dataset_name=self.dataset_name, dataset=dataset)
    #     self.model.wait_to_complete()

    def _complete(self):
        self.included_field_names = self.res["trained_model_configs"][0]["metadata"][
            "analytics_config"
        ]["analyzed_fields"]["includes"]
        self.input_field_names = self.res["trained_model_configs"][0]["input"]["field_names"]
        self.dependent_variable = self.res["trained_model_configs"][0]["metadata"][
            "analytics_config"
        ]["analysis"]["regression"]["dependent_variable"]

        self.feature_names_in_, preprocessor_field_names = self.get_feature_names_in_(
            self.definition["preprocessors"]
        )
        for input_field in self.input_field_names:
            if input_field not in preprocessor_field_names:
                self.feature_names_in_.append(input_field)
        self.feature_names_map = {
            name: i for i, name in enumerate(self.feature_names_in_)
        }
        self.n_features_in_ = len(self.feature_names_in_)
        print(f"n_features_in_ is {self.n_features_in_}")
        self.max_features_ = self.n_features_
        self.destination_index = self.res["trained_model_configs"][0]["metadata"][
            "analytics_config"
        ]["dest"]["index"]
        loss_function = self.res["trained_model_configs"][0]["metadata"]["analytics_config"][
            "analysis"
        ]["regression"]["loss_function"]
        for _, tree in enumerate(self.forest.trees):
            for _, node in tree.nodes.items():
                if node.is_leaf is False:
                    node.split_feature = self.feature_names_map[node.split_feature_name]

        self.constant = self.forest.trees[0].nodes[0].value
        self.n_outputs = 1

        self.init_ = self._initialize_init_()

        self.estimators_ = np.ndarray(
            (self.forest.num_trees() - 1, 1), dtype=sklearn.tree.DecisionTreeRegressor
        )
        self.n_estimators_ = self.estimators_.shape[0]

        for i in range(self.estimators_.shape[0]):
            self.estimators_[i, 0] = self._initialize_estimator(i)

        self.learning_rate = 1.0
        # TODO works only for a single case
        if loss_function == "mse":
            self.criterion = "squared_error"
            self._loss = LeastSquaresError()
        else:
            raise NotImplementedError(
                "Only MSE loss function is implemented at the moment"
            )

    def _initialize_estimator(self, i) -> sklearn.tree.DecisionTreeRegressor:
        estimator = sklearn.tree.DecisionTreeRegressor()
        estimator.tree_ = self.forest.trees[i + 1].to_sklearn()
        estimator.n_features_in_ = self.n_features_in_
        # estimator.n_features_in = self.n_features_in_
        estimator.max_depth = self.max_depth
        estimator.max_features_ = self.max_features_
        self.n_outputs = self.n_outputs
        return estimator

    def _initialize_init_(self) -> sklearn.dummy.DummyRegressor:
        estimator = sklearn.dummy.DummyRegressor(
            strategy="constant", constant=self.constant, 
        )
        estimator.constant_ = np.array([self.constant])
        estimator.n_outputs_ = 1
        return estimator

    def predict(self, X, **kwargs):
        feature_names_in = kwargs.get('feature_names_in', None)
        X = pd.DataFrame(X, columns=feature_names_in)
        return super().predict(X[self.feature_names_in_].to_numpy())
        # if isinstance(X, np.ndarray) and X.shape[1] == self.n_features_in_:
        #     return super().predict(X)
        # if isinstance(X, DataFrame):
        #     X = X[self.input_field_names].to_pandas()
        # elif isinstance(X, pd.core.frame.DataFrame):
        #     X = X[self.input_field_names]
        # else:
        #     raise NotImplementedError("Only pandas and eland DataFrame are supported.")
        # predictions = self.es_model.predict(X.to_numpy())
        # return predictions