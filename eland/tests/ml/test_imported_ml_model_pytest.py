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
#
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBRegressor, XGBClassifier

from eland.ml import ImportedMLModel
from eland.tests import ES_TEST_CLIENT


class TestImportedMLModel:

    def test_decision_tree_classifier(self):
        # Train model
        training_data = datasets.make_classification(n_features=5)
        classifier = DecisionTreeClassifier()
        classifier.fit(training_data[0], training_data[1])

        # Get some test results
        test_data = [[0.1, 0.2, 0.3, -0.5, 1.0], [1.6, 2.1, -10, 50, -1.0]]
        test_results = classifier.predict(test_data)

        # Serialise the models to Elasticsearch
        feature_names = ["f0", "f1", "f2", "f3", "f4"]
        model_id = "test_decision_tree_classifier"

        es_model = ImportedMLModel(ES_TEST_CLIENT, model_id, classifier, feature_names, overwrite=True)
        es_results = es_model.predict(test_data)

        np.testing.assert_almost_equal(test_results, es_results, decimal=4)

        # Clean up
        es_model.delete_model()

    def test_decision_tree_regressor(self):
        # Train model
        training_data = datasets.make_regression(n_features=5)
        regressor = DecisionTreeRegressor()
        regressor.fit(training_data[0], training_data[1])

        # Get some test results
        test_data = [[0.1, 0.2, 0.3, -0.5, 1.0], [1.6, 2.1, -10, 50, -1.0]]
        test_results = regressor.predict(test_data)

        # Serialise the models to Elasticsearch
        feature_names = ["f0", "f1", "f2", "f3", "f4"]
        model_id = "test_decision_tree_regressor"

        es_model = ImportedMLModel(ES_TEST_CLIENT, model_id, regressor, feature_names, overwrite=True)
        es_results = es_model.predict(test_data)

        np.testing.assert_almost_equal(test_results, es_results, decimal=4)

        # Clean up
        es_model.delete_model()

    def test_random_forest_classifier(self):
        # Train model
        training_data = datasets.make_classification(n_features=5)
        classifier = RandomForestClassifier()
        classifier.fit(training_data[0], training_data[1])

        # Get some test results
        test_data = [[0.1, 0.2, 0.3, -0.5, 1.0], [1.6, 2.1, -10, 50, -1.0]]
        test_results = classifier.predict(test_data)

        # Serialise the models to Elasticsearch
        feature_names = ["f0", "f1", "f2", "f3", "f4"]
        model_id = "test_random_forest_classifier"

        es_model = ImportedMLModel(ES_TEST_CLIENT, model_id, classifier, feature_names, overwrite=True)
        es_results = es_model.predict(test_data)

        np.testing.assert_almost_equal(test_results, es_results, decimal=4)

        # Clean up
        es_model.delete_model()

    def test_random_forest_regressor(self):
        # Train model
        training_data = datasets.make_regression(n_features=5)
        regressor = RandomForestRegressor()
        regressor.fit(training_data[0], training_data[1])

        # Get some test results
        test_data = [[0.1, 0.2, 0.3, -0.5, 1.0], [1.6, 2.1, -10, 50, -1.0]]
        test_results = regressor.predict(test_data)

        # Serialise the models to Elasticsearch
        feature_names = ["f0", "f1", "f2", "f3", "f4"]
        model_id = "test_random_forest_regressor"

        es_model = ImportedMLModel(ES_TEST_CLIENT, model_id, regressor, feature_names, overwrite=True)
        es_results = es_model.predict(test_data)

        np.testing.assert_almost_equal(test_results, es_results, decimal=4)

        # Clean up
        es_model.delete_model()

    def test_xgb_classifier(self):
        # Train model
        training_data = datasets.make_classification(n_features=5)
        classifier = XGBClassifier()
        classifier.fit(training_data[0], training_data[1])

        # Get some test results
        test_data = [[0.1, 0.2, 0.3, -0.5, 1.0], [1.6, 2.1, -10, 50, -1.0]]
        test_results = classifier.predict(np.asarray(test_data))

        # Serialise the models to Elasticsearch
        feature_names = ["f0", "f1", "f2", "f3", "f4"]
        model_id = "test_xgb_classifier"

        es_model = ImportedMLModel(ES_TEST_CLIENT, model_id, classifier, feature_names, overwrite=True)
        es_results = es_model.predict(test_data)

        np.testing.assert_almost_equal(test_results, es_results, decimal=4)

        # Clean up
        es_model.delete_model()

    def test_xgb_regressor(self):
        # Train model
        training_data = datasets.make_regression(n_features=5)
        regressor = XGBRegressor()
        regressor.fit(training_data[0], training_data[1])

        # Get some test results
        test_data = [[0.1, 0.2, 0.3, -0.5, 1.0], [1.6, 2.1, -10, 50, -1.0]]
        test_results = regressor.predict(np.asarray(test_data))

        # Serialise the models to Elasticsearch
        feature_names = ["f0", "f1", "f2", "f3", "f4"]
        model_id = "test_xgb_regressor"

        es_model = ImportedMLModel(ES_TEST_CLIENT, model_id, regressor, feature_names, overwrite=True)
        es_results = es_model.predict(test_data)

        np.testing.assert_almost_equal(test_results, es_results, decimal=4)

        # Clean up
        es_model.delete_model()
