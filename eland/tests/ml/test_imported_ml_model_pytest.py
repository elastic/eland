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

import pytest
import numpy as np

from eland.ml import ImportedMLModel
from eland.tests import ES_TEST_CLIENT


try:
    from sklearn import datasets
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from xgboost import XGBRegressor, XGBClassifier

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


requires_sklearn = pytest.mark.skipif(
    not HAS_SKLEARN, reason="This test requires 'scikit-learn' package to run"
)
requires_xgboost = pytest.mark.skipif(
    not HAS_XGBOOST, reason="This test requires 'xgboost' package to run"
)
requires_no_ml_extras = pytest.mark.skipif(
    HAS_SKLEARN or HAS_XGBOOST,
    reason="This test requires 'scikit-learn' and 'xgboost' to not be installed",
)


class TestImportedMLModel:
    @requires_no_ml_extras
    def test_import_ml_model_when_dependencies_are_not_available(self):
        from eland.ml import MLModel, ImportedMLModel  # noqa: F401

    @requires_sklearn
    def test_unpack_and_raise_errors_in_ingest_simulate(self, mocker):
        # Train model
        training_data = datasets.make_classification(n_features=5)
        classifier = DecisionTreeClassifier()
        classifier.fit(training_data[0], training_data[1])

        # Serialise the models to Elasticsearch
        feature_names = ["f0", "f1", "f2", "f3", "f4"]
        model_id = "test_decision_tree_classifier"
        test_data = [[0.1, 0.2, 0.3, -0.5, 1.0], [1.6, 2.1, -10, 50, -1.0]]

        es_model = ImportedMLModel(
            ES_TEST_CLIENT,
            model_id,
            classifier,
            feature_names,
            overwrite=True,
            es_compress_model_definition=True,
        )

        # Mock the ingest.simulate API to return an error within {'docs': [...]}
        mock = mocker.patch.object(ES_TEST_CLIENT.ingest, "simulate")
        mock.return_value = {
            "docs": [
                {
                    "error": {
                        "type": "x_content_parse_exception",
                        "reason": "[1:1052] [inference_model_definition] failed to parse field [trained_model]",
                    }
                }
            ]
        }

        with pytest.raises(RuntimeError) as err:
            es_model.predict(test_data)

        assert repr(err.value) == (
            'RuntimeError("Failed to run prediction for model ID '
            "'test_decision_tree_classifier'\", {'type': 'x_content_parse_exception', "
            "'reason': '[1:1052] [inference_model_definition] failed to parse "
            "field [trained_model]'})"
        )

    @requires_sklearn
    @pytest.mark.parametrize("compress_model_definition", [True, False])
    def test_decision_tree_classifier(self, compress_model_definition):
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

        es_model = ImportedMLModel(
            ES_TEST_CLIENT,
            model_id,
            classifier,
            feature_names,
            overwrite=True,
            es_compress_model_definition=compress_model_definition,
        )
        es_results = es_model.predict(test_data)

        np.testing.assert_almost_equal(test_results, es_results, decimal=2)

        # Clean up
        es_model.delete_model()

    @requires_sklearn
    @pytest.mark.parametrize("compress_model_definition", [True, False])
    def test_decision_tree_regressor(self, compress_model_definition):
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

        es_model = ImportedMLModel(
            ES_TEST_CLIENT,
            model_id,
            regressor,
            feature_names,
            overwrite=True,
            es_compress_model_definition=compress_model_definition,
        )
        es_results = es_model.predict(test_data)

        np.testing.assert_almost_equal(test_results, es_results, decimal=2)

        # Clean up
        es_model.delete_model()

    @requires_sklearn
    @pytest.mark.parametrize("compress_model_definition", [True, False])
    def test_random_forest_classifier(self, compress_model_definition):
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

        es_model = ImportedMLModel(
            ES_TEST_CLIENT,
            model_id,
            classifier,
            feature_names,
            overwrite=True,
            es_compress_model_definition=compress_model_definition,
        )
        es_results = es_model.predict(test_data)

        np.testing.assert_almost_equal(test_results, es_results, decimal=2)

        # Clean up
        es_model.delete_model()

    @requires_sklearn
    @pytest.mark.parametrize("compress_model_definition", [True, False])
    def test_random_forest_regressor(self, compress_model_definition):
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

        es_model = ImportedMLModel(
            ES_TEST_CLIENT,
            model_id,
            regressor,
            feature_names,
            overwrite=True,
            es_compress_model_definition=compress_model_definition,
        )
        es_results = es_model.predict(test_data)

        np.testing.assert_almost_equal(test_results, es_results, decimal=2)

        # Clean up
        es_model.delete_model()

    @requires_xgboost
    @pytest.mark.parametrize("compress_model_definition", [True, False])
    @pytest.mark.parametrize("multi_class", [True, False])
    def test_xgb_classifier(self, compress_model_definition, multi_class):
        # test both multiple and binary classification
        if multi_class:
            training_data = datasets.make_classification(
                n_features=5, n_classes=3, n_informative=3
            )
            classifier = XGBClassifier(booster="gbtree", objective="multi:softmax")
        else:
            training_data = datasets.make_classification(n_features=5)
            classifier = XGBClassifier(booster="gbtree")

        # Train model
        classifier.fit(training_data[0], training_data[1])

        # Get some test results
        test_data = [[0.1, 0.2, 0.3, -0.5, 1.0], [1.6, 2.1, -10, 50, -1.0]]
        test_results = classifier.predict(np.asarray(test_data))

        # Serialise the models to Elasticsearch
        feature_names = ["f0", "f1", "f2", "f3", "f4"]
        model_id = "test_xgb_classifier"

        es_model = ImportedMLModel(
            ES_TEST_CLIENT,
            model_id,
            classifier,
            feature_names,
            overwrite=True,
            es_compress_model_definition=compress_model_definition,
        )
        es_results = es_model.predict(test_data)

        np.testing.assert_almost_equal(test_results, es_results, decimal=2)

        # Clean up
        es_model.delete_model()

    @requires_xgboost
    @pytest.mark.parametrize("compress_model_definition", [True, False])
    def test_xgb_regressor(self, compress_model_definition):
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

        es_model = ImportedMLModel(
            ES_TEST_CLIENT,
            model_id,
            regressor,
            feature_names,
            overwrite=True,
            es_compress_model_definition=compress_model_definition,
        )

        es_results = es_model.predict(test_data)

        np.testing.assert_almost_equal(test_results, es_results, decimal=2)

        # Clean up
        es_model.delete_model()

    @requires_xgboost
    def test_predict_single_feature_vector(self):
        # Train model
        training_data = datasets.make_regression(n_features=1)
        regressor = XGBRegressor()
        regressor.fit(training_data[0], training_data[1])

        # Get some test results
        test_data = [[0.1]]
        test_results = regressor.predict(np.asarray(test_data))

        # Serialise the models to Elasticsearch
        feature_names = ["f0"]
        model_id = "test_xgb_regressor"

        es_model = ImportedMLModel(
            ES_TEST_CLIENT, model_id, regressor, feature_names, overwrite=True
        )

        # Single feature
        es_results = es_model.predict(test_data[0])

        np.testing.assert_almost_equal(test_results, es_results, decimal=2)

        # Clean up
        es_model.delete_model()
