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
from eland.tests import ES_TEST_CLIENT

from eland.ml.model import ExternalModel
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


def test_decision_tree_classifier():
    # Train model
    training_data = datasets.make_classification(n_features=5)

    classifier = DecisionTreeClassifier()

    classifier.fit(training_data[0], training_data[1])

    # Get some test results
    test_data = [[0.1, 0.2, 0.3, -0.5, 1.0], [1.6, 2.1, -10, 50, -1.0]]

    test_results = classifier.predict(test_data)

    # Serialise the models to Elasticsearch
    feature_names = ["f0", "f1", "f2", "f3", "f4"]

    model_id = "sklearn_decision_tree_classifier_pytest"

    es_model = ExternalModel(ES_TEST_CLIENT, model_id, classifier, feature_names)

    es_model.predict(test_data)

