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

from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union

import numpy as np

from .._model_serializer import Ensemble, Tree, TreeNode
from .._optional import import_optional_dependency
from ..common import TYPE_CLASSIFICATION, TYPE_REGRESSION
from .base import ModelTransformer

import_optional_dependency("sklearn", on_version="warn")

from sklearn.ensemble import (  # type: ignore
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore


class SKLearnTransformer(ModelTransformer):
    """
    Base class for SKLearn transformers.
    warning: Should not use this class directly. Use derived classes instead
    """

    def __init__(
        self,
        model: Any,
        feature_names: Sequence[str],
        classification_labels: Optional[Sequence[str]] = None,
        classification_weights: Optional[Sequence[float]] = None,
    ):
        """
        Base class for SKLearn transformations

        :param model: sklearn trained model
        :param feature_names: The feature names for the model
        :param classification_labels: Optional classification labels (if not encoded in the model)
        :param classification_weights: Optional classification weights
        """
        super().__init__(
            model, feature_names, classification_labels, classification_weights
        )
        self._node_decision_type = "lte"

    def build_tree_node(
        self,
        node_index: int,
        node_data: Tuple[Union[int, float], ...],
        value: np.ndarray,  # type: ignore
    ) -> TreeNode:
        """
        This builds out a TreeNode class given the sklearn tree node definition.

        Node decision types are defaulted to "lte" to match the behavior of SKLearn

        :param node_index: The node index
        :param node_data: Opaque node data contained in the sklearn tree state
        :param value: Opaque node value (i.e. leaf/node values) from tree state
        :return: TreeNode object
        """
        if value.shape[0] != 1:
            raise ValueError(
                f"unexpected multiple values returned from leaf node '{node_index}'"
            )
        if node_data[0] == -1:  # is leaf node
            if (
                value.shape[1] == 1
            ):  # classification requires more than one value, so assume regression
                leaf_value = [float(value[0][0])]
            else:
                # the classification value, which is the index of the largest value
                leaf_value = [float(np.argmax(value))]
            return TreeNode(
                node_index,
                decision_type=self._node_decision_type,
                leaf_value=leaf_value,
            )
        else:
            return TreeNode(
                node_index,
                decision_type=self._node_decision_type,
                left_child=int(node_data[0]),
                right_child=int(node_data[1]),
                split_feature=int(node_data[2]),
                threshold=float(node_data[3]),
            )


class SKLearnDecisionTreeTransformer(SKLearnTransformer):
    """
    class for transforming SKLearn decision tree models into Tree model formats supported by Elasticsearch.
    """

    def __init__(
        self,
        model: Union[DecisionTreeRegressor, DecisionTreeClassifier],
        feature_names: Sequence[str],
        classification_labels: Optional[Sequence[str]] = None,
    ):
        """
        Transforms a Decision Tree model (Regressor|Classifier) into a ES Supported Tree format
        :param model: fitted decision tree model
        :param feature_names: model feature names
        :param classification_labels: Optional classification labels
        """
        super().__init__(model, feature_names, classification_labels)

    def transform(self) -> Tree:
        """
        Transform the provided model into an ES supported Tree object
        :return: Tree object for ES storage and use
        """
        target_type = (
            "regression"
            if isinstance(self._model, DecisionTreeRegressor)
            else "classification"
        )
        check_is_fitted(self._model, ["tree_"])
        tree_classes = None
        if self._classification_labels:
            tree_classes = self._classification_labels
        if isinstance(self._model, DecisionTreeClassifier):
            check_is_fitted(self._model, ["classes_"])
            if tree_classes is None:
                tree_classes = [str(c) for c in self._model.classes_]
        nodes = []
        tree_state = self._model.tree_.__getstate__()
        for i in range(len(tree_state["nodes"])):
            nodes.append(
                self.build_tree_node(i, tree_state["nodes"][i], tree_state["values"][i])
            )

        return Tree(self._feature_names, target_type, nodes, tree_classes)

    @property
    def model_type(self) -> str:
        return (
            TYPE_REGRESSION
            if isinstance(self._model, DecisionTreeRegressor)
            else TYPE_CLASSIFICATION
        )


class SKLearnForestTransformer(SKLearnTransformer):
    """
    Base class for transforming SKLearn forest models into Ensemble model formats supported by Elasticsearch.

    warning: do not use this class directly. Use a derived class instead
    """

    def __init__(
        self,
        model: Union[RandomForestClassifier, RandomForestRegressor],
        feature_names: Sequence[str],
        classification_labels: Optional[Sequence[str]] = None,
        classification_weights: Optional[Sequence[float]] = None,
    ):
        super().__init__(
            model, feature_names, classification_labels, classification_weights
        )

    def build_aggregator_output(self) -> Dict[str, Any]:
        raise NotImplementedError("build_aggregator_output must be implemented")

    def determine_target_type(self) -> str:
        raise NotImplementedError("determine_target_type must be implemented")

    def transform(self) -> Ensemble:
        check_is_fitted(self._model, ["estimators_"])
        estimators = self._model.estimators_
        ensemble_classes = None
        if self._classification_labels:
            ensemble_classes = self._classification_labels
        if isinstance(self._model, RandomForestClassifier):
            check_is_fitted(self._model, ["classes_"])
            if ensemble_classes is None:
                ensemble_classes = [str(c) for c in self._model.classes_]
        ensemble_models: Sequence[Tree] = [
            SKLearnDecisionTreeTransformer(m, self._feature_names).transform()
            for m in estimators
        ]
        return Ensemble(
            self._feature_names,
            ensemble_models,
            self.build_aggregator_output(),
            target_type=self.determine_target_type(),
            classification_labels=ensemble_classes,
            classification_weights=self._classification_weights,
        )


class SKLearnForestRegressorTransformer(SKLearnForestTransformer):
    """
    Class for transforming RandomForestRegressor models into an ensemble model supported by Elasticsearch
    """

    def __init__(self, model: RandomForestRegressor, feature_names: Sequence[str]):
        super().__init__(model, feature_names)

    def build_aggregator_output(self) -> Dict[str, Any]:
        return {
            "weighted_sum": {
                "weights": [1.0 / len(self._model.estimators_)]
                * len(self._model.estimators_),
            }
        }

    def determine_target_type(self) -> str:
        return "regression"

    @property
    def model_type(self) -> str:
        return TYPE_REGRESSION


class SKLearnForestClassifierTransformer(SKLearnForestTransformer):
    """
    Class for transforming RandomForestClassifier models into an ensemble model supported by Elasticsearch
    """

    def __init__(
        self,
        model: RandomForestClassifier,
        feature_names: Sequence[str],
        classification_labels: Optional[Sequence[str]] = None,
    ):
        super().__init__(model, feature_names, classification_labels)

    def build_aggregator_output(self) -> Dict[str, Any]:
        return {"weighted_mode": {"num_classes": len(self._model.classes_)}}

    def determine_target_type(self) -> str:
        return "classification"

    @property
    def model_type(self) -> str:
        return TYPE_CLASSIFICATION


_MODEL_TRANSFORMERS: Dict[type, Type[ModelTransformer]] = {
    DecisionTreeRegressor: SKLearnDecisionTreeTransformer,
    DecisionTreeClassifier: SKLearnDecisionTreeTransformer,
    RandomForestRegressor: SKLearnForestRegressorTransformer,
    RandomForestClassifier: SKLearnForestClassifierTransformer,
}
