#  Copyright 2019 Elasticsearch BV
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

from typing import List, Union

import numpy as np

from eland.ml._optional import import_optional_dependency
from eland.ml._model_serializer import Tree, TreeNode, Ensemble

sklearn = import_optional_dependency("sklearn")
xgboost = import_optional_dependency("xgboost")

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted
from xgboost import Booster, XGBRegressor, XGBClassifier


class ModelTransformer:
    def __init__(self,
                 model,
                 feature_names: List[str],
                 classification_labels: List[str] = None,
                 classification_weights: List[float] = None
                 ):
        self._feature_names = feature_names
        self._model = model
        self._classification_labels = classification_labels
        self._classification_weights = classification_weights

    def is_supported(self):
        return isinstance(self._model, (DecisionTreeClassifier,
                                        DecisionTreeRegressor,
                                        RandomForestRegressor,
                                        RandomForestClassifier,
                                        XGBClassifier,
                                        XGBRegressor,
                                        Booster))


class SKLearnTransformer(ModelTransformer):
    """
    Base class for SKLearn transformers.
    warning: Should not use this class directly. Use derived classes instead
    """

    def __init__(self,
                 model,
                 feature_names: List[str],
                 classification_labels: List[str] = None,
                 classification_weights: List[float] = None
                 ):
        """
        Base class for SKLearn transformations

        :param model: sklearn trained model
        :param feature_names: The feature names for the model
        :param classification_labels: Optional classification labels (if not encoded in the model)
        :param classification_weights: Optional classification weights
        """
        super().__init__(model, feature_names, classification_labels, classification_weights)
        self._node_decision_type = "lte"

    def build_tree_node(self, node_index: int, node_data: dict, value) -> TreeNode:
        """
        This builds out a TreeNode class given the sklearn tree node definition.

        Node decision types are defaulted to "lte" to match the behavior of SKLearn

        :param node_index: The node index
        :param node_data: Opaque node data contained in the sklearn tree state
        :param value: Opaque node value (i.e. leaf/node values) from tree state
        :return: TreeNode object
        """
        if value.shape[0] != 1:
            raise ValueError("unexpected multiple values returned from leaf node '{0}'".format(node_index))
        if node_data[0] == -1:  # is leaf node
            if value.shape[1] == 1:  # classification requires more than one value, so assume regression
                leaf_value = float(value[0][0])
            else:
                # the classification value, which is the index of the largest value
                leaf_value = int(np.argmax(value))
            return TreeNode(node_index, decision_type=self._node_decision_type, leaf_value=leaf_value)
        else:
            return TreeNode(node_index,
                            decision_type=self._node_decision_type,
                            left_child=int(node_data[0]),
                            right_child=int(node_data[1]),
                            split_feature=int(node_data[2]),
                            threshold=float(node_data[3]))


class SKLearnDecisionTreeTransformer(SKLearnTransformer):
    """
    class for transforming SKLearn decision tree models into Tree model formats supported by Elasticsearch.
    """

    def __init__(self,
                 model: Union[DecisionTreeRegressor, DecisionTreeClassifier],
                 feature_names: List[str],
                 classification_labels: List[str] = None):
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
        target_type = "regression" if isinstance(self._model, DecisionTreeRegressor) else "classification"
        check_is_fitted(self._model, ["tree_"])
        tree_classes = None
        if self._classification_labels:
            tree_classes = self._classification_labels
        if isinstance(self._model, DecisionTreeClassifier):
            check_is_fitted(self._model, ["classes_"])
            if tree_classes is None:
                tree_classes = [str(c) for c in self._model.classes_]
        nodes = list()
        tree_state = self._model.tree_.__getstate__()
        for i in range(len(tree_state["nodes"])):
            nodes.append(self.build_tree_node(i, tree_state["nodes"][i], tree_state["values"][i]))

        return Tree(self._feature_names,
                    target_type,
                    nodes,
                    tree_classes)


class SKLearnForestTransformer(SKLearnTransformer):
    """
    Base class for transforming SKLearn forest models into Ensemble model formats supported by Elasticsearch.

    warning: do not use this class directly. Use a derived class instead
    """

    def __init__(self,
                 model: Union[RandomForestClassifier,
                              RandomForestRegressor],
                 feature_names: List[str],
                 classification_labels: List[str] = None,
                 classification_weights: List[float] = None
                 ):
        super().__init__(model, feature_names, classification_labels, classification_weights)

    def build_aggregator_output(self) -> dict:
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
        ensemble_models = [SKLearnDecisionTreeTransformer(m,
                                                          self._feature_names).transform() for m in estimators]
        return Ensemble(self._feature_names,
                        ensemble_models,
                        self.build_aggregator_output(),
                        target_type=self.determine_target_type(),
                        classification_labels=ensemble_classes,
                        classification_weights=self._classification_weights)


class SKLearnForestRegressorTransformer(SKLearnForestTransformer):
    """
    Class for transforming RandomForestRegressor models into an ensemble model supported by Elasticsearch
    """

    def __init__(self,
                 model: RandomForestRegressor,
                 feature_names: List[str]
                 ):
        super().__init__(model, feature_names)

    def build_aggregator_output(self) -> dict:
        return {
            "weighted_sum": {"weights": [1.0 / len(self._model.estimators_)] * len(self._model.estimators_), }
        }

    def determine_target_type(self) -> str:
        return "regression"


class SKLearnForestClassifierTransformer(SKLearnForestTransformer):
    """
    Class for transforming RandomForestClassifier models into an ensemble model supported by Elasticsearch
    """

    def __init__(self,
                 model: RandomForestClassifier,
                 feature_names: List[str],
                 classification_labels: List[str] = None,
                 ):
        super().__init__(model, feature_names, classification_labels)

    def build_aggregator_output(self) -> dict:
        return {"weighted_mode": {"num_classes": len(self._model.classes_)}}

    def determine_target_type(self) -> str:
        return "classification"


class XGBoostForestTransformer(ModelTransformer):
    """
    Base class for transforming XGBoost models into ensemble models supported by Elasticsearch

    warning: do not use directly. Use a derived classes instead
    """

    def __init__(self,
                 model: Booster,
                 feature_names: List[str],
                 base_score: float = 0.5,
                 objective: str = "reg:squarederror",
                 classification_labels: List[str] = None,
                 classification_weights: List[float] = None
                 ):
        super().__init__(model, feature_names, classification_labels, classification_weights)
        self._node_decision_type = "lt"
        self._base_score = base_score
        self._objective = objective

    def get_feature_id(self, feature_id: str) -> int:
        if feature_id[0] == "f":
            try:
                return int(feature_id[1:])
            except ValueError:
                raise RuntimeError("Unable to interpret '{0}'".format(feature_id))
        else:
            try:
                return int(feature_id)
            except ValueError:
                raise RuntimeError("Unable to interpret '{0}'".format(feature_id))

    def extract_node_id(self, node_id: str, curr_tree: int) -> int:
        t_id, n_id = node_id.split("-")
        if t_id is None or n_id is None:
            raise RuntimeError(
                "cannot determine node index or tree from '{0}' for tree {1}".format(node_id, curr_tree))
        try:
            t_id = int(t_id)
            n_id = int(n_id)
            if t_id != curr_tree:
                raise RuntimeError("extracted tree id {0} does not match current tree {1}".format(t_id, curr_tree))
            return n_id
        except ValueError:
            raise RuntimeError(
                "cannot determine node index or tree from '{0}' for tree {1}".format(node_id, curr_tree))

    def build_tree_node(self, row, curr_tree: int) -> TreeNode:
        node_index = row["Node"]
        if row["Feature"] == "Leaf":
            return TreeNode(node_idx=node_index, leaf_value=float(row["Gain"]))
        else:
            return TreeNode(node_idx=node_index,
                            decision_type=self._node_decision_type,
                            left_child=self.extract_node_id(row["Yes"], curr_tree),
                            right_child=self.extract_node_id(row["No"], curr_tree),
                            threshold=float(row["Split"]),
                            split_feature=self.get_feature_id(row["Feature"]))

    def build_tree(self, nodes: List[TreeNode]) -> Tree:
        return Tree(feature_names=self._feature_names,
                    tree_structure=nodes)

    def build_base_score_stump(self) -> Tree:
        return Tree(feature_names=self._feature_names,
                    tree_structure=[TreeNode(0, leaf_value=self._base_score)])

    def build_forest(self) -> List[Tree]:
        """
        This builds out the forest of trees as described by XGBoost into a format
        supported by Elasticsearch

        :return: A list of Tree objects
        """
        if self._model.booster not in {'dart', 'gbtree'}:
            raise ValueError("booster must exist and be of type dart or gbtree")

        tree_table = self._model.trees_to_dataframe()
        transformed_trees = list()
        curr_tree = None
        tree_nodes = list()
        for _, row in tree_table.iterrows():
            if row["Tree"] != curr_tree:
                if len(tree_nodes) > 0:
                    transformed_trees.append(self.build_tree(tree_nodes))
                curr_tree = row["Tree"]
                tree_nodes = list()
            tree_nodes.append(self.build_tree_node(row, curr_tree))
            # add last tree
        if len(tree_nodes) > 0:
            transformed_trees.append(self.build_tree(tree_nodes))
        # We add this stump as XGBoost adds the base_score to the regression outputs
        if self._objective.startswith("reg"):
            transformed_trees.append(self.build_base_score_stump())
        return transformed_trees

    def build_aggregator_output(self) -> dict:
        raise NotImplementedError("build_aggregator_output must be implemented")

    def determine_target_type(self) -> str:
        raise NotImplementedError("determine_target_type must be implemented")

    def is_objective_supported(self) -> bool:
        return False

    def transform(self) -> Ensemble:
        if self._model.booster not in {'dart', 'gbtree'}:
            raise ValueError("booster must exist and be of type dart or gbtree")

        if not self.is_objective_supported():
            raise ValueError("Unsupported objective '{0}'".format(self._objective))

        forest = self.build_forest()
        return Ensemble(feature_names=self._feature_names,
                        trained_models=forest,
                        output_aggregator=self.build_aggregator_output(),
                        classification_labels=self._classification_labels,
                        classification_weights=self._classification_weights,
                        target_type=self.determine_target_type())


class XGBoostRegressorTransformer(XGBoostForestTransformer):
    def __init__(self,
                 model: XGBRegressor,
                 feature_names: List[str]):
        super().__init__(model.get_booster(),
                         feature_names,
                         model.base_score,
                         model.objective)

    def determine_target_type(self) -> str:
        return "regression"

    def is_objective_supported(self) -> bool:
        return self._objective in {'reg:squarederror',
                                   'reg:linear',
                                   'reg:squaredlogerror',
                                   'reg:logistic'}

    def build_aggregator_output(self) -> dict:
        return {"weighted_sum": {}}


class XGBoostClassifierTransformer(XGBoostForestTransformer):
    def __init__(self,
                 model: XGBClassifier,
                 feature_names: List[str],
                 classification_labels: List[str] = None):
        super().__init__(model.get_booster(),
                         feature_names,
                         model.base_score,
                         model.objective,
                         classification_labels)

    def determine_target_type(self) -> str:
        return "classification"

    def is_objective_supported(self) -> bool:
        return self._objective in {'binary:logistic', 'binary:hinge'}

    def build_aggregator_output(self) -> dict:
        return {"logistic_regression": {}}
