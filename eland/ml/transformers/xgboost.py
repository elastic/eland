# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

from typing import Optional, List, Dict, Any, Type
from .base import ModelTransformer
import pandas as pd  # type: ignore
from .._model_serializer import Ensemble, Tree, TreeNode
from ..ml_model import MLModel
from .._optional import import_optional_dependency

import_optional_dependency("xgboost", on_version="warn")

from xgboost import Booster, XGBRegressor, XGBClassifier  # type: ignore


class XGBoostForestTransformer(ModelTransformer):
    """
    Base class for transforming XGBoost models into ensemble models supported by Elasticsearch

    warning: do not use directly. Use a derived classes instead
    """

    def __init__(
        self,
        model: Booster,
        feature_names: List[str],
        base_score: float = 0.5,
        objective: str = "reg:squarederror",
        classification_labels: Optional[List[str]] = None,
        classification_weights: Optional[List[float]] = None,
    ):
        super().__init__(
            model, feature_names, classification_labels, classification_weights
        )
        self._node_decision_type = "lt"
        self._base_score = base_score
        self._objective = objective

    def get_feature_id(self, feature_id: str) -> int:
        if feature_id[0] == "f":
            try:
                return int(feature_id[1:])
            except ValueError:
                raise RuntimeError(f"Unable to interpret '{feature_id}'")
        else:
            try:
                return int(feature_id)
            except ValueError:
                raise RuntimeError(f"Unable to interpret '{feature_id}'")

    def extract_node_id(self, node_id: str, curr_tree: int) -> int:
        t_id, n_id = node_id.split("-")
        if t_id is None or n_id is None:
            raise RuntimeError(
                f"cannot determine node index or tree from '{node_id}' for tree {curr_tree}"
            )
        try:
            l_id = int(t_id)
            r_id = int(n_id)
            if l_id != curr_tree:
                raise RuntimeError(
                    f"extracted tree id {l_id} does not match current tree {curr_tree}"
                )
            return r_id
        except ValueError:
            raise RuntimeError(
                f"cannot determine node index or tree from '{node_id}' for tree {curr_tree}"
            )

    def build_tree_node(self, row: pd.Series, curr_tree: int) -> TreeNode:
        node_index = row["Node"]
        if row["Feature"] == "Leaf":
            return TreeNode(node_idx=node_index, leaf_value=float(row["Gain"]))
        else:
            return TreeNode(
                node_idx=node_index,
                decision_type=self._node_decision_type,
                left_child=self.extract_node_id(row["Yes"], curr_tree),
                right_child=self.extract_node_id(row["No"], curr_tree),
                threshold=float(row["Split"]),
                split_feature=self.get_feature_id(row["Feature"]),
            )

    def build_tree(self, nodes: List[TreeNode]) -> Tree:
        return Tree(feature_names=self._feature_names, tree_structure=nodes)

    def build_base_score_stump(self) -> Tree:
        return Tree(
            feature_names=self._feature_names,
            tree_structure=[TreeNode(0, leaf_value=self._base_score)],
        )

    def build_forest(self) -> List[Tree]:
        """
        This builds out the forest of trees as described by XGBoost into a format
        supported by Elasticsearch

        :return: A list of Tree objects
        """
        self.check_model_booster()

        tree_table: pd.DataFrame = self._model.trees_to_dataframe()
        transformed_trees = []
        curr_tree: Optional[Any] = None
        tree_nodes: List[TreeNode] = []
        for _, row in tree_table.iterrows():
            if row["Tree"] != curr_tree:
                if len(tree_nodes) > 0:
                    transformed_trees.append(self.build_tree(tree_nodes))
                curr_tree = row["Tree"]
                tree_nodes = []
            tree_nodes.append(self.build_tree_node(row, curr_tree))
            # add last tree
        if len(tree_nodes) > 0:
            transformed_trees.append(self.build_tree(tree_nodes))
        # We add this stump as XGBoost adds the base_score to the regression outputs
        if self._objective.partition(":")[0] == "reg":
            transformed_trees.append(self.build_base_score_stump())
        return transformed_trees

    def build_aggregator_output(self) -> Dict[str, Any]:
        raise NotImplementedError("build_aggregator_output must be implemented")

    def determine_target_type(self) -> str:
        raise NotImplementedError("determine_target_type must be implemented")

    def is_objective_supported(self) -> bool:
        return False

    def check_model_booster(self) -> None:
        # xgboost v1 made booster default to 'None' meaning 'gbtree'
        if self._model.booster not in {"dart", "gbtree", None}:
            raise ValueError(
                f"booster must exist and be of type 'dart' or "
                f"'gbtree', was {self._model.booster!r}"
            )

    def transform(self) -> Ensemble:
        self.check_model_booster()

        if not self.is_objective_supported():
            raise ValueError(f"Unsupported objective '{self._objective}'")

        forest = self.build_forest()
        return Ensemble(
            feature_names=self._feature_names,
            trained_models=forest,
            output_aggregator=self.build_aggregator_output(),
            classification_labels=self._classification_labels,
            classification_weights=self._classification_weights,
            target_type=self.determine_target_type(),
        )


class XGBoostRegressorTransformer(XGBoostForestTransformer):
    def __init__(self, model: XGBRegressor, feature_names: List[str]):
        # XGBRegressor.base_score defaults to 0.5.
        base_score = model.base_score
        if base_score is None:
            base_score = 0.5
        super().__init__(
            model.get_booster(), feature_names, base_score, model.objective
        )

    def determine_target_type(self) -> str:
        return "regression"

    def is_objective_supported(self) -> bool:
        return self._objective in {
            "reg:squarederror",
            "reg:linear",
            "reg:squaredlogerror",
            "reg:logistic",
        }

    def build_aggregator_output(self) -> Dict[str, Any]:
        return {"weighted_sum": {}}

    @property
    def model_type(self) -> str:
        return MLModel.TYPE_REGRESSION


class XGBoostClassifierTransformer(XGBoostForestTransformer):
    def __init__(
        self,
        model: XGBClassifier,
        feature_names: List[str],
        classification_labels: Optional[List[str]] = None,
    ):
        super().__init__(
            model.get_booster(),
            feature_names,
            model.base_score,
            model.objective,
            classification_labels,
        )

    def determine_target_type(self) -> str:
        return "classification"

    def is_objective_supported(self) -> bool:
        return self._objective in {"binary:logistic", "binary:hinge"}

    def build_aggregator_output(self) -> Dict[str, Any]:
        return {"logistic_regression": {}}

    @property
    def model_type(self) -> str:
        return MLModel.TYPE_CLASSIFICATION


_MODEL_TRANSFORMERS: Dict[type, Type[ModelTransformer]] = {
    XGBRegressor: XGBoostRegressorTransformer,
    XGBClassifier: XGBoostClassifierTransformer,
}
