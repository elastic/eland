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

from typing import Optional, List, Dict, Any, Type
from .base import ModelTransformer
from .._model_serializer import Ensemble, Tree, TreeNode
from ..ml_model import MLModel
from .._optional import import_optional_dependency

import_optional_dependency("lightgbm", on_version="warn")

from lightgbm import Booster, LGBMRegressor  # type: ignore


def transform_decider(decider: str) -> str:
    if decider == "<=":
        return "lte"
    if decider == "<":
        return "lt"
    if decider == ">":
        return "gt"
    if decider == ">=":
        return "gte"
    raise ValueError(
        "Unsupported splitting decider: %s. Only <=, <, >=, and > are allowed."
    )


class Counter:
    def __init__(self, start: int = 0):
        self._value = start

    def inc(self) -> "Counter":
        self._value += 1
        return self

    def value(self) -> int:
        return self._value


class LGBMForestTransformer(ModelTransformer):
    """
    Base class for transforming LightGBM models into ensemble models supported by Elasticsearch

    warning: do not use directly. Use a derived classes instead
    """

    def __init__(
        self,
        model: Booster,
        feature_names: List[str],
        classification_labels: Optional[List[str]] = None,
        classification_weights: Optional[List[float]] = None,
    ):
        super().__init__(
            model, feature_names, classification_labels, classification_weights
        )
        self._node_decision_type = "lte"
        self._objective = model.params["objective"]

    def build_tree(self, tree_json_obj: Dict[str, Any]) -> Tree:
        tree_nodes = list()
        next_id = Counter()

        def add_tree_node(tree_node_json_obj: Dict[str, Any], counter: Counter) -> int:
            curr_id = counter.value()
            if "leaf_value" in tree_node_json_obj:
                tree_nodes.append(
                    TreeNode(
                        node_idx=curr_id,
                        leaf_value=[float(tree_node_json_obj["leaf_value"])],
                    )
                )
                return curr_id
            left_id = add_tree_node(tree_node_json_obj["left_child"], counter.inc())
            right_id = add_tree_node(tree_node_json_obj["right_child"], counter.inc())
            tree_nodes.append(
                TreeNode(
                    node_idx=curr_id,
                    default_left=tree_node_json_obj["default_left"],
                    split_feature=tree_node_json_obj["split_feature"],
                    threshold=float(tree_node_json_obj["threshold"]),
                    decision_type=transform_decider(
                        tree_node_json_obj["decision_type"]
                    ),
                    left_child=left_id,
                    right_child=right_id,
                )
            )
            return curr_id

        add_tree_node(tree_json_obj["tree_structure"], next_id)
        tree_nodes.sort(key=lambda n: n.node_idx)
        return Tree(
            feature_names=self._feature_names,
            target_type=self.determine_target_type(),
            tree_structure=tree_nodes,
        )

    def build_forest(self) -> List[Tree]:
        """
        This builds out the forest of trees as described by LightGBM into a format
        supported by Elasticsearch

        :return: A list of Tree objects
        """
        self.check_model_booster()
        json_dump = self._model.dump_model()
        return [self.build_tree(t) for t in json_dump["tree_info"]]

    def build_aggregator_output(self) -> Dict[str, Any]:
        raise NotImplementedError("build_aggregator_output must be implemented")

    def determine_target_type(self) -> str:
        raise NotImplementedError("determine_target_type must be implemented")

    def is_objective_supported(self) -> bool:
        return False

    def check_model_booster(self) -> None:
        raise NotImplementedError("check_model_booster must be implemented")

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


class LGBMRegressorTransformer(LGBMForestTransformer):
    def __init__(self, model: LGBMRegressor, feature_names: List[str]):
        super().__init__(model.booster_, feature_names)

    def is_objective_supported(self) -> bool:
        return self._objective in {
            "regression",
            "regression_l1",
            "huber",
            "fair",
            "quantile",
            "mape",
        }

    def check_model_booster(self) -> None:
        if self._model.params["boosting_type"] not in {"gbdt", "rf", "dart", "goss"}:
            raise ValueError(
                f"boosting type must exist and be of type 'gbdt', 'rf', 'dart', or 'goss'"
                f", was {self._model.params['boosting_type']!r}"
            )

    def determine_target_type(self) -> str:
        return "regression"

    def build_aggregator_output(self) -> Dict[str, Any]:
        return {"weighted_sum": {}}

    @property
    def model_type(self) -> str:
        return MLModel.TYPE_REGRESSION


_MODEL_TRANSFORMERS: Dict[type, Type[ModelTransformer]] = {
    LGBMRegressor: LGBMRegressorTransformer
}
