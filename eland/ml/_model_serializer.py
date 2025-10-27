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

import base64
import gzip
import json
from abc import ABC
from collections.abc import Sequence
from typing import Any


def add_if_exists(d: dict[str, Any], k: str, v: Any) -> None:
    if v is not None:
        d[k] = v


class ModelSerializer(ABC):
    def __init__(
        self,
        feature_names: Sequence[str],
        target_type: str | None = None,
        classification_labels: Sequence[str] | None = None,
    ):
        self._target_type = target_type
        self._feature_names = feature_names
        self._classification_labels = classification_labels

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        add_if_exists(d, "target_type", self._target_type)
        add_if_exists(d, "feature_names", self._feature_names)
        add_if_exists(d, "classification_labels", self._classification_labels)
        return d

    @property
    def feature_names(self) -> Sequence[str]:
        return self._feature_names

    def serialize_model(self) -> dict[str, Any]:
        return {"trained_model": self.to_dict()}

    def serialize_and_compress_model(self) -> str:
        json_string = json.dumps(self.serialize_model(), separators=(",", ":"))
        return base64.b64encode(gzip.compress(json_string.encode("utf-8"))).decode(
            "ascii"
        )

    def bounds(self) -> tuple[float, float]:
        raise NotImplementedError


class TreeNode:
    def __init__(
        self,
        node_idx: int,
        default_left: bool | None = None,
        decision_type: str | None = None,
        left_child: int | None = None,
        right_child: int | None = None,
        split_feature: int | None = None,
        threshold: float | None = None,
        leaf_value: list[float] | None = None,
        number_samples: int | None = None,
    ):
        self._node_idx = node_idx
        self._decision_type = decision_type
        self._left_child = left_child
        self._right_child = right_child
        self._split_feature = split_feature
        self._threshold = threshold
        self._leaf_value = leaf_value
        self._default_left = default_left
        self._number_samples = number_samples

    @property
    def node_idx(self) -> int:
        return self._node_idx

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        add_if_exists(d, "node_index", self._node_idx)
        add_if_exists(d, "decision_type", self._decision_type)
        if self._leaf_value is None:
            add_if_exists(d, "left_child", self._left_child)
            add_if_exists(d, "right_child", self._right_child)
            add_if_exists(d, "split_feature", self._split_feature)
            add_if_exists(d, "threshold", self._threshold)
            add_if_exists(d, "number_samples", self._number_samples)
            add_if_exists(d, "default_left", self._default_left)
        else:
            if len(self._leaf_value) == 1:
                # Support Elasticsearch 7.6 which only
                # singular leaf_values not in arrays
                add_if_exists(d, "leaf_value", self._leaf_value[0])
            else:
                add_if_exists(d, "leaf_value", self._leaf_value)
        return d


class Tree(ModelSerializer):
    def __init__(
        self,
        feature_names: Sequence[str],
        target_type: str | None = None,
        tree_structure: Sequence[TreeNode] | None = None,
        classification_labels: Sequence[str] | None = None,
    ):
        super().__init__(
            feature_names=feature_names,
            target_type=target_type,
            classification_labels=classification_labels,
        )
        if target_type == "regression" and classification_labels:
            raise ValueError("regression does not support classification_labels")
        self._tree_structure = tree_structure or []

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        add_if_exists(d, "tree_structure", [t.to_dict() for t in self._tree_structure])
        return {"tree": d}

    def bounds(self) -> tuple[float, float]:
        leaf_values = [
            tree_node._leaf_value[0]
            for tree_node in self._tree_structure
            if tree_node._leaf_value is not None
        ]
        return min(leaf_values), max(leaf_values)


class Ensemble(ModelSerializer):
    def __init__(
        self,
        feature_names: Sequence[str],
        trained_models: Sequence[ModelSerializer],
        output_aggregator: dict[str, Any],
        target_type: str | None = None,
        classification_labels: Sequence[str] | None = None,
        classification_weights: Sequence[float] | None = None,
    ):
        super().__init__(
            feature_names=feature_names,
            target_type=target_type,
            classification_labels=classification_labels,
        )
        self._trained_models = trained_models
        self._classification_weights = classification_weights
        self._output_aggregator = output_aggregator

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        trained_models = None
        if self._trained_models:
            trained_models = [t.to_dict() for t in self._trained_models]
        add_if_exists(d, "trained_models", trained_models)
        add_if_exists(d, "classification_weights", self._classification_weights)
        add_if_exists(d, "aggregate_output", self._output_aggregator)
        return {"ensemble": d}

    def bounds(self) -> tuple[float, float]:
        min_bound, max_bound = tuple(
            map(sum, zip(*[model.bounds() for model in self._trained_models]))
        )
        return min_bound, max_bound
