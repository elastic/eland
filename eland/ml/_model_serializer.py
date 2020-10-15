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
from typing import Any, Dict, List, Optional, Sequence


def add_if_exists(d: Dict[str, Any], k: str, v: Any) -> None:
    if v is not None:
        d[k] = v


class ModelSerializer(ABC):
    def __init__(
        self,
        feature_names: Sequence[str],
        target_type: Optional[str] = None,
        classification_labels: Optional[Sequence[str]] = None,
    ):
        self._target_type = target_type
        self._feature_names = feature_names
        self._classification_labels = classification_labels

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        add_if_exists(d, "target_type", self._target_type)
        add_if_exists(d, "feature_names", self._feature_names)
        add_if_exists(d, "classification_labels", self._classification_labels)
        return d

    @property
    def feature_names(self) -> Sequence[str]:
        return self._feature_names

    def serialize_model(self) -> Dict[str, Any]:
        return {"trained_model": self.to_dict()}

    def serialize_and_compress_model(self) -> str:
        json_string = json.dumps(self.serialize_model(), separators=(",", ":"))
        return base64.b64encode(gzip.compress(json_string.encode("utf-8"))).decode(
            "ascii"
        )


class TreeNode:
    def __init__(
        self,
        node_idx: int,
        default_left: Optional[bool] = None,
        decision_type: Optional[str] = None,
        left_child: Optional[int] = None,
        right_child: Optional[int] = None,
        split_feature: Optional[int] = None,
        threshold: Optional[float] = None,
        leaf_value: Optional[List[float]] = None,
    ):
        self._node_idx = node_idx
        self._decision_type = decision_type
        self._left_child = left_child
        self._right_child = right_child
        self._split_feature = split_feature
        self._threshold = threshold
        self._leaf_value = leaf_value
        self._default_left = default_left

    @property
    def node_idx(self) -> int:
        return self._node_idx

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        add_if_exists(d, "node_index", self._node_idx)
        add_if_exists(d, "decision_type", self._decision_type)
        if self._leaf_value is None:
            add_if_exists(d, "left_child", self._left_child)
            add_if_exists(d, "right_child", self._right_child)
            add_if_exists(d, "split_feature", self._split_feature)
            add_if_exists(d, "threshold", self._threshold)
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
        target_type: Optional[str] = None,
        tree_structure: Optional[Sequence[TreeNode]] = None,
        classification_labels: Optional[Sequence[str]] = None,
    ):
        super().__init__(
            feature_names=feature_names,
            target_type=target_type,
            classification_labels=classification_labels,
        )
        if target_type == "regression" and classification_labels:
            raise ValueError("regression does not support classification_labels")
        self._tree_structure = tree_structure or []

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        add_if_exists(d, "tree_structure", [t.to_dict() for t in self._tree_structure])
        return {"tree": d}


class Ensemble(ModelSerializer):
    def __init__(
        self,
        feature_names: Sequence[str],
        trained_models: Sequence[ModelSerializer],
        output_aggregator: Dict[str, Any],
        target_type: Optional[str] = None,
        classification_labels: Optional[Sequence[str]] = None,
        classification_weights: Optional[Sequence[float]] = None,
    ):
        super().__init__(
            feature_names=feature_names,
            target_type=target_type,
            classification_labels=classification_labels,
        )
        self._trained_models = trained_models
        self._classification_weights = classification_weights
        self._output_aggregator = output_aggregator

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        trained_models = None
        if self._trained_models:
            trained_models = [t.to_dict() for t in self._trained_models]
        add_if_exists(d, "trained_models", trained_models)
        add_if_exists(d, "classification_weights", self._classification_weights)
        add_if_exists(d, "aggregate_output", self._output_aggregator)
        return {"ensemble": d}
