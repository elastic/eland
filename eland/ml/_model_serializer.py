import base64
import gzip
import json
from abc import ABC
from typing import List


def add_if_exists(d: dict, k: str, v) -> dict:
    if v is not None:
        d[k] = v
    return d

class ModelSerializer(ABC):

    def __init__(self,
                 feature_names: List[str],
                 target_type: str = None,
                 classification_labels: List[str] = None):
        self._target_type = target_type
        self._feature_names = feature_names
        self._classification_labels = classification_labels

    def to_dict(self):
        d = dict()
        add_if_exists(d, "target_type", self._target_type)
        add_if_exists(d, "feature_names", self._feature_names)
        add_if_exists(d, "classification_labels", self._classification_labels)
        return d

    @property
    def feature_names(self):
        return self._feature_names

    def serialize_and_compress_model(self) -> str:
        json_string = json.dumps({'trained_model': self.to_dict()})
        return base64.b64encode(gzip.compress(bytes(json_string, 'utf-8')))


class TreeNode:
    def __init__(self,
                 node_idx: int,
                 default_left: bool = None,
                 decision_type: str = None,
                 left_child: int = None,
                 right_child: int = None,
                 split_feature: int = None,
                 threshold: float = None,
                 leaf_value: float = None):
        self._node_idx = node_idx
        self._decision_type = decision_type
        self._left_child = left_child
        self._right_child = right_child
        self._split_feature = split_feature
        self._threshold = threshold
        self._leaf_value = leaf_value
        self._default_left = default_left

    def to_dict(self):
        d = dict()
        add_if_exists(d, 'node_index', self._node_idx)
        add_if_exists(d, 'decision_type', self._decision_type)
        if self._leaf_value is None:
            add_if_exists(d, 'left_child', self._left_child)
            add_if_exists(d, 'right_child', self._right_child)
            add_if_exists(d, 'split_feature', self._split_feature)
            add_if_exists(d, 'threshold', self._threshold)
        else:
            add_if_exists(d, 'leaf_value', self._leaf_value)
        return d


class Tree(ModelSerializer):
    def __init__(self,
                 feature_names: List[str],
                 target_type: str = None,
                 tree_structure: List[TreeNode] = [],
                 classification_labels: List[str] = None):
        super().__init__(
            feature_names=feature_names,
            target_type=target_type,
            classification_labels=classification_labels
        )
        if target_type == 'regression' and classification_labels:
            raise ValueError("regression does not support classification_labels")
        self._tree_structure = tree_structure

    def to_dict(self):
        d = super().to_dict()
        add_if_exists(d, 'tree_structure', [t.to_dict() for t in self._tree_structure])
        return {'tree': d}


class Ensemble(ModelSerializer):
    def __init__(self,
                 feature_names: List[str],
                 trained_models: List[ModelSerializer],
                 output_aggregator: dict,
                 target_type: str = None,
                 classification_labels: List[str] = None,
                 classification_weights: List[float] = None):
        super().__init__(feature_names=feature_names,
                         target_type=target_type,
                         classification_labels=classification_labels)
        self._trained_models = trained_models
        self._classification_weights = classification_weights
        self._output_aggregator = output_aggregator

    def to_dict(self):
        d = super().to_dict()
        trained_models = None
        if self._trained_models:
            trained_models = [t.to_dict() for t in self._trained_models]
        add_if_exists(d, 'trained_models', trained_models)
        add_if_exists(d, 'classification_weights', self._classification_weights)
        add_if_exists(d, 'aggregate_output', self._output_aggregator)
        return {'ensemble': d}
