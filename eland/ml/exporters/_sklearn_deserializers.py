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

import sklearn
from sklearn.preprocessing import FunctionTransformer
import numpy as np


class Tree:
    def __init__(self, json_tree, feature_names_map):
        TREE_LEAF = -1

        node_count = len(json_tree["tree_structure"])
        children_left = np.ones((node_count,), dtype=int) * TREE_LEAF
        children_right = np.ones((node_count,), dtype=int) * TREE_LEAF
        feature = np.ones((node_count,), dtype=int) * -2
        threshold = np.ones((node_count,), dtype=float) * -2
        impurity = np.zeros((node_count,), dtype=float)
        # value works only for regression and binary classification
        value = np.zeros((node_count, 1, 1), dtype="<f8")
        n_node_samples = np.zeros((node_count,), dtype=int)

        # parse values from the JSON tree
        feature_names = json_tree["feature_names"]
        for json_node in json_tree["tree_structure"]:
            node_id = json_node["node_index"]
            n_node_samples[node_id] = json_node["number_samples"]

            if "leaf_value" not in json_node:
                children_left[node_id] = json_node["left_child"]
                children_right[node_id] = json_node["right_child"]
                feature[node_id] = feature_names_map[
                    feature_names[json_node["split_feature"]]
                ]
                threshold[node_id] = json_node["threshold"]
                if "split_gain" in json_node:
                    impurity[node_id] = json_node["split_gain"]
                else:
                    impurity[node_id] = -1
            else:
                value[node_id, 0, 0] = json_node["leaf_value"]

        # iterate through tree to get max depth and expected values
        weighted_n_node_samples = n_node_samples.copy()
        self.max_depth = Tree._compute_expectations(
            children_left=children_left,
            children_right=children_right,
            node_sample_weight=weighted_n_node_samples,
            values=value,
            node_index=0,
        )

        # initialize the sklearn tree
        self.tree_ = sklearn.tree._tree.Tree(
            len(feature_names), np.array([1], dtype=int), 1
        )
        node_state = np.array(
            [
                (
                    children_left[i],
                    children_right[i],
                    feature[i],
                    threshold[i],
                    impurity[i],
                    n_node_samples[i],
                    weighted_n_node_samples[i],
                )
                for i in range(node_count)
            ],
            dtype=[
                ("left_child", "<i8"),
                ("right_child", "<i8"),
                ("feature", "<i8"),
                ("threshold", "<f8"),
                ("impurity", "<f8"),
                ("n_node_samples", "<i8"),
                ("weighted_n_node_samples", "<f8"),
            ],
        )
        state = {
            "max_depth": self.max_depth,
            "node_count": node_count,
            "nodes": node_state,
            "values": value,
        }
        self.tree_.__setstate__(state)

    @staticmethod
    def _compute_expectations(
        children_left, children_right, node_sample_weight, values, node_index, depth=0
    ) -> int:
        if children_right[node_index] == -1:
            return 0

        left_index = children_left[node_index]
        right_index = children_right[node_index]
        depth_left = Tree._compute_expectations(
            children_left,
            children_right,
            node_sample_weight,
            values,
            left_index,
            depth + 1,
        )
        depth_right = Tree._compute_expectations(
            children_left,
            children_right,
            node_sample_weight,
            values,
            right_index,
            depth + 1,
        )
        left_weight = node_sample_weight[left_index]
        right_weight = node_sample_weight[right_index]

        v = (
            left_weight * values[left_index, :] + right_weight * values[right_index, :]
        ) / (left_weight + right_weight)
        values[node_index, :] = v
        return max(depth_left, depth_right) + 1


class TargetMeanEncoder(FunctionTransformer):
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        target_map = self.preprocessor["target_mean_encoding"]["target_map"]
        feature_name_out = self.preprocessor["target_mean_encoding"]["feature_name"]
        self.field_name_in = self.preprocessor["target_mean_encoding"]["field"]
        fallback_value = self.preprocessor["target_mean_encoding"]["default_value"]

        def func(column):
            return np.array(
                [
                    target_map[str(category)]
                    if category in target_map
                    else fallback_value
                    for category in column
                ]
            ).reshape(-1, 1)

        def feature_names_out(ft, carr):
            return [feature_name_out if c == self.field_name_in else c for c in carr]

        super().__init__(func=func, feature_names_out=feature_names_out)


class FrequencyEncoder(FunctionTransformer):
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        frequency_map = self.preprocessor["frequency_encoding"]["frequency_map"]
        feature_name_out = self.preprocessor["frequency_encoding"]["feature_name"]
        self.field_name_in = self.preprocessor["frequency_encoding"]["field"]
        fallback_value = 0.0
        func = lambda column: np.array(
            [
                frequency_map[str(category)]
                if category in frequency_map
                else fallback_value
                for category in column
            ]
        ).reshape(-1, 1)
        feature_names_out = lambda ft, carr: [
            feature_name_out if c == self.field_name_in else c for c in carr
        ]
        super().__init__(func=func, feature_names_out=feature_names_out)


class OneHotEncoder(sklearn.preprocessing.OneHotEncoder):
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.field_name_in = self.preprocessor["one_hot_encoding"]["field"]
        self.cats = [list(self.preprocessor["one_hot_encoding"]["hot_map"].keys())]
        super().__init__(categories=self.cats, handle_unknown="ignore")