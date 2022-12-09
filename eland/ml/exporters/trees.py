#
# Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
# or more contributor license agreements. Licensed under the Elastic License;
# you may not use this file except in compliance with the Elastic License.
#
import numpy as np
from math import isclose
import sklearn


class TreeNode:
    def __init__(self, json_node, feature_names):
        self.id = json_node["node_index"]
        self.number_samples = json_node["number_samples"]
        self.is_leaf = False
        self.left_child_id = json_node["left_child"]
        self.right_child_id = json_node["right_child"]
        self.threshold = json_node["threshold"]
        self.split_feature_name = feature_names[json_node["split_feature"]]
        self.split_feature = json_node["split_feature"]
        if 'split_gain' in json_node:
            self.split_gain = json_node["split_gain"]
        else:
            self.split_gain = -1 # TODO (valeriy42) Check if this value makes sense.

    def label(self):
        label = f"id={self.id}\nsamples={self.number_samples}\n{self.split_feature_name}<{self.threshold:.4f}"
        return label

    def __eq__(self, value):
        return (
            self.id == value.id
            and self.is_leaf == value.is_leaf
            and self.left_child_id == value.left_child_id
            and self.right_child_id == value.right_child_id
            and isclose(self.threshold, value.threshold, abs_tol=1e-6)
            and self.split_feature_name == value.split_feature_name
        )

    def __ne__(self, value):
        return not self == value

    def __str__(self):
        return (
            f"Node {self.id}: is_leaf {self.is_leaf}, left_child_id {self.left_child_id}, right_child_id {self.right_child_id}\n"
            f"threshold {self.threshold}, split_feature {self.split_feature_name}, split_gain {self.split_gain}"
        )


class Leaf:
    def __init__(self, json_node):
        self.id = json_node["node_index"]
        self.number_samples = json_node["number_samples"]
        self.value = json_node["leaf_value"]
        self.is_leaf = True

    def label(self):
        label = f"id={self.id}\nsamples={self.number_samples}\nvalue={self.value:.4f}"
        return label

    def __eq__(self, value):
        return (
            self.id == value.id
            and isclose(self.value, value.value, abs_tol=1e-6)
            and self.is_leaf == value.is_leaf
        )

    def __ne__(self, value):
        return not self == value

    def __str__(self):
        return (
            f"Leaf {self.id}: number_samples {self.number_samples}, value {self.value}"
        )


class Tree:
    def __init__(self, json_tree):
        self.feature_names = json_tree["feature_names"]
        self.nodes = {}
        self.number_nodes = 0
        self.number_leaves = 0

        for json_node in json_tree["tree_structure"]:
            if "leaf_value" in json_node:
                leaf = Leaf(json_node)
                self.nodes[leaf.id] = leaf
                self.number_leaves += 1
            else:
                node = TreeNode(json_node, self.feature_names)
                self.nodes[node.id] = node
                self.number_nodes += 1
        self.traverse(self.nodes[0], 0)

    def traverse(self, node, height=0):
        node.height = height
        if not node.is_leaf:
            self.traverse(self.nodes[node.left_child_id], height + 1)
            self.traverse(self.nodes[node.right_child_id], height + 1)

    def max_index(self):
        return max(map(lambda x: x.id, self.nodes.values()))

    # def min_gain(self):
    #     if self.number_nodes > 0:
    #         return min(map(lambda x: x.split_gain, self.nodes.values()))
    #     return 0

    # def max_gain(self):
    #     if self.number_nodes > 0:
    #         return max(map(lambda x: x.split_gain, self.nodes.values()))
    #     return 0

    # def total_gain(self):
    #     if self.number_nodes > 0:
    #         return sum(map(lambda x: x.split_gain, self.nodes.values()))
    #     return 0

    def size(self):
        return len(self.nodes)

    def depth(self):
        return max(map(lambda x: x.height, self.nodes.values())) + 1

    def get_leaf_path(self, x):
        path = []
        node = self.nodes[0]
        path.append(node.id)
        while not node.is_leaf:
            if x[node.split_feature] < node.threshold:
                node = self.nodes[node.left_child_id]
            else:
                node = self.nodes[node.right_child_id]
            path.append(node.id)
        return path

    def get_leaf_id(self, x):
        return self.get_leaf_path(x)[-1]

    def get_tree_distance(self, x1, x2):
        path1 = self.get_leaf_path(x1)
        path2 = self.get_leaf_path(x2)
        return len(np.setdiff1d(path1, path2)) + len(np.setdiff1d(path2, path1))

    def get_leaves_num_samples(self) -> dict:
        num_samples = {}  # np.zeros(self.number_leaves)
        for _, node in self.nodes.items():
            if node.is_leaf:
                num_samples[node.id] = node.number_samples
        return num_samples

    def __eq__(self, other):
        if (
            self.number_leaves != other.number_leaves
            or self.number_nodes != other.number_nodes
        ):
            return False
        for i, node in enumerate(self.nodes):
            if node != other.nodes[i]:
                return False
        return True

    def num_diff_nodes(self, other):
        diff_nodes = 0.0
        for i, node in self.nodes.items():
            if (
                i not in other.nodes.keys()
                or node.is_leaf != other.nodes[i].is_leaf
                or node != other.nodes[i]
            ):
                diff_nodes += 1.0
        return diff_nodes

    @staticmethod
    def compute_expectations(
        children_left, children_right, node_sample_weight, values, node_index, depth=0
    ):
        if children_right[node_index] == -1:
            return 0

        left_index = children_left[node_index]
        right_index = children_right[node_index]
        depth_left = Tree.compute_expectations(
            children_left,
            children_right,
            node_sample_weight,
            values,
            left_index,
            depth + 1,
        )
        depth_right = Tree.compute_expectations(
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

    def to_sklearn(self):
        """Convert to sklearn.tree._tree.Tree instance."""

        # this is only for regression tree
        node_count = len(self.nodes)
        TREE_LEAF = -1
        children_left = np.ones((node_count,), dtype=int) * TREE_LEAF
        children_right = np.ones((node_count,), dtype=int) * TREE_LEAF
        feature = np.ones((node_count,), dtype=int) * -2
        threshold = np.ones((node_count,), dtype=float) * -2
        impurity = np.zeros((node_count,), dtype=float)
        # value works only for regression
        value = np.zeros((node_count, 1, 1), dtype="<f8")
        n_node_samples = np.zeros((node_count,), dtype=int)
        # weighted_n_node_samples = np.zeros((node_count,), dtype=int)
        for node_id, node in self.nodes.items():
            n_node_samples[node_id] = node.number_samples

            if node.is_leaf is False:
                children_left[node_id] = node.left_child_id
                children_right[node_id] = node.right_child_id
                feature[node_id] = node.split_feature
                threshold[node_id] = node.threshold
                impurity[node_id] = node.split_gain
            else:
                value[node_id, 0, 0] = node.value
        weighted_n_node_samples = n_node_samples.copy()
        max_depth = Tree.compute_expectations(
            children_left=children_left,
            children_right=children_right,
            node_sample_weight=weighted_n_node_samples,
            values=value,
            node_index=0,
        )
        tree_ = sklearn.tree._tree.Tree(
            len(self.feature_names), np.array([1], dtype=int), 1
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
            "max_depth": max_depth,
            "node_count": node_count,
            "nodes": node_state,
            "values": value,
        }
        tree_.__setstate__(state)
        return tree_


class Forest:
    def __init__(self, trained_models):
        self.trees = []
        for trained_model in trained_models:
            self.trees.append(Tree(trained_model["tree"]))

    def __len__(self):
        return len(self.trees)

    def num_trees(self):
        return len(self.trees)

    def max_tree_length(self):
        return max(map(lambda x: x.number_nodes, self.trees))

    def min_gain(self):
        return min(map(lambda x: x.min_gain(), self.trees))

    def max_gain(self):
        return max(map(lambda x: x.max_gain(), self.trees))

    def max_depth(self) -> int:
        return max(map(lambda x: x.depth(), self.trees))

    def total_gain(self):
        return [x.total_gain() for x in self.trees]

    def tree_sizes(self):
        return [x.size() for x in self.trees]

    def tree_depths(self):
        return [x.depth() for x in self.trees]

    def show_tree(self, tree_index=0):
        if tree_index >= len(self):
            return None
        return self.trees[tree_index].dot
