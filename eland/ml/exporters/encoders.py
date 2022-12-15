import sklearn
from sklearn.preprocessing import FunctionTransformer
import numpy as np


class TargetMeanEncoder(FunctionTransformer):
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        target_map = self.preprocessor["target_mean_encoding"]["target_map"]
        feature_name_out = self.preprocessor["target_mean_encoding"]["feature_name"]
        self.field_name_in = self.preprocessor["target_mean_encoding"]["field"]
        fallback_value = self.preprocessor["target_mean_encoding"]["default_value"]
        func = lambda column: np.array(
            [
                target_map[str(category)] if category in target_map else fallback_value
                for category in column
            ]
        ).reshape(-1, 1)
        feature_names_out = lambda ft, carr: [
            feature_name_out if c == self.field_name_in else c for c in carr
        ]
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
                frequency_map[str(category)] if category in frequency_map else fallback_value
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
        self.field_name_in = self.preprocessor['one_hot_encoding']['field']
        self.cats = [list(self.preprocessor['one_hot_encoding']['hot_map'].keys())]
        super().__init__(categories=self.cats, handle_unknown='ignore')