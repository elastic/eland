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

from typing import Any, Dict, Mapping, List

TYPE_LEARNING_TO_RANK = "learning_to_rank"

class FeatureExtractor:
    def __init__(self, type: str, feature_name: str):
        self.feature_name = feature_name
        self.type = type

    def to_dict(self) -> Dict[str, Any]:
        return {
            self.type: {
                k: v.to_dict() if hasattr(v, "to_dict") else v
                for k, v in self.__dict__.items()
                if v is not None and k != "type"
            }
        }

class QueryFeatureExtractor(FeatureExtractor):
    def __init__(self, feature_name: str, query: Mapping[str, Any]):
        super().__init__(feature_name=feature_name, type="query_extractor")
        self.query = query

class LTRModelConfig:
    def __init__(self, feature_extractors: List[FeatureExtractor]):
        self.feature_extractors = feature_extractors

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "feature_extractors": [
                feature_extractor.to_dict()
                for feature_extractor in self.feature_extractors
            ]
        }

    def feature_names(self) -> List[str]:
        return [extractor.feature_name for extractor in self.feature_extractors]
