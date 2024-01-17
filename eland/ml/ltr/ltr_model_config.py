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

from functools import cached_property
from typing import Any, Dict, List, Mapping, Optional

from eland.ml.common import TYPE_LEARNING_TO_RANK


class FeatureExtractor:
    """
    A base class representing a generic feature extractor.
    """

    def __init__(self, type: str, feature_name: str):
        """
        Parameters
        ----------
        type: str
            Type of the feature extractor.

        feature_name: str
            Name of the extracted features.
        """
        self.feature_name = feature_name
        self.type = type

    def to_dict(self) -> Dict[str, Any]:
        """Convert the feature extractor into a dict that can be send to ES as part of the inference config."""
        return {
            self.type: {
                k: v.to_dict() if hasattr(v, "to_dict") else v
                for k, v in self.__dict__.items()
                if v is not None and k != "type"
            }
        }


class QueryFeatureExtractor(FeatureExtractor):
    """
    A class that allows to define a query feature extractor.
    """

    def __init__(
        self,
        feature_name: str,
        query: Mapping[str, Any],
        default_score: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        feature_name: str
            Name of the extracted features.

        query: Mapping[str, Any]
            Templated query used to extract the feature.

        default_score: str
            Scored used by default when the doc is not matching the query.

        Examples
        --------
        >>> from eland.ml.ltr import QueryFeatureExtractor

        >>> query_feature_extractor = QueryFeatureExtractor(
        ...    feature_name='title_bm25',
        ...    query={ "match": { "title": "{{query}}" } }
        ... )
        """
        super().__init__(feature_name=feature_name, type="query_extractor")
        self.query = query
        self.default_score = default_score


class LTRModelConfig:
    """
    A class representing LTR model configuration.
    """

    def __init__(self, feature_extractors: List[FeatureExtractor]):
        """
        Parameters
        ----------
        feature_extractors: List[FeatureExtractor]
            List of the feature extractors for the LTR model.

        Examples
        --------
        >>> from eland.ml.ltr import LTRModelConfig, QueryFeatureExtractor

        >>> ltr_model_config = LTRModelConfig(
        ...     feature_extractors=[
        ...        QueryFeatureExtractor(
        ...            feature_name='title_bm25',
        ...            query={ "match": { "title": "{{query}}" } }
        ...        ),
        ...        QueryFeatureExtractor(
        ...            feature_name='descritption_bm25',
        ...            query={ "match": { "description": "{{query}}" } }
        ...        )
        ...     ]
        ... )
        """
        self.feature_extractors = feature_extractors

    def to_dict(self) -> Mapping[str, Any]:
        """
        Convert the into a dict that can be send to ES as an inference config.
        """
        return {
            TYPE_LEARNING_TO_RANK: {
                "feature_extractors": [
                    feature_extractor.to_dict()
                    for feature_extractor in self.feature_extractors
                ]
            }
        }

    @cached_property
    def feature_names(self) -> List[str]:
        """
        List of the feature names for the model.
        """

        return [extractor.feature_name for extractor in self.feature_extractors]

    @cached_property
    def query_feature_extractors(self) -> List[QueryFeatureExtractor]:
        """
        List of query feature extractors for the model.
        """
        return [
            extractor
            for extractor in self.feature_extractors
            if isinstance(extractor, QueryFeatureExtractor)
        ]

    def feature_index(self, feature_name: str) -> int:
        "Returns the index of the feature in the feature lists."

        return self.feature_names.index(feature_name)
