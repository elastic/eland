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

from collections.abc import Mapping
from functools import cached_property
from typing import Any

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

    def to_dict(self) -> dict[str, Any]:
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
        default_score: float | None = None,
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

    def __init__(self, feature_extractors: list[FeatureExtractor]):
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
    def feature_names(self) -> list[str]:
        """
        List of the feature names for the model.
        """

        return [extractor.feature_name for extractor in self.feature_extractors]

    @cached_property
    def query_feature_extractors(self) -> list[QueryFeatureExtractor]:
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

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "LTRModelConfig":
        """
        Create an LTRModelConfig from a dict.

        Parameters
        ----------
        d: Mapping[str, Any]
            Dict representing the LTR model config.

        Examples
        --------
        >>> from eland.ml.ltr import LTRModelConfig

        >>> ltr_model_config_dict = {
        ...     "learning_to_rank": {
        ...         "feature_extractors": [
        ...             {
        ...                 "query_extractor": {
        ...                     "feature_name": "title_bm25",
        ...                     "query": { "match": { "title": "{{query}}" } }
        ...                 }
        ...             },
        ...             {
        ...                 "query_extractor": {
        ...                     "feature_name": "description_bm25",
        ...                     "query": { "match": { "description": "{{query}}" } }
        ...                 }
        ...             }
        ...         ]
        ...     }
        ... }

        >>> ltr_model_config = LTRModelConfig.from_dict(ltr_model_config_dict)
        """
        if TYPE_LEARNING_TO_RANK not in d:
            raise ValueError(
                f"Invalid LTR model config, missing '{TYPE_LEARNING_TO_RANK}' key"
            )

        feature_extractors = []
        for feature_extractor in d[TYPE_LEARNING_TO_RANK]["feature_extractors"]:
            if "query_extractor" in feature_extractor:
                fe = feature_extractor["query_extractor"]
                feature_extractors.append(
                    QueryFeatureExtractor(
                        feature_name=fe["feature_name"],
                        query=fe["query"],
                        default_score=fe.get("default_score"),
                    )
                )
            else:
                raise ValueError(
                    f"Unknown feature extractor type: {list(feature_extractor.keys())}"
                )

        return cls(feature_extractors=feature_extractors)
