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

import json
from typing import TYPE_CHECKING, Any, List, Mapping, Tuple, Union

from eland.common import ensure_es_client
from . import FeatureExtractor

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch

class FeatureLogger:
    def __init__(
        self,
        es_client: Union[str, List[str], Tuple[str, ...], "Elasticsearch"],
        es_index: str,
        feature_extractors: List["FeatureExtractor"]
    ):
        self._feature_extractors = feature_extractors
        self._client: Elasticsearch = ensure_es_client(es_client)
        self._index_name = es_index

    def extract_features(
        self, query_params: Mapping[str, Any], doc_ids: List[str]
    ) -> Mapping[str, List[float]]:
        doc_features = dict(
            (doc_id, [float(0)] * len(self._feature_extractors)) for doc_id in doc_ids
        )
        for feature_idx, feature_extractor in enumerate(self._feature_extractors):
            # TODO: we want to replace this with a single call to search and extract scores from named queries.
            doc_scores = self._extract_scores(feature_extractor, query_params, doc_ids)
            for doc_id, score in doc_scores.items():
                doc_features[doc_id][feature_idx] = score

        return doc_features

    def _extract_scores(
        self,
        feature_extractor: FeatureExtractor,
        query_params: Mapping[str, Any],
        doc_ids: List[str],
    ) -> Mapping[str, float]:
        script_source = (
            """{
          "query": {
            "bool": {
              "must": """
            + json.dumps(feature_extractor.query)
            + """,
              "filter": { "terms": {"_id" : {{#toJson}}doc_ids{{/toJson}} } }
            }
          },
          "size": {{size}},
          "_source": false
        }"""
        )

        params = {**query_params, "doc_ids": doc_ids, "size": len(doc_ids)}

        search_response = self._client.search_template(
            index=self._index_name, source=script_source, params=params
        )

        return dict(
            (hit["_id"], hit["_score"]) for hit in search_response["hits"]["hits"]
        )
