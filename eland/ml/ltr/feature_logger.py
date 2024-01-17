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
from functools import cached_property
from typing import TYPE_CHECKING, Any, List, Mapping, Tuple, Union

from eland.common import ensure_es_client
from eland.ml.ltr.ltr_model_config import LTRModelConfig

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch


class FeatureLogger:
    """
    A class that is used during model training to extract features from the judgment list.
    """

    def __init__(
        self,
        es_client: Union[str, List[str], Tuple[str, ...], "Elasticsearch"],
        es_index: str,
        ltr_model_config: LTRModelConfig,
    ):
        """
        Parameters
        ----------
        es_client: Elasticsearch client argument(s)
            - elasticsearch-py parameters or
            - elasticsearch-py instance

        es_index: str
            Name of the Elastcsearch index used for features extractions.

        ltr_model_config: LTRModelConfig
            LTR model config used to extract feature.
        """
        self._model_config = ltr_model_config
        self._client: Elasticsearch = ensure_es_client(es_client)
        self._index_name = es_index

    def extract_features(
        self, query_params: Mapping[str, Any], doc_ids: List[str]
    ) -> Mapping[str, List[float]]:
        """
        Extract document features.

        Parameters
        ----------
        query_params: Mapping[str, Any]
            List of templates params used during features extraction.

        doc_ids: List[str]
            List of doc ids.

        Example
        -------
        >>> from eland.ml.ltr import FeatureLogger, LTRModelConfig, QueryFeatureExtractor

        >>> ltr_model_config=LTRModelConfig(
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

        >>> feature_logger = FeatureLogger(
        ...     es_client='http://localhost:9200',
        ...     es_index='national_parks',
        ...     ltr_model_config=ltr_model_config
        ... )

        >>> doc_features = feature_logger.extract_features(query_params={"query": "yosemite"}, doc_ids=["park-yosemite", "park-everglade"])
        """

        doc_features = {
            doc_id: [float("nan")] * len(self._model_config.feature_extractors)
            for doc_id in doc_ids
        }

        for doc_id, query_features in self._extract_query_features(
            query_params, doc_ids
        ).items():
            for feature_name, feature_value in query_features.items():
                doc_features[doc_id][
                    self._model_config.feature_index(feature_name)
                ] = feature_value

        return doc_features

    def _to_named_query(
        self, query: Mapping[str, Mapping[str, any]], query_name: str
    ) -> Mapping[str, Mapping[str, any]]:
        return {"bool": {"must": query, "_name": query_name}}

    @cached_property
    def _script_source(self) -> str:
        query_extractors = self._model_config.query_feature_extractors
        queries = [
            self._to_named_query(extractor.query, extractor.feature_name)
            for extractor in query_extractors
        ]

        return (
            json.dumps(
                {
                    "query": {
                        "bool": {
                            "should": queries,
                            "filter": {"ids": {"values": "##DOC_IDS_JSON##"}},
                        }
                    },
                    "size": "##DOC_IDS_SIZE##",
                    "_source": False,
                }
            )
            .replace('"##DOC_IDS_JSON##"', "{{#toJson}}__doc_ids{{/toJson}}")
            .replace('"##DOC_IDS_SIZE##"', "{{__size}}")
        )

    def _extract_query_features(
        self, query_params: Mapping[str, Any], doc_ids: List[str]
    ):
        # When support for include_named_queries_score will be added,
        # this will be replaced by the call to the client search_template method.
        from elasticsearch._sync.client import _quote

        __path = f"/{_quote(self._index_name)}/_search/template"
        __query = {"include_named_queries_score": True}
        __headers = {"accept": "application/json", "content-type": "application/json"}
        __body = {
            "source": self._script_source,
            "params": {**query_params, "__doc_ids": doc_ids, "__size": len(doc_ids)},
        }

        return {
            hit["_id"]: hit["matched_queries"] if "matched_queries" in hit else {}
            for hit in self._client.perform_request(
                "GET", __path, params=__query, headers=__headers, body=__body
            )["hits"]["hits"]
        }
