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

from eland.ml.ltr import FeatureLogger, LTRModelConfig, QueryFeatureExtractor
from tests import ES_TEST_CLIENT, NATIONAL_PARKS_INDEX_NAME


class TestFeatureLogger:
    def test_extract_feature(self):
        # Create the feature logger and some document extract features for a query.
        ltr_model_config = self._ltr_model_config()
        feature_logger = FeatureLogger(
            ES_TEST_CLIENT, NATIONAL_PARKS_INDEX_NAME, ltr_model_config
        )

        doc_ids = ["park_yosemite", "park_hawaii-volcanoes", "park_death-valley"]

        doc_features = feature_logger.extract_features(
            query_params={"query": "yosemite"}, doc_ids=doc_ids
        )

        # Assert all docs are presents.
        assert len(doc_features) == len(doc_ids) and all(
            doc_id in doc_ids for doc_id in doc_features.keys()
        )

        # Check all features are extracted for all docs
        assert all(
            len(features) == len(ltr_model_config.feature_extractors)
            for features in doc_features.values()
        )
        print(doc_features)

        # "park_yosemite" document matches for title and is a world heritage site
        assert (
            doc_features["park_yosemite"][0] > 0
            and doc_features["park_yosemite"][1] > 1
        )

        # "park_hawaii-volcanoes" document does not matches for title but is a world heritage site
        assert (
            doc_features["park_hawaii-volcanoes"][0] == 0
            and doc_features["park_hawaii-volcanoes"][1] > 1
        )

        # "park_hawaii-volcanoes" document does not matches for title and is not a world heritage site
        assert doc_features["park_death-valley"] == [0, 0]

    def _ltr_model_config(self):
        # Returns an LTR config with 2 query feature extractors:
        # - title_bm25: BM25 score of the match query on the title field
        # - popularity: Value of the popularity field
        return LTRModelConfig(
            [
                QueryFeatureExtractor(
                    feature_name="title_bm25", query={"match": {"title": "{{query}}"}}
                ),
                QueryFeatureExtractor(
                    feature_name="world_heritage_site",
                    query={"term": {"world_heritage_site": True}},
                ),
            ]
        )
