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

import os

import pandas as pd
from elasticsearch import Elasticsearch

from eland.common import es_version
from eland.dataload import (  # noqa: F401
    ECOMMERCE_INDEX_NAME,
    ECOMMERCE_MAPPING,
    FLIGHTS_INDEX_NAME,
    FLIGHTS_MAPPING,
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define test files and indices
ELASTICSEARCH_HOST = os.environ.get(
    "ELASTICSEARCH_URL", os.environ.get("ELASTICSEARCH_HOST", "http://localhost:9200")
)

# Define client to use in tests
ES_TEST_CLIENT = Elasticsearch(ELASTICSEARCH_HOST)

ES_VERSION = es_version(ES_TEST_CLIENT)

FLIGHTS_FILE_NAME = ROOT_DIR + "/flights.json.gz"
FLIGHTS_DF_FILE_NAME = ROOT_DIR + "/flights_df.json.gz"

FLIGHTS_SMALL_INDEX_NAME = "flights_small"
FLIGHTS_SMALL_MAPPING = FLIGHTS_MAPPING
FLIGHTS_SMALL_FILE_NAME = ROOT_DIR + "/flights_small.json.gz"

ECOMMERCE_FILE_NAME = ROOT_DIR + "/ecommerce.json.gz"
ECOMMERCE_DF_FILE_NAME = ROOT_DIR + "/ecommerce_df.json.gz"

TEST_MAPPING1 = {
    "mappings": {
        "properties": {
            "city": {"type": "text", "fields": {"raw": {"type": "keyword"}}},
            "text": {
                "type": "text",
                "fields": {"english": {"type": "text", "analyzer": "english"}},
            },
            "origin_location": {
                "properties": {
                    "lat": {
                        "type": "text",
                        "index_prefixes": {},
                        "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                    },
                    "lon": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                    },
                }
            },
            "maps-telemetry": {
                "properties": {
                    "attributesPerMap": {
                        "properties": {
                            "dataSourcesCount": {
                                "properties": {
                                    "avg": {"type": "long"},
                                    "max": {"type": "long"},
                                    "min": {"type": "long"},
                                }
                            },
                            "emsVectorLayersCount": {
                                "dynamic": "true",
                                "properties": {
                                    "france_departments": {
                                        "properties": {
                                            "avg": {"type": "float"},
                                            "max": {"type": "long"},
                                            "min": {"type": "long"},
                                        }
                                    }
                                },
                            },
                        }
                    }
                }
            },
            "type": {"type": "keyword"},
            "name": {"type": "text"},
            "user_name": {"type": "keyword"},
            "email": {"type": "keyword"},
            "content": {"type": "text"},
            "tweeted_at": {"type": "date"},
            "dest_location": {"type": "geo_point"},
            "my_join_field": {
                "type": "join",
                "relations": {"question": ["answer", "comment"], "answer": "vote"},
            },
        }
    }
}

TEST_MAPPING1_INDEX_NAME = "mapping1"

TEST_MAPPING1_EXPECTED = {
    "city": "text",
    "city.raw": "keyword",
    "content": "text",
    "dest_location": "geo_point",
    "email": "keyword",
    "maps-telemetry.attributesPerMap.dataSourcesCount.avg": "long",
    "maps-telemetry.attributesPerMap.dataSourcesCount.max": "long",
    "maps-telemetry.attributesPerMap.dataSourcesCount.min": "long",
    "maps-telemetry.attributesPerMap.emsVectorLayersCount.france_departments.avg": "float",
    "maps-telemetry.attributesPerMap.emsVectorLayersCount.france_departments.max": "long",
    "maps-telemetry.attributesPerMap.emsVectorLayersCount.france_departments.min": "long",
    "my_join_field": "join",
    "name": "text",
    "origin_location.lat": "text",
    "origin_location.lat.keyword": "keyword",
    "origin_location.lon": "text",
    "origin_location.lon.keyword": "keyword",
    "text": "text",
    "text.english": "text",
    "tweeted_at": "date",
    "type": "keyword",
    "user_name": "keyword",
}

TEST_MAPPING1_EXPECTED_DF = pd.DataFrame.from_dict(
    data=TEST_MAPPING1_EXPECTED, orient="index", columns=["es_dtype"]
)
TEST_MAPPING1_EXPECTED_SOURCE_FIELD_DF = TEST_MAPPING1_EXPECTED_DF.drop(
    index=[
        "city.raw",
        "origin_location.lat.keyword",
        "origin_location.lon.keyword",
        "text.english",
    ]
)
TEST_MAPPING1_EXPECTED_SOURCE_FIELD_COUNT = len(
    TEST_MAPPING1_EXPECTED_SOURCE_FIELD_DF.index
)

TEST_NESTED_USER_GROUP_INDEX_NAME = "nested_user_group"
TEST_NESTED_USER_GROUP_MAPPING = {
    "mappings": {
        "properties": {
            "group": {"type": "keyword"},
            "user": {
                "properties": {
                    "first": {"type": "keyword"},
                    "last": {"type": "keyword"},
                    "address": {"type": "keyword"},
                }
            },
        }
    }
}

TEST_NESTED_USER_GROUP_DOCS = [
    {
        "_index": TEST_NESTED_USER_GROUP_INDEX_NAME,
        "_source": {
            "group": "amsterdam",
            "user": [
                {
                    "first": "Manke",
                    "last": "Nelis",
                    "address": ["Elandsgracht", "Amsterdam"],
                },
                {
                    "first": "Johnny",
                    "last": "Jordaan",
                    "address": ["Elandsstraat", "Amsterdam"],
                },
            ],
        },
    },
    {
        "_index": TEST_NESTED_USER_GROUP_INDEX_NAME,
        "_source": {
            "group": "london",
            "user": [
                {"first": "Alice", "last": "Monkton"},
                {"first": "Jimmy", "last": "White", "address": ["London"]},
            ],
        },
    },
    {
        "_index": TEST_NESTED_USER_GROUP_INDEX_NAME,
        "_source": {"group": "new york", "user": [{"first": "Bill", "last": "Jones"}]},
    },
]
