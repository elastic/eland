#  Copyright 2019 Elasticsearch BV
#
#      Licensed under the Apache License, Version 2.0 (the "License");
#      you may not use this file except in compliance with the License.
#      You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#      Unless required by applicable law or agreed to in writing, software
#      distributed under the License is distributed on an "AS IS" BASIS,
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#      See the License for the specific language governing permissions and
#      limitations under the License.

from elasticsearch import helpers
from elasticsearch.client import ClusterClient

from eland.tests import *

DATA_LIST = [
    (FLIGHTS_FILE_NAME, FLIGHTS_INDEX_NAME, FLIGHTS_MAPPING),
    (FLIGHTS_SMALL_FILE_NAME, FLIGHTS_SMALL_INDEX_NAME, FLIGHTS_MAPPING),
    (ECOMMERCE_FILE_NAME, ECOMMERCE_INDEX_NAME, ECOMMERCE_MAPPING)
]


def _setup_data(es):
    # Read json file and index records into Elasticsearch
    for data in DATA_LIST:
        json_file_name = data[0]
        index_name = data[1]
        mapping = data[2]

        # Delete index
        print("Deleting index:", index_name)
        es.indices.delete(index=index_name, ignore=[400, 404])
        print("Creating index:", index_name)
        es.indices.create(index=index_name, body=mapping)

        df = pd.read_json(json_file_name, lines=True)

        actions = []
        n = 0

        print("Adding", df.shape[0], "items to index:", index_name)
        for index, row in df.iterrows():
            values = row.to_dict()
            # make timestamp datetime 2018-01-01T12:09:35
            # values['timestamp'] = datetime.strptime(values['timestamp'], '%Y-%m-%dT%H:%M:%S')

            # Use integer as id field for repeatable results
            action = {'_index': index_name, '_source': values, '_id': str(n)}

            actions.append(action)

            n = n + 1

            if n % 10000 == 0:
                helpers.bulk(es, actions)
                actions = []

        helpers.bulk(es, actions)
        actions = []

        print("Done", index_name)


def _update_max_compilations_limit(es, limit="10000/1m"):
    print('Updating script.max_compilations_rate to ', limit)
    cluster_client = ClusterClient(es)
    body = {
        "transient": {
            "script.max_compilations_rate": limit
        }
    }
    cluster_client.put_settings(body=body)


def _setup_test_mappings(es):
    # Create a complex mapping containing many Elasticsearch features
    es.indices.delete(index=TEST_MAPPING1_INDEX_NAME, ignore=[400, 404])
    es.indices.create(index=TEST_MAPPING1_INDEX_NAME, body=TEST_MAPPING1)


def _setup_test_nested(es):
    es.indices.delete(index=TEST_NESTED_USER_GROUP_INDEX_NAME, ignore=[400, 404])
    es.indices.create(index=TEST_NESTED_USER_GROUP_INDEX_NAME, body=TEST_NESTED_USER_GROUP_MAPPING)

    helpers.bulk(es, TEST_NESTED_USER_GROUP_DOCS)


if __name__ == '__main__':
    # Create connection to Elasticsearch - use defaults
    print('Connecting to ES', ELASTICSEARCH_HOST)
    es = ES_TEST_CLIENT

    _setup_data(es)
    _setup_test_mappings(es)
    _setup_test_nested(es)
    _update_max_compilations_limit(es)
