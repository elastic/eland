import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch import helpers

from eland.tests import *

DATA_LIST = [
    (FLIGHTS_FILE_NAME, FLIGHTS_INDEX_NAME),
    (ECOMMERCE_FILE_NAME, ECOMMERCE_INDEX_NAME)
]

def _setup_data(es):
    # Read json file and index records into Elasticsearch
    for data in DATA_LIST:
        json_file_name = data[0]
        index_name = data[1]

        # Delete index
        print("Deleting index:", index_name)
        es.indices.delete(index=index_name, ignore=[400, 404])

        df = pd.read_json(json_file_name, lines=True)

        actions = []
        n = 0

        print("Adding", df.shape[0], "items to index:", index_name)
        for index, row in df.iterrows():
            values = row.to_dict()
            # make timestamp datetime 2018-01-01T12:09:35
            #values['timestamp'] = datetime.strptime(values['timestamp'], '%Y-%m-%dT%H:%M:%S')

            action = {'_index': index_name, '_source': values}

            actions.append(action)

            n = n + 1

            if n % 10000 == 0:
                helpers.bulk(es, actions)
                actions = []

        helpers.bulk(es, actions)
        actions = []

        print("Done", index_name)

def _setup_test_mappings(es):
    # Create a complex mapping containing many Elasticsearch features
    es.indices.delete(index=TEST_MAPPING1_INDEX_NAME, ignore=[400, 404])
    es.indices.create(index=TEST_MAPPING1_INDEX_NAME, body=TEST_MAPPING1)

if __name__ == '__main__':
    # Create connection to Elasticsearch - use defaults
    es = Elasticsearch(ELASTICSEARCH_HOST)

    _setup_data(es)
    _setup_test_mappings(es)
