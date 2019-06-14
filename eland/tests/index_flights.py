import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch import helpers

from eland.tests import FLIGHTS_FILE_NAME, FLIGHTS_INDEX_NAME, ECOMMERCE_FILE_NAME, ECOMMERCE_INDEX_NAME


DATA_LIST = [
    (FLIGHTS_FILE_NAME, FLIGHTS_INDEX_NAME),
    (ECOMMERCE_FILE_NAME, ECOMMERCE_INDEX_NAME)
]

if __name__ == '__main__':

    # Read json file and index records into Elasticsearch
    for data in DATA_LIST:
        json_file_name = data[0]
        index_name = data[1]

        # Create connection to Elasticsearch - use defaults1
        es = Elasticsearch()

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
