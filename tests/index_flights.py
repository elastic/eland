import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch import helpers

if __name__ == '__main__':

    # Read json file and index records into Elasticsearch
    json_file_name = 'flights.json.gz'
    index_name = 'flights'

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
        action = {"_index": index_name, '_source': row.to_dict()}

        actions.append(action)

        n = n + 1

        if n % 10000 == 0:
            helpers.bulk(es, actions)
            actions = []

    helpers.bulk(es, actions)
    actions = []

    print("Done", es.cat.indices(index_name))
