from eland import Client
from eland import DataFrame
from eland import Mappings


def read_es(es_params, index_pattern):
    return DataFrame(client=es_params, index_pattern=index_pattern)


def pandas_to_es(df, es_params, destination_index, if_exists='fail', chunk_size=10000, refresh=False):
    """
    Append a pandas DataFrame to an Elasticsearch index.
    Mainly used in testing.

    Parameters
    ----------
    es_params : Elasticsearch client argument
        elasticsearch-py parameters or
        elasticsearch-py instance or
        eland.Client instance

    destination_index : str
        Name of Elasticsearch index to be written

    if_exists : str, default 'fail'
        Behavior when the destination index exists. Value can be one of:
        ``'fail'``
            If table exists, do nothing.
        ``'replace'``
            If table exists, drop it, recreate it, and insert data.
        ``'append'``
                If table exists, insert data. Create if does not exist.
    """
    client = Client(es_params)

    mapping = Mappings._generate_es_mappings(df)

    # If table exists, check if_exists parameter
    if client.index_exists(index=destination_index):
        if if_exists == "fail":
            raise ValueError(
                "Could not create the index [{0}] because it "
                "already exists. "
                "Change the if_exists parameter to "
                "'append' or 'replace' data.".format(destination_index)
            )
        elif if_exists == "replace":
            client.index_delete(index=destination_index)
            client.index_create(index=destination_index, mapping=mapping)
        # elif if_exists == "append":
        # TODO validate mapping is compatible
    else:
        client.index_create(index=destination_index, mapping=mapping)

    # Now add data
    actions = []
    n = 0
    for row in df.iterrows():
        # Use index as _id
        id = row[0]
        values = row[1].to_dict()

        # Use integer as id field for repeatable results
        action = {'_index': destination_index, '_source': values, '_id': str(id)}

        actions.append(action)

        n = n + 1

        if n % chunk_size == 0:
            client.bulk(actions, refresh=refresh)
            actions = []

    client.bulk(actions, refresh=refresh)
