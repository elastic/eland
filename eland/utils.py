from eland import Client
from eland import DataFrame
from eland import Mappings

import pandas as pd


def read_es(es_params, index_pattern):
    """
    Utility method to create an eland.Dataframe from an Elasticsearch index_pattern.
    (Similar to pandas.read_csv, but source data is an Elasticsearch index rather than
    a csv file)

    Parameters
    ----------
    es_params: Elasticsearch client argument(s)
        - elasticsearch-py parameters or
        - elasticsearch-py instance or
        - eland.Client instance
    index_pattern: str
        Elasticsearch index pattern

    Returns
    -------
    eland.DataFrame

    See Also
    --------
    eland.pandas_to_eland: Create an eland.Dataframe from pandas.DataFrame
    eland.eland_to_pandas: Create a pandas.Dataframe from eland.DataFrame
    """
    return DataFrame(client=es_params, index_pattern=index_pattern)

def pandas_to_eland(pd_df, es_params, destination_index, if_exists='fail', chunk_size=10000, refresh=False, dropna=False,
                    geo_points=None):
    """
    Append a pandas DataFrame to an Elasticsearch index.
    Mainly used in testing.
    Modifies the elasticsearch destination index

    Parameters
    ----------
    es_params: Elasticsearch client argument(s)
        - elasticsearch-py parameters or
        - elasticsearch-py instance or
        - eland.Client instance
    destination_index: str
        Name of Elasticsearch index to be appended to
    if_exists : {'fail', 'replace', 'append'}, default 'fail'
        How to behave if the index already exists.

        - fail: Raise a ValueError.
        - replace: Delete the index before inserting new values.
        - append: Insert new values to the existing index. Create if does not exist.
    dropna: bool, default 'False'
        * True: Remove missing values (see pandas.Series.dropna)
        * False: Include missing values - may cause bulk to fail
    geo_points: list, default None
        List of columns to map to geo_point data type

    Returns
    -------
    eland.Dataframe
        eland.DataFrame referencing data in destination_index

    See Also
    --------
    eland.read_es: Create an eland.Dataframe from an Elasticsearch index
    eland.eland_to_pandas: Create a pandas.Dataframe from eland.DataFrame
    """
    client = Client(es_params)

    mapping = Mappings._generate_es_mappings(pd_df, geo_points)

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
            client.index_create(index=destination_index, body=mapping)
        # elif if_exists == "append":
        # TODO validate mapping is compatible
    else:
        client.index_create(index=destination_index, body=mapping)

    # Now add data
    actions = []
    n = 0
    for row in pd_df.iterrows():
        # Use index as _id
        id = row[0]

        if dropna:
            values = row[1].dropna().to_dict()
        else:
            values = row[1].to_dict()

        # Use integer as id field for repeatable results
        action = {'_index': destination_index, '_source': values, '_id': str(id)}

        actions.append(action)

        n = n + 1

        if n % chunk_size == 0:
            client.bulk(actions, refresh=refresh)
            actions = []

    client.bulk(actions, refresh=refresh)

    ed_df = DataFrame(client, destination_index)

    return ed_df

def eland_to_pandas(ed_df):
    """
    Convert an eland.Dataframe to a pandas.DataFrame

    **Note: this loads the entire Elasticsearch index into in core pandas.DataFrame structures. For large
    indices this can create significant load on the Elasticsearch cluster and require signficant memory**

    Parameters
    ----------
    ed_df: eland.DataFrame
        The source eland.Dataframe referencing the Elasticsearch index

    Returns
    -------
    pandas.Dataframe
        pandas.DataFrame contains all rows and columns in eland.DataFrame

    See Also
    --------
    eland.read_es: Create an eland.Dataframe from an Elasticsearch index
    eland.pandas_to_eland: Create an eland.Dataframe from pandas.DataFrame
    """
    return ed_df._to_pandas()

def _inherit_docstrings(parent, excluded=[]):
    """Creates a decorator which overwrites a decorated class' __doc__
    attribute with parent's __doc__ attribute. Also overwrites __doc__ of
    methods and properties defined in the class with the __doc__ of matching
    methods and properties in parent.

    Args:
        parent (object): Class from which the decorated class inherits __doc__.
        excluded (list): List of parent objects from which the class does not
            inherit docstrings.

    Returns:
        function: decorator which replaces the decorated class' documentation
            parent's documentation.
    """

    def decorator(cls):
        if parent not in excluded:
            cls.__doc__ = parent.__doc__
        for attr, obj in cls.__dict__.items():
            parent_obj = getattr(parent, attr, None)
            if parent_obj in excluded or (
                not callable(parent_obj) and not isinstance(parent_obj, property)
            ):
                continue
            if callable(obj):
                obj.__doc__ = parent_obj.__doc__
            elif isinstance(obj, property) and obj.fget is not None:
                p = property(obj.fget, obj.fset, obj.fdel, parent_obj.__doc__)
                setattr(cls, attr, p)
        return cls

    return decorator
