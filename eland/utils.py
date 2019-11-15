import pandas as pd
import csv

from pandas.io.parsers import _c_parser_defaults


from eland import Client
from eland import DataFrame
from eland import Mappings

_default_chunk_size = 10000


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


def pandas_to_eland(pd_df, es_params, destination_index, if_exists='fail', chunksize=None,
                    refresh=False,
                    dropna=False,
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
    refresh: bool, default 'False'
        Refresh destination_index after bulk index
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
    if chunksize is None:
        chunksize = _default_chunk_size

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

        if n % chunksize == 0:
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


def read_csv(filepath_or_buffer,
             es_client,
             es_dest_index,
             es_if_exists='fail',
             es_refresh=False,
             es_dropna=False,
             es_geo_points=None,
             sep=",",
             delimiter=None,
             # Column and Index Locations and Names
             header="infer",
             names=None,
             index_col=None,
             usecols=None,
             squeeze=False,
             prefix=None,
             mangle_dupe_cols=True,
             # General Parsing Configuration
             dtype=None,
             engine=None,
             converters=None,
             true_values=None,
             false_values=None,
             skipinitialspace=False,
             skiprows=None,
             skipfooter=0,
             nrows=None,
             # Iteration
             # iterator=False,
             chunksize=None,
             # NA and Missing Data Handling
             na_values=None,
             keep_default_na=True,
             na_filter=True,
             verbose=False,
             skip_blank_lines=True,
             # Datetime Handling
             parse_dates=False,
             infer_datetime_format=False,
             keep_date_col=False,
             date_parser=None,
             dayfirst=False,
             cache_dates=True,
             # Quoting, Compression, and File Format
             compression="infer",
             thousands=None,
             decimal=b".",
             lineterminator=None,
             quotechar='"',
             quoting=csv.QUOTE_MINIMAL,
             doublequote=True,
             escapechar=None,
             comment=None,
             encoding=None,
             dialect=None,
             # Error Handling
             error_bad_lines=True,
             warn_bad_lines=True,
             # Internal
             delim_whitespace=False,
             low_memory=_c_parser_defaults["low_memory"],
             memory_map=False,
             float_precision=None):
    """
    Read a comma-separated values (csv) file into eland.DataFrame (i.e. an Elasticsearch index).

    **Modifies an Elasticsearch index**

     **Note iteration not supported**

    Parameters
    ----------
    es_params: Elasticsearch client argument(s)
        - elasticsearch-py parameters or
        - elasticsearch-py instance or
        - eland.Client instance
    es_dest_index: str
        Name of Elasticsearch index to be appended to
    es_if_exists : {'fail', 'replace', 'append'}, default 'fail'
        How to behave if the index already exists.

        - fail: Raise a ValueError.
        - replace: Delete the index before inserting new values.
        - append: Insert new values to the existing index. Create if does not exist.
    es_dropna: bool, default 'False'
        * True: Remove missing values (see pandas.Series.dropna)
        * False: Include missing values - may cause bulk to fail
    es_geo_points: list, default None
        List of columns to map to geo_point data type
    iterator
        ignored
    chunksize
        number of csv rows to read before bulk index into Elasticsearch

    Other Parameters
    ----------------
    Parameters derived from :pandas_api_docs:`read_csv`.

    See Also
    --------
    :pandas_api_docs:`read_csv` - for all parameters

    Notes
    -----
    TODO - currently the eland.DataFrame may not retain the order of the data in the csv.
    """
    kwds = dict()

    kwds.update(
        delimiter=delimiter,
        engine=engine,
        dialect=dialect,
        compression=compression,
        # engine_specified=engine_specified,
        doublequote=doublequote,
        escapechar=escapechar,
        quotechar=quotechar,
        quoting=quoting,
        skipinitialspace=skipinitialspace,
        lineterminator=lineterminator,
        header=header,
        index_col=index_col,
        names=names,
        prefix=prefix,
        skiprows=skiprows,
        skipfooter=skipfooter,
        na_values=na_values,
        true_values=true_values,
        false_values=false_values,
        keep_default_na=keep_default_na,
        thousands=thousands,
        comment=comment,
        decimal=decimal,
        parse_dates=parse_dates,
        keep_date_col=keep_date_col,
        dayfirst=dayfirst,
        date_parser=date_parser,
        cache_dates=cache_dates,
        nrows=nrows,
        # iterator=iterator,
        chunksize=chunksize,
        converters=converters,
        dtype=dtype,
        usecols=usecols,
        verbose=verbose,
        encoding=encoding,
        squeeze=squeeze,
        memory_map=memory_map,
        float_precision=float_precision,
        na_filter=na_filter,
        delim_whitespace=delim_whitespace,
        warn_bad_lines=warn_bad_lines,
        error_bad_lines=error_bad_lines,
        low_memory=low_memory,
        mangle_dupe_cols=mangle_dupe_cols,
        infer_datetime_format=infer_datetime_format,
        skip_blank_lines=skip_blank_lines,
    )

    if chunksize is None:
        kwds.update(chunksize=_default_chunk_size)

    client = Client(es_client)

    # read csv in chunks to pandas DataFrame and dump to eland DataFrame (and Elasticsearch)
    reader = pd.read_csv(filepath_or_buffer, **kwds)

    first_write = True
    for chunk in reader:
        if first_write:
            pandas_to_eland(chunk, client, es_dest_index, if_exists=es_if_exists, chunksize=chunksize,
                            refresh=es_refresh, dropna=es_dropna, geo_points=es_geo_points)
            first_write = False
        else:
            pandas_to_eland(chunk, client, es_dest_index, if_exists='append', chunksize=chunksize,
                            refresh=es_refresh, dropna=es_dropna, geo_points=es_geo_points)

    # Now create an eland.DataFrame that references the new index
    ed_df = DataFrame(client, es_dest_index)

    return ed_df

