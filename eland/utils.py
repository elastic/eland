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

import csv

import pandas as pd
from pandas.io.parsers import _c_parser_defaults

from eland import Client, DEFAULT_CHUNK_SIZE
from eland import DataFrame
from eland import FieldMappings


def read_es(es_client, es_index_pattern):
    """
    Utility method to create an eland.Dataframe from an Elasticsearch index_pattern.
    (Similar to pandas.read_csv, but source data is an Elasticsearch index rather than
    a csv file)

    Parameters
    ----------
    es_client: Elasticsearch client argument(s)
        - elasticsearch-py parameters or
        - elasticsearch-py instance or
        - eland.Client instance
    es_index_pattern: str
        Elasticsearch index pattern

    Returns
    -------
    eland.DataFrame

    See Also
    --------
    eland.pandas_to_eland: Create an eland.Dataframe from pandas.DataFrame
    eland.eland_to_pandas: Create a pandas.Dataframe from eland.DataFrame
    """
    return DataFrame(client=es_client, index_pattern=es_index_pattern)


def pandas_to_eland(pd_df,
                    es_client,
                    es_dest_index,
                    es_if_exists='fail',
                    es_refresh=False,
                    es_dropna=False,
                    es_geo_points=None,
                    chunksize=None):
    """
    Append a pandas DataFrame to an Elasticsearch index.
    Mainly used in testing.
    Modifies the elasticsearch destination index

    Parameters
    ----------
    es_client: Elasticsearch client argument(s)
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
    es_refresh: bool, default 'False'
        Refresh es_dest_index after bulk index
    es_dropna: bool, default 'False'
        * True: Remove missing values (see pandas.Series.dropna)
        * False: Include missing values - may cause bulk to fail
    es_geo_points: list, default None
        List of columns to map to geo_point data type
    chunksize: int, default None
        number of pandas.DataFrame rows to read before bulk index into Elasticsearch

    Returns
    -------
    eland.Dataframe
        eland.DataFrame referencing data in destination_index

    Examples
    --------

    >>> pd_df = pd.DataFrame(data={'A': 3.141,
    ...                            'B': 1,
    ...                            'C': 'foo',
    ...                            'D': pd.Timestamp('20190102'),
    ...                            'E': [1.0, 2.0, 3.0],
    ...                            'F': False,
    ...                            'G': [1, 2, 3]},
    ...                      index=['0', '1', '2'])
    >>> type(pd_df)
    <class 'pandas.core.frame.DataFrame'>
    >>> pd_df
           A  B  ...      F  G
    0  3.141  1  ...  False  1
    1  3.141  1  ...  False  2
    2  3.141  1  ...  False  3
    <BLANKLINE>
    [3 rows x 7 columns]
    >>> pd_df.dtypes
    A           float64
    B             int64
    C            object
    D    datetime64[ns]
    E           float64
    F              bool
    G             int64
    dtype: object

    Convert `pandas.DataFrame` to `eland.DataFrame` - this creates an Elasticsearch index called `pandas_to_eland`.
    Overwrite existing Elasticsearch index if it exists `if_exists="replace"`, and sync index so it is
    readable on return `refresh=True`


    >>> ed_df = ed.pandas_to_eland(pd_df,
    ...                            'localhost',
    ...                            'pandas_to_eland',
    ...                            es_if_exists="replace",
    ...                            es_refresh=True)
    >>> type(ed_df)
    <class 'eland.dataframe.DataFrame'>
    >>> ed_df
           A  B  ...      F  G
    0  3.141  1  ...  False  1
    1  3.141  1  ...  False  2
    2  3.141  1  ...  False  3
    <BLANKLINE>
    [3 rows x 7 columns]
    >>> ed_df.dtypes
    A           float64
    B             int64
    C            object
    D    datetime64[ns]
    E           float64
    F              bool
    G             int64
    dtype: object

    See Also
    --------
    eland.read_es: Create an eland.Dataframe from an Elasticsearch index
    eland.eland_to_pandas: Create a pandas.Dataframe from eland.DataFrame
    """
    if chunksize is None:
        chunksize = DEFAULT_CHUNK_SIZE

    client = Client(es_client)

    mapping = FieldMappings._generate_es_mappings(pd_df, es_geo_points)

    # If table exists, check if_exists parameter
    if client.index_exists(index=es_dest_index):
        if es_if_exists == "fail":
            raise ValueError(
                "Could not create the index [{0}] because it "
                "already exists. "
                "Change the if_exists parameter to "
                "'append' or 'replace' data.".format(es_dest_index)
            )
        elif es_if_exists == "replace":
            client.index_delete(index=es_dest_index)
            client.index_create(index=es_dest_index, body=mapping)
        # elif if_exists == "append":
        # TODO validate mapping are compatible
    else:
        client.index_create(index=es_dest_index, body=mapping)

    # Now add data
    actions = []
    n = 0
    for row in pd_df.iterrows():
        # Use index as _id
        id = row[0]

        if es_dropna:
            values = row[1].dropna().to_dict()
        else:
            values = row[1].to_dict()

        # Use integer as id field for repeatable results
        action = {'_index': es_dest_index, '_source': values, '_id': str(id)}

        actions.append(action)

        n = n + 1

        if n % chunksize == 0:
            client.bulk(actions, refresh=es_refresh)
            actions = []

    client.bulk(actions, refresh=es_refresh)

    ed_df = DataFrame(client, es_dest_index)

    return ed_df


def eland_to_pandas(ed_df, show_progress=False):
    """
    Convert an eland.Dataframe to a pandas.DataFrame

    **Note: this loads the entire Elasticsearch index into in core pandas.DataFrame structures. For large
    indices this can create significant load on the Elasticsearch cluster and require signficant memory**

    Parameters
    ----------
    ed_df: eland.DataFrame
        The source eland.Dataframe referencing the Elasticsearch index
    show_progress: bool
        Output progress of option to stdout? By default False.

    Returns
    -------
    pandas.Dataframe
        pandas.DataFrame contains all rows and columns in eland.DataFrame

    Examples
    --------
    >>> ed_df = ed.DataFrame('localhost', 'flights').head()
    >>> type(ed_df)
    <class 'eland.dataframe.DataFrame'>
    >>> ed_df
       AvgTicketPrice  Cancelled  ... dayOfWeek           timestamp
    0      841.265642      False  ...         0 2018-01-01 00:00:00
    1      882.982662      False  ...         0 2018-01-01 18:27:00
    2      190.636904      False  ...         0 2018-01-01 17:11:14
    3      181.694216       True  ...         0 2018-01-01 10:33:28
    4      730.041778      False  ...         0 2018-01-01 05:13:00
    <BLANKLINE>
    [5 rows x 27 columns]

    Convert `eland.DataFrame` to `pandas.DataFrame` (Note: this loads entire Elasticsearch index into core memory)

    >>> pd_df = ed.eland_to_pandas(ed_df)
    >>> type(pd_df)
    <class 'pandas.core.frame.DataFrame'>
    >>> pd_df
       AvgTicketPrice  Cancelled  ... dayOfWeek           timestamp
    0      841.265642      False  ...         0 2018-01-01 00:00:00
    1      882.982662      False  ...         0 2018-01-01 18:27:00
    2      190.636904      False  ...         0 2018-01-01 17:11:14
    3      181.694216       True  ...         0 2018-01-01 10:33:28
    4      730.041778      False  ...         0 2018-01-01 05:13:00
    <BLANKLINE>
    [5 rows x 27 columns]

    Convert `eland.DataFrame` to `pandas.DataFrame` and show progress every 10000 rows

    >>> pd_df = ed.eland_to_pandas(ed.DataFrame('localhost', 'flights'), show_progress=True) # doctest: +SKIP
    2020-01-29 12:43:36.572395: read 10000 rows
    2020-01-29 12:43:37.309031: read 13059 rows

    See Also
    --------
    eland.read_es: Create an eland.Dataframe from an Elasticsearch index
    eland.pandas_to_eland: Create an eland.Dataframe from pandas.DataFrame
    """
    return ed_df._to_pandas(show_progress=show_progress)


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

    **Note pandas iteration options not supported**

    Parameters
    ----------
    es_client: Elasticsearch client argument(s)
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
    chunksize
        number of csv rows to read before bulk index into Elasticsearch

    Other Parameters
    ----------------
    Parameters derived from :pandas_api_docs:`pandas.read_csv`.

    See Also
    --------
    :pandas_api_docs:`pandas.read_csv`

    Notes
    -----
    iterator not supported

    Examples
    --------

    See if 'churn' index exists in Elasticsearch

    >>> from elasticsearch import Elasticsearch # doctest: +SKIP
    >>> es = Elasticsearch() # doctest: +SKIP
    >>> es.indices.exists(index="churn") # doctest: +SKIP
    False

    Read 'churn.csv' and use first column as _id (and eland.DataFrame index)
    ::

        # churn.csv
        ,state,account length,area code,phone number,international plan,voice mail plan,number vmail messages,total day minutes,total day calls,total day charge,total eve minutes,total eve calls,total eve charge,total night minutes,total night calls,total night charge,total intl minutes,total intl calls,total intl charge,customer service calls,churn
        0,KS,128,415,382-4657,no,yes,25,265.1,110,45.07,197.4,99,16.78,244.7,91,11.01,10.0,3,2.7,1,0
        1,OH,107,415,371-7191,no,yes,26,161.6,123,27.47,195.5,103,16.62,254.4,103,11.45,13.7,3,3.7,1,0
        ...

    >>>  ed.read_csv("churn.csv",
    ...             es_client='localhost',
    ...             es_dest_index='churn',
    ...             es_refresh=True,
    ...             index_col=0) # doctest: +SKIP
              account length  area code  churn  customer service calls  ... total night calls  total night charge total night minutes voice mail plan
    0                128        415      0                       1  ...                91               11.01               244.7             yes
    1                107        415      0                       1  ...               103               11.45               254.4             yes
    2                137        415      0                       0  ...               104                7.32               162.6              no
    3                 84        408      0                       2  ...                89                8.86               196.9              no
    4                 75        415      0                       3  ...               121                8.41               186.9              no
    ...              ...        ...    ...                     ...  ...               ...                 ...                 ...             ...
    3328             192        415      0                       2  ...                83               12.56               279.1             yes
    3329              68        415      0                       3  ...               123                8.61               191.3              no
    3330              28        510      0                       2  ...                91                8.64               191.9              no
    3331             184        510      0                       2  ...               137                6.26               139.2              no
    3332              74        415      0                       0  ...                77               10.86               241.4             yes
    <BLANKLINE>
    [3333 rows x 21 columns]

    Validate data now exists in 'churn' index:

    >>> es.search(index="churn", size=1) # doctest: +SKIP
    {'took': 1, 'timed_out': False, '_shards': {'total': 1, 'successful': 1, 'skipped': 0, 'failed': 0}, 'hits': {'total': {'value': 3333, 'relation': 'eq'}, 'max_score': 1.0, 'hits': [{'_index': 'churn', '_id': '0', '_score': 1.0, '_source': {'state': 'KS', 'account length': 128, 'area code': 415, 'phone number': '382-4657', 'international plan': 'no', 'voice mail plan': 'yes', 'number vmail messages': 25, 'total day minutes': 265.1, 'total day calls': 110, 'total day charge': 45.07, 'total eve minutes': 197.4, 'total eve calls': 99, 'total eve charge': 16.78, 'total night minutes': 244.7, 'total night calls': 91, 'total night charge': 11.01, 'total intl minutes': 10.0, 'total intl calls': 3, 'total intl charge': 2.7, 'customer service calls': 1, 'churn': 0}}]}}

    TODO - currently the eland.DataFrame may not retain the order of the data in the csv.
    """
    kwds = dict()

    kwds.update(
        sep=sep,
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
        kwds.update(chunksize=DEFAULT_CHUNK_SIZE)

    client = Client(es_client)

    # read csv in chunks to pandas DataFrame and dump to eland DataFrame (and Elasticsearch)
    reader = pd.read_csv(filepath_or_buffer, **kwds)

    first_write = True
    for chunk in reader:
        if first_write:
            pandas_to_eland(chunk, client, es_dest_index, es_if_exists=es_if_exists, chunksize=chunksize,
                            es_refresh=es_refresh, es_dropna=es_dropna, es_geo_points=es_geo_points)
            first_write = False
        else:
            pandas_to_eland(chunk, client, es_dest_index, es_if_exists='append', chunksize=chunksize,
                            es_refresh=es_refresh, es_dropna=es_dropna, es_geo_points=es_geo_points)

    # Now create an eland.DataFrame that references the new index
    ed_df = DataFrame(client, es_dest_index)

    return ed_df
