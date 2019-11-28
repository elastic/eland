import pandas as pd


def eland_date_to_pandas_date(date_format):
    """
    Given a specific Elasticsearch format for a date datatype, returns the
    'partial' `to_datetime` function to parse a given value in that format

    **Date Formats: https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping-date-format.html#built-in-date-formats

    Parameters
    ----------
    date_format: str
        The Elasticsearch date format (ex. 'epoch_millis', 'epoch_second', etc.)

    Returns
    -------
    function
        A function that parses the input (date datatype) according to the right format
    """

    if date_format == "epoch_millis":
        return lambda x: pd.to_datetime(x, unit='ms')
    elif date_format == "epoch_second":
        return lambda x: pd.to_datetime(x, unit='s')
    else:
        return lambda x: pd.to_datetime(x)
