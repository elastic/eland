from typing import Union
import warnings

import pandas as pd


def eland_date_to_pandas_date(value: Union[int, str], date_format: str):
    """
    Given a specific Elasticsearch format for a date datatype, returns the
    'partial' `to_datetime` function to parse a given value in that format

    **Date Formats: https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping-date-format.html#built-in-date-formats

    Parameters
    ----------
    value: Union[int, str]
        The date value.
    date_format: str
        The Elasticsearch date format (ex. 'epoch_millis', 'epoch_second', etc.)

    Returns
    -------
    datetime if parsing succeeded.
    """

    if date_format is None:
        try:
            value = int(value)
            return pd.to_datetime(value, unit='ms')
        except ValueError:
            return pd.to_datetime(value)
    elif date_format == "epoch_millis":
        return pd.to_datetime(value, unit='ms')
    elif date_format == "epoch_second":
        return pd.to_datetime(value, unit='s')
    else:
        warnings.warn("The '{}' format is not explicitly supported."
                      "The parsed date might be wrong.".format(date_format),
                      Warning)
        return pd.to_datetime(value)
