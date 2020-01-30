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

# Default number of rows displayed (different to pandas where ALL could be displayed)
import warnings
from enum import Enum
from typing import Union

import pandas as pd

DEFAULT_NUM_ROWS_DISPLAYED = 60

DEFAULT_CHUNK_SIZE = 10000
DEFAULT_CSV_BATCH_OUTPUT_SIZE = 10000
DEFAULT_PROGRESS_REPORTING_NUM_ROWS = 10000
DEFAULT_ES_MAX_RESULT_WINDOW = 10000  # index.max_result_window


def docstring_parameter(*sub):
    def dec(obj):
        obj.__doc__ = obj.__doc__.format(*sub)
        return obj

    return dec


class SortOrder(Enum):
    ASC = 0
    DESC = 1

    @staticmethod
    def reverse(order):
        if order == SortOrder.ASC:
            return SortOrder.DESC

        return SortOrder.ASC

    @staticmethod
    def to_string(order):
        if order == SortOrder.ASC:
            return "asc"

        return "desc"

    @staticmethod
    def from_string(order):
        if order == "asc":
            return SortOrder.ASC

        return SortOrder.DESC

def elasticsearch_date_to_pandas_date(value: Union[int, str], date_format: str) -> pd.Timestamp:
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
    datetime: pd.Timestamp
        From https://www.elastic.co/guide/en/elasticsearch/reference/current/date.html
        Date formats can be customised, but if no format is specified then it uses the default:
        "strict_date_optional_time||epoch_millis"
        Therefore if no format is specified we assume either strict_date_optional_time
        or epoch_millis.
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
    elif date_format == "strict_date_optional_time":
        return pd.to_datetime(value, format="%Y-%m-%dT%H:%M:%S.%f%z", exact=False)
    elif date_format == "basic_date":
        return pd.to_datetime(value, format="%Y%m%d")
    elif date_format == "basic_date_time":
        return pd.to_datetime(value, format="%Y%m%dT%H%M%S.%f", exact=False)
    elif date_format == "basic_date_time_no_millis":
        return pd.to_datetime(value, format="%Y%m%dT%H%M%S%z")
    elif date_format == "basic_ordinal_date":
        return pd.to_datetime(value, format="%Y%j")
    elif date_format == "basic_ordinal_date_time":
        return pd.to_datetime(value, format="%Y%jT%H%M%S.%f%z", exact=False)
    elif date_format == "basic_ordinal_date_time_no_millis":
        return pd.to_datetime(value, format="%Y%jT%H%M%S%z")
    elif date_format == "basic_time":
        return pd.to_datetime(value, format="%H%M%S.%f%z", exact=False)
    elif date_format == "basic_time_no_millis":
        return pd.to_datetime(value, format="%H%M%S%z")
    elif date_format == "basic_t_time":
        return pd.to_datetime(value, format="T%H%M%S.%f%z", exact=False)
    elif date_format == "basic_t_time_no_millis":
        return pd.to_datetime(value, format="T%H%M%S%z")
    elif date_format == "basic_week_date":
        return pd.to_datetime(value, format="%GW%V%u")
    elif date_format == "basic_week_date_time":
        return pd.to_datetime(value, format="%GW%V%uT%H%M%S.%f%z", exact=False)
    elif date_format == "basic_week_date_time_no_millis":
        return pd.to_datetime(value, format="%GW%V%uT%H%M%S%z")
    elif date_format == "strict_date":
        return pd.to_datetime(value, format="%Y-%m-%d")
    elif date_format == "date":
        return pd.to_datetime(value, format="%Y-%m-%d")
    elif date_format == "strict_date_hour":
        return pd.to_datetime(value, format="%Y-%m-%dT%H")
    elif date_format == "date_hour":
        return pd.to_datetime(value, format="%Y-%m-%dT%H")
    elif date_format == "strict_date_hour_minute":
        return pd.to_datetime(value, format="%Y-%m-%dT%H:%M")
    elif date_format == "date_hour_minute":
        return pd.to_datetime(value, format="%Y-%m-%dT%H:%M")
    elif date_format == "strict_date_hour_minute_second":
        return pd.to_datetime(value, format="%Y-%m-%dT%H:%M:%S")
    elif date_format == "date_hour_minute_second":
        return pd.to_datetime(value, format="%Y-%m-%dT%H:%M:%S")
    elif date_format == "strict_date_hour_minute_second_fraction":
        return pd.to_datetime(value, format="%Y-%m-%dT%H:%M:%S.%f", exact=False)
    elif date_format == "date_hour_minute_second_fraction":
        return pd.to_datetime(value, format="%Y-%m-%dT%H:%M:%S.%f", exact=False)
    elif date_format == "strict_date_hour_minute_second_millis":
        return pd.to_datetime(value, format="%Y-%m-%dT%H:%M:%S.%f", exact=False)
    elif date_format == "date_hour_minute_second_millis":
        return pd.to_datetime(value, format="%Y-%m-%dT%H:%M:%S.%f", exact=False)
    elif date_format == "strict_date_time":
        return pd.to_datetime(value, format="%Y-%m-%dT%H:%M:%S.%f%z", exact=False)
    elif date_format == "date_time":
        return pd.to_datetime(value, format="%Y-%m-%dT%H:%M:%S.%f%z", exact=False)
    elif date_format == "strict_date_time_no_millis":
        return pd.to_datetime(value, format="%Y-%m-%dT%H:%M:%S%z")
    elif date_format == "date_time_no_millis":
        return pd.to_datetime(value, format="%Y-%m-%dT%H:%M:%S%z")
    elif date_format == "strict_hour":
        return pd.to_datetime(value, format="%H")
    elif date_format == "hour":
        return pd.to_datetime(value, format="%H")
    elif date_format == "strict_hour_minute":
        return pd.to_datetime(value, format="%H:%M")
    elif date_format == "hour_minute":
        return pd.to_datetime(value, format="%H:%M")
    elif date_format == "strict_hour_minute_second":
        return pd.to_datetime(value, format="%H:%M:%S")
    elif date_format == "hour_minute_second":
        return pd.to_datetime(value, format="%H:%M:%S")
    elif date_format == "strict_hour_minute_second_fraction":
        return pd.to_datetime(value, format="%H:%M:%S.%f", exact=False)
    elif date_format == "hour_minute_second_fraction":
        return pd.to_datetime(value, format="%H:%M:%S.%f", exact=False)
    elif date_format == "strict_hour_minute_second_millis":
        return pd.to_datetime(value, format="%H:%M:%S.%f", exact=False)
    elif date_format == "hour_minute_second_millis":
        return pd.to_datetime(value, format="%H:%M:%S.%f", exact=False)
    elif date_format == "strict_ordinal_date":
        return pd.to_datetime(value, format="%Y-%j")
    elif date_format == "ordinal_date":
        return pd.to_datetime(value, format="%Y-%j")
    elif date_format == "strict_ordinal_date_time":
        return pd.to_datetime(value, format="%Y-%jT%H:%M:%S.%f%z", exact=False)
    elif date_format == "ordinal_date_time":
        return pd.to_datetime(value, format="%Y-%jT%H:%M:%S.%f%z", exact=False)
    elif date_format == "strict_ordinal_date_time_no_millis":
        return pd.to_datetime(value, format="%Y-%jT%H:%M:%S%z")
    elif date_format == "ordinal_date_time_no_millis":
        return pd.to_datetime(value, format="%Y-%jT%H:%M:%S%z")
    elif date_format == "strict_time":
        return pd.to_datetime(value, format="%H:%M:%S.%f%z", exact=False)
    elif date_format == "time":
        return pd.to_datetime(value, format="%H:%M:%S.%f%z", exact=False)
    elif date_format == "strict_time_no_millis":
        return pd.to_datetime(value, format="%H:%M:%S%z")
    elif date_format == "time_no_millis":
        return pd.to_datetime(value, format="%H:%M:%S%z")
    elif date_format == "strict_t_time":
        return pd.to_datetime(value, format="T%H:%M:%S.%f%z", exact=False)
    elif date_format == "t_time":
        return pd.to_datetime(value, format="T%H:%M:%S.%f%z", exact=False)
    elif date_format == "strict_t_time_no_millis":
        return pd.to_datetime(value, format="T%H:%M:%S%z")
    elif date_format == "t_time_no_millis":
        return pd.to_datetime(value, format="T%H:%M:%S%z")
    elif date_format == "strict_week_date":
        return pd.to_datetime(value, format="%G-W%V-%u")
    elif date_format == "week_date":
        return pd.to_datetime(value, format="%G-W%V-%u")
    elif date_format == "strict_week_date_time":
        return pd.to_datetime(value, format="%G-W%V-%uT%H:%M:%S.%f%z", exact=False)
    elif date_format == "week_date_time":
        return pd.to_datetime(value, format="%G-W%V-%uT%H:%M:%S.%f%z", exact=False)
    elif date_format == "strict_week_date_time_no_millis":
        return pd.to_datetime(value, format="%G-W%V-%uT%H:%M:%S%z")
    elif date_format == "week_date_time_no_millis":
        return pd.to_datetime(value, format="%G-W%V-%uT%H:%M:%S%z")
    elif date_format == "strict_weekyear" or date_format == "weekyear":
        # TODO investigate if there is a way of converting this
        raise NotImplementedError("strict_weekyear is not implemented due to support in pandas")
        return pd.to_datetime(value, format="%G")
        # Not supported in pandas
        # ValueError: ISO year directive '%G' must be used with the ISO week directive '%V'
        # and a weekday directive '%A', '%a', '%w', or '%u'.
    elif date_format == "strict_weekyear_week" or date_format == "weekyear_week":
        # TODO investigate if there is a way of converting this
        raise NotImplementedError("strict_weekyear_week is not implemented due to support in pandas")
        return pd.to_datetime(value, format="%G-W%V")
        # Not supported in pandas
        # ValueError: ISO year directive '%G' must be used with the ISO week directive '%V'
        # and a weekday directive '%A', '%a', '%w', or '%u'.
    elif date_format == "strict_weekyear_week_day":
        return pd.to_datetime(value, format="%G-W%V-%u")
    elif date_format == "weekyear_week_day":
        return pd.to_datetime(value, format="%G-W%V-%u")
    elif date_format == "strict_year":
        return pd.to_datetime(value, format="%Y")
    elif date_format == "year":
        return pd.to_datetime(value, format="%Y")
    elif date_format == "strict_year_month":
        return pd.to_datetime(value, format="%Y-%m")
    elif date_format == "year_month":
        return pd.to_datetime(value, format="%Y-%m")
    elif date_format == "strict_year_month_day":
        return pd.to_datetime(value, format="%Y-%m-%d")
    elif date_format == "year_month_day":
        return pd.to_datetime(value, format="%Y-%m-%d")
    else:
        warnings.warn("The '{}' format is not explicitly supported."
                      "Using pandas.to_datetime(value) to parse value".format(date_format),
                      Warning)
        # TODO investigate how we could generate this just once for a bulk read.
        return pd.to_datetime(value)

