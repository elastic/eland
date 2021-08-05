#  Licensed to Elasticsearch B.V. under one or more contributor
#  license agreements. See the NOTICE file distributed with
#  this work for additional information regarding copyright
#  ownership. Elasticsearch B.V. licenses this file to you under
#  the Apache License, Version 2.0 (the "License"); you may
#  not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.

import re
import warnings
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import pandas as pd  # type: ignore
from elasticsearch import Elasticsearch

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

# Default number of rows displayed (different to pandas where ALL could be displayed)
DEFAULT_NUM_ROWS_DISPLAYED = 60
DEFAULT_CHUNK_SIZE = 10000
DEFAULT_CSV_BATCH_OUTPUT_SIZE = 10000
DEFAULT_PROGRESS_REPORTING_NUM_ROWS = 10000
DEFAULT_SEARCH_SIZE = 5000
DEFAULT_PIT_KEEP_ALIVE = "3m"
DEFAULT_PAGINATION_SIZE = 5000  # for composite aggregations
PANDAS_VERSION: Tuple[int, ...] = tuple(
    int(part) for part in pd.__version__.split(".") if part.isdigit()
)[:2]


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    EMPTY_SERIES_DTYPE = pd.Series().dtype


def build_pd_series(
    data: Dict[str, Any], dtype: Optional["DTypeLike"] = None, **kwargs: Any
) -> pd.Series:
    """Builds a pd.Series while squelching the warning
    for unspecified dtype on empty series
    """
    dtype = dtype or (EMPTY_SERIES_DTYPE if not data else dtype)
    if dtype is not None:
        kwargs["dtype"] = dtype
    return pd.Series(data, **kwargs)


def docstring_parameter(*sub: Any) -> Callable[[Any], Any]:
    def dec(obj: Any) -> Any:
        obj.__doc__ = obj.__doc__.format(*sub)
        return obj

    return dec


class SortOrder(Enum):
    ASC = 0
    DESC = 1

    @staticmethod
    def reverse(order: "SortOrder") -> "SortOrder":
        if order == SortOrder.ASC:
            return SortOrder.DESC

        return SortOrder.ASC

    @staticmethod
    def to_string(order: "SortOrder") -> str:
        if order == SortOrder.ASC:
            return "asc"

        return "desc"

    @staticmethod
    def from_string(order: str) -> "SortOrder":
        if order == "asc":
            return SortOrder.ASC

        return SortOrder.DESC


def elasticsearch_date_to_pandas_date(
    value: Union[int, str, float], date_format: Optional[str]
) -> pd.Timestamp:
    """
    Given a specific Elasticsearch format for a date datatype, returns the
    'partial' `to_datetime` function to parse a given value in that format

    **Date Formats: https://www.elastic.co/guide/en/elasticsearch/reference/current/mapping-date-format.html#built-in-date-formats

    Parameters
    ----------
    value: Union[int, str, float]
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

    if date_format is None or isinstance(value, (int, float)):
        try:
            return pd.to_datetime(
                value, unit="s" if date_format == "epoch_second" else "ms"
            )
        except ValueError:
            return pd.to_datetime(value)
    elif date_format == "epoch_millis":
        return pd.to_datetime(value, unit="ms")
    elif date_format == "epoch_second":
        return pd.to_datetime(value, unit="s")
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
        raise NotImplementedError(
            "strict_weekyear is not implemented due to support in pandas"
        )
        return pd.to_datetime(value, format="%G")
        # Not supported in pandas
        # ValueError: ISO year directive '%G' must be used with the ISO week directive '%V'
        # and a weekday directive '%A', '%a', '%w', or '%u'.
    elif date_format == "strict_weekyear_week" or date_format == "weekyear_week":
        # TODO investigate if there is a way of converting this
        raise NotImplementedError(
            "strict_weekyear_week is not implemented due to support in pandas"
        )
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
        warnings.warn(
            f"The '{date_format}' format is not explicitly supported."
            f"Using pandas.to_datetime(value) to parse value",
            Warning,
        )
        # TODO investigate how we could generate this just once for a bulk read.
        return pd.to_datetime(value)


def ensure_es_client(
    es_client: Union[str, List[str], Tuple[str, ...], Elasticsearch]
) -> Elasticsearch:
    if not isinstance(es_client, Elasticsearch):
        es_client = Elasticsearch(es_client)
    return es_client


def es_version(es_client: Elasticsearch) -> Tuple[int, int, int]:
    """Tags the current ES client with a cached '_eland_es_version'
    property if one doesn't exist yet for the current Elasticsearch version.
    """
    eland_es_version: Tuple[int, int, int]
    if not hasattr(es_client, "_eland_es_version"):
        version_info = es_client.info()["version"]["number"]
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version_info)
        if match is None:
            raise ValueError(
                f"Unable to determine Elasticsearch version. "
                f"Received: {version_info}"
            )
        eland_es_version = cast(
            Tuple[int, int, int], tuple(int(x) for x in match.groups())
        )
        es_client._eland_es_version = eland_es_version  # type: ignore
    else:
        eland_es_version = es_client._eland_es_version  # type: ignore
    return eland_es_version
