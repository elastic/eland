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

import gzip
import json
import os
from datetime import timedelta

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

import eland as ed
from eland.common import PANDAS_VERSION

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Create pandas and eland data frames
from tests import (
    ECOMMERCE_DF_FILE_NAME,
    ECOMMERCE_INDEX_NAME,
    ES_TEST_CLIENT,
    FLIGHTS_FILE_NAME,
    FLIGHTS_INDEX_NAME,
    FLIGHTS_SMALL_INDEX_NAME,
)

_ed_flights = ed.DataFrame(ES_TEST_CLIENT, FLIGHTS_INDEX_NAME)
flight_records = []
with gzip.open(FLIGHTS_FILE_NAME) as f:
    for json_obj in f:
        flight_records.append(json.loads(json_obj))
_pd_flights = pd.DataFrame.from_records(flight_records).reindex(
    _ed_flights.columns, axis=1
)
if PANDAS_VERSION[0] >= 2:
    _pd_flights["timestamp"] = pd.to_datetime(_pd_flights["timestamp"], format="mixed")
else:
    _pd_flights["timestamp"] = pd.to_datetime(_pd_flights["timestamp"])
# Mimic what copy_to in an Elasticsearch mapping would do, combining the two fields in a list
_pd_flights["Cities"] = _pd_flights.apply(
    lambda x: list(sorted([x["OriginCityName"], x["DestCityName"]])), axis=1
)
_pd_flights.index = _pd_flights.index.map(str)  # make index 'object' not int

_pd_flights_small = _pd_flights.head(48)
_ed_flights_small = ed.DataFrame(ES_TEST_CLIENT, FLIGHTS_SMALL_INDEX_NAME)

_pd_ecommerce = pd.read_json(ECOMMERCE_DF_FILE_NAME).sort_index()
_pd_ecommerce["order_date"] = pd.to_datetime(_pd_ecommerce["order_date"])
_pd_ecommerce["products.created_on"] = _pd_ecommerce["products.created_on"].apply(
    lambda x: pd.to_datetime(x)
)
_pd_ecommerce.insert(2, "customer_birth_date", None)
_pd_ecommerce.index = _pd_ecommerce.index.map(str)  # make index 'object' not int
_pd_ecommerce["customer_birth_date"].astype("datetime64[ns]")
_ed_ecommerce = ed.DataFrame(ES_TEST_CLIENT, ECOMMERCE_INDEX_NAME)


class TestData:
    client = ES_TEST_CLIENT

    def pd_flights(self):
        return _pd_flights

    def ed_flights(self):
        return _ed_flights

    def pd_flights_small(self):
        return _pd_flights_small

    def ed_flights_small(self):
        return _ed_flights_small

    def pd_ecommerce(self):
        return _pd_ecommerce

    def ed_ecommerce(self):
        return _ed_ecommerce


def assert_pandas_eland_frame_equal(left, right, **kwargs):
    if not isinstance(left, pd.DataFrame):
        raise AssertionError(f"Expected type pd.DataFrame, found {type(left)} instead")

    if not isinstance(right, ed.DataFrame):
        raise AssertionError(f"Expected type ed.DataFrame, found {type(right)} instead")

    # Use pandas tests to check similarity
    assert_frame_equal(left, right.to_pandas(), **kwargs)


def assert_eland_frame_equal(left, right, **kwargs):
    if not isinstance(left, ed.DataFrame):
        raise AssertionError(f"Expected type ed.DataFrame, found {type(left)} instead")

    if not isinstance(right, ed.DataFrame):
        raise AssertionError(f"Expected type ed.DataFrame, found {type(right)} instead")

    # Use pandas tests to check similarity
    assert_frame_equal(left.to_pandas(), right.to_pandas(), **kwargs)


def assert_pandas_eland_series_equal(left, right, **kwargs):
    if not isinstance(left, pd.Series):
        raise AssertionError(f"Expected type pd.Series, found {type(left)} instead")

    if not isinstance(right, ed.Series):
        raise AssertionError(f"Expected type ed.Series, found {type(right)} instead")

    # Use pandas tests to check similarity
    assert_series_equal(left, right.to_pandas(), **kwargs)


def assert_almost_equal(left, right, **kwargs):
    """Asserts left and right are almost equal. Left and right
    can be scalars, series, dataframes, etc
    """
    if isinstance(left, (ed.DataFrame, ed.Series)):
        left = left.to_pandas()
    if isinstance(right, (ed.DataFrame, ed.Series)):
        right = right.to_pandas()

    if isinstance(right, pd.DataFrame):
        kwargs.setdefault("check_exact", True)
        assert_frame_equal(left, right)
    elif isinstance(right, pd.Series):
        kwargs.setdefault("check_exact", True)
        assert_series_equal(left, right)
    elif isinstance(right, float):
        assert right * 0.99 <= left <= right * 1.01
    elif isinstance(right, pd.Timestamp):
        assert isinstance(left, pd.Timestamp) and right - timedelta(
            seconds=0.1
        ) < left < right + timedelta(seconds=0.1)
    elif right is pd.NaT:
        assert left is pd.NaT
    else:
        assert left == right, f"{left} != {right}"
