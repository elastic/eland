# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

import os

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

import eland as ed

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Create pandas and eland data frames
from eland.tests import (
    ES_TEST_CLIENT,
    FLIGHTS_DF_FILE_NAME,
    FLIGHTS_INDEX_NAME,
    FLIGHTS_SMALL_INDEX_NAME,
    ECOMMERCE_DF_FILE_NAME,
    ECOMMERCE_INDEX_NAME,
)

_pd_flights = pd.read_json(FLIGHTS_DF_FILE_NAME).sort_index()
_pd_flights["timestamp"] = pd.to_datetime(_pd_flights["timestamp"])
_pd_flights.index = _pd_flights.index.map(str)  # make index 'object' not int
_ed_flights = ed.DataFrame(ES_TEST_CLIENT, FLIGHTS_INDEX_NAME)

_pd_flights_small = _pd_flights.head(48)
_ed_flights_small = ed.DataFrame(ES_TEST_CLIENT, FLIGHTS_SMALL_INDEX_NAME)

_pd_ecommerce = pd.read_json(ECOMMERCE_DF_FILE_NAME).sort_index()
_pd_ecommerce["order_date"] = pd.to_datetime(_pd_ecommerce["order_date"])
_pd_ecommerce["products.created_on"] = _pd_ecommerce["products.created_on"].apply(
    lambda x: pd.to_datetime(x)
)
_pd_ecommerce.insert(2, "customer_birth_date", None)
_pd_ecommerce.index = _pd_ecommerce.index.map(str)  # make index 'object' not int
_pd_ecommerce["customer_birth_date"].astype("datetime64")
_ed_ecommerce = ed.DataFrame(ES_TEST_CLIENT, ECOMMERCE_INDEX_NAME)


class TestData:
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


def assert_pandas_eland_frame_equal(left, right):
    if not isinstance(left, pd.DataFrame):
        raise AssertionError(f"Expected type pd.DataFrame, found {type(left)} instead")

    if not isinstance(right, ed.DataFrame):
        raise AssertionError(f"Expected type ed.DataFrame, found {type(right)} instead")

    # Use pandas tests to check similarity
    assert_frame_equal(left, right.to_pandas())


def assert_eland_frame_equal(left, right):
    if not isinstance(left, ed.DataFrame):
        raise AssertionError(f"Expected type ed.DataFrame, found {type(left)} instead")

    if not isinstance(right, ed.DataFrame):
        raise AssertionError(f"Expected type ed.DataFrame, found {type(right)} instead")

    # Use pandas tests to check similarity
    assert_frame_equal(left.to_pandas(), right.to_pandas())


def assert_pandas_eland_series_equal(left, right, check_less_precise=False):
    if not isinstance(left, pd.Series):
        raise AssertionError(f"Expected type pd.Series, found {type(left)} instead")

    if not isinstance(right, ed.Series):
        raise AssertionError(f"Expected type ed.Series, found {type(right)} instead")

    # Use pandas tests to check similarity
    assert_series_equal(left, right.to_pandas(), check_less_precise=check_less_precise)
