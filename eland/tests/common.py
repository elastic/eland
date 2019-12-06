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

import os

import pandas as pd
from pandas.util.testing import (assert_frame_equal, assert_series_equal)

import eland as ed

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Create pandas and eland data frames
from eland.tests import ES_TEST_CLIENT
from eland.tests import FLIGHTS_DF_FILE_NAME, FLIGHTS_INDEX_NAME, \
    FLIGHTS_SMALL_INDEX_NAME, \
    ECOMMERCE_DF_FILE_NAME, ECOMMERCE_INDEX_NAME

_pd_flights = pd.read_json(FLIGHTS_DF_FILE_NAME).sort_index()
_pd_flights['timestamp'] = \
    pd.to_datetime(_pd_flights['timestamp'])
_pd_flights.index = _pd_flights.index.map(str)  # make index 'object' not int
_ed_flights = ed.read_es(ES_TEST_CLIENT, FLIGHTS_INDEX_NAME)

_pd_flights_small = _pd_flights.head(48)
_ed_flights_small = ed.read_es(ES_TEST_CLIENT, FLIGHTS_SMALL_INDEX_NAME)

_pd_ecommerce = pd.read_json(ECOMMERCE_DF_FILE_NAME).sort_index()
_pd_ecommerce['order_date'] = \
    pd.to_datetime(_pd_ecommerce['order_date'])
_pd_ecommerce['products.created_on'] = \
    _pd_ecommerce['products.created_on'].apply(lambda x: pd.to_datetime(x))
_pd_ecommerce.insert(2, 'customer_birth_date', None)
_pd_ecommerce.index = _pd_ecommerce.index.map(str)  # make index 'object' not int
_pd_ecommerce['customer_birth_date'].astype('datetime64')
_ed_ecommerce = ed.read_es(ES_TEST_CLIENT, ECOMMERCE_INDEX_NAME)


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
        raise AssertionError("Expected type {exp_type}, found {act_type} instead".format(
            exp_type='pd.DataFrame', act_type=type(left)))

    if not isinstance(right, ed.DataFrame):
        raise AssertionError("Expected type {exp_type}, found {act_type} instead".format(
            exp_type='ed.DataFrame', act_type=type(right)))

    # Use pandas tests to check similarity
    assert_frame_equal(left, right._to_pandas())


def assert_eland_frame_equal(left, right):
    if not isinstance(left, ed.DataFrame):
        raise AssertionError("Expected type {exp_type}, found {act_type} instead".format(
            exp_type='ed.DataFrame', act_type=type(left)))

    if not isinstance(right, ed.DataFrame):
        raise AssertionError("Expected type {exp_type}, found {act_type} instead".format(
            exp_type='ed.DataFrame', act_type=type(right)))

    # Use pandas tests to check similarity
    assert_frame_equal(left._to_pandas(), right._to_pandas())


def assert_pandas_eland_series_equal(left, right, check_less_precise=False):
    if not isinstance(left, pd.Series):
        raise AssertionError("Expected type {exp_type}, found {act_type} instead".format(
            exp_type='pd.Series', act_type=type(left)))

    if not isinstance(right, ed.Series):
        raise AssertionError("Expected type {exp_type}, found {act_type} instead".format(
            exp_type='ed.Series', act_type=type(right)))

    # Use pandas tests to check similarity
    assert_series_equal(left, right._to_pandas(), check_less_precise=check_less_precise)
