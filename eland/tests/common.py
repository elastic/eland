import pytest

import eland as ed

import pandas as pd

from pandas.util.testing import (assert_frame_equal)

import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Create pandas and eland data frames
from eland.tests import ELASTICSEARCH_HOST
from eland.tests import FLIGHTS_DF_FILE_NAME, FLIGHTS_INDEX_NAME,\
    ECOMMERCE_DF_FILE_NAME, ECOMMERCE_INDEX_NAME

_pd_flights = pd.read_json(FLIGHTS_DF_FILE_NAME).sort_index()
_pd_flights['timestamp'] = \
    pd.to_datetime(_pd_flights['timestamp'])
_pd_flights.index = _pd_flights.index.map(str) # make index 'object' not int
_ed_flights = ed.read_es(ELASTICSEARCH_HOST, FLIGHTS_INDEX_NAME)

_pd_ecommerce = pd.read_json(ECOMMERCE_DF_FILE_NAME).sort_index()
_pd_ecommerce['order_date'] = \
    pd.to_datetime(_pd_ecommerce['order_date'])
_pd_ecommerce['products.created_on'] = \
    _pd_ecommerce['products.created_on'].apply(lambda x: pd.to_datetime(x))
_pd_ecommerce.insert(2, 'customer_birth_date', None)
_pd_ecommerce.index = _pd_ecommerce.index.map(str) # make index 'object' not int
_pd_ecommerce['customer_birth_date'].astype('datetime64')
_ed_ecommerce = ed.read_es(ELASTICSEARCH_HOST, ECOMMERCE_INDEX_NAME)

class TestData:

    def pd_flights(self):
        return _pd_flights

    def ed_flights(self):
        return _ed_flights

    def pd_ecommerce(self):
        return _pd_ecommerce

    def ed_ecommerce(self):
        return _ed_ecommerce

def assert_pandas_eland_frame_equal(left, right):
    if not isinstance(left, pd.DataFrame):
        raise AssertionError("Expected type {exp_type}, found {act_type} instead",
                             exp_type=type(pd.DataFrame), act_type=type(left))

    if not isinstance(right, ed.DataFrame):
        raise AssertionError("Expected type {exp_type}, found {act_type} instead",
                             exp_type=type(ed.DataFrame), act_type=type(right))

    # Use pandas tests to check similarity
    assert_frame_equal(left, right._to_pandas())


