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

import inspect
import pytest
import pandas as pd
from .common import (
    assert_pandas_eland_frame_equal,
    assert_pandas_eland_series_equal,
    assert_frame_equal,
    assert_series_equal,
    _ed_flights,
    _pd_flights,
    _ed_ecommerce,
    _pd_ecommerce,
    _ed_flights_small,
    _pd_flights_small,
)
import eland as ed


class SymmetricAPIChecker:
    def __init__(self, ed_obj, pd_obj):
        self.ed = ed_obj
        self.pd = pd_obj

    def load_dataset(self, dataset):
        if dataset == "flights":
            self.ed = _ed_flights
            self.pd = _pd_flights.copy()
        elif dataset == "flights_small":
            self.ed = _ed_flights_small
            self.pd = _pd_flights_small.copy()
        elif dataset == "ecommerce":
            self.ed = _ed_ecommerce
            self.pd = _pd_ecommerce.copy()
        else:
            raise ValueError(f"Unknown dataset {dataset!r}")

    def return_value_checker(self, func_name):
        """Returns a function which wraps the requested function
        and checks the return value when that function is inevitably
        called.
        """

        def f(*args, **kwargs):
            ed_exc = None
            try:
                ed_obj = getattr(self.ed, func_name)(*args, **kwargs)
            except Exception as e:
                ed_exc = e
            pd_exc = None
            try:
                if func_name == "to_pandas":
                    pd_obj = self.pd
                else:
                    pd_obj = getattr(self.pd, func_name)(*args, **kwargs)
            except Exception as e:
                pd_exc = e

            self.check_exception(ed_exc, pd_exc)
            self.check_values(ed_obj, pd_obj)

            if isinstance(ed_obj, (ed.DataFrame, ed.Series)):
                return SymmetricAPIChecker(ed_obj, pd_obj)
            return pd_obj

        return f

    def check_values(self, ed_obj, pd_obj):
        """Checks that any two values coming from eland and pandas are equal"""
        if isinstance(ed_obj, ed.DataFrame):
            assert_pandas_eland_frame_equal(pd_obj, ed_obj)
        elif isinstance(ed_obj, ed.Series):
            assert_pandas_eland_series_equal(pd_obj, ed_obj)
        elif isinstance(ed_obj, pd.DataFrame):
            assert_frame_equal(ed_obj, pd_obj)
        elif isinstance(ed_obj, pd.Series):
            assert_series_equal(ed_obj, pd_obj)
        elif isinstance(ed_obj, pd.Index):
            assert ed_obj.equals(pd_obj)
        else:
            assert ed_obj == pd_obj

    def check_exception(self, ed_exc, pd_exc):
        """Checks that either an exception was raised or not from both eland and pandas"""
        assert (ed_exc is None) == (pd_exc is None) and type(ed_exc) == type(pd_exc)
        if pd_exc is not None:
            raise pd_exc

    def __getitem__(self, item):
        if isinstance(item, SymmetricAPIChecker):
            pd_item = item.pd
            ed_item = item.ed
        else:
            pd_item = ed_item = item

        ed_exc = None
        pd_exc = None
        try:
            pd_obj = self.pd[pd_item]
        except Exception as e:
            pd_exc = e
        try:
            ed_obj = self.ed[ed_item]
        except Exception as e:
            ed_exc = e

        self.check_exception(ed_exc, pd_exc)
        if isinstance(ed_obj, (ed.DataFrame, ed.Series)):
            return SymmetricAPIChecker(ed_obj, pd_obj)
        return pd_obj

    def __getattr__(self, item):
        if item == "to_pandas":
            return self.return_value_checker("to_pandas")

        pd_obj = getattr(self.pd, item)
        if inspect.isfunction(pd_obj) or inspect.ismethod(pd_obj):
            return self.return_value_checker(item)
        ed_obj = getattr(self.ed, item)

        self.check_values(ed_obj, pd_obj)

        if isinstance(ed_obj, (ed.DataFrame, ed.Series)):
            return SymmetricAPIChecker(ed_obj, pd_obj)
        return pd_obj


@pytest.fixture(scope="function")
def df():
    return SymmetricAPIChecker(
        ed_obj=_ed_flights_small, pd_obj=_pd_flights_small.copy()
    )
