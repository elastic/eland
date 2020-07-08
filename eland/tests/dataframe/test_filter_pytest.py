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

# File called _pytest for PyCharm compatability

import pytest
from eland.tests.common import TestData
from eland.tests.common import assert_pandas_eland_frame_equal


class TestDataFrameFilter(TestData):
    def test_filter_arguments_mutually_exclusive(self):
        ed_flights_small = self.ed_flights_small()

        with pytest.raises(TypeError):
            ed_flights_small.filter(items=[], like="!", regex="!")
        with pytest.raises(TypeError):
            ed_flights_small.filter(items=[], regex="!")
        with pytest.raises(TypeError):
            ed_flights_small.filter(items=[], like="!")
        with pytest.raises(TypeError):
            ed_flights_small.filter(like="!", regex="!")
        with pytest.raises(TypeError):
            ed_flights_small.filter()

    @pytest.mark.parametrize(
        "items",
        [
            ["DestCountry", "Cancelled", "AvgTicketPrice"],
            [],
            ["notfound", "AvgTicketPrice"],
        ],
    )
    def test_flights_filter_columns_items(self, items):
        ed_flights_small = self.ed_flights_small()
        pd_flights_small = self.pd_flights_small()

        ed_df = ed_flights_small.filter(items=items)
        pd_df = pd_flights_small.filter(items=items)

        assert_pandas_eland_frame_equal(pd_df, ed_df)

    @pytest.mark.parametrize("like", ["Flight", "Nope"])
    def test_flights_filter_columns_like(self, like):
        ed_flights_small = self.ed_flights_small()
        pd_flights_small = self.pd_flights_small()

        ed_df = ed_flights_small.filter(like=like)
        pd_df = pd_flights_small.filter(like=like)

        assert_pandas_eland_frame_equal(pd_df, ed_df)

    @pytest.mark.parametrize("regex", ["^Flig", "^Flight.*r$", ".*", "^[^C]"])
    def test_flights_filter_columns_regex(self, regex):
        ed_flights_small = self.ed_flights_small()
        pd_flights_small = self.pd_flights_small()

        ed_df = ed_flights_small.filter(regex=regex)
        pd_df = pd_flights_small.filter(regex=regex)

        assert_pandas_eland_frame_equal(pd_df, ed_df)

    @pytest.mark.parametrize("items", [[], ["20"], [str(x) for x in range(30)]])
    def test_flights_filter_index_items(self, items):
        ed_flights_small = self.ed_flights_small()
        pd_flights_small = self.pd_flights_small()

        ed_df = ed_flights_small.filter(items=items, axis=0)
        pd_df = pd_flights_small.filter(items=items, axis=0)

        assert_pandas_eland_frame_equal(pd_df, ed_df)

    def test_flights_filter_index_like_and_regex(self):
        ed_flights_small = self.ed_flights_small()

        with pytest.raises(NotImplementedError):
            ed_flights_small.filter(like="2", axis=0)
        with pytest.raises(NotImplementedError):
            ed_flights_small.filter(regex="^2", axis=0)
