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
from pandas.testing import assert_series_equal

from tests.common import TestData


class TestDataFrameCount(TestData):
    filter_data = [
        "AvgTicketPrice",
        "Cancelled",
        "dayOfWeek",
        "timestamp",
        "DestCountry",
    ]

    def test_count(self, df):
        df.load_dataset("ecommerce")
        df.count()

    def test_count_flights(self):

        pd_flights = self.pd_flights().filter(self.filter_data)
        ed_flights = self.ed_flights().filter(self.filter_data)

        pd_count = pd_flights.count()
        ed_count = ed_flights.count()

        assert_series_equal(pd_count, ed_count)
