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

from tests.common import TestData, assert_eland_frame_equal


class TestDataKQLQuery(TestData):
    def test_flights_match_query(self):
        ed_flights = self.ed_flights()

        left = ed_flights.kql_query("OriginCityName:Rome")[
            ed_flights["Carrier"] == "Kibana Airlines"
        ]

        right = ed_flights[ed_flights["Carrier"] == "Kibana Airlines"].kql_query(
            "OriginCityName:Rome"
        )

        assert len(left) > 0
        assert_eland_frame_equal(left, right)
