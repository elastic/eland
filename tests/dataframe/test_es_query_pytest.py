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

from tests.common import TestData, assert_eland_frame_equal


class TestDataEsQuery(TestData):
    def test_flights_match_query(self):
        ed_flights = self.ed_flights()

        left = ed_flights.es_query({"match": {"OriginCityName": "Rome"}})[
            ed_flights["Carrier"] == "Kibana Airlines"
        ]

        right = ed_flights[ed_flights["Carrier"] == "Kibana Airlines"].es_query(
            {"match": {"OriginCityName": "Rome"}}
        )

        assert len(left) > 0
        assert_eland_frame_equal(left, right)

    def test_es_query_allows_query_in_dict(self):
        ed_flights = self.ed_flights()

        left = ed_flights.es_query({"match": {"OriginCityName": "Rome"}})
        right = ed_flights.es_query({"query": {"match": {"OriginCityName": "Rome"}}})

        assert len(left) > 0
        assert_eland_frame_equal(left, right)

    def test_es_query_geo_location(self):
        df = self.ed_ecommerce()
        cur_nearby = df.es_query(
            {
                "bool": {
                    "filter": {
                        "geo_distance": {
                            "distance": "10km",
                            "geoip.location": {"lon": 55.3, "lat": 25.3},
                        }
                    }
                }
            }
        )["currency"].value_counts()

        assert cur_nearby["EUR"] == 476

    @pytest.mark.parametrize("query", [(), [], 1, True])
    def test_es_query_wrong_type(self, query):
        ed_flights = self.ed_flights_small()

        with pytest.raises(TypeError):
            ed_flights.es_query(query)
