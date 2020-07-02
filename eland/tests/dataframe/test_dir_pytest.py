# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

# File called _pytest for PyCharm compatibility


from eland.tests.common import TestData


class TestDataFrameDir(TestData):
    def test_flights_dir(self):
        ed_flights = self.ed_flights()

        print(dir(ed_flights))

        autocomplete_attrs = dir(ed_flights)

        for c in ed_flights.columns:
            assert c in autocomplete_attrs
