# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

# File called _pytest for PyCharm compatability
import eland as ed
from eland.tests import ES_TEST_CLIENT
from eland.tests import FLIGHTS_INDEX_NAME
from eland.tests.common import TestData
from eland.tests.common import assert_pandas_eland_series_equal


class TestSeriesRename(TestData):
    def test_rename(self):
        pd_carrier = self.pd_flights()["Carrier"]
        ed_carrier = ed.Series(ES_TEST_CLIENT, FLIGHTS_INDEX_NAME, "Carrier")

        assert_pandas_eland_series_equal(pd_carrier, ed_carrier)

        pd_renamed = pd_carrier.rename("renamed")
        ed_renamed = ed_carrier.rename("renamed")

        print(pd_renamed)
        print(ed_renamed)

        print(ed_renamed.info_es())

        assert_pandas_eland_series_equal(pd_renamed, ed_renamed)

        pd_renamed2 = pd_renamed.rename("renamed2")
        ed_renamed2 = ed_renamed.rename("renamed2")

        print(ed_renamed2.info_es())

        assert "renamed2" == ed_renamed2.name

        assert_pandas_eland_series_equal(pd_renamed2, ed_renamed2)
