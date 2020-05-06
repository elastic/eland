# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

# File called _pytest for PyCharm compatability

from eland.tests.common import TestData
from eland.tests.common import assert_pandas_eland_frame_equal


class TestDataFrameHeadTail(TestData):
    def test_head(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        ed_head_10 = ed_flights.head(10)
        pd_head_10 = pd_flights.head(10)
        assert_pandas_eland_frame_equal(pd_head_10, ed_head_10)

        ed_head_8 = ed_head_10.head(8)
        pd_head_8 = pd_head_10.head(8)
        assert_pandas_eland_frame_equal(pd_head_8, ed_head_8)

        ed_head_20 = ed_head_10.head(20)
        pd_head_20 = pd_head_10.head(20)
        assert_pandas_eland_frame_equal(pd_head_20, ed_head_20)

    def test_tail(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        ed_tail_10 = ed_flights.tail(10)
        pd_tail_10 = pd_flights.tail(10)
        assert_pandas_eland_frame_equal(pd_tail_10, ed_tail_10)

        ed_tail_8 = ed_tail_10.tail(8)
        pd_tail_8 = pd_tail_10.tail(8)
        assert_pandas_eland_frame_equal(pd_tail_8, ed_tail_8)

        ed_tail_20 = ed_tail_10.tail(20)
        pd_tail_20 = pd_tail_10.tail(20)
        assert_pandas_eland_frame_equal(pd_tail_20, ed_tail_20)

    def test_head_tail(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        ed_head_10 = ed_flights.head(10)
        pd_head_10 = pd_flights.head(10)
        assert_pandas_eland_frame_equal(pd_head_10, ed_head_10)

        ed_tail_8 = ed_head_10.tail(8)
        pd_tail_8 = pd_head_10.tail(8)
        assert_pandas_eland_frame_equal(pd_tail_8, ed_tail_8)

        ed_tail_5 = ed_tail_8.tail(5)
        pd_tail_5 = pd_tail_8.tail(5)
        assert_pandas_eland_frame_equal(pd_tail_5, ed_tail_5)

        ed_tail_4 = ed_tail_5.tail(4)
        pd_tail_4 = pd_tail_5.tail(4)
        assert_pandas_eland_frame_equal(pd_tail_4, ed_tail_4)

    def test_tail_head(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        ed_tail_10 = ed_flights.tail(10)
        pd_tail_10 = pd_flights.tail(10)
        assert_pandas_eland_frame_equal(pd_tail_10, ed_tail_10)

        ed_head_8 = ed_tail_10.head(8)
        pd_head_8 = pd_tail_10.head(8)
        assert_pandas_eland_frame_equal(pd_head_8, ed_head_8)

        ed_tail_5 = ed_head_8.tail(5)
        pd_tail_5 = pd_head_8.tail(5)
        assert_pandas_eland_frame_equal(pd_tail_5, ed_tail_5)

        ed_head_4 = ed_tail_5.head(4)
        pd_head_4 = pd_tail_5.head(4)
        assert_pandas_eland_frame_equal(pd_head_4, ed_head_4)

    def test_head_0(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        ed_head_0 = ed_flights.head(0)
        pd_head_0 = pd_flights.head(0)
        assert_pandas_eland_frame_equal(pd_head_0, ed_head_0)

    def test_doc_test_tail(self):
        df = self.ed_flights()
        df = df[(df.OriginAirportID == "AMS") & (df.FlightDelayMin > 60)]
        df = df[["timestamp", "OriginAirportID", "DestAirportID", "FlightDelayMin"]]
        df = df.tail()
        print(df)

    def test_doc_test_tail_empty(self):
        df = self.ed_flights()
        df = df[df.OriginAirportID == "NADA"]
        df = df.tail()
        assert df.shape[0] == 0

    def test_doc_test_tail_single(self):
        df = self.ed_flights_small()
        df = df[(df.Carrier == "Kibana Airlines") & (df.DestAirportID == "ITM")].tail()
        assert df.shape[0] == 1
