# File called _pytest for PyCharm compatability

import pandas as pd

import eland as ed
from eland.tests.common import ELASTICSEARCH_HOST
from eland.tests.common import TestData
from eland.tests.common import assert_pandas_eland_frame_equal


class TestDataFrameQuery(TestData):

    def test_getitem_query(self):
        # Examples from:
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html
        pd_df = pd.DataFrame({'A': range(1, 6), 'B': range(10, 0, -2), 'C': range(10, 5, -1)},
                             index=['0', '1', '2', '3', '4'])

        # Now create index
        index_name = 'eland_test_query'

        ed_df = ed.pandas_to_eland(pd_df, ELASTICSEARCH_HOST, index_name, if_exists="replace", refresh=True)

        assert_pandas_eland_frame_equal(pd_df, ed_df)

        pd_df.info()
        ed_df.info()

        pd_q1 = pd_df[pd_df.A > 2]
        pd_q2 = pd_df[pd_df.A > pd_df.B]
        pd_q3 = pd_df[pd_df.B == pd_df.C]

        ed_q1 = ed_df[ed_df.A > 2]
        ed_q2 = ed_df[ed_df.A > ed_df.B]
        ed_q3 = ed_df[ed_df.B == ed_df.C]

        assert_pandas_eland_frame_equal(pd_q1, ed_q1)
        assert_pandas_eland_frame_equal(pd_q2, ed_q2)
        assert_pandas_eland_frame_equal(pd_q3, ed_q3)

        pd_q4 = pd_df[(pd_df.A > 2) & (pd_df.B > 3)]
        ed_q4 = ed_df[(ed_df.A > 2) & (ed_df.B > 3)]

        assert_pandas_eland_frame_equal(pd_q4, ed_q4)

    def test_simple_query(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        assert pd_flights.query('FlightDelayMin > 60').shape == \
               ed_flights.query('FlightDelayMin > 60').shape

    def test_isin_query(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        assert pd_flights[pd_flights.OriginAirportID.isin(['LHR','SYD'])].shape == \
               ed_flights[ed_flights.OriginAirportID.isin(['LHR','SYD'])].shape
