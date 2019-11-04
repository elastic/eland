# File called _pytest for PyCharm compatability

import ast

import pandas as pd
from pandas.util.testing import (assert_frame_equal)

from eland.tests.common import ROOT_DIR
from eland.tests.common import TestData


class TestDataFrameToCSV(TestData):

    def test_to_csv_head(self):
        results_file = ROOT_DIR + '/dataframe/results/test_to_csv_head.csv'

        ed_flights = self.ed_flights().head()
        pd_flights = self.pd_flights().head()
        ed_flights.to_csv(results_file)
        # Converting back from csv is messy as pd_flights is created from a json file
        pd_from_csv = pd.read_csv(results_file, index_col=0, converters={
            'DestLocation': lambda x: ast.literal_eval(x),
            'OriginLocation': lambda x: ast.literal_eval(x)})
        pd_from_csv.index = pd_from_csv.index.map(str)
        pd_from_csv.timestamp = pd.to_datetime(pd_from_csv.timestamp)

        assert_frame_equal(pd_flights, pd_from_csv)

    def test_to_csv_full(self):
        results_file = ROOT_DIR + '/dataframe/results/test_to_csv_full.csv'

        # Test is slow as it's for the full dataset, but it is useful as it goes over 10000 docs
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        ed_flights.to_csv(results_file)
        # Converting back from csv is messy as pd_flights is created from a json file
        pd_from_csv = pd.read_csv(results_file, index_col=0, converters={
            'DestLocation': lambda x: ast.literal_eval(x),
            'OriginLocation': lambda x: ast.literal_eval(x)})
        pd_from_csv.index = pd_from_csv.index.map(str)
        pd_from_csv.timestamp = pd.to_datetime(pd_from_csv.timestamp)

        assert_frame_equal(pd_flights, pd_from_csv)
