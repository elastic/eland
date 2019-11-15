# File called _pytest for PyCharm compatability

import ast
import time

import eland as ed

from elasticsearch import Elasticsearch

import pandas as pd
from pandas.util.testing import assert_frame_equal

from eland.tests.common import ROOT_DIR
from eland.tests.common import TestData

from eland.tests import ELASTICSEARCH_HOST
from eland.tests import FLIGHTS_INDEX_NAME

from eland.tests.common import assert_pandas_eland_frame_equal


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

        # Now read the csv to an index
        now_millis = int(round(time.time() * 1000))

        test_index = FLIGHTS_INDEX_NAME + '.' + str(now_millis)
        es = Elasticsearch(ELASTICSEARCH_HOST)

        ed_flights_from_csv = ed.read_csv(results_file, es, test_index, index_col=0, es_refresh=True,
                                          es_geo_points=['OriginLocation', 'DestLocation'],
                                          converters={
                                              'DestLocation': lambda x: ast.literal_eval(x),
                                              'OriginLocation': lambda x: ast.literal_eval(x)}
                                          )
        pd_flights_from_csv = ed.eland_to_pandas(ed_flights_from_csv)

        # TODO - there is a 'bug' where the Elasticsearch index returns data in a different order to the CSV
        print(ed_flights_from_csv.head())
        print(pd_flights_from_csv.head())
