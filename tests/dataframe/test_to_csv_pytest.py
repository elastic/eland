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

import ast
import time
from io import StringIO

import pandas as pd
from pandas.testing import assert_frame_equal

import eland as ed
from tests import ES_TEST_CLIENT, FLIGHTS_INDEX_NAME
from tests.common import ROOT_DIR, TestData


class TestDataFrameToCSV(TestData):
    def test_to_csv_head(self):
        results_file = ROOT_DIR + "/dataframe/results/test_to_csv_head.csv"

        ed_flights = self.ed_flights().head()
        pd_flights = self.pd_flights().head()
        ed_flights.to_csv(results_file)
        # Converting back from csv is messy as pd_flights is created from a json file
        pd_from_csv = pd.read_csv(
            results_file,
            index_col=0,
            converters={
                "DestLocation": lambda x: ast.literal_eval(x),
                "OriginLocation": lambda x: ast.literal_eval(x),
            },
        )
        pd_from_csv.index = pd_from_csv.index.map(str)
        pd_from_csv.timestamp = pd.to_datetime(pd_from_csv.timestamp)

        assert_frame_equal(pd_flights, pd_from_csv)

    def test_to_csv_full(self):
        results_file = ROOT_DIR + "/dataframe/results/test_to_csv_full.csv"

        # Test is slow as it's for the full dataset, but it is useful as it goes over 10000 docs
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        ed_flights.to_csv(results_file)
        # Converting back from csv is messy as pd_flights is created from a json file
        pd_from_csv = pd.read_csv(
            results_file,
            index_col=0,
            converters={
                "DestLocation": lambda x: ast.literal_eval(x),
                "OriginLocation": lambda x: ast.literal_eval(x),
            },
        )
        pd_from_csv.index = pd_from_csv.index.map(str)
        pd_from_csv.timestamp = pd.to_datetime(pd_from_csv.timestamp)

        assert_frame_equal(pd_flights, pd_from_csv)

        # Now read the csv to an index
        now_millis = int(round(time.time() * 1000))

        test_index = FLIGHTS_INDEX_NAME + "." + str(now_millis)

        ed_flights_from_csv = ed.csv_to_eland(
            results_file,
            ES_TEST_CLIENT,
            test_index,
            index_col=0,
            es_refresh=True,
            es_type_overrides={
                "OriginLocation": "geo_point",
                "DestLocation": "geo_point",
            },
            converters={
                "DestLocation": lambda x: ast.literal_eval(x),
                "OriginLocation": lambda x: ast.literal_eval(x),
            },
        )
        pd_flights_from_csv = ed.eland_to_pandas(ed_flights_from_csv)

        # TODO - there is a 'bug' where the Elasticsearch index returns data in a different order to the CSV
        print(ed_flights_from_csv.head())
        print(pd_flights_from_csv.head())

        # clean up index
        ES_TEST_CLIENT.indices.delete(index=test_index)

    def test_pd_to_csv_without_filepath(self):

        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        ret = ed_flights.to_csv()
        results = StringIO(ret)
        # Converting back from csv is messy as pd_flights is created from a json file
        pd_from_csv = pd.read_csv(
            results,
            index_col=0,
            converters={
                "DestLocation": lambda x: ast.literal_eval(x),
                "OriginLocation": lambda x: ast.literal_eval(x),
            },
        )
        pd_from_csv.index = pd_from_csv.index.map(str)
        pd_from_csv.timestamp = pd.to_datetime(pd_from_csv.timestamp)

        assert_frame_equal(pd_flights, pd_from_csv)
