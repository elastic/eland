# File called _pytest for PyCharm compatability

import numpy as np
import pandas as pd

import eland as ed
from eland.tests.common import ES_TEST_CLIENT
from eland.tests.common import TestData
from eland.tests.common import assert_pandas_eland_frame_equal


class TestDataFrameDateTime(TestData):

    def test_datetime_to_ms(self):
        df = pd.DataFrame(data={'A': np.random.rand(3),
                                'B': 1,
                                'C': 'foo',
                                'D': pd.Timestamp('20190102'),
                                'E': [1.0, 2.0, 3.0],
                                'F': False,
                                'G': [1, 2, 3]},
                          index=['0', '1', '2'])

        expected_mappings = {'mappings': {
            'properties': {'A': {'type': 'double'},
                           'B': {'type': 'long'},
                           'C': {'type': 'keyword'},
                           'D': {'type': 'date'},
                           'E': {'type': 'double'},
                           'F': {'type': 'boolean'},
                           'G': {'type': 'long'}}}}

        mappings = ed.Mappings._generate_es_mappings(df)

        assert expected_mappings == mappings

        # Now create index
        index_name = 'eland_test_generate_es_mappings'

        ed_df = ed.pandas_to_eland(df, ES_TEST_CLIENT, index_name, if_exists="replace", refresh=True)
        ed_df_head = ed_df.head()

        assert_pandas_eland_frame_equal(df, ed_df_head)

    def test_date_implicit_epoch_millis(self):
        index_name = 'date_implicit_epoch_millis'
        ed_df = ed.read_es(ELASTICSEARCH_HOST, index_name)

        ser1 = ed_df["date"]._to_pandas()
        ser2 = pd.Series(pd.to_datetime("1970-01-01T00:00:03"))

        assert ser1.values == ser2.values

    def test_date_implicit_strict_date_optional_time(self):
        index_name = 'date_implicit_strict_date_optional_time'
        ed_df = ed.read_es(ELASTICSEARCH_HOST, index_name)

        ser1 = ed_df["date"]._to_pandas()
        ser2 = pd.Series(pd.to_datetime("1970-01-01T00:00:03"))

        assert ser1.values == ser2.values

    def test_date_explicit_epoch_millis(self):
        index_name = 'date_explicit_epoch_millis'
        ed_df = ed.read_es(ELASTICSEARCH_HOST, index_name)

        ser1 = ed_df["date"]._to_pandas()
        ser2 = pd.Series(pd.to_datetime("1970-01-01T00:00:03"))

        assert ser1.values == ser2.values

    def test_date_explicit_epoch_second(self):
        index_name = 'date_explicit_epoch_second'
        ed_df = ed.read_es(ELASTICSEARCH_HOST, index_name)

        ser1 = ed_df["date"]._to_pandas()
        ser2 = pd.Series(pd.to_datetime("1970-01-01T00:00:03"))

        assert ser1.values == ser2.values

