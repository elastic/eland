# File called _pytest for PyCharm compatability

import numpy as np
import pandas as pd

import eland as ed
from eland.tests.common import ELASTICSEARCH_HOST
from eland.tests.common import TestData


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

        ed.pandas_to_es(df, ELASTICSEARCH_HOST, index_name, if_exists="replace", refresh=True)

        ed_df = ed.DataFrame(ELASTICSEARCH_HOST, index_name)
        ed_df_head = ed_df.head()

        # assert_frame_equal(df, ed_df_head)
