#  Copyright 2019 Elasticsearch BV
#
#      Licensed under the Apache License, Version 2.0 (the "License");
#      you may not use this file except in compliance with the License.
#      You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#      Unless required by applicable law or agreed to in writing, software
#      distributed under the License is distributed on an "AS IS" BASIS,
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#      See the License for the specific language governing permissions and
#      limitations under the License.

# File called _pytest for PyCharm compatability

import numpy as np
import pandas as pd

import eland as ed
from eland.tests.common import ES_TEST_CLIENT, assert_pandas_eland_frame_equal
from eland.tests.common import TestData


class TestDataFrameUtils(TestData):

    def test_generate_es_mappings(self):
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

        mappings = ed.FieldMappings._generate_es_mappings(df)

        assert expected_mappings == mappings

        # Now create index
        index_name = 'eland_test_generate_es_mappings'

        ed_df = ed.pandas_to_eland(df, ES_TEST_CLIENT, index_name, es_if_exists="replace", es_refresh=True)
        ed_df_head = ed_df.head()

        assert_pandas_eland_frame_equal(df, ed_df_head)

    def test_eland_to_pandas_performance(self):
        # TODO quantify this
        pd_df = ed.eland_to_pandas(self.ed_flights(), show_progress=True)

        # This test calls the same method so is redundant
        #assert_pandas_eland_frame_equal(pd_df, self.ed_flights())
