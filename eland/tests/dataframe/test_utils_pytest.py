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

import numpy as np
import pandas as pd

import eland as ed
from eland.field_mappings import FieldMappings
from eland.tests.common import ES_TEST_CLIENT, TestData, assert_pandas_eland_frame_equal


class TestDataFrameUtils(TestData):
    def test_generate_es_mappings(self):
        df = pd.DataFrame(
            data={
                "A": np.random.rand(3),
                "B": 1,
                "C": "foo",
                "D": pd.Timestamp("20190102"),
                "E": [1.0, 2.0, 3.0],
                "F": False,
                "G": [1, 2, 3],
            },
            index=["0", "1", "2"],
        )

        expected_mappings = {
            "mappings": {
                "properties": {
                    "A": {"type": "double"},
                    "B": {"type": "long"},
                    "C": {"type": "keyword"},
                    "D": {"type": "date"},
                    "E": {"type": "double"},
                    "F": {"type": "boolean"},
                    "G": {"type": "long"},
                }
            }
        }

        mappings = FieldMappings._generate_es_mappings(df)

        assert expected_mappings == mappings

        # Now create index
        index_name = "eland_test_generate_es_mappings"

        ed_df = ed.pandas_to_eland(
            df, ES_TEST_CLIENT, index_name, es_if_exists="replace", es_refresh=True
        )
        ed_df_head = ed_df.head()

        assert_pandas_eland_frame_equal(df, ed_df_head)

        ES_TEST_CLIENT.indices.delete(index=index_name)

    def test_pandas_to_eland_ignore_index(self):
        df = pd.DataFrame(
            data={
                "A": np.random.rand(3),
                "B": 1,
                "C": "foo",
                "D": pd.Timestamp("20190102"),
                "E": [1.0, 2.0, 3.0],
                "F": False,
                "G": [1, 2, 3],
                "H": "Long text",  # text
                "I": "52.36,4.83",  # geo point
            },
            index=["0", "1", "2"],
        )

        # Now create index
        index_name = "test_pandas_to_eland_ignore_index"

        ed_df = ed.pandas_to_eland(
            df,
            ES_TEST_CLIENT,
            index_name,
            es_if_exists="replace",
            es_refresh=True,
            use_pandas_index_for_es_ids=False,
            es_type_overrides={"H": "text", "I": "geo_point"},
        )

        # Check types
        expected_mapping = {
            "test_pandas_to_eland_ignore_index": {
                "mappings": {
                    "properties": {
                        "A": {"type": "double"},
                        "B": {"type": "long"},
                        "C": {"type": "keyword"},
                        "D": {"type": "date"},
                        "E": {"type": "double"},
                        "F": {"type": "boolean"},
                        "G": {"type": "long"},
                        "H": {"type": "text"},
                        "I": {"type": "geo_point"},
                    }
                }
            }
        }

        mapping = ES_TEST_CLIENT.indices.get_mapping(index_name)

        assert expected_mapping == mapping

        # Convert back to pandas and compare with original
        pd_df = ed.eland_to_pandas(ed_df)

        # Compare values excluding index
        assert df.values.all() == pd_df.values.all()

        # Ensure that index is populated by ES.
        assert not (df.index == pd_df.index).any()

        ES_TEST_CLIENT.indices.delete(index=index_name)

    def test_eland_to_pandas_performance(self):
        # TODO quantify this
        ed.eland_to_pandas(self.ed_flights(), show_progress=True)

        # This test calls the same method so is redundant
        # assert_pandas_eland_frame_equal(pd_df, self.ed_flights())
