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

from datetime import datetime, timedelta

import pandas as pd
import pytest
from elasticsearch.helpers import BulkIndexError

from eland import DataFrame, pandas_to_eland
from tests.common import (
    ES_TEST_CLIENT,
    assert_frame_equal,
    assert_pandas_eland_frame_equal,
)

dt = datetime.utcnow()
pd_df = pd.DataFrame(
    {
        "a": [1, 2, 3],
        "b": [1.0, 2.0, 3.0],
        "c": ["A", "B", "C"],
        "d": [dt, dt + timedelta(1), dt + timedelta(2)],
    },
    index=["0", "1", "2"],
)

pd_df2 = pd.DataFrame({"Z": [3, 2, 1], "a": ["C", "D", "E"]}, index=["0", "1", "2"])


@pytest.fixture(scope="function", autouse=True)
def delete_test_index():
    ES_TEST_CLIENT.indices.delete(index="test-index", ignore=404)
    yield
    ES_TEST_CLIENT.indices.delete(index="test-index", ignore=404)


class TestPandasToEland:
    def test_returns_eland_dataframe(self):
        df = pandas_to_eland(
            pd_df, es_client=ES_TEST_CLIENT, es_dest_index="test-index"
        )

        assert isinstance(df, DataFrame)
        assert "es_index_pattern: test-index" in df.es_info()

    def test_es_if_exists_fail(self):
        pandas_to_eland(pd_df, es_client=ES_TEST_CLIENT, es_dest_index="test-index")

        with pytest.raises(ValueError) as e:
            pandas_to_eland(pd_df, es_client=ES_TEST_CLIENT, es_dest_index="test-index")

        assert str(e.value) == (
            "Could not create the index [test-index] because it "
            "already exists. Change the 'es_if_exists' parameter "
            "to 'append' or 'replace' data."
        )

    def test_es_if_exists_replace(self):
        # Assert that 'replace' allows for creation
        df1 = pandas_to_eland(
            pd_df2,
            es_client=ES_TEST_CLIENT,
            es_dest_index="test-index",
            es_if_exists="replace",
            es_refresh=True,
        ).to_pandas()
        assert_frame_equal(pd_df2, df1)

        # Assert that 'replace' will replace existing mapping and entries
        df2 = pandas_to_eland(
            pd_df,
            es_client=ES_TEST_CLIENT,
            es_dest_index="test-index",
            es_if_exists="replace",
            es_refresh=True,
        )
        assert_pandas_eland_frame_equal(pd_df, df2)

        df3 = pandas_to_eland(
            pd_df2,
            es_client=ES_TEST_CLIENT,
            es_dest_index="test-index",
            es_if_exists="replace",
            es_refresh=True,
        ).to_pandas()
        assert_frame_equal(df1, df3)

    def test_es_if_exists_append(self):
        df1 = pandas_to_eland(
            pd_df,
            es_client=ES_TEST_CLIENT,
            es_dest_index="test-index",
            es_if_exists="append",
            es_refresh=True,
            # We use 'short' here specifically so that the
            # assumed type of 'long' is coerced into a 'short'
            # by append mode.
            es_type_overrides={"a": "short"},
        )
        assert_pandas_eland_frame_equal(pd_df, df1)
        assert df1.shape == (3, 4)

        pd_df2 = pd.DataFrame(
            {
                "a": [4, 5, 6],
                "b": [-1.0, -2.0, -3.0],
                "c": ["A", "B", "C"],
                "d": [dt, dt - timedelta(1), dt - timedelta(2)],
            },
            index=["3", "4", "5"],
        )
        df2 = pandas_to_eland(
            pd_df2,
            es_client=ES_TEST_CLIENT,
            es_dest_index="test-index",
            es_if_exists="append",
            es_refresh=True,
        )

        # Assert that the second pandas dataframe is actually appended
        assert df2.shape == (6, 4)
        pd_df3 = pd_df.append(pd_df2)
        assert_pandas_eland_frame_equal(pd_df3, df2)

    def test_es_if_exists_append_mapping_mismatch_schema_enforcement(self):
        df1 = pandas_to_eland(
            pd_df,
            es_client=ES_TEST_CLIENT,
            es_dest_index="test-index",
            es_if_exists="append",
            es_refresh=True,
        )

        with pytest.raises(ValueError) as e:
            pandas_to_eland(
                pd_df2,
                es_client=ES_TEST_CLIENT,
                es_dest_index="test-index",
                es_if_exists="append",
            )

        assert str(e.value) == (
            "DataFrame dtypes and Elasticsearch index mapping aren't compatible:\n"
            "- 'b' is missing from DataFrame columns\n"
            "- 'c' is missing from DataFrame columns\n"
            "- 'd' is missing from DataFrame columns\n"
            "- 'Z' is missing from ES index mapping\n"
            "- 'a' column type ('keyword') not compatible with ES index mapping type ('long')"
        )

        # Assert that the index isn't modified
        assert_pandas_eland_frame_equal(pd_df, df1)

    def test_es_if_exists_append_mapping_mismatch_no_schema_enforcement(self):
        pandas_to_eland(
            pd_df,
            es_client=ES_TEST_CLIENT,
            es_dest_index="test-index",
            es_if_exists="append",
            es_refresh=True,
        )

        pd_df2 = pd.DataFrame(
            {
                "a": [4, 5, 6],
                "b": [-1.0, -2.0, -3.0],
                "d": [dt, dt - timedelta(1), dt - timedelta(2)],
                "e": ["A", "B", "C"],
            },
            index=["3", "4", "5"],
        )

        pandas_to_eland(
            pd_df2,
            es_client=ES_TEST_CLIENT,
            es_dest_index="test-index",
            es_if_exists="append",
            es_refresh=True,
            es_verify_mapping_compatibility=False,
        )

        final_df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6],
                "b": [1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
                "c": ["A", "B", "C", None, None, None],
                "d": [
                    dt,
                    dt + timedelta(1),
                    dt + timedelta(2),
                    dt,
                    dt - timedelta(1),
                    dt - timedelta(2),
                ],
                "e": [None, None, None, "A", "B", "C"],
            },
            index=["0", "1", "2", "3", "4", "5"],
        )

        eland_df = DataFrame(ES_TEST_CLIENT, "test-index")
        # Assert that the index isn't modified
        assert_pandas_eland_frame_equal(final_df, eland_df)

    def test_es_if_exists_append_es_type_coerce_error(self):
        df1 = pandas_to_eland(
            pd_df,
            es_client=ES_TEST_CLIENT,
            es_dest_index="test-index",
            es_if_exists="append",
            es_refresh=True,
            es_type_overrides={"a": "byte"},
        )
        assert_pandas_eland_frame_equal(pd_df, df1)

        pd_df_short = pd.DataFrame(
            {
                "a": [128],  # This value is too large for 'byte'
                "b": [-1.0],
                "c": ["A"],
                "d": [dt],
            },
            index=["3"],
        )

        with pytest.raises(BulkIndexError) as e:
            pandas_to_eland(
                pd_df_short,
                es_client=ES_TEST_CLIENT,
                es_dest_index="test-index",
                es_if_exists="append",
            )

        # Assert that the value 128 caused the index error
        assert "Value [128] is out of range for a byte" in str(e.value.errors)

    def test_pandas_to_eland_text_inserts_keyword(self):
        es = ES_TEST_CLIENT
        df1 = pandas_to_eland(
            pd_df,
            es_client=es,
            es_dest_index="test-index",
            es_if_exists="append",
            es_refresh=True,
            es_type_overrides={
                "c": "text",
                "b": {"type": "float"},
                "d": {"type": "text"},
            },
        )
        assert es.indices.get_mapping(index="test-index") == {
            "test-index": {
                "mappings": {
                    "properties": {
                        "a": {"type": "long"},
                        "b": {"type": "float"},
                        "c": {
                            "fields": {"keyword": {"type": "keyword"}},
                            "type": "text",
                        },
                        "d": {"type": "text"},
                    }
                }
            }
        }

        # 'c' is aggregatable on 'keyword'
        assert df1.groupby("c").mean().to_dict() == {
            "a": {"A": 1.0, "B": 2.0, "C": 3.0},
            "b": {"A": 1.0, "B": 2.0, "C": 3.0},
        }

        # 'd' isn't aggregatable because it's missing the 'keyword'
        with pytest.raises(ValueError) as e:
            df1.groupby("d").mean()
        assert str(e.value) == (
            "Cannot use 'd' with groupby() because it has "
            "no aggregatable fields in Elasticsearch"
        )
