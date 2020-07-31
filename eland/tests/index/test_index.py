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

import pytest
import pandas as pd
from eland import DataFrame, Index
from eland.tests.common import (
    ES_TEST_CLIENT,
    ECOMMERCE_INDEX_NAME,
    FLIGHTS_SMALL_INDEX_NAME,
)


def test_default_index():
    ed_df = DataFrame(
        ES_TEST_CLIENT,
        FLIGHTS_SMALL_INDEX_NAME,
    )
    assert isinstance(ed_df.index, Index)
    assert ed_df.index.es_index_fields == ("_id",)
    assert ed_df.index.sort_fields == ("_doc",)
    assert ed_df.index.is_source_fields == (("_id", False),)


def test_column_index():
    ed_df = DataFrame(
        ES_TEST_CLIENT,
        FLIGHTS_SMALL_INDEX_NAME,
        es_index_field="Carrier",
    )
    assert isinstance(ed_df.index, Index)
    assert ed_df.index.sort_fields == ("Carrier",)
    assert ed_df.index.is_source_fields == (("Carrier", True),)

    pd_df = ed_df.to_pandas()
    assert isinstance(pd_df.index, pd.Index)


def test_shape_and_size(df):
    assert df.index.shape == (48,)
    assert df.index.size == 48


def test_drop_and_filter(df):
    ed_df = DataFrame(
        ES_TEST_CLIENT,
        ECOMMERCE_INDEX_NAME,
        es_index_field="order_id",
    )
    pd_df = ed_df.to_pandas()
    df = df.set_objects(ed_df, pd_df)

    assert df.shape == (4675, 45)
    df = df.drop(index=[550375])
    assert df.shape == (4674, 45)

    with pytest.raises(KeyError) as e:
        df.drop(index=[550375, 550412])
    assert e.value.args[0] == "[550375] not found in axis"

    df = df.filter(axis="index", items=[550412, 550425, 550426])
    assert df.shape == (2, 45)
