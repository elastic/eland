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
from eland import DataFrame, MultiIndex
from eland.tests.common import ES_TEST_CLIENT, FLIGHTS_SMALL_INDEX_NAME


def test_multi_index_type():
    ed_df = DataFrame(
        ES_TEST_CLIENT,
        FLIGHTS_SMALL_INDEX_NAME,
        es_index_field=("Carrier", "DestCountry"),
    )
    assert isinstance(ed_df.index, MultiIndex)
    assert ed_df.index.es_index_fields == ("Carrier", "DestCountry")
    assert ed_df.index.sort_fields == ("Carrier", "DestCountry")
    assert ed_df.index.is_source_fields == (("Carrier", True), ("DestCountry", True))

    pd_df = ed_df.to_pandas()
    assert isinstance(pd_df.index, pd.MultiIndex)


def test_multi_index_repr(df):
    ed_df = DataFrame(
        ES_TEST_CLIENT,
        FLIGHTS_SMALL_INDEX_NAME,
        es_index_field=("Carrier", "DestCountry"),
    )
    ed_df.drop(labels=["AvgTicketPrice"], axis="columns", inplace=True)
    print(ed_df._repr_html_())
    assert repr(ed_df) == (
        """                     Cancelled Carrier  ... dayOfWeek           timestamp
Carrier DestCountry                     ...                              
ES-Air  AR               False  ES-Air  ...         0 2018-01-01 01:30:47
        AR               False  ES-Air  ...         0 2018-01-01 23:43:02
        CA               False  ES-Air  ...         0 2018-01-01 01:08:20
        CN               False  ES-Air  ...         0 2018-01-01 07:58:17
        CN               False  ES-Air  ...         0 2018-01-01 21:30:40
...                        ...     ...  ...       ...                 ...
        AR               False  ES-Air  ...         0 2018-01-01 01:30:47
        AR               False  ES-Air  ...         0 2018-01-01 23:43:02
        CA               False  ES-Air  ...         0 2018-01-01 01:08:20
        CN               False  ES-Air  ...         0 2018-01-01 07:58:17
        CN               False  ES-Air  ...         0 2018-01-01 21:30:40

[48 rows x 26 columns]"""
    )
    assert ed_df._repr_html_() == (
        """<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Cancelled</th>
      <th>Carrier</th>
      <th>...</th>
      <th>dayOfWeek</th>
      <th>timestamp</th>
    </tr>
    <tr>
      <th>Carrier</th>
      <th>DestCountry</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="11" valign="top">ES-Air</th>
      <th>AR</th>
      <td>False</td>
      <td>ES-Air</td>
      <td>...</td>
      <td>0</td>
      <td>2018-01-01 01:30:47</td>
    </tr>
    <tr>
      <th>AR</th>
      <td>False</td>
      <td>ES-Air</td>
      <td>...</td>
      <td>0</td>
      <td>2018-01-01 23:43:02</td>
    </tr>
    <tr>
      <th>CA</th>
      <td>False</td>
      <td>ES-Air</td>
      <td>...</td>
      <td>0</td>
      <td>2018-01-01 01:08:20</td>
    </tr>
    <tr>
      <th>CN</th>
      <td>False</td>
      <td>ES-Air</td>
      <td>...</td>
      <td>0</td>
      <td>2018-01-01 07:58:17</td>
    </tr>
    <tr>
      <th>CN</th>
      <td>False</td>
      <td>ES-Air</td>
      <td>...</td>
      <td>0</td>
      <td>2018-01-01 21:30:40</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>AR</th>
      <td>False</td>
      <td>ES-Air</td>
      <td>...</td>
      <td>0</td>
      <td>2018-01-01 01:30:47</td>
    </tr>
    <tr>
      <th>AR</th>
      <td>False</td>
      <td>ES-Air</td>
      <td>...</td>
      <td>0</td>
      <td>2018-01-01 23:43:02</td>
    </tr>
    <tr>
      <th>CA</th>
      <td>False</td>
      <td>ES-Air</td>
      <td>...</td>
      <td>0</td>
      <td>2018-01-01 01:08:20</td>
    </tr>
    <tr>
      <th>CN</th>
      <td>False</td>
      <td>ES-Air</td>
      <td>...</td>
      <td>0</td>
      <td>2018-01-01 07:58:17</td>
    </tr>
    <tr>
      <th>CN</th>
      <td>False</td>
      <td>ES-Air</td>
      <td>...</td>
      <td>0</td>
      <td>2018-01-01 21:30:40</td>
    </tr>
  </tbody>
</table>
</div>
<p>48 rows × 26 columns</p>"""
    )


def test_multi_index_drop_filter_agg(df):
    ed_df = DataFrame(
        ES_TEST_CLIENT,
        FLIGHTS_SMALL_INDEX_NAME,
        es_index_field=("Carrier", "DestCountry"),
    )
    pd_df = ed_df.to_pandas()
    assert isinstance(pd_df.index, pd.MultiIndex)
    assert isinstance(ed_df.index, MultiIndex)
    assert list(pd_df.index) == [
        ("ES-Air", "AR"),
        ("ES-Air", "AR"),
        ("ES-Air", "CA"),
        ("ES-Air", "CN"),
        ("ES-Air", "CN"),
        ("ES-Air", "CO"),
        ("ES-Air", "JP"),
        ("ES-Air", "JP"),
        ("ES-Air", "US"),
        ("JetBeats", "AT"),
        ("JetBeats", "CH"),
        ("JetBeats", "FI"),
        ("JetBeats", "IN"),
        ("JetBeats", "IT"),
        ("JetBeats", "IT"),
        ("JetBeats", "JP"),
        ("JetBeats", "US"),
        ("JetBeats", "US"),
        ("JetBeats", "US"),
        ("JetBeats", "US"),
        ("Kibana Airlines", "AU"),
        ("Kibana Airlines", "CA"),
        ("Kibana Airlines", "CA"),
        ("Kibana Airlines", "CA"),
        ("Kibana Airlines", "CH"),
        ("Kibana Airlines", "CN"),
        ("Kibana Airlines", "CN"),
        ("Kibana Airlines", "CN"),
        ("Kibana Airlines", "DE"),
        ("Kibana Airlines", "GB"),
        ("Kibana Airlines", "IN"),
        ("Kibana Airlines", "IT"),
        ("Kibana Airlines", "IT"),
        ("Kibana Airlines", "JP"),
        ("Kibana Airlines", "JP"),
        ("Logstash Airways", "AT"),
        ("Logstash Airways", "AT"),
        ("Logstash Airways", "CA"),
        ("Logstash Airways", "CA"),
        ("Logstash Airways", "CH"),
        ("Logstash Airways", "CN"),
        ("Logstash Airways", "FR"),
        ("Logstash Airways", "IT"),
        ("Logstash Airways", "IT"),
        ("Logstash Airways", "IT"),
        ("Logstash Airways", "IT"),
        ("Logstash Airways", "PR"),
        ("Logstash Airways", "RU"),
    ]
    df = df.set_objects(ed_df, pd_df)
    assert df.shape == (48, 27)
    assert df.index.shape == (48,)
    assert df.index.size == 48

    df = df.drop(index=[("ES-Air", "AR")])
    df = df.drop(columns=["DestLocation", "OriginLocation"])
    assert df.size == 1150
    assert df.shape == (46, 25)
    assert df.mean(numeric_only=True).shape == (9,)

    # TODO: Uncomment when numeric_only=True works properly for types
    # df.max(numeric_only=True)
    # df.min(numeric_only=True)

    df = df.filter(
        axis="columns", items=["Carrier", "DestCountry", "AvgTicketPrice", "Cancelled"]
    )

    # TODO: Remove the get_objects() once min and max work for text fields.
    ed_df, pd_df = df.get_objects()
    ed_agg = ed_df.agg(["min", "max", "mean"])[["AvgTicketPrice", "Cancelled"]]
    pd_agg = pd_df.agg(["min", "max", "mean"])[["AvgTicketPrice", "Cancelled"]]
    df_agg = df.set_objects(ed_agg, pd_agg)
    assert df_agg.shape == (3, 2)


def test_multi_index_drop_in_index(df):
    ed_df = DataFrame(
        ES_TEST_CLIENT,
        FLIGHTS_SMALL_INDEX_NAME,
        es_index_field=("Carrier", "DestCountry"),
    )
    pd_df = ed_df.to_pandas()
    df = df.set_objects(ed_df, pd_df)
    assert df.index.size == 48

    df2 = df.drop([("ES-Air", "AR")])
    assert df2.index.size == 46

    df3 = df.drop([("ES-Air", "AR"), ("ES-Air", "CA"), ("Logstash Airways", "CN")])
    assert df3.index.size == 44

    # Dropping an item that eixsts and one that doesn't
    # shows both items in the error.
    with pytest.raises(KeyError) as e:
        df3.drop(index=[("ES-Air", "CA"), ("Logstash Airways", "RU")])
    assert (
        e.value.args[0]
        == "[('ES-Air', 'CA') ('Logstash Airways', 'RU')] not found in axis"
    )

    with pytest.raises(ValueError) as e:
        df.drop([("ES-Air", "CA", 1)])
    assert str(e.value) == "Length of names must match number of levels in MultiIndex."


def test_multiindex_drop_filter__id(df):
    ed_df = DataFrame(
        ES_TEST_CLIENT,
        FLIGHTS_SMALL_INDEX_NAME,
        es_index_field=("_id", "Carrier"),
    )
    pd_df = ed_df.to_pandas()
    assert list(pd_df.index) == [
        ("0", "Kibana Airlines"),
        ("1", "Logstash Airways"),
        ("10", "JetBeats"),
        ("11", "Logstash Airways"),
        ("12", "Logstash Airways"),
        ("13", "Logstash Airways"),
        ("14", "Logstash Airways"),
        ("15", "Kibana Airlines"),
        ("16", "Logstash Airways"),
        ("17", "ES-Air"),
        ("18", "ES-Air"),
        ("19", "JetBeats"),
        ("2", "Logstash Airways"),
        ("20", "JetBeats"),
        ("21", "ES-Air"),
        ("22", "JetBeats"),
        ("23", "Logstash Airways"),
        ("24", "Logstash Airways"),
        ("25", "ES-Air"),
        ("26", "Kibana Airlines"),
        ("27", "JetBeats"),
        ("28", "Kibana Airlines"),
        ("29", "Logstash Airways"),
        ("3", "Kibana Airlines"),
        ("30", "Kibana Airlines"),
        ("31", "ES-Air"),
        ("32", "Kibana Airlines"),
        ("33", "JetBeats"),
        ("34", "Logstash Airways"),
        ("35", "ES-Air"),
        ("36", "JetBeats"),
        ("37", "Kibana Airlines"),
        ("38", "ES-Air"),
        ("39", "JetBeats"),
        ("4", "Kibana Airlines"),
        ("40", "ES-Air"),
        ("41", "JetBeats"),
        ("42", "ES-Air"),
        ("43", "Logstash Airways"),
        ("44", "Kibana Airlines"),
        ("45", "Kibana Airlines"),
        ("46", "Kibana Airlines"),
        ("47", "Kibana Airlines"),
        ("5", "JetBeats"),
        ("6", "JetBeats"),
        ("7", "Kibana Airlines"),
        ("8", "Kibana Airlines"),
        ("9", "Logstash Airways"),
    ]

    df = df.set_objects(ed_df, pd_df)
    assert df.shape == (48, 27)

    df = df.drop(index=[("12", "Logstash Airways")])
    assert df.shape == (47, 27)

    df = df.filter(
        items=[("47", "Kibana Airlines"), ("6", "JetBeats"), ("7", "Kibana Airlines")],
        axis="index",
    )
    assert df.shape == (3, 27)
