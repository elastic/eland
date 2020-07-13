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


class TestDataFrameDrop:
    def test_drop(self, df):
        df.drop(["Carrier", "DestCityName"], axis=1)
        df.drop(columns=["Carrier", "DestCityName"])

        df.drop(["1", "2"])
        df.drop(["1", "2"], axis=0)
        df.drop(index=["1", "2"])

    def test_drop_all_columns(self, df):
        all_columns = list(df.columns)
        rows = df.shape[0]

        for dropped in (
            df.drop(labels=all_columns, axis=1),
            df.drop(columns=all_columns),
            df.drop(all_columns, axis=1),
        ):
            assert dropped.shape == (rows, 0)
            assert list(dropped.columns) == []

    def test_drop_all_index(self, df):
        all_index = list(df.pd.index)
        cols = df.shape[1]

        for dropped in (
            df.drop(all_index),
            df.drop(all_index, axis=0),
            df.drop(index=all_index),
        ):
            assert dropped.shape == (0, cols)
            assert list(dropped.to_pandas().index) == []
