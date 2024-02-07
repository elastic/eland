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

# File called _pytest for PyCharm compatibility

from io import StringIO
from pathlib import Path

import pandas
from pandas.testing import assert_frame_equal

from tests.common import ROOT_DIR, TestData


class TestDataFrameToJSON(TestData):

    def test_to_json_default_arguments(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()
        ed_flights.to_json(ROOT_DIR + "/dataframe/results/eland_to_json.jsonl")
        pd_flights.to_json(ROOT_DIR + "/dataframe/results/pandas_to_json.jsonl")

        assert_frame_equal(
            pandas.read_json(ROOT_DIR + "/dataframe/results/eland_to_json.jsonl"),
            pandas.read_json(ROOT_DIR + "/dataframe/results/pandas_to_json.jsonl"),
        )

    def test_to_json_streaming_mode(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()
        ed_flights.to_json(
            ROOT_DIR + "/dataframe/results/streaming_eland_to_json.jsonl",
            lines=True,
            orient="records",
        )
        pd_flights.to_json(
            ROOT_DIR + "/dataframe/results/streaming_pandas_to_json.jsonl",
            lines=True,
            orient="records",
        )

        assert_frame_equal(
            pandas.read_json(
                ROOT_DIR + "/dataframe/results/streaming_eland_to_json.jsonl",
                lines=True,
                orient="records",
            ),
            pandas.read_json(
                ROOT_DIR + "/dataframe/results/streaming_pandas_to_json.jsonl",
                lines=True,
                orient="records",
            ),
        )

    def test_to_json_streaming_mode_pathlib(self):
        root_dir = Path(ROOT_DIR)

        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()
        ed_flights.to_json(
            root_dir / "dataframe" / "results" / "pathlib_eland_to_json.jsonl",
            lines=True,
            orient="records",
        )
        pd_flights.to_json(
            root_dir / "dataframe" / "results" / "pathlib_pandas_to_json.jsonl",
            lines=True,
            orient="records",
        )

        assert_frame_equal(
            pandas.read_json(
                root_dir / "dataframe" / "results" / "pathlib_eland_to_json.jsonl",
                lines=True,
                orient="records",
            ),
            pandas.read_json(
                root_dir / "dataframe" / "results" / "pathlib_pandas_to_json.jsonl",
                lines=True,
                orient="records",
            ),
        )

    def test_to_json_with_other_buffer(self):
        root_dir = Path(ROOT_DIR)

        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()
        output_buffer = StringIO()
        ed_flights.to_json(output_buffer, lines=True, orient="records")
        output_string = pd_flights.to_json(lines=True, orient="records")

        output_buffer.seek(0)  # rewind our StringIO object

        assert_frame_equal(
            pandas.read_json(output_buffer, lines=True, orient="records"),
            pandas.read_json(
                StringIO(output_string),
                lines=True,
                orient="records",
            ),
        )

    def test_to_json_with_file_handle(self):
        root_dir = Path(ROOT_DIR)

        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()
        with open(
            root_dir / "dataframe" / "results" / "fh_eland_to_json.jsonl", "w"
        ) as w:
            ed_flights.to_json(w)
        pd_flights.to_json(
            root_dir / "dataframe" / "results" / "check_pandas_to_json.jsonl"
        )

        assert_frame_equal(
            pandas.read_json(
                ROOT_DIR + "/dataframe/results/fh_eland_to_json.jsonl",
                lines=True,
                orient="records",
            ),
            pandas.read_json(
                ROOT_DIR + "/dataframe/results/check_pandas_to_json.jsonl",
                lines=True,
                orient="records",
            ),
        )
