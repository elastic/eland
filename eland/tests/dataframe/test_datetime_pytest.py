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
from datetime import datetime

import numpy as np
import pandas as pd
from pandas.util.testing import assert_series_equal

import eland as ed
from eland.tests.common import ES_TEST_CLIENT
from eland.tests.common import TestData
from eland.tests.common import assert_pandas_eland_frame_equal
from eland.tests.common import assert_pandas_eland_series_equal


class TestDataFrameDateTime(TestData):
    times = ["2019-11-26T19:58:15.246+0000",
             "1970-01-01T00:00:03.000+0000"]
    time_index_name = 'test_time_formats'

    @classmethod
    def setup_class(cls):
        """ setup any state specific to the execution of the given class (which
        usually contains tests).
        """
        es = ES_TEST_CLIENT
        if es.indices.exists(cls.time_index_name):
            es.indices.delete(index=cls.time_index_name)
        dts = [datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%f%z")
               for time in cls.times]

        time_formats_docs = [TestDataFrameDateTime.get_time_values_from_datetime(dt)
                             for dt in dts]
        mappings = {'properties': {}}

        for field_name, field_value in time_formats_docs[0].items():
            mappings['properties'][field_name] = {}
            mappings['properties'][field_name]['type'] = 'date'
            mappings['properties'][field_name]['format'] = field_name

        body = {"mappings": mappings}
        index = 'test_time_formats'
        es.indices.delete(index=index, ignore=[400, 404])
        es.indices.create(index=index, body=body)

        for i, time_formats in enumerate(time_formats_docs):
            es.index(index=index, body=time_formats, id=i)
        es.indices.refresh(index=index)

    @classmethod
    def teardown_class(cls):
        """ teardown any state that was previously setup with a call to
        setup_class.
        """

        es = ES_TEST_CLIENT
        es.indices.delete(index=cls.time_index_name)

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

        mappings = ed.FieldMappings._generate_es_mappings(df)

        assert expected_mappings == mappings

        # Now create index
        index_name = 'eland_test_generate_es_mappings'

        ed_df = ed.pandas_to_eland(df, ES_TEST_CLIENT, index_name, es_if_exists="replace", es_refresh=True)

        #print(df.to_string())
        #print(ed_df.to_string())
        #print(ed_df.dtypes)
        #print(ed_df._to_pandas().dtypes)

        assert_series_equal(df.dtypes, ed_df.dtypes)

        assert_pandas_eland_frame_equal(df, ed_df)

    def test_all_formats(self):
        index_name = self.time_index_name
        ed_df = ed.read_es(ES_TEST_CLIENT, index_name)

        for format_name in self.time_formats.keys():
            times = [pd.to_datetime(datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S.%f%z")
                                    .strftime(self.time_formats[format_name]),
                                    format=self.time_formats[format_name])
                     for dt in self.times]

            ed_series = ed_df[format_name]
            pd_series = pd.Series(times,
                                  index=[str(i) for i in range(len(self.times))],
                                  name=format_name)

            assert_pandas_eland_series_equal(pd_series, ed_series)

    @staticmethod
    def get_time_values_from_datetime(dt: datetime) -> dict:
        time_formats = {
            "epoch_millis": int(dt.timestamp() * 1000),
            "epoch_second": int(dt.timestamp()),
            "strict_date_optional_time": dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + dt.strftime("%z"),
            "basic_date": dt.strftime("%Y%m%d"),
            "basic_date_time": dt.strftime("%Y%m%dT%H%M%S.%f")[:-3] + dt.strftime("%z"),
            "basic_date_time_no_millis": dt.strftime("%Y%m%dT%H%M%S%z"),
            "basic_ordinal_date": dt.strftime("%Y%j"),
            "basic_ordinal_date_time": dt.strftime("%Y%jT%H%M%S.%f")[:-3] + dt.strftime("%z"),
            "basic_ordinal_date_time_no_millis": dt.strftime("%Y%jT%H%M%S%z"),
            "basic_time": dt.strftime("%H%M%S.%f")[:-3] + dt.strftime("%z"),
            "basic_time_no_millis": dt.strftime("%H%M%S%z"),
            "basic_t_time": dt.strftime("T%H%M%S.%f")[:-3] + dt.strftime("%z"),
            "basic_t_time_no_millis": dt.strftime("T%H%M%S%z"),
            "basic_week_date": dt.strftime("%GW%V%u"),
            "basic_week_date_time": dt.strftime("%GW%V%uT%H%M%S.%f")[:-3] + dt.strftime("%z"),
            "basic_week_date_time_no_millis": dt.strftime("%GW%V%uT%H%M%S%z"),
            "strict_date": dt.strftime("%Y-%m-%d"),
            "date": dt.strftime("%Y-%m-%d"),
            "strict_date_hour": dt.strftime("%Y-%m-%dT%H"),
            "date_hour": dt.strftime("%Y-%m-%dT%H"),
            "strict_date_hour_minute": dt.strftime("%Y-%m-%dT%H:%M"),
            "date_hour_minute": dt.strftime("%Y-%m-%dT%H:%M"),
            "strict_date_hour_minute_second": dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "date_hour_minute_second": dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "strict_date_hour_minute_second_fraction": dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
            "date_hour_minute_second_fraction": dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
            "strict_date_hour_minute_second_millis": dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
            "date_hour_minute_second_millis": dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
            "strict_date_time": dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + dt.strftime("%z"),
            "date_time": dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + dt.strftime("%z"),
            "strict_date_time_no_millis": dt.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "date_time_no_millis": dt.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "strict_hour": dt.strftime("%H"),
            "hour": dt.strftime("%H"),
            "strict_hour_minute": dt.strftime("%H:%M"),
            "hour_minute": dt.strftime("%H:%M"),
            "strict_hour_minute_second": dt.strftime("%H:%M:%S"),
            "hour_minute_second": dt.strftime("%H:%M:%S"),
            "strict_hour_minute_second_fraction": dt.strftime("%H:%M:%S.%f")[:-3],
            "hour_minute_second_fraction": dt.strftime("%H:%M:%S.%f")[:-3],
            "strict_hour_minute_second_millis": dt.strftime("%H:%M:%S.%f")[:-3],
            "hour_minute_second_millis": dt.strftime("%H:%M:%S.%f")[:-3],
            "strict_ordinal_date": dt.strftime("%Y-%j"),
            "ordinal_date": dt.strftime("%Y-%j"),
            "strict_ordinal_date_time": dt.strftime("%Y-%jT%H:%M:%S.%f")[:-3] + dt.strftime("%z"),
            "ordinal_date_time": dt.strftime("%Y-%jT%H:%M:%S.%f")[:-3] + dt.strftime("%z"),
            "strict_ordinal_date_time_no_millis": dt.strftime("%Y-%jT%H:%M:%S%z"),
            "ordinal_date_time_no_millis": dt.strftime("%Y-%jT%H:%M:%S%z"),
            "strict_time": dt.strftime("%H:%M:%S.%f")[:-3] + dt.strftime("%z"),
            "time": dt.strftime("%H:%M:%S.%f")[:-3] + dt.strftime("%z"),
            "strict_time_no_millis": dt.strftime("%H:%M:%S%z"),
            "time_no_millis": dt.strftime("%H:%M:%S%z"),
            "strict_t_time": dt.strftime("T%H:%M:%S.%f")[:-3] + dt.strftime("%z"),
            "t_time": dt.strftime("T%H:%M:%S.%f")[:-3] + dt.strftime("%z"),
            "strict_t_time_no_millis": dt.strftime("T%H:%M:%S%z"),
            "t_time_no_millis": dt.strftime("T%H:%M:%S%z"),
            "strict_week_date": dt.strftime("%G-W%V-%u"),
            "week_date": dt.strftime("%G-W%V-%u"),
            "strict_week_date_time": dt.strftime("%G-W%V-%uT%H:%M:%S.%f")[:-3] + dt.strftime("%z"),
            "week_date_time": dt.strftime("%G-W%V-%uT%H:%M:%S.%f")[:-3] + dt.strftime("%z"),
            "strict_week_date_time_no_millis": dt.strftime("%G-W%V-%uT%H:%M:%S%z"),
            "week_date_time_no_millis": dt.strftime("%G-W%V-%uT%H:%M:%S%z"),
            "strict_weekyear": dt.strftime("%G"),
            "weekyear": dt.strftime("%G"),
            "strict_weekyear_week": dt.strftime("%G-W%V"),
            "weekyear_week": dt.strftime("%G-W%V"),
            "strict_weekyear_week_day": dt.strftime("%G-W%V-%u"),
            "weekyear_week_day": dt.strftime("%G-W%V-%u"),
            "strict_year": dt.strftime("%Y"),
            "year": dt.strftime("%Y"),
            "strict_year_month": dt.strftime("%Y-%m"),
            "year_month": dt.strftime("%Y-%m"),
            "strict_year_month_day": dt.strftime("%Y-%m-%d"),
            "year_month_day": dt.strftime("%Y-%m-%d"),
        }

        return time_formats

    time_formats = {
        "epoch_millis": "%Y-%m-%dT%H:%M:%S.%f",
        "epoch_second": "%Y-%m-%dT%H:%M:%S",
        "strict_date_optional_time": "%Y-%m-%dT%H:%M:%S.%f%z",
        "basic_date": "%Y%m%d",
        "basic_date_time": "%Y%m%dT%H%M%S.%f",
        "basic_date_time_no_millis": "%Y%m%dT%H%M%S%z",
        "basic_ordinal_date": "%Y%j",
        "basic_ordinal_date_time": "%Y%jT%H%M%S.%f%z",
        "basic_ordinal_date_time_no_millis": "%Y%jT%H%M%S%z",
        "basic_time": "%H%M%S.%f%z",
        "basic_time_no_millis": "%H%M%S%z",
        "basic_t_time": "T%H%M%S.%f%z",
        "basic_t_time_no_millis": "T%H%M%S%z",
        "basic_week_date": "%GW%V%u",
        "basic_week_date_time": "%GW%V%uT%H%M%S.%f%z",
        "basic_week_date_time_no_millis": "%GW%V%uT%H%M%S%z",
        "date": "%Y-%m-%d",
        "strict_date": "%Y-%m-%d",
        "strict_date_hour": "%Y-%m-%dT%H",
        "date_hour": "%Y-%m-%dT%H",
        "strict_date_hour_minute": "%Y-%m-%dT%H:%M",
        "date_hour_minute": "%Y-%m-%dT%H:%M",
        "strict_date_hour_minute_second": "%Y-%m-%dT%H:%M:%S",
        "date_hour_minute_second": "%Y-%m-%dT%H:%M:%S",
        "strict_date_hour_minute_second_fraction": "%Y-%m-%dT%H:%M:%S.%f",
        "date_hour_minute_second_fraction": "%Y-%m-%dT%H:%M:%S.%f",
        "strict_date_hour_minute_second_millis": "%Y-%m-%dT%H:%M:%S.%f",
        "date_hour_minute_second_millis": "%Y-%m-%dT%H:%M:%S.%f",
        "strict_date_time": "%Y-%m-%dT%H:%M:%S.%f%z",
        "date_time": "%Y-%m-%dT%H:%M:%S.%f%z",
        "strict_date_time_no_millis": "%Y-%m-%dT%H:%M:%S%z",
        "date_time_no_millis": "%Y-%m-%dT%H:%M:%S%z",
        "strict_hour": "%H",
        "hour": "%H",
        "strict_hour_minute": "%H:%M",
        "hour_minute": "%H:%M",
        "strict_hour_minute_second": "%H:%M:%S",
        "hour_minute_second": "%H:%M:%S",
        "strict_hour_minute_second_fraction": "%H:%M:%S.%f",
        "hour_minute_second_fraction": "%H:%M:%S.%f",
        "strict_hour_minute_second_millis": "%H:%M:%S.%f",
        "hour_minute_second_millis": "%H:%M:%S.%f",
        "strict_ordinal_date": "%Y-%j",
        "ordinal_date": "%Y-%j",
        "strict_ordinal_date_time": "%Y-%jT%H:%M:%S.%f%z",
        "ordinal_date_time": "%Y-%jT%H:%M:%S.%f%z",
        "strict_ordinal_date_time_no_millis": "%Y-%jT%H:%M:%S%z",
        "ordinal_date_time_no_millis": "%Y-%jT%H:%M:%S%z",
        "strict_time": "%H:%M:%S.%f%z",
        "time": "%H:%M:%S.%f%z",
        "strict_time_no_millis": "%H:%M:%S%z",
        "time_no_millis": "%H:%M:%S%z",
        "strict_t_time": "T%H:%M:%S.%f%z",
        "t_time": "T%H:%M:%S.%f%z",
        "strict_t_time_no_millis": "T%H:%M:%S%z",
        "t_time_no_millis": "T%H:%M:%S%z",
        "strict_week_date": "%G-W%V-%u",
        "week_date": "%G-W%V-%u",
        "strict_week_date_time": "%G-W%V-%uT%H:%M:%S.%f%z",
        "week_date_time": "%G-W%V-%uT%H:%M:%S.%f%z",
        "strict_week_date_time_no_millis": "%G-W%V-%uT%H:%M:%S%z",
        "week_date_time_no_millis": "%G-W%V-%uT%H:%M:%S%z",
        "strict_weekyear_week_day": "%G-W%V-%u",
        "weekyear_week_day": "%G-W%V-%u",
        "strict_year": "%Y",
        "year": "%Y",
        "strict_year_month": "%Y-%m",
        "year_month": "%Y-%m",
        "strict_year_month_day": "%Y-%m-%d",
        "year_month_day": "%Y-%m-%d"
    }

    # excluding these formats as pandas throws a ValueError
    # "strict_weekyear": ("%G", None) - not supported in pandas
    # "strict_weekyear_week": ("%G-W%V", None),
    # E   ValueError: ISO year directive '%G' must be used with the ISO week directive '%V' and a weekday directive '%A', '%a', '%w', or '%u'.
