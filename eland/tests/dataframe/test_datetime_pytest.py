# File called _pytest for PyCharm compatability
from datetime import datetime

from elasticsearch import Elasticsearch
import numpy as np
import pandas as pd

import eland as ed
from eland.tests.common import ELASTICSEARCH_HOST
from eland.tests.common import TestData
from eland.tests.common import assert_pandas_eland_frame_equal


class TestDataFrameDateTime(TestData):

    times = ["2019-11-26T19:58:15.246+0000",
             "1970-01-01T00:00:03.000+0000"]
    time_index_name = 'test_time_formats'

    @classmethod
    def setup_class(cls):
        """ setup any state specific to the execution of the given class (which
        usually contains tests).
        """
        es = Elasticsearch(ELASTICSEARCH_HOST)
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
        es = Elasticsearch(ELASTICSEARCH_HOST)
        #es.indices.delete(index=cls.time_index_name)

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

        ed_df = ed.pandas_to_eland(df, ELASTICSEARCH_HOST, index_name, if_exists="replace", refresh=True)
        ed_df_head = ed_df.head()

        assert_pandas_eland_frame_equal(df, ed_df_head)

    def test_all_formats(self):
        index_name = self.time_index_name
        ed_df = ed.read_es(ELASTICSEARCH_HOST, index_name)

        for format_name in self.time_formats.keys():
            times = [pd.to_datetime(datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S.%f%z")
                                    .strftime(self.time_formats[format_name][0]),
                                    format=self.time_formats[format_name][0],
                                    utc=self.time_formats[format_name][1])
                     for dt in self.times]

            ser1 = ed.eland_to_pandas(ed_df[format_name])
            ser2 = pd.Series(times,
                             index=[str(i) for i in range(len(self.times))],
                             name=format_name)
            #remove prints once debugged
            print(format_name)
            print(ser1)
            print(ser2)
            pd.testing.assert_series_equal(ser1, ser2)

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
            "strict_date_hour": dt.strftime("%Y-%m-%dT%H"),
            "strict_date_hour_minute": dt.strftime("%Y-%m-%dT%H:%M"),
            "strict_date_hour_minute_second": dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "strict_date_hour_minute_second_fraction": dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
            "strict_date_hour_minute_second_millis": dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
            "strict_date_time": dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + dt.strftime("%z"),
            "strict_date_time_no_millis": dt.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "strict_hour": dt.strftime("%H"),
            "strict_hour_minute": dt.strftime("%H:%M"),
            "strict_hour_minute_second": dt.strftime("%H:%M:%S"),
            "strict_hour_minute_second_fraction": dt.strftime("%H:%M:%S.%f")[:-3],
            "strict_hour_minute_second_millis": dt.strftime("%H:%M:%S.%f")[:-3],
            "strict_ordinal_date": dt.strftime("%Y-%j"),
            "strict_ordinal_date_time": dt.strftime("%Y-%jT%H:%M:%S.%f")[:-3] + dt.strftime("%z"),
            "strict_ordinal_date_time_no_millis": dt.strftime("%Y-%jT%H:%M:%S%z"),
            "strict_time": dt.strftime("%H:%M:%S.%f")[:-3] + dt.strftime("%z"),
            "strict_time_no_millis": dt.strftime("%H:%M:%S%z"),
            "strict_t_time": dt.strftime("T%H:%M:%S.%f")[:-3] + dt.strftime("%z"),
            "strict_t_time_no_millis": dt.strftime("T%H:%M:%S%z"),
            "strict_week_date": dt.strftime("%G-W%V-%u"),
            "strict_week_date_time": dt.strftime("%G-W%V-%uT%H:%M:%S.%f")[:-3] + dt.strftime("%z"),
            "strict_week_date_time_no_millis": dt.strftime("%G-W%V-%uT%H:%M:%S%z"),
            "strict_weekyear": dt.strftime("%G"),
            "strict_weekyear_week": dt.strftime("%G-W%V"),
            "strict_weekyear_week_day": dt.strftime("%G-W%V-%u"),
            "strict_year": dt.strftime("%Y"),
            "strict_year_month": dt.strftime("%Y-%m"),
            "strict_year_month_day": dt.strftime("%Y-%m-%d"),
        }

        return time_formats

    #TO DO
    # this need to be changed WIP
    time_formats = {
        "epoch_millis": ("%Y-%m-%dT%H:%M:%S.%f", None),
        "epoch_second": ("%Y-%m-%dT%H:%M:%S", None),
        "strict_date_optional_time": ("%Y-%m-%dT%H:%M:%S.%f%z", True),
        "basic_date": ("%Y%m%d", None),
        "basic_date_time": ("%Y%m%dT%H%M%S.%f", None),
        "basic_date_time_no_millis": ("%Y%m%dT%H%M%S%z", True),
        "basic_ordinal_date": ("%Y%j", None),
        "basic_ordinal_date_time": ("%Y%jT%H%M%S.%f%z", True),
        "basic_ordinal_date_time_no_millis": ("%Y%jT%H%M%S%z", None),
        "basic_time": ("%H%M%S.%f%z", None),
        "basic_time_no_millis": ("%H%M%S%z", None),
        "basic_t_time": ("T%H%M%S.%f%z", None),
        "basic_t_time_no_millis": ("T%H%M%S%z", True),
        "basic_week_date": ("%GW%V%u", False),
        "basic_week_date_time": ("%GW%V%uT%H%M%S.%f%z", True),
        "basic_week_date_time_no_millis": ("%GW%V%uT%H%M%S%z", True),
        "strict_date": ("%Y-%m-%d", False),
        "strict_date_hour": ("%Y-%m-%dT%H", False),
        "strict_date_hour_minute": ("%Y-%m-%dT%H:%M", False),
        "strict_date_hour_minute_second": ("%Y-%m-%dT%H:%M:%S", False),
        "strict_date_hour_minute_second_fraction": ("%Y-%m-%dT%H:%M:%S.%f", False),
        "strict_date_hour_minute_second_millis": ("%Y-%m-%dT%H:%M:%S.%f", False),
        "strict_date_time": ("%Y-%m-%dT%H:%M:%S.%f%z", True),
        "strict_date_time_no_millis": ("%Y-%m-%dT%H:%M:%S%z", True),
        "strict_hour": ("%H", False),
        "strict_hour_minute": ("%H:%M", False),
        "strict_hour_minute_second": ("%H:%M:%S", False),
        "strict_hour_minute_second_fraction": ("%H:%M:%S.%f", False),
        "strict_hour_minute_second_millis": ("%H:%M:%S.%f", False),
        "strict_ordinal_date": ("%Y-%j", False),
        "strict_ordinal_date_time": ("%Y-%jT%H:%M:%S.%f%z", True),
        "strict_ordinal_date_time_no_millis": ("%Y-%jT%H:%M:%S%z", True),
        "strict_time": ("%H:%M:%S.%f%z", True),
        "strict_time_no_millis": ("%H:%M:%S%z", True),
        "strict_t_time": ("T%H:%M:%S.%f", False),
        "strict_t_time_no_millis": ("T%H:%M:%S%z", True),
        "strict_week_date":("%G-W%V-%u", False),
        "strict_week_date_time": ("%G-W%V-%uT%H:%M:%S.%f%z", True),
        "strict_week_date_time_no_millis": ("%G-W%V-%uT%H:%M:%S%z", True),
        "strict_weekyear": ("%G", False),
        "strict_weekyear_week": ("%G-W%V", False),
        "strict_weekyear_week_day": ("%G-W%V-%u", False),
        "strict_year": ("%Y", False),
        "strict_year_month": ("%Y-%m", False),
        "strict_year_month_day": ("%Y-%m-%d", False),
    }
