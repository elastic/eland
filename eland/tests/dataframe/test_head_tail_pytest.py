# File called _pytest for PyCharm compatability
import pandas as pd
import io

import eland as ed

from pandas.util.testing import (
    assert_series_equal, assert_frame_equal)

class TestDataFrameHeadTail():

    def test_head(self):
        ed_flights = ed.read_es(es_params='localhost', index_pattern='flights')

        head_10 = ed_flights.head(10)
        print(head_10._query_compiler._operations._to_es_query())

        head_8 = head_10.head(8)
        print(head_8._query_compiler._operations._to_es_query())

        head_20 = head_10.head(20)
        print(head_20._query_compiler._operations._to_es_query())

    def test_tail(self):
        ed_flights = ed.read_es(es_params='localhost', index_pattern='flights')

        tail_10 = ed_flights.tail(10)
        print(tail_10._query_compiler._operations._to_es_query())
        print(tail_10)

        tail_8 = tail_10.tail(8)
        print(tail_8._query_compiler._operations._to_es_query())

        tail_20 = tail_10.tail(20)
        print(tail_20._query_compiler._operations._to_es_query())

    def test_head_tail(self):
        ed_flights = ed.read_es(es_params='localhost', index_pattern='flights')

        head_10 = ed_flights.head(10)
        print(head_10._query_compiler._operations._to_es_query())

        tail_8 = head_10.tail(8)
        print(tail_8._query_compiler._operations._to_es_query())

        tail_5 = tail_8.tail(5)
        print(tail_5._query_compiler._operations._to_es_query())

        head_4 = tail_5.head(4)
        print(head_4._query_compiler._operations._to_es_query())

    def test_tail_head(self):
        ed_flights = ed.read_es(es_params='localhost', index_pattern='flights')

        tail_10 = ed_flights.tail(10)
        print(tail_10._query_compiler._operations._to_es_query())

        head_8 = tail_10.head(8)
        print(head_8._query_compiler._operations._to_es_query())

        head_5 = head_8.head(5)
        print(head_5._query_compiler._operations._to_es_query())

        tail_4 = head_5.tail(4)
        print(tail_4._query_compiler._operations._to_es_query())

    def test_head_tail_print(self):
        ed_flights = ed.read_es(es_params='localhost', index_pattern='flights')

        tail_100 = ed_flights.tail(100)
        print(tail_100._query_compiler._operations._to_es_query())
        print(tail_100)

        head_10 = tail_100.head(10)
        print(head_10)

        tail_4 = head_10.tail(4)
        print(tail_4._query_compiler._operations._to_es_query())
        print(tail_4)
