# File called _pytest for PyCharm compatability
from io import StringIO

from eland.tests.common import TestData


class TestDataFrameInfo(TestData):

    def test_to_info1(self):
        ed_flights = self.ed_flights()
        pd_flights = self.pd_flights()

        ed_buf = StringIO()
        pd_buf = StringIO()

        # Ignore memory_usage and first line (class name)
        ed_flights.info(buf=ed_buf, memory_usage=False)
        pd_flights.info(buf=pd_buf, memory_usage=False)

        ed_buf_lines = ed_buf.getvalue().split('\n')
        pd_buf_lines = pd_buf.getvalue().split('\n')

        assert pd_buf_lines[1:] == ed_buf_lines[1:]

        print(self.ed_ecommerce().info())
