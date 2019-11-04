# File called _pytest for PyCharm compatability

from eland import Query
from eland.tests.common import TestData


class TestQueryCopy(TestData):

    def test_copy(self):
        q = Query()

        q.exists('field_a')
        q.exists('field_b', must=False)

        print(q.to_search_body())

        q1 = Query(q)

        q.exists('field_c', must=False)
        q1.exists('field_c1', must=False)

        print(q.to_search_body())
        print(q1.to_search_body())
