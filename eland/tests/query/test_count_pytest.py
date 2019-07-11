# File called _pytest for PyCharm compatability

from eland.tests.common import TestData

from eland import Query


class TestQueryCopy(TestData):

    def test_copy(self):
        q = Query()

        q.exists('field_a')
        q.exists('field_b', must=False)

        print(q.to_query())

        q1 = Query(q)

        q.exists('field_c', must=False)
        q1.exists('field_c1', must=False)

        print(q.to_query())
        print(q1.to_query())



