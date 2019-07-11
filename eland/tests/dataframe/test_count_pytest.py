# File called _pytest for PyCharm compatability

from eland.tests.common import TestData


class TestDataFrameCount(TestData):

    def test_to_count1(self):
        pd_ecommerce = self.pd_ecommerce()
        ed_ecommerce = self.ed_ecommerce()

        pd_count = pd_ecommerce.count()
        ed_count = ed_ecommerce.count()

        print(pd_count)
        print(ed_count)



