# File called _pytest for PyCharm compatability

from eland.tests.common import TestData


class TestDataFrameShape(TestData):

    def test_to_shape1(self):
        pd_ecommerce = self.pd_ecommerce()
        ed_ecommerce = self.ed_ecommerce()

        pd_shape = pd_ecommerce.shape
        ed_shape = ed_ecommerce.shape

        assert pd_shape == ed_shape

    def test_to_shape2(self):
        pd_flights = self.pd_flights()
        ed_flights = self.ed_flights()

        pd_shape = pd_flights.shape
        ed_shape = ed_flights.shape

        assert pd_shape == ed_shape
