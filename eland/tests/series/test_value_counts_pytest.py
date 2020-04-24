# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

# File called _pytest for PyCharm compatability
import pytest
from pandas.testing import assert_series_equal

from eland.tests.common import TestData


class TestSeriesValueCounts(TestData):
    def test_value_counts(self):
        pd_s = self.pd_flights()["Carrier"]
        ed_s = self.ed_flights()["Carrier"]

        pd_vc = pd_s.value_counts()
        ed_vc = ed_s.value_counts()

        assert_series_equal(pd_vc, ed_vc)

    def test_value_counts_size(self):
        pd_s = self.pd_flights()["Carrier"]
        ed_s = self.ed_flights()["Carrier"]

        pd_vc = pd_s.value_counts()[:1]
        ed_vc = ed_s.value_counts(es_size=1)

        assert_series_equal(pd_vc, ed_vc)

    def test_value_counts_keyerror(self):
        ed_f = self.ed_flights()
        with pytest.raises(KeyError):
            assert ed_f["not_a_column"].value_counts()

    def test_value_counts_dataframe(self):
        # value_counts() is a series method, should raise AttributeError if called on a DataFrame
        ed_f = self.ed_flights()
        with pytest.raises(AttributeError):
            assert ed_f.value_counts()

    def test_value_counts_non_int(self):
        ed_s = self.ed_flights()["Carrier"]
        with pytest.raises(TypeError):
            assert ed_s.value_counts(es_size="foo")

    def test_value_counts_non_positive_int(self):
        ed_s = self.ed_flights()["Carrier"]
        with pytest.raises(ValueError):
            assert ed_s.value_counts(es_size=-9)

    @pytest.mark.filterwarnings("ignore:Aggregations not supported")
    def test_value_counts_non_aggregatable(self):
        ed_s = self.ed_ecommerce()["customer_first_name"]
        pd_s = self.pd_ecommerce()["customer_first_name"]

        pd_vc = pd_s.value_counts().head(20).sort_index()
        ed_vc = ed_s.value_counts(es_size=20).sort_index()

        assert_series_equal(pd_vc, ed_vc)

        ed_s = self.ed_ecommerce()["customer_gender"]
        with pytest.raises(ValueError):
            assert ed_s.value_counts()
