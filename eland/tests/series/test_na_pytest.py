# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

from eland import eland_to_pandas
from eland.tests.common import TestData
from eland.tests.common import assert_pandas_eland_frame_equal


class TestSeriesNA(TestData):
    columns = [
        "currency",
        "customer_full_name",
        "geoip.country_iso_code",
        "geoip.region_name",
    ]

    def test_not_isna(self):
        ed_ecommerce = self.ed_ecommerce()
        pd_ecommerce = eland_to_pandas(ed_ecommerce)

        for column in self.columns:
            not_isna_ed_ecommerce = ed_ecommerce[~ed_ecommerce[column].isna()]
            not_isna_pd_ecommerce = pd_ecommerce[~pd_ecommerce[column].isna()]
            assert_pandas_eland_frame_equal(
                not_isna_pd_ecommerce, not_isna_ed_ecommerce
            )

    def test_isna(self):
        ed_ecommerce = self.ed_ecommerce()
        pd_ecommerce = eland_to_pandas(ed_ecommerce)

        isna_ed_ecommerce = ed_ecommerce[ed_ecommerce["geoip.region_name"].isna()]
        isna_pd_ecommerce = pd_ecommerce[pd_ecommerce["geoip.region_name"].isna()]
        assert_pandas_eland_frame_equal(isna_pd_ecommerce, isna_ed_ecommerce)

    def test_notna(self):
        ed_ecommerce = self.ed_ecommerce()
        pd_ecommerce = eland_to_pandas(ed_ecommerce)

        for column in self.columns:
            notna_ed_ecommerce = ed_ecommerce[ed_ecommerce[column].notna()]
            notna_pd_ecommerce = pd_ecommerce[pd_ecommerce[column].notna()]
            assert_pandas_eland_frame_equal(notna_pd_ecommerce, notna_ed_ecommerce)
