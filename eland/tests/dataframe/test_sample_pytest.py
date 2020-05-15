# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

# File called _pytest for PyCharm compatibility
import pytest
from pandas.testing import assert_frame_equal

from eland.tests.common import TestData
from eland import eland_to_pandas


class TestDataFrameSample(TestData):
    SEED = 42

    def build_from_index(self, sample_ed_flights):
        sample_pd_flights = self.pd_flights_small().loc[
            sample_ed_flights.index, sample_ed_flights.columns
        ]
        return sample_pd_flights

    def test_sample(self):
        ed_flights_small = self.ed_flights_small()
        first_sample = ed_flights_small.sample(n=10, random_state=self.SEED)
        second_sample = ed_flights_small.sample(n=10, random_state=self.SEED)

        assert_frame_equal(
            eland_to_pandas(first_sample), eland_to_pandas(second_sample)
        )

    def test_sample_raises(self):
        ed_flights_small = self.ed_flights_small()

        with pytest.raises(ValueError):
            ed_flights_small.sample(n=10, frac=0.1)

        with pytest.raises(ValueError):
            ed_flights_small.sample(frac=1.5)

        with pytest.raises(ValueError):
            ed_flights_small.sample(n=-1)

    def test_sample_basic(self):
        ed_flights_small = self.ed_flights_small()
        sample_ed_flights = ed_flights_small.sample(n=10, random_state=self.SEED)
        pd_from_eland = eland_to_pandas(sample_ed_flights)

        # build using index
        sample_pd_flights = self.build_from_index(pd_from_eland)

        assert_frame_equal(sample_pd_flights, pd_from_eland)

    def test_sample_frac_01(self):
        frac = 0.15
        ed_flights = self.ed_flights_small().sample(frac=frac, random_state=self.SEED)
        pd_from_eland = eland_to_pandas(ed_flights)
        pd_flights = self.build_from_index(pd_from_eland)

        assert_frame_equal(pd_flights, pd_from_eland)

        # assert right size from pd_flights
        size = len(self.pd_flights_small())
        assert len(pd_flights) == int(round(frac * size))

    def test_sample_on_boolean_filter(self):
        ed_flights = self.ed_flights_small()
        columns = ["timestamp", "OriginAirportID", "DestAirportID", "FlightDelayMin"]
        sample_ed_flights = ed_flights[columns].sample(n=5, random_state=self.SEED)
        pd_from_eland = eland_to_pandas(sample_ed_flights)
        sample_pd_flights = self.build_from_index(pd_from_eland)

        assert_frame_equal(sample_pd_flights, pd_from_eland)

    def test_sample_head(self):
        ed_flights = self.ed_flights_small()
        sample_ed_flights = ed_flights.sample(n=10, random_state=self.SEED)
        sample_pd_flights = self.build_from_index(eland_to_pandas(sample_ed_flights))

        pd_head_5 = sample_pd_flights.head(5)
        ed_head_5 = sample_ed_flights.head(5)
        assert_frame_equal(pd_head_5, eland_to_pandas(ed_head_5))

    def test_sample_shape(self):
        ed_flights = self.ed_flights_small()
        sample_ed_flights = ed_flights.sample(n=10, random_state=self.SEED)
        sample_pd_flights = self.build_from_index(eland_to_pandas(sample_ed_flights))

        assert sample_pd_flights.shape == sample_ed_flights.shape
