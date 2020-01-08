#  Copyright 2019 Elasticsearch BV
#
#      Licensed under the Apache License, Version 2.0 (the "License");
#      you may not use this file except in compliance with the License.
#      You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#      Unless required by applicable law or agreed to in writing, software
#      distributed under the License is distributed on an "AS IS" BASIS,
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#      See the License for the specific language governing permissions and
#      limitations under the License.

# File called _pytest for PyCharm compatability
import pytest
from matplotlib.testing.decorators import check_figures_equal

from eland.tests.common import TestData


@check_figures_equal(extensions=['png'])
def test_plot_hist(fig_test, fig_ref):
    test_data = TestData()

    pd_flights = test_data.pd_flights()['FlightDelayMin']
    ed_flights = test_data.ed_flights()['FlightDelayMin']

    pd_ax = fig_ref.subplots()
    ed_ax = fig_test.subplots()

    pd_flights.hist(ax=pd_ax)
    ed_flights.hist(ax=ed_ax)
