# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

# File called _pytest for PyCharm compatability
from matplotlib.testing.decorators import check_figures_equal

from eland.tests.common import TestData


@check_figures_equal(extensions=["png"])
def test_plot_hist(fig_test, fig_ref):
    test_data = TestData()

    pd_flights = test_data.pd_flights()["FlightDelayMin"]
    ed_flights = test_data.ed_flights()["FlightDelayMin"]

    pd_flights.hist(figure=fig_ref)
    ed_flights.hist(figure=fig_test)


@check_figures_equal(extensions=["png"])
def test_plot_multiple_hists(fig_test, fig_ref):
    test_data = TestData()

    pd_flights = test_data.pd_flights()
    ed_flights = test_data.ed_flights()

    pd_flights[pd_flights.AvgTicketPrice < 250]["FlightDelayMin"].hist(
        figure=fig_ref, alpha=0.5, density=True
    )
    pd_flights[pd_flights.AvgTicketPrice > 250]["FlightDelayMin"].hist(
        figure=fig_ref, alpha=0.5, density=True
    )

    ed_flights[ed_flights.AvgTicketPrice < 250]["FlightDelayMin"].hist(
        figure=fig_test, alpha=0.5, density=True
    )
    ed_flights[ed_flights.AvgTicketPrice > 250]["FlightDelayMin"].hist(
        figure=fig_test, alpha=0.5, density=True
    )


@check_figures_equal(extensions=["png"])
def test_plot_multiple_hists_pretty(fig_test, fig_ref):
    test_data = TestData()

    pd_flights = test_data.pd_flights()
    ed_flights = test_data.ed_flights()

    pd_flights[pd_flights.OriginWeather == "Sunny"]["FlightTimeMin"].hist(
        figure=fig_ref, alpha=0.5, density=True
    )
    pd_flights[pd_flights.OriginWeather != "Sunny"]["FlightTimeMin"].hist(
        figure=fig_ref, alpha=0.5, density=True
    )

    ed_flights[ed_flights.OriginWeather == "Sunny"]["FlightTimeMin"].hist(
        figure=fig_test, alpha=0.5, density=True
    )
    ed_flights[ed_flights.OriginWeather != "Sunny"]["FlightTimeMin"].hist(
        figure=fig_test, alpha=0.5, density=True
    )
