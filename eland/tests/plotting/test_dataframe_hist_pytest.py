# File called _pytest for PyCharm compatability

from eland.tests.common import TestData

from matplotlib.testing.decorators import check_figures_equal

@check_figures_equal(extensions=['png'])
def test_plot(fig_test, fig_ref):
    test_data = TestData()

    pd_flights = test_data.pd_flights()[['DistanceKilometers', 'DistanceMiles', 'FlightDelayMin', 'FlightTimeHour']]
    ed_flights = test_data.ed_flights()[['DistanceKilometers', 'DistanceMiles', 'FlightDelayMin', 'FlightTimeHour']]

    pd_ax = fig_ref.subplots()
    pd_flights.hist(ax=pd_ax)

    ed_ax = fig_test.subplots()
    ed_flights.hist(ax=ed_ax)

