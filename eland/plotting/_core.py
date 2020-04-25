# Licensed to Elasticsearch B.V under one or more agreements.
# Elasticsearch B.V licenses this file to you under the Apache 2.0 License.
# See the LICENSE file in the project root for more information

from eland.plotting._matplotlib.hist import hist_series, hist_frame


def ed_hist_series(
    self,
    by=None,
    ax=None,
    grid=True,
    xlabelsize=None,
    xrot=None,
    ylabelsize=None,
    yrot=None,
    figsize=None,
    bins=10,
    **kwds,
):
    """
    Draw histogram of the input series using matplotlib.

    See :pandas_api_docs:`pandas.Series.hist` for usage.

    Notes
    -----
    Derived from ``pandas.plotting._core.hist_frame 0.25.3``

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> df = ed.DataFrame('localhost', 'flights')
    >>> df[df.OriginWeather == 'Sunny']['FlightTimeMin'].hist(alpha=0.5, density=True) # doctest: +SKIP
    >>> df[df.OriginWeather != 'Sunny']['FlightTimeMin'].hist(alpha=0.5, density=True) # doctest: +SKIP
    >>> plt.show() # doctest: +SKIP

    """
    return hist_series(
        self,
        by=by,
        ax=ax,
        grid=grid,
        xlabelsize=xlabelsize,
        xrot=xrot,
        ylabelsize=ylabelsize,
        yrot=yrot,
        figsize=figsize,
        bins=bins,
        **kwds,
    )


def ed_hist_frame(
    data,
    column=None,
    by=None,
    grid=True,
    xlabelsize=None,
    xrot=None,
    ylabelsize=None,
    yrot=None,
    ax=None,
    sharex=False,
    sharey=False,
    figsize=None,
    layout=None,
    bins=10,
    **kwds,
):
    """
    Make a histogram of the DataFrame's.

    See :pandas_api_docs:`pandas.DataFrame.hist` for usage.

    Notes
    -----
    Derived from ``pandas.plotting._core.hist_frame 0.25.3``

    Ideally, we'd call the pandas method `hist_frame` directly
    with histogram data, but weights are applied to ALL series.
    For example, we can plot a histogram of pre-binned data via:

    .. code-block:: python

        counts, bins = np.histogram(data)
        plt.hist(bins[:-1], bins, weights=counts)

    However,

    .. code-block:: python

        ax.hist(data[col].dropna().values, bins=bins, **kwds)

    is for ``[col]`` and weights are a single array.

    Examples
    --------
    >>> df = ed.DataFrame('localhost', 'flights')
    >>> hist = df.select_dtypes(include=[np.number]).hist(figsize=[10,10]) # doctest: +SKIP
    """
    return hist_frame(
        data,
        column=column,
        by=by,
        grid=grid,
        xlabelsize=xlabelsize,
        xrot=xrot,
        ylabelsize=ylabelsize,
        yrot=yrot,
        ax=ax,
        sharex=sharex,
        sharey=sharey,
        figsize=figsize,
        layout=layout,
        bins=bins,
        **kwds,
    )
