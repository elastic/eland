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

import numpy as np

import pandas.core.common as com
from pandas.core.dtypes.generic import (
    ABCIndexClass)
from pandas.plotting._matplotlib.tools import _flatten, _set_ticks_props, _subplots


def ed_hist_frame(ed_df, column=None, by=None, grid=True, xlabelsize=None,
                  xrot=None, ylabelsize=None, yrot=None, ax=None, sharex=False,
                  sharey=False, figsize=None, layout=None, bins=10, **kwds):
    """
    See :pandas_api_docs:`pandas.DataFrame.hist` for usage.

    Notes
    -----
    Derived from ``pandas.plotting._core.hist_frame 0.24.2`` - TODO update to ``0.25.1``

    Ideally, we'd call `hist_frame` directly with histogram data,
    but weights are applied to ALL series. For example, we can
    plot a histogram of pre-binned data via:

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
    # Start with empty pandas data frame derived from
    ed_df_bins, ed_df_weights = ed_df._hist(num_bins=bins)

    if by is not None:
        raise NotImplementedError("TODO")

    if column is not None:
        if not isinstance(column, (list, np.ndarray, ABCIndexClass)):
            column = [column]
        ed_df_bins = ed_df_bins[column]
        ed_df_weights = ed_df_weights[column]
    naxes = len(ed_df_bins.columns)

    fig, axes = _subplots(naxes=naxes, ax=ax, squeeze=False,
                          sharex=sharex, sharey=sharey, figsize=figsize,
                          layout=layout)
    _axes = _flatten(axes)

    for i, col in enumerate(com.try_sort(ed_df_bins.columns)):
        ax = _axes[i]

        # pandas code
        # pandas / plotting / _core.py: 2410
        # ax.hist(data[col].dropna().values, bins=bins, **kwds)

        ax.hist(ed_df_bins[col][:-1], bins=ed_df_bins[col], weights=ed_df_weights[col], **kwds)
        ax.set_title(col)
        ax.grid(grid)

    _set_ticks_props(axes, xlabelsize=xlabelsize, xrot=xrot,
                     ylabelsize=ylabelsize, yrot=yrot)
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    return axes

def ed_hist_series(ed_s, column=None, by=None, grid=True, xlabelsize=None,
                  xrot=None, ylabelsize=None, yrot=None, ax=None,
                  figsize=None, layout=None, bins=10, **kwds):
    """
    See :pandas_api_docs:`pandas.Series.hist` for usage.

    Notes
    -----
    Derived from ``pandas.plotting._core.hist_frame 0.24.2`` - TODO update to ``0.25.1``


    Examples
    --------
    >>> df = ed.DataFrame('localhost', 'ecommerce')
    >>> hist = df['taxful_total_price'].hist(figsize=[10,10]) # doctest: +SKIP
    """
    # this is mostly the same code as above, it has been split out
    # to a series specific method now so we can expand series plotting


    # Start with empty pandas data frame derived from
    ed_s_bins, ed_s_weights = ed_s._hist(num_bins=bins)

    if by is not None:
        raise NotImplementedError("TODO")

    # raise error rather than warning when series is not plottable
    if ed_s_bins.empty:
        raise ValueError("{} has no meaningful histogram interval. All values 0."
                        .format(ed_s.name))

    naxes = len(ed_s_bins.columns)
    fig, axes = _subplots(naxes=naxes, ax=ax, squeeze=False, figsize=figsize, layout=layout)
    _axes = _flatten(axes)
    for i, col in enumerate(com.try_sort(ed_s_bins.columns)):
        ax = _axes[i]
        ax.hist(ed_s_bins[col][:-1], bins=ed_s_bins[col], weights=ed_s_weights[col], **kwds)
        ax.grid(grid)

    _set_ticks_props(axes, xlabelsize=xlabelsize, xrot=xrot,
                     ylabelsize=ylabelsize, yrot=yrot)
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    return axes