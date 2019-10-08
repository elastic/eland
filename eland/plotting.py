import numpy as np

import pandas.core.common as com
from pandas.core.dtypes.generic import (
    ABCIndexClass)


def ed_hist_frame(ed_df, column=None, by=None, grid=True, xlabelsize=None,
               xrot=None, ylabelsize=None, yrot=None, ax=None, sharex=False,
               sharey=False, figsize=None, layout=None, bins=10, **kwds):
    """
    Derived from pandas.plotting._core.hist_frame 0.24.2

    Ideally, we'd call hist_frame directly with histogram data,
    but weights are applied to ALL series. For example, we can
    plot a histogram of pre-binned data via:

    counts, bins = np.histogram(data)
    plt.hist(bins[:-1], bins, weights=counts)

    However,

    ax.hist(data[col].dropna().values, bins=bins, **kwds)

    is for [col] and weights are a single array.

    We therefore cut/paste code.
    """
    # Start with empty pandas data frame derived from
    ed_df_bins, ed_df_weights = ed_df._hist(num_bins=bins)

    _raise_if_no_mpl()
    _converter._WARN = False
    if by is not None:
        raise NotImplementedError("TODO")
        """
        axes = grouped_hist(data, column=column, by=by, ax=ax, grid=grid,
                            figsize=figsize, sharex=sharex, sharey=sharey,
                            layout=layout, bins=bins, xlabelsize=xlabelsize,
                            xrot=xrot, ylabelsize=ylabelsize,
                            yrot=yrot, **kwds)
        """
        return axes

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
