import numpy as np

import pandas.core.common as com
from pandas.core.dtypes.generic import (
    ABCIndexClass)
from pandas.plotting._core import (
    _raise_if_no_mpl, _converter, grouped_hist, _subplots, _flatten, _set_ticks_props)


def hist_frame(ed_df, column=None, by=None, grid=True, xlabelsize=None,
               xrot=None, ylabelsize=None, yrot=None, ax=None, sharex=False,
               sharey=False, figsize=None, layout=None, bins=10, **kwds):
    """
    Derived from pandas.plotting._core.hist_frame 0.24.2
    """
    # Start with empty pandas data frame derived from
    empty_pd_df = ed_df._empty_pd_df()

    _raise_if_no_mpl()
    _converter._WARN = False
    if by is not None:
        axes = grouped_hist(empty_pd_df, column=column, by=by, ax=ax, grid=grid,
                            figsize=figsize, sharex=sharex, sharey=sharey,
                            layout=layout, bins=bins, xlabelsize=xlabelsize,
                            xrot=xrot, ylabelsize=ylabelsize,
                            yrot=yrot, **kwds)
        return axes

    if column is not None:
        if not isinstance(column, (list, np.ndarray, ABCIndexClass)):
            column = [column]
        empty_pd_df = empty_pd_df[column]
    data = empty_pd_df._get_numeric_data()
    naxes = len(empty_pd_df.columns)

    fig, axes = _subplots(naxes=naxes, ax=ax, squeeze=False,
                          sharex=sharex, sharey=sharey, figsize=figsize,
                          layout=layout)
    _axes = _flatten(axes)

    for i, col in enumerate(com.try_sort(data.columns)):
        ax = _axes[i]
        ax.hist(empty_pd_df[col].dropna().values, bins=bins, **kwds)
        ax.set_title(col)
        ax.grid(grid)

    _set_ticks_props(axes, xlabelsize=xlabelsize, xrot=xrot,
                     ylabelsize=ylabelsize, yrot=yrot)
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    return axes
