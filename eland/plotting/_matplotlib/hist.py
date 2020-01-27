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
from pandas.core.dtypes.generic import ABCIndexClass
from pandas.plotting._matplotlib import converter
from pandas.plotting._matplotlib.tools import _flatten, _set_ticks_props, _subplots


def hist_series(
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
        **kwds
):
    import matplotlib.pyplot as plt

    if by is None:
        if kwds.get("layout", None) is not None:
            raise ValueError(
                "The 'layout' keyword is not supported when " "'by' is None"
            )
        # hack until the plotting interface is a bit more unified
        fig = kwds.pop(
            "figure", plt.gcf() if plt.get_fignums() else plt.figure(figsize=figsize)
        )
        if figsize is not None and tuple(figsize) != tuple(fig.get_size_inches()):
            fig.set_size_inches(*figsize, forward=True)
        if ax is None:
            ax = fig.gca()
        elif ax.get_figure() != fig:
            raise AssertionError("passed axis not bound to passed figure")

        self_bins, self_weights = self._hist(num_bins=bins)
        # As this is a series, squeeze Series to arrays
        self_bins = self_bins.squeeze()
        self_weights = self_weights.squeeze()

        ax.hist(self_bins[:-1], bins=self_bins, weights=self_weights, **kwds)
        ax.grid(grid)
        axes = np.array([ax])

        _set_ticks_props(
            axes, xlabelsize=xlabelsize, xrot=xrot, ylabelsize=ylabelsize, yrot=yrot
        )

    else:
        raise NotImplementedError("TODO")

    if hasattr(axes, "ndim"):
        if axes.ndim == 1 and len(axes) == 1:
            return axes[0]
    return axes


def hist_frame(
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
        **kwds
):
    # Start with empty pandas data frame derived from
    ed_df_bins, ed_df_weights = data._hist(num_bins=bins)

    converter._WARN = False  # no warning for pandas plots
    if by is not None:
        raise NotImplementedError("TODO")

    if column is not None:
        if not isinstance(column, (list, np.ndarray, ABCIndexClass)):
            column = [column]
        ed_df_bins = ed_df_bins[column]
        ed_df_weights = ed_df_weights[column]
    naxes = len(ed_df_bins.columns)

    if naxes == 0:
        raise ValueError("hist method requires numerical columns, " "nothing to plot.")

    fig, axes = _subplots(
        naxes=naxes,
        ax=ax,
        squeeze=False,
        sharex=sharex,
        sharey=sharey,
        figsize=figsize,
        layout=layout,
    )
    _axes = _flatten(axes)

    for i, col in enumerate(com.try_sort(data.columns)):
        ax = _axes[i]
        ax.hist(ed_df_bins[col][:-1], bins=ed_df_bins[col], weights=ed_df_weights[col], **kwds)
        ax.set_title(col)
        ax.grid(grid)

    _set_ticks_props(
        axes, xlabelsize=xlabelsize, xrot=xrot, ylabelsize=ylabelsize, yrot=yrot
    )
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    return axes
