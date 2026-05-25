from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


__all__ = ["feature_summary", "plot_feature"]


# some functions for summarising the relationship between a continuous feature and a target variable
def feature_summary(data, feature, target, bins=10, for_plotting=False):
    """
    Summarizes the relationship between a continuous feature and a target variable by binning the feature
    and calculating statistics for each bin.
    """
    data = data.copy()

    # Create new column to define the bins of the feature
    if isinstance(bins, Iterable):
        nbins = len(bins) - 1
        data[f"{feature}_{nbins}b"] = pd.cut(data[feature], bins=bins, include_lowest=True)
    else:
        nbins = bins
        data[f"{feature}_{nbins}b"], bins = pd.qcut(data[feature], q=bins, retbins=True, duplicates="drop")

    # Group by the bins and calculate the mean of the target for each bin
    agg_dict = {
        "count": (target, "count"),
        f"{target}_mean": (target, "mean"),
    }
    summary = data.groupby(f"{feature}_{nbins}b", observed=True).agg(**agg_dict)

    # produce useful plotting metrics if requested
    if for_plotting:
        summary["bin_start"] = bins[:-1]
        summary["bin_end"] = bins[1:]
        summary["bin_mid"] = (bins[:-1] + bins[1:]) / 2
        summary["bin_width"] = bins[1:] - bins[:-1]
    return summary.reset_index()


def plot_feature(summary, feature, target_cols, ax=None):
    """
    Plots the relationship between a continuous feature and a target variable by creating a bar plot
    of the frequency density of the feature and a line plot of the mean target value for each bin of the feature.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.bar(
        summary["bin_mid"],
        summary["count"] / summary["bin_width"],  # plot frequency density
        width=summary["bin_width"],
        alpha=0.2,
        edgecolor="black",
        color="k",
    )  # plot count

    ax2 = ax.twinx()

    for target_col in target_cols:
        ax2.plot(summary["bin_mid"], summary[target_col], label=target_col)  # plot target mean

    ax.set_xlabel(feature)
    ax.set_ylabel("Frequency density")
    ax2.set_ylabel("Target Mean")

    ax2.legend()
    plt.close(fig)  # close the figure to prevent it from displaying
    return fig