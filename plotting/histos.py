import numpy as np
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from typing import List, Tuple


def hist_n_particles(q: List[int], label: str) -> Figure:
    """Generate histogram for particle counts on events.

    :param q: Count per event.
    :type q: List[int]
    :param label: Plot title.
    :type label: str
    :return: Figure with histogram and histogram ratio.
    :rtype: Figure
    """

    fig, ax = plt.subplots(
        nrows=1,
        figsize=(15, 8)
    )

    bins, edges = np.histogram(q, bins=15, range=(0, 15))

    for idx, val in enumerate(bins[::-1]):
        if val > 0:
            max_idx = len(bins) - idx - 1
            break

    edges = edges[:max_idx]
    bins = bins[:max_idx]

    ax.bar(edges, bins, width=1, alpha=0.6)
    ax.set_title(label, fontsize=20, y=1.04)
    ax.set_xticks(edges)

    return fig


def hist_var(q: List[float], label: str) -> Figure:
    """Generate histogram for continuous quantity.

    :param q: Quantity value per event.
    :type q: List[float]
    :param label: Plot title.
    :type label: str
    :return: Figure with histogram and histogram ratio.
    :rtype: Figure
    """

    fig, ax = plt.subplots(
        nrows=1,
        figsize=(15, 8)
    )

    bins, edges, _ = ax.hist(q, alpha=0.6, label=label, bins=20)
    ax.set_title(label, fontsize=20, y=1.04)

    return fig


def ratio_hist(q1: List[float], q2: List[float], bins: int,
               hist_range: Tuple[int], label: str,
               hist1_label: str, hist2_label: str) -> Figure:
    """Generate histogram with ratio pad.

    :param q1: Quantity value per event.
    :type q1: List[float]
    :param q2: Quantity value per event.
    :type q2: List[float]
    :param bins: Number of bins for histogram.
    :type bins: int
    :param hist_range: Range for histogram bins.
    :type hist_range: Tuple[int]
    :param label: Plot title.
    :type label: str
    :param hist1_label: Label for q1.
    :type hist1_label: str
    :param hist2_label: Label for q2.
    :type hist2_label: str
    :return: Figure with histogram and histogram ratio.
    :rtype: Figure
    """

    bins1, edges1 = np.histogram(
        q1,
        bins=bins,
        range=hist_range
    )

    bins2, edges2 = np.histogram(
        q2,
        bins=bins,
        range=hist_range
    )
    bin_width = edges1[1] - edges1[0]

    error1 = np.sqrt(bins1)
    error2 = np.sqrt(bins2)
    frac_error = error1/bins1 + error2/bins2

    fig, (ax1, ax2) = plt.subplots(
            nrows=2,
            gridspec_kw={'height_ratios': [3, 1]},
            sharex=True,
            figsize=(15, 8)
    )

    ax1.bar(edges1[:-1], bins1, width=bin_width, alpha=0.6, label=hist1_label)
    ax1.bar(
        x=edges1[:-1],
        bottom=bins1,
        height=error1,
        width=bin_width,
        alpha=0.0,
        color='w',
        hatch='/',
        label='Stat. Uncertainty'
    )
    ax1.bar(
        x=edges1[:-1],
        bottom=bins1,
        height=-error1,
        width=bin_width,
        alpha=0.0,
        color='w',
        hatch='/'
    )

    ax1.bar(edges2[:-1], bins2, width=bin_width, alpha=0.6, label=hist2_label)
    ax1.bar(
        x=edges2[:-1],
        bottom=bins2,
        height=error2,
        width=bin_width,
        alpha=0.0,
        color='w',
        hatch='/'
    )
    ax1.bar(
        x=edges2[:-1],
        bottom=bins2,
        height=-error2,
        width=bin_width,
        alpha=0.0,
        color='w',
        hatch='/'
    )

    ratios = bins1/bins2
    error_ratio = ratios * frac_error
    ax2.bar(
        bottom=1.0,
        height=error_ratio,
        x=edges1[:-1],
        width=bin_width,
        alpha=0.5,
        color="blue"
    )
    ax2.bar(
        bottom=1.0,
        height=-error_ratio,
        x=edges1[:-1],
        width=bin_width,
        alpha=0.5,
        color="blue"
    )
    _ = ax2.scatter(edges1[:-1], ratios, marker='o', color="black")
    ax1.legend()
    ax1.set_title(label, fontsize=20)
    ax1.set_ylabel("Events", fontsize=15)
    ax2.set_ylabel(f"{hist1_label}/{hist2_label}")
