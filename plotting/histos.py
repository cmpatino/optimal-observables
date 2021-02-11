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


def hist_var(q: List[float], ax: plt.Axes, **kwargs) -> plt.Axes:
    """Create histogram with error bars.

    :param q: Values to create histogram.
    :type q: List[float]
    :param ax: Axes in which histrogram is plotted.
    :type ax: plt.Axes
    :return: Axes with histogram.
    :rtype: plt.Axes
    """

    bins, edges, _ = ax.hist(
        q,
        alpha=0.6,
        histtype="step",
        align="left",
        linewidth=4,
        **kwargs
    )
    errors = np.sqrt(bins)
    bin_width = edges[1] - edges[0]

    ax.bar(
        x=edges[:-1],
        bottom=bins,
        height=errors,
        width=bin_width,
        alpha=0.0,
        color='w',
        hatch='/'
    )
    ax.bar(
        x=edges[:-1],
        bottom=bins,
        height=-errors,
        width=bin_width,
        alpha=0.0,
        color='w',
        hatch='/'
    )
    return ax


def ratio_hist(processes_q: List[List[float]], hist_labels: List[str],
               reference_label: str, n_bins: int, hist_range: Tuple[int, int],
               title: str, figsize=(15, 8)) -> Figure:
    """Generate histrograms with ratio pad

    :param processes_q: Quantity for each event and process
    :type processes_q: List[List[float]]
    :param hist_labels: Labels for each process
    :type hist_labels: List[str]
    :param reference_label: Label of process taken as the denominator of ratios
    :type reference_label: str
    :param n_bins: Number of bins for histograms
    :type n_bins: int
    :param hist_range: Range for histogram bins
    :type hist_range: Tuple[int]
    :param title: Plot title
    :type title: str
    :param figsize: Figure size for output plot
    :type figsize: Tuple[int]
    :return: Figure with histogram and histogram ratio.
    :rtype: Figure
    """

    fig, ax = plt.subplots(
            nrows=len(processes_q),
            ncols=1,
            gridspec_kw={'height_ratios': [3] + [1]*(len(processes_q) - 1)},
            sharex=True,
            figsize=figsize
    )
    legends = []

    p_bins = {}
    p_edges = {}
    p_errors = {}
    for p, label in zip(processes_q, hist_labels):
        bins, edges, _ = ax[0].hist(
            x=p,
            bins=n_bins,
            range=hist_range,
            fill=False,
            label=label,
            align='left',
            histtype='step',
            linewidth=4
        )
        p_bins[label] = bins
        p_edges[label] = edges
        p_errors[label] = np.sqrt(bins)
        legends += [label]

    bin_width = edges[1] - edges[0]

    for label in hist_labels:
        ax[0].bar(
            x=p_edges[label][:-1],
            bottom=p_bins[label],
            height=p_errors[label],
            width=bin_width,
            alpha=0.0,
            color='w',
            hatch='/'
        )
        ax[0].bar(
            x=p_edges[label][:-1],
            bottom=p_bins[label],
            height=-p_errors[label],
            width=bin_width,
            alpha=0.0,
            color='w',
            hatch='/'
        )
        legends += ['_', '_']

    ax[0].set_ylabel("Events", fontsize=15)
    ax[0].set_title(title, fontsize=20)
    legends[-1] = "Stat. Uncertainty"
    ax[0].legend(legends)

    plot_idx = 1
    ref_bins = p_bins[reference_label]
    ref_edges = p_edges[reference_label]
    ref_frac_error = p_errors[reference_label]/ref_bins
    for label in hist_labels:
        if label == reference_label:
            continue
        ratios = p_bins[label]/ref_bins
        error_ratio = ratios * (ref_frac_error + p_errors[label]/p_bins[label])

        ax[plot_idx].bar(
            bottom=1.0,
            height=error_ratio,
            x=ref_edges[:-1],
            width=bin_width,
            alpha=0.3,
            color="blue"
        )
        ax[plot_idx].bar(
            bottom=1.0,
            height=-error_ratio,
            x=ref_edges[:-1],
            width=bin_width,
            alpha=0.3,
            color="blue"
        )
        ax[plot_idx].scatter(ref_edges[:-1], ratios, marker='o', color="black")
        ax[plot_idx].set_ylabel(f"{label}/{reference_label}")
        plot_idx += 1
