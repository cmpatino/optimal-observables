import numpy as np
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from typing import List


def hist_n_particles(q1: List[int], q2: List[int], label: str,
                     hist1_label: str, hist2_label: str) -> Figure:
    """Generate histogram with ratio pad for particle counts on events.

    :param q1: Count per event.
    :type q1: List[int]
    :param q2: Count per event.
    :type q2: List[float]
    :param label: Plot title.
    :type label: str
    :param hist1_label: Label for q1.
    :type hist1_label: str
    :param hist2_label: Label for q2.
    :type hist2_label: str
    :return: Figure with histogram and histogram ratio.
    :rtype: Figure
    """

    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True,
        figsize=(15, 8)
    )

    bins1, edges1 = np.histogram(q1, bins=15, range=(0, 15))
    bins2, edges2 = np.histogram(q2, bins=15, range=(0, 15))

    for idx, val in enumerate(bins1[::-1]):
        if val > 0:
            max_idx1 = len(bins1) - idx - 1
            break
    for idx, val in enumerate(bins2[::-1]):
        if val > 0:
            max_idx2 = len(bins2) - idx - 1
            break

    max_idx = max(max_idx1, max_idx2)
    edges = edges1[:max_idx]
    bins1 = bins1[:max_idx]
    bins2 = bins2[:max_idx]

    ax1.bar(edges, bins1, width=1, alpha=0.6, label=hist1_label)
    ax1.bar(edges, bins2, width=1, alpha=0.6, label=hist2_label)
    ax1.set_title(label, fontsize=20, y=1.04)
    ax1.set_xticks(edges)
    ax1.legend(fontsize=20)

    ratios = bins1/bins2
    _ = ax2.plot(edges, ratios, marker='o')
    ax2.set_ylabel(f"{hist1_label}/{hist2_label}", fontsize=20)

    return fig


def hist_var(q1: List[float], q2: List[float], label: str,
             hist1_label: str, hist2_label: str) -> Figure:
    """Generate histogram with ratio pad for continuous quantity.

    :param q1: Quantity value per event.
    :type q1: List[float]
    :param q2: Quantity value per event.
    :type q2: List[float]
    :param label: Plot title.
    :type label: str
    :param hist1_label: Label for q1.
    :type hist1_label: str
    :param hist2_label: Label for q2.
    :type hist2_label: str
    :return: Figure with histogram and histogram ratio.
    :rtype: Figure
    """

    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True,
        figsize=(15, 8)
    )

    bins1, edges1, _ = ax1.hist(q1, alpha=0.6, label=hist1_label, bins=20)
    bins2, edges2, _ = ax1.hist(q2, alpha=0.6, label=hist2_label, bins=20)
    ax1.set_title(label, fontsize=20, y=1.04)
    ax1.legend(fontsize=20)

    ratios = bins1/bins2
    _ = ax2.plot(edges1[:-1], ratios, marker='o')
    ax2.set_ylabel(f"{hist1_label}/{hist2_label}", fontsize=20)

    return fig
