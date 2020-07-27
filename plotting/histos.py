import numpy as np
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from typing import List


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
