#
# Author: Kevin Ingles
# File: my_plotting.py
# Description: User defined functions to facilitate plotting routines

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
rc('text', usetex=True)


def get_cmap(n: int, name: str = 'hsv'):
    '''
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.
    '''
    return plt.cm.get_cmap(name, n)


def costumize_axis(ax: plt.Axes, x_title: str, y_title: str):
    ax.set_xlabel(x_title, fontsize=24)
    ax.set_ylabel(y_title, fontsize=24)
    ax.tick_params(axis='both', labelsize=18, top=True, right=True)
    ax.tick_params(axis='both', which='major', direction='in', length=8)
    ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
    ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
    ax.tick_params(axis='both', which='minor',
                   direction='in', length=4, top=True, right=True)
    return ax


def autoscale_y(ax, margin=0.1):
    """
    This function rescales the y-axis based on the data that is visible given 
    the current xlim of the axis.\n
    Parameters:
    --------------
    ax: a matplotlib axes object
    margin:  the fraction of the total height of the y-data to pad the upper and lower ylims

    Returns:
    --------------
    None
    """

    def get_bottom_top(line):
        xd = line.get_xdata()
        yd = line.get_ydata()
        lo, hi = ax.get_xlim()
        y_displayed = yd[((xd > lo) & (xd < hi))]
        h = np.max(y_displayed) - np.min(y_displayed)
        bot = np.min(y_displayed)-margin*h
        top = np.max(y_displayed)+margin*h
        return bot, top

    lines = ax.get_lines()
    if len(lines) == 0:
        print('No lines in plot, leaving plot unchanged')
        return
    bot, top = np.inf, -np.inf

    for line in lines:
        new_bot, new_top = get_bottom_top(line)
        if new_bot < bot:
            bot = new_bot
        if new_top > top:
            top = new_top

    if np.isinf(bot) or np.isinf(top):
        print("inf encoutner, leaving y-axis unchanged: check for error")
        return
    ax.set_ylim(bot, top)


# def smooth_histogram(counts: np.ndarray,
#                      window_size: int) -> np.ndarray:
#     # Convert to cubic splice
#     new_counts = np.zeros_like(counts)
#     mid = int(window_size / 2)
#     for i in range(counts.size):
#         if i < mid or i > counts.size - mid - 1:
#             new_counts[i] = counts[i]
#         else:
#             mean = 0
#             for j in range(-mid, mid):
#                 mean += counts[i + j]
#             mean = mean / window_size
#             new_counts[i] = mean
#     return new_counts

def smooth_histogram(x: np.ndarray,
                     y: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.interpolate import CubicSpline

    low, high = x[0], x[-1]
    xs = np.linspace(low, high, x.size * 2)
    cs = CubicSpline(x, y)
    return xs, cs(xs)
