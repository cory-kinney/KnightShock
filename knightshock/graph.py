from matplotlib import pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.ticker import AutoMinorLocator
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.backend_bases import MouseEvent

from typing import Callable
import numpy as np


def data_selector(axes: Axes, line: Line2D, *, callback: Callable[[np.ndarray, np.ndarray], None], **kwargs)\
        -> RectangleSelector:
    """Passes data contained inside a user-selected graphical region to a callback function

    Parameters
    ----------
    axes: matplotlib.axes.Axes
        axes on which data is plotted
    line: matplotlib.lines.Line2D
        plotted data from which to select
    callback: Callable[[numpy.ndarray, numpy.ndarray], None]
        function to pass the selected x, y data to

    Returns
    -------
    RectangleSelector
        reference to the RectangleSelector used to select the data which is returned to avoid garbage collection

    """

    def select_callback(click_event: MouseEvent, release_event: MouseEvent):
        x1, y1 = click_event.xdata, click_event.ydata
        x2, y2 = release_event.xdata, release_event.ydata

        try:
            x, y = line.get_xdata(), line.get_ydata()
            data_indices = np.logical_and(np.logical_and(x >= x1, x <= x2), np.logical_and(y >= y1, y <= y2))
            callback(x[data_indices], y[data_indices])

        except ValueError:
            pass

    return RectangleSelector(axes, select_callback, useblit=True, drawtype='box', **kwargs)


def time_zero(t: np.ndarray, *, P: np.ndarray = None, schlieren: np.ndarray = None):
    """Returns the time zero as defined by either the maximum slope of the pressure or the minimum of the schlieren in a
    region graphically selected by the user

    Parameters
    ----------
    t: np.ndarray
        time array
    P: np.ndarray, optional
        pressure array
    schlieren: np.ndarray, optional
        schlieren array

    Returns
    -------
    t0: float
        time zero offset for initial t array

    """

    if P is None and schlieren is None:
        raise ValueError("Either pressure or schlieren data must be given")
    elif P is not None and schlieren is not None:
        figure, (axes_P, axes_schlieren) = plt.subplots(2, 1, sharex='col')
        axes_schlieren.set_xlabel("Time")
    else:
        figure, axes = plt.subplots()
        if P is not None:
            axes_P = axes
            axes_schlieren = None
            axes_P.set_xlabel("Time")
        else:
            axes_P = None
            axes_schlieren = axes
            axes_schlieren.set_xlabel("Time")

    figure.suptitle('Time Zero')
    plt.tight_layout()

    t0 = 0

    def update_t0(value):
        nonlocal t0
        t0 += value

        if axes_P:
            line_P.set_xdata(t - t0)

        if axes_schlieren:
            line_schlieren.set_xdata(t - t0)
            axes_schlieren.set_xlim([x - value for x in axes_schlieren.get_xlim()])
        else:
            axes_P.set_xlim([x - value for x in axes_P.get_xlim()])

        figure.canvas.draw()

    if axes_P:
        axes_P.set_ylabel("Pressure")
        axes_P.set_yticks([])
        axes_P.set_yticklabels([])
        axes_P.xaxis.set_minor_locator(AutoMinorLocator())
        axes_P.tick_params(which='both', direction='in')
        axes_P.axvline(0, color='r', linestyle='--', alpha=0.5, zorder=0)

        line_P, = axes_P.plot(t, P, color='k')
        axes_P.set_xlim([np.min(t), np.max(t)])
        selector_P = data_selector(axes_P, line_P,
                                   callback=lambda x, y: update_t0(x[np.argmax(np.diff(y) / np.diff(x))]))

    if axes_schlieren:
        axes_schlieren.set_ylabel("Schlieren")
        axes_schlieren.set_yticks([])
        axes_schlieren.set_yticklabels([])
        axes_schlieren.xaxis.set_minor_locator(AutoMinorLocator())
        axes_schlieren.tick_params(which='both', direction='in')
        axes_schlieren.axvline(0, color='r', linestyle='--', alpha=0.5, zorder=0)

        line_schlieren, = axes_schlieren.plot(t, schlieren, color='k')
        axes_schlieren.set_xlim([np.min(t), np.max(t)])
        selector_schlieren = data_selector(axes_schlieren, line_schlieren,
                                           callback=lambda x, y: update_t0(x[np.argmin(y)]))

    plt.show()
    return t0
