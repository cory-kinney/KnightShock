from typing import List, Tuple
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseEvent
from matplotlib.widgets import RectangleSelector


def find_feature(x: np.ndarray, y: np.ndarray, *, feature: str = 'max', guess: Tuple[float, float] = (None, None),
                 title: str = '', xlabel: str = '', ylabel: str = '') -> Tuple[float, float]:
    """Finds the specified feature in a user-selected region of plotted data

    Parameters
    ----------
    x: np.ndarray
        x values of feature data
    y: np.ndarray
        y values of feature data
    feature: str, optional
        method used for finding feature

        - 'max', default

        - 'min'

        - 'max slope'

    guess: Tuple[float, float], optional
        guess for feature x, y values
    title: str, optional
        title of plot
    xlabel: str, optional
        label for x axis
    ylabel: str, optional
        label for y axis

    Returns
    -------
    point: Tuple[float, float]
        feature x, y values

    """
    if feature not in {'max', 'min', 'max slope'}:
        raise ValueError(f"Invalid feature specified ('{feature}') - valid features are 'max', 'min', or 'max slope'")

    fig, ax = plt.subplots()

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.plot(x, y)
    sc = ax.scatter(*guess)

    point: Tuple[float, float] = guess

    def select_callback(click_event: MouseEvent, release_event: MouseEvent):
        nonlocal point

        x1, y1 = click_event.xdata, click_event.ydata
        x2, y2 = release_event.xdata, release_event.ydata

        try:
            region_indices = np.logical_and(np.logical_and(x >= x1, x <= x2), np.logical_and(y >= y1, y <= y2))

            if feature == 'max':
                feature_index = np.argmax(y[region_indices])
            elif feature == 'min':
                feature_index = np.argmin(y[region_indices])
            elif feature == 'max slope':
                slope = np.diff(y) / np.diff(x)
                feature_index = np.argmax(slope[region_indices[:-1]])

            point = x[region_indices][feature_index], y[region_indices][feature_index]

            sc.set_offsets(np.c_[point[0], point[1]])
            fig.canvas.draw()

        except ValueError:
            pass

    selector = RectangleSelector(ax, select_callback, useblit=True, drawtype='box',
                                 lineprops=dict(color='black', linestyle='-', alpha=0.5),
                                 rectprops=dict(color='gray', alpha=0.2, fill=True))
    plt.show()

    return point
