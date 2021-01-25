"""Shock tube experiment planning and data analysis package"""

from knightshock.gasdynamics import frozen_shock_conditions
from typing import Dict, Iterable, Union, Tuple
import cantera as ct
import numpy as np


Mixture = Union[str, Dict[str, float], np.ndarray]


def shock_velocity_from_ToF(x: np.ndarray, dt: np.ndarray) -> Tuple[float, float, float]:
    """Calculate the velocity and attenuation of the incident shock as a linear fit of the average velocities over a set
    of measurement intervals calculated by time of flight

    Parameters
    ----------
    x: numpy.ndarray
        measurement positions relative to end wall
    dt: numpy.ndarray
        time of flight across each measurement interval

    Returns
    -------
    u0: float
        velocity of the shock at the end wall
    attenuation: float
        change in shock velocity with respect to distance
    r2: float
        coefficient of determination of the linear fit

    """

    if x.ndim != 1:
        raise ValueError("Dimension of x (ndim = {}) must be 1".format(x.ndim))
    if dt.ndim != 1:
        raise ValueError("Dimension of dt array (ndim = {}) must be 1".format(dt.ndim))
    if dt.shape[0] != x.shape[0] - 1:
        raise ValueError("Size of dt array (size = {}) must be 1 less than x array (size = {})".format(dt.size, x.size))

    x = np.abs(x)
    if not np.all(x[:-1] < x[1:]):
        raise ValueError("Elements of x must be monotonic")

    x_midpoint = (x[1:] + x[:-1]) / 2
    u_average = np.diff(x) / dt

    A = np.vstack([x_midpoint, np.ones(x_midpoint.size)]).T
    model, residual = np.linalg.lstsq(A, u_average, rcond=None)[:2]

    u0: float = model[1]
    attenuation: float = model[0]
    r2: float = (1 - residual / (u_average.size * u_average.var()))[0]

    return u0, attenuation, r2


def ToF_from_pressure_series(t: np.ndarray, pressure_series: Iterable[np.ndarray], threshold: float) -> np.ndarray:
    """Calculates the time of flight across a measurement interval from the intersection of pressure series with a
    threshold value

    Parameters
    ----------
    t: numpy.ndarray
        time series
    pressure_series: iterable of numpy.ndarray
        list of pressure series at measurement positions for given time series
    threshold: float
        pressure value used to measure arrival time

    Returns
    -------
    dt: numpy.ndarray
        time of flight over each measurement interval

    """

    if t.ndim != 1:
        raise ValueError("Dimension of t (ndim = {}) must be 1".format(t.ndim))
    if any([P.ndim != 1 for P in pressure_series]):
        raise ValueError("Dimension of all pressure series must be 1")
    if any([P.size != t.size for P in pressure_series]):
        raise ValueError("Size of all pressure series must be the same as t (size = {})".format(t.size))

    def interpolation(x, x0, x1, y0, y1):
        return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

    indices = [np.argmax(P > threshold) for P in pressure_series]

    if any([i == -1 for i in indices]):
        raise ValueError("Threshold value does not intersect all pressure traces")

    dt = np.diff(np.array([interpolation(threshold, P[i - 1], P[i], t[i - 1], t[i])
                           for i, P in zip(indices, pressure_series)]))

    return dt


class ShockTubeState:
    """ """

    class Region:
        """ """

        def __init__(self, T: float, P: float, X: Mixture, thermo: ct.ThermoPhase):
            # Validate inputs
            if not isinstance(thermo, ct.ThermoPhase):
                raise TypeError
            self._thermo.TPX = T, P, X

            self._T = T
            self._P = P
            self._X = X
            self._thermo = thermo

        @property
        def thermo(self) -> ct.ThermoPhase:
            self._thermo.TPX = self.T, self.P, self.X
            return self._thermo

        @property
        def T(self) -> float:
            return self._T

        @property
        def P(self) -> float:
            return self._P

        @property
        def X(self) -> Mixture:
            return self._X

        @property
        def MW(self) -> float:
            return self.thermo.mean_molecular_weight

        @property
        def gamma(self) -> float:
            return self.thermo.cp / self.thermo.cv

        @property
        def a(self) -> float:
            return (self.gamma * ct.gas_constant / self.MW * self.T) ** 0.5

    def __init__(self, region1: Region, region2: Region, region4: Region, region5: Region):
        self._regions = {
            1: region1,
            2: region2,
            4: region4,
            5: region5
        }

    def __getitem__(self, region_num: int) -> Region:
        return self._regions[region_num]

    @classmethod
    def from_experiment(cls, T1: float, P1: float, X1: Mixture, T4: float, P4: float, X4: Mixture, u: float,
                        mechanism: str, *, method: str = 'EE'):
        solution = ct.Solution(mechanism)

        region1 = ShockTubeState.Region(T1, P1, X1, solution)
        region4 = ShockTubeState.Region(T4, P4, X4, solution)

        (T2, P2), (T5, P5) = frozen_shock_conditions(u / region1.a, region1.thermo, method)
        region2 = ShockTubeState.Region(T2, P2, X1, solution)
        region5 = ShockTubeState.Region(T5, P5, X1, solution)

        return cls(region1, region2, region4, region5)

