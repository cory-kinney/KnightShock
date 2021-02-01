"""Shock tube experiment planning and data analysis package"""

from knightshock.gasdynamics import frozen_shock_conditions, shock_tube_flow_properties
from typing import Dict, Iterable, Union, Tuple
from scipy.optimize import root_scalar
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
    if dt.size != x.size - 1:
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
        """Define the shock tube state from experimental data

        Parameters
        ----------
        T1: float
            initial driven temperature (K)
        P1: float
            initial driven pressure (Pa)
        X1: str, Dict[str, float], or numpy.ndarray
            initial driven mixture
        T4: float
            initial driver temperature (K)
        P4: float
            initial driver pressure (Pa)
        X4: str, Dict[str, float], or numpy.ndarray
            initial driver mixture
        u: float
            incident shock velocity at end wall (m/s)
        mechanism: str
            file path to mechanism
        method: str, optional
            method used for FROSH (default = 'EE')

        """
        solution = ct.Solution(mechanism)

        region1 = ShockTubeState.Region(T1, P1, X1, solution)
        region4 = ShockTubeState.Region(T4, P4, X4, solution)

        T2, P2, T5, P5 = frozen_shock_conditions(u / region1.a, region1.thermo, method)
        region2 = ShockTubeState.Region(T2, P2, X1, solution)
        region5 = ShockTubeState.Region(T5, P5, X1, solution)

        return cls(region1, region2, region4, region5)


def M_from_P4_P1(P4, P1, T4, T1, X4, X1, gas, *, area_ratio=1, bracket=None):
    """Predicts incident shock wave Mach number from initial conditions

    Parameters
    ----------
    P4: float
        driver pressure (Pa)
    P1: float
        driven pressure (Pa)
    T4: float
        driver temperature (K)
    T1: float
        driven temperature (K)
    X4: str, Dict[str, float], or numpy.ndarray
        driver mixture
    X1: str, Dict[str, float], or numpy.ndarray
        driven mixture
    gas: ct.ThermoPhase or str
        thermodynamic property object
    area_ratio: float, optional
        ratio of the area of driver to driven (default = 1)
    bracket: Tuple[float, float], optional
        bracket for Mach number iterative solver (default = 1.001, 5)

    Returns
    -------
    M: float
        incident shock wave Mach number

    """
    if not isinstance(gas, ct.ThermoPhase):
        gas = ct.ThermoPhase(gas)

    if bracket is None:
        bracket = [1.001, 5]

    gas.TPX = T4, P4, X4
    gamma4 = gas.cp / gas.cv
    MW4 = gas.mean_molecular_weight

    gas.TPX = T1, P1, X1
    gamma1 = gas.cp / gas.cv
    MW1 = gas.mean_molecular_weight

    def P4_P1_from_M(M):
        return shock_tube_flow_properties(M, T1, T4, MW1, MW4, gamma1, gamma4, area_ratio=area_ratio)[0]

    root_results = root_scalar(lambda M: P4 / P1 - P4_P1_from_M(M), bracket=bracket)
    if not root_results.converged:
        raise RuntimeError

    return root_results.root


def M_from_P5_P1(P5, P1, T1, X1, gas, *, method=None, bracket=None):
    """Predicts incident shock wave Mach number from initial driven pressure and post reflected shock pressure

    Parameters
    ----------
    P5: float
        post reflected shock pressure (Pa)
    P1: float
        initial pressure (Pa)
    T1: float
        initial temperature (K)
    X1: str, Dict[str, float], or numpy.ndarray
        initial mixture
    gas: ct.ThermoPhase or str
        thermodynamic property object
    method: str, optional
        method for FROSH
    bracket: Tuple[float, float], optional
        bracket for Mach number iterative solver (default = 1.001, 5)

    Returns
    -------
    M: float
        incident shock wave Mach number

    """
    if not isinstance(gas, ct.ThermoPhase):
        gas = ct.ThermoPhase(gas)

    if bracket is None:
        bracket = [1.001, 5]

    gas.X1 = X1

    def P5_from_M(M):
        gas.TP = T1, P1
        return frozen_shock_conditions(M, gas, method=method)[3]

    root_results = root_scalar(lambda M: P5 - P5_from_M(M), bracket=bracket)
    if not root_results.converged:
        raise RuntimeError

    return root_results.root


def M_from_T5_T1(T5, T1, P1, X1, gas, *, method=None, bracket=None):
    """Predicts incident shock wave Mach number from initial driven temperature and post reflected shock temperature

    Parameters
    ----------
    T5: float
        post reflected shock temperature (K)
    T1: float
        initial temperature (K)
    P1: float
        initial pressure (Pa)
    X1: str, Dict[str, float], or numpy.ndarray
        initial mixture
    gas: ct.ThermoPhase or str
        thermodynamic property object
    method: str, optional
        method for FROSH
    bracket: Tuple[float, float], optional
        bracket for Mach number iterative solver (default = 1.001, 5)

    Returns
    -------
    M: float
        incident shock wave Mach number

    """
    if not isinstance(gas, ct.ThermoPhase):
        gas = ct.ThermoPhase(gas)

    if bracket is None:
        bracket = [1.001, 5]

    gas.X = X1

    def T5_from_M(M):
        gas.TP = T1, P1
        return frozen_shock_conditions(M, gas, method=method)[2]

    root_results = root_scalar(lambda M: T5 - T5_from_M(M), bracket=bracket)
    if not root_results.converged:
        raise RuntimeError

    return root_results.root
