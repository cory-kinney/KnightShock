"""Shock tube experiment data analysis package"""

import cantera as ct
import numpy as np
import pyshock


class Experiment:
    """Shock tube experiment base class

    Attributes
    ----------
    T1 : float
        initial temperature [K] of the driven gas
    P1 : float
        initial pressure [Pa] of the driven gas
    T4 : float
        initial temperature [K] of the driver gas
    P4 : float
        initial pressure [Pa] of the driver gas
    driven_mixture : str
        Mixture composition of driven gas as comma-separated species: mole fraction pairs
    driver_mixture : str
        Mixture composition of driver gas as comma-separated species: mole fraction pairs
    T2 : float
        temperature [K] of the driven gas after the incident shock wave
    P2 : float
        pressure [Pa] of the driven gas after the incident shock wave
    T5 : float
        temperature [K] of the driver gas after the reflected shock wave
    P5 : float
        pressure [Pa] of the driver gas after the reflected shock wave

    """

    def __init__(self):
        self.T1 = None
        self.P1 = None
        self.T4 = None
        self.P4 = None

        self.driven_mixture = None
        self.driver_mixture = None

        self.T2 = None
        self.P2 = None
        self.T5 = None
        self.P5 = None

        self._thermo = None

    @property
    def thermo(self):
        """
        `cantera.ThermoPhase` object for temperature- and pressure-dependent thermodynamic property calculations

        .. Attention::
           Temperature, pressure, and mixture composition (`TPX`) of `cantera.ThermoPhase` object must be updated before
           accessing thermodynamic properties

        """
        return self._thermo

    @thermo.setter
    def thermo(self, value):
        if isinstance(value, ct.ThermoPhase):
            self._thermo = value
        elif isinstance(value, str):
            try:
                self._thermo = ct.ThermoPhase(value)
            except ct.CanteraError:
                raise ValueError("Invalid mechanism file")
        else:
            raise TypeError("Input must be a Cantera ThermoPhase object or file path to a valid mechanism file")

    @property
    def gamma1(self):
        """Specific heat ratio of driven gas at initial conditions"""
        self.thermo.TPX = self.T1, self.P1, self.driven_mixture
        return self.thermo.cp / self.thermo.cv

    @property
    def gamma4(self):
        """Specific heat ratio of driver gas at initial conditions"""
        self.thermo.TPX = self.T4, self.P4, self.driver_mixture
        return self.thermo.cp / self.thermo.cv

    @property
    def a1(self):
        """Speed of sound [m/s] in driven gas at initial conditions"""
        self.thermo.TPX = self.T1, self.P1, self.driven_mixture
        return (self.thermo.cp / self.thermo.cv * ct.gas_constant * self.T1 / self.thermo.mean_molecular_weight) ** 0.5

    @property
    def a4(self):
        """Speed of sound [m/s] in driver gas at initial conditions"""
        self.thermo.TPX = self.T4, self.P4, self.driver_mixture
        return (self.thermo.cp / self.thermo.cv * ct.gas_constant * self.T4 / self.thermo.mean_molecular_weight) ** 0.5

    def calculate_shock_conditions(self, u):
        """Calculates `T2`, `P2`, `T5`, and `P5` for the experiment using `pyshock.FROSH`

        Parameters
        ----------
        u : float
            incident shock wave velocity [m/s]

        """

        self.thermo.X = self.driven_mixture
        self.T2, self.P2, self.T5, self.P5 = pyshock.FROSH(self.T1, self.P1, u, thermo=self.thermo)


def calculate_shock_velocity(x, dt):
    """Computes the least-squares linear fit of average shock velocities over intervals

    Parameters
    ----------
    x : numpy.ndarray
        positions relative to end wall
    dt : numpy.ndarray
        differences in shock wave arrival times over intervals

    Returns
    -------
    u : float
        shock velocity extrapolated to end wall
    attenuation : float
        deceleration of the shock relative to end wall velocity
    r2 : float
        coefficient of determination of the linear fit

    Raises
    ------
    ValueError
        `x` values are not greater than zero
    ValueError
        `x` values are not strictly increasing

    """

    if np.any(x < 0):
        raise ValueError("x values must be positive")
    if not np.all(x[1:] > x[:-1]):
        raise ValueError("x values must be strictly increasing")

    x_midpoint = (x[1:] + x[:-1]) / 2
    u_avg = np.abs(np.diff(x)) / dt

    A = np.vstack([x_midpoint, np.ones(len(x_midpoint))]).T
    model, residual = np.linalg.lstsq(A, u_avg, rcond=None)[:2]
    r2 = (1 - residual / (u_avg.size * u_avg.var()))[0]

    u = model[1]
    attenuation = model[0] / model[1]

    return u, attenuation, r2

