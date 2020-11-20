"""
Shock tube experiment planning and data analysis

Notes
-----
The standard shock tube region notation followed in PyShock's code and documentation is:

1. initial driven gas

2. post-incident-shock driven gas

3. expanded driver gas

4. initial driver gas

5. post-reflected-shock driven gas

"""

import cantera as ct
import numpy as np
from scipy import optimize

from knightshock import gas_dynamics


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

        self.u = None

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
            raise TypeError("Value must be of type cantera.ThermoPhase or a file path to a valid mechanism file")

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

    @property
    def M(self):
        """Mach number of incident shock wave"""
        return self.u / self.a1

    def calculate_shock_conditions(self):
        """Calculates `T2`, `P2`, `T5`, and `P5` for the experiment using
        `knightshock.gas_dynamics.shock_conditions_FROSH`"""

        self.thermo.X = self.driven_mixture
        self.T2, self.P2, self.T5, self.P5 = \
            gas_dynamics.shock_conditions_FROSH(self.T1, self.P1, self.u, thermo=self.thermo)

    @staticmethod
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


class ExperimentPlan:
    def __init__(self, *, T1=None, P1=None, T4=None, P4=None, T5=None, P5=None, gamma1=None, gamma4=None,
                 MW1=None, MW4=None, area_ratio=1, bracket=None):
        def calc_P5(M):
            P2 = gas_dynamics.normal_shock_pressure_ratio(M, gamma1) * P1
            M_reflected = gas_dynamics.reflected_shock_Mach_number(M, gamma1)
            return gas_dynamics.normal_shock_pressure_ratio(M_reflected, gamma1) * P2

        def calc_T5(M):
            T2 = gas_dynamics.normal_shock_temperature_ratio(M, gamma1) * T1
            M_reflected = gas_dynamics.reflected_shock_Mach_number(M, gamma1)
            return gas_dynamics.normal_shock_temperature_ratio(M_reflected, gamma1) * T2

        if not bracket:
            bracket = [1.01, 5]

        if P1 and P4 and T1 and T4 and gamma1 and gamma4 and MW1 and MW4:
            root_results = optimize.root_scalar(
                lambda M: P4 / P1 - gas_dynamics.shock_tube_flow_properties(M, T1, T4, MW1, MW4, gamma1, gamma4)[0],
                bracket=bracket)
        elif P1 and P5 and gamma1:
            root_results = optimize.root_scalar(lambda M: P5 - calc_P5(M), bracket=bracket)
        elif T1 and T5 and gamma1:
            root_results = optimize.root_scalar(lambda M: T5 - calc_T5(M), bracket=bracket)
        else:
            raise ValueError

        if not root_results.converged:
            raise RuntimeError

        self.M = root_results.root


