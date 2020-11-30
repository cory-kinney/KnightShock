"""
Shock tube experiment planning and data analysis

Notes
-----
The standard shock tube region notation followed in KnightShock's code and documentation is:

1. initial driven gas

2. post-incident-shock driven gas

3. expanded driver gas

4. initial driver gas

5. post-reflected-shock driven gas

"""

import warnings
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
            `x` values are not strictly decreasing

        """

        if np.any(x < 0):
            raise ValueError("x values must be positive")
        if not np.all(x[1:] < x[:-1]):
            raise ValueError("x values must be strictly decreasing")

        x_midpoint = (x[1:] + x[:-1]) / 2
        u_avg = np.abs(np.diff(x)) / dt

        A = np.vstack([x_midpoint, np.ones(len(x_midpoint))]).T
        model, residual = np.linalg.lstsq(A, u_avg, rcond=None)[:2]
        r2 = (1 - residual / (u_avg.size * u_avg.var()))[0]

        u = model[1]
        attenuation = model[0] / model[1]

        return u, attenuation, r2


class ExperimentPlan:
    """Class for planning experiments with different combinations of initial conditions and target shock conditions"""

    def __init__(self):
        self.T1 = None
        self.T2 = None
        self.T4 = None
        self.T5 = None

        self.P1 = None
        self.P2 = None
        self.P4 = None
        self.P5 = None

        self.MW1 = None
        self.MW4 = None
        self.gamma1 = None
        self.gamma4 = None

        self.M = None

        self.area_ratio = 1
        self.constant_properties = True  # Not implemented
        self.bracket = [1.01, 5]

    def solve(self):
        """Attempts to solve for missing values that can be calculated from defined parameters."""

        if not (self.T4 is not None and self.MW1 is not None and self.MW4 is not None and self.gamma1 is not None
                and self.gamma4 is not None):
            raise ValueError("T4, MW1, MW4, gamma1, gamma4 must be specified")

        solved = False

        while solved is False:
            if self.M is None:
                try:
                    self.M = self._solve_M()
                    continue
                except ValueError:
                    pass

            if self.P1 is None:
                try:
                    self.P1 = self._solve_P1()
                    continue
                except ValueError:
                    pass

            if self.P2 is None:
                try:
                    self.P2 = self._solve_P2()
                    continue
                except ValueError:
                    pass

            if self.P4 is None:
                try:
                    self.P4 = self._solve_P4()
                    continue
                except ValueError:
                    pass

            if self.P5 is None:
                try:
                    self.P5 = self._solve_P5()
                    continue
                except ValueError:
                    pass

            if self.T1 is None:
                try:
                    self.T1 = self._solve_T1()
                    continue
                except ValueError:
                    pass

            if self.T2 is None:
                try:
                    self.T2 = self._solve_T2()
                    continue
                except ValueError:
                    pass

            if self.T5 is None:
                try:
                    self.T5 = self._solve_T5()
                    continue
                except ValueError:
                    pass

            solved = True

    def _solve_M(self):
        if self.P5 is not None and self.P1 is not None:
            def P5(M):
                P2 = gas_dynamics.normal_shock_pressure_ratio(M, self.gamma1) * self.P1
                M_reflected = gas_dynamics.reflected_shock_Mach_number(M, self.gamma1)
                return gas_dynamics.normal_shock_pressure_ratio(M_reflected, self.gamma1) * P2

            results = optimize.root_scalar(lambda M: self.P5 - P5(M), bracket=self.bracket)
            if not results.converged:
                raise RuntimeError

            return results.root

        if self.T5 is not None and self.T1 is not None:
            def T5(M):
                T2 = gas_dynamics.normal_shock_temperature_ratio(M, self.gamma1) * self.T1
                M_reflected = gas_dynamics.reflected_shock_Mach_number(M, self.gamma1)
                return gas_dynamics.normal_shock_temperature_ratio(M_reflected, self.gamma1) * T2

            results = optimize.root_scalar(lambda M: self.T5 - T5(M), bracket=self.bracket)
            if not results.converged:
                raise RuntimeError

            return results.root

        if self.P4 is not None and self.P1 is not None:
            def P4_P1(M):
                return gas_dynamics.shock_tube_flow_properties(M, self.T1, self.T4, self.MW1, self.MW4,
                                                               self.gamma1, self.gamma4, area_ratio=self.area_ratio)[0]

            results = optimize.root_scalar(lambda M: self.P4 / self.P1 - P4_P1(M), bracket=self.bracket)
            if not results.converged:
                raise RuntimeError

            return results.root

        else:
            raise ValueError

    def _solve_P1(self):
        if self.P5 is not None and self.M is not None:
            P2_P1 = gas_dynamics.normal_shock_pressure_ratio(self.M, self.gamma1)
            M_reflected = gas_dynamics.reflected_shock_Mach_number(self.M, self.gamma1)
            P5_P2 = gas_dynamics.normal_shock_pressure_ratio(M_reflected, self.gamma1)

            return self.P5 / P5_P2 / P2_P1

        elif self.P4 is not None and self.M is not None:
            P4_P1 = gas_dynamics.shock_tube_flow_properties(self.M, self.T1, self.T4, self.MW1, self.MW4,
                                                            self.gamma1, self.gamma4, area_ratio=self.area_ratio)[0]
            return P4_P1 * self.P1

        else:
            raise ValueError

    def _solve_P2(self):
        if self.P1 is not None and self.M is not None:
            P2_P1 = gas_dynamics.normal_shock_pressure_ratio(self.M, self.gamma1)
            return P2_P1 * self.P1

        else:
            raise ValueError

    def _solve_P4(self):
        if self.P1 is not None and self.M is not None:
            P4_P1 = gas_dynamics.shock_tube_flow_properties(self.M, self.T1, self.T4, self.MW1, self.MW4,
                                                            self.gamma1, self.gamma4, area_ratio=self.area_ratio)[0]
            return self.P1 * P4_P1

        else:
            raise ValueError

    def _solve_P5(self):
        if self.P1 is not None and self.M is not None:
            P2_P1 = gas_dynamics.normal_shock_pressure_ratio(self.M, self.gamma1)
            M_reflected = gas_dynamics.reflected_shock_Mach_number(self.M, self.gamma1)
            P5_P2 = gas_dynamics.normal_shock_pressure_ratio(M_reflected, self.gamma1)
            return P5_P2 * P2_P1 * self.P1

        else:
            raise ValueError

    def _solve_T1(self):
        if self.T5 is not None and self.M is not None:
            T2_T1 = gas_dynamics.normal_shock_temperature_ratio(self.M, self.gamma1)
            M_reflected = gas_dynamics.reflected_shock_Mach_number(self.M, self.gamma1)
            T5_T2 = gas_dynamics.normal_shock_temperature_ratio(M_reflected, self.gamma1)
            return self.T5 / T5_T2 / T2_T1

        else:
            raise ValueError

    def _solve_T2(self):
        if self.T1 is not None and self.M is not None:
            T2_T1 = gas_dynamics.normal_shock_temperature_ratio(self.M, self.gamma1)
            return T2_T1 * self.T1

        else:
            raise ValueError

    def _solve_T5(self):
        if self.T1 is not None and self.M is not None:
            T2_T1 = gas_dynamics.normal_shock_temperature_ratio(self.M, self.gamma1)
            M_reflected = gas_dynamics.reflected_shock_Mach_number(self.M, self.gamma1)
            T5_T2 = gas_dynamics.normal_shock_temperature_ratio(M_reflected, self.gamma1)
            return T5_T2 * T2_T1 * self.T1

        else:
            raise ValueError
