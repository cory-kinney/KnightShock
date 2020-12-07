"""Shock tube experiment planning and data analysis package

Notes
-----
The standard shock tube region notation followed in KnightShock's code and documentation is:

1. initial driven gas

2. post-incident-shock driven gas

3. expanded driver gas

4. initial driver gas

5. post-reflected-shock driven gas

"""

from knightshock import gas_dynamics
from matplotlib import pyplot as plt
from scipy import optimize
import cantera as ct
import numpy as np


class ShockTubeState:
    """

    Attributes
    ----------
    driven_mixture : str
        mixture composition of driven gas as comma-separated species and mole fraction pairs
    driver_mixture : str
        mixture composition of driver gas as comma-separated species and mole fraction pairs
    T1 : float
        initial temperature of the driven gas
    P1 : float
        initial pressure of the driven gas
    T4 : float
        initial temperature of the driver gas
    P4 : float
        initial pressure of the driver gas
    T2 : float
        temperature of the driven gas after the incident shock
    P2 : float
        pressure of the driven gas after the incident shock
    T5 : float
        temperature of the driver gas after the reflected shock
    P5 : float
        pressure of the driver gas after the reflected shock
    u : float
        velocity of incident shock
    M : float
        Mach number of incident shock
    area_ratio : float
        ratio of driver area to driven area

    Notes
    -----
    Temperature and pressure must be in units of K and Pa, respectively, for thermodynamic property calculations.

    """

    def __init__(self, thermo):
        """
        Parameters
        ----------
        thermo:
            object for temperature- and pressure-dependent thermodynamic property calculations

        """

        if not isinstance(thermo, ct.ThermoPhase):
            raise TypeError("Thermo must be instance of cantera.ThermoPhase")

        self.thermo = thermo
        self.driven_mixture = None
        self.driver_mixture = None

        self.T1 = None
        self.P1 = None
        self.T2 = None
        self.P2 = None
        self.T4 = None
        self.P4 = None
        self.T5 = None
        self.P5 = None

        self.u = None
        self.M = None

        self.area_ratio = 1

    @property
    def MW1(self):
        """Mean molecular weight of driven gas at initial conditions"""
        self.thermo.TPX = self.T1, self.P1, self.driven_mixture
        return self.thermo.mean_molecular_weight

    @property
    def MW4(self):
        """Mean molecular weight of driver gas at initial conditions"""
        self.thermo.TPX = self.T4, self.P4, self.driver_mixture
        return self.thermo.mean_molecular_weight

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
        """Speed of sound in driven gas at initial conditions"""
        self.thermo.TPX = self.T1, self.P1, self.driven_mixture
        return (self.thermo.cp / self.thermo.cv * ct.gas_constant * self.T1 / self.thermo.mean_molecular_weight) ** 0.5

    @property
    def a4(self):
        """Speed of sound in driver gas at initial conditions"""
        self.thermo.TPX = self.T4, self.P4, self.driver_mixture
        return (self.thermo.cp / self.thermo.cv * ct.gas_constant * self.T4 / self.thermo.mean_molecular_weight) ** 0.5


class Experiment(ShockTubeState):
    """Shock tube experiment base class that extends the `knightshock.ShockTubeState` class

    Attributes
    ----------
    x : numpy.ndarray
        positions relative to end wall
    dt : numpy.ndarray
        differences in shock wave arrival times over intervals

    """

    def __init__(self, thermo):
        super().__init__(thermo)

        # Data points for shock velocity time-of-flight calculation
        self._x = None
        self._dt = None

        self._x_midpoint = None
        self._u_average = None

        self.attenuation = None
        self.r2 = None

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = np.array(value)

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        self._dt = np.array(value)

    def update_shock_velocity(self):
        """Computes the least-squares linear fit of average shock velocities over intervals

        Raises
        ------
        ValueError
            `x` values are not greater than zero
        ValueError
            `x` values are not strictly decreasing

        """

        if np.any(self.x < 0):
            raise ValueError("x values must be positive")
        if not np.all(self.x[1:] < self.x[:-1]):
            raise ValueError("x values must be strictly decreasing")

        self._x_midpoint = (self.x[1:] + self.x[:-1]) / 2
        self._u_average = np.abs(np.diff(self.x)) / self.dt

        A = np.vstack([self._x_midpoint, np.ones(len(self._x_midpoint))]).T
        model, residual = np.linalg.lstsq(A, self._u_average, rcond=None)[:2]

        self.u = model[1]
        self.attenuation = model[0] / model[1]
        self.r2 = (1 - residual / (self._u_average.size * self._u_average.var()))[0]

    def update_shock_conditions(self, *, constant_properties=False):
        """
        Parameters
        ----------
        constant_properties : bool (optional)

        """
        if constant_properties:
            self.T2 = gas_dynamics.T2_ideal(self.T1, self.M, self.gamma1)
            self.P2 = gas_dynamics.P2_ideal(self.P1, self.M, self.gamma1)
            self.T5 = gas_dynamics.T5_ideal(self.T1, self.M, self.gamma1)
            self.P5 = gas_dynamics.P5_ideal(self.P1, self.M, self.gamma1)
        else:
            self.thermo.X = self.driven_mixture
            self.T2, self.P2, self.T5, self.P5 = \
                gas_dynamics.shock_conditions_FROSH(self.T1, self.P1, self.M, thermo=self.thermo)

    def plot_shock_velocity(self):
        plt.figure()
        plt.scatter(self._x_midpoint, self._u_average, color='r')
        plt.plot([0, self.x.max()], [self.u, self.u + self.attenuation * self.x.max()], '-k')
        plt.xlim([0, self.x.max()])
        plt.show()


class ExperimentPlan(ShockTubeState):
    """Class for planning experiments with different combinations of initial conditions and target shock conditions"""

    def __init__(self, thermo):
        super().__init__(thermo)

        self.area_ratio = 1

    def solve(self, method, *, bracket=None):
        if self.P1 is None or self.T1 is None or self.T4 is None:
            raise ValueError

        if bracket is None:
            bracket = [1.001, 5]

        if method == "M":
            if self.M is None:
                raise ValueError

        elif method == "P5/P1":
            if self.P5 is None or self.P1 is None:
                raise ValueError

            def P5(M):
                P2 = gas_dynamics.normal_shock_pressure_ratio(M, self.gamma1) * self.P1
                M_reflected = gas_dynamics.reflected_shock_Mach_number(M, self.gamma1)
                return gas_dynamics.normal_shock_pressure_ratio(M_reflected, self.gamma1) * P2

            results = optimize.root_scalar(lambda M: self.P5 - P5(M), bracket=bracket)
            if not results.converged:
                raise RuntimeError

            self.M = results.root

        elif method == "T5/T1":
            if self.T5 is None or self.T1 is None:
                raise ValueError

            def T5(M):
                T2 = gas_dynamics.normal_shock_temperature_ratio(M, self.gamma1) * self.T1
                M_reflected = gas_dynamics.reflected_shock_Mach_number(M, self.gamma1)
                return gas_dynamics.normal_shock_temperature_ratio(M_reflected, self.gamma1) * T2

            results = optimize.root_scalar(lambda M: self.T5 - T5(M), bracket=bracket)
            if not results.converged:
                raise RuntimeError

            self.M = results.root

        elif method == "P4/P1":
            if self.P4 is None or self.P1 is None:
                raise ValueError

            def P4_P1(M_guess):
                return gas_dynamics.shock_tube_flow_properties(M_guess, self.T1, self.T4, self.MW1, self.MW4,
                                                               self.gamma1, self.gamma4, area_ratio=self.area_ratio)[0]

            results = optimize.root_scalar(lambda M: self.P4 / self.P1 - P4_P1(M), bracket=bracket)
            if not results.converged:
                raise RuntimeError

            self.M = results.root

        else:
            raise ValueError("Invalid method specified")

        self.u = self.M * self.a1
        self.P2 = gas_dynamics.P2_ideal(self.P1, self.M, self.gamma1)
        self.T2 = gas_dynamics.T2_ideal(self.T1, self.M, self.gamma1)
        if method != "P5/P1":
            self.P5 = gas_dynamics.P5_ideal(self.P1, self.M, self.gamma1)
        if method != "T5/T1":
            self.T5 = gas_dynamics.T5_ideal(self.T1, self.M, self.gamma1)
        if method != "P4/P1":
            P4_P1 = gas_dynamics.shock_tube_flow_properties(self.M, self.T1, self.T4, self.MW1, self.MW4,
                                                            self.gamma1, self.gamma4, area_ratio=self.area_ratio)[0]
            self.P4 = P4_P1 * self.P1
