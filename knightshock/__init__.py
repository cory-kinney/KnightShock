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


# Optional unit conversion support
try:
    import pint
    units = pint.UnitRegistry()
except ImportError:
    pint = None
    units = None


class ShockVelocityError(Exception):
    """Raised when shock velocity is invalid"""
    pass


class ShockTubeState:
    """Base class defining the state of the shock tube with basic validation of parameter values"""

    def __init__(self, mechanism):
        self._gas = ct.Solution(mechanism)
        self._driven_mixture = None
        self._driver_mixture = None

        self._T1 = None
        self._P1 = None
        self._T2 = None
        self._P2 = None
        self._T4 = None
        self._P4 = None
        self._T5 = None
        self._P5 = None

        self._M = None
        self._area_ratio = None

    @property
    def driven_mixture(self):
        """"""
        return self._driven_mixture

    @driven_mixture.setter
    def driven_mixture(self, value):
        self._gas.X = value  # Attempts to set mixture property for Cantera to validate input
        self._driven_mixture = value

    @property
    def driver_mixture(self):
        """"""
        return self._driver_mixture

    @driver_mixture.setter
    def driver_mixture(self, value):
        self._gas.X = value  # Attempts to set mixture property for Cantera to validate input
        self._driver_mixture = value

    @property
    def T1(self):
        """Initial driven state temperature [K]"""
        return self._T1

    @T1.setter
    def T1(self, value):
        if not value > self._gas.min_temp:
            raise ValueError("Temperature must be greater than zero")
        self._T1 = value

    @property
    def T2(self):
        """Post-incident-shock driven state temperature [K]"""
        return self._T2

    @T2.setter
    def T2(self, value):
        if not value > 0:
            raise ValueError("Temperature must be greater than zero")
        self._T2 = value

    @property
    def T4(self):
        """Initial driver state temperature [K]"""
        return self._T4

    @T4.setter
    def T4(self, value):
        if not value > 0:
            raise ValueError("Temperature must be greater than zero")
        self._T4 = value

    @property
    def T5(self):
        """Post-reflected-shock driven state temperature [K]"""
        return self._T5

    @T5.setter
    def T5(self, value):
        if not value > 0:
            raise ValueError("Temperature must be greater than zero")
        self._T5 = value

    @property
    def P1(self):
        """"""
        return self._P1

    @P1.setter
    def P1(self, value):
        if not value > 0:
            raise ValueError("Pressure must be greater than zero")
        self._P1 = value

    @property
    def P2(self):
        """"""
        return self._P2

    @P2.setter
    def P2(self, value):
        if not value > 0:
            raise ValueError("Pressure must be greater than zero")
        self._P2 = value

    @property
    def P4(self):
        """"""
        return self._P4

    @P4.setter
    def P4(self, value):
        if not value > 0:
            raise ValueError("Pressure must be greater than zero")
        self._P4 = value

    @property
    def P5(self):
        """"""
        return self._P5

    @P5.setter
    def P5(self, value):
        if not value > 0:
            raise ValueError("Pressure must be greater than zero")
        self._P5 = value

    @property
    def M(self):
        """Mach number of incident shock wave"""
        return self._M

    @M.setter
    def M(self, value):
        if not value > 1:
            raise ValueError("Mach number must be greater than one")
        self._M = value

    @property
    def u(self):
        """Velocity of incident shock wave [m/s]"""
        return self._M * self.a1

    @property
    def area_ratio(self):
        """Ratio of area of driver to driven"""
        return self._area_ratio

    @area_ratio.setter
    def area_ratio(self, value):
        if value < 1:
            raise ValueError("Area ratio must be greater than or equal to one")

    @property
    def MW1(self):
        """Initial driven mixture mean molecular weight [kg/kmol]"""
        self._gas.TPX = self.T1, self.P1, self.driven_mixture
        return self._gas.mean_molecular_weight

    @property
    def MW4(self):
        """Initial driver mixture mean molecular weight [kg/kmol]"""
        self._gas.TPX = self.T4, self.P4, self.driver_mixture
        return self._gas.mean_molecular_weight

    @property
    def gamma1(self):
        """Specific heat ratio of driven gas at initial conditions"""
        self._gas.TPX = self.T1, self.P1, self.driven_mixture
        return self._gas.cp / self._gas.cv

    @property
    def gamma4(self):
        """Specific heat ratio of driver gas at initial conditions"""
        self._gas.TPX = self.T4, self.P4, self.driver_mixture
        return self._gas.cp / self._gas.cv

    @property
    def a1(self):
        """Speed of sound in driven gas at initial conditions [m/s]"""
        self._gas.TPX = self.T1, self.P1, self.driven_mixture
        return (self._gas.cp / self._gas.cv * ct.gas_constant / self._gas.mean_molecular_weight
                * self.T1) ** 0.5

    @property
    def a4(self):
        """Speed of sound in driver gas at initial conditions [m/s]"""
        self._gas.TPX = self.T4, self.P4, self.driver_mixture
        return (self._gas.cp / self._gas.cv * ct.gas_constant / self._gas.mean_molecular_weight
                * self.T4) ** 0.5


class Experiment(ShockTubeState):
    """Shock tube experiment base class that extends the `knightshock.ShockTubeState` class

    """

    def __init__(self, thermo):
        super().__init__(thermo)

        pass

    def calculate_shock_conditions(self, u, *, constant_properties=False):
        """
        Parameters
        ----------
        constant_properties : bool (optional)

        """
        M = u / self.a1
        if M < 1:
            raise ShockVelocityError

        if constant_properties:
            T2 = gas_dynamics.T2_ideal(self.T1, self.M, self.gamma1)
            P2 = gas_dynamics.P2_ideal(self.P1, self.M, self.gamma1)
            T5 = gas_dynamics.T5_ideal(self.T1, self.M, self.gamma1)
            P5 = gas_dynamics.P5_ideal(self.P1, self.M, self.gamma1)
        else:
            self._gas.X = self.driven_mixture
            T2, P2, T5, P5 = gas_dynamics.shock_conditions_FROSH(self.T1, self.P1, self.M, thermo=self._gas)

        return T2, P2, T5, P5

    @staticmethod
    def calculate_shock_velocity(x, dt):
        """Calculates the shock velocity by time-of-flight

        Parameters
        ----------
        x : numpy.ndarray
            measurement positions relative to endwall
        dt : numpy.ndarray
            time-of-flight over interval

        """

        x_midpoint = (x[1:] + x[:-1]) / 2
        u_average = np.abs(np.diff(x)) / dt

        A = np.vstack([x_midpoint, np.ones(len(x_midpoint))]).T
        model, residual = np.linalg.lstsq(A, u_average, rcond=None)[:2]

        u = model[1]
        attenuation = model[0] / model[1]
        r2 = (1 - residual / (u_average.size * u_average.var()))[0]

        return u, attenuation, r2


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
