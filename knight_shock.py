from pint import UnitRegistry
import cantera as ct
import numpy as np

units = UnitRegistry()


class Mixture:
    def __init__(self, mole_fractions):
        """ Initializes a mixture using a dict of species (key) and mole fractions/ratios (value) """
        self._mole_fractions = None
        self.mole_fractions = mole_fractions

    def __str__(self):
        return "".join("{}:{}, ".format(key, value) for key, value in self.mole_fractions.items())[:-2]

    @property
    def mole_fractions(self):
        return self._mole_fractions

    @mole_fractions.setter
    def mole_fractions(self, value):
        """ Normalizes and sets the mole fractions for the mixture """
        total = sum(value.values())
        self._mole_fractions = {species: mole_fraction / total for species, mole_fraction in value}

    def get_fill_pressures(self, P):
        return {species: x * P for species, x in self._mole_fractions}


class Conditions:
    def __init__(self):
        self._T = None
        self._P = None

    @property
    def T(self):
        return self._T

    @T.setter
    @units.check('[temperature]')
    def T(self, value):
        self._T = value.to(units.K)

    @property
    def P(self):
        return self._P

    @P.setter
    @units.check('[pressure]')
    def P(self, value):
        self._P = value


class Experiment:
    def __init__(self):
        self.mechanism = None

        self.driver_mixture = None
        self.region4 = None

        self.driven_mixture = None
        self.region1 = None
        self._region2 = None
        self._region5 = None

        self._x_timer_counters = None  # [m]
        self._t_timer_counters = None  # [s]

        self._shock_velocity_model = None
        self._shock_velocity_r2 = None

        self._M = None
        self._attenuation = None

    def set_x_timer_counters(self, x_timer_counters, unit=units.m):
        factor = (1 * unit).to(units.m).magnitude
        self._x_timer_counters = np.array([x * factor for x in x_timer_counters])

    def set_t_timer_counters(self, t_timer_counters, unit=units.s):
        factor = (1 * unit).to(units.s).magnitude
        self._t_timer_counters = np.array([t * factor for t in t_timer_counters])

    def _calculate_shock_speed(self):
        num_timer_counters = len(self._t_timer_counters)

        x = np.empty(num_timer_counters)
        velocity = np.empty(num_timer_counters)

        for i in range(num_timer_counters):
            x[i] = (self._x_timer_counters[i] + self._x_timer_counters[i + 1]) / 2
            velocity = (self._x_timer_counters[i + 1] - self._x_timer_counters[i]) \
                       / (self._t_timer_counters[i + 1] - self._t_timer_counters[i])

        A = np.vstack([x, np.ones(len(x))]).T
        model, residual = np.linalg.lstsq(A, velocity)[:2]

        self._shock_velocity_model = (model[0] / units.s, model[1] * units.m / units.s)
        self._shock_velocity_r2 = 1 - residual / (velocity.size * velocity.var())

        solution = ct.solution(self.mechanism)
        solution.TPX = self.region1.T.magnitude, self.region1.P.to(units.Pa).magnitude, str(self.driven_mixture)

        gamma = solution.cp_mass / solution.cv_mass
        a = np.sqrt(ct.gas_constant / solution.mean_molecular_weight * units.J / units.kg / units.K
                    * gamma * self.region1.T)

        self._M = self._shock_velocity_model[1] / a
        self._attenuation = self._shock_velocity_model[0] / self._shock_velocity_model[1]

