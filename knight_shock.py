from pint import UnitRegistry
import cantera as ct
import numpy as np

units = UnitRegistry()


class Experiment:
    """
    Regions:
        1 - initial conditions driven section
        2 - post-incident-shock conditions driven section
        3 - expanded conditions driver section
        4 - initial conditions driver section
        5 - post-reflected-shock driven section
    """

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

        solution = ct.Solution(self.mechanism)
        solution.TPX = self.region1.T.magnitude, self.region1.P.to(units.Pa).magnitude, str(self.driven_mixture)

        gamma = solution.cp_mass / solution.cv_mass
        a = np.sqrt(ct.gas_constant / solution.mean_molecular_weight * units.J / units.kg / units.K
                    * gamma * self.region1.T)

        self._M = self._shock_velocity_model[1] / a
        self._attenuation = self._shock_velocity_model[0] / self._shock_velocity_model[1]




