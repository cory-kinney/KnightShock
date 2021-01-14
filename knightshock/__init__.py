"""Shock tube experiment planning and data analysis package"""

from knightshock.gas_dynamics import *
from knightshock.kinetics import *
from dataclasses import dataclass

from typing import Dict, List, Union
from tabulate import tabulate
import cantera as ct
import numpy as np


@dataclass(frozen=True)
class State:
    """Base class encapsulating the state of the shock tube regions"""

    class Region:
        """"""

        def __init__(self, T: float, P: float, X: Union[str, Dict[str, float], np.ndarray], thermo: ct.ThermoPhase):
            self._T = T
            self._P = P
            self._X = X
            self._thermo = thermo

            # Validate inputs
            if not isinstance(thermo, ct.ThermoPhase):
                raise TypeError
            self._thermo.TPX = self.T, self.P, self.X

        @property
        def thermo(self):
            self._thermo.TPX = self.T, self.P, self.X
            return self._thermo

        @property
        def T(self):
            return self._T

        @property
        def P(self):
            return self._P

        @property
        def X(self):
            return self._X

        @property
        def MW(self):
            return self.thermo.mean_molecular_weight

        @property
        def gamma(self):
            return self.thermo.cp / self.thermo.cv

        @property
        def a(self):
            return (self.gamma * ct.gas_constant / self.MW * self.T) ** 0.5

    region1: Region
    region4: Region
    region2: Dict[str, Region]
    region5: Dict[str, Region]

    def __getitem__(self, region_num):
        if region_num == 1:
            return self.region1
        elif region_num == 2:
            return self.region2
        elif region_num == 4:
            return self.region4
        elif region_num == 5:
            return self.region5
        else:
            raise KeyError("Invalid region number")

    def __str__(self):
        initial_conditions_table = tabulate(
            [[region, self[region].T, self[region].P / 1e5] for region in [1, 4]],
            headers=['Region', 'T [K]', 'P [bar]'])

        shock_conditions_table = tabulate(
            [['T2 [K]', self[2]['FF'].T, self[2]['FE'].T, self[2]['EE'].T],
             ['P2 [bar]', self[2]['FF'].P / 1e5, self[2]['FE'].P / 1e5, self[2]['EE'].P / 1e5],
             ['T5 [K]', self[5]['FF'].T, self[5]['FE'].T, self[5]['EE'].T],
             ['P5 [bar]', self[5]['FF'].P / 1e5, self[5]['FE'].P / 1e5, self[5]['EE'].P / 1e5]],
            headers=['', 'FF', 'FE', 'EE'])

        return "Initial Conditions\n\n{}\n\nShock Conditions\n\n{}"\
            .format(initial_conditions_table, shock_conditions_table)

    @classmethod
    def from_experiment(cls, T1, P1, X1, T4, P4, X4, u, mechanism):
        solution = ct.Solution(mechanism)

        region1 = State.Region(T1, P1, X1, solution)
        region4 = State.Region(T4, P4, X4, solution)
        region2 = {}
        region5 = {}

        for method in ['FF', 'FE', 'EE']:
            (T2, P2), (T5, P5) = frozen_shock_conditions(u / region1.a, region1.thermo, method)
            region2[method] = State.Region(T2, P2, X1, solution)
            region5[method] = State.Region(T5, P5, X1, solution)

        return cls(region1, region4, region2, region5)


class Experiment:
    def __init__(self, mechanism):
        self.gas = ct.Solution(mechanism)
        self.state = None

        self.u = None
        self.attenuation = None
        self.r2 = None

    @staticmethod
    def shock_velocity_TOF(x, dt):
        x_midpoint = (x[1:] + x[:-1]) / 2
        u_average = np.abs(np.diff(x)) / dt

        A = np.vstack([x_midpoint, np.ones(len(x_midpoint))]).T
        model, residual = np.linalg.lstsq(A, u_average, rcond=None)[:2]

        u0: float = model[1]
        attenuation: float = model[0]
        r2: float = (1 - residual / (u_average.size * u_average.var()))[0]

        return u0, attenuation, r2

    @staticmethod
    def dt_pressure_trace(time: np.ndarray, pressure_traces: List[np.ndarray], threshold: float):
        def interpolation(x, x0, x1, y0, y1):
            return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

        indices = [np.argmax(P > threshold) for P in pressure_traces]
        t = np.array([interpolation(threshold, P[i - 1], P[i], time[i - 1], time[i])
                      for i, P in zip(indices, pressure_traces)])

        return np.diff(t)
