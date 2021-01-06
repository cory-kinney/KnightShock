"""Shock tube experiment planning and data analysis package"""

from matplotlib import pyplot as plt
from knightshock import gas_dynamics
from tabulate import tabulate
from tqdm import tqdm
import cantera as ct
import pandas as pd
import numpy as np


class ShockTubeState:
    """Base class encapsulating the state of the shock tube regions"""

    def __init__(self, mechanism):
        self.initial_driven = ct.Solution(mechanism)
        self.incident_driven = ct.Solution(mechanism)
        self.reflected_driven = ct.Solution(mechanism)
        self.initial_driver = ct.Solution(mechanism)

    def __getitem__(self, region_num):
        """Allows for indexing of the 'ShockTubeState' by region number for convenience"""
        if region_num == 1:
            return self.initial_driven
        elif region_num == 2:
            return self.incident_driven
        elif region_num == 4:
            return self.initial_driver
        elif region_num == 5:
            return self.initial_driven
        else:
            raise IndexError("Invalid region number")

    def __str__(self):
        """Tabulates pressure and temperature for states into a str for printing"""
        return tabulate([[i, self[i].T, self[i].P / 1e5] for i in [1, 2, 4, 5]],
                        headers=['State', 'Temperature [K]', 'Pressure [bar]'])

    @classmethod
    def prediction(cls, X_driver, X_driven, *, T1=None, T4=None, T5=None, P1=None, P4=None, P5=None, mechanism,
                   attenuation=None):
        """Creates a prediction of the shock tube state for given constraints"""
        state = cls(mechanism)
        state.driver_mixture = X_driver
        state.driven_mixture = X_driven

        raise NotImplementedError


class ShockTubeExperiment:
    def __init__(self, mechanism):
        self.state = ShockTubeState(mechanism)
        self._u = None

    # Read only properties

    @property
    def T1(self):
        return self.state[1].T

    @property
    def P1(self):
        return self.state[1].P

    @property
    def T2(self):
        return self.state[2].T

    @property
    def P2(self):
        return self.state[2].P

    @property
    def T4(self):
        return self.state[4].T

    @property
    def P4(self):
        return self.state[4].P

    @property
    def T5(self):
        return self.state[5].T

    @property
    def P5(self):
        return self.state[5].P

    @property
    def a1(self):
        return (self.state[1].cp / self.state[1].cv * ct.gas_constant / self.state[1].mean_molecular_weight
                * self.state[1].T) ** 0.5

    @property
    def M(self):
        return self._u / self.a1

    # Read and write accessible properties

    @property
    def driven_mixture(self):
        return self.state[1].X

    @driven_mixture.setter
    def driven_mixture(self, value):
        self.state[1].X = self.state[4].X = self.state[5].X = value

    @property
    def driver_mixture(self):
        return self.state[4].X

    @driver_mixture.setter
    def driver_mixture(self, value):
        self.state[4].X = value

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, value):
        if value / self.a1 < 1:
            raise ValueError

        self._u = value

        self.state[2].TP, self.state[5].TP = \
            gas_dynamics.shock_conditions_FROSH(self.state[1].T, self.state[1].P, self.M, thermo=self.state[1])

    # Utility methods

    @staticmethod
    def shock_velocity_TOF(x, dt):
        x_midpoint = (x[1:] + x[:-1]) / 2
        u_average = np.abs(np.diff(x)) / dt

        A = np.vstack([x_midpoint, np.ones(len(x_midpoint))]).T
        model, residual = np.linalg.lstsq(A, u_average, rcond=None)[:2]

        u0 = model[1]
        attenuation = model[0]
        r2 = (1 - residual / (u_average.size * u_average.var()))[0]

        return u0, attenuation, r2


class ShockTubeReactorModel:
    """Constant volume, adiabatic reactor model for shock tube experiment chemical kinetics simulations"""

    def __init__(self, gas):
        self.reactor = ct.Reactor(gas)
        self.reactor_net = ct.ReactorNet([self.reactor])
        self.states = ct.SolutionArray(gas, extra=['t'])

    def run(self, duration, *, log_frequency=10):
        t = 0
        iteration = 0

        with tqdm(total=duration, bar_format="|{bar:25}| {desc}") as progress_bar:
            while t < duration:
                t = self.reactor_net.step()

                # Saves state and updates progress bar for initial state, final state, and iteration numbers that are
                # multiples of the log frequency
                if iteration % log_frequency == 0 or t > duration:
                    self.states.append(self.reactor.thermo.state, t=t)

                    progress_bar.n = t if t < duration else duration
                    time = "{t:.{dec}f} {unit}".format(t=t * (1e3 if t > 1e-3 else 1e6), dec=3 if t > 1e-3 else 1,
                                                       unit="ms" if t > 1e-3 else "µs")
                    progress_bar.set_description_str("{t}, T = {T:.1f} K, P = {P:.2f} bar"
                                                     .format(t=time, T=self.reactor.T, P=self.reactor.thermo.P / 1e5))

                iteration += 1

    def IDT_max_pressure_rise(self):
        """Calculates the ignition delay time from the maximum pressure rise

        Returns
        -------
        float
            ignition delay time [s]

        """
        return self.states.t[np.argmax(np.diff(self.states.P) / np.diff(self.states.t))]

    def IDT_peak_species_concentration(self, species):
        """Calculates the ignition delay time from the peak of the species mole fraction

        Parameters
        ----------
        species : str

        Returns
        -------
        float
            ignition delay time [s]

        """
        return self.states.t[np.argmax(self.states(species).X)]

    def plot_temperature_history(self):
        plt.figure()
        plt.plot(self.states.t * 1e3, self.states.T)
        plt.xlabel("Time (ms)")
        plt.ylabel("Temperature (K)")
        plt.show()

    def plot_pressure_history(self):
        plt.figure()
        plt.plot(self.states.t * 1e3, self.states.P / 1e5)
        plt.xlabel("Time (ms)")
        plt.ylabel("Pressure (bar)")
        plt.show()

    def plot_concentration(self, species):
        if isinstance(species, str):
            species = [species]

        plt.figure()
        for s in species:
            plt.plot(self.states.t * 1e3, self.states(s).X, label=s)
        plt.xlabel("Time (ms)")
        plt.ylabel("Mole Fraction")
        plt.legend(loc='upper right')
        plt.show()


class ParameterStudy:
    def __init__(self, df, mechanism):
        self.gas = ct.Solution(mechanism)
        self.df = df

    @classmethod
    def product(cls, T, P, X, mechanism):
        index = pd.MultiIndex.from_product([X, P, T], names=["X", "P", "T"])
        return cls(pd.DataFrame(index=index).reset_index(), mechanism)

    def run(self, duration):
        IDT = []

        for index, row in self.df.iterrows():
            T, P, X = row["T"], row["P"], row["X"]
            print("{}/{} | T = {:.0f} K, P = {:.1f} bar, {{{}}}".format(index + 1, len(self.df.index), T, P / 1e5, X))

            self.gas.TPX = T, P, X
            model = ShockTubeReactorModel(self.gas)
            model.run(duration)

            IDT.append(model.IDT_max_pressure_rise())

        self.df["IDT"] = IDT

    def plot_IDT_profile(self, P, X):
        data = self.df.loc[(self.df["P"] == P) & (self.df["X"] == X)]

        plt.figure()
        plt.scatter(data["T"], data["IDT"])
        plt.show()
