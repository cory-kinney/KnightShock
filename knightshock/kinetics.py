""""""

from typing import Union, Iterable
from tqdm import tqdm
import cantera as ct
import pandas as pd
import numpy as np


class ShockTubeReactor:
    """Constant volume, adiabatic reactor model for shock tube experiment chemical kinetics simulations"""

    def __init__(self, gas: ct.Solution):
        """Initializes the reactor with the initial state of the gas

        Parameters
        ----------
        gas: cantera.Solution
            initial state of the reactor

        """
        self.reactor = ct.Reactor(gas)
        self.reactor_net = ct.ReactorNet([self.reactor])
        self.states = ct.SolutionArray(gas, extra=['t'])

    def run(self, duration: float, *, log_frequency: int = 10):
        """

        Parameters
        ----------
        duration: float
            duration of the simulation (s)
        log_frequency: int, optional
            number of iterations between logging data points

        """
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

    def IDT_max_pressure_rise(self) -> float:
        """Calculates the ignition delay time from the maximum pressure rise"""
        return self.states.t[np.argmax(np.diff(self.states.P) / np.diff(self.states.t))]

    def IDT_peak_species_concentration(self, species: str) -> float:
        """Calculates the ignition delay time from the peak mole fraction of a species

        Parameters
        ----------
        species: str
            name of species as listed in the mechanism

        """
        return self.states.t[np.argmax(self.states(species).X)]


class KineticsParameterStudy:
    """Class designed for convenient execution of many `knightshock.kinetics.ShockTubeReactor` simulations"""

    def __init__(self, df: pd.DataFrame, mechanism: str):
        self.gas = ct.Solution(mechanism)
        self.df = df

    @classmethod
    def from_product(cls, T: Union[float, Iterable[float]], P: Union[float, Iterable[float]],
                     X: Union[str, Iterable[str]], mechanism: str):
        """Defines a parameter study from the cartesian product of the given temperatures, pressures, and mixtures using
        a given mechanism

        Parameters
        ----------
        T: float or iterable of floats
            temperature
        P: float or iterable of floats
            pressure
        X: str or iterable of str
            mixture
        mechanism: str
            file path of Cantera-compatible mechanism

        """
        index = pd.MultiIndex.from_product([X, P, T], names=["X", "P", "T"])
        return cls(pd.DataFrame(index=index).reset_index(), mechanism)

    def run(self, duration: float):
        """Runs all cases using the `knightshock.kinetics.ShockTubeReactor` for the given duration and computes ignition
        delay time by maximum pressure rise

        Parameters
        ----------
        duration: float
            duration of the simulation (s)

        """
        IDT = []
        for index, row in self.df.iterrows():
            T, P, X = row["T"], row["P"], row["X"]
            print("{}/{} | T = {:.0f} K, P = {:.1f} bar, {{{}}}".format(index + 1, len(self.df.index), T, P / 1e5, X))

            self.gas.TPX = T, P, X
            model = ShockTubeReactor(self.gas)
            model.run(duration)

            IDT.append(model.IDT_max_pressure_rise())

        self.df["IDT"] = IDT
