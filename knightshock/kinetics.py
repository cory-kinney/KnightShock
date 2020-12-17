from matplotlib import pyplot as plt
import cantera as ct
import numpy as np


class ShockTubeSimulation:
    """Constant volume, adiabatic reactor for shock tube experiment simulations"""

    def __init__(self, gas):
        self.gas = gas
        self.reactor = ct.Reactor(self.gas)
        self.reactor_net = ct.ReactorNet([self.reactor])
        self.states = ct.SolutionArray(self.gas, extra=['t'])

    @classmethod
    def from_mixture(cls, T, P, X, mechanism):
        """Initializes a simulation from the mixture mole fractions

        Parameters
        ----------
        T : float
            initial temperature [K]
        P : float
            initial pressure [Pa]
        X
            initial mixture mole fractions
        mechanism
            file path for a valid mechanism for Cantera

        """
        gas = ct.Solution(mechanism)
        gas.TPX = T, P, X
        return cls(gas)

    @classmethod
    def from_shock_tube_state(cls, state):
        """Initializes a simulation from a `knightshock.ShockTubeState` object

        Parameters
        ----------
        state : knightshock.ShockTubeState

        """
        state._gas.TPX = state.T5, state.P5, state.driven_mixture
        return cls(state._gas)

    def run(self, time):
        """Runs the simulation

        Parameters
        ----------
        time : float
            duration of the simulation [s]

        """
        t = 0
        while t < time:
            t = self.reactor_net.step()
            self.states.append(self.reactor.thermo.state, t=t)
            print("t = {:10.3e} ms\tT = {:10.3f} K\tP = {:10.3f}"
                  .format(self.reactor_net.time, self.reactor.T, self.reactor.thermo.P))

    def ignition_delay_time_pressure_rise(self):
        """Calculates the ignition delay time from the maximum pressure rise

        Returns
        -------
        float
            ignition delay time [s]

        """
        return self.states.t[np.argmax(np.diff(self.states.P) / np.diff(self.states.t))]

    def ignition_delay_time_species(self, species):
        """Calculates the ignition delay time from the peak of the mole fraction of a species

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
