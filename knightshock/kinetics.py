from matplotlib import pyplot as plt
import cantera as ct
import numpy as np


class KineticsSimulation:
    def __init__(self, T, P, X, mechanism):
        self.gas = ct.Solution(mechanism)
        self.gas.TPX = T, P, X

        self.reactor = ct.Reactor(self.gas)
        self.reactor_net = ct.ReactorNet([self.reactor])

        self.states = ct.SolutionArray(self.gas, extra=['t'])

    def run(self, time, dt):
        t = 0
        while t < time:
            t += dt
            self.reactor_net.advance(t)
            self.states.append(self.reactor.thermo.state, t=t * 1e3)
            print("t = {:10.3e} ms\tT = {:10.3f} K\tP = {:10.3f}"
                  .format(self.reactor_net.time, self.reactor.T, self.reactor.thermo.P))

    def ignition_delay_time(self, species):
        if species not in self.gas.species_names:
            raise ValueError("Species name not found in mechanism")
        return self.states.t[np.argmax(self.states.X[:, self.gas.species_index(species)])]

    def plot_concentration(self, species):
        plt.figure()
        plt.plot(self.states.t, self.states.X[:, self.gas.species_index(species)])
        plt.xlabel('Time (ms)')
        plt.ylabel('{} Mole Fraction'.format(species))
        plt.show()
