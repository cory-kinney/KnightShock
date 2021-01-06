from knightshock import ShockTubeReactorModel
from tabulate import tabulate
import cantera as ct

T = 1300
P = 100e5
X = "CH4: 0.05, O2: 0.10, Ar: 0.85"
mechanism = "mechanisms\\aramco2.cti"
duration = 5e-3

gas = ct.Solution(mechanism)
gas.TPX = T, P, X

print("T = {:.0f} K, P = {:.1f} bar, {{{}}}".format(T, P / 1e5, X))
sim = ShockTubeReactorModel(gas)
sim.run(duration)

print("\nIgnition Delay Time (ms)\n")
print(tabulate([[sim.IDT_max_pressure_rise() * 1e3]
                + [sim.IDT_peak_species_concentration(species) * 1e3 for species in ["OH", "OHV", "CH", "CHV"]]],
               headers=["max dp/dt", "OH", "OH*", "CH", "CH*"]))

sim.plot_pressure_history()
