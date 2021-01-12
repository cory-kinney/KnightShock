from knightshock import State
from knightshock.kinetics import ShockTubeReactorModel
from tabulate import tabulate

driven = "AR: 0.872938, O2: 0.087062, CH4: 0.03694, C2H6: 0.001884, C3H8: 0.000704, IC4H10: 0.000236, C4H10: 0.000236"

exp = State.from_experiment(294, 4e5, driven, 294, 50e5, "N2: 1", 730.7, "mechanisms\\aramco2.cti")
print(exp)

print("\nSimulating IDT")
sim = ShockTubeReactorModel(exp.regions[5]['EE'].thermo)
sim.run(10e-3)

print("\nIgnition Delay Time [ms]\n")
print(tabulate([[sim.IDT_max_pressure_rise() * 1e3]
                + [sim.IDT_peak_species_concentration(species) * 1e3 for species in ["OH", "OHV", "CH", "CHV"]]],
               headers=["max dp/dt", "OH", "OH*", "CH", "CH*"]))



