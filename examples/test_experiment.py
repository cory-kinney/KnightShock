from knightshock import State, ShockTubeReactorModel
from tabulate import tabulate

driven = "AR: 0.872938, O2: 0.087062, CH4: 0.03694, C2H6: 0.001884, C3H8: 0.000704, IC4H10: 0.000236, C4H10: 0.000236"

exp = State.from_experiment(294, 4e5, driven, 294, 50e5, "N2: 1", 730.7, "mechanisms\\aramco2.cti")
print(exp)

print("\nSimulating IDT")
sim = ShockTubeReactorModel(exp[5]['EE'].thermo)
sim.run(10e-3)

IDT = {"Method": ["max(dp/dt)", "CH", "CH*", "OH", "OH*"],
       "IDT [ms]": [sim.IDT_max_pressure_rise() * 1e3, sim.IDT_peak_species_concentration("CH") * 1e3,
                    sim.IDT_peak_species_concentration("CHV") * 1e3, sim.IDT_peak_species_concentration("OH") * 1e3,
                    sim.IDT_peak_species_concentration("OHV") * 1e3]}
print("\n"+tabulate(IDT, headers='keys', tablefmt="rst"))

exp.region4 = ""

