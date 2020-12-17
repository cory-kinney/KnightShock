from knightshock.kinetics import ShockTubeSimulation

# Inputs
T = 1593.6
P = 100.8e5
X = "CH4: 0.0102, O2: 0.0204, Ar: 0.9694"

mechanism = "aramco2.cti"

time = 2e-3
dt = 1e-7

sim = ShockTubeSimulation.from_mixture(T, P, X, mechanism)
sim.run(time)

print("\nIgnition Delay Time (ms)")
print("OH\t\t\tCH\n{:.4f}\t\t{:.4f}".format(sim.ignition_delay_time_species("OH") * 1e3,
                                            sim.ignition_delay_time_species("CH") * 1e3))
print("OH*\t\t\tCH*\n{:.4f}\t\t{:.4f}".format(sim.ignition_delay_time_species("OHV") * 1e3,
                                              sim.ignition_delay_time_species("CHV") * 1e3))
print("Pressure Rise\n{:.4f}".format(sim.ignition_delay_time_pressure_rise() * 1e3))

sim.plot_pressure_history()
sim.plot_temperature_history()
sim.plot_concentration(["OH", "CH", "OHV", "CHV"])

