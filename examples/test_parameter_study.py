from knightshock import ParameterStudy

mechanism = "aramco2.cti"

T = [1000, 1050, 1100, 1150]
P = [100e5]
X = ["CH4: 0.05, O2: 0.10, Ar: 0.85"]

t = 8e-3

study = ParameterStudy.product(T, P, X, mechanism)
study.run(t)

print(study.df)
study.plot_IDT_profile(P[0], X[0])
