from knightshock import ParameterStudy

T = [1100, 1200, 1300, 1400]
P = [100e5, 200e5, 300e5]
X = ["CH4: 0.05, O2: 0.10, Ar: 0.85", "CH4: 0.10, O2: 0.20, Ar: 0.85"]
mechanism = "mechanisms\\aramco2.cti"
duration = 10e-3

study = ParameterStudy.product(T, P, X, mechanism)
study.run(duration)
study.df.to_csv("test.csv")
