import cantera as ct
import numpy as np

R = ct.gas_constant
_linear_polyatomic_molecules = frozenset({
    "BeCl2", "C3", "CCN", "CCO", "CH2", "CNC", "CNN", "CO2", "F2Xe", "HCC", "HCN", "N2O",
    "N3", "NCN", "NCO", "OCS", "C2H2", "C2N2", "CCCC", "HCCCl", "C3H2", "C3N2", "C3O2"
})


def frozen_shock_conditions(M, mixture, method=None, *, max_iter=1000, convergence_criteria=1e-6):
    """Implementation of the FROzen SHock (FROSH) algorithm"""

    method = 'EE' if None else method
    if method not in {"FF", "FE", "EE"}:
        raise ValueError("Invalid method")
    
    def NASA_polynomial(T, species):
        coefficients = mixture.species(species).thermo.coeffs
        return coefficients[1:8] if T > coefficients[0] else coefficients[8:15]

    def h(T):
        def h_species(species):
            a1, a2, a3, a4, a5, a6, _ = NASA_polynomial(T, species)
            return (a1 * T + a2 * T ** 2 / 2 + a3 * T ** 3 / 3 + a4 * T ** 4 / 4 + a5 * T ** 5 / 5 + a6) * R

        return np.sum(np.array([h_species(species) * x for species, x in mixture.mole_fraction_dict().items()])) \
               / mixture.mean_molecular_weight

    def cp(T):
        def cp_species(species):
            a1, a2, a3, a4, a5, _, _ = NASA_polynomial(T, species)
            return (a1 + a2 * T + a3 * T ** 2 + a4 * T ** 3 + a5 * T ** 4) * R

        return np.sum(np.array([cp_species(species) * x for species, x in mixture.mole_fraction_dict().items()])) \
               / mixture.mean_molecular_weight

    T1, P1 = mixture.TP
    h1 = h(T1)
    cp1 = cp(T1)
    gamma1 = cp1 / (cp1 - R / mixture.mean_molecular_weight)
    u1 = M * (R / mixture.mean_molecular_weight * gamma1 * T1) ** 0.5
    print(u1)
    v1 = R / mixture.mean_molecular_weight * T1 / P1

    def h_frozen(T):
        def rotational_DOF(species):
            if species in _linear_polyatomic_molecules:
                return 2
            else:
                num_atoms = sum(mixture.species(species).composition.values())
                if num_atoms == 1:
                    return 0
                elif num_atoms == 2:
                    return 2
                else:
                    return 3

        return h(T1) + R / mixture.mean_molecular_weight * (T - T1) \
               * np.sum(np.array([x * (5 + rotational_DOF(species)) / 2
                                  for species, x in mixture.mole_fraction_dict().items()]))

    # Define initial guess for iterative routine from ideal shock equations
    T2_guess = T1 * ((gamma1 * M ** 2 - (gamma1 - 1) / 2) * ((gamma1 - 1) / 2 * M ** 2 + 1)) \
               / (((gamma1 + 1) / 2) ** 2 * M ** 2)
    P2_guess = P1 * (2 * gamma1 * M ** 2 - (gamma1 - 1)) / (gamma1 + 1)

    for i in range(max_iter):
        h2 = h(T2_guess) if method[0] == 'E' else h_frozen(T2_guess)
        cp2 = cp(T2_guess) if method[0] == 'E' else cp1
        v2 = R / mixture.mean_molecular_weight * T2_guess / P2_guess

        f1 = (P2_guess / P1 - 1) + (u1 ** 2 / (P1 * v1)) * (v2 / v1 - 1)
        f2 = ((h2 - h1) / (1 / 2 * u1 ** 2)) + (v2 ** 2 / v1 ** 2 - 1)

        df1_dP2_T2 = (1 / P1) + (u1 ** 2 / (P1 * v1 ** 2)) * (-v2 / P2_guess)
        df1_dT2_P2 = (u1 ** 2 / (P1 * v1 ** 2)) * (v2 / T2_guess)
        df2_dP2_T2 = (2 * v2 / v1 ** 2) * (-v2 / P2_guess)
        df2_dT2_P2 = (2 / u1 ** 2) * cp2 + (2 * v2 / v1 ** 2) * (v2 / T2_guess)

        T2, P2 = tuple((np.array([[T2_guess], [P2_guess]])
                        - np.matmul(np.linalg.inv(np.array([[df1_dT2_P2, df1_dP2_T2], [df2_dT2_P2, df2_dP2_T2]])),
                                    np.array([[f1], [f2]]))).T[0])

        if abs(T2 - T2_guess) / T2_guess < convergence_criteria and \
                abs(P2 - P2_guess) / P2_guess < convergence_criteria:
            break
        elif i == max_iter - 1:
            raise RuntimeError("Convergence criteria not met within the maximum number of iterations")
        else:
            T2_guess = T2
            P2_guess = P2

    h2 = h(T2) if method[0] == 'E' else h_frozen(T2)
    cp2 = cp(T2) if method[0] == 'E' else cp1
    v2 = R / mixture.mean_molecular_weight * T2 / P2  # [m^3/kg]
    gamma2 = cp2 / (cp2 - R / mixture.mean_molecular_weight)
    u2 = u1 * v2 / v1

    # Define initial guess for iterative routine from ideal shock equations
    P5_guess = P2 * ((gamma2 + 1) / (gamma2 - 1) + 2 - P1 / P2) / (1 + (gamma2 + 1) / (gamma2 - 1) * P1 / P2)
    T5_guess = T2 * (P5_guess / P2) * ((gamma2 + 1) / (gamma2 - 1) + P5_guess / P2) \
               / (1 + (gamma2 + 1) / (gamma2 - 1) * P5_guess / P2)

    for i in range(max_iter):
        h5 = h(T5_guess) if method[1] == 'E' else h_frozen(T5_guess)
        cp5 = cp(T5_guess) if method[1] == 'E' else cp1
        v5 = R / mixture.mean_molecular_weight * T5_guess / P5_guess  # [m^3/kg]

        f3 = (P5_guess / P2 - 1) + ((u1 - u2) ** 2 / (P2 * (v5 - v2)))
        f4 = ((h5 - h2) / (1 / 2 * (u1 - u2) ** 2)) + ((v5 + v2) / (v5 - v2))

        df3_dP5_T5 = (1 / P2) + (-(u1 - u2) ** 2 / (P2 * (v5 - v2) ** 2)) * (-v5 / P5_guess)
        df3_dT5_P5 = (-(u1 - u2) ** 2 / (P2 * (v5 - v2) ** 2)) * (v5 / T5_guess)
        df4_dP5_T5 = (-2 * v2 / (v5 - v2) ** 2) * (-v5 / P5_guess)
        df4_dT5_P5 = (1 / (1 / 2 * (u1 - u2) ** 2)) * cp5 + (-2 * v2 / (v5 - v2) ** 2) * (v5 / T5_guess)

        T5, P5 = tuple((np.array([[T5_guess], [P5_guess]])
                        - np.matmul(np.linalg.inv(np.array([[df3_dT5_P5, df3_dP5_T5], [df4_dT5_P5, df4_dP5_T5]])),
                                    np.array([[f3], [f4]]))).T[0])

        if abs(T5 - T5_guess) / T5_guess < convergence_criteria and \
                abs(P5 - P5_guess) / P5_guess < convergence_criteria:
            break
        elif i == max_iter - 1:
            raise RuntimeError("Convergence criteria not met within the maximum number of iterations")
        else:
            T5_guess = T5
            P5_guess = P5

    return T2, P2, T5, P5
