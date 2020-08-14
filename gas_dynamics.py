import cantera as ct
import numpy as np


def T2_P2_constant_properties(T1, P1, M1, gamma1):
    P2 = P1 * (2 * gamma1 * M1 ** 2 - (gamma1 - 1)) / (gamma1 + 1)
    T2 = T1 * ((gamma1 * M1 ** 2 - (gamma1 - 1) / 2) * ((gamma1 - 1) / 2 * M1 ** 2 + 1)) / (((gamma1 + 1) / 2) ** 2 * M1 ** 2)
    return T2, P2


def T5_P5_constant_properties(P1, T2, P2, gamma2):
    P5 = P2 * ((gamma2 + 1) / (gamma2 - 1) + 2 - P1 / P2) / (1 + (gamma2 + 1) / (gamma2 - 1) * P1 / P2)
    T5 = T2 * (P5 / P2) * ((gamma2 + 1) / (gamma2 - 1) + P5 / P2) / (1 + (gamma2 + 1) / (gamma2 - 1) * P5 / P2)
    return T5, P5


def FROSH(T1, P1, u1, thermo, *, max_iterations=1000, convergence_criteria=1e-6):
    """
    Implementation of the FROzen SHock algorithm for calculating post-incident-shock and post-reflected-shock conditions
    from initial conditions and shock velocity for a shock tube experiment

    Args:
        T1: initial temperature in driven section [K]
        P1: initial pressure in driven section [Pa]
        u1: incident shock velocity [m/s]
        thermo: Cantera ThermoPhase object for calculating mixture properties at different temperatures and pressures
        max_iterations: maximum number of iterations for calculating region 2 and 5 temperature and pressure conditions
        convergence_criteria: maximum change between iterations for convergence

    Returns:
        T2: temperature in the post-incident-shock region [K]
        P2: pressure in the post-incident-shock region [Pa]
        T5: temperature in the post-reflected-shock region [K]
        P5: pressure in the post-reflected-shock region [Pa]

    """

    # Get relevant properties for mixture at T1 and P1
    thermo.TP = T1, P1

    gamma1 = thermo.cp_mass / thermo.cv_mass
    a1 = (ct.gas_constant / thermo.mean_molecular_weight * gamma1 * T1)**0.5  # [m/s]
    v1 = 1 / thermo.density_mass  # [m^3/kg]
    h1 = thermo.enthalpy_mass  # [J/kg]

    # Calculates ideal P2 and T2 for initial guesses
    M1 = u1 / a1
    T2, P2 = T2_P2_constant_properties(T1=T1, P1=P1, M1=M1, gamma1=gamma1)  # [K], [Pa]
    T2_guess = T2
    P2_guess = P2

    # Iterate to find P2 and T2
    for i in range(max_iterations):
        # Get relevant properties for the mixture at the guessed P2 and T2 values
        thermo.TP = T2_guess, P2_guess

        v2 = 1 / thermo.density_mass  # [m^3/kg]
        h2 = thermo.enthalpy_mass  # [J/kg]
        cp2 = thermo.cp_mass  # [J/kg/K]

        # Evaluate functions
        f1 = (P2_guess / P1 - 1) + (u1 ** 2 / (P1 * v1)) * (v2 / v1 - 1)
        f2 = ((h2 - h1) / (1 / 2 * u1 ** 2)) + (v2 ** 2 / v1 ** 2 - 1)

        df1_dP2_T2 = (1 / P1) + (u1 ** 2 / (P1 * v1 ** 2)) * (-v2 / P2_guess)
        df1_dT2_P2 = (u1 ** 2 / (P1 * v1 ** 2)) * (v2 / T2_guess)
        df2_dP2_T2 = (2 * v2 / v1 ** 2) * (-v2 / P2_guess)
        df2_dT2_P2 = (2 / u1 ** 2) * cp2 + (2 * v2 / v1 ** 2) * (v2 / T2_guess)

        T2, P2 = tuple((np.array([[T2_guess], [P2_guess]])
                        - np.matmul(np.linalg.inv(np.array([[df1_dT2_P2, df1_dP2_T2], [df2_dT2_P2, df2_dP2_T2]])),
                                    np.array([[f1], [f2]]))).T[0])

        # Break the loop when convergence criteria is met
        if abs(T2 - T2_guess) / T2_guess < convergence_criteria and \
           abs(P2 - P2_guess) / P2_guess < convergence_criteria:
            break

        # Update the P2 and T2 guesses
        T2_guess = T2
        P2_guess = P2

        # Raise an error if the maximum number of iterations has been reached
        if i == max_iterations - 1:
            raise RuntimeError("Convergence criteria not met within the maximum number of iterations")

    # Get relevant properties for mixture at T2 and P2
    thermo.TP = T2, P2

    gamma2 = thermo.cp_mass / thermo.cv_mass
    v2 = 1 / thermo.density_mass  # [m^3/kg]
    h2 = thermo.enthalpy_mass  # [J/kg]

    # Calculate u2 using continuity
    u2 = u1 * v2 / v1  # [m/s]

    # Calculates ideal P5 and T5 for initial guesses
    T5, P5 = T5_P5_constant_properties(P1=P1, T2=T2, P2=P2, gamma2=gamma2)  # [K], [Pa]
    T5_guess = T5
    P5_guess = P5

    # Iterate to find P5 and T5
    for i in range(max_iterations):
        # Get relevant properties for the mixture at the guessed P5 and T5 values
        thermo.TP = T5_guess, P5_guess

        v5 = 1 / thermo.density_mass  # [m^3/kg]
        h5 = thermo.enthalpy_mass  # [J/kg]
        cp5 = thermo.cp_mass  # [J/kg/K]

        # Evaluate functions
        f3 = (P5_guess / P2 - 1) + ((u1 - u2) ** 2 / (P2 * (v5 - v2)))
        f4 = ((h5 - h2) / (1 / 2 * (u1 - u2) ** 2)) + ((v5 + v2) / (v5 - v2))

        df3_dP5_T5 = (1 / P2) + (-(u1 - u2) ** 2 / (P2 * (v5 - v2) ** 2)) * (-v5 / P5_guess)
        df3_dT5_P5 = (-(u1 - u2) ** 2 / (P2 * (v5 - v2) ** 2)) * (v5 / T5_guess)
        df4_dP5_T5 = (-2 * v2 / (v5 - v2) ** 2) * (-v5 / P5_guess)
        df4_dT5_P5 = (1 / (1 / 2 * (u1 - u2) ** 2)) * cp5 + (-2 * v2 / (v5 - v2) ** 2) * (v5 / T5_guess)

        T5, P5 = tuple((np.array([[T5_guess], [P5_guess]])
                        - np.matmul(np.linalg.inv(np.array([[df3_dT5_P5, df3_dP5_T5], [df4_dT5_P5, df4_dP5_T5]])),
                                    np.array([[f3], [f4]]))).T[0])

        # Break the loop when convergence criteria is met
        if abs(T5 - T5_guess) / T5_guess < convergence_criteria and \
           abs(P5 - P5_guess) / P5_guess < convergence_criteria:
            break

        # Update the P5 and T5 guesses
        T5_guess = T5
        P5_guess = P5

        # Raise an error if the maximum number of iterations has been reached
        if i == max_iterations - 1:
            raise RuntimeError("Convergence criteria not met within the maximum number of iterations")

    return T2, P2, T5, P5
