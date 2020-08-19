from scipy import optimize
import cantera as ct
import numpy as np

R = 8314.4621  # [J/kmol/K]


def T2_P2_constant_properties(T1, P1, M1, gamma1):
    P2 = P1 * (2 * gamma1 * M1 ** 2 - (gamma1 - 1)) / (gamma1 + 1)
    T2 = T1 * ((gamma1 * M1 ** 2 - (gamma1 - 1) / 2) * ((gamma1 - 1) / 2 * M1 ** 2 + 1)) \
         / (((gamma1 + 1) / 2) ** 2 * M1 ** 2)
    return T2, P2


def T5_P5_constant_properties(P1, T2, P2, gamma2):
    P5 = P2 * ((gamma2 + 1) / (gamma2 - 1) + 2 - P1 / P2) / (1 + (gamma2 + 1) / (gamma2 - 1) * P1 / P2)
    T5 = T2 * (P5 / P2) * ((gamma2 + 1) / (gamma2 - 1) + P5 / P2) / (1 + (gamma2 + 1) / (gamma2 - 1) * P5 / P2)
    return T5, P5


def M_constant_properties():
    pass


def _area_ratio(gamma4, M3a, Me):
    return Me / M3a * ((2 + (gamma4 - 1) * M3a ** 2) / (2 + (gamma4 - 1) * Me ** 2)) ** (
            (gamma4 + 1) / (gamma4 - 1) / 2)


def _equivalence_factor(gamma4, M3a, Me):
    return (((2 + (gamma4 - 1) * M3a ** 2) / (2 + (gamma4 - 1) * Me ** 2)) ** 0.5
            * (2 + (gamma4 - 1) * Me) / (2 + (gamma4 - 1) * M3a)) ** (2 * gamma4 / (gamma4 - 1))


def _M3_ideal_expansion(gamma1, gamma4, a1, a4, Ms, M3a, Me):
    return ((gamma1 + 1) / 2 * Ms / (Ms ** 2 - 1) * a4 / a1
            * _equivalence_factor(gamma4, M3a, Me) ** ((gamma4 - 1) / gamma4 / 2) - (gamma4 - 1) / 2) ** -1


def tailored_mixture_constant_properties(Ms, gamma1, MW1, T1, species, specific_heat_ratios, molecular_weights, T4,
                                         *, area_ratio=1, minimum_Mach=1e-3):
    a1 = (R / MW1 * gamma1 * T1) ** 0.5

    a = specific_heat_ratios[0] / (specific_heat_ratios[0] - 1)
    b = specific_heat_ratios[1] / (specific_heat_ratios[1] - 1)

    def _M3_tailoring_error(x_guess):
        gamma4 = (x_guess * a + (1 - x_guess) * b) / (x_guess * a + (1 - x_guess) * b - 1)
        MW4 = x_guess * molecular_weights[0] + (1 - x_guess) * molecular_weights[1]
        a4 = (R / MW4 * gamma4 * T4) ** 0.5
        alpha4 = (gamma4 + 1) / (gamma4 - 1)
        p52 = (Ms ** 2 * (3 * gamma1 - 1) - 2 * (gamma1 - 1)) / (Ms ** 2 * (gamma1 - 1) + 2)
        M3_tailored = (alpha4 - 1) * (p52 - 1) / ((1 + alpha4) * (1 + alpha4 * p52)) ** 0.5

        def _M3a_error(M3a_guess, Me):
            return area_ratio - _area_ratio(gamma4, M3a_guess, Me)

        # Assume supersonic case (Me = 1, M3 > 1)
        M3a = optimize.root_scalar(lambda M3a_guess: _M3a_error(M3a_guess, 1), bracket=[minimum_Mach, 1]).root
        M3 = _M3_ideal_expansion(gamma1, gamma4, a1, a4, Ms, M3a, 1)

        # Subsonic case (M3 = Me)
        if M3 < 1:
            def _M3_expansion_error(M3_guess):
                M3a = optimize.root_scalar(lambda M3a_guess: _M3a_error(M3a_guess, M3_guess),
                                           bracket=[minimum_Mach, M3_guess]).root
                return M3_guess - _M3_ideal_expansion(gamma1, gamma4, a1, a4, Ms, M3a, M3_guess)

            M3 = optimize.root_scalar(_M3_expansion_error, bracket=[minimum_Mach, 1]).root

        return M3 - M3_tailored

    x = optimize.root_scalar(_M3_tailoring_error, bracket=[0, 1]).root
    return "{}: {}, {}: {}".format(species[0], x, species[1], 1 - x)


def FROSH(T1, P1, u1, mixture, *, mechanism, max_iterations=1000, convergence_criteria=1e-6):
    """
    Implementation of the FROzen SHock (FROSH) algorithm for calculating post-incident-shock and post-reflected-shock
    conditions from initial conditions and shock velocity for a shock tube experiment

    Args:
        T1: initial temperature in driven section [K]
        P1: initial pressure in driven section [Pa]
        u1: incident shock velocity [m/s]
        mixture: string of comma-separated species: mole fraction pairs
        mechanism: file name of mechanism for thermodynamic property calculation with Cantera
        max_iterations: maximum number of iterations for calculating region 2 and 5 temperature and pressure conditions
        convergence_criteria: maximum change between iterations for convergence

    Returns:
        T2: temperature in the post-incident-shock region [K]
        P2: pressure in the post-incident-shock region [Pa]
        T5: temperature in the post-reflected-shock region [K]
        P5: pressure in the post-reflected-shock region [Pa]

    Reference:


    """

    thermo = ct.ThermoPhase(mechanism)

    # Get relevant properties for mixture at T1 and P1
    thermo.TPX = T1, P1, mixture

    gamma1 = thermo.cp_mass / thermo.cv_mass
    a1 = (R / thermo.mean_molecular_weight * gamma1 * T1) ** 0.5  # [m/s]
    v1 = 1 / thermo.density_mass  # [m^3/kg]
    h1 = thermo.enthalpy_mass  # [J/kg]

    # Calculates ideal P2 and T2 for initial guesses
    M1 = u1 / a1
    T2, P2 = T2_P2_constant_properties(T1=T1, P1=P1, M1=M1, gamma1=gamma1)  # [K], [Pa]
    T2_guess = T2
    P2_guess = P2

    # Iterate to find P2 and T2
    for i in range(max_iterations):
        thermo.TP = T2_guess, P2_guess

        # Get relevant properties for the mixture at the guessed P2 and T2 values
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


def M_FROSH():
    pass


def tailored_mixture_FROSH():
    pass
