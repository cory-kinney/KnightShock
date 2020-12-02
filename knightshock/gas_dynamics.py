"""Shock tube gas dynamics equations and solvers"""

import numpy as np
import cantera as ct
from scipy import optimize


def normal_shock_pressure_ratio(M, gamma):
    """Calculates the pressure ratio across a normal shock under ideal shock assumptions

    $$ \\frac {P_2} {P_1} = \\frac {2 \\gamma M^2-(\\gamma - 1)} {\\gamma + 1} $$

    Parameters
    ----------
    M : float
        Mach number
    gamma : float
        specific heat ratio

    Returns
    -------
    float

    Raises
    ------
    ValueError
        `M` is less than one

    """

    if M < 1:
        raise ValueError("M must be greater than or equal to one")

    return (2 * gamma * M ** 2 - (gamma - 1)) / (gamma + 1)


def normal_shock_temperature_ratio(M, gamma):
    """Calculates the temperature ratio across a normal shock under ideal shock assumptions

    $$ \\frac {T_2} {T_1} = \\frac {\\left(\\gamma M^2 - \\frac {\\gamma - 1} {2}\\right)
    \\left(\\frac{\\gamma - 1}{2} M^2 + 1\\right)} {\\left(\\frac {\\gamma + 1} {2}\\right)^2 M^2} $$

    Parameters
    ----------
    M : float
        Mach number
    gamma : float
        specific heat ratio

    Returns
    -------
    float

    Raises
    ------
    ValueError
        `M` is less than one

    """

    if M < 1:
        raise ValueError("M must be greater than or equal to one")

    return ((gamma * M ** 2 - (gamma - 1) / 2) * ((gamma - 1) / 2 * M ** 2 + 1)) / (((gamma + 1) / 2) ** 2 * M ** 2)


def reflected_shock_Mach_number(M, gamma):
    """Calculates the reflected shock Mach number under ideal assumptions

    $$ \\frac {M_r} {M_r^2 - 1} = \\frac {M_s} {M_s^2 - 1} \\sqrt {1 + \\frac {2 \\left(\\gamma - 1 \\right)}
    {\\left(\\gamma + 1 \\right)^2} \\left(M_s^2 - 1 \\right) \\left(\\gamma + \\frac {1} {M_s^2} \\right)} $$

    Parameters
    ----------
    M : float
        incident shock Mach number
    gamma : float
        specific heat ratio

    Returns
    -------
    float
        reflected shock Mach number

    Raises
    ------
    ValueError
        `M` is not greater than one

    """

    if not M > 1:
        raise ValueError("M must be greater than one")

    a = M / (M ** 2 - 1) * (1 + 2 * (gamma - 1) / (gamma + 1) ** 2 * (M ** 2 - 1) * (gamma + 1 / M ** 2)) ** 0.5

    return (1 + (1 + 4 * a ** 2) ** 0.5) / a / 2


def T2_ideal(T1, M, gamma):
    """Calculates the ideal post-incident-shock temperature using
    `knightshock.gas_dynamics.normal_shock_temperature_ratio`.

    Parameters
    ----------
    T1 : float
        initial temperature
    M : float
        incident shock Mach number
    gamma : float
        specific heat ratio

    Returns
    -------
    T2 : float
        ideal post-incident-shock temperature

    """
    return T1 * normal_shock_temperature_ratio(M, gamma)


def P2_ideal(P1, M, gamma):
    """Calculates the ideal post-incident-shock pressure using `knightshock.gas_dynamics.normal_shock_pressure_ratio`.

    Parameters
    ----------
    P1 : float
        initial pressure
    M : float
        incident shock Mach number
    gamma : float
        specific heat ratio

    Returns
    -------
    P2 : float
        ideal post-incident-shock pressure

    """
    return P1 * normal_shock_pressure_ratio(M, gamma)


def T5_ideal(T1, M, gamma):
    """Calculates the ideal post-reflected-shock temperature using
    `knightshock.gas_dynamics.normal_shock_temperature_ratio` and `knightshock.gas_dynamics.reflected_Mach_number`.

    Parameters
    ----------
    T1 : float
        initial temperature
    M : float
        incident shock Mach number
    gamma : float
        specific heat ratio

    Returns
    -------
    T5 : float
        ideal post-reflected-shock temperature

    """
    return T2_ideal(T1, M, gamma) * normal_shock_temperature_ratio(reflected_shock_Mach_number(M, gamma), gamma)


def P5_ideal(P1, M, gamma):
    """Calculates the ideal post-reflected-shock pressure using `knightshock.gas_dynamics.normal_shock_pressure_ratio`
    and `knightshock.gas_dynamics.reflected_Mach_number`.

    Parameters
    ----------
    P1 : float
        initial pressure
    M : float
        incident shock Mach number
    gamma : float
        specific heat ratio

    Returns
    -------
    P5 : float
        ideal post-reflected-shock pressure

    """
    return P2_ideal(P1, M, gamma) * normal_shock_pressure_ratio(reflected_shock_Mach_number(M, gamma), gamma)


def shock_conditions_FROSH(T1, P1, M, *, thermo, max_iter=1000, convergence_criteria=1e-6):
    """Implementation of the FROzen SHock (FROSH) algorithm for calculating post-incident-shock and post-reflected-shock
    conditions from initial conditions and shock velocity. The two-dimensional iterative Newton-Raphson algorithm
    implemented for solving the Rankine-Hugoniot relations is derived by Campbell et al.[^1]. Uses
    `knightshock.gas_dynamics.T2_ideal` and `knightshock.gas_dynamics.P2_ideal` as initial guesses for algorithm
    stability.

    Parameters
    ----------
    T1 : float
        initial temperature [K]
    P1 : float
        initial pressure [Pa]
    M : float
        incident shock Mach number
    thermo : `cantera.ThermoPhase`
        thermodynamic properties for driven mixture
    max_iter : int, optional
        maximum number of iterations for solvers
    convergence_criteria : float, optional
        maximum absolute difference between iterations for convergence

    Returns
    -------
    T2 : float
        post-incident-shock temperature [K]
    P2 : float
        post-incident-shock pressure [Pa]
    T5 : float
        post-reflected-shock temperature [K]
    P5 : float
        post-reflected-shock pressure [Pa]

    Raises
    ------
    ValueError
        `T1` is not greater than zero
    ValueError
        `P1` is not greater than zero
    ValueError
        `M` is not greater than zero
    ValueError
        Incident shock Mach number is not greater than one
    RuntimeError
        Convergence criteria is not met within the maximum number of iterations

    References
    ----------
    [^1]: Campbell, Owen, Davidson, Hanson: Dependence of Calculated Postshock Thermodynamic Variables on Vibrational
    Equilibrium and Input Uncertainty. AIAA Journal of Thermophysics and Heat Transfer 31, 586-608 (2017).
    https://doi.org/10.2514/1.T4952

    """

    if not T1 > 0:
        raise ValueError("T1 must be positive")
    if not P1 > 0:
        raise ValueError("P1 must be positive")
    if not M > 1:
        raise ValueError("M must be greater than 1")

    # Get relevant properties for mixture at T1 and P1
    thermo.TP = T1, P1

    gamma1 = thermo.cp_mass / thermo.cv_mass
    a1 = (ct.gas_constant / thermo.mean_molecular_weight * gamma1 * T1) ** 0.5  # [m/s]
    v1 = 1 / thermo.density_mass  # [m^3/kg]
    h1 = thermo.enthalpy_mass  # [J/kg]

    # Calculates ideal P2 and T2 for initial guesses
    u = a1 * M  # [m/s]

    T2_guess = T2 = T2_ideal(T1, M, gamma1)  # [K]
    P2_guess = P2 = P2_ideal(P1, M, gamma1)  # [Pa]

    # Iterate to find P2 and T2
    for i in range(max_iter):
        thermo.TP = T2_guess, P2_guess

        # Get relevant properties for the mixture at the guessed P2 and T2 values
        v2 = 1 / thermo.density_mass  # [m^3/kg]
        h2 = thermo.enthalpy_mass  # [J/kg]
        cp2 = thermo.cp_mass  # [J/kg/K]

        # Evaluate functions
        f1 = (P2_guess / P1 - 1) + (u ** 2 / (P1 * v1)) * (v2 / v1 - 1)
        f2 = ((h2 - h1) / (1 / 2 * u ** 2)) + (v2 ** 2 / v1 ** 2 - 1)

        df1_dP2_T2 = (1 / P1) + (u ** 2 / (P1 * v1 ** 2)) * (-v2 / P2_guess)
        df1_dT2_P2 = (u ** 2 / (P1 * v1 ** 2)) * (v2 / T2_guess)
        df2_dP2_T2 = (2 * v2 / v1 ** 2) * (-v2 / P2_guess)
        df2_dT2_P2 = (2 / u ** 2) * cp2 + (2 * v2 / v1 ** 2) * (v2 / T2_guess)

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
        if i == max_iter - 1:
            raise RuntimeError("Convergence criteria not met within the maximum number of iterations")

    # Get relevant properties for mixture at T2 and P2
    thermo.TP = T2, P2

    gamma2 = thermo.cp_mass / thermo.cv_mass
    v2 = 1 / thermo.density_mass  # [m^3/kg]
    h2 = thermo.enthalpy_mass  # [J/kg]

    # Calculate u2 using continuity
    u2 = u * v2 / v1  # [m/s]

    # Calculates ideal P5 and T5 for initial guesses
    P5 = P2 * ((gamma2 + 1) / (gamma2 - 1) + 2 - P1 / P2) / (1 + (gamma2 + 1) / (gamma2 - 1) * P1 / P2)  # [Pa]
    T5 = T2 * (P5 / P2) * ((gamma2 + 1) / (gamma2 - 1) + P5 / P2) / (1 + (gamma2 + 1) / (gamma2 - 1) * P5 / P2)  # [K]

    T5_guess = T5
    P5_guess = P5

    # Iterate to find P5 and T5
    for i in range(max_iter):
        # Get relevant properties for the mixture at the guessed P5 and T5 values
        thermo.TP = T5_guess, P5_guess

        v5 = 1 / thermo.density_mass  # [m^3/kg]
        h5 = thermo.enthalpy_mass  # [J/kg]
        cp5 = thermo.cp_mass  # [J/kg/K]

        # Evaluate functions
        f3 = (P5_guess / P2 - 1) + ((u - u2) ** 2 / (P2 * (v5 - v2)))
        f4 = ((h5 - h2) / (1 / 2 * (u - u2) ** 2)) + ((v5 + v2) / (v5 - v2))

        df3_dP5_T5 = (1 / P2) + (-(u - u2) ** 2 / (P2 * (v5 - v2) ** 2)) * (-v5 / P5_guess)
        df3_dT5_P5 = (-(u - u2) ** 2 / (P2 * (v5 - v2) ** 2)) * (v5 / T5_guess)
        df4_dP5_T5 = (-2 * v2 / (v5 - v2) ** 2) * (-v5 / P5_guess)
        df4_dT5_P5 = (1 / (1 / 2 * (u - u2) ** 2)) * cp5 + (-2 * v2 / (v5 - v2) ** 2) * (v5 / T5_guess)

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
        if i == max_iter - 1:
            raise RuntimeError("Convergence criteria not met within the maximum number of iterations")

    return T2, P2, T5, P5


def shock_tube_flow_properties(M, T1, T4, MW1, MW4, gamma1, gamma4, *, area_ratio=1):
    """Calculates the characteristics associated with the formation of an ideal incident shock for a shock tube flow
    with monotonic convergence at the diaphragm section. The equations derived by Alpher and White[^1] were optimized
    for use with `scipy.optimize.root_scalar`.

    Parameters
    ----------
    M : float
        incident shock Mach number
    T1 : float
        absolute temperature of driven gas
    T4 : float
        absolute temperature of driver gas
    MW1 : float
        mean molecular weight of driven gas
    MW4 : float
        mean molecular weight of driver gas
    gamma1 : float
        specific heat ratio of driven gas
    gamma4 : float
        specific heat ratio of driver gas
    area_ratio : float, optional
        ratio of driver to driven area

    Returns
    -------
    P4_P1, M3a, Me, M3 : float, float, float, float
        ratio of driver to driven initial pressures
        Mach number of flow before nozzle
        Mach number of flow at nozzle exit
        Mach number of fully expanded driver flow

    Raises
    ------
    ValueError
        `M` is not greater than one
    ValueError
        `area_ratio` is less than one
    RuntimeError
        Solution does not converge

    References
    ----------
    [^1]: Alpher, R., & White, D. (1958). Flow in shock tubes with area change at the diaphragm section. Journal of
    Fluid Mechanics, 3(5), 457-470. https://doi.org/10.1017/S0022112058000124

    """

    if not M > 1:
        raise ValueError("M must be greater than one")
    if area_ratio < 1:
        raise ValueError("Area ratio must be greater than or equal to one")

    a1_u2 = (gamma1 + 1) / 2 * M / (M ** 2 - 1)
    a4_a1 = (gamma4 / gamma1 * MW1 / MW4 * T4 / T1) ** 0.5
    P2_P1 = normal_shock_pressure_ratio(M, gamma1)

    def equivalence_factor(_M3a, _Me):
        return (((2 + (gamma4 - 1) * _M3a ** 2) / (2 + (gamma4 - 1) * _Me ** 2)) ** 0.5
                * (2 + (gamma4 - 1) * _Me) / (2 + (gamma4 - 1) * _M3a)) ** (2 * gamma4 / (gamma4 - 1))

    def calc_M3(_M3a, _Me):
        return 1 / (a1_u2 * a4_a1 * equivalence_factor(_M3a, _Me) ** ((gamma4 - 1) / gamma4 / 2) - (gamma4 - 1) / 2)

    def solve_M3a(_Me):
        def area_ratio_error(_M3a):
            return area_ratio * _M3a - _Me * ((2 + (gamma4 - 1) * _M3a ** 2) / (2 + (gamma4 - 1) * _Me ** 2)) \
                   ** ((gamma4 + 1) / (gamma4 - 1) / 2)

        _root_results = optimize.root_scalar(area_ratio_error, bracket=[0, 1])
        if not _root_results.converged:
            raise RuntimeError("Root finding routine for M3a did not converge")
        return _root_results.root

    # Assume supersonic case (Me = 1)
    Me = 1
    M3a = solve_M3a(Me)
    M3 = calc_M3(M3a, Me)

    # Subsonic case (M3 = Me)
    if M3 < 1:
        root_results = optimize.root_scalar(lambda _M3: _M3 - calc_M3(solve_M3a(_Me=_M3), _Me=_M3), bracket=[0, 1])
        if not root_results.converged:
            raise RuntimeError("Root finding routine for M3 did not converge")
        M3 = root_results.root
        Me = M3
        M3a = solve_M3a(Me)

    P4_P1 = P2_P1 / equivalence_factor(M3a, Me) * (1 + (gamma4 - 1) / 2 * M3) ** (2 * gamma4 / (gamma4 - 1))

    return P4_P1, M3a, Me, M3


def tailored_mixture(M, T1, T4, MW1, MW4, gamma1, gamma4, *, area_ratio=1):
    """Calculates the species mole fractions that tailor the interaction between the reflected shock wave and the
    contact surface. The mole fractions for which the output of
    `knightshock.gas_dynamics.shock_tube_flow_properties` satisfies the tailoring condition described by Hong et al.[^1]
    is iteratively calculated using `scipy.optimize.root_scalar`.

    Parameters
    ----------
    M : float
        incident shock Mach number
    T1 : float
        absolute temperature of driven gas
    T4 : float
        absolute temperature of driver gas
    MW1 : float
        mean molecular weight of driven gas
    MW4 : tuple
        mean molecular weights of driver species used for tailoring
    gamma1 : float
        specific heat ratio of driven gas
    gamma4 : tuple
        specific heat ratios of driver species used for tailoring
    area_ratio : float
        ratio of driver to driven area

    Returns
    -------
    x0 : float
        mole fraction of first driver species
    x1 : float
        mole fraction of second driver species

    Raises
    ------
    ValueError
        Tailoring solution does not exist for the given inputs
    RuntimeError
        Root finding routine does not converge

    References
    ----------
    [^1]: Hong, Z., Davidson, D.F. & Hanson, R.K. Contact surface tailoring condition for shock tubes with different
    driver and driven section diameters. Shock Waves 19, 331–336 (2009). https://doi.org/10.1007/s00193-009-0212-z

    """

    P5_P2 = (M ** 2 * (3 * gamma1 - 1) - 2 * (gamma1 - 1)) / (M ** 2 * (gamma1 - 1) + 2)

    a0 = gamma4[0] / (gamma4[0] - 1)
    a1 = gamma4[1] / (gamma4[1] - 1)

    def _M3_error(_x0):
        _x1 = 1 - _x0

        _MW4 = _x0 * MW4[0] + _x1 * MW4[1]
        _gamma4 = (_x0 * a0 + _x1 * a1) / (_x0 * a0 + _x1 * a1 - 1)

        _alpha4 = (_gamma4 + 1) / (_gamma4 - 1)
        _M3_tailored = (_alpha4 - 1) * (P5_P2 - 1) / ((1 + _alpha4) * (1 + _alpha4 * P5_P2)) ** 0.5

        return _M3_tailored - shock_tube_flow_properties(M, T1, T4, MW1, _MW4, gamma1, _gamma4,
                                                         area_ratio=area_ratio)[3]

    try:
        root_results = optimize.root_scalar(lambda _x0: _M3_error(_x0), bracket=[0, 1])
    except ValueError:
        raise ValueError("Tailoring solution does not exist for the given inputs")

    if not root_results.converged:
        raise RuntimeError("Root finding routine for species mole fractions did not converge")

    x0 = root_results.root
    x1 = 1 - x0

    return x0, x1

