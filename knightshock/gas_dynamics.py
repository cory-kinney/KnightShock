"""Shock tube gas dynamics equations and solvers"""

from scipy import optimize
from typing import Tuple
import cantera as ct
import numpy as np


def frozen_shock_conditions(M: float, gas: ct.ThermoPhase, method: str = None, *, max_iter: int = 1000,
                            epsilon: float = 1e-6) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Calculates the region 2 and region 5 conditions using the FROzen SHock (FROSH) algorithm. The two-dimensional
    iterative Newton-Raphson solver for the Rankine-Hugoniot relations is derived by Campbell et al.[^1].

    Parameters
    ----------
    M
        incident shock Mach number
    gas
        driven gas at initial conditions
    method
        method used for thermodynamic property calculation assumptions
        - 'FF' (Frozen-Frozen)
        - 'FE' (Frozen-Equilibrium)
        - 'EE' (Equilibrium-Equilibrium)
    max_iter
        maximum number of iterations for region conditions iterative solvers
    epsilon
        relative error convergence criteria for region conditions iterative solvers

    Returns
    -------
    shock_conditions:
        T2, P2, T5, P5

    References
    ----------
    [^1]: Campbell, Owen, Davidson, Hanson: Dependence of Calculated Postshock Thermodynamic Variables on Vibrational
    Equilibrium and Input Uncertainty. AIAA Journal of Thermophysics and Heat Transfer 31, 586-608 (2017).
    https://doi.org/10.2514/1.T4952

    """

    method = 'EE' if method is None else method
    if method not in {"FF", "FE", "EE"}:
        raise ValueError("Invalid method - valid options are \'FF\', \'FE\', and \'EE\'")

    R = ct.gas_constant / gas.mean_molecular_weight

    # Get properties of mixture at region 1 conditions
    T1, P1 = gas.TP

    h1 = gas.enthalpy_mass
    cp1 = gas.cp_mass
    gamma1 = cp1 / (cp1 - R)
    u1 = M * (R * gamma1 * T1) ** 0.5
    v1 = R * T1 / P1

    # Define functions for frozen enthalpy computation
    _linear_polyatomic_molecules = frozenset({
        "BeCl2", "C3", "CCN", "CCO", "CH2", "CNC", "CNN", "CO2", "F2Xe", "HCC", "HCN", "N2O",
        "N3", "NCN", "NCO", "OCS", "C2H2", "C2N2", "CCCC", "HCCCl", "C3H2", "C3N2", "C3O2"
    })

    def rotational_DOF(species):
        """Gets the number of rotational degrees of freedom of a species molecule using Tables 6 and 7 in
        reference [^1]"""
        if species in _linear_polyatomic_molecules:
            return 2
        else:
            num_atoms = sum(gas.species(species).composition.values())
            if num_atoms == 1:
                return 0
            elif num_atoms == 2:
                return 2
            else:
                return 3

    translational_rotational_energy = np.sum(np.array([x * (5 + rotational_DOF(species)) / 2
                                                       for species, x in gas.mole_fraction_dict().items()]))

    def h_frozen(T):
        """Calculates the vibrationally frozen enthalpy of the mixture"""
        return h1 + R * (T - T1) * translational_rotational_energy

    # Calculate initial guess for region 2 conditions from ideal shock equations
    T2_guess = T1 * ((gamma1 * M ** 2 - (gamma1 - 1) / 2) * ((gamma1 - 1) / 2 * M ** 2 + 1)) \
               / (((gamma1 + 1) / 2) ** 2 * M ** 2)
    P2_guess = P1 * (2 * gamma1 * M ** 2 - (gamma1 - 1)) / (gamma1 + 1)

    for i in range(max_iter):
        # Get properties of mixture at guessed region 2 conditions
        gas.TP = T2_guess, P2_guess

        h2 = gas.enthalpy_mass if method[0] == 'E' else h_frozen(T2_guess)
        cp2 = gas.cp_mass if method[0] == 'E' else cp1
        v2 = R * T2_guess / P2_guess

        # Calculate iterative solver terms
        f1 = (P2_guess / P1 - 1) + (u1 ** 2 / (P1 * v1)) * (v2 / v1 - 1)
        f2 = ((h2 - h1) / (1 / 2 * u1 ** 2)) + (v2 ** 2 / v1 ** 2 - 1)

        df1_dP2_T2 = (1 / P1) + (u1 ** 2 / (P1 * v1 ** 2)) * (-v2 / P2_guess)
        df1_dT2_P2 = (u1 ** 2 / (P1 * v1 ** 2)) * (v2 / T2_guess)
        df2_dP2_T2 = (2 * v2 / v1 ** 2) * (-v2 / P2_guess)
        df2_dT2_P2 = (2 / u1 ** 2) * cp2 + (2 * v2 / v1 ** 2) * (v2 / T2_guess)

        # Calculate next iteration for region 2 conditions
        T2, P2 = tuple((np.array([[T2_guess], [P2_guess]])
                        - np.matmul(np.linalg.inv(np.array([[df1_dT2_P2, df1_dP2_T2], [df2_dT2_P2, df2_dP2_T2]])),
                                    np.array([[f1], [f2]]))).T[0])

        # Check for convergence and maximum iteration conditions
        if abs(T2 - T2_guess) / T2_guess < epsilon and abs(P2 - P2_guess) / P2_guess < epsilon:
            break
        elif i == max_iter - 1:
            raise RuntimeError("Convergence criteria not met within the maximum number of iterations")
        else:
            T2_guess = T2
            P2_guess = P2

    # Get properties of mixture at solved region 2 conditions
    gas.TP = T2, P2

    h2 = gas.enthalpy_mass if method[0] == 'E' else h_frozen(T2)
    cp2 = gas.cp_mass if method[0] == 'E' else cp1
    v2 = R * T2 / P2
    gamma2 = cp2 / (cp2 - R)
    u2 = u1 * v2 / v1

    # Calculate initial guess for region 5 conditions from ideal shock equations
    P5_guess = P2 * ((gamma2 + 1) / (gamma2 - 1) + 2 - P1 / P2) / (1 + (gamma2 + 1) / (gamma2 - 1) * P1 / P2)
    T5_guess = T2 * (P5_guess / P2) * ((gamma2 + 1) / (gamma2 - 1) + P5_guess / P2) \
               / (1 + (gamma2 + 1) / (gamma2 - 1) * P5_guess / P2)

    for i in range(max_iter):
        # Get properties of mixture at guessed region 5 conditions
        gas.TP = T5_guess, P5_guess
        h5 = gas.enthalpy_mass if method[1] == 'E' else h_frozen(T5_guess)
        cp5 = gas.cp_mass if method[1] == 'E' else cp1
        v5 = R * T5_guess / P5_guess

        # Calculate iterative solver terms
        f3 = (P5_guess / P2 - 1) + ((u1 - u2) ** 2 / (P2 * (v5 - v2)))
        f4 = ((h5 - h2) / (1 / 2 * (u1 - u2) ** 2)) + ((v5 + v2) / (v5 - v2))

        df3_dP5_T5 = (1 / P2) + (-(u1 - u2) ** 2 / (P2 * (v5 - v2) ** 2)) * (-v5 / P5_guess)
        df3_dT5_P5 = (-(u1 - u2) ** 2 / (P2 * (v5 - v2) ** 2)) * (v5 / T5_guess)
        df4_dP5_T5 = (-2 * v2 / (v5 - v2) ** 2) * (-v5 / P5_guess)
        df4_dT5_P5 = (1 / (1 / 2 * (u1 - u2) ** 2)) * cp5 + (-2 * v2 / (v5 - v2) ** 2) * (v5 / T5_guess)

        # Calculate next iteration for region 5 conditions
        T5, P5 = tuple((np.array([[T5_guess], [P5_guess]])
                        - np.matmul(np.linalg.inv(np.array([[df3_dT5_P5, df3_dP5_T5], [df4_dT5_P5, df4_dP5_T5]])),
                                    np.array([[f3], [f4]]))).T[0])

        # Check for convergence and maximum iteration conditions
        if abs(T5 - T5_guess) / T5_guess < epsilon and abs(P5 - P5_guess) / P5_guess < epsilon:
            break
        elif i == max_iter - 1:
            raise RuntimeError("Convergence criteria not met within the maximum number of iterations")
        else:
            T5_guess = T5
            P5_guess = P5

    # Reset object to region 1 conditions
    gas.TP = T1, P1

    return (T2, P2), (T5, P5)


def shock_tube_flow_properties(M: float, T1: float, T4: float, MW1: float, MW4: float, gamma1: float, gamma4: float,
                               *, area_ratio: float = 1) -> Tuple[float, float, float, float]:
    """Calculates the characteristics associated with the formation of an ideal incident shock for a shock tube flow
    with monotonic convergence at the diaphragm section. The equations derived by Alpher and White[^1] were optimized
    for use with `scipy.optimize.root_scalar`.

    Parameters
    ----------
    M
        incident shock Mach number
    T1
        absolute temperature of driven gas
    T4
        absolute temperature of driver gas
    MW1
        mean molecular weight of driven gas
    MW4
        mean molecular weight of driver gas
    gamma1
        specific heat ratio of driven gas
    gamma4
        specific heat ratio of driver gas
    area_ratio
        ratio of driver to driven area

    Returns
    -------
    P4_P1
        ratio of driver to driven initial pressures
    M3a
        Mach number of flow before nozzle
    Me
        Mach number of flow at nozzle exit
    M3
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
    P2_P1 = (2 * gamma1 * M ** 2 - (gamma1 - 1)) / (gamma1 + 1)

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


def tailored_mixture(M: float, T1: float, T4: float, MW1: float, MW4: Tuple[float, float], gamma1: float,
                     gamma4: Tuple[float, float], *, area_ratio: float = 1) -> Tuple[float, float]:
    """Calculates the species mole fractions that tailor the interaction between the reflected shock wave and the
    contact surface. The mole fractions for which the output of
    `knightshock.gas_dynamics.shock_tube_flow_properties` satisfies the tailoring condition described by Hong et al.[^1]
    is iteratively calculated using `scipy.optimize.root_scalar`.

    Parameters
    ----------
    M
        incident shock Mach number
    T1
        absolute temperature of driven gas
    T4
        absolute temperature of driver gas
    MW1
        mean molecular weight of driven gas
    MW4
        mean molecular weights of driver species used for tailoring
    gamma1
        specific heat ratio of driven gas
    gamma4
        specific heat ratios of driver species used for tailoring
    area_ratio
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
