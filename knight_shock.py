import cantera as ct
import numpy as np

R = ct.gas_constant


def FROSH(T1, P1, u1, *, thermo, max_iterations=1000, convergence_criteria=1e-6):
    """
    Implementation of the FROzen SHock (FROSH) algorithm for calculating post-incident-shock and post-reflected-shock
    conditions from initial conditions and shock velocity for a shock tube experiment

    Args:
        T1: initial temperature in driven section [K]
        P1: initial pressure in driven section [Pa]
        u1: incident shock velocity [m/s]
        thermo: object for temperature- and pressure-dependent thermodynamic property calculation
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
    a1 = (ct.gas_constant / thermo.mean_molecular_weight * gamma1 * T1) ** 0.5  # [m/s]
    v1 = 1 / thermo.density_mass  # [m^3/kg]
    h1 = thermo.enthalpy_mass  # [J/kg]

    # Calculates ideal P2 and T2 for initial guesses
    Ms = u1 / a1

    T2 = T1 * ((gamma1 * Ms ** 2 - (gamma1 - 1) / 2) * ((gamma1 - 1) / 2 * Ms ** 2 + 1)) \
         / (((gamma1 + 1) / 2) ** 2 * Ms ** 2)  # [K]
    P2 = P1 * (2 * gamma1 * Ms ** 2 - (gamma1 - 1)) / (gamma1 + 1)  # [Pa]

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
    P5 = P2 * ((gamma2 + 1) / (gamma2 - 1) + 2 - P1 / P2) / (1 + (gamma2 + 1) / (gamma2 - 1) * P1 / P2)  # [Pa]
    T5 = T2 * (P5 / P2) * ((gamma2 + 1) / (gamma2 - 1) + P5 / P2) / (1 + (gamma2 + 1) / (gamma2 - 1) * P5 / P2)  # [K]

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


class Experiment:
    """
    Regions:
        1 - initial conditions driven section
        2 - post-incident-shock conditions driven section
        3 - expanded conditions driver section
        4 - initial conditions driver section
        5 - post-reflected-shock driven section
    """

    def __init__(self, mechanism):
        self.T1 = None  # [K]
        self.T4 = None  # [K]
        self.P1 = None  # [Pa]
        self.P4 = None  # [Pa]

        self.driver_mixture = None
        self.driven_mixture = None

        self._u = None  # [m/s]

        self.T2 = None  # [K]
        self.P2 = None  # [Pa]
        self.T5 = None  # [K]
        self.P5 = None  # [Pa]

        self.thermo = ct.Solution(mechanism)

    @property
    def gamma1(self):
        self.thermo.TPX = self.T1, self.P1, self.driven_mixture
        return self.thermo.cp / self.thermo.cv

    @property
    def gamma4(self):
        self.thermo.TPX = self.T4, self.P4, self.driver_mixture
        return self.thermo.cp / self.thermo.cv

    @property
    def a1(self):
        self.thermo.TPX = self.T1, self.P1, self.driven_mixture
        return (self.thermo.cp / self.thermo.cv * ct.gas_constant * self.T1 / self.thermo.mean_molecular_weight) ** 0.5

    @property
    def a4(self):
        self.thermo.TPX = self.T4, self.P4, self.driver_mixture
        return (self.thermo.cp / self.thermo.cv * ct.gas_constant * self.T4 / self.thermo.mean_molecular_weight) ** 0.5

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, value):
        if value < 0:
            raise ValueError

        self._u = value
        self.T2, self.P2, self.T5, self.P5 = FROSH(self.T1, self.P1, self.u, thermo=self.thermo)

    @staticmethod
    def get_shock_speed(x, t):
        n = len(x)

        if len(t) == n:
            t = np.diff(t)

        if len(t) != n - 1:
            raise ValueError

        x_midpoint = np.empty(n - 1)
        u_avg = np.empty(n - 1)

        for i in range(n - 1):
            x_midpoint[i] = (x[i] + x[i + 1]) / 2
            u_avg[i] = (x[i] - x[i + 1]) / t[i]

        A = np.vstack([x_midpoint, np.ones(n - 1)]).T
        model, residual = np.linalg.lstsq(A, u_avg, rcond=None)[:2]
        r2 = 1 - residual / (u_avg.size * u_avg.var())

        u = model[1]
        attenuation = model[0] / model[1]

        return u, attenuation, r2


