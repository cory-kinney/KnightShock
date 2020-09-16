from scipy import optimize

R = 8314.4621  # [J/kmol/K]


def _area_ratio(M3a, Me, gamma4):
    return Me / M3a * ((2 + (gamma4 - 1) * M3a ** 2) / (2 + (gamma4 - 1) * Me ** 2)) ** (
            (gamma4 + 1) / (gamma4 - 1) / 2)


def _equivalence_factor(gamma4, M3a, Me):
    return (((2 + (gamma4 - 1) * M3a ** 2) / (2 + (gamma4 - 1) * Me ** 2)) ** 0.5
            * (2 + (gamma4 - 1) * Me) / (2 + (gamma4 - 1) * M3a)) ** (2 * gamma4 / (gamma4 - 1))


def _M3_ideal_expansion(Ms, M3a, Me, gamma1, a1, gamma4, a4):
    return ((gamma1 + 1) / 2 * Ms / (Ms ** 2 - 1) * a4 / a1
            * _equivalence_factor(gamma4, M3a, Me) ** ((gamma4 - 1) / gamma4 / 2) - (gamma4 - 1) / 2) ** -1


def tailored_mixture_ideal(Ms, T1, gamma1, MW1, T4, specific_heat_ratios, molecular_weights,
                           *, area_ratio=1, minimum_Mach=1e-3):
    """
    Calculates the driver mixture mole fractions for a tailored contact-surface using ideal assumptions

    Args:
        Ms: incident shock wave Mach number
        T1: initial temperature in the driven section [K]
        gamma1: specific heat ratio of driven gas
        MW1: mean molecular weight of driven gas
        T4: initial temperature in the driver section [K]
        specific_heat_ratios: specific heat ratios of driver species
        molecular_weights: mean molecular weights of driver species
        area_ratio: ratio of the area of the driver section to the area of the driven section
        minimum_Mach: lower bound for Mach number for iterative solvers

    Returns:
        x: mole fraction of the first species
        1 - x: mole fraction of the second species

    """

    assert area_ratio >= 1, "Area ratio must be greater than or equal to one"

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
            return area_ratio - _area_ratio(M3a_guess, Me, gamma4)

        # Assume supersonic case (Me = 1, M3 > 1)
        M3a = optimize.root_scalar(lambda M3a_guess: _M3a_error(M3a_guess, 1), bracket=[minimum_Mach, 1]).root
        M3 = _M3_ideal_expansion(Ms, M3a, 1, gamma1, 1, gamma4, a4)

        # Subsonic case (M3 = Me)
        if M3 < 1:
            def _M3_expansion_error(M3_guess):
                M3a = optimize.root_scalar(lambda M3a_guess: _M3a_error(M3a_guess, M3_guess),
                                           bracket=[minimum_Mach, M3_guess]).root
                return M3_guess - _M3_ideal_expansion(Ms, M3a, M3_guess, gamma1, a1, gamma4, a4)

            M3 = optimize.root_scalar(_M3_expansion_error, bracket=[minimum_Mach, 1]).root

        return M3 - M3_tailored

    x = optimize.root_scalar(_M3_tailoring_error, bracket=[0, 1]).root
    return x, 1 - x
