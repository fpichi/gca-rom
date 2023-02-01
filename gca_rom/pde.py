import numpy as np


def problem(argument):
    """
    problem(argument: int) -> Tuple[str, str, np.ndarray, np.ndarray]

    This function takes in an integer argument and returns a tuple containing the problem name (str), variable name (str), mu1 (np.ndarray) and mu2 (np.ndarray) for the specified case.

    The possible values of argument are:

    Poisson problem
    Advection problem
    Graetz problem
    Navier-Stokes problem (variable = VX)
    Navier-Stokes problem (variable = VY)
    Navier-Stokes problem (variable = P)
    Returns:
    Tuple containing the problem name (str), variable name (str), mu1 (np.ndarray) and mu2 (np.ndarray) for the specified case.
    """

    match argument:
        case 1:
            problem = "poisson"
            variable = 'U'
            mu1 = np.linspace(0.01, 10., 10)
            mu2 = np.linspace(0.01, 10., 10)
        case 2:
            problem = "advection"
            variable = 'U'
            mu1 = np.linspace(0., 6., 10)
            mu2 = np.linspace(-1.0, 1.0, 10)
        case 3:
            problem = "graetz"
            variable = 'U'
            mu1 = np.linspace(1., 3., 10)
            mu2 = np.linspace(0.01, 0.1, 20)
        case 4:
            problem = "navier_stokes"
            variable = 'VX'
            mu1 = np.linspace(0.5, 2., 21)[::2]
            mu2 = np.linspace(2., 0.5, 151)[::5]
        case 5:
            problem = "navier_stokes"
            variable = 'VY'
            mu1 = np.linspace(0.5, 2., 21)[::2]
            mu2 = np.linspace(2., 0.5, 151)[::5]
        case 6:
            problem = "navier_stokes"
            variable = 'P'
            mu1 = np.linspace(0.5, 2., 21)[::2]
            mu2 = np.linspace(2., 0.5, 151)[::5]
    return problem, variable, mu1, mu2