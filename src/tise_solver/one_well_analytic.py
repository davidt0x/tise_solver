import warnings
import numpy as np
from scipy.optimize import fsolve


def solve(func, x0):
    """
    Wrapper for fsolve that catches warnings and returns NaN if the solve failed.
    """
    with warnings.catch_warnings(record=True) as w:
        x, infodict, ier, mesg = fsolve(func=func, x0=x0, full_output=True)

    if ier == 1 and len(w) == 0:
        return x[0]
    else:
        return np.nan


def cot(x):
    """
    Compute the cotangent of x.
    """
    return np.cos(x)/np.sin(x)


def uniquetol(r, tol=1e-6) -> np.array:
    """
    Compute the unique elements of vector within some floating point tolerance

    Args:
        r: Vector to find unique elements from.
        tol: The tolerance to apply

    Returns:
        The unique elements of r.
    """
    return r[~(np.triu(np.abs(r[:, None] - r) <= tol, 1)).any(0)]


def one_well_energies(depth: float, width: float, tolerance=1e-6):
    """
    Find the analytic energies of a one-well square potential.
    """
    # Figure out how many energies we are supposed to have
    v_0 = np.sqrt(width ** 2 * depth / 2.0)
    N = int(np.floor(0.5 + v_0/np.pi) + np.floor(v_0/np.pi) + 1)

    # The two objective functions we are trying to find roots for.
    obj1 = lambda E: np.sqrt((depth - E) / E) - np.tan(np.sqrt(E * width**2 / 2.0))
    obj2 = lambda E: np.sqrt((depth - E) / E) + cot(np.sqrt(E * width**2 / 2.0))

    # Sample a grid of points, we will use these as starting points for fsolve
    E_grid = np.linspace(-depth, 0.0, num=1000)

    # Run fsolve for each starting point for each objective function
    r1 = np.array([solve(func=obj1, x0=x0) for x0 in np.linspace(0.0, depth, num=1000)])
    r2 = np.array([solve(func=obj2, x0=x0) for x0 in np.linspace(0.0, depth, num=1000)])

    # Remove any bad solutions (any solve that made a warning or didn't converge)
    s1 = r1[~np.isnan(r1)]
    s2 = r2[~np.isnan(r2)]

    # Remove any duplicate solutions up to some floating point
    s1 = uniquetol(s1, tol=1e-6)
    s2 = uniquetol(s2, tol=1e-6)

    # Combine and sort
    s = np.sort(np.concatenate((s1, s2)))

    return s