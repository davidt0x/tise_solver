import math
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from typing import List, Union, Optional

from tise_solver.potentials import n_square_wells, n_square_wells_bounds, calc_min_debroglie_wavelength, calc_background_width


def numerov(widths: List[float],
            depths: List[float],
            separations: List[float] = None,
            width_bg: Optional[float] = None,
            dx: Optional[float] = None,
            method: str = 'dense'):
    """
    Solve the TISE for a potential with N square wells via the matrix numerov method.

    Args:
        widths: A list of widths for each well.
        depths: A list of depths for each well. Must be the same length as widths.
        separations: A list of N - 1 seperations. seperations[i] is the distance between well_i and well_i+1.
        width_bg: The width from the lower bound of the domain and the leftmost edge of the first well. Similarly,
            the width from the rightmost edge of the last well and the upper bound of the domain. If None, then
            width_bg = 2.0 * tise_solver.potentials.
        dx: The step size to use when discretized the potential. Defaults to
            calc_min_debroglie_wavelength(max(depths)) / (2.0 * math.pi)
        method: A string (either 'sparse' or 'dense'), which tells the function to use a dense or sparse eigensolver
            for computing the solution. Default is dense.

    Returns:

    """

    if separations is None:
        if len(depths) != 1:
            raise ValueError("Must pass a list separation widths for more than one well!")
        else:
            separations = []

    if width_bg is None:
        width_bg = calc_background_width(max(depths))

    if dx is None:
        dx = calc_min_debroglie_wavelength(max(depths)) / (2.0 * math.pi)

    # Concatenate all the widths (background, wells, separations)
    pot_widths = [width_bg] + [x for t in zip(widths, separations) for x in t] + [widths[-1], width_bg]
    w_tot = np.sum(pot_widths)

    # Discretize the domain of the potential
    n_steps = int(sum([np.ceil(w/dx) for w in pot_widths]))
    x = np.linspace(0, w_tot, n_steps)
    del_x = x[1] - x[0]

    # Get the potential function and evaluate it at all discrete x positions
    v = n_square_wells(widths=widths, depths=depths, separations=separations, width_bg=width_bg)
    v = v(x)

    # Get the bounds of each well
    bounds = n_square_wells_bounds(widths=widths, depths=depths, separations=separations, width_bg=width_bg)

    # solve the schrodinger equation using the numerov matrix method.
    beta = 2.0

    if method == 'dense':
        V = np.diag(v)
        A = (-1.0 / beta) * (1.0 / dx ** 2.0) * (np.diag(-2.0 * np.ones(n_steps)) +
                                                 np.diag(np.ones(n_steps - 1), -1) +
                                                 np.diag(np.ones(n_steps - 1), 1))
        B = (1 / 12) * (np.diag(10 * np.ones(n_steps)) +
                        np.diag(np.ones(n_steps - 1), -1) +
                        np.diag(np.ones(n_steps - 1), 1))
        sys_eq = np.linalg.lstsq(B, A, rcond=None)[0] + V
        E, psi = np.linalg.eig(sys_eq)

    elif method == 'sparse':
        V = scipy.sparse.diags([v], [0], format='csc')
        A = (-1.0 / beta) * (1.0 / dx**2.0) * scipy.sparse.diags([-2.0 * np.ones(n_steps),
                                                                  np.ones(n_steps - 1),
                                                                  np.ones(n_steps - 1)],
                                                                 [0, -1, 1], format='csc')
        B = (1.0 / 12.0) * scipy.sparse.diags([10 * np.ones(n_steps), np.ones(n_steps - 1), np.ones(n_steps - 1)],
                                              [0, -1, 1], format='csc')
        sys_eq = scipy.sparse.linalg.spsolve(B, A) + V

        # FIXME: We need the equation that Lena said she would provide for k, the number of eigen values to compute
        X = scipy.sparse.diags([sys_eq.diagonal(k) for k in range(20, -1, -1)], range(20, -1, -1))
        E, psi = scipy.linalg.eig_banded(X.data, select='v', select_range=(0, max(depths)), max_ev=10)
        #E, psi = scipy.sparse.linalg.eigsh(A=sys_eq, k=20, sigma=0)
    else:
        raise ValueError("Invalid string passed to method argument, must be either 'sparse' or 'dense'")

    # Sort the eigen values (and corresponsding eigenvectors) from least to greatest
    idx = E.argsort()
    E = E[idx]
    psi = psi[:, idx]

    # Take only the eigenvectors greater than 0 and less than bg
    inds = np.where((E > 0) & (E < max(depths)))[0]
    E = E[inds]
    psi = psi[:, inds]

    # Compute the probability density by squaring
    dens = psi * psi

    # normalizing the density
    dens = dens / np.trapz(dens, axis=0)

    # Compute the individual density for each well and barrier\separation between wells
    p_wells = np.zeros((len(widths), len(E)))
    p_int = np.zeros((len(separations), len(E)))
    for well_i, (lower, upper) in enumerate(bounds):
        lower_i = int(np.ceil(lower / dx)) + 1
        upper_i = int(np.ceil(upper / dx)) + 2

        if widths[well_i] != 0:
            p_wells[well_i, :] = np.trapz(dens[lower_i:upper_i, :], axis=0)

        # Compute the separation between this well and the last, don't do this for the first
        # well because that is the background
        if well_i > 0 and separations[well_i-1] > 0.0:
            barrier_upper = bounds[well_i-1][1] # Upper bound of the previous well is the lower bound of the barrier
            barrier_upperi = int(np.ceil(barrier_upper / dx)) + 2
            p_int[well_i-1, :] = np.trapz(dens[barrier_upperi:lower_i, :], axis=0)

    # Compute the probability for the background
    p_bg = 1 - (np.sum(p_wells, axis=0) + np.sum(p_int, axis=0))

    return dict(v=v, E=E, psi=psi, dens=dens, p_wells=p_wells, p_int=p_int, p_bg=p_bg, n_steps=n_steps)


def main():
    r = numerov(widths=[3.0], depths=[10.0], dx=0.01, method='sparse')


if __name__ == "__main__":
    main()


