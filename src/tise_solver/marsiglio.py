#%%
from tise_solver.potentials import n_square_wells, n_square_wells_bounds, calc_background_width
from tise_solver.potentials import calc_min_debroglie_wavelength

from typing import List, Union

import math
import numpy as np
from scipy.linalg.lapack import dsyevd


def harmonic(nt: int = 900, omega: float = 50.0):
    """
    A complete port of the Fortran code contained in tise_solver/fortran/harmonic.f

    Args:
        nt: The number of terms
        omega: FIXME: Something, in units of E_1 === (pi^2*hbar^2)/(2*m_0*a^2)

    Returns:
        FIXME: What does this return?
    """

    # FIXME: This constant is not used in the Fortran code but it is set.
    aa = 1.0   # square well width is the unit of length

    # This corresponds to the construct the matrix nested loop that starts on line 19 of harmonic.f
    # Construct the matrix
    # Get the indices for the upper triangle, without diagonal. This code doesn't set the lower triangle
    # of the matrix like the FORTRAN code does. The matrix is symmetric and I think LAPACK dsyevd doesn't
    # need the lower part so no need to set it.
    n, m = np.triu_indices(n=nt, k=1)
    nmm = (n+1) - (m+1)
    npm = (n+1) + (m+1)
    amm = (np.power(-1.0, nmm) + 1.0) / np.power(nmm, 2)
    app = (np.power(-1.0, npm) + 1.0) / np.power(npm, 2)
    aham = np.zeros((nt,nt))
    aham[n, m] = 0.25 * omega * omega * (amm - app)
    n_idx = np.arange(nt)+1
    np.fill_diagonal(aham, n_idx*n_idx + math.pi**2 * omega * omega * (1.0 - 6.0 / (math.pi*n_idx)**2) / 48.0)

    # Get the eigen values and vectors using LAPACK
    eig_values, eig_vectors, info = dsyevd(aham)

    # Check if things went well
    if info < 0:
        raise ValueError("Invalid argument passed to LAPACK dsyevd")
    elif info > 0:
        raise ValueError("Eigensolver failed to converge.")

    # This corresponds to loop in harmonic.f, line 36, that computes some kind of analytical solution or something.
    nx = np.arange(nt) + 1.0
    analytic = omega * (nx - 0.5)

    # Compute the ground state and 1st excited state wave function and probability density
    # This is vectorized version of the nested loop in harmonic.f, at line 45
    ix = np.arange(0, 200) + 1.0
    xx = ix * 0.005  # in units of a
    sin_n_pi_x = np.sin(nx[None, :] * math.pi * xx[:, None])
    psi0 = np.sum(eig_vectors[:, 0] * sin_n_pi_x, axis=1) * np.sqrt(2.0)
    psi1 = np.sum(eig_vectors[:, 1] * sin_n_pi_x, axis=1) * np.sqrt(2.0)
    dens0 = psi0 * psi0
    dens1 = psi1 * psi1

    # some constants
    factanal = (math.pi * omega / 2.0) ** (0.25)
    expanal = 0.25 * omega * math.pi ** 2

    anal0 = factanal * np.exp(-expanal * (xx - 0.5) ** 2)
    anal1 = -anal0 * 2.0 * np.sqrt(expanal) * (xx - 0.5)

    return eig_values, eig_vectors, analytic, dens0, dens1, anal0, anal1


def marsiglio(
    widths: List[float],
    depths: List[float],
    separations: List[float],
    width_bg: Union[float, None] = None,
    nt: int = None,
):
    """
    Solve the TISE for a potential of N square wells using Frank's method.

    Args:
        widths: A list of widths for each well.
        depths: A list of depths for each well. Must be the same length as widths.
        separations: A list of N - 1 seperations. seperations[i] is the distance between well_i and well_i+1.
        width_bg: The width from the lower bound of the domain and the leftmost edge of the first well. Similarly,
            the width from the rightmost edge of the last well and the upper bound of the domain. If None, then
            width_bg = int(np.ceil(10.0 * 2 * math.pi * (1 / np.sqrt(2.0 * max(depths))))
        nt: The number of terms. This will default to

    Returns:
    
    """

    if separations is None:
        if len(depths) != 1:
            raise ValueError("Must pass a list separation widths for more than one well!")
        else:
            separations = []

    if width_bg is None:
        width_bg = calc_background_width(max(depths))

    bounds = n_square_wells_bounds(widths=widths, depths=depths, separations=separations, width_bg=width_bg)

    # Find the width of the infinite well that all the wells are embedded into
    w_tot = np.sum(widths) + np.sum(separations) + 2.0 * width_bg

    # Concatenate all the widths (background, wells, separations)
    pot_widths = [width_bg] + [x for t in zip(widths, separations) for x in t] + [widths[-1], width_bg]

    # Pretend like we are doing the Numerov method and discretize the domain of the potential, this is only used if
    # nt is not specified. This is will atleast make comparing to the Numerov method easier.
    if nt is None:
        dx = calc_min_debroglie_wavelength(max(depths)) / (2.0 * math.pi)
        nt = int(sum([np.ceil(w / dx) for w in pot_widths]))

    # Lets fix the infinite well width to unitary, we will normalize x by w_tot
    a = w_tot

    V_max = max(depths)

    H = np.zeros((nt, nt))

    # Construct the non-diagonal portion of the matrix first.
    p, q = np.triu_indices(n=nt, k=1)
    pp = p + 1.0
    qq = q + 1.0
    qmp = (qq - pp)
    ppq = (pp + qq)

    # Now add the contribution for each well
    for well_i, (lower, upper) in enumerate(bounds):
        if lower != upper:
            pi_a = (math.pi / a)
            H_w = (1 / (pp - qq)) * (np.sin(upper * qmp * pi_a) - np.sin(lower * qmp * pi_a)) + \
                  (1 / (pp + qq)) * (np.sin(upper * ppq * pi_a) - np.sin(lower * ppq * pi_a))

            H[p, q] = H[p, q] + (depths[well_i] / math.pi) * H_w

    # Now do the same for the diagonal
    p = np.arange(nt)
    pp = p + 1
    pp2pi = 2.0 * pp * math.pi

    H[p, p] = (1 / 2) * (pp * math.pi / a)**2 + V_max
    for well_i, (lower, upper) in enumerate(bounds):
        if lower != upper:
            H_w = (pp2pi / a) * (upper - lower) + np.sin(pp2pi * (lower/a)) - np.sin(pp2pi * (upper/a))
            H[p, p] = H[p, p] - (depths[well_i] / pp2pi) * H_w


    # Get the eigen values and vectors using LAPACK
    E, c, info = dsyevd(H)

    # Check if things went well
    if info < 0:
        raise ValueError("Invalid argument passed to LAPACK dsyevd")
    elif info > 0:
        raise ValueError("Eigensolver failed to converge.")

    # Take only the eigenvalues greater than 0 and less than the V_max
    #E = np.array([np.sum(c[:, i] * E[i]) for i in range(nt)])
    inds = np.where((E > 0) & (E < V_max))[0]
    E = E[inds]

    return dict(E=E, c=c)


def main():

    #%%
    # Now lets time things
    import timeit
    NUM_TIMES = 100
    print(f"Average Execution Time: {timeit.timeit(harmonic, number=NUM_TIMES)/NUM_TIMES*1000} milliseconds")


if __name__ == "__main__":
    main()

