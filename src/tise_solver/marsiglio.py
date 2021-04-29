#%%
from tise_solver.potentials import n_square_wells, n_square_wells_bounds, calc_background_width

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
    nt: int = 900,
    omega: float = 50.0
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
        nt: The number of terms
        omega: FIXME: Something, in units of E_1 === (pi^2*hbar^2)/(2*m_0*a^2)

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

    # Lets fix the infinite well width to unitary, we will normalize x by w_tot
    a = w_tot

    V_max = max(depths)

    # Normalize the bounds of the wells
    # We are normalizing the x domain to be between 0 and 1.
    normed_bounds = [(lower/a, upper/a) for (lower, upper) in bounds]

    # Compute the eigen energies of the infinite embedding well
    E_0 = (((np.arange(nt) + 1)) ** 2 * math.pi ** 2) / (2.0 * a ** 2)

    H = np.zeros((nt, nt))

    # Construct the non-diagonal portion of the matrix first.
    n, m = np.triu_indices(n=nt, k=1)
    m_prime = ((m+1) * math.pi) / a
    n_prime = ((n+1) * math.pi) / a
    npm = n_prime + m_prime
    nmm = n_prime - m_prime

    # Add the contribution of the background
    H[n, m] = H[n, m] + V_max*(np.sin(nmm)/nmm - np.sin(npm)/npm)

    # Now add the contribution for each well
    for well_i, (lower, upper) in enumerate(bounds):
        if lower != upper:
            H[n, m] = H[n, m] + depths[well_i] * ((np.sin(nmm*lower) - np.sin(nmm*upper))/nmm +
                                         (np.sin(npm*upper) - np.sin(npm*lower))/npm)

    # Now do the same for the diagonal
    n = np.arange(nt)
    n_prime = ((n + 1) * math.pi) / a
    n_prime2 = 2 * n_prime
    H[n, n] = V_max * (1 - np.sin(n_prime2)/n_prime2)
    for well_i, (lower, upper) in enumerate(bounds):
        if lower != upper:
             H[n, n] = H[n, n] + depths[well_i] * (lower - upper + (np.sin(n_prime2*upper) - np.sin(n_prime2*lower))/n_prime2)

    H[n, n] = E_0 + H[n, n]

    # Get the eigen values and vectors using LAPACK
    E, c, info = dsyevd(H)

    # Check if things went well
    if info < 0:
        raise ValueError("Invalid argument passed to LAPACK dsyevd")
    elif info > 0:
        raise ValueError("Eigensolver failed to converge.")

    # Take only the eigenvalues greater than 0 and less than the V_max
    E = np.array([np.sum(c[:, i] * E[i]) for i in range(nt)])
    inds = np.where((E > 0) & (E < V_max))[0]
    E = E[inds]

    # Define the wave function
    m = np.arange(nt)
    m_prime = ((m + 1) * math.pi) / a
    def psi(x):
        return np.sqrt(2.0 * a) * np.matmul(np.sin(m_prime * x[:, None]), c)

    return E, psi


def main():

    #%%
    # Now lets time things
    import timeit
    NUM_TIMES = 100
    print(f"Average Execution Time: {timeit.timeit(harmonic, number=NUM_TIMES)/NUM_TIMES*1000} milliseconds")


if __name__ == "__main__":
    main()

