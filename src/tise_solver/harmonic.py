#%%
from typing import List

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


def main():

    #%%
    # Now lets time things
    import timeit
    NUM_TIMES = 100
    print(f"Average Execution Time: {timeit.timeit(harmonic, number=NUM_TIMES)/NUM_TIMES*1000} milliseconds")


if __name__ == "__main__":
    main()

