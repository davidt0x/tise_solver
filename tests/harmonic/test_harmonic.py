import numpy as np

from tise_solver.harmonic import harmonic


def test_harmonic():
    """
    Test harmonic tise example against Fortran code output
    """

    # Run and check that things are working like the Fortran code
    eig_values, eig_vectors, analytic, dens0, dens1, anal0, anal1 = harmonic(nt=900, omega=50.0)

    # Let check things against the fortran results
    f16 = np.loadtxt('tests/harmonic/fort.16', skiprows=1)
    assert np.allclose(f16[:, 2], eig_values)
    assert np.allclose(f16[:, 3], analytic)

    f17 = np.loadtxt('tests/harmonic/fort.17', skiprows=1)
    assert np.allclose(f17[:, 2], dens0)
    assert np.allclose(f17[:, 5], dens1)

    f18 = np.loadtxt('tests/harmonic/fort.18', skiprows=1)
    assert np.allclose(f18[:, 2], anal0)
    assert np.allclose(f18[:, 4], anal1)
