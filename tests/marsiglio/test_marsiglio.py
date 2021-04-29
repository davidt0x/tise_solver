from tise_solver.marsiglio import harmonic, marsiglio
from tise_solver.one_well_analytic import one_well_energies
from tise_solver.potentials import calc_background_width

import pytest
import numpy as np


def test_harmonic():
    """
    Test harmonic tise example against Fortran code output
    """

    # Run and check that things are working like the Fortran code
    eig_values, eig_vectors, analytic, dens0, dens1, anal0, anal1 = harmonic(nt=900, omega=50.0)

    # Let check things against the fortran results
    f16 = np.loadtxt('tests/marsiglio/fort.16', skiprows=1)
    assert np.allclose(f16[:, 2], eig_values)
    assert np.allclose(f16[:, 3], analytic)

    f17 = np.loadtxt('tests/marsiglio/fort.17', skiprows=1)
    assert np.allclose(f17[:, 2], dens0)
    assert np.allclose(f17[:, 5], dens1)

    f18 = np.loadtxt('tests/marsiglio/fort.18', skiprows=1)
    assert np.allclose(f18[:, 2], anal0)
    assert np.allclose(f18[:, 4], anal1)


@pytest.mark.parametrize("widths,depths,separations,width_bg", [
    ([7.0], [10.0], [], 2.0),
    ([7.0, 5.0], [10.0, 12.0], [2.5], None),
    ([1.0, 2.0, 3.0], [100.0, 12.0, 3.0], [5.0, 6.0], 20.0),
    ([1.0, 2.0, 3.0], [100.0, 12.0, 3.0], [0.0, 6.0], 20.0),
])
def test_marsiglio_n_wells(widths, depths, separations, width_bg):
    vals, vecs = marsiglio(widths=widths, depths=depths, separations=separations, width_bg=width_bg, nt=900)


N_MANY_WELLS = 100
@pytest.mark.parametrize("widths,depths,separations,width_bg", [
    ([i*5.0 for i in range(1,N_MANY_WELLS+1)], [float(i) for i in range(1,N_MANY_WELLS+1)], [2.0]*(N_MANY_WELLS-1), 10.0)
])
def test_many_wells(widths, depths, separations, width_bg):
    vals, vecs = marsiglio(widths=widths, depths=depths, separations=separations, width_bg=width_bg, nt=900)


@pytest.mark.parametrize("width,depth,separation,width_bg", [
    (3.0, 10.0, [], None),
])
def test_one_well_marsiglio(width, depth, separation, width_bg):
    s = one_well_energies(depth=depth, width=width)
    E, psi = marsiglio(widths=[width], depths=[depth], separations=separation, width_bg=width_bg)

    x = np.linspace(0.0, 2*calc_background_width(depth) + width, num=100)
    psi_x = psi(x)

    assert np.allclose(E, s, atol=1e-1)
