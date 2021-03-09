import pytest
import numpy as np

from scipy.io import loadmat

from tise_solver.numerov import numerov


def test_get_p_2W():
    """
    Test the output for the matrix numerov method against the MATLAB implementation.
    """
    mat = loadmat('tests/numerov/get_p_2W.mat', squeeze_me=True)

    # Get the parameters we are going to test
    d1 = mat['d1']
    d2 = mat['d2']
    w1 = mat['w1']
    w2 = mat['w2']
    w_sep = mat['w_sep']

    r = numerov(widths=[w1, w2], depths=[d1, d2], separations=[w_sep])

    assert np.allclose(mat['v'], r['v'])
    assert np.allclose(mat['E'], r['E'])
    #assert np.allclose(mat['psi'], r['psi'])
    assert np.allclose(mat['dens'], r['dens'])
    assert np.allclose(mat['p_1'], r['p_wells'][0, :])
    assert np.allclose(mat['p_2'], r['p_wells'][1, :])
    assert np.allclose(mat['p_int'], r['p_int'][0, :])
    assert np.allclose(mat['p_bg'], r['p_bg'])


@pytest.mark.parametrize("widths,depths,separations,width_bg", [
    ([7.0], [10.0], [], 2.0),
    ([7.0, 5.0], [10.0, 12.0], [2.5], None),
    ([1.0, 2.0, 3.0], [100.0, 12.0, 3.0], [5.0, 6.0], 20.0),
    ([1.0, 2.0, 3.0], [100.0, 12.0, 3.0], [0.0, 6.0], 20.0),
])
def test_N_wells_numerov(widths, depths, separations, width_bg):
    r = numerov(widths=widths, depths=depths, separations=separations, width_bg=width_bg)
