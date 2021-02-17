import numpy as np
from scipy.io import loadmat

from tise_solver.get_p_2W import get_p_2W


def test_get_p_2W():
    """
    Test the output for the matrix numerov method against the MATLAB implementation.
    """
    mat = loadmat('tests/get_p_2W/get_p_2W.mat', squeeze_me=True)

    # Get the parameters we are going to test
    d1 = mat['d1']
    d2 = mat['d2']
    w1 = mat['w1']
    w2 = mat['w2']
    w_sep = mat['w_sep']

    r = get_p_2W(d1=d1, d2=d2, w1=w1, w2=w2, w_sep=w_sep)

    # Make check to see that all the values that the function returns match
    for val_name in r:
        # Check any value with the same name as the matlab code, except psi. These
        # are the eigenvectors which can have opposite direction. dens is their elementwise
        # square which is checked.
        if val_name in mat and val_name != 'psi':
            assert np.allclose(mat[val_name], r[val_name]), f"{val_name} did not match"



