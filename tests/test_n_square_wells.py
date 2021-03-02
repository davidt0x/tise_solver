import math
import pytest
import numpy as np
from tise_solver.potentials import n_square_wells, two_square_wells


def test_two_square_wells():
    """Test the generalized N wells function againts Lena's 2 well code."""

    # Lets test 2 wells
    widths = [7.0, 5.0]
    depths = [10.0, 12.0]
    separations = [2.5]

    v = n_square_wells(widths=widths, depths=depths, separations=separations)

    x, v2 = two_square_wells(d1=depths[0], d2=depths[1], w1=widths[0], w2=widths[1], w_sep=separations[0])

    # This should be exactly equal
    assert np.all(v(x) == v2)


def test_exceptions():

    """Check that improper parameters generate proper exceptions."""
    with pytest.raises(ValueError) as ex:
        v = n_square_wells(widths=[1, 2], depths=[1], separations=[1])

    assert "widths and depths must be equal" in str(ex.value)

    """Check that improper parameters generate proper exceptions."""
    with pytest.raises(ValueError) as ex:
        v = n_square_wells(widths=[1, 2], depths=[1, 2], separations=[1, 2])

    assert "separations must be one less than the number of wells" in str(ex.value)