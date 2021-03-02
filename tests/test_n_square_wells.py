import math
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

