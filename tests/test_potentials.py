import math
import pytest
import numpy as np
from tise_solver.potentials import n_square_wells, n_square_wells_bounds, two_square_wells


def test_two_square_wells():
    """Test the generalized N wells function against Lena's 2 well code."""

    # Lets test 2 wells
    widths = [7.0, 5.0]
    depths = [10.0, 12.0]
    separations = [2.5]

    v = n_square_wells(widths=widths, depths=depths, separations=separations)

    x, v2 = two_square_wells(d1=depths[0], d2=depths[1], w1=widths[0], w2=widths[1], w_sep=separations[0])

    # This should be exactly equal
    assert np.all(v(x) == v2)


@pytest.mark.parametrize("func", [n_square_wells, n_square_wells_bounds])
def test_n_square_wells_exceptions(func):
    """Check that improper parameters generate proper exceptions."""

    with pytest.raises(ValueError) as ex:
        v = func(widths=[1, 2], depths=[1], separations=[1])

    assert "widths and depths must be equal" in str(ex.value)

    """Check that improper parameters generate proper exceptions."""
    with pytest.raises(ValueError) as ex:
        v = func(widths=[1, 2], depths=[1, 2], separations=[1, 2])

    assert "separations must be one less than the number of wells" in str(ex.value)


def test_scalar_x():
    """Test that the potential function works with scalar parameters."""
    v = n_square_wells(widths=[7.0, 5.0], depths=[10.0, 12.0], separations=[2.5])
    assert v(10.0) == v([10.0])[0]
    assert type(v(10.0)) == np.float64


@pytest.mark.parametrize("widths,depths,separations,width_bg", [
    # ([7.0], [10.0], [], 2.0),
    # ([7.0, 5.0], [10.0, 12.0], [2.5], None),
    # ([1.0, 2.0, 3.0], [100.0, 12.0, 3.0], [5.0, 6.0], 20.0),
    ([1.0, 2.0, 3.0], [100.0, 12.0, 3.0], [0.0, 6.0], 20.0),
])
def test_square_well_bounds(widths, depths, separations, width_bg):
    v = n_square_wells(widths=widths, depths=depths, separations=separations, width_bg=width_bg)
    bounds = n_square_wells_bounds(widths=widths, depths=depths, separations=separations, width_bg=width_bg)

    # Make sure we have the same number of wells and bounds tuples.
    assert len(widths) == len(bounds)

    # Check that the values sampled from between well upper and lower bounds equal correct values
    max_depth = max(depths)
    for i, (lower, upper) in enumerate(bounds):
        assert np.all(v(np.linspace(lower+1e-8, upper, num=100)) == (max_depth - depths[i]))

        # Check the values between each well, if this is the first well, check the background
        if i == 0:
            assert np.all(v(np.linspace(0.0, lower, num=100)) == max_depth)
        else:
            if bounds[i-1][1] != lower:
                assert np.all(v(np.linspace(bounds[i-1][1]+1e-8, lower, num=100)) == max_depth)

    # Check the rightmost side background, just check an aribitrary amount outside the domain
    assert np.all(v(np.linspace(bounds[-1][1] + 1e-8, bounds[-1][1] + 100.0, num=100)) == max_depth)
