import math
import numpy as np

from typing import List, Union, Callable


def n_square_wells(widths: List[float],
                   depths: List[float],
                   separations: List[float],
                   width_bg: Union[float, None] = None) -> Callable:
    """
    Return a potential function V(x) for N non-uniform square wells.

    Args:
        widths: A list of widths for each well.
        depths: A list of depths for each well. Must be the same length as widths.
        separations: A list of N - 1 seperations. seperations[i] is the distance between well_i and well_i+1.
        width_bg: The width from the lower bound of the domain and the leftmost edge of the first well. Similarly,
            the width from the rightmost edge of the last well and the upper bound of the domain. If None, then
            width_bg = int(np.ceil(10.0 * 2 * math.pi * (1 / np.sqrt(2.0 * max(depths)))), FIXME: Why does Lena do this?

    Returns:
        A vectorized function for calculating the potential at any position.
    """

    if len(depths) != len(widths):
        raise ValueError("Length of wells widths and depths must be equal.")

    if len(depths)-1 != len(separations):
        raise ValueError("The length of separations must be one less than the number of wells (len(depths))")

    max_depth = max(depths)

    # This is how Lena's code computes width of background, do that for now if nothing is passed.
    if width_bg is None:
        dx1 = 1 / np.sqrt(2.0 * max_depth)
        lamb = 2 * math.pi * dx1
        width_bg = np.ceil(10.0 * lamb)

    # Specify the edges of each part of the function. The list comprehension below interleaves widths and
    # separations
    edges = [0, width_bg] + [x for t in zip(widths, separations) for x in t] + [widths[-1], width_bg]

    # The edges are the cumulative sum of all the interleaved widths
    edges = np.cumsum(edges)

    # Now construct a value array for each set of edges above.
    values = np.array([max_depth, max_depth] +
                      [max_depth - x for t in zip(depths, len(separations)*[0]) for x in t] +
                      [max_depth - depths[-1], max_depth, max_depth])

    def V(x):
        return values[np.searchsorted(edges, x)]

    return V


def two_square_wells(d1: float = 10.0, d2: float = 12.0, w1: float = 7.0, w2: float = 5.0, w_sep: float = 2.5):
    """
    Generate a potential function with two square wells. This is based off Lena's original MATLAB code
    (matlab/get_p_2W.mat) and is mostly for testing purposes.

    Args:
        d1: Depth of the first well.
        d2: Depth of the second well.
        w1: Width of the first well
        w2: Width of the second well.
        w_sep: Width of the barrier between two wells

    Returns:
        A tuple with the following elements:
         - 1D array of positions
         - 1D array of potential at those positions

    """
    w = np.array([w1, w_sep, w2])

    bg = max(d1, d2)
    pot_heights = np.array([bg, bg - d1, bg, bg - d2, bg])
    beta = 2

    # find the minimum debroglie wavelength:
    dx1 = 1 / np.sqrt(beta * bg)
    dx2 = np.nanmin(w) / 5.0
    # note temporary difference: 020421
    # dx = min(dx1, dx2) / 100;
    dx = min(dx1, dx2)

    lamb = 2 * math.pi * dx1

    pw = w
    pw[np.isnan(pw)] = 0.0
    w1, w_sep, w2 = pw
    n_t1 = int(np.ceil(w1 / dx))
    n_t2 = int(np.ceil(w2 / dx))
    n_sep = int(np.ceil(w_sep / dx))

    # w_bg = ceil(2.5 * lamb)
    # temp change, 020121
    w_bg = int(np.ceil(10.0 * lamb))
    n_bg = int(np.ceil(w_bg / dx))

    pot_widths = np.concatenate(([w_bg], pw, [w_bg]))
    w_tot = np.sum(pot_widths)

    n_steps = n_t1 + n_t2 + n_sep + 2 * n_bg
    x = np.linspace(0, w_tot, n_steps)
    del_x = x[1] - x[0]
    v = np.zeros(n_steps)
    # build your v
    bounds = np.zeros(len(pot_widths) + 1)
    bounds[-1] = w_tot

    for j in range(1, len(pot_widths)):
        bounds[j] = np.sum(pot_widths[0:j])

    # background ends at n_bg+1, t1 begins at n_bg+2
    t1b_int = n_bg + 2
    t1e_int = t1b_int + n_t1 - 1
    t2b_int = t1e_int + n_sep
    t2e_int = t2b_int + n_t2 - 1

    for i in range(n_steps):
        if x[i] <= bounds[1]:
            v[i] = pot_heights[0]
        elif x[i] <= np.sum(bounds[2]):
            v[i] = pot_heights[1]
        elif x[i] <= np.sum(bounds[3]):
            v[i] = pot_heights[2]
        elif x[i] <= np.sum(bounds[4]):
            v[i] = pot_heights[3]
        else:
            v[i] = pot_heights[4]

    return x, v