
import numpy as np
from typing import List

def n_square_wells(widths: List[float], depths: List[float], separations: List[float], width_bg: float):

    max_depth = max(depths)

    # Specify the edges of each part of the function. The list comprehension below interleaves widths and
    # separations
    edges = [0, width_bg] + [x for t in zip(widths, separations) for x in t] + [widths[-1], width_bg]

    # Now construct a value array for each set of edges above.
    values = np.array([max_depth, max_depth] +
                      [max_depth - x for t in zip(depths, len(separations)*[0]) for x in t] +
                      [max_depth - depths[-1], max_depth, max_depth])

    def V(x):
        return values[np.searchsorted(edges, x)]

    return V
