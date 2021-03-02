from tise_solver.potentials import n_square_wells

def test_n_square_wells():

    width_bg = 3
    widths = [2, 2, 3]
    depths = [4, 5]
    separations = [1, 2, 1]

    V = n_square_wells(widths=widths, depths=depths, separations=separations, width_bg=width_bg)

