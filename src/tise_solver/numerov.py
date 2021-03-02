import math
import numpy as np


def numerov(d1: float = 10.0, d2: float = 12.0, w1: float = 7.0, w2: float = 5.0, w_sep: float = 2.5):
    """
    Solve the TISE via the matrix numerov method.

    Args:
        d1: Depth of the first well.
        d2: Depth of the second well.
        w1: Width of the first well
        w2: Width of the second well.
        w_sep: Width of the barrier between two wells

    Returns:

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

    # solve the schrodinger equation using the numerov matrix method.
    V = np.diag(v)
    A = (-1.0 / beta) * (1.0 / dx**2.0) * (np.diag(-2.0 * np.ones(n_steps)) + np.diag(np.ones(n_steps - 1), -1) + np.diag(np.ones(n_steps - 1), 1))
    B = (1/12) * (np.diag(10 * np.ones(n_steps)) + np.diag(np.ones(n_steps - 1), -1) + np.diag(np.ones(n_steps - 1), 1))
    sys_eq = np.linalg.lstsq(B, A, rcond=None)[0] + V
    E, psi  = np.linalg.eig(sys_eq)

    # Sort the eigen values (and corresponsding eigenvectors) from least to greatest
    idx = E.argsort()
    E = E[idx]
    psi = psi[:, idx]

    # Take only the eigenvectors greater than 0 and less than bg
    inds = np.where((E > 0) & (E < bg))[0]
    E = E[inds]
    psi = psi[:, inds]

    # Compute the probability density by squaring
    dens = psi * psi

    # normalizing the densities
    p_1 = np.zeros(len(E))
    p_2 = np.zeros(len(E))
    p_int = np.zeros(len(E))
    p_bg = np.zeros(len(E))
    E_dom = np.empty((3, len(E)))
    E_dom[:] = np.NaN

    for i in range(len(E)):
        d = dens[:, i]
        integral = np.trapz(d)
        d = d / integral
        dens[:, i] = d

        if pw[0] == 0:
            p_1[0] = 0
        else:
            p_1[i] = np.trapz(d[t1b_int-1:t1e_int])

        if pw[2] == 0:
            p_2[i] = 0
        else:
            p_2[i] = np.trapz(d[t2b_int-1:t2e_int])

        if pw[1] == 0:
            p_int[i] = 0
        else:
            p_int[i] = np.trapz(d[t1e_int:t2b_int - 1])

        p_bg[i] = 1 - np.sum([p_1[i], p_2[i], p_int[i]])

        if p_1[i] > p_2[i] and p_1[i] > p_bg[i]:
            E_dom[0, i] = E[i]
        elif p_2[i] > p_1[i] and p_2[i] > p_bg[i]:
            E_dom[1, i] = E[i]
        else:
            E_dom[2, i] = E[i]

    t1_E = E_dom[0, ~np.isnan(E_dom[0, :])]
    t2_E = E_dom[1, ~np.isnan(E_dom[1, :])]
    t3_E = E_dom[2, ~np.isnan(E_dom[2, :])]

    return dict(v=v, E=E, t1_E=t1_E, t2_E=t2_E, t3_E=t3_E, psi=psi, dens=dens, p_1=p_1, p_2=p_2, p_int=p_int, p_bg=p_bg)


def main():
    r = numerov()


if __name__ == "__main__":
    main()


