"""Reference values for `tests/bessel.rs`: the six Bessel functions of complex argument and
real order in their SciPy scalings, from mpmath at 40 digits.

    python bessel_reference.py > rows.txt

Rows are `(nu, re z, im z, ive, kve, jve, yve, h1e, h2e)` with each complex value as a
`(re, im)` pair, where

    ive = e^{-|Re z|} I_nu(z)      kve = e^{z} K_nu(z)
    jve = e^{-|Im z|} J_nu(z)      yve = e^{-|Im z|} Y_nu(z)
    h1e = e^{-iz} H1_nu(z)         h2e = e^{iz} H2_nu(z)

The grid covers all four quadrants, both axes (the negative real axis approached from
above, which is where the principal branch puts it), the near-origin corner, and `|z|` out
to 1000, at whole, half, third and arbitrary orders of both signs.
"""

import mpmath as mp

mp.mp.dps = 40

NUS = [0, 1, 3, mp.mpf(1) / 3, mp.mpf(1) / 2, mp.mpf(7) / 4, mp.mpf(9) / 4, mp.mpf(37) / 5,
       -mp.mpf(1) / 3, -mp.mpf(5) / 2, -mp.mpf(3) / 4, -mp.mpf(6) / 5]

ZS = [
    # right half-plane
    (0.1, 0.1), (0.0, 0.001), (0.5, 0.3), (0.0, 1.2), (1.5, 1.5), (2.5, 1.0), (0.0, 3.0),
    (4.0, -0.5), (8.0, 5.0), (15.0, -4.0), (0.0, 20.0), (25.0, 25.0), (0.0, 60.0),
    (100.0, 1.0), (300.0, -200.0), (1000.0, 0.5), (0.0, 500.0), (2.0, 0.0), (1e-3, 0.0),
    (0.0, -2.0), (0.0, -45.0), (7.0, 0.0), (60.0, 0.0),
    # left half-plane, both quadrants, and the negative real axis from above
    (-0.3, 0.4), (-0.3, -0.4), (-2.0, 1.0), (-2.0, -1.0), (-5.0, 12.0), (-5.0, -12.0),
    (-30.0, 3.0), (-30.0, -3.0), (-1.0, 0.0), (-9.0, 0.0), (-200.0, 150.0), (-0.01, -0.02),
]


def f(x):
    return mp.nstr(x, 17, min_fixed=-3, max_fixed=3, strip_zeros=False)


def c(w):
    return f"{f(w.real)}, {f(w.imag)}"


def rows():
    for nu in NUS:
        for re, im in ZS:
            z = mp.mpc(re, im)
            ive = mp.exp(-abs(z.real)) * mp.besseli(nu, z)
            kve = mp.exp(z) * mp.besselk(nu, z)
            # mpmath forms the decaying Hankel function as `J +- iY`, two terms of size
            # `e^{|Im z|}` cancelling to `e^{-|Im z|}`: at `z = 500i` that is 434 digits of
            # cancellation, and at 40 it returned 1e383. Give the oscillating family the
            # digits that cancellation eats.
            with mp.workdps(60 + int(1.2 * abs(z.imag))):
                jve = mp.exp(-abs(z.imag)) * mp.besselj(nu, z)
                yve = mp.exp(-abs(z.imag)) * mp.bessely(nu, z)
                h1e = mp.exp(-1j * z) * mp.hankel1(nu, z)
                h2e = mp.exp(1j * z) * mp.hankel2(nu, z)
            yield nu, z, ive, kve, jve, yve, h1e, h2e


if __name__ == "__main__":
    for nu, z, ive, kve, jve, yve, h1e, h2e in rows():
        print(f"    ({f(mp.mpf(nu))}, {c(z)}, {c(ive)}, {c(kve)}, {c(jve)}, {c(yve)}, {c(h1e)}, {c(h2e)}),")
