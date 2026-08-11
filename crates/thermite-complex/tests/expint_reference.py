"""Reference values for the complex exponential integral E_N(z).

Regenerates the tables in `expint.rs`. mpmath is an independent implementation at
arbitrary precision, so it shares no code with thermite.

    python expint_reference.py

The point set deliberately covers both regimes of the implementation (|z| < 1 takes
the power series, |z| >= 1 the Stieltjes continued fraction) and both sides of the
branch cut on (-inf, 0].
"""

import mpmath as mp

mp.mp.dps = 30

POINTS = [
    # inside the unit disc - power series
    (0.5, 0.25),
    (0.3, -0.6),
    (0.9, 0.1),
    # outside - continued fraction
    (2.0, 1.0),
    (3.5, -2.0),
    (8.0, 0.0),
    # far off the real axis, small real part: the case a lexicographic `re < 1`
    # regime test would misroute into the (divergent there) power series
    (0.5, 100.0),
    # straddling the branch cut - these two must come out conjugate
    (-2.0, 0.1),
    (-2.0, -0.1),
    # on the cut, and just above it
    (-0.5, 0.0),
    (-1.0, 1e-8),
    # left half-plane, large |z|: e^-z is huge here rather than underflowing,
    # which is what makes the order recurrence's error amplification reachable
    (-20.0, 5.0),
    (-50.0, 0.5),
    (-100.0, 1.0),
    # hard against the cut from both sides, at several magnitudes. These are what
    # the Stieltjes continued fraction cannot do (it degrades by |Arg z| ~ 177 deg
    # and fails outright on the cut), hence the left half-plane going to the series.
    (-5.0, 0.01),
    (-8.0, -1e-6),
    (-0.999, 0.0),
    (-40.0, 0.0),
]


def main():
    for n in (1, 2, 3):
        print(f"const EXPINT_E{n}: &[(f64, f64, f64, f64)] = &[")
        for a, b in POINTS:
            v = mp.expint(n, mp.mpc(a, b))
            print(f"    ({a!r}, {b!r}, {float(v.real):.17e}, {float(v.imag):.17e}),")
        print("];")
        print()


if __name__ == "__main__":
    main()
