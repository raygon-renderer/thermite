#!/usr/bin/env python3
"""Generates the `REF` table in `faddeeva.rs` - reference values of w(z).

    pip install mpmath && python faddeeva_reference.py

Paste the output over the existing `const REF` block.

The oracle is `w(z) = exp(-z^2) * erfc(-i z)` evaluated at 60 digits, which shares
nothing with the implementation (a rational approximation in the Mobius variable
`(L + iz)/(L - iz)`), so agreement is real evidence rather than a tautology.

The grid is chosen to hit every structural feature of the implementation:

- `y = 0` exactly, and `y = 1e-8`, either side of nothing in particular - the method
  has no special case there, and that is the point worth pinning.
- `x` out to 1000, where `Re w` has underflowed to zero but `Im w ~ 1/(sqrt(pi) x)`
  still carries full relative accuracy.
- `x < 0` at several `y`, for the even/odd conjugate symmetry.
- `y < 0`, which is the reflection `w(z) = 2exp(-z^2) - w(-z)` and the only branch in
  the function.
- The first zero of `w` at `1.99146684283 - 1.35481012811i`, where that reflection
  cancels catastrophically and only absolute accuracy survives. The test skips it for
  relative comparison and checks it absolutely.
"""

import mpmath as mp

mp.mp.dps = 60

XS = (0.0, 1e-8, 0.001, 0.1, 0.5, 1.0, 2.0, 3.5, 5.0, 8.0, 15.0, 50.0, 1000.0)
YS = (0.0, 1e-8, 0.001, 0.1, 1.0, 5.0, 30.0)

FIRST_ZERO = (1.99146684283, -1.35481012811)


def points():
    for x in XS:
        for y in YS:
            yield x, y

    # negative real part: Re w even in x, Im w odd in x
    for x in (-0.5, -2.0, -6.0):
        for y in (0.0, 0.25, 3.0):
            yield x, y

    # the lower half-plane, i.e. the reflection
    for x in (0.0, 0.5, 2.0, 6.0):
        for y in (-0.001, -0.5, -2.0, -8.0):
            yield x, y

    yield FIRST_ZERO
    yield (1.9, -1.3)


def w(x, y):
    z = mp.mpc(x, y)
    return mp.exp(-z * z) * mp.erfc(-1j * z)


def main():
    pts = list(points())

    print("/// Reference values `(Re z, Im z, Re w, Im w)`, from mpmath at 60 digits.")
    print("///")
    print("/// Regenerate with `faddeeva_reference.py`, in this directory.")
    print("#[rustfmt::skip]")
    print(f"const REF: [(f64, f64, f64, f64); {len(pts)}] = [")
    for x, y in pts:
        v = w(x, y)
        print(f"    ({x!r}, {y!r}, {float(v.real)!r}, {float(v.imag)!r}),")
    print("];")


if __name__ == "__main__":
    main()
