"""Reference values for `tests/airy.rs`: the Airy functions of complex argument in SciPy's
`airye` scalings, from mpmath.

    python airy_reference.py > rows.txt

Rows are `(re z, im z, eAi, eAip, eBi, eBip)` with each complex value as a `(re, im)`
pair, where with the principal `zeta = (2/3) z sqrt(z)`

    eAi = e^{zeta} Ai(z)      eAip = e^{zeta} Ai'(z)
    eBi = e^{-|Re zeta|} Bi   eBip = e^{-|Re zeta|} Bi'

The grid covers the whole plane: both sectors of the implementation, points on and just
either side of the `+-2pi/3` rays where it switches route, the negative real axis from
above, and `|z|` out to 300, where `Ai` alone would have underflowed long ago.
"""

import mpmath as mp

mp.mp.dps = 40


def f(x):
    return mp.nstr(x, 17, min_fixed=-3, max_fixed=3, strip_zeros=False)


def c(w):
    return f"{f(w.real)}, {f(w.imag)}"


def polar(r, deg):
    a = mp.mpf(deg) * mp.pi / 180
    return (float(r * mp.cos(a)), float(r * mp.sin(a)))


ZS = [
    (0.1, 0.1), (0.0, 0.001), (0.5, 0.3), (1.5, 1.5), (2.5, -1.0), (0.0, 3.0), (0.0, -3.0),
    (4.0, 0.5), (8.0, 5.0), (15.0, -4.0), (30.0, 30.0), (100.0, 1.0), (300.0, -100.0),
    (2.0, 0.0), (7.0, 0.0), (40.0, 0.0), (-2.0, 0.0), (-9.0, 0.0), (-40.0, 0.0), (-0.5, 0.2),
    (-3.0, 2.0), (-3.0, -2.0), (-12.0, 5.0), (-12.0, -5.0), (-50.0, 20.0), (-0.01, -0.02),
    (-5.0, 60.0), (-80.0, -1.0),
]

# The +-2pi/3 rays and their neighbourhood, at several radii: the seam between the sectors.
for r in (0.7, 3.0, 12.0, 45.0):
    for deg in (118.0, 119.9, 120.0, 120.1, 122.0, -118.0, -119.9, -120.0, -120.1, -122.0):
        ZS.append(polar(mp.mpf(r), deg))


def rows():
    for re, im in ZS:
        z = mp.mpc(re, im)
        zeta = mp.mpf(2) / 3 * z * mp.sqrt(z)
        # The decaying function is a difference of two growing ones inside mpmath, so give it
        # the digits that cancellation eats, as the Bessel generator does.
        with mp.workdps(60 + int(1.5 * abs(zeta))):
            ai = mp.airyai(z)
            aip = mp.airyai(z, derivative=1)
            bi = mp.airybi(z)
            bip = mp.airybi(z, derivative=1)
            eai = mp.exp(zeta) * ai
            eaip = mp.exp(zeta) * aip
            ebi = mp.exp(-abs(zeta.real)) * bi
            ebip = mp.exp(-abs(zeta.real)) * bip
        yield z, eai, eaip, ebi, ebip


if __name__ == "__main__":
    for z, eai, eaip, ebi, ebip in rows():
        print(f"    ({c(z)}, {c(eai)}, {c(eaip)}, {c(ebi)}, {c(ebip)}),")
