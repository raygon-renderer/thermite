"""Reference values for the power-transform family, at 60 digits.

Generates the tables embedded in `tests/power_transform.rs`. Each transform is written
in the cancellation-free form (`expm1`/`log1p` rather than `power(...) - 1`), because at
60 digits the naive spelling still cancels for the same reason it does in f64 - it just
takes a smaller lambda to show it, and the tables deliberately include those rows.

    python scripts/power_transform_ref.py
"""

from mpmath import mp, mpf, exp, expm1, log1p

mp.dps = 60


def boxcox_1p(x, lam):
    x, lam = mpf(repr(x)), mpf(repr(lam))
    return log1p(x) if lam == 0 else expm1(lam * log1p(x)) / lam


def inv_boxcox(y, lam):
    y, lam = mpf(repr(y)), mpf(repr(lam))
    return exp(y) if lam == 0 else exp(log1p(lam * y) / lam)


def inv_boxcox_1p(y, lam):
    y, lam = mpf(repr(y)), mpf(repr(lam))
    return expm1(y) if lam == 0 else expm1(log1p(lam * y) / lam)


def yeo_johnson(y, lam):
    """The four published cases, written out rather than folded, so that the reference
    does not share the kernel's sign-fold identity."""
    y, lam = mpf(repr(y)), mpf(repr(lam))
    if y >= 0:
        return log1p(y) if lam == 0 else expm1(lam * log1p(y)) / lam
    if lam == 2:
        return -log1p(-y)
    return -expm1((2 - lam) * log1p(-y)) / (2 - lam)


def inv_yeo_johnson(z, lam):
    z, lam = mpf(repr(z)), mpf(repr(lam))
    if z >= 0:
        return expm1(z) if lam == 0 else expm1(log1p(lam * z) / lam)
    if lam == 2:
        return 1 - exp(-z)
    return 1 - exp(log1p((lam - 2) * z) / (2 - lam))


TABLES = {
    "BOXCOX_1P": (boxcox_1p, [
        (0.0, 0.5), (1.0, 0.0), (1e-20, 0.5), (-1e-20, 2.0), (0.5, 1e-18), (2.0, 0.5),
        (-0.5, 3.0), (1e5, 0.25), (-0.9, 2.0), (3.0, -1.5), (1e-12, 7.0),
    ]),
    "INV_BOXCOX": (inv_boxcox, [
        (0.0, 0.5), (1.0, 0.0), (-0.6931471805599453, 0.0), (0.8284271247461901, 0.5),
        (49.5, 2.0), (0.99, -1.0), (-1.21895141649746, -1.5), (2.3025851195035365, 1e-8),
        (0.6931471805601855, 1e-12), (-0.6931471805599453, 1e-300), (2.0, 3.0),
        (10.0, 0.1), (-0.5, -2.0), (-1.999, 0.5),
    ]),
    "INV_BOXCOX_1P": (inv_boxcox_1p, [
        (0.0, 0.5), (1.0, 0.0), (1e-20, 0.5), (0.5, 1e-18), (1.4641016151377546, 0.5),
        (2.0, 3.0), (-0.5, -2.0), (1e-18, 7.0), (-1.5, 0.5),
    ]),
    "YEO_JOHNSON": (yeo_johnson, [
        (0.0, 0.5), (1.0, 0.0), (-1.0, 2.0), (2.0, 0.5), (-2.0, 0.5), (3.0, 1.5),
        (-3.0, 1.5), (0.5, -1.0), (-0.5, -1.0), (1e-20, 0.5), (-1e-20, 0.5),
        (10.0, 0.0), (-10.0, 2.0), (5.0, 3.0), (-5.0, 3.0), (1e5, 0.25), (-1e5, 0.25),
        (0.25, 1e-12), (-0.25, 1e-12), (7.0, 2.0), (-7.0, 0.0),
    ]),
    "INV_YEO_JOHNSON": (inv_yeo_johnson, [
        (0.0, 0.5), (0.6931471805599453, 0.0), (-0.6931471805599453, 2.0),
        (1.4641016151377546, 0.5), (-2.797434948471088, 0.5), (4.666666666666667, 1.5),
        (-2.0, 1.5), (1e-20, 0.5), (-1e-20, 0.5), (71.66666666666667, 3.0),
        (-0.8333333333333334, 3.0),
    ]),
}

if __name__ == "__main__":
    for name, (fn, grid) in TABLES.items():
        print(f"{name}")
        for a, lam in grid:
            print("    (%r, %r, %r)," % (a, lam, float(fn(a, lam))))
        print()
