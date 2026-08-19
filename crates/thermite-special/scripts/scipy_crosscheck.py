"""Differential cross-check of thermite-special / thermite-complex against `scipy.special`.

Every reference table in this crate was generated with mpmath by the same person who wrote
the kernel it checks. That makes those tables precise but not independent: a systematic
misunderstanding of a definition, a normalization convention or a branch cut agrees with
itself at any number of digits. SciPy is a second implementation by unrelated authors, so it
catches the class of error mpmath tables structurally cannot.

Usage:

    cargo run --release -p thermite-special --example scipy_dump > grid.csv
    python scripts/scipy_crosscheck.py grid.csv

Reports the worst relative deviation per function. This is a validation tool, not a test:
the suite stays self-contained with embedded tables and CI grows no Python dependency.

Note on tolerances. A disagreement here is not automatically a thermite bug - several
scipy routines are themselves only good to a few ulp, and a couple of these functions are
*more* accurate here than in scipy by construction (`erfcx` past the `erfc` underflow,
`poisson_pmf` through Loader's saddle-point form rather than `exp(k ln l - l - lgamma)`).
The value of the check is the shape of the disagreement, not a pass/fail line.
"""

import csv
import sys
from collections import defaultdict

import numpy as np
import scipy.special as sp
from scipy.stats import yeojohnson


def w0(x):
    return sp.lambertw(x, 0).real


def wm1(x):
    return sp.lambertw(x, -1).real


def poisson(k, lam):
    # Real k: the Gamma density is the same function, which is why thermite takes a real k.
    return np.exp(k * np.log(lam) - lam - sp.gammaln(k + 1))


# name -> (arity, callable). Arity 1 uses column `a`, 2 uses `a` and `b`.
REAL = {
    "erf": (1, sp.erf),
    "erfc": (1, sp.erfc),
    "erfcx": (1, sp.erfcx),
    "erfinv": (1, sp.erfinv),
    "probit": (1, sp.ndtri),
    "tgamma": (1, sp.gamma),
    "lgamma": (1, lambda x: sp.gammaln(x)),
    "digamma": (1, sp.psi),
    "beta": (2, sp.beta),
    "lbeta": (2, sp.betaln),
    "lambertw0": (1, w0),
    "lambertwm1": (1, wm1),
    "expint1": (1, lambda x: sp.expn(1, x)),
    "expint2": (1, lambda x: sp.expn(2, x)),
    "expint3": (1, lambda x: sp.expn(3, x)),
    "expit": (1, sp.expit),
    "logit": (1, sp.logit),
    "entr": (1, sp.entr),
    "xlogy": (2, sp.xlogy),
    "xlog1py": (2, sp.xlog1py),
    "rel_entr": (2, sp.rel_entr),
    "kl_div": (2, sp.kl_div),
    "boxcox": (2, sp.boxcox),
    "boxcox1p": (2, sp.boxcox1p),
    "inv_boxcox": (2, sp.inv_boxcox),
    "inv_boxcox1p": (2, sp.inv_boxcox1p),
    "yeojohnson": (2, lambda y, l: yeojohnson(np.array([y, 0.0]), lmbda=l)[0]),
    "hermite3": (1, lambda x: sp.eval_hermite(3, x)),
    "hermite8": (1, lambda x: sp.eval_hermite(8, x)),
    "legendre5": (1, lambda x: sp.eval_legendre(5, x)),
    "legendre7_2": (1, lambda x: sp.lpmv(2, 7, x)),
    "laguerre4": (2, lambda x, a: sp.eval_genlaguerre(4, 0.0, x)),
    "laguerre6a": (2, lambda x, a: sp.eval_genlaguerre(6, a, x)),
    "jacobi4": (2, lambda x, _: sp.eval_jacobi(4, 0.5, 1.5, x)),
    "poisson_pmf": (2, poisson),
    "voigt_xy": (2, lambda x, y: sp.wofz(complex(x, y)).real),
}

COMPLEX = {"wofz": lambda x, y: sp.wofz(complex(x, y))}


def rel(got, want):
    if not np.isfinite(want) or not np.isfinite(got):
        # inf == inf and nan == nan both count as agreement.
        same = (np.isnan(got) and np.isnan(want)) or (got == want)
        return 0.0 if same else float("inf")
    scale = abs(want)
    if scale < 1e-290:
        return abs(got - want)
    return abs(got - want) / scale


def main(path):
    worst = defaultdict(lambda: (0.0, None))
    counts = defaultdict(int)

    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            name = row["fn"]
            a, b = float(row["a"]), float(row["b"])
            got_re, got_im = float(row["re"]), float(row["im"])

            if name in COMPLEX:
                want = COMPLEX[name](a, b)
                e = max(rel(got_re, want.real), rel(got_im, want.imag))
                got = complex(got_re, got_im)
            elif name in REAL:
                arity, fn = REAL[name]
                try:
                    want = float(fn(a) if arity == 1 else fn(a, b))
                except (ValueError, ZeroDivisionError):
                    continue
                e = rel(got_re, want)
                got = got_re
            else:
                continue

            counts[name] += 1
            if e > worst[name][0]:
                worst[name] = (e, f"a={a!r} b={b!r} got={got!r} scipy={want!r}")

    print(f"{'function':14} {'n':>5} {'worst rel':>11}   note")
    for name in sorted(worst):
        e, where = worst[name]
        note = "" if e < 1e-12 else "  <-- inspect"
        print(f"{name:14} {counts[name]:5} {e:11.3e}{note}")
        if e >= 1e-12:
            print(f"    {where}")

    total = sum(counts.values())
    bad = sum(1 for n in worst if worst[n][0] >= 1e-12)
    print(f"\n{total} comparisons, {len(counts)} functions, {bad} above 1e-12")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "grid.csv"))
