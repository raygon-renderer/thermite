"""References for `bessel_i_ratio` / `inv_bessel_i_ratio` and Gauss-Legendre nodes.

Generates `tests/bessel_ratio_ref/table.rs`:

- `RATIO`: `(nu, x, A)` with `A = I_nu(x)/I_{nu-1}(x)` at 50 digits, `x` the exact f64.
- `INV_RATIO`: `(nu, r, kappa)` with `r = A(kappa0)` rounded to f64 and `kappa` the exact
  inverse of that `r` by `findroot`, so the kernel is compared to the true inverse of the
  argument it was given.
- `RATIO_1M` / `INV_RATIO_1M`: the complement `1 - A` and its inverse on `t = 1 - r`, the
  inverse rows capped at `kappa = 1e5` where mpmath's `besseli` stops converging.
- `GAUSS_LEGENDRE`, `GAUSS_HERMITE`, `GAUSS_LAGUERRE`: `(n, k, node, weight)` (Laguerre with
  `alpha`) for the k-th root (0-based, descending), each by sign bisection in a bracket set
  in the WKB angle where the roots are evenly spaced, weights from the derivative.

Every root uses the local `bisect`, not mpmath's: its secant walked to neighbouring roots,
its bisector judges convergence by `|f|`, and at exact resolution it escalates precision on
a value that is zero to 50 digits.

    python scripts/bessel_ratio_ref.py
"""
import os

from mpmath import mp, mpf, besseli, findroot, legendre, diff, cos, pi

mp.dps = 50
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "tests", "bessel_ratio_ref", "table.rs")


def bisect(f, bracket):
    """Plain sign bisection to the working precision. mpmath's `findroot` bisector judges
    convergence by |f|, which a degree-100 polynomial never satisfies, and its secant
    wanders to neighbouring roots."""
    lo, hi = bracket
    flo, fhi = f(lo), f(hi)
    assert flo * fhi < 0, ("no sign change", lo, hi, flo, fhi)
    for _ in range(200):
        mid = (lo + hi) / 2
        # Stop well before the working precision: at exact resolution mpmath tries to
        # resolve a polynomial that is zero to 50 digits and escalates to thousands of bits.
        if mid == lo or mid == hi or hi - lo < mpf(10) ** -44 * max(abs(mid), mpf(1)):
            break
        fm = f(mid)
        if (fm < 0) == (flo < 0):
            lo, flo = mid, fm
        else:
            hi, fhi = mid, fm
    return (lo + hi) / 2


def ratio(nu, x):
    nu, x = mpf(nu), mpf(x)
    if x == 0:
        return mpf(0)
    # scaled to survive large x
    return besseli(nu, x) / besseli(nu - 1, x)


NUS = [1.0, 1.5, 2.0, 3.5, 5.0, 25.0, 150.0]
XS = [1e-8, 1e-3, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0, 1000.0, 1e5]

ratio_rows = []
for nu in NUS:
    for x in XS:
        ratio_rows.append((nu, x, float(ratio(nu, x))))

inv_rows = []
for nu in NUS:
    for k0 in [1e-6, 0.01, 0.3, 1.0, 3.0, 10.0, 40.0, 300.0, 3000.0, 1e5]:
        r = float(ratio(nu, k0))
        if r == 0.0 or r >= 1.0:
            continue
        rm = mpf(r)
        kappa = findroot(lambda t: ratio(nu, t) - rm, mpf(k0))
        inv_rows.append((nu, r, float(kappa)))

# Complement rows: the far tail, where 1 - A is what the kernel evaluates directly, plus a
# few inside x < 8 nu where it is 1 - A from the forward.
ratio_1m_rows = []
for nu in NUS:
    for x in [5.0, 20.0, 30.0, 50.0, 100.0, 300.0, 1000.0, 1e4, 1e5]:
        ratio_1m_rows.append((nu, x, float(1 - ratio(nu, x))))

inv_1m_rows = []
for nu in NUS:
    # mpmath's besseli series stops converging past kappa ~ 1e5 at these orders, so the
    # smallest t per order is the one with kappa ~ (2 nu - 1)/(2 t) = 1e5.
    t_small = max(1e-8, (2 * nu - 1) / 2e5)
    for t in [t_small, 1e-2, 0.05, 0.2, 0.6, 1.0, 1.5]:
        tm = mpf(t)
        p = 2 * mpf(nu)
        # Banerjee in the complement, then bracketed bisection on 1 - A = t.
        r = 1 - tm
        k0 = r * (p - r * r) / (1 - r * r) if t < 1 else mpf(1)
        f = lambda k, nu=nu, tm=tm: (1 - ratio(nu, k)) - tm
        if t == 1:
            kappa = mpf(0)
        elif t < 1:
            try:
                kappa = bisect(f, (k0 / 3, 3 * k0))
            except ValueError as e:
                raise SystemExit(f"inverse complement bracket failed at nu={nu}, t={t}, k0={k0}: {e}")
        else:
            # t in (1, 2]: r < 0, the mirrored root
            r2 = tm - 1
            k1 = r2 * (p - r2 * r2) / (1 - r2 * r2)
            kappa = -bisect(lambda k, nu=nu, r2=r2: ratio(nu, k) - r2, (k1 / 2, 2 * k1))
        inv_1m_rows.append((nu, t, float(kappa)))

gl_rows = []
for n in [1, 2, 3, 5, 8, 16, 33, 64, 128]:
    for k in range(n):
        if n > 8 and k not in (0, 1, n // 2, n - 1):
            continue
        # Tricomi's seed for the k-th root from the right. Bracketed bisection, not the
        # default secant: the secant jumped to a neighbouring root at n = 8, k = 6. The
        # roots are spaced about pi/n apart and the seed is within a fraction of that.
        # Bracket in the angle, where the roots are evenly spaced (pi/(n + 1/2) apart). In x
        # they cluster at the ends, and a bracket of +-1/n holds several roots at n = 128.
        theta = pi * (mpf(k) + mpf(3) / 4) / (mpf(n) + mpf(1) / 2)
        delta = pi * mpf("0.45") / (mpf(n) + mpf(1) / 2)
        seed = cos(theta)
        lo, hi = cos(theta + delta), cos(theta - delta)
        x = bisect(lambda t: legendre(n, t), (lo, hi)) if n > 1 else mpf(0)
        assert lo < x < hi, (n, k, x, seed)
        d = diff(lambda t: legendre(n, t), x)
        w = 2 / ((1 - x * x) * d * d)
        gl_rows.append((n, k, float(x), float(w)))

from mpmath import hermite, laguerre, gamma, factorial, sqrt, sin, mpmathify

# Gauss-Hermite: roots of H_n from the WKB phase phi - sin(2 phi)/2 = 2 pi (k + 3/4)/(2n + 1),
# x = sqrt(2n + 1) cos(phi), k from the right, the left half by symmetry.
gh_rows = []
for n in [1, 2, 3, 5, 8, 16, 32, 64, 100]:
    for k in range(n):
        if n > 8 and k not in (0, 1, n // 2, n - 1):
            continue
        kk = min(k, n - 1 - k)
        if n == 1 or 2 * kk == n - 1:
            x = mpf(0)  # the middle root of an odd degree is exactly zero
        else:
            c = 2 * pi * (mpf(kk) + mpf(3) / 4) / (2 * n + 1)
            phi = findroot(lambda p: p - sin(2 * p) / 2 - c, (mpf("1.5") * c) ** (mpf(1) / 3))
            # bracket half a root spacing in phi: d phi = 2 pi / ((2n+1) (1 - cos 2 phi))
            dphi = 2 * pi / ((2 * n + 1) * (1 - cos(2 * phi)))
            lo, hi = sqrt(2 * n + 1) * cos(phi + mpf("0.45") * dphi), sqrt(2 * n + 1) * cos(phi - mpf("0.45") * dphi)
            x = bisect(lambda t: hermite(n, t), (lo, hi))
            if k != kk:
                x = -x
        w = 2 ** (n - 1) * factorial(n) * sqrt(pi) / (n * n * hermite(n - 1, x) ** 2) if n > 1 else sqrt(pi)
        gh_rows.append((n, k, float(x), float(w)))

# Gauss-Laguerre: roots of L_n^alpha from psi - sin psi = 4 pi (k + 3/4)/nu, nu = 4n + 2 alpha + 2,
# x = nu cos^2(psi/2), k from the right.
glag_rows = []
for n in [1, 2, 3, 5, 8, 16, 32, 64]:
    for alpha in [0.0, 0.5, 2.0]:
        for k in range(n):
            if n > 8 and k not in (0, 1, n // 2, n - 1):
                continue
            a = mpf(alpha)
            nuv = 4 * n + 2 * a + 2
            if n == 1:
                x = 1 + a
            else:
                c = 4 * pi * (mpf(k) + mpf(3) / 4) / nuv
                psi = findroot(lambda p: p - sin(p) - c, min((6 * c) ** (mpf(1) / 3), pi - mpf("0.1")))
                dpsi = 4 * pi / (nuv * (1 - cos(psi)))
                lo = nuv * (1 + cos(min(psi + mpf("0.45") * dpsi, pi))) / 2
                hi = nuv * (1 + cos(max(psi - mpf("0.45") * dpsi, mpf(0)))) / 2
                x = bisect(lambda t: laguerre(n, a, t), (lo, hi))
            d = diff(lambda t: laguerre(n, a, t), x)
            w = gamma(n + a + 1) / (factorial(n) * x * d * d)
            glag_rows.append((n, alpha, k, float(x), float(w)))

out = ["// Generated by scripts/bessel_ratio_ref.py (mpmath, 50 dps). Do not edit.", ""]
out += ["// (nu, x, 1 - I_nu(x)/I_{nu-1}(x))", "#[rustfmt::skip]", "#[allow(clippy::excessive_precision, clippy::approx_constant, dead_code)]",
        f"const RATIO_1M: [(f64, f64, f64); {len(ratio_1m_rows)}] = ["]
out += [f"    ({a!r}, {b!r}, {c!r})," for a, b, c in ratio_1m_rows]
out += ["];", ""]
out += ["// (nu, t, kappa with 1 - I_nu(kappa)/I_{nu-1}(kappa) = t exactly)", "#[rustfmt::skip]",
        "#[allow(clippy::excessive_precision, clippy::approx_constant, dead_code)]", f"const INV_RATIO_1M: [(f64, f64, f64); {len(inv_1m_rows)}] = ["]
out += [f"    ({a!r}, {b!r}, {c!r})," for a, b, c in inv_1m_rows]
out += ["];", ""]
out += ["// (n, k, x_k, w_k): k-th root of H_n from the right, 0-based", "#[rustfmt::skip]",
        "#[allow(clippy::excessive_precision, clippy::approx_constant, dead_code)]", f"const GAUSS_HERMITE: [(u32, u32, f64, f64); {len(gh_rows)}] = ["]
out += [f"    ({n}, {k}, {x!r}, {w!r})," for n, k, x, w in gh_rows]
out += ["];", ""]
out += ["// (n, alpha, k, x_k, w_k): k-th root of L_n^alpha from the right, 0-based", "#[rustfmt::skip]",
        "#[allow(clippy::excessive_precision, clippy::approx_constant, dead_code)]", f"const GAUSS_LAGUERRE: [(u32, f64, u32, f64, f64); {len(glag_rows)}] = ["]
out += [f"    ({n}, {a!r}, {k}, {x!r}, {w!r})," for n, a, k, x, w in glag_rows]
out += ["];", ""]
out += ["// (nu, x, I_nu(x)/I_{nu-1}(x))", "#[rustfmt::skip]", "#[allow(clippy::excessive_precision, clippy::approx_constant, dead_code)]",
        f"const RATIO: [(f64, f64, f64); {len(ratio_rows)}] = ["]
out += [f"    ({a!r}, {b!r}, {c!r})," for a, b, c in ratio_rows]
out += ["];", ""]
out += ["// (nu, r, kappa with I_nu(kappa)/I_{nu-1}(kappa) = r exactly)", "#[rustfmt::skip]",
        "#[allow(clippy::excessive_precision, clippy::approx_constant, dead_code)]", f"const INV_RATIO: [(f64, f64, f64); {len(inv_rows)}] = ["]
out += [f"    ({a!r}, {b!r}, {c!r})," for a, b, c in inv_rows]
out += ["];", ""]
out += ["// (n, k, x_k, w_k): k-th root of P_n from the right, 0-based", "#[rustfmt::skip]",
        "#[allow(clippy::excessive_precision, clippy::approx_constant, dead_code)]", f"const GAUSS_LEGENDRE: [(u32, u32, f64, f64); {len(gl_rows)}] = ["]
out += [f"    ({n}, {k}, {x!r}, {w!r})," for n, k, x, w in gl_rows]
out += ["];"]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="\n") as f:
    f.write("\n".join(out) + "\n")
print("wrote", OUT, len(ratio_rows), len(inv_rows), len(gl_rows))
