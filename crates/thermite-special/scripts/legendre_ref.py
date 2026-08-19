"""Exact reference values for the Legendre and Hermite tests.

Everything is computed in `fractions.Fraction` from the defining recurrences and
rounded to f64 exactly once at print time, so the emitted tables carry no
floating-point history of their own. The x values are dyadic rationals, which are
exact in binary64, so the only rounding anywhere is that final one.

Usage:
    python crates/thermite-special/scripts/legendre_ref.py

Paste the output into crates/thermite-special/tests/{legendre,hermite}.rs.
"""

from fractions import Fraction as F

# Dyadic, exact in f64, and none of them a root of a low-degree P_n (which would
# make a relative-error check measure the reference's own conditioning instead).
XS = [
    F(1, 8),
    F(1, 4),
    F(3, 8),
    F(1, 2),
    F(5, 8),
    F(3, 4),
    F(7, 8),
    F(-5, 16),
    F(-11, 16),
]

MAX_LEGENDRE = 16  # past the hard-coded degree 13, into the runtime recurrence
MAX_HERMITE = 12


def legendre(n, x):
    """(n+1) P_{n+1} = (2n+1) x P_n - n P_{n-1}, exactly."""
    if n == 0:
        return F(1)
    p0, p1 = F(1), x
    for k in range(1, n):
        p0, p1 = p1, (F(2 * k + 1) * x * p1 - F(k) * p0) / F(k + 1)
    return p1


def hermite(n, x):
    """Physicists': H_{n+1} = 2x H_n - 2n H_{n-1}, exactly."""
    if n == 0:
        return F(1)
    h0, h1 = F(1), 2 * x
    for k in range(1, n):
        h0, h1 = h1, 2 * x * h1 - 2 * F(k) * h0
    return h1


def assoc_legendre(n, m, x):
    """Condon-Shortley: P_n^m = (-1)^m (1-x^2)^{m/2} d^m/dx^m P_n.

    Returns (rational_part, half_powers) where the value is
    rational_part * sqrt(1-x^2)^half_powers, with half_powers in {0, 1}. Keeping the
    irrational factor symbolic means the emitted constant stays exact for even m and
    the test applies a single sqrt for odd m.
    """
    # Coefficients of P_n in the monomial basis, exactly.
    coeffs = [F(0)] * (n + 1)
    if n == 0:
        coeffs[0] = F(1)
    else:
        prev = [F(1)] + [F(0)] * n
        cur = [F(0), F(1)] + [F(0)] * (n - 1)
        for k in range(1, n):
            nxt = [F(0)] * (n + 1)
            for i in range(n):
                nxt[i + 1] += F(2 * k + 1) * cur[i]
            for i in range(n + 1):
                nxt[i] -= F(k) * prev[i]
            nxt = [c / F(k + 1) for c in nxt]
            prev, cur = cur, nxt
        coeffs = cur

    # m-th derivative.
    for _ in range(m):
        coeffs = [coeffs[i] * F(i) for i in range(1, len(coeffs))] or [F(0)]

    d = sum(c * x**i for i, c in enumerate(coeffs))
    sign = F((-1) ** m)
    one_minus = 1 - x * x
    # (1-x^2)^{m/2} = (1-x^2)^{m//2} * sqrt(1-x^2)^{m%2}
    return sign * d * one_minus ** (m // 2), m % 2


def legendre_coeffs(n):
    """Monomial coefficients of P_n, exactly."""
    if n == 0:
        return [F(1)]
    prev = [F(1)] + [F(0)] * n
    cur = [F(0), F(1)] + [F(0)] * (n - 1)
    for k in range(1, n):
        nxt = [F(0)] * (n + 1)
        for i in range(n):
            nxt[i + 1] += F(2 * k + 1) * cur[i]
        for i in range(n + 1):
            nxt[i] -= F(k) * prev[i]
        nxt = [c / F(k + 1) for c in nxt]
        prev, cur = cur, nxt
    return cur


def condition(n, x):
    """sum |c_k| |x|^k - what any monomial-basis evaluation of P_n must cancel through.

    The kernel's Estrin forms are these same coefficients regrouped, so this is the
    right error scale for degrees 1..13. It is the reason a flat ulp tolerance is wrong:
    P_12(0.875) cancels terms of order 2000 down to 0.23.
    """
    return sum(abs(c) * abs(x) ** k for k, c in enumerate(legendre_coeffs(n)))


def hermite_coeffs(n):
    """Monomial coefficients of the physicists' H_n, exactly."""
    prev = [F(1)] + [F(0)] * n
    cur = [F(0), F(1) * 2] + [F(0)] * (n - 1) if n >= 1 else [F(1)]
    if n == 0:
        return [F(1)]
    for k in range(1, n):
        nxt = [F(0)] * (n + 1)
        for i in range(n):
            nxt[i + 1] += 2 * cur[i]
        for i in range(n + 1):
            nxt[i] -= 2 * F(k) * prev[i]
        prev, cur = cur, nxt
    return cur


def cond_of(coeffs, x):
    return sum(abs(c) * abs(x) ** k for k, c in enumerate(coeffs))


# (alpha, beta) pairs: the Legendre case, the two Chebyshev cases, and asymmetric
# integer and half-integer weights that no symmetry could accidentally satisfy.
AB = [(F(0), F(0)), (F(-1, 2), F(-1, 2)), (F(1, 2), F(1, 2)), (F(1), F(2)), (F(3, 2), F(-1, 2))]


def jacobi_coeffs(n, a, b):
    """Monomial coefficients of P_n^{(a,b)} via the three-term recurrence, exactly."""
    if n == 0:
        return [F(1)]
    polys = [[F(1)], [(a - b) / 2, (a + b + 2) / 2]]
    for k in range(2, n + 1):
        c0 = 2 * k * (k + a + b) * (2 * k + a + b - 2)
        c1x = (2 * k + a + b - 1) * (2 * k + a + b) * (2 * k + a + b - 2)
        c1 = (2 * k + a + b - 1) * (a * a - b * b)
        c2 = 2 * (k + a - 1) * (k + b - 1) * (2 * k + a + b)

        prev, prev2 = polys[k - 1], polys[k - 2]
        out = [F(0)] * (k + 1)
        for i, c in enumerate(prev):
            out[i + 1] += c1x * c
            out[i] += c1 * c
        for i, c in enumerate(prev2):
            out[i] -= c2 * c
        polys.append([c / c0 for c in out])
    return polys[n]


def deriv(coeffs, m):
    for _ in range(m):
        coeffs = [coeffs[i] * F(i) for i in range(1, len(coeffs))] or [F(0)]
    return coeffs


def evaluate(coeffs, x):
    return sum(c * x**i for i, c in enumerate(coeffs))


def f64(v):
    return repr(float(v))


def emit_table(name, rows, width):
    print(f"const {name}: [[f64; {width}]; {len(rows)}] = [")
    for row in rows:
        print("    [" + ", ".join(f64(v) for v in row) + "],")
    print("];")
    print()


def main():
    print("// Generated by scripts/legendre_ref.py: exact rational arithmetic,")
    print("// rounded to f64 once. Do not hand-edit.")
    print()
    print(f"const XS: [f64; {len(XS)}] = [" + ", ".join(f64(x) for x in XS) + "];")
    print()

    emit_table(
        "LEGENDRE",
        [[legendre(n, x) for x in XS] for n in range(1, MAX_LEGENDRE + 1)],
        len(XS),
    )

    emit_table(
        "LEGENDRE_COND",
        [[condition(n, x) for x in XS] for n in range(1, MAX_LEGENDRE + 1)],
        len(XS),
    )

    # Degree-independent upper bound on the same quantity (|x| <= 1), for tests that
    # sweep x off the table grid.
    sigmas = [condition(n, F(1)) for n in range(0, 15)]
    print(f"const LEGENDRE_SIGMA: [f64; {len(sigmas)}] = [" + ", ".join(f64(s) for s in sigmas) + "];")
    print()

    emit_table(
        "HERMITE",
        [[hermite(n, x) for x in XS] for n in range(1, MAX_HERMITE + 1)],
        len(XS),
    )

    emit_table(
        "HERMITE_COND",
        [[cond_of(hermite_coeffs(n), x) for x in XS] for n in range(1, MAX_HERMITE + 1)],
        len(XS),
    )

    # Jacobi: value and conditioning together, since `jacobi` with m > 0 is the m-th
    # derivative and its cancellation is worse than the m = 0 case.
    print("// (alpha, beta, n, m, [values], [conditioning])")
    jrows = []
    for a, b in AB:
        for n in range(1, 8):
            for m in range(0, min(n, 3) + 1):
                c = deriv(jacobi_coeffs(n, a, b), m)
                jrows.append(
                    (
                        a,
                        b,
                        n,
                        m,
                        [evaluate(c, x) for x in XS],
                        [cond_of(c, x) for x in XS],
                    )
                )

    print(f"const JACOBI: [(f64, f64, u32, u32, [f64; {len(XS)}], [f64; {len(XS)}]); {len(jrows)}] = [")
    for a, b, n, m, vals, conds in jrows:
        v = ", ".join(f64(t) for t in vals)
        c = ", ".join(f64(t) for t in conds)
        print(f"    ({f64(a)}, {f64(b)}, {n}, {m}, [{v}], [{c}]),")
    print("];")
    print()

    # Associated Legendre, (n, m) with 1 <= m <= n <= 6. Emitted as the rational part
    # only, and the test multiplies back sqrt(1-x^2) where half_powers is 1.
    print("// (n, m, [values]) where the value is the entry times sqrt(1-x^2) iff m is odd.")
    rows = []
    for n in range(1, 7):
        for m in range(1, n + 1):
            vals = []
            for x in XS:
                v, half = assoc_legendre(n, m, x)
                assert half == m % 2
                vals.append(v)
            rows.append((n, m, vals))

    print(f"const ASSOC: [(u32, u32, [f64; {len(XS)}]); {len(rows)}] = [")
    for n, m, vals in rows:
        print(f"    ({n}, {m}, [" + ", ".join(f64(v) for v in vals) + "]),")
    print("];")


if __name__ == "__main__":
    main()
