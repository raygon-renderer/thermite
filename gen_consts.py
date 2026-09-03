"""
Single source of truth for every mathematical constant in the workspace.

    python gen_consts.py            # regenerate all six files
    python gen_consts.py --check    # exit 1 if anything is stale (CI)

Generates, in full:

    crates/thermite/src/math/consts/mod.rs          FloatConsts, f32/f64/Vector impls
    crates/thermite-compensated/src/consts/mod.rs   double-double splits + log tables
    crates/thermite-interval/src/consts/mod.rs      BoundedFloatConsts wiring
    crates/thermite-interval/src/consts_table.rs    exact enclosure pairs
    crates/thermite-special/src/tables/bernoulli.rs        BernoulliNumbers, f32/f64 tables
    crates/thermite-compensated/src/consts/bernoulli.rs    the same tables, double-double

The reusable machinery each of those needs lives in a hand-written `macros.rs`
beside it and is never touched here. Adding a constant is therefore one edit:
append it to `CONSTS` below and re-run this script. Nothing else in the
workspace lists constant names. Every other site goes through the
`for_each_float_const!` / `for_each_math_const!` callback macros emitted into
thermite's `consts/mod.rs`.

Numerics: every value is computed from its mathematical definition with mpmath
at 80 significant digits, then converted with EXACT rational arithmetic
(`fractions.Fraction`), so nothing rounds twice.

  * thermite gets a 40-digit decimal literal, which rounds correctly to nearest
    in both f32 and f64 (Rust parses float literals with correct rounding),
  * thermite-compensated gets `(RN(c), RN(c - RN(c)))` in the target format, as
    hex, so the split is unambiguous,
  * thermite-interval gets the two consecutive floats bracketing the true value,
    decided by exact rational comparison, degenerate when representable.

thermite-interval's `consts` test binary is the check on all of it: it audits
every constant for correct rounding and cross-checks the three representations
against each other.
"""

import os
import struct
import sys
from fractions import Fraction

import mpmath as mp

mp.mp.dps = 80

ROOT = os.path.dirname(os.path.abspath(__file__))

pi = +mp.pi
e = +mp.e
euler = +mp.euler
phi = (1 + mp.sqrt(5)) / 2
plastic = mp.findroot(lambda x: x**3 - x - 1, mp.mpf("1.3247"))


def reciprocal_fibonacci():
    """Sum of 1/F(k), k >= 1. Terms fall off like phi^-k, so 400 of them are
    far below the 80-digit working precision."""
    return mp.fsum(1 / mp.fib(k) for k in range(1, 401))


def laplace_limit():
    """Root of x*exp(sqrt(1+x^2)) / (1 + sqrt(1+x^2)) = 1."""
    f = lambda x: x * mp.exp(mp.sqrt(1 + x**2)) / (1 + mp.sqrt(1 + x**2)) - 1
    return mp.findroot(f, mp.mpf("0.6627434193"))


def erdos_borwein():
    """Sum of 1/(2^k - 1), k >= 1. Geometric, so 400 terms is far past 80 digits."""
    return mp.fsum(1 / (mp.mpf(2) ** k - 1) for k in range(1, 401))


def niven():
    """1 + sum of (1 - 1/zeta(k)), k >= 2. Terms fall off like 2^-k."""
    return 1 + mp.fsum(1 - 1 / mp.zeta(k) for k in range(2, 401))


def fransen_robinson():
    """Integral of 1/Gamma(x) over (0, inf). Split at the peak and out along the
    super-exponential tail; the node list is redundant enough that halving the
    spacing does not move a single one of the 80 digits."""
    return mp.quad(lambda x: 1 / mp.gamma(x), [0, 1, 2, 3, 5, 10, 20, 50, mp.inf])


def golomb_dickman():
    """lambda = integral of exp(li(t)) over (0, 1)."""
    return mp.quad(lambda t: mp.exp(mp.li(t)), [0, mp.mpf("0.5"), 1])


def artin(small_primes=100, terms=60):
    """A = prod_p (1 - 1/(p(p-1))), over ALL primes.

    The bare product converges like 1/p, so the tail is reached through the prime
    zeta function instead: 1/(p(p-1)) = sum_{j>=2} p^-j, hence

        log(1 - u) = -sum_{m>=1} u^m/m,   u^m = sum_{i>=0} C(m+i-1, i) p^-(2m+i)

    and collecting by exponent s = 2m+i turns the p-sum into P(s), which mpmath
    provides directly. Primes below `small_primes` are multiplied in exactly and
    subtracted out of each P(s): expanding them would need s in the thousands,
    because C(m+i-1, i) grows far faster than 2^-s shrinks. With the head taken
    exactly, u <= 1e-4 and 60 terms already agree past 40 digits."""
    small = list(mp.libmp.libintmath.list_primes(small_primes))
    head = mp.mpf(1)
    for p in small:
        head *= 1 - mp.mpf(1) / (p * (p - 1))
    log_tail = mp.mpf(0)
    for s in range(2, terms + 1):
        c = mp.fsum(mp.binomial(m + (s - 2 * m) - 1, s - 2 * m) / m for m in range(1, s // 2 + 1))
        log_tail -= c * (mp.primezeta(s) - mp.fsum(mp.mpf(p) ** -s for p in small))
    return head * mp.exp(log_tail)


# Defined by the Feigenbaum functional equation, which mpmath does not solve.
# OEIS A006890, digits transcribed.
FEIGENBAUM_DELTA = mp.mpf(
    "4.6692016091029906718532038204662016172581855774757686327456513430041343302113147371386897440239480138171659848551898"
)

# Marker for the three constants whose value depends on the format's precision.
# `EPS[name](p)` takes the mantissa bit count of the REPRESENTATION: 24/53 for
# a plain f32/f64, and 47/105 for a double-double built out of them.
EPS = {
    "EPSILON": lambda p: mp.mpf(2) ** -(p - 1),
    "SQRT_EPSILON": lambda p: mp.sqrt(mp.mpf(2) ** -(p - 1)),
    "FOURTH_ROOT_EPSILON": lambda p: mp.sqrt(mp.sqrt(mp.mpf(2) ** -(p - 1))),
}

# ---------------------------------------------------------------------------
# THE CONSTANTS. (name, value, doc). `None` as the value marks an EPS entry.
# Order here is the order everywhere: trait declaration, macro expansion, tables.
# ---------------------------------------------------------------------------

CONSTS = [
    ("NEG_ZERO", mp.mpf("-0"), "Negative zero (-0) (only sign bit set)"),
    ("E", e, "Euler's number (e)"),
    ("EULER_GAMMA", euler, "Euler-Mascheroni constant (γ)"),
    ("PI_SQUARED", pi**2, r"`$\pi^2$`"),
    ("PI_CUBED", pi**3, r"`$\pi^3$`"),
    ("PI_FOURTH", pi**4, r"`$\pi^4$`"),
    ("FRAC_1_PI", 1 / pi, r"`$1/\pi$`"),
    ("FRAC_1_SQRT_2", 1 / mp.sqrt(2), r"`$1/\sqrt{2}$`"),
    ("FRAC_1_SQRT_3", 1 / mp.sqrt(3), r"`$1/\sqrt{3}$`"),
    ("FRAC_1_SQRT_5", 1 / mp.sqrt(5), r"`$1/\sqrt{5}$`"),
    ("FRAC_2_PI", 2 / pi, r"`$2/\pi$`"),
    ("FRAC_1_SQRT_PI", 1 / mp.sqrt(pi), r"`$1/\sqrt{\pi}$`"),
    ("FRAC_1_SQRT_SQRT_PI", 1 / mp.sqrt(mp.sqrt(pi)), r"`$\pi^{-1/4}$`, the normalization of the Hermite functions"),
    ("FRAC_2_SQRT_PI", 2 / mp.sqrt(pi), r"`$2/\sqrt{\pi}$`"),
    ("FRAC_SQRT_PI_2", mp.sqrt(pi) / 2, r"`$\sqrt{\pi}/2$`"),
    ("FRAC_1_SQRT_TAU", 1 / mp.sqrt(2 * pi), r"`$1/\sqrt{2\pi}$`"),
    ("FRAC_PI_2", pi / 2, r"`$\pi/2$`"),
    ("FRAC_PI_3", pi / 3, r"`$\pi/3$`"),
    ("FRAC_PI_4", pi / 4, r"`$\pi/4$`"),
    ("FRAC_PI_6", pi / 6, r"`$\pi/6$`"),
    ("FRAC_PI_8", pi / 8, r"`$\pi/8$`"),
    ("FRAC_PI_180", pi / 180, r"`$\pi/180$`"),
    ("FRAC_180_PI", 180 / pi, r"`$180/\pi$`"),
    ("LN_2", mp.log(2), r"`$\ln 2$`"),
    ("LN_10", mp.log(10), r"`$\ln 10$`"),
    ("LN_PI", mp.log(pi), r"`$\ln \pi$`"),
    ("LN_TAU", mp.log(2 * pi), r"`$\ln 2\pi$`"),
    ("FRAC_LN_PI_2", mp.log(pi) / 2, r"`$\frac{1}{2}\ln \pi$`"),
    (
        "FRAC_LN_TAU_2",
        mp.log(2 * pi) / 2,
        r"`$\frac{1}{2}\ln 2\pi$`, the constant term of the Stirling series for `$\ln \Gamma$`",
    ),
    ("LOG2_10", mp.log(10, 2), r"`$\log_2 10$`"),
    ("LOG2_E", mp.log(e, 2), r"`$\log_2 e$`"),
    ("LOG2_PI", mp.log(pi, 2), r"`$\log_2 \pi$`"),
    ("LOG10_2", mp.log(2, 10), r"`$\log_{10} 2$`"),
    ("LOG10_E", mp.log(e, 10), r"`$\log_{10} e$`"),
    ("PI", pi, "Archimedes' constant (π)"),
    ("SQRT_2", mp.sqrt(2), r"`$\sqrt{2}$`"),
    ("SQRT_3", mp.sqrt(3), r"`$\sqrt{3}$`"),
    ("SQRT_5", mp.sqrt(5), r"`$\sqrt{5}$`"),
    ("SQRT_E", mp.sqrt(e), r"`$\sqrt{e}$`"),
    ("EPSILON", None, "The machine epsilon"),
    ("SQRT_EPSILON", None, r"The square root of the machine epsilon (`$\sqrt{\varepsilon}$`)"),
    ("FOURTH_ROOT_EPSILON", None, r"The fourth root of the machine epsilon (`$\sqrt[4]{\varepsilon}$`)"),
    ("TAU", 2 * pi, "The full circle constant (τ)"),
    ("SQRT_FRAC_PI_2", mp.sqrt(pi / 2), r"`$\sqrt{\pi/2}$`"),
    ("SQRT_TAU", mp.sqrt(2 * pi), r"`$\sqrt{2\pi}$`"),
    ("PHI", phi, "The golden ratio (φ)"),
    ("FRAC_1_PHI", 1 / phi, r"`$1/\varphi = \varphi - 1$`, the 1D golden-ratio low-discrepancy increment"),
    ("FRAC_1_PHI_SQUARED", 1 / phi**2, r"`$1/\varphi^2$`"),
    (
        "GOLDEN_ANGLE",
        2 * pi / phi**2,
        r"""The golden angle `$2\pi/\varphi^2 = \pi(3 - \sqrt{5})$` in radians

The rotation between successive samples in a Vogel (sunflower) disk spiral.""",
    ),
    ("FRAC_1_3", mp.mpf(1) / 3, r"`$1/3$`"),
    ("FRAC_2_3", mp.mpf(2) / 3, r"`$2/3$`"),
    ("FRAC_1_4", mp.mpf(1) / 4, r"`$1/4$`"),
    ("FRAC_1_6", mp.mpf(1) / 6, r"`$1/6$`"),
    ("FRAC_1_E", 1 / e, r"`$1/e$`"),
    ("FRAC_NEG_1_E", -1 / e, r"`$-1/e$`"),
    ("FRAC_1_2", mp.mpf(1) / 2, r"`$1/2$`"),
    ("FRAC_3_4", mp.mpf(3) / 4, r"`$3/4$`"),
    ("LN_LN_2", mp.log(mp.log(2)), r"`$\ln(\ln 2)$`, the median of the Gumbel distribution"),
    ("SQRT_LN_4", mp.sqrt(mp.log(4)), r"`$\sqrt{\ln 4}$`"),
    ("FRAC_2PI_3", 2 * pi / 3, r"`$2\pi/3$`"),
    ("FRAC_3PI_4", 3 * pi / 4, r"`$3\pi/4$`"),
    ("FRAC_4PI_3", 4 * pi / 3, r"`$4\pi/3$`, the volume of the unit sphere"),
    ("FOUR_PI", 4 * pi, r"`$4\pi$`, the solid angle of the whole sphere in steradians"),
    ("FRAC_1_4PI", 1 / (4 * pi), r"`$1/(4\pi)$`, the density of the uniform distribution on the sphere"),
    ("FRAC_1_TAU", 1 / (2 * pi), r"`$1/(2\pi)$`"),
    ("SQRT_PI", mp.sqrt(pi), r"`$\sqrt{\pi}$`"),
    ("PI_MINUS_3", pi - 3, r"`$\pi - 3$`"),
    ("FOUR_MINUS_PI", 4 - pi, r"`$4 - \pi$`"),
    ("PI_POW_E", pi**e, r"`$\pi^e$`"),
    ("CBRT_2", mp.cbrt(2), r"`$\sqrt[3]{2}$`"),
    ("CBRT_3", mp.cbrt(3), r"`$\sqrt[3]{3}$`"),
    ("CBRT_PI", mp.cbrt(pi), r"`$\sqrt[3]{\pi}$`"),
    ("FRAC_1_CBRT_PI", 1 / mp.cbrt(pi), r"`$1/\sqrt[3]{\pi}$`"),
    ("FRAC_1_SQRT_E", 1 / mp.sqrt(e), r"`$1/\sqrt{e} = e^{-1/2}$`"),
    ("E_POW_PI", e**pi, r"`$e^\pi$`, Gelfond's constant"),
    (
        "GELFOND_SCHNEIDER",
        mp.mpf(2) ** mp.sqrt(2),
        r"`$2^{\sqrt{2}}$`, the Gelfond-Schneider constant (also called Hilbert's number)",
    ),
    ("SIN_1", mp.sin(1), r"`$\sin 1$`"),
    ("COS_1", mp.cos(1), r"`$\cos 1$`"),
    ("TAN_1", mp.tan(1), r"`$\tan 1$`"),
    ("SINH_1", mp.sinh(1), r"`$\sinh 1$`"),
    ("COSH_1", mp.cosh(1), r"`$\cosh 1$`"),
    ("TANH_1", mp.tanh(1), r"`$\tanh 1$`"),
    ("LN_PHI", mp.log(phi), r"`$\ln \varphi$`"),
    ("FRAC_1_LN_PHI", 1 / mp.log(phi), r"`$1/\ln \varphi$`"),
    ("FRAC_1_EULER_GAMMA", 1 / euler, r"`$1/\gamma$`"),
    ("EULER_GAMMA_SQUARED", euler**2, r"`$\gamma^2$`"),
    ("ZETA_2", mp.zeta(2), r"`$\zeta(2) = \pi^2/6$`"),
    ("ZETA_3", mp.zeta(3), r"`$\zeta(3)$`, Apery's constant"),
    ("ZETA_4", mp.zeta(4), r"`$\zeta(4) = \pi^4/90$`"),
    ("CATALAN", +mp.catalan, r"Catalan's constant `$K$`"),
    ("GLAISHER", +mp.glaisher, r"The Glaisher-Kinkelin constant `$A$`"),
    ("KHINCHIN", +mp.khinchin, r"Khinchin's constant `$K_0$`"),
    (
        "LEVY",
        mp.exp(pi**2 / (12 * mp.log(2))),
        r"""Levy's constant `$e^{\pi^2/(12\ln 2)}$`

The limit of `$q_n^{1/n}$` for the denominators of almost every real number's
continued fraction expansion, the companion to [`KHINCHIN`].

[`KHINCHIN`]: FloatConsts::KHINCHIN""",
    ),
    (
        "EXTREME_VALUE_SKEWNESS",
        12 * mp.sqrt(6) * mp.zeta(3) / pi**3,
        r"`$12\sqrt{6}\,\zeta(3)/\pi^3$`, the skewness of the extreme value distribution",
    ),
    (
        "RAYLEIGH_SKEWNESS",
        2 * mp.sqrt(pi) * (pi - 3) / (4 - pi) ** mp.mpf("1.5"),
        r"`$2\sqrt{\pi}(\pi-3)/(4-\pi)^{3/2}$`, the skewness of the Rayleigh distribution",
    ),
    (
        "RAYLEIGH_KURTOSIS_EXCESS",
        -(6 * pi**2 - 24 * pi + 16) / (4 - pi) ** 2,
        r"`$-(6\pi^2 - 24\pi + 16)/(4-\pi)^2$`, the excess kurtosis of the Rayleigh distribution",
    ),
    (
        "RAYLEIGH_KURTOSIS",
        3 - (6 * pi**2 - 24 * pi + 16) / (4 - pi) ** 2,
        r"""`$3 - (6\pi^2 - 24\pi + 16)/(4-\pi)^2$`, the kurtosis of the Rayleigh distribution

Boost's constants table prints this formula with the opposite sign, but lists the
(correct) value 3.245089. Kurtosis is `3 + `[`RAYLEIGH_KURTOSIS_EXCESS`].

[`RAYLEIGH_KURTOSIS_EXCESS`]: FloatConsts::RAYLEIGH_KURTOSIS_EXCESS""",
    ),
    ("FEIGENBAUM_DELTA", FEIGENBAUM_DELTA, r"The first Feigenbaum constant `$\delta$`"),
    (
        "PLASTIC_RATIO",
        plastic,
        r"The plastic ratio `$\rho$`, the real root of `$x^3 = x + 1$`",
    ),
    (
        "FRAC_1_PLASTIC_RATIO",
        1 / plastic,
        r"""`$1/\rho$`, the first increment of the 2D R2 low-discrepancy sequence

The plastic ratio is to two dimensions what `$\varphi$` is to one: the pair
`$(1/\rho, 1/\rho^2)$` generates the additive-recurrence R2 sequence.""",
    ),
    ("FRAC_1_PLASTIC_RATIO_SQUARED", 1 / plastic**2, r"`$1/\rho^2$`, the second R2 increment"),
    ("GAUSS", 1 / mp.agm(1, mp.sqrt(2)), r"Gauss's constant `$G = 1/\mathrm{agm}(1, \sqrt{2})$`"),
    (
        "LEMNISCATE",
        pi / mp.agm(1, mp.sqrt(2)),
        r"The lemniscate constant `$\varpi = \pi G = 2\int_0^1 dt/\sqrt{1-t^4}$`",
    ),
    ("DOTTIE", mp.findroot(lambda x: mp.cos(x) - x, mp.mpf("0.739")), r"The Dottie number, the unique real solution of `$\cos x = x$`"),
    (
        "OMEGA",
        mp.findroot(lambda x: x * mp.exp(x) - 1, mp.mpf("0.5671")),
        r"The omega constant `$\Omega$`, the solution of `$\Omega e^{\Omega} = 1$`, i.e. `$W(1)$`",
    ),
    (
        "PSI",
        reciprocal_fibonacci(),
        r"The reciprocal Fibonacci constant `$\psi = \sum_{k=1}^{\infty} 1/F_k$`",
    ),
    ("LAPLACE_LIMIT", laplace_limit(), r"The Laplace limit, the root of `$x e^{\sqrt{1+x^2}} / (1 + \sqrt{1+x^2}) = 1$`"),
    (
        "ERDOS_BORWEIN",
        erdos_borwein(),
        r"The Erdos-Borwein constant `$E = \sum_{k=1}^{\infty} 1/(2^k - 1)$`",
    ),
    (
        "NIVEN",
        niven(),
        r"Niven's constant `$1 + \sum_{k=2}^{\infty} (1 - 1/\zeta(k))$`, the average maximum prime exponent",
    ),
    (
        "SOLDNER",
        mp.findroot(mp.li, mp.mpf("1.4513")),
        r"The Ramanujan-Soldner constant `$\mu$`, the positive root of the logarithmic integral `$\mathrm{li}(x)$`",
    ),
    (
        "FRANSEN_ROBINSON",
        fransen_robinson(),
        r"The Fransen-Robinson constant `$\int_0^{\infty} dx/\Gamma(x)$`",
    ),
    (
        "GOLOMB_DICKMAN",
        golomb_dickman(),
        r"The Golomb-Dickman constant `$\lambda = \int_0^1 e^{\mathrm{li}(t)}\,dt$`",
    ),
    (
        "TWIN_PRIME",
        +mp.twinprime,
        r"The twin prime constant `$C_2 = \prod_{p \ge 3} (1 - 1/(p-1)^2)$`",
    ),
    (
        "MERTENS",
        +mp.mertens,
        r"The Meissel-Mertens constant `$M = \gamma + \sum_p (\ln(1 - 1/p) + 1/p)$`",
    ),
    (
        "ARTIN",
        artin(),
        r"""Artin's constant `$A = \prod_p (1 - 1/(p(p-1)))$`

The conjectured density of primes admitting a given non-square integer > 1 as a
primitive root.""",
    ),
]

NAMES = [c[0] for c in CONSTS]
# Names whose value is the same in every format, i.e. everything but the epsilons.
MATH_NAMES = [c[0] for c in CONSTS if c[1] is not None]

# thermite-compensated's reciprocal-log table: 1/ln(b) for b = 3..32, used to
# turn a natural log into log-base-b with one multiply.
LOG_TABLE_BASES = range(3, 33)

# Cody-Waite pieces for exp's argument reduction. HAND-TUNED, not "ln 2 to more
# digits" (see the comment emitted beside them). Deliberately narrow so that
# `k * piece` is exact for the whole exponent range, which is what buys the
# reduction its accuracy. Do not "improve" these by widening them.
LN_2_EXTENDED = {
    "f32": ["0x1.62e4000000000p-1", "0x1.7f7e000000000p-20", "-0x1.c610ca0000000p-37"],
    "f64": ["0x1.62e42fefa3800p-1", "0x1.ef35793c76800p-45", "-0x1.9ff0342542fc3p-90"],
}

LN_2_EXTENDED_DOC = {
    "f32": """// LN_2_EXTENDED: Cody-Waite pieces, NOT simply ln(2) to ever more digits.
//
// `exp_internal` reduces with `r -= k * LN_2_EXTENDED[i]`, where both `k` and the piece
// are plain (uncompensated) vectors, so each product is a single rounded multiply. The
// pieces therefore have to be narrow enough that those products are exact, which is
// what buys the reduction its accuracy. The compensated subtraction around them is
// already exact on its own.
//
// f32 carries 24 mantissa bits and `exp` overflows near 88.7, so |k| <= 128 needs 8 of
// them: the leading pieces get 24 - 8 = 16 bits each and the tail takes the remainder.
// Verified exact for |k| <= 160. Worst-case reduction error 5.2e-17, against the ~3.6e-15
// that double-single needs.
//
// Widening these to "more accurate" full-precision values silently makes `exp` WORSE.
// At ~21 bits per piece the product is inexact past about k = 16.""",
    "f64": """// LN_2_EXTENDED: Cody-Waite pieces. See the f32 table above for why these are narrow.
//
// f64 carries 53 mantissa bits and `exp` overflows near 709.8, so |k| <= 1024 needs 11
// of them: 53 - 11 = 42 bits per leading piece, tail takes the remainder. Verified exact
// for |k| <= 1100. Worst-case reduction error 2.9e-42, against the ~1.2e-32 that
// double-double needs.
//
// Full-precision f64 values here make every `k * piece` a rounded multiply and cap `exp`
// at about 50 bits (1.1e-15 relative at x = 29, growing with |k|). `ln` inherits that
// ceiling through its Halley step, and everything built on the pair (`powf`, the gamma
// family) inherits it in turn.""",
}


# --- exact conversions -------------------------------------------------------


def exact(x):
    """mpmath value -> exact Fraction (an mpf is a dyadic rational, so lossless)."""
    x = mp.mpf(x)
    if x == 0:
        return Fraction(0)
    from mpmath.libmp import to_rational

    return Fraction(*to_rational(x._mpf_))


def f64_bits(v):
    return struct.unpack("<Q", struct.pack("<d", v))[0]


def f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def f32_from_bits(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


def f32_next(v, up):
    b = f32_bits(v)
    if v == 0.0:
        return f32_from_bits(1 if up else 0x80000001)
    neg = b >> 31
    b += 1 if ((up and not neg) or (not up and neg)) else -1
    return f32_from_bits(b)


def f64_next(v, up):
    import math

    return math.nextafter(v, math.inf if up else -math.inf)


def nearest_f64(true):
    # float(Fraction) is correctly rounded (half-even) in CPython.
    return float(true)


def nearest_f32(true):
    # A cast through f64 can double-round by at most one ulp, so fix by exact search.
    c = struct.unpack("<f", struct.pack("<f", float(true)))[0]
    cands = [f32_next(c, False), c, f32_next(c, True)]
    return min(cands, key=lambda v: (abs(Fraction(v) - true), f32_bits(v) & 1))


FMT = {
    # name: (mantissa bits, nearest, next, bits, bit width)
    "f32": (24, nearest_f32, f32_next, f32_bits, 8),
    "f64": (53, nearest_f64, f64_next, f64_bits, 16),
}


def value(name, val, mantissa_bits):
    """The true value of `name` in a format with `mantissa_bits` of precision."""
    return EPS[name](mantissa_bits) if val is None else val


def split(true, nearest):
    """Double-double split: (hi, lo) with hi = RN(c), lo = RN(c - hi)."""
    hi = nearest(true)
    lo = nearest(true - Fraction(hi))
    return hi, lo


def bracket(true, fmt):
    """The two consecutive floats enclosing `true`; equal when representable."""
    _, nearest, step, _, _ = FMT[fmt]
    c = nearest(true)
    fc = Fraction(c)
    if fc == true:
        return c, c
    lo, hi = (c, step(c, True)) if fc < true else (step(c, False), c)
    assert Fraction(lo) < true < Fraction(hi), "bracket does not enclose"
    assert step(lo, True) == hi, "bracket is not one ulp wide"
    return lo, hi


def bits_lit(v, fmt):
    _, _, _, bits, width = FMT[fmt]
    return f"{fmt}::from_bits(0x{bits(v):0{width}X})"


def decimal(v):
    """40 significant digits: rounds correctly into both f32 and f64."""
    return mp.nstr(v, 40, strip_zeros=False)


# --- shared emission helpers -------------------------------------------------

GENERATED = "// @generated by gen_consts.py at the workspace root. DO NOT EDIT BY HAND."


def doc_lines(doc, indent):
    return "\n".join(f"{indent}///{' ' + line if line else ''}" for line in doc.split("\n"))


# --- 1. thermite: crates/thermite/src/math/consts/mod.rs ---------------------


def gen_thermite():
    w = []
    p = w.append
    p(GENERATED)
    p("//")
    p("// Add or change a constant in `CONSTS` in that script and re-run it. The reusable")
    p("// `impl_consts!` machinery lives in `macros.rs` beside this file and is hand-written.")
    p("//")
    p("// The literals ARE the constants, at full precision, so `approx_constant` (which wants")
    p("// `core::f64::consts::PI` written instead) and `excessive_precision` are both noise here.")
    p("#![allow(clippy::approx_constant, clippy::excessive_precision)]")
    p("")
    p("#[macro_use]")
    p("mod macros;")
    p("")
    p("use crate::vector::{SplatConst, SplatVector, VectorValue};")
    p("use crate::{Vector, register::FloatRegister};")
    p("")
    p("/// Extensive set of constant special values used in float operations.")
    p("pub trait FloatConsts {")
    for i, (name, _, doc) in enumerate(CONSTS):
        if i:
            p("")
        p(doc_lines(doc, "    "))
        p(f"    const {name}: Self;")
    p("}")
    p("")
    p("/// Expands `$mac!` with the name of every [`FloatConsts`] constant, in declaration")
    p("/// order, so no other crate ever has to keep a list of them in sync.")
    p("///")
    p("/// Tokens after the macro name are passed through ahead of the names, which is how a")
    p("/// callback takes arguments of its own:")
    p("///")
    p("/// ```ignore")
    p("/// thermite::for_each_float_const!(check_consts, V, E;);  // check_consts!(V, E; PI, TAU, ...)")
    p("/// ```")
    p("#[macro_export]")
    p("#[doc(hidden)]")
    p("macro_rules! for_each_float_const {")
    p("    ($mac:ident $(, $($pre:tt)*)?) => {")
    p("        $mac!($($($pre)*)? " + ", ".join(NAMES) + ");")
    p("    };")
    p("}")
    p("")
    p("/// Like [`for_each_float_const!`], minus the three constants whose value depends on")
    p("/// the representation's precision (`EPSILON`, `SQRT_EPSILON`, `FOURTH_ROOT_EPSILON`).")
    p("///")
    p("/// Use this for anything that compares one representation's constants against")
    p("/// another's: a `Compensated<f64>` carries its own, much smaller, epsilon.")
    p("#[macro_export]")
    p("#[doc(hidden)]")
    p("macro_rules! for_each_math_const {")
    p("    ($mac:ident $(, $($pre:tt)*)?) => {")
    p("        $mac!($($($pre)*)? " + ", ".join(MATH_NAMES) + ");")
    p("    };")
    p("}")
    p("")
    p("")
    p("crate::for_each_float_const!(impl_consts);")
    for fmt in ("f32", "f64"):
        mant = FMT[fmt][0]
        p("")
        p("impl_consts! {@")
        p(f"    {fmt} {{")
        for name, val, _ in CONSTS:
            if name == "NEG_ZERO":
                p(f"        NEG_ZERO = -0.0{fmt},")
            else:
                p(f"        {name} = {decimal(value(name, val, mant))},")
        p("    }")
        p("}")
    p("")
    return "\n".join(w)


# --- 2. thermite-compensated: crates/thermite-compensated/src/consts/mod.rs --


def gen_compensated():
    w = []
    p = w.append
    p(GENERATED)
    p("//")
    p("// Double-double `(value, error)` splits of every `FloatConsts` constant, plus the")
    p("// reciprocal-log table `log_base` reduces through. The carrier/macro machinery is")
    p("// hand-written in `macros.rs` beside this file.")
    p("#![allow(clippy::approx_constant)]")
    p("")
    p("#[macro_use]")
    p("mod macros;")
    p("#[cfg(feature = \"special\")]")
    p("mod bernoulli;")
    p("")
    p("use core::marker::PhantomData;")
    p("")
    p("use thermite::{")
    p("    Vector,")
    p("    math::FloatConsts,")
    p("    vector::{SplatConst, const_splat},")
    p("};")
    p("")
    p("use super::{Compensated, ScalarValue};")
    p("")
    p("use self::macros::{Ln2Extended, LogTableError, LogTableValue};")
    p("")
    p(f"pub const LOG_TABLE_SIZE: usize = {len(LOG_TABLE_BASES)};")
    p("")
    p("/// Helper trait to store precomputed compensated logarithm table for small integer bases.")
    p("pub trait CompensatedLogTable<T = Self>: FloatConsts {")
    p(f"    /// The log table entries for bases {LOG_TABLE_BASES[0]}..={LOG_TABLE_BASES[-1]} as (high, low) pairs.")
    p("    const LOG_TABLE: [Compensated<T>; LOG_TABLE_SIZE];")
    p("")
    p("    /// Third part of extended precision ln(2), where the first two parts are provided by `LN_2`.")
    p("    const LN_2_EXTENDED: [T; 3];")
    p("}")
    p("")
    p("impl<R: thermite::register::FloatRegister> CompensatedLogTable<Self> for Vector<R>")
    p("where")
    p("    R::Element: CompensatedLogTable<R::Element>,")
    p("{")
    p("    #[rustfmt::skip]")
    p("    const LOG_TABLE: [Compensated<Self>; LOG_TABLE_SIZE] = log_table!(")
    for row in range(0, len(LOG_TABLE_BASES), 10):
        idx = range(row, min(row + 10, len(LOG_TABLE_BASES)))
        p("        " + " ".join(f"{i:2}," for i in idx))
    p("    );")
    p("")
    p("    const LN_2_EXTENDED: [Self; 3] = [")
    for i in range(3):
        p(f"        const_splat::<Self, Ln2Extended<R::Element, {i}>>(),")
    p("    ];")
    p("}")
    p("")
    p("thermite::for_each_float_const!(impl_consts);")
    for fmt in ("f32", "f64"):
        # A double-double built from `fmt` carries twice the mantissa, so its
        # epsilon is 2^-105 for f64 (not 2^-52) and 2^-47 for f32.
        mant = 2 * FMT[fmt][0]
        nearest = FMT[fmt][1]
        p("")
        p(f"impl_consts!({fmt} {{")
        for name, val, _ in CONSTS:
            hi, lo = split(exact(value(name, val, mant)), nearest)
            if name == "NEG_ZERO":
                hi = -0.0
            p(f'    {name} = ("{hi.hex()}", "{lo.hex()}"),')
        p("});")
    for fmt in ("f32", "f64"):
        nearest = FMT[fmt][1]
        p("")
        p(f"impl_consts!(LOG {fmt} [")
        for b in LOG_TABLE_BASES:
            hi, lo = split(exact(1 / mp.log(b)), nearest)
            p(f'    ("{hi.hex()}", "{lo.hex()}"),  // 1/ln({b})')
        p("],")
        p(LN_2_EXTENDED_DOC[fmt])
        p("[" + ", ".join(f'"{v}"' for v in LN_2_EXTENDED[fmt]) + "]);")
    p("")
    return "\n".join(w)


# --- 3/4. thermite-interval --------------------------------------------------


def gen_interval_consts():
    w = []
    p = w.append
    p(GENERATED)
    p("//! Enclosure constants: exact `(RD(c), RU(c))` pairs for every [`FloatConsts`]")
    p("//! name, meaning the two consecutive floats bracketing the true mathematical")
    p("//! value, so that pi _as an interval_ actually contains pi, one ulp wide, and an")
    p("//! exactly representable constant (`EPSILON`, `FRAC_1_4`, ...) is degenerate.")
    p("//!")
    p("//! The pairs live in `consts_table.rs`. Both that file and this one come from")
    p("//! `gen_consts.py` at the workspace root, which computes each constant from its")
    p("//! mathematical definition with mpmath at 80 digits and brackets it by exact")
    p("//! rational comparison, with no float rounding and no \"step one ulp and hope\"")
    p("//! anywhere. The `bounded_consts!` machinery is hand-written in `macros.rs`.")
    p("//!")
    p("//! Structure mirrors thermite-compensated's `SplitFloatConsts`: element impls")
    p("//! (`f32`/`f64`) from the table, lifted to `Vector<R>` through one")
    p("//! splat-carrier pair per constant name.")
    p("")
    p("#[macro_use]")
    p("mod macros;")
    p("")
    p("use core::marker::PhantomData;")
    p("")
    p("use thermite::prelude::*;")
    p("use thermite::register::FloatRegister;")
    p("use thermite::vector::{SplatConst, const_splat};")
    p("")
    p("thermite::for_each_float_const!(bounded_consts);")
    p("")
    return "\n".join(w)


def gen_interval_table():
    w = []
    p = w.append
    p(GENERATED)
    p("//")
    p("// Exact enclosure pairs (RD(c), RU(c)) for every `FloatConsts` constant,")
    p("// decided by exact rational comparison against 80-digit mpmath values.")
    p("// Degenerate pairs mark exactly representable constants.")
    p("")
    for fmt in ("f32", "f64"):
        mant = FMT[fmt][0]
        p("#[rustfmt::skip]")
        p(f"pub(crate) mod {fmt}_table {{")
        for name, val, _ in CONSTS:
            if name == "NEG_ZERO":
                lit = bits_lit(-0.0, fmt)
                p(f"    pub const NEG_ZERO: ({fmt}, {fmt}) = ({lit}, {lit}); // -0.0 (exact)")
                continue
            true = exact(value(name, val, mant))
            lo, hi = bracket(true, fmt)
            tag = "exact" if lo == hi else "RD/RU"
            approx = mp.nstr(mp.mpf(true.numerator) / true.denominator, 25)
            p(f"    pub const {name}: ({fmt}, {fmt}) = ({bits_lit(lo, fmt)}, {bits_lit(hi, fmt)}); // {approx} ({tag})")
        p("}")
        p("")
    p("/// Correctly rounded (nearest, ties-to-even) point values of every constant,")
    p("/// i.e. what a `FloatConsts` value should be bit-for-bit. Audit reference only.")
    p("#[doc(hidden)]")
    p("pub mod nearest {")
    for fmt in ("f32", "f64"):
        mant, nearest, _, _, _ = FMT[fmt]
        p("    #[rustfmt::skip]")
        p(f"    pub mod {fmt} {{")
        for name, val, _ in CONSTS:
            v = -0.0 if name == "NEG_ZERO" else nearest(exact(value(name, val, mant)))
            p(f"        pub const {name}: {fmt} = {bits_lit(v, fmt)};")
        p("    }")
    p("}")
    p("")
    return "\n".join(w)


# --- driver ------------------------------------------------------------------

# --- 5/6. Bernoulli numbers --------------------------------------------------
#
# Unlike everything above, these are not mpmath values: Bernoulli numbers are exact
# rationals, so they are built with `Fraction` and rounded once, the same way the rest of
# the file converts.
#
# Generation lives here rather than in Rust because the recurrence needs bignums even
# where the results do not. Boost's tangent-number recurrence overflows any fixed-width
# integer far earlier than the outputs do: T_20 already needs 39 digits, while reduced
# B_2n numerators stay inside i128 through B_58. Boost therefore runs that recurrence in
# floating point with a scale factor to keep it from overflowing; Python just uses
# bignums and emits the values.


def tangent_numbers(m):
    """`T_1 .. T_m`, exact integers, by the standard triangle recurrence."""
    t = [0] * (m + 2)
    t[1] = 1
    out = [0] * (m + 1)
    out[1] = 1
    for i in range(2, m + 1):
        t[1] *= i - 1
        for j in range(2, i + 1):
            t[j] = t[j] * (i - j) + t[j - 1] * (i - j + 2)
        out[i] = t[i]
    return out


def bernoulli_even(count):
    """`B_0, B_2, ..., B_{2(count-1)}` as exact Fractions, via the tangent numbers:

        B_2n = (-1)^(n+1) * 2n * T_n / (2^2n * (2^2n - 1))

    Odd-index Bernoulli numbers past B_1 are all zero and B_1 is convention-dependent,
    so neither is generated - see the emitted module docs."""
    t = tangent_numbers(count)
    out = [Fraction(1)]
    for i in range(1, count):
        p2 = 1 << (2 * i)
        b = Fraction(2 * i * t[i], p2 * (p2 - 1))
        out.append(b if i % 2 else -b)
    return out


def finite_in(fmt, frac):
    """`frac` rounded to `fmt`, or None if it rounds to infinity."""
    try:
        v = FMT[fmt][1](frac)
    except (OverflowError, ValueError):
        return None
    return v if abs(v) != float("inf") else None


_BERNOULLI = None


def bernoulli_table(fmt):
    """`B_2, B_4, ...`, every `B_2n` with `n >= 1` that is finite in `fmt`, as exact
    Fractions. `|B_2n|` grows factorially, so this terminates: the last finite one is
    B_64 for f32 and B_258 for f64.

    B_0 = 1 is skipped along with B_1. B_0 and B_1 co-occur in practice. Any formula that
    indexes the sequence from 0 (Faulhaber, the binomial recurrence, the Bernoulli
    polynomials) reaches B_1 at k = 1, and a formula that skips B_1 (Euler-Maclaurin, the
    asymptotic series, the zeta identity) starts at B_2 and never wanted B_0 either. So
    anything needing B_0 already special-cases the head of the sequence for B_1's sake."""
    global _BERNOULLI
    if _BERNOULLI is None:
        # 140 entries reaches B_278, comfortably past f64's last finite B_258.
        _BERNOULLI = bernoulli_even(140)
    out = []
    for fr in _BERNOULLI[1:]:
        if finite_in(fmt, fr) is None:
            break
        out.append(fr)
    return out


def rust_float_literal(v, fmt):
    """Shortest decimal literal that parses back to exactly `v` in `fmt`.

    Rust parses float literals with correct rounding, so a shortest round-tripping
    decimal is exact. Hex is reserved for the double-double splits, where it is what
    keeps the two limbs unambiguous."""
    if fmt == "f64":
        s = repr(v)
    else:
        s = next(
            f"{v:.{p}g}"
            for p in range(1, 18)
            if struct.unpack("<f", struct.pack("<f", float(f"{v:.{p}g}")))[0] == v
        )
    if "." not in s and "e" not in s and "E" not in s:
        s += ".0"
    return s


BERNOULLI_MODULE_DOC = r"""//! Bernoulli numbers `$B_{2n}$`, as one static table per float format.
//!
//! **The tables start at `$B_2$`**, so entry `i` is `$B_{2i+2}$`. They are even-index
//! only: every odd-index Bernoulli number past `$B_1$` is zero, so storing them would
//! double the table for nothing.
//!
//! ```
//! use thermite_special::tables::bernoulli::BernoulliNumbers;
//!
//! assert_eq!(f64::B2N[0], 1.0 / 6.0);           // B_2
//! assert_eq!(f64::B2N[2], 1.0 / 42.0);          // B_6
//! ```
//!
//! # `$B_0$` and `$B_1$` are deliberately absent
//!
//! `$B_1$` is the single value the two competing conventions disagree on: `$-1/2$` if
//! you take the generating function to be `$x/(e^x - 1)$`, `$+1/2$` if you take it to be
//! `$x/(1 - e^{-x})$` (Knuth has argued in print for the latter). Every _other_ Bernoulli
//! number is identical under both, so omitting `$B_1$` makes this table convention-free
//! rather than convention-bearing.
//!
//! `$B_0 = 1$` is not contested, but it goes with it, because the two co-occur. A formula
//! that indexes the sequence from zero (Faulhaber's sum of powers, the binomial
//! recurrence, the Bernoulli polynomials) reaches `$B_1$` at `$k = 1$` and therefore
//! already special-cases the head of the sequence. A formula that skips `$B_1$`
//! (Euler-Maclaurin, the `$\ln\Gamma$` / `$\psi$` asymptotic series, the `$\zeta(2n)$`
//! identity) starts at `$B_2$` and never wanted `$B_0$` either. Nothing sits in the gap,
//! so the table holds exactly the values that need a table.
//!
//! Callers who want the head should let their heart guide them on `$B_1 = \pm\tfrac{1}{2}$`
//! and write `$B_0 = 1$` beside it.
//!
//! # The table ends where the format does
//!
//! `$|B_{2n}|$` grows factorially,
//!
//! ```math
//! |B_{2n}| = \frac{2\,(2n)!}{(2\pi)^{2n}}\,\zeta(2n)
//!     \sim 4\sqrt{\pi n}\left(\frac{n}{\pi e}\right)^{2n}
//! ```
//!
//! with consecutive terms growing by roughly `$n^2/\pi^2$`, so each format has a _last_
//! representable Bernoulli number and nothing beyond it to return.
//!
//! The tables stop exactly there, which means **`B2N.len()` is the overflow boundary**:
//! `B2N.get(i)` is `None` precisely where the value would be infinite. There is no
//! overflow policy, no error type and no limit constant, because the slice length
//! already carries that information.
//!
//! There is no underflow at the other end. `$|B_{2n}|$` bottoms out at
//! `$B_6 = 1/42$` and grows monotonically after it, so no entry is denormal.
//!
//! | format | entries | first | last finite | first to overflow |
//! |---|---|---|---|---|
//! | `f32` | 32 | `$B_2$` | `$B_{64}$` | `$B_{66}$` |
//! | `f64` | 129 | `$B_2$` | `$B_{258}$` | `$B_{260}$` |
//!
//! `thermite-compensated` implements the same trait for `Compensated<f32>` and
//! `Compensated<f64>` under its `special` feature. Those tables are the SAME lengths: a
//! double-double carries twice the mantissa but the same exponent range, so widening the
//! type buys precision, not reach."""


def gen_special_bernoulli():
    w = []
    p = w.append
    p(GENERATED)
    p(BERNOULLI_MODULE_DOC)
    p("")
    p("use thermite::element::FloatElement;")
    p("")
    p("/// Static Bernoulli number tables for a float format.")
    p("///")
    p("/// Implemented for `f32` and `f64` here, and for `Compensated<f32>` /")
    p("/// `Compensated<f64>` by `thermite-compensated` under its `special` feature.")
    p("pub trait BernoulliNumbers: FloatElement {")
    p(r"    /// `$B_2, B_4, B_6, \ldots$`, every even-index Bernoulli number finite in `Self`,")
    p(r"    /// starting at `$B_2$`, so that entry `i` is `$B_{2i+2}$`.")
    p("    ///")
    p(r"    /// See the [module docs](self) for why the table ends where it does, and why")
    p(r"    /// `$B_0$` and `$B_1$` are not in it.")
    p("    const B2N: &'static [Self];")
    p("}")
    p("")
    p(r"/// `$B_{2n}$`, or `None` when it is not tabulated: either `$n = 0$`, or `$B_{2n}$`")
    p("/// overflows `E`.")
    p("///")
    p(r"/// The argument is `n` as in `$B_{2n}$`, following the mathematics rather than the")
    p(r"/// slice index, so `bernoulli_b2n::<f64>(1)` is `$B_2$`, the first entry. Reach for")
    p(r"/// [`BernoulliNumbers::B2N`] directly when iterating, where entry `i` is `$B_{2i+2}$`.")
    p("#[inline]")
    p("#[must_use]")
    p("pub fn bernoulli_b2n<E: BernoulliNumbers>(n: usize) -> Option<E> {")
    p("    E::B2N.get(n.checked_sub(1)?).copied()")
    p("}")
    for fmt in ("f32", "f64"):
        table = bernoulli_table(fmt)
        nearest = FMT[fmt][1]
        p("")
        p(f"impl BernoulliNumbers for {fmt} {{")
        p("    #[rustfmt::skip]")
        p(f"    const B2N: &'static [{fmt}] = &[")
        for k, fr in enumerate(table):
            lit = rust_float_literal(nearest(fr), fmt)
            p(f"        {lit},  // B_{2 * (k + 1)}")
        p("    ];")
        p("}")
    p("")
    return "\n".join(w)


def factorial_table(fmt):
    """`0!, 1!, 2!, ...`, every factorial finite in `fmt`, as exact Fractions. `n!`
    grows factorially by definition, so this terminates: the last finite one is 34!
    for f32 and 170! for f64."""
    out = []
    f = Fraction(1)
    n = 0
    while finite_in(fmt, f) is not None:
        out.append(f)
        n += 1
        f *= n
    return out


FACTORIAL_MODULE_DOC = r"""//! Factorials `$n!$`, as one static table per float format.
//!
//! Entry `i` is `$i!$` (so the table starts `1, 1, 2, 6, ...`), correctly rounded once
//! from the exact integer. Tabulating rather than accumulating is an accuracy decision
//! as much as a speed one: a running product rounds at every step, while the table entry
//! is the nearest float to the true value.
//!
//! # The table ends where the format does
//!
//! Each format has a last representable factorial and nothing beyond it to return, so
//! **`FACTORIALS.len()` is the overflow boundary**: `FACTORIALS.get(n)` is `None`
//! precisely where `$n!$` would be infinite. There is no overflow policy, no error type
//! and no limit constant, because the slice length already carries that information.
//!
//! | format | entries | last finite | first to overflow |
//! |---|---|---|---|
//! | `f32` | 35 | `$34!$` | `$35!$` |
//! | `f64` | 171 | `$170!$` | `$171!$` |"""


def gen_special_factorial():
    w = []
    p = w.append
    p(GENERATED)
    p(FACTORIAL_MODULE_DOC)
    p("")
    p("use thermite::element::FloatElement;")
    p("")
    p("/// Static factorial tables for a float format.")
    p("pub trait Factorials: FloatElement {")
    p(r"    /// `$0!, 1!, 2!, \ldots$`, every factorial finite in `Self`, so entry `i` is `$i!$`.")
    p("    ///")
    p(r"    /// See the [module docs](self) for why the table ends where it does.")
    p("    const FACTORIALS: &'static [Self];")
    p("}")
    p("")
    p(r"/// `$n!$`, or `None` when it overflows `E`.")
    p("#[inline]")
    p("#[must_use]")
    p("pub fn factorial<E: Factorials>(n: usize) -> Option<E> {")
    p("    E::FACTORIALS.get(n).copied()")
    p("}")
    for fmt in ("f32", "f64"):
        table = factorial_table(fmt)
        nearest = FMT[fmt][1]
        p("")
        p(f"impl Factorials for {fmt} {{")
        p("    #[rustfmt::skip]")
        p(f"    const FACTORIALS: &'static [{fmt}] = &[")
        for k, fr in enumerate(table):
            lit = rust_float_literal(nearest(fr), fmt)
            p(f"        {lit},  // {k}!")
        p("    ];")
        p("}")
    p("")
    return "\n".join(w)


def gen_compensated_bernoulli():
    w = []
    p = w.append
    p(GENERATED)
    p("//! Bernoulli numbers at double-double precision.")
    p("//!")
    p("//! The `thermite-special` [`BernoulliNumbers`] tables, split into `(value, error)`")
    p(r"//! limbs. Entry `i` is `$B_{2i+2}$`, same as there.")
    p("//!")
    p(r"//! The lengths match the underlying format's exactly: a double-double carries twice")
    p(r"//! the mantissa but the SAME exponent range, so `Compensated<f64>` runs out at")
    p(r"//! `$B_{260}$` just as `f64` does. Widening buys precision, not reach.")
    p("")
    p("use thermite_special::tables::bernoulli::BernoulliNumbers;")
    p("")
    p("use crate::Compensated;")
    for fmt in ("f32", "f64"):
        table = bernoulli_table(fmt)
        nearest = FMT[fmt][1]
        p("")
        p(f"impl BernoulliNumbers for Compensated<{fmt}> {{")
        p("    #[rustfmt::skip]")
        p(f"    const B2N: &'static [Compensated<{fmt}>] = &[")
        for k, fr in enumerate(table):
            hi, lo = split(fr, nearest)
            p(
                f"        Compensated {{ value: hexf::hex{fmt}!(\"{hi.hex()}\"),"
                f" error: hexf::hex{fmt}!(\"{lo.hex()}\") }},  // B_{2 * (k + 1)}"
            )
        p("    ];")
        p("}")
    p("")
    return "\n".join(w)


POLYLOG_MODULE_DOC = r"""//! Constants behind the polylogarithm kernel: `$\zeta$` at the positive integers and the
//! Stieltjes constants, one table per float format.
//!
//! # `ZETA_INT` ends where the format does
//!
//! Entry `i` is `$\zeta(i + 2)$`. The table stops at the last `$n$` for which
//! `$\zeta(n)$` is distinguishable from 1 in the format: `$\zeta(n) - 1 \approx 2^{-n}$`, so
//! past the table the nearest float is exactly `1.0`, and a kernel reads that value for
//! every order beyond it without a branch worth having. `ZETA_INT.len() + 1` is the last
//! tabulated `$n$`.
//!
//! | format | entries | last tabulated |
//! |---|---|---|
//! | `f32` | 23 | `$\zeta(24)$` |
//! | `f64` | 52 | `$\zeta(53)$` |
//!
//! # `STIELTJES`
//!
//! `$\gamma_0, \gamma_1, \ldots$` in the Laurent expansion about the pole,
//!
//! ```math
//! \zeta(1 + \varepsilon) - \frac{1}{\varepsilon} = \sum_{k \ge 0} \frac{(-1)^k}{k!}\gamma_k\,\varepsilon^k ,
//! ```
//!
//! which is entire, so the sum converges for every `$\varepsilon$`. The kernel uses it for
//! `$|\varepsilon| \le 1/10$`, where 24 terms hold binary64 accuracy."""


def gen_special_polylog():
    w = []
    p = w.append
    p(GENERATED)
    p(POLYLOG_MODULE_DOC)
    p("")
    p("use thermite::element::FloatElement;")
    p("")
    p("/// Static polylogarithm constant tables for a float format.")
    p("pub trait PolylogConsts: FloatElement {")
    p(r"    /// `$\zeta(n)$` for `$n = 2, 3, \ldots$`. Entry `i` is `$\zeta(i + 2)$`. See the [module docs](self).")
    p("    const ZETA_INT: &'static [Self];")
    p(r"    /// The Stieltjes constants `$\gamma_0, \gamma_1, \ldots$`. Entry `i` is `$\gamma_i$`.")
    p("    const STIELTJES: &'static [Self];")
    p("}")
    for fmt in ("f32", "f64"):
        nearest = FMT[fmt][1]
        p("")
        p(f"impl PolylogConsts for {fmt} {{")
        p("    #[rustfmt::skip]")
        p(f"    const ZETA_INT: &'static [{fmt}] = &[")
        with mp.workdps(60):
            n = 2
            while True:
                v = nearest(exact(mp.zeta(n)))
                if v == 1.0:
                    break
                p(f"        {rust_float_literal(v, fmt)},  // zeta({n})")
                n += 1
        p("    ];")
        p("")
        p("    #[rustfmt::skip]")
        p(f"    const STIELTJES: &'static [{fmt}] = &[")
        with mp.workdps(60):
            for k in range(24):
                v = nearest(exact(mp.stieltjes(k)))
                p(f"        {rust_float_literal(v, fmt)},  // gamma_{k}")
        p("    ];")
        p("}")
    p("")
    return "\n".join(w)


OUTPUTS = [
    ("crates/thermite/src/math/consts/mod.rs", gen_thermite),
    ("crates/thermite-compensated/src/consts/mod.rs", gen_compensated),
    ("crates/thermite-interval/src/consts/mod.rs", gen_interval_consts),
    ("crates/thermite-interval/src/consts_table.rs", gen_interval_table),
    ("crates/thermite-special/src/tables/bernoulli.rs", gen_special_bernoulli),
    ("crates/thermite-special/src/tables/factorial.rs", gen_special_factorial),
    ("crates/thermite-special/src/tables/polylog.rs", gen_special_polylog),
    ("crates/thermite-compensated/src/consts/bernoulli.rs", gen_compensated_bernoulli),
]


def main():
    check = "--check" in sys.argv[1:]
    stale = []
    for rel, gen in OUTPUTS:
        path = os.path.join(ROOT, rel)
        text = gen()
        old = None
        if os.path.exists(path):
            with open(path, encoding="utf8", newline="") as f:
                old = f.read()
        if check:
            if old != text:
                stale.append(rel)
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf8", newline="\n") as f:
            f.write(text)
        print(f"{'unchanged' if old == text else 'wrote    '} {rel}")
    if check:
        if stale:
            print("stale (re-run gen_consts.py):\n  " + "\n  ".join(stale))
            return 1
        print(f"up to date ({len(CONSTS)} constants)")
    else:
        print(f"{len(CONSTS)} constants")
    return 0


if __name__ == "__main__":
    sys.exit(main())
