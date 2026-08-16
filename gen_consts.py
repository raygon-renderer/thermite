"""
Single source of truth for every mathematical constant in the workspace.

    python gen_consts.py            # regenerate all four files
    python gen_consts.py --check    # exit 1 if anything is stale (CI)

Generates, in full:

    crates/thermite/src/math/consts/mod.rs          FloatConsts, f32/f64/Vector impls
    crates/thermite-compensated/src/consts/mod.rs   double-double splits + log tables
    crates/thermite-interval/src/consts/mod.rs      BoundedFloatConsts wiring
    crates/thermite-interval/src/consts_table.rs    exact enclosure pairs

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


def reciprocal_fibonacci():
    """Sum of 1/F(k), k >= 1. Terms fall off like phi^-k, so 400 of them are
    far below the 80-digit working precision."""
    return mp.fsum(1 / mp.fib(k) for k in range(1, 401))


def laplace_limit():
    """Root of x*exp(sqrt(1+x^2)) / (1 + sqrt(1+x^2)) = 1."""
    f = lambda x: x * mp.exp(mp.sqrt(1 + x**2)) / (1 + mp.sqrt(1 + x**2)) - 1
    return mp.findroot(f, mp.mpf("0.6627434193"))


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
    ("FRAC_2_PI", 2 / pi, r"`$2/\pi$`"),
    ("FRAC_1_SQRT_PI", 1 / mp.sqrt(pi), r"`$1/\sqrt{\pi}$`"),
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
    ("FRAC_LN_PI_2", mp.log(pi) / 2, r"`$\frac{1}{2}\ln \pi$`"),
    ("LOG2_10", mp.log(10, 2), r"`$\log_2 10$`"),
    ("LOG2_E", mp.log(e, 2), r"`$\log_2 e$`"),
    ("LOG10_2", mp.log(2, 10), r"`$\log_{10} 2$`"),
    ("LOG10_E", mp.log(e, 10), r"`$\log_{10} e$`"),
    ("PI", pi, "Archimedes' constant (π)"),
    ("SQRT_2", mp.sqrt(2), r"`$\sqrt{2}$`"),
    ("SQRT_3", mp.sqrt(3), r"`$\sqrt{3}$`"),
    ("SQRT_E", mp.sqrt(e), r"`$\sqrt{e}$`"),
    ("EPSILON", None, "The machine epsilon"),
    ("SQRT_EPSILON", None, r"The square root of the machine epsilon (`$\sqrt{\varepsilon}$`)"),
    ("FOURTH_ROOT_EPSILON", None, r"The fourth root of the machine epsilon (`$\sqrt[4]{\varepsilon}$`)"),
    ("TAU", 2 * pi, "The full circle constant (τ)"),
    ("SQRT_FRAC_PI_2", mp.sqrt(pi / 2), r"`$\sqrt{\pi/2}$`"),
    ("SQRT_TAU", mp.sqrt(2 * pi), r"`$\sqrt{2\pi}$`"),
    ("PHI", phi, "The golden ratio (φ)"),
    ("FRAC_1_3", mp.mpf(1) / 3, r"`$1/3$`"),
    ("FRAC_2_3", mp.mpf(2) / 3, r"`$2/3$`"),
    ("FRAC_1_4", mp.mpf(1) / 4, r"`$1/4$`"),
    ("FRAC_1_6", mp.mpf(1) / 6, r"`$1/6$`"),
    ("FRAC_NEG_1_E", -1 / e, r"`$-1/e$`"),
    ("FRAC_1_2", mp.mpf(1) / 2, r"`$1/2$`"),
    ("FRAC_3_4", mp.mpf(3) / 4, r"`$3/4$`"),
    ("LN_LN_2", mp.log(mp.log(2)), r"`$\ln(\ln 2)$`, the median of the Gumbel distribution"),
    ("SQRT_LN_4", mp.sqrt(mp.log(4)), r"`$\sqrt{\ln 4}$`"),
    ("FRAC_2PI_3", 2 * pi / 3, r"`$2\pi/3$`"),
    ("FRAC_3PI_4", 3 * pi / 4, r"`$3\pi/4$`"),
    ("FRAC_4PI_3", 4 * pi / 3, r"`$4\pi/3$`, the volume of the unit sphere"),
    ("FRAC_1_TAU", 1 / (2 * pi), r"`$1/(2\pi)$`"),
    ("SQRT_PI", mp.sqrt(pi), r"`$\sqrt{\pi}$`"),
    ("PI_MINUS_3", pi - 3, r"`$\pi - 3$`"),
    ("FOUR_MINUS_PI", 4 - pi, r"`$4 - \pi$`"),
    ("PI_POW_E", pi**e, r"`$\pi^e$`"),
    ("CBRT_PI", mp.cbrt(pi), r"`$\sqrt[3]{\pi}$`"),
    ("FRAC_1_CBRT_PI", 1 / mp.cbrt(pi), r"`$1/\sqrt[3]{\pi}$`"),
    ("FRAC_1_SQRT_E", 1 / mp.sqrt(e), r"`$1/\sqrt{e} = e^{-1/2}$`"),
    ("E_POW_PI", e**pi, r"`$e^\pi$`, Gelfond's constant"),
    ("SIN_1", mp.sin(1), r"`$\sin 1$`"),
    ("COS_1", mp.cos(1), r"`$\cos 1$`"),
    ("SINH_1", mp.sinh(1), r"`$\sinh 1$`"),
    ("COSH_1", mp.cosh(1), r"`$\cosh 1$`"),
    ("LN_PHI", mp.log(phi), r"`$\ln \varphi$`"),
    ("FRAC_1_LN_PHI", 1 / mp.log(phi), r"`$1/\ln \varphi$`"),
    ("FRAC_1_EULER_GAMMA", 1 / euler, r"`$1/\gamma$`"),
    ("EULER_GAMMA_SQUARED", euler**2, r"`$\gamma^2$`"),
    ("ZETA_2", mp.zeta(2), r"`$\zeta(2) = \pi^2/6$`"),
    ("ZETA_3", mp.zeta(3), r"`$\zeta(3)$`, Apery's constant"),
    ("CATALAN", +mp.catalan, r"Catalan's constant `$K$`"),
    ("GLAISHER", +mp.glaisher, r"The Glaisher-Kinkelin constant `$A$`"),
    ("KHINCHIN", +mp.khinchin, r"Khinchin's constant `$K_0$`"),
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
        mp.findroot(lambda x: x**3 - x - 1, mp.mpf("1.3247")),
        r"The plastic ratio `$\rho$`, the real root of `$x^3 = x + 1$`",
    ),
    ("GAUSS", 1 / mp.agm(1, mp.sqrt(2)), r"Gauss's constant `$G = 1/\mathrm{agm}(1, \sqrt{2})$`"),
    ("DOTTIE", mp.findroot(lambda x: mp.cos(x) - x, mp.mpf("0.739")), r"The Dottie number, the unique real solution of `$\cos x = x$`"),
    (
        "PSI",
        reciprocal_fibonacci(),
        r"The reciprocal Fibonacci constant `$\psi = \sum_{k=1}^{\infty} 1/F_k$`",
    ),
    ("LAPLACE_LIMIT", laplace_limit(), r"The Laplace limit, the root of `$x e^{\sqrt{1+x^2}} / (1 + \sqrt{1+x^2}) = 1$`"),
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
// It was previously ~21 bits per piece, which is inexact past about k = 16.""",
    "f64": """// LN_2_EXTENDED: Cody-Waite pieces. See the f32 table above for why these are narrow.
//
// f64 carries 53 mantissa bits and `exp` overflows near 709.8, so |k| <= 1024 needs 11
// of them: 53 - 11 = 42 bits per leading piece, tail takes the remainder. Verified exact
// for |k| <= 1100. Worst-case reduction error 2.9e-42, against the ~1.2e-32 that
// double-double needs.
//
// These were previously full-precision f64 values, which made every `k * piece` a rounded
// multiply and capped `exp` at about 50 bits (1.1e-15 relative at x = 29, growing with
// |k|). `ln` inherited that ceiling through its Halley step, and everything built on the
// pair (`powf`, the gamma family) inherited it in turn.""",
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

OUTPUTS = [
    ("crates/thermite/src/math/consts/mod.rs", gen_thermite),
    ("crates/thermite-compensated/src/consts/mod.rs", gen_compensated),
    ("crates/thermite-interval/src/consts/mod.rs", gen_interval_consts),
    ("crates/thermite-interval/src/consts_table.rs", gen_interval_table),
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
