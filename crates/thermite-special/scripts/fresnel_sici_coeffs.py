"""Coefficients for `fresnel` (C, S) and `sici` (Si, Ci).

Generates `src/tables/fresnel.rs` and `src/tables/sici.rs`.

Two fits per function per precision:

  small |x| <= x0   a Chebyshev series in w = x^4 (Fresnel) or v = x^2 (sici),
                    summed by Clenshaw. Chebyshev rather than a monomial Horner
                    because the monomial condition number reaches 3482 at the
                    crossover the auxiliaries want, against 5.8 for Clenshaw.

  large |x| > x0    monomial Horner in u = 1/(pi x^2)^2 (Fresnel) or v = 1/x^2
                    (sici), for the auxiliaries f and g of

                        C = 1/2 + f sin t - g cos t,  S = 1/2 - f cos t - g sin t
                        Si = pi/2 - f cos x - g sin x,  Ci = f sin x - g cos x

                    with t = pi x^2 / 2. Horner is fine there: its condition
                    number is ~1.0 once x0 is far enough out.

Nothing is sized by a fit-error bound. Each candidate is evaluated the way the
Rust kernel will evaluate it (Clenshaw or Horner, in numpy float32/float64,
including the reconstruction and the two-word phase) and scored in ulps against
mpmath at 45 digits. That is what picks the degrees below.

    python scripts/fresnel_sici_coeffs.py [--sweep]
"""

import os
import sys

import numpy as np
from mpmath import mp, mpf, pi as MP_PI, sqrt, cos, sin, log, euler

mp.dps = 45
HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# oracles
# --------------------------------------------------------------------------- #

def fres_aux(x):
    """(f, g) with C = 1/2 + f sin t - g cos t, S = 1/2 - f cos t - g sin t."""
    C, S = mp.fresnelc(x), mp.fresnels(x)
    t = MP_PI * x * x / 2
    a, b = C - mpf(1) / 2, S - mpf(1) / 2
    return a * sin(t) - b * cos(t), -a * cos(t) - b * sin(t)


def sici_aux(x):
    """(f, g) with Si = pi/2 - f cos x - g sin x, Ci = f sin x - g cos x."""
    a, b = mp.si(x) - MP_PI / 2, mp.ci(x)
    return -a * cos(x) + b * sin(x), -a * sin(x) - b * cos(x)


def cin(x):
    """gamma + ln x - Ci(x), the entire part; Ci = (gamma + ln x) - Cin."""
    return euler + log(x) - mp.ci(x)


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #

def cheb_coeffs(f, lo, hi, n):
    """Chebyshev coefficients a_0..a_n of f on [lo, hi], by the discrete cosine transform."""
    N = n + 1
    nodes = [cos(MP_PI * (k + mpf(1) / 2) / N) for k in range(N)]
    vals = [f(lo + (hi - lo) * (t + 1) / 2) for t in nodes]
    out = []
    for j in range(N):
        s = sum(vals[k] * cos(MP_PI * j * (k + mpf(1) / 2) / N) for k in range(N))
        out.append((2 if j else 1) * s / N)
    return out


def mono_coeffs(f, lo, hi, n):
    """Near-minimax monomial coefficients c_0..c_n (ascending) of f on [lo, hi]."""
    co = mp.chebyfit(f, [lo, hi], n + 1, error=False)
    return list(co[::-1])


def clenshaw(a, t):
    """Sum a_k T_k(t) by the same backward recurrence the Rust kernel uses."""
    b1 = b2 = np.zeros_like(t)
    t2 = t + t
    for c in a[:0:-1]:
        b1, b2 = t2 * b1 - b2 + c, b1
    return t * b1 - b2 + a[0]


def horner(c, x):
    """Sum c_k x^k, c ascending."""
    acc = np.full_like(x, c[-1])
    for ck in c[-2::-1]:
        acc = acc * x + ck
    return acc


def as_dtype(coeffs, dtype):
    return [dtype(float(c)) for c in coeffs]


# --------------------------------------------------------------------------- #
# the evaluation models: these mirror the Rust kernels op for op
# --------------------------------------------------------------------------- #

def two_word_phase(x, dtype):
    """`x^2/2 mod 2`, to full precision, for `sincos_pi`.

    `x*x = p + e` exactly (`e` is always representable), and halving is exact, so
    the phase is `p/2 + e/2` exactly. Both halves must be reduced mod 2
    *separately before being added*: `|e/2|` reaches `ulp(x^2)/4`, which is 32 at
    x = 1e9, and adding that to a reduced `p/2` in [-1,1] rounds the latter's low
    bits straight off (measured: 3.6e-15 of phase, ~25 ulp of the result). Each
    reduction is exact by Sterbenz."""
    def rem2(v):
        return v - dtype(2.0) * np.rint(v * dtype(0.5))

    p = x * x
    if dtype is np.float64:
        e = _two_product_err(x, x)
    else:
        e = (x.astype(np.float64) * x.astype(np.float64) - p.astype(np.float64)).astype(np.float32)
    return rem2(rem2(p * dtype(0.5)) + rem2(e * dtype(0.5)))


def _two_product_err(a, b):
    """err(a*b) in float64 via Dekker's split (numpy exposes no fma)."""
    split = np.float64(134217729.0)  # 2^27 + 1
    ca, cb = split * a, split * b
    ah, bh = ca - (ca - a), cb - (cb - b)
    al, bl = a - ah, b - bh
    p = a * b
    return ((ah * bh - p) + ah * bl + al * bh) + al * bl


def eval_fresnel(x, cfg, dtype):
    """Model of the Rust fresnel kernel. x: positive float array."""
    x = x.astype(dtype)
    w0 = dtype(cfg["x0"]) ** 4
    small = x <= dtype(cfg["x0"])

    q = x * x
    w = q * q
    t_small = dtype(2.0) * w / w0 - dtype(1.0)
    A = clenshaw(as_dtype(cfg["cheb_c"], dtype), t_small)
    B = clenshaw(as_dtype(cfg["cheb_s"], dtype), t_small)
    C_small = x * A
    S_small = x * q * B

    pi_d = dtype(float(MP_PI))
    r = dtype(1.0) / (pi_d * q)
    u = r * r
    P = horner(as_dtype(cfg["aux_p"], dtype), u)
    Q = horner(as_dtype(cfg["aux_q"], dtype), u)
    f = P * r * x
    g = Q * u * x

    ph = two_word_phase(x, dtype)
    sin_t = np.sin(np.float64(np.pi) * ph.astype(np.float64)).astype(dtype)
    cos_t = np.cos(np.float64(np.pi) * ph.astype(np.float64)).astype(dtype)
    half = dtype(0.5)
    C_large = half + (f * sin_t - g * cos_t)
    S_large = half - (f * cos_t + g * sin_t)

    C = np.where(small, C_small, C_large)
    S = np.where(small, S_small, S_large)
    cut = x > dtype(cfg["cutoff"])
    return np.where(cut, half, C), np.where(cut, half, S)


def eval_sici(x, cfg, dtype):
    x = x.astype(dtype)
    v0 = dtype(cfg["x0"]) ** 2
    small = x <= dtype(cfg["x0"])

    v = x * x
    t_small = dtype(2.0) * v / v0 - dtype(1.0)
    A = clenshaw(as_dtype(cfg["cheb_si"], dtype), t_small)
    B = clenshaw(as_dtype(cfg["cheb_cin"], dtype), t_small)
    Si_small = x * A
    gamma = dtype(float(euler))
    Ci_small = (gamma + np.log(x.astype(np.float64)).astype(dtype)) - v * B

    rx = dtype(1.0) / x
    vv = rx * rx
    P = horner(as_dtype(cfg["aux_p"], dtype), vv)
    Q = horner(as_dtype(cfg["aux_q"], dtype), vv)
    f = P * rx
    g = Q * vv

    sx = np.sin(x.astype(np.float64)).astype(dtype)
    cx = np.cos(x.astype(np.float64)).astype(dtype)
    half_pi = dtype(float(MP_PI / 2))
    Si_large = half_pi - (f * cx + g * sx)
    Ci_large = f * sx - g * cx

    Si = np.where(small, Si_small, Si_large)
    Ci = np.where(small, Ci_small, Ci_large)
    Si = np.where(x > dtype(cfg["cutoff"]), half_pi, Si)
    return Si, Ci


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #

def ulp_err(got, want_mp, dtype):
    """Relative error in ulps of the result's own magnitude."""
    out = []
    for g, w in zip(got, want_mp):
        w = mpf(w)
        if w == 0:
            out.append(mpf(0) if g == 0 else mpf(1e30))
            continue
        eps = mpf(2) ** (-24 if dtype is np.float32 else -53)
        out.append(abs((mpf(float(g)) - w) / w) / (2 * eps))
    return out


def envelope_ulp_err(got, want_mp, env_mp, dtype):
    """Error relative to an envelope, for functions with zeros (Ci)."""
    eps = mpf(2) ** (-24 if dtype is np.float32 else -53)
    return [abs(mpf(float(g)) - mpf(w)) / mpf(e) / (2 * eps) for g, w, e in zip(got, want_mp, env_mp)]


def grid(lo, hi, n):
    return np.exp(np.linspace(np.log(lo), np.log(hi), n))


def _probe(x0, dtype, n):
    """Grid, ROUNDED TO THE TARGET TYPE FIRST so the oracle is evaluated at the same
    binary value the kernel sees. Grading against the unrounded grid point instead
    charges the kernel for `x`'s own rounding - `dC/dx = cos t` is O(1), so at
    x = 2744 that alone reads as 1675 ulp and hides everything real."""
    # Out to just under the cutoff, not to some comfortable 1e6: the phase and the
    # auxiliaries are only interesting where they are hard.
    top = 1e7 if dtype is np.float32 else 1e15
    xs = np.concatenate([grid(1e-4, x0, n // 2), grid(x0, top, n // 2)])
    return xs.astype(dtype)


def score_fresnel(cfg, dtype, n=260):
    xs = _probe(cfg["x0"], dtype, n)
    C, S = eval_fresnel(xs, cfg, dtype)
    wc = [mp.fresnelc(mpf(float(x))) for x in xs]
    ws = [mp.fresnels(mpf(float(x))) for x in xs]
    return max(ulp_err(C, wc, dtype)), max(ulp_err(S, ws, dtype))


def score_sici(cfg, dtype, n=260):
    xs = _probe(cfg["x0"], dtype, n)
    Si, Ci = eval_sici(xs, cfg, dtype)
    wsi = [mp.si(mpf(float(x))) for x in xs]
    wci = [mp.ci(mpf(float(x))) for x in xs]
    # Ci has zeros (the first at x ~ 0.6165) and no method is relatively accurate at
    # one. Grade against the magnitudes actually combined: |gamma + ln x| + |Cin| on
    # the small branch, |f| + |g| ~ 1/x on the auxiliary branch.
    env = []
    for x, w in zip(xs, wci):
        xm = mpf(float(x))
        env.append(abs(euler + log(xm)) + abs(cin(xm)) if float(x) <= cfg["x0"] else mpf(1) / xm)
    return max(ulp_err(Si, wsi, dtype)), max(envelope_ulp_err(Ci, wci, env, dtype))


# --------------------------------------------------------------------------- #
# building a config
# --------------------------------------------------------------------------- #

def build_fresnel(x0, dc, ds, dp, dq, cutoff):
    x0 = mpf(x0)
    w0 = x0 ** 4
    A = lambda w: mp.fresnelc(w ** mpf(0.25)) / w ** mpf(0.25) if w > 0 else mpf(1)
    B = lambda w: mp.fresnels(w ** mpf(0.25)) / w ** mpf(0.75) if w > 0 else MP_PI / 6
    u0 = 1 / (MP_PI * x0 * x0) ** 2
    P = lambda u: MP_PI * _xf(u) * fres_aux(_xf(u))[0] if u > 0 else mpf(1)
    Q = lambda u: MP_PI ** 2 * _xf(u) ** 3 * fres_aux(_xf(u))[1] if u > 0 else mpf(1)
    return {
        "x0": float(x0), "cutoff": cutoff,
        "cheb_c": cheb_coeffs(A, 0, w0, dc),
        "cheb_s": cheb_coeffs(B, 0, w0, ds),
        "aux_p": mono_coeffs(P, 0, u0, dp),
        "aux_q": mono_coeffs(Q, 0, u0, dq),
    }


def _xf(u):
    return sqrt(1 / (MP_PI * sqrt(u)))


def build_sici(x0, dsi, dcin, dp, dq, cutoff):
    x0 = mpf(x0)
    v0 = x0 * x0
    A = lambda v: mp.si(sqrt(v)) / sqrt(v) if v > 0 else mpf(1)
    B = lambda v: cin(sqrt(v)) / v if v > 0 else mpf(1) / 4
    w0 = 1 / v0
    P = lambda v: (1 / sqrt(v)) * sici_aux(1 / sqrt(v))[0] if v > 0 else mpf(1)
    Q = lambda v: (1 / v) * sici_aux(1 / sqrt(v))[1] if v > 0 else mpf(1)
    return {
        "x0": float(x0), "cutoff": cutoff,
        "cheb_si": cheb_coeffs(A, 0, v0, dsi),
        "cheb_cin": cheb_coeffs(B, 0, v0, dcin),
        "aux_p": mono_coeffs(P, 0, w0, dp),
        "aux_q": mono_coeffs(Q, 0, w0, dq),
    }


# --------------------------------------------------------------------------- #
# emit
# --------------------------------------------------------------------------- #

def fmt(c, ty):
    """A Rust float literal. numpy 2 reprs a scalar as `np.float32(x)`, so round
    through `float` - the f64 repr of an f32 value round-trips back to that f32."""
    v = float(c)
    return repr(float(np.float32(v))) if ty == "f32" else repr(v)


def emit_array(name, ty, coeffs, doc):
    body = ",\n    ".join(fmt(c, ty) for c in coeffs)
    return f"/// {doc}\npub const {name}: [{ty}; {len(coeffs)}] = [\n    {body},\n];\n"


def write_tables(path, header, blocks):
    with open(path, "w", newline="\n") as fh:
        fh.write(header)
        for b in blocks:
            fh.write("\n")
            fh.write(b)
    print(f"wrote {path}")


FRESNEL_HEADER = '''//! Coefficients for the Fresnel integrals `C(x)` and `S(x)`.
//!
//! GENERATED by `scripts/fresnel_sici_coeffs.py`. Do not edit by hand.
//!
//! `CHEB_*` are Chebyshev coefficients in `T_k` on `w = x^4` mapped from
//! `[0, X0^4]` to `[-1, 1]`, summed by Clenshaw. `AUX_*` are ascending monomial
//! coefficients in `u = 1/(pi x^2)^2`, summed by Horner, for the auxiliaries
//!
//! ```text
//! f = P(u)/(pi x),  g = Q(u)/(pi^2 x^3)
//! C = 1/2 + f sin t - g cos t,  S = 1/2 - f cos t - g sin t,  t = pi x^2/2
//! ```
//!
//! The auxiliary fits are sized at CONTRIBUTION-weighted targets, not plain
//! relative ones: `C` and `S` sit in `[0.32, 0.72]`, so `P` need only be accurate
//! to `eps*1.26x` and `Q` to `eps*3.9x^3`. Re-fitting them at plain relative
//! accuracy adds about five degrees and buys nothing.
#![allow(clippy::excessive_precision)]
'''

SICI_HEADER = '''//! Coefficients for the trigonometric integrals `Si(x)` and `Ci(x)`.
//!
//! GENERATED by `scripts/fresnel_sici_coeffs.py`. Do not edit by hand.
//!
//! `CHEB_*` are Chebyshev coefficients in `T_k` on `v = x^2` mapped from
//! `[0, X0^2]` to `[-1, 1]`, summed by Clenshaw; `CHEB_CIN_*` fits the entire
//! part `Cin = gamma + ln x - Ci`, so `Ci = (gamma + ln x) - x^2 B(v)`.
//! `AUX_*` are ascending monomial coefficients in `v = 1/x^2` for
//!
//! ```text
//! f = P(v)/x,  g = Q(v)/x^2
//! Si = pi/2 - f cos x - g sin x,  Ci = f sin x - g cos x
//! ```
//!
//! `P` is fitted at plain relative accuracy because `|Ci| ~ f`; `Q` is relaxed by
//! a factor `x` because it contributes at `1/x^2` against a `1/x` result. The
//! crossover is far out (12 in f64) because these auxiliaries are Stieltjes
//! functions whose cut reaches `v = 0`, so polynomial convergence near the
//! endpoint is sub-geometric: degree 33 at `x >= 6` against 17 at `x >= 12`.
#![allow(clippy::excessive_precision)]
'''


# --------------------------------------------------------------------------- #

# Cutoffs: |f| ~ 1/(pi x) (Fresnel) and 1/x (sici) fall under eps/4 of the limit.
CUT64 = 1.147e16
CUT32 = 2.136e7

CONFIGS = {
    "fresnel": {
        "f64": dict(x0=2.5265, d=(17, 16, 16, 16), cutoff=CUT64),
        "f32": dict(x0=2.5265, d=(11, 11, 3, 3), cutoff=CUT32),
    },
    "sici": {
        "f64": dict(x0=12.0, d=(19, 19, 17, 16), cutoff=CUT64),
        "f32": dict(x0=6.0, d=(7, 7, 8, 7), cutoff=CUT32),
    },
}


def main():
    sweep = "--sweep" in sys.argv
    built = {}

    for ty, dt in (("f64", np.float64), ("f32", np.float32)):
        c = CONFIGS["fresnel"][ty]
        cfg = build_fresnel(c["x0"], *c["d"], c["cutoff"])
        ec, es = score_fresnel(cfg, dt)
        print(f"fresnel {ty}: x0={c['x0']} deg={c['d']}  C {float(ec):.2f} ulp   S {float(es):.2f} ulp")
        built[("fresnel", ty)] = cfg

        c = CONFIGS["sici"][ty]
        cfg = build_sici(c["x0"], *c["d"], c["cutoff"])
        esi, eci = score_sici(cfg, dt)
        print(f"sici    {ty}: x0={c['x0']} deg={c['d']}  Si {float(esi):.2f} ulp   Ci {float(eci):.2f} ulp (envelope)")
        built[("sici", ty)] = cfg

    if sweep:
        print("\n-- sweep --")
        for ty, dt in (("f64", np.float64), ("f32", np.float32)):
            for x0 in ([2.0, 2.5265, 3.0] if ty == "f64" else [2.0, 2.5265]):
                for d in ([(13, 13, 25, 25), (17, 16, 16, 16), (20, 20, 12, 12)] if ty == "f64"
                          else [(8, 8, 5, 5), (11, 11, 3, 3)]):
                    cfg = build_fresnel(x0, *d, CUT64 if ty == "f64" else CUT32)
                    ec, es = score_fresnel(cfg, dt, 120)
                    print(f"  fresnel {ty} x0={x0:<7} d={d}  C {float(ec):8.2f}  S {float(es):8.2f}")
            for x0 in ([8.0, 10.0, 12.0] if ty == "f64" else [6.0, 8.0]):
                for d in ([(14, 14, 25, 24), (17, 17, 20, 19), (19, 19, 17, 16)] if ty == "f64"
                          else [(7, 7, 8, 7), (9, 8, 6, 5)]):
                    cfg = build_sici(x0, *d, CUT64 if ty == "f64" else CUT32)
                    esi, eci = score_sici(cfg, dt, 120)
                    print(f"  sici    {ty} x0={x0:<7} d={d}  Si {float(esi):8.2f}  Ci {float(eci):8.2f}")
        return

    f64, f32 = built[("fresnel", "f64")], built[("fresnel", "f32")]
    write_tables(os.path.join(HERE, "..", "src", "tables", "fresnel.rs"), FRESNEL_HEADER, [
        f"/// Crossover between the small-argument series and the auxiliary form.\npub const X0_F64: f64 = {f64['x0']!r};\n",
        f"/// `2/X0^4`: maps `w = x^4` onto the Chebyshev domain as `w*MAP - 1`.\npub const MAP_F64: f64 = {2.0 / f64['x0'] ** 4!r};\n",
        f"/// Above this, `C` and `S` are `1/2` to within half an ulp.\npub const CUTOFF_F64: f64 = {f64['cutoff']!r};\n",
        emit_array("CHEB_C_F64", "f64", f64["cheb_c"], "`C(x)/x` in `T_k(w)`."),
        emit_array("CHEB_S_F64", "f64", f64["cheb_s"], "`S(x)/x^3` in `T_k(w)`."),
        emit_array("AUX_P_F64", "f64", f64["aux_p"], "`P(u) = pi x f(x)`, ascending."),
        emit_array("AUX_Q_F64", "f64", f64["aux_q"], "`Q(u) = pi^2 x^3 g(x)`, ascending."),
        f"/// Crossover between the small-argument series and the auxiliary form.\npub const X0_F32: f32 = {fmt(f32["x0"], "f32")};\n",
        f"/// `2/X0^4`: maps `w = x^4` onto the Chebyshev domain as `w*MAP - 1`.\npub const MAP_F32: f32 = {fmt(2.0 / f32["x0"] ** 4, "f32")};\n",
        f"/// Above this, `C` and `S` are `1/2` to within half an ulp.\npub const CUTOFF_F32: f32 = {fmt(f32["cutoff"], "f32")};\n",
        emit_array("CHEB_C_F32", "f32", f32["cheb_c"], "`C(x)/x` in `T_k(w)`."),
        emit_array("CHEB_S_F32", "f32", f32["cheb_s"], "`S(x)/x^3` in `T_k(w)`."),
        emit_array("AUX_P_F32", "f32", f32["aux_p"], "`P(u) = pi x f(x)`, ascending."),
        emit_array("AUX_Q_F32", "f32", f32["aux_q"], "`Q(u) = pi^2 x^3 g(x)`, ascending."),
    ])

    f64, f32 = built[("sici", "f64")], built[("sici", "f32")]
    write_tables(os.path.join(HERE, "..", "src", "tables", "sici.rs"), SICI_HEADER, [
        f"/// Crossover between the small-argument series and the auxiliary form.\npub const X0_F64: f64 = {f64['x0']!r};\n",
        f"/// `2/X0^2`: maps `v = x^2` onto the Chebyshev domain as `v*MAP - 1`.\npub const MAP_F64: f64 = {2.0 / f64['x0'] ** 2!r};\n",
        f"/// Above this, `Si` is `pi/2` to within half an ulp. `Ci` has no cutoff.\npub const CUTOFF_F64: f64 = {f64['cutoff']!r};\n",
        emit_array("CHEB_SI_F64", "f64", f64["cheb_si"], "`Si(x)/x` in `T_k(v)`."),
        emit_array("CHEB_CIN_F64", "f64", f64["cheb_cin"], "`Cin(x)/x^2` in `T_k(v)`."),
        emit_array("AUX_P_F64", "f64", f64["aux_p"], "`P(v) = x f(x)`, ascending."),
        emit_array("AUX_Q_F64", "f64", f64["aux_q"], "`Q(v) = x^2 g(x)`, ascending."),
        f"/// Crossover between the small-argument series and the auxiliary form.\npub const X0_F32: f32 = {fmt(f32["x0"], "f32")};\n",
        f"/// `2/X0^2`: maps `v = x^2` onto the Chebyshev domain as `v*MAP - 1`.\npub const MAP_F32: f32 = {fmt(2.0 / f32["x0"] ** 2, "f32")};\n",
        f"/// Above this, `Si` is `pi/2` to within half an ulp. `Ci` has no cutoff.\npub const CUTOFF_F32: f32 = {fmt(f32["cutoff"], "f32")};\n",
        emit_array("CHEB_SI_F32", "f32", f32["cheb_si"], "`Si(x)/x` in `T_k(v)`."),
        emit_array("CHEB_CIN_F32", "f32", f32["cheb_cin"], "`Cin(x)/x^2` in `T_k(v)`."),
        emit_array("AUX_P_F32", "f32", f32["aux_p"], "`P(v) = x f(x)`, ascending."),
        emit_array("AUX_Q_F32", "f32", f32["aux_q"], "`Q(v) = x^2 g(x)`, ascending."),
    ])


if __name__ == "__main__":
    main()
