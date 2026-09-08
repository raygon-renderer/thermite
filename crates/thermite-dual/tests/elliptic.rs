//! The elliptic family on `Dual`: Carlson symmetric integrals and Legendre forms, plus the
//! Jacobi zeta and Heuman lambda companions, through the `carlson` / `ellint` request
//! structs.
//!
//! Nothing in `thermite-dual` is specific to these: the kernels are generic over any float
//! vector whose element carries `EllipticConsts`, and lifting that one constant onto
//! `Dual<E, N>` is the whole implementation. The derivative is the chain rule through the
//! Carlson duplication and the AGM (contractive algebraic iterations), so the value must
//! track the plain vector and the derivative must match a central difference on the plain
//! function in every argument.
//!
//! How closely the value tracks is a policy-tier question, and both tiers are checked here.
//! Below `Best`, a real vector lowers a polynomial with Estrin (`poly_n_internal`) to buy
//! ILP, while a composite takes the plain Horner default - a composite already saturates
//! the ILP Estrin exists to expose, so it is deliberately left out of that path. Same
//! polynomial, different bracketing, so the primal agrees to an ulp rather than to the
//! bit: measured against mpmath at 50 digits over 52 R_C/R_J/Z/Lambda_0 values, the two
//! disagree 8 times, always by one ulp, and every time it is the composite's Horner that
//! is the closer of the two (mean 0.59 ulp against Estrin's 0.73). At `Best` and above the
//! real vector switches back to Horner for exactly that reason, the two brackets coincide,
//! and the primal IS bit-identical - checked at 0 ulp on the same grid.

#![cfg(feature = "special")]

use thermite::math::policy::policies::Precision;
use thermite::prelude::*;
use thermite::vector::ops::MulAddExt;
use thermite_dual::Dual;
use thermite_special::{SpecialMath, SpecialMathWithPolicy};
use thermite_special::elliptic::{
    CarlsonRc, CarlsonRd, CarlsonRf, CarlsonRg, CarlsonRj, EllintD, EllintDInc, EllintE, EllintEInc, EllintF, EllintK,
    EllintPi, EllintPiInc, HeumanLambda, JacobiZeta,
};

type V = Vector<f64>;
type D = Dual<V, 1>;

fn v(x: f64) -> V {
    V::splat(x)
}

/// A constant (zero derivative) dual.
fn c(x: f64) -> D {
    D::constant(v(x))
}

/// The seeded variable.
fn var(x: f64) -> D {
    D::variable(v(x), 0)
}

fn central<F: Fn(f64) -> f64>(f: F, x: f64) -> f64 {
    let h = 1e-6 * x.abs().max(1.0);
    (f(x + h) - f(x - h)) / (2.0 * h)
}

/// The dual and the plain vector must agree on the kernels' FMA gate,
/// `matches!(HAS_NATIVE_FMA, True)`, or the Carlson duplication runs its fused reduction for
/// one and the add-then-scale form for the other: same math, two roundings, and the primal
/// stops being bit-identical. `Dual` forwards the inner vector's answer for exactly this
/// reason - the flag is about FMA speed, and a dual FMA is `N + 1` inner FMAs.
const _: () = assert!(matches!(
    (<D as MulAddExt>::HAS_NATIVE_FMA, <V as MulAddExt>::HAS_NATIVE_FMA),
    (thermite::tribool::True, thermite::tribool::True)
        | (thermite::tribool::False, thermite::tribool::False)
        | (thermite::tribool::Indeterminate, thermite::tribool::Indeterminate)
));

/// Total order on the f64 line, so an ulp gap is a subtraction. Negative lanes reflect
/// (`Z` goes negative past `pi/2`), and the two halves join at zero.
fn ord(x: f64) -> i64 {
    let b = x.to_bits() as i64;
    if b < 0 { i64::MIN.wrapping_sub(b) } else { b }
}

fn ulp_gap(a: f64, b: f64) -> u64 {
    ord(a).wrapping_sub(ord(b)).unsigned_abs()
}

/// `got` is the dual evaluation with argument `i` seeded. `plain` evaluates the real function
/// with argument `i` replaced. Value within `max_ulp` (0 = to the bit; see the module docs
/// for why the default tier is not 0), derivative against a central difference.
fn check(name: &str, args: &[f64], i: usize, got: D, plain: impl Fn(f64) -> f64, max_ulp: u64) {
    let value = plain(args[i]);
    let gv = got.re.extract::<0>();
    let gap = ulp_gap(gv, value);
    assert!(gap <= max_ulp, "{name}{args:?} value: dual {gv:e}, plain {value:e}, {gap} ulp > {max_ulp}");

    let want = central(&plain, args[i]);
    let gd = got.dual[0].extract::<0>();
    let scale = want.abs().max(1e-3);
    let e = (gd - want).abs() / scale;
    assert!(e <= 2e-7, "d/d[{i}] {name}{args:?}: dual {gd:e}, central {want:e}, rel {e:e}");
}

/// Stamps a check over every argument of one request kind, at both policy tiers: `$entry` is
/// the default-tier entry point and `$entry_p` its policy-taking twin, run at `Precision`.
/// The default tier allows an ulp on the value, `Precision` demands the bit (module docs).
macro_rules! all_args {
    ($name:literal, $args:expr, $entry:ident, $entry_p:ident, $kind:ident { $($field:ident),* }) => {{
        let args: &[f64] = &$args;
        let n = args.len();
        for i in 0..n {
            let mut d: Vec<D> = args.iter().map(|&a| c(a)).collect();
            d[i] = var(args[i]);

            let mut it = d.iter().copied();
            let got = D::$entry($kind { $($field: it.next().unwrap()),* });
            let plain = |t: f64| {
                let mut p: Vec<f64> = args.to_vec();
                p[i] = t;
                let mut it = p.iter().map(|&a| v(a));
                V::$entry($kind { $($field: it.next().unwrap()),* }).extract::<0>()
            };
            check($name, args, i, got, plain, 2);

            let mut it = d.iter().copied();
            let got = SpecialMathWithPolicy::$entry_p::<Precision, _>($kind { $($field: it.next().unwrap()),* });
            let plain = |t: f64| {
                let mut p: Vec<f64> = args.to_vec();
                p[i] = t;
                let mut it = p.iter().map(|&a| v(a));
                SpecialMathWithPolicy::$entry_p::<Precision, _>($kind { $($field: it.next().unwrap()),* })
                    .extract::<0>()
            };
            check($name, args, i, got, plain, 0);
        }
    }};
}

#[test]
fn carlson_on_dual_matches_central_differences() {
    for &(x, y, z, p) in &[(1.0, 2.0, 4.0, 0.75), (0.5, 0.25, 8.0, 3.0), (3.0, 3.0, 0.125, 0.5), (2.0, 2.0, 2.0, 2.0)] {
        all_args!("R_F", [x, y, z], carlson, carlson_p, CarlsonRf { x, y, z });
        all_args!("R_D", [x, y, z], carlson, carlson_p, CarlsonRd { x, y, z });
        all_args!("R_G", [x, y, z], carlson, carlson_p, CarlsonRg { x, y, z });
        all_args!("R_J", [x, y, z, p], carlson, carlson_p, CarlsonRj { x, y, z, p });
        all_args!("R_C", [x, y], carlson, carlson_p, CarlsonRc { x, y });
    }
}

#[test]
fn legendre_complete_on_dual_matches_central_differences() {
    for &k in &[0.1, 0.25, 0.5, 0.75, 0.9, 0.97] {
        all_args!("K", [k], ellint, ellint_p, EllintK { k });
        all_args!("E", [k], ellint, ellint_p, EllintE { k });
        all_args!("D", [k], ellint, ellint_p, EllintD { k });
        for &n in &[0.25, -0.5, 0.6] {
            all_args!("Pi", [n, k], ellint, ellint_p, EllintPi { n, k });
        }
    }
}

#[test]
fn legendre_incomplete_on_dual_matches_central_differences() {
    for &phi in &[0.2, 0.75, 1.25, 2.0, 4.0] {
        for &k in &[0.25, 0.5, 0.875] {
            all_args!("F", [phi, k], ellint, ellint_p, EllintF { phi, k });
            all_args!("E", [phi, k], ellint, ellint_p, EllintEInc { phi, k });
            all_args!("D", [phi, k], ellint, ellint_p, EllintDInc { phi, k });
            all_args!("Z", [phi, k], ellint, ellint_p, JacobiZeta { phi, k });
            for &n in &[0.25, -0.5] {
                all_args!("Pi", [n, phi, k], ellint, ellint_p, EllintPiInc { n, phi, k });
            }
        }
    }
    // Heuman lambda's usual domain is |phi| <= pi/2, but the far arm is exercised at 2.0.
    for &phi in &[0.2, 0.75, 1.25, 2.0] {
        for &k in &[0.25, 0.5, 0.875] {
            all_args!("Lambda0", [phi, k], ellint, ellint_p, HeumanLambda { phi, k });
        }
    }
}

#[test]
fn a_known_derivative_dk_dk() {
    // dK/dk = E/(k k'^2) - K/k (DLMF 19.4.1 in the modulus), checked against the dual.
    for &k in &[0.25, 0.5, 0.75] {
        let kk = var(k);
        let dk = D::ellint(EllintK { k: kk }).dual[0].extract::<0>();
        let big_k = V::ellint(EllintK { k: v(k) }).extract::<0>();
        let big_e = V::ellint(EllintE { k: v(k) }).extract::<0>();
        let want = big_e / (k * (1.0 - k * k)) - big_k / k;
        let e = ((dk - want) / want).abs();
        assert!(e <= 1e-13, "dK/dk at {k}: dual {dk:e}, closed form {want:e}, rel {e:e}");
    }
}
