//! The elliptic family on `Dual`: Carlson symmetric integrals and Legendre forms, plus the
//! Jacobi zeta and Heuman lambda companions, through the `carlson` / `ellint` request
//! structs.
//!
//! Nothing in `thermite-dual` is specific to these: the kernels are generic over any float
//! vector whose element carries `EllipticConsts`, and lifting that one constant onto
//! `Dual<E, N>` is the whole implementation. The derivative is the chain rule through the
//! Carlson duplication and the AGM (contractive algebraic iterations), so the value must
//! match the plain vector to the bit and the derivative must match a central difference on
//! the plain function in every argument.

#![cfg(feature = "special")]

use thermite::prelude::*;
use thermite_dual::Dual;
use thermite_special::SpecialMath;
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

/// `got` is the dual evaluation with argument `i` seeded. `plain` evaluates the real function
/// with argument `i` replaced. Value to the bit, derivative against a central difference.
fn check(name: &str, args: &[f64], i: usize, got: D, plain: impl Fn(f64) -> f64) {
    let value = plain(args[i]);
    let gv = got.re.extract::<0>();
    assert_eq!(gv.to_bits(), value.to_bits(), "{name}{args:?} value: dual {gv:e}, plain {value:e}");

    let want = central(&plain, args[i]);
    let gd = got.dual[0].extract::<0>();
    let scale = want.abs().max(1e-3);
    let e = (gd - want).abs() / scale;
    assert!(e <= 2e-7, "d/d[{i}] {name}{args:?}: dual {gd:e}, central {want:e}, rel {e:e}");
}

/// Stamps a check over every argument of one request kind: `$call` builds the request from a
/// slice of `D`, `$plain` from a slice of `f64`.
macro_rules! all_args {
    ($name:literal, $args:expr, $entry:ident, $kind:ident { $($field:ident),* }) => {{
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
            check($name, args, i, got, plain);
        }
    }};
}

#[test]
fn carlson_on_dual_matches_central_differences() {
    for &(x, y, z, p) in &[(1.0, 2.0, 4.0, 0.75), (0.5, 0.25, 8.0, 3.0), (3.0, 3.0, 0.125, 0.5), (2.0, 2.0, 2.0, 2.0)] {
        all_args!("R_F", [x, y, z], carlson, CarlsonRf { x, y, z });
        all_args!("R_D", [x, y, z], carlson, CarlsonRd { x, y, z });
        all_args!("R_G", [x, y, z], carlson, CarlsonRg { x, y, z });
        all_args!("R_J", [x, y, z, p], carlson, CarlsonRj { x, y, z, p });
        all_args!("R_C", [x, y], carlson, CarlsonRc { x, y });
    }
}

#[test]
fn legendre_complete_on_dual_matches_central_differences() {
    for &k in &[0.1, 0.25, 0.5, 0.75, 0.9, 0.97] {
        all_args!("K", [k], ellint, EllintK { k });
        all_args!("E", [k], ellint, EllintE { k });
        all_args!("D", [k], ellint, EllintD { k });
        for &n in &[0.25, -0.5, 0.6] {
            all_args!("Pi", [n, k], ellint, EllintPi { n, k });
        }
    }
}

#[test]
fn legendre_incomplete_on_dual_matches_central_differences() {
    for &phi in &[0.2, 0.75, 1.25, 2.0, 4.0] {
        for &k in &[0.25, 0.5, 0.875] {
            all_args!("F", [phi, k], ellint, EllintF { phi, k });
            all_args!("E", [phi, k], ellint, EllintEInc { phi, k });
            all_args!("D", [phi, k], ellint, EllintDInc { phi, k });
            all_args!("Z", [phi, k], ellint, JacobiZeta { phi, k });
            for &n in &[0.25, -0.5] {
                all_args!("Pi", [n, phi, k], ellint, EllintPiInc { n, phi, k });
            }
        }
    }
    // Heuman lambda's usual domain is |phi| <= pi/2, but the far arm is exercised at 2.0.
    for &phi in &[0.2, 0.75, 1.25, 2.0] {
        for &k in &[0.25, 0.5, 0.875] {
            all_args!("Lambda0", [phi, k], ellint, HeumanLambda { phi, k });
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
