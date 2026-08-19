//! `zernike_basis_d`: the value basis plus `dZ/dx` and `dZ/dy` for every mode.
//!
//! The primary anchor is `Dual<V, 2>` seeded with an identity Jacobian, run through the
//! *value* kernel. That is a genuinely independent route to the same numbers, since dual
//! arithmetic differentiates every operation of the ladder mechanically, where
//! `zernike_basis_d` differentiates the recurrence analytically and shares it between the
//! value and both slopes, so agreement is evidence rather than tautology.
//!
//! It is also the comparison the docs make a performance claim about, which makes having
//! it in the suite worth more than the closed forms alone.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]

use thermite::Vector;
use thermite::prelude::*;
use thermite_dual::Dual;
use thermite_special::specialized::MAX_ZERNIKE_DEGREE;
use thermite_special::zernike::ansi_index;
use thermite_special::{RealPrimalMath, SpecialMath, ZERNIKE_ORTHONORMAL, ZERNIKE_UNIT_PEAK};

type D = Vector<f64>;
type D2 = Dual<D, 2>;

#[track_caller]
fn close(name: &str, got: f64, want: f64, tol: f64) {
    let err = (got - want).abs() / want.abs().max(1.0);
    assert!(err <= tol, "{name}: got {got:?}, want {want:?} (err {err:e})");
}

/// The pupil centre, the rim, both axes, and interior points in all four quadrants.
const SAMPLES: &[(f64, f64)] = &[
    (0.0, 0.0),
    (0.3, 0.4),
    (-0.3, 0.4),
    (0.3, -0.4),
    (-0.3, -0.4),
    (0.8, 0.0),
    (0.0, 0.8),
    (-0.6, 0.0),
    (0.0, -0.6),
    (0.6, 0.8),
    (0.02, 0.05),
];

/// Checks values and both gradients at degree `L` against `Dual<V, 2>` through the value
/// kernel, in the given normalization.
fn check_against_dual<const L: usize, const NORM: u8, const N: usize>() {
    for &(x, y) in SAMPLES {
        let mut out = [D::ZERO; N];
        let mut ddx = [D::ZERO; N];
        let mut ddy = [D::ZERO; N];

        D::zernike_basis_d::<L, NORM, N>(D::splat(x), D::splat(y), &mut out, &mut ddx, &mut ddy);

        // The same basis through dual arithmetic, seeded with the identity Jacobian.
        let mut dual = [D2::ZERO; N];
        D2::zernike_basis::<L, NORM, N>(D2::variable(D::splat(x), 0), D2::variable(D::splat(y), 1), &mut dual);

        // And the plain value kernel, so the `_d` form's values are pinned to the form
        // everything else in the suite already tests rather than only to the dual.
        let mut plain = [D::ZERO; N];
        D::zernike_basis::<L, NORM, N>(D::splat(x), D::splat(y), &mut plain);

        for j in 0..N {
            let ctx = format!("L={L} NORM={NORM} j={j} at ({x}, {y})");
            let grad = dual[j].gradient();

            close(
                &format!("value {ctx}"),
                out[j].extract::<0>(),
                plain[j].extract::<0>(),
                0.0,
            );
            close(
                &format!("value-vs-dual {ctx}"),
                out[j].extract::<0>(),
                dual[j].value().extract::<0>(),
                1e-14,
            );
            close(
                &format!("ddx {ctx}"),
                ddx[j].extract::<0>(),
                grad[0].extract::<0>(),
                1e-13,
            );
            close(
                &format!("ddy {ctx}"),
                ddy[j].extract::<0>(),
                grad[1].extract::<0>(),
                1e-13,
            );
        }
    }
}

#[test]
fn gradients_match_forward_mode_autodiff() {
    check_against_dual::<0, ZERNIKE_UNIT_PEAK, 1>();
    check_against_dual::<1, ZERNIKE_UNIT_PEAK, 3>();
    check_against_dual::<2, ZERNIKE_UNIT_PEAK, 6>();
    check_against_dual::<3, ZERNIKE_UNIT_PEAK, 10>();
    check_against_dual::<4, ZERNIKE_UNIT_PEAK, 15>();
    check_against_dual::<6, ZERNIKE_UNIT_PEAK, 28>();
    check_against_dual::<9, ZERNIKE_UNIT_PEAK, 55>();

    check_against_dual::<2, ZERNIKE_ORTHONORMAL, 6>();
    check_against_dual::<4, ZERNIKE_ORTHONORMAL, 15>();
    check_against_dual::<7, ZERNIKE_ORTHONORMAL, 36>();
    check_against_dual::<11, ZERNIKE_ORTHONORMAL, 78>();
}

#[test]
fn gradients_match_autodiff_across_the_ladder_boundary() {
    assert_eq!(MAX_ZERNIKE_DEGREE, 16, "ladder cap moved; update the degrees below");

    check_against_dual::<16, ZERNIKE_ORTHONORMAL, 153>();
    check_against_dual::<17, ZERNIKE_ORTHONORMAL, 171>();
}

#[test]
fn low_order_gradients_are_the_closed_forms() {
    // The named aberrations, unit-peak, written out as bare polynomials in (x, y):
    //   Z_1^1  = x            Z_1^-1 = y
    //   Z_2^0  = 2(x^2+y^2)-1 Z_2^2  = x^2 - y^2      Z_2^-2 = 2xy
    //   Z_3^1  = (3(x^2+y^2) - 2) x                   (coma)
    const L: usize = 3;
    const N: usize = 10;

    for &(x, y) in SAMPLES {
        let mut out = [D::ZERO; N];
        let mut ddx = [D::ZERO; N];
        let mut ddy = [D::ZERO; N];

        D::zernike_basis_d::<L, ZERNIKE_UNIT_PEAK, N>(D::splat(x), D::splat(y), &mut out, &mut ddx, &mut ddy);

        let s = x * x + y * y;

        let cases: &[(u32, i32, f64, f64, f64)] = &[
            (1, 1, x, 1.0, 0.0),
            (1, -1, y, 0.0, 1.0),
            (2, 0, 2.0 * s - 1.0, 4.0 * x, 4.0 * y),
            (2, 2, x * x - y * y, 2.0 * x, -2.0 * y),
            (2, -2, 2.0 * x * y, 2.0 * y, 2.0 * x),
            (3, 1, (3.0 * s - 2.0) * x, 9.0 * x * x + 3.0 * y * y - 2.0, 6.0 * x * y),
            (3, -1, (3.0 * s - 2.0) * y, 6.0 * x * y, 3.0 * x * x + 9.0 * y * y - 2.0),
        ];

        for &(n, m, v, dx, dy) in cases {
            let j = ansi_index(n, m) as usize;
            let ctx = format!("Z_{n}^{m} at ({x}, {y})");

            close(&format!("value {ctx}"), out[j].extract::<0>(), v, 1e-14);
            close(&format!("ddx {ctx}"), ddx[j].extract::<0>(), dx, 1e-14);
            close(&format!("ddy {ctx}"), ddy[j].extract::<0>(), dy, 1e-14);
        }
    }
}

#[test]
fn the_pupil_centre_gradient_is_finite() {
    // The whole point of the Cartesian formulation. In polar coordinates the tangential
    // derivative is dZ/dtheta / rho, which is 0/0 here and is why hand-rolled polar
    // implementations special-case the origin.
    //
    // The modes with a nonzero gradient at the centre are exactly the |m| = 1 ones, not
    // just tilt: Z_n^{+-1} = Q_{k,1}(s) * (x or y), whose linear term survives. Coma
    // (Z_3^1) has slope -2 there. Everything with |m| != 1 is at least quadratic, since
    // its azimuthal factor already is.
    //
    // The nonzero value is Q_{k,1}(0) = P_k^{(0,1)}(-1) = (-1)^k (k+1), with k = (n-1)/2.
    const L: usize = 8;
    const N: usize = 45;

    let mut out = [D::ZERO; N];
    let mut ddx = [D::ZERO; N];
    let mut ddy = [D::ZERO; N];

    D::zernike_basis_d::<L, ZERNIKE_UNIT_PEAK, N>(D::ZERO, D::ZERO, &mut out, &mut ddx, &mut ddy);

    for j in 0..N {
        let gx = ddx[j].extract::<0>();
        let gy = ddy[j].extract::<0>();

        assert!(
            gx.is_finite() && gy.is_finite(),
            "j={j}: gradient is not finite ({gx}, {gy})"
        );
    }

    for j in 0..N {
        let (n, m) = thermite_special::zernike::ansi_to_nm(j as u32);

        let gx = ddx[j].extract::<0>();
        let gy = ddy[j].extract::<0>();

        let (want_x, want_y) = match m {
            1 | -1 => {
                let k = (n - 1) / 2;
                let q0 = if k % 2 == 0 { (k + 1) as f64 } else { -((k + 1) as f64) };

                if m == 1 { (q0, 0.0) } else { (0.0, q0) }
            }
            _ => (0.0, 0.0),
        };

        close(&format!("d(Z_{n}^{m})/dx at centre"), gx, want_x, 0.0);
        close(&format!("d(Z_{n}^{m})/dy at centre"), gy, want_y, 0.0);
    }

    // Spelled out for the two everyone knows, so the rule above is anchored to something
    // recognizable: unit tilt, and coma's -2.
    close("d(x-tilt)/dx", ddx[ansi_index(1, 1) as usize].extract::<0>(), 1.0, 0.0);
    close("d(y-tilt)/dy", ddy[ansi_index(1, -1) as usize].extract::<0>(), 1.0, 0.0);
    close("d(coma)/dx", ddx[ansi_index(3, 1) as usize].extract::<0>(), -2.0, 0.0);
}

#[test]
fn gradients_match_forward_mode_autodiff_in_f32() {
    // The same dual cross-check at single precision. The gradient recurrence carries one
    // more accumulated term than the value recurrence, so it is the more precision-
    // sensitive of the two and worth running at f32 on its own account.
    type F = Vector<f32>;
    type F2 = Dual<F, 2>;

    fn check<const L: usize, const N: usize>() {
        for &(x, y) in SAMPLES {
            let (xf, yf) = (x as f32, y as f32);

            let mut out = [F::ZERO; N];
            let mut ddx = [F::ZERO; N];
            let mut ddy = [F::ZERO; N];

            F::zernike_basis_d::<L, ZERNIKE_ORTHONORMAL, N>(F::splat(xf), F::splat(yf), &mut out, &mut ddx, &mut ddy);

            let mut dual = [F2::ZERO; N];
            F2::zernike_basis::<L, ZERNIKE_ORTHONORMAL, N>(
                F2::variable(F::splat(xf), 0),
                F2::variable(F::splat(yf), 1),
                &mut dual,
            );

            for j in 0..N {
                let ctx = format!("f32 L={L} j={j} at ({xf}, {yf})");
                let grad = dual[j].gradient();

                close(
                    &format!("value {ctx}"),
                    out[j].extract::<0>() as f64,
                    dual[j].value().extract::<0>() as f64,
                    1e-5,
                );
                close(
                    &format!("ddx {ctx}"),
                    ddx[j].extract::<0>() as f64,
                    grad[0].extract::<0>() as f64,
                    1e-5,
                );
                close(
                    &format!("ddy {ctx}"),
                    ddy[j].extract::<0>() as f64,
                    grad[1].extract::<0>() as f64,
                    1e-5,
                );
            }
        }
    }

    check::<2, 6>();
    check::<4, 15>();
    check::<6, 28>();
    check::<10, 66>();
}
