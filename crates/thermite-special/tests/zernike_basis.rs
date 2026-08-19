//! The batch Zernike kernel, `zernike_basis::<L, NORM, N>`.
//!
//! The anchor here is different from `tests/zernike.rs`: rather than an external
//! reference, the batch form is checked mode-by-mode against the *single-mode*
//! `zernike`, which that suite already pins against the exact factorial sum. The two
//! share no arithmetic (the single-mode form goes through `jacobi` in `2rho^2 - 1` and
//! a `sin_cos`, the batch form through a reduced recurrence in `s = x^2 + y^2` and a
//! complex power ladder), so agreement between them is a real cross-check rather than a
//! restatement.
//!
//! What that leaves uncovered is anything both forms could get wrong together, so the
//! orthonormality integral is repeated here through the batch path, and the pupil centre
//! (which the polar form reaches only through a degenerate `atan2`) gets its own test.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]

use std::f64::consts::{FRAC_1_SQRT_2, PI};

use thermite::Vector;
use thermite::prelude::*;
use thermite_special::specialized::MAX_ZERNIKE_DEGREE;
use thermite_special::zernike::{
    ansi_index, ansi_to_nm, count_up_to_degree, fringe_to_ansi, fringe_to_nm, noll_to_ansi, noll_to_nm,
};
use thermite_special::{SpecialMath, ZERNIKE_ORTHONORMAL, ZERNIKE_UNIT_PEAK};

type D = Vector<f64>;

#[track_caller]
fn close(name: &str, got: f64, want: f64, tol: f64) {
    let err = (got - want).abs() / want.abs().max(1.0);
    assert!(err <= tol, "{name}: got {got:?}, want {want:?} (err {err:e})");
}

/// Cartesian pupil samples: the centre, the rim, both axes, and interior points in every
/// quadrant, so each branch of the complex ladder's sign handling is exercised.
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
    (0.05, 0.02),
    (FRAC_1_SQRT_2, FRAC_1_SQRT_2), // exactly on the rim, at 45 degrees
];

/// The single-mode form at the same point, reached through polar coordinates.
fn single<const NORM: u8>(x: f64, y: f64, n: u32, m: i32) -> f64 {
    let rho = x.hypot(y);
    let theta = y.atan2(x);

    D::splat(rho).zernike::<NORM>(D::splat(theta), n, m).extract::<0>()
}

/// Checks the whole basis at degree `L` against the single-mode form, in both
/// normalizations. `N` must be `(L+1)(L+2)/2`.
fn check_degree<const L: usize, const N: usize>() {
    assert_eq!(N, count_up_to_degree(L as u32) as usize, "test setup: N is wrong for L");

    for &(x, y) in SAMPLES {
        let mut peak = [D::ZERO; N];
        let mut ortho = [D::ZERO; N];

        D::zernike_basis::<L, ZERNIKE_UNIT_PEAK, N>(D::splat(x), D::splat(y), &mut peak);
        D::zernike_basis::<L, ZERNIKE_ORTHONORMAL, N>(D::splat(x), D::splat(y), &mut ortho);

        for (j, (&got_peak, &got_ortho)) in peak.iter().zip(ortho.iter()).enumerate() {
            let (n, m) = ansi_to_nm(j as u32);

            // Every slot must be written: an untouched one still holds the ZERO the
            // caller's buffer came in with, which a value comparison against a mode that
            // happens to vanish would not catch. The rim samples make every mode nonzero.
            close(
                &format!("L={L} peak j={j} (n={n}, m={m}) at ({x}, {y})"),
                got_peak.extract::<0>(),
                single::<ZERNIKE_UNIT_PEAK>(x, y, n, m),
                1e-13,
            );

            close(
                &format!("L={L} ortho j={j} (n={n}, m={m}) at ({x}, {y})"),
                got_ortho.extract::<0>(),
                single::<ZERNIKE_ORTHONORMAL>(x, y, n, m),
                1e-13,
            );
        }
    }
}

#[test]
fn batch_matches_the_single_mode_form_on_the_unrolled_path() {
    check_degree::<0, 1>();
    check_degree::<1, 3>();
    check_degree::<2, 6>();
    check_degree::<3, 10>();
    check_degree::<4, 15>();
    check_degree::<5, 21>();
    check_degree::<6, 28>();
    check_degree::<8, 45>();
    check_degree::<11, 78>();
}

#[test]
fn batch_matches_the_single_mode_form_at_the_ladder_boundary() {
    // The last stamped degree and the first rolled one, so both sides of the
    // `L > MAX_ZERNIKE_DEGREE` split are covered and the fallback cannot rot unnoticed.
    assert_eq!(MAX_ZERNIKE_DEGREE, 16, "ladder cap moved; update the degrees below");

    check_degree::<16, 153>();
    check_degree::<17, 171>();
    check_degree::<18, 190>();
}

#[test]
fn the_pupil_centre_is_finite_and_exact() {
    // The point the polar form can only reach through atan2(0, 0). Here it is just the
    // origin of a polynomial: every |m| > 0 mode vanishes because (x + iy)^m does, and
    // R_n^0(0) = (-1)^{n/2}.
    const L: usize = 8;
    const N: usize = 45;

    let mut basis = [D::ZERO; N];
    D::zernike_basis::<L, ZERNIKE_UNIT_PEAK, N>(D::ZERO, D::ZERO, &mut basis);

    for (j, &value) in basis.iter().enumerate() {
        let (n, m) = ansi_to_nm(j as u32);
        let got = value.extract::<0>();

        assert!(got.is_finite(), "j={j} (n={n}, m={m}) is not finite: {got}");

        let want = if m != 0 {
            0.0
        } else if (n / 2) % 2 == 0 {
            1.0
        } else {
            -1.0
        };

        close(&format!("centre j={j} (n={n}, m={m})"), got, want, 1e-15);
    }
}

#[test]
fn batch_modes_are_orthonormal() {
    // The same integral as `tests/zernike.rs`, driven through the batch path. Repeated
    // rather than assumed: this is the one property the cross-check against the
    // single-mode form cannot establish, since both forms would have to be wrong
    // together and they share no arithmetic.
    const N_U: usize = 2048;
    const N_T: usize = 64;
    const L: usize = 4;
    const N: usize = 15;

    let mut gram = [[0.0f64; N]; N];
    let mut basis = [D::ZERO; N];

    for iu in 0..N_U {
        let u = (iu as f64 + 0.5) / N_U as f64;
        let rho = u.sqrt();

        for it in 0..N_T {
            let theta = 2.0 * PI * (it as f64 + 0.5) / N_T as f64;

            let (x, y) = (rho * theta.cos(), rho * theta.sin());

            D::zernike_basis::<L, ZERNIKE_ORTHONORMAL, N>(D::splat(x), D::splat(y), &mut basis);

            for (a, row) in gram.iter_mut().enumerate() {
                let va = basis[a].extract::<0>();

                for (b, cell) in row.iter_mut().enumerate() {
                    *cell += va * basis[b].extract::<0>();
                }
            }
        }
    }

    let scale = 1.0 / (N_U * N_T) as f64;

    for (a, row) in gram.iter().enumerate() {
        for (b, &cell) in row.iter().enumerate() {
            let got = cell * scale;
            let want = if a == b { 1.0 } else { 0.0 };

            assert!(
                (got - want).abs() < 2e-4,
                "gram[{:?}][{:?}] = {got}, want {want}",
                ansi_to_nm(a as u32),
                ansi_to_nm(b as u32)
            );
        }
    }
}

#[test]
fn the_gathers_agree_with_the_compositions_they_replace() {
    for j in 1..=231 {
        assert_eq!(
            noll_to_ansi(j),
            {
                let (n, m) = noll_to_nm(j);
                ansi_index(n, m)
            },
            "noll_to_ansi({j})"
        );

        assert_eq!(
            fringe_to_ansi(j),
            {
                let (n, m) = fringe_to_nm(j);
                ansi_index(n, m)
            },
            "fringe_to_ansi({j})"
        );
    }
}

#[test]
fn the_gathers_pick_the_named_aberrations() {
    // Reading a basis buffer the way an instrument would: the classical names, through
    // whichever index scheme the coefficient set arrived in.
    const L: usize = 4;
    const N: usize = 15;

    let (x, y) = (0.36, -0.48); // rho = 0.6
    let rho: f64 = 0.6;

    let mut basis = [D::ZERO; N];
    D::zernike_basis::<L, ZERNIKE_UNIT_PEAK, N>(D::splat(x), D::splat(y), &mut basis);

    let at = |j: usize| basis[j].extract::<0>();

    // Noll 4 and Fringe 4 are both defocus, Z_2^0 = 2 rho^2 - 1.
    let defocus = 2.0 * rho * rho - 1.0;
    close("noll 4", at(noll_to_ansi(4) as usize), defocus, 1e-14);
    close("fringe 4", at(fringe_to_ansi(4) as usize), defocus, 1e-14);
    close("ansi 4", at(4), defocus, 1e-14);

    // Fringe 9 is primary spherical, Z_4^0 = 6 rho^4 - 6 rho^2 + 1 - and it lands at
    // ANSI 12, not 8. The reordering the Fringe scheme exists to produce.
    let r2 = rho * rho;
    assert_eq!(fringe_to_ansi(9), 12);
    close(
        "fringe 9",
        at(fringe_to_ansi(9) as usize),
        6.0 * r2 * r2 - 6.0 * r2 + 1.0,
        1e-14,
    );

    // Noll 2 is x-tilt, Z_1^1 = rho cos(theta) = x.
    close("noll 2", at(noll_to_ansi(2) as usize), x, 1e-15);
    // Noll 3 is y-tilt, Z_1^-1 = rho sin(theta) = y.
    close("noll 3", at(noll_to_ansi(3) as usize), y, 1e-15);
}

#[test]
fn the_batch_form_works_on_composite_types() {
    use thermite_compensated::Compensated;
    use thermite_complex::Complex;
    use thermite_dual::Dual;

    // The kernel is a default method over vector arithmetic, so every composite gets it
    // without an implementation. "Gets it" is checked here rather than assumed: a default
    // method is only monomorphized where it is called, so nothing before this test had
    // ever instantiated the ladder for anything but a plain vector.
    const L: usize = 3;
    const N: usize = 10;

    let (x, y) = (0.3, 0.4); // rho^2 = 0.25

    let mut plain = [D::ZERO; N];
    D::zernike_basis::<L, ZERNIKE_ORTHONORMAL, N>(D::splat(x), D::splat(y), &mut plain);

    let mut dual = [Dual::<D, 1>::ZERO; N];
    Dual::<D, 1>::zernike_basis::<L, ZERNIKE_ORTHONORMAL, N>(
        Dual::variable(D::splat(x), 0),
        Dual::constant(D::splat(y)),
        &mut dual,
    );

    let mut complex = [Complex::<D>::ZERO; N];
    Complex::<D>::zernike_basis::<L, ZERNIKE_ORTHONORMAL, N>(
        Complex::new(D::splat(x), D::ZERO),
        Complex::new(D::splat(y), D::ZERO),
        &mut complex,
    );

    let mut comp = [Compensated::<D>::ZERO; N];
    Compensated::<D>::zernike_basis::<L, ZERNIKE_ORTHONORMAL, N>(
        Compensated::new(D::splat(x)),
        Compensated::new(D::splat(y)),
        &mut comp,
    );

    for j in 0..N {
        let want = plain[j].extract::<0>();
        let (n, m) = ansi_to_nm(j as u32);
        let ctx = format!("j={j} (n={n}, m={m})");

        close(&format!("dual {ctx}"), dual[j].value().extract::<0>(), want, 1e-14);
        close(&format!("complex re {ctx}"), complex[j].re.extract::<0>(), want, 1e-14);
        close(&format!("complex im {ctx}"), complex[j].im.extract::<0>(), 0.0, 1e-14);
        close(
            &format!("compensated {ctx}"),
            comp[j].value().extract::<0>(),
            want,
            1e-14,
        );
    }

    // Defocus, orthonormal: Z_2^0 = sqrt(3)(2 rho^2 - 1), and d/dx of it is 4 sqrt(3) x.
    let defocus = ansi_index(2, 0) as usize;
    close("defocus", plain[defocus].extract::<0>(), -3f64.sqrt() / 2.0, 1e-14);
    close(
        "d(defocus)/dx",
        dual[defocus].gradient()[0].extract::<0>(),
        4.0 * 3f64.sqrt() * x,
        1e-13,
    );
}

#[test]
fn the_batch_form_agrees_with_the_single_mode_form_in_f32() {
    // The whole cross-check at single precision. The batch and single-mode routes now
    // share the reduced recurrence, so what this really guards is the complex power
    // ladder and the normalization constants at f32 - the places where a coefficient
    // that is exact in f64 can stop being exact.
    type F = Vector<f32>;

    fn check<const L: usize, const N: usize>() {
        for &(x, y) in SAMPLES {
            let (xf, yf) = (x as f32, y as f32);

            let mut basis = [F::ZERO; N];
            F::zernike_basis::<L, ZERNIKE_ORTHONORMAL, N>(F::splat(xf), F::splat(yf), &mut basis);

            for (j, &got) in basis.iter().enumerate() {
                let (n, m) = ansi_to_nm(j as u32);

                let rho = xf.hypot(yf);
                let theta = yf.atan2(xf);

                let want = F::splat(rho)
                    .zernike::<ZERNIKE_ORTHONORMAL>(F::splat(theta), n, m)
                    .extract::<0>();

                close(
                    &format!("f32 L={L} j={j} (n={n}, m={m}) at ({xf}, {yf})"),
                    got.extract::<0>() as f64,
                    want as f64,
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
