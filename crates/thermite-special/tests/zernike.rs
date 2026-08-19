//! Zernike polynomials: the radial kernel, the two normalizations, and the three
//! single-index conventions.
//!
//! Radial reference values are the *direct* factorial sum
//! `$\sum_k (-1)^k \frac{(n-k)!}{k!\,((n+m)/2-k)!\,((n-m)/2-k)!}\rho^{n-2k}$` evaluated in
//! exact rational arithmetic (Python `fractions`) and rounded once. That makes the
//! reference genuinely independent of the shifted-Jacobi route the kernel takes: the two
//! agree exactly as rationals for every mode through `n = 14`, so any disagreement in
//! floating point is the kernel's.
//!
//! The orthonormality test is the one that ties everything together at once: radial
//! shape, angular factor, and the `$N_n^m$` scale all have to be right simultaneously for
//! the Gram matrix to come out as the identity.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]
#![allow(clippy::excessive_precision)]

use std::f64::consts::PI;

use thermite::Vector;
use thermite::prelude::*;
use thermite_special::zernike::{
    ansi_index, ansi_to_nm, count_up_to_degree, fringe_index, fringe_to_nm, is_valid, noll_index, noll_to_nm,
};
use thermite_special::{SpecialMath, ZERNIKE_ORTHONORMAL, ZERNIKE_UNIT_PEAK};

type D = Vector<f64>;

#[track_caller]
fn close(name: &str, got: f64, want: f64, tol: f64) {
    let err = (got - want).abs() / want.abs().max(1.0);
    assert!(err <= tol, "{name}: got {got:?}, want {want:?} (err {err:e})");
}

fn radial(rho: f64, n: u32, m: u32) -> f64 {
    D::splat(rho).zernike_r(n, m).extract::<0>()
}

fn peak(rho: f64, theta: f64, n: u32, m: i32) -> f64 {
    D::splat(rho)
        .zernike::<ZERNIKE_UNIT_PEAK>(D::splat(theta), n, m)
        .extract::<0>()
}

fn ortho(rho: f64, theta: f64, n: u32, m: i32) -> f64 {
    D::splat(rho)
        .zernike::<ZERNIKE_ORTHONORMAL>(D::splat(theta), n, m)
        .extract::<0>()
}

// --- Radial polynomial ---

#[test]
fn zernike_r_matches_the_exact_factorial_sum() {
    // (n, m, rho, R_n^m(rho)); rho spans both dyadic values (exact in binary) and 0.9,
    // which is not, so a rounding-sensitive path cannot hide in exactly-representable
    // arguments.
    let probes: &[(u32, u32, f64, f64)] = &[
        (0, 0, 0.25, 1.0),
        (1, 1, 0.25, 2.50000000000000000e-01),
        (1, 1, 0.9, 9.00000000000000022e-01),
        (2, 0, 0.25, -8.75000000000000000e-01),
        (2, 0, 0.5, -5.00000000000000000e-01),
        (2, 0, 0.9, 6.19999999999999996e-01),
        (2, 2, 0.75, 5.62500000000000000e-01),
        (3, 1, 0.25, -4.53125000000000000e-01),
        (3, 1, 0.5, -6.25000000000000000e-01),
        (3, 1, 0.9, 3.87000000000000011e-01),
        (3, 3, 0.75, 4.21875000000000000e-01),
        (4, 0, 0.25, 6.48437500000000000e-01),
        (4, 0, 0.75, -4.76562500000000000e-01),
        (4, 0, 0.9, 7.66000000000000014e-02),
        (4, 2, 0.5, -5.00000000000000000e-01),
        (4, 2, 0.9, 1.94399999999999989e-01),
        (4, 4, 0.75, 3.16406250000000000e-01),
        (5, 1, 0.25, 5.72265625000000000e-01),
        (5, 1, 0.75, -4.39453125000000000e-01),
        (5, 1, 0.9, -1.43100000000000005e-01),
        (6, 0, 0.5, 4.37500000000000000e-01),
        (6, 0, 0.9, -3.34179999999999977e-01),
        (8, 4, 0.25, 4.87670898437500000e-02),
        (8, 4, 0.75, 7.41577148437500000e-02),
        (8, 4, 0.9, -4.25940119999999978e-01),
        (10, 2, 0.25, 4.85673904418945312e-01),
        (10, 2, 0.75, 2.60530471801757812e-01),
        (10, 2, 0.9, -2.31781418999999989e-01),
        (12, 0, 0.25, -3.49054574966430664e-01),
        (12, 0, 0.75, -2.14712381362915039e-01),
        (12, 0, 0.9, 1.21087251243999994e-01),
        (14, 6, 0.25, 3.37936021387577057e-02),
        (14, 6, 0.75, -2.81021710485219955e-01),
        (14, 6, 0.9, 2.09078035499609988e-01),
    ];

    for &(n, m, rho, want) in probes {
        close(&format!("R_{n}^{m}({rho})"), radial(rho, n, m), want, 1e-14);
    }
}

#[test]
fn zernike_r_is_unity_at_the_rim() {
    // R_n^m(1) = 1 for every valid mode, by construction. A structural check that
    // reaches degrees no value table covers, and the property that makes the unit-peak
    // normalization mean what it says.
    for n in 0..=20 {
        for m in 0..=n {
            if !is_valid(n, m as i32) {
                continue;
            }

            close(&format!("R_{n}^{m}(1)"), radial(1.0, n, m), 1.0, 1e-12);
        }
    }
}

#[test]
fn zernike_r_low_order_closed_forms() {
    for &rho in &[0.0, 0.125, 0.5, 0.875, 1.0] {
        let r2 = rho * rho;

        close("R_0^0", radial(rho, 0, 0), 1.0, 0.0);
        close("R_1^1", radial(rho, 1, 1), rho, 1e-15);
        close("R_2^0", radial(rho, 2, 0), 2.0 * r2 - 1.0, 1e-15);
        close("R_2^2", radial(rho, 2, 2), r2, 1e-15);
        close("R_3^1", radial(rho, 3, 1), 3.0 * rho * r2 - 2.0 * rho, 1e-15);
        close("R_3^3", radial(rho, 3, 3), rho * r2, 1e-15);
        close("R_4^0", radial(rho, 4, 0), 6.0 * r2 * r2 - 6.0 * r2 + 1.0, 1e-15);
        close("R_4^2", radial(rho, 4, 2), 4.0 * r2 * r2 - 3.0 * r2, 1e-15);
    }
}

#[test]
fn nonexistent_modes_are_zero() {
    // |m| > n, and n - |m| odd. Both are outside the family rather than merely awkward,
    // so the answer is zero and not whatever the recurrence would produce.
    for &(n, m) in &[(0u32, 1u32), (1, 2), (2, 3), (4, 7), (2, 1), (3, 0), (4, 1), (5, 2)] {
        assert_eq!(radial(0.7, n, m), 0.0, "R_{n}^{m} should be zero");
        assert!(!is_valid(n, m as i32), "is_valid disagrees for ({n}, {m})");
    }

    for &(n, m) in &[(1i32, 2i32), (2, -3), (3, 0), (4, -1)] {
        assert_eq!(peak(0.7, 0.4, n as u32, m), 0.0, "Z_{n}^{m} should be zero");
        assert_eq!(ortho(0.7, 0.4, n as u32, m), 0.0, "normalized Z_{n}^{m} should be zero");
    }
}

// --- Angular factor and normalization ---

#[test]
fn zernike_splits_into_radial_and_angular_parts() {
    for &(n, m) in &[
        (1i32, 1i32),
        (1, -1),
        (2, 0),
        (2, 2),
        (2, -2),
        (3, 3),
        (3, -1),
        (5, -5),
        (6, 2),
    ] {
        let n = n as u32;
        let am = m.unsigned_abs();

        for &rho in &[0.0, 0.3, 0.75, 1.0] {
            for &theta in &[0.0, 0.4, 1.7, PI, 4.9, 2.0 * PI - 0.1] {
                let angular = if m >= 0 {
                    (am as f64 * theta).cos()
                } else {
                    (am as f64 * theta).sin()
                };

                close(
                    &format!("Z_{n}^{m}({rho}, {theta})"),
                    peak(rho, theta, n, m),
                    radial(rho, n, am) * angular,
                    1e-14,
                );
            }
        }
    }
}

#[test]
fn orthonormal_scales_by_the_ansi_factor() {
    // N_n^m = sqrt(2(n+1)/(1 + delta_{m,0})), i.e. sqrt(n+1) on the rotationally
    // symmetric modes and sqrt(2(n+1)) on the rest.
    for &(n, m) in &[
        (0i32, 0i32),
        (1, 1),
        (1, -1),
        (2, 0),
        (2, -2),
        (4, 0),
        (4, 4),
        (7, 3),
        (8, 0),
    ] {
        let n = n as u32;

        let want_factor = if m == 0 {
            ((n + 1) as f64).sqrt()
        } else {
            (2.0 * (n + 1) as f64).sqrt()
        };

        for &(rho, theta) in &[(0.4, 0.9), (0.85, 2.6), (1.0, 0.0)] {
            let unit = peak(rho, theta, n, m);
            close(
                &format!("N_{n}^{m}"),
                ortho(rho, theta, n, m),
                unit * want_factor,
                1e-14,
            );
        }
    }
}

#[test]
fn orthonormal_modes_have_an_identity_gram_matrix() {
    // The test that constrains radial shape, angular factor and scale simultaneously:
    //
    //     (1/pi) \int_0^{2pi} \int_0^1 Z_a Z_b rho drho dtheta = delta_ab
    //
    // Substituting u = rho^2 turns that into a plain average over uniform (u, theta),
    // which the midpoint rule handles well: exactly in theta, since the integrand is a
    // trigonometric polynomial over a full period, and to O(h^2) in u.
    const N_U: usize = 2048;
    const N_T: usize = 64;
    const MAX_DEGREE: u32 = 4;

    let modes: Vec<(u32, i32)> = (0..count_up_to_degree(MAX_DEGREE)).map(ansi_to_nm).collect();
    let count = modes.len();

    let mut gram = vec![0.0f64; count * count];

    for iu in 0..N_U {
        let u = (iu as f64 + 0.5) / N_U as f64;
        let rho = u.sqrt();

        for it in 0..N_T {
            let theta = 2.0 * PI * (it as f64 + 0.5) / N_T as f64;

            let values: Vec<f64> = modes.iter().map(|&(n, m)| ortho(rho, theta, n, m)).collect();

            for a in 0..count {
                for b in 0..count {
                    gram[a * count + b] += values[a] * values[b];
                }
            }
        }
    }

    let scale = 1.0 / (N_U * N_T) as f64;

    for a in 0..count {
        for b in 0..count {
            let got = gram[a * count + b] * scale;
            let want = if a == b { 1.0 } else { 0.0 };

            assert!(
                (got - want).abs() < 2e-4,
                "gram[{:?}][{:?}] = {got}, want {want}",
                modes[a],
                modes[b]
            );
        }
    }
}

// --- Index conventions ---

#[test]
fn ansi_indices_match_the_published_ordering() {
    // ANSI Z80.28 / OSA: zero-based, m ascending within each degree.
    let table: &[(u32, i32)] = &[
        (0, 0),
        (1, -1),
        (1, 1),
        (2, -2),
        (2, 0),
        (2, 2),
        (3, -3),
        (3, -1),
        (3, 1),
        (3, 3),
        (4, -4),
        (4, -2),
        (4, 0),
        (4, 2),
        (4, 4),
    ];

    for (j, &(n, m)) in table.iter().enumerate() {
        assert_eq!(ansi_index(n, m), j as u32, "ansi_index({n}, {m})");
        assert_eq!(ansi_to_nm(j as u32), (n, m), "ansi_to_nm({j})");
    }
}

#[test]
fn noll_indices_match_the_published_ordering() {
    // Noll: one-based, |m| ascending, and the sign of the leading member of each +-m
    // pair flipping with n mod 4. Rows 5/6 and 12/13 are where a naive "cosine always
    // first" rule diverges from the standard.
    let table: &[(u32, u32, i32)] = &[
        (1, 0, 0),
        (2, 1, 1),
        (3, 1, -1),
        (4, 2, 0),
        (5, 2, -2),
        (6, 2, 2),
        (7, 3, -1),
        (8, 3, 1),
        (9, 3, -3),
        (10, 3, 3),
        (11, 4, 0),
        (12, 4, 2),
        (13, 4, -2),
        (14, 4, 4),
        (15, 4, -4),
    ];

    for &(j, n, m) in table {
        assert_eq!(noll_index(n, m), j, "noll_index({n}, {m})");
        assert_eq!(noll_to_nm(j), (n, m), "noll_to_nm({j})");
    }
}

#[test]
fn fringe_indices_match_the_published_ordering() {
    // Fringe / Air Force: one-based, ordered by spatial frequency n + |m|. Index 9 is
    // primary spherical (4, 0) while (3, 3) is only 10 - the reordering that makes a
    // 37-term Fringe truncation different from a 37-term ANSI one.
    let table: &[(u32, u32, i32)] = &[
        (1, 0, 0),
        (2, 1, 1),
        (3, 1, -1),
        (4, 2, 0),
        (5, 2, 2),
        (6, 2, -2),
        (7, 3, 1),
        (8, 3, -1),
        (9, 4, 0),
        (10, 3, 3),
        (11, 3, -3),
        (12, 4, 2),
        (13, 4, -2),
        (14, 5, 1),
        (15, 5, -1),
        (16, 6, 0),
    ];

    for &(j, n, m) in table {
        assert_eq!(fringe_index(n, m), j, "fringe_index({n}, {m})");
        assert_eq!(fringe_to_nm(j), (n, m), "fringe_to_nm({j})");
    }
}

/// Every valid mode with radial degree at most `max_degree`, in ANSI order.
fn modes_to_degree(max_degree: u32) -> Vec<(u32, i32)> {
    (0..=max_degree)
        .flat_map(|n| (-(n as i32)..=(n as i32)).map(move |m| (n, m)))
        .filter(|&(n, m)| is_valid(n, m))
        .collect()
}

#[test]
fn every_scheme_round_trips() {
    for (n, m) in modes_to_degree(20) {
        assert_eq!(ansi_to_nm(ansi_index(n, m)), (n, m), "ansi round trip for ({n}, {m})");
        assert_eq!(noll_to_nm(noll_index(n, m)), (n, m), "noll round trip for ({n}, {m})");
        assert_eq!(
            fringe_to_nm(fringe_index(n, m)),
            (n, m),
            "fringe round trip for ({n}, {m})"
        );
    }
}

#[test]
fn degree_ordered_schemes_number_a_degree_truncation_contiguously() {
    // ANSI and Noll both order by radial degree, so the modes with n <= N are exactly
    // the first (N+1)(N+2)/2 indices, with no gaps and no repeats. That is what makes a
    // degree truncation and an index truncation the same set in those two schemes.
    const MAX_DEGREE: u32 = 20;

    let modes = modes_to_degree(MAX_DEGREE);
    let count = count_up_to_degree(MAX_DEGREE) as usize;
    assert_eq!(modes.len(), count);

    let mut ansi_seen = vec![false; count];
    let mut noll_seen = vec![false; count];

    for &(n, m) in &modes {
        for (index, seen, name) in [
            (ansi_index(n, m) as usize, &mut ansi_seen, "ansi"),
            (noll_index(n, m) as usize - 1, &mut noll_seen, "noll"),
        ] {
            assert!(index < count, "{name} index {index} out of range for ({n}, {m})");
            assert!(!seen[index], "{name} index {index} claimed twice, at ({n}, {m})");

            seen[index] = true;
        }
    }
}

#[test]
fn fringe_numbers_a_spatial_frequency_truncation_contiguously() {
    // Fringe orders by spatial frequency n + |m| instead, so a *degree* truncation is
    // full of holes under it - (16, -16) lands at 258, well past the 231 modes of degree
    // 20. Its natural blocks are the frequency groups q = (n + |m|)/2, and the modes with
    // q <= Q are exactly indices 1..=(Q+1)^2.
    //
    // This is the concrete reason a 37-term Fringe set is not a 37-term ANSI set, and
    // why swapping the two silently reassigns every coefficient past the third.
    const MAX_GROUP: u32 = 10;

    let count = ((MAX_GROUP + 1) * (MAX_GROUP + 1)) as usize;

    let modes: Vec<(u32, i32)> = modes_to_degree(2 * MAX_GROUP)
        .into_iter()
        .filter(|&(n, m)| (n + m.unsigned_abs()) / 2 <= MAX_GROUP)
        .collect();

    assert_eq!(modes.len(), count);

    let mut seen = vec![false; count];

    for &(n, m) in &modes {
        let index = fringe_index(n, m) as usize - 1;

        assert!(index < count, "fringe index {index} out of range for ({n}, {m})");
        assert!(!seen[index], "fringe index {index} claimed twice, at ({n}, {m})");

        seen[index] = true;
    }
}

#[test]
fn indices_are_usable_in_const_context() {
    // These exist to be evaluated once at configuration time, often to lay out a static
    // coefficient table, so they have to hold up as `const fn` rather than merely be
    // cheap at runtime.
    const DEFOCUS_NOLL: u32 = noll_index(2, 0);
    const SPHERICAL_FRINGE: u32 = fringe_index(4, 0);
    const COMA_ANSI: u32 = ansi_index(3, 1);
    const MODES_TO_6: u32 = count_up_to_degree(6);

    assert_eq!(DEFOCUS_NOLL, 4);
    assert_eq!(SPHERICAL_FRINGE, 9);
    assert_eq!(COMA_ANSI, 8);
    assert_eq!(MODES_TO_6, 28);
}

// --- Composite inheritance ---

#[test]
fn zernike_differentiates_through_dual() {
    use thermite_dual::Dual;

    // dR/drho for the low-order modes, against closed forms. `zernike_r` is a default
    // method over vector arithmetic, so `Dual` inherits it and the slope comes out of
    // the same code that produced the value.
    type DD = Dual<D, 1>;

    for &rho in &[0.2, 0.55, 0.9] {
        let r = DD::variable(D::splat(rho), 0);

        let r2 = rho * rho;

        // R_2^0 = 2rho^2 - 1
        let d = r.zernike_r(2, 0);
        close("R_2^0 value", d.value().extract::<0>(), 2.0 * r2 - 1.0, 1e-14);
        close("R_2^0 slope", d.gradient()[0].extract::<0>(), 4.0 * rho, 1e-13);

        // R_3^1 = 3rho^3 - 2rho
        let d = r.zernike_r(3, 1);
        close(
            "R_3^1 value",
            d.value().extract::<0>(),
            3.0 * rho * r2 - 2.0 * rho,
            1e-14,
        );
        close("R_3^1 slope", d.gradient()[0].extract::<0>(), 9.0 * r2 - 2.0, 1e-13);

        // R_4^0 = 6rho^4 - 6rho^2 + 1
        let d = r.zernike_r(4, 0);
        close(
            "R_4^0 value",
            d.value().extract::<0>(),
            6.0 * r2 * r2 - 6.0 * r2 + 1.0,
            1e-14,
        );
        close(
            "R_4^0 slope",
            d.gradient()[0].extract::<0>(),
            24.0 * rho * r2 - 12.0 * rho,
            1e-13,
        );
    }
}

// --- f32 ---

#[test]
fn zernike_r_f32_tracks_the_reference() {
    // The same exact-rational references as the f64 table. f32 carries ~7 digits, and the
    // recurrence is run at the working precision throughout, so the bound is a few ulp
    // rather than the 1e-14 the f64 path holds.
    type F = Vector<f32>;

    let probes: &[(u32, u32, f32, f64)] = &[
        (2, 0, 0.25, -8.75000000000000000e-01),
        (3, 1, 0.5, -6.25000000000000000e-01),
        (4, 0, 0.75, -4.76562500000000000e-01),
        (4, 2, 0.9, 1.94399999999999989e-01),
        (5, 1, 0.25, 5.72265625000000000e-01),
        (6, 0, 0.9, -3.34179999999999977e-01),
        (8, 4, 0.75, 7.41577148437500000e-02),
        (10, 2, 0.25, 4.85673904418945312e-01),
        (12, 0, 0.75, -2.14712381362915039e-01),
        (14, 6, 0.9, 2.09078035499609988e-01),
    ];

    for &(n, m, rho, want) in probes {
        let got = F::splat(rho).zernike_r(n, m).extract::<0>();
        close(&format!("f32 R_{n}^{m}({rho})"), got as f64, want, 2e-6);
    }

    // The rim identity holds in f32 too, and it is the one that would expose a
    // coefficient that only rounds badly in single precision.
    for n in 0..=14u32 {
        for m in 0..=n {
            if !is_valid(n, m as i32) {
                continue;
            }

            let got = F::splat(1.0).zernike_r(n, m).extract::<0>();
            close(&format!("f32 R_{n}^{m}(1)"), got as f64, 1.0, 5e-6);
        }
    }
}

#[test]
fn zernike_f32_matches_the_f64_path() {
    type F = Vector<f32>;

    for &(n, m) in &[(1i32, 1i32), (2, 0), (2, -2), (3, 3), (4, 2), (6, -4), (9, 1)] {
        let n = n as u32;

        for &(rho, theta) in &[(0.25f32, 0.4f32), (0.75, 2.6), (1.0, 5.1), (0.5, -1.2)] {
            let got = F::splat(rho)
                .zernike::<ZERNIKE_ORTHONORMAL>(F::splat(theta), n, m)
                .extract::<0>();

            let want = ortho(rho as f64, theta as f64, n, m);

            close(&format!("f32 Z_{n}^{m}({rho}, {theta})"), got as f64, want, 5e-6);
        }
    }
}
