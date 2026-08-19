//! The Gaussian shape function `a * exp(-(x/c)^2 / 2)`.
//!
//! Small enough that the risk is not the algorithm but the plumbing: the amplitude and
//! width are vectors, the width appears as a reciprocal under `Worst` precision (a
//! different code path from the division every other policy takes), and the whole thing
//! is one of the few kernels here with three vector arguments that could be transposed.
//!
//! The reference is the defining expression evaluated in scalar f64. That shares the
//! `exp` with the kernel, which is deliberate: the point is the shape and the argument
//! handling, and `exp` itself is covered against libm in `special_vs_libm.rs`.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]

use thermite::math::policy::policies::{Performance, Precision, UltraPerformance};
use thermite::prelude::*;
use thermite_special::{SpecialMath, SpecialMathWithPolicy};

type D = Vector<f64>;
type F = Vector<f32>;

fn want(x: f64, a: f64, c: f64) -> f64 {
    let t = x / c;
    a * (-0.5 * t * t).exp()
}

#[track_caller]
fn close(name: &str, got: f64, w: f64, tol: f64) {
    let err = (got - w).abs() / w.abs().max(f64::MIN_POSITIVE);
    assert!(err <= tol, "{name}: got {got:?}, want {w:?} (rel err {err:e})");
}

const XS: [f64; 11] = [-4.0, -2.5, -1.0, -0.5, -0.125, 0.0, 0.125, 0.5, 1.0, 2.5, 4.0];
const AS: [f64; 4] = [1.0, 0.5, 3.25, -2.0];
const CS: [f64; 5] = [0.25, 0.5, 1.0, 2.0, 7.5];

#[test]
fn matches_the_defining_expression() {
    for &a in &AS {
        for &c in &CS {
            for &x in &XS {
                let got = D::splat(x).gaussian(D::splat(a), D::splat(c)).extract::<0>();
                close(&format!("gaussian(x={x}, a={a}, c={c})"), got, want(x, a, c), 1e-14);
            }
        }
    }
}

#[test]
fn the_peak_is_the_amplitude_exactly() {
    // At x = 0 the exponent is exactly zero and exp(0) is exactly 1, so this should be
    // the amplitude bit for bit, at every width and every policy.
    for &a in &AS {
        for &c in &CS {
            let v = D::ZERO;
            assert_eq!(v.gaussian(D::splat(a), D::splat(c)).extract::<0>(), a, "a={a} c={c}");
            assert_eq!(
                v.gaussian_p::<Precision>(D::splat(a), D::splat(c)).extract::<0>(),
                a,
                "Precision a={a} c={c}"
            );
        }
    }
}

#[test]
fn is_even_in_the_argument() {
    // The kernel squares x/c, so this should hold bit for bit rather than approximately.
    for &a in &AS {
        for &c in &CS {
            for &x in &XS {
                let pos = D::splat(x).gaussian(D::splat(a), D::splat(c)).extract::<0>();
                let neg = D::splat(-x).gaussian(D::splat(a), D::splat(c)).extract::<0>();
                assert_eq!(pos, neg, "symmetry at x={x}, a={a}, c={c}");
            }
        }
    }
}

#[test]
fn the_half_maximum_lands_where_it_should() {
    // x = c * sqrt(2 ln 2) is the half-maximum by construction, an independent handle on
    // the factor of 1/2 in the exponent that a self-consistent reference cannot check.
    let hwhm = (2.0 * 2.0f64.ln()).sqrt();
    for &c in &CS {
        for &a in &AS {
            let got = D::splat(c * hwhm).gaussian(D::splat(a), D::splat(c)).extract::<0>();
            close(&format!("half max at c={c}, a={a}"), got, a * 0.5, 1e-13);
        }
    }
}

#[test]
fn one_sigma_and_two_sigma_match_the_closed_form() {
    // Another fix on the exponent's factor: exp(-1/2) and exp(-2) at x = c and x = 2c.
    for &c in &CS {
        let one = D::splat(c).gaussian(D::ONE, D::splat(c)).extract::<0>();
        let two = D::splat(2.0 * c).gaussian(D::ONE, D::splat(c)).extract::<0>();
        close(&format!("1 sigma at c={c}"), one, (-0.5f64).exp(), 1e-14);
        close(&format!("2 sigma at c={c}"), two, (-2.0f64).exp(), 1e-14);
    }
}

#[test]
fn every_policy_agrees_to_its_own_precision() {
    // `Worst` precision swaps the division by c for a reciprocal-multiply, which is a
    // genuinely different expression and the only policy-dependent branch in here.
    for &a in &AS {
        for &c in &CS {
            for &x in &XS {
                let v = D::splat(x);
                let (av, cv) = (D::splat(a), D::splat(c));
                let w = want(x, a, c);

                close(
                    &format!("Precision x={x} c={c}"),
                    v.gaussian_p::<Precision>(av, cv).extract::<0>(),
                    w,
                    1e-14,
                );
                close(
                    &format!("Performance x={x} c={c}"),
                    v.gaussian_p::<Performance>(av, cv).extract::<0>(),
                    w,
                    1e-10,
                );
                // The reciprocal path is an approximation by design, so this is a sanity
                // bound on it, not a precision claim.
                close(
                    &format!("Ultra x={x} c={c}"),
                    v.gaussian_p::<UltraPerformance>(av, cv).extract::<0>(),
                    w,
                    1e-2,
                );
            }
        }
    }
}

#[test]
fn far_tails_underflow_to_zero_rather_than_misbehaving() {
    for &c in &CS {
        let far = D::splat(c * 100.0).gaussian(D::ONE, D::splat(c)).extract::<0>();
        assert!(
            (0.0..1e-300).contains(&far),
            "far tail at c={c} should vanish, got {far}"
        );
        assert!(far.is_finite(), "far tail at c={c} must stay finite, got {far}");
    }
}

#[test]
fn amplitude_width_and_argument_can_all_vary_per_lane() {
    // Three vector arguments in one call, where a transposition between them would still give
    // plausible values in any test that splats them together.
    use thermite::backend::scalar::Scalar;
    type D4 = thermite::simd::f64x4<Scalar>;

    let xs = [0.5, -1.25, 2.0, 0.0];
    let as_ = [1.0, 2.5, -0.75, 4.0];
    let cs = [1.0, 0.5, 3.0, 0.25];

    let got = D4::new(xs).gaussian(D4::new(as_), D4::new(cs));
    for lane in 0..4 {
        close(
            &format!("lane {lane}"),
            got.as_slice()[lane],
            want(xs[lane], as_[lane], cs[lane]),
            1e-14,
        );
    }
}

#[test]
fn f32_tracks_the_same_shape() {
    for &a in &AS {
        for &c in &CS {
            for &x in &XS {
                let got = F::splat(x as f32)
                    .gaussian(F::splat(a as f32), F::splat(c as f32))
                    .extract::<0>();
                let w = want(x, a, c);
                if w.abs() < 1e-30 {
                    continue; // below f32's range, nothing to compare
                }
                close(&format!("f32 x={x} a={a} c={c}"), got as f64, w, 1e-5);
            }
        }
    }
}
