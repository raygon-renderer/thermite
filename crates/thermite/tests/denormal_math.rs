//! Denormal inputs through the math kernels, under `PreserveDenormals`.
//!
//! Everything with `f(x) = x + O(x^3)` near zero must return a denormal input
//! EXACTLY: the correction term sits far below the denormal ULP (2^-149 for
//! f32), so the answer is the input. That makes this family self-checking and a
//! cheap, strong exercise of the subnormal paths.
//!
//! `cbrt` is the interesting counter-case, since it leaves the denormal range, and
//! its fast refinement cubes the root, which for a denormal input lands back in
//! the denormal range with ~1 significant bit. Preserving denormals therefore
//! routes it to the extended-precision refinement.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use thermite::math::policy::policies::{AveragePrecision, Performance, PreserveDenormals};
use thermite::prelude::*;
use thermite::simd::Simd;

/// Preserve denormals at the DEFAULT precision, the tier `cbrt` is most exposed
/// on, so the tests below catch a regression in the policy gate.
type Preserve = AveragePrecision<PreserveDenormals<Performance>>;

/// Subnormals spanning the f32 range, both signs.
fn f32_denormals() -> Vec<f32> {
    let mut v = vec![
        f32::from_bits(0x0000_0001), // smallest
        f32::from_bits(0x0000_0002),
        f32::from_bits(0x0000_0100),
        f32::from_bits(0x0000_FFFF),
        f32::from_bits(0x0040_0000),
        f32::from_bits(0x007F_FFFF), // largest
    ];
    for k in 0..23 {
        v.push(f32::from_bits(1 << k));
    }
    let neg: Vec<f32> = v.iter().map(|x| -x).collect();
    v.extend(neg);
    v
}

fn f64_denormals() -> Vec<f64> {
    vec![
        f64::from_bits(0x0000_0000_0000_0001),
        f64::from_bits(0x0000_0000_0001_0000),
        f64::from_bits(0x0008_0000_0000_0000),
        f64::from_bits(0x000F_FFFF_FFFF_FFFF),
        -f64::from_bits(0x0000_0000_0000_0001),
        -f64::from_bits(0x000F_FFFF_FFFF_FFFF),
    ]
}

/// `f(x) == x` bit-for-bit, for every denormal.
macro_rules! identity_family {
    ($b:ty, $($m:ident),+ $(,)?) => {$(
        for x in f32_denormals() {
            let got = Vector::<<S as Simd>::f32x8>::splat(x).$m::<Preserve>().extract::<0>();
            assert_eq!(
                got.to_bits(),
                x.to_bits(),
                concat!(stringify!($m), "({:e}) should be exactly the input, got {:e}"),
                x,
                got
            );
        }
    )+};
}

for_each_backend_concrete! {

    /// `f(x) = x + O(x^3)`: a denormal must round-trip unchanged.
    fn near_identity_preserves_denormals() {
        identity_family!(
            S, sin_p, tan_p, asin_p, atan_p, sinh_p, tanh_p, asinh_p, atanh_p, ln_1p_p, exp_m1_p,
        );
    }

    /// `cbrt` leaves the denormal range, so check it stays accurate there.
    /// The fast refinement cubes the root back into the denormal range,
    /// so `Preserve` must route to the extended-precision form.
    fn cbrt_denormals_f32() {
        for x in f32_denormals() {
            let got = Vector::<<S as Simd>::f32x8>::splat(x)
                .cbrt_p::<Preserve>()
                .extract::<0>();
            let want = libm::cbrtf(x);

            let rel = ((got - want) / want).abs();
            assert!(
                rel <= 1e-6,
                "cbrt({x:e}): got {got:e}, want {want:e} (rel err {rel:e})"
            );
        }
    }

    fn cbrt_denormals_f64() {
        for x in f64_denormals() {
            let got = Vector::<<S as Simd>::f64x4>::splat(x)
                .cbrt_p::<Preserve>()
                .extract::<0>();
            let want = libm::cbrt(x);

            let rel = ((got - want) / want).abs();
            assert!(
                rel <= 1e-14,
                "cbrt({x:e}): got {got:e}, want {want:e} (rel err {rel:e})"
            );
        }
    }

    /// `ln` leaves the denormal range downward. A subnormal has no exponent
    /// field for the reduction to split, so `Preserve` must rescale it first
    /// - without that every denormal comes back `-inf`, which is only the
    /// right answer under the flushing tiers.
    fn ln_denormals_f32() {
        for x in f32_denormals().into_iter().filter(|x| *x > 0.0) {
            let got = Vector::<<S as Simd>::f32x8>::splat(x)
                .ln_p::<Preserve>()
                .extract::<0>();
            let want = libm::logf(x);

            let rel = ((got - want) / want).abs();
            assert!(
                got.is_finite() && rel <= 1e-6,
                "ln({x:e}): got {got:e}, want {want:e} (rel err {rel:e})"
            );
        }
    }

    fn ln_denormals_f64() {
        for x in f64_denormals().into_iter().filter(|x| *x > 0.0) {
            let got = Vector::<<S as Simd>::f64x4>::splat(x)
                .ln_p::<Preserve>()
                .extract::<0>();
            let want = libm::log(x);

            let rel = ((got - want) / want).abs();
            assert!(
                got.is_finite() && rel <= 1e-14,
                "ln({x:e}): got {got:e}, want {want:e} (rel err {rel:e})"
            );
        }
    }

    /// Zero is still `-inf`, and a negative denormal is still NaN: the
    /// rescale must not swallow the edge cases the tail hands out.
    fn ln_zero_and_negative_denormals() {
        let zero = Vector::<<S as Simd>::f64x4>::ZERO
            .ln_p::<Preserve>()
            .extract::<0>();
        assert_eq!(zero, f64::NEG_INFINITY, "ln(0) under Preserve");

        for x in f64_denormals().into_iter().filter(|x| *x < 0.0) {
            let got = Vector::<<S as Simd>::f64x4>::splat(x)
                .ln_p::<Preserve>()
                .extract::<0>();
            assert!(got.is_nan(), "ln({x:e}) should be NaN, got {got:e}");
        }
    }

    /// The flushing tiers are untouched: a denormal is a zero there, so
    /// `-inf` is what they should keep returning.
    ///
    /// **Mode-aware, not `#[cfg]`-skipped.** The `preserve_denormals` feature
    /// flips the default `denormal_behavior`, so `Performance` stops flushing and
    /// this assertion inverts. Written as a `cfg!` switch rather than a
    /// `#[cfg(not(...))]` on the test, so `--features preserve_denormals` is a
    /// clean run instead of a run with silent holes in it, the same reason
    /// `exp_range::powf_of_a_subnormal_base` is written that way.
    fn ln_denormals_still_flush_by_default() {
        for x in f64_denormals().into_iter().filter(|x| *x > 0.0) {
            let got = Vector::<<S as Simd>::f64x4>::splat(x)
                .ln_p::<Performance>()
                .extract::<0>();

            if cfg!(feature = "preserve_denormals") {
                // Preserved: the true log of a denormal, near -708 to -745.
                let want = libm::log(x);
                assert!(
                    got.is_finite() && (got - want).abs() <= 1e-9 * want.abs(),
                    "ln({x:e}) under Performance with preserve_denormals: got {got:e}, want {want:e}"
                );
            } else {
                assert_eq!(got, f64::NEG_INFINITY, "ln({x:e}) under Performance");
            }
        }
    }

    /// `sqrt` also leaves the denormal range, and is exact there.
    fn sqrt_denormals() {
        for x in f32_denormals().into_iter().filter(|x| *x > 0.0) {
            let got = Vector::<<S as Simd>::f32x8>::splat(x).sqrt().extract::<0>();
            assert_eq!(got.to_bits(), libm::sqrtf(x).to_bits(), "sqrt({x:e})");
        }
    }
}
