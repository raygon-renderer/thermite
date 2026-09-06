//! `cbrt` at the extremes of the range.
//!
//! Two guard bugs lived here: the f64 kernel compared its HIGH word against
//! `0x7F800000` (f32's infinity pattern) instead of `0x7FF00000`, so every
//! finite value with exponent >= 1017 was treated as non-finite and returned
//! unchanged. A `>` rather than `>=` in either kernel lets infinity itself
//! fell through to the algorithm and came back NaN.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use thermite::math::policy::policies::{MediumPrecision, Performance, Precision};
use thermite::prelude::*;
use thermite::simd::Simd;

for_each_backend_concrete! {

    /// Non-finite inputs are returned as-is, at every tier.
    fn non_finite_passthrough() {
        for x in [f32::INFINITY, f32::NEG_INFINITY] {
            let v = Vector::<<S as Simd>::f32x8>::splat(x);
            assert_eq!(v.cbrt().extract::<0>(), x, "f32 cbrt({x})");
            assert_eq!(v.cbrt_p::<Precision>().extract::<0>(), x, "f32 cbrt({x}) @Best");
        }
        for x in [f64::INFINITY, f64::NEG_INFINITY] {
            let v = Vector::<<S as Simd>::f64x4>::splat(x);
            assert_eq!(v.cbrt().extract::<0>(), x, "f64 cbrt({x})");
            assert_eq!(v.cbrt_p::<Precision>().extract::<0>(), x, "f64 cbrt({x}) @Best");
        }
        assert!(
            Vector::<<S as Simd>::f32x8>::splat(f32::NAN)
                .cbrt()
                .extract::<0>()
                .is_nan(),
            "f32 cbrt(NaN)"
        );
        assert!(
            Vector::<<S as Simd>::f64x4>::splat(f64::NAN)
                .cbrt()
                .extract::<0>()
                .is_nan(),
            "f64 cbrt(NaN)"
        );
    }

    /// Large FINITE f64 must be cube-rooted, not mistaken for non-finite.
    /// `2^1017` is the exact value the old threshold cut at.
    fn large_finite_f64() {
        for x in [1e300f64, f64::from_bits(0x7F80_0000_0000_0000), 1e307, -1e307] {
            let got = Vector::<<S as Simd>::f64x4>::splat(x).cbrt().extract::<0>();
            let want = libm::cbrt(x);
            let rel = ((got - want) / want).abs();
            assert!(
                rel <= 1e-14,
                "cbrt({x:e}): got {got:e}, want {want:e} (rel {rel:e})"
            );
        }
    }

    /// The top of the range must work at Average (the default) and at
    /// Best. The fast refinement forms `2t^3 + x` ~ 3x, which overflows
    /// above ~MAX/3. Average scales the ratio by 1/4 to avoid it, Best
    /// never forms `t^3` at all.
    fn top_of_range() {
        for x in [1.2e38f32, 3.0e38, f32::MAX, -f32::MAX] {
            let v = Vector::<<S as Simd>::f32x8>::splat(x);
            let want = libm::cbrtf(x);

            for (tier, got) in [
                ("default", v.cbrt().extract::<0>()),
                ("Best", v.cbrt_p::<Precision>().extract::<0>()),
            ] {
                let rel = ((got - want) / want).abs();
                assert!(rel <= 1e-6, "f32 cbrt({x:e}) @{tier}: got {got:e}, want {want:e}");
            }
        }
        for x in [1e308f64, f64::MAX, -f64::MAX] {
            let v = Vector::<<S as Simd>::f64x4>::splat(x);
            let want = libm::cbrt(x);

            for (tier, got) in [
                ("default", v.cbrt().extract::<0>()),
                ("Best", v.cbrt_p::<Precision>().extract::<0>()),
            ] {
                let rel = ((got - want) / want).abs();
                assert!(
                    rel <= 1e-14,
                    "f64 cbrt({x:e}) @{tier}: got {got:e}, want {want:e}"
                );
            }
        }
    }
}

/// Medium and below keep the raw ratio, which gives up the top binade, a
/// deliberate tier trade pinned here so it stays deliberate rather than
/// drifting. Only observable where the fast branch is actually reachable:
/// without true FMA the kernel is forced onto the exact form regardless of
/// precision, so this is an x86_v3-only property.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn medium_gives_up_the_top_binade_on_fma() {
    use thermite::backend::x86_v3::X86V3;

    let got = Vector::<<X86V3 as Simd>::f32x8>::splat(f32::MAX)
        .cbrt_p::<MediumPrecision<Performance>>()
        .extract::<0>();

    // Mode-aware rather than `#[cfg]`-skipped: under `preserve_denormals` the kernel
    // already forces the extended-precision path at every tier (see the `Preserve` arm
    // in `ps.rs::cbrt`), which never forms the ~3x intermediate, so the top binade works
    // and this trade does not exist there. Skipping would hide the assertion, not state
    // it.
    if cfg!(feature = "preserve_denormals") {
        let want = libm::cbrtf(f32::MAX);
        assert!(
            (got / want - 1.0).abs() < 1e-3,
            "cbrt(f32::MAX) @Medium under preserve_denormals takes the extended path; want ~{want:e}, got {got:e}"
        );
    } else {
        assert!(
            got.is_nan(),
            "cbrt(f32::MAX) @Medium is documented as overflowing (the raw ratio forms \
             ~3x); got {got:e}. If this now works, update the tier comments in ps.rs."
        );
    }
}
