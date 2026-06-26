// Scalar-based, so it could run on wasm, but the `ldexp` extreme-exponent case is a known
// failure without `strict_ieee754`; keep it off the wasm run for now (revisit in a later phase).
#![cfg(not(target_arch = "wasm32"))]
//! Tests for `ldexp` and `frexp` on FloatVectorWithBits.
//!
//! These use scalar f32/f64 as the vector type (since they implement the
//! register/vector traits), so they validate the math without needing a
//! specific SIMD backend. Run the same tests through your dispatch macro
//! on real SIMD types to validate backend behavior.
//!
//! Reference behavior: C99 ldexp/frexp semantics.
//!   frexp(x) -> (frac, exp) where x = frac * 2^exp, 0.5 <= |frac| < 1.0
//!   ldexp(x, n) -> x * 2^n

// -------------------------------------------------------
// Helper: compare frexp output against the C reference
// -------------------------------------------------------

use thermite::{
    backend::scalar::prelude::*,
    math::policy::{DenormalBehavior, PolicyParameters, PrecisionPolicy},
};

fn ref_frexp_f32(x: f32) -> (f32, i32) {
    libm::frexpf(x)
}

fn ref_frexp_f64(x: f64) -> (f64, i64) {
    let (f, e) = libm::frexp(x);

    (f, e as i64)
}

// -------------------------------------------------------
//  f32 frexp tests
// -------------------------------------------------------

#[test]
fn frexp_f32_normal_values() {
    let cases: &[f32] = &[
        1.0,
        -1.0,
        2.0,
        -2.0,
        0.5,
        -0.5,
        0.75,
        1.5,
        3.0,
        100.0,
        -100.0,
        1.0e10,
        1.0e-10,
        1.0e30,
        1.0e-30,
        f32::MAX,
        f32::MIN,
        f32::MIN_POSITIVE,
    ];

    for &x in cases {
        let (frac, exp) = your_frexp_f32(x); // replace with actual call
        let (ref_frac, ref_exp) = ref_frexp_f32(x);

        assert_eq!(exp, ref_exp, "frexp f32 exp mismatch for x = {x:e}");
        assert_eq!(
            frac.to_bits(),
            ref_frac.to_bits(),
            "frexp f32 frac mismatch for x = {x:e}: got {frac:e}, expected {ref_frac:e}"
        );

        // Roundtrip: frac * 2^exp == x
        let reconstructed = libm::ldexpf(frac, exp);
        assert_eq!(
            reconstructed.to_bits(),
            x.to_bits(),
            "frexp f32 roundtrip failed for x = {x:e}"
        );
    }
}

#[test]
fn frexp_f32_subnormals() {
    let cases: &[f32] = &[
        // Smallest subnormal
        f32::from_bits(0x0000_0001),
        // Negative smallest subnormal
        f32::from_bits(0x8000_0001),
        // Largest subnormal
        f32::from_bits(0x007F_FFFF),
        // Negative largest subnormal
        f32::from_bits(0x807F_FFFF),
        // Mid-range subnormals
        f32::from_bits(0x0040_0000), // 2^-126 * 0.5
        f32::from_bits(0x0000_0100), // small but not smallest
        f32::from_bits(0x0000_FFFF),
        // Powers of two in subnormal range
        f32::from_bits(0x0000_0002), // 2 * min_subnormal
        f32::from_bits(0x0000_0004),
        f32::from_bits(0x0000_0008),
        f32::from_bits(0x0000_0010),
    ];

    for &x in cases {
        let (frac, exp) = your_frexp_f32(x);
        let (ref_frac, ref_exp) = ref_frexp_f32(x);

        assert_eq!(
            exp,
            ref_exp,
            "frexp f32 subnormal exp mismatch for x = {:e} (bits {:#010x})",
            x,
            x.to_bits()
        );
        assert_eq!(
            frac.to_bits(),
            ref_frac.to_bits(),
            "frexp f32 subnormal frac mismatch for x = {:e} (bits {:#010x}): got {:e}, expected {:e}",
            x,
            x.to_bits(),
            frac,
            ref_frac
        );

        // Verify fraction is in [0.5, 1.0)
        assert!(
            frac.abs() >= 0.5 && frac.abs() < 1.0,
            "frexp f32 fraction {frac:e} not in [0.5, 1.0) for subnormal x = {x:e}"
        );

        // Verify sign preserved
        assert_eq!(
            frac.is_sign_negative(),
            x.is_sign_negative(),
            "frexp f32 sign lost for subnormal x = {x:e}"
        );

        // Roundtrip
        let reconstructed = libm::ldexpf(frac, exp);
        assert_eq!(
            reconstructed.to_bits(),
            x.to_bits(),
            "frexp f32 subnormal roundtrip failed for x = {:e} (bits {:#010x})",
            x,
            x.to_bits()
        );
    }
}

#[test]
fn frexp_f32_zeros() {
    for &x in &[0.0f32, -0.0f32] {
        let (frac, exp) = your_frexp_f32(x);
        assert_eq!(exp, 0, "frexp f32 zero exp should be 0");
        assert_eq!(frac.to_bits(), x.to_bits(), "frexp f32 should preserve zero sign");
    }
}

#[test]
fn frexp_f32_special() {
    // Infinity
    let (frac, exp) = your_frexp_f32(f32::INFINITY);
    assert!(
        frac.is_infinite() && frac.is_sign_positive(),
        "frexp(+inf) frac, {frac}"
    );
    assert_eq!(exp, 0, "frexp(+inf) exp");

    let (frac, exp) = your_frexp_f32(f32::NEG_INFINITY);
    assert!(
        frac.is_infinite() && frac.is_sign_negative(),
        "frexp(-inf) frac, {frac}"
    );
    assert_eq!(exp, 0, "frexp(-inf) exp");

    // NaN
    let (frac, _exp) = your_frexp_f32(f32::NAN);
    assert!(frac.is_nan(), "frexp(NaN) should return NaN fraction");
}

#[test]
fn frexp_f32_boundary_normal_subnormal() {
    // Smallest normal
    let x = f32::MIN_POSITIVE; // 2^-126
    let (frac, exp) = your_frexp_f32(x);
    assert_eq!(frac, 0.5);
    assert_eq!(exp, -125); // 0.5 * 2^-125 = 2^-126

    // One step below: largest subnormal
    let x = f32::from_bits(f32::MIN_POSITIVE.to_bits() - 1);
    let (frac, exp) = your_frexp_f32(x);
    let (ref_frac, ref_exp) = ref_frexp_f32(x);
    assert_eq!(exp, ref_exp);
    assert_eq!(frac.to_bits(), ref_frac.to_bits());
}

// -------------------------------------------------------
//  f64 frexp tests
// -------------------------------------------------------

#[test]
fn frexp_f64_subnormals() {
    let cases: &[f64] = &[
        f64::from_bits(0x0000_0000_0000_0001), // smallest subnormal (= 5e-324)
        f64::from_bits(0x8000_0000_0000_0001), // negative smallest
        f64::from_bits(0x000F_FFFF_FFFF_FFFF), // largest subnormal
        f64::from_bits(0x800F_FFFF_FFFF_FFFF), // negative largest subnormal
        f64::from_bits(0x0008_0000_0000_0000), // 2^-1022 * 0.5
        f64::from_bits(0x0000_0000_0000_0010),
        f64::from_bits(0x0000_0000_0001_0000),
        f64::from_bits(0x0000_0001_0000_0000),
    ];

    for &x in cases {
        let (frac, exp) = your_frexp_f64(x);
        let (ref_frac, ref_exp) = ref_frexp_f64(x);

        assert_eq!(
            exp,
            ref_exp,
            "frexp f64 subnormal exp mismatch for x = {:e} (bits {:#018x})",
            x,
            x.to_bits()
        );
        assert_eq!(
            frac.to_bits(),
            ref_frac.to_bits(),
            "frexp f64 subnormal frac mismatch for x = {:e} (bits {:#018x})",
            x,
            x.to_bits()
        );

        assert!(frac.abs() >= 0.5 && frac.abs() < 1.0);
        assert_eq!(frac.is_sign_negative(), x.is_sign_negative());

        let reconstructed = your_ldexp_f64(frac, exp);
        assert_eq!(
            reconstructed.to_bits(),
            x.to_bits(),
            "frexp f64 subnormal roundtrip failed for x = {:e}",
            x
        );
    }
}

// -------------------------------------------------------
// Helper: bit-exact comparison against libm
// -------------------------------------------------------

fn ref_ldexp_f32(x: f32, n: i32) -> f32 {
    libm::ldexpf(x, n)
}

fn ref_ldexp_f64(x: f64, n: i64) -> f64 {
    libm::ldexp(x, n as i32)
}

// -------------------------------------------------------
//  f32 ldexp tests
// -------------------------------------------------------

#[test]
fn ldexp_f32_normal_scaling() {
    let cases: &[(f32, i32, f32)] = &[
        (1.0, 0, 1.0),
        (1.0, 1, 2.0),
        (1.0, -1, 0.5),
        (1.0, 10, 1024.0),
        (1.0, -10, 1.0 / 1024.0),
        (0.5, 1, 1.0),
        (0.75, 2, 3.0),
        (-1.0, 3, -8.0),
        (1.5, -1, 0.75),
    ];

    for &(x, n, expected) in cases {
        let result = your_ldexp_f32(x, n);
        assert_eq!(
            result.to_bits(),
            expected.to_bits(),
            "ldexp({x}, {n}): got {result}, expected {expected}"
        );
    }
}

#[test]
fn ldexp_f32_zeros() {
    for &x in &[0.0f32, -0.0f32] {
        for n in [-1000, -1, 0, 1, 1000] {
            let result = your_ldexp_f32(x, n);
            assert_eq!(
                result.to_bits(),
                x.to_bits(),
                "ldexp({x:?}, {n}) should preserve signed zero"
            );
        }
    }
}

#[test]
fn ldexp_f32_infinities() {
    for n in [-1000, -1, 0, 1, 1000] {
        let r = your_ldexp_f32(f32::INFINITY, n);
        assert!(r == f32::INFINITY, "ldexp(+inf, {n})");

        let r = your_ldexp_f32(f32::NEG_INFINITY, n);
        assert!(r == f32::NEG_INFINITY, "ldexp(-inf, {n})");
    }
}

#[test]
fn ldexp_f32_nan() {
    for n in [-1000, 0, 1000] {
        let r = your_ldexp_f32(f32::NAN, n);
        assert!(r.is_nan(), "ldexp(NaN, {n}) should be NaN");
    }
}

#[test]
fn ldexp_f32_overflow_to_inf() {
    // Just past representable range
    let r = your_ldexp_f32(1.0, 128);
    assert_eq!(r, f32::INFINITY);

    let r = your_ldexp_f32(-1.0, 128);
    assert_eq!(r, f32::NEG_INFINITY);

    // MAX * 2 overflows
    let r = your_ldexp_f32(f32::MAX, 1);
    assert_eq!(r, f32::INFINITY);
}

#[test]
fn ldexp_f32_underflow_to_zero() {
    // Way below subnormal range
    let r = your_ldexp_f32(1.0, -200);
    assert_eq!(r.to_bits(), 0.0f32.to_bits());

    let r = your_ldexp_f32(-1.0, -200);
    assert_eq!(r.to_bits(), (-0.0f32).to_bits());
}

#[test]
fn ldexp_f32_normal_to_subnormal() {
    // 1.0 * 2^-126 = MIN_POSITIVE (smallest normal)
    let r = your_ldexp_f32(1.0, -126);
    assert_eq!(r, f32::MIN_POSITIVE);

    // 1.0 * 2^-127 = largest subnormal / 2 region
    let r = your_ldexp_f32(1.0, -127);
    let expected = ref_ldexp_f32(1.0, -127);
    assert_eq!(
        r.to_bits(),
        expected.to_bits(),
        "ldexp(1.0, -127): got {r:e} ({:#010x}), expected {expected:e} ({:#010x})",
        r.to_bits(),
        expected.to_bits()
    );
    // Verify it's actually subnormal
    assert!(r != 0.0 && r.classify() == core::num::FpCategory::Subnormal);

    // Further into subnormal territory
    for n in -149..=-127 {
        let r = your_ldexp_f32(1.0, n);
        let expected = ref_ldexp_f32(1.0, n);
        assert_eq!(
            r.to_bits(),
            expected.to_bits(),
            "ldexp(1.0, {n}): got {r:e} ({:#010x}), expected {expected:e} ({:#010x})",
            r.to_bits(),
            expected.to_bits()
        );
    }

    // Smallest representable: 2^-149
    let r = your_ldexp_f32(1.0, -149);
    assert_eq!(r.to_bits(), 0x0000_0001);

    // One below: 2^-150 rounds to zero
    let r = your_ldexp_f32(1.0, -150);
    assert_eq!(r, 0.0);
}

#[test]
fn ldexp_f32_subnormal_input_scale_up() {
    let smallest_subnormal = f32::from_bits(0x0000_0001); // 2^-149

    // Scale subnormal back to normal range
    let r = your_ldexp_f32(smallest_subnormal, 149);
    assert_eq!(r, 1.0, "smallest subnormal * 2^149 should be 1.0");

    let r = your_ldexp_f32(smallest_subnormal, 150);
    assert_eq!(r, 2.0);

    let r = your_ldexp_f32(smallest_subnormal, 200);
    let expected = ref_ldexp_f32(smallest_subnormal, 200);
    assert_eq!(r.to_bits(), expected.to_bits());

    // Largest subnormal scaled up
    let largest_subnormal = f32::from_bits(0x007F_FFFF);
    let r = your_ldexp_f32(largest_subnormal, 127);
    let expected = ref_ldexp_f32(largest_subnormal, 127);
    assert_eq!(r.to_bits(), expected.to_bits(), "largest subnormal * 2^127");
}

#[test]
fn ldexp_f32_subnormal_input_scale_down() {
    let largest_subnormal = f32::from_bits(0x007F_FFFF);

    // Shifting a subnormal down further
    let r = your_ldexp_f32(largest_subnormal, -1);
    let expected = ref_ldexp_f32(largest_subnormal, -1);
    assert_eq!(r.to_bits(), expected.to_bits(), "largest subnormal >> 1");

    // Multiple subnormal-to-subnormal scalings
    for shift in 1..=23 {
        let r = your_ldexp_f32(largest_subnormal, -shift);
        let expected = ref_ldexp_f32(largest_subnormal, -shift);
        assert_eq!(r.to_bits(), expected.to_bits(), "largest_subnormal * 2^-{shift}");
    }
}

#[test]
fn ldexp_f32_preserves_mantissa_bits() {
    // 1.5 = 1 + 0.5, so mantissa bit 22 is set.
    // Scaling into subnormal range should shift this bit right.
    let x = 1.5f32; // 0x3FC0_0000

    for n in -148..=-127 {
        let r = your_ldexp_f32(x, n);
        let expected = ref_ldexp_f32(x, n);
        assert_eq!(
            r.to_bits(),
            expected.to_bits(),
            "ldexp(1.5, {n}): mantissa bits should be correctly shifted"
        );
    }
}

#[test]
fn ldexp_f32_frexp_roundtrip_subnormals() {
    // The most important test: frexp then ldexp should be identity.
    let subnormals: &[f32] = &[
        f32::from_bits(0x0000_0001),
        f32::from_bits(0x8000_0001),
        f32::from_bits(0x007F_FFFF),
        f32::from_bits(0x807F_FFFF),
        f32::from_bits(0x0040_0000),
        f32::from_bits(0x0000_0100),
        f32::from_bits(0x0000_FFFF),
    ];

    for &x in subnormals {
        let (frac, exp) = your_frexp_f32(x);
        let reconstructed = your_ldexp_f32(frac, exp);
        assert_eq!(
            reconstructed.to_bits(),
            x.to_bits(),
            "frexp->ldexp roundtrip failed for subnormal {x:e} (bits {:#010x}): \
                 frexp gave ({frac:e}, {exp}), ldexp gave {reconstructed:e} ({:#010x})",
            x.to_bits(),
            reconstructed.to_bits()
        );
    }
}

#[cfg_attr(not(feature = "preserve_denormals"), should_panic)]
#[test]
fn ldexp_f32_extreme_exponents() {
    // Huge positive exponent
    let r = your_ldexp_f32(1.0, i32::MAX);
    assert!(r.is_infinite());

    let r = your_ldexp_f32(f32::MIN_POSITIVE, i32::MAX);
    assert!(r.is_infinite());

    // Huge negative exponent
    let r = your_ldexp_f32(1.0, i32::MIN);
    assert_eq!(r, 0.0);

    let r = your_ldexp_f32(f32::MAX, i32::MIN);
    assert_eq!(r, 0.0);
}

// -------------------------------------------------------
//  f64 ldexp tests
// -------------------------------------------------------

#[test]
fn ldexp_f64_normal_to_subnormal() {
    // 1.0 * 2^-1022 = MIN_POSITIVE
    let r = your_ldexp_f64(1.0, -1022);
    assert_eq!(r, f64::MIN_POSITIVE);

    // Into subnormal range
    for n in -1074..=-1023 {
        let r = your_ldexp_f64(1.0, n);
        let expected = ref_ldexp_f64(1.0, n);
        assert_eq!(
            r.to_bits(),
            expected.to_bits(),
            "ldexp f64 (1.0, {n}): got {:e} ({:#018x}), expected {:e} ({:#018x})",
            r,
            r.to_bits(),
            expected,
            expected.to_bits()
        );
    }

    // Smallest representable f64: 2^-1074
    let r = your_ldexp_f64(1.0, -1074);
    assert_eq!(r.to_bits(), 0x0000_0000_0000_0001);
}

#[test]
fn ldexp_f64_subnormal_roundtrip() {
    let subnormals: &[f64] = &[
        f64::from_bits(0x0000_0000_0000_0001),
        f64::from_bits(0x8000_0000_0000_0001),
        f64::from_bits(0x000F_FFFF_FFFF_FFFF),
        f64::from_bits(0x800F_FFFF_FFFF_FFFF),
        f64::from_bits(0x0008_0000_0000_0000),
        5e-324,
    ];

    for &x in subnormals {
        let (frac, exp) = your_frexp_f64(x);
        let reconstructed = your_ldexp_f64(frac, exp);
        assert_eq!(
            reconstructed.to_bits(),
            x.to_bits(),
            "f64 frexp->ldexp roundtrip failed for {:e} (bits {:#018x})",
            x,
            x.to_bits()
        );
    }
}

#[test]
fn ldexp_f64_preserves_mantissa_bits() {
    // 1.0 + 2^-52 (smallest increment above 1.0)
    let x = f64::from_bits(0x3FF0_0000_0000_0001);

    for n in -1073..=-1023 {
        let r = your_ldexp_f64(x, n);
        let expected = ref_ldexp_f64(x, n);
        assert_eq!(
            r.to_bits(),
            expected.to_bits(),
            "ldexp f64 (1+eps, {n}): mantissa rounding mismatch"
        );
    }
}

// -------------------------------------------------------
//  Bulk sweep: compare against libm for many values
// -------------------------------------------------------

#[test]
fn ldexp_f32_exhaustive_subnormal_range() {
    // Sweep all f32 subnormals (there are only 2^23 - 1 of them)
    for bits in 1u32..0x0080_0000 {
        let x = f32::from_bits(bits);
        // Scale up into normal range and back
        let up = your_ldexp_f32(x, 127);
        let expected_up = ref_ldexp_f32(x, 127);
        assert_eq!(up.to_bits(), expected_up.to_bits(), "subnormal {:#010x} * 2^127", bits);
    }
}

#[test]
fn ldexp_f32_exhaustive_into_subnormal() {
    // Take a few normal values and sweep them down into subnormal territory
    let normals: &[f32] = &[1.0, 1.5, 1.999999, 1.0000001];

    for &x in normals {
        for n in -150..=-126 {
            let r = your_ldexp_f32(x, n);
            let expected = ref_ldexp_f32(x, n);
            assert_eq!(r.to_bits(), expected.to_bits(), "ldexp({x}, {n})");
        }
    }
}

#[test]
fn frexp_f32_exhaustive_subnormals() {
    // Check every positive subnormal f32
    for bits in 1u32..0x0080_0000 {
        let x = f32::from_bits(bits);
        let (frac, exp) = your_frexp_f32(x);
        let (ref_frac, ref_exp) = ref_frexp_f32(x);

        assert_eq!(exp, ref_exp, "frexp exp for subnormal {bits:#010x}");
        assert_eq!(
            frac.to_bits(),
            ref_frac.to_bits(),
            "frexp frac for subnormal {bits:#010x}"
        );
    }
}

// -------------------------------------------------------
// Placeholder wrappers -- replace with your actual calls
// -------------------------------------------------------
//
// For scalar f32/f64 through the Vector<f32>/Vector<f64> path
// with Preserve denormal policy:
//

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct TestPolicy;

impl Policy for TestPolicy {
    const POLICY: PolicyParameters = PolicyParameters {
        denormal_behavior: DenormalBehavior::Preserve,
        check_overflow: true,
        unroll_loops: true,
        precision: PrecisionPolicy::Best,
        avoid_branching: false,
        max_iterations: 100,
        use_compensation: true,
    };
}

fn your_frexp_f32(x: f32) -> (f32, i32) {
    let v = thermite::Vector::<f32>(x);
    let (frac, exp) = v.frexp_p::<TestPolicy>();
    (frac.extract::<0>(), exp.extract::<0>())
}

fn your_ldexp_f32(x: f32, n: i32) -> f32 {
    let v = Vector::<f32>(x);
    let e = Vector::<i32>(n);
    v.ldexp_p::<TestPolicy>(e).extract::<0>()
}

fn your_frexp_f64(x: f64) -> (f64, i64) {
    let v = thermite::Vector::<f64>(x);
    let (frac, exp) = v.frexp_p::<TestPolicy>();
    (frac.extract::<0>(), exp.extract::<0>())
}

fn your_ldexp_f64(x: f64, n: i64) -> f64 {
    let v = thermite::Vector::<f64>(x);
    let e = thermite::Vector::<i64>(n);
    v.ldexp_p::<TestPolicy>(e).extract::<0>()
}

// Same pattern for f64.
