//! `frexp` must satisfy BOTH halves of its contract on every finite nonzero
//! input, including subnormals: `x == frac * 2^exp` **and** `0.5 <= |frac| < 1`.
//!
//! The flush path used to return `(x, 0)` for a subnormal - the identity held,
//! the normalization bound did not, silently. These check every policy against
//! libm.
#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use thermite::math::policy::policies::{AvoidBranching, Performance, PreserveDenormals};
use thermite::math::policy::{DenormalBehavior, Policy, PolicyParameters, PrecisionPolicy};
use thermite::prelude::*;
use thermite::simd::Simd;

/// Explicit flush-to-zero, independent of the `preserve_denormals` /
/// `strict_ieee754` features (which flip the default).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct FlushPolicy;

impl Policy for FlushPolicy {
    const POLICY: PolicyParameters = PolicyParameters {
        check_overflow: true,
        unroll_loops: true,
        precision: PrecisionPolicy::Average,
        avoid_branching: false,
        max_iterations: 10000,
        use_compensation: false,
        denormal_behavior: DenormalBehavior::FlushToZero,
    };
}

fn f32_cases() -> Vec<f32> {
    let mut v = vec![
        1.0,
        1.5,
        -1.5,
        0.75,
        100.0,
        -100.0,
        f32::MAX,
        f32::MIN,
        f32::MIN_POSITIVE,               // smallest normal
        f32::from_bits(0x007F_FFFF),     // largest subnormal
        f32::from_bits(0x0000_0001),     // smallest subnormal
        f32::from_bits(0x8000_0001),     // negative smallest subnormal
        f32::from_bits(0x0040_0000),
        f32::from_bits(0x0000_FFFF),
        1.0e-40,
        -1.0e-40,
        core::f32::consts::PI,
    ];
    // a sweep across the subnormal range, where the old code was silently wrong
    for k in 0..23 {
        v.push(f32::from_bits(1 << k));
        v.push(f32::from_bits((1 << k) | 0x8000_0000));
    }
    v
}

fn f64_cases() -> Vec<f64> {
    vec![
        1.0,
        -1.5,
        f64::MAX,
        f64::MIN_POSITIVE,
        f64::from_bits(0x000F_FFFF_FFFF_FFFF), // largest subnormal
        f64::from_bits(0x0000_0000_0000_0001), // smallest subnormal
        f64::from_bits(0x8000_0000_0000_0001),
        f64::from_bits(0x0008_0000_0000_0000),
        5.0e-324,
        1.0e-320,
        core::f64::consts::PI,
    ]
}

macro_rules! check_f32 {
    ($b:ty, $p:ty, $label:expr) => {{
        for x in f32_cases() {
            let (f, e) = Vector::<<$b as Simd>::f32x8>::splat(x).frexp_p::<$p>();
            let (gf, ge) = (f.extract::<0>(), e.extract::<0>());
            let (wf, we) = libm::frexpf(x);

            assert_eq!(
                gf.to_bits(),
                wf.to_bits(),
                "{} frexp({x:e}) frac: got {gf:e}, want {wf:e}",
                $label
            );
            assert_eq!(ge, we, "{} frexp({x:e}) exp: got {ge}, want {we}", $label);

            // both halves of the contract, stated directly
            assert!(
                gf.abs() >= 0.5 && gf.abs() < 1.0,
                "{} frexp({x:e}): fraction {gf:e} outside [0.5, 1.0)",
                $label
            );
            assert_eq!(
                libm::ldexpf(gf, ge).to_bits(),
                x.to_bits(),
                "{} frexp({x:e}) round-trip",
                $label
            );
        }
    }};
}

macro_rules! check_f64 {
    ($b:ty, $p:ty, $label:expr) => {{
        for x in f64_cases() {
            let (f, e) = Vector::<<$b as Simd>::f64x4>::splat(x).frexp_p::<$p>();
            let (gf, ge) = (f.extract::<0>(), e.extract::<0>());
            let (wf, we) = libm::frexp(x);

            assert_eq!(
                gf.to_bits(),
                wf.to_bits(),
                "{} frexp({x:e}) frac: got {gf:e}, want {wf:e}",
                $label
            );
            assert_eq!(ge, we as i64, "{} frexp({x:e}) exp: got {ge}, want {we}", $label);
            assert!(
                gf.abs() >= 0.5 && gf.abs() < 1.0,
                "{} frexp({x:e}): fraction {gf:e} outside [0.5, 1.0)",
                $label
            );
        }
    }};
}

/// Zeros and non-finites are unchanged by the fixup: `(+-0, 0)`, inf/NaN through.
macro_rules! check_specials {
    ($b:ty, $p:ty, $label:expr) => {{
        for x in [0.0f32, -0.0] {
            let (f, e) = Vector::<<$b as Simd>::f32x8>::splat(x).frexp_p::<$p>();
            assert_eq!(f.extract::<0>().to_bits(), x.to_bits(), "{} frexp({x}) frac", $label);
            assert_eq!(e.extract::<0>(), 0, "{} frexp({x}) exp", $label);
        }
        for x in [f32::INFINITY, f32::NEG_INFINITY] {
            let (f, e) = Vector::<<$b as Simd>::f32x8>::splat(x).frexp_p::<$p>();
            assert_eq!(f.extract::<0>(), x, "{} frexp({x}) frac", $label);
            assert_eq!(e.extract::<0>(), 0, "{} frexp({x}) exp", $label);
        }
        let (f, _) = Vector::<<$b as Simd>::f32x8>::splat(f32::NAN).frexp_p::<$p>();
        assert!(f.extract::<0>().is_nan(), "{} frexp(NaN)", $label);
    }};
}

macro_rules! suite {
    ($m:ident, $b:ty, $l:expr) => {
        mod $m {
            use super::*;

            // the branchy fast path
            #[test]
            fn flush_f32() {
                check_f32!($b, FlushPolicy, concat!($l, " flush"));
            }
            #[test]
            fn flush_f64() {
                check_f64!($b, FlushPolicy, concat!($l, " flush"));
            }

            // the straight-line arm (Preserve)
            #[test]
            fn preserve_f32() {
                check_f32!($b, PreserveDenormals<Performance>, concat!($l, " preserve"));
            }
            #[test]
            fn preserve_f64() {
                check_f64!($b, PreserveDenormals<Performance>, concat!($l, " preserve"));
            }

            // the branchless arm under a flushing policy
            #[test]
            fn branchless_f32() {
                check_f32!($b, AvoidBranching<FlushPolicy, true>, concat!($l, " branchless"));
            }

            #[test]
            fn specials() {
                check_specials!($b, FlushPolicy, concat!($l, " flush"));
                check_specials!($b, PreserveDenormals<Performance>, concat!($l, " preserve"));
                check_specials!($b, AvoidBranching<FlushPolicy, true>, concat!($l, " branchless"));
            }
        }
    };
}

suite!(scalar, thermite::backend::scalar::Scalar, "scalar");
suite!(x86_v1, thermite::backend::x86_v1::X86V1, "x86_v1");
suite!(x86_v2, thermite::backend::x86_v2::X86V2, "x86_v2");
suite!(x86_v3, thermite::backend::x86_v3::X86V3, "x86_v3");
