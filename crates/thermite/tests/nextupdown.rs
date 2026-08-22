//! Differential tests for `FloatRegister::next_up` / `next_down`.
//!
//! Two distinct implementations exist here, and both are checked bit-exactly
//! against Rust's `f32/f64::next_up`/`next_down`:
//!   - the generic bit-twiddling default in `register/mod.rs` (every CPU
//!     backend, X86V1/V2/V3 included, plus the wide `ArrayRegister` types via
//!     lane delegation; the hand-written v1/v3 polyfills were deleted in
//!     favor of it),
//!   - the scalar element seed (`FloatElement::next_up`/`next_down`).
//!
//! A third, the x86-v4 k-mask polyfills, only compiles under an
//! `avx512-tier*` feature and is not exercised by this suite.
//!
//! Beyond the random/edge corpus, `regions` walks every interesting area of
//! the IEEE-754 domain explicitly: NaNs (payloads and signs), both infinities,
//! +/-MAX, exponent-rollover boundaries, the normal/subnormal boundary, the
//! tiniest subnormals, and both zeros.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use harness::Tol;

use thermite::register::FloatRegister as _;
use thermite::simd::Simd;

use thermite::backend::scalar::Scalar;

// ---------------------------------------------------------------------------
// Random + standard-edge corpus (bit-exact, NaN-aware via Tol::Exact).
// ---------------------------------------------------------------------------

macro_rules! corpus_suite {
    ($name:ident, $backend:ty, [$($reg:ident: $e:ty),* $(,)?], $label:expr) => {
        mod $name {
            use super::*;

            #[test]
            fn next_up() {
                $( oracle_unary!(concat!($label, " ", stringify!($reg)), <$backend as Simd>::$reg, $e,
                    next_up, |x| x.next_up(), Tol::Exact); )*
            }

            #[test]
            fn next_down() {
                $( oracle_unary!(concat!($label, " ", stringify!($reg)), <$backend as Simd>::$reg, $e,
                    next_down, |x| x.next_down(), Tol::Exact); )*
            }
        }
    };
}

corpus_suite!(scalar, Scalar, [f32x4: f32, f64x2: f64], "scalar");

// ---------------------------------------------------------------------------
// Explicit region walk: every value below is tested splatted across all lanes,
// and the whole list is also streamed through lane-sized chunks to check lane
// independence.
// ---------------------------------------------------------------------------

macro_rules! region_values {
    ($e:ty, $bits:ty) => {{
        let nz: $e = -0.0;
        [
            // NaNs: quiet, payload, negative
            <$e>::NAN,
            <$e>::from_bits(<$e>::NAN.to_bits() | 1), // payload NaN
            -<$e>::NAN,
            // All-ones-mantissa NaNs: the inputs where a broken NaN guard
            // wraps the increment out of NaN-space entirely (0x7F..F + 1 =
            // sign bit = -0.0; 0xFF..F + 1 = 0 = +0.0). Caught a real v3 bug:
            // its `is_nan` used `_CMP_NEQ_OQ`, which is constant-false.
            <$e>::from_bits(<$bits>::MAX >> 1),
            <$e>::from_bits(<$bits>::MAX),
            // infinities and the largest finite values
            <$e>::INFINITY,
            <$e>::NEG_INFINITY,
            <$e>::MAX,
            <$e>::MIN, // == -MAX
            // exponent rollover: the values straddling 1.0 and 2.0
            <$e>::from_bits((1.0 as $e).to_bits() - 1),
            1.0,
            <$e>::from_bits((2.0 as $e).to_bits() - 1),
            2.0,
            -1.0,
            // ordinary values
            1.5,
            -2.5,
            // normal/subnormal boundary
            <$e>::MIN_POSITIVE,                            // smallest normal
            -<$e>::MIN_POSITIVE,
            <$e>::from_bits(<$e>::MIN_POSITIVE.to_bits() - 1), // largest subnormal
            -<$e>::from_bits(<$e>::MIN_POSITIVE.to_bits() - 1),
            // the tiniest subnormals (next_up/down of these reach +/-0)
            <$e>::from_bits(1),
            <$e>::from_bits(1 | (1 as $bits) << (size_of::<$e>() * 8 - 1)), // -tiniest
            // both zeros
            0.0,
            nz,
        ]
    }};
}

macro_rules! region_check {
    ($label:expr, $ut:ty, $e:ty, $values:expr) => {{
        let lanes = <<$ut as ::thermite::register::CoreRegister>::Lanes as ::generic_array::typenum::Unsigned>::USIZE;
        let values: &[$e] = &$values;

        let mut cases: Vec<Vec<$e>> = Vec::new();
        // each special value in every lane at once
        for &v in values {
            cases.push(vec![v; lanes]);
        }
        // and mixed lane-sized chunks (wrapping) for lane independence
        for chunk in values.chunks(lanes) {
            let mut case: Vec<$e> = chunk.to_vec();
            case.resize(lanes, 1.0);
            cases.push(case);
        }

        for input in cases {
            let got_up = harness::read::<$ut>(&<$ut>::next_up(harness::make_array::<$ut>(&input)));
            let want_up: Vec<$e> = input.iter().map(|x| x.next_up()).collect();
            harness::assert_lanes_eq(
                concat!($label, " [next_up regions]"),
                &[input.as_slice()],
                &got_up,
                &want_up,
                Tol::Exact,
            );

            let got_down = harness::read::<$ut>(&<$ut>::next_down(harness::make_array::<$ut>(&input)));
            let want_down: Vec<$e> = input.iter().map(|x| x.next_down()).collect();
            harness::assert_lanes_eq(
                concat!($label, " [next_down regions]"),
                &[input.as_slice()],
                &got_down,
                &want_down,
                Tol::Exact,
            );
        }
    }};
}

macro_rules! region_suite {
    ($name:ident, $backend:ty, $label:expr) => {
        mod $name {
            use super::*;

            #[test]
            fn regions_f32() {
                region_check!($label, <$backend as Simd>::f32x4, f32, region_values!(f32, u32));
                region_check!($label, <$backend as Simd>::f32x8, f32, region_values!(f32, u32));
            }

            #[test]
            fn regions_f64() {
                region_check!($label, <$backend as Simd>::f64x2, f64, region_values!(f64, u64));
                region_check!($label, <$backend as Simd>::f64x4, f64, region_values!(f64, u64));
            }
        }
    };
}

region_suite!(regions_scalar, Scalar, "scalar");

// ---------------------------------------------------------------------------
// Round-trip property: for finite values, next_down(next_up(x)) == x
// (every value in the region list except NaN and the infinities).
// ---------------------------------------------------------------------------

macro_rules! roundtrip_check {
    ($label:expr, $ut:ty, $e:ty, $values:expr) => {{
        let lanes = <<$ut as ::thermite::register::CoreRegister>::Lanes as ::generic_array::typenum::Unsigned>::USIZE;
        let values: &[$e] = &$values;

        for &v in values
            .iter()
            .filter(|v| v.is_finite() && **v != <$e>::MAX && **v != <$e>::MIN)
        {
            let input = vec![v; lanes];
            let reg = harness::make_array::<$ut>(&input);
            let rt = harness::read::<$ut>(&<$ut>::next_down(<$ut>::next_up(reg)));

            // next_up(-tiniest) is -0.0, whose next_down is -tiniest again, but
            // next_up(+0.0) is +tiniest whose next_down is +0.0 - both equal the
            // input bitwise except for the -0.0 -> +0.0 -> +tiniest -> +0.0 hop,
            // so the property is plain bitwise equality everywhere but -0.0.
            let want: Vec<$e> = input.iter().map(|x| x.next_up().next_down()).collect();
            harness::assert_lanes_eq(
                concat!($label, " [next_up/next_down roundtrip]"),
                &[input.as_slice()],
                &rt,
                &want,
                Tol::Exact,
            );
        }
    }};
}

#[test]
fn roundtrip() {
    roundtrip_check!("scalar f32x4", <Scalar as Simd>::f32x4, f32, region_values!(f32, u32));
}

// x86 backends: native polyfills (v1/v3) and the generic default (v2).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    use thermite::backend::{x86_v1::X86V1, x86_v2::X86V2, x86_v3::X86V3};

    // v1: the SSE2 polyfills natively, plus ArrayRegister delegation on the wide types
    corpus_suite!(v1, X86V1, [f32x4: f32, f64x2: f64, f32x8: f32, f64x4: f64], "x86_v1");
    // v2: the generic FloatRegister default implementation
    corpus_suite!(v2, X86V2, [f32x4: f32, f64x2: f64], "x86_v2");
    // v3: the AVX2 polyfills on native widths, generic default on the 128-bit types
    corpus_suite!(v3, X86V3, [f32x8: f32, f64x4: f64, f32x4: f32, f64x2: f64], "x86_v3");

    region_suite!(regions_v1, X86V1, "x86_v1");
    region_suite!(regions_v2, X86V2, "x86_v2");
    region_suite!(regions_v3, X86V3, "x86_v3");

    #[test]
    fn roundtrip() {
        roundtrip_check!("x86_v1 f32x4", <X86V1 as Simd>::f32x4, f32, region_values!(f32, u32));
        roundtrip_check!("x86_v1 f64x2", <X86V1 as Simd>::f64x2, f64, region_values!(f64, u64));
        roundtrip_check!("x86_v2 f32x4", <X86V2 as Simd>::f32x4, f32, region_values!(f32, u32));
        roundtrip_check!("x86_v2 f64x2", <X86V2 as Simd>::f64x2, f64, region_values!(f64, u64));
        roundtrip_check!("x86_v3 f32x8", <X86V3 as Simd>::f32x8, f32, region_values!(f32, u32));
        roundtrip_check!("x86_v3 f64x4", <X86V3 as Simd>::f64x4, f64, region_values!(f64, u64));
    }
}

// wasm: generic FloatRegister default on native f32x4/f64x2, ArrayRegister on the wide types.
#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use thermite::backend::wasm::Wasm;

    corpus_suite!(wasm, Wasm, [f32x4: f32, f64x2: f64, f32x8: f32, f64x4: f64], "wasm");
    region_suite!(regions_wasm, Wasm, "wasm");

    #[test]
    fn roundtrip() {
        roundtrip_check!("wasm f32x4", <Wasm as Simd>::f32x4, f32, region_values!(f32, u32));
        roundtrip_check!("wasm f64x2", <Wasm as Simd>::f64x2, f64, region_values!(f64, u64));
        roundtrip_check!("wasm f32x8", <Wasm as Simd>::f32x8, f32, region_values!(f32, u32));
        roundtrip_check!("wasm f64x4", <Wasm as Simd>::f64x4, f64, region_values!(f64, u64));
    }
}

// neon: generic FloatRegister default on native f32x4/f64x2, ArrayRegister on the wide types.
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use thermite::backend::neon::Neon;

    corpus_suite!(neon, Neon, [f32x4: f32, f64x2: f64, f32x8: f32, f64x4: f64], "neon");
    region_suite!(regions_neon, Neon, "neon");

    #[test]
    fn roundtrip() {
        roundtrip_check!("neon f32x4", <Neon as Simd>::f32x4, f32, region_values!(f32, u32));
        roundtrip_check!("neon f64x2", <Neon as Simd>::f64x2, f64, region_values!(f64, u64));
        roundtrip_check!("neon f32x8", <Neon as Simd>::f32x8, f32, region_values!(f32, u32));
        roundtrip_check!("neon f64x4", <Neon as Simd>::f64x4, f64, region_values!(f64, u64));
    }
}
