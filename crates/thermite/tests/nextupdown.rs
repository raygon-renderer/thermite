//! Differential tests for `FloatRegister::next_up` / `next_down`.
//!
//! Three distinct implementations exist here, and all are checked bit-exactly
//! against Rust's `f32/f64::next_up`/`next_down`:
//!   - the generic bit-twiddling default in `register/mod.rs` (every CPU
//!     backend, X86V1/V2/V3 included, plus the wide `ArrayRegister` types via
//!     lane delegation; the hand-written v1/v3 polyfills were deleted in
//!     favor of it),
//!   - the scalar element seed (`FloatElement::next_up`/`next_down`),
//!   - the x86-v4 k-mask polyfills (only under an `avx512-tier*` feature, so
//!     only under Intel SDE on this host).
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

// ---------------------------------------------------------------------------
// Random + standard-edge corpus (bit-exact, NaN-aware via Tol::Exact).
// ---------------------------------------------------------------------------
macro_rules! corpus {
    ($S:ty, $method:ident, [$($reg:ident: $e:ty),* $(,)?]) => {
        $( oracle_unary!(harness::label::<$S>(stringify!($reg)), <$S as Simd>::$reg, $e,
            $method, |x| x.$method(), Tol::Exact); )*
    };
}

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
            <$e>::INFINITY,
            <$e>::NEG_INFINITY,
            <$e>::MAX,
            <$e>::MIN, // == -MAX
            <$e>::from_bits((1.0 as $e).to_bits() - 1),
            1.0,
            <$e>::from_bits((2.0 as $e).to_bits() - 1),
            2.0,
            -1.0,
            1.5,
            -2.5,
            <$e>::MIN_POSITIVE,                            // smallest normal
            -<$e>::MIN_POSITIVE,
            <$e>::from_bits(<$e>::MIN_POSITIVE.to_bits() - 1), // largest subnormal
            -<$e>::from_bits(<$e>::MIN_POSITIVE.to_bits() - 1),
            <$e>::from_bits(1),
            <$e>::from_bits(1 | (1 as $bits) << (size_of::<$e>() * 8 - 1)), // -tiniest
            0.0,
            nz,
        ]
    }};
}

macro_rules! region_check {
    ($S:ty, $reg:ident, $e:ty, $values:expr) => {{
        let label = harness::label::<$S>(stringify!($reg));
        let lanes =
            <<<$S as Simd>::$reg as ::thermite::register::CoreRegister>::Lanes as ::generic_array::typenum::Unsigned>::USIZE;
        let values: &[$e] = &$values;

        let mut cases: Vec<Vec<$e>> = Vec::new();
        for &v in values {
            cases.push(vec![v; lanes]);
        }
        for chunk in values.chunks(lanes) {
            let mut case: Vec<$e> = chunk.to_vec();
            case.resize(lanes, 1.0);
            cases.push(case);
        }

        for input in cases {
            let got_up = harness::read::<<$S as Simd>::$reg>(&<<$S as Simd>::$reg>::next_up(
                harness::make_array::<<$S as Simd>::$reg>(&input),
            ));
            let want_up: Vec<$e> = input.iter().map(|x| x.next_up()).collect();
            harness::assert_lanes_eq(
                &format!("{label} [next_up regions]"),
                &[input.as_slice()],
                &got_up,
                &want_up,
                Tol::Exact,
            );

            let got_down = harness::read::<<$S as Simd>::$reg>(&<<$S as Simd>::$reg>::next_down(
                harness::make_array::<<$S as Simd>::$reg>(&input),
            ));
            let want_down: Vec<$e> = input.iter().map(|x| x.next_down()).collect();
            harness::assert_lanes_eq(
                &format!("{label} [next_down regions]"),
                &[input.as_slice()],
                &got_down,
                &want_down,
                Tol::Exact,
            );
        }
    }};
}

// next_down(next_up(x)) == x for every finite x short of +-MAX.
macro_rules! roundtrip_check {
    ($S:ty, $reg:ident, $e:ty, $values:expr) => {{
        let label = harness::label::<$S>(stringify!($reg));
        let lanes =
            <<<$S as Simd>::$reg as ::thermite::register::CoreRegister>::Lanes as ::generic_array::typenum::Unsigned>::USIZE;
        let values: &[$e] = &$values;

        for &v in values
            .iter()
            .filter(|v| v.is_finite() && **v != <$e>::MAX && **v != <$e>::MIN)
        {
            let input = vec![v; lanes];
            let reg = harness::make_array::<<$S as Simd>::$reg>(&input);
            let rt = harness::read::<<$S as Simd>::$reg>(&<<$S as Simd>::$reg>::next_down(
                <<$S as Simd>::$reg>::next_up(reg),
            ));

            let want: Vec<$e> = input.iter().map(|x| x.next_up().next_down()).collect();
            harness::assert_lanes_eq(
                &format!("{label} [next_up/next_down roundtrip]"),
                &[input.as_slice()],
                &rt,
                &want,
                Tol::Exact,
            );
        }
    }};
}

for_each_backend! {
    fn next_up<S: Simd>() {
        corpus!(S, next_up, [f32x4: f32, f32x8: f32, f32x16: f32, f64x2: f64, f64x4: f64, f64x8: f64]);
    }
    fn next_down<S: Simd>() {
        corpus!(S, next_down, [f32x4: f32, f32x8: f32, f32x16: f32, f64x2: f64, f64x4: f64, f64x8: f64]);
    }
    fn regions_f32<S: Simd>() {
        region_check!(S, f32x4, f32, region_values!(f32, u32));
        region_check!(S, f32x8, f32, region_values!(f32, u32));
        region_check!(S, f32x16, f32, region_values!(f32, u32));
    }
    fn regions_f64<S: Simd>() {
        region_check!(S, f64x2, f64, region_values!(f64, u64));
        region_check!(S, f64x4, f64, region_values!(f64, u64));
        region_check!(S, f64x8, f64, region_values!(f64, u64));
    }
    fn roundtrip<S: Simd>() {
        roundtrip_check!(S, f32x4, f32, region_values!(f32, u32));
        roundtrip_check!(S, f32x8, f32, region_values!(f32, u32));
        roundtrip_check!(S, f64x2, f64, region_values!(f64, u64));
        roundtrip_check!(S, f64x4, f64, region_values!(f64, u64));
    }
}
