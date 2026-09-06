//! Polyfill-focused differential tests.
//!
//! Operations with no native hardware instruction are emulated by a *polyfill*
//! (`backend/*/polyfills/`). These are the highest-risk code in the library -
//! this audit found **six** production bugs here (all since fixed). Every
//! polyfill-backed register op is checked against an **independent pure-Rust
//! oracle** (not the scalar backend, which may route through the same generic
//! polyfill and hide a shared bug).
//!
//! `fixed_*` are regression tests for the defects this file's audit found and
//! that have been fixed (P1-P6). Everything runs on every backend: an op that
//! is native on one backend is a polyfill on another, and the oracle does not
//! care which.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use harness::Tol;

use thermite::register::{
    BitshiftRegister as _, FloatRegister as _, IntegerRegister as _, Register as _, SignedIntegerRegister as _,
    SignedRegister as _, UnsignedIntegerRegister as _,
};
use thermite::simd::Simd;

/// `"X86V3 i32x4"`-style label for a slot of `$S`.
macro_rules! l {
    ($S:ty, $reg:ident) => {
        harness::label::<$S>(stringify!($reg))
    };
}

// ===========================================================================
// Verified-correct polyfills, always-green.
// ===========================================================================

/// `mullo` (low half of the product), exercising the 64-bit
/// `_mm{,256}_mullo_epi64x` emulation. Correct for every width.
macro_rules! mullo_for {
    ($S:ty; $($reg:ident: $e:ty),* $(,)?) => {
        $( oracle_binary!(l!($S, $reg), <$S as Simd>::$reg, $e, mullo, |a, b| a.wrapping_mul(b), Tol::Exact); )*
    };
}

/// Bit population / scan ops that are correct everywhere they're tested here.
macro_rules! popcount_for {
    ($S:ty; $($reg:ident: $e:ty),* $(,)?) => {
        $(
            oracle_unary!(l!($S, $reg), <$S as Simd>::$reg, $e, count_ones, |x| x.count_ones() as $e, Tol::Exact);
            oracle_unary!(l!($S, $reg), <$S as Simd>::$reg, $e, count_zeros, |x| x.count_zeros() as $e, Tol::Exact);
        )*
    };
}

/// `roli`/`rori` (const-generic amounts) vs Rust's `rotate_left`/`rotate_right`,
/// for each literal amount that fits the element width.
macro_rules! check_rot_const {
    ($l:expr, $reg:ty, $e:ty, $input:expr, $regv:expr, $bits:expr; $($imm:literal),* $(,)?) => {$(
        if ($imm as u32) < $bits {
            let got = harness::read::<$reg>(&<$reg>::roli::<$imm>($regv));
            let want: Vec<$e> = $input.iter().map(|&x| x.rotate_left($imm as u32)).collect();
            harness::assert_lanes_eq(
                &format!("{} [roli<{}>]", $l, stringify!($imm)),
                &[$input.as_slice()], &got, &want, Tol::Exact);

            let got = harness::read::<$reg>(&<$reg>::rori::<$imm>($regv));
            let want: Vec<$e> = $input.iter().map(|&x| x.rotate_right($imm as u32)).collect();
            harness::assert_lanes_eq(
                &format!("{} [rori<{}>]", $l, stringify!($imm)),
                &[$input.as_slice()], &got, &want, Tol::Exact);
        }
    )*};
}

/// Byte/bit reversal and rotates, correct for every width.
macro_rules! bitperm_for {
    ($S:ty; $($reg:ident: $e:ty),* $(,)?) => {
        $(
            oracle_unary!(l!($S, $reg), <$S as Simd>::$reg, $e, swap_bytes, |x| x.swap_bytes(), Tol::Exact);
            oracle_unary!(l!($S, $reg), <$S as Simd>::$reg, $e, reverse_bits, |x| x.reverse_bits(), Tol::Exact);
            oracle_shift!(l!($S, $reg), <$S as Simd>::$reg, $e, rol, |x, s| x.rotate_left(s));
            oracle_shift!(l!($S, $reg), <$S as Simd>::$reg, $e, ror, |x, s| x.rotate_right(s));
        )*
    };
}

/// Rotates by an amount >= the element width, and the const-generic
/// `roli`/`rori` forms.
///
/// Rust's `rotate_left`/`rotate_right` (and hence the scalar backend,
/// which literally _is_ those) reduce the amount modulo the width. An
/// unmasked `shr(v, width - shift)` underflows for `shift >= width` and
/// makes every vector backend return zeros, a silent divergence from the
/// scalar oracle. This pins the masked amount down.
macro_rules! rotate_wrap_for {
    ($S:ty; $($reg:ident: $e:ty),* $(,)?) => {
        $({
            let label = l!($S, $reg);
            let bits = (core::mem::size_of::<$e>() * 8) as u32;
            let lanes = <<<$S as Simd>::$reg as thermite::register::CoreRegister>::Lanes
                as generic_array::typenum::Unsigned>::USIZE;
            let mut rng = harness::rng();

            for input in harness::corpus::<$e>(lanes, &mut rng) {
                let reg = harness::make_array::<<$S as Simd>::$reg>(&input);

                // out-of-range amounts, including exactly `bits` and past it
                for extra in [0u32, 1, 3, bits / 2] {
                    let sh = bits + extra;
                    let got = harness::read::<<$S as Simd>::$reg>(&<<$S as Simd>::$reg>::rol(reg, sh));
                    let want: Vec<$e> = input.iter().map(|&x| x.rotate_left(sh)).collect();
                    harness::assert_lanes_eq(
                        &format!("{label} [rol out-of-range]"), &[input.as_slice()], &got, &want, Tol::Exact);

                    let got = harness::read::<<$S as Simd>::$reg>(&<<$S as Simd>::$reg>::ror(reg, sh));
                    let want: Vec<$e> = input.iter().map(|&x| x.rotate_right(sh)).collect();
                    harness::assert_lanes_eq(
                        &format!("{label} [ror out-of-range]"), &[input.as_slice()], &got, &want, Tol::Exact);
                }

                // const-generic forms (a representative spread, incl. 0)
                check_rot_const!(label, <$S as Simd>::$reg, $e, input, reg, bits; 0, 1, 3, 7, 8, 15, 16, 31, 32, 63);
            }
        })*
    };
}

macro_rules! for_lztz {
    ($S:ty; $($reg:ident: $e:ty),* $(,)?) => {$(
        oracle_unary!(l!($S, $reg), <$S as Simd>::$reg, $e, leading_zeros, |x| x.leading_zeros() as $e, Tol::Exact);
        oracle_unary!(l!($S, $reg), <$S as Simd>::$reg, $e, trailing_zeros, |x| x.trailing_zeros() as $e, Tol::Exact);
    )*};
}
macro_rules! for_signed {
    ($S:ty; $($reg:ident: $e:ty, $w:ty),* $(,)?) => {$(
        oracle_shift!(l!($S, $reg), <$S as Simd>::$reg, $e, sra, |x, s| x >> s);
        oracle_binary!(l!($S, $reg), <$S as Simd>::$reg, $e, avg_floor,
            |a, b| (((a as $w) + (b as $w)) >> 1) as $e, Tol::Exact);
        oracle_binary!(l!($S, $reg), <$S as Simd>::$reg, $e, avg_ceil,
            |a, b| (((a as $w) + (b as $w) + 1) >> 1) as $e, Tol::Exact);
    )*};
}
macro_rules! for_unsigned_avg {
    ($S:ty; $($reg:ident: $e:ty, $w:ty),* $(,)?) => {$(
        oracle_binary!(l!($S, $reg), <$S as Simd>::$reg, $e, avg,
            |a, b| (((a as $w) + (b as $w) + 1) >> 1) as $e, Tol::Exact);
    )*};
}
macro_rules! for_float {
    ($S:ty; $($reg:ident: $e:ty),* $(,)?) => {$(
        oracle_binary!(l!($S, $reg), <$S as Simd>::$reg, $e, copysign, |a, b| a.copysign(b), Tol::Exact);
        oracle_unary!(l!($S, $reg), <$S as Simd>::$reg, $e, fract, |x| x - x.trunc(), Tol::Exact);
    )*};
}
macro_rules! for_rounding {
    ($S:ty; $($reg:ident: $e:ty),* $(,)?) => {$(
        oracle_unary!(l!($S, $reg), <$S as Simd>::$reg, $e, floor, |x: $e| x.floor(), Tol::Exact);
        oracle_unary!(l!($S, $reg), <$S as Simd>::$reg, $e, ceil, |x: $e| x.ceil(), Tol::Exact);
        oracle_unary!(l!($S, $reg), <$S as Simd>::$reg, $e, trunc, |x: $e| x.trunc(), Tol::Exact);
        oracle_unary!(l!($S, $reg), <$S as Simd>::$reg, $e, round, |x: $e| x.round_ties_even(), Tol::Exact);
    )*};
}
macro_rules! for_mulhi {
    ($S:ty; $($reg:ident: $e:ty, $w:ty),* $(,)?) => {$(
        oracle_binary!(l!($S, $reg), <$S as Simd>::$reg, $e, mulhi,
            |a, b| (((a as $w) * (b as $w)) >> (core::mem::size_of::<$e>() * 8)) as $e, Tol::Exact);
    )*};
}
macro_rules! for_saturating {
    ($S:ty; $($reg:ident: $e:ty),* $(,)?) => {$(
        oracle_binary!(l!($S, $reg), <$S as Simd>::$reg, $e, saturating_add, |a, b| a.saturating_add(b), Tol::Exact);
        oracle_binary!(l!($S, $reg), <$S as Simd>::$reg, $e, saturating_sub, |a, b| a.saturating_sub(b), Tol::Exact);
    )*};
}

// ===========================================================================
// Generic `_limited` f64 <-> 64-bit-int magic-number conversions.
// ===========================================================================

/// Integer-valued probes spanning the signed domain `[-2^51, 2^51]`, weighted
/// toward the negatives and the endpoints.
///
/// Negatives are the whole point: the bias trick is `x + 1.5 * 2^52`, and the
/// non-negative half of the domain agrees with the oracle under *either* a
/// correct unbias or a wrong one, so a corpus of naturals proves nothing here.
const LIMITED_SIGNED: &[i64] = &[
    0,
    1,
    -1,
    2,
    -2,
    3,
    -3,
    -42,
    42,
    -1000,
    1000,
    1 << 31,
    -(1 << 31),
    1 << 32,
    -(1 << 32),
    1 << 50,
    -(1 << 50),
    (1 << 51) - 1,
    -((1 << 51) - 1),
    1 << 51,
    -(1 << 51),
];

/// Integer-valued probes spanning the unsigned domain `[0, 2^52)`.
const LIMITED_UNSIGNED: &[u64] = &[
    0,
    1,
    2,
    3,
    42,
    1000,
    1 << 31,
    1 << 32,
    1 << 50,
    1 << 51,
    (1 << 52) - 2,
    (1 << 52) - 1,
];

/// The four `backend/generic/polyfills/casts.rs` magic-number conversions,
/// checked against a pure-Rust oracle on **every** backend's f64 registers -
/// not only the one backend that routes to them in production.
///
/// That last part is the point of this test existing. These four are generic
/// over `R`, but the wasm backend is the only caller, so until now they were
/// never executed on x86 at all.
macro_rules! limited_casts {
    ($S:ty, $f:ident, $i:ident, $u:ident) => {{
        use generic_array::typenum::Unsigned;
        use thermite::backend::generic::polyfills::casts as gc;
        use thermite::register::CoreRegister;

        let lanes = <<<$S as Simd>::$f as CoreRegister>::Lanes as Unsigned>::USIZE;
        let label = l!($S, $f);

        for start in 0..LIMITED_SIGNED.len() {
            let ints: Vec<i64> = (0..lanes)
                .map(|k| LIMITED_SIGNED[(start + k) % LIMITED_SIGNED.len()])
                .collect();
            let floats: Vec<f64> = ints.iter().map(|&x| x as f64).collect();

            let got = harness::read::<<$S as Simd>::$i>(&gc::convert_pd_epi64_limited::<<$S as Simd>::$f>(
                harness::make_array::<<$S as Simd>::$f>(&floats),
            ));
            harness::assert_lanes_eq(
                &format!("convert_pd_epi64_limited {label}"),
                &[ints.as_slice()],
                &got,
                &ints,
                Tol::Exact,
            );

            let got = harness::read::<<$S as Simd>::$f>(&gc::convert_epi64_pd_limited::<<$S as Simd>::$f>(
                harness::make_array::<<$S as Simd>::$i>(&ints),
            ));
            harness::assert_lanes_eq(
                &format!("convert_epi64_pd_limited {label}"),
                &[floats.as_slice()],
                &got,
                &floats,
                Tol::Exact,
            );

            let rt = harness::read::<<$S as Simd>::$i>(&gc::convert_pd_epi64_limited::<<$S as Simd>::$f>(
                gc::convert_epi64_pd_limited::<<$S as Simd>::$f>(harness::make_array::<<$S as Simd>::$i>(&ints)),
            ));
            harness::assert_lanes_eq(
                &format!("limited i64 round trip {label}"),
                &[ints.as_slice()],
                &rt,
                &ints,
                Tol::Exact,
            );
        }

        for start in 0..LIMITED_UNSIGNED.len() {
            let ints: Vec<u64> = (0..lanes)
                .map(|k| LIMITED_UNSIGNED[(start + k) % LIMITED_UNSIGNED.len()])
                .collect();
            let floats: Vec<f64> = ints.iter().map(|&x| x as f64).collect();

            let got = harness::read::<<$S as Simd>::$u>(&gc::convert_pd_epu64_limited::<<$S as Simd>::$f>(
                harness::make_array::<<$S as Simd>::$f>(&floats),
            ));
            harness::assert_lanes_eq(
                &format!("convert_pd_epu64_limited {label}"),
                &[],
                &got,
                &ints,
                Tol::Exact,
            );

            let got = harness::read::<<$S as Simd>::$f>(&gc::convert_epu64_pd_limited::<<$S as Simd>::$f>(
                harness::make_array::<<$S as Simd>::$u>(&ints),
            ));
            harness::assert_lanes_eq(
                &format!("convert_epu64_pd_limited {label}"),
                &[floats.as_slice()],
                &got,
                &floats,
                Tol::Exact,
            );

            let rt = harness::read::<<$S as Simd>::$u>(&gc::convert_pd_epu64_limited::<<$S as Simd>::$f>(
                gc::convert_epu64_pd_limited::<<$S as Simd>::$f>(harness::make_array::<<$S as Simd>::$u>(&ints)),
            ));
            harness::assert_lanes_eq(
                &format!("limited u64 round trip {label}"),
                &[ints.as_slice()],
                &rt,
                &ints,
                Tol::Exact,
            );
        }
    }};
}

for_each_backend! {
    fn mullo<S: Simd>() {
        mullo_for!(S; i32x4: i32, i32x8: i32, i32x16: i32, u32x4: u32, u32x8: u32, u32x16: u32,
            i64x2: i64, i64x4: i64, i64x8: i64, u64x2: u64, u64x4: u64, u64x8: u64);
    }
    fn popcount<S: Simd>() {
        popcount_for!(S; i32x4: i32, i32x8: i32, i32x16: i32, u32x4: u32, u32x8: u32, u32x16: u32,
            i64x2: i64, i64x4: i64, i64x8: i64, u64x2: u64, u64x4: u64, u64x8: u64);
    }
    fn bitperm<S: Simd>() {
        bitperm_for!(S; i32x4: i32, i32x8: i32, i32x16: i32, u32x4: u32, u32x8: u32, u32x16: u32,
            i64x2: i64, i64x4: i64, i64x8: i64, u64x2: u64, u64x4: u64, u64x8: u64);
    }
    fn rotate_wraparound_and_const<S: Simd>() {
        rotate_wrap_for!(S; i32x4: i32, i32x8: i32, u32x4: u32, u32x8: u32,
            i64x2: i64, i64x4: i64, u64x2: u64, u64x4: u64);
    }
    /// 32-bit `mulhi` (P1) and the 64-bit limb decompositions.
    fn mulhi<S: Simd>() {
        for_mulhi!(S; i32x4: i32, i64, u32x4: u32, u64, i32x8: i32, i64, u32x8: u32, u64,
            i32x16: i32, i64, u32x16: u32, u64,
            i64x2: i64, i128, u64x2: u64, u128, i64x4: i64, i128, u64x4: u64, u128,
            i64x8: i64, i128, u64x8: u64, u128);
    }
    /// Signed (P2) and unsigned saturating add/sub.
    fn saturating<S: Simd>() {
        for_saturating!(S; i32x4: i32, i32x8: i32, i32x16: i32, u32x4: u32, u32x8: u32, u32x16: u32,
            i64x2: i64, i64x4: i64, i64x8: i64, u64x2: u64, u64x4: u64, u64x8: u64);
    }
    /// leading/trailing zeros, incl. the u64 forms (P3).
    fn bitscan<S: Simd>() {
        for_lztz!(S; i32x4: i32, i32x8: i32, i32x16: i32, u32x4: u32, u32x8: u32, u32x16: u32,
            i64x2: i64, i64x4: i64, i64x8: i64, u64x2: u64, u64x4: u64, u64x8: u64);
    }
    fn signed_extras<S: Simd>() {
        for_signed!(S; i32x4: i32, i64, i32x8: i32, i64, i32x16: i32, i64,
            i64x2: i64, i128, i64x4: i64, i128, i64x8: i64, i128);
    }
    fn unsigned_avg<S: Simd>() {
        for_unsigned_avg!(S; u32x4: u32, u64, u32x8: u32, u64, u32x16: u32, u64,
            u64x2: u64, u128, u64x4: u64, u128, u64x8: u64, u128);
    }
    fn float_ops<S: Simd>() {
        for_float!(S; f32x4: f32, f32x8: f32, f32x16: f32, f64x2: f64, f64x4: f64, f64x8: f64);
    }
    /// floor/ceil/trunc/round vs Rust (round = ties-to-even, deliberately
    /// unlike std's `round`, and the SSE2 forms are polyfilled).
    fn rounding<S: Simd>() {
        for_rounding!(S; f32x4: f32, f32x8: f32, f64x2: f64, f64x4: f64);
    }
    /// P5: integer `copysign`.
    fn fixed_copysign_int<S: Simd>() {
        fn cs32(a: i32, b: i32) -> i32 {
            if (a < 0) != (b < 0) { a.wrapping_neg() } else { a }
        }
        fn cs64(a: i64, b: i64) -> i64 {
            if (a < 0) != (b < 0) { a.wrapping_neg() } else { a }
        }
        oracle_binary!(l!(S, i32x4), <S as Simd>::i32x4, i32, copysign, cs32, Tol::Exact);
        oracle_binary!(l!(S, i32x8), <S as Simd>::i32x8, i32, copysign, cs32, Tol::Exact);
        oracle_binary!(l!(S, i64x2), <S as Simd>::i64x2, i64, copysign, cs64, Tol::Exact);
        oracle_binary!(l!(S, i64x4), <S as Simd>::i64x4, i64, copysign, cs64, Tol::Exact);
    }
    /// P6: the reduced `u32x2 -> u64x2` widen.
    fn fixed_cast_u32x2_to_u64x2<S: Simd>() {
        use thermite::register::CastRegister;

        let label = l!(S, u32x2);
        let mut rng = harness::rng();
        for input in harness::corpus::<u32>(2, &mut rng) {
            let half = harness::make_array::<<S as Simd>::u32x2>(&input);
            let full = <<S as Simd>::u64x2 as CastRegister<<S as Simd>::u32x2>>::cast_from(half);

            let got = harness::read::<<S as Simd>::u64x2>(&full);
            let want: Vec<u64> = input.iter().map(|&x| x as u64).collect();

            harness::assert_lanes_eq(
                &format!("{label} [u32x2 -> u64x2 cast]"),
                &[want.as_slice()],
                &got,
                &want,
                Tol::Exact,
            );
        }
    }
    fn limited_casts_x2<S: Simd>() {
        limited_casts!(S, f64x2, i64x2, u64x2);
    }
    fn limited_casts_x4<S: Simd>() {
        limited_casts!(S, f64x4, i64x4, u64x4);
    }
}
