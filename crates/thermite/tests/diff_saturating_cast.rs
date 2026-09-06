//! Saturating-narrow cast (`CastRegister`) differential audit.
//!
//! The x86 backends lower these to the hardware saturating pack instructions
//! (`vpackssdw`/`vpacksswb`, and `vpminu* + vpackus*` for the unsigned-source
//! pairs. SSE2 has no `packusdw`, so u32 sources clamp + truncate there. WASM
//! and NEON use their narrowing forms. AVX-512 has `vpmovs*`/`vpmovus*`). The
//! scalar backend's `saturating_cast_from` is the reference oracle
//! (`value.clamp(INTO::MIN, INTO::MAX) as INTO`), so each test is a differential
//! of the pack path against the clamp-then-truncate path on an identical corpus.
//! The corpus spans the full source range, so out-of-range inputs (which is what
//! exercises the saturation) dominate.
//!
//! Every (source, destination) pair the `Simd` grid requires is run on every
//! backend. The pairs are grouped by source width.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use harness::Tol;
use thermite::backend::scalar::Scalar;
use thermite::register::{CastRegister, CoreRegister};
use thermite::simd::Simd;

/// One saturating-narrow pair: backend pack vs scalar clamp oracle, bit-exact.
/// `$se` is the source element type the corpus is generated over.
macro_rules! sat_diff {
    ($S:ty, $src:ident, $dst:ident, $se:ty) => {{
        let label = harness::label::<$S>(concat!(stringify!($src), "->", stringify!($dst)));
        let mut rng = harness::rng();
        let lanes = <<<$S as Simd>::$src as CoreRegister>::Lanes as generic_array::typenum::Unsigned>::USIZE;
        for input in harness::corpus::<$se>(lanes, &mut rng) {
            let got =
                harness::read::<<$S as Simd>::$dst>(
                    &<<$S as Simd>::$dst as CastRegister<<$S as Simd>::$src>>::saturating_cast_from(
                        harness::make_array::<<$S as Simd>::$src>(&input),
                    ),
                );
            let want = harness::read::<<Scalar as Simd>::$dst>(&<<Scalar as Simd>::$dst as CastRegister<
                <Scalar as Simd>::$src,
            >>::saturating_cast_from(
                harness::make_array::<<Scalar as Simd>::$src>(&input),
            ));
            harness::assert_lanes_eq(
                &format!("{label} [sat cast vs scalar clamp]"),
                &[],
                &got,
                &want,
                Tol::Exact,
            );
        }
    }};
}

for_each_backend! {
    /// 16 -> 8 (single- and two-source packs, plus the reduced x4/x2 widths).
    fn from_16<S: Simd>() {
        sat_diff!(S, i16x16, i8x16, i16);
        sat_diff!(S, i16x8, i8x8, i16);
        sat_diff!(S, i16x4, i8x4, i16);
        sat_diff!(S, u16x16, u8x16, u16);
        sat_diff!(S, u16x8, u8x8, u16);
        sat_diff!(S, u16x4, u8x4, u16);
    }

    /// 32 -> 16 and the skip-level 32 -> 8 (composed packs).
    fn from_32<S: Simd>() {
        sat_diff!(S, i32x16, i16x16, i32);
        sat_diff!(S, i32x8, i16x8, i32);
        sat_diff!(S, i32x4, i16x4, i32);
        sat_diff!(S, i32x2, i16x2, i32);
        sat_diff!(S, u32x16, u16x16, u32);
        sat_diff!(S, u32x8, u16x8, u32);
        sat_diff!(S, u32x4, u16x4, u32);
        sat_diff!(S, u32x2, u16x2, u32);

        sat_diff!(S, i32x16, i8x16, i32);
        sat_diff!(S, i32x8, i8x8, i32);
        sat_diff!(S, i32x4, i8x4, i32);
        sat_diff!(S, i32x2, i8x2, i32);
        sat_diff!(S, u32x16, u8x16, u32);
        sat_diff!(S, u32x8, u8x8, u32);
        sat_diff!(S, u32x4, u8x4, u32);
        sat_diff!(S, u32x2, u8x2, u32);
    }

    /// 64 -> 32/16/8 (clamp + narrow, no 64-bit pack before AVX-512).
    fn from_64<S: Simd>() {
        sat_diff!(S, i64x16, i32x16, i64);
        sat_diff!(S, i64x8, i32x8, i64);
        sat_diff!(S, i64x4, i32x4, i64);
        sat_diff!(S, i64x2, i32x2, i64);
        sat_diff!(S, u64x16, u32x16, u64);
        sat_diff!(S, u64x8, u32x8, u64);
        sat_diff!(S, u64x4, u32x4, u64);
        sat_diff!(S, u64x2, u32x2, u64);

        sat_diff!(S, i64x16, i16x16, i64);
        sat_diff!(S, i64x8, i16x8, i64);
        sat_diff!(S, i64x4, i16x4, i64);
        sat_diff!(S, i64x2, i16x2, i64);
        sat_diff!(S, u64x16, u16x16, u64);
        sat_diff!(S, u64x8, u16x8, u64);
        sat_diff!(S, u64x4, u16x4, u64);
        sat_diff!(S, u64x2, u16x2, u64);

        sat_diff!(S, i64x16, i8x16, i64);
        sat_diff!(S, i64x8, i8x8, i64);
        sat_diff!(S, i64x4, i8x4, i64);
        sat_diff!(S, i64x2, i8x2, i64);
        sat_diff!(S, u64x16, u8x16, u64);
        sat_diff!(S, u64x8, u8x8, u64);
        sat_diff!(S, u64x4, u8x4, u64);
        sat_diff!(S, u64x2, u8x2, u64);
    }
}
