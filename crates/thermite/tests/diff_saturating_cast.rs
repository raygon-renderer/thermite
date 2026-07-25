//! Saturating-narrow cast (`SaturatingCastRegister`) differential audit.
//!
//! The x86 backends lower these to the hardware saturating pack instructions
//! (`vpackssdw`/`vpacksswb`, and `vpminu* + vpackus*` for the unsigned-source
//! pairs). The scalar backend's `saturating_cast_from` is the reference oracle
//! (`value.clamp(INTO::MIN, INTO::MAX) as INTO`), so each test is a differential
//! of the pack path against the clamp-then-truncate path on an identical corpus.
//! The corpus spans the full source range, so out-of-range inputs (which is what
//! exercises the saturation) dominate.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use harness::Tol;
use thermite::backend::scalar::Scalar;
use thermite::register::{CoreRegister, SaturatingCastRegister};
use thermite::simd::Simd;

/// One saturating-narrow pair: x86 backend pack vs scalar clamp oracle, bit-exact.
/// `$se` is the source element type the corpus is generated over.
macro_rules! sat_diff {
    ($label:expr, $b:ty, $src:ident, $dst:ident, $se:ty) => {{
        let mut rng = harness::rng();
        let lanes = <<<$b as Simd>::$src as CoreRegister>::Lanes as generic_array::typenum::Unsigned>::USIZE;
        for input in harness::corpus::<$se>(lanes, &mut rng) {
            let got = harness::read::<<$b as Simd>::$dst>(&<<$b as Simd>::$dst as SaturatingCastRegister<
                <$b as Simd>::$src,
            >>::saturating_cast_from(harness::make_array::<
                <$b as Simd>::$src,
            >(&input)));
            let want = harness::read::<<Scalar as Simd>::$dst>(&<<Scalar as Simd>::$dst as SaturatingCastRegister<
                <Scalar as Simd>::$src,
            >>::saturating_cast_from(
                harness::make_array::<<Scalar as Simd>::$src>(&input),
            ));
            harness::assert_lanes_eq(
                concat!($label, " [sat cast vs scalar clamp]"),
                &[],
                &got,
                &want,
                Tol::Exact,
            );
        }
    }};
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    use thermite::backend::x86_v1::X86V1;
    use thermite::backend::x86_v2::X86V2;
    use thermite::backend::x86_v3::X86V3;

    #[test]
    fn v3_saturating_narrows() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        // signed: vpackssdw / vpacksswb
        sat_diff!("v3 i32x8->i16x8", X86V3, i32x8, i16x8, i32);
        sat_diff!("v3 i16x16->i8x16", X86V3, i16x16, i8x16, i16);
        sat_diff!("v3 i32x4->i16x4", X86V3, i32x4, i16x4, i32);
        sat_diff!("v3 i16x8->i8x8", X86V3, i16x8, i8x8, i16);

        // unsigned: vpminu* + vpackus*
        sat_diff!("v3 u32x8->u16x8", X86V3, u32x8, u16x8, u32);
        sat_diff!("v3 u16x16->u8x16", X86V3, u16x16, u8x16, u16);
        sat_diff!("v3 u32x4->u16x4", X86V3, u32x4, u16x4, u32);
        sat_diff!("v3 u16x8->u8x8", X86V3, u16x8, u8x8, u16);

        // adjacent 16 -> 8 reduced widths
        sat_diff!("v3 i16x4->i8x4", X86V3, i16x4, i8x4, i16);
        sat_diff!("v3 u16x4->u8x4", X86V3, u16x4, u8x4, u16);

        // skip-level 32 -> 8 (composed packs)
        sat_diff!("v3 i32x4->i8x4", X86V3, i32x4, i8x4, i32);
        sat_diff!("v3 i32x8->i8x8", X86V3, i32x8, i8x8, i32);
        sat_diff!("v3 u32x4->u8x4", X86V3, u32x4, u8x4, u32);
        sat_diff!("v3 u32x8->u8x8", X86V3, u32x8, u8x8, u32);

        // 64 -> 32 (clamp + narrow; no AVX2 64-bit pack)
        sat_diff!("v3 i64x2->i32x2", X86V3, i64x2, i32x2, i64);
        sat_diff!("v3 u64x2->u32x2", X86V3, u64x2, u32x2, u64);

        // pack-less i64 / sub-native x2 combos (clamp + truncating narrow)
        sat_diff!("v3 i64x4->i16x4", X86V3, i64x4, i16x4, i64);
        sat_diff!("v3 i64x4->i8x4", X86V3, i64x4, i8x4, i64);
        sat_diff!("v3 u64x4->u16x4", X86V3, u64x4, u16x4, u64);
        sat_diff!("v3 u64x4->u8x4", X86V3, u64x4, u8x4, u64);
        sat_diff!("v3 i32x2->i16x2", X86V3, i32x2, i16x2, i32);
        sat_diff!("v3 i32x2->i8x2", X86V3, i32x2, i8x2, i32);
        sat_diff!("v3 u32x2->u16x2", X86V3, u32x2, u16x2, u32);
        sat_diff!("v3 u32x2->u8x2", X86V3, u32x2, u8x2, u32);
        sat_diff!("v3 i64x2->i16x2", X86V3, i64x2, i16x2, i64);
        sat_diff!("v3 i64x2->i8x2", X86V3, i64x2, i8x2, i64);
        sat_diff!("v3 u64x2->u16x2", X86V3, u64x2, u16x2, u64);
        sat_diff!("v3 u64x2->u8x2", X86V3, u64x2, u8x2, u64);

        // wide emulated narrows: 32->16/8 via packs, 64->* via clamp-to-i32 + composed packs
        sat_diff!("v3 i64x4->i32x4", X86V3, i64x4, i32x4, i64);
        sat_diff!("v3 i64x8->i32x8", X86V3, i64x8, i32x8, i64);
        sat_diff!("v3 i64x16->i32x16", X86V3, i64x16, i32x16, i64);
        sat_diff!("v3 i32x16->i16x16", X86V3, i32x16, i16x16, i32);
        sat_diff!("v3 i32x16->i8x16", X86V3, i32x16, i8x16, i32);
        sat_diff!("v3 i64x8->i16x8", X86V3, i64x8, i16x8, i64);
        sat_diff!("v3 i64x16->i16x16", X86V3, i64x16, i16x16, i64);
        sat_diff!("v3 i64x16->i8x16", X86V3, i64x16, i8x16, i64);
        sat_diff!("v3 i64x8->i8x8", X86V3, i64x8, i8x8, i64);
        sat_diff!("v3 u64x4->u32x4", X86V3, u64x4, u32x4, u64);
        sat_diff!("v3 u64x8->u32x8", X86V3, u64x8, u32x8, u64);
        sat_diff!("v3 u64x16->u32x16", X86V3, u64x16, u32x16, u64);
        sat_diff!("v3 u32x16->u16x16", X86V3, u32x16, u16x16, u32);
        sat_diff!("v3 u32x16->u8x16", X86V3, u32x16, u8x16, u32);
        sat_diff!("v3 u64x8->u16x8", X86V3, u64x8, u16x8, u64);
        sat_diff!("v3 u64x16->u16x16", X86V3, u64x16, u16x16, u64);
        sat_diff!("v3 u64x16->u8x16", X86V3, u64x16, u8x16, u64);
        sat_diff!("v3 u64x8->u8x8", X86V3, u64x8, u8x8, u64);
    }

    #[test]
    fn v2_saturating_narrows() {
        if !is_x86_feature_detected!("sse4.2") {
            return;
        }

        // signed: packssdw / packsswb (single- and two-source 128-bit packs)
        sat_diff!("v2 i32x8->i16x8", X86V2, i32x8, i16x8, i32);
        sat_diff!("v2 i16x16->i8x16", X86V2, i16x16, i8x16, i16);
        sat_diff!("v2 i32x4->i16x4", X86V2, i32x4, i16x4, i32);
        sat_diff!("v2 i16x8->i8x8", X86V2, i16x8, i8x8, i16);
        sat_diff!("v2 i16x4->i8x4", X86V2, i16x4, i8x4, i16);

        // unsigned: pminu* + packus*
        sat_diff!("v2 u32x8->u16x8", X86V2, u32x8, u16x8, u32);
        sat_diff!("v2 u16x16->u8x16", X86V2, u16x16, u8x16, u16);
        sat_diff!("v2 u32x4->u16x4", X86V2, u32x4, u16x4, u32);
        sat_diff!("v2 u16x8->u8x8", X86V2, u16x8, u8x8, u16);
        sat_diff!("v2 u16x4->u8x4", X86V2, u16x4, u8x4, u16);

        // skip-level 32 -> 8 (composed packs)
        sat_diff!("v2 i32x4->i8x4", X86V2, i32x4, i8x4, i32);
        sat_diff!("v2 i32x8->i8x8", X86V2, i32x8, i8x8, i32);
        sat_diff!("v2 i32x16->i8x16", X86V2, i32x16, i8x16, i32);
        sat_diff!("v2 u32x4->u8x4", X86V2, u32x4, u8x4, u32);
        sat_diff!("v2 u32x8->u8x8", X86V2, u32x8, u8x8, u32);
        sat_diff!("v2 u32x16->u8x16", X86V2, u32x16, u8x16, u32);

        // 64 -> 32 (clamp + narrow)
        sat_diff!("v2 i64x2->i32x2", X86V2, i64x2, i32x2, i64);
        sat_diff!("v2 i64x4->i32x4", X86V2, i64x4, i32x4, i64);
        sat_diff!("v2 u64x2->u32x2", X86V2, u64x2, u32x2, u64);
        sat_diff!("v2 u64x4->u32x4", X86V2, u64x4, u32x4, u64);

        // pack-less i64 -> 16/8 and sub-native x2 combos (clamp + truncating narrow)
        sat_diff!("v2 i64x4->i16x4", X86V2, i64x4, i16x4, i64);
        sat_diff!("v2 i64x4->i8x4", X86V2, i64x4, i8x4, i64);
        sat_diff!("v2 i64x8->i16x8", X86V2, i64x8, i16x8, i64);
        sat_diff!("v2 i64x8->i8x8", X86V2, i64x8, i8x8, i64);
        sat_diff!("v2 i64x16->i8x16", X86V2, i64x16, i8x16, i64);
        sat_diff!("v2 i64x16->i16x16", X86V2, i64x16, i16x16, i64);
        sat_diff!("v2 i32x2->i16x2", X86V2, i32x2, i16x2, i32);
        sat_diff!("v2 i32x2->i8x2", X86V2, i32x2, i8x2, i32);
        sat_diff!("v2 i64x2->i16x2", X86V2, i64x2, i16x2, i64);
        sat_diff!("v2 i64x2->i8x2", X86V2, i64x2, i8x2, i64);
        sat_diff!("v2 u64x4->u16x4", X86V2, u64x4, u16x4, u64);
        sat_diff!("v2 u64x4->u8x4", X86V2, u64x4, u8x4, u64);
        sat_diff!("v2 u64x8->u16x8", X86V2, u64x8, u16x8, u64);
        sat_diff!("v2 u64x8->u8x8", X86V2, u64x8, u8x8, u64);
        sat_diff!("v2 u64x16->u8x16", X86V2, u64x16, u8x16, u64);
        sat_diff!("v2 u64x16->u16x16", X86V2, u64x16, u16x16, u64);
        sat_diff!("v2 u32x2->u16x2", X86V2, u32x2, u16x2, u32);
        sat_diff!("v2 u32x2->u8x2", X86V2, u32x2, u8x2, u32);
        sat_diff!("v2 u64x2->u16x2", X86V2, u64x2, u16x2, u64);
        sat_diff!("v2 u64x2->u8x2", X86V2, u64x2, u8x2, u64);
    }

    #[test]
    fn v1_saturating_narrows() {
        if !is_x86_feature_detected!("sse2") {
            return;
        }

        // signed: packssdw / packsswb (SSE2; single, two-source, composed)
        sat_diff!("v1 i32x8->i16x8", X86V1, i32x8, i16x8, i32);
        sat_diff!("v1 i16x16->i8x16", X86V1, i16x16, i8x16, i16);
        sat_diff!("v1 i32x4->i16x4", X86V1, i32x4, i16x4, i32);
        sat_diff!("v1 i16x8->i8x8", X86V1, i16x8, i8x8, i16);
        sat_diff!("v1 i16x4->i8x4", X86V1, i16x4, i8x4, i16);
        sat_diff!("v1 i32x4->i8x4", X86V1, i32x4, i8x4, i32);
        sat_diff!("v1 i32x8->i8x8", X86V1, i32x8, i8x8, i32);
        sat_diff!("v1 i32x16->i8x16", X86V1, i32x16, i8x16, i32);

        // unsigned u16->u8: min_epu16x_v1 + packuswb (SSE2)
        sat_diff!("v1 u16x16->u8x16", X86V1, u16x16, u8x16, u16);
        sat_diff!("v1 u16x8->u8x8", X86V1, u16x8, u8x8, u16);
        sat_diff!("v1 u16x4->u8x4", X86V1, u16x4, u8x4, u16);

        // unsigned u32->* : no packusdw on SSE2, so clamp + truncating narrow
        sat_diff!("v1 u32x4->u16x4", X86V1, u32x4, u16x4, u32);
        sat_diff!("v1 u32x8->u16x8", X86V1, u32x8, u16x8, u32);
        sat_diff!("v1 u32x4->u8x4", X86V1, u32x4, u8x4, u32);
        sat_diff!("v1 u32x8->u8x8", X86V1, u32x8, u8x8, u32);
        sat_diff!("v1 u32x16->u8x16", X86V1, u32x16, u8x16, u32);

        // 64 -> 32 (clamp + narrow)
        sat_diff!("v1 i64x2->i32x2", X86V1, i64x2, i32x2, i64);
        sat_diff!("v1 i64x4->i32x4", X86V1, i64x4, i32x4, i64);
        sat_diff!("v1 u64x2->u32x2", X86V1, u64x2, u32x2, u64);
        sat_diff!("v1 u64x4->u32x4", X86V1, u64x4, u32x4, u64);

        // pack-less i64 -> 16/8 and sub-native x2 combos (clamp + truncating narrow)
        sat_diff!("v1 i64x4->i16x4", X86V1, i64x4, i16x4, i64);
        sat_diff!("v1 i64x4->i8x4", X86V1, i64x4, i8x4, i64);
        sat_diff!("v1 i64x8->i16x8", X86V1, i64x8, i16x8, i64);
        sat_diff!("v1 i64x8->i8x8", X86V1, i64x8, i8x8, i64);
        sat_diff!("v1 i64x16->i8x16", X86V1, i64x16, i8x16, i64);
        sat_diff!("v1 i64x16->i16x16", X86V1, i64x16, i16x16, i64);
        sat_diff!("v1 i32x2->i16x2", X86V1, i32x2, i16x2, i32);
        sat_diff!("v1 i32x2->i8x2", X86V1, i32x2, i8x2, i32);
        sat_diff!("v1 i64x2->i16x2", X86V1, i64x2, i16x2, i64);
        sat_diff!("v1 i64x2->i8x2", X86V1, i64x2, i8x2, i64);
        sat_diff!("v1 u64x4->u16x4", X86V1, u64x4, u16x4, u64);
        sat_diff!("v1 u64x4->u8x4", X86V1, u64x4, u8x4, u64);
        sat_diff!("v1 u64x8->u16x8", X86V1, u64x8, u16x8, u64);
        sat_diff!("v1 u64x8->u8x8", X86V1, u64x8, u8x8, u64);
        sat_diff!("v1 u64x16->u8x16", X86V1, u64x16, u8x16, u64);
        sat_diff!("v1 u64x16->u16x16", X86V1, u64x16, u16x16, u64);
        sat_diff!("v1 u32x2->u16x2", X86V1, u32x2, u16x2, u32);
        sat_diff!("v1 u32x2->u8x2", X86V1, u32x2, u8x2, u32);
        sat_diff!("v1 u64x2->u16x2", X86V1, u64x2, u16x2, u64);
        sat_diff!("v1 u64x2->u8x2", X86V1, u64x2, u8x2, u64);
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use thermite::backend::wasm::Wasm;

    #[test]
    fn wasm_saturating_narrows() {
        // signed: i16x8.narrow_i32x4_s / i8x16.narrow_i16x8_s (single, two-source, composed)
        sat_diff!("wasm i32x8->i16x8", Wasm, i32x8, i16x8, i32);
        sat_diff!("wasm i16x16->i8x16", Wasm, i16x16, i8x16, i16);
        sat_diff!("wasm i32x4->i16x4", Wasm, i32x4, i16x4, i32);
        sat_diff!("wasm i16x8->i8x8", Wasm, i16x8, i8x8, i16);
        sat_diff!("wasm i16x4->i8x4", Wasm, i16x4, i8x4, i16);
        sat_diff!("wasm i32x4->i8x4", Wasm, i32x4, i8x4, i32);
        sat_diff!("wasm i32x8->i8x8", Wasm, i32x8, i8x8, i32);
        sat_diff!("wasm i32x16->i8x16", Wasm, i32x16, i8x16, i32);

        // unsigned: *.min + *.narrow_*_u
        sat_diff!("wasm u32x8->u16x8", Wasm, u32x8, u16x8, u32);
        sat_diff!("wasm u16x16->u8x16", Wasm, u16x16, u8x16, u16);
        sat_diff!("wasm u32x4->u16x4", Wasm, u32x4, u16x4, u32);
        sat_diff!("wasm u16x8->u8x8", Wasm, u16x8, u8x8, u16);
        sat_diff!("wasm u16x4->u8x4", Wasm, u16x4, u8x4, u16);
        sat_diff!("wasm u32x4->u8x4", Wasm, u32x4, u8x4, u32);
        sat_diff!("wasm u32x8->u8x8", Wasm, u32x8, u8x8, u32);
        sat_diff!("wasm u32x16->u8x16", Wasm, u32x16, u8x16, u32);

        // 64 -> 32 (clamp + narrow; no 64-bit narrow on WASM)
        sat_diff!("wasm i64x2->i32x2", Wasm, i64x2, i32x2, i64);
        sat_diff!("wasm i64x4->i32x4", Wasm, i64x4, i32x4, i64);
        sat_diff!("wasm u64x2->u32x2", Wasm, u64x2, u32x2, u64);
        sat_diff!("wasm u64x4->u32x4", Wasm, u64x4, u32x4, u64);

        // pack-less i64 -> 16/8 and sub-native x2 combos (clamp + truncating narrow)
        sat_diff!("wasm i64x4->i16x4", Wasm, i64x4, i16x4, i64);
        sat_diff!("wasm i64x4->i8x4", Wasm, i64x4, i8x4, i64);
        sat_diff!("wasm i64x8->i16x8", Wasm, i64x8, i16x8, i64);
        sat_diff!("wasm i64x8->i8x8", Wasm, i64x8, i8x8, i64);
        sat_diff!("wasm i64x16->i8x16", Wasm, i64x16, i8x16, i64);
        sat_diff!("wasm i64x16->i16x16", Wasm, i64x16, i16x16, i64);
        sat_diff!("wasm i32x2->i16x2", Wasm, i32x2, i16x2, i32);
        sat_diff!("wasm i32x2->i8x2", Wasm, i32x2, i8x2, i32);
        sat_diff!("wasm i64x2->i16x2", Wasm, i64x2, i16x2, i64);
        sat_diff!("wasm i64x2->i8x2", Wasm, i64x2, i8x2, i64);
        sat_diff!("wasm u64x4->u16x4", Wasm, u64x4, u16x4, u64);
        sat_diff!("wasm u64x4->u8x4", Wasm, u64x4, u8x4, u64);
        sat_diff!("wasm u64x8->u16x8", Wasm, u64x8, u16x8, u64);
        sat_diff!("wasm u64x8->u8x8", Wasm, u64x8, u8x8, u64);
        sat_diff!("wasm u64x16->u8x16", Wasm, u64x16, u8x16, u64);
        sat_diff!("wasm u64x16->u16x16", Wasm, u64x16, u16x16, u64);
        sat_diff!("wasm u32x2->u16x2", Wasm, u32x2, u16x2, u32);
        sat_diff!("wasm u32x2->u8x2", Wasm, u32x2, u8x2, u32);
        sat_diff!("wasm u64x2->u16x2", Wasm, u64x2, u16x2, u64);
        sat_diff!("wasm u64x2->u8x2", Wasm, u64x2, u8x2, u64);

        // wide emulated narrows generated by `impl_casts!` (cross-N ArrayRegister recursion)
        sat_diff!("wasm i64x8->i32x8", Wasm, i64x8, i32x8, i64);
        sat_diff!("wasm i64x16->i32x16", Wasm, i64x16, i32x16, i64);
        sat_diff!("wasm i32x16->i16x16", Wasm, i32x16, i16x16, i32);
        sat_diff!("wasm u64x8->u32x8", Wasm, u64x8, u32x8, u64);
        sat_diff!("wasm u64x16->u32x16", Wasm, u64x16, u32x16, u64);
        sat_diff!("wasm u32x16->u16x16", Wasm, u32x16, u16x16, u32);
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use thermite::backend::neon::Neon;

    #[test]
    fn neon_saturating_narrows() {
        // signed: i16x8.narrow_i32x4_s / i8x16.narrow_i16x8_s (single, two-source, composed)
        // (inherited from the wasm section; revisit for NEON)
        sat_diff!("neon i32x8->i16x8", Neon, i32x8, i16x8, i32);
        sat_diff!("neon i16x16->i8x16", Neon, i16x16, i8x16, i16);
        sat_diff!("neon i32x4->i16x4", Neon, i32x4, i16x4, i32);
        sat_diff!("neon i16x8->i8x8", Neon, i16x8, i8x8, i16);
        sat_diff!("neon i16x4->i8x4", Neon, i16x4, i8x4, i16);
        sat_diff!("neon i32x4->i8x4", Neon, i32x4, i8x4, i32);
        sat_diff!("neon i32x8->i8x8", Neon, i32x8, i8x8, i32);
        sat_diff!("neon i32x16->i8x16", Neon, i32x16, i8x16, i32);

        // unsigned: *.min + *.narrow_*_u (inherited from the wasm section; revisit for NEON)
        sat_diff!("neon u32x8->u16x8", Neon, u32x8, u16x8, u32);
        sat_diff!("neon u16x16->u8x16", Neon, u16x16, u8x16, u16);
        sat_diff!("neon u32x4->u16x4", Neon, u32x4, u16x4, u32);
        sat_diff!("neon u16x8->u8x8", Neon, u16x8, u8x8, u16);
        sat_diff!("neon u16x4->u8x4", Neon, u16x4, u8x4, u16);
        sat_diff!("neon u32x4->u8x4", Neon, u32x4, u8x4, u32);
        sat_diff!("neon u32x8->u8x8", Neon, u32x8, u8x8, u32);
        sat_diff!("neon u32x16->u8x16", Neon, u32x16, u8x16, u32);

        // 64 -> 32 (clamp + narrow; no 64-bit narrow on WASM)
        // (inherited from the wasm section; revisit for NEON)
        sat_diff!("neon i64x2->i32x2", Neon, i64x2, i32x2, i64);
        sat_diff!("neon i64x4->i32x4", Neon, i64x4, i32x4, i64);
        sat_diff!("neon u64x2->u32x2", Neon, u64x2, u32x2, u64);
        sat_diff!("neon u64x4->u32x4", Neon, u64x4, u32x4, u64);

        // pack-less i64 -> 16/8 and sub-native x2 combos (clamp + truncating narrow)
        sat_diff!("neon i64x4->i16x4", Neon, i64x4, i16x4, i64);
        sat_diff!("neon i64x4->i8x4", Neon, i64x4, i8x4, i64);
        sat_diff!("neon i64x8->i16x8", Neon, i64x8, i16x8, i64);
        sat_diff!("neon i64x8->i8x8", Neon, i64x8, i8x8, i64);
        sat_diff!("neon i64x16->i8x16", Neon, i64x16, i8x16, i64);
        sat_diff!("neon i64x16->i16x16", Neon, i64x16, i16x16, i64);
        sat_diff!("neon i32x2->i16x2", Neon, i32x2, i16x2, i32);
        sat_diff!("neon i32x2->i8x2", Neon, i32x2, i8x2, i32);
        sat_diff!("neon i64x2->i16x2", Neon, i64x2, i16x2, i64);
        sat_diff!("neon i64x2->i8x2", Neon, i64x2, i8x2, i64);
        sat_diff!("neon u64x4->u16x4", Neon, u64x4, u16x4, u64);
        sat_diff!("neon u64x4->u8x4", Neon, u64x4, u8x4, u64);
        sat_diff!("neon u64x8->u16x8", Neon, u64x8, u16x8, u64);
        sat_diff!("neon u64x8->u8x8", Neon, u64x8, u8x8, u64);
        sat_diff!("neon u64x16->u8x16", Neon, u64x16, u8x16, u64);
        sat_diff!("neon u64x16->u16x16", Neon, u64x16, u16x16, u64);
        sat_diff!("neon u32x2->u16x2", Neon, u32x2, u16x2, u32);
        sat_diff!("neon u32x2->u8x2", Neon, u32x2, u8x2, u32);
        sat_diff!("neon u64x2->u16x2", Neon, u64x2, u16x2, u64);
        sat_diff!("neon u64x2->u8x2", Neon, u64x2, u8x2, u64);

        // wide emulated narrows generated by `impl_casts!` (cross-N ArrayRegister recursion)
        sat_diff!("neon i64x8->i32x8", Neon, i64x8, i32x8, i64);
        sat_diff!("neon i64x16->i32x16", Neon, i64x16, i32x16, i64);
        sat_diff!("neon i32x16->i16x16", Neon, i32x16, i16x16, i32);
        sat_diff!("neon u64x8->u32x8", Neon, u64x8, u32x8, u64);
        sat_diff!("neon u64x16->u32x16", Neon, u64x16, u32x16, u64);
        sat_diff!("neon u32x16->u16x16", Neon, u32x16, u16x16, u32);
    }
}
