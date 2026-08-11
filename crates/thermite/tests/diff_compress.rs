//! `Register::compress` / `compress_z` on the real x86 backends, checked against
//! a scalar partition oracle. Exercises the wired `compress_via_table!` (<= 8
//! lanes) and `compress_via_wide!` (16/32-lane) overrides.
#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use generic_array::{GenericArray, sequence::GenericSequence, typenum::Unsigned};

use thermite::backend::x86_v2::X86V2;
use thermite::backend::x86_v3::X86V3;
use thermite::register::{Element, Register, Storage};
use thermite::simd::{NativeSimd, Simd};

/// Compare `R::compress` (stable partition) and `R::compress_z` (zeroing) to
/// scalar oracles over the given mask bit patterns.
fn check<R: Register>(patterns: impl Iterator<Item = u64>) {
    let n = <R::Lanes as Unsigned>::USIZE;

    // Distinct nonzero lane values so a dropped/misplaced lane is visible.
    let v: Storage<R> = R::new(GenericArray::generate(|i| {
        <R::Element as Element>::from_u8((i + 1) as u8)
    }));
    let vals: Vec<R::Element> = R::as_slice(&v).to_vec();

    for bits in patterns {
        let sel: GenericArray<R::Element, R::Lanes> =
            GenericArray::generate(|i| <R::Element as Element>::from_u8(((bits >> i) & 1) as u8));
        let mask = R::into_mask(R::new(sel));

        // Non-zeroing oracle: selected in order, then unselected in order.
        let mut part = vec![R::Element::default(); n];
        // Zeroing oracle: selected in order, then zeros.
        let mut zero = vec![R::Element::default(); n];
        let mut pos = 0;
        for l in 0..n {
            if (bits >> l) & 1 == 1 {
                part[pos] = vals[l];
                zero[pos] = vals[l];
                pos += 1;
            }
        }
        for l in 0..n {
            if (bits >> l) & 1 == 0 {
                part[pos] = vals[l];
                pos += 1;
            }
        }

        let got = R::compress(v, mask);
        let got_z = R::compress_z(v, mask);
        assert_eq!(R::as_slice(&got), &part[..], "compress n={n} bits={bits:b}");
        assert_eq!(R::as_slice(&got_z), &zero[..], "compress_z n={n} bits={bits:b}");
    }
}

/// Deterministic sample of `count` mask patterns of `lanes` bits.
fn sample(lanes: usize, count: usize) -> impl Iterator<Item = u64> {
    let mask = if lanes >= 64 { u64::MAX } else { (1u64 << lanes) - 1 };
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    (0..count).map(move |_| {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s & mask
    })
}

#[test]
fn v3_table() {
    check::<<X86V3 as Simd>::f32x8>(0..256);
    check::<<X86V3 as Simd>::i32x8>(0..256);
    check::<<X86V3 as Simd>::u16x8>(0..256);
    check::<<X86V3 as Simd>::f32x4>(0..16);
    check::<<X86V3 as Simd>::f64x4>(0..16);
    // The integer 64x4 registers route compress through their own `permutev`
    // override (the doubled-index `vpermd`), not f64x4's - cover both.
    check::<<X86V3 as Simd>::i64x4>(0..16);
    check::<<X86V3 as Simd>::u64x4>(0..16);
    check::<<X86V3 as Simd>::i64x2>(0..4);
    check::<<X86V3 as Simd>::f64x2>(0..4);
}

#[test]
fn v3_wide() {
    check::<<X86V3 as Simd>::i16x16>(0..(1 << 16));
    check::<<X86V3 as Simd>::u16x16>(0..(1 << 16));
    check::<<X86V3 as Simd>::i8x16>(0..(1 << 16));
    check::<<X86V3 as Simd>::u8x16>(0..(1 << 16));
    // Native-width byte vectors (32 lanes on AVX2) - exercises 4-group routing.
    check::<<X86V3 as NativeSimd>::i8xN>(sample(32, 20000));
    check::<<X86V3 as NativeSimd>::u8xN>(sample(32, 20000));
}

#[test]
fn v3_emulated() {
    // f32x16 is `ArrayRegister<F32x8V3, 2>` - no per-register override, so this
    // exercises the `HAS_PERMUTEV` default path on an emulated wide register.
    check::<<X86V3 as Simd>::f32x16>(0..(1 << 16));
    check::<<X86V3 as Simd>::i32x16>(0..(1 << 16));
}

#[test]
fn v2() {
    check::<<X86V2 as Simd>::f32x4>(0..16);
    // The 64-bit v2 registers gained pshufb-based `permutev` overrides
    // (2026-08-08); compress routes through them.
    check::<<X86V2 as Simd>::f64x2>(0..4);
    check::<<X86V2 as Simd>::i64x2>(0..4);
    check::<<X86V2 as Simd>::u64x2>(0..4);
    check::<<X86V2 as Simd>::i16x8>(0..256);
    check::<<X86V2 as Simd>::i8x16>(0..(1 << 16));
    check::<<X86V2 as Simd>::u8x16>(0..(1 << 16));
}
