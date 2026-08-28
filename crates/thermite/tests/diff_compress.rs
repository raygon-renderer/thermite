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

/// Compare `R::expand` (the inverse stable partition) and `R::expand_z`
/// (zeroing) to scalar oracles over the given mask bit patterns.
///
/// The `ArrayRegister` shapes below route both through the branchless wide
/// scatter (`expand_permute_wide`) above 8 lanes. Before that they took the
/// data-dependent branchy scalar default, which this test also covers on any
/// shape the guard excludes.
fn check_expand<R: Register>(patterns: impl Iterator<Item = u64>) {
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

        // Non-zeroing oracle: selected lanes read the packed front in order,
        // unselected lanes read the tail in order.
        let mut full = vec![R::Element::default(); n];
        // Zeroing oracle: selected lanes only, everything else zero.
        let mut zero = vec![R::Element::default(); n];
        let mut pos = 0;
        for l in 0..n {
            if (bits >> l) & 1 == 1 {
                full[l] = vals[pos];
                zero[l] = vals[pos];
                pos += 1;
            }
        }
        for l in 0..n {
            if (bits >> l) & 1 == 0 {
                full[l] = vals[pos];
                pos += 1;
            }
        }

        let got = R::expand(v, mask);
        let got_z = R::expand_z(v, mask);
        assert_eq!(R::as_slice(&got), &full[..], "expand n={n} bits={bits:b}");
        assert_eq!(R::as_slice(&got_z), &zero[..], "expand_z n={n} bits={bits:b}");
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
    // override (the doubled-index `vpermd`), not f64x4's, so cover both.
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
    // Native-width byte vectors (32 lanes on AVX2), exercising 4-group routing.
    check::<<X86V3 as NativeSimd>::i8xN>(sample(32, 20000));
    check::<<X86V3 as NativeSimd>::u8xN>(sample(32, 20000));
}

#[test]
fn v3_emulated() {
    // f32x16 is `ArrayRegister<F32x8V3, 2>`: `compress_z` takes the merge tree
    // (2x 8-lane chunk compress + one count-indexed merge), `compress` the
    // stable-partition default.
    check::<<X86V3 as Simd>::f32x16>(0..(1 << 16));
    check::<<X86V3 as Simd>::i32x16>(0..(1 << 16));
}

#[test]
fn merge_tree_shapes() {
    // v2 f32x16 = ArrayRegister<F32x4V2, 4>: 4-lane chunks, N=4, a two-level
    // pairwise tree, so the M=2 chunk-shift stage as well as M=1.
    check::<<X86V2 as Simd>::f32x16>(0..(1 << 16));
    check::<<X86V2 as Simd>::i32x16>(0..(1 << 16));
    // v2 i16x16 = ArrayRegister<I16x8V2, 2>: 8-lane chunks, N=2.
    check::<<X86V2 as Simd>::i16x16>(0..(1 << 16));
    // Emulated 32-lane bytes: 16-lane chunks, N=2, one M=1 merge with each
    // chunk taking its own native wide compress.
    check::<thermite::register::array::ArrayRegister<<X86V2 as NativeSimd>::u8xN, 2>>(sample(32, 20000));
    check::<thermite::register::array::ArrayRegister<<X86V3 as NativeSimd>::u8xN, 2>>(sample(64, 20000));
}

/// `expand` / `expand_z` on the emulated-wide `ArrayRegister` shapes, whose
/// >8-lane arms route onto `expand_permute_wide` (branchless) rather than the
/// scalar default. Same shape set as `merge_tree_shapes` plus the v3 pairs.
#[test]
fn array_expand_shapes() {
    // v3 f32x16 = ArrayRegister<F32x8V3, 2>: 8-lane chunks.
    check_expand::<<X86V3 as Simd>::f32x16>(0..(1 << 16));
    check_expand::<<X86V3 as Simd>::i32x16>(0..(1 << 16));
    // v2 f32x16 = ArrayRegister<F32x4V2, 4>: 4-lane chunks.
    check_expand::<<X86V2 as Simd>::f32x16>(0..(1 << 16));
    check_expand::<<X86V2 as Simd>::i32x16>(0..(1 << 16));
    // v2 i16x16 = ArrayRegister<I16x8V2, 2>.
    check_expand::<<X86V2 as Simd>::i16x16>(0..(1 << 16));
    // Emulated byte arrays: 32 lanes (v2 chunks) and 64 lanes (v3 chunks).
    check_expand::<thermite::register::array::ArrayRegister<<X86V2 as NativeSimd>::u8xN, 2>>(sample(32, 20000));
    check_expand::<thermite::register::array::ArrayRegister<<X86V3 as NativeSimd>::u8xN, 2>>(sample(64, 20000));
}

/// `expand` / `expand_z` on the NATIVE wide registers, the `compress_via_wide!`
/// shapes, where both arms are now grouped kernels (`expand_grouped` composing
/// two `expand_z_grouped` trees). The mirror of `v3_wide`, which is the
/// corresponding gate for `compress`/`compress_z`.
#[test]
fn native_wide_expand_shapes() {
    check_expand::<<X86V3 as Simd>::i16x16>(0..(1 << 16));
    check_expand::<<X86V3 as Simd>::u16x16>(0..(1 << 16));
    check_expand::<<X86V3 as Simd>::i8x16>(0..(1 << 16));
    check_expand::<<X86V3 as Simd>::u8x16>(0..(1 << 16));
    check_expand::<<X86V2 as Simd>::i8x16>(0..(1 << 16));
    check_expand::<<X86V2 as Simd>::u8x16>(0..(1 << 16));
    // Native-width byte vectors (32 lanes on AVX2), 4-group routing.
    check_expand::<<X86V3 as NativeSimd>::i8xN>(sample(32, 20000));
    check_expand::<<X86V3 as NativeSimd>::u8xN>(sample(32, 20000));
}

/// The `_n` family: `compress_n` / `compress_z_n` / `expand_n` / `expand_z_n`
/// must be **bit-identical** to `N` separate single-vector calls under the same
/// mask, for every mask pattern.
///
/// That is the whole contract. The `_n` forms only share the mask-derived plan
/// (movemask, table rows, index registers, merge controls), never change the
/// permutation, so comparing against the single-vector ops is a complete
/// check, and it transitively inherits those ops' own oracle coverage above.
///
/// The `N` value registers carry disjoint value ranges so a lane leaking
/// between values is visible, not just a lane leaking between positions.
fn check_n<R: Register, const N: usize>(patterns: impl Iterator<Item = u64>) {
    let n = <R::Lanes as Unsigned>::USIZE;

    let mut vals = [R::EMPTY; N];
    for (k, slot) in vals.iter_mut().enumerate() {
        *slot = R::new(GenericArray::generate(|i| {
            // Distinct per (value, lane), wrapped into the element's range.
            <R::Element as Element>::from_u8(((k * n + i + 1) % 251 + 1) as u8)
        }));
    }

    for bits in patterns {
        let sel: GenericArray<R::Element, R::Lanes> =
            GenericArray::generate(|i| <R::Element as Element>::from_u8(((bits >> i) & 1) as u8));
        let mask = R::into_mask(R::new(sel));

        let got_c = R::compress_n::<N>(vals, mask);
        let got_cz = R::compress_z_n::<N>(vals, mask);
        let got_e = R::expand_n::<N>(vals, mask);
        let got_ez = R::expand_z_n::<N>(vals, mask);

        for k in 0..N {
            assert_eq!(
                R::as_slice(&got_c[k]),
                R::as_slice(&R::compress(vals[k], mask)),
                "compress_n N={N} k={k} lanes={n} bits={bits:b}"
            );
            assert_eq!(
                R::as_slice(&got_cz[k]),
                R::as_slice(&R::compress_z(vals[k], mask)),
                "compress_z_n N={N} k={k} lanes={n} bits={bits:b}"
            );
            assert_eq!(
                R::as_slice(&got_e[k]),
                R::as_slice(&R::expand(vals[k], mask)),
                "expand_n N={N} k={k} lanes={n} bits={bits:b}"
            );
            assert_eq!(
                R::as_slice(&got_ez[k]),
                R::as_slice(&R::expand_z(vals[k], mask)),
                "expand_z_n N={N} k={k} lanes={n} bits={bits:b}"
            );
        }
    }
}

/// `_n` on the <= 8-lane table path (one row fetch, `N` `permutev_row`s).
#[test]
fn n_family_table() {
    check_n::<<X86V3 as Simd>::i32x8, 2>(0..256);
    check_n::<<X86V3 as Simd>::i32x8, 4>(0..256);
}

/// `_n` on the native grouped kernels (`compress_via_wide!`): one plan for the
/// zeroing forms, two plus a shared shift control for the non-zeroing ones.
#[test]
fn n_family_grouped() {
    check_n::<<X86V3 as Simd>::u16x16, 2>(0..(1 << 16));
    check_n::<<X86V3 as Simd>::u16x16, 4>(0..(1 << 16));

    // 32 lanes, 4 groups and two merge levels.
    check_n::<<X86V3 as NativeSimd>::u8xN, 2>(sample(32, 20000));
    check_n::<<X86V3 as NativeSimd>::u8xN, 4>(sample(32, 20000));
}

/// `_n` on the emulated `ArrayRegister` shapes: `compress_z_n` takes the merge
/// tree with shared counts/`merge_pair` controls and chunk-recursive `_n`
/// calls, and the other three take the shared-index wide scatter.
#[test]
fn n_family_array() {
    // v3 f32x16 = ArrayRegister<F32x8V3, 2>: 8-lane chunks, merge2.
    check_n::<<X86V3 as Simd>::f32x16, 2>(0..(1 << 16));
    check_n::<<X86V3 as Simd>::f32x16, 4>(0..(1 << 16));
    // v2 f32x16 = ArrayRegister<F32x4V2, 4>: 4-lane chunks, merge4 (the M=2
    // chunk-shift stage as well as M=1).
    check_n::<<X86V2 as Simd>::f32x16, 2>(0..(1 << 16));
    // 64 lanes over 8 chunks: merge8, the M=4 stage.
    check_n::<thermite::register::array::ArrayRegister<<X86V2 as NativeSimd>::u8xN, 4>, 2>(sample(64, 20000));
}

#[test]
fn v2() {
    check::<<X86V2 as Simd>::f32x4>(0..16);
    // The 64-bit v2 registers gained pshufb-based `permutev` overrides, and
    // compress routes through them.
    check::<<X86V2 as Simd>::f64x2>(0..4);
    check::<<X86V2 as Simd>::i64x2>(0..4);
    check::<<X86V2 as Simd>::u64x2>(0..4);
    check::<<X86V2 as Simd>::i16x8>(0..256);
    check::<<X86V2 as Simd>::i8x16>(0..(1 << 16));
    check::<<X86V2 as Simd>::u8x16>(0..(1 << 16));
}
