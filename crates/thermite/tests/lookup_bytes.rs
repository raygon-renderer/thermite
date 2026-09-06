//! `lookup` on the BYTE registers, where x86 v2/v3 have `pshufb`/`vpshufb`
//! overrides for tables of exactly 16/32/48/64 entries (and a scalar fallback
//! otherwise). Oracled against a plain Rust indexing loop.
//!
//! Every index used here is in range, as `Register::lookup`'s safety contract
//! requires. The overrides do not clamp or mask, so out-of-range indices are
//! deliberately not exercised.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use thermite::Vector;
use thermite::simd::{NativeSimd, Simd};
use thermite::vector::GenericVector;

/// Distinct table values in `0..=127`, so the same generator serves `u8` and `i8`
/// and any block/lane mix-up shows up as a mismatched value.
fn table_value(i: usize) -> i8 {
    ((i * 5 + 11) % 128) as i8
}

#[inline(always)]
fn check_lookup<V>(label: &str)
where
    V: GenericVector<Element: TryFrom<i8> + PartialEq + core::fmt::Debug + Copy>,
    <V::Unsigned as GenericVector>::Element: TryFrom<usize>,
{
    let lanes = V::LANES;

    // 16/32/48/64 hit the hardware arms, 1/17/40 hit the scalar fallback.
    for &len in &[16usize, 32, 48, 64, 1, 17, 40] {
        let table: Vec<V::Element> = (0..len)
            .map(|i| <V::Element as TryFrom<i8>>::try_from(table_value(i)).ok().unwrap())
            .collect();

        // Rotations cover every (lane, index) pair. Strides 1/3/7 and the two
        // constant rows pin the block boundaries (0 and len - 1).
        let mut rows: Vec<Vec<usize>> = Vec::new();
        for stride in [1usize, 3, 7] {
            for offset in 0..len {
                rows.push((0..lanes).map(|j| (offset + j * stride) % len).collect());
            }
        }
        rows.push(vec![0; lanes]);
        rows.push(vec![len - 1; lanes]);

        for idxs in &rows {
            let elems: Vec<<V::Unsigned as GenericVector>::Element> = idxs
                .iter()
                .map(|&u| {
                    <<V::Unsigned as GenericVector>::Element as TryFrom<usize>>::try_from(u)
                        .ok()
                        .unwrap()
                })
                .collect();

            let got = V::lookup(&table, <V::Unsigned as GenericVector>::from_slice(&elems));
            let got = got.into_array().as_slice().to_vec();

            let want: Vec<V::Element> = idxs.iter().map(|&i| table[i]).collect();

            assert_eq!(got, want, "{label} len={len} indices={idxs:?}");
        }
    }
}

for_each_backend! {
    fn u8x16<S: Simd>() {
        check_lookup::<Vector<<S as Simd>::u8x16>>(&harness::label::<S>("u8x16"));
    }

    fn i8x16<S: Simd>() {
        check_lookup::<Vector<<S as Simd>::i8x16>>(&harness::label::<S>("i8x16"));
    }

    // The native byte width: 128-bit on SSE/NEON/WASM, u8x32 on AVX2, u8x64 on AVX-512.
    fn u8xn<S: NativeSimd>() {
        check_lookup::<Vector<<S as NativeSimd>::u8xN>>(&harness::label::<S>("u8xN"));
    }

    fn i8xn<S: NativeSimd>() {
        check_lookup::<Vector<<S as NativeSimd>::i8xN>>(&harness::label::<S>("i8xN"));
    }
}
