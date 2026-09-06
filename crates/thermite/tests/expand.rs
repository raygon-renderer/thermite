//! `GenericVector::expand` / `expand_z` / `expand_m` / `compress_m` correctness.
//!
//! `expand(mask)` is defined as the exact inverse permutation of
//! `compress(mask)`, so beyond per-op scalar oracles this file checks the laws
//! that pin the whole family together, per backend:
//!
//! - `v.compress(m).expand(m) == v` and `v.expand(m).compress(m) == v` (plain
//!   forms are mutual inverses, for every mask)
//! - `v.expand_z(m) == m.select(v.expand(m), zero)`
//! - `v.expand_m(src, m) == m.select(v.expand(m), src)`
//! - `v.compress_z(m).expand_m(src, m) == m.select(v, src)` - the full
//!   wavefront round trip (compact, then scatter back over a background),
//!   chaining four ops through the trusted plain permutation.
//!
//! Exhaustive over all `2^LANES` mask patterns per width. Backends: scalar
//! (trait defaults) plus every native backend for the target (table / wide
//! macro paths, `vpexpand*` on AVX-512). The lowerings must agree lane-for-lane.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use thermite::prelude::*;

macro_rules! check {
    ($vty:ty, $elem:ty, $n:literal) => {{
        type V = $vty;
        const N: usize = $n;

        // Distinct nonzero values, with `src` disjoint from `data` so a lane
        // pulled from the wrong source is visible.
        let mut data = [0 as $elem; N];
        let mut back = [0 as $elem; N];
        for i in 0..N {
            data[i] = ((i + 1) * 10) as $elem;
            back[i] = ((i + 1) * 10 + 5) as $elem;
        }
        let v = V::new(data.into());
        let src = V::new(back.into());

        for bits in 0u32..(1 << N) {
            let mut sel = [0 as $elem; N];
            for lane in 0..N {
                if (bits >> lane) & 1 == 1 {
                    sel[lane] = 1 as $elem;
                }
            }
            let m = V::new(sel.into()).cmp_ne(V::ZERO);
            let cnt = (bits & ((1u32 << N) - 1)).count_ones() as usize;

            // --- scalar oracles ---------------------------------------
            let mut want = [0 as $elem; N];
            let mut pos = 0;
            for lane in 0..N {
                if (bits >> lane) & 1 == 1 {
                    want[lane] = data[pos];
                    pos += 1;
                }
            }
            for lane in 0..N {
                if (bits >> lane) & 1 == 0 {
                    want[lane] = data[pos];
                    pos += 1;
                }
            }

            let got = v.expand(m);
            let got_z = v.expand_z(m);
            let got_m = v.expand_m(src, m);
            let got_cm = v.compress_m(src, m);

            for lane in 0..N {
                let selected = (bits >> lane) & 1 == 1;

                assert_eq!(
                    got.as_slice()[lane],
                    want[lane],
                    "expand N={N} bits={bits:b} lane={lane}"
                );

                let want_z = if selected { want[lane] } else { 0 as $elem };
                assert_eq!(
                    got_z.as_slice()[lane],
                    want_z,
                    "expand_z N={N} bits={bits:b} lane={lane}"
                );

                let want_m = if selected { want[lane] } else { back[lane] };
                assert_eq!(
                    got_m.as_slice()[lane],
                    want_m,
                    "expand_m N={N} bits={bits:b} lane={lane}"
                );

                let want_cm = if lane < cnt {
                    let mut seen = 0;
                    let mut val = 0 as $elem;
                    for j in 0..N {
                        if (bits >> j) & 1 == 1 {
                            if seen == lane {
                                val = data[j];
                                break;
                            }
                            seen += 1;
                        }
                    }
                    val
                } else {
                    back[lane]
                };
                assert_eq!(
                    got_cm.as_slice()[lane],
                    want_cm,
                    "compress_m N={N} bits={bits:b} lane={lane}"
                );
            }

            let there = v.compress(m).expand(m);
            let and_back = v.expand(m).compress(m);
            let round = v.compress_z(m).expand_m(src, m);
            let select = m.select(v, src);
            for lane in 0..N {
                assert_eq!(
                    there.as_slice()[lane],
                    data[lane],
                    "expand(compress) N={N} bits={bits:b} lane={lane}"
                );
                assert_eq!(
                    and_back.as_slice()[lane],
                    data[lane],
                    "compress(expand) N={N} bits={bits:b} lane={lane}"
                );
                assert_eq!(
                    round.as_slice()[lane],
                    select.as_slice()[lane],
                    "wavefront round trip N={N} bits={bits:b} lane={lane}"
                );
            }
        }
    }};
}

for_each_backend_concrete! {
    fn i32x4() { check!(thermite::simd::i32x4<S>, i32, 4) }
    fn u32x4() { check!(thermite::simd::u32x4<S>, u32, 4) }
    fn i32x8() { check!(thermite::simd::i32x8<S>, i32, 8) }
    fn u32x16() { check!(thermite::simd::u32x16<S>, u32, 16) }
    fn u16x8() { check!(thermite::simd::u16x8<S>, u16, 8) }
    fn u64x2() { check!(thermite::simd::u64x2<S>, u64, 2) }
    fn i64x4() { check!(thermite::simd::i64x4<S>, i64, 4) }
    fn u64x8() { check!(thermite::simd::u64x8<S>, u64, 8) }
    fn f32x4() { check!(thermite::simd::f32x4<S>, f32, 4) }
    fn f32x8() { check!(thermite::simd::f32x8<S>, f32, 8) }
    fn f32x16() { check!(thermite::simd::f32x16<S>, f32, 16) }
    fn f64x2() { check!(thermite::simd::f64x2<S>, f64, 2) }
    fn f64x4() { check!(thermite::simd::f64x4<S>, f64, 4) }
    fn f64x8() { check!(thermite::simd::f64x8<S>, f64, 8) }

    fn u8x16() { check!(thermite::simd::u8x16<S>, u8, 16) }
    fn i16x16() { check!(thermite::simd::i16x16<S>, i16, 16) }
}
