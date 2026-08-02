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
//! macro paths) - the two lowerings must agree lane-for-lane.

use thermite::backend::scalar::Scalar;
use thermite::prelude::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use thermite::backend::{x86_v1::X86V1, x86_v2::X86V2, x86_v3::X86V3};

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use thermite::backend::wasm::Wasm;

#[cfg(target_arch = "aarch64")]
use thermite::backend::neon::Neon;

macro_rules! check {
    ($name:ident, $vty:ty, $elem:ty, $n:literal) => {
        #[test]
        fn $name() {
            type V = $vty;
            const N: usize = $n;

            // Distinct nonzero values; `src` disjoint from `data` so a lane
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
                // expand: selected lanes read the packed front in order, the
                // unselected lanes read the tail in order.
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

                    assert_eq!(got.as_slice()[lane], want[lane], "expand N={N} bits={bits:b} lane={lane}");

                    let want_z = if selected { want[lane] } else { 0 as $elem };
                    assert_eq!(got_z.as_slice()[lane], want_z, "expand_z N={N} bits={bits:b} lane={lane}");

                    let want_m = if selected { want[lane] } else { back[lane] };
                    assert_eq!(got_m.as_slice()[lane], want_m, "expand_m N={N} bits={bits:b} lane={lane}");

                    // compress_m: packed selected values below the count,
                    // src's own lanes at and above it.
                    let want_cm = if lane < cnt {
                        // lane-th selected element of `data`
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
                    assert_eq!(got_cm.as_slice()[lane], want_cm, "compress_m N={N} bits={bits:b} lane={lane}");
                }

                // --- laws -------------------------------------------------
                let there = v.compress(m).expand(m);
                let and_back = v.expand(m).compress(m);
                let round = v.compress_z(m).expand_m(src, m);
                let select = m.select(v, src);
                for lane in 0..N {
                    assert_eq!(
                        there.as_slice()[lane],
                        data[lane],
                        "expand∘compress N={N} bits={bits:b} lane={lane}"
                    );
                    assert_eq!(
                        and_back.as_slice()[lane],
                        data[lane],
                        "compress∘expand N={N} bits={bits:b} lane={lane}"
                    );
                    assert_eq!(
                        round.as_slice()[lane],
                        select.as_slice()[lane],
                        "wavefront round trip N={N} bits={bits:b} lane={lane}"
                    );
                }
            }
        }
    };
}

macro_rules! suite {
    ($modname:ident, $backend:ty) => {
        mod $modname {
            use super::*;

            // <= 8 lanes: the table path on native backends.
            check!(i32x4, thermite::simd::i32x4<$backend>, i32, 4);
            check!(u32x4, thermite::simd::u32x4<$backend>, u32, 4);
            check!(i32x8, thermite::simd::i32x8<$backend>, i32, 8);
            check!(u16x8, thermite::simd::u16x8<$backend>, u16, 8);
            check!(u64x2, thermite::simd::u64x2<$backend>, u64, 2);
            check!(i64x4, thermite::simd::i64x4<$backend>, i64, 4);
            check!(f32x4, thermite::simd::f32x4<$backend>, f32, 4);
            check!(f32x8, thermite::simd::f32x8<$backend>, f32, 8);
            check!(f64x2, thermite::simd::f64x2<$backend>, f64, 2);
            check!(f64x4, thermite::simd::f64x4<$backend>, f64, 4);

            // > 8 lanes: the wide path (per-group assembly + one permute).
            check!(u8x16, thermite::simd::u8x16<$backend>, u8, 16);
            check!(i16x16, thermite::simd::i16x16<$backend>, i16, 16);
        }
    };
}

suite!(scalar, Scalar);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
suite!(v1, X86V1);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
suite!(v2, X86V2);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
suite!(v3, X86V3);

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
suite!(wasm, Wasm);
#[cfg(target_arch = "aarch64")]
suite!(neon, Neon);
