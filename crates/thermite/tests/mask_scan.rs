//! `GenericMask::first_set` / `last_set` / `count_set` correctness.
//!
//! These turn a mask into scanning primitives (find-first / find-last / popcount
//! of true lanes). Verified against a per-lane oracle over a battery of bit
//! patterns (empty, full, every single bit, and full-minus-one-bit), which
//! pins down the first/last/count edges. Exercised across the scalar backend
//! (native 1-lane, `ArrayRegister`, and reduced register masks), on x86 the
//! native movemask paths of v1/v2/v3, and on aarch64 the NEON ones.

use thermite::backend::scalar::Scalar;
use thermite::mask::GenericMask;
use thermite::prelude::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use thermite::backend::{x86_v1::X86V1, x86_v2::X86V2, x86_v3::X86V3};

macro_rules! check {
    ($name:ident, $vty:ty) => {
        #[test]
        fn $name() {
            type V = $vty;
            const LANES: usize = <V as GenericVector>::LANES;

            // Build the mask for a bit pattern: lane set -> element 1, else 0,
            // then `!= 0` lifts it into the mask domain.
            let make = |bits: u64| -> <V as GenericVector>::Mask {
                let mut data = [<V as GenericVector>::Element::default(); LANES];
                for lane in 0..LANES {
                    if (bits >> lane) & 1 == 1 {
                        data[lane] = 1 as _;
                    }
                }
                V::new(data.into()).cmp_ne(V::ZERO)
            };

            let full: u64 = if LANES >= 64 {
                u64::MAX
            } else {
                (1u64 << LANES) - 1
            };

            // Patterns: empty, full, each single bit, and full with one bit cleared.
            let mut patterns = alloc_patterns(full, LANES);
            for bits in patterns.drain(..) {
                let m = make(bits);

                // Oracle over the LANES low bits.
                let mut first = None;
                let mut last = None;
                let mut count = 0usize;
                for lane in 0..LANES {
                    if (bits >> lane) & 1 == 1 {
                        if first.is_none() {
                            first = Some(lane);
                        }
                        last = Some(lane);
                        count += 1;
                    }
                }

                assert_eq!(m.first_set(), first, "first_set bits={bits:#x} lanes={LANES}");
                assert_eq!(m.last_set(), last, "last_set bits={bits:#x} lanes={LANES}");
                assert_eq!(m.count_set(), count, "count_set bits={bits:#x} lanes={LANES}");
            }

            // N-ary forms, over every ordered combination of a small pattern
            // set. These take a merged path on several backends (a saturating
            // narrowing pack on x86, a horizontal add on NEON), so N=2 and N=4
            // exercise the merge and N=1/3 the odd-count leftovers it falls
            // back to. `count_set` is free to scramble lane order internally;
            // `first_set`/`last_set` are not, and only the latter two would
            // notice if it did.
            type M = <$vty as GenericVector>::Mask;
            const P: [usize; 4] = [0, 1, 2, 3];

            let pats: [u64; 4] = [0, full, 1, full & !1];

            for &a in &P {
                let sel = [pats[a]];
                let ms = [make(sel[0])];
                let (f, l, c) = oracle(&sel, LANES);
                assert_eq!(M::count_set_many(ms), c, "count_set_many({sel:x?})");
                assert_eq!(M::first_set_many(ms), f, "first_set_many({sel:x?})");
                assert_eq!(M::last_set_many(ms), l, "last_set_many({sel:x?})");

                for &b in &P {
                    let sel = [pats[a], pats[b]];
                    let ms = [make(sel[0]), make(sel[1])];
                    let (f, l, c) = oracle(&sel, LANES);
                    assert_eq!(M::count_set_many(ms), c, "count_set_many({sel:x?})");
                    assert_eq!(M::first_set_many(ms), f, "first_set_many({sel:x?})");
                    assert_eq!(M::last_set_many(ms), l, "last_set_many({sel:x?})");

                    for &c_ in &P {
                        let sel = [pats[a], pats[b], pats[c_]];
                        let ms = [make(sel[0]), make(sel[1]), make(sel[2])];
                        let (f, l, c) = oracle(&sel, LANES);
                        assert_eq!(M::count_set_many(ms), c, "count_set_many({sel:x?})");
                        assert_eq!(M::first_set_many(ms), f, "first_set_many({sel:x?})");
                        assert_eq!(M::last_set_many(ms), l, "last_set_many({sel:x?})");

                        for &d in &P {
                            let sel = [pats[a], pats[b], pats[c_], pats[d]];
                            let ms = [make(sel[0]), make(sel[1]), make(sel[2]), make(sel[3])];
                            let (f, l, c) = oracle(&sel, LANES);
                            assert_eq!(M::count_set_many(ms), c, "count_set_many({sel:x?})");
                            assert_eq!(M::first_set_many(ms), f, "first_set_many({sel:x?})");
                            assert_eq!(M::last_set_many(ms), l, "last_set_many({sel:x?})");
                        }
                    }
                }
            }
        }
    };
}

/// Per-lane oracle over a concatenation of bit patterns, `bits[i]` occupying
/// lanes `i * lanes .. (i + 1) * lanes`.
fn oracle(bits: &[u64], lanes: usize) -> (Option<usize>, Option<usize>, usize) {
    let mut first = None;
    let mut last = None;
    let mut count = 0usize;

    for (i, &pat) in bits.iter().enumerate() {
        for lane in 0..lanes {
            if (pat >> lane) & 1 == 1 {
                let idx = i * lanes + lane;
                if first.is_none() {
                    first = Some(idx);
                }
                last = Some(idx);
                count += 1;
            }
        }
    }

    (first, last, count)
}

// Collect the test patterns into a Vec so the closure body stays simple.
fn alloc_patterns(full: u64, lanes: usize) -> Vec<u64> {
    let mut v = vec![0u64, full];
    for k in 0..lanes {
        v.push(1u64 << k); // single bit
        v.push(full & !(1u64 << k)); // full minus one bit
    }
    v
}

macro_rules! suite {
    ($modname:ident, $backend:ty) => {
        mod $modname {
            use super::*;
            // f32 masks live in float registers and take their own path into
            // the merged count.
            check!(f32x4, thermite::simd::f32x4<$backend>);
            check!(f32x8, thermite::simd::f32x8<$backend>);
            check!(f32x16, thermite::simd::f32x16<$backend>);
            check!(u8x8, thermite::simd::u8x8<$backend>);
            check!(u8x16, thermite::simd::u8x16<$backend>);
            check!(u16x8, thermite::simd::u16x8<$backend>);
            check!(u16x16, thermite::simd::u16x16<$backend>);
            check!(u32x3, thermite::simd::u32x3<$backend>); // reduced register
            check!(u32x4, thermite::simd::u32x4<$backend>);
            check!(u32x8, thermite::simd::u32x8<$backend>);
            check!(u32x16, thermite::simd::u32x16<$backend>);
            check!(u64x2, thermite::simd::u64x2<$backend>);
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

#[cfg(target_arch = "aarch64")]
suite!(neon, thermite::backend::neon::Neon);
