//! Coverage for `GenericVector`/`NumericVector` methods in `vector/vector.rs`
//! that the differential suites don't reach: the scalar-fallback `map`/`fold`/
//! `reduce`, dynamic lane access (`extractv`/`insertv`/`broadcastv`) and the
//! const-index forms, horizontal reductions (`sum`/`prod`/`min`/`max`/
//! `min_max_element`/`arg_minmax`), `is_zero`/`is_all_zero`, `indexed`/`single`/
//! `reverse`, and the integer `avg` family.
//!
//! Deterministic distinct-value inputs (no ties), values exactly representable in
//! f32/i32, so comparisons are exact.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use thermite::Vector;
use thermite::prelude::*;
use thermite::simd::{Simd, Simd3A};

macro_rules! common_methods {
    ($V:ty, $e:ty, $L:expr) => {{
        type V = $V;
        let rd = |v: V| -> Vec<$e> { v.into_array().as_slice().to_vec() };
        // a = [3, 1, 4, 2]: distinct, min at lane 1, max at lane 2.
        let a = V::new([3 as $e, 1 as $e, 4 as $e, 2 as $e]);

        // --- scalar-fallback map / fold / reduce ---
        assert_eq!(rd(a.map(|x| x * 2 as $e + 1 as $e)), vec![7 as $e, 3 as $e, 9 as $e, 5 as $e], "{} map", $L);
        assert_eq!(a.fold(10 as $e, |acc, x| acc + x), 20 as $e, "{} fold", $L);
        assert_eq!(a.reduce(|x, y| x + y), 10 as $e, "{} reduce(+)", $L);
        assert_eq!(a.reduce(|x, y| if x > y { x } else { y }), 4 as $e, "{} reduce(max)", $L);

        // --- dynamic lane access ---
        assert_eq!(a.extractv(2), 4 as $e, "{} extractv", $L);
        assert_eq!(rd(a.insertv(0, 9 as $e)), vec![9 as $e, 1 as $e, 4 as $e, 2 as $e], "{} insertv", $L);
        assert_eq!(rd(a.broadcastv(1)), vec![1 as $e; 4], "{} broadcastv", $L);

        // --- const-index lane access ---
        assert_eq!(a.extract::<3>(), 2 as $e, "{} extract", $L);
        assert_eq!(rd(a.insert::<1>(9 as $e)), vec![3 as $e, 9 as $e, 4 as $e, 2 as $e], "{} insert", $L);
        assert_eq!(rd(a.broadcast::<2>()), vec![4 as $e; 4], "{} broadcast", $L);

        // --- horizontal reductions ---
        assert_eq!(a.sum_elements(), 10 as $e, "{} sum_elements", $L);
        assert_eq!(a.prod_elements(), 24 as $e, "{} prod_elements", $L);
        assert_eq!(a.min_element(), 1 as $e, "{} min_element", $L);
        assert_eq!(a.max_element(), 4 as $e, "{} max_element", $L);
        assert_eq!(a.min_max_element(), (1 as $e, 4 as $e), "{} min_max_element", $L);
        assert_eq!(a.arg_minmax(), (1usize, 2usize), "{} arg_minmax", $L);

        // --- predicates ---
        assert_eq!(rd(a.is_zero().select(V::ONE, V::ZERO)), vec![0 as $e; 4], "{} is_zero (none)", $L);
        assert_eq!(
            rd(V::new([0 as $e, 1 as $e, 0 as $e, 2 as $e]).is_zero().select(V::ONE, V::ZERO)),
            vec![1 as $e, 0 as $e, 1 as $e, 0 as $e], "{} is_zero (some)", $L
        );
        assert!(!a.is_all_zero(), "{} is_all_zero (false)", $L);
        assert!(V::ZERO.is_all_zero(), "{} is_all_zero (true)", $L);

        // --- construction / routing ---
        assert_eq!(rd(V::indexed()), vec![0 as $e, 1 as $e, 2 as $e, 3 as $e], "{} indexed", $L);
        assert_eq!(V::single(5 as $e).extractv(0), 5 as $e, "{} single lane0", $L);
        assert_eq!(rd(a.reverse()), vec![2 as $e, 4 as $e, 1 as $e, 3 as $e], "{} reverse", $L);
    }};
}

/// `reverse` across widths/types/backends (`indexed()` reversed). Catches the
/// per-register shuffle-immediate bugs (e.g. f32x4 swapping only lanes 1<->2).
macro_rules! rev {
    ($reg:ty) => {{
        type V = Vector<$reg>;
        let n = <V as GenericVector>::LANES;
        let got: Vec<f64> = V::indexed()
            .reverse()
            .into_array()
            .as_slice()
            .iter()
            .map(|&x| x as f64)
            .collect();
        let want: Vec<f64> = (0..n).rev().map(|i| i as f64).collect();
        assert_eq!(got, want, concat!("reverse ", stringify!($reg)));
    }};
}

/// Radix round-trips only: `deinterleave(interleave(a, b)) == (a, b)` and
/// `deinterleave_radix . interleave_radix == id` for radix 2/3/5. Radix 2 forwards
/// to the native `interleave`, radix 3 to the native radix-3 sequence, radix 5 to
/// the permute+blend gather. No pair-granularity `interleave2` (ill-defined on odd
/// lane counts), so this variant is valid for the 3-lane `ReducedRegister` types too.
macro_rules! ilv_radix {
    ($reg:ty) => {{
        type V = Vector<$reg>;
        let a = V::indexed();
        let b = a + a; // distinct from a (except lane 0), enough to expose lane mixups
        let to = |v: V| -> Vec<f64> { v.into_array().as_slice().iter().map(|&x| x as f64).collect() };

        let (lo, hi) = a.interleave(b);
        let (a2, b2) = lo.deinterleave(hi);
        assert_eq!(to(a2), to(a), concat!("deinterleave a ", stringify!($reg)));
        assert_eq!(to(b2), to(b), concat!("deinterleave b ", stringify!($reg)));

        // `c/d/e` are distinct from a/b (except lane 0) to expose lane mixups.
        let c = b + a; // 3 * indexed
        let d = c + a; // 4 * indexed
        let e = d + a; // 5 * indexed
        let [r2a, r2b] = V::deinterleave_radix(V::interleave_radix([a, b]));
        assert_eq!(to(r2a), to(a), concat!("radix2 a ", stringify!($reg)));
        assert_eq!(to(r2b), to(b), concat!("radix2 b ", stringify!($reg)));
        let [r3a, r3b, r3c] = V::deinterleave_radix(V::interleave_radix([a, b, c]));
        assert_eq!(to(r3a), to(a), concat!("radix3 a ", stringify!($reg)));
        assert_eq!(to(r3b), to(b), concat!("radix3 b ", stringify!($reg)));
        assert_eq!(to(r3c), to(c), concat!("radix3 c ", stringify!($reg)));
        let [r5a, r5b, r5c, r5d, r5e] = V::deinterleave_radix(V::interleave_radix([a, b, c, d, e]));
        assert_eq!(to(r5a), to(a), concat!("radix5 a ", stringify!($reg)));
        assert_eq!(to(r5b), to(b), concat!("radix5 b ", stringify!($reg)));
        assert_eq!(to(r5c), to(c), concat!("radix5 c ", stringify!($reg)));
        assert_eq!(to(r5d), to(d), concat!("radix5 d ", stringify!($reg)));
        assert_eq!(to(r5e), to(e), concat!("radix5 e ", stringify!($reg)));
    }};
}

/// [`ilv_radix!`] plus the group-granularity `interleave_by`/`deinterleave_by`
/// round-trip at `GROUP == 1` (== `interleave`) and `GROUP == 2` (pairs, with a native
/// override on wide backends, the lane-wise polyfill default on the emulated ones).
/// Even lane counts only.
macro_rules! ilv {
    ($reg:ty) => {{
        ilv_radix!($reg);

        type V = Vector<$reg>;
        let a = V::indexed();
        let b = a + a;
        let to = |v: V| -> Vec<f64> { v.into_array().as_slice().iter().map(|&x| x as f64).collect() };

        let (lo1, hi1) = a.interleave_by::<1>(b);
        let (a1, b1) = lo1.deinterleave_by::<1>(hi1);
        assert_eq!(to(a1), to(a), concat!("deinterleave_by::<1> a ", stringify!($reg)));
        assert_eq!(to(b1), to(b), concat!("deinterleave_by::<1> b ", stringify!($reg)));

        let (lo2, hi2) = a.interleave_by::<2>(b);
        let (a2, b2) = lo2.deinterleave_by::<2>(hi2);
        assert_eq!(to(a2), to(a), concat!("deinterleave_by::<2> a ", stringify!($reg)));
        assert_eq!(to(b2), to(b), concat!("deinterleave_by::<2> b ", stringify!($reg)));
    }};
}

/// Semantic check (NOT a round-trip): `interleave_radix::<3>` must place stream
/// `s`, element `q` at flat AoS position `q*3 + s`. A round-trip can hide a
/// self-cancelling interleave/deinterleave index bug (esp. in the `ArrayRegister`
/// chunk-scatter and the `ReducedRegister` gather); this pins the absolute layout.
/// Stream `s` is `indexed() + s*LANES`, so `v_s[q] == q + s*LANES` (distinct).
macro_rules! radix3_semantic {
    ($reg:ty, $e:ty) => {{
        type V = Vector<$reg>;
        let n = <V as GenericVector>::LANES;
        let step = V::splat(n as $e);
        let a = V::indexed();
        let b = a + step;
        let c = b + step;
        let [o0, o1, o2] = V::interleave_radix([a, b, c]);
        let (a0, a1, a2) = (o0.into_array(), o1.into_array(), o2.into_array());
        let flat: Vec<f64> = a0
            .as_slice()
            .iter()
            .chain(a1.as_slice().iter())
            .chain(a2.as_slice().iter())
            .map(|&x| x as f64)
            .collect();
        for m in 0..(3 * n) {
            let (q, s) = (m / 3, m % 3);
            assert_eq!(
                flat[m],
                (q + s * n) as f64,
                concat!("radix3 semantic pos {} ", stringify!($reg)),
                m
            );
        }
    }};
}

/// Semantic check (NOT a round-trip) for pair-granularity `interleave_by::<2>`:
/// `concat(lo, hi) == [a.P0, b.P0, a.P1, b.P1, ...]` - group `k` is pair `k/2` of
/// `a` (k even) or `b` (k odd). `a[q] == q`, `b[q] == q + LANES` (distinct), so the
/// expected element at flat position `e` (group `k = e/2`, offset `sub = e%2`,
/// source element `(k/2)*2 + sub`) is that source, `+ LANES` when `k` is odd.
macro_rules! group2_semantic {
    ($reg:ty, $e:ty) => {{
        type V = Vector<$reg>;
        let n = <V as GenericVector>::LANES;
        let a = V::indexed();
        let b = a + V::splat(n as $e);
        let (lo, hi) = a.interleave_by::<2>(b);
        let (al, ah) = (lo.into_array(), hi.into_array());
        let flat: Vec<f64> = al
            .as_slice()
            .iter()
            .chain(ah.as_slice().iter())
            .map(|&x| x as f64)
            .collect();
        for e in 0..(2 * n) {
            let (k, sub) = (e / 2, e % 2);
            let src = (k / 2) * 2 + sub;
            let expect = if k % 2 == 0 { src } else { src + n };
            assert_eq!(
                flat[e], expect as f64,
                concat!("group2 semantic pos {} ", stringify!($reg)),
                e
            );
        }
    }};
}

/// Semantic check (NOT a round-trip) for the **square** group-radix transpose
/// `deinterleave_radix_by::<N, GROUP>` with `N == LANES / GROUP`:
/// `out[r].group[q] == inputs[q].group[r]`, each group `GROUP` consecutive
/// elements. Input `i` is `indexed() + i*LANES` (all distinct), so element
/// `(q*GROUP + sub)` of `out[r]` must equal `q*LANES + (r*GROUP + sub)`. Then
/// `interleave_radix_by` must invert it. Covers the native (4,2)/(4,1) AVX2 paths
/// and the `ArrayRegister`/lane-wise fallbacks.
macro_rules! radix_by_transpose {
    ($reg:ty, $e:ty, $N:expr, $GROUP:expr) => {{
        type V = Vector<$reg>;
        let n = <V as GenericVector>::LANES;
        assert_eq!(
            $N,
            n / $GROUP,
            concat!("square transpose needs N==LANES/GROUP ", stringify!($reg))
        );
        let mut inputs = [V::ZERO; $N];
        for i in 0..$N {
            inputs[i] = V::indexed() + V::splat((i * n) as $e);
        }
        let out = V::deinterleave_radix_by::<$N, $GROUP>(inputs);
        let groups = n / $GROUP;
        for r in 0..$N {
            let arr = out[r].into_array();
            let o = arr.as_slice();
            for q in 0..groups {
                for sub in 0..$GROUP {
                    let got = o[q * $GROUP + sub] as f64;
                    let want = (q * n + (r * $GROUP + sub)) as f64;
                    assert_eq!(got, want, concat!("radix_by transpose ", stringify!($reg)));
                }
            }
        }
        let back = V::interleave_radix_by::<$N, $GROUP>(out);
        for i in 0..$N {
            let g: Vec<f64> = back[i].into_array().as_slice().iter().map(|&x| x as f64).collect();
            let w: Vec<f64> = inputs[i].into_array().as_slice().iter().map(|&x| x as f64).collect();
            assert_eq!(g, w, concat!("radix_by transpose roundtrip ", stringify!($reg)));
        }
    }};
}

/// Full semantic check for **any** `(N, GROUP)` shape, both directions, the
/// non-square generalization of [`radix_by_transpose!`]. Round-trips alone cannot
/// catch two directions that are consistently wrong together, so each direction is
/// checked against its index formula independently (input `i` = `indexed() + i*LANES`,
/// so every element value encodes its (register, lane) coordinates):
///
/// - deinterleave: `out[r].group[q] == flat group (q*N + r)`, i.e. element
///   `q*GROUP + sub` of `out[r]` is `(c/groups)*LANES + (c%groups)*GROUP + sub`
///   with `c = q*N + r` and `groups = LANES/GROUP`.
/// - interleave: `out[t].group[lg] == inputs[c % N].group[c / N]` with
///   `c = t*groups + lg`, i.e. element `lg*GROUP + sub` is
///   `(c%N)*LANES + (c/N)*GROUP + sub`.
macro_rules! radix_by_semantic {
    ($reg:ty, $e:ty, $N:expr, $GROUP:expr) => {{
        type V = Vector<$reg>;
        let n = <V as GenericVector>::LANES;
        let groups = n / $GROUP;
        let mut inputs = [V::ZERO; $N];
        for i in 0..$N {
            inputs[i] = V::indexed() + V::splat((i * n) as $e);
        }

        let de = V::deinterleave_radix_by::<$N, $GROUP>(inputs);
        for r in 0..$N {
            let arr = de[r].into_array();
            let o = arr.as_slice();
            for q in 0..groups {
                let c = q * $N + r;
                for sub in 0..$GROUP {
                    let got = o[q * $GROUP + sub] as f64;
                    let want = ((c / groups) * n + (c % groups) * $GROUP + sub) as f64;
                    assert_eq!(
                        got, want,
                        concat!(
                            "radix_by deinterleave semantic ",
                            stringify!($reg),
                            " N=",
                            stringify!($N),
                            " G=",
                            stringify!($GROUP)
                        ),
                    );
                }
            }
        }

        let il = V::interleave_radix_by::<$N, $GROUP>(inputs);
        for t in 0..$N {
            let arr = il[t].into_array();
            let o = arr.as_slice();
            for lg in 0..groups {
                let c = t * groups + lg;
                for sub in 0..$GROUP {
                    let got = o[lg * $GROUP + sub] as f64;
                    let want = ((c % $N) * n + (c / $N) * $GROUP + sub) as f64;
                    assert_eq!(
                        got, want,
                        concat!(
                            "radix_by interleave semantic ",
                            stringify!($reg),
                            " N=",
                            stringify!($N),
                            " G=",
                            stringify!($GROUP)
                        ),
                    );
                }
            }
        }
    }};
}

/// **Differential: a register's `(de)interleave_radix_by` override vs the portable
/// engine it replaced.** Runs the register's own path and
/// `(de)interleave_radix_by_default::<R, N, GROUP>` (which never re-enters the
/// override, its arms reaching only `deinterleave`/`deinterleave_by`/`deinterleave_radix`)
/// on identical input and pins them equal, both directions.
///
/// This is the check an override actually needs: [`radix_by_semantic!`] pins the index
/// formulas, but this pins "the specialization did not change behaviour", which is the
/// failure mode a round-trip provably cannot see (both directions wrong together still
/// round-trip).
macro_rules! radix_by_vs_default {
    ($reg:ty, $e:ty, $N:expr, $GROUP:expr) => {{
        type V = Vector<$reg>;
        let n = <V as GenericVector>::LANES;

        let mut inputs = [V::ZERO; $N];
        for i in 0..$N {
            inputs[i] = V::indexed() + V::splat((i * n) as $e);
        }

        // The register's own path (the override under test).
        let spec_de = V::deinterleave_radix_by::<$N, $GROUP>(inputs);
        let spec_il = V::interleave_radix_by::<$N, $GROUP>(inputs);

        // The portable default, called directly = exactly what the override replaced.
        let mut raw = [inputs[0].0; $N];
        for i in 0..$N {
            raw[i] = inputs[i].0;
        }
        let ref_de = thermite::backend::generic::polyfills::deinterleave_radix_by_default::<$reg, $N, $GROUP>(raw);
        let ref_il = thermite::backend::generic::polyfills::interleave_radix_by_default::<$reg, $N, $GROUP>(raw);

        for i in 0..$N {
            let g: Vec<f64> = spec_de[i].into_array().as_slice().iter().map(|&x| x as f64).collect();
            let w: Vec<f64> = Vector::<$reg>(ref_de[i])
                .into_array()
                .as_slice()
                .iter()
                .map(|&x| x as f64)
                .collect();
            assert_eq!(
                g, w,
                concat!(
                    "deinterleave override != default: ",
                    stringify!($reg),
                    " N=",
                    stringify!($N),
                    " GROUP=",
                    stringify!($GROUP)
                ),
            );

            let g: Vec<f64> = spec_il[i].into_array().as_slice().iter().map(|&x| x as f64).collect();
            let w: Vec<f64> = Vector::<$reg>(ref_il[i])
                .into_array()
                .as_slice()
                .iter()
                .map(|&x| x as f64)
                .collect();
            assert_eq!(
                g, w,
                concat!(
                    "interleave override != default: ",
                    stringify!($reg),
                    " N=",
                    stringify!($N),
                    " GROUP=",
                    stringify!($GROUP)
                ),
            );
        }
    }};
}

/// Round-trip only: `deinterleave_radix_by . interleave_radix_by == id` for a
/// possibly non-square `(N, GROUP)`. Exercises the general lane-wise group path and
/// the `N == 2` / `GROUP == 1` forwarding arms that the transpose macro doesn't hit.
macro_rules! radix_by_roundtrip {
    ($reg:ty, $e:ty, $N:expr, $GROUP:expr) => {{
        type V = Vector<$reg>;
        let n = <V as GenericVector>::LANES;
        let mut inputs = [V::ZERO; $N];
        for i in 0..$N {
            inputs[i] = V::indexed() + V::splat((i * n) as $e);
        }
        let back = V::deinterleave_radix_by::<$N, $GROUP>(V::interleave_radix_by::<$N, $GROUP>(inputs));
        for i in 0..$N {
            let g: Vec<f64> = back[i].into_array().as_slice().iter().map(|&x| x as f64).collect();
            let w: Vec<f64> = inputs[i].into_array().as_slice().iter().map(|&x| x as f64).collect();
            assert_eq!(g, w, concat!("radix_by roundtrip ", stringify!($reg)));
        }
    }};
}

for_each_backend_concrete! {
    fn float_methods() {
        common_methods!(Vector<<S as Simd>::f32x4>, f32, harness::label::<S>("f32x4"));
    }

    fn int_methods() {
        common_methods!(Vector<<S as Simd>::i32x4>, i32, harness::label::<S>("i32x4"));
    }

    /// The group-radix transpose primitive across native paths and the
    /// emulated/lane-wise fallbacks (which of those a shape takes varies by backend).
    fn radix_by_transpose_widths() {
        // (4,2) f32 pair transpose: native on AVX2, ArrayRegister-emulated elsewhere.
        radix_by_transpose!(<S as Simd>::f32x8, f32, 4, 2);
        // (8,1) full f32 8x8 transpose: native on AVX2 (transpose256_w32), emulated elsewhere.
        radix_by_transpose!(<S as Simd>::f32x8, f32, 8, 1);
        // Integer 256-bit family via bit-cast into the same bodies.
        radix_by_transpose!(<S as Simd>::i32x8, i32, 8, 1);
        radix_by_transpose!(<S as Simd>::u32x8, u32, 8, 1);
        radix_by_transpose!(<S as Simd>::i32x8, i32, 4, 2);
        radix_by_transpose!(<S as Simd>::u32x8, u32, 4, 2);
        // (4,1) 64-bit transpose: f64x4 direct, i64x4/u64x4 via bit-cast.
        radix_by_transpose!(<S as Simd>::f64x4, f64, 4, 1);
        radix_by_transpose!(<S as Simd>::i64x4, i64, 4, 1);
        radix_by_transpose!(<S as Simd>::u64x4, u64, 4, 1);
        // Lane-wise / forwarding fallbacks.
        radix_by_transpose!(<S as Simd>::f32x4, f32, 2, 2); // N==2 -> deinterleave_by::<2>
        radix_by_transpose!(<S as Simd>::f32x4, f32, 4, 1); // GROUP==1 -> deinterleave_radix::<4>
        radix_by_transpose!(<S as Simd>::f64x4, f64, 2, 2); // N==2 -> f64 deinterleave_by::<2>

        // Non-square round-trips (general path, N != LANES/GROUP).
        radix_by_roundtrip!(<S as Simd>::f32x8, f32, 3, 2); // 3 inputs, 4 pair-groups each
        radix_by_roundtrip!(<S as Simd>::f32x8, f32, 2, 2); // N==2 forward
        radix_by_roundtrip!(<S as Simd>::f32x8, f32, 5, 1); // GROUP==1 radix-5 forward
        // GROUP==1 large N routes to the deinterleave_n stage engine, not the
        // single-gather. Both pow-2 (16, 32) and a non-pow-2 (12) case.
        radix_by_roundtrip!(<S as Simd>::f32x8, f32, 16, 1);
        radix_by_roundtrip!(<S as Simd>::f32x8, f32, 32, 1);
        radix_by_roundtrip!(<S as Simd>::f32x8, f32, 12, 1);
        // Group-radix, large N (staged-vs-lane-wise dispatch): pow-2 N with GROUP 2 and 4.
        radix_by_roundtrip!(<S as Simd>::f32x8, f32, 8, 2);
        radix_by_roundtrip!(<S as Simd>::f32x8, f32, 16, 2);
        radix_by_roundtrip!(<S as Simd>::f32x8, f32, 16, 4);
        radix_by_roundtrip!(<S as Simd>::f32x8, f32, 8, 4);
    }

    /// Exact semantics of `(de)interleave_radix_by` for every pow-2 shape the certified
    /// ladder engine (`x86_v3::polyfills::transpose256`) may cover, both directions,
    /// plus non-pow-2 fallback shapes that validate the check itself against the
    /// untouched lane-wise reference. A shape the ladder does not certify silently
    /// falls back, so this sweep is correct regardless of which path each shape
    /// actually takes on each backend. It pins the SEMANTICS, not the route.
    fn radix_by_ladder_shapes() {
        // f32x8: the full pow-2 (N, GROUP) battery.
        radix_by_semantic!(<S as Simd>::f32x8, f32, 2, 2);
        radix_by_semantic!(<S as Simd>::f32x8, f32, 2, 4);
        radix_by_semantic!(<S as Simd>::f32x8, f32, 4, 1);
        radix_by_semantic!(<S as Simd>::f32x8, f32, 4, 2); // native square on AVX2
        radix_by_semantic!(<S as Simd>::f32x8, f32, 4, 4);
        radix_by_semantic!(<S as Simd>::f32x8, f32, 8, 1); // native square on AVX2
        radix_by_semantic!(<S as Simd>::f32x8, f32, 8, 2);
        radix_by_semantic!(<S as Simd>::f32x8, f32, 8, 4);
        radix_by_semantic!(<S as Simd>::f32x8, f32, 16, 1);
        radix_by_semantic!(<S as Simd>::f32x8, f32, 16, 2);
        radix_by_semantic!(<S as Simd>::f32x8, f32, 16, 4);
        radix_by_semantic!(<S as Simd>::f32x8, f32, 32, 1);
        radix_by_semantic!(<S as Simd>::f32x8, f32, 32, 2);
        radix_by_semantic!(<S as Simd>::f32x8, f32, 32, 4);
        // Non-pow-2 N: never certified, exercises the lane-wise reference and thereby
        // validates the semantic formulas themselves.
        radix_by_semantic!(<S as Simd>::f32x8, f32, 3, 2);
        radix_by_semantic!(<S as Simd>::f32x8, f32, 6, 2);
        radix_by_semantic!(<S as Simd>::f32x8, f32, 12, 1);
        // Integer 32-bit family through the si cast adapter.
        radix_by_semantic!(<S as Simd>::i32x8, i32, 8, 2);
        radix_by_semantic!(<S as Simd>::i32x8, i32, 16, 1);
        radix_by_semantic!(<S as Simd>::u32x8, u32, 16, 4);
        // 64-bit family through the pd/si adapters (8-byte elements: GROUP scales x2).
        radix_by_semantic!(<S as Simd>::f64x4, f64, 4, 2);
        radix_by_semantic!(<S as Simd>::f64x4, f64, 8, 1);
        radix_by_semantic!(<S as Simd>::f64x4, f64, 8, 2);
        radix_by_semantic!(<S as Simd>::f64x4, f64, 16, 1);
        radix_by_semantic!(<S as Simd>::i64x4, i64, 8, 1);
        radix_by_semantic!(<S as Simd>::u64x4, u64, 4, 2);
    }

    /// `ArrayRegister`'s `(de)interleave_radix_by` chunk-chain (`register/array.rs`):
    /// output chunk `i` is one inner `radix_by::<S, GROUP>` of flat chunks `S*i..S*i+S`,
    /// valid whenever a group fits a chunk (`GROUP` divides the inner `L`). That
    /// decomposition is subtle enough that round-trips are not enough, so these check both
    /// directions against the index formulas at every emulated width.
    ///
    /// This is also the path that carries native shuffles + the ladder up to the emulated
    /// widths: on AVX2 `f32x16 = ArrayRegister<F32x8V3, 2>` reaches `F32x8V3` through
    /// here. On 128-bit backends `f32x8` reaches `F32x4`.
    fn radix_by_array_register_chunk_chain() {
        // f32x16 (2-chunk on AVX2 with inner L = 8, 4-chunk on 128-bit backends with inner L = 4).
        radix_by_semantic!(<S as Simd>::f32x16, f32, 2, 1);
        radix_by_semantic!(<S as Simd>::f32x16, f32, 4, 1);
        radix_by_semantic!(<S as Simd>::f32x16, f32, 8, 1);
        radix_by_semantic!(<S as Simd>::f32x16, f32, 16, 1);
        radix_by_semantic!(<S as Simd>::f32x16, f32, 4, 2);
        radix_by_semantic!(<S as Simd>::f32x16, f32, 8, 2); // square: N == LANES/GROUP
        radix_by_semantic!(<S as Simd>::f32x16, f32, 16, 2);
        radix_by_semantic!(<S as Simd>::f32x16, f32, 4, 4); // square
        radix_by_semantic!(<S as Simd>::f32x16, f32, 8, 4);
        radix_by_semantic!(<S as Simd>::f32x16, f32, 2, 8); // square
        radix_by_semantic!(<S as Simd>::f32x16, f32, 4, 8);
        radix_by_semantic!(<S as Simd>::f32x16, f32, 8, 8); // GROUP == AVX2 inner L
        // GROUP == 16 spans every chunk: must take the lane-wise fallback.
        radix_by_semantic!(<S as Simd>::f32x16, f32, 2, 16);
        // Non-pow2 radix through the chunk-chain (inner picks its own strategy).
        radix_by_semantic!(<S as Simd>::f32x16, f32, 3, 2);
        radix_by_semantic!(<S as Simd>::f32x16, f32, 6, 4);
        // f64x8 (2-chunk on AVX2, inner L = 4).
        radix_by_semantic!(<S as Simd>::f64x8, f64, 4, 2);
        radix_by_semantic!(<S as Simd>::f64x8, f64, 8, 1);
        radix_by_semantic!(<S as Simd>::f64x8, f64, 2, 4); // GROUP == inner L
        radix_by_semantic!(<S as Simd>::f64x8, f64, 2, 8); // GROUP > inner L -> fallback
        // f32x8 (native on AVX2, ArrayRegister<F32x4, 2> on 128-bit backends).
        radix_by_semantic!(<S as Simd>::f32x8, f32, 4, 2);
        radix_by_semantic!(<S as Simd>::f32x8, f32, 8, 1);
        radix_by_semantic!(<S as Simd>::f32x8, f32, 2, 4);
        // 4-chunk arrays: f64x16 = ArrayRegister<F64x4, 4> on AVX2 (inner L = 4). The ONLY
        // configs there that exercise the flat-chunk `c / N`, `c % N` arithmetic with
        // N > 2, since a 2-chunk array cannot distinguish several index mistakes.
        radix_by_semantic!(<S as Simd>::f64x16, f64, 2, 1);
        radix_by_semantic!(<S as Simd>::f64x16, f64, 4, 1);
        radix_by_semantic!(<S as Simd>::f64x16, f64, 8, 1);
        radix_by_semantic!(<S as Simd>::f64x16, f64, 16, 1);
        radix_by_semantic!(<S as Simd>::f64x16, f64, 4, 2);
        radix_by_semantic!(<S as Simd>::f64x16, f64, 8, 2); // square: N == LANES/GROUP
        radix_by_semantic!(<S as Simd>::f64x16, f64, 2, 4);
        radix_by_semantic!(<S as Simd>::f64x16, f64, 4, 4); // square
        radix_by_semantic!(<S as Simd>::f64x16, f64, 3, 2); // non-pow2 through the chain
        radix_by_semantic!(<S as Simd>::f64x16, f64, 2, 8); // GROUP > inner L -> fallback
        // 4-chunk integer array too.
        radix_by_semantic!(<S as Simd>::i64x16, i64, 8, 2);
        radix_by_semantic!(<S as Simd>::i64x16, i64, 4, 1);
    }

    /// **The `ArrayRegister` chunk-chain override must equal the engine it replaced.**
    /// `register/array.rs` overrides `(de)interleave_radix_by` to delegate per chunk
    /// position to the INNER register. Without the override every one of these shapes
    /// runs `(de)interleave_radix_by_default` at the full array width, and a
    /// specialization that silently diverges from what it replaces is the exact bug
    /// this pins down. Covers 2-chunk and 4-chunk configs plus the `GROUP > inner L`
    /// shapes where the override declines and both sides must take the same fallback.
    fn radix_by_array_register_matches_default() {
        // f32x16: every GROUP dividing the inner width.
        radix_by_vs_default!(<S as Simd>::f32x16, f32, 4, 1);
        radix_by_vs_default!(<S as Simd>::f32x16, f32, 8, 1);
        radix_by_vs_default!(<S as Simd>::f32x16, f32, 16, 1);
        radix_by_vs_default!(<S as Simd>::f32x16, f32, 4, 2);
        radix_by_vs_default!(<S as Simd>::f32x16, f32, 8, 2);
        radix_by_vs_default!(<S as Simd>::f32x16, f32, 16, 2);
        radix_by_vs_default!(<S as Simd>::f32x16, f32, 4, 4);
        radix_by_vs_default!(<S as Simd>::f32x16, f32, 8, 4);
        radix_by_vs_default!(<S as Simd>::f32x16, f32, 2, 8);
        radix_by_vs_default!(<S as Simd>::f32x16, f32, 3, 2); // non-pow2 radix
        radix_by_vs_default!(<S as Simd>::f32x16, f32, 6, 4);
        radix_by_vs_default!(<S as Simd>::f32x16, f32, 2, 16); // GROUP > L -> both fall back
        // f64x16: the N > 2 flat-chunk arithmetic.
        radix_by_vs_default!(<S as Simd>::f64x16, f64, 4, 1);
        radix_by_vs_default!(<S as Simd>::f64x16, f64, 8, 1);
        radix_by_vs_default!(<S as Simd>::f64x16, f64, 16, 1);
        radix_by_vs_default!(<S as Simd>::f64x16, f64, 4, 2);
        radix_by_vs_default!(<S as Simd>::f64x16, f64, 8, 2);
        radix_by_vs_default!(<S as Simd>::f64x16, f64, 2, 4);
        radix_by_vs_default!(<S as Simd>::f64x16, f64, 4, 4);
        radix_by_vs_default!(<S as Simd>::f64x16, f64, 3, 2);
        radix_by_vs_default!(<S as Simd>::f64x16, f64, 2, 8); // GROUP > L -> both fall back
        radix_by_vs_default!(<S as Simd>::i64x16, i64, 8, 2);
        // 64-bit + integer 2-chunk (on AVX2) widths.
        radix_by_vs_default!(<S as Simd>::f64x8, f64, 4, 2);
        radix_by_vs_default!(<S as Simd>::f64x8, f64, 8, 1);
        radix_by_vs_default!(<S as Simd>::f64x8, f64, 2, 4);
        radix_by_vs_default!(<S as Simd>::i32x16, i32, 8, 2);
        radix_by_vs_default!(<S as Simd>::i32x16, i32, 16, 1);
        // f32x8 = ArrayRegister<F32x4, 2> on 128-bit backends (native on AVX2: trivial).
        radix_by_vs_default!(<S as Simd>::f32x8, f32, 4, 2);
        radix_by_vs_default!(<S as Simd>::f32x8, f32, 8, 1);
        radix_by_vs_default!(<S as Simd>::f32x8, f32, 2, 4);
    }

    fn interleave_roundtrip() {
        ilv!(<S as Simd>::f32x4);
        ilv!(<S as Simd>::f32x8);
        ilv!(<S as Simd>::f32x16);
        ilv!(<S as Simd>::f64x2);
        ilv!(<S as Simd>::f64x4);
        ilv!(<S as Simd>::i32x4);
        ilv!(<S as Simd>::i32x8);
        ilv!(<S as Simd>::u32x4);
        ilv!(<S as Simd>::i64x2);

        // ReducedRegister (3-in-4 padded) types: radix-2 uses the native prefix
        // trick, radix 3/5 fall to the gather over the logical 3 lanes. This is
        // the live path behind 3D `load_deinterleaved::<3>` (ReducedRegister has
        // no memory override, so it routes through the radix engine). Only the
        // radix ops are exercised, since `interleave2` (pairs) is ill-defined on an
        // odd lane count.
        ilv_radix!(<S as Simd3A>::f32x3A);
        ilv_radix!(<S as Simd3A>::i32x3A);
        ilv_radix!(<S as Simd3A>::f64x3A);
    }

    /// Absolute-layout check for `interleave_radix::<3>`: native sequences, the
    /// `ArrayRegister` per-chunk delegation, and the `ReducedRegister` gather.
    fn interleave_radix3_semantic() {
        radix3_semantic!(<S as Simd>::f32x8, f32);
        radix3_semantic!(<S as Simd>::f32x4, f32);
        radix3_semantic!(<S as Simd>::f32x16, f32);
        radix3_semantic!(<S as Simd>::i32x8, i32);
        radix3_semantic!(<S as Simd>::i32x4, i32);
        radix3_semantic!(<S as Simd3A>::f32x3A, f32); // ReducedRegister gather
    }

    /// Absolute-layout check for pair-granularity `interleave_by::<2>`: the native
    /// override where one exists, the lane-wise polyfill, and the `ArrayRegister` path.
    fn interleave_by2_semantic() {
        group2_semantic!(<S as Simd>::f32x8, f32);
        group2_semantic!(<S as Simd>::f32x4, f32);
        group2_semantic!(<S as Simd>::f32x16, f32);
        group2_semantic!(<S as Simd>::f64x4, f64);
        group2_semantic!(<S as Simd>::i32x8, i32);
    }

    fn reverse_widths() {
        rev!(<S as Simd>::f32x4);
        rev!(<S as Simd>::f32x8);
        rev!(<S as Simd>::f32x16);
        rev!(<S as Simd>::f64x2);
        rev!(<S as Simd>::f64x4);
        rev!(<S as Simd>::f64x8);
        rev!(<S as Simd>::i32x4);
        rev!(<S as Simd>::i32x8);
        rev!(<S as Simd>::i32x16);
        rev!(<S as Simd>::i64x2);
        rev!(<S as Simd>::i64x4);
        rev!(<S as Simd>::i16x8);
        rev!(<S as Simd>::u8x16);
    }
}
