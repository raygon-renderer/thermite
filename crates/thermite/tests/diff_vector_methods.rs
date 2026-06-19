//! Coverage for `GenericVector`/`NumericVector` methods in `vector/vector.rs`
//! that the differential suites don't reach: the scalar-fallback `map`/`fold`/
//! `reduce`, dynamic lane access (`extractv`/`insertv`/`broadcastv`) and the
//! const-index forms, horizontal reductions (`sum`/`prod`/`min`/`max`/
//! `min_max_element`/`arg_minmax`), `is_zero`/`is_all_zero`, `indexed`/`single`/
//! `reverse`, and the integer `avg` family.
//!
//! Deterministic distinct-value inputs (no ties), values exactly representable in
//! f32/i32, so comparisons are exact.
#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use thermite::Vector;
use thermite::prelude::*;
use thermite::simd::Simd;

use thermite::backend::scalar::Scalar;
use thermite::backend::x86_v1::X86V1;
use thermite::backend::x86_v2::X86V2;
use thermite::backend::x86_v3::X86V3;

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

macro_rules! methods_suite {
    ($mod:ident, $backend:ty) => {
        mod $mod {
            use super::*;

            #[test]
            fn float_methods() {
                common_methods!(
                    Vector<<$backend as Simd>::f32x4>,
                    f32,
                    concat!(stringify!($mod), " f32x4")
                );
            }

            #[test]
            fn int_methods() {
                common_methods!(
                    Vector<<$backend as Simd>::i32x4>,
                    i32,
                    concat!(stringify!($mod), " i32x4")
                );
            }
        }
    };
}

methods_suite!(v3, X86V3);
methods_suite!(v2, X86V2);
methods_suite!(v1, X86V1);
methods_suite!(scalar, Scalar);

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

/// `deinterleave(interleave(a, b)) == (a, b)` across widths/types/backends.
/// Catches per-register deinterleave bugs (e.g. f32x4 reading a shadowed operand).
macro_rules! ilv {
    ($reg:ty) => {{
        type V = Vector<$reg>;
        let a = V::indexed();
        let b = a + a; // distinct from a (except lane 0), enough to expose lane mixups
        let (lo, hi) = a.interleave(b);
        let (a2, b2) = lo.deinterleave(hi);
        let to = |v: V| -> Vec<f64> { v.into_array().as_slice().iter().map(|&x| x as f64).collect() };
        assert_eq!(to(a2), to(a), concat!("deinterleave a ", stringify!($reg)));
        assert_eq!(to(b2), to(b), concat!("deinterleave b ", stringify!($reg)));
    }};
}

#[test]
fn interleave_roundtrip() {
    ilv!(<X86V3 as Simd>::f32x4);
    ilv!(<X86V3 as Simd>::f32x8);
    ilv!(<X86V3 as Simd>::f64x2);
    ilv!(<X86V3 as Simd>::f64x4);
    ilv!(<X86V3 as Simd>::i32x4);
    ilv!(<X86V3 as Simd>::i32x8);
    ilv!(<X86V3 as Simd>::u32x4);
    ilv!(<X86V3 as Simd>::i64x2);
    ilv!(<X86V2 as Simd>::f32x4);
    ilv!(<X86V2 as Simd>::f64x2);
    ilv!(<X86V2 as Simd>::i32x4);
    ilv!(<X86V2 as Simd>::i64x2);
    ilv!(<X86V1 as Simd>::f32x4);
    ilv!(<X86V1 as Simd>::f32x8); // ArrayRegister-emulated on v1
    ilv!(<X86V1 as Simd>::f64x2);
    ilv!(<X86V1 as Simd>::i32x4);
    ilv!(<X86V1 as Simd>::i64x2);
}

#[test]
fn reverse_widths() {
    rev!(<X86V3 as Simd>::f32x4);
    rev!(<X86V3 as Simd>::f32x8);
    rev!(<X86V3 as Simd>::f64x2);
    rev!(<X86V3 as Simd>::f64x4);
    rev!(<X86V3 as Simd>::i32x8);
    rev!(<X86V3 as Simd>::i64x4);
    rev!(<X86V2 as Simd>::f32x4);
    rev!(<X86V2 as Simd>::f32x8); // ArrayRegister-emulated on v2
    rev!(<X86V2 as Simd>::f64x2);
    rev!(<X86V2 as Simd>::i64x2);
    rev!(<X86V1 as Simd>::f32x4);
    rev!(<X86V1 as Simd>::f32x8); // ArrayRegister-emulated on v1
    rev!(<X86V1 as Simd>::f64x2);
    rev!(<X86V1 as Simd>::i64x2);
}
