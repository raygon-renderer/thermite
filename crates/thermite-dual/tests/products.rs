//! `difference_of_products` / `sum_of_products` on `Dual<V, N>`: `a*b - b*a` must be
//! exactly zero on the value part and every derivative part, or a gradient picks up
//! noise exactly where a sign test is about to look.
//!
//! Dispatches to `X86V3`: without hardware FMA the inner call lowers to naive
//! `a*b - c*d`, every grouping gives exact zero, and the bug this catches (the wrong
//! pairing loses the zero on about 22% of inputs) is invisible.

#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use thermite::isa::InstructionSet;
use thermite::math::policy::policies::Precision;
use thermite::math::{CoreMath, CoreMathWithPolicy};
use thermite::prelude::*;
use thermite::simd::{HasIsa, Simd};
use thermite::vector::ops::MulAddExt;

use thermite_dual::Dual;

type S = thermite::backend::x86_v3::X86V3;
type V = Vector<<S as Simd>::f64x4>;
/// One partial is enough; the derivative parts are independent.
type D = Dual<V, 1>;

// Without a fused multiply every grouping passes and the suite is vacuous.
const _: () = assert!(
    matches!(<V as MulAddExt>::HAS_NATIVE_FMA, thermite::tribool::True),
    "these tests only exercise the compensated lowering, which needs native FMA"
);

/// `(a.re, a.dual, b.re, b.dual)` quadruples, each checked to make the wrong grouping
/// return nonzero for both degenerate cases. Re-check if edited.
const PAIRS: &[(f64, f64, f64, f64)] = &[
    (0.1, 0.3, 1.0 / 3.0, core::f64::consts::PI),
    (0.1, 0.7, 1.0 / 7.0, core::f64::consts::PI),
    (core::f64::consts::PI, 6.02214076e23, -1.380649e-23, -0.875),
    (0.3, 9.007199254740992e15, 2.5e-5, 3.7e12),
    (1.0 / 3.0, core::f64::consts::PI, 2.2250738585072013e-152, 2.2250738585072013e-152),
];

#[inline(always)]
fn d(re: f64, du: f64) -> D {
    Dual { re: V::splat(re), dual: [V::splat(du)] }
}

#[inline(always)]
fn parts(x: D) -> (f64, f64) {
    (x.re.extract::<0>(), x.dual[0].extract::<0>())
}

#[thermite::dispatch(S)]
fn self_cross_product_imp() {
    for &(ar, ad, br, bd) in PAIRS {
        let (a, b) = (d(ar, ad), d(br, bd));

        let (re, du) = parts(a.difference_of_products(b, b, a));
        assert_eq!(re, 0.0, "default policy, value: ({ar:e},{ad:e}) x ({br:e},{bd:e})");
        assert_eq!(du, 0.0, "default policy, derivative: ({ar:e},{ad:e}) x ({br:e},{bd:e})");

        let (re, du) = parts(a.difference_of_products_p::<Precision>(b, b, a));
        assert_eq!(re, 0.0, "Precision, value: ({ar:e},{ad:e}) x ({br:e},{bd:e})");
        assert_eq!(du, 0.0, "Precision, derivative: ({ar:e},{ad:e}) x ({br:e},{bd:e})");
    }
}

#[thermite::dispatch(S)]
fn sum_of_products_cancels_imp() {
    for &(ar, ad, br, bd) in PAIRS {
        let (a, b) = (d(ar, ad), d(br, bd));

        // `a*b + (-b)*a` is zero. The compensation enters with the opposite sign here.
        let (re, du) = parts(a.sum_of_products(b, -b, a));
        assert_eq!(re, 0.0, "default policy, value: ({ar:e},{ad:e}) + ({br:e},{bd:e})");
        assert_eq!(du, 0.0, "default policy, derivative: ({ar:e},{ad:e}) + ({br:e},{bd:e})");

        let (re, du) = parts(a.sum_of_products_p::<Precision>(b, -b, a));
        assert_eq!(re, 0.0, "Precision, value: ({ar:e},{ad:e}) + ({br:e},{bd:e})");
        assert_eq!(du, 0.0, "Precision, derivative: ({ar:e},{ad:e}) + ({br:e},{bd:e})");
    }
}

#[thermite::dispatch(S)]
fn value_part_matches_the_plain_vector_imp() {
    // The dual's value must be bit-identical to evaluating on the plain vector: a
    // dual is supposed to change what else you learn, never the answer.
    for &(ar, ad, br, bd) in PAIRS {
        let (a, b) = (d(ar, ad), d(br, bd));
        let (cc, dd) = (d(bd, ar), d(ad, br));

        let got = a.difference_of_products(b, cc, dd).re.extract::<0>();
        let plain = V::splat(ar)
            .difference_of_products(V::splat(br), V::splat(bd), V::splat(ad))
            .extract::<0>();

        assert_eq!(got.to_bits(), plain.to_bits(), "value drifted from the plain vector");
    }
}

fn skip() -> bool {
    if <S as HasIsa>::ISA > InstructionSet::get() {
        eprintln!("skipped: {:?} unavailable (host is {:?})", <S as HasIsa>::ISA, InstructionSet::get());
        return true;
    }
    false
}

#[test]
fn self_cross_product_is_zero() {
    if skip() {
        return;
    }
    self_cross_product_imp();
}

#[test]
fn sum_of_products_cancels_exactly() {
    if skip() {
        return;
    }
    sum_of_products_cancels_imp();
}

#[test]
fn value_part_matches_the_plain_vector() {
    if skip() {
        return;
    }
    value_part_matches_the_plain_vector_imp();
}
