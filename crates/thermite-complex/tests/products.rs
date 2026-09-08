//! `difference_of_products` / `sum_of_products` on `Complex<V>`: `a*b - b*a` must be
//! exactly zero in both components.
//!
//! Dispatches to `X86V3` rather than the 1-lane scalar backend: without hardware FMA the
//! inner call lowers to naive `a*b - c*d`, every grouping gives exact zero, and the bug
//! this catches (the wrong pairing loses the zero on about 22% of inputs) is invisible.

#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use thermite::isa::InstructionSet;
use thermite::math::policy::policies::Precision;
use thermite::math::{CoreMath, CoreMathWithPolicy};
use thermite::prelude::*;
use thermite::simd::{HasIsa, Simd};
use thermite::vector::ops::MulAddExt;

use thermite_complex::Complex;

type S = thermite::backend::x86_v3::X86V3;
type V = Vector<<S as Simd>::f64x4>;
type C = Complex<V>;

// Without a fused multiply every grouping passes and the suite is vacuous.
const _: () = assert!(
    matches!(<V as MulAddExt>::HAS_NATIVE_FMA, thermite::tribool::True),
    "these tests only exercise the compensated lowering, which needs native FMA"
);

/// `(a.re, a.im, b.re, b.im)` quadruples, each checked to make the wrong grouping return
/// nonzero for both degenerate cases. Random constants mostly pass either way (only ~22%
/// discriminate), so re-check these if edited. Spans ~500 binades.
const PAIRS: &[(f64, f64, f64, f64)] = &[
    (0.1, 0.3, 1.0 / 3.0, core::f64::consts::PI),
    (0.1, 0.7, 1.0 / 7.0, core::f64::consts::PI),
    (core::f64::consts::PI, 6.02214076e23, -1.380649e-23, -0.875),
    (0.3, 9.007199254740992e15, 2.5e-5, 3.7e12),
    (1.0 / 3.0, core::f64::consts::PI, 2.2250738585072013e-152, 2.2250738585072013e-152),
];

#[inline(always)]
fn c(re: f64, im: f64) -> C {
    Complex::new(V::splat(re), V::splat(im))
}

#[inline(always)]
fn parts(z: C) -> (f64, f64) {
    (z.re.extract::<0>(), z.im.extract::<0>())
}

#[thermite::dispatch(S)]
fn self_cross_product_imp() {
    for &(ar, ai, br, bi) in PAIRS {
        let (a, b) = (c(ar, ai), c(br, bi));

        let (re, im) = parts(a.difference_of_products(b, b, a));
        assert_eq!(re, 0.0, "default policy, real: ({ar:e},{ai:e}) x ({br:e},{bi:e})");
        assert_eq!(im, 0.0, "default policy, imag: ({ar:e},{ai:e}) x ({br:e},{bi:e})");

        let (re, im) = parts(a.difference_of_products_p::<Precision>(b, b, a));
        assert_eq!(re, 0.0, "Precision, real: ({ar:e},{ai:e}) x ({br:e},{bi:e})");
        assert_eq!(im, 0.0, "Precision, imag: ({ar:e},{ai:e}) x ({br:e},{bi:e})");
    }
}

#[thermite::dispatch(S)]
fn sum_of_products_cancels_imp() {
    for &(ar, ai, br, bi) in PAIRS {
        let (a, b) = (c(ar, ai), c(br, bi));

        // `a*b + (-b)*a` is zero. The compensation enters with the opposite sign here.
        let (re, im) = parts(a.sum_of_products(b, -b, a));
        assert_eq!(re, 0.0, "default policy, real: ({ar:e},{ai:e}) + ({br:e},{bi:e})");
        assert_eq!(im, 0.0, "default policy, imag: ({ar:e},{ai:e}) + ({br:e},{bi:e})");

        let (re, im) = parts(a.sum_of_products_p::<Precision>(b, -b, a));
        assert_eq!(re, 0.0, "Precision, real: ({ar:e},{ai:e}) + ({br:e},{bi:e})");
        assert_eq!(im, 0.0, "Precision, imag: ({ar:e},{ai:e}) + ({br:e},{bi:e})");
    }
}

#[thermite::dispatch(S)]
fn agrees_with_the_naive_form_imp() {
    // Same value as the naive expansion to within a couple of ulp.
    for &(ar, ai, br, bi) in PAIRS {
        let (a, b) = (c(ar, ai), c(br, bi));
        let (cc, d) = (c(bi, ar), c(ai, br));

        let (gr, gi) = parts(a.difference_of_products(b, cc, d));
        let (nr, ni) = parts(a * b - cc * d);

        for (got, naive, half) in [(gr, nr, "real"), (gi, ni, "imag")] {
            let scale = naive.abs().max(got.abs());
            assert!(
                (got - naive).abs() <= 1e-12 * scale,
                "{half}: compensated {got:e} vs naive {naive:e}"
            );
        }
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
fn agrees_with_the_naive_form() {
    if skip() {
        return;
    }
    agrees_with_the_naive_form_imp();
}
