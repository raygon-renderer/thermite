//! `Dual`'s Hermite functions keep the plain vector's value, bit for bit.
//!
//! `Dual` passes `EXACT_FMA = true` where the trait defaults pass `false`; see the comment
//! on its `hermite_function_n` override for why that is sound (0.24 ulp value / 0.30
//! derivative with the correction, 87.35 without, over `|x| = 32..53`).
//!
//! Two ways this goes vacuous: without hardware FMA the correction is off for everyone,
//! so this dispatches to `X86V3`; and an integer `x` has an exact square with no residual,
//! so every argument is a non-integer large enough that `x^2 eps` is several ulp.

#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use thermite::isa::InstructionSet;
use thermite::math::policy::policies::Precision;
use thermite::prelude::*;
use thermite::simd::{HasIsa, Simd};
use thermite::vector::ops::MulAddExt;
use thermite_special::SpecialMathWithPolicy;

use thermite_dual::Dual;

type S = thermite::backend::x86_v3::X86V3;
type V = Vector<<S as Simd>::f64x4>;
type D = Dual<V, 1>;

const _: () = assert!(
    matches!(<V as MulAddExt>::HAS_NATIVE_FMA, thermite::tribool::True),
    "the seed correction is gated on native FMA; without it this suite proves nothing"
);

/// Non-integer, spread across the bands where the correction is worth 1, 6, 25 and 87 ulp.
const XS: &[f64] = &[
    3.7000000000000002,
    5.123456789,
    -9.87654321,
    13.333333333333334,
    21.7182818284,
    -27.1828182845,
    39.1415926535,
    47.7724538509,
];

#[thermite::dispatch(S)]
fn value_matches_the_plain_vector_imp() {
    for &xv in XS {
        let x = V::splat(xv);
        let d = Dual::new(x, [V::ONE]);

        // Const degree.
        let plain = <V as SpecialMathWithPolicy>::hermite_function_n_p::<Precision, 6>(x);
        let dual = <D as SpecialMathWithPolicy>::hermite_function_n_p::<Precision, 6>(d);
        assert_eq!(
            dual.re.extract::<0>().to_bits(),
            plain.extract::<0>().to_bits(),
            "hermite_function_n at x = {xv:e}: dual value drifted from the plain vector"
        );

        // Runtime degree.
        let plain = <V as SpecialMathWithPolicy>::hermite_function_p::<Precision>(x, 6);
        let dual = <D as SpecialMathWithPolicy>::hermite_function_p::<Precision>(d, 6);
        assert_eq!(
            dual.re.extract::<0>().to_bits(),
            plain.extract::<0>().to_bits(),
            "hermite_function at x = {xv:e}: dual value drifted from the plain vector"
        );
    }
}

#[thermite::dispatch(S)]
fn series_value_matches_the_plain_vector_imp() {
    const C: [f64; 5] = [0.5, -1.25, 0.75, 2.0, -0.125];

    // Dual constants (zero derivative), so the values must agree bit for bit.
    const CD: [Dual<f64, 1>; 5] = [
        Dual::constant(C[0]),
        Dual::constant(C[1]),
        Dual::constant(C[2]),
        Dual::constant(C[3]),
        Dual::constant(C[4]),
    ];

    for &xv in XS {
        let x = V::splat(xv);
        let d = Dual::new(x, [V::ONE]);

        let plain = <V as SpecialMathWithPolicy>::hermite_function_series_n_p::<Precision, 5>(x, &C);
        let dual = <D as SpecialMathWithPolicy>::hermite_function_series_n_p::<Precision, 5>(d, &CD);
        assert_eq!(
            dual.re.extract::<0>().to_bits(),
            plain.extract::<0>().to_bits(),
            "hermite_function_series_n at x = {xv:e}"
        );

        let plain = <V as SpecialMathWithPolicy>::hermite_function_series_p::<Precision>(x, &C);
        let dual = <D as SpecialMathWithPolicy>::hermite_function_series_p::<Precision>(d, &CD);
        assert_eq!(
            dual.re.extract::<0>().to_bits(),
            plain.extract::<0>().to_bits(),
            "hermite_function_series at x = {xv:e}"
        );
    }
}

#[thermite::dispatch(S)]
fn derivative_follows_the_recurrence_imp() {
    // psi_n'(x) = sqrt(n/2) psi_{n-1}(x) - sqrt((n+1)/2) psi_{n+1}(x), verified against
    // 40-digit mpmath (worst relative mismatch 5.4e-41). The dual traces the recurrence
    // rather than using this rule, so it is an independent check on the derivative.
    const N: usize = 6;

    for &xv in XS {
        let x = V::splat(xv);
        let d = Dual::new(x, [V::ONE]);

        let got = <D as SpecialMathWithPolicy>::hermite_function_n_p::<Precision, N>(d).dual[0]
            .extract::<0>();

        let lo = <V as SpecialMathWithPolicy>::hermite_function_n_p::<Precision, { N - 1 }>(x).extract::<0>();
        let hi = <V as SpecialMathWithPolicy>::hermite_function_n_p::<Precision, { N + 1 }>(x).extract::<0>();
        let want = (N as f64 / 2.0).sqrt() * lo - ((N + 1) as f64 / 2.0).sqrt() * hi;

        // `psi_n` decays like `e^{-x^2/2}`, so the largest arguments land in the denormals
        // (`psi_6(39.14)` is 7e-323) where a relative tolerance is meaningless. The
        // bit-identity tests above still cover them.
        if want.abs() < 1e-280 {
            continue;
        }

        let scale = want.abs().max(got.abs());
        assert!(
            (got - want).abs() <= 1e-11 * scale,
            "psi_{N}'({xv:e}): dual gave {got:e}, recurrence gives {want:e}"
        );
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
fn value_matches_the_plain_vector() {
    if skip() {
        return;
    }
    value_matches_the_plain_vector_imp();
}

#[test]
fn series_value_matches_the_plain_vector() {
    if skip() {
        return;
    }
    series_value_matches_the_plain_vector_imp();
}

#[test]
fn derivative_follows_the_recurrence() {
    if skip() {
        return;
    }
    derivative_follows_the_recurrence_imp();
}
