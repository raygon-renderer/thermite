//! Double-double arithmetic keeps its error terms under `algebraic-scalar`.
//!
//! Run both ways:
//!
//! ```text
//! cargo nextest run -p thermite-compensated --features thermite/algebraic-scalar
//! ```
//!
//! Every assertion is on an error term. A folded transformation leaves the value plausible
//! and the low word zero, which is why this went unnoticed for so long.
//!
//! These are regression pins, not guards. Reverting the strict-join fixes in `lib.rs` with
//! the feature on leaves this file passing (the folding needs the surrounding kernel).
//! The suites that actually catch those, under `--features thermite/algebraic-scalar`:
//!
//! | operator | suite |
//! |---|---|
//! | `Div<V>`, `Div<Self>`, `div_scalar` | `trig_pi`, `erfinv`, `expint`, `elliptic`, `erfc_tail` |
//! | `Sub<Self>` | `gamma` (`digamma`, `trigamma`, `beta`) |

use thermite::prelude::*;
use thermite_compensated::Compensated;

/// The 1-lane scalar backend, the only one the feature touches. `ArrayRegister` widths
/// are covered by thermite's own `eft` suite.
type V = Vector<f64>;
type C = Compensated<V>;

fn c(x: f64) -> C {
    C::new(V::splat(x))
}

fn parts(x: C) -> (f64, f64) {
    (x.uncompensated().extract::<0>(), x.error().extract::<0>())
}

/// `1 + 2^-60` rounds to `1`, so everything is in the error term.
#[test]
fn addition_keeps_its_error_term() {
    let (hi, lo) = parts(c(1.0) + c(2f64.powi(-60)));

    assert_eq!(hi, 1.0);
    assert_eq!(
        lo,
        2f64.powi(-60),
        "error term folded to zero - `ScalarValue for Vector<R>` is not delegating two_sum \
         to FloatVectorWithBits"
    );
}

/// The subtractive twin.
#[test]
fn subtraction_keeps_its_error_term() {
    let (hi, lo) = parts(c(1.0) - c(2f64.powi(-60)));

    assert_eq!(hi, 1.0);
    assert_eq!(lo, -2f64.powi(-60), "error term folded to zero");
}

/// `0.1 * 0.1` is not representable, so the product has a genuine second word.
#[test]
fn multiplication_keeps_its_error_term() {
    let (hi, lo) = parts(c(0.1) * c(0.1));

    assert_eq!(hi, 0.1f64 * 0.1);
    assert_ne!(lo, 0.0, "product error term folded to zero - check two_prod's delegation");

    // Exact square of the f64 nearest 0.1, checked with mpmath at 60 digits.
    assert_eq!(hi, 0.010000000000000002f64);
    assert_eq!(
        lo, -8.326672684688674e-19,
        "second word of 0.1*0.1 is {lo:e}, expected -8.326672684688674e-19"
    );
}

/// Division depends on association as well as on the transformations: the remainder
/// chain subtracts the two product words in order.
#[test]
fn division_keeps_its_error_term() {
    let (hi, lo) = parts(c(1.0) / c(3.0));

    assert_eq!(hi, 1.0f64 / 3.0);
    assert_ne!(lo, 0.0, "division error term folded to zero");
    assert!(
        (lo - 1.850371707708594e-17).abs() < 1e-32,
        "second word of 1/3 is {lo:e}, expected 1.850371707708594e-17 - a wrong-but-nonzero \
         value here means the remainder chain `(a - p_hi) - p_lo` was re-bracketed rather \
         than the transformation being folded"
    );
}

/// Same remainder chain, different entry point. The compile-time rational constants are
/// built on this.
#[test]
fn from_fraction_keeps_its_error_term() {
    let (hi, lo) = parts(C::from_fraction(V::splat(1.0), V::splat(3.0)));

    assert_eq!(hi, 1.0f64 / 3.0);
    assert!(
        (lo - 1.850371707708594e-17).abs() < 1e-32,
        "second word of from_fraction(1, 3) is {lo:e}"
    );
}

/// `Compensated / V`, the first operator found to fold. Dividing by two is the sharpest
/// case: the quotient is exact, so the remainder is zero and the whole second word is the
/// `+ self.error` join. It came back `0.0` under the feature.
///
/// Regression pin only. Reverting the fix leaves this passing; `tests/trig_pi.rs` under
/// the feature is what fails.
#[test]
fn division_by_a_plain_vector_keeps_its_error_term() {
    let (hi, lo) = parts(C::from_fraction(V::splat(1.0), V::splat(3.0)) / V::splat(2.0));

    assert_eq!(hi, 1.0f64 / 6.0);
    assert_ne!(lo, 0.0, "error term folded to zero - check `Div<V>`'s remainder join");
    assert_eq!(
        lo, 9.25185853854297e-18,
        "second word of (1/3)/2 is {lo:e}, expected 9.25185853854297e-18"
    );
}

/// A chain of subtractions against a running accumulator, the shape of the kernels that
/// failed. Regression pin only: this passes even with `Sub`'s inner
/// `self.error - rhs.error` left reassociable. `tests/gamma.rs` under the feature is what
/// catches that (`digamma`/`trigamma` dropped to 5.5e-17 relative).
///
/// `10 - H_30` checked against `fractions.Fraction`, pair exact to 2.4e-33.
#[test]
fn a_chain_of_subtractions_keeps_its_error_term() {
    let mut x = c(10.0);

    for k in 1..=30u32 {
        x = x - C::from_fraction(V::splat(1.0), V::splat(f64::from(k)));
    }

    let (hi, lo) = parts(x);

    assert_eq!(hi, 6.005012869079609);
    assert_eq!(
        lo, -3.194409176879754e-17,
        "second word of 10 - H_30 is {lo:e}, expected -3.194409176879754e-17"
    );
}
