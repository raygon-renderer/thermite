//! The `Compensated` Bernoulli tables, checked for the two things a generated split can
//! get wrong without failing to compile: a limb that does not belong to the value beside
//! it, and a table whose length was taken from the wrong format.
//!
//! The length check is the interesting one. It is easy to assume a wider type reaches
//! further into a factorially growing sequence, and it does not: a double-double carries
//! twice the mantissa of its base format but the SAME exponent range, so
//! `Compensated<f64>` overflows at `B_260` exactly where `f64` does. The tables are
//! therefore the same lengths, and this file is what stops someone "fixing" that.

#![cfg(feature = "special")]

use thermite_compensated::Compensated;
use thermite_special::tables::bernoulli::BernoulliNumbers;

/// Slice index holding `B_2n`. The tables omit `B_0` and `B_1`, so entry 0 is `B_2`.
const fn entry(two_n: usize) -> usize {
    two_n / 2 - 1
}

/// Same exact rationals the `thermite-special` test uses, over the range where both the
/// numerator and the denominator are exactly representable.
const EXACT: &[(usize, i64, i64)] = &[
    (2, 1, 6),
    (4, -1, 30),
    (6, 1, 42),
    (8, -1, 30),
    (10, 5, 66),
    (12, -691, 2730),
    (14, 7, 6),
    (16, -3617, 510),
    (18, 43867, 798),
    (20, -174611, 330),
    (22, 854513, 138),
    (24, -236364091, 2730),
    (26, 8553103, 6),
    (28, -23749461029, 870),
];

/// Widening the type buys precision, not reach.
#[test]
fn compensated_tables_are_the_same_length_as_their_base_format() {
    assert_eq!(<Compensated<f32> as BernoulliNumbers>::B2N.len(), f32::B2N.len());
    assert_eq!(<Compensated<f64> as BernoulliNumbers>::B2N.len(), f64::B2N.len());
}

/// The high limb of a double-double split is the base format's correctly rounded value,
/// so it must be the plain table entry bit for bit. A swapped or shifted limb shows up
/// here immediately.
#[test]
fn high_limb_is_the_plain_table_entry() {
    for (i, c) in <Compensated<f64> as BernoulliNumbers>::B2N.iter().enumerate() {
        assert_eq!(c.value, f64::B2N[i], "B_{} high limb", 2 * i + 2);
    }
    for (i, c) in <Compensated<f32> as BernoulliNumbers>::B2N.iter().enumerate() {
        assert_eq!(c.value, f32::B2N[i], "B_{} high limb", 2 * i + 2);
    }
}

/// A well-formed split has `|error| <= 1/2 ulp(value)`, which is what makes `value` the
/// correctly rounded leading limb rather than an arbitrary decomposition.
#[test]
fn error_limb_is_within_half_an_ulp() {
    for (i, c) in <Compensated<f64> as BernoulliNumbers>::B2N.iter().enumerate() {
        let half_ulp = ulp(c.value) * 0.5;
        assert!(
            c.error.abs() <= half_ulp,
            "B_{}: |error| {} exceeds half an ulp {}",
            2 * i + 2,
            c.error.abs(),
            half_ulp
        );
    }
}

fn ulp(v: f64) -> f64 {
    let next = f64::from_bits(v.abs().to_bits() + 1);
    next - v.abs()
}

/// The split must actually carry the value, not just look well formed. For an entry
/// `num/den`, the residual `den * value - num` is computed with one fused multiply-add
/// (so it is exact to a single rounding) and must be cancelled by `den * error`.
///
/// This is the check that would fail if the low limbs came from the wrong entries while
/// still satisfying the half-ulp bound above.
#[test]
fn split_reconstructs_the_exact_rational() {
    for &(two_n, num, den) in EXACT {
        let c = <Compensated<f64> as BernoulliNumbers>::B2N[entry(two_n)];
        let residual = c.value.mul_add(den as f64, -(num as f64));
        let corrected = residual + c.error * den as f64;

        // The residual itself is an f64 computation on quantities of size |error * den|,
        // so it carries ~1e-16 relative error of its own, about 1e-32 relative to num.
        let tol = (num as f64).abs() * 1e-28;
        assert!(
            corrected.abs() <= tol,
            "B_{two_n}: split leaves {corrected} of {num}/{den} uncorrected (tol {tol})"
        );
    }
}

/// The sequence iterator is written once over `RealPrimalMath`, so it must reach
/// `Compensated` with no double-double-specific code. This is the whole point of the
/// extension-trait shape: the head and the odd zeros come from the iterator, the values
/// from whichever table the element type carries.
#[test]
fn sequence_iterator_reaches_compensated() {
    use thermite::prelude::*;
    use thermite_special::bernoulli::BernoulliMath;

    type C = Compensated<Vector<f64>>;

    let b1 = Compensated {
        value: -0.5,
        error: 0.0,
    };
    let seq: Vec<Compensated<f64>> = C::bernoulli_numbers(b1).take(5).map(|c| c.extract::<0>()).collect();

    assert_eq!(seq[0].value, 1.0, "B_0");
    assert_eq!(seq[1].value, -0.5, "B_1 as passed");
    assert_eq!(seq[2].value, 1.0 / 6.0, "B_2 high limb");
    assert_eq!(seq[3].value, 0.0, "B_3");
    assert_eq!(seq[4].value, -1.0 / 30.0, "B_4 high limb");

    // The double-double table is what makes this worth doing: the low limb carries the
    // bits an f64 table cannot.
    assert_ne!(seq[2].error, 0.0, "B_2 should carry a nonzero low limb");
    assert_eq!(seq[0].error, 0.0, "B_0 = 1 is exact");
}
