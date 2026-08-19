//! The generated `B2N` tables are data, so nothing about them fails to compile. A
//! permuted entry, a dropped row, a table truncated early, or an off-by-one in the
//! even-index convention would all still build and still return plausible numbers.
//!
//! The tables start at `B_2`, so entry `i` holds `B_{2i+2}`. That offset is exactly the
//! kind of thing a refactor slips on, so every test below states the subscript it means
//! rather than reusing the slice index.
//!
//! Two independent checks do the work. The small entries are exact rationals of integers
//! that fit comfortably in an `f64` mantissa, so they are checked for bit equality. The
//! whole tail is checked against the asymptotic ratio
//!
//!     B_2n / B_2n-2 = -(2n)(2n-1)/(4 pi^2) * zeta(2n)/zeta(2n-2)
//!
//! whose zeta factor is within 1e-6 of 1 past n = 10. That covers every entry rather
//! than a hand-picked few, and it is sensitive to exactly the failures a spot check is
//! blind to: a swapped pair, a missing row, or a table that stops in the wrong place.

use thermite_special::tables::bernoulli::{BernoulliNumbers, bernoulli_b2n};

/// Slice index holding `B_2n`. The tables omit `B_0` and `B_1`, so entry 0 is `B_2`.
const fn entry(two_n: usize) -> usize {
    two_n / 2 - 1
}

/// `(2n, numerator, denominator)` for the entries whose numerator and denominator both
/// fit in an `f64` mantissa. `num as f64 / den as f64` is then a correctly rounded
/// division of two exactly representable integers, i.e. the true `B_2n` rounded once -
/// so `f64` entries are compared for bit equality, not with a tolerance.
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

#[test]
fn f64_small_entries_are_exact() {
    for &(two_n, num, den) in EXACT {
        let expected = num as f64 / den as f64;
        assert_eq!(f64::B2N[entry(two_n)], expected, "B_{two_n}");
    }
}

#[test]
fn f32_small_entries_round_the_same_value() {
    for &(two_n, num, den) in EXACT {
        let expected = (num as f64 / den as f64) as f32;
        assert_eq!(f32::B2N[entry(two_n)], expected, "B_{two_n}");
    }
}

/// The head of each table is `B_2`, not `B_0` and not `B_4`. Reintroducing `B_0` or
/// slicing one too many off the front shifts every other test by one, so pin it here
/// where the failure message says what actually happened.
#[test]
fn tables_start_at_b2() {
    assert_eq!(f64::B2N[0], 1.0 / 6.0, "first f64 entry should be B_2");
    assert_eq!(f32::B2N[0], (1.0f64 / 6.0) as f32, "first f32 entry should be B_2");
}

/// Documented in the module: 32 entries spanning `B_2..=B_64` for `f32`, 129 spanning
/// `B_2..=B_258` for `f64`. The upper ends are the format's real boundaries, not a chosen
/// cutoff, so they are asserted literally. If a table changes length, the docs are wrong.
#[test]
fn table_lengths_match_the_documented_boundaries() {
    assert_eq!(f32::B2N.len(), 32, "f32 table should span B_2..=B_64");
    assert_eq!(f64::B2N.len(), 129, "f64 table should span B_2..=B_258");
}

#[test]
fn no_entry_is_infinite_or_nan() {
    assert!(f32::B2N.iter().all(|b| b.is_finite()));
    assert!(f64::B2N.iter().all(|b| b.is_finite()));
}

/// `sign(B_2n) = (-1)^(n+1)`, so with entry `i` holding `B_{2i+2}` the even entries are
/// the positive ones. A swapped adjacent pair survives the magnitude checks but not this.
#[test]
fn signs_alternate() {
    for (i, &b) in f64::B2N.iter().enumerate() {
        assert_eq!(b > 0.0, i % 2 == 0, "sign of B_{}", 2 * i + 2);
    }
}

/// `|B_2n|` falls to a minimum at `B_6 = 1/42` and grows monotonically after it. The
/// module docs lean on this to claim no entry is denormal.
#[test]
fn magnitude_bottoms_out_at_b6_then_grows() {
    let min_index = (0..f64::B2N.len())
        .min_by(|&a, &b| f64::B2N[a].abs().total_cmp(&f64::B2N[b].abs()))
        .expect("table is non-empty");
    assert_eq!(min_index, entry(6), "minimum |B_2n| should be B_6 = 1/42");
    assert_eq!(f64::B2N[entry(6)], 1.0 / 42.0);

    for i in (entry(6) + 1)..f64::B2N.len() {
        assert!(
            f64::B2N[i].abs() > f64::B2N[i - 1].abs(),
            "|B_{}| should exceed |B_{}|",
            2 * i + 2,
            2 * i
        );
    }

    assert!(f64::B2N.iter().all(|b| b.is_normal()));
}

/// The structural check on the whole tail: consecutive entries must sit in the ratio the
/// asymptotic form demands. Past `n = 10` the zeta correction is under 1e-5, so this
/// pins every remaining entry to five digits without needing a table of references.
#[test]
fn tail_follows_the_asymptotic_ratio() {
    const FOUR_PI_SQUARED: f64 = 4.0 * core::f64::consts::PI * core::f64::consts::PI;

    // Entry `i` is B_{2i+2}, so `n >= 10` starts at i = 9.
    for i in 9..f64::B2N.len() {
        let two_n = (2 * i + 2) as f64;
        let expected = -(two_n * (two_n - 1.0)) / FOUR_PI_SQUARED;
        let actual = f64::B2N[i] / f64::B2N[i - 1];
        let rel = ((actual - expected) / expected).abs();
        assert!(
            rel < 1e-5,
            "B_{}/B_{} = {actual}, asymptotic form wants {expected} (rel {rel:e})",
            2 * i + 2,
            2 * i
        );
    }
}

/// The table is not cut short: one more entry would genuinely overflow. Uses the same
/// ratio, which is an underestimate of the true growth here, so clearing the format's
/// maximum with it is conclusive.
#[test]
fn one_past_the_end_would_overflow() {
    const FOUR_PI_SQUARED: f64 = 4.0 * core::f64::consts::PI * core::f64::consts::PI;

    // Last entry is B_{2*len}; the one that did not fit is B_{2*len+2}.
    let two_n = (2 * f64::B2N.len() + 2) as f64;
    let next = f64::B2N.last().expect("non-empty").abs() / FOUR_PI_SQUARED * (two_n * (two_n - 1.0));
    assert!(next.is_infinite(), "B_{two_n} should overflow f64, got {next}");

    let two_n = (2 * f32::B2N.len() + 2) as f64;
    let last = f32::B2N.last().expect("non-empty").abs() as f64;
    let next = last / FOUR_PI_SQUARED * (two_n * (two_n - 1.0));
    assert!(next > f32::MAX as f64, "B_{two_n} should overflow f32, got {next}");
}

/// The accessor takes `n` as in `B_2n`, not the slice index, and answers `None` for
/// everything the table does not carry: `B_0` at the bottom, overflow at the top.
#[test]
fn lookup_takes_the_subscript_and_is_none_off_the_ends() {
    assert_eq!(bernoulli_b2n::<f64>(0), None, "B_0 is not tabulated");
    assert_eq!(bernoulli_b2n::<f64>(1), Some(1.0 / 6.0), "B_2");
    assert_eq!(bernoulli_b2n::<f64>(129), Some(f64::B2N[128]), "B_258");
    assert_eq!(bernoulli_b2n::<f64>(130), None, "B_260 overflows");

    assert_eq!(bernoulli_b2n::<f32>(0), None);
    assert_eq!(bernoulli_b2n::<f32>(32), Some(f32::B2N[31]), "B_64");
    assert_eq!(bernoulli_b2n::<f32>(33), None, "B_66 overflows");
}

/// The `f32` table is the `f64` one rounded, entry for entry, over its shorter range.
/// Catches a table generated at the wrong width or indexed with the wrong stride.
#[test]
fn f32_table_is_the_f64_table_rounded() {
    for (i, &b) in f32::B2N.iter().enumerate() {
        assert_eq!(b, f64::B2N[i] as f32, "B_{}", 2 * i + 2);
    }
}

// ---------------------------------------------------------------------------
// The sequence iterator: puts B_0, B_1 and the odd zeros back around the table.
// ---------------------------------------------------------------------------

mod sequence {
    use super::{EXACT, entry};
    use thermite::prelude::*;
    // `BernoulliNumbers` comes through the `bernoulli` module's re-export rather than
    // `tables`, which is the import path the public API is meant to be reached by.
    use thermite_special::bernoulli::{BernoulliMath, BernoulliNumbers, BernoulliSequence};

    type V = Vector<f64>;

    fn lane0(v: V) -> f64 {
        v.extract::<0>()
    }

    /// The head is the caller's business, so both conventions must come back verbatim
    /// while everything after them stays identical.
    #[test]
    fn head_is_the_callers_choice_and_nothing_else_moves() {
        for b1 in [-0.5, 0.5] {
            let seq: Vec<f64> = V::bernoulli_numbers(b1).take(5).map(lane0).collect();
            assert_eq!(seq[0], 1.0, "B_0 is always 1");
            assert_eq!(seq[1], b1, "B_1 is whatever the caller passed");
            assert_eq!(seq[2], 1.0 / 6.0, "B_2");
            assert_eq!(seq[3], 0.0, "B_3");
            assert_eq!(seq[4], -1.0 / 30.0, "B_4");
        }
    }

    /// Odd subscripts past 1 are zero, even ones are the table entry. This is the check
    /// that the subscript-to-table-index mapping (`n / 2 - 1`) is right at every step,
    /// not just at the two spots the doctest looks at.
    #[test]
    fn subscripts_line_up_with_the_table() {
        for (n, v) in V::bernoulli_numbers(-0.5).enumerate().skip(2) {
            let got = lane0(v);
            if n % 2 == 1 {
                assert_eq!(got, 0.0, "B_{n} should be zero");
            } else {
                assert_eq!(got, f64::B2N[entry(n)], "B_{n}");
            }
        }
    }

    /// Cross-check the even terms against the exact rationals, through the iterator this
    /// time rather than by indexing the table directly.
    #[test]
    fn even_terms_match_the_exact_rationals() {
        let seq: Vec<f64> = V::bernoulli_numbers(-0.5).map(lane0).collect();
        for &(two_n, num, den) in EXACT {
            assert_eq!(seq[two_n], num as f64 / den as f64, "B_{two_n}");
        }
    }

    /// Ends after the last representable `B_2n`, and does not trail a spurious zero for
    /// the odd subscript beyond it.
    #[test]
    fn stops_after_the_last_representable_even_term() {
        let seq: Vec<f64> = V::bernoulli_numbers(-0.5).map(lane0).collect();
        assert_eq!(seq.len(), 2 * f64::B2N.len() + 1, "B_0 ..= B_258");
        assert_eq!(seq.len(), 259);
        assert_eq!(*seq.last().expect("non-empty"), f64::B2N[128], "last item is B_258");

        let seq32: Vec<f32> = Vector::<f32>::bernoulli_numbers(-0.5)
            .map(|v| v.extract::<0>())
            .collect();
        assert_eq!(seq32.len(), 65, "B_0 ..= B_64");
    }

    /// `size_hint` has to stay honest as the iterator advances, since `ExactSizeIterator`
    /// lets callers preallocate off it.
    #[test]
    fn exact_size_is_exact_at_every_step() {
        let mut it = V::bernoulli_numbers(-0.5);
        let mut expected = 259;
        loop {
            assert_eq!(it.len(), expected, "len with {expected} left");
            assert_eq!(it.size_hint(), (expected, Some(expected)));
            if it.next().is_none() {
                break;
            }
            expected -= 1;
        }
        assert_eq!(expected, 0);
    }

    /// Fused: an exhausted iterator keeps saying `None` rather than wrapping or running
    /// the subscript away past the end.
    #[test]
    fn exhausted_iterator_stays_exhausted() {
        let mut it = V::bernoulli_numbers(-0.5);
        while it.next().is_some() {}
        for _ in 0..4 {
            assert_eq!(it.next().map(lane0), None);
            // `len()` reads the same counter `next` guards, so a runaway subscript would
            // show up here as a wrapped or nonzero remaining count.
            assert_eq!(it.len(), 0, "exhausted iterator must stay at zero remaining");
        }
    }

    /// Every lane carries the value, not just lane 0 - these are splats.
    #[test]
    fn all_lanes_are_equal() {
        for (n, v) in V::bernoulli_numbers(-0.5).enumerate().take(8) {
            for lane in 0..V::LANES {
                assert_eq!(v.extractv(lane), lane0(v), "B_{n}: lane {lane}");
            }
        }
    }

    /// `Copy` means a partially consumed iterator can be forked without disturbing the
    /// original, which is the reason the state is a bare counter plus a value.
    #[test]
    fn is_copy_and_forks_cleanly() {
        let mut it: BernoulliSequence<V> = V::bernoulli_numbers(0.5);
        let _ = it.next();
        let forked = it;

        let a: Vec<f64> = it.take(3).map(lane0).collect();
        let b: Vec<f64> = forked.take(3).map(lane0).collect();
        assert_eq!(a, b);
        assert_eq!(a[0], 0.5, "fork resumes at B_1");
    }
}
