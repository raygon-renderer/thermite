//! The slice-taking series against their const-length originals.
//!
//! A series carries `k`-dependent state and does not partition, so unlike the slice
//! reductions in `thermite` these cannot be folds over the const kernel. Either they share
//! one body or they duplicate the recurrence, and both shapes are represented here:
//!
//! - **Chebyshev shares.** `chebyshev_n` and `chebyshev` are one function taking a slice,
//!   with `N = 0` meaning "length unknown" and a nonzero `N` supplied to LLVM as an
//!   `assert_unchecked`. Bit-for-bit agreement is then nearly tautological, which is the
//!   point, but not entirely: it still pins that the `assume` and the folded-away branches
//!   do not change the arithmetic the runtime path performs.
//! - **Legendre, Hermite and Laguerre duplicate**, deliberately. Their per-step weights are
//!   `from_ratio` calls that become literals only if the loop unrolls, so merging them
//!   would put a division or a square root per step behind an `assume`, with no test able
//!   to see the loss. Until that is measured they stay split, and for them the bit-for-bit
//!   assertion is load-bearing: duplicated code drifts, and anything less than equality
//!   means one of the two was edited alone.

use thermite::math::policy::policies::{Performance, Precision};
use thermite::prelude::*;
use thermite_special::SpecialMathWithPolicy;

type D = Vector<f64>;
type F = Vector<f32>;

const C: [f64; 8] = [1.5, -0.75, 0.25, 2.0, -1.25, 0.5, 0.125, -0.0625];

/// Chebyshev, all four kinds, at both policies. `Precision` is the one that takes the
/// Reinsch form, so it is a different arm and needs its own check.
///
/// Both spellings now reach one body, so what this actually pins is that the const path's
/// `assert_unchecked` and its folded-away `n == 1`/`n == 2` branches leave the arithmetic
/// alone. `n = 1` and `n = 2` are in the list because those are the shortcut lengths, where
/// a divergence would show up first.
#[test]
fn chebyshev_slice_is_bit_identical() {
    for &x in &[-1.0, -0.999, -0.5, 0.0, 0.25, 0.9, 0.999_999, 1.0] {
        let v = D::splat(x);

        macro_rules! check {
            ($k:literal, $n:literal, $policy:ty, $name:literal) => {{
                let c: [f64; $n] = core::array::from_fn(|i| C[i]);
                let want = v.chebyshev_n_p::<$policy, $k, $n>(&c).extract::<0>();
                let got = v.chebyshev_p::<$policy, $k>(&c[..]).extract::<0>();
                assert_eq!(got, want, "chebyshev kind {} n={} {} at x={x}", $k, $n, $name);
            }};
        }

        check!(1, 1, Precision, "precision");
        check!(1, 2, Precision, "precision");
        check!(1, 5, Precision, "precision");
        check!(1, 8, Precision, "precision");
        check!(2, 8, Precision, "precision");
        check!(3, 8, Precision, "precision");
        check!(4, 8, Precision, "precision");

        check!(1, 8, Performance, "performance");
        check!(2, 5, Performance, "performance");
        check!(3, 2, Performance, "performance");
        check!(4, 1, Performance, "performance");
    }
}

#[test]
fn legendre_series_slice_is_bit_identical() {
    for &x in &[-1.0, -0.5, 0.0, 0.375, 1.0] {
        let v = D::splat(x);

        macro_rules! check {
            ($n:literal) => {{
                let c: [f64; $n] = core::array::from_fn(|i| C[i]);
                let want = v.legendre_series_n_p::<Precision, $n>(&c).extract::<0>();
                let got = v.legendre_series_p::<Precision>(&c[..]).extract::<0>();
                assert_eq!(got, want, "legendre_series n={} at x={x}", $n);
            }};
        }

        check!(1);
        check!(2);
        check!(3);
        check!(6);
        check!(8);
    }
}

#[test]
fn hermite_function_series_slice_is_bit_identical() {
    for &x in &[-4.0, -1.0, 0.0, 0.5, 3.25] {
        let v = D::splat(x);

        macro_rules! check {
            ($n:literal) => {{
                let c: [f64; $n] = core::array::from_fn(|i| C[i]);
                let want = v.hermite_function_series_n_p::<Precision, $n>(&c).extract::<0>();
                let got = v.hermite_function_series_p::<Precision>(&c[..]).extract::<0>();
                assert_eq!(got, want, "hermite_function_series n={} at x={x}", $n);
            }};
        }

        check!(1);
        check!(2);
        check!(4);
        check!(8);
    }
}

#[test]
fn laguerre_function_series_slice_is_bit_identical() {
    for &x in &[0.0, 0.5, 2.0, 9.0] {
        let v = D::splat(x);

        macro_rules! check {
            ($n:literal, $alpha:expr) => {{
                let c: [f64; $n] = core::array::from_fn(|i| C[i]);
                let a = D::splat($alpha);

                let want = v.laguerre_function_series_n_p::<Precision, $n>(a, &c).extract::<0>();
                let got = v.laguerre_function_series_p::<Precision>(a, &c[..]).extract::<0>();
                assert_eq!(got, want, "laguerre_function_series n={} alpha={} at x={x}", $n, $alpha);
            }};
        }

        check!(1, 0.0);
        check!(2, 0.0);
        check!(5, 1.5);
        check!(8, 2.25);
    }
}

/// The integer-weight form is a separate monomorph (`INT_ALPHA = true`), so it needs its
/// own check rather than riding on the general one.
#[test]
fn laguerre_function_series_i_slice_is_bit_identical() {
    for &x in &[0.0, 0.75, 4.5] {
        let v = D::splat(x);

        macro_rules! check {
            ($n:literal, $alpha:literal) => {{
                let c: [f64; $n] = core::array::from_fn(|i| C[i]);

                let want = v.laguerre_function_series_i_n_p::<Precision, $n>($alpha, &c).extract::<0>();
                let got = v.laguerre_function_series_i_p::<Precision>($alpha, &c[..]).extract::<0>();
                assert_eq!(got, want, "laguerre_function_series_i n={} alpha={} at x={x}", $n, $alpha);
            }};
        }

        check!(1, 0);
        check!(4, 0);
        check!(8, 3);
    }
}

/// The const forms reject an empty coefficient array at compile time and the slice forms
/// cannot, so each has to answer for the case. Zero is the empty sum.
#[test]
fn empty_coefficients_are_zero() {
    let v = D::splat(0.5);
    let none: [f64; 0] = [];

    assert_eq!(v.chebyshev_p::<Precision, 1>(&none).extract::<0>(), 0.0);
    assert_eq!(v.legendre_series_p::<Precision>(&none).extract::<0>(), 0.0);
    assert_eq!(v.hermite_function_series_p::<Precision>(&none).extract::<0>(), 0.0);
    assert_eq!(
        v.laguerre_function_series_p::<Precision>(D::ZERO, &none).extract::<0>(),
        0.0
    );
    assert_eq!(
        v.laguerre_function_series_i_p::<Precision>(0, &none).extract::<0>(),
        0.0
    );
}

/// float32 takes a different specialized body, and for `chebyshev` a different Reinsch
/// decision, so it is checked separately rather than assumed.
#[test]
fn float32_agrees() {
    let c: [f32; 6] = [1.5, -0.75, 0.25, 2.0, -1.25, 0.5];

    for &x in &[-1.0f32, -0.25, 0.0, 0.875, 1.0] {
        let v = F::splat(x);

        assert_eq!(
            v.chebyshev_p::<Precision, 1>(&c[..]).extract::<0>(),
            v.chebyshev_n_p::<Precision, 1, 6>(&c).extract::<0>(),
            "f32 chebyshev at x={x}"
        );

        assert_eq!(
            v.legendre_series_p::<Precision>(&c[..]).extract::<0>(),
            v.legendre_series_n_p::<Precision, 6>(&c).extract::<0>(),
            "f32 legendre_series at x={x}"
        );
    }
}
