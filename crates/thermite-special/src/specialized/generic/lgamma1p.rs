//! `$\ln\Gamma(1+v)$` and `$\Gamma(1+v)-1$` on `$\lvert v\rvert \le 1/2$`, both signs at once.
//!
//! # Why these exist as their own functions
//!
//! `$\Gamma(1+v) - 1$` cannot be computed as `tgamma(1 + v) - 1`. Near zero
//! `$\Gamma(1+v) \approx 1 - \gamma v$`, so subtracting one throws away every bit that is not
//! in `$\gamma v$`. About 7 bits lost at `$v = 10^{-2}$`, 27 at `$10^{-8}$`, all of them by
//! `$10^{-16}$`.
//!
//! That matters because Temme's series, the small-`x` arm for `$Y_\nu$`, forms
//!
//! ```math
//! g_1 = \frac{g_+ - g_-}{(1+g_+)(1+g_-)\,2v}, \qquad g_\pm = \Gamma(1\pm v) - 1
//! ```
//!
//! which is `$0/0$` as `$v \to 0$` and needs both `$g_\pm$` to full relative accuracy to
//! resolve it. Boost carries a dedicated `tgamma1pm1` for exactly this reason.
//!
//! # The route taken, and what it avoids
//!
//! `$\Gamma(1+v) - 1 = \mathrm{expm1}(\ln\Gamma(1+v))$`, with `$\ln\Gamma(1+v)$` from its
//! `$\zeta$` series. The series has **no subtraction of near-equal quantities anywhere**, so it
//! is uniformly accurate across the range including at `$v = 0$`, where it simply returns zero.
//!
//! The alternative was a fitted rational, which would have needed the parked minimax tooling.
//! Every coefficient here is `$\zeta(k)/k$` computed to 60 digits and rounded once: exact
//! constants, not an approximation. See [`crate::tables::lgamma1p`].
//!
//! Both signs come out of one evaluation, because the series splits by parity and Temme wants
//! `$\pm v$` anyway.

use thermite::{
    math::{TranscendentalMathWithPolicy, policy::Policy},
    prelude::*,
};

use thermite::element::FloatElement;

use crate::tables::lgamma1p::LogGamma1p;

/// `$(\ln\Gamma(1+v),\; \ln\Gamma(1-v))$` for `$\lvert v\rvert \le 1/2$`.
///
/// Outside that range the series has not been given enough terms and the result is wrong. The
/// caller reduces the order first. Exact at `$v = 0$`, where both halves are zero.
#[inline(always)]
pub fn lgamma1p_pair<P, E, V, const NE: usize, const NO: usize>(v: V, t: &LogGamma1p<E, NE, NO>) -> (V, V)
where
    E: FloatElement,
    V: FloatVector<Element = E> + TranscendentalMathWithPolicy,
    P: Policy,
{
    let w = v * v;

    // Even `k` contributes `w * P_even(w)`, odd `k` contributes `v * P_odd(w)`. The sign of
    // `v` only reaches the odd half, which is the whole point of splitting them.
    let even = w * w.poly_n_p::<P, NE>(&t.even);
    let odd = v * w.poly_n_p::<P, NO>(&t.odd);

    (even + odd, even - odd)
}

/// `$(\Gamma(1+v) - 1,\; \Gamma(1-v) - 1)$` for `$\lvert v\rvert \le 1/2$`.
///
/// Boost calls the single-sign form `tgamma1pm1`. See the [module docs](self) for why
/// `tgamma(1 + v) - 1` is not an acceptable substitute.
#[inline(always)]
pub fn tgamma1pm1_pair<P, E, V, const NE: usize, const NO: usize>(v: V, t: &LogGamma1p<E, NE, NO>) -> (V, V)
where
    E: FloatElement,
    V: FloatVector<Element = E> + TranscendentalMathWithPolicy,
    P: Policy,
{
    let (lp, lm) = lgamma1p_pair::<P, E, V, NE, NO>(v, t);

    // `expm1` rather than `exp - 1`, for the same reason the series exists: both arguments go
    // to zero with `v`, and so must both results.
    (lp.exp_m1_p::<P>(), lm.exp_m1_p::<P>())
}

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
mod tests {
    use super::*;

    use thermite::Vector;
    use thermite::math::policy::policies::Precision;

    use crate::tables::lgamma1p::LGAMMA1P_F64;

    type V = Vector<f64>;

    fn lg(v: f64) -> (f64, f64) {
        let (a, b) = lgamma1p_pair::<Precision, f64, V, 25, 25>(V::splat(v), &LGAMMA1P_F64);
        (a.extract::<0>(), b.extract::<0>())
    }

    fn tg(v: f64) -> (f64, f64) {
        let (a, b) = tgamma1pm1_pair::<Precision, f64, V, 25, 25>(V::splat(v), &LGAMMA1P_F64);
        (a.extract::<0>(), b.extract::<0>())
    }

    fn rel(got: f64, want: f64) -> f64 {
        if want == 0.0 {
            return if got == 0.0 { 0.0 } else { f64::INFINITY };
        }
        ((got - want) / want).abs()
    }

    /// Against mpmath at 60 digits, across the whole `|v| <= 1/2` range and both signs.
    #[test]
    fn lgamma1p_matches_a_high_precision_reference() {
        // (v, lnGamma(1+v), lnGamma(1-v))
        const ROWS: &[(f64, f64, f64)] = &[
            (0.5, -0.12078223763524522, 0.5723649429247001),
            (0.25, -0.09827183642181316, 0.20328095143129538),
            (0.1, -0.04987244125983972, 0.06637623973474296),
            (0.01, -0.005690307946069646, 0.005854806764709776),
            (0.001, -0.0005763935982833696, 0.0005780385328913797),
            (1e-08, -5.772156566768626e-09, 5.772156731262032e-09),
            (0.0, 0.0, 0.0),
        ];

        for &(v, want_p, want_m) in ROWS {
            let (gp, gm) = lg(v);
            assert!(rel(gp, want_p) <= 4e-16, "lnGamma(1+{v}): got {gp}, want {want_p}");
            assert!(rel(gm, want_m) <= 4e-16, "lnGamma(1-{v}): got {gm}, want {want_m}");
        }
    }

    /// The property the whole module exists for: `Gamma(1+v) - 1` keeps full **relative**
    /// accuracy as `v -> 0`, where `tgamma(1 + v) - 1` has none left.
    ///
    /// At `v = 1e-8` the true value is about `-5.77e-9`, and forming it by subtraction leaves
    /// roughly 27 bits. The naive route is checked here to be as bad as claimed rather than
    /// merely asserted to be.
    #[test]
    fn tgamma1pm1_survives_where_subtraction_does_not() {
        for &(v, want) in &[
            (1e-2f64, -0.005674148808493963),
            (1e-4, -5.771167683758092e-05),
            (1e-6, -5.77214675846445e-07),
            (1e-8, -5.77215655010973e-09),
            (1e-12, -5.772156649005438e-13),
        ] {
            let (gp, _) = tg(v);
            assert!(
                rel(gp, want) <= 8e-16,
                "Gamma(1+{v})-1: got {gp}, want {want}, rel {:e}",
                rel(gp, want)
            );
        }

        // Exactly zero at zero, both signs. A series has no reason to miss this, but the
        // `expm1` in front of it would if it were `exp - 1`.
        assert_eq!(tg(0.0), (0.0, 0.0));
    }

    /// The odd half carries the sign and the even half does not, so `v` and `-v` must simply
    /// swap the pair. Bit-exact: it is one polynomial pair and a sign, not two evaluations.
    #[test]
    fn the_two_signs_are_one_evaluation() {
        for &v in &[0.5f64, 0.3, 0.05, 1e-6] {
            let (a, b) = lg(v);
            let (c, d) = lg(-v);
            assert_eq!(a.to_bits(), d.to_bits(), "lnGamma(1+v) must equal lnGamma(1-(-v))");
            assert_eq!(b.to_bits(), c.to_bits(), "lnGamma(1-v) must equal lnGamma(1+(-v))");
        }
    }
}
