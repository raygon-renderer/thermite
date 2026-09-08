#![doc = include_str!("../README.md")]
#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(clippy::needless_arbitrary_self_type, clippy::needless_range_loop)]
#![recursion_limit = "256"]

use thermite::{
    element::{Element, ElementExt, FloatElementWithBits},
    math::{
        PrimalMathWithPolicy, PrimalProjection, TranscendentalMathWithPolicy,
        policy::{DefaultPolicy, Policy},
        scalar::Unwrap,
    },
    vector::{FloatVector, FloatVectorWithBits},
};

pub mod specialized;

// Raw approximation coefficients, shared with the sibling crates. Documented on the
// module itself rather than here: an outer doc at this declaration site is merged with
// the module's own and then resolved in THIS scope, which breaks every link it makes
// to its own submodules.
#[doc(hidden)]
pub mod tables;

pub use tables::bernoulli::BernoulliNumbers;
pub use tables::cot_pi::CotPiDerivatives;
pub use tables::factorial::Factorials;

pub mod bernoulli;
pub mod bessel;
pub mod polylog;
pub mod zernike;

// `BesselOrder` appears in the signature of every runtime-order Bessel entry point below,
// so a caller has to be able to name it without reaching into the module.
pub use crate::bessel::BesselOrder;

// The marker-selected Bessel entry points (`bessel_n`, `bessel`, `sph_bessel_n`,
// `sph_bessel`, `airy`) are bounded on these. The markers themselves stay in
// `bessel::{J, Y, I, K, Scaled, Ai, ..}`.
use crate::bessel::{AiryFn, BesselFamily, BesselRatioFamily};

// Likewise `PolylogOrder`, the order argument of `polylog`.
pub use crate::polylog::PolylogOrder;

// The two normalization flags appear in the `zernike` signature below as a const
// generic, so a caller has to be able to name them without reaching into the module.
pub use crate::zernike::{ZERNIKE_ORTHONORMAL, ZERNIKE_UNIT_PEAK};

use crate::specialized::{CarlsonKind, EllipticKind, WrapTo};

// Spherical-harmonic support: `ShTable` appears in the public signatures below, and
// `MAX_SH_DEGREE` is the documented degree at which they leave the unrolled path.
pub use crate::specialized::{MAX_SH_DEGREE, ShTable};

/// Elliptic integral request structs and the traits they implement:
///
/// - Carlson symmetric integrals (for [`SpecialMath::carlson`]): [`CarlsonRf`](elliptic::CarlsonRf),
///   [`CarlsonRc`](elliptic::CarlsonRc), [`CarlsonRd`](elliptic::CarlsonRd), [`CarlsonRj`](elliptic::CarlsonRj),
///   [`CarlsonRg`](elliptic::CarlsonRg), implementing [`CarlsonKind`].
/// - Legendre integrals (for [`SpecialMath::ellint`]): [`EllintK`](elliptic::EllintK)/[`EllintF`](elliptic::EllintF),
///   [`EllintE`](elliptic::EllintE)/[`EllintEInc`](elliptic::EllintEInc),
///   [`EllintD`](elliptic::EllintD)/[`EllintDInc`](elliptic::EllintDInc),
///   [`EllintPi`](elliptic::EllintPi)/[`EllintPiInc`](elliptic::EllintPiInc), implementing
///   [`EllipticKind`]. Completeness is encoded by the struct: a complete integral has no `phi` field.
///
/// The request structs are implemented for every float vector whose element carries
/// [`EllipticConsts`](elliptic::EllipticConsts): real `f32`/`f64` vectors, `Dual` (the
/// derivative is the chain rule through the Carlson duplication and the AGM, contractive
/// algebraic iterations) and `Compensated` (which supplies its own, tighter, convergence
/// thresholds and holds full double-double). `Complex` does not implement the constants, so
/// an elliptic integral of a complex vector is a compile error rather than a wrong answer:
/// the kernels' region decisions are real-line comparisons.
pub mod elliptic {
    pub use crate::specialized::EllipticConsts;

    pub use crate::specialized::{CarlsonKind, CarlsonRc, CarlsonRd, CarlsonRf, CarlsonRg, CarlsonRj};

    pub use crate::specialized::{
        EllintD, EllintDInc, EllintE, EllintEInc, EllintF, EllintK, EllintPi, EllintPiInc, EllipticKind,
    };

    /// The two members of the family that are not Legendre integrals, dispatched through the
    /// same [`EllipticKind`] entry point as the rest.
    pub use crate::specialized::{HeumanLambda, JacobiZeta};
}

thermite::math_traits! {
    #![thermite(thermite)]
    #![scalar(ScalarSpecialMath)]
    #![surface]

    /// Special math functions that are valid for both real and complex floating-point vectors.
    #[diagnostic::on_unimplemented(
        message = "`{Self}` does not provide special math (`erf`, `gamma`, activations, ...)",
        note = "The special-math traits are auto-implemented for every float vector (any `FloatVector` whose element is `f32`/`f64`) and for composite float types. A bare `f32`/`f64` does not qualify. Wrap it in `Vector::<f32>::splat(x)`, or use `ScalarSpecialMath`'s `scalar_`-prefixed methods.",
        note = "If `{Self}` already is a `FloatVector` and only the method call fails to resolve, bring the trait into scope: `use thermite_special::SpecialMath;` (or the relevant `RealSpecialMath` / `RealPrimalMath`)."
    )]
    pub trait SpecialMath: TranscendentalMathWithPolicy {
        /// Computes the error function.
        ///
        /// For f32 vectors, this is still decently accurate even with the `Medium` and `Worst` precision policies,
        /// thanks to good approximations that don't rely on the precision of `exp`. Subsequently, performance
        /// of the lower precision policies is excellent. Furthermore, if using on a GPU with native `exp` support,
        /// all precision policies will have good performance and accuracy.
        ///
        /// Below `Best`, the f64 kernel forms `erf` as `$1 - m\,e^{-x^2}$`, whose error is a fixed
        /// absolute ulp of 1: `erf(0)` comes out `2.2e-16` and `erf(1e-8)` is only 2e-8 relative.
        /// From `Best` up, `|x| < 0.84375` takes a direct `$x + x\,R(x^2)/S(x^2)$` arm that is
        /// exact at zero and relatively accurate down to the denormals. The f32 kernel carries
        /// that arm from `Average`.
        fn erf(self) -> Self;

        /// Computes the complementary error function.
        ///
        /// The f64 kernel is one product of six rationals times `$e^{-x^2}$` over the whole
        /// line, within about 3 ulp everywhere on hardware with a fused multiply-add: the one
        /// error that grows, the rounding of `$x^2$` under the exponential amplified by `$x^2$`,
        /// is removed with the exact residual of the product at every tier. Without a native
        /// FMA that residual is unavailable, so `Best` removes the growth with a bit-split of `x`
        /// instead, and the lower tiers keep it (47 ulp at `x = 14`, 237 at `x = 24`).
        fn erfc(self) -> Self;

        /// Computes the scaled complementary error function,
        /// `$\operatorname{erfcx}(x) = e^{x^2}\operatorname{erfc}(x)$`.
        ///
        /// `erfc` underflows to zero at `x ~ 27` in `f64` and `x ~ 9` in `f32`,
        /// where the true value is `$e^{-x^2}/(x\sqrt{\pi})$`, nonzero and merely too small to
        /// represent. Anything reading a Gaussian tail past that point silently gets zero:
        /// importance weights, log-likelihoods, censored-data models, the Voigt profile.
        /// `erfcx` removes the exponential and decays only as `$1/(x\sqrt{\pi})$`, so it is
        /// representable for every finite argument and keeps full relative accuracy.
        ///
        /// Computed on the real backends as the Faddeeva function restricted to the imaginary
        /// axis, `$w(ix) = \operatorname{erfcx}(x)$`, where Weideman's rational approximation
        /// degenerates to real arithmetic: one reciprocal and one Horner, no transcendental at
        /// all for `x >= 0`. That makes it cheaper than the `erfc` it complements, and
        /// measures 1.22 ulp worst over `$x \in [0, 10^{15}]$` at the `Best` tier and above.
        ///
        /// Negative arguments use `$\operatorname{erfcx}(-x) = 2e^{x^2} - \operatorname{erfcx}(x)$`
        /// and legitimately overflow below about `-26.6` (`f64`), the function itself growing
        /// like `$e^{x^2}$` in that direction.
        ///
        /// The two are related by `$\operatorname{erfc}(x) = e^{-x^2}\operatorname{erfcx}(x)$`,
        /// which is the numerically sound way to recover a tail value that `erfc` alone cannot
        /// hold. Keep the `$-x^2$` in the log domain rather than exponentiating it.
        fn erfcx(self) -> Self;

        /// Computes the Logistic sigmoid function, defined as `$\sigma(x) = \frac{1}{1 + e^{-x}}$`.
        ///
        /// It's worth mentioning that the derivative of the logistic sigmoid can be computed very cheaply
        /// from the output of the logistic sigmoid itself, in the form of:
        ///
        /// ```rust,ignore
        /// let s = x.logistic_sigmoid();
        /// let derivative = s * (1.0 - s); // or s.nmul_adde(s, s), which may be slightly faster
        /// ```
        ///
        /// Notably, for `f32` and `f64` this implementation still has good precision for the `Worst`
        /// precision policy, and for the `Best` precision policies handles very large positive and negative
        /// inputs without overflow or underflow issues.
        #[doc(alias = "expit")]
        fn logistic_sigmoid(self) -> Self;

        /// Computes the logit `$\ln\!\frac{p}{1-p}$`, the inverse of
        /// [`logistic_sigmoid`](SpecialMath::logistic_sigmoid).
        ///
        /// Evaluated as `$\ln(p) - \ln_{1p}(-p)$`, which is accurate for small `p` where the direct
        /// quotient is not. For `p` approaching 1 no evaluation order helps. `$1 - p$` has already
        /// lost its low digits inside the input itself, and the information is not recoverable from
        /// `p`. A caller who knows `$q = 1 - p$` should pass it to
        /// [`logit_1m`](SpecialMath::logit_1m) instead, which is exact at the far end of the range.
        ///
        /// `p = 0` gives `-∞`, `p = 1` gives `+∞`, and `p` outside `[0, 1]` is out of domain.
        fn logit(self) -> Self;

        /// Computes `$\mathrm{logit}(1 - q) = \ln\!\frac{1-q}{q}$` from the complement `q` directly.
        ///
        /// The companion entry point to [`logit`](SpecialMath::logit), in the same relationship as
        /// [`langevin_1m`](RealSpecialMath::langevin_1m) has to
        /// [`langevin`](RealSpecialMath::langevin). The logit diverges as its argument approaches 1,
        /// and near that end `$1 - p$` cannot be formed from `p` without losing every digit that
        /// matters. Working in `q` throughout sidesteps that: evaluated as
        /// `$\ln_{1p}(-q) - \ln(q)$`, accurate to a few ulp however small `q` is.
        ///
        /// Note the sign convention follows the substitution, so `logit_1m(q) == -logit(q)` as
        /// functions of the same number. The two differ in _which_ probability the argument names.
        fn logit_1m(self) -> Self;

        /// Computes the softplus function, defined as `$\frac{1}{k}\ln(1 + e^{kx})$`.
        ///
        /// This is a smooth approximation to the ReLU function
        /// that is more numerically stable for large inputs.
        ///
        /// The parameter `k` controls the steepness of the curve, with larger values approaching ReLU more closely.
        /// Pass `k = 1` and `rcp_k = 1` for the standard softplus with no steepness scaling.
        ///
        /// `rcp_k` must equal `1/k`. It is passed explicitly so callers that invoke softplus repeatedly
        /// with the same `k` can pre-compute the reciprocal once rather than recomputing it per call.
        ///
        /// To also obtain the derivative with respect to `x`, use
        /// [`softplus_d`](crate::RealPrimalMath::softplus_d).
        fn softplus(self, k: Self, rcp_k: Self) -> Self;

        /// Computes the Gamma function (`$\Gamma(z)$`) for any real input, for each value in a vector.
        ///
        /// This implementation uses a few different behaviors to ensure the greatest precision where possible.
        ///
        /// * For non-integer positive inputs, it uses the Lanczos approximation.
        /// * For small non-integer negative inputs, it uses the recursive identity `$\Gamma(z) = \Gamma(z+1)/z$` until `z` is positive.
        /// * For large non-integer negative inputs, it uses the reflection formula `$-\pi / (\Gamma(z)\sin(\pi z)\,z)$`.
        /// * For positive integers, it simply computes the factorial in a tight loop to ensure precision. Lookup tables could not be used with SIMD.
        /// * At zero, the result will be positive or negative infinity based on the input sign (signed zero is a thing).
        ///
        /// **NOTE**: The Gamma function is not defined for negative integers.
        #[doc(alias = "gamma")]
        fn tgamma(self) -> Self;

        /// Computes the natural log of the Gamma function (`$\ln|\Gamma(x)|$`) for any real input, for each value in a vector.
        #[doc(alias = "gammaln")]
        #[doc(alias = "lngamma")]
        fn lgamma(self) -> Self;

        /// The Poisson probability mass `$P(k; \lambda) = e^{-\lambda}\lambda^k / k!$` at `k = self`,
        /// for real `$k \ge 0$` and mean `$\lambda \ge 0$`.
        ///
        /// Not `exp(k ln lambda - lambda - lgamma(k+1))`: that forms an `$O(1)$` answer as the
        /// exponential of a difference of large numbers, and half an ulp of
        /// `$\ln\Gamma(k+1) = O(k \ln k)$` becomes that many ulp of the mass. For `$k \ge 9$` this
        /// uses Loader's saddle-point form (the one R's `dpois` uses),
        ///
        /// ```math
        /// P(k; \lambda) = \frac{e^{-\mathrm{stirlerr}(k) - \mathrm{bd0}(k, \lambda)}}{\sqrt{2\pi k}}
        /// ```
        ///
        /// with `stirlerr` the Stirling remainder (a short `$1/k^2$` series) and `bd0` the
        /// deviance `$k \ln(k/\lambda) + \lambda - k$` (a series in `$(k-\lambda)/(k+\lambda)$` near
        /// the peak, where the direct form cancels): both are small where the mass is not
        /// negligible, so the exponential amplifies nothing, and there is no `lgamma` and no
        /// `ln` at all near the peak. Below `$k = 9$` the same machinery is used after shifting
        /// `k` up by an integer, with the exact product `$(k+1)\cdots(k+m)$` taken back out, so
        /// there is no `lgamma` anywhere, and mixed vectors share one `ln`, one `stirlerr` and
        /// one `exp`. Real `k` is allowed because
        /// the Gamma density is the same function: `$f(x; a) = P(a-1; x)$` for shape `$a \ge 1$`
        /// (unit scale).
        ///
        /// Edges: `$\lambda = 0$` gives `1` at `$k = 0$` and `0` above; `$k = 0$` is `$e^{-\lambda}$`.
        fn poisson_pmf(self, lambda: Self) -> Self;

        /// `$\ln P(k; \lambda)$`, the log of [`poisson_pmf`](SpecialMath::poisson_pmf), formed
        /// directly (no `exp` then `ln`) so it stays finite far in the tails where the mass
        /// itself underflows.
        fn poisson_log_pmf(self, lambda: Self) -> Self;

        /// Computes the digamma function `$\psi(x) = \frac{\mathrm{d}}{\mathrm{d}x}\ln\Gamma(x) = \frac{\Gamma'(x)}{\Gamma(x)}$`
        /// for any real input, for each value in a vector.
        ///
        /// The argument is handled in three regimes:
        ///
        /// * For `x >= 10`, an asymptotic expansion in `$1/x^2$` is used.
        /// * For smaller `x`, the recurrence `$\psi(x) = \psi(x+1) - 1/x$` shifts the argument into
        ///   `[1, 2]`, where a rational minimax approximation `$\psi(x) = (x - x_0)(Y + R(x-1))$` is used
        ///   (`$x_0$` is the positive root of `$\psi$`).
        /// * For `x <= -1`, the reflection formula `$\psi(1-x) = \psi(x) + \pi\cot(\pi x)$` is applied.
        ///
        /// **NOTE**: The digamma function is not defined at zero or the negative integers. Those inputs
        /// yield NaN when overflow checking is enabled.
        #[doc(alias = "psi")]
        fn digamma(self) -> Self;

        /// Computes the trigamma function `$\psi_1(x) = \frac{\mathrm{d}}{\mathrm{d}x}\psi(x)$`,
        /// the second derivative of `$\ln\Gamma$`.
        ///
        /// Real vectors run a dedicated kernel (three minimax rational regions with a single
        /// recurrence step and the `$\pi^2/\sin^2(\pi x)$` reflection) that is a little tighter
        /// than the general [`polygamma`](crate::SpecialMath::polygamma) machinery at
        /// order 1. `polygamma(1)` routes here, so the two spellings agree exactly. Complex
        /// vectors have their own implementation, which is the reason this lives on
        /// `SpecialMath` while `polygamma` is real-only.
        ///
        /// The poles at zero and the negative integers evaluate to `+inf`: `$\psi_1$` has
        /// double poles, so unlike [`digamma`](SpecialMath::digamma) the two one-sided limits
        /// agree.
        fn trigamma(self) -> Self;

        /// Computes the polygamma function `$\psi_n(x) = \frac{\mathrm{d}^n}{\mathrm{d}x^n}\psi(x)$`,
        /// the n-th derivative of [`digamma`](SpecialMath::digamma) (`n = 0` **is** digamma,
        /// `n = 1` is [`trigamma`](SpecialMath::trigamma)).
        ///
        /// The order `n` is a runtime scalar shared by every lane. That is a deliberate design
        /// choice: it closes the Gamma family under differentiation, since
        /// `$\psi_n'(x) = \psi_{n+1}(x)$` is reachable by passing `n + 1`, which is what lets
        /// forward-mode AD (`Dual`) differentiate through any member of the family to any depth.
        /// All order-dependent coefficients are scalar work splatted once, so uniform `n`
        /// costs a vector nothing.
        ///
        /// For `n >= 2`, real vectors run a masked recurrence up to the transition point
        /// `$N = 0.4\,d_{10} + 4n$` and then the Bernoulli asymptotic series on the positive
        /// axis. Negative arguments reflect through the n-th derivative of `$\cot(\pi x)$`
        /// (tabulated to `n = 20`, above which negative arguments return NaN). At zero
        /// and the negative integers, odd `n` returns `+inf` (the correct two-sided limit)
        /// and even `n` has one-sided limits of opposite sign, so it returns NaN when
        /// overflow checking is enabled.
        ///
        /// Complex vectors run the same recurrence-plus-series in complex arithmetic, gated
        /// on `$\operatorname{Re} z$`, reflecting the half-plane `$\operatorname{Re} z < 1/2$`
        /// through the same tabulated `$\cot$` derivative (so the `n <= 20` reflection reach
        /// applies there too). Only `psi_n` of a _real_ variable is real, so this is the
        /// family member that makes `polygamma` complex-capable at all orders.
        ///
        /// Orders where `$n!$` overflows the element type (`n >= 171` for f64, `n >= 35` for
        /// f32) return the signed infinity carried by the leading term on the real positive
        /// axis, and NaN over C.
        fn polygamma(self, n: u32) -> Self;

        /// Computes the Riemann zeta function `$\zeta(s) = \sum_{n\ge1} n^{-s}$`.
        ///
        /// Evaluated as `1 + `[`zetac`](SpecialMath::zetac), which is where the accuracy
        /// argument lives (see there). Worst relative error measured against mpmath at 40
        /// digits: 4.4e-16 for `s` in `[1.5, 5]`, 4.3e-16 for `[5, 40]`, 2.3e-15 through the
        /// critical strip `[0.1, 0.9]`, and 4.6e-16 approaching the pole at `s = 1`, which
        /// returns infinity.
        ///
        /// Negative `s` goes through the functional equation
        /// `$\zeta(s) = 2^s\pi^{s-1}\sin(\pi s/2)\,\Gamma(1-s)\,\zeta(1-s)$`, landing back at
        /// `$1-s > 1$` where the series is at its most accurate. That arm costs a `tgamma` and
        /// a `sin_pi` beyond the main path, so it is gated on a lane needing it.
        ///
        /// This is the Riemann zeta of one real argument. The two-argument Hurwitz form
        /// `$\zeta(s, q)$` is **not** provided: it generalizes the same expansion but loses the
        /// prime factorization that makes this one cheap, so it is a separate and materially
        /// more expensive function rather than a special case of this one.
        #[doc(alias = "riemann_zeta")]
        fn zeta(self) -> Self;

        /// Computes `$\zeta(s) - 1$`, accurately where `$\zeta(s)$` is within rounding of 1.
        ///
        /// `$\zeta$` approaches 1 quickly: `$\zeta(40) - 1$` is about `9.1e-13`, already below
        /// the mantissa of `$\zeta$` itself, and `$\zeta(80) - 1$` is `8.3e-25`. Forming
        /// [`zeta`](SpecialMath::zeta) and subtracting 1 therefore destroys the answer: at
        /// `s = 40` it is off by `9e-8` relative, at `s = 80` by **100%**, and past `s = 200` it
        /// returns a flat zero.
        ///
        /// This is not a wrapper around that subtraction. The Euler-Maclaurin sum underneath
        /// opens with the `$n = 1$` term, which _is_ the 1, so the complement is obtained by
        /// **omitting** it, with no cancellation anywhere and still full relative accuracy
        /// at `s = 700`, where the value is around `1e-211`. `$\zeta$` is the derived form here,
        /// the same way `exp` relates to [`exp_m1`](thermite::math::TranscendentalMath::exp_m1).
        ///
        /// Same accuracy and the same negative-`s` handling as `zeta`.
        #[doc(alias = "zeta_minus_one")]
        fn zetac(self) -> Self;

        /// Computes the polylogarithm `$\mathrm{Li}_s(z) = \sum_{k \ge 1} z^k / k^s$`, continued
        /// to the whole plane, at a scalar real order given as a [`PolylogOrder`].
        ///
        /// The order is uniform across the packet and tagged by class, because whole-number
        /// order is a different, far cheaper algorithm than arbitrary real order and every
        /// order-dependent coefficient is a per-call scalar precompute. See the
        /// [order module](crate::polylog) for why it is not a vector. [`Integer`](PolylogOrder::Integer)
        /// covers both signs: `$n \le 0$` is the closed rational form (a polynomial in
        /// `$z/(1-z)$`), `$n = 1$` is `$-\ln(1-z)$`, and `$n \ge 2$` runs entirely on tabulated
        /// `$\zeta$` values. [`Real`](PolylogOrder::Real) is the general algorithm (Wood 1992,
        /// Roughan 2026): the defining series, the unity series about `$z = 1$` with its two
        /// cancelling poles fused algebraically so orders arbitrarily close to an integer cost
        /// nothing extra, and Wood's m-th-root identity in the far field.
        ///
        /// On a real vector the argument is real and the result is the **real part** of the
        /// principal value, which for `$z > 1$` (the cut) is the same from either side. Complex
        /// vectors return the full value. On the cut it follows the sign of `$\mathrm{Im}\,z$`'s
        /// zero, C99 style, with `-0` giving mpmath's and Wood's convention for a bare real.
        ///
        /// ```rust,ignore
        /// let li2 = z.polylog(PolylogOrder::Integer(2));   // the dilogarithm
        /// let fd  = (-x.exp()).polylog(PolylogOrder::Real(1.5)); // -F_{1/2}(x)/Gamma(3/2)
        /// ```
        ///
        /// The order is spelled in the vector's own element types: `Real` carries
        /// `Self::Element` (a complex element on a complex vector, of which only a real value
        /// is implemented and anything else answers NaN, or a dual element on a dual vector, whose
        /// derivative part must be zero) and `Integer` carries the signed lane element
        /// (`i64` on an `f64` vector, `i32` on an `f32` one). Every order-dependent coefficient
        /// is computed once per call in that element type through the scalar math surface.
        ///
        /// Special values: `$\mathrm{Li}_s(1) = \zeta(s)$` for `$s > 1$` and `$+\infty$` below,
        /// `$\mathrm{Li}_s(-1) = -\eta(s)$`, `$\mathrm{Li}_s(0) = 0$`. Every arm is a fixed-length
        /// series whose length follows the policy's precision tier. Whole-number orders past
        /// `$n = 79$` (binary64) or `$n = 34$` (binary32, where `$n!$` overflows) return NaN in
        /// the far field (`$|\ln z| > 3.2$`). The series and unity arms have no such limit. Cost
        /// grows with `$\ln|z|$` in the far field at real order (one unity series per root,
        /// `$m \approx \ln|z| / 2.08$` roots).
        ///
        /// Measured against mpmath on 4952 points (real and complex `$z$`, orders from `-6` to
        /// `30` and a dozen real ones including `$2 + 10^{-9}$`), binary64 at `Precision`:
        /// whole-number orders `$n \ge 0$` within 1.3e-14 relative on the real line. Negative
        /// whole orders within 1.5e-13 (the alternating defining series on the negative axis
        /// peaks at ~2500x its sum). Real orders within 3.1e-13, with the far field's m-th-root
        /// sum cancelling by `$m^{s-1}$`, which is what makes binary32 real order 1.1e-4 there
        /// and 2e-5 elsewhere. On the cut the real part is accurate normwise (the imaginary part
        /// can be a millionth of it near `$z = 1$` at `$s = 1 + 10^{-6}$`).
        ///
        /// Autodiff closes by `$\mathrm{Li}_s'(z) = \mathrm{Li}_{s-1}(z)/z$` with the order
        /// lowered by one, which is why the runtime order is what the trait carries.
        #[scalar_form((self, order: PolylogOrder<Self, Self::Signed>) -> Self)]
        fn polylog(self, order: PolylogOrder<Self::Element, <Self::Signed as thermite::vector::GenericVector>::Element>) -> Self;

        /// A cylindrical Bessel function at compile-time order, selected by family marker:
        /// [`J`](bessel::J), [`Y`](bessel::Y), [`I`](bessel::I), [`K`](bessel::K), or any of
        /// them under [`Scaled`](bessel::Scaled). `N` is signed and the families reflect at
        /// negative order (`$J_{-n} = (-1)^n J_n$`, `$I_{-n} = I_n$`).
        ///
        /// ```rust,ignore
        /// let j2 = x.bessel_n::<J, 2>();                 // J_2(x)
        /// let ke = x.bessel_n::<Scaled<K>, 0>();         // e^x K_0(x)
        /// ```
        ///
        /// The marker only selects: each spelling is a one-line route into the kernel for that
        /// family, scaling and order form, with nothing evaluated that was not asked for. `Scaled<J>` and
        /// `Scaled<Y>` are the SciPy `jve`/`yve` scalings by `$e^{-|\mathrm{Im}\,z|}$`, which
        /// is 1 on the real axis, so on a real vector they are `J` and `Y` unchanged. On a
        /// complex vector they are the scaled values.
        fn bessel_n<F: BesselFamily, const N: i32>(self) -> Self;

        /// [`bessel_n`](SpecialMath::bessel_n) with the order taken **per lane**, at runtime,
        /// as a [`BesselOrder`] of any class.
        ///
        /// ```rust,ignore
        /// let iv = x.bessel::<Scaled<I>>(BesselOrder::Real(nu));   // e^{-|x|} I_nu(x)
        /// let jh = x.bessel::<J>(BesselOrder::HalfInteger(k));     // J_{k/2}(x), elementary
        /// ```
        fn bessel<F: BesselFamily>(self, order: BesselOrder<Self, Self::Signed>) -> Self;

        /// A spherical Bessel function at compile-time order, the twin of
        /// [`bessel_n`](SpecialMath::bessel_n) for `$j_n$`, `$y_n$`, `$i_n$`, `$k_n$`.
        ///
        /// ```rust,ignore
        /// let j3 = x.sph_bessel_n::<J, 3>();             // j_3(x)
        /// let ke = x.sph_bessel_n::<Scaled<K>, 1>();     // e^x k_1(x)
        /// ```
        fn sph_bessel_n<F: BesselFamily, const N: usize>(self) -> Self;

        /// [`sph_bessel_n`](SpecialMath::sph_bessel_n) for an order known only at runtime.
        fn sph_bessel<F: BesselFamily>(self, n: u32) -> Self;

        /// One Airy function selected by marker: [`Ai`](bessel::Ai), [`AiPrime`](bessel::AiPrime),
        /// [`Bi`](bessel::Bi), [`BiPrime`](bessel::BiPrime), or any of them under
        /// [`Scaled`](bessel::Scaled).
        ///
        /// Not a slice of [`airy_all`](SpecialMath::airy_all): the four outputs come from two
        /// Bessel passes (order 1/3 for the values, 2/3 for the derivatives), and asking for
        /// one runs one pass (`Ai` skips the `I` half of it too, so it is roughly a quarter
        /// of the tuple). Take the tuple when you want more than one of them.
        ///
        /// ```rust,ignore
        /// let ai = x.airy::<Ai>();
        /// let bp = x.airy::<Scaled<BiPrime>>();      // e^{-zeta} Bi'(x) on the positive axis
        /// ```
        fn airy<W: AiryFn>(self) -> Self;

        /// `$(\mathrm{Ai}, \mathrm{Ai}', \mathrm{Bi}, \mathrm{Bi}')$`, all four, with the
        /// exponential factored out on the positive axis when `SCALED` (SciPy `airy` / `airye`).
        ///
        /// Prefer the scaled form on **accuracy** grounds, not only range: on the positive
        /// axis the kernel produces `$e^{\zeta}K$` natively, so it evaluates no exponential
        /// anywhere and holds 1-3 eps where the unscaled one reaches 684 at `x = 100`
        /// (`$\zeta = \tfrac{2}{3}x^{3/2}$`). Unscaled, `Ai` underflows past `x ~ 104` and
        /// `Bi` overflows past `x ~ 104.5`. For `x < 0` the functions oscillate, nothing is
        /// factored out, and the phase error grows like `$|x|^{3/2}$` in every library.
        fn airy_all<const SCALED: bool>(self) -> (Self, Self, Self, Self);

        /// Computes the Beta function `$\mathrm{B}(x, y)$`
        fn beta(self, y: Self) -> Self;

        /// Computes `$\ln\left|\mathrm{B}(x, y)\right|$`, the log of the absolute Beta function.
        ///
        /// [`beta`](SpecialMath::beta) itself underflows to zero for quite ordinary arguments
        /// (`$\mathrm{B}(200, 200)$` is about `1e-121`, already gone in f32) and overflows for
        /// arguments straddling the poles. The log form has range to spare in both directions and is
        /// what the surrounding computation usually wants anyway, since Beta almost always appears
        /// inside a product of Gammas that is about to be logged.
        ///
        /// Evaluated as `$\ln\Gamma(x) + \ln\Gamma(y) - \ln\Gamma(x+y)$`. The absolute value follows
        /// [`lgamma`](SpecialMath::lgamma), so recover the sign from
        /// [`lgamma_r`](RealSpecialMath::lgamma_r) if the arguments can be negative.
        ///
        /// This buys range at some cost in relative accuracy. The three `lgamma` terms cancel
        /// against each other, shedding roughly `$\log_{10}\frac{\ln\Gamma(x+y)}{|\ln \mathrm{B}|}$`
        /// digits. That is under one digit at `$x = y = 200$`, and a little over two at
        /// `$x = 200,\ y = 1$` where the terms are near 860 and the answer is near -5.3. It remains
        /// far better conditioned than [`beta`](SpecialMath::beta), which simply has no value to
        /// return across most of that domain.
        fn lbeta(self, y: Self) -> Self;

        /// Computes the m-th derivative of the n-th degree Jacobi polynomial
        ///
        /// A the special case where α and β are both zero, the Jacobi polynomial reduces to a
        /// Legendre polynomial.
        ///
        /// **NOTE**: Given constant α, β or `n`, LLVM will happily optimize those away and unroll loops.
        fn jacobi(self, alpha: Self, beta: Self, n: u32, m: u32) -> Self;

        /// Computes the N-th degree physicists' [Hermite polynomial](https://en.wikipedia.org/wiki/Hermite_polynomials)
        /// `$H_N(x)$` where `x` is `self` and `N` is the polynomial degree.
        ///
        /// Evaluated by the three-term recurrence
        ///
        /// ```math
        /// H_{n+1}(x) = 2x\,H_n(x) - 2n\,H_{n-1}(x)
        /// ```
        ///
        /// seeded with `$H_0 = 1$` and `$H_1(x) = 2x$`. The trip count is `N`, with no data
        /// dependence, so LLVM unrolls the whole thing into straight-line FMA.
        ///
        /// The derivative is another member of the same family, `$H_n'(x) = 2n\,H_{n-1}(x)$`, so a
        /// value-and-slope pair costs one extra call rather than a separate kernel. The
        /// probabilists' polynomials are a rescaling, `$He_n(x) = 2^{-n/2} H_n(x/\sqrt{2})$`.
        ///
        /// **NOTE**: this is the raw polynomial, which grows fast: `$H_n(0) = (-2)^{n/2} (n-1)!!$` for
        /// even `n`, and `$H_n(x) \sim (2x)^n$` in the tails. It leaves binary32 range at the origin
        /// around degree 48 and binary64 around 300, and much earlier for `|x|` of a few units. If
        /// what you actually want is the *normalized* Hermite function (the quantum harmonic
        /// oscillator eigenstate, a Hermite-Gauss beam mode, or the basis of a Hermite spectral
        /// method), use [`hermite_function`](SpecialMath::hermite_function), which folds the
        /// Gaussian weight and the normalization into the recurrence and stays `$O(1)$` at every
        /// degree. The raw polynomial is the right primitive for Gauss-Hermite quadrature
        /// node-finding at modest `n` and for anything that genuinely wants `$H_n$` itself.
        fn hermite_n<const N: usize>(self) -> Self;

        /// Computes the n-th degree physicists' [Hermite polynomial](https://en.wikipedia.org/wiki/Hermite_polynomials)
        /// `H_n(x)` where `x` is `self` and `n` is a vector of unsigned integers representing the polynomial degree.
        ///
        /// The polynomial is calculated independently per-lane with the given degree in `n`.
        ///
        /// This uses the recurrence relation to compute the polynomial iteratively.
        fn hermitev(self, n: Self::Unsigned) -> Self;

        /// `$H_n(x)$` for a degree known only at runtime: [`hermitev`](SpecialMath::hermitev)
        /// with the degree splatted, which is the cheapest correct spelling of a uniform degree.
        /// The runtime twin of [`hermite_n`](SpecialMath::hermite_n).
        fn hermite(self, n: u32) -> Self;

        /// Computes the orthonormal [Hermite function](https://en.wikipedia.org/wiki/Hermite_polynomials#Hermite_functions)
        ///
        /// ```math
        /// \psi_N(x) = \frac{1}{\sqrt{2^N N! \sqrt{\pi}}}\, e^{-x^2/2}\, H_N(x)
        /// ```
        ///
        /// where `x` is `self`. These are the eigenfunctions of the quantum harmonic oscillator
        /// and of the Fourier transform, the Hermite-Gauss modes of a paraxial beam, and the
        /// basis of Hermite spectral methods. They are orthonormal on the whole line,
        /// `$\int \psi_m \psi_n\, dx = \delta_{mn}$`.
        ///
        /// Evaluated by the recurrence on the functions themselves,
        ///
        /// ```math
        /// \psi_{n+1}(x) = \sqrt{\tfrac{2}{n+1}}\, x\, \psi_n(x) - \sqrt{\tfrac{n}{n+1}}\, \psi_{n-1}(x)
        /// ```
        ///
        /// which keeps every intermediate `$O(1)$` (the polynomial's growth and the Gaussian's
        /// decay cancel inside each step), so unlike [`hermite`](SpecialMath::hermite) it does not
        /// overflow at high degree. Both square roots are literals under the unrolled loop. The
        /// per-step cost is one FMA on the critical path.
        ///
        /// # Range
        ///
        /// The only quantity that can leave the exponent range is the Gaussian seed, which is
        /// carried as `$e^{-x^2/4}$` in two halves to double the reach. Full accuracy at every
        /// degree holds for `$|x|$` under about 18.7 (binary32) or 53 (binary64), which covers
        /// every degree up to about 175 / 1400 everywhere on the line, since past the turning
        /// point `$\sqrt{2n+1}$` the true value decays faster than the seed. Beyond that the result
        /// is still correct wherever `$e^{-x^2/4}$` is representable, and zero past it.
        ///
        /// Under a `Best`-or-better precision policy on true-FMA hardware, the rounding of `$x^2$`
        /// (which is the entire error budget of a Gaussian at large `x`) is recovered exactly and
        /// corrected to first order.
        fn hermite_function_n<const N: usize>(self) -> Self;

        /// `$\psi_n(x)$` for a degree known only at runtime. The runtime twin of
        /// [`hermite_function_n`](SpecialMath::hermite_function_n): the same seed and recurrence,
        /// with the per-step constants computed rather than folded.
        fn hermite_function(self, n: u32) -> Self;

        /// Evaluates a finite series of Hermite functions at `x = self`:
        ///
        /// ```math
        /// \sum_{k=0}^{N-1} \mathrm{coeffs}[k] \cdot \psi_k(x)
        /// ```
        ///
        /// with `$\psi_k$` as in [`hermite_function`](SpecialMath::hermite_function). Evaluated by
        /// Clenshaw's backward recurrence, which is more stable than summing the functions one at
        /// a time and never forms them individually. `N` is the *length* of the coefficient array,
        /// so the highest function is `$\psi_{N-1}$`; `N = 0` is rejected.
        ///
        /// Same range as [`hermite_function`](SpecialMath::hermite_function): the coefficients are
        /// pre-scaled by half of the Gaussian and the outer factor carries the other half, so the
        /// running Clenshaw values grow no faster than `$e^{x^2/4}$`.
        #[skip_dispatch] #[compose] fn hermite_function_series_n<const N: usize>(self, coeffs: &[Self::Element; N]) -> Self;

        /// [`hermite_function_series_n`](SpecialMath::hermite_function_series_n) over a
        /// runtime-length coefficient slice.
        ///
        /// Same recurrence, same pre-scaling, same range. The length is the only difference,
        /// and it costs real work rather than only unrolling: the recurrence coefficients
        /// `$\sqrt{2/(k+1)}$` and `$\sqrt{k/(k+1)}$` fold to literals when `N` is a constant
        /// and become per-step square roots when it is not. Prefer the const form when the
        /// degree is known.
        ///
        /// An empty coefficient slice is `0`, where the const form rejects `N = 0` at compile
        /// time.
        #[skip_dispatch] #[compose] fn hermite_function_series(self, coeffs: &[Self::Element]) -> Self;

        /// Computes the generalized (associated) [Laguerre polynomial](https://en.wikipedia.org/wiki/Laguerre_polynomials)
        /// `$L_N^{(\alpha)}(x)$`, where `x` is `self` and `N` is the polynomial degree.
        ///
        /// Passing `alpha = Self::ZERO` gives the ordinary Laguerre polynomial `$L_N(x)$`; because
        /// `alpha` is an ordinary argument rather than a const generic, that case folds away
        /// completely when the zero is visible at the call site.
        ///
        /// Evaluated by the three-term recurrence
        ///
        /// ```math
        /// (n+1)\,L_{n+1}^{(\alpha)}(x) = (2n + \alpha + 1 - x)\,L_n^{(\alpha)}(x) - (n + \alpha)\,L_{n-1}^{(\alpha)}(x)
        /// ```
        ///
        /// seeded with `$L_0^{(\alpha)} = 1$` and `$L_1^{(\alpha)}(x) = 1 + \alpha - x$`. The trip count
        /// is `N`, with no data dependence, so LLVM unrolls the whole thing into straight-line FMA.
        ///
        /// The derivative is another member of the same family,
        /// `$\frac{\mathrm{d}}{\mathrm{d}x} L_n^{(\alpha)}(x) = -L_{n-1}^{(\alpha+1)}(x)$`, so a
        /// value-and-slope pair costs one extra call rather than a separate kernel.
        ///
        /// **NOTE**: the forward recurrence is the standard evaluation route (Boost and GSL both use
        /// it) and is well behaved across the oscillatory region `$0 \le x \lesssim 4n$`. Past that
        /// `$L_n^{(\alpha)}$` itself grows like `$(-x)^n/n!$` and will overflow for large `N` and `x`
        /// on its own account.
        ///
        /// Laguerre-Gaussian beam modes, the radial part of the hydrogen wavefunction, the quantum
        /// harmonic oscillator and coherent-state expansions, and Gauss-Laguerre quadrature.
        fn laguerre_n<const N: usize>(self, alpha: Self) -> Self;

        /// Computes the generalized (associated) [Laguerre polynomial](https://en.wikipedia.org/wiki/Laguerre_polynomials)
        /// `$L_n^{(\alpha)}(x)$` where `n` is a vector of unsigned integers giving the degree per lane.
        ///
        /// The per-lane counterpart of [`laguerre`](SpecialMath::laguerre), in the same relation to it
        /// as [`hermitev`](SpecialMath::hermitev) is to [`hermite`](SpecialMath::hermite). The
        /// recurrence runs to the largest `n` in the vector and lanes freeze at their own degree, so
        /// the cost is set by `max(n)` rather than by any one lane.
        fn laguerrev(self, alpha: Self, n: Self::Unsigned) -> Self;

        /// `$L_n^{(\alpha)}(x)$` for a degree known only at runtime:
        /// [`laguerrev`](SpecialMath::laguerrev) with the degree splatted. The runtime twin of
        /// [`laguerre_n`](SpecialMath::laguerre_n).
        fn laguerre(self, alpha: Self, n: u32) -> Self;

        /// Computes the orthonormal generalized [Laguerre function](https://en.wikipedia.org/wiki/Laguerre_polynomials#Generalized_Laguerre_polynomials)
        ///
        /// ```math
        /// l_N^{(\alpha)}(x) = \sqrt{\frac{N!}{\Gamma(N+\alpha+1)}}\; x^{\alpha/2} e^{-x/2}\, L_N^{(\alpha)}(x)
        /// ```
        ///
        /// where `x` is `self`. Orthonormal on the half-line, `$\int_0^\infty l_m l_n\, dx = \delta_{mn}$`.
        /// This is the radial factor of Laguerre-Gauss beam modes and (up to a power of `x` from the
        /// spherical measure) of the hydrogen wavefunctions. Defined for `$x \ge 0$` and
        /// `$\alpha > -1$`, and nothing is checked outside that.
        ///
        /// Evaluated by the recurrence on the functions themselves, with
        /// `$s_k = \sqrt{(k+1)(k+\alpha+1)}$`:
        ///
        /// ```math
        /// l_{k+1} = \frac{(2k + \alpha + 1 - x)\, l_k - s_{k-1}\, l_{k-1}}{s_k}
        /// ```
        ///
        /// which keeps every intermediate `$O(1)$`, so unlike [`laguerre`](SpecialMath::laguerre)
        /// it does not overflow at high degree or large `x`. `alpha` is a runtime vector, so each
        /// step also carries a `sqrt` and a reciprocal, beside the recurrence rather than on its
        /// critical path, and folded to literals when `alpha` is a visible constant. The seed
        /// is skipped outright by a uniform branch when every lane has `alpha = 0`, which is the
        /// ordinary Laguerre function and by far the common case.
        ///
        /// # Range
        ///
        /// The Gaussian-like seed `$x^{\alpha/2} e^{-x/2}$` is carried as `$e^{-x/4}$` in two
        /// halves, as in [`hermite_function`](SpecialMath::hermite_function). Full accuracy at
        /// every degree for `x` under about 350 (binary32) or 2800 (binary64), covering every
        /// degree up to roughly 87 / 700 everywhere on the half-line (the turning point of
        /// `$l_n^{(\alpha)}$` is near `4n`).
        ///
        /// `alpha` is unrestricted over the same `x` range. The seed's whole parameter
        /// dependence, `$x^{\alpha/2}/\sqrt{\Gamma(\alpha+1)}$`, is the square root of the Poisson
        /// mass `$P(\alpha; x)$` and is evaluated as [`poisson_pmf`](SpecialMath::poisson_pmf)
        /// is (Loader's saddle-point form, one exponential of a small exponent), so neither
        /// factor materializes (separately `$x^{\alpha/2}$` overflows binary64 near
        /// `$\alpha = 250$` and `$1/\sqrt{\Gamma(\alpha+1)}$` underflows near `$\alpha = 320$`,
        /// and their overlap would be `inf * 0`) and nothing large is exponentiated: 0-3 ulp
        /// at the peak `x ~ alpha` out to `$\alpha = 1400$`, against a 50-digit oracle.
        fn laguerre_function_n<const N: usize>(self, alpha: Self) -> Self;

        /// `$\ell_n^{(\alpha)}(x)$` for a degree known only at runtime. The runtime twin of
        /// [`laguerre_function_n`](SpecialMath::laguerre_function_n): the same seed and
        /// recurrence, with the per-step scales computed rather than folded.
        fn laguerre_function(self, alpha: Self, n: u32) -> Self;

        /// [`laguerre_function`](SpecialMath::laguerre_function) at an integer weight, taken as a
        /// **scalar** `i32` rather than a vector.
        ///
        /// Same function and same range. What changes is what the compiler can see. Every
        /// quantity the recurrence derives from the weight (the `$s_k = \sqrt{(k+1)(k+\alpha+1)}$`
        /// and their reciprocals, and the `$2k+\alpha+1$` offsets) becomes a scalar constant
        /// instead of a vector `sqrt` and reciprocal per step, and folds to a literal outright
        /// when `alpha` is compile-time known.
        ///
        /// The seed changes too. Up to `$\alpha = 170$` (binary64) / `29` (binary32) the
        /// normalization `$x^{\alpha/2}/\sqrt{\alpha!}$` is a scalar factorial, a `powi` and at
        /// most one `sqrt`, with no `ln`, `lgamma` or second `exp` at all, and a few ulp *more*
        /// accurate than the log form, whose `lgamma` error is amplified by the exponential.
        /// `$\alpha = 0$` is a scalar test that skips even that. Beyond the cap it takes
        /// the vector form's saddle-point seed. Measured on AVX2 f64x4 at degree 4:
        /// about 5x faster than the vector form at a literal small weight, 2x at a runtime one.
        ///
        /// Prefer this whenever the weight is a non-negative integer, which every classical
        /// application has: the hydrogen radial functions use `$\alpha = 2\ell+1$` and the
        /// Laguerre-Gauss beam modes use `$\alpha = |\ell|$`. Negative values are out of domain,
        /// as `$\alpha \le -1$` is for the general form.
        ///
        /// Like the series forms this is inlined into the caller rather than given its own
        /// dispatch trampoline: the weight is a plain `i32` argument, and a shared
        /// out-of-line copy would take it at runtime, which both defeats the folding above
        /// and (measured) stops LLVM overlapping consecutive evaluations, at 7x the cost.
        /// Call it from inside a `#[thermite::dispatch]` body.
        #[skip_dispatch] #[compose] fn laguerre_function_i_n<const N: usize>(self, alpha: i32) -> Self;

        /// [`laguerre_function_i_n`](SpecialMath::laguerre_function_i_n) for a degree known only
        /// at runtime.
        #[skip_dispatch] #[compose] fn laguerre_function_i(self, alpha: i32, n: u32) -> Self;

        /// Evaluates a finite series of generalized Laguerre functions at `x = self`:
        ///
        /// ```math
        /// \sum_{k=0}^{N-1} \mathrm{coeffs}[k] \cdot l_k^{(\alpha)}(x)
        /// ```
        ///
        /// with `$l_k^{(\alpha)}$` as in [`laguerre_function`](SpecialMath::laguerre_function).
        /// Clenshaw's backward recurrence, same range as the single function; `N` is the
        /// coefficient count and `N = 0` is rejected.
        #[skip_dispatch] #[compose] fn laguerre_function_series_n<const N: usize>(self, alpha: Self, coeffs: &[Self::Element; N]) -> Self;

        /// [`laguerre_function_series_n`](SpecialMath::laguerre_function_series_n) over a
        /// runtime-length coefficient slice.
        ///
        /// Same recurrence, same pre-scaling, same range. The per-step weights are computed
        /// rather than folded, as in
        /// [`hermite_function_series`](SpecialMath::hermite_function_series). An empty
        /// coefficient slice is `0`.
        #[skip_dispatch] #[compose] fn laguerre_function_series(self, alpha: Self, coeffs: &[Self::Element]) -> Self;

        /// [`laguerre_function_series`](SpecialMath::laguerre_function_series) at a scalar integer
        /// weight, in the same relation to it as
        /// [`laguerre_function_i`](SpecialMath::laguerre_function_i) is to
        /// [`laguerre_function`](SpecialMath::laguerre_function). See there for what the integer
        /// form buys.
        #[skip_dispatch] #[compose] fn laguerre_function_series_i_n<const N: usize>(self, alpha: i32, coeffs: &[Self::Element; N]) -> Self;

        /// [`laguerre_function_series_i_n`](SpecialMath::laguerre_function_series_i_n) over a
        /// runtime-length coefficient slice.
        ///
        /// The `_n` is the coefficient count and the `_i` is the integer weight, in that
        /// order because the length is the newer axis, and both mean what they do everywhere else.
        /// An empty coefficient slice is `0`.
        #[skip_dispatch] #[compose] fn laguerre_function_series_i(self, alpha: i32, coeffs: &[Self::Element]) -> Self;

        /// Evaluates a finite series of [Chebyshev polynomials](https://en.wikipedia.org/wiki/Chebyshev_polynomials)
        /// of the `K`-th kind at `x = self`:
        ///
        /// ```math
        /// \sum_{k=0}^{N-1} \mathrm{coeffs}[k] \cdot P_k(x)
        /// ```
        ///
        /// where `P_k` is `T_k`, `U_k`, `V_k`, or `W_k` depending on `K`. All four kinds share the
        /// recurrence `$P_{k+1}(x) = 2x \cdot P_k(x) - P_{k-1}(x)$` with `P_0(x) = 1`, and differ only in
        /// `P_1(x)`:
        ///
        /// | `K` | Kind   | `P_1(x)`   | Notes |
        /// |-----|--------|------------|-------|
        /// | `1` | First  (`T_k`) | `x`        | Most common, the minimax/approximation basis on `[-1, 1]`. |
        /// | `2` | Second (`U_k`) | `2x`       | Related to `$\sin((k+1)\theta)/\sin(\theta)$` under `$x = \cos\theta$`. |
        /// | `3` | Third  (`V_k`) | `2x - 1`   | "Airfoil" polynomials; `$\cos((k+\tfrac12)\theta)/\cos(\theta/2)$`. |
        /// | `4` | Fourth (`W_k`) | `2x + 1`   | `$\sin((k+\tfrac12)\theta)/\sin(\theta/2)$`. |
        ///
        /// Any other value of `K` is a compile-time error.
        ///
        /// There is deliberately no single-polynomial `T_n(x)` entry point beside this, unlike
        /// [`legendre`](SpecialMath::legendre) or [`hermite`](SpecialMath::hermite). Chebyshev
        /// polynomials are used almost exclusively as an approximation basis, i.e. as a series;
        /// their quadrature nodes and weights are closed-form, so nothing needs to iterate on a
        /// lone `$T_n$`; and the one genuine single-`$T_n$` application (Chebyshev filter response,
        /// Dolph-Chebyshev windows) needs `$|x| > 1$`, where the right evaluation is
        /// `$\cosh(n \cosh^{-1} x)$` and not this recurrence at all. A unit coefficient array
        /// recovers `$T_n$` if it is ever wanted.
        ///
        /// Evaluation is done via Clenshaw's backward recurrence with FMA, which is
        /// more numerically stable than a forward sum when the partial sums of
        /// `$\sum c_k P_k$` are much smaller than `$\max_k |c_k P_k|$` (e.g. fitted minimax series
        /// with alternating-sign coefficients). `N` is the *length* of the coefficient
        /// slice, so the highest polynomial term is `P_{N-1}`; `N = 0` is rejected,
        /// `N = 1` evaluates to `coeffs[0]`.
        ///
        /// `coeffs[0]` multiplies `P_0 = 1`, `coeffs[1]` multiplies `P_1(x)` (which depends on `K`),
        /// and so on. Because LLVM sees both `K` and `N` as constants, the recurrence loop and the
        /// `P_1` selection are fully unrolled and specialized at monomorphization time.
        ///
        /// # Accuracy near `$x = \pm 1$`
        ///
        /// The plain recurrence forms `$2x b_{k+1} - b_{k+2}$` with consecutive `$b_k$` of nearly
        /// equal magnitude as `x` approaches either endpoint, and cancels. This is a property of
        /// the *recurrence*, not of the series: measured against a 60-digit oracle at `N = 24`,
        /// it costs up to 37 ulp on sums whose own condition number is about 1, and up to 230 ulp
        /// on unstructured coefficients.
        ///
        /// Under a `Best`-or-better precision policy, real vectors instead take Reinsch's
        /// modification, which recurs on the differences (near `+1`) or sums (near `-1`) so the
        /// small quantity is never formed by subtraction. On the same grid that bounds the error
        /// envelope 2.5x to 17x tighter across all four kinds. It is an envelope improvement
        /// rather than a pointwise one (individual arguments can land worse), and costs
        /// roughly 2x on the recurrence's dependency chain, which is why it is gated.
        ///
        /// binary32 gains the same way, 2.6x to 13.5x on its own grid. Measuring it needs an
        /// f32-native one: `1 - 2^-j` rounds to exactly `1.0` for every `j >= 24`, so an f64
        /// grid piles two thirds of its points onto the endpoint itself, where the endpoint
        /// form degenerates into a plain running sum and the two policies agree, and never
        /// samples the f32 neighbourhood where the cancellation actually bites.
        ///
        /// Coefficients from a minimax or least-squares *fit* decay geometrically and barely
        /// notice either way (about 3 ulp to 1). The gap opens on slowly-decaying or
        /// non-decaying spectra: truncated expansions, near-singular functions, or coefficients
        /// that came from somewhere other than a fit.
        ///
        /// `Complex` and the composite arithmetics keep the plain recurrence at every policy,
        /// since Reinsch needs a real `copysign` and a meaningful nearest endpoint.
        #[skip_dispatch] #[compose] fn chebyshev_n<const K: usize, const N: usize>(self, coeffs: &[Self::Element; N]) -> Self;

        /// [`chebyshev_n`](SpecialMath::chebyshev_n) over a runtime-length coefficient slice.
        ///
        /// `K` stays a const generic, since it selects *which* Chebyshev kind, not how many
        /// coefficients, and there are exactly four. Only the length becomes dynamic.
        ///
        /// Same recurrence and the same `Best`-precision Reinsch form near `$x = \pm 1$`; what
        /// the runtime length costs is the unrolling and the folded `coeffs` indices. An empty
        /// coefficient slice is `0`.
        #[skip_dispatch] #[compose] fn chebyshev<const K: usize>(self, coeffs: &[Self::Element]) -> Self;

        /// Computes the Gaussian function with amplitude `a` and standard deviation `c`, defined as `$a\, e^{-\frac{1}{2}(x/c)^2}$`.
        ///
        /// The position `b` is assumed to be zero. For a non-zero position, use `self - b` as the input.
        fn gaussian(self, a: Self, c: Self) -> Self;

        /// Computes the Planck shape factor `$\frac{x^3}{e^x - 1}$`, finite at `x = 0` where it
        /// vanishes like `$x^2$`.
        ///
        /// The dimensionless kernel of Planck's law: substituting `$x = h\nu/kT$` recovers the
        /// spectral radiance up to a scale factor, so this is the part worth computing carefully and
        /// the constants are left to the caller. Radiative transfer, climate radiation budgets, and
        /// stellar atmospheres.
        ///
        /// The denominator cancels for small `x` and the quotient is `$0/0$` at the origin.
        /// Evaluated here as `$x^2/\varphi_1(x)$` using `phi_n::<1>`, which is finite and equal
        /// to 1 there, so the singularity never forms rather than being patched after the fact.
        fn planck(self) -> Self;

        /// Computes the m-th associated n-th degree Legendre polynomial,
        /// where m=0 signifies the regular n-th degree Legendre polynomial.
        ///
        /// If `m` is odd, the input is only valid between -1 and 1
        ///
        /// **NOTE**: Given constant `n` and/or `m`, LLVM will happily unroll and optimize inner loops.
        ///
        /// Internally, this is computed with [`jacobi`](SpecialMath::jacobi) when m > 0.
        fn legendre(self, n: u32, m: u32) -> Self;

        /// Evaluates a finite [Legendre series](https://en.wikipedia.org/wiki/Legendre_polynomials)
        /// at `x = self`:
        ///
        /// ```math
        /// \sum_{k=0}^{N-1} \mathrm{coeffs}[k] \cdot P_k(x)
        /// ```
        ///
        /// The form a Legendre-moment expansion takes: Mie and Henyey-Greenstein scattering
        /// phase functions tabulated by their moments, multipole expansions in `$\cos\theta$`, and
        /// the polar factor of a spherical-harmonic expansion at fixed order.
        ///
        /// Evaluated by Clenshaw's backward recurrence on the Legendre three-term relation, which
        /// is more stable than building each `$P_k$` with [`legendre`](SpecialMath::legendre) and
        /// summing, and does `$O(N)$` work rather than `$O(N^2)$`. The recurrence ratios
        /// `$(2k+1)/(k+1)$` and `$k/(k+1)$` are literals under the unrolled loop, so the per-step
        /// cost matches [`chebyshev`](SpecialMath::chebyshev): one FMA on the critical path. `N`
        /// is the coefficient count; `N = 0` is rejected, `N = 1` evaluates to `coeffs[0]`.
        ///
        /// Plain Clenshaw at every policy: the endpoint cancellation that `chebyshev` treats
        /// under `Best` precision exists here too (`$P_n(1) = 1$` for every `n`), but its
        /// Reinsch-style rewrite for the Legendre ratios has not been derived or measured.
        #[skip_dispatch] #[compose] fn legendre_series_n<const N: usize>(self, coeffs: &[Self::Element; N]) -> Self;

        /// [`legendre_series_n`](SpecialMath::legendre_series_n) over a runtime-length
        /// coefficient slice.
        ///
        /// Plain Clenshaw here too. The recurrence ratios `$(2k+1)/(k+1)$` and `$k/(k+1)$` are
        /// literals only when `N` is a constant, so this pays a division per step where the
        /// const form pays none, the widest const-versus-slice gap of the series family.
        /// An empty coefficient slice is `0`.
        #[skip_dispatch] #[compose] fn legendre_series(self, coeffs: &[Self::Element]) -> Self;

        /// Computes the [Zernike](https://en.wikipedia.org/wiki/Zernike_polynomials) radial
        /// polynomial `$R_n^m(\rho)$`, where `rho` is `self`.
        ///
        /// Returns zero unless `$m \le n$` with `$n - m$` even, the condition for the mode to
        /// exist. `m` is the *absolute* azimuthal frequency here. The sign only affects the
        /// angular factor, which lives in [`zernike`](SpecialMath::zernike).
        ///
        /// Evaluated through the shifted Jacobi identity
        ///
        /// ```math
        /// R_n^m(\rho) = \rho^m\, P_{(n-m)/2}^{(0,\,m)}\!\left(2\rho^2 - 1\right)
        /// ```
        ///
        /// rather than the textbook sum
        /// `$\sum_k (-1)^k \frac{(n-k)!}{k!\,((n+m)/2 - k)!\,((n-m)/2 - k)!} \rho^{n-2k}$`, which
        /// alternates factorials of size `$(n-k)!$` against an answer bounded by 1 and loses all
        /// precision somewhere around `n = 10-15`. That is well inside the range adaptive optics,
        /// ophthalmology and surface metrology actually use.
        ///
        /// The `$(-1)^{(n-m)/2}$` prefactor usually seen with this identity is absent because the
        /// argument is written `$2\rho^2 - 1$` rather than `$1 - 2\rho^2$`: reflecting a Jacobi
        /// polynomial swaps its two parameters and absorbs exactly that sign.
        ///
        /// The polynomial is only orthogonal on `$\rho \in [0, 1]$` and grows quickly outside it.
        /// Nothing clamps the argument, so an unnormalized pupil coordinate stays the caller's
        /// problem.
        fn zernike_r(self, n: u32, m: u32) -> Self;

        /// Computes the Zernike polynomial `$Z_n^m(\rho, \theta)$` on the unit disc, with `rho`
        /// as `self`:
        ///
        /// ```math
        /// Z_n^m(\rho, \theta) = N_n^m\, R_n^{|m|}(\rho) \times
        ///   \begin{cases} \cos(m\theta) & m \ge 0 \\ \sin(|m|\theta) & m < 0 \end{cases}
        /// ```
        ///
        /// Returns zero unless `$|m| \le n$` with `$n - |m|$` even.
        ///
        /// `NORM` selects the normalization `$N_n^m$` and must be either
        /// [`ZERNIKE_UNIT_PEAK`] (`$N = 1$`, so `$R_n^m(1) = 1$` and coefficients read as peak
        /// amplitude) or [`ZERNIKE_ORTHONORMAL`]
        /// (`$N_n^m = \sqrt{2(n+1)/(1 + \delta_{m,0})}$`, the ANSI Z80.28 and Noll convention,
        /// under which coefficients read as RMS contributions). Any other value is a compile-time
        /// error. There is deliberately no default: the two differ by a factor of up to
        /// `$\sqrt{2(n+1)}$` per mode, and picking one silently is how coefficient sets get
        /// misinterpreted.
        ///
        /// `(n, m)` is a runtime pair rather than a const generic on purpose. The workload is a
        /// basis, not a function. A wavefront fit evaluates tens to hundreds of modes over
        /// thousands of pupil samples, with the mode list coming from a config or a sensor
        /// geometry, so the degree is loop-invariant across the vector axis and const-generic
        /// specialization would buy a jump table rather than an unrolled loop.
        ///
        /// The single-index conventions (ANSI Z80.28 / OSA, Noll, Fringe) and the conversions
        /// between them are in [`crate::zernike`]. They disagree from the second term
        /// onward, so convert at the boundary rather than assuming.
        fn zernike<const NORM: u8>(self, theta: Self, n: u32, m: i32) -> Self;

        /// Evaluates **all** Zernike modes through degree `L` at the Cartesian pupil point
        /// `(x, y)`, into `out[j]` for the ANSI Z80.28 / OSA index `$j = (n(n+2) + m)/2$`.
        ///
        /// `N` must equal `(L+1)(L+2)/2` (compile-time checked), and `NORM` is
        /// [`ZERNIKE_UNIT_PEAK`] or [`ZERNIKE_ORTHONORMAL`] as on
        /// [`zernike`](SpecialMath::zernike).
        ///
        /// This is the entry point a wavefront fit or reconstruction wants. It is not merely
        /// a loop over [`zernike`](SpecialMath::zernike). Substituting `$s = x^2+y^2$`
        /// splits every mode into a polynomial in `s` times `$\operatorname{Re}$` or
        /// `$\operatorname{Im}$` of `$(x+iy)^{|m|}$`, which is where the `$\rho^{|m|}$` and the
        /// `$\cos m\theta$` both come from at once. Evaluation is then **pure polynomial
        /// arithmetic**: no `atan2`, no `sqrt`, no trigonometry, no division, `$O(L^2)$` FMAs
        /// for the entire basis, and no singularity at the pupil centre. Calling the
        /// single-mode form per mode instead costs a `sin_cos` and a `powi` each and restarts
        /// the radial recurrence every time, for `$O(L^3)$` work.
        ///
        /// Cartesian input is part of that, not a convenience: pupil samples arrive as
        /// `(x, y)`, and a polar entry point would charge an `atan2` per sample for an angle
        /// this kernel immediately dissolves.
        ///
        /// Fully unrolled at compile time for each `L` up to
        /// [`MAX_ZERNIKE_DEGREE`](specialized::MAX_ZERNIKE_DEGREE); above that it takes a
        /// rolled path that is correct at any degree and substantially slower.
        ///
        /// Nothing normalizes `(x, y)` onto the unit disc. Outside it the polynomials still
        /// evaluate correctly and simply are not orthogonal.
        ///
        /// The layout is ANSI because it is the scheme whose index has a closed form *and*
        /// whose degree truncation is contiguous. Noll and Fringe callers gather through
        /// [`noll_to_ansi`](crate::zernike::noll_to_ansi) /
        /// [`fringe_to_ansi`](crate::zernike::fringe_to_ansi).
        ///
        /// ```
        /// use thermite::prelude::*;
        /// use thermite_special::{SpecialMath, ZERNIKE_ORTHONORMAL};
        /// use thermite_special::zernike::noll_to_ansi;
        ///
        /// type V = Vector<f64>;
        /// const L: usize = 4;
        /// const N: usize = 15; // (L+1)(L+2)/2
        ///
        /// let mut basis = [V::ZERO; N];
        /// V::zernike_basis::<L, ZERNIKE_ORTHONORMAL, N>(V::splat(0.3), V::splat(0.4), &mut basis);
        ///
        /// // Noll 4 is defocus, Z_2^0 = sqrt(3) (2 rho^2 - 1) orthonormal.
        /// let defocus = basis[noll_to_ansi(4) as usize].extract::<0>();
        /// assert!((defocus - 3f64.sqrt() * (2.0 * 0.25 - 1.0)).abs() < 1e-14);
        /// ```
        #[skip_dispatch] #[compose] fn zernike_basis<const L: usize, const NORM: u8, const N: usize>(x: Self, y: Self, out: &mut [Self; N]) -> ();

        /// Computes both branches of the Lambert W function simultaneously: (`$W_0(x)$`, `$W_{-1}(x)$`).
        ///
        /// The `$W_0$` result is valid for `x >= -1/e`; the `$W_{-1}$` result is valid for `-1/e <= x < 0`.
        /// Outside these domains, the respective result is NaN (when overflow checking is enabled).
        fn lambert_w(self) -> (Self, Self);

        // TEMP(bessel_j): disabled until orders beyond J_0 exist. Only f32 `J_0` was
        // ever implemented, so every composite type (Dual, Complex, Compensated) could
        // do nothing but `todo!()`. Re-enable this line and the ones marked
        // TEMP(bessel_j) elsewhere together.
        //fn bessel_j<const N: i32>(self) -> Self;

        /// Computes the generalized exponential integral `E_n(x)` for integer order `n`.
        #[doc(alias = "expn")]
        #[doc(alias = "exp1")]
        fn expint_n<const N: usize>(self) -> Self;

        /// `E_n(x)` for an order known only at runtime. The runtime twin of
        /// [`expint_n`](SpecialMath::expint_n): the same `E_1` kernel, the same recurrence and the
        /// same continued-fraction handover, so the two agree to the bit.
        fn expint(self, n: u32) -> Self;

        /// Returns `$\varphi_N(x)$`, the `N`-th phi-function of exponential integrators.
        ///
        /// ```math
        /// \varphi_0(x) = e^x, \qquad
        /// \varphi_{k+1}(x) = \frac{\varphi_k(x) - 1/k!}{x}, \qquad
        /// \varphi_k(x) = \sum_{n \ge 0} \frac{x^n}{(n + k)!}, \qquad
        /// \varphi_k(0) = \frac{1}{k!}
        /// ```
        ///
        /// `phi_n::<0>` is `exp`. `phi_n::<1>` is `$(e^x - 1)/x$`, which written out
        /// directly is `$0/0$` at the origin and loses most of the mantissa near it, so it is
        /// evaluated as `$\mathrm{expm1}(x)/x$` with the removable singularity filled in (the
        /// value is 1), which is accurate across the whole line. Outside the
        /// exponential-integrator literature `phi_n::<1>` goes by **`exprel`**, which is the name
        /// SciPy, Boost and the statistics literature use for it. There is no separate
        /// `exprel` here because this is it. Beyond that the recurrence is
        /// the wrong way to compute them: each step subtracts `1/k!` from a value that is barely
        /// larger while `|x|` is small, so `$\varphi_2 = (\mathrm{expm1}(x) - x)/x^2$` loses twice the bits
        /// `phi_n::<1>` would have, and gets worse with `N`. Below `|x| = N` this sums the series
        /// instead (its terms are monotone there, so nothing cancels), and above it runs the
        /// recurrence upward from `expm1`, where the amplification per step is bounded. Measured
        /// against mpmath, both arms sit within a few ulp for `N <= 8`.
        ///
        /// The series arm's length is bounded by the policy's `max_iterations`. The primitive
        /// float types know their precision statically and use a fixed count instead. Nothing
        /// caps `N`, though nothing needs it large: ETDRK4 wants `phi_1..phi_3`, and exponential
        /// Rosenbrock methods rarely go past `phi_4`.
        ///
        /// `phi_n::<1>` alone is the coefficient that keeps appearing wherever an exponential is
        /// integrated over a finite step:
        ///
        /// * The in-scattering integral through a homogeneous medium,
        ///   `$\int_0^t e^{-\sigma s}\,ds = t\,\varphi_1(-\sigma t)$`. The singular case is the empty
        ///   medium, which is not an edge case in practice.
        /// * Exact stepping of an Ornstein-Uhlenbeck process, and the Langevin thermostat's
        ///   mean-reversion factor.
        /// * Frame-rate-independent exponential smoothing, usually written `1 - exp(-k * dt)` and then
        ///   divided by `k`.
        ///
        /// The higher orders are the coefficients of exponential time differencing: integrating
        /// `y' = Ly + N(y)` exactly over a step gives `$y(h) = e^{hL} y_0 + h\,\varphi_1(hL)\,N$`, and
        /// expanding `N` in time along the step brings in `$\varphi_2, \varphi_3, \ldots$` as the
        /// weights of the higher-order terms.
        #[doc(alias = "exprel")]
        fn phi_n<const N: usize>(self) -> Self;

        /// `$\varphi_n(x)$` for an order known only at runtime. The runtime twin of
        /// [`phi_n`](SpecialMath::phi_n): the same series and recurrence arms, with the series
        /// length worked out from `n` per call rather than at compile time.
        fn phi(self, n: u32) -> Self;

        /// Carlson symmetric elliptic integral, selected by a [`CarlsonKind`] request struct
        /// with named fields. The arity (and which argument is the parameter / repeated one)
        /// is fixed per kind, so the wrong shape is a compile error.
        ///
        /// ```rust,ignore
        /// let rf = V::carlson(CarlsonRf { x, y, z });
        /// let rj = V::carlson_p::<Precision, _>(CarlsonRj { x, y, z, p });
        /// ```
        #[kind]
        fn carlson<K: CarlsonKind<Output = Self>>(kind: K) -> Self;

        /// Legendre elliptic integral, selected by an [`EllipticKind`] request struct. Each
        /// form ([`EllintK`](elliptic::EllintK)/[`EllintF`](elliptic::EllintF)/[`EllintE`](elliptic::EllintE)/
        /// [`EllintEInc`](elliptic::EllintEInc)/[`EllintD`](elliptic::EllintD)/[`EllintDInc`](elliptic::EllintDInc)/
        /// [`EllintPi`](elliptic::EllintPi)/[`EllintPiInc`](elliptic::EllintPiInc)) carries exactly
        /// its own arguments, and completeness is encoded by whether the struct has a `phi` field.
        ///
        /// Two family members that are _not_ Legendre integrals dispatch through here as well,
        /// because they are built from the same Carlson forms and belong beside their siblings:
        /// [`JacobiZeta`](elliptic::JacobiZeta), the oscillating part of `$E(\varphi, k)$`, and
        /// [`HeumanLambda`](elliptic::HeumanLambda), its complementary-modulus companion.
        ///
        /// ```rust,ignore
        /// let k_int = V::ellint(EllintK { k });                       // K(k)
        /// let e_inc = V::ellint_p::<Precision, _>(EllintEInc { phi, k }); // E(phi, k)
        /// let z     = V::ellint(JacobiZeta { phi, k });               // Z(phi, k)
        /// ```
        #[kind]
        fn ellint<K: EllipticKind<Output = Self>>(kind: K) -> Self;
    }

    /// Special math functions that are only defined for real-valued floating-point vectors.
    ///
    /// These functions either rely on ordering/sign information that has no complex analogue
    /// (e.g. `erfinv`, `probit`, `lgamma_r`), or use the real absolute value in a way that
    /// makes them non-holomorphic (e.g. `algebraic_sigmoid`).
    #[diagnostic::on_unimplemented(
        message = "`{Self}` does not provide real-valued special math (`erfinv`, `probit`, `lgamma_r`, ...)",
        note = "`RealSpecialMath` is only meaningful for real-valued float vectors. Complex number types deliberately do not implement it. A bare `f32`/`f64` does not qualify either. Wrap it in `Vector::<f32>::splat(x)`, or use `ScalarSpecialMath`."
    )]
    pub trait RealSpecialMath: SpecialMathWithPolicy {
        /// Computes the inverse error function.
        fn erfinv(self) -> Self;

        /// Computes the Probit function, the inverse of the cumulative distribution function
        /// of the standard normal distribution.
        #[doc(alias = "ndtri")]
        fn probit(self) -> Self;

        /// Computes the cumulative distribution function of the standard normal
        /// distribution, the inverse of [`probit`](RealSpecialMath::probit):
        ///
        /// ```math
        /// \Phi(x) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{x} e^{-t^2/2}\,dt
        ///         = \tfrac12 \operatorname{erfc}\!\left(-\frac{x}{\sqrt 2}\right)
        /// ```
        ///
        /// The probability that a standard normal variable falls below `x`: z-scores to
        /// p-values, the `N(d_1)`/`N(d_2)` terms of Black-Scholes, the probit link, and
        /// `x * ndtr(x)` is GELU. The name is Cephes/SciPy's.
        ///
        /// Underflows to zero below about `x = -38.6` (`f64`) and `-14.4` (`f32`). When
        /// the tail probability itself is the quantity of interest, use
        /// [`log_ndtr`](RealSpecialMath::log_ndtr), which is finite there.
        #[doc(alias = "norm_cdf")]
        #[doc(alias = "Phi")]
        fn ndtr(self) -> Self;

        /// Computes `$\ln \Phi(x)$`, the logarithm of the standard normal CDF, finite
        /// for every finite `x`.
        ///
        /// `ln(ndtr(x))` is `-inf` below `x ~ -38.6` in `f64` (`-14.4` in `f32`), exactly
        /// where a probit or censored-regression likelihood, a truncated-normal density,
        /// or an expected-improvement acquisition needs the tail: `log_ndtr(-100)` is an
        /// ordinary `-5004.6`. The kernel keeps `$-x^2/2$` in the log domain and takes
        /// the rest from [`erfcx`](SpecialMath::erfcx), which has no underflow, so the
        /// left tail carries full relative accuracy to the largest `x` whose square is
        /// representable. On the right it is `ln_1p` of the complement, so
        /// `log_ndtr(10) = -7.6e-24` rather than a rounded zero.
        ///
        /// Costs one `erfcx`, one `ln_1p`, and an `exp` for the lanes with `x > 0`.
        #[doc(alias = "log_norm_cdf")]
        fn log_ndtr(self) -> Self;

        /// Computes `$\ln \operatorname{erfc}(x)$`, finite for every finite `x`.
        ///
        /// `erfc` underflows at `x ~ 27` (`f64`) / `9.3` (`f32`) and its logarithm does
        /// not: `logerfc(100) = -10004.8`. This is the log-domain form of a Gaussian
        /// tail wherever `erfc` rather than the normal CDF is the natural quantity
        /// (Ewald sums, Gaussian-smeared edges, the Mills ratio in the log domain), and
        /// it is `log_ndtr` with `x = -\sqrt 2 x'`. Built on
        /// [`erfcx`](SpecialMath::erfcx) with `$-x^2$` kept in the log domain. On the
        /// left, where `erfc(x)` is between 1 and 2, it is `ln_1p(erf(|x|))`, so the
        /// result stays accurate down to `logerfc(-1e-20) = 1.13e-20`.
        #[doc(alias = "log_erfc")]
        fn logerfc(self) -> Self;

        /// The Fresnel integrals `$S(x) = \int_0^x \sin(\pi t^2/2)\,dt$` and
        /// `$C(x) = \int_0^x \cos(\pi t^2/2)\,dt$`, together.
        ///
        /// **Returns `(S, C)`**, the same order as SciPy's `fresnel` and this crate's own
        /// [`sici`](RealSpecialMath::sici).
        ///
        /// Both are odd, both tend to `1/2`, and both stay in `[0.32, 0.72]` past the
        /// first oscillation. Measured against a 45-digit oracle over `x` from `1e-4` to
        /// `1e15`: 2.80 ulp (`C`) and 2.64 (`S`) in `f64`, 2.14 and 3.40 in `f32` out to
        /// `1e7`.
        ///
        /// The phase `$\pi x^2/2$` is carried in two words and reduced exactly, which is
        /// not a refinement but the whole of the large-argument accuracy: computed the
        /// obvious way as `x*x*0.5`, the phase is already 5.3e-6 wrong at `x = 98765` and
        /// returns the wrong _sign_ by `$x \approx 10^9$`, and since `C` and `S` are
        /// `1/2` plus a term of size `$1/(\pi x)$` that error lands straight on the
        /// result. Below `Average` the residual is dropped and that behaviour returns.
        ///
        /// Above `x = 1.147e16` (`f64`) / `2.136e7` (`f32`) the oscillating correction is
        /// under half an ulp of `1/2`, and both are exactly `1/2`.
        #[doc(alias = "fresnels")]
        #[doc(alias = "fresnelc")]
        fn fresnel(self) -> (Self, Self);

        /// `$C(x)$` alone. See [`fresnel`](RealSpecialMath::fresnel).
        ///
        /// Unlike `airy::<Ai>` this is not a cheaper evaluation by
        /// much: `C` and `S` share the argument reduction, the phase and both
        /// auxiliaries, so asking for one drops a single Chebyshev series and one
        /// reconstruction: roughly a third, not three quarters.
        fn fresnel_c(self) -> Self;

        /// `$S(x)$` alone. See [`fresnel_c`](RealSpecialMath::fresnel_c).
        fn fresnel_s(self) -> Self;

        /// The trigonometric integrals `$\mathrm{Si}(x) = \int_0^x \frac{\sin t}{t}\,dt$`
        /// and `$\mathrm{Ci}(x) = \gamma + \ln x + \int_0^x \frac{\cos t - 1}{t}\,dt$`,
        /// together. Returns `(Si, Ci)`.
        ///
        /// `Si` is odd. `Ci` is real only on the positive axis (`$\mathrm{Ci}(-x) =
        /// \mathrm{Ci}(x) + i\pi$`), so this returns `Ci(|x|)`, dropping the imaginary
        /// part, which is what SciPy's `sici` does. `Ci(0)` is `$-\infty$`.
        ///
        /// Measured 2.03 ulp (`Si`) and 1.42 (`Ci`, against its envelope) in `f64` over
        /// `x` from `1e-4` to `1e15`. In `f32`, 1.34 and 1.99.
        ///
        /// Two things worth knowing before relying on `Ci`:
        ///
        /// - **It has zeros**, the first near `x = 0.6165`, and no algorithm is
        ///   relatively accurate at one. The accuracy above is relative to
        ///   `$\lvert\gamma + \ln x\rvert + \lvert\mathrm{Cin}\rvert$` below the
        ///   crossover and to the `$1/x$` envelope above it.
        /// - **Its large-argument accuracy is
        ///   [`sin_cos`](thermite::math::TranscendentalMath::sin_cos)'s**: for `Ci` the
        ///   oscillation _is_ the value, so a phase error is a relative error, and full
        ///   argument reduction is a `Best`-tier property. `Si` is insulated, tending to
        ///   `$\pi/2$` with the oscillation only a `$1/x$` correction, and is `$\pi/2$`
        ///   exactly above `x = 1.147e16` (`f64`) / `2.136e7` (`f32`). `Ci` has no such
        ///   cutoff: it decays like `$1/x$` and stays representable for every finite `x`.
        #[doc(alias = "si")]
        #[doc(alias = "ci")]
        fn sici(self) -> (Self, Self);

        /// `$\mathrm{Si}(x)$` alone. See [`sici`](RealSpecialMath::sici), and
        /// [`fresnel_c`](RealSpecialMath::fresnel_c) for what a single accessor saves.
        #[doc(alias = "Si")]
        fn sinint(self) -> Self;

        /// `$\mathrm{Ci}(x)$` alone. See [`sici`](RealSpecialMath::sici).
        #[doc(alias = "Ci")]
        fn cosint(self) -> Self;

        /// Computes the inverse of [`log_ndtr`](RealSpecialMath::log_ndtr): the `x` with
        /// `$\ln \Phi(x) = y$`, for `y <= 0`. The quantile of a log-probability.
        ///
        /// [`probit`](RealSpecialMath::probit) of `$e^y$` stops working once `$e^y$`
        /// underflows (`y < -745` in `f64`), which is exactly where a log-likelihood, a
        /// truncated-normal EM step or an extreme-value fit needs the quantile. This
        /// inverts `log_ndtr` directly, by Newton with the inverse Mills ratio as the
        /// derivative, from a `probit(e^y)` seed one precision tier down where that
        /// exists and from the tail asymptotic below. Within a few ulp of the true inverse
        /// of the given `y` over the whole domain. `y = 0` gives `+inf`, `y = -inf` gives
        /// `-inf`, and `y > 0` is NaN.
        #[doc(alias = "ndtri_exp")]
        fn inv_log_ndtr(self) -> Self;

        /// Computes the inverse of the digamma function on `$(0, \infty)$`: the `x` with
        /// `$\psi(x) = y$`.
        ///
        /// The maximum-likelihood estimate of a gamma shape or a Dirichlet concentration is
        /// this function of a mean log. Newton on `digamma` with `trigamma` from Minka's
        /// seed (`$e^y + 1/2$` above `y = -2.22`, `$-1/(y + \gamma)$` below). Above `y = 6`
        /// the Stirling series is solved for `x` directly, since there Newton on `digamma`
        /// cannot see past `digamma`'s own rounding. `+inf` maps to `+inf` and `-inf` to `0`.
        #[doc(alias = "digammainv")]
        fn inv_digamma(self) -> Self;

        /// Computes the Wright omega function, the `$\omega > 0$` with
        /// `$\omega + \ln \omega = x$`.
        ///
        /// This is `$W_0(e^x)$`, the principal Lambert W of an exponential, evaluated without
        /// forming `$e^x$`: `$W_0(e^x)$` overflows past `x = 709` where `$\omega(x) \approx x - \ln x$`
        /// is ordinary. Newton on `$\omega + \ln \omega - x$` from a cheap seed per region.
        /// Below `x = -7` the Lagrange series in `$e^x$` is the answer outright.
        #[doc(alias = "wrightomega")]
        fn wright_omega(self) -> Self;

        /// Computes the modified Bessel ratio `$A_\nu(x) = I_\nu(x) / I_{\nu-1}(x)$` for
        /// `nu >= 1`, odd in `x`.
        ///
        /// With `$p = 2\nu$` this is the mean resultant length of a von Mises-Fisher
        /// distribution on `$S^{p-1}$` at concentration `x`. `nu = 1` is the von Mises circle
        /// `$I_1/I_0$`, and `nu = 3/2` is the [`langevin`](RealSpecialMath::langevin) function.
        /// Never forms the two Bessel functions where they would underflow: a series pair for
        /// small `x`, the continued fraction for the ratio in the middle, and the scaled
        /// quotient only where `x` dominates the order. The order is a plain vector, but whole
        /// and half-integer orders reach their fast Bessel kernels through the order simplifier.
        #[doc(alias = "vmf_a")]
        fn bessel_ratio<F: BesselRatioFamily>(self, nu: Self) -> Self;

        /// Computes the inverse of [`bessel_ratio`](RealSpecialMath::bessel_ratio): the
        /// concentration `$\kappa$` with `$I_\nu(\kappa)/I_{\nu-1}(\kappa) = r$`, for
        /// `0 <= r < 1`, odd in `r`.
        ///
        /// The maximum-likelihood concentration of a von Mises-Fisher distribution from its
        /// observed mean resultant length, in any dimension `$p = 2\nu$`. Banerjee's
        /// `$r(p - r^2)/(1 - r^2)$` seeds a Newton whose derivative is the closed form
        /// `$1 - A^2 - (2\nu - 1)A/\kappa$`, so each step is one ratio evaluation. `r = 1`
        /// gives `+inf`, `r > 1` NaN.
        ///
        /// As `r -> 1` the problem itself is ill-conditioned: `$\kappa \sim (p-1)/(2(1-r))$`,
        /// and an ulp of `r` is a relative `$2\kappa\epsilon/(p-1)$` of `$\kappa$`. The result is
        /// the exact inverse of the given `r` to that extent.
        #[doc(alias = "vmf_kappa")]
        fn inv_bessel_ratio<F: BesselRatioFamily>(self, nu: Self) -> Self;

        /// Computes `$1 - A_\nu(x)$`, the complement of
        /// [`bessel_ratio`](RealSpecialMath::bessel_ratio), to full relative accuracy
        /// where the ratio itself is within an ulp of 1.
        ///
        /// `1 - bessel::ratio::<I>(x)` is gone once `$A$` rounds to 1 (`x` past `1e16 (p-1)/2`),
        /// and is only accurate to `$\epsilon/(1 - A)$` before that. This evaluates the
        /// complement directly for `x >= 8 nu`, from the Hankel expansions at a reduced order
        /// and the ratio recurrence walked upward in complement form. `$A$` is odd, so
        /// `$1 - A(-x) = 2 - (1 - A(x))$`.
        #[doc(alias = "vmf_a_1m")]
        fn bessel_ratio_1m<F: BesselRatioFamily>(self, nu: Self) -> Self;

        /// Computes the inverse of [`bessel_ratio_1m`](RealSpecialMath::bessel_ratio_1m):
        /// the concentration `$\kappa$` with `$1 - I_\nu(\kappa)/I_{\nu-1}(\kappa) = t$`, for
        /// `0 < t <= 2` (`t = 1 - r`).
        ///
        /// The complement form of [`inv_bessel_ratio`](RealSpecialMath::inv_bessel_ratio)
        /// for nearly concentrated data: `$\kappa \sim (p-1)/(2t)$` as `t -> 0`. This form
        /// keeps full relative accuracy there instead of losing `$2\kappa\epsilon/(p-1)$`
        /// to the rounding of `r`. It is the [`inv_langevin_1m`](RealSpecialMath::inv_langevin_1m)
        /// move in every dimension. `t = 0` gives `+inf`. `t` in `(1, 2]` is a negative `r`
        /// and returns the mirrored `$\kappa$`.
        #[doc(alias = "vmf_kappa_1m")]
        fn inv_bessel_ratio_1m<F: BesselRatioFamily>(self, nu: Self) -> Self;

        /// Computes the `k`-th node and weight of the `n`-point Gauss-Legendre quadrature
        /// rule on `$[-1, 1]$`, with the root index `k` taken **per lane**.
        ///
        /// The rule integrates every polynomial through degree `$2n - 1$` exactly:
        /// `$\int_{-1}^{1} f \approx \sum_k w_k f(x_k)$`, `$x_k$` the roots of `$P_n$` in
        /// descending order (`k = 0` is the largest, `$x_{n-1-k} = -x_k$`) and
        /// `$w_k = 2 / ((1 - x_k^2) P_n'(x_k)^2)$`. The packet _is_ the rule: sweep `k` over
        /// `0..n` in packets of consecutive indices and store the two vectors. Every lane
        /// runs the same `O(n)` recurrence, so a packet of roots costs one root.
        ///
        /// Tricomi's `$\cos(\pi(k + 3/4)/(n + 1/2))$` seeds a Newton on `$P_n$` from the
        /// recurrence, and nodes land within a few `$\epsilon$` absolute. A non-integer or
        /// out-of-range `k` gives NaN in both.
        ///
        /// ```rust,ignore
        /// let n = 16;
        /// for base in (0..n).step_by(V::LANES) {
        ///     let k = V::from_array(core::array::from_fn(|i| (base + i) as f64));
        ///     let (x, w) = k.gauss_legendre(n as u32); // lanes past n - 1 are NaN
        /// }
        /// ```
        fn gauss_legendre(self, n: u32) -> (Self, Self);

        /// Computes the `k`-th node and weight of the `n`-point Gauss-Hermite rule, for
        /// `$\int_{-\infty}^{\infty} f(x) e^{-x^2}\,dx \approx \sum_k w_k f(x_k)$`, the root
        /// index `k` per lane (`k = 0` the largest root, `$x_{n-1-k} = -x_k$`).
        ///
        /// Same shape as [`gauss_legendre`](RealSpecialMath::gauss_legendre): a packet of
        /// consecutive indices is the rule. Seeded from the WKB phase of the Hermite equation
        /// and finished by Newton on `$H_n/n!$`, whose recurrence stays in range where the raw
        /// `$H_n$` overflows at degree 48. The weights are the unscaled ones, which reach
        /// `$e^{-x_k^2}$` at the outer nodes. The scalar factor in them underflows past
        /// `n = 170` in `f64` and `n = 40` in `f32`, which bounds the rule.
        fn gauss_hermite(self, n: u32) -> (Self, Self);

        /// Computes the `k`-th node and weight of the `n`-point Gauss-Laguerre rule, for
        /// `$\int_0^{\infty} f(x)\, x^\alpha e^{-x}\,dx \approx \sum_k w_k f(x_k)$`, the root
        /// index `k` and `alpha > -1` per lane (`k = 0` the largest root).
        ///
        /// Same shape as [`gauss_legendre`](RealSpecialMath::gauss_legendre). Seeded from the
        /// WKB phase of the Laguerre equation, whose phase count between the turning points
        /// carries the Bessel-zero offset on the left and the Airy offset on the right, and
        /// finished by Newton on the raw `$L_n^\alpha$` with Hildebrand's weight
        /// `$\Gamma(n+\alpha+1)/(n!\,x_k\,L_n^{\alpha\prime}(x_k)^2)$`. Unscaled weights, which
        /// reach `$e^{-x_k}$` at the outer nodes. `$L_{n-1}$` at the largest root grows like
        /// `$e^{x/2}$`, which bounds the rule near `n = 170` in `f64` and `n = 20` in `f32`.
        fn gauss_laguerre(self, alpha: Self, n: u32) -> (Self, Self);

        /// Computes the Pochhammer symbol `$(z)_m = \dfrac{\Gamma(z+m)}{\Gamma(z)}$`.
        ///
        /// Combinatorics calls this the **rising factorial**, and for a non-negative integer
        /// `m` it is exactly the ascending product `$z(z+1)\cdots(z+m-1)$`. The name here is
        /// the special-function one because the function is not restricted to integers: `m`
        /// is any real, which is what the hypergeometric series need and what "factorial"
        /// would misdescribe.
        ///
        /// Note that the notation `$(z)_m$` is **ambiguous in the literature**: it means the
        /// rising factorial in special functions and the _falling_ factorial through much of
        /// combinatorics and statistics. This function is the rising one. The falling
        /// factorial is `pochhammer(z - n + 1, n)`, and the two are related by
        /// `$z^{(\bar n)} = (-1)^n (-z)^{(\underline n)}$`. Neither is shipped separately,
        /// being an argument transform away.
        ///
        /// # Accuracy
        ///
        /// The obvious spelling `exp(lgamma(z+m) - lgamma(z))` cancels catastrophically
        /// whenever `m` is small beside `z`: at `z = 1e8, m = 1e-4` it has **no correct
        /// digits**. This does not use it.
        ///
        /// At `Average` precision and above (which includes the default policy), integer `m`
        /// up to 20 in absolute value takes an exact product, 0.00 ulp median and 4.2 worst.
        /// That path also covers negative `z` and returns exact zeros at the poles: `$(-2)_3$`
        /// is 0.
        ///
        /// Below `Average` it is compiled out and integer `m` goes through the Stirling
        /// difference like anything else, which measures 4.2 ulp median and 172 worst. The
        /// difference that shows is the exactness rather than the ulp count: `$(3)_1$` comes
        /// back as `3.0000000000000018` there, and `$(200)_2$` as `40200.00000000002`.
        ///
        /// Any other `m` with `z` and `z+m` both positive takes a Stirling difference
        /// arranged so nothing large is ever subtracted from anything large. Its error is the
        /// floor for anything exponentiating a logarithm, tracking
        /// `$|\ln (z)_m|\cdot\epsilon$`. Over 6924 measured points with `z` in `[0.1, 8.9]`
        /// that is a median of 2.6 ulp and a 99th percentile of 25. Individual points scale
        /// with the result's own logarithm, reaching 259 ulp where the value is near `1e163`,
        /// and falling to nothing as the result approaches 1.
        ///
        /// A non-integer `m` with `z` or `z+m` non-positive (a ratio taken across Gamma's
        /// poles) has no cheap rearrangement and does fall back to the logarithmic form,
        /// inheriting its cancellation.
        #[doc(alias = "poch")]
        #[doc(alias = "rising_factorial")]
        fn pochhammer(self, m: Self) -> Self;

        /// Computes the Jacobi elliptic functions `$(\mathrm{sn}, \mathrm{cn}, \mathrm{dn})$`
        /// at argument `self` and modulus `k`, all three from one evaluation.
        ///
        /// All three are made from a single angle, the **amplitude**
        /// `$\varphi = \mathrm{am}(u, k)$`, defined by `$F(\varphi, k) = u$`, so this function
        /// inverts the incomplete integral of the first kind that
        /// [`ellint`](SpecialMath::ellint) evaluates:
        ///
        /// ```math
        /// \mathrm{sn}(u, k) = \sin\varphi, \qquad
        /// \mathrm{cn}(u, k) = \cos\varphi, \qquad
        /// \mathrm{dn}(u, k) = \sqrt{1 - k^2 \sin^2\varphi}
        /// ```
        ///
        /// Hence their names: sine amplitude, cosine amplitude and delta amplitude. At
        /// `k = 0` the amplitude is `u` and they collapse to `$(\sin u, \cos u, 1)$`. At
        /// `k = 1` they stop being periodic and become
        /// `$(\tanh u, \operatorname{sech} u, \operatorname{sech} u)$`.
        ///
        /// # Why one function and not three
        ///
        /// The triple is closed under differentiation in `u`, each derivative a product
        /// of the other two:
        ///
        /// ```math
        /// \frac{d\,\mathrm{sn}}{du} = \mathrm{cn}\,\mathrm{dn}, \qquad
        /// \frac{d\,\mathrm{cn}}{du} = -\mathrm{sn}\,\mathrm{dn}, \qquad
        /// \frac{d\,\mathrm{dn}}{du} = -k^2\,\mathrm{sn}\,\mathrm{cn}
        /// ```
        ///
        /// so they are one object the way `$(\sin, \cos)$` are, and
        /// [`Dual`](https://docs.rs/thermite-dual) differentiates them without touching the
        /// iteration underneath. It also costs nothing to return all three: they share the
        /// entire computation, and only the last few operations differ.
        ///
        /// The other nine Jacobi functions in Glaisher's notation (`ns`, `nc`, `nd`, `sc`,
        /// `sd`, `cs`, `cd`, `ds`, `dc`) are reciprocals and ratios of these three, so this
        /// gives all twelve.
        ///
        /// # Domain and accuracy
        ///
        /// Only `$k^2$` enters, so the sign of `k` does not matter. `|k| > 1` is out of
        /// domain and gives NaN. Worst absolute error measured against mpmath at 40 digits
        /// over `|u| <= 8` and `k` in `[0, 1)` is 8.3 eps for `sn`, 4.1 for `cn` and 3.8 for
        /// `dn`. Absolute is the meaningful metric: all three are bounded by 1 and all three
        /// have zeros, so relative accuracy at a zero depends on how well that zero's
        /// location is known, exactly as for `sin`. For the same reason accuracy falls off
        /// slowly with `|u|`, that being the argument of the single trigonometric call
        /// inside.
        #[doc(alias = "sn")]
        #[doc(alias = "cn")]
        #[doc(alias = "dn")]
        #[doc(alias = "ellipj")]
        #[doc(alias = "sncndn")]
        fn jacobi_elliptic(self, k: Self) -> (Self, Self, Self);

        /// Computes the arithmetic-geometric mean `$\mathrm{AGM}(a, b)$` of two non-negative
        /// arguments.
        ///
        /// Iterating `$a \mapsto (a + b)/2$` against `$b \mapsto \sqrt{ab}$` drives the two
        /// sequences to a common limit, quadratically: the pair closes to within a factor of
        /// a few in a handful of passes from any starting ratio, and the correct digits then
        /// double per pass. The loop is branchless and costs one `sqrt` per iteration, with no
        /// transcendentals anywhere, which is why it is also the engine behind the complete
        /// elliptic integrals, `$K(k) = \pi / (2\,\mathrm{AGM}(1, k'))$`, reached through
        /// [`ellint`](SpecialMath::ellint) rather than by calling this directly.
        ///
        /// Symmetric in its arguments and homogeneous, `$\mathrm{AGM}(ca, cb) =
        /// c\,\mathrm{AGM}(a, b)$`. `AGM(a, 0)` is `0` and `AGM(inf, b)` is `inf`. A negative
        /// argument is outside the domain (the geometric mean's sign becomes ambiguous after
        /// the first pass) and returns NaN under overflow checking, as does a zero paired with
        /// an infinity.
        ///
        /// The geometric mean is formed as one product, so two arguments both above
        /// `$\sqrt{\text{MAX}}$` (about 1.3e154 in f64, 1.8e19 in f32) overflow to infinity
        /// even where the mean is representable. Scale both by a common power of two first if
        /// that range matters. Homogeneity makes it exact.
        fn agm(self, other: Self) -> Self;

        /// Computes the Langevin function `$L(x) = \coth x - \frac{1}{x}$`.
        ///
        /// Odd, strictly increasing, `L(0) = 0`, `L'(0) = 1/3`, `L(x) -> 1` as `x -> ∞`.
        /// This is the mean resultant length `$A_3(\kappa)$` of a von Mises-Fisher
        /// distribution on the sphere, and the freely-jointed-chain force-extension law
        /// in polymer physics.
        ///
        /// Evaluated as an odd minimax polynomial for `|x| <= 2` (the direct form
        /// `coth x - 1/x` cancels catastrophically there, losing `3u/x^2`), and as
        /// `1 - 1/x + 2/(e^{2x} - 1)` beyond. Both branches are accurate to a few ulp
        /// at every precision policy. The policy mainly selects the `exp`.
        ///
        /// To also obtain the derivative `L'(x)`, use
        /// [`langevin_d`](crate::RealPrimalMath::langevin_d).
        fn langevin(self) -> Self;

        /// Computes the inverse Langevin function `$L^{-1}(y)$` for `|y| < 1`.
        ///
        /// Odd, with a simple pole at `y = 1`: `L^-1(y) ~ 1/(1-y)`. `|y| = 1` returns
        /// `±∞`, and `|y| > 1` returns NaN under overflow checking (an unspecified
        /// value otherwise). Its condition number is `1/(1-y)`, so near the pole the
        /// result cannot be more accurate than that, however exact the arithmetic. A
        /// consumer that knows `1 - y` should form it before rounding.
        ///
        /// A rational seed (the same family as Cohen's Pade approximant, which the vMF
        /// literature knows as the Banerjee et al. concentration estimator) is refined by
        /// Newton (f32) or Halley (f64) steps whose count follows the precision policy:
        ///
        /// | precision | steps | relative error |
        /// |---|---|---|
        /// | `Worst` | 0 | ~2e-5 |
        /// | `Medium`, `Average`, `Best` | 1 | full (a few ulp) |
        /// | `Reference` | 2 | full |
        fn inv_langevin(self) -> Self;

        /// Computes `1 - L(x)`, the complement of the [Langevin function](RealSpecialMath::langevin),
        /// accurately where `L(x)` is within rounding of 1.
        ///
        /// `1 - L(x) ~ 1/x`, so once `x > 1/u` (sharpness ~1e7 in f32, ~1e16 in f64)
        /// `langevin(x)` rounds to exactly 1 and its complement is gone. This returns it
        /// to full relative precision at any `x`, from the same intermediates. Same cost
        /// as `langevin`. Negative `x` gives `1 + L(|x|)`.
        ///
        /// Pairs with [`inv_langevin_1m`](RealSpecialMath::inv_langevin_1m): the vMF
        /// convolution `kappa' = L^-1(L(k1) L(k2))` should be formed as
        /// `inv_langevin_1m(a + b - a*b)` with `a = langevin_1m(k1)`, `b = langevin_1m(k2)`,
        /// which is cancellation-free at every sharpness.
        fn langevin_1m(self) -> Self;

        /// Computes `L^-1(1 - t)` from the complement `t` directly.
        ///
        /// The [inverse Langevin function](RealSpecialMath::inv_langevin) has a pole at
        /// `y = 1` and a condition number of `1/(1-y)`, so a caller that knows `1 - y`
        /// (see [`langevin_1m`](RealSpecialMath::langevin_1m)) should pass it here rather
        /// than form `y` and lose its low digits: this entry point works in `t` throughout
        /// and is accurate to a few ulp at any sharpness. `t = 0` returns `+∞`, `t > 1`
        /// gives the negative branch, and `t < 0` is out of the domain (NaN under
        /// overflow checking). Same cost as `inv_langevin`.
        fn inv_langevin_1m(self) -> Self;

        /// GELU activation function, defined as `$\tfrac{1}{2} x \left(1 + \operatorname{erf}\!\left(\frac{\alpha x}{\sqrt{2}}\right)\right)$`,
        /// where `alpha` helps control the shape of the curve. The standard GELU function
        /// is recovered when `alpha` is 1.
        ///
        /// For f32 vectors, this remains decently accurate even with the `Medium` and `Worst` precision policies,
        /// thanks to good `erf` implementations at the various precision levels. See `erf` for more details.
        ///
        /// To also obtain the derivative with respect to `x` (which shares most of the computation), use
        /// [`gelu_d`](crate::RealPrimalMath::gelu_d).
        fn gelu(self, alpha: Self) -> Self;

        /// Swish activation function, defined as `$x\,\sigma(\beta x) = \frac{x}{1 + e^{-\beta x}}$`,
        /// where `beta` controls the sharpness of the gate. The standard Swish/SiLU function
        /// is recovered when `beta` is 1. As `beta -> 0`, the output approaches `x/2` (half-identity);
        /// as `beta -> inf`, Swish approaches ReLU.
        ///
        /// To also obtain the derivative with respect to `x`, use
        /// [`swish_d`](crate::RealPrimalMath::swish_d).
        fn swish(self, beta: Self) -> Self;

        /// Computes the algebraic sigmoid function, defined as `$\frac{x}{(1 + |x|^N)^{1/N}}$`, where
        /// `N` is a positive integer parameter that controls the steepness of the curve.
        ///
        /// This also has the unique behavior where for `N=0`, the function is just the identity function,
        /// and for `N=1` it is the [softsign function](https://en.wikipedia.org/wiki/Activation_function#Softsign).
        ///
        /// **Note**: This function uses `$|x|^N$` (the real absolute value), so it is non-holomorphic
        /// and only meaningful for real-valued inputs.
        ///
        /// To also obtain the derivative with respect to `x`, use
        /// [`algebraic_sigmoid_d`](crate::RealPrimalMath::algebraic_sigmoid_d).
        fn algebraic_sigmoid_n<const N: usize>(self) -> Self;

        /// The algebraic sigmoid for a degree known only at runtime. The runtime twin of
        /// [`algebraic_sigmoid_n`](RealSpecialMath::algebraic_sigmoid_n), same arithmetic.
        fn algebraic_sigmoid(self, n: u32) -> Self;

        /// Algebraic analogue of the [Swish](https://en.wikipedia.org/wiki/Swish_function) activation,
        /// defined as `$x\left(\frac{1}{2} + \frac{x}{2\sqrt{1 + x^2}}\right)$`. Equivalent to gating `x` by
        /// `(1 + algebraic_sigmoid_n::<2>(x)) / 2`, the `[0, 1]`-rescaled `N=2` algebraic sigmoid.
        ///
        /// Like standard Swish/SiLU, this is smooth and non-monotonic (it dips slightly below zero
        /// for moderately negative `x` before rising) and shares the same asymptotes (`f(x) -> x` as
        /// `x -> ∞`, `f(x) -> 0` as `x -> -∞`). Unlike Swish, it requires no `exp` or `log`, which
        /// is substantially cheaper on hardware without fast transcendentals.
        ///
        /// To also obtain the derivative with respect to `x` (which shares most of the underlying
        /// computation, notably `$1/\sqrt{1 + x^2}$`), use
        /// [`algebraic_swish_d`](crate::RealPrimalMath::algebraic_swish_d).
        ///
        /// # Historical note
        ///
        /// Algebraic gating functions of this form are effectively unknown in modern deep learning,
        /// which standardized on `exp`-based activations (sigmoid, Swish/SiLU, GELU) once GPUs made
        /// `exp` essentially free, a single-cycle special-function-unit op on most modern hardware.
        /// On CPUs the calculus is different: a vectorized `exp` still costs ~20+ cycles even with
        /// good polynomial approximations, while `sqrt`/`rsqrt` are cheap hardware ops (often
        /// approximated in 4-7 cycles). For CPU-side inference, training on CPU, or embedded targets
        /// without a transcendental SFU, this remains a competitive Swish-shaped activation at a
        /// fraction of the cost.
        fn algebraic_swish(self) -> Self;

        /// Computes the natural log of the Gamma function (`$\ln|\Gamma(x)|$`) for any real input, for each value in a vector,
        /// and returns the sign of the Gamma function from before the absolute value was taken.
        fn lgamma_r(self) -> (Self, Self);

        /// Computes the definite integral of the Gaussian function from `x0` to `x1`, with amplitude `a` and standard deviation `c`.
        /// This is more efficient than evaluating the indefinite integral at both limits and subtracting.
        ///
        /// The position `b` is assumed to be zero, so offset the limits accordingly for a non-zero position.
        fn gaussian_integral(x0: Self, x1: Self, a: Self, c: Self) -> Self;

        /// The [Box-Cox transform](https://en.wikipedia.org/wiki/Power_transform) of `x = self`
        /// with parameter `lambda`.
        ///
        /// ```math
        /// \mathrm{boxcox}(x, \lambda) = \begin{cases} \dfrac{x^\lambda - 1}{\lambda} & \lambda \ne 0 \\[6pt] \ln x & \lambda = 0\end{cases}
        /// ```
        ///
        /// The variance-stabilizing power transform of applied statistics: `$\lambda$` is fitted
        /// to make skewed data as close to normal as possible before a model sees it, and the
        /// family interpolates the transforms people otherwise pick by hand: `$\lambda = 1$`
        /// leaves the data alone up to a shift, `$1/2$` is a square root, `$0$` a logarithm,
        /// `$-1$` a reciprocal. A fixture of statistical software since Box and Cox introduced
        /// it in 1964.
        ///
        /// The two cases are one function: `$\ln x$` is the limit as `$\lambda \to 0$`, not a
        /// separate rule. Written out, `$(x^\lambda - 1)/\lambda$` is `$0/0$` there, and the
        /// trouble is not confined to the point. Computing `$x^\lambda$` and subtracting one
        /// cancels, so the naive form is already wrong in the fifth digit at
        /// `$\lambda = 10^{-12}$` and returns a flat zero by `$10^{-300}$`. That matters because
        /// a fitting routine searches `$\lambda$` near zero, which is the usual answer for
        /// right-skewed data.
        ///
        /// Evaluated as [`powf_m1`](thermite::math::TranscendentalMath::powf_m1)`(x, lambda)/lambda`,
        /// which forms `$x^\lambda - 1$` without ever forming `$x^\lambda$`, so there is nothing to
        /// cancel and **no series or crossover is needed**. Measured against a 60-digit oracle,
        /// it holds a few ulp from `$\lambda = 10^{-300}$` to `$\lambda = \pm 8$`. Only the exact
        /// `$\lambda = 0$` is selected apart.
        ///
        /// Domain is `$x > 0$`, and a negative `x` gives NaN. At `$x = 0$` the limits are taken:
        /// `$-1/\lambda$` for `$\lambda > 0$` and `$-\infty$` otherwise, which is the
        /// conventional choice. That needs no special case: `powf_m1(0, lambda)` is `$-1$`
        /// above zero and `$+\infty$` below, and the division does the rest.
        fn boxcox(self, lambda: Self) -> Self;

        /// The Box-Cox transform of `$1 + x$`, where `x = self`.
        ///
        /// ```math
        /// \mathrm{boxcox1p}(x, \lambda) = \begin{cases} \dfrac{(1 + x)^\lambda - 1}{\lambda} & \lambda \ne 0 \\[6pt] \ln (1 + x) & \lambda = 0\end{cases}
        /// ```
        ///
        /// The shifted form exists for the same reason [`ln_1p`](thermite::math::TranscendentalMath::ln_1p)
        /// does: when `x` is small, `$1 + x$` rounds it away, and every digit of the answer
        /// with it. Calling [`boxcox`](crate::RealSpecialMath::boxcox)`(1 + x, lambda)` loses `x` entirely once
        /// `$|x| < \varepsilon$`, where this returns `$\lambda x$` to full precision. Built on
        /// [`compound_m1`](thermite::math::TranscendentalMath::compound_m1), which forms
        /// `$(1 + x)^\lambda - 1$` without forming either `$1 + x$` or `$(1+x)^\lambda$`.
        ///
        /// This is also the kernel underneath [`yeo_johnson`](crate::RealSpecialMath::yeo_johnson), whose
        /// argument is data centered near zero by construction.
        ///
        /// Domain is `$x > -1$`; below that the result is NaN. At `$x = -1$` the limits are
        /// `$-1/\lambda$` for `$\lambda > 0$` and `$-\infty$` otherwise.
        fn boxcox_1p(self, lambda: Self) -> Self;

        /// The inverse [Box-Cox transform](https://en.wikipedia.org/wiki/Power_transform) of
        /// `y = self` with parameter `lambda`, undoing [`boxcox`](crate::RealSpecialMath::boxcox).
        ///
        /// ```math
        /// \mathrm{boxcox}^{-1}(y, \lambda) = \begin{cases} (\lambda y + 1)^{1/\lambda} & \lambda \ne 0 \\[6pt] e^y & \lambda = 0\end{cases}
        /// ```
        ///
        /// Wanted by anyone who uses the forward transform: a model fitted on transformed
        /// data predicts in transformed units, and the prediction has to come back.
        ///
        /// Evaluated as `$\exp\!\left(\ln(1 + \lambda y)/\lambda\right)$` rather than as a
        /// literal power, which is not merely a rearrangement. The whole
        /// point of [`boxcox`](crate::RealSpecialMath::boxcox) is that it stays accurate as `$\lambda \to 0$`,
        /// and `$\lambda$` fitted near zero is the common case. There `$\lambda y$` is tiny,
        /// so forming `$\lambda y + 1$` and raising it to the power `$1/\lambda$` throws away
        /// exactly the digits the forward transform took care to keep. Through `ln_1p` the
        /// exponent tends smoothly to `y`, so the `$\lambda = 0$` case is the limit rather
        /// than a discontinuity, and only the exact zero is selected apart.
        ///
        /// The range of the forward transform is `$\lambda y + 1 > 0$`. Outside it the result
        /// is NaN, and on the boundary it is `$0$` for `$\lambda > 0$` and `$+\infty$` below.
        fn inv_boxcox(self, lambda: Self) -> Self;

        /// The inverse of [`boxcox_1p`](crate::RealSpecialMath::boxcox_1p).
        ///
        /// ```math
        /// \mathrm{boxcox1p}^{-1}(y, \lambda) = \begin{cases} (\lambda y + 1)^{1/\lambda} - 1 & \lambda \ne 0 \\[6pt] e^y - 1 & \lambda = 0\end{cases}
        /// ```
        ///
        /// The same exponent as [`inv_boxcox`](crate::RealSpecialMath::inv_boxcox) with `expm1` outside it
        /// instead of `exp`, so a result near zero keeps its relative accuracy, which, this
        /// being the inverse of a transform applied to data centered near zero, is the
        /// ordinary case rather than an edge one. Also the kernel underneath
        /// [`inv_yeo_johnson`](crate::RealSpecialMath::inv_yeo_johnson).
        fn inv_boxcox_1p(self, lambda: Self) -> Self;

        /// The [Yeo-Johnson transform](https://en.wikipedia.org/wiki/Power_transform) of
        /// `y = self` with parameter `lambda`.
        ///
        /// ```math
        /// \psi(y, \lambda) = \begin{cases}
        ///   \dfrac{(y + 1)^\lambda - 1}{\lambda} & y \ge 0,\ \lambda \ne 0 \\[6pt]
        ///   \ln(y + 1) & y \ge 0,\ \lambda = 0 \\[6pt]
        ///   -\dfrac{(1 - y)^{2 - \lambda} - 1}{2 - \lambda} & y < 0,\ \lambda \ne 2 \\[6pt]
        ///   -\ln(1 - y) & y < 0,\ \lambda = 2
        /// \end{cases}
        /// ```
        ///
        /// Box-Cox's sibling, and the one that gets used more, since it is defined on the whole
        /// real line rather than on `$x > 0$`. Same job (fit `$\lambda$` by maximum likelihood
        /// to make skewed data as close to normal as a power transform can) without the "add a
        /// constant to make everything positive first" step, which is an arbitrary choice that
        /// changes the fitted `$\lambda$`. Introduced by Yeo and Johnson in 2000.
        ///
        /// # One kernel, not four
        ///
        /// The four cases are one function seen twice. The `$y < 0$` branch is the `$y \ge 0$`
        /// branch applied to `$|y|$` with `$\lambda$` reflected to `$2 - \lambda$` and the
        /// result negated, which is what makes `$\psi$` smooth in `$\lambda$` across `$y = 0$`
        /// in the first place. Folding the sign out first therefore collapses the two
        /// logarithmic special cases (`$\lambda = 0$` above zero, `$\lambda = 2$` below) into
        /// the single seam that [`boxcox_1p`](crate::RealSpecialMath::boxcox_1p) already handles, and the whole
        /// transform is `$\pm\,\mathrm{boxcox1p}(|y|, \lambda\ \mathrm{or}\ 2 - \lambda)$`.
        ///
        /// That the kernel is the `1p` form and not [`boxcox`](crate::RealSpecialMath::boxcox) applied to
        /// `$1 + |y|$` matters here more than anywhere else. `$\psi(y, \lambda) \approx y$`
        /// near the origin for every `$\lambda$`, and the origin is where the data is: the
        /// transform's reason for existing is samples that straddle zero. Forming `$1 + |y|$`
        /// would round away everything below `$\varepsilon$` and return a flat zero there.
        ///
        /// The value is finite for every finite `y`, so there is nothing to guard: the two
        /// domain edges of the kernel are at `$|y| = -1$`, which the fold never reaches.
        fn yeo_johnson(self, lambda: Self) -> Self;

        /// The inverse [Yeo-Johnson transform](https://en.wikipedia.org/wiki/Power_transform),
        /// undoing [`yeo_johnson`](crate::RealSpecialMath::yeo_johnson).
        ///
        /// ```math
        /// \psi^{-1}(z, \lambda) = \begin{cases}
        ///   (\lambda z + 1)^{1/\lambda} - 1 & z \ge 0,\ \lambda \ne 0 \\[6pt]
        ///   e^z - 1 & z \ge 0,\ \lambda = 0 \\[6pt]
        ///   1 - \left((\lambda - 2) z + 1\right)^{1/(2 - \lambda)} & z < 0,\ \lambda \ne 2 \\[6pt]
        ///   1 - e^{-z} & z < 0,\ \lambda = 2
        /// \end{cases}
        /// ```
        ///
        /// The same sign fold as the forward transform, over
        /// [`inv_boxcox_1p`](crate::RealSpecialMath::inv_boxcox_1p). `$\psi$` is increasing and fixes the origin,
        /// so the branch on the way back is the sign of the transformed value, which is the
        /// sign of `y`.
        ///
        /// Unlike the forward direction this one has a range to respect: for `$\lambda > 0$`
        /// the transform's image is bounded below by `$-1/\lambda$`, and a `z` past that came
        /// from no `y`. Such an input gives NaN rather than a plausible-looking number.
        fn inv_yeo_johnson(self, lambda: Self) -> Self;

        /// Evaluates **all** real spherical harmonics through degree `L` at the unit
        /// direction `(x, y, z)`, into `out[l * (l + 1) + m]` for `m` in `-l..=l`.
        ///
        /// Orthonormal real harmonics. Evaluation is pure polynomial arithmetic:
        /// no trigonometry, no division, `O(L^2)` FMAs total, exact zeros for every
        /// `m != 0` harmonic at the poles, fully unrolled at compile time for each
        /// `L` up to [`MAX_SH_DEGREE`] (above that it takes the rolled general path,
        /// which is correct at any degree but roughly 10x slower).
        ///
        /// `CS` picks the phase convention. `false` gives the standard real-SH
        /// tables (`$Y_{11} = \sqrt{3/4\pi}\,x$`); `true` applies the Condon-Shortley
        /// `$(-1)^{|m|}$` phase, negating every odd-`|m|` harmonic to match Sloan's
        /// `SHEval` and the physics convention (`$Y_{11} = -\sqrt{3/4\pi}\,x$`). The
        /// choice is baked into a constant table, so neither costs an instruction,
        /// but mixing the two silently corrupts any projection/reconstruction
        /// round-trip, which is why it must be named.
        ///
        /// `N` must equal `(L + 1)^2` (compile-time checked). The direction is
        /// assumed unit-length, and nothing renormalizes. See
        /// [`sh_impl`](specialized::sh_impl) for the full convention, algorithm,
        /// and domain notes.
        ///
        /// ```
        /// use thermite::prelude::*;
        /// use thermite_special::RealSpecialMath;
        ///
        /// type V = Vector<f64>;
        /// let (x, y, z) = (V::splat(0.6), V::splat(0.0), V::splat(0.8));
        ///
        /// let mut sh = [V::ZERO; 9];
        /// V::spherical_harmonics::<2, 9, false>(x, y, z, &mut sh);
        /// // Y(1,1) = sqrt(3/4pi) * x
        /// assert!((sh[3].extract::<0>() - 0.48860251190292 * 0.6).abs() < 1e-14);
        ///
        /// // Condon-Shortley negates odd |m|, and agrees on even |m|.
        /// let mut cs = [V::ZERO; 9];
        /// V::spherical_harmonics::<2, 9, true>(x, y, z, &mut cs);
        /// assert_eq!(cs[3].extract::<0>(), -sh[3].extract::<0>());
        /// assert_eq!(cs[8].extract::<0>(), sh[8].extract::<0>());
        /// ```
        #[skip_dispatch] #[compose] fn spherical_harmonics<const L: usize, const N: usize, const CS: bool>(x: Self, y: Self, z: Self, out: &mut [Self; N]) -> ();

        /// Builds the runtime coefficient table that [`spherical_harmonics_with`](RealSpecialMath::spherical_harmonics_with)
        /// and [`spherical_harmonics_d_with`](RealPrimalMath::spherical_harmonics_d_with) evaluate.
        ///
        /// The table depends only on `L` and `CS`, never on the direction, so a caller
        /// sweeping many directions should build it once rather than calling the
        /// one-shot [`spherical_harmonics`](RealSpecialMath::spherical_harmonics)
        /// per direction. The phase is baked in here, which is why the evaluators take
        /// no `CS`.
        ///
        /// The table is typed by `Self::Primal`, the unaugmented value type: the
        /// recurrence coefficients are constants, so a `Dual`'s derivative parts and a
        /// `Complex`'s imaginary part would only store zeros. For plain vectors and
        /// `Compensated` the primal is `Self` and nothing changes. For `Dual` the table
        /// is a fraction of the size and its entries multiply as reals.
        ///
        /// ```
        /// use thermite::prelude::*;
        /// use thermite_special::{RealSpecialMath, ShTable};
        ///
        /// type V = Vector<f64>;
        /// const L: usize = 3;
        /// const N: usize = (L + 1) * (L + 1);
        ///
        /// let mut table = ShTable::<V, N>::zeroed();
        /// V::spherical_harmonics_table::<L, N, false>(&mut table);
        ///
        /// let mut sh = [V::ZERO; N];
        /// for &(x, y, z) in &[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)] {
        ///     V::spherical_harmonics_with::<L, N>(
        ///         &table, V::splat(x), V::splat(y), V::splat(z), &mut sh,
        ///     );
        /// }
        /// assert!((sh[1].extract::<0>() - 0.48860251190292).abs() < 1e-14);
        /// ```
        #[skip_dispatch] #[scalar_form((table: &mut ShTable<Self, N>) -> ())]
        fn spherical_harmonics_table<const L: usize, const N: usize, const CS: bool>(table: &mut ShTable<<Self as PrimalProjection>::Primal, N>) -> ();

        /// Evaluates all harmonics through degree `L` from a prebuilt table.
        ///
        /// The table holds `Self::Primal` coefficients. See
        /// [`spherical_harmonics_table`](RealSpecialMath::spherical_harmonics_table)
        /// for how to build it and why, and
        /// [`spherical_harmonics`](RealSpecialMath::spherical_harmonics) for the
        /// conventions and layout.
        #[skip_dispatch] #[scalar_form((table: &ShTable<Self, N>, x: Self, y: Self, z: Self, out: &mut [Self; N]) -> ())]
        fn spherical_harmonics_with<const L: usize, const N: usize>(table: &ShTable<<Self as PrimalProjection>::Primal, N>, x: Self, y: Self, z: Self, out: &mut [Self; N]) -> ();

    }

    /// "Primal" special functions: the value-and-derivative (`_d`) forms of the activation
    /// functions, returning `(value, derivative)` together.
    ///
    /// These exist for *single-value* real numbers (`f32`, `f64`, `Compensated`, ...) where the
    /// analytic derivative is a useful, cheaply-shared byproduct of the value. They are **not**
    /// implemented for derivative-carrying numbers such as `Dual`: an automatic-differentiation
    /// type already produces the derivative from the plain value form (e.g. [`gelu`](RealSpecialMath::gelu)),
    /// so the bundled `_d` derivative would be redundant work at the wrong level of abstraction.
    ///
    /// Each `*_d` method mirrors the like-named value-only function in [`SpecialMath`] /
    /// [`RealSpecialMath`], returning that same value as the first tuple element.
    #[diagnostic::on_unimplemented(
        message = "`{Self}` does not provide value-and-derivative special math (`softplus_d`, `gelu_d`, `spherical_harmonics_d`, ...)",
        note = "`RealPrimalMath` builds on `RealSpecialMath` and is implemented only for primal real vectors (plain float vectors and `Compensated`), never for `Dual` or `Complex`, which get their derivatives from the value form instead. A bare `f32`/`f64` does not qualify either. Wrap it in `Vector::<f32>::splat(x)`, or use `ScalarSpecialMath`."
    )]
    pub trait RealPrimalMath: RealSpecialMathWithPolicy + PrimalMathWithPolicy {
        /// [`spherical_harmonics`](RealSpecialMath::spherical_harmonics) plus the
        /// ambient Cartesian gradient of every harmonic, into `ddx`/`ddy`/`ddz`.
        ///
        /// Lives on [`RealPrimalMath`] rather than [`RealSpecialMath`], so `Dual` does
        /// not get it, and should not want it. If you need `$\partial/\partial(x,y,z)$`,
        /// call this directly rather than evaluating
        /// [`spherical_harmonics`](RealSpecialMath::spherical_harmonics) on a
        /// `Dual<V, 3>` seeded with an identity Jacobian: this shares the recurrence
        /// between the value and all three gradients, whereas dual arithmetic carries a
        /// derivative through every operation and costs roughly twice as much.
        ///
        /// `Dual` earns its keep on the _value_ form instead, where `(x, y, z)` are
        /// themselves functions of upstream parameters and the chain rule has real work
        /// to do. Even there, going the other way (contracting these three gradients
        /// against an upstream Jacobian) loses: spherical harmonics cost about two
        /// operations per harmonic to evaluate but three per harmonic per parameter to
        /// contract, because one recurrence produces the whole basis.
        ///
        /// The derivatives are those of the polynomial form at the given (unit)
        /// input. Project out the radial component (`g - (g . n) n`) for the
        /// tangential gradient. Shares all recurrence work with the value pass, since
        /// the gradients come from tabulated norm ratios, not new recurrences.
        #[skip_dispatch] #[compose] fn spherical_harmonics_d<const L: usize, const N: usize, const CS: bool>(x: Self, y: Self, z: Self, out: &mut [Self; N], ddx: &mut [Self; N], ddy: &mut [Self; N], ddz: &mut [Self; N]) -> ();

        /// [`zernike_basis`](SpecialMath::zernike_basis) plus `$\partial Z_n^m/\partial x$`
        /// and `$\partial Z_n^m/\partial y$` for every mode, in the same ANSI layout.
        ///
        /// This is what a Shack-Hartmann wavefront reconstruction integrates against. The
        /// sensor measures local wavefront *slopes*, not the wavefront itself, so the fit
        /// matrix is built from the gradient basis and the value basis never appears in it.
        ///
        /// Lives on [`RealPrimalMath`] rather than [`SpecialMath`] for the same reason
        /// [`spherical_harmonics_d`](RealPrimalMath::spherical_harmonics_d) does: `Dual`
        /// should not get it and should not want it. Seeding a `Dual<V, 2>` and calling the
        /// value form carries two derivative components through every operation of the whole
        /// ladder, where this differentiates only the two factors that depend on the point
        /// and shares the radial recurrence between the value and both gradients.
        ///
        /// The gradient is finite everywhere, including the pupil centre. That is the
        /// practical dividend of the Cartesian formulation: the polar
        /// `$\partial_\theta Z/\rho$` is singular there, and hand-rolled polar
        /// implementations guard the origin with a special case.
        ///
        /// `N` must equal `(L+1)(L+2)/2`, and `NORM` is as on
        /// [`zernike_basis`](SpecialMath::zernike_basis). All three output buffers are
        /// written in full.
        #[skip_dispatch] #[compose] fn zernike_basis_d<const L: usize, const NORM: u8, const N: usize>(x: Self, y: Self, out: &mut [Self; N], ddx: &mut [Self; N], ddy: &mut [Self; N]) -> ();

        /// [`spherical_harmonics_with`](RealSpecialMath::spherical_harmonics_with) plus
        /// the ambient Cartesian gradients, from a prebuilt table.
        #[skip_dispatch] #[compose] fn spherical_harmonics_d_with<const L: usize, const N: usize>(table: &ShTable<Self, N>, x: Self, y: Self, z: Self, out: &mut [Self; N], ddx: &mut [Self; N], ddy: &mut [Self; N], ddz: &mut [Self; N]) -> ();

        /// [`softplus`](SpecialMath::softplus) together with its derivative w.r.t. `x`
        /// (the logistic sigmoid `$\sigma(kx)$`).
        fn softplus_d(self, k: Self, rcp_k: Self) -> (Self, Self);

        /// [`gelu`](RealSpecialMath::gelu) together with its derivative w.r.t. `x`.
        fn gelu_d(self, alpha: Self) -> (Self, Self);

        /// [`swish`](RealSpecialMath::swish) together with its derivative w.r.t. `x`.
        fn swish_d(self, beta: Self) -> (Self, Self);

        /// [`algebraic_sigmoid`](RealSpecialMath::algebraic_sigmoid) together with its derivative w.r.t. `x`.
        fn algebraic_sigmoid_d_n<const N: usize>(self) -> (Self, Self);

        /// [`algebraic_sigmoid_d_n`](RealPrimalMath::algebraic_sigmoid_d_n) for a degree known
        /// only at runtime.
        fn algebraic_sigmoid_d(self, n: u32) -> (Self, Self);

        /// [`algebraic_swish`](RealSpecialMath::algebraic_swish) together with its derivative w.r.t. `x`.
        fn algebraic_swish_d(self) -> (Self, Self);

        /// [`langevin`](RealSpecialMath::langevin) together with its derivative
        /// `$L'(x) = \frac{1}{x^2} - \operatorname{csch}^2 x$`.
        ///
        /// The derivative shares every intermediate with the value, so this costs a
        /// handful of arithmetic ops over `langevin` alone.
        fn langevin_d(self) -> (Self, Self);
    }
}
