//! Math kernels for [`Complex`].
//!
//! Implements the `Specialized*Math` traits. Complex vectors thereby get
//! [`CoreMath`](thermite::math::CoreMath),
//! [`TranscendentalMath`](thermite::math::TranscendentalMath) and
//! [`SpatialMath`](thermite::math::SpatialMath) (with their `_p` policy forms)
//! from the blanket impls in `thermite::math`, plus
//! [`SpecializedComplexMath`] for the operations
//! over the real part.
//!
//! Each kernel evaluates its complex function through the *inner* vector's policy
//! math (`sin_cos_p`, `exp_p`, `atan2_p`, ...), carrying the caller's policy all the
//! way down. Anything expressible by composition (`sin`, `cos`, `tan_pi`,
//! `powi`, `sqrt1pm1`, `compound`, ...) comes from the trait defaults, which are
//! already correct over C.
//!
//! [`RealMath`](thermite::math::RealMath) is not implemented. See the note at the
//! bottom of this file.

use thermite::math::PrimalProjection;
use thermite::math::policy::{DefaultPolicy, Policy, PrecisionPolicy};
use thermite::math::specialized::{SpecializedCoreMath, SpecializedSpatialMath, SpecializedTranscendentalMath};
use thermite::prelude::*;

use self::specialized::{ComplexVector, SpecializedComplexMath};
use crate::Complex;
use crate::vector::RealFloatVector;

thermite::math_traits! {
    #![thermite(thermite)]
    #![surface(__complex_math_surface)]

    /// Operations whose result is real (modulus, argument, polar form) or whose
    /// argument is (a real power, base, or logarithm base), which the `Self -> Self`
    /// core families cannot express.
    ///
    /// The purely complex operations (`exp`, `ln`, `sin`, `sqrt`, `powf`, ...) fit
    /// the core families and come from
    /// [`TranscendentalMath`](thermite::math::TranscendentalMath) as they do for
    /// any other vector.
    #[element(FloatElement)]
    pub trait ComplexMath: ComplexVector {
        /// The modulus `$|z|$`, as a real value.
        ///
        /// Uses `hypot`, so it does not overflow for large components the way
        /// `sqrt(norm_sqr())` would.
        fn norm(self) -> Self::Real;

        /// The principal argument `arg(z)`, in `(-pi, pi]`, as a real value.
        fn arg(self) -> Self::Real;

        /// Converts to polar form `(r, theta)`, such that `self == r * exp(i*theta)`.
        fn to_polar(self) -> (Self::Real, Self::Real);

        /// Builds a complex number from a polar representation `r * exp(i*theta)`.
        fn from_polar(r: Self::Real, theta: Self::Real) -> Self;

        /// The unit complex number at angle `theta`, `$e^{i\theta} = \cos\theta + i\sin\theta$`.
        ///
        /// The `r = 1` case of [`from_polar`](ComplexMath::from_polar), and cheaper than
        /// doing it that way, it's just one `sin_cos` and nothing else. This is the
        /// rotation/phasor constructor, for twiddle factors, unit-circle sampling, and the
        /// angle term of an [`expf`](ComplexMath::expf).
        fn from_angle(theta: Self::Real) -> Self;

        /// Raises `self` to a real power.
        ///
        /// The complex-exponent form is [`powf`](thermite::math::TranscendentalMath::powf).
        fn powfr(self, e: Self::Real) -> Self;

        /// Raises a real base to the complex power `self`.
        fn expf(self, base: Self::Real) -> Self;

        /// The logarithm of `self` in an arbitrary real base.
        ///
        /// The complex-base form is [`log`](thermite::math::TranscendentalMath::log).
        fn logr(self, base: Self::Real) -> Self;

        /// `1/self`, scaling by the modulus and not its square.
        ///
        /// Survives the magnitudes where [`inv`](ComplexVector::inv) would have
        /// `norm_sqr()` overflow to infinity or underflow to zero.
        fn finv(self) -> Self;

        /// `self/rhs`, scaling by the modulus and not its square.
        ///
        /// Survives the magnitudes where `/` would have `rhs.norm_sqr()` overflow
        /// to infinity or underflow to zero.
        fn fdiv(self, rhs: Self) -> Self;
    }
}

pub mod specialized;

#[cfg(feature = "special")]
pub mod special;

// --- SpecializedComplexMath: the real-valued and real-argument operations ---

impl<V: RealFloatVector> SpecializedComplexMath<Complex<V::Element>> for Complex<V> {
    #[inline(always)]
    fn norm<P: Policy>(self) -> V {
        self.re.hypot_p::<P>(self.im)
    }

    #[inline(always)]
    fn arg<P: Policy>(self) -> V {
        self.im.atan2_p::<P>(self.re)
    }

    #[inline(always)]
    fn from_polar<P: Policy>(r: V, theta: V) -> Self {
        // C99 takes `e^(-inf + iy)` to +-0 for every non-finite `y`, and the same rule
        // is what every caller here wants: once the modulus has collapsed to zero the
        // result is the origin whatever direction it was approached from, but
        // `r * cos(theta)` still reads `0 * NaN`. `z^(1 + i)` at z = 0 has a genuine
        // -inf angle (the spiral never settles) and comes out NaN without this.
        //
        // The `is_finite` test is what keeps it honest: a finite angle is left exactly
        // as it was, signed zeros included, so nothing well-defined moves.
        let theta = if const { P::POLICY.check_overflow } {
            theta.nz(r.cmp_eq(V::ZERO).bitandnot(theta.is_finite()))
        } else {
            theta
        };

        let (s, c) = theta.sin_cos_p::<P>();

        Self::new(r * c, r * s)
    }

    #[inline(always)]
    fn powfr<P: Policy>(self, e: V) -> Self {
        // z^y = (r e^(i t))^y = r^y e^(i t y)
        let (r, theta) = self.to_polar_p::<P>();

        Self::from_polar_p::<P>(r.powf_p::<P>(e), theta * e)
    }

    #[inline(always)]
    fn expf<P: Policy>(self, base: V) -> Self {
        // b^(a + ci) = b^a * e^(i c ln b)
        //
        // `ln b` only ever reaches the angle here, so it can be masked outright.
        let ln_b = finite_log_term::<P, V>(base.ln_p::<P>(), self.im);

        Self::from_polar_p::<P>(base.powf_p::<P>(self.re), self.im * ln_b)
    }

    #[inline(always)]
    fn logr<P: Policy>(self, base: V) -> Self {
        // log_b(z) = ln(z) / ln(b); one real reciprocal, then scale both components.
        let (r, theta) = self.to_polar_p::<P>();
        let d = base.ln_p::<P>().approx_reciprocal_p::<P>();

        Self::new(r.ln_p::<P>() * d, theta * d)
    }

    #[inline(always)]
    fn finv<P: Policy>(self) -> Self {
        // Scaling twice by 1/|z| keeps every intermediate in range, where
        // conj(z) / |z|^2 would overflow the square.
        //
        // Taking the reciprocal explicitly, rather than writing `(conj/n)/n`:
        // `Div<V>` is already a reciprocal-and-scale, so the quotient form spent
        // two divisions computing the same 1/|z| twice. `digamma` calls this in a
        // loop.
        let inv = self.norm_p::<P>().approx_reciprocal_p::<P>();

        self.conj() * inv * inv
    }
}

// --- SpecializedCoreMath ---

impl<V: RealFloatVector> PrimalProjection for Complex<V> {
    // A real coefficient's imaginary part is identically zero, so tables and
    // cached coefficients live in the real vector's primal, recursively.
    type Primal = V::Primal;

    #[inline(always)]
    fn from_primal(p: Self::Primal) -> Self {
        Self::new(V::from_primal(p), V::ZERO)
    }

    #[inline(always)]
    fn to_primal(self) -> Self::Primal {
        self.re.to_primal()
    }
}

impl<V: RealFloatVector> SpecializedCoreMath<Complex<V::Element>> for Complex<V> {
    /// Same shape as the `Dual` override: the multiply is a full complex one, but a
    /// real addend touches only the real part, where the default would add an
    /// explicit zero to the imaginary part on every Horner step.
    ///
    /// The addend is _fused_ into the real part's inner FMA rather than added after a
    /// complete complex multiply. Both spell `re*m.re - im*m.im + a`, but
    /// `mul_adde(re, m.re, nmul_adde(im, m.im, a))` is two FMAs where multiply-then-add
    /// is an FMA, a multiply and an add. That saves one instruction per Horner step,
    /// measured at 48 vs 60 vector ops over a 13-term complex polynomial
    /// (`bin/poly_n_primal_probe`). It also rounds once less.
    #[inline(always)]
    fn mul_add_primal<P: Policy>(self, m: Self, a: Self::Primal) -> Self {
        let re = self.re.mul_adde(m.re, self.im.nmul_adde(m.im, V::from_primal(a)));
        let im = self.re.mul_adde(m.im, self.im * m.re);

        Complex::new(re, im)
    }

    /// `$ab - cd$` over `$\mathbb{C}$`, as four real differences of products.
    ///
    /// Kahan's compensation only works when the multiply-add is a single rounding, which
    /// a complex multiply-add is not. So each half is a sum of two real differences,
    /// with the compensation on the inner real vector:
    ///
    /// ```math
    /// \Re = (a_r b_r - c_r d_r) + (c_i d_i - a_i b_i) \qquad
    /// \Im = (a_r b_i - c_i d_r) + (a_i b_r - c_r d_i)
    /// ```
    ///
    /// The pairing matters. `difference_of_products` is exact when its two products are
    /// equal, which is what makes `$ab - ba$` a true zero. For that to survive, each term
    /// pairs with the one it maps onto under `$c = b, d = a$` (`$a_r b_i$` with
    /// `$c_i d_r$`). Pairing the two `$ab$` terms together loses the zero on about 22% of
    /// inputs (200k random pairs).
    ///
    /// Against a 60-digit reference over 200k inputs, imaginary half: max 533 ulp, mean
    /// 0.414, against naive's 795 and 0.473.
    #[inline(always)]
    fn difference_of_products<P: Policy>(self, b: Self, c: Self, d: Self) -> Self {
        let a = self;

        Complex::new(
            a.re.difference_of_products_p::<P>(b.re, c.re, d.re)
                + c.im.difference_of_products_p::<P>(d.im, a.im, b.im),
            a.re.difference_of_products_p::<P>(b.im, c.im, d.re)
                + a.im.difference_of_products_p::<P>(b.re, c.re, d.im),
        )
    }

    /// `$ab + cd$` over `$\mathbb{C}$`, the companion to
    /// [`difference_of_products`](Self::difference_of_products) and the same argument.
    ///
    /// ```math
    /// \Re = (a_r b_r - c_i d_i) + (c_r d_r - a_i b_i) \qquad
    /// \Im = (a_r b_i + c_i d_r) + (c_r d_i + a_i b_r)
    /// ```
    ///
    /// The real half stays a pair of differences, since a complex product's real part is
    /// one. Mirror-paired against the degenerate case `$ab + (-b)a$`, so `$c = -b, d = a$`.
    ///
    /// Over 100k random inputs: real half max 257 ulp / mean 0.410, imaginary max 444 /
    /// mean 0.422, against naive's 8247 / 0.599 and 15698 / 0.673.
    #[inline(always)]
    fn sum_of_products<P: Policy>(self, b: Self, c: Self, d: Self) -> Self {
        let a = self;

        Complex::new(
            a.re.difference_of_products_p::<P>(b.re, c.im, d.im)
                + c.re.difference_of_products_p::<P>(d.re, a.im, b.im),
            a.re.sum_of_products_p::<P>(b.im, c.im, d.re) + c.re.sum_of_products_p::<P>(d.im, a.im, b.re),
        )
    }

    /// `P(z)/Q(z)`, evaluated directly or through `1/z` depending on which is better
    /// conditioned.
    ///
    /// Overridden only to change *which quantity* that decision is made on. The
    /// generic default tests `x.cmp_gt(ONE)`, which over C is the lexicographic order
    /// on `(re, im)`, so it keys off the real part alone and will happily evaluate
    /// the direct form at `z = 10^150 i`, overflowing, while reporting that `z` is
    /// "not greater than one". The condition that actually matters is `|z| > 1`.
    ///
    /// Both forms are the same rational function (`P_rev(1/z)/Q_rev(1/z)` differs from
    /// `P(z)/Q(z)` only by `z^(D-N)`, corrected below), so this is a conditioning fix,
    /// not a correctness one, except where the wrong choice overflows outright.
    #[inline(always)]
    fn poly_rational_n<P: Policy, const N: usize, const D: usize>(
        self,
        numerator: &[Complex<V::Element>; N],
        denominator: &[Complex<V::Element>; D],
    ) -> Self {
        let x = self;

        if const { P::POLICY.precision.le(thermite::math::policy::PrecisionPolicy::Average) } {
            let n = SpecializedCoreMath::poly_n::<P, N>(x, numerator);
            let d = SpecializedCoreMath::poly_n::<P, D>(x, denominator);

            return n.approx_div_p::<P>(d);
        }

        let invert = x.norm_sqr().cmp_gt(V::ONE);

        let mut n0 = Self::EMPTY;
        let mut d0 = Self::EMPTY;

        if const { P::POLICY.avoid_branching } || !invert.all() {
            n0 = SpecializedCoreMath::poly_n::<P, N>(x, numerator);
            d0 = SpecializedCoreMath::poly_n::<P, D>(x, denominator);
        }

        let mut z = Self::EMPTY;
        let mut n1 = Self::EMPTY;
        let mut d1 = Self::EMPTY;

        if const { P::POLICY.avoid_branching } || invert.any() {
            z = SpecializedCoreMath::approx_reciprocal::<P>(x);
            n1 = SpecializedCoreMath::poly_rev_n::<P, N>(z, numerator);
            d1 = SpecializedCoreMath::poly_rev_n::<P, D>(z, denominator);
        }

        let n = invert.select(n1, n0);
        let d = invert.select(d1, d0);

        let res = n.approx_div_p::<P>(d);

        // Same degree: the reversed form is the same value, nothing to undo.
        if const { N == D } {
            return res;
        }

        if const { P::POLICY.avoid_branching } || invert.any() {
            let (u, e) = if const { N < D } { (z, D - N) } else { (x, N - D) };

            return invert.select(res * SpecializedCoreMath::powi::<P>(u, e as i32), res);
        }

        res
    }

    // 1/sqrt(z) = conj(sqrt(z)) / |z|, since |sqrt(z)|^2 == |z|.
    #[inline(always)]
    fn inverse_sqrt<P: Policy>(self) -> Self {
        let s = self.sqrt();
        let inv = self.norm_p::<P>().approx_reciprocal_p::<P>();

        Complex::new(s.re * inv, -(s.im * inv))
    }
}

// --- SpecializedTranscendentalMath ---

/// `$iz = -\operatorname{Im} z + i\operatorname{Re} z$`: a component swap and a sign flip.
///
/// Worth a helper because the natural spelling is not free. `z.mul_add(Self::I, w)`
/// is a complex FMA (four inner FMAs) over a constant of zeros and ones, and IEEE
/// forbids folding `a*0.0 + b` to `b` (`a` may be infinite, and the zero has a sign),
/// so all four survive into the assembly.
#[inline(always)]
fn mul_i<V: RealFloatVector>(z: Complex<V>) -> Complex<V> {
    Complex::new(-z.im, z.re)
}

/// `$-iz = \operatorname{Im} z - i\operatorname{Re} z$`. See [`mul_i`].
#[inline(always)]
fn mul_neg_i<V: RealFloatVector>(z: Complex<V>) -> Complex<V> {
    Complex::new(z.im, -z.re)
}

/// Replaces the lanes where a hyperbolic quotient has overflowed with its limit.
///
/// `tan`/`tanh` divide by `cosh` plus a bounded term. Past `$|2x| \approx 710$` in
/// binary64 that denominator is infinite while the numerator's `sinh` is too, so the
/// quotient is `inf/inf`, giving `NaN` in the component that should have saturated and a
/// signed zero in the other. The function itself is perfectly well behaved there and
/// tends to a unit along one axis.
///
/// Keyed on the denominator rather than a magnitude threshold, so it fires exactly on
/// the lanes that lost the value and needs no per-element constant. `is_infinite`, not
/// `!is_finite`: a `NaN` argument must still produce `NaN`.
///
/// A no-op unless the policy sets
/// [`check_overflow`](thermite::math::policy::PolicyParameters::check_overflow).
#[inline(always)]
fn saturate<P: Policy, V: RealFloatVector>(res: Complex<V>, denom: V, limit: Complex<V>) -> Complex<V> {
    if const { !P::POLICY.check_overflow } {
        return res;
    }

    let lost = denom.is_infinite();

    if thermite::unlikely(lost.any()) {
        return lost.select(limit, res);
    }

    res
}

/// `$\operatorname{asinh}(p) = \ln(p + \sqrt{1 + p^2})$`, conditioned at both ends.
///
/// Serves `asin` too: `$(iz)^2 = -z^2$`, so `$\sqrt{1 - z^2}$` is this function's
/// `$\sqrt{1 + p^2}$` at `$p = iz$`, and `$\operatorname{asin}(z) = -i\operatorname{asinh}(iz)$`
/// exactly. One kernel, two functions.
///
/// The naive `$\ln(p + s)$` loses everything at both extremes:
///
/// - **Large `$|p|$`**: `$s \to \mp p$` and the sum cancels. At `$z = 10^8 i$` the
///   `asin` sum is `$-10^8 + 10^8$`, *exactly* zero, so `$\ln 0$` returned an
///   infinity where the true value is `19.11i`. The companion `$s - p$` is the other
///   root, and `$(p + s)(s - p) = s^2 - p^2 = 1$`, so it is both exact and the
///   well-conditioned one. `$|w| < 1$` tests which cancelled, and `$\ln w = -\ln w'$`
///   holds outright rather than up to `$2\pi i$`, the companion lying in the right
///   half-plane exactly when it is selected.
/// - **Small `$|p|$`**: `$w = 1 + O(p)$`, and forming that sum rounds away the very
///   `$p$` the answer consists of: `asinh(1e-8)` kept 8 of its 16 digits. Feeding
///   `$w - 1$` to `ln_1p` instead fixes it, provided `$w - 1$` is *not* formed by
///   subtracting: `$s - 1 = p^2/(s + 1)$` has no cancellation of its own.
///
/// # Policy
///
/// Both corrections are gated at [`Best`](PrecisionPolicy::Best) and above, together
/// costing one complex division and, only where a lane actually cancels, a second
/// logarithm. Below that the plain `$\ln(p + s)$` stands, as before.
#[inline(always)]
fn log_asinh<P: Policy, V: RealFloatVector>(p: Complex<V>) -> Complex<V> {
    let s = p.mul_add(p, Complex::ONE).sqrt();
    let w = p + s;

    if const { !P::POLICY.precision.ge(PrecisionPolicy::Best) } {
        return w.ln_p::<P>();
    }

    // w - 1, without ever forming the difference.
    let mut res = (p + p.square() / (s + Complex::ONE)).ln_1p_p::<P>();

    // `|w| < 1` alone is too eager. For tiny `p` with a negative real part `w = 1 + p`
    // sits just under one without anything having cancelled, and the log branch would
    // undo the `ln_1p` correction above. Genuine cancellation needs the two terms to be
    // large and nearly opposite, so `|p| > 1` as well, which is also where `u` itself
    // stops being trustworthy, `u = w - 1 ~ -1` then being a difference of two terms of
    // size `|p|`.
    let flip = w.norm_sqr().cmp_lt(V::ONE) & p.norm_sqr().cmp_gt(V::ONE);

    if thermite::unlikely(flip.any()) {
        let l = (s - p).ln_p::<P>();

        res = flip.select(Complex::new(-l.re, -l.im), res);
    }

    res
}

/// `$\ln w$`, taking the reciprocal companion where `w` has cancelled.
///
/// The inverse trigonometric and hyperbolic functions are each `$\ln$` of a sum of two
/// terms whose *difference* is the algebraically conjugate root, and in every case the
/// product of the two is exactly one:
///
/// ```text
/// asin :  (iz + s)(s - iz) = s^2 + z^2 = 1      s = sqrt(1 - z^2)
/// acos :  (z + is)(z - is) = z^2 + s^2 = 1      s = sqrt(1 - z^2)
/// asinh:  (z + s)(s - z)   = s^2 - z^2 = 1      s = sqrt(1 + z^2)
/// ```
///
/// So exactly one of the pair is well conditioned: for large `$|z|$` the two terms are
/// nearly equal in magnitude, one sum cancels to nothing and the other doubles. At
/// `$z = 10^8 i$` the `asin` sum is `$-10^8 + 10^8$`, *exactly zero*, and `$\ln 0$`
/// returned an infinity for a true value of `19.11i`.
///
/// Since `$ww' = 1$`, `$|w| < 1$` is an exact test for which one cancelled, and
/// `$\ln w = -\ln w'$` unambiguously, not merely up to `$2\pi i$`, because the
/// companion is in the right half-plane exactly when it is the one being selected.
///
/// One `ln` either way: the argument is blended *before* the logarithm, so the cost
/// over the naive form is a complex add, a `norm_sqr` and two selects.
///
/// # Policy
///
/// Gated at [`Best`](PrecisionPolicy::Best) and above. Below it the cancelling form is
/// used unconditionally, as it was before. The failure needs `$|z| \gg 1$`, and the
/// lower tiers do not promise the digits that are lost.
#[inline(always)]
fn ln_reciprocal_pair<P: Policy, V: RealFloatVector>(w: Complex<V>, companion: Complex<V>) -> Complex<V> {
    if const { !P::POLICY.precision.ge(PrecisionPolicy::Best) } {
        return w.ln_p::<P>();
    }

    let flip = w.norm_sqr().cmp_lt(V::ONE);

    let l = flip.select(companion, w).ln_p::<P>();

    Complex::new(l.re.neg_c(flip), l.im.neg_c(flip))
}

/// `$b^{a+ci} - 1$` from the real `$b^a - 1$` and the angle `$\phi = c\ln b$`.
///
/// ```text
/// b^(a+ci) - 1 = (b^a cos(phi) - 1) + i b^a sin(phi)
///              = (bm1 cos(phi) + (cos(phi) - 1)) + i (bm1 + 1) sin(phi)
/// ```
///
/// `exp_m1` and `cos_m1` are the cancellation-free primitives, keeping the relative
/// accuracy near `z = 0` that the naive form loses entirely. Shared by the `e`, `2`
/// and `10` bases, which differ only in which real `*_m1` supplies `bm1` and in the
/// scale of `phi`.
#[inline(always)]
fn expm1_from<P: Policy, V: RealFloatVector>(bm1: V, phi: V) -> Complex<V> {
    let (s, c) = phi.sin_cos_p::<P>();
    let cm1 = phi.cos_m1_p::<P>();

    Complex::new(bm1.mul_adde(c, cm1), s.mul_adde(bm1, s))
}

/// Masks `+-inf` out of a `d * ln r` angle term.
///
/// `ln r` is -inf at r = 0 and +inf at r = inf, so a real exponent (`d == 0`, the
/// overwhelmingly common case) forms `0 * inf` where the limit is plainly 0, and one
/// NaN there takes the whole result with it: `(0+0i)^2` was NaN on the strength of it.
/// Killing the infinity before the product exists is a compare and an AND, no branch,
/// and it leaves the term identical wherever `d` is not zero.
///
/// This is what reduces `powf` to the `powfr` formula, which never forms `ln r` at all.
#[inline(always)]
fn finite_log_term<P: Policy, V: RealFloatVector>(ln_r: V, d: V) -> V {
    if const { !P::POLICY.check_overflow } {
        return ln_r;
    }

    ln_r.zz(d.cmp_ne(V::ZERO))
}

impl<V: RealFloatVector> SpecializedTranscendentalMath<Complex<V::Element>> for Complex<V> {
    /// `sin(a + bi) = sin(a)cosh(b) + i*cos(a)sinh(b)`,
    /// `cos(a + bi) = cos(a)cosh(b) - i*sin(a)sinh(b)`.
    #[inline(always)]
    fn sin_cos<P: Policy>(self) -> (Self, Self) {
        let (s, c) = self.re.sin_cos_p::<P>();
        let (sh, ch) = self.im.sinh_cosh_p::<P>();

        (Complex::new(s * ch, c * sh), Complex::new(c * ch, -(s * sh)))
    }

    /// `$\sin(\pi z)$` and `$\cos(\pi z)$`, from the *real* `sincos_pi`.
    ///
    /// Must be overridden rather than left to the default. That default is
    /// `sin_cos(z * pi)`, which rounds `pi * Re z` before doing any reduction and so
    /// throws away the exact argument reduction real `sincos_pi` performs near the
    /// integers, precisely where the Gamma reflection formulas put their poles, and
    /// where `sin(pi z)` passes through zero. It is also no more work: one real
    /// `sincos_pi` and one real `sinh_cosh`, the same two calls the default makes.
    #[inline(always)]
    fn sincos_pi<P: Policy>(self) -> (Self, Self) {
        let (s, c) = self.re.sincos_pi_p::<P>();
        let (sh, ch) = (self.im * <V as thermite::math::FloatConsts>::PI).sinh_cosh_p::<P>();

        (Complex::new(s * ch, c * sh), Complex::new(c * ch, -(s * sh)))
    }

    /// `$\tan(a + bi) = \frac{\sin 2a + i\sinh 2b}{\cos 2a + \cosh 2b}$`
    ///
    /// The doubled-angle form takes one real division, where the default
    /// (`sin_cos` then a complex divide) takes a complex one.
    ///
    /// Under [`check_overflow`](thermite::math::policy::PolicyParameters::check_overflow)
    /// the saturation is handled: `$\tan(z) \to i\,\mathrm{sign}(b)$` as `$|b|$` grows,
    /// but `$\sinh$` and `$\cosh$` both overflow past `$|2b| \approx 710$` and the
    /// quotient becomes `inf/inf`. `tan(1 + 400i)` was `NaN`.
    #[inline(always)]
    fn tan<P: Policy>(self) -> Self {
        let (two_re, two_im) = (self.re + self.re, self.im + self.im);

        let (s, c) = two_re.sin_cos_p::<P>();
        let (sh, ch) = two_im.sinh_cosh_p::<P>();

        let denom = c + ch;
        let res = Complex::new(s, sh) / denom;

        saturate::<P, V>(res, denom, Complex::new(V::ZERO, V::ONE.mul_sign(self.im)))
    }

    /// `sinh(a + bi) = sinh(a)cos(b) + i*cosh(a)sin(b)`,
    /// `cosh(a + bi) = cosh(a)cos(b) + i*sinh(a)sin(b)`.
    #[inline(always)]
    fn sinh_cosh<P: Policy>(self) -> (Self, Self) {
        let (s, c) = self.im.sin_cos_p::<P>();
        let (sh, ch) = self.re.sinh_cosh_p::<P>();

        (Complex::new(sh * c, ch * s), Complex::new(ch * c, sh * s))
    }

    /// `tanh(a + bi) = (sinh(2a) + i*sin(2b)) / (cosh(2a) + cos(2b))`.
    ///
    /// Saturates to `$\mathrm{sign}(a)$` for large `$|a|$` under `check_overflow`; see
    /// [`tan`](Self::tan), of which this is the transpose.
    #[inline(always)]
    fn tanh<P: Policy>(self) -> Self {
        let (two_re, two_im) = (self.re + self.re, self.im + self.im);

        let (s, c) = two_im.sin_cos_p::<P>();
        let (sh, ch) = two_re.sinh_cosh_p::<P>();

        let denom = ch + c;
        let res = Complex::new(sh, s) / denom;

        saturate::<P, V>(res, denom, Complex::new(V::ONE.mul_sign(self.re), V::ZERO))
    }

    /// `$\mathrm{sinc}(z) = \sin(z)/z$`, with the removable singularity filled in.
    #[inline(always)]
    fn sinc<P: Policy>(self) -> Self {
        let is_zero = self.is_zero();

        // 0/0 = NaN at the origin, so the guard has to be a select. The quotient
        // cannot be patched up after the fact.
        let q = self.sin_p::<P>() / self;

        is_zero.select(Self::ONE, q)
    }

    /// `$\mathrm{atanhc}(z) = \operatorname{atanh}(z)/z$`, singularity filled in.
    #[inline(always)]
    fn atanhc<P: Policy>(self) -> Self {
        let is_zero = self.is_zero();

        // 0/0 = NaN at the origin, so the guard has to be a select. The quotient
        // cannot be patched up after the fact.
        let q = self.atanh_p::<P>() / self;

        is_zero.select(Self::ONE, q)
    }

    /// `$\mathrm{sinhc}(z) = \sinh(z)/z$`, with the removable singularity filled in.
    ///
    /// Note `$\mathrm{sinhc}(z) = \mathrm{sinc}(iz)$`, so on the imaginary axis this is the
    /// ordinary `sinc` and it has the same zeros, at `$z = ik\pi$`.
    #[inline(always)]
    fn sinhc<P: Policy>(self) -> Self {
        let is_zero = self.is_zero();

        // 0/0 = NaN at the origin, so the guard has to be a select. The quotient
        // cannot be patched up after the fact.
        let q = self.sinh_p::<P>() / self;

        is_zero.select(Self::ONE, q)
    }

    /// `$\mathrm{sinc}_\pi(z) = \frac{\sin(\pi z)}{\pi z}$`, singularity filled in.
    ///
    /// Overridden so the zeros are *exact*. The default is `sinc(z * pi)`, which
    /// rounds `pi * Re z` before reducing. The subsequent division by `pi z` cancels
    /// most of that error, so the default is accurate to about an ulp, but at a
    /// non-zero integer it returns ~1e-16 rather than zero. Going through the real
    /// `sin_pi`, which is exactly zero there, makes this exactly zero too.
    ///
    /// That is the property that makes `sinc_pi` an *interpolating* kernel: Lanczos
    /// and sinc resampling reproduce their samples only if the kernel vanishes at
    /// every non-zero integer.
    #[inline(always)]
    fn sinc_pi<P: Policy>(self) -> Self {
        let is_zero = self.is_zero();

        // As in `sinc`: 0/0 is NaN at the origin, so the guard must be a select.
        // Trait-qualified: the blanket `TranscendentalMath::sin_pi` is equally in scope.
        let q = SpecializedTranscendentalMath::sin_pi::<P>(self) / (self * <V as thermite::math::FloatConsts>::PI);

        is_zero.select(Self::ONE, q)
    }

    /// `e^(a + bi) = e^a * (cos(b) + i*sin(b))`.
    #[inline(always)]
    fn exp<P: Policy>(self) -> Self {
        Self::from_polar_p::<P>(self.re.exp_p::<P>(), self.im)
    }

    /// `exph(z) = e^z / 2`.
    #[inline(always)]
    fn exph<P: Policy>(self) -> Self {
        Self::from_polar_p::<P>(self.re.exph_p::<P>(), self.im)
    }

    /// `$2^z = 2^a e^{ib\ln 2}$`.
    ///
    /// # Policy
    ///
    /// Above [`Average`](PrecisionPolicy::Average) the real part goes through the real
    /// `exp2`. Rescaling it as `exp(a ln 2)` instead rounds `a ln 2` first, and `exp`
    /// then amplifies that rounding by the argument: `exp2(1000)` is wrong in its
    /// tenth digit (~300 ulp) that way. The imaginary part can afford the multiply
    /// either way, feeding a `sincos` that reduces its own argument.
    ///
    /// At or below `Average` the rescaled form is used, `exp2` being the more
    /// expensive kernel and 300 ulp being well inside that tier's budget.
    #[inline(always)]
    fn exp2<P: Policy>(self) -> Self {
        if const { P::POLICY.precision.le(PrecisionPolicy::Average) } {
            return (self * V::LN_2).exp_p::<P>();
        }

        Self::from_polar_p::<P>(self.re.exp2_p::<P>(), self.im * V::LN_2)
    }

    /// `$10^z = 10^a e^{ib\ln 10}$`. See [`exp2`](Self::exp2), including the policy split.
    #[inline(always)]
    fn exp10<P: Policy>(self) -> Self {
        if const { P::POLICY.precision.le(PrecisionPolicy::Average) } {
            return (self * V::LN_10).exp_p::<P>();
        }

        Self::from_polar_p::<P>(self.re.exp10_p::<P>(), self.im * V::LN_10)
    }

    /// `$e^z - 1$`, without the cancellation of forming `$e^z$` and subtracting one.
    #[inline(always)]
    fn exp_m1<P: Policy>(self) -> Self {
        expm1_from::<P, V>(self.re.exp_m1_p::<P>(), self.im)
    }

    /// `$2^z - 1$`. Through the real `exp2_m1` above `Average`, as [`exp2`](Self::exp2) is.
    #[inline(always)]
    fn exp2_m1<P: Policy>(self) -> Self {
        if const { P::POLICY.precision.le(PrecisionPolicy::Average) } {
            return (self * V::LN_2).exp_m1_p::<P>();
        }

        expm1_from::<P, V>(self.re.exp2_m1_p::<P>(), self.im * V::LN_2)
    }

    /// `$10^z - 1$`. Through the real `exp10_m1` above `Average`, as [`exp10`](Self::exp10) is.
    #[inline(always)]
    fn exp10_m1<P: Policy>(self) -> Self {
        if const { P::POLICY.precision.le(PrecisionPolicy::Average) } {
            return (self * V::LN_10).exp_m1_p::<P>();
        }

        expm1_from::<P, V>(self.re.exp10_m1_p::<P>(), self.im * V::LN_10)
    }

    /// `z^w`, the principal value.
    #[inline(always)]
    fn powf<P: Policy>(self, e: Self) -> Self {
        // z^w = (r e^(i t))^(c + di)
        //     = r^c e^(-d t) * (cos(c t + d ln r) + i sin(c t + d ln r))
        //     = from_polar(e^(c ln r - d t), c t + d ln r)
        //
        // The angle needs `ln r` at every policy, so it is computed once up front.
        let (r, theta) = self.to_polar_p::<P>();
        let ln_r = r.ln_p::<P>();

        // Both products against `ln r` are `0 * inf` at the degenerate points, each for
        // its own exponent part: `d ln r` in the angle, `c ln r` in the fused exponent.
        // `0^0` is 1 and `0^2` is 0 only once both are masked.
        let ln_r_angle = finite_log_term::<P, V>(ln_r, e.im);
        let ln_r_mod = finite_log_term::<P, V>(ln_r, e.re);

        let angle = e.im.mul_adde(ln_r_angle, e.re * theta);

        let mut modulus = if const { P::POLICY.precision.le(PrecisionPolicy::Average) } {
            // Fused exponent: one `exp` for the whole modulus. `ln r` is already in
            // hand and `powf` is `exp(c ln r)` underneath, so the split form costs a
            // second `exp` and a second `ln` for an extended-precision `c ln r` that
            // this tier is not paying for.
            e.im.nmul_adde(theta, e.re * ln_r_mod).exp_p::<P>()
        } else {
            // `powf` carries more precision through `c ln r` than the fused exponent
            // can, which is what keeps `|z^w|` near an ulp once `|c ln r|` is large.
            r.powf_p::<P>(e.re) * (-e.im * theta).exp_p::<P>()
        };

        if const { P::POLICY.check_overflow && !P::POLICY.precision.le(PrecisionPolicy::Average) } {
            // `r^c` leaves the range on its own where `e^{-dt}` would have brought the
            // product back. `(-1e200)^(2 + 300i)` is about 1e-9 and the split form
            // gives `inf`, or `NaN` from the mirror-image `0 * inf`. The fused exponent
            // has no such intermediate, so it covers those lanes.
            let lost = !modulus.is_finite();

            if thermite::unlikely(lost.any()) {
                modulus = lost.select(e.im.nmul_adde(theta, e.re * ln_r_mod).exp_p::<P>(), modulus);
            }
        }

        Self::from_polar_p::<P>(modulus, angle)
    }

    /// The principal cube root.
    ///
    /// This does not agree with the real cube root of a negative real: the real
    /// cube root of -8 is -2, the principal complex one `$1 + i\sqrt{3}$`.
    #[inline(always)]
    fn cbrt<P: Policy>(self) -> Self {
        let (r, theta) = self.to_polar_p::<P>();

        // 1/3 is not representable, so divide. Multiplying by a rounded reciprocal
        // loses a bit at the higher policies.
        let three: V = thermite::const_splat!(int <V::Element>: 3);

        Self::from_polar_p::<P>(r.cbrt_p::<P>(), theta / three)
    }

    /// The principal `N`th root, `$z^{1/N} = |z|^{1/N} e^{i\arg(z)/N}$`.
    ///
    /// The trait default is real-only: for odd `N` it takes `abs()` and restores the
    /// sign afterwards, which over C collapses `z` to its modulus and returns a real
    /// root.
    #[inline(always)]
    fn nth_root_n<P: Policy, const N: usize>(self) -> Self {
        let (r, theta) = self.to_polar_p::<P>();

        // Not `const_splat!`: `N` is a generic parameter, which cannot appear in the
        // const operation that arm expands to.
        let n = V::splat(<V::Element as FloatElement>::from_int(N as thermite::LargeInt));

        Self::from_polar_p::<P>(r.powf_p::<P>(n.approx_reciprocal_p::<P>()), theta / n)
    }

    /// [`nth_root_n`](Self::nth_root_n) for a degree known only at runtime.
    #[inline(always)]
    fn nth_root<P: Policy>(self, n: u32) -> Self {
        let (r, theta) = self.to_polar_p::<P>();
        let n = V::splat(<V::Element as FloatElement>::from_int(n as thermite::LargeInt));

        Self::from_polar_p::<P>(r.powf_p::<P>(n.approx_reciprocal_p::<P>()), theta / n)
    }

    /// The principal natural logarithm: `ln(z) = ln|z| + i*arg(z)`.
    ///
    /// Branch cut on `(-inf, 0]`, continuous from above; `-pi <= Im(ln z) <= pi`.
    #[inline(always)]
    fn ln<P: Policy>(self) -> Self {
        let (r, theta) = self.to_polar_p::<P>();

        Complex::new(r.ln_p::<P>(), theta)
    }

    /// `$\ln(1 + z)$`, without the cancellation of forming `1 + z` first.
    #[inline(always)]
    fn ln_1p<P: Policy>(self) -> Self {
        // |1 + z|^2 - 1 = 2a + a^2 + b^2 = a(a + 2) + b^2, so
        //   Re = 0.5 * ln_1p(a(a + 2) + b^2),  Im = atan2(b, 1 + a)
        // The real part goes through ln_1p to keep the accuracy near z = 0 that
        // ln(1 + z) would lose.
        let t = self.re.mul_adde(self.re + V::TWO, self.im * self.im);

        Complex::new(
            t.ln_1p_p::<P>() * <V as FloatVector>::HALF,
            self.im.atan2_p::<P>(self.re + V::ONE),
        )
    }

    /// `$\log_2 z = \log_2|z| + i\arg(z)\log_2 e$`.
    ///
    /// Through the real `log2` rather than `ln(z) * log2(e)`, which is the same work
    /// (one multiply fewer, in fact, as the argument is scaled but `ln|z|` is not) and
    /// picks up whatever the element's own `log2` does. Measured identical to the
    /// rescaled form on f64, where thermite's `log2` *is* `ln * LOG2_E`; f32 has a
    /// dedicated kernel, so no policy gate is warranted either way.
    #[inline(always)]
    fn log2<P: Policy>(self) -> Self {
        let (r, theta) = self.to_polar_p::<P>();

        Complex::new(r.log2_p::<P>(), theta * V::LOG2_E)
    }

    /// `$\log_{10} z = \log_{10}|z| + i\arg(z)\log_{10} e$`. See [`log2`](Self::log2).
    #[inline(always)]
    fn log10<P: Policy>(self) -> Self {
        let (r, theta) = self.to_polar_p::<P>();

        Complex::new(r.log10_p::<P>(), theta * V::LOG10_E)
    }

    /// `log_N(z) = ln(z) / ln(N)` for a compile-time integer base.
    #[inline(always)]
    fn log_n_n<P: Policy, const N: usize>(self) -> Self {
        // See `nth_root`: a generic `N` rules `const_splat!` out here.
        let ln_n = V::splat(<V::Element as FloatElement>::from_int(N as thermite::LargeInt)).ln_p::<P>();

        self.ln_p::<P>() / ln_n
    }

    /// `log_n(z) = ln(z) / ln(n)` for a base known only at runtime.
    #[inline(always)]
    fn log_n<P: Policy>(self, n: u32) -> Self {
        let ln_n = V::splat(<V::Element as FloatElement>::from_int(n as thermite::LargeInt)).ln_p::<P>();

        self.ln_p::<P>() / ln_n
    }

    /// `$\ln(1 - e^{-z})$`.
    ///
    /// The `_ext` form lets a real vector reuse an already-computed `ln(x)`. There
    /// is no such shortcut over C, so it forwards to the plain form.
    #[inline(always)]
    fn ln1m_expnx_ext<P: Policy>(self, _lnx: Self) -> Self {
        self.ln1m_expnx_p::<P>()
    }

    // --- inverse trigonometric / hyperbolic functions ---

    /// `asin(z) = -i ln(iz + sqrt(1 - z^2))`.
    ///
    /// Branch cuts on `(-inf, -1)` (continuous from above) and `(1, inf)`
    /// (continuous from below); `-pi/2 <= Re(asin z) <= pi/2`.
    #[inline(always)]
    fn asin<P: Policy>(self) -> Self {
        // asin(z) = -i*asinh(iz); see [`log_asinh`] for the conditioning.
        mul_neg_i(log_asinh::<P, V>(mul_i(self)))
    }

    /// `acos(z) = -i ln(z + i sqrt(1 - z^2))`.
    ///
    /// Branch cuts on `(-inf, -1)` and `(1, inf)`; `0 <= Re(acos z) <= pi`.
    #[inline(always)]
    fn acos<P: Policy>(self) -> Self {
        let is = mul_i(self.nmul_add(self, Self::ONE).sqrt());

        mul_neg_i(ln_reciprocal_pair::<P, V>(self + is, self - is))
    }

    /// `atan(z) = (ln(1 + iz) - ln(1 - iz)) / (2i)`.
    ///
    /// Branch cuts on `(-inf*i, -i]` and `[i, inf*i)`; `-pi/2 <= Re(atan z) <= pi/2`.
    #[inline(always)]
    fn atan<P: Policy>(self) -> Self {
        // 1 +- iz = (1 -+ Im z) +- i*Re z: two adds, no multiply by the unit.
        let a = Complex::new(V::ONE - self.im, self.re);
        let b = Complex::new(V::ONE + self.im, -self.re);

        // z / (2i) == -0.5i * z, and the -i is the free swap-and-negate.
        mul_neg_i(a.ln_p::<P>() - b.ln_p::<P>()) * <V as FloatVector>::HALF
    }

    /// `asinh(z) = ln(z + sqrt(1 + z^2))`.
    #[inline(always)]
    fn asinh<P: Policy>(self) -> Self {
        log_asinh::<P, V>(self)
    }

    /// `acosh(z) = 2 ln(sqrt((z+1)/2) + sqrt((z-1)/2))`.
    ///
    /// Branch cut on `(-inf, 1)`, continuous from above.
    #[inline(always)]
    fn acosh<P: Policy>(self) -> Self {
        // (z +- 1)/2 scales by a *real* half, so the complex FMA (four inner FMAs,
        // half of them against a zero imaginary part) is one FMA and one multiply.
        let h = <V as FloatVector>::HALF;
        let half_im = self.im * h;

        let a = Complex::new(self.re.mul_adde(h, h), half_im).sqrt();
        let b = Complex::new(self.re.mul_sube(h, h), half_im).sqrt();

        let half_res = (a + b).ln_p::<P>();

        half_res + half_res
    }

    /// `atanh(z) = (ln(1 + z) - ln(1 - z)) / 2`.
    ///
    /// Branch cuts on `(-inf, -1]` and `[1, inf)`.
    #[inline(always)]
    fn atanh<P: Policy>(self) -> Self {
        ((Self::ONE + self).ln_p::<P>() - (Self::ONE - self).ln_p::<P>()) * <V as FloatVector>::HALF
    }
}

// --- SpecializedSpatialMath: the norms, as real-valued complex numbers ---

/// The modulus of each term, for the `hypot` family.
#[inline(always)]
fn moduli<V: RealFloatVector, P: Policy, const N: usize>(values: [Complex<V>; N]) -> [V; N] {
    let mut out = [V::ZERO; N];

    let mut i = 0;
    while i < N {
        out[i] = values[i].norm_p::<P>();
        i += 1;
    }

    out
}

impl<V: RealFloatVector> SpecializedSpatialMath<Complex<V::Element>> for Complex<V> {
    /// `$\sqrt{\sum_i |z_i|^2}$`, as a real complex number.
    ///
    /// Must be overridden, and not only for tuning. The generic `hypot_n` changes
    /// *meaning* over C depending on the precision policy: its high-precision path
    /// opens with `abs()`, which here is the modulus, so everything after it is real
    /// and the result is the norm, but the `PrecisionPolicy::Worst` path skips that
    /// and squares directly, giving the analytic continuation `sqrt(sum z_i^2)`
    /// instead. Two different functions behind one name, chosen by a policy.
    ///
    /// This pins the norm, matching [`l2_norm`](Self::l2_norm) and the crate docs.
    /// Taking the modulus of each term first costs `N` extra square roots and buys
    /// the same overflow safety the real `hypot_n` has.
    #[inline(always)]
    fn hypot_n<P: Policy, const N: usize>(values: [Self; N]) -> Self {
        Self::real(<V as thermite::math::SpatialMathWithPolicy>::hypot_n_p::<P, N>(
            moduli::<V, P, N>(values),
        ))
    }

    /// `$1/\sqrt{\sum_i |z_i|^2}$`, as a real complex number. See [`hypot_n`](Self::hypot_n).
    #[inline(always)]
    fn inv_hypot_n<P: Policy, const N: usize>(values: [Self; N]) -> Self {
        Self::real(<V as thermite::math::SpatialMathWithPolicy>::inv_hypot_n_p::<P, N>(
            moduli::<V, P, N>(values),
        ))
    }

    /// `|re| + |im|`, as a real complex number.
    #[inline(always)]
    fn l1_norm<P: Policy>(self) -> Self {
        Self::real(self.norm_l1())
    }

    /// `$|z|^2 = z\bar{z}$`, as a real complex number.
    #[inline(always)]
    fn l2_norm_squared<P: Policy>(self) -> Self {
        Self::real(self.norm_sqr())
    }

    /// The modulus `$|z|$`, as a real complex number.
    ///
    /// The default `sqrt(l2_norm_squared())` squares the range and overflows for
    /// large components; `hypot` does not.
    #[inline(always)]
    fn l2_norm<P: Policy>(self) -> Self {
        Self::real(self.norm_p::<P>())
    }
}

// --- No SpecializedRealMath ---
//
// RealMath is the part of the math library that assumes an ordered field: atan2 (a
// quadrant of the real plane), wrap_angle, to_degrees/to_radians, step, smoothstep
// and its inverse, logaddexp, rescale. Their defaults are written over max, abs and
// clamp as orderings. Over C they would return a plausible-looking result with no
// meaning.
//
// Each public math trait blankets off its own specialized trait, and leaving this
// one unimplemented costs nothing: CoreMath, TranscendentalMath and SpatialMath are
// unaffected. The argument of a complex number is ComplexMath::arg, returning a real
// value where atan2 would have to return a complex one.
