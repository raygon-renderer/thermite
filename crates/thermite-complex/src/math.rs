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
//! [`RealMath`](thermite::math::RealMath) is not implemented; see the note at the
//! bottom of this file.

use thermite::math::policy::Policy;
use thermite::math::specialized::{SpecializedCoreMath, SpecializedSpatialMath, SpecializedTranscendentalMath};
use thermite::prelude::*;

use crate::Complex;
use crate::specialized::{ComplexMathWithPolicy, ComplexVector, SpecializedComplexMath};
use crate::vector::ComplexFloatVector;

// --- SpecializedComplexMath: the real-valued and real-argument operations ---

impl<V: ComplexFloatVector> SpecializedComplexMath<Complex<V::Element>> for Complex<V> {
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
        Self::from_polar_p::<P>(base.powf_p::<P>(self.re), self.im * base.ln_p::<P>())
    }

    #[inline(always)]
    fn logr<P: Policy>(self, base: V) -> Self {
        // log_b(z) = ln(z) / ln(b); one real reciprocal, then scale both components.
        let (r, theta) = self.to_polar_p::<P>();
        let d = base.ln_p::<P>().reciprocal_p::<P>();

        Self::new(r.ln_p::<P>() * d, theta * d)
    }

    #[inline(always)]
    fn finv<P: Policy>(self) -> Self {
        // (conj(z) / |z|) / |z| keeps every intermediate in range, where
        // conj(z) / |z|^2 would overflow the square.
        let n = self.norm_p::<P>();

        (self.conj() / n) / n
    }
}

// --- SpecializedCoreMath ---

impl<V: ComplexFloatVector> SpecializedCoreMath<Complex<V::Element>> for Complex<V> {
    // 1/sqrt(z) = conj(sqrt(z)) / |z|, since |sqrt(z)|^2 == |z|.
    #[inline(always)]
    fn inverse_sqrt<P: Policy>(self) -> Self {
        let s = self.sqrt();
        let inv = self.norm_p::<P>().reciprocal_p::<P>();

        Complex::new(s.re * inv, -(s.im * inv))
    }
}

// --- SpecializedTranscendentalMath ---

impl<V: ComplexFloatVector> SpecializedTranscendentalMath<Complex<V::Element>> for Complex<V> {
    /// `sin(a + bi) = sin(a)cosh(b) + i*cos(a)sinh(b)`,
    /// `cos(a + bi) = cos(a)cosh(b) - i*sin(a)sinh(b)`.
    #[inline(always)]
    fn sin_cos<P: Policy>(self) -> (Self, Self) {
        let (s, c) = self.re.sin_cos_p::<P>();
        let (sh, ch) = self.im.sinh_cosh_p::<P>();

        (Complex::new(s * ch, c * sh), Complex::new(c * ch, -(s * sh)))
    }

    /// `$\tan(a + bi) = \frac{\sin 2a + i\sinh 2b}{\cos 2a + \cosh 2b}$`
    ///
    /// The doubled-angle form takes one real division, where the default
    /// (`sin_cos` then a complex divide) takes a complex one.
    #[inline(always)]
    fn tan<P: Policy>(self) -> Self {
        let (two_re, two_im) = (self.re + self.re, self.im + self.im);

        let (s, c) = two_re.sin_cos_p::<P>();
        let (sh, ch) = two_im.sinh_cosh_p::<P>();

        Complex::new(s, sh) / (c + ch)
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
    #[inline(always)]
    fn tanh<P: Policy>(self) -> Self {
        let (two_re, two_im) = (self.re + self.re, self.im + self.im);

        let (s, c) = two_im.sin_cos_p::<P>();
        let (sh, ch) = two_re.sinh_cosh_p::<P>();

        Complex::new(sh, s) / (ch + c)
    }

    /// `$\mathrm{sinc}(z) = \sin(z)/z$`, with the removable singularity filled in.
    #[inline(always)]
    fn sinc<P: Policy>(self) -> Self {
        let is_zero = self.is_zero();

        // 0/0 = NaN at the origin, so the guard has to be a select; the quotient
        // cannot be patched up after the fact.
        let q = self.sin_p::<P>() / self;

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

    /// `2^z = e^(z ln 2)`.
    #[inline(always)]
    fn exp2<P: Policy>(self) -> Self {
        (self * V::LN_2).exp_p::<P>()
    }

    /// `10^z = e^(z ln 10)`.
    #[inline(always)]
    fn exp10<P: Policy>(self) -> Self {
        (self * V::LN_10).exp_p::<P>()
    }

    /// `$e^z - 1$`, without the cancellation of forming `$e^z$` and subtracting one.
    #[inline(always)]
    fn exp_m1<P: Policy>(self) -> Self {
        // e^(a+bi) - 1 = (e^a cos(b) - 1) + i e^a sin(b)
        //              = (expm1(a) cos(b) + (cos(b) - 1)) + i (expm1(a) + 1) sin(b)
        //
        // expm1 and cos_m1 are the cancellation-free primitives, keeping the
        // relative accuracy near z = 0 that the naive form loses entirely.
        let em1 = self.re.exp_m1_p::<P>();
        let (s, c) = self.im.sin_cos_p::<P>();
        let cm1 = self.im.cos_m1_p::<P>();

        Complex::new(em1.mul_adde(c, cm1), s.mul_adde(em1, s))
    }

    /// `2^z - 1`.
    #[inline(always)]
    fn exp2_m1<P: Policy>(self) -> Self {
        (self * V::LN_2).exp_m1_p::<P>()
    }

    /// `10^z - 1`.
    #[inline(always)]
    fn exp10_m1<P: Policy>(self) -> Self {
        (self * V::LN_10).exp_m1_p::<P>()
    }

    /// `z^w`, the principal value.
    #[inline(always)]
    fn powf<P: Policy>(self, e: Self) -> Self {
        // z^w = (r e^(i t))^(c + di)
        //     = r^c e^(-d t) * (cos(c t + d ln r) + i sin(c t + d ln r))
        //     = from_polar(r^c e^(-d t), c t + d ln r)
        let (r, theta) = self.to_polar_p::<P>();

        Self::from_polar_p::<P>(
            r.powf_p::<P>(e.re) * (-e.im * theta).exp_p::<P>(),
            e.im.mul_adde(r.ln_p::<P>(), e.re * theta),
        )
    }

    /// The principal cube root.
    ///
    /// This does not agree with the real cube root of a negative real: the real
    /// cube root of -8 is -2, the principal complex one `$1 + i\sqrt{3}$`.
    #[inline(always)]
    fn cbrt<P: Policy>(self) -> Self {
        let (r, theta) = self.to_polar_p::<P>();

        // 1/3 is not representable, so divide; multiplying by a rounded reciprocal
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
    fn nth_root<P: Policy, const N: usize>(self) -> Self {
        let (r, theta) = self.to_polar_p::<P>();

        let n = V::splat(<V::Element as FloatElement>::from_int(N as thermite::LargeInt));

        Self::from_polar_p::<P>(r.powf_p::<P>(n.reciprocal_p::<P>()), theta / n)
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

    /// `log2(z) = ln(z) * log2(e)`.
    #[inline(always)]
    fn log2<P: Policy>(self) -> Self {
        self.ln_p::<P>() * V::LOG2_E
    }

    /// `log10(z) = ln(z) * log10(e)`.
    #[inline(always)]
    fn log10<P: Policy>(self) -> Self {
        self.ln_p::<P>() * V::LOG10_E
    }

    /// `log_N(z) = ln(z) / ln(N)` for a compile-time integer base.
    #[inline(always)]
    fn log_n<P: Policy, const N: usize>(self) -> Self {
        let ln_n = V::splat(<V::Element as FloatElement>::from_int(N as thermite::LargeInt)).ln_p::<P>();

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
        let w = self.mul_add(Self::I, self.nmul_add(self, Self::ONE).sqrt());

        -(Self::I * w.ln_p::<P>())
    }

    /// `acos(z) = -i ln(z + i sqrt(1 - z^2))`.
    ///
    /// Branch cuts on `(-inf, -1)` and `(1, inf)`; `0 <= Re(acos z) <= pi`.
    #[inline(always)]
    fn acos<P: Policy>(self) -> Self {
        let w = Self::I.mul_add(self.nmul_add(self, Self::ONE).sqrt(), self);

        -(Self::I * w.ln_p::<P>())
    }

    /// `atan(z) = (ln(1 + iz) - ln(1 - iz)) / (2i)`.
    ///
    /// Branch cuts on `(-inf*i, -i]` and `[i, inf*i)`; `-pi/2 <= Re(atan z) <= pi/2`.
    #[inline(always)]
    fn atan<P: Policy>(self) -> Self {
        let a = self.mul_add(Self::I, Self::ONE);
        let b = self.nmul_add(Self::I, Self::ONE);

        // z / (2i) == -0.5i * z
        (a.ln_p::<P>() - b.ln_p::<P>()) * Self::imag(-<V as FloatVector>::HALF)
    }

    /// `asinh(z) = ln(z + sqrt(1 + z^2))`.
    #[inline(always)]
    fn asinh<P: Policy>(self) -> Self {
        (self + self.mul_add(self, Self::ONE).sqrt()).ln_p::<P>()
    }

    /// `acosh(z) = 2 ln(sqrt((z+1)/2) + sqrt((z-1)/2))`.
    ///
    /// Branch cut on `(-inf, 1)`, continuous from above.
    #[inline(always)]
    fn acosh<P: Policy>(self) -> Self {
        let half = Self::real(<V as FloatVector>::HALF);

        let a = self.mul_add(half, half).sqrt();
        let b = self.mul_sub(half, half).sqrt();

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

impl<V: ComplexFloatVector> SpecializedSpatialMath<Complex<V::Element>> for Complex<V> {
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
