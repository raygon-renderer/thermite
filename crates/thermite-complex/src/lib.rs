#![no_std]

use thermite::{
    math::{
        RealMathWithPolicy, SpatialMathWithPolicy, TranscendentalMathWithPolicy,
        policy::{DefaultPolicy, Policy},
    },
    register::FloatElement,
};

/// A trait for vectors that support the necessary mathematical operations
/// to be used as the real and imaginary parts of a complex number.
pub trait MathVector: TranscendentalMathWithPolicy + SpatialMathWithPolicy + RealMathWithPolicy {}
impl<V> MathVector for V where V: TranscendentalMathWithPolicy + SpatialMathWithPolicy + RealMathWithPolicy {}

pub struct Complex<V: MathVector, P: Policy = DefaultPolicy> {
    pub re: V,
    pub im: V,
    _policy: core::marker::PhantomData<P>,
}

impl<V: MathVector, P: Policy> Clone for Complex<V, P> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<V: MathVector, P: Policy> Copy for Complex<V, P> {}

impl<V: MathVector, P: Policy> Complex<V, P> {
    /// Creates a complex number with the given real and imaginary parts.
    #[inline(always)]
    pub const fn new(re: V, im: V) -> Self {
        Self {
            re,
            im,
            _policy: core::marker::PhantomData,
        }
    }

    /// Creates a complex number with the given real and imaginary parts splatted across all lanes.
    #[inline(always)]
    pub fn splat(re: V::Element, im: V::Element) -> Self {
        Self::new(V::splat(re), V::splat(im))
    }

    /// Changes the policy of the complex number operations.
    #[inline(always)]
    pub const fn with_policy<Q: Policy>(self) -> Complex<V, Q> {
        Complex::<V, Q>::new(self.re, self.im)
    }
}

impl<V: MathVector, P: Policy> Complex<V, P> {
    /// Creates a complex number with the given real part and zero imaginary part.
    #[inline(always)]
    pub const fn real(re: V) -> Self {
        Self::new(re, V::ZERO)
    }

    /// Creates a complex number with zero real part and the given imaginary part.
    #[inline(always)]
    pub const fn imag(im: V) -> Self {
        Self::new(V::ZERO, im)
    }

    pub const I: Self = Self::new(V::ZERO, V::ONE);
    pub const NEG_I: Self = Self::new(V::ZERO, V::NEG_ONE);
    pub const ZERO: Self = Self::new(V::ZERO, V::ZERO);
    pub const ONE: Self = Self::new(V::ONE, V::ZERO);
}

impl<V: MathVector, P: Policy> Complex<V, P> {
    /// Computes the squared norm (magnitude) of the complex number.
    #[inline(always)]
    pub fn norm_sqr(self) -> V {
        self.re.mul_adde(self.re, self.im * self.im)
    }

    /// Computes the norm (magnitude) of the complex number.
    #[inline(always)]
    pub fn norm(self) -> V {
        self.re.hypot_p::<P>(self.im)
    }

    /// Scales/multiplies the complex number by the given vector.
    #[inline(always)]
    pub fn scale(self, t: V) -> Self {
        self * t
    }

    /// Unscales/divides the complex number by the given vector.
    #[inline(always)]
    pub fn unscale(self, t: V) -> Self {
        self / t
    }

    /// Computes the complex conjugate of the complex number.
    #[inline(always)]
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    /// Computes the multiplicative inverse of the complex number.
    #[inline(always)]
    pub fn inv(self) -> Self {
        self.unscale(self.norm_sqr()).conj()
    }

    /// Returns `self * m + a`
    #[inline(always)]
    pub fn mul_add(self, m: Self, a: Self) -> Self {
        Self::new(
            self.im.nmul_adde(m.im, self.re.mul_adde(m.re, a.re)),
            self.re.mul_adde(m.im, self.im.mul_adde(m.re, a.im)),
        )
    }

    /// Returns the L1 norm `|re| + |im|` -- the [Manhattan distance] from the origin.
    ///
    /// [Manhattan distance]: https://en.wikipedia.org/wiki/Taxicab_geometry
    #[inline(always)]
    pub fn l1_norm(self) -> V {
        self.re.abs() + self.im.abs()
    }

    /// Calculate the principal Arg of self.
    #[inline(always)]
    pub fn arg(self) -> V {
        self.im.atan2_p::<P>(self.re)
    }

    /// Convert to polar form (r, theta), such that
    /// `self = r * exp(i * theta)`
    #[inline(always)]
    pub fn to_polar(self) -> (V, V) {
        (self.norm(), self.arg())
    }

    /// Convert a polar representation into a complex number.
    #[inline(always)]
    pub fn from_polar(r: V, theta: V) -> Self {
        let (s, c) = theta.sin_cos_p::<P>();
        Self::new(r * c, r * s)
    }

    /// Computes `e^(self)`, where `e` is the base of the natural logarithm.
    #[inline]
    pub fn exp(self) -> Self {
        // formula: e^(a + bi) = e^a (cos(b) + i*sin(b))
        // = from_polar(e^a, b)
        Self::from_polar(self.re.exp_p::<P>(), self.im)
    }

    /// Computes the principal value of natural logarithm of `self`.
    ///
    /// This function has one branch cut:
    ///
    /// * `(-∞, 0]`, continuous from above.
    ///
    /// The branch satisfies `-π ≤ arg(ln(z)) ≤ π`.
    #[inline]
    pub fn ln(self) -> Self {
        // formula: ln(z) = ln|z| + i*arg(z)
        let (r, theta) = self.to_polar();
        Self::new(r.ln_p::<P>(), theta)
    }

    /// Computes the principal value of the square root of `self`.
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        // Old formula: sqrt(r e^(it)) = sqrt(r) e^(it/2)
        // let (r, theta) = self.to_polar();
        // Self::from_polar(r.sqrt(), theta * V::splat_as(0.5))

        // New formula from: http://stanleyrabinowitz.com/bibliography/complexSquareRoot.pdf
        let half = V::HALF;
        let m = self.norm() * half;

        let r = self.re.mul_adde(half, m).sqrt(); // sqrt(0.5 * (m + re))
        let i = self.re.nmul_adde(half, m).sqrt(); // sqrt(0.5 * (m - re))

        Complex::new(r, i.mul_sign(self.im))
    }

    /// Computes the principal value of the cube root of `self`.
    ///
    /// Note that this does not match the usual result for the cube root of
    /// negative real numbers. For example, the real cube root of `-8` is `-2`,
    /// but the principal complex cube root of `-8` is `1 + i√3`.
    #[inline]
    pub fn cbrt(self) -> Self {
        // formula: cbrt(r e^(it)) = cbrt(r) e^(it/3)
        let (r, theta) = self.to_polar();
        // 1/3 isn't well-represented in float, so an exact inverse can't work with all precisions
        Self::from_polar(r.cbrt_p::<P>(), theta / V::splat(FloatElement::from_i64(3)))
    }

    /// Raises `self` to a floating point power.
    #[inline]
    pub fn powf(self, exp: V) -> Self {
        // formula: x^y = (ρ e^(i θ))^y = ρ^y e^(i θ y)
        // = from_polar(ρ^y, θ y)
        let (r, theta) = self.to_polar();
        Self::from_polar(r.powf_p::<P>(exp), theta * exp)
    }

    /// Returns the logarithm of `self` with respect to an arbitrary base.
    #[inline]
    pub fn log(self, base: V) -> Self {
        // formula: log_y(x) = log_y(ρ e^(i θ))
        // = log_y(ρ) + log_y(e^(i θ)) = log_y(ρ) + ln(e^(i θ)) / ln(y)
        // = log_y(ρ) + i θ / ln(y)
        let (r, theta) = self.to_polar();
        let d = V::ONE / base.ln_p::<P>();
        Self::new(r.ln_p::<P>() * d, theta * d)
    }

    /// Raises `self` to a complex power.
    #[inline]
    pub fn powc(self, exp: Self) -> Self {
        // formula: x^y = (a + i b)^(c + i d)
        // = (ρ e^(i θ))^c (ρ e^(i θ))^(i d)
        //    where ρ=|x| and θ=arg(x)
        // = ρ^c e^(−d θ) e^(i c θ) ρ^(i d)
        // = p^c e^(−d θ) (cos(c θ)
        //   + i sin(c θ)) (cos(d ln(ρ)) + i sin(d ln(ρ)))
        // = p^c e^(−d θ) (
        //   cos(c θ) cos(d ln(ρ)) − sin(c θ) sin(d ln(ρ))
        //   + i(cos(c θ) sin(d ln(ρ)) + sin(c θ) cos(d ln(ρ))))
        // = p^c e^(−d θ) (cos(c θ + d ln(ρ)) + i sin(c θ + d ln(ρ)))
        // = from_polar(p^c e^(−d θ), c θ + d ln(ρ))
        let (r, theta) = self.to_polar();
        Self::from_polar(
            r.powf_p::<P>(exp.re) * (-exp.im * theta).exp_p::<P>(),
            exp.im.mul_adde(r.ln_p::<P>(), exp.re * theta),
        )
    }

    /// Raises a floating point number to the complex power `self`.
    #[inline]
    pub fn expf(self, base: V) -> Self {
        // formula: x^(a+bi) = x^a x^bi = x^a e^(b ln(x) i)
        // = from_polar(x^a, b ln(x))
        Self::from_polar(base.powf_p::<P>(self.re), self.im * base.ln_p::<P>())
    }

    /// Computes sine and cosine of `self` together, improving efficiency.
    #[inline]
    pub fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.re.sin_cos_p::<P>();
        let (sh, ch) = (self.im.sinh_p::<P>(), self.im.cosh_p::<P>());

        (Self::new(s * ch, c * sh), Self::new(c * ch, -s * sh))
    }

    /// Computes the sine of `self`.
    #[inline]
    pub fn sin(self) -> Self {
        // formula: sin(a + bi) = sin(a)cosh(b) + i*cos(a)sinh(b)
        let (s, c) = self.re.sin_cos_p::<P>();
        Self::new(s * self.im.cosh_p::<P>(), c * self.im.sinh_p::<P>())
    }

    /// Computes the cosine of `self`.
    #[inline]
    pub fn cos(self) -> Self {
        // formula: cos(a + bi) = cos(a)cosh(b) - i*sin(a)sinh(b)
        let (s, c) = self.re.sin_cos_p::<P>();
        Self::new(c * self.im.cosh_p::<P>(), -s * self.im.sinh_p::<P>())
    }

    /// Computes the tangent of `self`.
    #[inline]
    pub fn tan(self) -> Self {
        // formula: tan(a + bi) = (sin(2a) + i*sinh(2b))/(cos(2a) + cosh(2b))
        let (two_re, two_im) = (self.re + self.re, self.im + self.im);
        let (s, c) = two_re.sin_cos_p::<P>();
        Self::new(s, two_im.sinh_p::<P>()).unscale(c + two_im.cosh_p::<P>())
    }

    /// Computes the principal value of the inverse sine of `self`.
    ///
    /// This function has two branch cuts:
    ///
    /// * `(-∞, -1)`, continuous from above.
    /// * `(1, ∞)`, continuous from below.
    ///
    /// The branch satisfies `-π/2 ≤ Re(asin(z)) ≤ π/2`.
    #[inline]
    pub fn asin(self) -> Self {
        // formula: arcsin(z) = -i ln(sqrt(1-z^2) + iz)
        Self::NEG_I * self.mul_add(Self::I, self.mul_add(-self, Self::ONE).sqrt()).ln()
    }

    /// Computes the principal value of the inverse cosine of `self`.
    ///
    /// This function has two branch cuts:
    ///
    /// * `(-∞, -1)`, continuous from above.
    /// * `(1, ∞)`, continuous from below.
    ///
    /// The branch satisfies `0 ≤ Re(acos(z)) ≤ π`.
    #[inline]
    pub fn acos(self) -> Self {
        // formula: arccos(z) = -i ln(i sqrt(1-z^2) + z)
        Self::NEG_I * Self::I.mul_add(self.mul_add(-self, Self::ONE).sqrt(), self).ln()
    }

    /// Computes the principal value of the inverse tangent of `self`.
    ///
    /// This function has two branch cuts:
    ///
    /// * `(-∞i, -i]`, continuous from the left.
    /// * `[i, ∞i)`, continuous from the right.
    ///
    /// The branch satisfies `-π/2 ≤ Re(atan(z)) ≤ π/2`.
    #[inline]
    pub fn atan(self) -> Self {
        // formula: arctan(z) = (ln(1+iz) - ln(1-iz))/(2i)
        let a = self.mul_add(Self::I, Self::ONE);
        let b = self.mul_add(Self::NEG_I, Self::ONE);

        // z/(2i) == -0.5i * z
        (a.ln() - b.ln()) * Self::imag(-V::HALF)
    }

    /// Computes the hyperbolic sine of `self`.
    #[inline]
    pub fn sinh(self) -> Self {
        // formula: sinh(a + bi) = sinh(a)cos(b) + i*cosh(a)sin(b)
        let (s, c) = self.im.sin_cos_p::<P>();
        let (sh, ch) = self.re.sinh_cosh_p::<P>();
        Self::new(sh * c, ch * s)
    }

    /// Computes the hyperbolic cosine of `self`.
    #[inline]
    pub fn cosh(self) -> Self {
        // formula: cosh(a + bi) = cosh(a)cos(b) + i*sinh(a)sin(b)
        let (s, c) = self.im.sin_cos_p::<P>();
        let (sh, ch) = self.re.sinh_cosh_p::<P>();
        Self::new(ch * c, sh * s)
    }

    /// Computes the hyperbolic tangent of `self`.
    #[inline]
    pub fn tanh(self) -> Self {
        // formula: tanh(a + bi) = (sinh(2a) + i*sin(2b))/(cosh(2a) + cos(2b))
        let (two_re, two_im) = (self.re + self.re, self.im + self.im);
        let (s, c) = two_im.sin_cos_p::<P>();
        let (sh, ch) = two_re.sinh_cosh_p::<P>();
        Self::new(sh + s, ch + c).unscale(ch + c)
    }

    /// Computes the principal value of inverse hyperbolic sine of `self`.
    ///
    /// This function has two branch cuts:
    ///
    /// * `(-∞i, -i)`, continuous from the left.
    /// * `(i, ∞i)`, continuous from the right.
    ///
    /// The branch satisfies `-π/2 ≤ Im(asinh(z)) ≤ π/2`.
    #[inline]
    pub fn asinh(self) -> Self {
        // formula: arcsinh(z) = ln(z + sqrt(1+z^2))
        //(self + (one + self * self).sqrt()).ln()
        (self + self.mul_add(self, Self::ONE).sqrt()).ln()
    }

    /// Computes the principal value of inverse hyperbolic cosine of `self`.
    ///
    /// This function has one branch cut:
    ///
    /// * `(-∞, 1)`, continuous from above.
    ///
    /// The branch satisfies `-π ≤ Im(acosh(z)) ≤ π` and `0 ≤ Re(acosh(z)) < ∞`.
    #[inline]
    pub fn acosh(self) -> Self {
        // formula: arccosh(z) = 2 ln(sqrt((z+1)/2) + sqrt((z-1)/2))
        let one_half = Self::real(V::HALF);

        let a = self.mul_add(one_half, one_half).sqrt();
        let b = self.mul_add(one_half, -one_half).sqrt();
        let half_res = (a + b).ln();

        half_res + half_res // res * 2
    }

    /// Computes the principal value of inverse hyperbolic tangent of `self`.
    ///
    /// This function has two branch cuts:
    ///
    /// * `(-∞, -1]`, continuous from above.
    /// * `[1, ∞)`, continuous from below.
    ///
    /// The branch satisfies `-π/2 ≤ Im(atanh(z)) ≤ π/2`.
    #[inline]
    pub fn atanh(self) -> Self {
        // formula: arctanh(z) = (ln(1+z) - ln(1-z))/2

        //if self == one {
        //    return Self::new(T::infinity(), T::zero());
        //} else if self == -one {
        //    return Self::new(-T::infinity(), T::zero());
        //}
        Self::real(V::HALF) * ((Self::ONE + self).ln() - (Self::ONE - self).ln())
    }

    /// Returns `1/self` using floating-point operations.
    ///
    /// This may be more accurate than the generic `self.inv()` in cases
    /// where `self.norm_sqr()` would overflow to ∞ or underflow to 0.
    #[inline(always)]
    pub fn finv(self) -> Self {
        let norm = Self::real(self.norm());
        // TODO: Maybe extract 1/n and multiply?
        (self.conj() / norm) / norm
    }

    /// Returns `self/other` using floating-point operations.
    ///
    /// This may be more accurate than the generic `Div` implementation in cases
    /// where `other.norm_sqr()` would overflow to ∞ or underflow to 0.
    #[inline(always)]
    pub fn fdiv(self, rhs: Self) -> Self {
        self * rhs.finv()
    }
}

impl<V: MathVector, P: Policy> core::ops::Add for Complex<V, P> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl<V: MathVector, P: Policy> core::ops::Sub for Complex<V, P> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl<V: MathVector, P: Policy> core::ops::Mul for Complex<V, P> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.re.mul_sube(rhs.re, self.im * rhs.im),
            self.re.mul_adde(rhs.im, self.im * rhs.re),
        )
    }
}

impl<V: MathVector, P: Policy> core::ops::Div for Complex<V, P> {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self::Output {
        let denom = rhs.re.mul_adde(rhs.re, rhs.im * rhs.im);
        Self::new(
            self.re.mul_adde(rhs.re, -self.im * rhs.im) / denom,
            self.im.mul_adde(rhs.re, self.re * rhs.im) / denom,
        )
    }
}

impl<V: MathVector, P: Policy> core::ops::Mul<V> for Complex<V, P> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: V) -> Self::Output {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

impl<V: MathVector, P: Policy> core::ops::Div<V> for Complex<V, P> {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: V) -> Self::Output {
        Self::new(self.re / rhs, self.im / rhs)
    }
}

impl<V: MathVector, P: Policy> core::ops::Neg for Complex<V, P> {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self::new(-self.re, -self.im)
    }
}
