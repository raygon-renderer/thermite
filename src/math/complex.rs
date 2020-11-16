//! Complex Number Vectors

use crate::*;

use std::marker::PhantomData;

#[derive(Debug, Clone, Copy)]
pub struct Complex<S: Simd, V: SimdFloatVector<S>> {
    pub re: V,
    pub im: V,
    _simd: PhantomData<S>,
}

impl<S: Simd, V: SimdFloatVector<S>> Complex<S, V> {
    #[inline(always)]
    pub fn new(re: V, im: V) -> Self {
        Self {
            re,
            im,
            _simd: PhantomData,
        }
    }

    /// Creates a new Complex with all lanes of `re` and `im` set to the inputs, respectively.
    #[inline(always)]
    pub fn splat(re: V::Element, im: V::Element) -> Self {
        Self::new(V::splat(re), V::splat(im))
    }

    /// Create a new Complex
    #[inline(always)]
    pub fn real(re: V) -> Self {
        Self::new(re, V::zero())
    }

    /// Returns imaginary unit
    #[inline(always)]
    pub fn i() -> Self {
        Self::new(V::zero(), V::one())
    }

    /// real(1)
    #[inline(always)]
    pub fn one() -> Self {
        Self::real(V::one())
    }

    /// real(0)
    #[inline(always)]
    pub fn zero() -> Self {
        Self::new(V::zero(), V::zero())
    }
}

impl<S: Simd, V: SimdFloatVector<S>> Complex<S, V>
where
    V: SimdVectorizedMath<S>,
{
    /// Returns the square of the norm
    #[inline(always)]
    pub fn norm_sqr(self) -> V {
        self.re.mul_add(self.re, self.im * self.im)
    }

    /// Calculate |self|
    #[inline(always)]
    pub fn norm(self) -> V {
        self.re.hypot(self.im)
    }

    /// Multiplies `self` by the scalar `t`.
    #[inline(always)]
    pub fn scale(mut self, t: V) -> Self {
        self.re *= t;
        self.im *= t;
        self
    }

    /// Divides `self` by the scalar `t`.
    #[inline(always)]
    pub fn unscale(self, t: V) -> Self {
        self.scale(V::one() / t)
    }

    /// Returns the complex conjugate. i.e. `re - i im`
    #[inline(always)]
    pub fn conj(mut self) -> Self {
        self.im = -self.im;
        self
    }

    /// Returns `1/self`
    #[inline(always)]
    pub fn inv(self) -> Self {
        self.unscale(self.norm_sqr()).conj()
    }

    pub fn mul_add(self, m: Self, a: Self) -> Self {
        Self::new(
            self.im.nmul_add(m.im, self.re.mul_add(m.re, a.re)),
            self.re.mul_add(m.im, self.im.mul_add(m.re, a.im)),
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
        self.im.atan2(self.re)
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
        let (s, c) = theta.sin_cos();
        Self::new(r * c, r * s)
    }

    /// Computes `e^(self)`, where `e` is the base of the natural logarithm.
    #[inline(always)]
    pub fn exp(self) -> Self {
        // formula: e^(a + bi) = e^a (cos(b) + i*sin(b))
        // = from_polar(e^a, b)
        Self::from_polar(self.re.exp(), self.im)
    }

    /// Computes the principal value of natural logarithm of `self`.
    ///
    /// This function has one branch cut:
    ///
    /// * `(-∞, 0]`, continuous from above.
    ///
    /// The branch satisfies `-π ≤ arg(ln(z)) ≤ π`.
    pub fn ln(self) -> Self {
        // formula: ln(z) = ln|z| + i*arg(z)
        let (r, theta) = self.to_polar();
        Self::new(r.ln(), theta)
    }

    pub fn sqrt(self) -> Self {
        // TODO: Convert reference to branchless
        unimplemented!()
    }

    pub fn cbrt(self) -> Self {
        // TODO: Convert reference to branchless
        unimplemented!()
    }

    /// Raises `self` to a floating point power.
    pub fn powf(self, exp: V) -> Self {
        // formula: x^y = (ρ e^(i θ))^y = ρ^y e^(i θ y)
        // = from_polar(ρ^y, θ y)
        let (r, theta) = self.to_polar();
        Self::from_polar(r.powf(exp), theta * exp)
    }

    /// Returns the logarithm of `self` with respect to an arbitrary base.
    #[inline(always)]
    pub fn log(self, base: V) -> Self {
        // formula: log_y(x) = log_y(ρ e^(i θ))
        // = log_y(ρ) + log_y(e^(i θ)) = log_y(ρ) + ln(e^(i θ)) / ln(y)
        // = log_y(ρ) + i θ / ln(y)
        let (r, theta) = self.to_polar();
        let d = V::one() / base.ln();
        Self::new(r.ln() * d, theta * d)
    }

    /// Raises `self` to a complex power.
    #[inline(always)]
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
            r.powf(exp.re) * (-exp.im * theta).exp(),
            exp.re * theta + exp.im * r.ln(),
        )
    }

    /// Raises a floating point number to the complex power `self`.
    #[inline(always)]
    pub fn expf(self, base: V) -> Self {
        // formula: x^(a+bi) = x^a x^bi = x^a e^(b ln(x) i)
        // = from_polar(x^a, b ln(x))
        Self::from_polar(base.powf(self.re), self.im * base.ln())
    }
    /// Computes the sine of `self`.
    #[inline(always)]
    pub fn sin(self) -> Self {
        // formula: sin(a + bi) = sin(a)cosh(b) + i*cos(a)sinh(b)
        let (s, c) = self.re.sin_cos();
        Self::new(s * self.im.cosh(), c * self.im.sinh())
    }

    /// Computes the cosine of `self`.
    #[inline]
    pub fn cos(self) -> Self {
        // formula: cos(a + bi) = cos(a)cosh(b) - i*sin(a)sinh(b)
        let (s, c) = self.re.sin_cos();
        Self::new(c * self.im.cosh(), -s * self.im.sinh())
    }

    /// Computes the tangent of `self`.
    #[inline]
    pub fn tan(self) -> Self {
        // formula: tan(a + bi) = (sin(2a) + i*sinh(2b))/(cos(2a) + cosh(2b))
        let (two_re, two_im) = (self.re + self.re, self.im + self.im);
        let (s, c) = two_re.sin_cos();
        Self::new(s, two_im.sinh()).unscale(c + two_im.cosh())
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
        let i = Self::i();
        -i * ((Self::one() - self * self).sqrt() + i * self).ln()
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
        let i = Self::i();
        -i * (i * (Self::one() - self * self).sqrt() + self).ln()
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
        let i = Self::i();
        let one = Self::one();
        let two = one + one;
        //if self == i {
        //    return Self::new(T::zero(), T::infinity());
        //} else if self == -i {
        //    return Self::new(T::zero(), -T::infinity());
        //}
        ((one + i * self).ln() - (one - i * self).ln()) / (two * i)
    }

    /// Computes the hyperbolic sine of `self`.
    #[inline]
    pub fn sinh(self) -> Self {
        // formula: sinh(a + bi) = sinh(a)cos(b) + i*cosh(a)sin(b)
        let (s, c) = self.im.sin_cos();
        Self::new(self.re.sinh() * c, self.re.cosh() * s)
    }

    /// Computes the hyperbolic cosine of `self`.
    #[inline]
    pub fn cosh(self) -> Self {
        // formula: cosh(a + bi) = cosh(a)cos(b) + i*sinh(a)sin(b)
        let (s, c) = self.im.sin_cos();
        Self::new(self.re.cosh() * c, self.re.sinh() * s)
    }

    /// Computes the hyperbolic tangent of `self`.
    #[inline]
    pub fn tanh(self) -> Self {
        // formula: tanh(a + bi) = (sinh(2a) + i*sin(2b))/(cosh(2a) + cos(2b))
        let (two_re, two_im) = (self.re + self.re, self.im + self.im);
        let (s, c) = two_im.sin_cos();
        Self::new(two_re.sinh(), s).unscale(two_re.cosh() + c)
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
        let one = Self::one();
        (self + (one + self * self).sqrt()).ln()
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
        let one = Self::one();
        let two = one + one;
        two * (((self + one) / two).sqrt() + ((self - one) / two).sqrt()).ln()
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
        let one = Self::one();
        let two = one + one;
        //if self == one {
        //    return Self::new(T::infinity(), T::zero());
        //} else if self == -one {
        //    return Self::new(-T::infinity(), T::zero());
        //}
        ((one + self).ln() - (one - self).ln()) / two
    }

    /// Returns `1/self` using floating-point operations.
    ///
    /// This may be more accurate than the generic `self.inv()` in cases
    /// where `self.norm_sqr()` would overflow to ∞ or underflow to 0.
    #[inline]
    pub fn finv(self) -> Self {
        let norm = Self::real(self.norm());
        self.conj() / norm / norm
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

impl<S: Simd, V: SimdFloatVector<S>> Add<Self> for Complex<S, V> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl<S: Simd, V: SimdFloatVector<S>> Sub<Self> for Complex<S, V> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl<S: Simd, V: SimdFloatVector<S>> Mul<Self> for Complex<S, V> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re.mul_sub(rhs.re, self.im * rhs.im),
            self.re.mul_add(rhs.im, self.im * rhs.re),
        )
    }
}

impl<S: Simd, V: SimdFloatVector<S>> Div<Self> for Complex<S, V>
where
    V: SimdVectorizedMath<S>,
{
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        let re = self.re.mul_add(rhs.re, self.im * rhs.im);
        let im = self.im.mul_sub(rhs.re, self.re * rhs.im);

        Self::new(re, im).unscale(rhs.norm_sqr())
    }
}

impl<S: Simd, V: SimdFloatVector<S>> Neg for Complex<S, V> {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}
