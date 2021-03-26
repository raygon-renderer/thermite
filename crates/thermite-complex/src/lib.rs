//! Complex Number Vectors

#![no_std]
use thermite::*;

use core::{
    fmt,
    marker::PhantomData,
    ops::{Add, Div, Mul, Neg, Sub},
};

/// A vectorized (SoA) complex number in Cartesian form.
pub struct Complex<S: Simd, V: SimdFloatVector<S>, P: Policy = policies::Performance> {
    /// Real part
    pub re: V,
    /// Imaginary part
    pub im: V,
    _simd: PhantomData<(S, P)>,
}

impl<S: Simd, V: SimdFloatVector<S>, P: Policy> Clone for Complex<S, V, P> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<S: Simd, V: SimdFloatVector<S>, P: Policy> Copy for Complex<S, V, P> {}

impl<S: Simd, V: SimdFloatVector<S>, P: Policy> fmt::Debug for Complex<S, V, P>
where
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Complex")
            .field("re", &self.re)
            .field("im", &self.im)
            .finish()
    }
}

#[dispatch(S)]
impl<S: Simd, V: SimdFloatVector<S>, P: Policy> Complex<S, V, P> {
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

    /// Create a new Complex `a+0i`
    #[inline(always)]
    pub fn real(re: V) -> Self {
        Self::new(re, V::zero())
    }

    /// Create a new Complex `0+bi`
    #[inline(always)]
    pub fn imag(im: V) -> Self {
        Self::new(V::zero(), im)
    }

    /// Returns imaginary unit
    #[inline(always)]
    pub fn i() -> Self {
        Self::new(V::zero(), V::one())
    }

    /// Return negative imaginary unit
    #[inline(always)]
    pub fn neg_i() -> Self {
        Self::new(V::zero(), V::neg_one())
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

#[dispatch(S)]
impl<S: Simd, V: SimdFloatVector<S>, P: Policy> Complex<S, V, P>
where
    V: SimdVectorizedMath<S>,
{
    /// Returns the square of the norm
    #[inline(always)]
    pub fn norm_sqr(self) -> V {
        self.re.mul_adde(self.re, self.im * self.im)
    }

    /// Calculate |self|
    #[inline(always)]
    pub fn norm(self) -> V {
        self.re.hypot_p::<P>(self.im)
    }

    /// Multiplies `self` by the scalar `t`.
    #[inline(always)]
    pub fn scale(mut self, t: V) -> Self {
        self.re *= t;
        self.im *= t;
        self
    }

    /// Divides `self` by the scalar `t`.
    ///
    /// NOTE: This will result in undefined values if `t` is zero.
    #[inline(always)]
    pub fn unscale(self, t: V) -> Self {
        self.scale(t.reciprocal_p::<P>())
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

    /// Returns `self * m + a`
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
    #[inline(always)]
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
    #[inline(always)]
    pub fn ln(self) -> Self {
        // formula: ln(z) = ln|z| + i*arg(z)
        let (r, theta) = self.to_polar();
        Self::new(r.ln_p::<P>(), theta)
    }

    /// Computes the principal value of the square root of `self`.
    #[inline(always)]
    pub fn sqrt(self) -> Self {
        // formula: sqrt(r e^(it)) = sqrt(r) e^(it/2)
        let (r, theta) = self.to_polar();
        Self::from_polar(r.sqrt(), theta * V::splat_as(0.5))
    }

    /// Computes the principal value of the cube root of `self`.
    ///
    /// Note that this does not match the usual result for the cube root of
    /// negative real numbers. For example, the real cube root of `-8` is `-2`,
    /// but the principal complex cube root of `-8` is `1 + i√3`.
    #[inline(always)]
    pub fn cbrt(self) -> Self {
        // formula: cbrt(r e^(it)) = cbrt(r) e^(it/3)
        let (r, theta) = self.to_polar();
        // 1/3 isn't well-represented in float, so an exact inverse can't work with all precisions
        Self::from_polar(r.cbrt(), theta / V::splat_as(3))
    }

    /// Raises `self` to a floating point power.
    pub fn powf(self, exp: V) -> Self {
        // formula: x^y = (ρ e^(i θ))^y = ρ^y e^(i θ y)
        // = from_polar(ρ^y, θ y)
        let (r, theta) = self.to_polar();
        Self::from_polar(r.powf_p::<P>(exp), theta * exp)
    }

    /// Returns the logarithm of `self` with respect to an arbitrary base.
    #[inline(always)]
    pub fn log(self, base: V) -> Self {
        // formula: log_y(x) = log_y(ρ e^(i θ))
        // = log_y(ρ) + log_y(e^(i θ)) = log_y(ρ) + ln(e^(i θ)) / ln(y)
        // = log_y(ρ) + i θ / ln(y)
        let (r, theta) = self.to_polar();
        let d = V::one() / base.ln_p::<P>();
        Self::new(r.ln_p::<P>() * d, theta * d)
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
            r.powf_p::<P>(exp.re) * (-exp.im * theta).exp_p::<P>(),
            exp.im.mul_adde(r.ln_p::<P>(), exp.re * theta),
        )
    }

    /// Raises a floating point number to the complex power `self`.
    #[inline(always)]
    pub fn expf(self, base: V) -> Self {
        // formula: x^(a+bi) = x^a x^bi = x^a e^(b ln(x) i)
        // = from_polar(x^a, b ln(x))
        Self::from_polar(base.powf_p::<P>(self.re), self.im * base.ln_p::<P>())
    }
    /// Computes the sine of `self`.
    #[inline(always)]
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
        Self::new(s, two_im.sinh()).unscale(c + two_im.cosh_p::<P>())
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
        Self::neg_i() * self.mul_add(Self::i(), self.mul_add(-self, Self::one()).sqrt()).ln()
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
        Self::neg_i() * Self::i().mul_add(self.mul_add(-self, Self::one()).sqrt(), self).ln()
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
        let one = Self::one();

        let a = self.mul_add(Self::i(), one);
        let b = self.mul_add(Self::neg_i(), one);

        // z/(2i) == -0.5i * z
        (a.ln() - b.ln()) * Self::imag(V::splat_as(-0.5))
    }

    /// Computes the hyperbolic sine of `self`.
    #[inline]
    pub fn sinh(self) -> Self {
        // formula: sinh(a + bi) = sinh(a)cos(b) + i*cosh(a)sin(b)
        let (s, c) = self.im.sin_cos_p::<P>();
        Self::new(self.re.sinh_p::<P>() * c, self.re.cosh_p::<P>() * s)
    }

    /// Computes the hyperbolic cosine of `self`.
    #[inline]
    pub fn cosh(self) -> Self {
        // formula: cosh(a + bi) = cosh(a)cos(b) + i*sinh(a)sin(b)
        let (s, c) = self.im.sin_cos_p::<P>();
        Self::new(self.re.cosh_p::<P>() * c, self.re.sinh_p::<P>() * s)
    }

    /// Computes the hyperbolic tangent of `self`.
    #[inline]
    pub fn tanh(self) -> Self {
        // formula: tanh(a + bi) = (sinh(2a) + i*sin(2b))/(cosh(2a) + cos(2b))
        let (two_re, two_im) = (self.re + self.re, self.im + self.im);
        let (s, c) = two_im.sin_cos_p::<P>();
        Self::new(two_re.sinh_p::<P>(), s).unscale(two_re.cosh_p::<P>() + c)
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
        (self + self.mul_add(self, Self::one()).sqrt()).ln()
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
        let one_half = Self::real(V::splat_as(0.5));

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
        let one = Self::one();
        let one_half = Self::real(V::splat_as(0.5));
        //if self == one {
        //    return Self::new(T::infinity(), T::zero());
        //} else if self == -one {
        //    return Self::new(-T::infinity(), T::zero());
        //}
        one_half * ((one + self).ln() - (one - self).ln())
    }

    /// Returns `1/self` using floating-point operations.
    ///
    /// This may be more accurate than the generic `self.inv()` in cases
    /// where `self.norm_sqr()` would overflow to ∞ or underflow to 0.
    #[inline]
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

#[dispatch(S)]
impl<S: Simd, V: SimdFloatVector<S>, P: Policy> Add<Self> for Complex<S, V, P> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

#[dispatch(S)]
impl<S: Simd, V: SimdFloatVector<S>, P: Policy> Sub<Self> for Complex<S, V, P> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

#[dispatch(S)]
impl<S: Simd, V: SimdFloatVector<S>, P: Policy> Mul<Self> for Complex<S, V, P> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re.mul_sube(rhs.re, self.im * rhs.im),
            self.re.mul_adde(rhs.im, self.im * rhs.re),
        )
    }
}

#[dispatch(S)]
impl<S: Simd, V: SimdFloatVector<S>, P: Policy> Div<Self> for Complex<S, V, P>
where
    V: SimdVectorizedMath<S>,
{
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        let re = self.re.mul_adde(rhs.re, self.im * rhs.im);
        let im = self.im.mul_sube(rhs.re, self.re * rhs.im);

        Self::new(re, im).unscale(rhs.norm_sqr())
    }
}

#[dispatch(S)]
impl<S: Simd, V: SimdFloatVector<S>, P: Policy> Neg for Complex<S, V, P> {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}
