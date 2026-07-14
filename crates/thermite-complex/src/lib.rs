#![no_std]

//! # SIMD complex numbers
//!
//! [`Complex<V>`] stores a real and an imaginary part, each an inner value `V`.
//! With `V` a Thermite [`FloatVector`](thermite::prelude::FloatVector) each lane
//! is an independent complex number (struct-of-arrays); with `V` an `f32`/`f64`
//! it is a complex scalar, which is the [`Element`](thermite::element::Element)
//! of the vector form.
//!
//! ```text
//! Complex<f32>        => a complex scalar
//! Complex<Vector<R>>  => LANES complex numbers, SIMD-parallel
//! ```
//!
//! `Complex<V>` implements the [`GenericVector`] -> [`FloatVector`] stack and the
//! `Specialized*Math` traits. [`CoreMath`], [`TranscendentalMath`] and
//! [`SpatialMath`] (with their `_p::<P>()` policy forms) then come from the same
//! blanket impls that serve `Vector<R>`:
//!
//! ```
//! use thermite::prelude::*;
//! use thermite::math::TranscendentalMath;
//! use thermite_complex::Complex;
//!
//! fn gaussian<V: FloatVector + TranscendentalMath>(x: V) -> V {
//!     (-(x * x)).exp()
//! }
//!
//! type V = Vector<f64>;
//!
//! // e^(-i^2) = e^1 = e
//! let z = gaussian(Complex::<V>::I);
//! assert!((z.re.extract::<0>() - core::f64::consts::E).abs() < 1e-12);
//! assert!(z.im.extract::<0>().abs() < 1e-12);
//! ```
//!
//! The operations whose result or argument is *real* (`norm`, `arg`, polar form,
//! real powers and bases) have no place in those families and get their own; see
//! [`specialized`] for [`ComplexVector`] and [`ComplexMath`].
//!
//! The inner `V` need not be a plain vector. Anything implementing [`ComplexValue`]
//! will do, including the other composites:
//!
//! ```text
//! Complex<Dual<V, N>>      => complex arithmetic carrying N derivatives  (`dual` feature)
//! Complex<Compensated<V>>  => complex arithmetic in double-double        (`compensated`)
//! ```
//!
//! # Ordering, sign and rounding
//!
//! C is neither ordered nor signed, but the vector traits require both:
//!
//! - Ordering ([`cmp_lt`] and friends, [`min`], [`max`], [`clamp`],
//!   [`arg_minmax`], the derived [`PartialOrd`]) is lexicographic by `(re, im)`.
//!   It is a tiebreak rule, not a statement about magnitudes.
//! - [`abs`] and [`signum`] are modulus-based: `$|z|$` (as a real complex) and
//!   `$z/|z|$`, preserving `abs(z) * signum(z) == z`. The spatial norms
//!   ([`l1_norm`], [`l2_norm`], [`hypot`]) are likewise the real quantities.
//! - The sign-bit ops ([`copysign`], [`mul_sign`], [`signed_zero`]) are
//!   componentwise. [`is_negative`]/[`is_positive`] report the sign of `re`, a
//!   mask having only one bit per lane.
//! - Rounding ([`floor`], [`ceil`], [`round`], [`trunc`], [`fract`]) is
//!   componentwise, and `%` is `z - trunc(z/w)*w` with that truncation. These
//!   satisfy the traits; they are not complex-analytic operations.
//!
//! [`RealMath`] is *not* implemented: `atan2`, `wrap_angle`, `step`, `smoothstep`
//! and the rest of that family are defined over an ordered field, so a
//! `V: RealMath` bound will not accept a complex vector. For the argument of `z`,
//! use [`ComplexMath::arg`], which returns the real vector it is.
//!
//! [`GenericVector`]: thermite::prelude::GenericVector
//! [`FloatVector`]: thermite::prelude::FloatVector
//! [`CoreMath`]: thermite::math::CoreMath
//! [`TranscendentalMath`]: thermite::math::TranscendentalMath
//! [`SpatialMath`]: thermite::math::SpatialMath
//! [`RealMath`]: thermite::math::RealMath
//! [`cmp_lt`]: thermite::prelude::PartialOrdVector::cmp_lt
//! [`min`]: thermite::prelude::NumericVector::min
//! [`max`]: thermite::prelude::NumericVector::max
//! [`clamp`]: thermite::prelude::NumericVector::clamp
//! [`arg_minmax`]: thermite::prelude::NumericVector::arg_minmax
//! [`abs`]: thermite::prelude::SignedVector::abs
//! [`signum`]: thermite::prelude::SignedVector::signum
//! [`l1_norm`]: thermite::math::SpatialMath::l1_norm
//! [`l2_norm`]: thermite::math::SpatialMath::l2_norm
//! [`hypot`]: thermite::math::SpatialMath::hypot
//! [`copysign`]: thermite::prelude::SignedVector::copysign
//! [`mul_sign`]: thermite::prelude::FloatVector::mul_sign
//! [`signed_zero`]: thermite::prelude::FloatVector::signed_zero
//! [`is_negative`]: thermite::prelude::SignedVector::is_negative
//! [`is_positive`]: thermite::prelude::SignedVector::is_positive
//! [`floor`]: thermite::prelude::FloatVector::floor
//! [`ceil`]: thermite::prelude::FloatVector::ceil
//! [`round`]: thermite::prelude::FloatVector::round
//! [`trunc`]: thermite::prelude::FloatVector::trunc
//! [`fract`]: thermite::prelude::FloatVector::fract

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};

use thermite::vector::ops::{MulAddAssignExt, MulAddExt, Square};

pub mod math;
pub mod specialized;
pub mod vector;

#[cfg(feature = "special")]
pub mod special;

pub use specialized::{ComplexMath, ComplexMathWithPolicy, ComplexVector, SpecializedComplexMath};
pub use vector::ComplexFloatVector;

/// A value usable as the real/imaginary storage of a [`Complex`].
///
/// Implemented for `f32`/`f64` and for every Thermite float
/// [`Vector`](thermite::prelude::Vector). The arithmetic below is written once
/// against it and serves both the element level (`Complex<f32>`) and the vector
/// level (`Complex<Vector<R>>`). The math library wants the stronger
/// [`ComplexFloatVector`].
pub trait ComplexValue:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + MulAddExt<Self, Self, Output = Self>
{
    /// The additive identity in this value type.
    const VAL_ZERO: Self;
    /// The multiplicative identity in this value type.
    const VAL_ONE: Self;

    /// Truncate towards zero. Used to give [`Complex`] a (componentwise) `Rem`.
    fn val_trunc(self) -> Self;
}

impl ComplexValue for f32 {
    const VAL_ZERO: Self = 0.0;
    const VAL_ONE: Self = 1.0;

    #[inline(always)]
    fn val_trunc(self) -> Self {
        thermite::register::FloatElement::trunc(self)
    }
}

impl ComplexValue for f64 {
    const VAL_ZERO: Self = 0.0;
    const VAL_ONE: Self = 1.0;

    #[inline(always)]
    fn val_trunc(self) -> Self {
        thermite::register::FloatElement::trunc(self)
    }
}

impl<R: thermite::register::FloatRegister> ComplexValue for thermite::prelude::Vector<R> {
    const VAL_ZERO: Self = <Self as thermite::prelude::NumericVector>::ZERO;
    const VAL_ONE: Self = <Self as thermite::prelude::NumericVector>::ONE;

    #[inline(always)]
    fn val_trunc(self) -> Self {
        thermite::prelude::FloatVector::trunc(self)
    }
}

/// `Complex<Dual<V, N>>`: a complex number whose parts each carry `N` derivative
/// components, giving forward-mode AD through the complex functions.
///
/// Everything here is written against [`ComplexValue`], which [`Dual`] satisfies,
/// so this impl is all it takes. Seeded along the real axis (`dz = 1`), the dual
/// parts of `f(z)` are `f'(z)` for holomorphic `f`.
///
/// [`Dual`]: thermite_dual::Dual
#[cfg(feature = "dual")]
impl<V: thermite_dual::DualValue, const N: usize> ComplexValue for thermite_dual::Dual<V, N> {
    const VAL_ZERO: Self = Self::ZERO;
    const VAL_ONE: Self = Self::ONE;

    #[inline(always)]
    fn val_trunc(self) -> Self {
        thermite_dual::DualValue::val_trunc(self)
    }
}

/// `Complex<Compensated<V>>`: a complex number whose parts are each a double-double,
/// roughly doubling the mantissa of the complex arithmetic and of every kernel built
/// on it.
///
/// [`Compensated`] carries no error term of its own through a complex multiply; the
/// compensation is per component, and the cross terms of `(a + bi)(c + di)` are
/// summed in double-double, which is where the precision comes from.
///
/// [`Compensated`]: thermite_compensated::Compensated
#[cfg(feature = "compensated")]
impl<V: thermite_compensated::ScalarValue> ComplexValue for thermite_compensated::Compensated<V> {
    const VAL_ZERO: Self = thermite_compensated::Compensated {
        value: V::SCALAR_ZERO,
        error: V::SCALAR_ZERO,
    };

    const VAL_ONE: Self = thermite_compensated::Compensated {
        value: V::SCALAR_ONE,
        error: V::SCALAR_ZERO,
    };

    // Truncating the folded value+error, as `Compensated`'s own `Rem` and
    // `FloatElement::trunc` do.
    #[inline(always)]
    fn val_trunc(self) -> Self {
        thermite_compensated::Compensated::new(self.value().scalar_trunc())
    }
}

/// A complex number `re + im*i`.
///
/// The derived [`PartialOrd`] is lexicographic on `(re, im)`, matching
/// [`PartialOrdVector`](thermite::prelude::PartialOrdVector). The [crate docs](crate)
/// cover the rest of the ordering/sign/rounding semantics.
#[derive(Debug, Default, Clone, Copy, PartialEq, PartialOrd)]
#[repr(C)]
pub struct Complex<V> {
    /// The real part.
    pub re: V,
    /// The imaginary part.
    pub im: V,
}

impl<V: ComplexValue> thermite::const_default::ConstDefault for Complex<V> {
    const DEFAULT: Self = Self::ZERO;
}

impl<V: ComplexValue> Complex<V> {
    /// Zero: `0 + 0i`.
    pub const ZERO: Self = Self::new(V::VAL_ZERO, V::VAL_ZERO);
    /// One: `1 + 0i`.
    pub const ONE: Self = Self::new(V::VAL_ONE, V::VAL_ZERO);
    /// The imaginary unit: `0 + 1i`.
    pub const I: Self = Self::new(V::VAL_ZERO, V::VAL_ONE);

    /// Creates a complex number with the given real and imaginary parts.
    #[inline(always)]
    pub const fn new(re: V, im: V) -> Self {
        Self { re, im }
    }

    /// Creates a complex number with the given real part and zero imaginary part.
    #[inline(always)]
    pub const fn real(re: V) -> Self {
        Self::new(re, V::VAL_ZERO)
    }

    /// Creates a complex number with zero real part and the given imaginary part.
    #[inline(always)]
    pub const fn imag(im: V) -> Self {
        Self::new(V::VAL_ZERO, im)
    }

    /// The complex conjugate: `re - im*i`.
    #[inline(always)]
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    /// The squared modulus `$|z|^2 = re^2 + im^2$`.
    ///
    /// Cheaper than the modulus (no square root), but it squares the range, so it
    /// overflows or underflows near the limits of the format.
    #[inline(always)]
    pub fn norm_sqr(self) -> V {
        self.re.mul_adde(self.re, self.im * self.im)
    }

    /// The multiplicative inverse `$1/z = \bar{z}/|z|^2$`.
    ///
    /// Inherits the range limits of [`norm_sqr`](Complex::norm_sqr); the scaled
    /// form is [`finv`](crate::ComplexMath::finv).
    #[inline(always)]
    pub fn inv(self) -> Self {
        self.conj() / self.norm_sqr()
    }
}

// --- Arithmetic: Complex op Complex ---

impl<V: ComplexValue> Neg for Complex<V> {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

impl<V: ComplexValue> Add for Complex<V> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl<V: ComplexValue> Sub for Complex<V> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl<V: ComplexValue> Mul for Complex<V> {
    type Output = Self;

    // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re.mul_sube(rhs.re, self.im * rhs.im),
            self.re.mul_adde(rhs.im, self.im * rhs.re),
        )
    }
}

impl<V: ComplexValue> Div for Complex<V> {
    type Output = Self;

    // (a + bi)/(c + di) = ((ac + bd) + (bc - ad)i) / (c^2 + d^2), taking one
    // reciprocal of the real denominator, so there is only one division.
    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        let denom = rhs.re.mul_adde(rhs.re, rhs.im * rhs.im);
        let inv = V::VAL_ONE / denom;

        Self::new(
            self.re.mul_adde(rhs.re, self.im * rhs.im) * inv,
            self.im.mul_sube(rhs.re, self.re * rhs.im) * inv,
        )
    }
}

// z % w = z - trunc(z/w)*w, truncating the quotient componentwise. Required by
// num_traits::NumOps for NumericVector; not a complex-analytic operation.
#[allow(clippy::suspicious_arithmetic_impl)]
impl<V: ComplexValue> Rem for Complex<V> {
    type Output = Self;

    #[inline(always)]
    fn rem(self, rhs: Self) -> Self {
        let q = self / rhs;
        let k = Complex::new(q.re.val_trunc(), q.im.val_trunc());

        k.nmul_adde(rhs, self) // self - k*rhs
    }
}

// --- Arithmetic: Complex op real value ---

impl<V: ComplexValue> Add<V> for Complex<V> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: V) -> Self {
        Self::new(self.re + rhs, self.im)
    }
}

impl<V: ComplexValue> Sub<V> for Complex<V> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: V) -> Self {
        Self::new(self.re - rhs, self.im)
    }
}

impl<V: ComplexValue> Mul<V> for Complex<V> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: V) -> Self {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

impl<V: ComplexValue> Div<V> for Complex<V> {
    type Output = Self;

    // single reciprocal, then multiply through
    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline(always)]
    fn div(self, rhs: V) -> Self {
        // single reciprocal, then multiply through
        let inv = V::VAL_ONE / rhs;

        Self::new(self.re * inv, self.im * inv)
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl<V: ComplexValue> Rem<V> for Complex<V> {
    type Output = Self;

    #[inline(always)]
    fn rem(self, rhs: V) -> Self {
        let q = self / rhs;
        let k = Complex::new(q.re.val_trunc(), q.im.val_trunc());

        k.nmul_adde(Complex::real(rhs), self)
    }
}

// --- Fused multiply-add by a real value ---
//
// z*r + w with real r is a single fused op per component (re*r + w.re,
// im*r + w.im). Unlike the complex-by-complex form it is therefore a true
// single-rounding FMA whenever the inner type has one.
macro_rules! complex_real_fma {
    ($($name:ident),* $(,)?) => {
        $(
            #[inline(always)]
            fn $name(self, a: V, b: Self) -> Self {
                Self::new(self.re.$name(a, b.re), self.im.$name(a, b.im))
            }
        )*
    };
}

#[rustfmt::skip]
impl<V: ComplexValue> MulAddExt<V, Self> for Complex<V> {
    type Output = Self;

    const HAS_TRUE_FMA: bool = <V as MulAddExt<V, V>>::HAS_TRUE_FMA;

    complex_real_fma!(mul_add, mul_sub, nmul_add, nmul_sub, mul_adde, mul_sube, nmul_adde, nmul_sube);
}

// --- Assignment variants ---

macro_rules! impl_assign {
    ($($assign_trait:ident::$assign_method:ident => $op_trait:ident::$op_method:ident),* $(,)?) => {$(
        impl<V: ComplexValue, T> $assign_trait<T> for Complex<V>
        where
            Self: $op_trait<T, Output = Self>,
        {
            #[inline(always)]
            fn $assign_method(&mut self, rhs: T) {
                *self = $op_trait::$op_method(*self, rhs);
            }
        }
    )*};
}

#[rustfmt::skip]
impl_assign! {
    AddAssign::add_assign => Add::add,
    SubAssign::sub_assign => Sub::sub,
    MulAssign::mul_assign => Mul::mul,
    DivAssign::div_assign => Div::div,
    RemAssign::rem_assign => Rem::rem,
}

// --- Fused multiply-add ---
//
// (a + bi)(c + di) + (e + fi) expands to
//
//   re = a*c - b*d + e  =  fnma(b, d, fma(a, c, e))
//   im = a*d + b*c + f  =  fma(a, d, fma(b, c, f))
//
// i.e. two nested FMAs of the inner type per component. Composing the complex Mul
// and Add instead would round the product first. Each component still rounds more
// than once, so HAS_TRUE_FMA is false.

// The eight methods are the (product sign, addend sign) pairs over the exact or
// the estimating inner FMA. Both negations fold into the inner FMA's sign bits.
macro_rules! complex_mul_add {
    ($($name:ident => $neg_self:expr, $neg_addend:expr, $fma:ident, $nfma:ident);* $(;)?) => {
        $(
            #[inline(always)]
            fn $name(self, a: Self, b: Self) -> Self {
                let p = if $neg_self { -self } else { self };
                let c = if $neg_addend { -b } else { b };

                // re = p.re*a.re - p.im*a.im + c.re ; im = p.re*a.im + p.im*a.re + c.im
                Self::new(
                    p.im.$nfma(a.im, p.re.$fma(a.re, c.re)),
                    p.re.$fma(a.im, p.im.$fma(a.re, c.im)),
                )
            }
        )*
    };
}

#[rustfmt::skip]
impl<V: ComplexValue> MulAddExt<Self, Self> for Complex<V> {
    type Output = Self;

    // A complex "FMA" rounds each component several times whatever the inner FMA
    // does. It is never a single-rounding operation.
    const HAS_TRUE_FMA: bool = false;

    complex_mul_add! {
        mul_add   => false, false, mul_add,  nmul_add;
        mul_sub   => false, true,  mul_add,  nmul_add;
        nmul_add  => true,  false, mul_add,  nmul_add;
        nmul_sub  => true,  true,  mul_add,  nmul_add;
        mul_adde  => false, false, mul_adde, nmul_adde;
        mul_sube  => false, true,  mul_adde, nmul_adde;
        nmul_adde => true,  false, mul_adde, nmul_adde;
        nmul_sube => true,  true,  mul_adde, nmul_adde;
    }
}

#[rustfmt::skip]
impl<V: ComplexValue, A, B> MulAddAssignExt<A, B> for Complex<V>
where
    Self: MulAddExt<A, B, Output = Self>,
{
    #[inline(always)] fn mul_add_assign(&mut self, a: A, b: B) { *self = self.mul_add(a, b); }
    #[inline(always)] fn mul_sub_assign(&mut self, a: A, b: B) { *self = self.mul_sub(a, b); }
    #[inline(always)] fn nmul_add_assign(&mut self, a: A, b: B) { *self = self.nmul_add(a, b); }
    #[inline(always)] fn nmul_sub_assign(&mut self, a: A, b: B) { *self = self.nmul_sub(a, b); }
    #[inline(always)] fn mul_adde_assign(&mut self, a: A, b: B) { *self = self.mul_adde(a, b); }
    #[inline(always)] fn mul_sube_assign(&mut self, a: A, b: B) { *self = self.mul_sube(a, b); }
    #[inline(always)] fn nmul_adde_assign(&mut self, a: A, b: B) { *self = self.nmul_adde(a, b); }
    #[inline(always)] fn nmul_sube_assign(&mut self, a: A, b: B) { *self = self.nmul_sube(a, b); }
}

impl<V: ComplexValue> Square for Complex<V> {
    type Output = Self;

    // z^2 = (re^2 - im^2) + 2*re*im*i. The imaginary part is one add and one
    // multiply, versus the FMA over two products a general self*self would take.
    #[inline(always)]
    fn square(self) -> Self {
        Self::new(
            self.re.mul_sube(self.re, self.im * self.im),
            (self.re + self.re) * self.im,
        )
    }
}

// --- Iterator reductions ---

impl<V: ComplexValue> core::iter::Sum for Complex<V> {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |a, b| a + b)
    }
}

impl<V: ComplexValue> core::iter::Product for Complex<V> {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |a, b| a * b)
    }
}

// --- Float constants (real-valued) ---

macro_rules! impl_float_consts {
    ($($name:ident),* $(,)?) => {
        impl<V: ComplexValue + thermite::math::FloatConsts> thermite::math::FloatConsts for Complex<V> {
            $(const $name: Self = Self::real(<V as thermite::math::FloatConsts>::$name);)*
        }
    };
}

impl_float_consts!(
    NEG_ZERO,
    E,
    EULER_GAMMA,
    PI_SQUARED,
    PI_CUBED,
    PI_FOURTH,
    FRAC_1_PI,
    FRAC_1_SQRT_2,
    FRAC_1_SQRT_3,
    FRAC_2_PI,
    FRAC_1_SQRT_PI,
    FRAC_2_SQRT_PI,
    FRAC_SQRT_PI_2,
    FRAC_1_SQRT_TAU,
    FRAC_PI_2,
    FRAC_PI_3,
    FRAC_PI_4,
    FRAC_PI_6,
    FRAC_PI_8,
    FRAC_PI_180,
    FRAC_180_PI,
    LN_2,
    LN_10,
    LN_PI,
    FRAC_LN_PI_2,
    LOG2_10,
    LOG2_E,
    LOG10_2,
    LOG10_E,
    PI,
    SQRT_2,
    SQRT_3,
    SQRT_E,
    EPSILON,
    SQRT_EPSILON,
    FOURTH_ROOT_EPSILON,
    TAU,
    SQRT_FRAC_PI_2,
    SQRT_TAU,
    PHI,
    FRAC_1_3,
    FRAC_2_3,
    FRAC_1_4,
    FRAC_1_6,
    FRAC_NEG_1_E
);
