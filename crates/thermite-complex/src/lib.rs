#![doc = include_str!("../README.md")]
#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};

use thermite::tribool::{self, Tribool};
use thermite::vector::ops::{MulAddAssignExt, MulAddExt, Square};

pub mod math;
mod vector;

pub use crate::vector::RealFloatVector;

/// Everything needed to work with [`Complex`], in one glob.
///
/// ```
/// use thermite::prelude::*;
/// use thermite_complex::prelude::*;
/// ```
///
/// [`Complex`] itself lives at the crate root, being the type this crate is about;
/// the traits are spread across [`math`] and its submodules, and that layout is an
/// implementation detail. Import from here.
pub mod prelude {
    pub use crate::RealFloatVector;
    pub use crate::math::specialized::{ComplexVector, SpecializedComplexMath};
    pub use crate::math::{ComplexMath, ComplexMathWithPolicy};
    pub use crate::{Complex, RealValue};

    #[cfg(feature = "special")]
    pub use crate::math::special::{ComplexSpecialMath, ComplexSpecialMathWithPolicy, SpecializedComplexSpecialMath};
}

/// A value usable as the real/imaginary storage of a [`Complex`].
///
/// Implemented for `f32`/`f64` and for every Thermite float
/// [`Vector`](thermite::prelude::Vector). The arithmetic below is written once
/// against it and serves both the element level (`Complex<f32>`) and the vector
/// level (`Complex<Vector<R>>`). The math library wants the stronger
/// [`RealFloatVector`].
pub trait RealValue:
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

impl RealValue for f32 {
    const VAL_ZERO: Self = 0.0;
    const VAL_ONE: Self = 1.0;

    #[inline(always)]
    fn val_trunc(self) -> Self {
        thermite::register::FloatElement::trunc(self)
    }
}

impl RealValue for f64 {
    const VAL_ZERO: Self = 0.0;
    const VAL_ONE: Self = 1.0;

    #[inline(always)]
    fn val_trunc(self) -> Self {
        thermite::register::FloatElement::trunc(self)
    }
}

impl<R: thermite::register::FloatRegister> RealValue for thermite::prelude::Vector<R> {
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
/// Everything here is written against [`RealValue`], which [`Dual`] satisfies,
/// so this impl is all it takes. Seeded along the real axis (`dz = 1`), the dual
/// parts of `f(z)` are `f'(z)` for holomorphic `f`.
///
/// [`Dual`]: thermite_dual::Dual
#[cfg(feature = "dual")]
impl<V: thermite_dual::DualValue, const N: usize> RealValue for thermite_dual::Dual<V, N> {
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
impl<V: thermite_compensated::ScalarValue> RealValue for thermite_compensated::Compensated<V> {
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

impl<V: RealValue> thermite::const_default::ConstDefault for Complex<V> {
    const DEFAULT: Self = Self::ZERO;
}

impl<V: RealValue> Complex<V> {
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
    /// form is [`finv`](crate::math::ComplexMath::finv).
    #[inline(always)]
    pub fn inv(self) -> Self {
        self.conj() / self.norm_sqr()
    }
}

// --- Arithmetic: Complex op Complex ---

impl<V: RealValue> Neg for Complex<V> {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

impl<V: RealValue> Add for Complex<V> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl<V: RealValue> Sub for Complex<V> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl<V: RealValue> Mul for Complex<V> {
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

impl<V: RealValue> Div for Complex<V> {
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
impl<V: RealValue> Rem for Complex<V> {
    type Output = Self;

    #[inline(always)]
    fn rem(self, rhs: Self) -> Self {
        let q = self / rhs;
        let k = Complex::new(q.re.val_trunc(), q.im.val_trunc());

        k.nmul_adde(rhs, self) // self - k*rhs
    }
}

// --- Arithmetic: Complex op real value ---

impl<V: RealValue> Add<V> for Complex<V> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: V) -> Self {
        Self::new(self.re + rhs, self.im)
    }
}

impl<V: RealValue> Sub<V> for Complex<V> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: V) -> Self {
        Self::new(self.re - rhs, self.im)
    }
}

impl<V: RealValue> Mul<V> for Complex<V> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: V) -> Self {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

impl<V: RealValue> Div<V> for Complex<V> {
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
impl<V: RealValue> Rem<V> for Complex<V> {
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
impl<V: RealValue> MulAddExt<V, Self> for Complex<V> {
    type Output = Self;

    const HAS_NATIVE_FMA: Tribool = <V as MulAddExt<V, V>>::HAS_NATIVE_FMA;

    complex_real_fma!(mul_add, mul_sub, nmul_add, nmul_sub, mul_adde, mul_sube, nmul_adde, nmul_sube);
}

// --- Assignment variants ---

macro_rules! impl_assign {
    ($($assign_trait:ident::$assign_method:ident => $op_trait:ident::$op_method:ident),* $(,)?) => {$(
        impl<V: RealValue, T> $assign_trait<T> for Complex<V>
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
// than once, so HAS_NATIVE_FMA is False.

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
impl<V: RealValue> MulAddExt<Self, Self> for Complex<V> {
    type Output = Self;

    // A complex "FMA" rounds each component several times whatever the inner FMA
    // does. It is never a single-rounding operation.
    const HAS_NATIVE_FMA: Tribool = tribool::False;

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
impl<V: RealValue, A, B> MulAddAssignExt<A, B> for Complex<V>
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

impl<V: RealValue> Square for Complex<V> {
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

impl<V: RealValue> core::iter::Sum for Complex<V> {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |a, b| a + b)
    }
}

impl<V: RealValue> core::iter::Product for Complex<V> {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |a, b| a * b)
    }
}

// --- Float constants (real-valued) ---

macro_rules! impl_float_consts {
    ($($name:ident),* $(,)?) => {
        impl<V: RealValue + thermite::math::FloatConsts> thermite::math::FloatConsts for Complex<V> {
            $(const $name: Self = Self::real(<V as thermite::math::FloatConsts>::$name);)*
        }
    };
}

thermite::for_each_float_const!(impl_float_consts);
