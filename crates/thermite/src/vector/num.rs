#![allow(missing_docs)]

use core::ops::Deref;

use crate::{
    math::{FloatConsts, Math},
    register::{FloatElement, FloatRegister},
    vector::{
        Vector,
        generic::{
            BitshiftVector, FloatVector, GenericCastMask, GenericSelectable, GenericVector, NumericVector, SignedVector,
        },
    },
};

#[repr(transparent)]
pub struct NumVector<V: GenericVector>(pub V);

impl<V: GenericVector> Copy for NumVector<V> {}
impl<V: GenericVector> Clone for NumVector<V> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<V: GenericVector> Deref for NumVector<V> {
    type Target = V;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<V: GenericVector> GenericSelectable for NumVector<V>
where
    V: GenericSelectable,
{
    type SelectableMask = V::SelectableMask;

    #[inline(always)]
    fn select<M>(mask: M, t: Self, f: Self) -> Self
    where
        Self::SelectableMask: GenericCastMask<M>,
    {
        Self(<V as GenericSelectable>::select(mask, t.0, f.0))
    }
}

macro_rules! fwd_ops {
    (BINARY: $bound:ident => $($trait:ident::$op:ident),* $(,)?) => {paste::paste! {$(
        impl<V: $bound> core::ops::$trait for NumVector<V> {
            type Output = Self; #[inline(always)] fn $op(self, rhs: Self) -> Self::Output {
                Self(core::ops::$trait::$op(self.0, rhs.0))
            }
        }
        impl<V: $bound> core::ops::[<$trait Assign>] for NumVector<V> {
            #[inline(always)] fn [<$op _assign>](&mut self, rhs: Self) {
                core::ops::[<$trait Assign>]::[<$op _assign>](&mut self.0, rhs.0);
            }
        }
    )*}};
    (UNARY: $bound:ident => $($trait:ident::$op:ident),* $(,)?) => {paste::paste! {$(
        impl<V: $bound> core::ops::$trait for NumVector<V> {
            type Output = Self; #[inline(always)] fn $op(self) -> Self::Output {
                Self(core::ops::$trait::$op(self.0))
            }
        }
    )*}};
    (SHIFTS: $bound:ident => $($trait:ident::$op:ident),* $(,)?) => {paste::paste! {$(
        impl<V: $bound> core::ops::$trait<u32> for NumVector<V> {
            type Output = Self; #[inline(always)] fn $op(self, rhs: u32) -> Self::Output {
                Self(core::ops::$trait::$op(self.0, rhs))
            }
        }
        impl<V: $bound> core::ops::[<$trait Assign>]<u32> for NumVector<V> {
            #[inline(always)] fn [<$op _assign>](&mut self, rhs: u32) {
                core::ops::[<$trait Assign>]::[<$op _assign>](&mut self.0, rhs);
            }
        }
        impl<V: $bound> core::ops::$trait<NumVector<V::USize>> for NumVector<V> {
            type Output = Self; #[inline(always)] fn $op(self, rhs: NumVector<V::USize>) -> Self::Output {
                Self(core::ops::$trait::$op(self.0, rhs.0))
            }
        }
        impl<V: $bound> core::ops::[<$trait Assign>]<NumVector<V::USize>> for NumVector<V> {
            #[inline(always)] fn [<$op _assign>](&mut self, rhs: NumVector<V::USize>) {
                core::ops::[<$trait Assign>]::[<$op _assign>](&mut self.0, rhs.0);
            }
        }
    )*}};
}

fwd_ops!(BINARY: GenericVector => BitAnd::bitand, BitOr::bitor, BitXor::bitxor);
fwd_ops!(BINARY: NumericVector => Add::add, Sub::sub, Mul::mul, Div::div, Rem::rem);
fwd_ops!(SHIFTS: BitshiftVector => Shl::shl, Shr::shr);
fwd_ops!(UNARY: SignedVector => Neg::neg);

impl<V: NumericVector> num_traits::Num for NumVector<V> {
    type FromStrRadixErr = <V::Element as num_traits::Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        todo!()
    }
}

#[rustfmt::skip]
impl<V: FloatVector> num_traits::float::FloatCore for NumVector<V> {
    #[inline(always)] fn infinity()             -> Self { Self(V::INFINITY) }
    #[inline(always)] fn neg_infinity()         -> Self { Self(V::NEG_INFINITY) }
    #[inline(always)] fn nan()                  -> Self { Self(V::NAN) }
    #[inline(always)] fn neg_zero()             -> Self { Self(V::NEG_ZERO) }
    #[inline(always)] fn min_value()            -> Self { Self(V::MIN) }
    #[inline(always)] fn min_positive_value()   -> Self { Self(V::MIN_POSITIVE) }
    #[inline(always)] fn epsilon()              -> Self { Self(V::EPSILON) }
    #[inline(always)] fn max_value()            -> Self { Self(V::MAX) }

    /// Follows the logic of most important classification in the order:
    /// NaN (any) > Infinite (any) > Zero (all) > Subnormal (any) > Normal
    #[inline(always)]
    fn classify(self) -> core::num::FpCategory {
        if self.is_nan().any() {
            core::num::FpCategory::Nan
        } else if self.is_infinite().any() {
            core::num::FpCategory::Infinite
        } else if self.is_zero().all() {
            core::num::FpCategory::Zero
        } else if self.is_subnormal().any() {
            core::num::FpCategory::Subnormal
        } else {
            core::num::FpCategory::Normal
        }
    }

    #[inline(always)] fn to_degrees(self) -> Self { self * Self::FRAC_180_PI }
    #[inline(always)] fn to_radians(self) -> Self { self * Self::FRAC_PI_180 }
    #[inline(always)] fn integer_decode(self) -> (u64, i16, i8) { self.extract::<0>().integer_decode() }

    #[inline(always)] fn is_nan(self) -> bool { self.is_nan().any()}
    #[inline(always)] fn is_finite(self) -> bool { self.is_finite().all() }
    #[inline(always)] fn is_infinite(self) -> bool { self.is_infinite().any() }
    #[inline(always)] fn is_normal(self) -> bool { self.is_normal().all() }
    #[inline(always)] fn is_subnormal(self) -> bool { self.is_subnormal().any() }

    /// Returns true if **any** lane is negative, false otherwise.
    #[inline(always)] fn is_sign_negative(self) -> bool { self.is_negative().any() }
    /// Returns true if **all** lanes are positive, false otherwise.
    #[inline(always)] fn is_sign_positive(self) -> bool { self.is_positive().all() }

    #[inline(always)] fn floor(self) -> Self { self.floor() }
    #[inline(always)] fn ceil(self) -> Self { self.ceil() }
    #[inline(always)] fn round(self) -> Self { self.round() }
    #[inline(always)] fn trunc(self) -> Self { self.trunc() }
    #[inline(always)] fn fract(self) -> Self { self.fract() }

    #[inline(always)] fn abs(self) -> Self { self.abs() }
    #[inline(always)] fn signum(self) -> Self { self.signum() }

    #[inline(always)] fn min(self, other: Self) -> Self { self.min(other) }
    #[inline(always)] fn max(self, other: Self) -> Self { self.max(other) }
    #[inline(always)] fn clamp(self, min: Self, max: Self) -> Self { self.clamp(min, max) }

    /// Returns the reciprocal (1/x) of each lane. If you want to use an approximate
    /// reciprocal, check the [`Math`](crate::math::Math) trait.
    #[inline(always)] fn recip(self) -> Self { Self::ONE / self }
}

#[cfg(feature = "std")]
#[rustfmt::skip]
impl<V: FloatVector> num_traits::float::Float for NumVector<V>
where
    Self: VectorMath,
{
    #[inline(always)] fn epsilon() -> Self { Self::EPSILON }
    #[inline(always)] fn is_subnormal(self) -> bool { self.is_subnormal().any() }

    #[inline(always)] fn to_degrees(self) -> Self { self * Self::FRAC_180_PI }
    #[inline(always)] fn to_radians(self) -> Self { self * Self::FRAC_PI_180 }

    #[inline(always)] fn clamp(self, min: Self, max: Self) -> Self { self.clamp(min, max) }
    #[inline(always)] fn copysign(self, sign: Self) -> Self { self.copysign(sign) }

    #[inline(always)] fn nan() -> Self { Self::NAN }
    #[inline(always)] fn infinity() -> Self { Self::INFINITY }
    #[inline(always)] fn neg_infinity() -> Self { Self::NEG_INFINITY }
    #[inline(always)] fn neg_zero() -> Self { Self::NEG_ZERO }
    #[inline(always)] fn min_value() -> Self { Self::MIN }
    #[inline(always)] fn min_positive_value() -> Self { Self::MIN_POSITIVE }
    #[inline(always)] fn max_value() -> Self { Self::MAX }
    #[inline(always)] fn is_nan(self) -> bool { self.is_nan().any() }
    #[inline(always)] fn is_infinite(self) -> bool { self.is_infinite().any() }
    #[inline(always)] fn is_finite(self) -> bool { self.is_finite().all() }
    #[inline(always)] fn is_normal(self) -> bool { self.is_normal().all() }

    /// Follows the logic of most important classification in the order:
    /// NaN (any) > Infinite (any) > Zero (all) > Subnormal (any) > Normal
    #[inline(always)] fn classify(self) -> core::num::FpCategory { num_traits::float::FloatCore::classify(self) }

    #[inline(always)] fn floor(self) -> Self { self.floor() }
    #[inline(always)] fn ceil(self) -> Self { self.ceil() }
    #[inline(always)] fn round(self) -> Self { self.round() }
    #[inline(always)] fn trunc(self) -> Self { self.trunc() }
    #[inline(always)] fn fract(self) -> Self { self.fract() }
    #[inline(always)] fn abs(self) -> Self { self.abs() }
    #[inline(always)] fn signum(self) -> Self { self.signum() }

    /// Returns true if **any** lane is negative, false otherwise.
    #[inline(always)] fn is_sign_negative(self) -> bool { self.is_negative().any() }
    /// Returns true if **all** lanes are positive, false otherwise.
    #[inline(always)] fn is_sign_positive(self) -> bool { self.is_positive().all() }

    #[inline(always)] fn mul_add(self, a: Self, b: Self) -> Self { self.mul_add(a, b) }
    #[inline(always)] fn recip(self) -> Self { Self::ONE / self }
    #[inline(always)] fn powi(self, n: i32) -> Self { Math::powi(self, n) }
    #[inline(always)] fn powf(self, n: Self) -> Self { Math::powf(self, n) }
    #[inline(always)] fn sqrt(self) -> Self { self.sqrt() }
    #[inline(always)] fn exp(self) -> Self { Math::exp(self) }
    #[inline(always)] fn exp2(self) -> Self { Math::exp2(self) }
    #[inline(always)] fn ln(self) -> Self { Math::ln(self) }
    #[inline(always)] fn log(self, base: Self) -> Self { Math::log(self, base) }
    #[inline(always)] fn log2(self) -> Self { Math::log2(self) }
    #[inline(always)] fn log10(self) -> Self { Math::log10(self) }

    #[inline(always)] fn max(self, other: Self) -> Self { self.max(other) }
    #[inline(always)] fn min(self, other: Self) -> Self { self.min(other) }

    #[inline(always)] fn abs_sub(self, other: Self) -> Self { (self - other).max(Self::ZERO) }

    #[inline(always)] fn cbrt(self) -> Self { Math::cbrt(self) }
    #[inline(always)] fn hypot(self, other: Self) -> Self { Math::hypot(self, other) }
    #[inline(always)] fn sin(self) -> Self { Math::sin(self) }
    #[inline(always)] fn cos(self) -> Self { Math::cos(self) }
    #[inline(always)] fn tan(self) -> Self { Math::tan(self) }
    #[inline(always)] fn asin(self) -> Self { Math::asin(self) }
    #[inline(always)] fn acos(self) -> Self { Math::acos(self) }
    #[inline(always)] fn atan(self) -> Self { Math::atan(self) }
    #[inline(always)] fn atan2(self, other: Self) -> Self { Math::atan2(self, other) }
    #[inline(always)] fn sin_cos(self) -> (Self, Self) { Math::sin_cos(self) }
    #[inline(always)] fn exp_m1(self) -> Self { Math::exp_m1(self) }
    #[inline(always)] fn ln_1p(self) -> Self { Math::ln_1p(self) }
    #[inline(always)] fn sinh(self) -> Self { Math::sinh(self) }
    #[inline(always)] fn cosh(self) -> Self { Math::cosh(self) }
    #[inline(always)] fn tanh(self) -> Self { Math::tanh(self) }
    #[inline(always)] fn asinh(self) -> Self { Math::asinh(self) }
    #[inline(always)] fn acosh(self) -> Self { Math::acosh(self) }
    #[inline(always)] fn atanh(self) -> Self { Math::atanh(self) }

    #[inline(always)] fn integer_decode(self) -> (u64, i16, i8) { num_traits::float::FloatCore::integer_decode(self) }
}
