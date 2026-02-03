//! Compatibility wrapper for a generic vector to implement `num_traits` traits.

use core::ops::Deref;

use crate::{
    generic::{
        BitshiftVector, BitwiseVector, CastMask, FloatVector, GenericMask, GenericSelectable, GenericVector,
        NumVector as NumVectorTrait, NumericVector, PartialOrdVector, SignedVector,
    },
    math::{CoreMath, FloatConsts, RealMath, SpatialMath, TranscendentalMath},
    register::{FloatElement, FloatRegister},
    vector::Vector,
};

/// Wraps a generic vector to provide implementations of `num_traits` traits.
///
/// `num_traits` is a widely used crate that provides numeric traits for Rust types.
/// However, unless you want to use these, implementing them on all vector types
/// would cause conflicts with more inherent methods provided by this library.
///
/// Therefore, this wrapper type allows you to opt-in to using `num_traits`
/// when needed, without causing conflicts in the main vector types.
///
/// Notably, the implementation of `MulAdd` uses `mul_adde` under the hood,
/// which may fall back to separate multiply and add operations if a fused multiply-add
/// is not available for the target architecture.
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
        Self::SelectableMask: CastMask<M>,
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

fwd_ops!(BINARY: BitwiseVector => BitAnd::bitand, BitOr::bitor, BitXor::bitxor);
fwd_ops!(BINARY: NumericVector => Add::add, Sub::sub, Mul::mul, Div::div, Rem::rem);
fwd_ops!(SHIFTS: BitshiftVector => Shl::shl, Shr::shr);
fwd_ops!(UNARY: SignedVector => Neg::neg);

#[rustfmt::skip]
impl<V: NumericVector> num_traits::Zero for NumVector<V> {
    #[inline(always)] fn zero() -> Self { Self(V::ZERO) }
    /// Returns true if **all** lanes are zero, false otherwise.
    #[inline(always)] fn is_zero(&self) -> bool { self.0.is_zero().all() }
}

#[rustfmt::skip]
impl<V: NumericVector> num_traits::ConstZero for NumVector<V> {
    const ZERO: Self = Self(V::ZERO);
}

#[rustfmt::skip]
impl<V: NumericVector> num_traits::One for NumVector<V> {
    #[inline(always)] fn one() -> Self { Self(V::ONE) }
    /// Returns true if **all** lanes are one, false otherwise.
    #[inline(always)] fn is_one(&self) -> bool { self.0.cmp_eq(V::ONE).all() }
}

#[rustfmt::skip]
impl<V: NumericVector> num_traits::ConstOne for NumVector<V> {
    const ONE: Self = Self(V::ONE);
}

#[rustfmt::skip]
impl<V: NumericVector> num_traits::Bounded for NumVector<V> {
    #[inline(always)] fn min_value() -> Self { Self(V::MIN) }
    #[inline(always)] fn max_value() -> Self { Self(V::MAX) }
}

#[allow(clippy::partialeq_ne_impl)]
impl<V: PartialOrdVector> PartialEq for NumVector<V> {
    /// Returns true if **all** lanes are equal, false otherwise.
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.0.cmp_eq(other.0).all()
    }

    /// Returns true if **any** lane is not equal, false otherwise.
    #[inline(always)]
    fn ne(&self, other: &Self) -> bool {
        self.0.cmp_ne(other.0).any()
    }
}

impl<V: NumericVector> num_traits::Num for NumVector<V>
where
    V::Element: num_traits::Num,
{
    type FromStrRadixErr = <V::Element as num_traits::Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Ok(Self(V::splat(num_traits::Num::from_str_radix(str, radix)?)))
    }
}

impl<V: NumericVector> num_traits::NumCast for NumVector<V>
where
    V::Element: num_traits::NumCast,
{
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        <V::Element as num_traits::NumCast>::from(n).map(|val| Self(V::splat(val)))
    }
}

/// Implements conversion from `NumVector<V>` to primitive types by extracting
/// the first lane and converting that. Other lanes are ignored.
#[rustfmt::skip]
impl<V: GenericVector> num_traits::ToPrimitive for NumVector<V>
where
    V::Element: num_traits::ToPrimitive,
{
    #[inline(always)] fn to_isize(&self) -> Option<isize> { self.extract::<0>().to_isize() }
    #[inline(always)] fn to_i8(&self) -> Option<i8> { self.extract::<0>().to_i8() }
    #[inline(always)] fn to_i16(&self) -> Option<i16> { self.extract::<0>().to_i16() }
    #[inline(always)] fn to_i32(&self) -> Option<i32> { self.extract::<0>().to_i32() }
    #[inline(always)] fn to_i128(&self) -> Option<i128> { self.extract::<0>().to_i128() }
    #[inline(always)] fn to_usize(&self) -> Option<usize> { self.extract::<0>().to_usize() }
    #[inline(always)] fn to_u8(&self) -> Option<u8> { self.extract::<0>().to_u8() }
    #[inline(always)] fn to_u16(&self) -> Option<u16> { self.extract::<0>().to_u16() }
    #[inline(always)] fn to_u32(&self) -> Option<u32> { self.extract::<0>().to_u32() }
    #[inline(always)] fn to_u128(&self) -> Option<u128> { self.extract::<0>().to_u128() }
    #[inline(always)] fn to_f32(&self) -> Option<f32> { self.extract::<0>().to_f32() }
    #[inline(always)] fn to_f64(&self) -> Option<f64> { self.extract::<0>().to_f64() }
    #[inline(always)] fn to_i64(&self) -> Option<i64> { self.extract::<0>().to_i64() }
    #[inline(always)] fn to_u64(&self) -> Option<u64> { self.extract::<0>().to_u64() }
}

impl<V: PartialOrdVector> PartialOrd for NumVector<V> {
    /// Partial comparison between two vectors, returning `None` if
    /// the vectors are not fully ordered. Only returns `Some(Ordering)` if
    /// all lanes are less than, greater than, or equal.
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        let is_less = V::cmp_lt(self.0, other.0).all();
        let is_greater = V::cmp_gt(self.0, other.0).all();
        let is_equal = V::cmp_eq(self.0, other.0).all();

        match (is_less, is_greater, is_equal) {
            (true, false, false) => Some(core::cmp::Ordering::Less),
            (false, true, false) => Some(core::cmp::Ordering::Greater),
            (false, false, true) => Some(core::cmp::Ordering::Equal),
            _ => None,
        }
    }
}

#[rustfmt::skip]
impl<V: FloatVector> num_traits::float::FloatCore for NumVector<V>
    where V::Element: num_traits::float::FloatCore,
{
    #[inline(always)] fn infinity()             -> Self { Self(<V as FloatVector>::INFINITY) }
    #[inline(always)] fn neg_infinity()         -> Self { Self(<V as FloatVector>::NEG_INFINITY) }
    #[inline(always)] fn nan()                  -> Self { Self(<V as FloatVector>::NAN) }
    #[inline(always)] fn neg_zero()             -> Self { Self(<V as FloatVector>::NEG_ZERO) }
    #[inline(always)] fn min_value()            -> Self { Self(<V as NumericVector>::MIN) }
    #[inline(always)] fn min_positive_value()   -> Self { Self(<V as SignedVector>::MIN_POSITIVE) }
    #[inline(always)] fn epsilon()              -> Self { Self(<V as FloatVector>::EPSILON) }
    #[inline(always)] fn max_value()            -> Self { Self(<V as NumericVector>::MAX) }

    /// Follows the logic of most important classification in the order:
    /// NaN (any) > Infinite (any) > Zero (all) > Subnormal (any) > Normal
    #[inline(always)]
    fn classify(self) -> core::num::FpCategory {
        let x = self.0;

        if x.is_nan().any() {
            core::num::FpCategory::Nan
        } else if x.is_infinite().any() {
            core::num::FpCategory::Infinite
        } else if x.is_zero().all() {
            core::num::FpCategory::Zero
        } else if x.is_subnormal().any() {
            core::num::FpCategory::Subnormal
        } else {
            core::num::FpCategory::Normal
        }
    }

    #[inline(always)] fn to_degrees(self) -> Self { self * Self(V::FRAC_180_PI) }
    #[inline(always)] fn to_radians(self) -> Self { self * Self(V::FRAC_PI_180) }
    #[inline(always)] fn integer_decode(self) -> (u64, i16, i8) { self.extract::<0>().integer_decode() }

    #[inline(always)] fn is_nan(self) -> bool { self.0.is_nan().any()}
    #[inline(always)] fn is_finite(self) -> bool { self.0.is_finite().all() }
    #[inline(always)] fn is_infinite(self) -> bool { self.0.is_infinite().any() }
    #[inline(always)] fn is_normal(self) -> bool { self.0.is_normal().all() }
    #[inline(always)] fn is_subnormal(self) -> bool { self.0.is_subnormal().any() }

    /// Returns true if **any** lane is negative, false otherwise.
    #[inline(always)] fn is_sign_negative(self) -> bool { self.0.is_negative().any() }
    /// Returns true if **all** lanes are positive, false otherwise.
    #[inline(always)] fn is_sign_positive(self) -> bool { self.0.is_positive().all() }

    #[inline(always)] fn floor(self) -> Self { Self(self.0.floor()) }
    #[inline(always)] fn ceil(self) -> Self { Self(self.0.ceil()) }
    #[inline(always)] fn round(self) -> Self { Self(self.0.round()) }
    #[inline(always)] fn trunc(self) -> Self { Self(self.0.trunc()) }
    #[inline(always)] fn fract(self) -> Self { Self(self.0.fract()) }

    #[inline(always)] fn abs(self) -> Self { Self(self.0.abs()) }
    #[inline(always)] fn signum(self) -> Self { Self(self.0.signum()) }

    #[inline(always)] fn min(self, other: Self) -> Self { Self(self.0.min(other.0)) }
    #[inline(always)] fn max(self, other: Self) -> Self { Self(self.0.max(other.0)) }

    /// Returns the reciprocal (1/x) of each lane. If you want to use an approximate
    /// reciprocal, check the [`RealMath`] trait for that.
    #[inline(always)] fn recip(self) -> Self { <Self as num_traits::ConstOne>::ONE / self }
}

#[cfg(feature = "std")]
#[rustfmt::skip]
impl<V: FloatVector> num_traits::float::Float for NumVector<V>
where
    V::Element: num_traits::float::Float + num_traits::float::FloatCore,
    V: RealMath,
{
    #[inline(always)] fn is_subnormal(self) -> bool { self.0.is_subnormal().any() }

    #[inline(always)] fn to_degrees(self) -> Self { Self(self.0 * V::FRAC_180_PI) }
    #[inline(always)] fn to_radians(self) -> Self { Self(self.0 * V::FRAC_PI_180) }

    #[inline(always)] fn clamp(self, min: Self, max: Self)  -> Self { Self(self.0.clamp(min.0, max.0)) }
    #[inline(always)] fn copysign(self, sign: Self)         -> Self { Self(self.0.copysign(sign.0)) }

    #[inline(always)] fn infinity()             -> Self { Self(<V as FloatVector>::INFINITY) }
    #[inline(always)] fn neg_infinity()         -> Self { Self(<V as FloatVector>::NEG_INFINITY) }
    #[inline(always)] fn nan()                  -> Self { Self(<V as FloatVector>::NAN) }
    #[inline(always)] fn neg_zero()             -> Self { Self(<V as FloatVector>::NEG_ZERO) }
    #[inline(always)] fn min_value()            -> Self { Self(<V as NumericVector>::MIN) }
    #[inline(always)] fn min_positive_value()   -> Self { Self(<V as SignedVector>::MIN_POSITIVE) }
    #[inline(always)] fn epsilon()              -> Self { Self(<V as FloatVector>::EPSILON) }
    #[inline(always)] fn max_value()            -> Self { Self(<V as NumericVector>::MAX) }

    #[inline(always)] fn is_nan(self) -> bool { self.0.is_nan().any() }
    #[inline(always)] fn is_infinite(self) -> bool { self.0.is_infinite().any() }
    #[inline(always)] fn is_finite(self) -> bool { self.0.is_finite().all() }
    #[inline(always)] fn is_normal(self) -> bool { self.0.is_normal().all() }

    /// Follows the logic of most important classification in the order:
    /// NaN (any) > Infinite (any) > Zero (all) > Subnormal (any) > Normal
    #[inline(always)] fn classify(self) -> core::num::FpCategory { num_traits::float::FloatCore::classify(self) }

    #[inline(always)] fn floor(self) -> Self { Self(self.0.floor()) }
    #[inline(always)] fn ceil(self) -> Self { Self(self.0.ceil()) }
    #[inline(always)] fn round(self) -> Self { Self(self.0.round()) }
    #[inline(always)] fn trunc(self) -> Self { Self(self.0.trunc()) }
    #[inline(always)] fn fract(self) -> Self { Self(self.0.fract()) }
    #[inline(always)] fn abs(self) -> Self { Self(self.0.abs()) }
    #[inline(always)] fn signum(self) -> Self { Self(self.0.signum()) }

    /// Returns true if **any** lane is negative, false otherwise.
    #[inline(always)] fn is_sign_negative(self) -> bool { self.0.is_negative().any() }
    /// Returns true if **all** lanes are positive, false otherwise.
    #[inline(always)] fn is_sign_positive(self) -> bool { self.0.is_positive().all() }

    #[inline(always)] fn mul_add(self, a: Self, b: Self) -> Self { Self(FloatVector::mul_adde(self.0, a.0, b.0)) }
    #[inline(always)] fn recip(self)            -> Self { Self(V::ONE / self.0) }
    #[inline(always)] fn powi(self, n: i32)     -> Self { Self(CoreMath::powi(self.0, n)) }
    #[inline(always)] fn powf(self, n: Self)    -> Self { Self(TranscendentalMath::powf(self.0, n.0)) }
    #[inline(always)] fn sqrt(self)             -> Self { Self(FloatVector::sqrt(self.0)) }
    #[inline(always)] fn exp(self)              -> Self { Self(TranscendentalMath::exp(self.0)) }
    #[inline(always)] fn exp2(self)             -> Self { Self(TranscendentalMath::exp2(self.0)) }
    #[inline(always)] fn ln(self)               -> Self { Self(TranscendentalMath::ln(self.0)) }
    #[inline(always)] fn log(self, base: Self)  -> Self { Self(TranscendentalMath::log(self.0, base.0)) }
    #[inline(always)] fn log2(self)             -> Self { Self(TranscendentalMath::log2(self.0)) }
    #[inline(always)] fn log10(self)            -> Self { Self(TranscendentalMath::log10(self.0)) }

    #[inline(always)] fn max(self, other: Self) -> Self { Self(NumericVector::max(self.0, other.0)) }
    #[inline(always)] fn min(self, other: Self) -> Self { Self(NumericVector::min(self.0, other.0)) }

    #[inline(always)] fn abs_sub(self, other: Self) -> Self { Self(NumericVector::max(V::ZERO, self.0 - other.0)) }

    #[inline(always)] fn sin_cos(self)              -> (Self, Self) {
        let (s, c) = TranscendentalMath::sin_cos(self.0);
        (Self(s), Self(c))
    }

    #[inline(always)] fn cbrt(self)                 -> Self { Self(TranscendentalMath::cbrt(self.0)) }
    #[inline(always)] fn hypot(self, other: Self)   -> Self { Self(SpatialMath::hypot(self.0, other.0)) }
    #[inline(always)] fn sin(self)                  -> Self { Self(TranscendentalMath::sin(self.0)) }
    #[inline(always)] fn cos(self)                  -> Self { Self(TranscendentalMath::cos(self.0)) }
    #[inline(always)] fn tan(self)                  -> Self { Self(TranscendentalMath::tan(self.0)) }
    #[inline(always)] fn asin(self)                 -> Self { Self(TranscendentalMath::asin(self.0)) }
    #[inline(always)] fn acos(self)                 -> Self { Self(TranscendentalMath::acos(self.0)) }
    #[inline(always)] fn atan(self)                 -> Self { Self(TranscendentalMath::atan(self.0)) }
    #[inline(always)] fn atan2(self, other: Self)   -> Self { Self(TranscendentalMath::atan2(self.0, other.0)) }
    #[inline(always)] fn exp_m1(self)               -> Self { Self(TranscendentalMath::exp_m1(self.0)) }
    #[inline(always)] fn ln_1p(self)                -> Self { Self(TranscendentalMath::ln_1p(self.0)) }
    #[inline(always)] fn sinh(self)                 -> Self { Self(TranscendentalMath::sinh(self.0)) }
    #[inline(always)] fn cosh(self)                 -> Self { Self(TranscendentalMath::cosh(self.0)) }
    #[inline(always)] fn tanh(self)                 -> Self { Self(TranscendentalMath::tanh(self.0)) }
    #[inline(always)] fn asinh(self)                -> Self { Self(TranscendentalMath::asinh(self.0)) }
    #[inline(always)] fn acosh(self)                -> Self { Self(TranscendentalMath::acosh(self.0)) }
    #[inline(always)] fn atanh(self)                -> Self { Self(TranscendentalMath::atanh(self.0)) }

    #[inline(always)] fn integer_decode(self) -> (u64, i16, i8) { num_traits::float::FloatCore::integer_decode(self) }
}

#[rustfmt::skip]
impl<V: FloatVector> num_traits::MulAdd<Self, Self> for NumVector<V> {
    type Output = Self;

    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self::Output { Self(self.0.mul_adde(a.0, b.0)) }
}

impl<V: FloatVector> num_traits::MulAddAssign<Self, Self> for NumVector<V> {
    #[inline(always)]
    fn mul_add_assign(&mut self, a: Self, b: Self) {
        self.0 = self.0.mul_adde(a.0, b.0);
    }
}
