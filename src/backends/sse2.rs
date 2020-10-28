#![allow(unused)]

use crate::*;

use std::{
    fmt,
    marker::PhantomData,
    mem::{transmute, transmute_copy},
    ops::*,
};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SSE2;

impl Simd for SSE2 {
    type Vi32 = i32x4<SSE2>;
    type Vf32 = f32x4<SSE2>;
    //type Vf64 = f64x4<SSE2>;
}

macro_rules! decl {
    ($($name:ident: $ety:ty => $ty:ty),*) => {$(
        #[derive(Clone, Copy)]
        #[repr(transparent)]
        pub struct $name<S: Simd> {
            pub(crate) value: $ty,
            _is: PhantomData<S>,
        }

        impl<S: Simd> $name<S> {
            #[inline(always)]
            pub(crate) fn new(value: $ty) -> Self {
                Self { value, _is: PhantomData }
            }
        }

        impl<S: Simd> $name<S> where Self: SimdVectorBase<S, Element = $ety> {
            #[inline(always)]
            pub(crate) unsafe fn map<F>(mut self, f: F) -> Self
            where F: Fn($ety) -> $ety {
                for i in 0..Self::NUM_ELEMENTS {
                    let ptr = transmute::<&mut _, *mut $ety>(&mut self).add(i);
                    *ptr = f(*ptr);
                }
                self
            }

            pub(crate) unsafe fn zip<F>(a: Self, b: Self, f: F) -> Self
            where F: Fn($ety, $ety) -> $ety {
                let mut out = Self::default();
                for i in 0..Self::NUM_ELEMENTS {
                    *transmute::<&mut _, *mut $ety>(&mut out).add(i) =
                        f(a.extract_unchecked(i), b.extract_unchecked(i));
                }
                out
            }

            #[inline(always)]
            pub(crate) unsafe fn reduce<F>(self, mut init: $ety, f: F) -> $ety
            where F: Fn($ety, $ety) -> $ety {
                for i in 0..Self::NUM_ELEMENTS {
                    init = f(init, self.extract_unchecked(i));
                }
                init
            }

            #[inline(always)]
            pub(crate) unsafe fn reduce2<F>(self, f: F) -> $ety
            where F: Fn($ety, $ety) -> $ety {
                let mut accum = self.extract_unchecked(0);
                for i in 1..Self::NUM_ELEMENTS {
                    accum = f(accum, self.extract_unchecked(i));
                }
                accum
            }
        }

        impl<S: Simd> fmt::Debug for $name<S> where Self: SimdVectorBase<S> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let mut t = f.debug_tuple(stringify!($name));
                for i in 0..Self::NUM_ELEMENTS {
                    t.field(unsafe { &*transmute::<&_, *const $ety>(&self).add(i) });
                }
                t.finish()
            }
        }
    )*};
}

decl!(i32x4: i32 => __m128i);
impl<S: Simd> Default for i32x4<S> {
    #[inline(always)]
    fn default() -> Self {
        Self::new(unsafe { _mm_setzero_si128() })
    }
}

decl!(f32x4: f32 => __m128);
impl<S: Simd> Default for f32x4<S> {
    #[inline(always)]
    fn default() -> Self {
        Self::new(unsafe { _mm_setzero_ps() })
    }
}

/*
decl!(f64x4: f64 => (__m128d, __m128d));
impl<S: Simd> Default for f64x4<S> {
    #[inline(always)]
    fn default() -> Self {
        Self::new(unsafe { (_mm_setzero_pd(), _mm_setzero_pd()) })
    }
}*/

impl SimdVectorBase<SSE2> for i32x4<SSE2> {
    type Element = i32;

    #[inline(always)]
    fn splat(value: i32) -> Self {
        Self::new(unsafe { _mm_set1_epi32(value) })
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn extract_unchecked(self, index: usize) -> i32 {
        *transmute::<&_, *const i32>(&self).add(index)
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn replace_unchecked(mut self, index: usize, value: i32) -> Self {
        *transmute::<&_, *mut i32>(&mut self).add(index) = value;
        self
    }
}

impl SimdVectorBase<SSE2> for f32x4<SSE2> {
    type Element = f32;

    #[inline(always)]
    fn splat(value: f32) -> Self {
        Self::new(unsafe { _mm_set1_ps(value) })
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn extract_unchecked(self, index: usize) -> f32 {
        *transmute::<&_, *const f32>(&self).add(index)
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn replace_unchecked(mut self, index: usize, value: f32) -> Self {
        *transmute::<&_, *mut f32>(&mut self).add(index) = value;
        self
    }
}

impl SimdBitwise<SSE2> for i32x4<SSE2> {
    #[inline(always)]
    fn and_not(self, other: Self) -> Self {
        Self::new(unsafe { _mm_andnot_si128(self.value, other.value) })
    }

    const FULL_BITMASK: u16 = 0b1111;

    #[inline(always)]
    fn bitmask(self) -> u16 {
        unsafe { _mm_movemask_ps(transmute(self)) as u16 }
    }

    #[inline(always)]
    unsafe fn _mm_not(self) -> Self {
        self ^ Self::splat(!0)
    }

    #[inline(always)]
    unsafe fn _mm_bitand(self, rhs: Self) -> Self {
        Self::new(_mm_and_si128(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_bitor(self, rhs: Self) -> Self {
        Self::new(_mm_or_si128(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_bitxor(self, rhs: Self) -> Self {
        Self::new(_mm_xor_si128(self.value, rhs.value))
    }
}

impl SimdBitwise<SSE2> for f32x4<SSE2> {
    fn and_not(self, other: Self) -> Self {
        Self::new(unsafe { _mm_andnot_ps(self.value, other.value) })
    }

    const FULL_BITMASK: u16 = 0b1111;

    #[inline(always)]
    fn bitmask(self) -> u16 {
        unsafe { _mm_movemask_ps(transmute(self)) as u16 }
    }

    #[inline(always)]
    unsafe fn _mm_not(self) -> Self {
        self ^ Self::splat(f32::from_bits(!0))
    }

    #[inline(always)]
    unsafe fn _mm_bitand(self, rhs: Self) -> Self {
        Self::new(_mm_and_ps(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_bitor(self, rhs: Self) -> Self {
        Self::new(_mm_or_ps(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_bitxor(self, rhs: Self) -> Self {
        Self::new(_mm_xor_ps(self.value, rhs.value))
    }
}

impl PartialEq<Self> for i32x4<SSE2> {
    fn eq(&self, other: &Self) -> bool {
        <Self as SimdVector<SSE2>>::eq(*self, *other).all()
    }

    fn ne(&self, other: &Self) -> bool {
        <Self as SimdVector<SSE2>>::ne(*self, *other).any()
    }
}

impl PartialEq<Self> for f32x4<SSE2> {
    fn eq(&self, other: &Self) -> bool {
        <Self as SimdVector<SSE2>>::eq(*self, *other).all()
    }

    fn ne(&self, other: &Self) -> bool {
        <Self as SimdVector<SSE2>>::ne(*self, *other).any()
    }
}

impl Eq for i32x4<SSE2> {}

impl SimdMask<SSE2> for i32x4<SSE2> {}
impl SimdMask<SSE2> for f32x4<SSE2> {}

impl SimdVector<SSE2> for i32x4<SSE2> {
    #[inline(always)]
    fn zero() -> Self {
        Self::new(unsafe { _mm_setzero_si128() })
    }

    #[inline(always)]
    fn one() -> Self {
        Self::splat(1)
    }

    #[inline(always)]
    fn min_value() -> Self {
        Self::splat(i32::MIN)
    }

    #[inline(always)]
    fn max_value() -> Self {
        Self::splat(i32::MAX)
    }

    #[inline]
    fn min_element(self) -> Self::Element {
        unsafe { self.reduce2(|a, x| a.min(x)) }
    }

    #[inline]
    fn max_element(self) -> Self::Element {
        unsafe { self.reduce2(|a, x| a.max(x)) }
    }

    #[inline(always)]
    fn eq(self, other: Self) -> Mask<SSE2, Self> {
        Mask::new(Self::new(unsafe { _mm_cmpeq_epi32(self.value, other.value) }))
    }

    #[inline(always)]
    fn lt(self, other: Self) -> Mask<SSE2, Self> {
        Mask::new(Self::new(unsafe { _mm_cmplt_epi32(self.value, other.value) }))
    }

    #[inline(always)]
    fn gt(self, other: Self) -> Mask<SSE2, Self> {
        Mask::new(Self::new(unsafe { _mm_cmpgt_epi32(self.value, other.value) }))
    }

    #[inline(always)]
    unsafe fn _mm_add(self, rhs: Self) -> Self {
        Self::new(_mm_add_epi32(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_sub(self, rhs: Self) -> Self {
        Self::new(_mm_sub_epi32(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_mul(self, rhs: Self) -> Self {
        Self::new({
            let tmp1 = _mm_mul_epu32(self.value, rhs.value);
            let tmp2 = _mm_mul_epu32(_mm_srli_si128(self.value, 4), _mm_srli_si128(rhs.value, 4));

            _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp1, 8), _mm_shuffle_epi32(tmp2, 8))
        })
    }

    #[inline(always)]
    unsafe fn _mm_div(self, rhs: Self) -> Self {
        Self::zip(self, rhs, Div::div)
    }

    #[inline(always)]
    unsafe fn _mm_rem(self, rhs: Self) -> Self {
        Self::zip(self, rhs, Rem::rem)
    }
}

impl SimdVector<SSE2> for f32x4<SSE2> {
    #[inline(always)]
    fn zero() -> Self {
        Self::new(unsafe { _mm_setzero_ps() })
    }

    #[inline(always)]
    fn one() -> Self {
        Self::splat(1.0)
    }

    #[inline(always)]
    fn min_value() -> Self {
        Self::splat(f32::MIN)
    }

    #[inline(always)]
    fn max_value() -> Self {
        Self::splat(f32::MAX)
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        Self::new(unsafe { _mm_min_ps(self.value, other.value) })
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        Self::new(unsafe { _mm_max_ps(self.value, other.value) })
    }

    #[inline]
    fn min_element(self) -> Self::Element {
        // TODO: Replace with log-reduce
        unsafe { self.reduce2(|a, x| a.min(x)) }
    }

    #[inline]
    fn max_element(self) -> Self::Element {
        // TODO: Replace with log-reduce
        unsafe { self.reduce2(|a, x| a.max(x)) }
    }

    #[inline(always)]
    fn eq(self, other: Self) -> Mask<SSE2, Self> {
        Mask::new(Self::new(unsafe { _mm_cmpeq_ps(self.value, other.value) }))
    }

    #[inline(always)]
    fn lt(self, other: Self) -> Mask<SSE2, Self> {
        Mask::new(Self::new(unsafe { _mm_cmplt_ps(self.value, other.value) }))
    }

    #[inline(always)]
    fn gt(self, other: Self) -> Mask<SSE2, Self> {
        Mask::new(Self::new(unsafe { _mm_cmpgt_ps(self.value, other.value) }))
    }

    #[inline(always)]
    unsafe fn _mm_add(self, rhs: Self) -> Self {
        Self::new(_mm_add_ps(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_sub(self, rhs: Self) -> Self {
        Self::new(_mm_sub_ps(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_mul(self, rhs: Self) -> Self {
        Self::new(_mm_mul_ps(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_div(self, rhs: Self) -> Self {
        Self::new(_mm_div_ps(self.value, rhs.value))
    }

    #[inline(always)]
    unsafe fn _mm_rem(self, rhs: Self) -> Self {
        Self::zip(self, rhs, Rem::rem)
    }
}

impl SimdIntVector<SSE2> for i32x4<SSE2> {}

impl SimdSignedVector<SSE2> for i32x4<SSE2> {
    #[inline(always)]
    fn neg_one() -> Self {
        Self::splat(-1)
    }

    #[inline(always)]
    fn min_positive() -> Self {
        Self::splat(0)
    }

    #[inline(always)]
    fn abs(self) -> Self {
        // https://graphics.stanford.edu/~seander/bithacks.html#ConditionalNegate
        Self::new(unsafe {
            let should_negate = _mm_cmplt_epi32(self.value, _mm_setzero_si128());
            _mm_add_epi32(
                _mm_xor_si128(should_negate, self.value),
                // add 1 if negative
                _mm_and_si128(should_negate, _mm_set1_epi32(1)),
            )
        })
    }

    #[inline(always)]
    unsafe fn _mm_neg(self) -> Self {
        self ^ Self::neg_one() + Self::one()
    }
}

impl SimdSignedVector<SSE2> for f32x4<SSE2> {
    #[inline(always)]
    fn neg_one() -> Self {
        Self::splat(-1.0)
    }

    #[inline(always)]
    fn min_positive() -> Self {
        Self::splat(f32::MIN_POSITIVE)
    }

    #[inline(always)]
    fn signum(self) -> Self {
        Self::one() | (self & Self::neg_zero())
    }

    #[inline(always)]
    fn abs(self) -> Self {
        // clear sign bit
        self & Self::splat(f32::from_bits(0x7fffffff))
    }

    #[inline(always)]
    unsafe fn _mm_neg(self) -> Self {
        // Xor sign bit using -0.0 as a shorthand for the sign bit
        self ^ Self::neg_zero()
    }
}

impl SimdFloatVector<SSE2> for f32x4<SSE2> {
    #[inline(always)]
    fn epsilon() -> Self {
        Self::splat(f32::EPSILON)
    }
    #[inline(always)]
    fn infinity() -> Self {
        Self::splat(f32::INFINITY)
    }
    #[inline(always)]
    fn neg_infinity() -> Self {
        Self::splat(f32::NEG_INFINITY)
    }
    #[inline(always)]
    fn neg_zero() -> Self {
        Self::splat(-0.0)
    }
    #[inline(always)]
    fn nan() -> Self {
        Self::splat(f32::NAN)
    }

    fn sum(self) -> Self::Element {
        // TODO: Replace with log-reduce
        unsafe { self.reduce2(|sum, x| sum + x) }
    }

    fn product(self) -> Self::Element {
        // TODO: Replace with log-reduce
        unsafe { self.reduce2(|prod, x| x * prod) }
    }

    fn floor(self) -> Self {
        unsafe { self.map(|x| x.floor()) }
    }

    fn ceil(self) -> Self {
        unsafe { self.map(|x| x.ceil()) }
    }

    fn round(self) -> Self {
        unsafe { self.map(|x| x.round()) }
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        Self::new(unsafe { _mm_sqrt_ps(self.value) })
    }

    #[inline(always)]
    fn rsqrt(self) -> Self {
        Self::new(unsafe { _mm_rsqrt_ps(self.value) })
    }

    #[inline(always)]
    fn rsqrt_precise(self) -> Self {
        let y = self.rsqrt();
        let nx2 = self * Self::splat(-0.5);
        let threehalfs = Self::splat(1.5);

        y * (y * y).mul_add(nx2, threehalfs)
    }

    #[inline(always)]
    fn recepr(self) -> Self {
        Self::new(unsafe { _mm_rcp_ps(self.value) })
    }
}

macro_rules! impl_ops {
    (@UNARY $name:ident => $($op_trait:ident::$op:ident),*) => {paste::paste! {$(
        impl $op_trait for $name<SSE2> {
            type Output = Self;
            #[inline(always)] fn $op(self) -> Self { unsafe { self. [<_mm_ $op>]() } }
        }
    )*}};

    (@BINARY $name:ident => $($op_trait:ident::$op:ident),*) => {paste::paste! {$(
        impl $op_trait<Self> for $name<SSE2> {
            type Output = Self;
            #[inline(always)] fn $op(self, rhs: Self) -> Self { unsafe { self. [<_mm_ $op>](rhs) } }
        }

        impl [<$op_trait Assign>]<Self> for $name<SSE2> {
            #[inline(always)] fn [<$op _assign>](&mut self, rhs: Self) { *self = $op_trait::$op(*self, rhs); }
        }
    )*}};
}

impl_ops!(@UNARY i32x4 => Not::not, Neg::neg);
impl_ops!(@BINARY i32x4 => Add::add, Sub::sub, Mul::mul, Div::div, Rem::rem, BitAnd::bitand, BitOr::bitor, BitXor::bitxor);

impl_ops!(@UNARY f32x4 => Not::not, Neg::neg);
impl_ops!(@BINARY f32x4 => Add::add, Sub::sub, Mul::mul, Div::div, Rem::rem, BitAnd::bitand, BitOr::bitor, BitXor::bitxor);

impl SimdCastFrom<i32x4<SSE2>> for f32x4<SSE2> {
    #[inline(always)]
    fn from_cast(from: i32x4<SSE2>) -> Self {
        Self::new(unsafe { _mm_cvtepi32_ps(from.value) })
    }
}

impl SimdCastFrom<f32x4<SSE2>> for i32x4<SSE2> {
    #[inline(always)]
    fn from_cast(from: f32x4<SSE2>) -> Self {
        Self::new(unsafe { _mm_cvttps_epi32(from.value) })
    }
}
