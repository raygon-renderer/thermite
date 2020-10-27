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
    type Vf64 = f64x4<SSE2>;
}

macro_rules! decl {
    ($($name:ident: $ety:ty => $ty:ty),*) => {$(
        #[derive(Clone, Copy)]
        #[repr(transparent)]
        pub struct $name<S: Simd> {
            value: $ty,
            _is: PhantomData<S>,
        }

        impl<S: Simd> $name<S> {
            #[inline(always)]
            fn new(value: $ty) -> Self {
                Self { value, _is: PhantomData }
            }
        }

        impl<S: Simd> fmt::Debug for $name<S> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let mut t = f.debug_tuple(stringify!($name));
                for i in 0..S::Vi32::NUM_ELEMENTS {
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

decl!(f64x4: f64 => (__m128d, __m128d));
impl<S: Simd> Default for f64x4<S> {
    #[inline(always)]
    fn default() -> Self {
        Self::new(unsafe { (_mm_setzero_pd(), _mm_setzero_pd()) })
    }
}

impl SimdVectorBase<SSE2, i32> for i32x4<SSE2> {
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

impl SimdBitwise<SSE2, i32> for i32x4<SSE2> {
    #[inline(always)]
    fn and_not(self, other: Self) -> Self {
        Self::new(unsafe { _mm_andnot_si128(self.value, other.value) })
    }

    const FULL_BITMASK: u16 = 0b1111;

    #[inline(always)]
    unsafe fn _mm_not(self) -> Self {
        self ^ Self::splat(!0)
    }

    #[inline(always)]
    unsafe fn _mm_bitand(self, rhs: Self) -> Self {
        Self::new(unsafe { _mm_and_si128(self.value, rhs.value) })
    }

    #[inline(always)]
    unsafe fn _mm_bitor(self, rhs: Self) -> Self {
        Self::new(unsafe { _mm_or_si128(self.value, rhs.value) })
    }

    #[inline(always)]
    unsafe fn _mm_bitxor(self, rhs: Self) -> Self {
        Self::new(unsafe { _mm_xor_si128(self.value, rhs.value) })
    }
}

macro_rules! impl_ops {
    (@UNARY $name:ident => $($op_trait:ident::$op:ident),*) => {paste::paste! {$(
        impl<S: Simd> $op_trait for $name<S> {
            type Output = Self;
            #[inline(always)] fn $op(self) -> Self { unsafe { self. [<_mm_ $op>]() } }
        }
    )*}};

    (@BINARY $name:ident => $($op_trait:ident::$op:ident),*) => {paste::paste! {$(
        impl<S: Simd> $op_trait<Self> for $name<S> {
            type Output = Self;
            #[inline(always)] fn $op(self, rhs: Self) -> Self { unsafe { self. [<_mm_ $op>](rhs) } }
        }

        impl<S: Simd> [<$op_trait Assign>]<Self> for $name<S> {
            #[inline(always)] fn [<$op _assign>](&mut self, rhs: Self) { *self = $op_trait::$op(*self, rhs); }
        }
    )*}};
}

impl_ops!(@UNARY i32x4 => Not::not);
impl_ops!(@BINARY i32x4 => BitAnd::bitand, BitOr::bitor, BitXor::bitxor);
