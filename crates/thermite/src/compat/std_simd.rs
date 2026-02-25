use generic_array::typenum::{Const, Unsigned};

use std::simd::cmp::{SimdPartialEq, SimdPartialOrd};
use std::simd::{MaskElement as StdMaskElement, SimdElement as StdSimdElement};
//
use crate::generic::ops::{BitAndNot, BitAndNotAssign, NotMasked};
use crate::prelude::{BitwiseVector, FloatVector, GenericMask, NumericVector, PartialOrdVector, SignedVector};

pub trait MaskElement: crate::element::MaskElement + StdMaskElement {}
impl<T> MaskElement for T where T: crate::element::MaskElement + StdMaskElement {}

pub trait SimdElement:
    Element<Signed: StdSimdElement<Mask: MaskElement>, Unsigned: StdSimdElement<Mask: MaskElement>>
    + StdSimdElement<Mask: MaskElement>
{
}
impl<T> SimdElement for T where
    T: Element<Signed: StdSimdElement<Mask: MaskElement>, Unsigned: StdSimdElement<Mask: MaskElement>>
        + StdSimdElement<Mask: MaskElement>
{
}

use core::ops::Neg;

use std::simd::{
    LaneCount, Mask, Simd, SimdCast, SupportedLaneCount,
    num::{SimdFloat, SimdInt, SimdUint},
};

use crate::{
    generic::GenericSelectable,
    isa::InstructionSet,
    prelude::{BitCastVector, CastMask, Element, GenericVector},
    register::Lanes,
};

pub trait SimdNum<T: StdSimdElement>: Copy {
    type Cast<U: StdSimdElement>;

    const TWO: T;
    const MIN: T;
    const MAX: T;

    fn cast<U>(self) -> Self::Cast<U>
    where
        U: SimdCast + StdSimdElement;

    fn reduce_sum(self) -> T;
    fn reduce_product(self) -> T;
    fn reduce_min(self) -> T;
    fn reduce_max(self) -> T;

    fn simd_min(self, other: Self) -> Self;
    fn simd_max(self, other: Self) -> Self;
    fn simd_clamp(self, min: Self, max: Self) -> Self;

    fn swap_bytes(self) -> Self;
    fn reverse_bits(self) -> Self;
}

pub trait SimdSigned<T: StdSimdElement>: SimdNum<T> + Neg<Output = Self> {
    type Mask;

    const NEG_ONE: T;
    const MIN_POSITIVE: T;

    fn abs(self) -> Self;
    fn signum(self) -> Self;

    fn is_sign_positive(self) -> Self::Mask;
    fn is_sign_negative(self) -> Self::Mask;
    fn copysign(self, sign: Self) -> Self;
}

#[rustfmt::skip]
macro_rules! impl_float_simd {
    ($ty:ty: $bits:ty) => {
        impl<const N: usize> SimdNum<$ty> for Simd<$ty, N>
        where
            LaneCount<N>: SupportedLaneCount,
            Self: SimdFloat<Bits = Simd<$bits, N>, Scalar = $ty, Mask = Mask<<$ty as StdSimdElement>::Mask, N>>,
        {
            type Cast<U: StdSimdElement> = <Self as SimdFloat>::Cast<U>;

            const TWO: $ty = 2.0;
            const MIN: $ty = <$ty>::MIN;
            const MAX: $ty = <$ty>::MAX;

            #[inline(always)]
            fn cast<U>(self) -> Self::Cast<U> where U: SimdCast + StdSimdElement { <Self as SimdFloat>::cast(self) }

            #[inline(always)] fn reduce_sum(self) -> $ty { <Self as SimdFloat>::reduce_sum(self) }
            #[inline(always)] fn reduce_product(self) -> $ty { <Self as SimdFloat>::reduce_product(self) }
            #[inline(always)] fn reduce_min(self) -> $ty { <Self as SimdFloat>::reduce_min(self) }
            #[inline(always)] fn reduce_max(self) -> $ty { <Self as SimdFloat>::reduce_max(self) }
            #[inline(always)] fn simd_min(self, other: Self) -> Self { <Self as SimdFloat>::simd_min(self, other) }
            #[inline(always)] fn simd_max(self, other: Self) -> Self { <Self as SimdFloat>::simd_max(self, other) }
            #[inline(always)] fn simd_clamp(self, min: Self, max: Self) -> Self { <Self as SimdFloat>::simd_clamp(self, min, max) }

            #[inline(always)] fn swap_bytes(self) -> Self {
                let bits = self.to_bits();
                let swapped = SimdUint::swap_bytes(bits);
                Self::from_bits(swapped)
            }

            #[inline(always)] fn reverse_bits(self) -> Self {
                let bits = self.to_bits();
                let reversed = SimdUint::reverse_bits(bits);
                Self::from_bits(reversed)
            }
        }

        impl <const N: usize> SimdSigned<$ty> for Simd<$ty, N>
        where
            LaneCount<N>: SupportedLaneCount,
            Self: SimdFloat<Bits = Simd<$bits, N>, Scalar = $ty, Mask = Mask<<$ty as StdSimdElement>::Mask, N>>,
        {
            type Mask = Mask<<$ty as StdSimdElement>::Mask, N>;

            const NEG_ONE: $ty = -1.0;
            const MIN_POSITIVE: $ty = <$ty>::MIN_POSITIVE;

            #[inline(always)] fn abs(self) -> Self { <Self as SimdFloat>::abs(self) }
            #[inline(always)] fn signum(self) -> Self { <Self as SimdFloat>::signum(self) }
            #[inline(always)] fn is_sign_positive(self) -> Self::Mask { <Self as SimdFloat>::is_sign_positive(self) }
            #[inline(always)] fn is_sign_negative(self) -> Self::Mask { <Self as SimdFloat>::is_sign_negative(self) }
            #[inline(always)] fn copysign(self, sign: Self) -> Self { <Self as SimdFloat>::copysign(self, sign) }
        }
    };
}

impl_float_simd!(f32: u32);
impl_float_simd!(f64: u64);

#[rustfmt::skip]
macro_rules! impl_signed_int_simd {
    ($ty:ty) => {
        impl<const N: usize> SimdNum<$ty> for Simd<$ty, N>
        where
            LaneCount<N>: SupportedLaneCount,
            Self: SimdInt<Scalar = $ty, Mask = Mask<<$ty as StdSimdElement>::Mask, N>>,
        {
            type Cast<U: StdSimdElement> = <Self as SimdInt>::Cast<U>;

            const TWO: $ty = 2;
            const MIN: $ty = <$ty>::MIN;
            const MAX: $ty = <$ty>::MAX;

            #[inline(always)]
            fn cast<U>(self) -> Self::Cast<U> where U: SimdCast { <Self as SimdInt>::cast(self) }

            #[inline(always)] fn reduce_sum(self) -> $ty { <Self as SimdInt>::reduce_sum(self) }
            #[inline(always)] fn reduce_product(self) -> $ty { <Self as SimdInt>::reduce_product(self) }
            #[inline(always)] fn reduce_min(self) -> $ty { <Self as SimdInt>::reduce_min(self) }
            #[inline(always)] fn reduce_max(self) -> $ty { <Self as SimdInt>::reduce_max(self) }
            #[inline(always)] fn simd_min(self, other: Self) -> Self { Simd::min(self, other) }
            #[inline(always)] fn simd_max(self, other: Self) -> Self { Simd::max(self, other) }
            #[inline(always)] fn simd_clamp(self, min: Self, max: Self) -> Self { Simd::clamp(self, min, max) }

            #[inline(always)] fn swap_bytes(self) -> Self { <Self as SimdInt>::swap_bytes(self) }
            #[inline(always)] fn reverse_bits(self) -> Self { <Self as SimdInt>::reverse_bits(self) }
        }

        impl<const N: usize> SimdSigned<$ty> for Simd<$ty, N>
        where
            LaneCount<N>: SupportedLaneCount,
            Self: SimdInt<Scalar = $ty, Mask = Mask<<$ty as StdSimdElement>::Mask, N>>,
        {
            type Mask = Mask<<$ty as StdSimdElement>::Mask, N>;

            const NEG_ONE: $ty = -1;
            const MIN_POSITIVE: $ty = 1;

            #[inline(always)] fn abs(self) -> Self { <Self as SimdInt>::abs(self) }
            #[inline(always)] fn signum(self) -> Self { <Self as SimdInt>::signum(self) }
            #[inline(always)] fn is_sign_positive(self) -> Self::Mask { <Self as SimdInt>::is_positive(self) }
            #[inline(always)] fn is_sign_negative(self) -> Self::Mask { <Self as SimdInt>::is_negative(self) }
            #[inline(always)] fn copysign(self, sign: Self) -> Self {
                let abs = SimdSigned::abs(self);
                sign.is_sign_negative().select(abs.neg(), abs)
            }
        }
    };
}

impl_signed_int_simd!(i8);
impl_signed_int_simd!(i16);
impl_signed_int_simd!(i32);
impl_signed_int_simd!(i64);
impl_signed_int_simd!(isize);

// pub trait GenericSimdVector<T: SimdElement, const N: usize>: SimdNum<T> {
//     type UnsignedSimd:
// }

impl<T: SimdElement, const N: usize> GenericVector for Simd<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Const<N>: Lanes,
    Self: SimdNum<T>,
    Simd<<T as Element>::Unsigned, N>: SimdNum<<T as Element>::Unsigned>,
    Simd<<T as Element>::Signed, N>: SimdSigned<<T as Element>::Signed>,
{
    type Element = T;

    const EMPTY: Self = Simd::splat(T::ZERO);

    const LANES: usize = N;
    const ISA: InstructionSet = InstructionSet::Unknown;

    type Lanes = Const<N>;

    type Unsigned = Simd<<T as Element>::Unsigned, N>;
    type Signed = Simd<<T as Element>::Signed, N>;

    type Mask = Mask<<T as StdSimdElement>::Mask, N>;

    fn splat(value: Self::Element) -> Self {
        Simd::splat(value)
    }

    fn single(value: Self::Element) -> Self {
        todo!("set only the first element")
    }

    fn splat_const<C>() -> Self
    where
        C: crate::prelude::SplatConst<Self::Element>,
    {
        const { Simd::splat(C::VALUE) }
    }

    fn extract<const I: usize>(self) -> Self::Element {
        self[I]
    }

    fn extractv(self, idx: usize) -> Self::Element {
        self[idx]
    }

    fn insert<const I: usize>(mut self, value: Self::Element) -> Self {
        self[I] = value;
        self
    }

    fn insertv(mut self, idx: usize, value: Self::Element) -> Self {
        self[idx] = value;
        self
    }

    fn reverse(mut self) -> Self {
        self.as_mut_array().reverse();
        self
    }

    unsafe fn load(ptr: *const Self::Element) -> Self {
        unsafe { *(ptr as *const Self) }
    }

    unsafe fn load_unaligned(ptr: *const Self::Element) -> Self {
        unsafe { core::ptr::read_unaligned(ptr as *const Self) }
    }

    unsafe fn load_streaming(ptr: *const Self::Element) -> Self {
        // regular aligned load since we don't have streaming loads with std simd
        unsafe { Self::load(ptr) }
    }

    unsafe fn store(self, ptr: *mut Self::Element) {
        unsafe { *(ptr as *mut Self) = self }
    }

    unsafe fn store_unaligned(self, ptr: *mut Self::Element) {
        unsafe { core::ptr::write_unaligned(ptr as *mut Self, self) }
    }

    unsafe fn store_streaming(self, ptr: *mut Self::Element) {
        // regular aligned store since we don't have streaming stores with std simd
        unsafe { Self::store(self, ptr) }
    }

    fn broadcast<const I: usize>(self) -> Self {
        Self::splat(GenericVector::extract::<I>(self))
    }

    fn broadcastv(self, idx: usize) -> Self {
        Self::splat(GenericVector::extractv(self, idx))
    }

    fn swap_bytes(self) -> Self {
        SimdNum::swap_bytes(self)
    }

    const HAS_SIMPLE_UNPACK: bool = true;

    fn unpack(self, other: Self) -> (Self, Self) {
        self.deinterleave(other)
    }

    fn z(self, mask: Self::Mask) -> Self {
        mask.select(self, Self::EMPTY)
    }

    fn nz(self, mask: Self::Mask) -> Self {
        mask.select(Self::EMPTY, self)
    }

    fn map<F>(mut self, f: F) -> Self
    where
        F: Fn(Self::Element) -> Self::Element,
    {
        for i in 0..N {
            self[i] = f(self[i]);
        }

        self
    }

    fn fold<F>(self, mut init: Self::Element, f: F) -> Self::Element
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element,
    {
        for i in 0..N {
            init = f(init, self[i]);
        }

        init
    }

    fn reduce<F>(self, f: F) -> Self::Element
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element,
    {
        let mut acc = self[0];
        for i in 1..N {
            acc = f(acc, self[i]);
        }
        acc
    }

    fn splat_m(src: Self, mask: Self::Mask, value: Self::Element) -> Self {
        mask.select(Self::splat(value), src)
    }

    fn splat_z(mask: Self::Mask, value: Self::Element) -> Self {
        mask.select(Self::splat(value), Self::EMPTY)
    }

    unsafe fn load_m(src: Self, mask: Self::Mask, ptr: *const Self::Element) -> Self {
        todo!()
    }

    unsafe fn load_z(mask: Self::Mask, ptr: *const Self::Element) -> Self {
        todo!()
    }

    fn broadcast_c<const I: usize>(self, mask: Self::Mask) -> Self {
        mask.select(Self::splat(GenericVector::extract::<I>(self)), self)
    }

    fn broadcast_m<const I: usize>(self, src: Self, mask: Self::Mask) -> Self {
        mask.select(Self::splat(GenericVector::extract::<I>(self)), src)
    }

    fn broadcast_z<const I: usize>(self, mask: Self::Mask) -> Self {
        mask.select(Self::splat(GenericVector::extract::<I>(self)), Self::EMPTY)
    }

    fn broadcastv_c(self, mask: Self::Mask, idx: usize) -> Self {
        mask.select(Self::splat(GenericVector::extractv(self, idx)), self)
    }

    fn broadcastv_m(self, src: Self, mask: Self::Mask, idx: usize) -> Self {
        mask.select(Self::splat(GenericVector::extractv(self, idx)), src)
    }

    fn broadcastv_z(self, mask: Self::Mask, idx: usize) -> Self {
        mask.select(Self::splat(GenericVector::extractv(self, idx)), Self::EMPTY)
    }

    fn reverse_c(self, mask: Self::Mask) -> Self {
        todo!()
    }

    fn reverse_m(self, src: Self, mask: Self::Mask) -> Self {
        todo!()
    }

    fn reverse_z(self, mask: Self::Mask) -> Self {
        todo!()
    }

    fn swap_bytes_c(self, mask: Self::Mask) -> Self {
        todo!()
    }

    fn swap_bytes_m(self, src: Self, mask: Self::Mask) -> Self {
        todo!()
    }

    fn swap_bytes_z(self, mask: Self::Mask) -> Self {
        todo!()
    }
}

impl<T: MaskElement, U: MaskElement, const N: usize> CastMask<Mask<U, N>> for Mask<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Const<N>: Lanes,
{
    #[inline(always)]
    fn mask_from(from: Mask<U, N>) -> Self {
        from.cast()
    }
}

impl<T: SimdElement, const N: usize> GenericSelectable for Simd<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Const<N>: Lanes,
{
    type SelectableMask = Mask<<T as StdSimdElement>::Mask, N>;

    #[inline(always)]
    fn select<M>(mask: M, t: Self, f: Self) -> Self
    where
        Self::SelectableMask: CastMask<M>,
    {
        Self::SelectableMask::mask_from(mask).select(t, f)
    }
}

trait SameSizeElement<T> {}

macro_rules! impl_same_size_element {
    ($($a:ty = $b:ty),*) => {
        $(
            impl SameSizeElement<$b> for $a {}
            impl SameSizeElement<$a> for $b {}
        )*
    };
}

impl_same_size_element!(
    u8 = i8,
    u16 = i16,
    u32 = i32,
    u64 = i64,
    f32 = i32,
    f32 = u32,
    f64 = i64,
    f64 = u64
);

impl<T: SimdElement, U: SimdElement, const N: usize> BitCastVector<Simd<U, N>> for Simd<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Const<N>: Lanes,
    T: SameSizeElement<U>,
{
    #[inline(always)]
    fn from_bits(bits: Simd<U, N>) -> Self {
        unsafe { core::mem::transmute_copy(&bits) }
    }
}

impl<T: MaskElement, const N: usize> GenericMask for Mask<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Const<N>: Lanes,
{
    // It's downright stupid Mask::splat isn't const
    // const FALSY: Self = Mask::splat(false);
    // const TRUTHY: Self = Mask::splat(true);

    const FALSY: Self = unsafe { core::mem::zeroed() };
    const TRUTHY: Self = unsafe {
        // no matter what kind of mask it is, all bits set to 1 should be the "true" value
        let mut m: Self = core::mem::zeroed();
        core::ptr::write_bytes(&mut m, !0, size_of::<Self>());
        m
    };

    fn all(self) -> bool {
        self.all()
    }

    fn any(self) -> bool {
        self.any()
    }

    fn none(self) -> bool {
        !self.any()
    }

    fn native_bitmask(&self) -> Option<u64> {
        Some(self.to_bitmask())
    }

    fn bitmask(&self) -> bitvec::prelude::BitArray<impl bitvec::view::BitViewSized<Store = u32>> {
        let mut bitmask = bitvec::array::BitArray::<<Const<N> as Lanes>::BitmaskStorage>::ZERO;

        let bits = unsafe { core::mem::transmute::<u64, [u32; 2]>(self.to_bitmask()) };
        let bits = bitvec::slice::BitSlice::<u32>::from_slice(&bits);
        bitmask[..N].copy_from_bitslice(&bits[..N]);

        bitmask
    }

    fn ternlog<const IMM: i32>(a: Self, b: Self, c: Self) -> Self {
        todo!()
    }
}

impl<T: MaskElement, const N: usize> BitAndNot for Mask<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Const<N>: Lanes,
{
    type Output = Self;

    #[inline(always)]
    fn bitandnot(self, rhs: Self) -> Self {
        self & !rhs
    }
}

impl<T: MaskElement, const N: usize> BitAndNotAssign for Mask<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Const<N>: Lanes,
{
    #[inline(always)]
    fn bitandnot_assign(&mut self, rhs: Self) {
        *self &= !rhs;
    }
}

macro_rules! impl_masked_ops {
    (BINARY $trait:ident::$method:ident => $($ty:ty),*) => {paste::paste! {$(
        const _: () = {
            use core::ops::$trait;

            type Mask<const N: usize> = std::simd::Mask<<$ty as StdSimdElement>::Mask, N>;

            impl<const N: usize> crate::generic::ops::[<$trait Masked>]<Mask<N>> for Simd<$ty, N>
            where
                LaneCount<N>: SupportedLaneCount,
            {
                #[inline(always)] fn [<$method _c>](self, mask: Mask<N>, rhs: Self) -> Self {
                    mask.select(self.$method(rhs), self)
                }
                #[inline(always)] fn [<$method _m>](self, src: Self, mask: Mask<N>, rhs: Self) -> Self {
                    mask.select(self.$method(rhs), src)
                }
                #[inline(always)] fn [<$method _z>](self, mask: Mask<N>, rhs: Self) -> Self {
                    mask.select(self.$method(rhs), Simd::splat(0 as _))
                }
            }

            impl<const N: usize> crate::generic::ops::[<$trait AssignMasked>]<Mask<N>> for Simd<$ty, N>
            where
                LaneCount<N>: SupportedLaneCount,
            {
                #[inline(always)] fn [<$method _assign_c>](&mut self, mask: Mask<N>, rhs: Self) {
                    *self = mask.select(self.$method(rhs), *self);
                }
                #[inline(always)] fn [<$method _assign_m>](&mut self, src: Self, mask: Mask<N>, rhs: Self) {
                    *self = mask.select(self.$method(rhs), src);
                }
                #[inline(always)] fn [<$method _assign_z>](&mut self, mask: Mask<N>, rhs: Self) {
                    *self = mask.select(self.$method(rhs), Simd::splat(0 as _));
                }
            }
        };
    )*}};

    (UNARY $trait:ident::$method:ident => $($ty:ty),*) => {paste::paste! {$(
        const _: () = {
            use core::ops::$trait;

            type Mask<const N: usize> = std::simd::Mask<<$ty as StdSimdElement>::Mask, N>;

            impl<const N: usize> crate::generic::ops::[<$trait Masked>]<Mask<N>> for Simd<$ty, N>
            where
                LaneCount<N>: SupportedLaneCount,
            {
                #[inline(always)] fn [<$method _c>](self, mask: Mask<N>) -> Self {
                    mask.select(self.$method(), self)
                }
                #[inline(always)] fn [<$method _m>](self, src: Self, mask: Mask<N>) -> Self {
                    mask.select(self.$method(), src)
                }
                #[inline(always)] fn [<$method _z>](self, mask: Mask<N>) -> Self {
                    mask.select(self.$method(), Simd::splat(0 as _))
                }
            }
        };
    )*}};

    (SQUARE $($ty:ty),*) => {$(
        const _: () = {
            use crate::generic::ops::{Square, SquareMasked};

            type Mask<const N: usize> = std::simd::Mask<<$ty as StdSimdElement>::Mask, N>;

            impl<const N: usize> Square for Simd<$ty, N>
            where
                LaneCount<N>: SupportedLaneCount,
                Self: core::ops::Mul<Self, Output = Self>,
            {
                type Output = Self;
                #[inline(always)] fn square(self) -> Self { self * self }
            }

            impl<const N: usize> SquareMasked<Mask<N>> for Simd<$ty, N>
            where
                LaneCount<N>: SupportedLaneCount,
                Self: core::ops::Mul<Self, Output = Self>,
            {
                #[inline(always)] fn square_c(self, mask: Mask<N>) -> Self {
                    mask.select(self.square(), self)
                }
                #[inline(always)] fn square_m(self, src: Self, mask: Mask<N>) -> Self {
                    mask.select(self.square(), src)
                }
                #[inline(always)] fn square_z(self, mask: Mask<N>) -> Self {
                    mask.select(self.square(), Simd::splat(0 as _))
                }
            }
        };
    )*};
}

impl_masked_ops!(BINARY BitAnd::bitand => u8, i8, u16, i16, u32, i32, u64, i64);
impl_masked_ops!(BINARY BitOr::bitor => u8, i8, u16, i16, u32, i32, u64, i64);
impl_masked_ops!(BINARY BitXor::bitxor => u8, i8, u16, i16, u32, i32, u64, i64);
impl_masked_ops!(BINARY Add::add => u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);
impl_masked_ops!(BINARY Sub::sub => u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);
impl_masked_ops!(BINARY Mul::mul => u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);
impl_masked_ops!(BINARY Div::div => u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);
impl_masked_ops!(BINARY Rem::rem => u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);
impl_masked_ops!(UNARY Neg::neg => i8, i16, i32, i64, f32, f64);
impl_masked_ops!(UNARY Not::not => u8, i8, u16, i16, u32, i32, u64, i64);

impl_masked_ops!(SQUARE u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);

#[rustfmt::skip]
impl<T: SimdElement, const N: usize> BitwiseVector for Simd<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Const<N>: Lanes,
    Self: GenericVector,
    Self: crate::generic::ops::NotMasked<Self::Mask, Output = Self>,
    Self: crate::generic::ops::BitAndMasked<Self::Mask, Self, Output = Self>,
    Self: crate::generic::ops::BitAndAssignMasked<Self::Mask, Self>,
    Self: crate::generic::ops::BitAndNotMasked<Self::Mask, Self, Output = Self>,
    Self: crate::generic::ops::BitAndNotAssignMasked<Self::Mask, Self>,
    Self: crate::generic::ops::BitOrMasked<Self::Mask, Self, Output = Self>,
    Self: crate::generic::ops::BitOrAssignMasked<Self::Mask, Self>,
    Self: crate::generic::ops::BitXorMasked<Self::Mask, Self, Output = Self>,
    Self: crate::generic::ops::BitXorAssignMasked<Self::Mask, Self>,
    Self: crate::generic::ops::NotMasked<Self::Mask, Output = Self>,
{
    fn ternlog<const IMM: i32>(a: Self, b: Self, c: Self) -> Self {
        todo!()
    }

    fn bilog<const IMM: i32>(a: Self, b: Self) -> Self {
        todo!()
    }

    #[inline(always)] fn ternlog_c<const IMM: i32>(mask: Self::Mask, a: Self, b: Self, c: Self) -> Self {
        mask.select(Self::ternlog::<IMM>(a, b, c), a)
    }

    #[inline(always)] fn ternlog_m<const IMM: i32>(src: Self, mask: Self::Mask, a: Self, b: Self, c: Self) -> Self {
        mask.select(Self::ternlog::<IMM>(a, b, c), src)
    }

    #[inline(always)] fn ternlog_z<const IMM: i32>(mask: Self::Mask, a: Self, b: Self, c: Self) -> Self {
        mask.select(Self::ternlog::<IMM>(a, b, c), Self::EMPTY)
    }

    #[inline(always)] fn bilog_c<const IMM: i32>(mask: Self::Mask, a: Self, b: Self) -> Self {
        mask.select(Self::bilog::<IMM>(a, b), a)
    }

    #[inline(always)] fn bilog_m<const IMM: i32>(src: Self, mask: Self::Mask, a: Self, b: Self) -> Self {
        mask.select(Self::bilog::<IMM>(a, b), src)
    }

    #[inline(always)] fn bilog_z<const IMM: i32>(mask: Self::Mask, a: Self, b: Self) -> Self {
        mask.select(Self::bilog::<IMM>(a, b), Self::EMPTY)
    }
}

#[rustfmt::skip]
impl<T: SimdElement, const N: usize> PartialOrdVector for Simd<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Const<N>: Lanes,
    Self: GenericVector + SimdPartialOrd<Mask = <Self as GenericVector>::Mask>,
{
    #[inline(always)] fn cmp_eq(self, other: Self) -> Self::Mask { self.simd_eq(other) }
    #[inline(always)] fn cmp_ne(self, other: Self) -> Self::Mask { self.simd_ne(other) }
    #[inline(always)] fn cmp_lt(self, other: Self) -> Self::Mask { self.simd_lt(other) }
    #[inline(always)] fn cmp_le(self, other: Self) -> Self::Mask { self.simd_le(other) }
    #[inline(always)] fn cmp_gt(self, other: Self) -> Self::Mask { self.simd_gt(other) }
    #[inline(always)] fn cmp_ge(self, other: Self) -> Self::Mask { self.simd_ge(other) }
}

impl<T: SimdElement, const N: usize> NumericVector for Simd<T, N>
where
    T: num_traits::NumOps,
    LaneCount<N>: SupportedLaneCount,
    Const<N>: Lanes,
    Self: SimdNum<T> + GenericVector<Element = T> + PartialOrdVector,
    Self: crate::generic::ops::AddMasked<Self::Mask, Self, Output = Self>,
    Self: crate::generic::ops::AddAssignMasked<Self::Mask, Self>,
    Self: crate::generic::ops::SubMasked<Self::Mask, Self, Output = Self>,
    Self: crate::generic::ops::SubAssignMasked<Self::Mask, Self>,
    Self: crate::generic::ops::MulMasked<Self::Mask, Self, Output = Self>,
    Self: crate::generic::ops::MulAssignMasked<Self::Mask, Self>,
    Self: crate::generic::ops::DivMasked<Self::Mask, Self, Output = Self>,
    Self: crate::generic::ops::DivAssignMasked<Self::Mask, Self>,
    Self: crate::generic::ops::RemMasked<Self::Mask, Self, Output = Self>,
    Self: crate::generic::ops::RemAssignMasked<Self::Mask, Self>,
    Self: crate::generic::ops::SquareMasked<Self::Mask, Output = Self>,
    Self: num_traits::NumOps<Self>,
    Self: num_traits::NumAssignOps<Self>,
    Self: core::iter::Sum,
    Self: core::iter::Product,
{
    const ZERO: Self = Simd::splat(T::ZERO);
    const ONE: Self = Simd::splat(T::ONE);
    const TWO: Self = Simd::splat(<Self as SimdNum<T>>::TWO);

    const MIN: Self = Simd::splat(<Self as SimdNum<T>>::MIN);
    const MAX: Self = Simd::splat(<Self as SimdNum<T>>::MAX);

    fn is_zero(self) -> Self::Mask {
        self.cmp_eq(Self::ZERO)
    }

    fn min(self, other: Self) -> Self {
        self.simd_min(other)
    }

    fn max(self, other: Self) -> Self {
        self.simd_max(other)
    }

    fn clamp(self, min: Self, max: Self) -> Self {
        self.simd_clamp(min, max)
    }

    fn min_element(self) -> Self::Element {
        self.reduce_min()
    }

    fn max_element(self) -> Self::Element {
        self.reduce_max()
    }

    fn sum_elements(self) -> Self::Element {
        self.reduce_sum()
    }

    fn prod_elements(self) -> Self::Element {
        self.reduce_product()
    }

    fn offset() -> Self {
        Self::splat(T::from_u16(Self::LANES as u16))
    }

    fn indexed() -> Self {
        let mut arr = [T::ZERO; N];
        for i in 0..N {
            arr[i] = T::from_u16(i as u16);
        }
        Self::from_array(arr)
    }

    fn min_c(self, mask: Self::Mask, other: Self) -> Self {
        mask.select(self.simd_min(other), self)
    }

    fn min_m(self, src: Self, mask: Self::Mask, other: Self) -> Self {
        mask.select(self.simd_min(other), src)
    }

    fn min_z(self, mask: Self::Mask, other: Self) -> Self {
        mask.select(self.simd_min(other), Self::EMPTY)
    }

    fn max_c(self, mask: Self::Mask, other: Self) -> Self {
        mask.select(self.simd_max(other), self)
    }

    fn max_m(self, src: Self, mask: Self::Mask, other: Self) -> Self {
        mask.select(self.simd_max(other), src)
    }

    fn max_z(self, mask: Self::Mask, other: Self) -> Self {
        mask.select(self.simd_max(other), Self::EMPTY)
    }
}

impl<T: SimdElement, const N: usize> SignedVector for Simd<T, N>
where
    T: num_traits::NumOps,
    LaneCount<N>: SupportedLaneCount,
    Const<N>: Lanes,
    Self: NumericVector,
    Self: crate::generic::ops::NegMasked<Self::Mask, Output = Self>,
    Self: SimdSigned<T, Mask = Self::Mask>,
{
    const NEG_ONE: Self = Simd::splat(<Self as SimdSigned<T>>::NEG_ONE);
    const MIN_POSITIVE: Self = Simd::splat(<Self as SimdSigned<T>>::MIN_POSITIVE);

    fn abs(self) -> Self {
        SimdSigned::abs(self)
    }

    fn signum(self) -> Self {
        SimdSigned::signum(self)
    }

    fn copysign(self, sign: Self) -> Self {
        SimdSigned::copysign(self, sign)
    }

    fn is_positive(self) -> Self::Mask {
        SimdSigned::is_sign_positive(self)
    }

    fn is_negative(self) -> Self::Mask {
        SimdSigned::is_sign_negative(self)
    }

    fn select_negative(self, if_neg: Self, if_pos: Self) -> Self {
        self.is_negative().select(if_neg, if_pos)
    }

    fn abs_c(self, mask: Self::Mask) -> Self {
        mask.select(SimdSigned::abs(self), self)
    }

    fn abs_m(self, src: Self, mask: Self::Mask) -> Self {
        mask.select(SimdSigned::abs(self), src)
    }

    fn abs_z(self, mask: Self::Mask) -> Self {
        mask.select(SimdSigned::abs(self), Self::EMPTY)
    }

    fn copysign_c(self, mask: Self::Mask, sign: Self) -> Self {
        mask.select(SimdSigned::copysign(self, sign), self)
    }

    fn copysign_m(self, src: Self, mask: Self::Mask, sign: Self) -> Self {
        mask.select(SimdSigned::copysign(self, sign), src)
    }

    fn copysign_z(self, mask: Self::Mask, sign: Self) -> Self {
        mask.select(SimdSigned::copysign(self, sign), Self::EMPTY)
    }
}
