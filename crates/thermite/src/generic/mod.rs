#![allow(missing_docs, clippy::missing_safety_doc)]
#![deny(unconditional_recursion)] // just in case we miss one

use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, Index, Mul, Neg, Not, Rem,
    Shl, ShlAssign, Shr, ShrAssign, Sub,
};

use generic_array::GenericArray;

use crate::{
    BranchfreeDivider, Divider, Mask, Swizzle, Vector,
    divider::{Denominator, vector::VectorDivider},
    math::FloatConsts,
    register::{CastMaskRegister, Element, FloatElement, Lanes},
};

/// Simple associated constant splat trait.
///
/// Used with `GenericVector::splat_const` to splat compile-time constant values into vectors.
///
/// This is effectively a workaround for the lack of `const generics` for generic types.
pub trait SplatConst<E> {
    const VALUE: E;
}

/// Macro to splat a compile-time constant value into all lanes of a generic vector.
///
/// There are three forms of this macro:
/// ```ignore
/// // 1. Generic type parameters with bounds
/// // used when the type depends on generic parameters, and especially `Self`
/// let exp_lsb_mask: Self::Bits = crate::generic_splat!(
///     <Self> = <S: FloatVector>
///     <S::Bits as GenericVector>::Element: <S::Element as FloatElement>::EXP_LSB_MASK
/// );
///
/// // 2. Static type, direct value
/// // used when the type is known and the value is a literal or const expression
/// let zero: Vector<u32x4> = crate::generic_splat!(u32: 0);
///
/// // 3. Associated const value
/// // used when the value is an associated constant of a type
/// let infinity: Vector<f32x4> = crate::generic_splat!(<f32>::INFINITY);
/// ```
#[macro_export]
macro_rules! generic_splat {
    (
        <$($real_param:ident),+> = <$($gen_param:ident $(: $bound:path)?),+ $(,)?>
        $ty:ty : $value:expr
    ) => {{
        use core::marker::PhantomData;

        struct __GenericSplatValue<$($gen_param $(: $bound)?),+>(
            PhantomData<($($gen_param),+)>
        );

        impl<$($gen_param $(: $bound)?),+> $crate::generic::SplatConst<$ty>
        for __GenericSplatValue<$($gen_param),+> {
            const VALUE: $ty = const { $value };
        }

        $crate::generic::GenericVector::splat_const::<
            __GenericSplatValue<$($real_param),+>
        >()
    }};

    // Static type, direct value
    ($ty:ty: $value:expr) => {{
        struct __ConstSplatValue;
        impl $crate::generic::SplatConst<$ty> for __ConstSplatValue {
            const VALUE: $ty = const { $value };
        }
        $crate::generic::GenericVector::splat_const::<__ConstSplatValue>()
    }};

    // Associated const value
    (<$ty:ty $(as $trait:path)?>::$associated:ident) => {{
        struct __ConstSplatValue;
        impl $crate::generic::SplatConst<$ty> for __ConstSplatValue {
            const VALUE: $ty = const { <$ty $(as $trait)?>::$associated };
        }
        $crate::generic::GenericVector::splat_const::<__ConstSplatValue>()
    }};
}

pub trait MaskInteroperable<A, B>: GenericVector<Mask: CastMask<A::Mask> + CastMask<B::Mask>>
where
    A: GenericVector<Lanes = Self::Lanes, Mask: CastMask<Self::Mask> + CastMask<B::Mask>>,
    B: GenericVector<Lanes = Self::Lanes, Mask: CastMask<Self::Mask> + CastMask<A::Mask>>,
{
}

pub trait PartiallyInteroperable<A, B>:
    GenericVector<Mask: CastMask<A::Mask> + CastMask<B::Mask>>
    // casts
    + CastVector<Self>
    + CastVector<A>
    + CastVector<B>
where
    A: CastVector<Self> + GenericVector<Lanes = Self::Lanes, Mask: CastMask<Self::Mask> + CastMask<B::Mask>>,
    B: CastVector<Self> + GenericVector<Lanes = Self::Lanes, Mask: CastMask<Self::Mask> + CastMask<A::Mask>>,
{
}

impl<V, A, B> PartiallyInteroperable<A, B> for V
where
    V: GenericVector<Mask: CastMask<A::Mask> + CastMask<B::Mask>>
        // casts
        + CastVector<V>
        + CastVector<A>
        + CastVector<B>,
    A: CastVector<V> + GenericVector<Lanes = V::Lanes, Mask: CastMask<V::Mask> + CastMask<B::Mask>>,
    B: CastVector<V> + GenericVector<Lanes = V::Lanes, Mask: CastMask<V::Mask> + CastMask<A::Mask>>,
{
}

pub trait FullyInteroperable<A, B>:
    GenericVector<Mask: CastMask<A::Mask> + CastMask<B::Mask>>
    // bits
    + BitsVector<Self>
    + BitsVector<A>
    + BitsVector<B>
    // casts
    + CastVector<Self>
    + CastVector<A>
    + CastVector<B>
where
    A: BitsVector<Self> + CastVector<Self> + GenericVector<Lanes = Self::Lanes, Mask: CastMask<Self::Mask> + CastMask<B::Mask>>,
    B: BitsVector<Self> + CastVector<Self> + GenericVector<Lanes = Self::Lanes, Mask: CastMask<Self::Mask> + CastMask<A::Mask>>,
{
}

impl<V, A, B> FullyInteroperable<A, B> for V
where
    V: GenericVector<Mask: CastMask<A::Mask> + CastMask<B::Mask>>
        // bits
        + BitsVector<V>
        + BitsVector<A>
        + BitsVector<B>
        // casts
        + CastVector<V>
        + CastVector<A>
        + CastVector<B>,
    A: BitsVector<V> + CastVector<V> + GenericVector<Lanes = V::Lanes, Mask: CastMask<V::Mask> + CastMask<B::Mask>>,
    B: BitsVector<V> + CastVector<V> + GenericVector<Lanes = V::Lanes, Mask: CastMask<V::Mask> + CastMask<A::Mask>>,
{
}

/// Core trait for generic vector types.
///
/// Provides the basis for further specialized vector traits.
pub trait GenericVector:
    Sized
    + Copy
    + core::fmt::Debug
    + 'static
    + BitAnd<Self, Output = Self>
    + BitAndAssign<Self>
    + BitOr<Self, Output = Self>
    + BitOrAssign<Self>
    + BitXor<Self, Output = Self>
    + BitXorAssign<Self>
    + Not<Output = Self>
    + Index<usize, Output = Self::Element>
    + GenericSelectable<SelectableMask = Self::Mask>
{
    type Element: Element;

    const EMPTY: Self;
    const LANES: usize;

    type Lanes: Lanes;

    type USize: UnsignedIntegerVector<
            ISize = Self::ISize,
            USize = Self::USize,
            Lanes = Self::Lanes,
            Element = <Self::Element as Element>::USize,
            Mask: CastMask<Self::Mask>,
        > + CastVector<Self::ISize>
        + BitsVector<Self::ISize>;

    type ISize: SignedIntegerVector<
            ISize = Self::ISize,
            USize = Self::USize,
            Lanes = Self::Lanes,
            Element = <Self::Element as Element>::ISize,
            Mask: CastMask<Self::Mask>,
        > + CastVector<Self::USize>
        + BitsVector<Self::USize>;

    type Mask: GenericMask<Self>
        + GenericSelectable<SelectableMask = Self::Mask>
        + CastMask<<Self::USize as GenericVector>::Mask>
        + CastMask<<Self::ISize as GenericVector>::Mask>;

    fn splat(value: Self::Element) -> Self;

    /// Splat a compile-time constant value into all lanes of the vector.
    ///
    /// Use the `generic_splat!` macro to call this function with easier syntax.
    #[inline(always)]
    fn splat_const<C>() -> Self
    where
        C: SplatConst<Self::Element>,
    {
        Self::splat(C::VALUE)
    }

    fn single(value: Self::Element) -> Self;

    unsafe fn load(ptr: *const Self::Element) -> Self;
    unsafe fn load_unaligned(ptr: *const Self::Element) -> Self;
    unsafe fn load_streaming(ptr: *const Self::Element) -> Self;

    unsafe fn store(self, ptr: *mut Self::Element);
    unsafe fn store_unaligned(self, ptr: *mut Self::Element);
    unsafe fn store_streaming(self, ptr: *mut Self::Element);

    fn broadcast<const I: usize>(self) -> Self;
    fn broadcastv(self, idx: usize) -> Self;

    fn as_slice(&self) -> &[Self::Element];
    fn as_mut_slice(&mut self) -> &mut [Self::Element];

    fn extract<const I: usize>(self) -> Self::Element;
    fn insert<const I: usize>(self, value: Self::Element) -> Self;
    fn reverse(self) -> Self;
    fn swap_bytes(self) -> Self;

    /// !self & other
    fn bitandnot(self, other: Self) -> Self;

    const HAS_SIMPLE_UNPACK: bool;

    fn unpack(self, other: Self) -> (Self, Self);

    fn map<F>(self, f: F) -> Self
    where
        F: Fn(Self::Element) -> Self::Element;

    fn fold<F>(self, init: Self::Element, f: F) -> Self::Element
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element;

    fn reduce<F>(self, f: F) -> Self::Element
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element;

    #[inline(always)]
    fn cast<INTO>(self) -> INTO
    where
        INTO: CastVector<Self>,
    {
        INTO::cast_from(self)
    }

    #[inline(always)]
    fn fast_cast<INTO>(self) -> INTO
    where
        INTO: CastVector<Self>,
    {
        INTO::fast_cast_from(self)
    }

    #[inline(always)]
    fn into_bits<INTO>(self) -> INTO
    where
        INTO: BitsVector<Self>,
    {
        INTO::from_bits(self)
    }

    const HAS_MSB_BLENDV: bool;
}

pub trait BitshiftVector:
    GenericVector
    + Shr<Self::USize, Output = Self>
    + ShrAssign<Self::USize>
    + Shl<Self::USize, Output = Self>
    + ShlAssign<Self::USize>
    + Shr<u32, Output = Self>
    + ShrAssign<u32>
    + Shl<u32, Output = Self>
    + ShlAssign<u32>
{
    const HAS_TRUE_SHIFTV: bool;
    const HAS_WIDE_BYTE_SHIFTS: bool;

    fn bshli<const I: i32>(self) -> Self;
    fn bshri<const I: i32>(self) -> Self;
    fn shli<const I: i32>(self) -> Self;
    fn shri<const I: i32>(self) -> Self;
    fn shlv(self, counts: Self::USize) -> Self;
    fn shrv(self, counts: Self::USize) -> Self;
}

pub trait SwizzleVector: GenericVector {
    fn swizzle(self, other: Self, indices: GenericArray<u32, Self::Lanes>) -> Self;
    fn permute(self, indices: GenericArray<u32, Self::Lanes>) -> Self;
}

pub trait CastVector<FROM: GenericVector>: GenericVector {
    fn cast_from(from: FROM) -> Self;

    #[inline(always)]
    fn fast_cast_from(from: FROM) -> Self {
        Self::cast_from(from)
    }
}

pub trait BitsVector<FROM: GenericVector>: GenericVector {
    fn from_bits(bits: FROM) -> Self;
}

pub trait GenericMask<V: GenericVector>:
    'static
    + Sized
    + Copy
    + core::fmt::Debug
    + GenericSelectable<SelectableMask = Self>
    + CastMask<Self>
    + BitAnd<Self, Output = Self>
    + BitAndAssign<Self>
    + BitOr<Self, Output = Self>
    + BitOrAssign<Self>
    + BitXor<Self, Output = Self>
    + BitXorAssign<Self>
    + Not<Output = Self>
{
    const TRUTHY: Self;
    const FALSY: Self;

    fn from_unchecked(vector: V) -> Self;

    fn all(self) -> bool;
    fn any(self) -> bool;
    fn none(self) -> bool;
    fn value(self) -> V;

    fn native_bitmask(&self) -> Option<u64>;

    #[inline(always)]
    fn select<S>(self, t: S, f: S) -> S
    where
        S: GenericSelectable<SelectableMask: CastMask<Self>>,
    {
        S::select(self, t, f)
    }

    #[inline(always)]
    fn cast_mask<INTO>(self) -> INTO
    where
        INTO: CastMask<Self>,
    {
        INTO::mask_from(self)
    }

    #[inline(always)]
    fn swap<S>(self, a: &mut S, b: &mut S)
    where
        S: GenericSelectable<SelectableMask: CastMask<Self> + CastMask<S::SelectableMask>>,
    {
        let mask = S::SelectableMask::mask_from(self);

        let a2 = S::select(mask, *a, *b);
        let b2 = S::select(mask, *b, *a);

        *a = a2;
        *b = b2;
    }
}

pub trait CastMask<FROM>: Sized {
    fn mask_from(from: FROM) -> Self;
}

pub trait GenericSelectable: Copy {
    type SelectableMask: Copy;

    fn select<M>(mask: M, t: Self, f: Self) -> Self
    where
        Self::SelectableMask: CastMask<M>;
}

pub trait PartialOrdVector: GenericVector + PartialEq {
    fn cmp_lt(self, other: Self) -> Self::Mask;
    fn cmp_le(self, other: Self) -> Self::Mask;
    fn cmp_gt(self, other: Self) -> Self::Mask;
    fn cmp_ge(self, other: Self) -> Self::Mask;
    fn cmp_eq(self, other: Self) -> Self::Mask;
    fn cmp_ne(self, other: Self) -> Self::Mask;
}

#[rustfmt::skip]
pub trait NumericVector:
    PartialOrdVector<
        Element: num_traits::Num,
        // Mask: GenericCastMask<<Self::ISize as GenericVector>::Mask> + GenericCastMask<<Self::USize as GenericVector>::Mask>,
    >
    + num_traits::NumOps
    + num_traits::NumAssignOps
    + core::iter::Sum
    + core::iter::Product
    + num_traits::Bounded
{
    const ZERO: Self;
    const ONE: Self;
    const TWO: Self;
    const MIN: Self;
    const MAX: Self;

    fn is_zero(self) -> Self::Mask;

    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn clamp(self, min: Self, max: Self) -> Self;

    fn min_element(self) -> Self::Element;
    fn max_element(self) -> Self::Element;

    fn sum_elements(self) -> Self::Element;
    fn prod_elements(self) -> Self::Element;

    fn offset() -> Self;
    fn indexed() -> Self;
}

pub trait NumVector:
    NumericVector
    + num_traits::Num
    + num_traits::NumCast
    + num_traits::NumAssign
    + num_traits::ConstOne
    + num_traits::ConstZero
{
}

pub trait SignedVector: NumericVector<Element: num_traits::Signed> + Neg<Output = Self> {
    const NEG_ONE: Self;
    const MIN_POSITIVE: Self;

    fn abs(self) -> Self;

    fn signum(self) -> Self;
    fn copysign(self, sign: Self) -> Self;

    fn is_positive(self) -> Self::Mask;
    fn is_negative(self) -> Self::Mask;

    /// Based on if self is negative, select between `if_neg` and `if_pos`.
    fn select_negative(self, if_neg: Self, if_pos: Self) -> Self;
}

pub trait NumSignedVector: SignedVector + NumVector + num_traits::Signed {}

pub trait IntegerVector:
    NumericVector<Element: Denominator>
    + BitshiftVector
    + Div<Self::Divider, Output = Self>
    + Div<Self::BranchfreeDivider, Output = Self>
{
    type Divider: Copy;
    type BranchfreeDivider: Copy;
    type VectorizedDivider: Copy;

    fn wrapping_add(self, other: Self) -> Self;
    fn wrapping_sub(self, other: Self) -> Self;
    fn wrapping_mul(self, other: Self) -> Self;

    fn create_divider(d: Self::Element) -> Self::Divider;
    fn create_branchfree_divider(d: Self::Element) -> Self::BranchfreeDivider;

    fn to_divider(self) -> Self::VectorizedDivider;

    fn rotate_right(self, n: u32) -> Self;
    fn rotate_left(self, n: u32) -> Self;

    fn reverse_bits(self) -> Self;

    fn count_ones(self) -> Self;
    fn count_zeros(self) -> Self;
    fn leading_ones(self) -> Self;
    fn leading_zeros(self) -> Self;
}

pub trait SignedIntegerVector: SignedVector + IntegerVector {
    fn srai<const I: i32>(self) -> Self;
    fn sra(self, count: u32) -> Self;
    fn srav(self, counts: Self::USize) -> Self;
}

pub trait UnsignedIntegerVector: IntegerVector {
    fn is_power_of_two(self) -> Self::Mask;

    fn next_power_of_two_m1(self) -> Self;
    fn ilog2p1(self) -> Self;
    fn parity(self) -> Self;
}

use num_traits::float::FloatCore as CoreFloatTrait;

pub trait NumFloatVector:
    FloatVector
    + NumSignedVector
    + num_traits::FloatConst
    + CoreFloatTrait
    + num_traits::MulAdd<Self, Self, Output = Self>
    + num_traits::MulAddAssign<Self, Self>
{
}

pub trait FloatVector: SignedVector<Element: FloatElement> + FloatConsts + CastVector<Self::ExtendedPrecision> {
    const HALF: Self;
    const NEG_ZERO: Self;
    const INFINITY: Self;
    const NEG_INFINITY: Self;
    const NAN: Self;
    const EPSILON: Self;

    type ExtendedPrecision: FloatVector<Lanes = Self::Lanes> + CastVector<Self>;

    const HAS_TRUE_FMA: bool;

    fn is_infinite(self) -> Self::Mask;
    fn is_finite(self) -> Self::Mask;
    fn is_nan(self) -> Self::Mask;
    fn is_zero_or_subnormal(self) -> Self::Mask;
    fn is_normal(self) -> Self::Mask;
    fn is_subnormal(self) -> Self::Mask;

    fn mul_adde(self, a: Self, b: Self) -> Self;
    fn mul_sube(self, a: Self, b: Self) -> Self;
    fn nmul_adde(self, a: Self, b: Self) -> Self;
    fn nmul_sube(self, a: Self, b: Self) -> Self;

    fn mul_add(self, a: Self, b: Self) -> Self;
    fn mul_sub(self, a: Self, b: Self) -> Self;
    fn nmul_add(self, a: Self, b: Self) -> Self;
    fn nmul_sub(self, a: Self, b: Self) -> Self;

    const HAS_APPROX_RCP: bool;
    const HAS_APPROX_RSQRT: bool;

    fn sqrt(self) -> Self;
    fn rsqrt(self) -> Self;
    fn rcp(self) -> Self;

    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn round(self) -> Self;
    fn trunc(self) -> Self;
    fn fract(self) -> Self;

    fn mul_sign(self, sign: Self) -> Self;
    fn signed_zero(self) -> Self;

    fn next_up(self) -> Self;
    fn next_down(self) -> Self;
}

pub trait FloatVectorWithBits: FloatVector + FullyInteroperable<Self::Signed, Self::Bits> {
    type Signed: SignedIntegerVector<
            Lanes = Self::Lanes,
            Divider = Divider<<Self::Element as FloatElement>::Signed>,
            BranchfreeDivider = BranchfreeDivider<<Self::Element as FloatElement>::Signed>,
            Element = <Self::Element as FloatElement>::Signed,
        > + FullyInteroperable<Self, Self::Bits>;

    type Bits: UnsignedIntegerVector<
            Lanes = Self::Lanes,
            Divider = Divider<<Self::Element as FloatElement>::Bits>,
            BranchfreeDivider = BranchfreeDivider<<Self::Element as FloatElement>::Bits>,
            Element = <Self::Element as FloatElement>::Bits,
        > + FullyInteroperable<Self, Self::Signed>;

    const HAS_NATIVE_LDEXP: bool;
    const HAS_NATIVE_FREXP: bool;

    unsafe fn native_ldexp(self, exp: Self::Signed) -> Self;
    unsafe fn native_frexp(self) -> (Self, Self::Signed);

    fn total_order(self) -> Self::Signed;
}

mod scalar;
mod vector;
