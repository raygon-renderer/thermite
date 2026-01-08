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
    register::{
        BitsRegister, BitshiftRegister, CastMaskRegister, CastRegister, Element, FloatElement, FloatRegister,
        IntegerRegister, Lanes, MaskRegister, NumericRegister, PartialMaskRegister, PartialOrdRegister, Register,
        SignedIntegerRegister, SignedRegister, Storage, SwizzleRegister, UnsignedIntegerRegister,
    },
};

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

        impl<$($gen_param $(: $bound)?),+> $crate::vector::generic::SplatConst<$ty>
        for __GenericSplatValue<$($gen_param),+> {
            const VALUE: $ty = const { $value };
        }

        $crate::vector::generic::GenericVector::splat_const::<
            __GenericSplatValue<$($real_param),+>
        >()
    }};

    // Static type, direct value
    ($ty:ty: $value:expr) => {{
        struct __ConstSplatValue;
        impl $crate::vector::generic::SplatConst<$ty> for __ConstSplatValue {
            const VALUE: $ty = const { $value };
        }
        $crate::vector::generic::GenericVector::splat_const::<__ConstSplatValue>()
    }};

    // Associated const value
    (<$ty:ty $(as $trait:path)?>::$associated:ident) => {{
        struct __ConstSplatValue;
        impl $crate::vector::generic::SplatConst<$ty> for __ConstSplatValue {
            const VALUE: $ty = const { <$ty $(as $trait)?>::$associated };
        }
        $crate::vector::generic::GenericVector::splat_const::<__ConstSplatValue>()
    }};
}

/// Simple associated constant splat trait.
///
/// Used with `GenericVector::splat_const` to splat compile-time constant values into vectors.
///
/// This is effectively a workaround for the lack of `const generics` for generic types.
pub trait SplatConst<E> {
    const VALUE: E;
}

pub trait GenericInteroperable<A, B>:
    MaskedVector<Mask: GenericCastMask<A::Mask> + GenericCastMask<B::Mask>>
    // bits
    + BitsVector<Self>
    + BitsVector<A>
    + BitsVector<B>
    // casts
    + CastVector<Self>
    + CastVector<A>
    + CastVector<B>
where
    A: BitsVector<Self> + CastVector<Self> + MaskedVector<Lanes = Self::Lanes, Mask: GenericCastMask<Self::Mask> + GenericCastMask<B::Mask>>,
    B: BitsVector<Self> + CastVector<Self> + MaskedVector<Lanes = Self::Lanes, Mask: GenericCastMask<Self::Mask> + GenericCastMask<A::Mask>>,
{
}

impl<V, A, B> GenericInteroperable<A, B> for V
where
    V: MaskedVector<Mask: GenericCastMask<A::Mask> + GenericCastMask<B::Mask>>
        // bits
        + BitsVector<V>
        + BitsVector<A>
        + BitsVector<B>
        // casts
        + CastVector<V>
        + CastVector<A>
        + CastVector<B>,
    A: BitsVector<V>
        + CastVector<V>
        + MaskedVector<Lanes = V::Lanes, Mask: GenericCastMask<V::Mask> + GenericCastMask<B::Mask>>,
    B: BitsVector<V>
        + CastVector<V>
        + MaskedVector<Lanes = V::Lanes, Mask: GenericCastMask<V::Mask> + GenericCastMask<A::Mask>>,
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
{
    type Element: Element;
    type Register: Register<Element = Self::Element, Lanes = Self::Lanes>;

    /// Get the underlying register storage.
    fn register(self) -> Storage<Self::Register>;
    /// Create a vector from the underlying register storage.
    fn from_register(storage: Storage<Self::Register>) -> Self;

    const EMPTY: Self;
    const LANES: usize;

    type Lanes: Lanes;

    type USize: UnsignedIntegerVector<
            ISize = Self::ISize,
            USize = Self::USize,
            Lanes = Self::Lanes,
            Element = <Self::Element as Element>::USize,
            Register = <Self::Register as Register>::USize,
        > + CastVector<Self::ISize>
        + BitsVector<Self::ISize>;

    type ISize: SignedIntegerVector<
            ISize = Self::ISize,
            USize = Self::USize,
            Lanes = Self::Lanes,
            Element = <Self::Element as Element>::ISize,
            Register = <Self::Register as Register>::ISize,
        > + CastVector<Self::USize>
        + BitsVector<Self::USize>;

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
}

pub trait BitshiftVector:
    GenericVector<Register: BitshiftRegister>
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

pub trait SwizzleVector: GenericVector<Register: SwizzleRegister> {
    fn swizzle(self, other: Self, indices: GenericArray<u32, Self::Lanes>) -> Self;
    fn permute(self, indices: GenericArray<u32, Self::Lanes>) -> Self;
}

pub trait CastVector<FROM: GenericVector>: GenericVector<Register: CastRegister<FROM::Register>> {
    fn cast_from(from: FROM) -> Self;

    #[inline(always)]
    fn fast_cast_from(from: FROM) -> Self {
        Self::cast_from(from)
    }
}

impl<FROM, INTO> CastVector<Vector<FROM>> for Vector<INTO>
where
    FROM: Register,
    INTO: CastRegister<FROM>,
{
    #[inline(always)]
    fn cast_from(from: Vector<FROM>) -> Self {
        Vector::<INTO>::from(from)
    }

    #[inline(always)]
    fn fast_cast_from(from: Vector<FROM>) -> Self {
        Vector::<INTO>::fast_from(from)
    }
}

pub trait BitsVector<FROM: GenericVector>: GenericVector<Register: BitsRegister<FROM::Register>> {
    fn from_bits(bits: FROM) -> Self;
}

impl<FROM, INTO> BitsVector<Vector<FROM>> for Vector<INTO>
where
    FROM: Register,
    INTO: BitsRegister<FROM>,
{
    #[inline(always)]
    fn from_bits(bits: Vector<FROM>) -> Self {
        Vector::<INTO>::from_bits(bits)
    }
}

pub trait GenericMask<V: GenericVector>:
    GenericSelectable<SelectableMask = Self>
    + GenericCastMask<Self>
    + Sized
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
        S: GenericSelectable<SelectableMask: GenericCastMask<Self>>,
    {
        S::select(self, t, f)
    }

    #[inline(always)]
    fn cast_mask<INTO>(self) -> INTO
    where
        INTO: GenericCastMask<Self>,
    {
        INTO::mask_from(self)
    }

    #[inline(always)]
    fn swap<S>(self, a: &mut S, b: &mut S)
    where
        S: GenericSelectable<SelectableMask: GenericCastMask<Self> + GenericCastMask<S::SelectableMask>>,
    {
        let mask = S::SelectableMask::mask_from(self);

        let a2 = S::select(mask, *a, *b);
        let b2 = S::select(mask, *b, *a);

        *a = a2;
        *b = b2;
    }
}

pub trait GenericCastMask<FROM>: Sized {
    fn mask_from(from: FROM) -> Self;
}

impl<FROM, INTO> GenericCastMask<Mask<FROM>> for Mask<INTO>
where
    FROM: PartialMaskRegister,
    INTO: CastMaskRegister<FROM>,
{
    #[inline(always)]
    fn mask_from(from: Mask<FROM>) -> Self {
        Mask::<INTO>::from_mask(from)
    }
}

impl<R: MaskRegister> GenericMask<Vector<R>> for Mask<R> {
    const FALSY: Self = Mask::<R>::FALSY;
    const TRUTHY: Self = Mask::<R>::TRUTHY;

    #[inline(always)]
    fn from_unchecked(vector: Vector<R>) -> Self {
        Mask::<R>::from_unchecked(vector)
    }

    #[inline(always)]
    fn all(self) -> bool {
        Mask::<R>::all(self)
    }

    #[inline(always)]
    fn any(self) -> bool {
        Mask::<R>::any(self)
    }

    #[inline(always)]
    fn none(self) -> bool {
        Mask::<R>::none(self)
    }

    #[inline(always)]
    fn value(self) -> Vector<R> {
        self.value()
    }

    #[inline(always)]
    fn native_bitmask(&self) -> Option<u64> {
        self.native_bitmask()
    }
}

pub trait GenericSelectable: Copy {
    type SelectableMask: Copy;

    fn select<M>(mask: M, t: Self, f: Self) -> Self
    where
        Self::SelectableMask: GenericCastMask<M>;
}

impl<R> GenericSelectable for Vector<R>
where
    R: MaskRegister,
{
    type SelectableMask = Mask<R>;

    #[inline(always)]
    fn select<M>(mask: M, t: Self, f: Self) -> Self
    where
        Mask<R>: GenericCastMask<M>,
    {
        Mask::mask_from(mask).select(t, f)
    }
}

impl<R> GenericSelectable for Mask<R>
where
    R: MaskRegister,
{
    type SelectableMask = Mask<R>;

    #[inline(always)]
    fn select<M>(mask: M, t: Self, f: Self) -> Self
    where
        Mask<R>: GenericCastMask<M>,
    {
        Mask::mask_from(mask).select(t, f)
    }
}

pub trait MaskedVector: GenericVector<Register: MaskRegister> + GenericSelectable<SelectableMask = Self::Mask> {
    type Mask: GenericMask<Self> + GenericSelectable<SelectableMask = Self::Mask>;
}

pub trait PartialOrdVector: MaskedVector<Register: PartialOrdRegister> + PartialEq {
    fn cmp_lt(self, other: Self) -> Self::Mask;
    fn cmp_le(self, other: Self) -> Self::Mask;
    fn cmp_gt(self, other: Self) -> Self::Mask;
    fn cmp_ge(self, other: Self) -> Self::Mask;
    fn cmp_eq(self, other: Self) -> Self::Mask;
    fn cmp_ne(self, other: Self) -> Self::Mask;
}

pub trait NumericVector:
    PartialOrdVector<Element: num_traits::Num, Register: NumericRegister>
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

pub trait SignedVector:
    NumericVector<Element: num_traits::Signed, Register: SignedRegister> + Neg<Output = Self>
{
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
    NumericVector<Element: Denominator, Register: IntegerRegister>
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

pub trait SignedIntegerVector: SignedVector<Register: SignedIntegerRegister> + IntegerVector {
    fn srai<const I: i32>(self) -> Self;
    fn sra(self, count: u32) -> Self;
    fn srav(self, counts: Self::USize) -> Self;
}

pub trait UnsignedIntegerVector: IntegerVector<Register: UnsignedIntegerRegister> {
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

pub trait FloatVector:
    SignedVector<Element: FloatElement, Register: FloatRegister>
    + FloatConsts
    + GenericInteroperable<Self::Signed, Self::Bits>
    + CastVector<Self::ExtendedPrecision>
{
    const HALF: Self;
    const NEG_ZERO: Self;
    const INFINITY: Self;
    const NEG_INFINITY: Self;
    const NAN: Self;
    const EPSILON: Self;

    type Signed: SignedIntegerVector<
            Lanes = Self::Lanes,
            Divider = Divider<<Self::Element as FloatElement>::Signed>,
            BranchfreeDivider = BranchfreeDivider<<Self::Element as FloatElement>::Signed>,
            VectorizedDivider = VectorDivider<<Self::Register as FloatRegister>::Signed>,
            Element = <Self::Element as FloatElement>::Signed,
            Register = <Self::Register as FloatRegister>::Signed,
        > + GenericInteroperable<Self, Self::Bits>;

    type Bits: UnsignedIntegerVector<
            Lanes = Self::Lanes,
            Divider = Divider<<Self::Element as FloatElement>::Bits>,
            BranchfreeDivider = BranchfreeDivider<<Self::Element as FloatElement>::Bits>,
            VectorizedDivider = VectorDivider<<Self::Register as FloatRegister>::Bits>,
            Element = <Self::Element as FloatElement>::Bits,
            Register = <Self::Register as FloatRegister>::Bits,
        > + GenericInteroperable<Self, Self::Signed>;

    type ExtendedPrecision: FloatVector<Lanes = Self::Lanes> + CastVector<Self>;

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

    fn total_order(self) -> Self::Signed;
}

#[rustfmt::skip]
impl<R: Register> GenericVector for Vector<R> {
    type Element = R::Element;

    const EMPTY: Self = Vector::<R>::EMPTY;
    const LANES: usize = Vector::<R>::LANES;

    type Lanes = R::Lanes;
    type Register = R;

    type USize = Vector<R::USize>;
    type ISize = Vector<R::ISize>;

    #[inline(always)]
    fn register(self) -> Storage<Self::Register> {
        self.0
    }

    #[inline(always)]
    fn from_register(storage: Storage<Self::Register>) -> Self {
        Vector(storage)
    }

    #[inline(always)]
    fn splat_const<C>() -> Self where C: SplatConst<Self::Element> {
        const { Self::splat_const(C::VALUE) }
    }

    #[inline(always)] fn splat(value: Self::Element) -> Self { Vector::<R>::splat(value) }
    #[inline(always)] fn broadcast<const I: usize>(self) -> Self { Vector::<R>::broadcast::<I>(self) }
    #[inline(always)] fn broadcastv(self, idx: usize) -> Self { Vector::<R>::broadcastv(self, idx) }
    #[inline(always)] fn as_slice(&self) -> &[Self::Element] { Vector::<R>::as_slice(self) }
    #[inline(always)] fn as_mut_slice(&mut self) -> &mut [Self::Element] { Vector::<R>::as_mut_slice(self) }
    #[inline(always)] fn extract<const I: usize>(self) -> Self::Element { Vector::<R>::extract::<I>(self) }
    #[inline(always)] fn insert<const I: usize>(self, value: Self::Element) -> Self { Vector::<R>::insert::<I>(self, value) }
    #[inline(always)] fn reverse(self) -> Self { Vector::<R>::reverse(self) }
    #[inline(always)] fn swap_bytes(self) -> Self { Vector::<R>::swap_bytes(self) }
    #[inline(always)] fn bitandnot(self, other: Self) -> Self { Vector::<R>::bitandnot(self, other) }

    const HAS_SIMPLE_UNPACK: bool = R::HAS_SIMPLE_UNPACK;

    #[inline(always)] fn unpack(self, other: Self) -> (Self, Self) { Vector::<R>::unpack(self, other) }

    #[inline(always)] fn map<F>(self, f: F) -> Self
    where F: Fn(Self::Element) -> Self::Element,
    { Vector::<R>::map(self, f) }

    #[inline(always)] fn fold<F>(self, init: Self::Element, f: F) -> Self::Element
    where F: Fn(Self::Element, Self::Element) -> Self::Element,
    { Vector::<R>::fold(self, init, f) }

    #[inline(always)] fn reduce<F>(self, f: F) -> Self::Element
    where F: Fn(Self::Element, Self::Element) -> Self::Element,
    { Vector::<R>::reduce(self, f) }

    #[inline(always)] fn single(value: Self::Element) -> Self { Vector::<R>::single(value) }

    #[inline(always)] unsafe fn load(ptr: *const Self::Element) -> Self { unsafe { Vector::<R>::load(ptr) } }
    #[inline(always)] unsafe fn load_unaligned(ptr: *const Self::Element) -> Self { unsafe { Vector::<R>::load_unaligned(ptr) } }
    #[inline(always)] unsafe fn load_streaming(ptr: *const Self::Element) -> Self { unsafe { Vector::<R>::load_streaming(ptr) } }
    #[inline(always)] unsafe fn store(self, ptr: *mut Self::Element) { unsafe { Vector::<R>::store(self, ptr) } }
    #[inline(always)] unsafe fn store_unaligned(self, ptr: *mut Self::Element) { unsafe { Vector::<R>::store_unaligned(self, ptr) } }
    #[inline(always)] unsafe fn store_streaming(self, ptr: *mut Self::Element) { unsafe { Vector::<R>::store_streaming(self, ptr) } }
}

#[rustfmt::skip]
impl<R: BitshiftRegister> BitshiftVector for Vector<R> {
    const HAS_TRUE_SHIFTV: bool = R::HAS_TRUE_SHIFTV;
    const HAS_WIDE_BYTE_SHIFTS: bool = R::HAS_WIDE_BYTE_SHIFTS;

    #[inline(always)] fn bshli<const I: i32>(self) -> Self { Vector::<R>::bshli::<I>(self) }
    #[inline(always)] fn bshri<const I: i32>(self) -> Self { Vector::<R>::bshri::<I>(self) }
    #[inline(always)] fn shli<const I: i32>(self) -> Self { Vector::<R>::shli::<I>(self) }
    #[inline(always)] fn shri<const I: i32>(self) -> Self { Vector::<R>::shri::<I>(self) }
    #[inline(always)] fn shlv(self, shifts: Self::USize) -> Self { Vector::<R>::shlv(self, shifts) }
    #[inline(always)] fn shrv(self, shifts: Self::USize) -> Self { Vector::<R>::shrv(self, shifts) }
}

#[rustfmt::skip]
impl<R: MaskRegister> MaskedVector for Vector<R> {
    type Mask = Mask<R>;
}

#[rustfmt::skip]
impl<R: PartialOrdRegister> PartialOrdVector for Vector<R> {
    #[inline(always)] fn cmp_lt(self, other: Self) -> Self::Mask { Vector::<R>::cmp_lt(self, other) }
    #[inline(always)] fn cmp_le(self, other: Self) -> Self::Mask { Vector::<R>::cmp_le(self, other) }
    #[inline(always)] fn cmp_gt(self, other: Self) -> Self::Mask { Vector::<R>::cmp_gt(self, other) }
    #[inline(always)] fn cmp_ge(self, other: Self) -> Self::Mask { Vector::<R>::cmp_ge(self, other) }
    #[inline(always)] fn cmp_eq(self, other: Self) -> Self::Mask { Vector::<R>::cmp_eq(self, other) }
    #[inline(always)] fn cmp_ne(self, other: Self) -> Self::Mask { Vector::<R>::cmp_ne(self, other) }
}

#[rustfmt::skip]
impl<R: NumericRegister> NumericVector for Vector<R>
where
    R::Element: num_traits::Num,
{
    const ZERO: Self = Vector::<R>::ZERO;
    const ONE: Self = Vector::<R>::ONE;
    const TWO: Self = Vector::<R>::TWO;
    const MIN: Self = Vector::<R>::MIN;
    const MAX: Self = Vector::<R>::MAX;

    #[inline(always)] fn is_zero(self) -> Self::Mask { Vector::<R>::is_zero(self) }

    #[inline(always)] fn min(self, other: Self) -> Self { Vector::<R>::min(self, other) }
    #[inline(always)] fn max(self, other: Self) -> Self { Vector::<R>::max(self, other) }
    #[inline(always)] fn clamp(self, min: Self, max: Self) -> Self { Vector::<R>::clamp(self, min, max) }

    #[inline(always)] fn min_element(self) -> Self::Element { Vector::<R>::min_element(self) }
    #[inline(always)] fn max_element(self) -> Self::Element { Vector::<R>::max_element(self) }

    #[inline(always)] fn sum_elements(self) -> Self::Element { Vector::<R>::sum_elements(self) }
    #[inline(always)] fn prod_elements(self) -> Self::Element { Vector::<R>::prod_elements(self) }

    #[inline(always)] fn offset() -> Self { Vector::<R>::offset() }
    #[inline(always)] fn indexed() -> Self { Vector::<R>::indexed() }
}

impl<R: NumericRegister> NumVector for Vector<R> where R::Element: num_traits::Num + num_traits::NumCast {}

#[rustfmt::skip]
impl<R: SignedRegister> SignedVector for Vector<R>
where
    R::Element: num_traits::Signed,
{
    const NEG_ONE: Self = Vector::<R>::NEG_ONE;
    const MIN_POSITIVE: Self = Vector::<R>::MIN_POSITIVE;

    #[inline(always)] fn abs(self) -> Self { Vector::<R>::abs(self) }

    #[inline(always)] fn signum(self) -> Self { Vector::<R>::signum(self) }
    #[inline(always)] fn copysign(self, sign: Self) -> Self { Vector::<R>::copysign(self, sign) }

    #[inline(always)] fn is_positive(self) -> Self::Mask { Vector::<R>::is_positive(self) }
    #[inline(always)] fn is_negative(self) -> Self::Mask { Vector::<R>::is_negative(self) }

    #[inline(always)] fn select_negative(self, if_neg: Self, if_pos: Self) -> Self {
        Vector::<R>::select_negative(self, if_neg, if_pos)
    }
}

impl<R: SignedRegister> NumSignedVector for Vector<R> where R::Element: num_traits::Signed + num_traits::NumCast {}

#[rustfmt::skip]
impl<R: IntegerRegister> IntegerVector for Vector<R>
where
    R::Element: Denominator,
{
    type Divider = crate::divider::Divider<R::Element>;
    type BranchfreeDivider = crate::divider::BranchfreeDivider<R::Element>;
    type VectorizedDivider = crate::divider::vector::VectorDivider<R>;

    #[inline(always)] fn wrapping_add(self, other: Self) -> Self { <Vector<R>>::wrapping_add(self, other) }
    #[inline(always)] fn wrapping_sub(self, other: Self) -> Self { <Vector<R>>::wrapping_sub(self, other) }
    #[inline(always)] fn wrapping_mul(self, other: Self) -> Self { <Vector<R>>::wrapping_mul(self, other) }

    #[inline(always)] fn create_divider(d: Self::Element) -> Self::Divider { Denominator::to_divider(d) }
    #[inline(always)] fn create_branchfree_divider(d: Self::Element) -> Self::BranchfreeDivider { Denominator::to_branchfree_divider(d) }

    #[inline(always)] fn to_divider(self) -> Self::VectorizedDivider { Vector::<R>::to_divider(self) }

    #[inline(always)] fn rotate_right(self, n: u32) -> Self { Vector::<R>::rotate_right(self, n) }
    #[inline(always)] fn rotate_left(self, n: u32) -> Self { Vector::<R>::rotate_left(self, n) }

    #[inline(always)] fn reverse_bits(self) -> Self { Vector::<R>::reverse_bits(self) }

    #[inline(always)] fn count_ones(self) -> Self { Vector::<R>::count_ones(self) }
    #[inline(always)] fn count_zeros(self) -> Self { Vector::<R>::count_zeros(self) }
    #[inline(always)] fn leading_ones(self) -> Self { Vector::<R>::leading_ones(self) }
    #[inline(always)] fn leading_zeros(self) -> Self { Vector::<R>::leading_zeros(self) }
}

#[rustfmt::skip]
impl<R: SignedIntegerRegister> SignedIntegerVector for Vector<R>
where
    R::Element: Denominator + num_traits::Signed,
{
    #[inline(always)] fn srai<const I: i32>(self) -> Self { Vector::<R>::srai::<I>(self) }
    #[inline(always)] fn sra(self, count: u32) -> Self { Vector::<R>::sra(self, count) }
    #[inline(always)] fn srav(self, counts: Self::USize) -> Self { Vector::<R>::srav(self, counts) }
}

#[rustfmt::skip]
impl<R: UnsignedIntegerRegister> UnsignedIntegerVector for Vector<R>
where
    R::Element: Denominator,
{
    #[inline(always)] fn is_power_of_two(self) -> Self::Mask { Vector::<R>::is_power_of_two(self) }
    #[inline(always)] fn next_power_of_two_m1(self) -> Self { Vector::<R>::next_power_of_two_m1(self) }
    #[inline(always)] fn ilog2p1(self) -> Self { Vector::<R>::ilog2p1(self) }
    #[inline(always)] fn parity(self) -> Self { Vector::<R>::parity(self) }
}

#[rustfmt::skip]
impl<R: FloatRegister> FloatVector for Vector<R> {
    const HALF: Self = Vector::<R>::HALF;
    const NEG_ZERO: Self = Vector::<R>::NEG_ZERO;
    const INFINITY: Self = Vector::<R>::INFINITY;
    const NEG_INFINITY: Self = Vector::<R>::NEG_INFINITY;
    const NAN: Self = Vector::<R>::NAN;
    const EPSILON: Self = Vector::<R>::EPSILON;

    type Signed = Vector<R::Signed>;
    type Bits = Vector<R::Bits>;
    type ExtendedPrecision = Vector<R::ExtendedPrecision>;

    #[inline(always)] fn is_infinite(self) -> Self::Mask { Vector::<R>::is_infinite(self) }
    #[inline(always)] fn is_finite(self) -> Self::Mask { Vector::<R>::is_finite(self) }
    #[inline(always)] fn is_nan(self) -> Self::Mask { Vector::<R>::is_nan(self) }
    #[inline(always)] fn is_zero_or_subnormal(self) -> Self::Mask { Vector::<R>::is_zero_or_subnormal(self) }
    #[inline(always)] fn is_normal(self) -> Self::Mask { Vector::<R>::is_normal(self) }
    #[inline(always)] fn is_subnormal(self) -> Self::Mask { Vector::<R>::is_subnormal(self) }
    #[inline(always)] fn mul_adde(self, a: Self, b: Self) -> Self { Vector::<R>::mul_adde(self, a, b) }
    #[inline(always)] fn mul_sube(self, a: Self, b: Self) -> Self { Vector::<R>::mul_sube(self, a, b) }
    #[inline(always)] fn nmul_adde(self, a: Self, b: Self) -> Self { Vector::<R>::nmul_adde(self, a, b) }
    #[inline(always)] fn nmul_sube(self, a: Self, b: Self) -> Self { Vector::<R>::nmul_sube(self, a, b) }
    #[inline(always)] fn mul_add(self, a: Self, b: Self) -> Self { Vector::<R>::mul_add(self, a, b) }
    #[inline(always)] fn mul_sub(self, a: Self, b: Self) -> Self { Vector::<R>::mul_sub(self, a, b) }
    #[inline(always)] fn nmul_add(self, a: Self, b: Self) -> Self { Vector::<R>::nmul_add(self, a, b) }
    #[inline(always)] fn nmul_sub(self, a: Self, b: Self) -> Self { Vector::<R>::nmul_sub(self, a, b) }
    #[inline(always)] fn sqrt(self) -> Self { Vector::<R>::sqrt(self) }
    #[inline(always)] fn rsqrt(self) -> Self { Vector::<R>::rsqrt(self) }
    #[inline(always)] fn rcp(self) -> Self { Vector::<R>::rcp(self) }
    #[inline(always)] fn floor(self) -> Self { Vector::<R>::floor(self) }
    #[inline(always)] fn ceil(self) -> Self { Vector::<R>::ceil(self) }
    #[inline(always)] fn round(self) -> Self { Vector::<R>::round(self) }
    #[inline(always)] fn trunc(self) -> Self { Vector::<R>::trunc(self) }
    #[inline(always)] fn fract(self) -> Self { Vector::<R>::fract(self) }
    #[inline(always)] fn mul_sign(self, sign: Self) -> Self { Vector::<R>::mul_sign(self, sign) }
    #[inline(always)] fn signed_zero(self) -> Self { Vector::<R>::signed_zero(self) }
    #[inline(always)] fn next_up(self) -> Self { Vector::<R>::next_up(self) }
    #[inline(always)] fn next_down(self) -> Self { Vector::<R>::next_down(self) }
    #[inline(always)] fn total_order(self) -> Self::Signed { Vector::<R>::total_order(self) }
}
