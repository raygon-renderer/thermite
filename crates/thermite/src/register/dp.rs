//! Double-pumped registers for wider SIMD operations.

#![warn(missing_docs, clippy::missing_safety_doc)]

use crate::{
    divider::vector::VectorDivider,
    isa::InstructionSet,
    register::{Element, LinAlg4Register, SignedIntegerRegister, ValidLinAlg3Length},
};

use super::{
    BitsRegister, BitshiftRegister, CastMaskRegister, CastRegister, FloatRegister, IntegerRegister, Lanes,
    LinAlg3Register, MaskRegister, NumericRegister, PartialOrdRegister, Register, SignedRegister, Storage,
    SwizzleRegister, UnsignedIntegerRegister,
};

use generic_array::{
    ArrayLength, GenericArray,
    typenum::{self, Unsigned},
};

/// Combines two registers of the same type into a single register with double the lanes.
///
/// This is used to create wider registers for SIMD operations when the underlying
/// architectures are not wide enough.
///
/// Don't use this trait directly, use the [`DoublePump`](super::DoublePump) alias instead:
///
/// ```ignore
/// type f32x32 = DoublePump<f32x16>;
/// ```
#[repr(C)]
pub struct DoublePumpRegister<R: Register>(pub(crate) R::Storage, pub(crate) R::Storage);

const _: () = {
    use core::fmt;

    impl<R: Register> fmt::Debug for DoublePumpRegister<R>
    where
        R::Storage: fmt::Debug,
    {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_tuple("DoublePumpRegister")
                .field(&self.0)
                .field(&self.1)
                .finish()
        }
    }
};

#[doc(hidden)]
pub trait DoublePumpVector {
    type DoublePumped;
}

impl<R: Register> DoublePumpVector for crate::vector::Vector<R>
where
    R::DoubleRegister: Register,
{
    type DoublePumped = crate::vector::Vector<R::DoubleRegister>;
}

impl<R: Register> Clone for DoublePumpRegister<R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: Register> Copy for DoublePumpRegister<R> {}

impl<R: Register> DoublePumpRegister<R>
where
    typenum::Double<R::Lanes>: Lanes,
{
    #[inline(always)]
    fn split_array<T>(input: GenericArray<T, <Self as Register>::Lanes>) -> [GenericArray<T, R::Lanes>; 2] {
        unsafe { generic_array::const_transmute(input) }
    }
}

impl<R: Register> Register for DoublePumpRegister<R>
where
    typenum::Double<R::Lanes>: Lanes,
{
    type Lanes = typenum::Double<R::Lanes>;
    type Element = R::Element;
    type Storage = DoublePumpRegister<R>;

    const IS_EMULATED: bool = true; // sad, but true.

    const ISA: InstructionSet = R::ISA;

    type HalfRegister = R;
    type DoubleRegister = DoublePumpRegister<Self>;

    type ISize = DoublePumpRegister<R::ISize>;
    type USize = DoublePumpRegister<R::USize>;

    const EMPTY: Storage<Self> = Self(R::EMPTY, R::EMPTY);

    #[inline(always)]
    fn split(
        value: Storage<Self>,
    ) -> (
        <Self::HalfRegister as Register>::Storage,
        <Self::HalfRegister as Register>::Storage,
    )
    where
        Self::HalfRegister: Register,
    {
        (value.0, value.1)
    }

    #[inline(always)]
    fn join(
        lo: <Self::HalfRegister as Register>::Storage,
        hi: <Self::HalfRegister as Register>::Storage,
    ) -> Storage<Self>
    where
        Self::HalfRegister: Register,
    {
        Self(lo, hi)
    }

    #[inline(always)]
    fn new(value: generic_array::GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        // SAFETY: With arrays always being repr(C), we can safely transmute
        // the double-length array into the two half arrays for each register.
        let [lhs, rhs] = Self::split_array(value);

        Self(R::new(lhs), R::new(rhs))
    }

    #[inline(always)]
    fn single(value: Self::Element) -> Storage<Self> {
        Self(R::single(value), R::EMPTY)
    }

    #[inline(always)]
    fn splat(value: Self::Element) -> Storage<Self> {
        Self(R::splat(value), R::splat(value))
    }

    #[inline(always)]
    fn broadcast<const I: usize>(value: Storage<Self>) -> Storage<Self> {
        // NOTE: using `broadcast::<I>(value)` doesn't work
        // because the const index is propagated before the conditional
        // check, leading to out-of-bounds errors.
        Self::broadcastv(value, I)
    }

    #[inline(always)]
    fn broadcastv(value: Storage<Self>, idx: usize) -> Storage<Self> {
        let r = if idx < R::Lanes::USIZE {
            R::broadcastv(value.0, idx)
        } else {
            R::broadcastv(value.1, idx - R::Lanes::USIZE)
        };

        Self(r, r)
    }

    #[inline(always)]
    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { Self(R::load(ptr), R::load(ptr.add(R::Lanes::USIZE))) }
    }

    #[inline(always)]
    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { Self(R::load_unaligned(ptr), R::load_unaligned(ptr.add(R::Lanes::USIZE))) }
    }

    #[inline(always)]
    unsafe fn load_stream(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { Self(R::load_stream(ptr), R::load_stream(ptr.add(R::Lanes::USIZE))) }
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe {
            R::store(ptr, value.0);
            R::store(ptr.add(R::Lanes::USIZE), value.1);
        }
    }

    #[inline(always)]
    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe {
            R::store_unaligned(ptr, value.0);
            R::store_unaligned(ptr.add(R::Lanes::USIZE), value.1);
        }
    }

    #[inline(always)]
    unsafe fn store_stream(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe {
            R::store_stream(ptr, value.0);
            R::store_stream(ptr.add(R::Lanes::USIZE), value.1);
        }
    }

    #[inline(always)]
    fn fold<F>(first: Self::Element, value: Storage<Self>, f: F) -> Self::Element
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element,
    {
        R::fold(R::fold(first, value.0, &f), value.1, &f)
    }

    #[inline(always)]
    fn reduce<F>(value: Storage<Self>, f: F) -> Self::Element
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element,
    {
        let lo = R::reduce(value.0, &f);
        let hi = R::reduce(value.1, &f);

        f(lo, hi)
    }

    #[inline(always)]
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self(R::bitxor(lhs.0, rhs.0), R::bitxor(lhs.1, rhs.1))
    }

    #[inline(always)]
    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self(R::bitand(lhs.0, rhs.0), R::bitand(lhs.1, rhs.1))
    }

    #[inline(always)]
    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self(R::bitandnot(lhs.0, rhs.0), R::bitandnot(lhs.1, rhs.1))
    }

    #[inline(always)]
    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self(R::bitor(lhs.0, rhs.0), R::bitor(lhs.1, rhs.1))
    }

    #[inline(always)]
    fn not(value: Storage<Self>) -> Storage<Self> {
        Self(R::not(value.0), R::not(value.1))
    }

    #[inline(always)]
    fn blendv(mask: Storage<Self>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self(R::blendv(mask.0, lhs.0, rhs.0), R::blendv(mask.1, lhs.1, rhs.1))
    }

    const HAS_MSB_BLENDV: bool = R::HAS_MSB_BLENDV;

    #[inline(always)]
    fn reverse(mut value: Storage<Self>) -> Storage<Self> {
        Self(R::reverse(value.1), R::reverse(value.0))
    }

    #[inline(always)]
    fn unpack(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        let (r1_lo, r1_hi) = R::unpack(a.0, b.0);
        let (r2_lo, r2_hi) = R::unpack(a.1, b.1);

        (DoublePumpRegister(r1_lo, r1_hi), DoublePumpRegister(r2_lo, r2_hi))
    }

    #[inline(always)]
    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        Self(R::swap_bytes(value.0), R::swap_bytes(value.1))
    }
}

#[rustfmt::skip]
impl<R: BitshiftRegister> BitshiftRegister for DoublePumpRegister<R>
where
    typenum::Double<R::Lanes>: Lanes,
{
    const HAS_TRUE_SHIFTV: bool = R::HAS_TRUE_SHIFTV;

    #[inline(always)] fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> { Self(R::shli::<IMM8>(value.0), R::shli::<IMM8>(value.1)) }
    #[inline(always)] fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> { Self(R::shri::<IMM8>(value.0), R::shri::<IMM8>(value.1)) }
    #[inline(always)] fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> { Self(R::shl(value.0, shift), R::shl(value.1, shift)) }
    #[inline(always)] fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> { Self(R::shr(value.0, shift), R::shr(value.1, shift)) }
    #[inline(always)] fn shlv(value: Storage<Self>, shifts: Storage<Self::USize>) -> Storage<Self> { Self(R::shlv(value.0, shifts.0), R::shlv(value.1, shifts.1)) }
    #[inline(always)] fn shrv(value: Storage<Self>, shifts: Storage<Self::USize>) -> Storage<Self> { Self(R::shrv(value.0, shifts.0), R::shrv(value.1, shifts.1)) }
    #[inline(always)] fn rol(value: Storage<Self>, shift: u32) -> Storage<Self> { Self(R::rol(value.0, shift), R::rol(value.1, shift)) }
    #[inline(always)] fn ror(value: Storage<Self>, shift: u32) -> Storage<Self> { Self(R::ror(value.0, shift), R::ror(value.1, shift)) }
    #[inline(always)] fn roli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> { Self(R::roli::<IMM8>(value.0), R::roli::<IMM8>(value.1)) }
    #[inline(always)] fn rori<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> { Self(R::rori::<IMM8>(value.0), R::rori::<IMM8>(value.1)) }
    #[inline(always)] fn rolv(value: Storage<Self>, shifts: Storage<Self::USize>) -> Storage<Self> { Self(R::rolv(value.0, shifts.0), R::rolv(value.1, shifts.1)) }
    #[inline(always)] fn rorv(value: Storage<Self>, shifts: Storage<Self::USize>) -> Storage<Self> { Self(R::rorv(value.0, shifts.0), R::rorv(value.1, shifts.1)) }
    #[inline(always)] fn reverse_bits(value: Storage<Self>) -> Storage<Self> { Self(R::reverse_bits(value.0), R::reverse_bits(value.1)) }
}

impl<R: MaskRegister> MaskRegister for DoublePumpRegister<R>
where
    typenum::Double<R::Lanes>: Lanes,
{
    const TRUTHY: Storage<Self> = Self(R::TRUTHY, R::TRUTHY);
    const FALSY: Storage<Self> = Self(R::FALSY, R::FALSY);

    #[inline(always)]
    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        let [lhs, rhs] = Self::split_array(value);
        Self(R::new_mask(lhs), R::new_mask(rhs))
    }

    #[inline(always)]
    fn debug_iter_bool(value: &Storage<Self>) -> impl Iterator<Item = bool> {
        let iter_lo = R::debug_iter_bool(&value.0);
        let iter_hi = R::debug_iter_bool(&value.1);
        iter_lo.chain(iter_hi)
    }

    #[inline(always)]
    fn all(value: Storage<Self>) -> bool {
        R::all(value.0) && R::all(value.1)
    }

    #[inline(always)]
    fn any(value: Storage<Self>) -> bool {
        R::any(value.0) || R::any(value.1)
    }

    #[inline(always)]
    fn none(value: Storage<Self>) -> bool {
        R::none(value.0) && R::none(value.1)
    }

    #[inline(always)]
    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        if Self::Lanes::USIZE <= 64 {
            let lo = R::native_bitmask(value.0)?;
            let hi = R::native_bitmask(value.1)?;

            Some(lo | (hi << R::Lanes::U64))
        } else {
            None
        }
    }

    #[inline(always)]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        // try to use native bitmask if available
        if let Some(native) = Self::native_bitmask(value) {
            use bitvec::slice::BitSlice;

            let bits = unsafe { core::mem::transmute::<u64, [u32; 2]>(native) };
            let bits = BitSlice::<u32>::from_slice(&bits);

            view[..Self::Lanes::USIZE].copy_from_bitslice(&bits[..Self::Lanes::USIZE]);
        } else {
            // otherwise divide and conquer
            let lane_count = R::Lanes::USIZE;
            R::fill_bitmask(value.0, &mut view[..lane_count]);
            R::fill_bitmask(value.1, &mut view[lane_count..]);
        }
    }
}

impl<FROM: MaskRegister, INTO: CastMaskRegister<FROM>> CastMaskRegister<DoublePumpRegister<FROM>>
    for DoublePumpRegister<INTO>
where
    typenum::Double<INTO::Lanes>: Lanes,
    typenum::Double<FROM::Lanes>: Lanes,
{
    #[inline(always)]
    fn mask_from(value: <DoublePumpRegister<FROM> as Register>::Storage) -> Storage<Self> {
        Self(INTO::mask_from(value.0), INTO::mask_from(value.1))
    }
}

#[rustfmt::skip]
impl<R: PartialOrdRegister> PartialOrdRegister for DoublePumpRegister<R>
where
    typenum::Double<R::Lanes>: Lanes,
{
    #[inline(always)] fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Self(R::lt(lhs.0, rhs.0), R::lt(lhs.1, rhs.1)) }
    #[inline(always)] fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Self(R::le(lhs.0, rhs.0), R::le(lhs.1, rhs.1)) }
    #[inline(always)] fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Self(R::gt(lhs.0, rhs.0), R::gt(lhs.1, rhs.1)) }
    #[inline(always)] fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Self(R::ge(lhs.0, rhs.0), R::ge(lhs.1, rhs.1)) }
    #[inline(always)] fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Self(R::eq(lhs.0, rhs.0), R::eq(lhs.1, rhs.1)) }
    #[inline(always)] fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Self(R::ne(lhs.0, rhs.0), R::ne(lhs.1, rhs.1)) }
}

#[rustfmt::skip]
impl<R: NumericRegister> NumericRegister for DoublePumpRegister<R>
where
    typenum::Double<R::Lanes>: Lanes,
{
    const ZERO: Storage<Self> = Self(R::ZERO, R::ZERO);
    const ONE: Storage<Self> = Self(R::ONE, R::ONE);
    const TWO: Storage<Self> = Self(R::TWO, R::TWO);

    const MIN: Storage<Self> = Self(R::MIN, R::MIN);
    const MAX: Storage<Self> = Self(R::MAX, R::MAX);

    #[inline(always)] fn max_element(value: Storage<Self>) -> Self::Element { R::max_element(R::max(value.0, value.1)) }
    #[inline(always)] fn min_element(value: Storage<Self>) -> Self::Element { R::min_element(R::min(value.0, value.1)) }
    #[inline(always)] fn sum_elements(value: Storage<Self>) -> Self::Element { R::sum_elements(R::add(value.0, value.1)) }
    #[inline(always)] fn prod_elements(value: Storage<Self>) -> Self::Element { R::prod_elements(R::mul(value.0, value.1)) }
    #[inline(always)] fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Self(R::add(lhs.0, rhs.0), R::add(lhs.1, rhs.1)) }
    #[inline(always)] fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Self(R::sub(lhs.0, rhs.0), R::sub(lhs.1, rhs.1)) }
    #[inline(always)] fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Self(R::mul(lhs.0, rhs.0), R::mul(lhs.1, rhs.1)) }
    #[inline(always)] fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Self(R::div(lhs.0, rhs.0), R::div(lhs.1, rhs.1)) }
    #[inline(always)] fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Self(R::rem(lhs.0, rhs.0), R::rem(lhs.1, rhs.1)) }
    #[inline(always)] fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Self(R::min(lhs.0, rhs.0), R::min(lhs.1, rhs.1)) }
    #[inline(always)] fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Self(R::max(lhs.0, rhs.0), R::max(lhs.1, rhs.1)) }

    #[inline(always)]
    fn offset() -> Storage<Self> {
        // Because we're doubling up each time, we can just x2 the offset
        let mut offset = R::offset();
        offset = R::add(offset, offset); // via addition because it's faster
        Self(offset, offset)
    }

    #[inline(always)]
    fn indexed() -> Storage<Self> {
        // offset the second index to remain consistent
        let indexed = R::indexed();
        Self(indexed, R::add(indexed, R::offset()))
    }
}

#[rustfmt::skip]
impl<R: SignedRegister> SignedRegister for DoublePumpRegister<R>
where
    typenum::Double<R::Lanes>: Lanes,
{
    const NEG_ONE: Storage<Self> = Self(R::NEG_ONE, R::NEG_ONE);
    const MIN_POSITIVE: Storage<Self> = Self(R::MIN_POSITIVE, R::MIN_POSITIVE);

    #[inline(always)] fn neg(value: Storage<Self>) -> Storage<Self> { Self(R::neg(value.0), R::neg(value.1)) }
    #[inline(always)] fn abs(value: Storage<Self>) -> Storage<Self> { Self(R::abs(value.0), R::abs(value.1)) }
    #[inline(always)] fn copysign(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Self(R::copysign(lhs.0, rhs.0), R::copysign(lhs.1, rhs.1)) }
    #[inline(always)] fn signum(value: Storage<Self>) -> Storage<Self> { Self(R::signum(value.0), R::signum(value.1)) }
    #[inline(always)] fn is_negative(value: Storage<Self>) -> Storage<Self> { Self(R::is_negative(value.0), R::is_negative(value.1)) }
    #[inline(always)] fn is_positive(value: Storage<Self>) -> Storage<Self> { Self(R::is_positive(value.0), R::is_positive(value.1)) }

    #[inline(always)]
    fn conditional_negate(value: Storage<Self>, mask: Storage<Self>) -> Storage<Self> {
        Self(
            R::conditional_negate(value.0, mask.0),
            R::conditional_negate(value.1, mask.1),
        )
    }
}

#[rustfmt::skip]
impl<R: FloatRegister> FloatRegister for DoublePumpRegister<R>
where
    typenum::Double<R::Lanes>: Lanes,
{
    const HAS_TRUE_FMA: bool = R::HAS_TRUE_FMA;

    type Bits = DoublePumpRegister<R::Bits>;
    type Signed = DoublePumpRegister<R::Signed>;
    type ExtendedPrecision = DoublePumpRegister<R::ExtendedPrecision>;

    const HALF: Storage<Self> = Self(R::HALF, R::HALF);
    const NEG_ZERO: Storage<Self> = Self(R::NEG_ZERO, R::NEG_ZERO);
    const INFINITY: Storage<Self> = Self(R::INFINITY, R::INFINITY);
    const NEG_INFINITY: Storage<Self> = Self(R::NEG_INFINITY, R::NEG_INFINITY);
    const NAN: Storage<Self> = Self(R::NAN, R::NAN);
    const EPSILON: Storage<Self> = Self(R::EPSILON, R::EPSILON);

    const EXP_MASK: Storage<Self::Bits> = DoublePumpRegister(R::EXP_MASK, R::EXP_MASK);

    #[inline(always)] fn is_nan(value: Storage<Self>) -> Storage<Self> { Self(R::is_nan(value.0), R::is_nan(value.1)) }
    #[inline(always)] fn is_infinite(value: Storage<Self>) -> Storage<Self> { Self(R::is_infinite(value.0), R::is_infinite(value.1)) }
    #[inline(always)] fn is_finite(value: Storage<Self>) -> Storage<Self> { Self(R::is_finite(value.0), R::is_finite(value.1)) }
    #[inline(always)] fn is_subnormal(value: Storage<Self>) -> Storage<Self> { Self(R::is_subnormal(value.0), R::is_subnormal(value.1)) }
    #[inline(always)] fn is_zero_or_subnormal(value: Storage<Self>) -> Storage<Self> { Self(R::is_zero_or_subnormal(value.0), R::is_zero_or_subnormal(value.1)) }
    #[inline(always)] fn is_normal(value: Storage<Self>) -> Storage<Self> { Self(R::is_normal(value.0), R::is_normal(value.1)) }
    #[inline(always)] fn mul_adde(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> { Self(R::mul_adde(lhs.0, rhs.0, acc.0), R::mul_adde(lhs.1, rhs.1, acc.1)) }
    #[inline(always)] fn mul_sube(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> { Self(R::mul_sube(lhs.0, rhs.0, acc.0), R::mul_sube(lhs.1, rhs.1, acc.1)) }
    #[inline(always)] fn nmul_adde(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> { Self(R::nmul_adde(lhs.0, rhs.0, acc.0), R::nmul_adde(lhs.1, rhs.1, acc.1)) }
    #[inline(always)] fn nmul_sube(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> { Self(R::nmul_sube(lhs.0, rhs.0, acc.0), R::nmul_sube(lhs.1, rhs.1, acc.1)) }
    #[inline(always)] fn mul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> { Self(R::mul_add(lhs.0, rhs.0, acc.0), R::mul_add(lhs.1, rhs.1, acc.1)) }
    #[inline(always)] fn mul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> { Self(R::mul_sub(lhs.0, rhs.0, acc.0), R::mul_sub(lhs.1, rhs.1, acc.1)) }
    #[inline(always)] fn nmul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> { Self(R::nmul_add(lhs.0, rhs.0, acc.0), R::nmul_add(lhs.1, rhs.1, acc.1)) }
    #[inline(always)] fn nmul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> { Self(R::nmul_sub(lhs.0, rhs.0, acc.0), R::nmul_sub(lhs.1, rhs.1, acc.1)) }
    #[inline(always)] fn sqrt(value: Storage<Self>) -> Storage<Self> { Self(R::sqrt(value.0), R::sqrt(value.1)) }
    #[inline(always)] fn rsqrt(value: Storage<Self>) -> Storage<Self> { Self(R::rsqrt(value.0), R::rsqrt(value.1)) }

    const HAS_APPROX_RSQRT: bool = R::HAS_APPROX_RSQRT;
    const HAS_APPROX_RCP: bool = R::HAS_APPROX_RCP;

    #[inline(always)] fn rcp(value: Storage<Self>) -> Storage<Self> { Self(R::rcp(value.0), R::rcp(value.1)) }
    #[inline(always)] fn floor(value: Storage<Self>) -> Storage<Self> { Self(R::floor(value.0), R::floor(value.1)) }
    #[inline(always)] fn ceil(value: Storage<Self>) -> Storage<Self> { Self(R::ceil(value.0), R::ceil(value.1)) }
    #[inline(always)] fn round(value: Storage<Self>) -> Storage<Self> { Self(R::round(value.0), R::round(value.1)) }
    #[inline(always)] fn trunc(value: Storage<Self>) -> Storage<Self> { Self(R::trunc(value.0), R::trunc(value.1)) }
    #[inline(always)] fn fract(value: Storage<Self>) -> Storage<Self> { Self(R::fract(value.0), R::fract(value.1)) }
    #[inline(always)] fn next_up(value: Storage<Self>) -> Storage<Self> { Self(R::next_up(value.0), R::next_up(value.1)) }
    #[inline(always)] fn next_down(value: Storage<Self>) -> Storage<Self> { Self(R::next_down(value.0), R::next_down(value.1)) }
}

#[rustfmt::skip]
impl<R: IntegerRegister> IntegerRegister for DoublePumpRegister<R>
where
    typenum::Double<R::Lanes>: Lanes,
{
    #[inline(always)] fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Self(R::mulhi(lhs.0, rhs.0), R::mulhi(lhs.1, rhs.1)) }
    #[inline(always)] fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Self(R::mullo(lhs.0, rhs.0), R::mullo(lhs.1, rhs.1)) }

    #[inline(always)] fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Self(R::saturating_add(lhs.0, rhs.0), R::saturating_add(lhs.1, rhs.1)) }
    #[inline(always)] fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Self(R::saturating_sub(lhs.0, rhs.0), R::saturating_sub(lhs.1, rhs.1)) }
    #[inline(always)] fn wrapping_sum(value: Storage<Self>) -> Self::Element { R::wrapping_sum(R::add(value.0, value.1)) }
    #[inline(always)] fn wrapping_product(value: Storage<Self>) -> Self::Element { R::wrapping_product(R::mul(value.0, value.1)) }

    #[inline(always)]
    fn div_branched(value: Storage<Self>, divider: crate::divider::Divider<Self::Element>) -> Storage<Self> {
        Self(R::div_branched(value.0, divider), R::div_branched(value.1, divider))
    }

    #[inline(always)]
    fn div_branchfree(
        value: Storage<Self>,
        divider: crate::divider::BranchfreeDivider<Self::Element>,
    ) -> Storage<Self> {
        Self(R::div_branchfree(value.0, divider), R::div_branchfree(value.1, divider))
    }

    #[inline(always)]
    fn divv_branchfree(value: Storage<Self>, dividers: VectorDivider<Self>) -> Storage<Self> {
        let (lo, hi) = dividers.split();
        Self(R::divv_branchfree(value.0, lo), R::divv_branchfree(value.1, hi))
    }

    const HAS_HARDWARE_POPCNT: bool = R::HAS_HARDWARE_POPCNT;

    #[inline(always)] fn count_ones(value: Storage<Self>) -> Storage<Self> { Self(R::count_ones(value.0), R::count_ones(value.1)) }
    #[inline(always)] fn count_zeros(value: Storage<Self>) -> Storage<Self> { Self(R::count_zeros(value.0), R::count_zeros(value.1)) }
    #[inline(always)] fn leading_zeros(value: Storage<Self>) -> Storage<Self> { Self(R::leading_zeros(value.0), R::leading_zeros(value.1)) }
    #[inline(always)] fn leading_ones(value: Storage<Self>) -> Storage<Self> { Self(R::leading_ones(value.0), R::leading_ones(value.1)) }
    #[inline(always)] fn trailing_ones(value: Storage<Self>) -> Storage<Self> { Self(R::trailing_ones(value.0), R::trailing_ones(value.1)) }
    #[inline(always)] fn trailing_zeros(value: Storage<Self>) -> Storage<Self> { Self(R::trailing_zeros(value.0), R::trailing_zeros(value.1)) }
}

#[rustfmt::skip]
impl<R: UnsignedIntegerRegister> UnsignedIntegerRegister for DoublePumpRegister<R>
where
    typenum::Double<R::Lanes>: Lanes,
{
    #[inline(always)] fn ilog2p1(value: Storage<Self>) -> Storage<Self> { Self(R::ilog2p1(value.0), R::ilog2p1(value.1)) }
    #[inline(always)] fn next_power_of_two_m1(value: Storage<Self>) -> Storage<Self> { Self(R::next_power_of_two_m1(value.0), R::next_power_of_two_m1(value.1)) }
    #[inline(always)] fn is_power_of_two(value: Storage<Self>) -> Storage<Self> { Self(R::is_power_of_two(value.0), R::is_power_of_two(value.1)) }
    #[inline(always)] fn parity(value: Storage<Self>) -> Storage<Self> { Self(R::parity(value.0), R::parity(value.1)) }
}

#[rustfmt::skip]
impl<R: SignedIntegerRegister> SignedIntegerRegister for DoublePumpRegister<R>
where
    typenum::Double<R::Lanes>: Lanes,
{
    #[inline(always)] fn srai<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> { Self(R::srai::<IMM8>(value.0), R::srai::<IMM8>(value.1)) }
    #[inline(always)] fn sra(value: Storage<Self>, shift: u32) -> Storage<Self> { Self(R::sra(value.0, shift), R::sra(value.1, shift)) }
    #[inline(always)] fn srav(value: Storage<Self>, shifts: Storage<Self::USize>) -> Storage<Self> { Self(R::srav(value.0, shifts.0), R::srav(value.1, shifts.1)) }
}

impl<R: SwizzleRegister> SwizzleRegister for DoublePumpRegister<R>
where
    typenum::Double<R::Lanes>: Lanes,
{
    const HAS_PERMUTEV: bool = R::HAS_PERMUTEV;

    #[inline(always)]
    fn permutev(value: Storage<Self>, mut idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        if const { !Self::HAS_PERMUTEV } {
            return Self::scalar_permutev(value, idxs);
        }

        // mask out all indices to be within the range of R
        idxs.iter_mut().for_each(|idx| *idx &= <R::Lanes as Unsigned>::U32 - 1);

        let (blend_lo, blend_hi) = {
            let mut blends = Self::EMPTY; // mask register

            idxs.iter()
                .zip(Self::as_array_mut(&mut blends))
                .for_each(|(idx, blend)| {
                    // hopefully compiles to cmov or similar
                    *blend = if *idx < <R::Lanes as Unsigned>::U32 {
                        Element::FALSY
                    } else {
                        Element::TRUTHY
                    };
                });

            Self::split(blends)
        };

        let [pidx_lo, pidx_hi] = Self::split_array(idxs);

        let DoublePumpRegister(lo, hi) = value;

        let res_lo_from_lo: R::Storage = R::permutev(lo, pidx_lo.clone());
        let res_lo_from_hi: R::Storage = R::permutev(hi, pidx_lo);

        let res_hi_from_lo: R::Storage = R::permutev(lo, pidx_hi.clone());
        let res_hi_from_hi: R::Storage = R::permutev(hi, pidx_hi);

        let low: R::Storage = R::blendv(blend_lo, res_lo_from_lo, res_lo_from_hi);
        let high: R::Storage = R::blendv(blend_hi, res_hi_from_lo, res_hi_from_hi);

        Self(low, high)
    }
}

impl<FROM: Register, INTO: CastRegister<FROM>> CastRegister<DoublePumpRegister<FROM>> for DoublePumpRegister<INTO>
where
    typenum::Double<FROM::Lanes>: Lanes,
    typenum::Double<INTO::Lanes>: Lanes,
{
    #[inline(always)]
    fn cast_from(value: Storage<DoublePumpRegister<FROM>>) -> Storage<Self> {
        Self(INTO::cast_from(value.0), INTO::cast_from(value.1))
    }

    #[inline(always)]
    fn fast_cast_from(value: Storage<DoublePumpRegister<FROM>>) -> Storage<Self> {
        Self(INTO::fast_cast_from(value.0), INTO::fast_cast_from(value.1))
    }
}

impl<FROM: Register, INTO: BitsRegister<FROM>> BitsRegister<DoublePumpRegister<FROM>> for DoublePumpRegister<INTO>
where
    typenum::Double<FROM::Lanes>: Lanes,
    typenum::Double<INTO::Lanes>: Lanes,
{
    #[inline(always)]
    fn from_bits(value: Storage<DoublePumpRegister<FROM>>) -> Storage<Self> {
        Self(INTO::from_bits(value.0), INTO::from_bits(value.1))
    }
}

impl<R: FloatRegister> LinAlg4Register for DoublePumpRegister<R>
where
    Self: FloatRegister<Lanes = typenum::U4> + SwizzleRegister,
{
    // default implementations are fine
}

// TODO: Improve the swizzling here when some generic variant is available
impl<R: FloatRegister> LinAlg3Register for DoublePumpRegister<R>
where
    Self: FloatRegister<Lanes: ValidLinAlg3Length<Self>> + SwizzleRegister,
{
    #[inline(always)]
    fn min_element3(value: Storage<Self>) -> Self::Element {
        let mut a = Self::extract::<0>(value);
        let b = Self::extract::<1>(value);
        let c = Self::extract::<2>(value);

        if b < a {
            a = b;
        }

        if c < a {
            a = c;
        }

        a
    }

    #[inline(always)]
    fn max_element3(value: Storage<Self>) -> Self::Element {
        let mut a = Self::extract::<0>(value);
        let b = Self::extract::<1>(value);
        let c = Self::extract::<2>(value);

        if b > a {
            a = b;
        }

        if c > a {
            a = c;
        }

        a
    }

    #[inline(always)]
    fn sum_elements3(value: Storage<Self>) -> Self::Element {
        let a = Self::extract::<0>(value);
        let b = Self::extract::<1>(value);
        let c = Self::extract::<2>(value);

        a + b + c
    }

    #[inline(always)]
    fn prod_elements3(value: Storage<Self>) -> Self::Element {
        let a = Self::extract::<0>(value);
        let b = Self::extract::<1>(value);
        let c = Self::extract::<2>(value);

        a * b * c
    }
}
