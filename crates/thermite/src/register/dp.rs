//! Double-pumped registers for wider SIMD operations.

#![warn(missing_docs, clippy::missing_safety_doc)]

use crate::{
    divider::vector::VectorDivider,
    isa::InstructionSet,
    register::{LinAlg4Register, SignedIntegerRegister, ValidLinAlg3Length},
};

use super::{
    BitCastRegister, BitshiftRegister, BitwiseRegister, CastMaskRegister, CastRegister, CoreRegister, FloatRegister,
    IntegerRegister, Lanes, LinAlg3Register, MaskRegister, NumericRegister, PartialOrdRegister, Register,
    SignedRegister, Storage, SwizzleRegister, UnsignedIntegerRegister,
};

use generic_array::{
    GenericArray,
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
pub struct DoublePumpRegister<R: CoreRegister>(pub(crate) Storage<R>, pub(crate) Storage<R>);

const _: () = {
    use core::fmt;

    impl<R: CoreRegister> fmt::Debug for DoublePumpRegister<R> {
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

impl<R: CoreRegister> Clone for DoublePumpRegister<R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: CoreRegister> Copy for DoublePumpRegister<R> {}

impl<R: CoreRegister> DoublePumpRegister<R>
where
    typenum::Double<R::Lanes>: Lanes,
{
    #[inline(always)]
    fn split_array<T>(input: GenericArray<T, <Self as CoreRegister>::Lanes>) -> [GenericArray<T, R::Lanes>; 2] {
        unsafe { generic_array::const_transmute(input) }
    }
}

#[thermite_macros::double_pump_impl]
#[skip_masked]
impl<R: CoreRegister> CoreRegister for DoublePumpRegister<R>
where
    typenum::Double<R::Lanes>: Lanes,
{
    type Lanes = typenum::Double<R::Lanes>;
    type Storage = DoublePumpRegister<R>;
    type Mask = DoublePumpRegister<R::Mask>;

    const IS_EMULATED: bool = true; // sad, but true.

    const ISA: InstructionSet = R::ISA;

    const EMPTY: Storage<Self> = Self(R::EMPTY, R::EMPTY);

    fn blendv(mask: Storage<Self::Mask>, on_false: Storage<Self>, on_true: Storage<Self>) -> Storage<Self> {
        Self(
            R::blendv(mask.0, on_false.0, on_true.0),
            R::blendv(mask.1, on_false.1, on_true.1),
        )
    }

    fn z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        Self(R::z(mask.0, value.0), R::z(mask.1, value.1))
    }
}

#[thermite_macros::double_pump_impl]
#[skip_masked]
impl<R: MaskRegister> MaskRegister for DoublePumpRegister<R>
where
    typenum::Double<R::Lanes>: Lanes,
{
    fn set(mask: Storage<Self>, lane: usize, value: bool) -> Storage<Self> {
        let lane_count = R::Lanes::USIZE;
        if lane < lane_count {
            Self(R::set(mask.0, lane, value), mask.1)
        } else {
            Self(mask.0, R::set(mask.1, lane - lane_count, value))
        }
    }

    fn test(mask: Storage<Self>, lane: usize) -> bool {
        let lane_count = R::Lanes::USIZE;
        if lane < lane_count {
            R::test(mask.0, lane)
        } else {
            R::test(mask.1, lane - lane_count)
        }
    }

    const TRUTHY: Storage<Self> = Self(R::TRUTHY, R::TRUTHY);
    const FALSY: Storage<Self> = Self(R::FALSY, R::FALSY);

    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        let [lhs, rhs] = Self::split_array(value);
        Self(R::new_mask(lhs), R::new_mask(rhs))
    }

    fn all(value: Storage<Self>) -> bool {
        R::all(value.0) && R::all(value.1)
    }

    fn any(value: Storage<Self>) -> bool {
        R::any(value.0) || R::any(value.1)
    }

    fn none(value: Storage<Self>) -> bool {
        R::none(value.0) && R::none(value.1)
    }

    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        if Self::Lanes::USIZE <= 64 {
            let lo = R::native_bitmask(value.0)?;
            let hi = R::native_bitmask(value.1)?;

            Some(lo | (hi << R::Lanes::U64))
        } else {
            None
        }
    }

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

#[thermite_macros::double_pump_impl]
#[conditional]
impl<R: BitwiseRegister> BitwiseRegister for DoublePumpRegister<R>
where
    typenum::Double<R::Lanes>: Lanes,
{
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    fn not(value: Storage<Self>) -> Storage<Self> {}
    fn ternlog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {}
    fn bilog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {}
}

#[thermite_macros::double_pump_impl]
impl<R: Register> Register for DoublePumpRegister<R>
where
    typenum::Double<R::Lanes>: Lanes,
{
    type Element = R::Element;

    type HalfRegister = R;
    type DoubleRegister = DoublePumpRegister<Self>;

    type ISize = DoublePumpRegister<R::ISize>;
    type USize = DoublePumpRegister<R::USize>;

    const HAS_EQUAL_SIZE_MASK: bool = R::HAS_EQUAL_SIZE_MASK;

    #[skip_masked]
    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        Self(R::from_mask(mask.0), R::from_mask(mask.1))
    }

    #[skip_masked]
    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        DoublePumpRegister(R::into_mask(value.0), R::into_mask(value.1))
    }

    #[skip_masked]
    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        DoublePumpRegister(R::into_mask_unchecked(value.0), R::into_mask_unchecked(value.1))
    }

    #[skip_masked]
    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        DoublePumpRegister(R::msb_to_mask(value.0), R::msb_to_mask(value.1))
    }

    #[skip_masked]
    fn split(value: Storage<Self>) -> (Storage<Self::HalfRegister>, Storage<Self::HalfRegister>)
    where
        Self::HalfRegister: Register,
    {
        (value.0, value.1)
    }

    #[skip_masked]
    fn join(lo: Storage<Self::HalfRegister>, hi: Storage<Self::HalfRegister>) -> Storage<Self>
    where
        Self::HalfRegister: Register,
    {
        Self(lo, hi)
    }

    #[skip_masked]
    fn new(value: generic_array::GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        // SAFETY: With arrays always being repr(C), we can safely transmute
        // the double-length array into the two half arrays for each register.
        let [lhs, rhs] = Self::split_array(value);

        Self(R::new(lhs), R::new(rhs))
    }

    fn single(value: Self::Element) -> Storage<Self> {
        Self(R::single(value), R::EMPTY)
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        Self(R::splat(value), R::splat(value))
    }

    #[conditional]
    fn broadcast<const I: usize>(value: Storage<Self>) -> Storage<Self> {
        // NOTE: using `broadcast::<I>(value)` doesn't work
        // because the const index is propagated before the conditional
        // check, leading to out-of-bounds errors.
        Self::broadcastv(value, I)
    }

    #[conditional]
    fn broadcastv(value: Storage<Self>, idx: usize) -> Storage<Self> {
        let r = if idx < R::Lanes::USIZE {
            R::broadcastv(value.0, idx)
        } else {
            R::broadcastv(value.1, idx - R::Lanes::USIZE)
        };

        Self(r, r)
    }

    #[skip_masked]
    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { Self(R::load(ptr), R::load(ptr.add(R::Lanes::USIZE))) }
    }

    #[skip_masked]
    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { Self(R::load_unaligned(ptr), R::load_unaligned(ptr.add(R::Lanes::USIZE))) }
    }

    #[skip_masked]
    unsafe fn load_stream(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { Self(R::load_stream(ptr), R::load_stream(ptr.add(R::Lanes::USIZE))) }
    }

    #[skip_masked]
    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe {
            R::store(ptr, value.0);
            R::store(ptr.add(R::Lanes::USIZE), value.1);
        }
    }

    #[skip_masked]
    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe {
            R::store_unaligned(ptr, value.0);
            R::store_unaligned(ptr.add(R::Lanes::USIZE), value.1);
        }
    }

    #[skip_masked]
    unsafe fn store_stream(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe {
            R::store_stream(ptr, value.0);
            R::store_stream(ptr.add(R::Lanes::USIZE), value.1);
        }
    }

    fn fold<F>(first: Self::Element, value: Storage<Self>, f: F) -> Self::Element
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element,
    {
        R::fold(R::fold(first, value.0, &f), value.1, &f)
    }

    fn reduce<F>(value: Storage<Self>, f: F) -> Self::Element
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element,
    {
        let lo = R::reduce(value.0, &f);
        let hi = R::reduce(value.1, &f);

        f(lo, hi)
    }

    // NOTE: This has masked variants, but it's too complicated
    // to actually mask here, due to the lane swapping, so
    // we just let the default impls handle it.
    #[skip_masked]
    fn reverse(value: Storage<Self>) -> Storage<Self> {
        Self(R::reverse(value.1), R::reverse(value.0))
    }

    // double-pump logic for unpack does not add extra complexity,
    // so this is determined solely by the underlying register.
    const HAS_SIMPLE_UNPACK: bool = R::HAS_SIMPLE_UNPACK;

    #[skip_masked]
    fn unpack(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        let (r1_lo, r1_hi) = R::unpack(a.0, b.0);
        let (r2_lo, r2_hi) = R::unpack(a.1, b.1);

        (DoublePumpRegister(r1_lo, r1_hi), DoublePumpRegister(r2_lo, r2_hi))
    }

    #[conditional]
    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        Self(R::swap_bytes(value.0), R::swap_bytes(value.1))
    }
}

#[thermite_macros::double_pump_impl]
#[conditional]
impl<R: BitshiftRegister> BitshiftRegister for DoublePumpRegister<R>
where
    typenum::Double<R::Lanes>: Lanes,
{
    const HAS_WIDE_BYTE_SHIFTS: bool = false;
    const HAS_TRUE_SHIFTV: bool = R::HAS_TRUE_SHIFTV;

    fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {}
    fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {}
    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {}
    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {}
    fn shlv(value: Storage<Self>, shifts: Storage<Self::USize>) -> Storage<Self> {}
    fn shrv(value: Storage<Self>, shifts: Storage<Self::USize>) -> Storage<Self> {}
    fn rol(value: Storage<Self>, shift: u32) -> Storage<Self> {}
    fn ror(value: Storage<Self>, shift: u32) -> Storage<Self> {}
    fn roli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {}
    fn rori<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {}
    fn rolv(value: Storage<Self>, shifts: Storage<Self::USize>) -> Storage<Self> {}
    fn rorv(value: Storage<Self>, shifts: Storage<Self::USize>) -> Storage<Self> {}
    fn reverse_bits(value: Storage<Self>) -> Storage<Self> {}
}

impl<FROM: MaskRegister, INTO: CastMaskRegister<FROM>> CastMaskRegister<DoublePumpRegister<FROM>>
    for DoublePumpRegister<INTO>
where
    typenum::Double<INTO::Lanes>: Lanes,
    typenum::Double<FROM::Lanes>: Lanes,
{
    #[inline(always)]
    fn mask_from(value: Storage<DoublePumpRegister<FROM>>) -> Storage<Self> {
        Self(INTO::mask_from(value.0), INTO::mask_from(value.1))
    }
}

#[rustfmt::skip]
#[thermite_macros::double_pump_impl] #[skip_masked]
impl<R: PartialOrdRegister> PartialOrdRegister for DoublePumpRegister<R>
where
    typenum::Double<R::Lanes>: Lanes,
{
    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {}
    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {}
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {}
    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {}
    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {}
    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {}
}

#[rustfmt::skip]
#[thermite_macros::double_pump_impl]
impl<R: NumericRegister> NumericRegister for DoublePumpRegister<R>
where
    typenum::Double<R::Lanes>: Lanes,
{
    const ZERO: Storage<Self> = Self(R::ZERO, R::ZERO);
    const ONE: Storage<Self> = Self(R::ONE, R::ONE);
    const TWO: Storage<Self> = Self(R::TWO, R::TWO);

    const MIN: Storage<Self> = Self(R::MIN, R::MIN);
    const MAX: Storage<Self> = Self(R::MAX, R::MAX);

    fn max_element(value: Storage<Self>) -> Self::Element { R::max_element(R::max(value.0, value.1)) }
    fn min_element(value: Storage<Self>) -> Self::Element { R::min_element(R::min(value.0, value.1)) }
    fn sum_elements(value: Storage<Self>) -> Self::Element { R::sum_elements(R::add(value.0, value.1)) }
    fn prod_elements(value: Storage<Self>) -> Self::Element { R::prod_elements(R::mul(value.0, value.1)) }

    #[conditional] fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}

    fn sort(value: Storage<Self>) -> Storage<Self> {
        let lo = R::min(value.0, value.1);
        let hi = R::max(value.0, value.1);

        Self(R::sort(lo), R::sort(hi))
    }

    fn offset() -> Storage<Self> {
        // Because we're doubling up each time, we can just x2 the offset
        let mut offset = R::offset();
        offset = R::add(offset, offset); // via addition because it's faster
        Self(offset, offset)
    }

    fn indexed() -> Storage<Self> {
        // offset the second index to remain consistent
        let indexed = R::indexed();
        Self(indexed, R::add(indexed, R::offset()))
    }
}

#[rustfmt::skip]
#[thermite_macros::double_pump_impl] #[conditional]
impl<R: SignedRegister> SignedRegister for DoublePumpRegister<R>
where
    typenum::Double<R::Lanes>: Lanes,
{
    const NEG_ONE: Storage<Self> = Self(R::NEG_ONE, R::NEG_ONE);
    const MIN_POSITIVE: Storage<Self> = Self(R::MIN_POSITIVE, R::MIN_POSITIVE);

    fn neg(value: Storage<Self>) -> Storage<Self> {}
    fn abs(value: Storage<Self>) -> Storage<Self> {}
    fn copysign(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}

    #[skip_masked] fn signum(value: Storage<Self>) -> Storage<Self> {}

    #[skip_masked] fn is_negative(value: Storage<Self>) -> Storage<Self::Mask> {}
    #[skip_masked] fn is_positive(value: Storage<Self>) -> Storage<Self::Mask> {}
}

#[rustfmt::skip]
#[thermite_macros::double_pump_impl] #[conditional]
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

    #[skip_masked] fn is_nan(value: Storage<Self>) -> Storage<Self::Mask> {}
    #[skip_masked] fn is_infinite(value: Storage<Self>) -> Storage<Self::Mask> {}
    #[skip_masked] fn is_finite(value: Storage<Self>) -> Storage<Self::Mask> {}
    #[skip_masked] fn is_subnormal(value: Storage<Self>) -> Storage<Self::Mask> {}
    #[skip_masked] fn is_zero_or_subnormal(value: Storage<Self>) -> Storage<Self::Mask> {}
    #[skip_masked] fn is_normal(value: Storage<Self>) -> Storage<Self::Mask> {}
    fn mul_adde(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {}
    fn mul_sube(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> { }
    fn nmul_adde(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {}
    fn nmul_sube(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {}
    fn mul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {}
    fn mul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {}
    fn nmul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> { }
    fn nmul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {}
    fn sqrt(value: Storage<Self>) -> Storage<Self> {}
    fn rsqrt(value: Storage<Self>) -> Storage<Self> {}

    const HAS_APPROX_RSQRT: bool = R::HAS_APPROX_RSQRT;
    const HAS_APPROX_RCP: bool = R::HAS_APPROX_RCP;

    fn rcp(value: Storage<Self>) -> Storage<Self> {}
    fn floor(value: Storage<Self>) -> Storage<Self> {}
    fn ceil(value: Storage<Self>) -> Storage<Self> {}
    fn round(value: Storage<Self>) -> Storage<Self> {}
    fn trunc(value: Storage<Self>) -> Storage<Self> {}
    fn fract(value: Storage<Self>) -> Storage<Self> {}
    fn next_up(value: Storage<Self>) -> Storage<Self> {}
    fn next_down(value: Storage<Self>) -> Storage<Self> {}

    const HAS_NATIVE_LDEXP: bool = R::HAS_NATIVE_LDEXP;
    const HAS_NATIVE_FREXP: bool = R::HAS_NATIVE_FREXP;

    #[skip_masked]
    unsafe fn native_ldexp(value: Storage<Self>, exp: Storage<Self::Signed>) -> Storage<Self> {
        unsafe { Self(R::native_ldexp(value.0, exp.0), R::native_ldexp(value.1, exp.1)) }
    }

    #[skip_masked]
    unsafe fn native_frexp(value: Storage<Self>) -> (Storage<Self>, Storage<Self::Signed>) {
        let (lo_val, lo_exp) = unsafe { R::native_frexp(value.0) };
        let (hi_val, hi_exp) = unsafe { R::native_frexp(value.1) };

        (Self(lo_val, hi_val), DoublePumpRegister(lo_exp, hi_exp))
    }
}

#[rustfmt::skip]
#[thermite_macros::double_pump_impl] #[conditional]
impl<R: IntegerRegister> IntegerRegister for DoublePumpRegister<R>
where
    typenum::Double<R::Lanes>: Lanes,
{
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}

    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}

    fn wrapping_sum(value: Storage<Self>) -> Self::Element { R::wrapping_sum(R::add(value.0, value.1)) }
    fn wrapping_product(value: Storage<Self>) -> Self::Element { R::wrapping_product(R::mul(value.0, value.1)) }

    fn div_branched(value: Storage<Self>, divider: crate::divider::Divider<Self::Element>) -> Storage<Self> {}

    fn div_branchfree(
        value: Storage<Self>,
        divider: crate::divider::BranchfreeDivider<Self::Element>,
    ) -> Storage<Self> {}

    // these divv forms are explicitly implemented for the dividers' split logic

    #[skip_masked]
    fn divv_branchfree(value: Storage<Self>, dividers: VectorDivider<Self>) -> Storage<Self> {
        let (lo, hi) = dividers.split();
        Self(R::divv_branchfree(value.0, lo), R::divv_branchfree(value.1, hi))
    }

    #[skip_masked]
    fn divv_branchfree_c(mask: Storage<Self::Mask>, value: Storage<Self>, dividers: VectorDivider<Self>) -> Storage<Self> {
        let (lo, hi) = dividers.split();
        Self(R::divv_branchfree_c(mask.0, value.0, lo), R::divv_branchfree_c(mask.1, value.1, hi))
    }

    #[skip_masked]
    fn divv_branchfree_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>, dividers: VectorDivider<Self>) -> Storage<Self> {
        let (lo, hi) = dividers.split();
        Self(R::divv_branchfree_m(src.0, mask.0, value.0, lo), R::divv_branchfree_m(src.1, mask.1, value.1, hi))
    }

    #[skip_masked]
    fn divv_branchfree_z(mask: Storage<Self::Mask>, value: Storage<Self>, dividers: VectorDivider<Self>) -> Storage<Self> {
        let (lo, hi) = dividers.split();
        Self(R::divv_branchfree_z(mask.0, value.0, lo), R::divv_branchfree_z(mask.1, value.1, hi))
    }

    const HAS_HARDWARE_POPCNT: bool = R::HAS_HARDWARE_POPCNT;

    fn count_ones(value: Storage<Self>) -> Storage<Self> {}
    fn count_zeros(value: Storage<Self>) -> Storage<Self> {}
    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {}
    fn leading_ones(value: Storage<Self>) -> Storage<Self> {}
    fn trailing_ones(value: Storage<Self>) -> Storage<Self> {}
    fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {}
}

#[thermite_macros::double_pump_impl]
#[conditional]
impl<R: UnsignedIntegerRegister> UnsignedIntegerRegister for DoublePumpRegister<R>
where
    typenum::Double<R::Lanes>: Lanes,
{
    fn ilog2p1(value: Storage<Self>) -> Storage<Self> {}
    fn next_power_of_two_m1(value: Storage<Self>) -> Storage<Self> {}
    fn parity(value: Storage<Self>) -> Storage<Self> {}

    #[skip_masked]
    fn is_power_of_two(value: Storage<Self>) -> Storage<Self::Mask> {}
}

#[thermite_macros::double_pump_impl]
#[conditional]
impl<R: SignedIntegerRegister> SignedIntegerRegister for DoublePumpRegister<R>
where
    typenum::Double<R::Lanes>: Lanes,
{
    fn srai<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {}
    fn sra(value: Storage<Self>, shift: u32) -> Storage<Self> {}
    fn srav(value: Storage<Self>, shifts: Storage<Self::USize>) -> Storage<Self> {}
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

        let mut blends = <Self::Mask as MaskRegister>::FALSY; // mask register

        for (i, &idx) in idxs.iter().enumerate() {
            if idx > <R::Lanes as Unsigned>::U32 - 1 {
                // TODO: Optimize?
                blends = <Self::Mask as MaskRegister>::set(blends, i, true);
            }
        }

        let [pidx_lo, pidx_hi] = Self::split_array(idxs);

        let DoublePumpRegister(lo, hi) = value;

        let res_lo_from_lo: R::Storage = R::permutev(lo, pidx_lo.clone());
        let res_lo_from_hi: R::Storage = R::permutev(hi, pidx_lo);

        let res_hi_from_lo: R::Storage = R::permutev(lo, pidx_hi.clone());
        let res_hi_from_hi: R::Storage = R::permutev(hi, pidx_hi);

        let low: R::Storage = R::blendv(blends.0, res_lo_from_lo, res_lo_from_hi);
        let high: R::Storage = R::blendv(blends.1, res_hi_from_lo, res_hi_from_hi);

        Self(low, high)
    }
}

impl<FROM: CoreRegister, INTO: CastRegister<FROM>> CastRegister<DoublePumpRegister<FROM>> for DoublePumpRegister<INTO>
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

impl<FROM: CoreRegister, INTO: BitCastRegister<FROM>> BitCastRegister<DoublePumpRegister<FROM>>
    for DoublePumpRegister<INTO>
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
    Self: LinAlg3Register + FloatRegister<Lanes = typenum::U4> + SwizzleRegister,
{
    // default implementations are fine
}

// TODO: Improve the swizzling here when some generic variant is available
impl<R: FloatRegister> LinAlg3Register for DoublePumpRegister<R>
where
    Self: FloatRegister<Lanes: ValidLinAlg3Length<Self>, Storage = Self, Element = R::Element> + SwizzleRegister,
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
        let lo = if <R::Lanes as Unsigned>::USIZE == 2 {
            R::sum_elements(value.0)
        } else {
            let a = Self::extract::<0>(value);
            let b = Self::extract::<1>(value);

            a + b
        };

        let c = Self::extract::<2>(value);

        lo + c
    }

    #[inline(always)]
    fn prod_elements3(value: Storage<Self>) -> Self::Element {
        let lo = if <R::Lanes as Unsigned>::USIZE == 2 {
            R::prod_elements(value.0)
        } else {
            let a = Self::extract::<0>(value);
            let b = Self::extract::<1>(value);

            a * b
        };

        let c = Self::extract::<2>(value);

        lo * c
    }
}
