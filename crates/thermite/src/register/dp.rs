use core::ops::Shl;

use super::{
    BitsRegister, CastMaskRegister, CastRegister, FloatRegister, IntegerRegister, Lanes, LinAlg3Register, MaskElement,
    MaskRegister, NumericRegister, PartialOrdRegister, Register, ShiftRegister, SignedRegister, SwizzleRegister,
    UnsignedIntegerRegister,
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
pub struct DoublePumpRegister<R: Register>(R::Storage, R::Storage);

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

pub trait DoublePumpVector {
    type DoublePump;
}

impl<R: Register> DoublePumpVector for crate::vector::Vector<R>
where
    R::DoubleRegister: Register,
{
    type DoublePump = crate::vector::Vector<R::DoubleRegister>;
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

    type HalfRegister = R;
    type DoubleRegister = DoublePumpRegister<Self>;

    const EMPTY: Self::Storage = Self(R::EMPTY, R::EMPTY);

    #[inline(always)]
    fn split(
        value: Self::Storage,
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
    ) -> Self::Storage
    where
        Self::HalfRegister: Register,
    {
        Self(lo, hi)
    }

    #[inline(always)]
    fn new(value: generic_array::GenericArray<Self::Element, Self::Lanes>) -> Self::Storage {
        // SAFETY: With arrays always being repr(C), we can safely transmute
        // the double-length array into the two half arrays for each register.
        let [lhs, rhs] = Self::split_array(value);

        Self(R::new(lhs), R::new(rhs))
    }

    #[inline(always)]
    fn splat(value: Self::Element) -> Self::Storage {
        Self(R::splat(value), R::splat(value))
    }

    #[inline(always)]
    unsafe fn load(ptr: *const Self::Element) -> Self::Storage {
        unsafe { Self(R::load(ptr), R::load(ptr.add(core::mem::size_of::<R::Storage>()))) }
    }

    #[inline(always)]
    unsafe fn load_unaligned(ptr: *const Self::Element) -> Self::Storage {
        unsafe {
            Self(
                R::load_unaligned(ptr),
                R::load_unaligned(ptr.add(core::mem::size_of::<R::Storage>())),
            )
        }
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut Self::Element, value: Self::Storage) {
        unsafe {
            R::store(ptr, value.0);
            R::store(ptr.add(core::mem::size_of::<R::Storage>()), value.1);
        }
    }

    #[inline(always)]
    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Self::Storage) {
        unsafe {
            R::store_unaligned(ptr, value.0);
            R::store_unaligned(ptr.add(core::mem::size_of::<R::Storage>()), value.1);
        }
    }

    #[inline(always)]
    fn bitxor(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        Self(R::bitxor(lhs.0, rhs.0), R::bitxor(lhs.1, rhs.1))
    }

    #[inline(always)]
    fn bitand(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        Self(R::bitand(lhs.0, rhs.0), R::bitand(lhs.1, rhs.1))
    }

    #[inline(always)]
    fn bitandnot(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        Self(R::bitandnot(lhs.0, rhs.0), R::bitandnot(lhs.1, rhs.1))
    }

    #[inline(always)]
    fn bitor(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        Self(R::bitor(lhs.0, rhs.0), R::bitor(lhs.1, rhs.1))
    }

    #[inline(always)]
    fn not(value: Self::Storage) -> Self::Storage {
        Self(R::not(value.0), R::not(value.1))
    }

    #[inline(always)]
    fn blendv(mask: Self::Storage, lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        Self(R::blendv(mask.0, lhs.0, rhs.0), R::blendv(mask.1, lhs.1, rhs.1))
    }

    const HAS_MSB_BLENDV: bool = R::HAS_MSB_BLENDV;

    #[inline(always)]
    fn shl(value: Self::Storage, shift: u32) -> Self::Storage {
        Self(R::shl(value.0, shift), R::shl(value.1, shift))
    }

    #[inline(always)]
    fn shr(value: Self::Storage, shift: u32) -> Self::Storage {
        Self(R::shr(value.0, shift), R::shr(value.1, shift))
    }

    #[inline(always)]
    fn shlv(value: Self::Storage, shifts: impl Into<GenericArray<u32, Self::Lanes>>) -> Self::Storage {
        let [shift_lo, shift_hi] = Self::split_array(shifts.into());
        Self(R::shlv(value.0, shift_lo), R::shlv(value.1, shift_hi))
    }

    #[inline(always)]
    fn shrv(value: Self::Storage, shifts: impl Into<GenericArray<u32, Self::Lanes>>) -> Self::Storage {
        let [shift_lo, shift_hi] = Self::split_array(shifts.into());
        Self(R::shrv(value.0, shift_lo), R::shrv(value.1, shift_hi))
    }

    #[inline(always)]
    fn reverse(mut value: Self::Storage) -> Self::Storage {
        Self(R::reverse(value.1), R::reverse(value.0))
    }
}

impl<R: ShiftRegister> ShiftRegister for DoublePumpRegister<R>
where
    typenum::Double<R::Lanes>: Lanes,
{
    #[inline(always)]
    fn shli<const IMM8: i32>(value: Self::Storage) -> Self::Storage {
        Self(R::shli::<IMM8>(value.0), R::shli::<IMM8>(value.1))
    }

    #[inline(always)]
    fn shri<const IMM8: i32>(value: Self::Storage) -> Self::Storage {
        Self(R::shri::<IMM8>(value.0), R::shri::<IMM8>(value.1))
    }
}

impl<R: MaskRegister> MaskRegister for DoublePumpRegister<R>
where
    typenum::Double<R::Lanes>: Lanes,
{
    const TRUTHY: Self::Storage = Self(R::TRUTHY, R::TRUTHY);
    const FALSY: Self::Storage = Self(R::FALSY, R::FALSY);

    #[inline(always)]
    fn new_mask(value: impl Into<GenericArray<bool, Self::Lanes>>) -> Self::Storage {
        let [lhs, rhs] = Self::split_array(value.into());
        Self(R::new_mask(lhs), R::new_mask(rhs))
    }

    #[inline(always)]
    fn debug_iter_bool(value: &Self::Storage) -> impl Iterator<Item = bool> {
        let iter_lo = R::debug_iter_bool(&value.0);
        let iter_hi = R::debug_iter_bool(&value.1);
        iter_lo.chain(iter_hi)
    }

    #[inline(always)]
    fn all(value: Self::Storage) -> bool {
        R::all(value.0) && R::all(value.1)
    }

    #[inline(always)]
    fn any(value: Self::Storage) -> bool {
        R::any(value.0) || R::any(value.1)
    }

    #[inline(always)]
    fn none(value: Self::Storage) -> bool {
        R::none(value.0) && R::none(value.1)
    }
}

impl<FROM: MaskRegister, INTO: CastMaskRegister<FROM>> CastMaskRegister<DoublePumpRegister<FROM>>
    for DoublePumpRegister<INTO>
where
    typenum::Double<INTO::Lanes>: Lanes,
    typenum::Double<FROM::Lanes>: Lanes,
{
    #[inline(always)]
    fn mask_from(value: <DoublePumpRegister<FROM> as Register>::Storage) -> Self::Storage {
        Self(INTO::mask_from(value.0), INTO::mask_from(value.1))
    }
}

impl<R: PartialOrdRegister> PartialOrdRegister for DoublePumpRegister<R>
where
    typenum::Double<R::Lanes>: Lanes,
{
    #[inline(always)]
    fn lt(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        Self(R::lt(lhs.0, rhs.0), R::lt(lhs.1, rhs.1))
    }

    #[inline(always)]
    fn le(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        Self(R::le(lhs.0, rhs.0), R::le(lhs.1, rhs.1))
    }

    #[inline(always)]
    fn gt(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        Self(R::gt(lhs.0, rhs.0), R::gt(lhs.1, rhs.1))
    }

    #[inline(always)]
    fn ge(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        Self(R::ge(lhs.0, rhs.0), R::ge(lhs.1, rhs.1))
    }

    #[inline(always)]
    fn eq(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        Self(R::eq(lhs.0, rhs.0), R::eq(lhs.1, rhs.1))
    }

    #[inline(always)]
    fn ne(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        Self(R::ne(lhs.0, rhs.0), R::ne(lhs.1, rhs.1))
    }
}

impl<R: NumericRegister> NumericRegister for DoublePumpRegister<R>
where
    typenum::Double<R::Lanes>: Lanes,
{
    const ZERO: Self::Storage = Self(R::ZERO, R::ZERO);
    const ONE: Self::Storage = Self(R::ONE, R::ONE);
    const TWO: Self::Storage = Self(R::TWO, R::TWO);

    const MIN: Self::Storage = Self(R::MIN, R::MIN);
    const MAX: Self::Storage = Self(R::MAX, R::MAX);

    #[inline(always)]
    fn max_element(value: Self::Storage) -> Self::Element {
        R::max_element(R::max(value.0, value.1))
    }

    #[inline(always)]
    fn min_element(value: Self::Storage) -> Self::Element {
        R::min_element(R::min(value.0, value.1))
    }

    #[inline(always)]
    fn sum_elements(value: Self::Storage) -> Self::Element {
        R::sum_elements(R::add(value.0, value.1))
    }

    #[inline(always)]
    fn prod_elements(value: Self::Storage) -> Self::Element {
        R::prod_elements(R::mul(value.0, value.1))
    }

    #[inline(always)]
    fn add(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        Self(R::add(lhs.0, rhs.0), R::add(lhs.1, rhs.1))
    }

    #[inline(always)]
    fn sub(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        Self(R::sub(lhs.0, rhs.0), R::sub(lhs.1, rhs.1))
    }

    #[inline(always)]
    fn mul(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        Self(R::mul(lhs.0, rhs.0), R::mul(lhs.1, rhs.1))
    }

    #[inline(always)]
    fn div(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        Self(R::div(lhs.0, rhs.0), R::div(lhs.1, rhs.1))
    }

    #[inline(always)]
    fn rem(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        Self(R::rem(lhs.0, rhs.0), R::rem(lhs.1, rhs.1))
    }

    #[inline(always)]
    fn min(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        Self(R::min(lhs.0, rhs.0), R::min(lhs.1, rhs.1))
    }

    #[inline(always)]
    fn max(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        Self(R::max(lhs.0, rhs.0), R::max(lhs.1, rhs.1))
    }

    fn offset() -> Self::Storage {
        // Because we're doubling up each time, we can just x2 the offset
        let mut offset = R::offset();
        offset = R::add(offset, offset); // via addition because it's faster
        Self(offset, offset)
    }

    fn indexed() -> Self::Storage {
        // offset the second index to remain consistent
        let indexed = R::indexed();
        Self(indexed, R::add(indexed, R::offset()))
    }
}

impl<R: SignedRegister> SignedRegister for DoublePumpRegister<R>
where
    typenum::Double<R::Lanes>: Lanes,
{
    const NEG_ONE: Self::Storage = Self(R::NEG_ONE, R::NEG_ONE);

    #[inline(always)]
    fn neg(value: Self::Storage) -> Self::Storage {
        Self(R::neg(value.0), R::neg(value.1))
    }

    #[inline(always)]
    fn abs(value: Self::Storage) -> Self::Storage {
        Self(R::abs(value.0), R::abs(value.1))
    }

    #[inline(always)]
    fn copysign(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        Self(R::copysign(lhs.0, rhs.0), R::copysign(lhs.1, rhs.1))
    }

    #[inline(always)]
    fn signum(value: Self::Storage) -> Self::Storage {
        Self(R::signum(value.0), R::signum(value.1))
    }

    #[inline(always)]
    fn is_negative(value: Self::Storage) -> Self::Storage {
        Self(R::is_negative(value.0), R::is_negative(value.1))
    }

    #[inline(always)]
    fn is_positive(value: Self::Storage) -> Self::Storage {
        Self(R::is_positive(value.0), R::is_positive(value.1))
    }

    #[inline(always)]
    fn conditional_negate(value: Self::Storage, mask: Self::Storage) -> Self::Storage {
        Self(
            R::conditional_negate(value.0, mask.0),
            R::conditional_negate(value.1, mask.1),
        )
    }
}

impl<R: FloatRegister> FloatRegister for DoublePumpRegister<R>
where
    typenum::Double<R::Lanes>: Lanes,
{
    const HAS_TRUE_FMA: bool = R::HAS_TRUE_FMA;

    type Bits = DoublePumpRegister<R::Bits>;
    type Signed = DoublePumpRegister<R::Signed>;

    const HALF: Self::Storage = Self(R::HALF, R::HALF);
    const NEG_ZERO: Self::Storage = Self(R::NEG_ZERO, R::NEG_ZERO);
    const INFINITY: Self::Storage = Self(R::INFINITY, R::INFINITY);
    const NEG_INFINITY: Self::Storage = Self(R::NEG_INFINITY, R::NEG_INFINITY);
    const NAN: Self::Storage = Self(R::NAN, R::NAN);
    const EPSILON: Self::Storage = Self(R::EPSILON, R::EPSILON);

    #[inline(always)]
    fn is_nan(value: Self::Storage) -> Self::Storage {
        Self(R::is_nan(value.0), R::is_nan(value.1))
    }

    #[inline(always)]
    fn is_infinite(value: Self::Storage) -> Self::Storage {
        Self(R::is_infinite(value.0), R::is_infinite(value.1))
    }

    #[inline(always)]
    fn is_finite(value: Self::Storage) -> Self::Storage {
        Self(R::is_finite(value.0), R::is_finite(value.1))
    }

    #[inline(always)]
    fn is_subnormal(value: Self::Storage) -> Self::Storage {
        Self(R::is_subnormal(value.0), R::is_subnormal(value.1))
    }

    #[inline(always)]
    fn is_zero_or_subnormal(value: Self::Storage) -> Self::Storage {
        Self(R::is_zero_or_subnormal(value.0), R::is_zero_or_subnormal(value.1))
    }

    #[inline(always)]
    fn is_normal(value: Self::Storage) -> Self::Storage {
        Self(R::is_normal(value.0), R::is_normal(value.1))
    }

    #[inline(always)]
    fn mul_adde(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        Self(R::mul_adde(lhs.0, rhs.0, acc.0), R::mul_adde(lhs.1, rhs.1, acc.1))
    }

    #[inline(always)]
    fn mul_sube(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        Self(R::mul_sube(lhs.0, rhs.0, acc.0), R::mul_sube(lhs.1, rhs.1, acc.1))
    }

    #[inline(always)]
    fn nmul_adde(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        Self(R::nmul_adde(lhs.0, rhs.0, acc.0), R::nmul_adde(lhs.1, rhs.1, acc.1))
    }

    #[inline(always)]
    fn nmul_sube(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        Self(R::nmul_sube(lhs.0, rhs.0, acc.0), R::nmul_sube(lhs.1, rhs.1, acc.1))
    }

    #[inline(always)]
    fn mul_add(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        Self(R::mul_add(lhs.0, rhs.0, acc.0), R::mul_add(lhs.1, rhs.1, acc.1))
    }

    #[inline(always)]
    fn mul_sub(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        Self(R::mul_sub(lhs.0, rhs.0, acc.0), R::mul_sub(lhs.1, rhs.1, acc.1))
    }

    #[inline(always)]
    fn nmul_add(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        Self(R::nmul_add(lhs.0, rhs.0, acc.0), R::nmul_add(lhs.1, rhs.1, acc.1))
    }

    #[inline(always)]
    fn nmul_sub(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        Self(R::nmul_sub(lhs.0, rhs.0, acc.0), R::nmul_sub(lhs.1, rhs.1, acc.1))
    }

    #[inline(always)]
    fn sqrt(value: Self::Storage) -> Self::Storage {
        Self(R::sqrt(value.0), R::sqrt(value.1))
    }

    #[inline(always)]
    fn rsqrt(value: Self::Storage) -> Self::Storage {
        Self(R::rsqrt(value.0), R::rsqrt(value.1))
    }

    const HAS_APPROX_RSQRT: bool = R::HAS_APPROX_RSQRT;
    const HAS_APPROX_RCP: bool = R::HAS_APPROX_RCP;

    #[inline(always)]
    fn rcp(value: Self::Storage) -> Self::Storage {
        Self(R::rcp(value.0), R::rcp(value.1))
    }

    #[inline(always)]
    fn floor(value: Self::Storage) -> Self::Storage {
        Self(R::floor(value.0), R::floor(value.1))
    }

    #[inline(always)]
    fn ceil(value: Self::Storage) -> Self::Storage {
        Self(R::ceil(value.0), R::ceil(value.1))
    }

    #[inline(always)]
    fn round(value: Self::Storage) -> Self::Storage {
        Self(R::round(value.0), R::round(value.1))
    }

    #[inline(always)]
    fn trunc(value: Self::Storage) -> Self::Storage {
        Self(R::trunc(value.0), R::trunc(value.1))
    }

    #[inline(always)]
    fn fract(value: Self::Storage) -> Self::Storage {
        Self(R::fract(value.0), R::fract(value.1))
    }

    #[inline(always)]
    fn next_up(value: Self::Storage) -> Self::Storage {
        Self(R::next_up(value.0), R::next_up(value.1))
    }

    #[inline(always)]
    fn next_down(value: Self::Storage) -> Self::Storage {
        Self(R::next_down(value.0), R::next_down(value.1))
    }
}

impl<R: IntegerRegister> IntegerRegister for DoublePumpRegister<R>
where
    typenum::Double<R::Lanes>: Lanes,
{
    #[inline(always)]
    fn saturating_add(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        Self(R::saturating_add(lhs.0, rhs.0), R::saturating_add(lhs.1, rhs.1))
    }

    #[inline(always)]
    fn saturating_sub(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        Self(R::saturating_sub(lhs.0, rhs.0), R::saturating_sub(lhs.1, rhs.1))
    }

    #[inline(always)]
    fn wrapping_sum(value: Self::Storage) -> Self::Element {
        R::wrapping_sum(R::add(value.0, value.1))
    }

    #[inline(always)]
    fn wrapping_product(value: Self::Storage) -> Self::Element {
        R::wrapping_product(R::mul(value.0, value.1))
    }

    #[inline(always)]
    fn rol(value: Self::Storage, shift: u32) -> Self::Storage {
        Self(R::rol(value.0, shift), R::rol(value.1, shift))
    }

    #[inline(always)]
    fn ror(value: Self::Storage, shift: u32) -> Self::Storage {
        Self(R::ror(value.0, shift), R::ror(value.1, shift))
    }

    #[inline(always)]
    fn roli<const IMM8: i32>(value: Self::Storage) -> Self::Storage {
        Self(R::roli::<IMM8>(value.0), R::roli::<IMM8>(value.1))
    }

    #[inline(always)]
    fn rori<const IMM8: i32>(value: Self::Storage) -> Self::Storage {
        Self(R::rori::<IMM8>(value.0), R::rori::<IMM8>(value.1))
    }

    #[inline(always)]
    fn rolv(value: Self::Storage, shifts: impl Into<GenericArray<u32, Self::Lanes>>) -> Self::Storage {
        let [shift_lo, shift_hi] = Self::split_array(shifts.into());
        Self(R::rolv(value.0, shift_lo), R::rolv(value.1, shift_hi))
    }

    #[inline(always)]
    fn rorv(value: Self::Storage, shifts: impl Into<GenericArray<u32, Self::Lanes>>) -> Self::Storage {
        let [shift_lo, shift_hi] = Self::split_array(shifts.into());
        Self(R::rorv(value.0, shift_lo), R::rorv(value.1, shift_hi))
    }

    #[inline(always)]
    fn reverse_bits(value: Self::Storage) -> Self::Storage {
        Self(R::reverse_bits(value.0), R::reverse_bits(value.1))
    }

    #[inline(always)]
    fn count_ones(value: Self::Storage) -> Self::Storage {
        Self(R::count_ones(value.0), R::count_ones(value.1))
    }

    #[inline(always)]
    fn count_zeros(value: Self::Storage) -> Self::Storage {
        Self(R::count_zeros(value.0), R::count_zeros(value.1))
    }

    #[inline(always)]
    fn leading_zeros(value: Self::Storage) -> Self::Storage {
        Self(R::leading_zeros(value.0), R::leading_zeros(value.1))
    }

    #[inline(always)]
    fn leading_ones(value: Self::Storage) -> Self::Storage {
        Self(R::leading_ones(value.0), R::leading_ones(value.1))
    }

    #[inline(always)]
    fn trailing_ones(value: Self::Storage) -> Self::Storage {
        Self(R::trailing_ones(value.0), R::trailing_ones(value.1))
    }

    #[inline(always)]
    fn trailing_zeros(value: Self::Storage) -> Self::Storage {
        Self(R::trailing_zeros(value.0), R::trailing_zeros(value.1))
    }
}

impl<R: UnsignedIntegerRegister> UnsignedIntegerRegister for DoublePumpRegister<R>
where
    typenum::Double<R::Lanes>: Lanes,
{
    #[inline(always)]
    fn ilog2p1(value: Self::Storage) -> Self::Storage {
        Self(R::ilog2p1(value.0), R::ilog2p1(value.1))
    }

    #[inline(always)]
    fn next_power_of_two_m1(value: Self::Storage) -> Self::Storage {
        Self(R::next_power_of_two_m1(value.0), R::next_power_of_two_m1(value.1))
    }

    #[inline(always)]
    fn is_power_of_two(value: Self::Storage) -> Self::Storage {
        Self(R::is_power_of_two(value.0), R::is_power_of_two(value.1))
    }

    #[inline(always)]
    fn parity(value: Self::Storage) -> Self::Storage {
        Self(R::parity(value.0), R::parity(value.1))
    }
}

impl<R: SwizzleRegister> SwizzleRegister for DoublePumpRegister<R>
where
    typenum::Double<R::Lanes>: Lanes,
{
    #[inline(always)]
    fn permutev(value: Self::Storage, idxs: impl Into<GenericArray<u32, Self::Lanes>>) -> Self::Storage {
        let idxs = idxs.into();

        let [pidx_lo, pidx_hi] = {
            let mut idxs = idxs.clone();

            // lanes will always be a power of two, so subtracting one creates a mask
            let mask = <R::Lanes as Unsigned>::U32 - 1;

            // mask out all indices to be within the range of R
            idxs.iter_mut().for_each(|idx| *idx &= mask);

            Self::split_array(idxs)
        };

        let (blend_lo, blend_hi) = {
            let mut blends = Self::EMPTY; // mask register

            idxs.iter()
                .zip(Self::as_array_mut(&mut blends))
                .for_each(|(idx, blend)| {
                    // hopefully compiles to cmov or similar
                    *blend = if *idx < <R::Lanes as Unsigned>::U32 {
                        MaskElement::FALSY
                    } else {
                        MaskElement::TRUTHY
                    };
                });

            Self::split(blends)
        };

        let DoublePumpRegister(lo, hi) = value;

        let res_lo_from_lo: R::Storage = R::permutev(lo, pidx_lo.clone());
        let res_lo_from_hi: R::Storage = R::permutev(hi, pidx_lo);

        let res_hi_from_lo: R::Storage = R::permutev(lo, pidx_hi.clone());
        let res_hi_from_hi: R::Storage = R::permutev(hi, pidx_hi);

        let low: R::Storage = R::blendv(blend_lo, res_lo_from_lo, res_lo_from_hi);
        let high: R::Storage = R::blendv(blend_hi, res_hi_from_lo, res_hi_from_hi);

        Self(low, high)
    }

    // fn swizzle_i<const AIMM8: i32, const BIMM8: i32, const BLEND: i32>(
    //     _a: Self::Storage,
    //     _b: Self::Storage,
    // ) -> Self::Storage {
    //     unimplemented!()
    // }
}

impl<FROM: Register, INTO: CastRegister<FROM>> CastRegister<DoublePumpRegister<FROM>> for DoublePumpRegister<INTO>
where
    typenum::Double<FROM::Lanes>: Lanes,
    typenum::Double<INTO::Lanes>: Lanes,
{
    #[inline(always)]
    fn cast_from(value: <DoublePumpRegister<FROM> as Register>::Storage) -> Self::Storage {
        Self(INTO::cast_from(value.0), INTO::cast_from(value.1))
    }

    #[inline(always)]
    fn fast_cast_from(value: <DoublePumpRegister<FROM> as Register>::Storage) -> Self::Storage {
        Self(INTO::fast_cast_from(value.0), INTO::fast_cast_from(value.1))
    }
}

impl<FROM: Register, INTO: BitsRegister<FROM>> BitsRegister<DoublePumpRegister<FROM>> for DoublePumpRegister<INTO>
where
    typenum::Double<FROM::Lanes>: Lanes,
    typenum::Double<INTO::Lanes>: Lanes,
{
    #[inline(always)]
    fn from_bits(value: <DoublePumpRegister<FROM> as Register>::Storage) -> Self::Storage {
        Self(INTO::from_bits(value.0), INTO::from_bits(value.1))
    }
}

// TODO: Improve the swizzling here when some generic variant is available
impl<R: FloatRegister> LinAlg3Register for DoublePumpRegister<R>
where
    Self: FloatRegister<Lanes = typenum::U4> + SwizzleRegister,
{
    #[inline(always)]
    fn zero4(value: Self::Storage) -> Self::Storage {
        Self::insert::<3>(value, num_traits::Zero::zero())
    }

    #[inline(always)]
    fn one4(value: Self::Storage) -> Self::Storage {
        Self::insert::<3>(value, num_traits::One::one())
    }

    #[inline(always)]
    fn min_element3(value: Self::Storage) -> Self::Element {
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
    fn max_element3(value: Self::Storage) -> Self::Element {
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
    fn sum_elements3(value: Self::Storage) -> Self::Element {
        let a = Self::extract::<0>(value);
        let b = Self::extract::<1>(value);
        let c = Self::extract::<2>(value);

        a + b + c
    }

    #[inline(always)]
    fn prod_elements3(value: Self::Storage) -> Self::Element {
        let a = Self::extract::<0>(value);
        let b = Self::extract::<1>(value);
        let c = Self::extract::<2>(value);

        a * b * c
    }
}
