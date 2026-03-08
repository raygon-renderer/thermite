use core::mem::MaybeUninit;
use core::ops::Mul;

use generic_array::functional::FunctionalSequence;
use generic_array::typenum::{self, Const, Prod, ToUInt, Unsigned};

use crate::Vector;

use super::*;

// generates array_zip2, array_zip3, array_zip4, array_zip5, etc., to combine many arrays using a provided function
macro_rules! decl_array_zips {
    ($($count:literal => ($($part:ident,)+)),* $(,)?) => {paste::paste! {$(
        #[inline(always)]
        fn [<array_zip $count>]<$($part: Copy),+, U, F, const N: usize>($([<$part:lower>]: [$part; N]),+, mut f: F) -> [U; N]
        where F: FnMut($($part),+) -> U {
            let mut result: [MaybeUninit<U>; N] = unsafe { MaybeUninit::uninit().assume_init() };
            for i in 0..N { result[i].write(f( $( [<$part:lower>][i] ),+)); }
            unsafe { MaybeUninit::assume_init(result.into()) }
        }
    )*}};
}

decl_array_zips! {
    2 => (A, B,),
    3 => (A, B, C,),
    4 => (A, B, C, D,),
    5 => (A, B, C, D, E,),
    6 => (A, B, C, D, E, G,), // skip F since the function is bound to F
}

#[inline(always)]
fn array_unzip2<T: Copy, A: Copy, B: Copy, F, const N: usize>(x: [T; N], mut f: F) -> ([A; N], [B; N])
where
    F: FnMut(T) -> (A, B),
{
    let mut a: [MaybeUninit<A>; N] = unsafe { MaybeUninit::uninit().assume_init() };
    let mut b: [MaybeUninit<B>; N] = unsafe { MaybeUninit::uninit().assume_init() };

    for i in 0..N {
        let (aa, bb) = f(x[i]);
        a[i].write(aa);
        b[i].write(bb);
    }

    unsafe { (MaybeUninit::assume_init(a.into()), MaybeUninit::assume_init(b.into())) }
}

#[repr(transparent)]
pub struct ArrayRegister<R: CoreRegister, const N: usize>(pub [Storage<R>; N]);

impl<R: CoreRegister, const N: usize> Clone for ArrayRegister<R, N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: CoreRegister, const N: usize> Copy for ArrayRegister<R, N> {}

impl<R: CoreRegister, const N: usize> ArrayRegister<R, N> {
    #[inline(always)]
    pub const fn idx(lane: usize) -> (usize, usize) {
        (lane / R::Lanes::USIZE, lane % R::Lanes::USIZE)
    }
}

impl<R: CoreRegister, const N: usize> CoreRegister for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: Mul<R::Lanes, Output: Lanes>>,
{
    type Lanes = Prod<typenum::U<N>, R::Lanes>;
    type Storage = Self;
    type Mask = ArrayRegister<R::Mask, N>;

    const IS_EMULATED: bool = true;
    const ISA: InstructionSet = R::ISA;

    const EMPTY: Storage<Self> = Self([R::EMPTY; N]);

    fn blendv(mask: Storage<Self::Mask>, on_false: Storage<Self>, on_true: Storage<Self>) -> Storage<Self> {
        todo!()
    }

    fn z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        todo!()
    }

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        todo!()
    }
}

#[rustfmt::skip] #[thermite_macros::array_impl]
impl<R: BitwiseRegister, const N: usize> BitwiseRegister for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: Mul<R::Lanes, Output: Lanes>>,
{
    #[conditional] fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn not(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn ternlog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn bilog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {}
}

impl<FROM: CoreRegister, INTO: CastMaskRegister<FROM, Lanes = FROM::Lanes>, const N: usize>
    CastMaskRegister<ArrayRegister<FROM, N>> for ArrayRegister<INTO, N>
where
    Const<N>: ToUInt<Output: Mul<FROM::Lanes, Output: Lanes>>,
{
    #[inline(always)]
    fn mask_from(value: Storage<ArrayRegister<FROM, N>>) -> Storage<Self> {
        Self(value.0.map(INTO::mask_from))
    }
}

impl<FROM: CoreRegister, INTO: CastRegister<FROM, Lanes = FROM::Lanes>, const N: usize>
    CastRegister<ArrayRegister<FROM, N>> for ArrayRegister<INTO, N>
where
    Const<N>: ToUInt<Output: Mul<FROM::Lanes, Output: Lanes>>,
{
    #[inline(always)]
    fn cast_from(value: Storage<ArrayRegister<FROM, N>>) -> Storage<Self> {
        Self(value.0.map(INTO::cast_from))
    }

    #[inline(always)]
    fn fast_cast_from(value: Storage<ArrayRegister<FROM, N>>) -> Storage<Self> {
        Self(value.0.map(INTO::fast_cast_from))
    }
}

impl<FROM: CoreRegister, INTO: BitCastRegister<FROM, Lanes = FROM::Lanes>, const N: usize>
    BitCastRegister<ArrayRegister<FROM, N>> for ArrayRegister<INTO, N>
where
    Const<N>: ToUInt<Output: Mul<FROM::Lanes, Output: Lanes>>,
{
    #[inline(always)]
    fn from_bits(value: Storage<ArrayRegister<FROM, N>>) -> Storage<Self> {
        Self(value.0.map(INTO::from_bits))
    }
}

#[rustfmt::skip] #[thermite_macros::array_impl]
impl<R: MaskRegister, const N: usize> MaskRegister for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: Mul<R::Lanes, Output: Lanes>>,
{
    const TRUTHY: Storage<Self> = Self([R::TRUTHY; N]);
    const FALSY: Storage<Self> = Self([R::FALSY; N]);

    fn set(mut mask: Storage<Self>, lane: usize, value: bool) -> Storage<Self> {
        let (idx, lane_in_reg) = Self::idx(lane);
        mask.0[idx] = R::set(mask.0[idx], lane_in_reg, value);
        mask
    }

    fn test(mask: Storage<Self>, lane: usize) -> bool {
        let (idx, lane_in_reg) = Self::idx(lane);
        R::test(mask.0[idx], lane_in_reg)
    }

    fn all(mut value: Storage<Self>) -> bool {
        // O(log2(N))) reduction of bitwise AND across all registers
        crate::math::algorithms::reduce_in_place(&mut value.0, R::bitand);
        R::all(value.0[0])
    }

    fn any(mut value: Storage<Self>) -> bool {
        crate::math::algorithms::reduce_in_place(&mut value.0, R::bitor);
        R::any(value.0[0])
    }

    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        if const { Self::Lanes::USIZE <= 64 } {
            let mut bitmask = 0u64;
            for i in 0..N {
                if let Some(reg_bitmask) = R::native_bitmask(value.0[i]) {
                    bitmask |= reg_bitmask << (i * R::Lanes::USIZE);
                } else {
                    return None;
                }
            }
            Some(bitmask)
        } else {
            None
        }
    }

    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        todo!()
    }
}

#[rustfmt::skip] #[thermite_macros::array_impl]
impl<R: Register, const N: usize> Register for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: Mul<R::Lanes, Output: Lanes>>,
{
    type Element = R::Element;
    type Signed = ArrayRegister<R::Signed, N>;
    type Unsigned = ArrayRegister<R::Unsigned, N>;

    const HAS_EQUAL_SIZE_MASK: bool = R::HAS_EQUAL_SIZE_MASK;

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        ArrayRegister(mask.0.map(R::from_mask))
    }

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        ArrayRegister(value.0.map(R::into_mask))
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        ArrayRegister(value.0.map(R::msb_to_mask))
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        let ptr = value.as_ptr() as *const GenericArray<R::Element, R::Lanes>;

        let mut res = [R::EMPTY; N];
        for i in 0..N {
            res[i] = unsafe { R::new(ptr.add(i).read()) };
        }
        Self(res)
    }

    fn single(value: Self::Element) -> Storage<Self> {
        let mut res = [R::EMPTY; N];
        res[0] = R::single(value);
        Self(res)
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        Self([R::splat(value); N])
    }

    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        todo!()
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        todo!()
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        Self(value.0.map(R::swap_bytes))
    }
}

#[rustfmt::skip]
impl<R: PartialOrdRegister, const N: usize> PartialOrdRegister for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: Mul<R::Lanes, Output: Lanes>>,
{
    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> { ArrayRegister(array_zip2(lhs.0, rhs.0, R::eq)) }
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> { ArrayRegister(array_zip2(lhs.0, rhs.0, R::gt)) }
    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> { ArrayRegister(array_zip2(lhs.0, rhs.0, R::ge)) }
    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> { ArrayRegister(array_zip2(lhs.0, rhs.0, R::lt)) }
    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> { ArrayRegister(array_zip2(lhs.0, rhs.0, R::le)) }
    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> { ArrayRegister(array_zip2(lhs.0, rhs.0, R::ne)) }
}

#[rustfmt::skip] #[thermite_macros::array_impl]
impl<R: NumericRegister, const N: usize> NumericRegister for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: Mul<R::Lanes, Output: Lanes>>,
{
    const ZERO: Storage<Self> = Self([R::ZERO; N]);
    const ONE: Storage<Self> = Self([R::ONE; N]);
    const TWO: Storage<Self> = Self([R::TWO; N]);

    const MIN: Storage<Self> = Self([R::MIN; N]);
    const MAX: Storage<Self> = Self([R::MAX; N]);

    #[conditional] fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Self(array_zip2(lhs.0, rhs.0, R::add)) }
    #[conditional] fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Self(array_zip2(lhs.0, rhs.0, R::sub)) }
    #[conditional] fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Self(array_zip2(lhs.0, rhs.0, R::mul)) }
    #[conditional] fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Self(array_zip2(lhs.0, rhs.0, R::div)) }
    #[conditional] fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Self(array_zip2(lhs.0, rhs.0, R::rem)) }
    #[conditional] fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Self(array_zip2(lhs.0, rhs.0, R::min)) }
    #[conditional] fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Self(array_zip2(lhs.0, rhs.0, R::max)) }
    #[conditional] fn square(lhs: Storage<Self>) -> Storage<Self> {}

    fn min_element(mut value: Storage<Self>) -> Self::Element {
        crate::math::algorithms::reduce_in_place(&mut value.0, R::min);
        R::min_element(value.0[0])
    }

    fn max_element(mut value: Storage<Self>) -> Self::Element {
        crate::math::algorithms::reduce_in_place(&mut value.0, R::max);
        R::max_element(value.0[0])
    }

    fn sum_elements(mut value: Storage<Self>) -> Self::Element {
        crate::math::algorithms::reduce_in_place(&mut value.0, R::add);
        R::sum_elements(value.0[0])
    }

    fn prod_elements(mut value: Storage<Self>) -> Self::Element {
        crate::math::algorithms::reduce_in_place(&mut value.0, R::mul);
        R::prod_elements(value.0[0])
    }

    fn offset() -> Storage<Self> {
        let mut offset = R::offset();
        for _ in 0..N {
            offset = R::add(offset, R::offset());
        }
        Self([offset; N])
    }

    fn indexed() -> Storage<Self> {
        let mut result = [R::ZERO; N];
        let mut indexed = R::indexed();
        result[0] = indexed;

        for i in 1..N {
            result[i] = R::add(result[i - 1], indexed);
        }

        Self(result)
    }
}

#[rustfmt::skip] #[thermite_macros::array_impl]
impl<R: SignedRegister, const N: usize> SignedRegister for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: Mul<R::Lanes, Output: Lanes>>,
{
    const NEG_ONE: Storage<Self> = Self([R::NEG_ONE; N]);
    const MIN_POSITIVE: Storage<Self> = Self([R::MIN_POSITIVE; N]);

    #[conditional] fn neg(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn abs(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn copysign(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}

    fn signum(value: Storage<Self>) -> Storage<Self> {}
    fn is_negative(value: Storage<Self>) -> Storage<Self::Mask> {}
    fn is_positive(value: Storage<Self>) -> Storage<Self::Mask> {}
    fn select_negative(value: Storage<Self>, on_neg: Storage<Self>, on_pos: Storage<Self>) -> Storage<Self> {}
}

#[rustfmt::skip] #[thermite_macros::array_impl]
impl<R: BitshiftRegister, const N: usize> BitshiftRegister for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: Mul<R::Lanes, Output: Lanes>>,
{
    const HAS_WIDE_BYTE_SHIFTS: bool = false;
    const HAS_TRUE_SHIFTV: bool = R::HAS_TRUE_SHIFTV;

    #[conditional] fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {}
    #[conditional] fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {}
    #[conditional] fn shrv(mut value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {}
    #[conditional] fn shlv(mut value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {}
    #[conditional] fn rol(value: Storage<Self>, shift: u32) -> Storage<Self> {}
    #[conditional] fn ror(value: Storage<Self>, shift: u32) -> Storage<Self> {}
    #[conditional] fn roli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn rori<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn rorv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {}
    #[conditional] fn rolv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {}
    #[conditional] fn reverse_bits(mut value: Storage<Self>) -> Storage<Self> {}
}

#[rustfmt::skip] #[thermite_macros::array_impl]
impl<R: IntegerRegister, const N: usize> IntegerRegister for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: Mul<R::Lanes, Output: Lanes>>,
{
    #[conditional] fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}

    fn wrapping_product(mut value: Storage<Self>) -> Self::Element {
        crate::math::algorithms::reduce_in_place(&mut value.0, R::mullo);
        R::wrapping_product(value.0[0])
    }

    fn wrapping_sum(mut value: Storage<Self>) -> Self::Element {
        crate::math::algorithms::reduce_in_place(&mut value.0, R::add);
        R::wrapping_sum(value.0[0])
    }

    #[conditional] fn div_branched(value: Storage<Self>, divider: Divider<Self::Element>) -> Storage<Self> {}
    #[conditional] fn div_branchfree(value: Storage<Self>, divider: BranchfreeDivider<Self::Element>) -> Storage<Self> {}

    fn divv_branchfree(value: Storage<Self>, dividers: VectorDivider<Self>) -> Storage<Self> {
        let multipliers = dividers.multipliers.0;
        let shifts = dividers.shifts.0;

        Self(array_zip3(value.0, multipliers.0, shifts.0, |value, multiplier, shift| {
            R::divv_branchfree(value, VectorDivider { multipliers: Vector(multiplier), shifts: Vector(shift) })
        }))
    }

    fn divv_branchfree_c(mask: Storage<Self::Mask>, value: Storage<Self>, dividers: VectorDivider<Self>) -> Storage<Self> {
        let multipliers = dividers.multipliers.0;
        let shifts = dividers.shifts.0;

        Self(array_zip4(mask.0, value.0, multipliers.0, shifts.0, |mask_reg, value_reg, multiplier, shift| {
            R::divv_branchfree_c(mask_reg, value_reg, VectorDivider { multipliers: Vector(multiplier), shifts: Vector(shift) })
        }))
    }

    fn divv_branchfree_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>, dividers:VectorDivider<Self>) -> Storage<Self> {
        let multipliers = dividers.multipliers.0;
        let shifts = dividers.shifts.0;

        Self(array_zip5(src.0, mask.0, value.0, multipliers.0, shifts.0, |src_reg, mask_reg, value_reg, multiplier, shift| {
            R::divv_branchfree_m(src_reg, mask_reg, value_reg, VectorDivider { multipliers: Vector(multiplier), shifts: Vector(shift) })
        }))
    }

    fn divv_branchfree_z(mask: Storage<Self::Mask>, value: Storage<Self>, dividers: VectorDivider<Self>) -> Storage<Self> {
        let multipliers = dividers.multipliers.0;
        let shifts = dividers.shifts.0;

        Self(array_zip4(mask.0, value.0, multipliers.0, shifts.0, |mask_reg, value_reg, multiplier, shift| {
            R::divv_branchfree_z(mask_reg, value_reg, VectorDivider { multipliers: Vector(multiplier), shifts: Vector(shift) })
        }))
    }

    const HAS_HARDWARE_POPCNT: bool = R::HAS_HARDWARE_POPCNT;

    #[conditional] fn count_ones(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn count_zeros(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn leading_zeros(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn leading_ones(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn trailing_ones(value: Storage<Self>) -> Storage<Self> {}
}

#[rustfmt::skip] #[thermite_macros::array_impl]
impl<R: UnsignedIntegerRegister, const N: usize> UnsignedIntegerRegister for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: Mul<R::Lanes, Output: Lanes>>,
{
    #[conditional] fn ilog2p1(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn next_power_of_two_m1(mut value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn parity(mut value: Storage<Self>) -> Storage<Self> {}
    fn is_power_of_two(value: Storage<Self>) -> Storage<Self::Mask> {}
}

#[rustfmt::skip] #[thermite_macros::array_impl]
impl<R: SignedIntegerRegister, const N: usize> SignedIntegerRegister for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: Mul<R::Lanes, Output: Lanes>>,
{
    #[conditional] fn sra(value: Storage<Self>, shift: u32) -> Storage<Self> {}
    #[conditional] fn srai<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn srav(mut value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {}
}

#[rustfmt::skip] #[thermite_macros::array_impl]
impl<R: FloatRegister, const N: usize> FloatRegister for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: Mul<R::Lanes, Output: Lanes>>,
{
    const NAN: Storage<Self> = Self([R::NAN; N]);
    const INFINITY: Storage<Self> = Self([R::INFINITY; N]);
    const NEG_INFINITY: Storage<Self> = Self([R::NEG_INFINITY; N]);
    const EPSILON: Storage<Self> = Self([R::EPSILON; N]);
    const EXP_MASK: Storage<Self::Bits> = ArrayRegister([R::EXP_MASK; N]);
    const HALF: Storage<Self> = Self([R::HALF; N]);
    const NEG_ZERO: Storage<Self> = Self([R::NEG_ZERO; N]);

    type Bits = ArrayRegister<R::Bits, N>;
    type SignedBits = ArrayRegister<R::SignedBits, N>;
    type ExtendedPrecision = ArrayRegister<R::ExtendedPrecision, N>;

    const HAS_APPROX_RCP: bool = R::HAS_APPROX_RCP;
    const HAS_APPROX_RSQRT: bool = R::HAS_APPROX_RSQRT;
    const HAS_TRUE_FMA: bool = R::HAS_TRUE_FMA;

    const NATIVE_CAP: NativeCapability = R::NATIVE_CAP;

    unsafe fn block_autovectorization(value: &mut Storage<Self>) {
        unsafe { R::block_autovectorization(&mut value.0[0]) };
    }

    unsafe fn native_ldexp(value: Storage<Self>, exp: Storage<Self::SignedBits>) -> Storage<Self> {
        Self(array_zip2(value.0, exp.0, |v, e| unsafe { R::native_ldexp(v, e) }))
    }

    unsafe fn native_frexp(value: Storage<Self>) -> (Storage<Self>, Storage<Self::SignedBits>) {
        let (v, e) = array_unzip2(value.0, |v| unsafe { R::native_frexp(v) });

        (Self(v), ArrayRegister(e))
    }

    unsafe fn native_sin_cos(value: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        let (s, c) = array_unzip2(value.0, |v| unsafe { R::native_sin_cos(v) });

        (Self(s), Self(c))
    }

    unsafe fn native_sin(value: Storage<Self>) -> Storage<Self> {}
    unsafe fn native_cos(value: Storage<Self>) -> Storage<Self> {}
    unsafe fn native_exp(value: Storage<Self>) -> Storage<Self> {}
    unsafe fn native_exp2(value: Storage<Self>) -> Storage<Self> {}
    unsafe fn native_ln(value: Storage<Self>) -> Storage<Self> {}
    unsafe fn native_log2(value: Storage<Self>) -> Storage<Self> {}
    unsafe fn native_powf(base: Storage<Self>, exp: Storage<Self>) -> Storage<Self> {}
    unsafe fn native_tan(value: Storage<Self>) -> Storage<Self> {}

    fn total_order(value: Storage<Self>) -> Storage<Self::SignedBits> {}
    fn is_nan(value: Storage<Self>) -> Storage<Self::Mask> {}
    fn is_finite(value: Storage<Self>) -> Storage<Self::Mask> {}
    fn is_infinite(value: Storage<Self>) -> Storage<Self::Mask> {}
    fn is_normal(value: Storage<Self>) -> Storage<Self::Mask> {}
    fn is_subnormal(value: Storage<Self>) -> Storage<Self::Mask> {}
    fn is_zero_or_subnormal(value: Storage<Self>) -> Storage<Self::Mask> {}

    #[conditional] fn mul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn mul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn nmul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn nmul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn mul_adde(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn mul_sube(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn nmul_adde(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn nmul_sube(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {}

    #[conditional] fn sqrt(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn rsqrt(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn rcp(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn floor(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn ceil(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn round(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn trunc(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn fract(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn signed_zero(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn mul_sign(value: Storage<Self>, sign: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn next_down(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn next_up(value: Storage<Self>) -> Storage<Self> {}
}

#[rustfmt::skip] #[thermite_macros::array_impl]
impl<IDX, R: IndexableRegister<IDX>, const N: usize> IndexableRegister<ArrayRegister<IDX, N>> for ArrayRegister<R, N>
where
    IDX: UnsignedIntegerRegister<Lanes = R::Lanes>,
    Const<N>: ToUInt<Output: Mul<R::Lanes, Output: Lanes>>,
{
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<ArrayRegister<IDX, N>>) -> Storage<Self> {
        Self(indices.0.map(|idx| unsafe { R::gather(ptr, idx) }))
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<ArrayRegister<IDX, N>>,
    ) -> Storage<Self> {
        Self(array_zip3(src.0, mask.0, indices.0, |src_reg, mask_reg, idx| unsafe {
            R::gather_m(src_reg, mask_reg, ptr, idx)
        }))
    }

    unsafe fn gather_z(
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<ArrayRegister<IDX, N>>,
    ) -> Storage<Self> {
        Self(array_zip2(mask.0, indices.0, |mask_reg, idx| unsafe {
            R::gather_z(mask_reg, ptr, idx)
        }))
    }

    unsafe fn scatter(value: Storage<Self>, ptr: *mut Self::Element, indices: Storage<ArrayRegister<IDX, N>>) {
        for i in 0..N {
            unsafe { R::scatter(value.0[i], ptr, indices.0[i]) };
        }
    }

    unsafe fn scatter_m(
        value: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *mut Self::Element,
        indices: Storage<ArrayRegister<IDX, N>>,
    ) {
        for i in 0..N {
            unsafe { R::scatter_m(value.0[i], mask.0[i], ptr, indices.0[i]) };
        }
    }
}
