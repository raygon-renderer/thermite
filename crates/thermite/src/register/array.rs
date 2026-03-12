use core::mem::MaybeUninit;
use core::ops::Mul;

use generic_array::ArrayLength;
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

impl<R: CoreRegister, const N: usize> core::fmt::Debug for ArrayRegister<R, N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let arr = self.0;

        f.debug_tuple("ArrayRegister").field(&arr).finish()
    }
}

impl<R: CoreRegister, const N: usize> ArrayRegister<R, N> {
    #[inline(always)]
    pub const fn idx(lane: usize) -> (usize, usize) {
        (lane / R::Lanes::USIZE, lane % R::Lanes::USIZE)
    }

    #[inline(always)]
    pub const fn into_array(self) -> [Storage<R>; N] {
        self.0
    }

    #[inline(always)]
    pub const fn from_array(arr: [Storage<R>; N]) -> Self {
        Self(arr)
    }
}

#[rustfmt::skip] #[thermite_macros::array_impl]
impl<R: CoreRegister, const N: usize> CoreRegister for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
{
    type Lanes = Prod<typenum::U<N>, R::Lanes>;
    type Storage = Self;
    type Mask = ArrayRegister<R::Mask, N>;

    const IS_EMULATED: bool = true;
    const ISA: InstructionSet = R::ISA;

    const EMPTY: Storage<Self> = Self([R::EMPTY; N]);

    fn blendv(mask: Storage<Self::Mask>, on_false: Storage<Self>, on_true: Storage<Self>) -> Storage<Self> {}
    fn z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {}
    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {}

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        panic!("ArrayRegister does not support zeroupper operations");
    }
}

#[rustfmt::skip] #[thermite_macros::array_impl]
impl<R: BitwiseRegister, const N: usize> BitwiseRegister for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
{
    #[conditional] fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn not(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn ternlog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn bilog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {}
}

#[rustfmt::skip] #[thermite_macros::array_impl]
impl<R: MaskRegister, const N: usize> MaskRegister for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
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

    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        let ptr = value.as_ptr() as *const GenericArray<bool, R::Lanes>;
        let mut res = [R::FALSY; N];
        for (i, r) in res.iter_mut().enumerate() {
            *r = R::new_mask(unsafe { ptr.add(i).read() });
        }
        Self(res)
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
        for i in 0..N {
            R::fill_bitmask(value.0[i], &mut view[i * R::Lanes::USIZE..(i + 1) * R::Lanes::USIZE]);
        }
    }
}

#[rustfmt::skip] #[thermite_macros::array_impl]
impl<R: Register, const N: usize> Register for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
{
    type Element = R::Element;
    type Signed = ArrayRegister<R::Signed, N>;
    type Unsigned = ArrayRegister<R::Unsigned, N>;

    const HAS_EQUAL_SIZE_MASK: bool = R::HAS_EQUAL_SIZE_MASK;

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {}
    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {}
    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {}
    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {}

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        let ptr = value.as_ptr() as *const GenericArray<R::Element, R::Lanes>;

        let mut res = [R::EMPTY; N];

        for (i, r) in res.iter_mut().enumerate() {
            *r = unsafe { R::new(ptr.add(i).read()) };
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

    fn splat_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {}
    fn splat_z(mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {}

    fn broadcast<const I: usize>(mut value: Storage<Self>) -> Storage<Self> {
        let e = R::splat(Self::as_array(&value)[I]);
        value.0.fill(e);
        value
    }

    fn broadcast_c<const I: usize>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let e = R::splat(Self::as_array(&value)[I]);
        Self(value.0.map(|reg| R::blendv(mask.0[0], reg, e)))
    }

    fn broadcast_m<const I: usize>(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let e = R::splat(Self::as_array(&value)[I]);
        Self(src.0.map(|reg| R::blendv(mask.0[0], reg, e)))
    }

    fn broadcast_z<const I: usize>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let e = R::splat(Self::as_array(&value)[I]);
        Self(mask.0.map(|mask_reg| R::blendv(mask_reg, R::EMPTY, e)))
    }

    fn broadcastv(mut value: Storage<Self>, idx: usize) -> Storage<Self> {
        let e = R::splat(Self::as_array(&value)[idx]);
        value.0.fill(e);
        value
    }

    fn broadcastv_c(mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        let e = R::splat(Self::as_array(&value)[idx]);
        Self(value.0.map(|reg| R::blendv(mask.0[0], reg, e)))
    }

    fn broadcastv_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        let e = R::splat(Self::as_array(&value)[idx]);
        Self(src.0.map(|reg| R::blendv(mask.0[0], reg, e)))
    }

    fn broadcastv_z(mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        let e = R::splat(Self::as_array(&value)[idx]);
        Self(mask.0.map(|mask_reg| R::blendv(mask_reg, R::EMPTY, e)))
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        let mut res = [R::EMPTY; N];

        for (i, r) in res.iter_mut().enumerate() {
            *r = unsafe { R::load(ptr.add(i * R::Lanes::USIZE)) };
        }

        Self(res)
    }

    unsafe fn load_m(src: Storage<Self>, mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        let mut res = [R::EMPTY; N];

        for (i, r) in res.iter_mut().enumerate() {
            *r = unsafe { R::load_m(src.0[i], mask.0[i], ptr.add(i * R::Lanes::USIZE)) };
        }

        Self(res)
    }

    unsafe fn load_z(mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        let mut res = [R::EMPTY; N];

        for (i, r) in res.iter_mut().enumerate() {
            *r = unsafe { R::load_z(mask.0[i], ptr.add(i * R::Lanes::USIZE)) };
        }

        Self(res)
    }

    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        let mut res = [R::EMPTY; N];

        for (i, r) in res.iter_mut().enumerate() {
            *r = unsafe { R::load_unaligned(ptr.add(i * R::Lanes::USIZE)) };
        }

        Self(res)
    }

    unsafe fn load_stream(ptr: *const Self::Element) -> Storage<Self> {
        let mut res = [R::EMPTY; N];

        for (i, r) in res.iter_mut().enumerate() {
            *r = unsafe { R::load_stream(ptr.add(i * R::Lanes::USIZE)) };
        }

        Self(res)
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        for i in 0..N {
            unsafe { R::store(ptr.add(i * R::Lanes::USIZE), value.0[i]) };
        }
    }

    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Storage<Self>) {
        for i in 0..N {
            unsafe { R::store_unaligned(ptr.add(i * R::Lanes::USIZE), value.0[i]) };
        }
    }

    unsafe fn store_stream(ptr: *mut Self::Element, value: Storage<Self>) {
        for i in 0..N {
            unsafe { R::store_stream(ptr.add(i * R::Lanes::USIZE), value.0[i]) };
        }
    }

    unsafe fn store_masked(ptr: *mut Self::Element, mask: Storage<Self::Mask>, value: Storage<Self>) {
        for (i, r) in value.0.iter().enumerate() {
            unsafe { R::store_masked(ptr.add(i * R::Lanes::USIZE), mask.0[i], *r) };
        }
    }

    fn reverse(mut value: Storage<Self>) -> Storage<Self> {
        value.0.reverse();
        for r in &mut value.0 {
            *r = R::reverse(*r);
        }

        value
    }

    fn reverse_c(mask: Storage<Self::Mask>, mut value: Storage<Self>) -> Storage<Self> {
        value.0.reverse();
        for (r, mask_reg) in value.0.iter_mut().zip(mask.0.iter()) {
            *r = R::reverse_c(*mask_reg, *r);
        }

        value
    }

    fn reverse_m(src: Storage<Self>, mask: Storage<Self::Mask>, mut value: Storage<Self>) -> Storage<Self> {
        value.0.reverse();
        for (r, mask_reg) in value.0.iter_mut().zip(mask.0.iter()) {
            *r = R::reverse_m(src.0[0], *mask_reg, *r);
        }

        value
    }

    fn reverse_z(mask: Storage<Self::Mask>, mut value: Storage<Self>) -> Storage<Self> {
        value.0.reverse();
        for (r, mask_reg) in value.0.iter_mut().zip(mask.0.iter()) {
            *r = R::reverse_z(*mask_reg, *r);
        }

        value
    }

    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        todo!()
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        todo!()
    }

    #[conditional] fn swap_bytes(value: Storage<Self>) -> Storage<Self> {}
}

#[rustfmt::skip]
impl<R: PartialOrdRegister, const N: usize> PartialOrdRegister for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
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
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
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
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
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
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
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
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
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
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
{
    #[conditional] fn ilog2p1(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn next_power_of_two_m1(mut value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn parity(mut value: Storage<Self>) -> Storage<Self> {}
    fn is_power_of_two(value: Storage<Self>) -> Storage<Self::Mask> {}
}

#[rustfmt::skip] #[thermite_macros::array_impl]
impl<R: SignedIntegerRegister, const N: usize> SignedIntegerRegister for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
{
    #[conditional] fn sra(value: Storage<Self>, shift: u32) -> Storage<Self> {}
    #[conditional] fn srai<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn srav(mut value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {}
}

#[rustfmt::skip] #[thermite_macros::array_impl]
impl<R: FloatRegister, const N: usize> FloatRegister for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
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
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
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

/// ArrayRegister implements ExtendRegister and ConcatRegister in the most non-strict way possible, not requiring
/// either any specific relationship between N and M, or even that the total number of lanes is the same.
/// It just copies as many lanes as it can, and fills the rest with empty registers.
impl<R: CoreRegister, const N: usize, const M: usize> ExtendRegister<ArrayRegister<R, N>> for ArrayRegister<R, M>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
    Const<M>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
{
    #[inline(always)]
    fn extend(value: Storage<ArrayRegister<R, N>>) -> Storage<Self> {
        let mut result = [R::EMPTY; M];
        let min = const { if N < M { N } else { M } };

        result.copy_from_slice(&value.0[..min]);

        Self(result)
    }

    #[inline(always)]
    fn narrow(value: Storage<Self>) -> Storage<ArrayRegister<R, N>> {
        let mut result = [R::EMPTY; N];
        let min = const { if N < M { N } else { M } };

        result.copy_from_slice(&value.0[..min]);

        ArrayRegister(result)
    }
}

impl<R: CoreRegister, const N: usize> ExtendRegister<R> for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
{
    #[inline(always)]
    fn extend(value: Storage<R>) -> Storage<Self> {
        let mut result = [R::EMPTY; N];
        result[0] = value;
        Self(result)
    }

    #[inline(always)]
    fn narrow(value: Storage<Self>) -> Storage<R> {
        value.0[0]
    }
}

/// ArrayRegister implements ExtendRegister and ConcatRegister in the most non-strict way possible, not requiring
/// either any specific relationship between N and M, or even that the total number of lanes is the same.
/// It just copies as many lanes as it can, and fills the rest with empty registers.
impl<R: CoreRegister, const N: usize, const M: usize> ConcatRegister<ArrayRegister<R, N>> for ArrayRegister<R, M>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
    Const<M>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
{
    #[inline(always)]
    fn concat(lo: Storage<ArrayRegister<R, N>>, hi: Storage<ArrayRegister<R, N>>) -> Storage<Self> {
        let mut result = [R::EMPTY; M];
        let min = const { if N < M { N } else { M } };

        result[..min].copy_from_slice(&lo.0[..min]);
        if min < M {
            result[..(M - min)].copy_from_slice(&hi.0[..(M - min)]);
        }

        Self(result)
    }

    #[inline(always)]
    fn split(value: Storage<Self>) -> (Storage<ArrayRegister<R, N>>, Storage<ArrayRegister<R, N>>) {
        let mut lo = [R::EMPTY; N];
        let mut hi = [R::EMPTY; N];
        let min = const { if N < M { N } else { M } };

        lo.copy_from_slice(&value.0[..min]);
        if min < M {
            hi.copy_from_slice(&value.0[..(M - min)]);
        }

        (ArrayRegister(lo), ArrayRegister(hi))
    }
}

impl<R: CoreRegister> ConcatRegister<R> for ArrayRegister<R, 2>
where
    typenum::U2: Mul<R::Lanes, Output: Lanes>,
{
    #[inline(always)]
    fn concat(lo: Storage<R>, hi: Storage<R>) -> Storage<Self> {
        Self([lo, hi])
    }

    #[inline(always)]
    fn split(value: Storage<Self>) -> (Storage<R>, Storage<R>) {
        (value.0[0], value.0[1])
    }
}

pub trait PrecomputedSwizzleIndices<TotalLanes: ArrayLength, Lanes: ArrayLength, Chunks: ArrayLength>:
    SwizzleIndices<TotalLanes>
{
    const PRECOMPUTED: GenericArray<PrecomputedChunk<Lanes, Chunks>, Chunks>;
}

// --- Precomputed Data Structures ---

/// Holds the exact routing instructions for a SINGLE output chunk.
pub struct PrecomputedChunk<Lanes: ArrayLength, Chunks: ArrayLength> {
    /// The local indices to pass to the hardware `permutev`
    pub indices: GenericArray<u32, Lanes>,

    /// Flags to instantly skip input chunks if they aren't needed.
    pub has_any_lo: GenericArray<bool, Chunks>,
    pub has_any_hi: GenericArray<bool, Chunks>,

    /// The exact boolean blend masks for each input chunk.
    pub blend_mask_lo: GenericArray<GenericArray<bool, Lanes>, Chunks>,
    pub blend_mask_hi: GenericArray<GenericArray<bool, Lanes>, Chunks>,
}

impl<Lanes: ArrayLength, Chunks: ArrayLength> PrecomputedChunk<Lanes, Chunks> {
    #[inline(always)]
    pub const fn has_any(&self, idx: usize) -> bool {
        if idx < Chunks::USIZE {
            self.has_any_lo.as_slice()[idx]
        } else {
            self.has_any_hi.as_slice()[idx - Chunks::USIZE]
        }
    }

    #[inline(always)]
    pub const fn blend_mask(&self, idx: usize) -> &GenericArray<bool, Lanes> {
        if idx < Chunks::USIZE {
            &self.blend_mask_lo.as_slice()[idx]
        } else {
            &self.blend_mask_hi.as_slice()[idx - Chunks::USIZE]
        }
    }
}

// --- Blanket Implementation ---
// This executes the O(N^2) routing algorithm entirely at compile time for ANY given index array.

impl<T, TotalLanes, Lanes, Chunks> PrecomputedSwizzleIndices<TotalLanes, Lanes, Chunks> for T
where
    T: SwizzleIndices<TotalLanes>,
    TotalLanes: ArrayLength,
    Lanes: ArrayLength,
    Chunks: ArrayLength,
{
    const PRECOMPUTED: GenericArray<PrecomputedChunk<Lanes, Chunks>, Chunks> = {
        // Start with a fully zeroed nested array. (bools default to false, u32s to 0)
        let mut result: GenericArray<PrecomputedChunk<Lanes, Chunks>, Chunks> =
            unsafe { MaybeUninit::zeroed().assume_init() };

        let indices = T::INDICES;

        let total_lanes = TotalLanes::USIZE;
        let chunks = Chunks::USIZE;
        let lanes = Lanes::USIZE;

        let total_input_lanes = 2 * total_lanes; // Double bound for Swizzle
        let max_idx = total_input_lanes - 1;
        let is_pow2 = (total_input_lanes & (total_input_lanes - 1)) == 0;

        let result_slice = result.as_mut_slice();

        let mut i = 0; // Evaluate for each output chunk
        while i < chunks {
            let chunk_data = &mut result_slice[i];

            let local_idxs_slice = chunk_data.indices.as_mut_slice();
            let has_any_lo_slice = chunk_data.has_any_lo.as_mut_slice();
            let has_any_hi_slice = chunk_data.has_any_hi.as_mut_slice();
            let mask_lo_slice = chunk_data.blend_mask_lo.as_mut_slice();
            let mask_hi_slice = chunk_data.blend_mask_hi.as_mut_slice();

            let mut lane = 0;
            while lane < lanes {
                let mut global_idx = indices.as_slice()[i * lanes + lane] as usize;

                global_idx = if is_pow2 {
                    global_idx & max_idx
                } else if global_idx > max_idx {
                    max_idx
                } else {
                    global_idx
                };

                let target_chunk = global_idx / lanes;
                let local_idx = (global_idx % lanes) as u32;

                // 1. Assign local permute index
                local_idxs_slice[lane] = local_idx;

                // 2. Mark chunk routing masks
                if target_chunk < chunks {
                    has_any_lo_slice[target_chunk] = true;
                    mask_lo_slice[target_chunk].as_mut_slice()[lane] = true;
                } else {
                    let hi_chunk = target_chunk - chunks;
                    has_any_hi_slice[hi_chunk] = true;
                    mask_hi_slice[hi_chunk].as_mut_slice()[lane] = true;
                }

                lane += 1;
            }

            i += 1;
        }

        core::mem::forget(indices);

        result
    };
}

impl<R: SwizzleRegister, const N: usize> SwizzleRegister for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
{
    const HAS_PERMUTEV: bool = R::HAS_PERMUTEV;

    #[inline(always)]
    fn permutev(value: Storage<Self>, mut idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        if const { !Self::HAS_PERMUTEV } {
            return Self::scalar_permutev(value, idxs);
        }

        let mut result = [R::EMPTY; N];

        let max_idx = Self::Lanes::U32 * 2 - 1;

        for idx in idxs.iter_mut() {
            *idx = if const { Self::Lanes::IS_POWER_OF_TWO } {
                *idx & max_idx
            } else {
                (*idx).min(max_idx)
            };
        }

        for i in 0..N {
            let mut out_reg = R::EMPTY;

            for j in 0..N {
                let mut blend_mask = <R::Mask as MaskRegister>::FALSY;
                let mut local_idxs: GenericArray<u32, R::Lanes> = Default::default();
                let mut has_any = false;

                for lane in 0..<R::Lanes as Unsigned>::USIZE {
                    let mut global_idx = idxs[i * <R::Lanes as Unsigned>::USIZE + lane] as usize;

                    let target_chunk = global_idx / <R::Lanes as Unsigned>::USIZE;
                    let local_idx = (global_idx % <R::Lanes as Unsigned>::USIZE) as u32;

                    local_idxs[lane] = local_idx;

                    if target_chunk == j {
                        blend_mask = <R::Mask as MaskRegister>::set(blend_mask, lane, true);
                        has_any = true;
                    }
                }

                // Compile-time branching eliminates unused permute/blend calls
                if has_any {
                    let permuted = R::permutev(value.0[j], local_idxs);
                    out_reg = R::blendv(blend_mask, out_reg, permuted);
                }
            }

            result[i] = out_reg;
        }

        Self(result)
    }

    #[inline(always)]
    fn swizzle(a: Storage<Self>, b: Storage<Self>, mut idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        if const { !Self::HAS_PERMUTEV } {
            return Self::scalar_swizzle(a, b, idxs);
        }

        let mut result = [R::EMPTY; N];

        let max_idx = Self::Lanes::U32 * 2 - 1;

        for idx in idxs.iter_mut() {
            *idx = if const { Self::Lanes::IS_POWER_OF_TWO } {
                *idx & max_idx
            } else {
                (*idx).min(max_idx)
            };
        }

        for i in 0..N {
            let mut out_reg = R::EMPTY;

            // Check across all 2N input registers
            for j in 0..(2 * N) {
                let mut blend_mask = <R::Mask as MaskRegister>::FALSY;
                let mut local_idxs: GenericArray<u32, R::Lanes> = Default::default();
                let mut has_any = false;

                for lane in 0..<R::Lanes as Unsigned>::USIZE {
                    let mut global_idx = idxs[i * <R::Lanes as Unsigned>::USIZE + lane] as usize;

                    let target_chunk = global_idx / <R::Lanes as Unsigned>::USIZE;
                    local_idxs[lane] = (global_idx % <R::Lanes as Unsigned>::USIZE) as u32;

                    if target_chunk == j {
                        blend_mask = <R::Mask as MaskRegister>::set(blend_mask, lane, true);
                        has_any = true;
                    }
                }

                if has_any {
                    // Route to array `a` or `b` depending on chunk location
                    let src_reg = if j < N { a.0[j] } else { b.0[j - N] };
                    let permuted = R::permutev(src_reg, local_idxs);
                    out_reg = R::blendv(blend_mask, out_reg, permuted);
                }
            }

            result[i] = out_reg;
        }

        Self(result)
    }

    #[inline(always)]
    fn swizzle_const<I: SwizzleIndices<Self::Lanes>>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        if const { !Self::HAS_PERMUTEV } {
            // Forward the compile-time indices to the scalar fallback
            return Self::scalar_swizzle(a, b, I::INDICES);
        }

        let mut result = [R::EMPTY; N];
        let precomputed = <I as PrecomputedSwizzleIndices<Self::Lanes, R::Lanes, typenum::U<N>>>::PRECOMPUTED;

        for i in 0..N {
            let mut out_reg = R::EMPTY;
            let chunk_data = &precomputed[i];

            // Scan all 2N input chunks
            for j in 0..(2 * N) {
                if chunk_data.has_any(j) {
                    let mut blend_mask = <R::Mask as MaskRegister>::new_mask(chunk_data.blend_mask(j).clone());

                    // Route to `a` or `b` appropriately
                    let src_reg = if j < N { a.0[j] } else { b.0[j - N] };
                    let permuted = R::permutev(src_reg, chunk_data.indices.clone());
                    out_reg = R::blendv(blend_mask, out_reg, permuted);
                }
            }

            result[i] = out_reg;
        }

        Self(result)
    }

    #[inline(always)]
    fn permutev_const<I: SwizzleIndices<Self::Lanes>>(value: Storage<Self>) -> Storage<Self> {
        if const { !Self::HAS_PERMUTEV } {
            // Forward the compile-time indices to the scalar fallback
            return Self::scalar_permutev(value, I::INDICES);
        }

        let mut result = [R::EMPTY; N];
        let precomputed = <I as PrecomputedSwizzleIndices<Self::Lanes, R::Lanes, typenum::U<N>>>::PRECOMPUTED;

        for i in 0..N {
            let mut out_reg = R::EMPTY;
            let chunk_data = &precomputed[i];

            for j in 0..N {
                if chunk_data.has_any(j) {
                    let blend_mask = <R::Mask as MaskRegister>::new_mask(chunk_data.blend_mask(j).clone());
                    let permuted = R::permutev(value.0[j], chunk_data.indices.clone());
                    out_reg = R::blendv(blend_mask, out_reg, permuted);
                }
            }

            result[i] = out_reg;
        }

        Self(result)
    }
}

impl<R: FloatRegister, const N: usize> LinAlg4Register for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
    Self: LinAlg3Register + FloatRegister<Lanes = typenum::U4> + SwizzleRegister,
{
    // default implementations are fine
}

impl<R: FloatRegister, const N: usize> LinAlg3Register for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
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
            R::sum_elements(value.0[0])
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
            R::prod_elements(value.0[0])
        } else {
            let a = Self::extract::<0>(value);
            let b = Self::extract::<1>(value);

            a * b
        };

        let c = Self::extract::<2>(value);

        lo * c
    }
}

impl<FROM: CoreRegister, INTO: CastMaskRegister<FROM, Lanes = FROM::Lanes>, const N: usize>
    CastMaskRegister<ArrayRegister<FROM, N>> for ArrayRegister<INTO, N>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<FROM::Lanes, Output: Lanes>>,
{
    #[inline(always)]
    fn mask_from(value: Storage<ArrayRegister<FROM, N>>) -> Storage<Self> {
        Self(value.0.map(INTO::mask_from))
    }
}

impl<FROM: CoreRegister, INTO: CastRegister<FROM, Lanes = FROM::Lanes>, const N: usize>
    CastRegister<ArrayRegister<FROM, N>> for ArrayRegister<INTO, N>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<FROM::Lanes, Output: Lanes>>,
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
    Const<N>: ToUInt<Output: ArrayLength + Mul<FROM::Lanes, Output: Lanes>>,
{
    #[inline(always)]
    fn from_bits(value: Storage<ArrayRegister<FROM, N>>) -> Storage<Self> {
        Self(value.0.map(INTO::from_bits))
    }
}

macro_rules! impl_casts {
    ($a:literal $b:literal $c:literal => $trait:ident :: $($method:ident),+) => {paste::paste! {
        const _: () = {
            impl<FROM: CoreRegister, INTO: CoreRegister> $trait<ArrayRegister<FROM, $b>> for ArrayRegister<INTO, $c>
            where
                typenum::[<U $a>]: Mul<FROM::Lanes, Output: Lanes>,
                typenum::[<U $b>]: Mul<FROM::Lanes, Output: Lanes> + Mul<INTO::Lanes, Output: Lanes>,
                typenum::[<U $c>]: Mul<INTO::Lanes, Output: Lanes>,
                ArrayRegister<INTO, $b>: $trait<ArrayRegister<FROM, $a>> + CoreRegister<Storage = ArrayRegister<INTO, $b>>,
            {$(
                #[inline(always)] fn $method(value: Storage<ArrayRegister<FROM, $b>>) -> Storage<Self> {
                    let [lo, hi] = unsafe { generic_array::const_transmute(value.into_array()) };
                    let lo = ArrayRegister::<INTO, $b>::$method(ArrayRegister::<FROM, $a>::from_array(lo));
                    let hi = ArrayRegister::<INTO, $b>::$method(ArrayRegister::<FROM, $a>::from_array(hi));
                    Self::from_array(unsafe { generic_array::const_transmute([lo, hi]) })
                }
            )+}

            impl<FROM: CoreRegister, INTO: CoreRegister> $trait<ArrayRegister<FROM, $c>> for ArrayRegister<INTO, $b>
            where
                typenum::[<U $a>]: Mul<INTO::Lanes, Output: Lanes>,
                typenum::[<U $b>]: Mul<FROM::Lanes, Output: Lanes> + Mul<INTO::Lanes, Output: Lanes>,
                typenum::[<U $c>]: Mul<FROM::Lanes, Output: Lanes>,
                ArrayRegister<INTO, $a>: $trait<ArrayRegister<FROM, $b>>,
                ArrayRegister<FROM, $b>: CoreRegister<Storage = ArrayRegister<FROM, $b>>,
            {$(
                #[inline(always)] fn $method(value: Storage<ArrayRegister<FROM, $c>>) -> Storage<Self> {
                    let [lo, hi] = unsafe { generic_array::const_transmute(value.into_array()) };
                    let lo = ArrayRegister::<INTO, $a>::$method(ArrayRegister::<FROM, $b>::from_array(lo));
                    let hi = ArrayRegister::<INTO, $a>::$method(ArrayRegister::<FROM, $b>::from_array(hi));
                    Self::from_array(unsafe { generic_array::const_transmute([lo, hi]) })
                }
            )+}
        };
    }};

    ($trait:ident :: $($method:ident),+) => {
        // base cases
        const _: () = {
            impl<FROM: CoreRegister, INTO: CoreRegister> $trait<ArrayRegister<FROM, 2>> for ArrayRegister<INTO, 4>
            where
                typenum::U2: Mul<FROM::Lanes, Output: Lanes> + Mul<INTO::Lanes, Output: Lanes>,
                typenum::U4: Mul<INTO::Lanes, Output: Lanes>,
                ArrayRegister<INTO, 2>: $trait<FROM> + CoreRegister<Storage = ArrayRegister<INTO, 2>>,
            {$(
                #[inline(always)] fn $method(value: Storage<ArrayRegister<FROM, 2>>) -> Storage<Self> {
                    let [lo, hi] = unsafe { generic_array::const_transmute(value.into_array()) };
                    let lo = ArrayRegister::<INTO, 2>::$method(lo).into_array();
                    let hi = ArrayRegister::<INTO, 2>::$method(hi).into_array();
                    Self::from_array(unsafe { generic_array::const_transmute([lo, hi]) })
                }
            )+}

            impl<FROM: CoreRegister, INTO: CoreRegister> $trait<ArrayRegister<FROM, 4>> for ArrayRegister<INTO, 2>
            where
                typenum::U2: Mul<INTO::Lanes, Output: Lanes>,
                typenum::U4: Mul<FROM::Lanes, Output: Lanes>,
                INTO: $trait<ArrayRegister<FROM, 2>>,
                ArrayRegister<FROM, 2>: CoreRegister<Storage = ArrayRegister<FROM, 2>>,
            {$(
                #[inline(always)] fn $method(value: Storage<ArrayRegister<FROM, 4>>) -> Storage<Self> {
                    let [lo, hi] = unsafe { generic_array::const_transmute(value.into_array()) };
                    let lo = INTO::$method(ArrayRegister::<FROM, 2>::from_array(lo));
                    let hi = INTO::$method(ArrayRegister::<FROM, 2>::from_array(hi));
                    Self::from_array(unsafe { generic_array::const_transmute([lo, hi]) })
                }
            )+}
        };

        impl_casts!(2 4 8 => $trait::$($method),+);
        impl_casts!(4 8 16 => $trait::$($method),+);
        impl_casts!(8 16 32 => $trait::$($method),+);
    }
}

impl_casts!(CastMaskRegister::mask_from);
impl_casts!(BitCastRegister::from_bits);
impl_casts!(CastRegister::cast_from, fast_cast_from);

macro_rules! impl_indexable {
    ($a:literal $b:literal $c:literal) => {paste::paste! {
        const _: () = {
            impl<IDX: UnsignedIntegerRegister, R: Register> IndexableRegister<ArrayRegister<IDX, $b>> for ArrayRegister<R, $c>
            where
                ArrayRegister<IDX, $a>: UnsignedIntegerRegister<Lanes = <ArrayRegister<R, $b> as CoreRegister>::Lanes>,
                ArrayRegister<R, $b>: IndexableRegister<ArrayRegister<IDX, $a>, Element = R::Element>,
                typenum::[<U $b>]: Mul<R::Lanes, Output: Lanes> + Mul<IDX::Lanes, Output: Lanes>
                    + Mul<<ArrayRegister<R, $b> as CoreRegister>::Lanes, Output: Lanes>,
                typenum::[<U $c>]: Mul<R::Lanes, Output: Lanes> + Mul<IDX::Lanes, Output: Lanes>
                    + Mul<<ArrayRegister<R, $b> as CoreRegister>::Lanes, Output: Lanes>
                    + Mul<R::Lanes, Output = <ArrayRegister<IDX, $b> as CoreRegister>::Lanes>,
            {
                unsafe fn gather(ptr: *const Self::Element, indices: Storage<ArrayRegister<IDX, $b>>) -> Storage<Self> {
                    let [lo_idx, hi_idx] = unsafe { generic_array::const_transmute(indices) };
                    let lo_val = unsafe { <ArrayRegister<R, $b> as IndexableRegister<ArrayRegister<IDX, $a>>>::gather(ptr, lo_idx) };
                    let hi_val = unsafe { <ArrayRegister<R, $b> as IndexableRegister<ArrayRegister<IDX, $a>>>::gather(ptr, hi_idx) };
                    ArrayRegister(unsafe { generic_array::const_transmute([lo_val, hi_val]) })
                }
            }

            impl<IDX: UnsignedIntegerRegister, R: Register> IndexableRegister<ArrayRegister<IDX, $c>> for ArrayRegister<R, $b>
            where
                ArrayRegister<IDX, $b>: UnsignedIntegerRegister<Lanes = <ArrayRegister<R, $a> as CoreRegister>::Lanes>,
                ArrayRegister<IDX, $c>: UnsignedIntegerRegister<Lanes = <ArrayRegister<R, $b> as CoreRegister>::Lanes>,
                ArrayRegister<R, $a>: IndexableRegister<ArrayRegister<IDX, $b>, Element = R::Element>,
                typenum::[<U $b>]: Mul<R::Lanes, Output: Lanes> + Mul<IDX::Lanes, Output: Lanes>,
                typenum::[<U $c>]: Mul<R::Lanes, Output: Lanes> + Mul<IDX::Lanes, Output: Lanes>,
            {
                unsafe fn gather(ptr: *const Self::Element, indices: Storage<ArrayRegister<IDX, $c>>) -> Storage<Self> {
                    let [lo_idx, hi_idx] = unsafe { generic_array::const_transmute(indices) };
                    let lo_val = unsafe { <ArrayRegister<R, $a> as IndexableRegister<ArrayRegister<IDX, $b>>>::gather(ptr, lo_idx) };
                    let hi_val = unsafe { <ArrayRegister<R, $a> as IndexableRegister<ArrayRegister<IDX, $b>>>::gather(ptr, hi_idx) };
                    ArrayRegister(unsafe { generic_array::const_transmute([lo_val, hi_val]) })
                }
            }
        };
    }};

    () => {
        // base case
        impl<IDX, R: Register> IndexableRegister<ArrayRegister<IDX, 2>> for ArrayRegister<R, 4>
        where
            IDX: UnsignedIntegerRegister<Lanes = <ArrayRegister<R, 2> as CoreRegister>::Lanes>,
            ArrayRegister<R, 2>: IndexableRegister<IDX, Element = R::Element>,
            typenum::U2: Mul<R::Lanes, Output: Lanes> + Mul<IDX::Lanes, Output: Lanes>
                + Mul<<ArrayRegister<R, 2> as CoreRegister>::Lanes, Output: Lanes>,
            typenum::U4: Mul<R::Lanes, Output: Lanes> + Mul<IDX::Lanes, Output: Lanes>
                + Mul<<ArrayRegister<R, 2> as CoreRegister>::Lanes, Output: Lanes>
                + Mul<R::Lanes, Output = <ArrayRegister<IDX, 2> as CoreRegister>::Lanes>,
        {
            unsafe fn gather(ptr: *const Self::Element, indices: Storage<ArrayRegister<IDX, 2>>) -> Storage<Self> {
                let [lo_idx, hi_idx] = unsafe { generic_array::const_transmute(indices) };
                let lo_val = unsafe { <ArrayRegister<R, 2> as IndexableRegister<IDX>>::gather(ptr, lo_idx) };
                let hi_val = unsafe { <ArrayRegister<R, 2> as IndexableRegister<IDX>>::gather(ptr, hi_idx) };
                ArrayRegister(unsafe { generic_array::const_transmute([lo_val, hi_val]) })
            }
        }

        impl<IDX: UnsignedIntegerRegister, R: Register> IndexableRegister<ArrayRegister<IDX, 4>> for ArrayRegister<R, 2>
        where
            ArrayRegister<IDX, 2>: UnsignedIntegerRegister<Lanes = R::Lanes>,
            ArrayRegister<IDX, 4>: UnsignedIntegerRegister<Lanes = <ArrayRegister<R, 2> as CoreRegister>::Lanes>,
            R: IndexableRegister<ArrayRegister<IDX, 2>>,
            typenum::U2: Mul<R::Lanes, Output: Lanes> + Mul<IDX::Lanes, Output: Lanes>,
            typenum::U4: Mul<R::Lanes, Output: Lanes> + Mul<IDX::Lanes, Output: Lanes>,
        {
            unsafe fn gather(ptr: *const Self::Element, indices: Storage<ArrayRegister<IDX, 4>>) -> Storage<Self> {
                let [lo_idx, hi_idx] = unsafe { generic_array::const_transmute(indices) };
                let lo_val = unsafe { <R as IndexableRegister<ArrayRegister<IDX, 2>>>::gather(ptr, lo_idx) };
                let hi_val = unsafe { <R as IndexableRegister<ArrayRegister<IDX, 2>>>::gather(ptr, hi_idx) };
                ArrayRegister(unsafe { generic_array::const_transmute([lo_val, hi_val]) })
            }
        }

        impl_indexable!(2 4 8);
        impl_indexable!(4 8 16);
        impl_indexable!(8 16 32);
        // impl_indexable!(16 32 64);
    }
}

impl_indexable!();
