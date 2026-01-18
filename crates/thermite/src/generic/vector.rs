use super::*;

use crate::register::{
    BitsRegister, BitshiftRegister, CastMaskRegister, CastRegister, Element, FloatElement, FloatRegister,
    IntegerRegister, Lanes, NumericRegister, PartialOrdRegister, Register, SignedIntegerRegister, SignedRegister,
    Storage, SwizzleRegister, UnsignedIntegerRegister,
};

impl<FROM, INTO> CastVector<Vector<FROM>> for Vector<INTO>
where
    FROM: Register,
    INTO: Register + CastRegister<FROM>,
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

impl<FROM, INTO> BitsVector<Vector<FROM>> for Vector<INTO>
where
    FROM: Register,
    INTO: Register + BitsRegister<FROM>,
{
    #[inline(always)]
    fn from_bits(bits: Vector<FROM>) -> Self {
        Vector::<INTO>::from_bits(bits)
    }
}

impl<FROM, INTO> CastMask<Mask<FROM>> for Mask<INTO>
where
    FROM: Register,
    INTO: Register + CastMaskRegister<FROM>,
{
    #[inline(always)]
    fn mask_from(from: Mask<FROM>) -> Self {
        Mask::<INTO>::from_mask(from)
    }
}

impl<R> GenericSelectable for Vector<R>
where
    R: Register,
{
    type SelectableMask = Mask<R>;

    #[inline(always)]
    fn select<M>(mask: M, t: Self, f: Self) -> Self
    where
        Mask<R>: CastMask<M>,
    {
        Mask::mask_from(mask).select(t, f)
    }
}

impl<R> GenericSelectable for Mask<R>
where
    R: Register,
{
    type SelectableMask = Mask<R>;

    #[inline(always)]
    fn select<M>(mask: M, t: Self, f: Self) -> Self
    where
        Mask<R>: CastMask<M>,
    {
        Mask::mask_from(mask).select(t, f)
    }
}

impl<R: Register> GenericMask<Vector<R>> for Mask<R> {
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

#[rustfmt::skip]
impl<R: Register> GenericVector for Vector<R> {
    type Element = R::Element;

    const EMPTY: Self = Vector::<R>::EMPTY;
    const LANES: usize = Vector::<R>::LANES;

    type Lanes = R::Lanes;

    type USize = Vector<R::USize>;
    type ISize = Vector<R::ISize>;

    type Mask = Mask<R>;

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

    const HAS_MSB_BLENDV: bool = R::HAS_MSB_BLENDV;
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


    type ExtendedPrecision = Vector<R::ExtendedPrecision>;

    const HAS_TRUE_FMA: bool = R::HAS_TRUE_FMA;


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

    const HAS_APPROX_RCP: bool = R::HAS_APPROX_RCP;
    const HAS_APPROX_RSQRT: bool = R::HAS_APPROX_RSQRT;

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

    #[inline(always)]
    unsafe fn block_autovectorization(&mut self) {
        unsafe { R::block_autovectorization(&mut self.0) };
    }

}

#[rustfmt::skip]
impl<R: FloatRegister> FloatVectorWithBits for Vector<R> {
    type Signed = Vector<R::Signed>;
    type Bits = Vector<R::Bits>;

    const HAS_NATIVE_LDEXP: bool = R::HAS_NATIVE_LDEXP;
    const HAS_NATIVE_FREXP: bool = R::HAS_NATIVE_FREXP;

    #[inline(always)] unsafe fn native_ldexp(self, exp: Self::Signed) -> Self {
        unsafe { Vector(R::native_ldexp(self.0, exp.0)) }
    }

    #[inline(always)] unsafe fn native_frexp(self) -> (Self, Self::Signed) {
        let (mantissa, exp) = unsafe { R::native_frexp(self.0) };
        (Vector(mantissa), Vector(exp))
    }


    #[inline(always)] fn total_order(self) -> Self::Signed { Vector::<R>::total_order(self) }
}
