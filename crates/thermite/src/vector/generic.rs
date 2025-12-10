#![allow(missing_docs)]

use core::ops::{Add, Div, Mul, Neg, Rem, Sub};

use num_traits::Signed;

use crate::{
    Mask, Vector,
    divider::Denominator,
    register::{
        BitshiftRegister, Element, FloatRegister, IntegerRegister, MaskRegister, NumericRegister, PartialOrdRegister,
        Register, SignedIntegerRegister, SignedRegister, UnsignedIntegerRegister,
    },
};

pub trait VectorOperation<Args> {
    type Output;
    fn call(args: Args) -> Self::Output;
}

pub trait GenericVector: Sized + Copy + core::fmt::Debug + 'static {
    type Element: Element;

    const EMPTY: Self;
    const LANES: usize;

    type USize: UnsignedIntegerVector<Element = <Self::Element as Element>::USize>;
    type ISize: SignedIntegerVector<Element = <Self::Element as Element>::ISize>;

    fn splat(value: Self::Element) -> Self;
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
}

pub trait GenericMask<V: GenericVector>: Sized + Copy + core::fmt::Debug + 'static {
    fn all(self) -> bool;
    fn any(self) -> bool;
    fn none(self) -> bool;
    fn select(self, t: V, f: V) -> V;
}

impl<R: MaskRegister> GenericMask<Vector<R>> for Mask<R> {
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
    fn select(self, t: Vector<R>, f: Vector<R>) -> Vector<R> {
        Vector(R::blendv(self.0, f.0, t.0))
    }
}

pub trait BitshiftVector: GenericVector {
    fn shli<const I: i32>(self) -> Self;
    fn shri<const I: i32>(self) -> Self;
    fn shlv(self, counts: Self::USize) -> Self;
    fn shrv(self, counts: Self::USize) -> Self;
}

pub trait MaskedVector: GenericVector {
    type Mask: GenericMask<Self>;
}

pub trait PartialOrdVector: MaskedVector + PartialEq {
    fn cmp_lt(self, other: Self) -> Self::Mask;
    fn cmp_le(self, other: Self) -> Self::Mask;
    fn cmp_gt(self, other: Self) -> Self::Mask;
    fn cmp_ge(self, other: Self) -> Self::Mask;
    fn cmp_eq(self, other: Self) -> Self::Mask;
    fn cmp_ne(self, other: Self) -> Self::Mask;
}

pub trait NumericVector:
    PartialOrdVector
    + num_traits::Num
    + num_traits::Bounded
    + num_traits::ConstOne
    + num_traits::ConstZero
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

    fn min_element(self) -> Self::Element;
    fn max_element(self) -> Self::Element;

    fn sum_elements(self) -> Self::Element;
    fn prod_elements(self) -> Self::Element;

    fn offset() -> Self;
    fn indexed() -> Self;
}

pub trait SignedVector: NumericVector<Element: Signed> + Signed {
    const NEG_ONE: Self;
    const MIN_POSITIVE: Self;

    fn abs(self) -> Self;

    fn signum(self) -> Self;
    fn copysign(self, sign: Self) -> Self;

    fn is_positive(self) -> Self::Mask;
    fn is_negative(self) -> Self::Mask;
}

pub trait IntegerVector:
    NumericVector<Element: Denominator> + Div<Self::Divider, Output = Self> + Div<Self::BranchfreeDivider, Output = Self>
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

pub trait FloatVector: SignedVector + num_traits::FloatConst {
    const HALF: Self;
    const NEG_ZERO: Self;
    const INFINITY: Self;
    const NEG_INFINITY: Self;
    const NAN: Self;
    const EPSILON: Self;

    type Signed: SignedIntegerVector;
    type Bits: UnsignedIntegerVector;
    type ExtendedPrecision: FloatVector;

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

    type USize = Vector<R::USize>;
    type ISize = Vector<R::ISize>;

    #[inline(always)] fn splat(value: Self::Element) -> Self { Vector::<R>::splat(value) }
    #[inline(always)] fn broadcast<const I: usize>(self) -> Self { Vector::<R>::broadcast::<I>(self) }
    #[inline(always)] fn broadcastv(self, idx: usize) -> Self { Vector::<R>::broadcastv(self, idx) }
    #[inline(always)] fn as_slice(&self) -> &[Self::Element] { Vector::<R>::as_slice(self) }
    #[inline(always)] fn as_mut_slice(&mut self) -> &mut [Self::Element] { Vector::<R>::as_mut_slice(self) }
    #[inline(always)] fn extract<const I: usize>(self) -> Self::Element { Vector::<R>::extract::<I>(self) }
    #[inline(always)] fn insert<const I: usize>(self, value: Self::Element) -> Self { Vector::<R>::insert::<I>(self, value) }
    #[inline(always)] fn reverse(self) -> Self { Vector::<R>::reverse(self) }
    #[inline(always)] fn swap_bytes(self) -> Self { Vector::<R>::swap_bytes(self) }

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

    fn single(value: Self::Element) -> Self {
        Vector::<R>::single(value)
    }
    #[inline(always)] unsafe fn load(ptr: *const Self::Element) -> Self { unsafe { Vector::<R>::load(ptr) } }
    #[inline(always)] unsafe fn load_unaligned(ptr: *const Self::Element) -> Self { unsafe { Vector::<R>::load_unaligned(ptr) } }
    #[inline(always)] unsafe fn load_streaming(ptr: *const Self::Element) -> Self { unsafe { Vector::<R>::load_streaming(ptr) } }
    #[inline(always)] unsafe fn store(self, ptr: *mut Self::Element) { unsafe { Vector::<R>::store(self, ptr) } }
    #[inline(always)] unsafe fn store_unaligned(self, ptr: *mut Self::Element) { unsafe { Vector::<R>::store_unaligned(self, ptr) } }
    #[inline(always)] unsafe fn store_streaming(self, ptr: *mut Self::Element) { unsafe { Vector::<R>::store_streaming(self, ptr) } }
}

#[rustfmt::skip]
impl<R: BitshiftRegister> BitshiftVector for Vector<R> {
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

    #[inline(always)] fn min_element(self) -> Self::Element { Vector::<R>::min_element(self) }
    #[inline(always)] fn max_element(self) -> Self::Element { Vector::<R>::max_element(self) }

    #[inline(always)] fn sum_elements(self) -> Self::Element { Vector::<R>::sum_elements(self) }
    #[inline(always)] fn prod_elements(self) -> Self::Element { Vector::<R>::prod_elements(self) }

    #[inline(always)] fn offset() -> Self { Vector::<R>::offset() }
    #[inline(always)] fn indexed() -> Self { Vector::<R>::indexed() }
}

#[rustfmt::skip]
impl<R: SignedRegister> SignedVector for Vector<R>
where
    R::Element: Signed,
{
    const NEG_ONE: Self = Vector::<R>::NEG_ONE;
    const MIN_POSITIVE: Self = Vector::<R>::MIN_POSITIVE;

    #[inline(always)] fn abs(self) -> Self { Vector::<R>::abs(self) }

    #[inline(always)] fn signum(self) -> Self { Vector::<R>::signum(self) }
    #[inline(always)] fn copysign(self, sign: Self) -> Self { Vector::<R>::copysign(self, sign) }

    #[inline(always)] fn is_positive(self) -> Self::Mask { Vector::<R>::is_positive(self) }
    #[inline(always)] fn is_negative(self) -> Self::Mask { Vector::<R>::is_negative(self) }
}

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

impl<R: SignedIntegerRegister> SignedIntegerVector for Vector<R>
where
    R::Element: Denominator + Signed,
{
    #[inline(always)]
    fn srai<const I: i32>(self) -> Self {
        Vector::<R>::srai::<I>(self)
    }

    #[inline(always)]
    fn sra(self, count: u32) -> Self {
        Vector::<R>::sra(self, count)
    }

    #[inline(always)]
    fn srav(self, counts: Self::USize) -> Self {
        Vector::<R>::srav(self, counts)
    }
}

impl<R: UnsignedIntegerRegister> UnsignedIntegerVector for Vector<R>
where
    R::Element: Denominator,
{
    #[inline(always)]
    fn is_power_of_two(self) -> Self::Mask {
        Vector::<R>::is_power_of_two(self)
    }

    #[inline(always)]
    fn next_power_of_two_m1(self) -> Self {
        Vector::<R>::next_power_of_two_m1(self)
    }

    #[inline(always)]
    fn ilog2p1(self) -> Self {
        Vector::<R>::ilog2p1(self)
    }

    #[inline(always)]
    fn parity(self) -> Self {
        Vector::<R>::parity(self)
    }
}

#[rustfmt::skip]
impl<R: FloatRegister> FloatVector for Vector<R>
where
    R::Element: num_traits::FloatConst,
{
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
