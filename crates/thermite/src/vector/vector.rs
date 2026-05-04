#![warn(missing_docs, clippy::missing_safety_doc)]

//! Vector type wrapping low-level registers with a vector-like interface.
//!
//! Most vector features are provided by the [`GenericVector`](crate::generic) traits,
//! but this is the underlying type that most vectors are based on, using low-level
//! registers for various architectures.

use super::ops::*;
use super::*;

use crate::{
    divider::{BranchfreeDivider, Denominator, Divider, UnsupportedDivisor, vector::VectorDivider},
    mask::{CastMask, GenericSelectable, Mask},
    math::{FloatConsts, policy::Policy},
    register::{
        self, BitCastRegister, BitshiftRegister, BitwiseRegister, CastMaskRegister, CastRegister, ConcatRegister,
        ExtendRegister, FloatRegister, IndexableRegister, IntegerRegister, LinAlg3Register, LinAlg4Register,
        MaskRegister, NumericRegister, PartialOrdRegister, PermuteRegister, Register, ShuffleRegister,
        SignedIntegerRegister, SignedRegister, Storage, SwizzleRegister, UnsignedIntegerRegister,
    },
};

use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign, Index, IndexMut,
    Mul, MulAssign, Neg, Not, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};

use num_traits::{
    ConstOne, ConstZero, MulAdd, MulAddAssign, Num, One, Saturating, SaturatingAdd, SaturatingSub, Signed, WrappingAdd,
    WrappingMul, WrappingSub, Zero,
};

use generic_array::{GenericArray, typenum::Unsigned};

// pub mod streaming;
// pub mod unaligned;

/// SIMD Vector type.
///
/// This wraps a low-level register type and provides a vector-like interface, including
/// operator overloading and element-wise operations.
#[repr(transparent)]
pub struct Vector<R: Register>(#[doc(hidden)] pub Storage<R>);

impl<R: Register> Clone for Vector<R> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: Register> Copy for Vector<R> {}

const _: () = {
    use core::fmt;

    impl<R: Register> fmt::Debug for Vector<R> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let mut t = f.debug_tuple("Vector");

            for v in R::as_array(&self.0) {
                t.field(&v);
            }

            t.finish()
        }
    }
};

impl<R: Register> const_default::ConstDefault for Vector<R> {
    const DEFAULT: Self = Self::EMPTY;
}

impl<R: Register> Default for Vector<R> {
    #[inline(always)]
    fn default() -> Self {
        Self::EMPTY
    }
}

impl<R: Register> Vector<R> {
    /// Splat a value in a const context.
    ///
    /// This is temporarily marked as deprecated until we can figure out a better way.
    /// Only use within a `const {}` block.
    #[deprecated]
    #[inline(never)]
    pub const fn splat_const(value: R::Element) -> Self {
        Vector(register::reg_splat::<R>(value))
    }

    /// Create a new vector from an array of elements.
    #[inline(always)]
    pub fn from_array(values: impl Into<GenericArray<R::Element, R::Lanes>>) -> Self {
        Self(R::new(values.into()))
    }

    /// Returns a reference to the vector's elements as an array.
    #[inline(always)]
    pub fn as_array(&self) -> &GenericArray<R::Element, R::Lanes> {
        R::as_array(&self.0)
    }

    /// Returns a mutable reference to the vector's elements as an array.
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut GenericArray<R::Element, R::Lanes> {
        R::as_array_mut(&mut self.0)
    }

    /// Returns a slice of the vector's elements.
    #[inline(always)]
    pub fn as_slice(&self) -> &[R::Element] {
        R::as_array(&self.0).as_slice()
    }

    /// Returns a mutable slice of the vector's elements.
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [R::Element] {
        R::as_array_mut(&mut self.0).as_mut_slice()
    }

    /// Convert the vector to an array of elements.
    #[inline(always)]
    pub fn to_array(self) -> GenericArray<R::Element, R::Lanes> {
        R::as_array(&self.0).clone()
    }
}

impl<FROM, INTO> CastVector<Vector<FROM>> for Vector<INTO>
where
    FROM: Register + CastRegister<INTO>,
    INTO: Register + CastRegister<FROM>,
{
    #[inline(always)]
    fn cast_from(from: Vector<FROM>) -> Self {
        Vector(<INTO as CastRegister<FROM>>::cast_from(from.0))
    }

    #[inline(always)]
    fn cast_into(self) -> Vector<FROM> {
        Vector(<FROM as CastRegister<INTO>>::cast_from(self.0))
    }

    #[inline(always)]
    fn fast_cast_from(from: Vector<FROM>) -> Self {
        Vector(<INTO as CastRegister<FROM>>::fast_cast_from(from.0))
    }

    #[inline(always)]
    fn fast_cast_into(self) -> Vector<FROM> {
        Vector(<FROM as CastRegister<INTO>>::fast_cast_from(self.0))
    }
}

impl<FROM, INTO> BitCastVector<Vector<FROM>> for Vector<INTO>
where
    FROM: Register,
    INTO: Register + BitCastRegister<FROM>,
{
    #[inline(always)]
    fn from_bits(bits: Vector<FROM>) -> Self {
        Vector(<INTO as BitCastRegister<FROM>>::from_bits(bits.0))
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
        Vector(R::blendv(Mask::mask_from(mask).0, f.0, t.0))
    }
}

impl<R: Register> crate::simd::HasIsa for Vector<R> {
    const ISA: InstructionSet = R::ISA;
}

impl<T, R: Register> SplatVectorValue<T, Vector<R>> for Vector<R>
where
    T: SplatConst<R::Element>,
{
    const VALUE: Vector<R> = const { Vector(register::reg_splat::<R>(T::VALUE)) };
}

impl<R: Register> SplatVector<R::Element> for Vector<R> {
    type Splat<T: SplatConst<R::Element>> = Self;
}

impl<R: Register> Interleave for Vector<R> {
    #[inline(always)]
    fn interleave(self, other: Self) -> (Self, Self) {
        let (a, b) = R::interleave(self.0, other.0);
        (Vector(a), Vector(b))
    }

    #[inline(always)]
    fn deinterleave(self, other: Self) -> (Self, Self) {
        let (a, b) = R::deinterleave(self.0, other.0);
        (Vector(a), Vector(b))
    }
}

#[rustfmt::skip] #[thermite_macros::vector_impl]
impl<R: Register> GenericVector for Vector<R> {
    type Element = R::Element;

    const EMPTY: Self = Vector(R::EMPTY);
    const LANES: usize = <R::Lanes as generic_array::typenum::Unsigned>::USIZE;

    type Lanes = R::Lanes;

    type Unsigned = Vector<R::Unsigned>;
    type Signed = Vector<R::Signed>;

    type Mask = Mask<R>;

    fn new<const N: usize>(values: [R::Element; N]) -> Self
    where
        generic_array::typenum::Const<N>: generic_array::IntoArrayLength<ArrayLength = R::Lanes>,
    {
        Self(R::new(values.into()))
    }

    #[masked] fn splat(value: Self::Element) -> Self { Vector(R::splat(value)) }

    fn single(value: Self::Element) -> Self { Vector(R::single(value)) }

    #[conditional] fn broadcast<const I: usize>(self) -> Self {}
    #[conditional] fn broadcastv(self, idx: usize) -> Self {}

    fn extract<const I: usize>(self) -> Self::Element { R::extract::<I>(self.0) }
    fn extractv(self, idx: usize) -> Self::Element { R::as_array(&self.0)[idx] }

    fn insert<const I: usize>(self, value: Self::Element) -> Self { Vector(R::insert::<I>(self.0, value)) }

    fn insertv(mut self, idx: usize, value: Self::Element) -> Self {
        R::as_array_mut(&mut self.0)[idx] = value;
        self
    }

    unsafe fn lookup_unchecked(values: &[Self::Element], indices: Self::Unsigned) -> Self {
        unsafe { Self(R::lookup(values, indices.0)) }
    }

    #[conditional] fn reverse(self) -> Self {}
    #[conditional] fn swap_bytes(self) -> Self {}

    // The arguments of these are reversed for the register
    fn z(self, mask: Self::Mask) -> Self { Vector(R::z(mask.0, self.0)) }
    fn nz(self, mask: Self::Mask) -> Self { Vector(R::nz(mask.0, self.0)) }

    fn map<F>(self, f: F) -> Self where F: Fn(Self::Element) -> Self::Element { Vector(R::map(self.0, f)) }
    fn fold<F>(self, init: Self::Element, f: F) -> Self::Element where F: Fn(Self::Element, Self::Element) -> Self::Element { R::fold(init, self.0, f) }
    fn reduce<F>(self, f: F) -> Self::Element where F: Fn(Self::Element, Self::Element) -> Self::Element { R::reduce(self.0, f) }

    #[masked] unsafe fn load(ptr: *const Self::Element) -> Self {}

    unsafe fn load_unaligned(ptr: *const Self::Element) -> Self { unsafe { Vector(R::load_unaligned(ptr)) } }
    unsafe fn load_streaming(ptr: *const Self::Element) -> Self { unsafe { Vector(R::load_stream(ptr)) } }

    unsafe fn store(self, ptr: *mut Self::Element) { unsafe { R::store(ptr, self.0) } }
    unsafe fn store_masked(self, mask: Self::Mask, ptr: *mut Self::Element) { unsafe { R::store_masked(ptr, mask.0, self.0) } }
    unsafe fn store_unaligned(self, ptr: *mut Self::Element) { unsafe { R::store_unaligned(ptr, self.0) } }
    unsafe fn store_streaming(self, ptr: *mut Self::Element) { unsafe { R::store_stream(ptr, self.0) } }
}

#[rustfmt::skip]
impl<R, I> IndexableVector<Vector<I>> for Vector<R>
where
    R: IndexableRegister<I>,
    I: UnsignedIntegerRegister<Lanes = R::Lanes>,
{
    #[inline(always)]
    unsafe fn gather_ptr(ptr: *const Self::Element, indices: Vector<I>) -> Self {
        unsafe { Vector(R::gather(ptr, indices.0)) }
    }

    #[inline(always)]
    unsafe fn gather_ptr_m(src: Self, mask: Self::Mask, ptr: *const Self::Element, indices: Vector<I>) -> Self {
        unsafe { Vector(R::gather_m(src.0, mask.0, ptr, indices.0)) }
    }

    #[inline(always)]
    unsafe fn gather_ptr_z(mask: Self::Mask, ptr: *const Self::Element, indices: Vector<I>) -> Self {
        unsafe { Vector(R::gather_z(mask.0, ptr, indices.0)) }
    }

    #[inline(always)]
    unsafe fn scatter_ptr(value: Self, ptr: *mut Self::Element, indices: Vector<I>) {
        unsafe { R::scatter(value.0, ptr, indices.0) };
    }

    #[inline(always)]
    unsafe fn scatter_ptr_m(value: Self, mask: Self::Mask, ptr: *mut Self::Element, indices: Vector<I>) {
        unsafe { R::scatter_m(value.0, mask.0, ptr, indices.0) };
    }
}

#[rustfmt::skip] #[thermite_macros::vector_impl]
impl<R: BitwiseRegister + Register> BitwiseVector for Vector<R> {
    #[conditional] fn ternlog<const IMM: i32>(a: Self, b: Self, c: Self) -> Self {}
    #[conditional] fn bilog<const IMM: i32>(a: Self, b: Self) -> Self {}
}

#[rustfmt::skip] #[thermite_macros::vector_impl]
impl<R: BitshiftRegister> BitshiftVector for Vector<R> {
    const HAS_TRUE_SHIFTV: bool = R::HAS_TRUE_SHIFTV;
    const HAS_WIDE_BYTE_SHIFTS: bool = R::HAS_WIDE_BYTE_SHIFTS;

    #[conditional] fn bshli<const I: i32>(self) -> Self {}
    #[conditional] fn bshri<const I: i32>(self) -> Self {}
    #[conditional] fn shli<const I: i32>(self) -> Self {}
    #[conditional] fn shri<const I: i32>(self) -> Self {}
    #[conditional] fn shlv(self, shifts: Self::Unsigned) -> Self {}
    #[conditional] fn shrv(self, shifts: Self::Unsigned) -> Self {}

    #[conditional] fn rol(self, shift: u32) -> Self {}
    #[conditional] fn ror(self, shift: u32) -> Self {}
    #[conditional] fn roli<const I: i32>(self) -> Self {}
    #[conditional] fn rori<const I: i32>(self) -> Self {}
    #[conditional] fn rolv(self, counts: Self::Unsigned) -> Self {}
    #[conditional] fn rorv(self, counts: Self::Unsigned) -> Self {}

    #[conditional] fn reverse_bits(self) -> Self {}
}

#[rustfmt::skip]
impl<R: PartialOrdRegister> PartialOrdVector for Vector<R> {
    #[inline(always)] fn cmp_lt(self, other: Self) -> Self::Mask { Mask(R::lt(self.0, other.0)) }
    #[inline(always)] fn cmp_le(self, other: Self) -> Self::Mask { Mask(R::le(self.0, other.0)) }
    #[inline(always)] fn cmp_gt(self, other: Self) -> Self::Mask { Mask(R::gt(self.0, other.0)) }
    #[inline(always)] fn cmp_ge(self, other: Self) -> Self::Mask { Mask(R::ge(self.0, other.0)) }
    #[inline(always)] fn cmp_eq(self, other: Self) -> Self::Mask { Mask(R::eq(self.0, other.0)) }
    #[inline(always)] fn cmp_ne(self, other: Self) -> Self::Mask { Mask(R::ne(self.0, other.0)) }
}

#[rustfmt::skip] #[thermite_macros::vector_impl]
impl<R: NumericRegister> NumericVector for Vector<R> {
    const ZERO: Self = Vector(R::ZERO);
    const ONE: Self = Vector(R::ONE);
    const TWO: Self = Vector(R::TWO);
    const MIN: Self = Vector(R::MIN);
    const MAX: Self = Vector(R::MAX);

    fn is_zero(self) -> Self::Mask { self.cmp_eq(Self::ZERO) }

    fn is_all_zero(self) -> bool { R::is_all_zero(self.0) }

    #[conditional] fn min(self, other: Self) -> Self {}
    #[conditional] fn max(self, other: Self) -> Self {}

    fn clamp(self, min: Self, max: Self) -> Self { self.min(max).max(min) }

    fn min_element(self) -> Self::Element { R::min_element(self.0) }
    fn max_element(self) -> Self::Element { R::max_element(self.0) }

    #[conditional] fn scale(self, factor: Self::Element) -> Self {}

    fn pairwise_sum(lo: Self, hi: Self) -> Self {}
    fn relaxed_pairwise_sum(lo: Self, hi: Self) -> Self {}

    fn sum_elements(self) -> Self::Element { R::sum_elements(self.0) }
    fn prod_elements(self) -> Self::Element { R::prod_elements(self.0) }

    fn offset() -> Self { Vector(R::offset()) }
    fn indexed() -> Self { Vector(R::indexed()) }
}

impl<R: NumericRegister> Square for Vector<R> {
    type Output = Self;

    #[inline(always)]
    fn square(self) -> Self {
        Vector(R::square(self.0))
    }
}

impl<R: NumericRegister> SquareMasked<Mask<R>> for Vector<R> {
    #[inline(always)]
    fn square_c(self, mask: Mask<R>) -> Self {
        Vector(R::square_c(mask.0, self.0))
    }

    #[inline(always)]
    fn square_m(self, src: Self, mask: Mask<R>) -> Self {
        Vector(R::square_m(src.0, mask.0, self.0))
    }

    #[inline(always)]
    fn square_z(self, mask: Mask<R>) -> Self {
        Vector(R::square_z(mask.0, self.0))
    }
}

impl<R: NumericRegister> core::iter::Sum for Vector<R> {
    #[inline(always)]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Vector(R::ZERO), Add::add)
    }
}

impl<R: NumericRegister> core::iter::Product for Vector<R> {
    #[inline(always)]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Vector(R::ONE), Mul::mul)
    }
}

impl<R: NumericRegister> num_traits::Bounded for Vector<R> {
    #[inline(always)]
    fn max_value() -> Self {
        Vector(R::MAX)
    }

    #[inline(always)]
    fn min_value() -> Self {
        Vector(R::MIN)
    }
}

#[rustfmt::skip] #[thermite_macros::vector_impl]
impl<R: SignedRegister> SignedVector for Vector<R> {
    const NEG_ONE: Self = Vector(R::NEG_ONE);
    const MIN_POSITIVE: Self = Vector(R::MIN_POSITIVE);

    #[conditional] fn abs(self) -> Self {}

    fn signum(self) -> Self {}

    #[conditional] fn copysign(self, sign: Self) -> Self {}

    fn is_positive(self) -> Self::Mask { Mask(R::is_positive(self.0)) }
    fn is_negative(self) -> Self::Mask { Mask(R::is_negative(self.0)) }

    fn select_negative(self, if_neg: Self, if_pos: Self) -> Self {}
}

#[rustfmt::skip] #[thermite_macros::vector_impl]
impl<R: IntegerRegister> IntegerVector for Vector<R>
where
    R::Element: Denominator,
{
    type Divider = Divider<R::Element>;
    type BranchfreeDivider = BranchfreeDivider<R::Element>;
    type VectorizedDivider = VectorDivider<R>;

    #[conditional] fn mulhi(self, other: Self) -> Self {}
    #[conditional] fn mullo(self, other: Self) -> Self {}

    // fn wrapping_add(self, other: Self) -> Self {}
    // fn wrapping_sub(self, other: Self) -> Self {}
    // fn wrapping_mul(self, other: Self) -> Self {}

    #[conditional] fn saturating_add(self, other: Self) -> Self {}
    #[conditional] fn saturating_sub(self, other: Self) -> Self {}

    #[conditional] fn wrapping_sum(self) -> Self::Element { R::wrapping_sum(self.0) }
    #[conditional] fn wrapping_prod(self) -> Self::Element { R::wrapping_product(self.0) }

    fn create_divider(d: Self::Element) -> Self::Divider { Denominator::to_divider(d) }
    fn create_branchfree_divider(d: Self::Element) -> Self::BranchfreeDivider { Denominator::to_branchfree_divider(d) }

    fn to_divider(self) -> Self::VectorizedDivider { VectorDivider::new(self) }

    #[conditional] fn count_ones(self) -> Self {}
    #[conditional] fn count_zeros(self) -> Self {}
    #[conditional] fn leading_ones(self) -> Self {}
    #[conditional] fn leading_zeros(self) -> Self {}
}

impl<R: IntegerRegister> Div<Divider<R::Element>> for Vector<R> {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Divider<R::Element>) -> Self::Output {
        Self(R::div_branched(self.0, rhs))
    }
}

impl<R: IntegerRegister> Div<BranchfreeDivider<R::Element>> for Vector<R> {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: BranchfreeDivider<R::Element>) -> Self::Output {
        Self(R::div_branchfree(self.0, rhs))
    }
}

impl<R: IntegerRegister> Div<VectorDivider<R>> for Vector<R> {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: VectorDivider<R>) -> Self::Output {
        Self(R::divv_branchfree(self.0, rhs))
    }
}

impl<R: IntegerRegister> DivMasked<Mask<R>, Divider<R::Element>> for Vector<R>
where
    R::Element: Denominator,
{
    #[inline(always)]
    fn div_c(self, mask: Mask<R>, rhs: Divider<R::Element>) -> Self::Output {
        Vector(R::div_branched_c(mask.0, self.0, rhs))
    }

    #[inline(always)]
    fn div_m(self, src: Self, mask: Mask<R>, rhs: Divider<R::Element>) -> Self::Output {
        Vector(R::div_branched_m(src.0, mask.0, self.0, rhs))
    }

    #[inline(always)]
    fn div_z(self, mask: Mask<R>, rhs: Divider<R::Element>) -> Self::Output {
        Vector(R::div_branched_z(mask.0, self.0, rhs))
    }
}

impl<R: IntegerRegister> DivMasked<Mask<R>, BranchfreeDivider<R::Element>> for Vector<R>
where
    R::Element: Denominator,
{
    #[inline(always)]
    fn div_c(self, mask: Mask<R>, rhs: BranchfreeDivider<R::Element>) -> Self::Output {
        Vector(R::div_branchfree_c(mask.0, self.0, rhs))
    }

    #[inline(always)]
    fn div_m(self, src: Self, mask: Mask<R>, rhs: BranchfreeDivider<R::Element>) -> Self::Output {
        Vector(R::div_branchfree_m(src.0, mask.0, self.0, rhs))
    }

    #[inline(always)]
    fn div_z(self, mask: Mask<R>, rhs: BranchfreeDivider<R::Element>) -> Self::Output {
        Vector(R::div_branchfree_z(mask.0, self.0, rhs))
    }
}

impl<R: IntegerRegister> DivMasked<Mask<R>, VectorDivider<R>> for Vector<R>
where
    R::Element: Denominator,
{
    #[inline(always)]
    fn div_c(self, mask: Mask<R>, rhs: VectorDivider<R>) -> Self::Output {
        Vector(R::divv_branchfree_c(mask.0, self.0, rhs))
    }

    #[inline(always)]
    fn div_m(self, src: Self, mask: Mask<R>, rhs: VectorDivider<R>) -> Self::Output {
        Vector(R::divv_branchfree_m(src.0, mask.0, self.0, rhs))
    }

    #[inline(always)]
    fn div_z(self, mask: Mask<R>, rhs: VectorDivider<R>) -> Self::Output {
        Vector(R::divv_branchfree_z(mask.0, self.0, rhs))
    }
}

#[rustfmt::skip] #[thermite_macros::vector_impl]
impl<R: SignedIntegerRegister> SignedIntegerVector for Vector<R>
where
    R::Element: Denominator,
{
    #[conditional] fn srai<const I: i32>(self) -> Self {}
    #[conditional] fn sra(self, count: u32) -> Self {}
    #[conditional] fn srav(self, counts: Self::Unsigned) -> Self {}
    #[conditional] fn avg_floor(self, other: Self) -> Self {}
    #[conditional] fn avg_ceil(self, other: Self) -> Self {}
}

#[rustfmt::skip] #[thermite_macros::vector_impl]
impl<R: UnsignedIntegerRegister> UnsignedIntegerVector for Vector<R>
where
    R::Element: Denominator,
{
    fn is_power_of_two(self) -> Self::Mask { Mask(R::is_power_of_two(self.0)) }

    #[conditional] fn next_power_of_two_m1(self) -> Self {}
    #[conditional] fn ilog2p1(self) -> Self {}
    #[conditional] fn parity(self) -> Self {}
    #[conditional] fn avg(self, other: Self) -> Self {}
}

#[rustfmt::skip] #[thermite_macros::vector_impl]
impl<R: FloatRegister> FloatVector for Vector<R> {
    const HALF: Self = Vector(R::HALF);
    const NEG_ZERO: Self = Vector(R::NEG_ZERO);
    const INFINITY: Self = Vector(R::INFINITY);
    const NEG_INFINITY: Self = Vector(R::NEG_INFINITY);
    const NAN: Self = Vector(R::NAN);
    const EPSILON: Self = Vector(R::EPSILON);

    type ExtendedPrecision = Vector<R::ExtendedPrecision>;

    fn is_infinite(self)     -> Self::Mask { Mask(R::is_infinite(self.0)) }
    fn is_finite(self)       -> Self::Mask { Mask(R::is_finite(self.0)) }
    fn is_nan(self)          -> Self::Mask { Mask(R::is_nan(self.0)) }
    fn is_zero_or_subnormal(self) -> Self::Mask { Mask(R::is_zero_or_subnormal(self.0)) }
    fn is_normal(self)       -> Self::Mask { Mask(R::is_normal(self.0)) }
    fn is_subnormal(self)    -> Self::Mask { Mask(R::is_subnormal(self.0)) }

    const HAS_APPROX_RCP: bool = R::HAS_APPROX_RCP;
    const HAS_APPROX_RSQRT: bool = R::HAS_APPROX_RSQRT;

    #[conditional] fn sqrt(self) -> Self {}
    #[conditional] fn rsqrt(self) -> Self {}
    #[conditional] fn rcp(self) -> Self {}
    #[conditional] fn floor(self) -> Self {}
    #[conditional] fn ceil(self) -> Self {}
    #[conditional] fn round(self) -> Self {}
    #[conditional] fn trunc(self) -> Self {}
    #[conditional] fn fract(self) -> Self {}
    #[conditional] fn mul_sign(self, sign: Self) -> Self {}
    #[conditional] fn signed_zero(self) -> Self {}
    #[conditional] fn next_up(self) -> Self {}
    #[conditional] fn next_down(self) -> Self {}

    fn mix(self, a: Self, b: Self) -> Self { Vector(R::mix(a.0, b.0, self.0)) }

    unsafe fn block_autovectorization(&mut self) {
        unsafe { R::block_autovectorization(&mut self.0) };
    }

    fn with_bits<const N: usize, K: AsFloatVectorWithBitsKernel<Self, N>>(
        values: [Self; N],
        kernel: K,
    ) -> Option<<K as AsFloatVectorWithBitsKernel<Self, N>>::Output> {
        Some(kernel.with_bits(values))
    }
}

#[rustfmt::skip]
impl<R: FloatRegister> FloatVectorWithBits for Vector<R> {
    type SignedBits = Vector<R::SignedBits>;
    type Bits = Vector<R::Bits>;

    const NATIVE_CAP: NativeCapability = R::NATIVE_CAP;

    #[inline(always)] unsafe fn native_ldexp(self, exp: Self::SignedBits) -> Self {
        unsafe { Vector(R::native_ldexp(self.0, exp.0)) }
    }

    #[inline(always)] unsafe fn native_frexp(self) -> (Self, Self::SignedBits) {
        let (mantissa, exp) = unsafe { R::native_frexp(self.0) };
        (Vector(mantissa), Vector(exp))
    }

    #[inline(always)] unsafe fn native_sin_cos<P: Policy>(self) -> (Self, Self) {
        let (sin, cos) = unsafe { R::native_sin_cos::<P>(self.0) };
        (Vector(sin), Vector(cos))
    }

    #[inline(always)] unsafe fn native_sin<P: Policy>(self) -> Self { unsafe { Vector(R::native_sin::<P>(self.0)) } }
    #[inline(always)] unsafe fn native_cos<P: Policy>(self) -> Self { unsafe { Vector(R::native_cos::<P>(self.0)) } }
    #[inline(always)] unsafe fn native_tan<P: Policy>(self) -> Self { unsafe { Vector(R::native_tan::<P>(self.0)) } }
    #[inline(always)] unsafe fn native_exp2<P: Policy>(self) -> Self { unsafe { Vector(R::native_exp2::<P>(self.0)) } }
    #[inline(always)] unsafe fn native_log2<P: Policy>(self) -> Self { unsafe { Vector(R::native_log2::<P>(self.0)) } }
    #[inline(always)] unsafe fn native_exp<P: Policy>(self) -> Self { unsafe { Vector(R::native_exp::<P>(self.0)) } }
    #[inline(always)] unsafe fn native_ln<P: Policy>(self) -> Self { unsafe { Vector(R::native_ln::<P>(self.0)) } }
    #[inline(always)] unsafe fn native_powf<P: Policy>(self, exp: Self) -> Self { unsafe { Vector(R::native_powf::<P>(self.0, exp.0)) } }

    #[inline(always)] fn total_order(self) -> Self::SignedBits { Vector(R::total_order(self.0)) }
    #[inline(always)] fn linear_order(self) -> Self::SignedBits { Vector(R::linear_order(self.0)) }
}

#[rustfmt::skip]
impl<R: LinAlg3Register> LinAlg3Vector for Vector<R> {
    #[inline(always)] fn dot3(self, other: Self) -> Self::Element { R::dot3(self.0, other.0) }

    #[inline(always)]
    fn cross3<const DOP: bool>(self, other: Self) -> Self { Vector(R::cross3::<DOP>(self.0, other.0)) }

    #[inline(always)] fn zero4(self) -> Self { Vector(R::zero4(self.0)) }
    #[inline(always)] fn one4(self) -> Self { Vector(R::one4(self.0)) }
    #[inline(always)] fn min_element3(self) -> Self::Element { R::min_element3(self.0) }
    #[inline(always)] fn max_element3(self) -> Self::Element { R::max_element3(self.0) }
    #[inline(always)] fn sum_elements3(self) -> Self::Element { R::sum_elements3(self.0) }
    #[inline(always)] fn prod_elements3(self) -> Self::Element { R::prod_elements3(self.0) }
}

impl<R: LinAlg4Register> LinAlg4Vector for Vector<R> {
    #[inline(always)]
    fn dot4(self, other: Self) -> Self::Element {
        R::dot4(self.0, other.0)
    }

    #[inline(always)]
    fn quat4_product(self, other: Self) -> Self {
        Vector(R::quat4_product(self.0, other.0))
    }

    #[inline(always)]
    fn quat4_vec3_product<const DOP: bool>(self, vec: Self) -> Self {
        Vector(R::quat4_vec3_product::<DOP>(self.0, vec.0))
    }

    #[inline(always)]
    fn mat4_transpose(m: &[Self; 4]) -> [Self; 4] {
        // SAFETY: transmute &[Vector<R>; 4] to &[Storage<R>; 4] is safe
        // because Vector<R> is repr(transparent) around Storage<R>
        R::mat4_transpose(unsafe { core::mem::transmute(m) }).map(Vector)
    }

    #[inline(always)]
    fn mat4_vec4_product<const COLUMN_MAJOR: bool>(self, m: &[Self; 4]) -> Self {
        Self(R::mat4_vec4_product::<COLUMN_MAJOR>(
            // SAFETY: transmute &[Vector<R>; 4] to &[Storage<R>; 4] is safe
            // because Vector<R> is repr(transparent) around Storage<R>
            unsafe { core::mem::transmute(m) },
            self.0,
        ))
    }

    #[inline(always)]
    fn mat4_product<const COLUMN_MAJOR: bool>(lhs: &[Self; 4], rhs: &[Self; 4]) -> [Self; 4] {
        // SAFETY: transmute &[Vector<R>; 4] to &[Storage<R>; 4] is safe
        // because Vector<R> is repr(transparent) around Storage<R>
        R::mat4_product::<COLUMN_MAJOR>(
            unsafe { core::mem::transmute(lhs) }, //
            unsafe { core::mem::transmute(rhs) },
        )
        .map(Vector)
    }

    #[inline(always)]
    fn mat4_inverse_inplace(m: &mut [Self; 4]) -> bool {
        // SAFETY: transmute &[Vector<R>; 4] to &[Storage<R>; 4] is safe
        // because Vector<R> is repr(transparent) around Storage<R>
        R::mat4_inverse(unsafe { core::mem::transmute(m) })
    }
}

impl<R: Register> VectorWithRegister<R> for Vector<R> {
    #[inline(always)]
    fn into_register(self) -> Storage<R> {
        self.0
    }

    #[inline(always)]
    fn from_register(reg: Storage<R>) -> Self {
        Vector(reg)
    }
}

impl<R> FloatVectorWithRegister for Vector<R>
where
    R: FloatRegister,
{
    type Register = R;
}

impl<R> SignedIntegerVectorWithRegister for Vector<R>
where
    R: SignedIntegerRegister,
{
    type Register = R;
}

impl<R> UnsignedIntegerVectorWithRegister for Vector<R>
where
    R: UnsignedIntegerRegister,
{
    type Register = R;
}

impl<R: Register, B: Register> Extend<Mask<R>> for Mask<B>
where
    B::Mask: ExtendRegister<R::Mask>,
{
    #[inline(always)]
    fn extend(v: Mask<R>) -> Self {
        Mask(B::Mask::extend(v.0))
    }

    #[inline(always)]
    fn narrow(self) -> Mask<R> {
        Mask(B::Mask::narrow(self.0))
    }
}

impl<R: Register, B: Register> Concat<Mask<R>> for Mask<B>
where
    B::Mask: ConcatRegister<R::Mask>,
{
    #[inline(always)]
    fn concat(lo: Mask<R>, hi: Mask<R>) -> Self {
        Mask(B::Mask::concat(lo.0, hi.0))
    }

    #[inline(always)]
    fn split(self) -> (Mask<R>, Mask<R>) {
        let (lo, hi) = B::Mask::split(self.0);
        (Mask(lo), Mask(hi))
    }
}

impl<R: Register, B: Register> Extend<Vector<R>> for Vector<B>
where
    B: ExtendRegister<R>,
{
    #[inline(always)]
    fn extend(v: Vector<R>) -> Self {
        Vector(B::extend(v.0))
    }

    #[inline(always)]
    fn narrow(self) -> Vector<R> {
        Vector(B::narrow(self.0))
    }
}

impl<R: Register, B: Register> Concat<Vector<R>> for Vector<B>
where
    B: ConcatRegister<R>,
{
    #[inline(always)]
    fn concat(lo: Vector<R>, hi: Vector<R>) -> Self {
        Vector(B::concat(lo.0, hi.0))
    }

    #[inline(always)]
    fn split(self) -> (Vector<R>, Vector<R>) {
        let (lo, hi) = B::split(self.0);
        (Vector(lo), Vector(hi))
    }
}

impl<R: PartialOrdRegister> PartialEq for Vector<R> {
    /// Compare two vectors for equality, returning true only if all elements are equal.
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        Mask::<R>(R::eq(self.0, other.0)).all()
    }

    /// Compare two vectors for inequality, returning true if any element is not equal.
    #[allow(clippy::partialeq_ne_impl)] // sometimes might have better underlying implementation
    #[inline(always)]
    fn ne(&self, other: &Self) -> bool {
        Mask::<R>(R::ne(self.0, other.0)).any()
    }
}

impl<R: Register> Index<usize> for Vector<R> {
    type Output = R::Element;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &R::as_array(&self.0)[index]
    }
}

impl<R: Register> IndexMut<usize> for Vector<R> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut R::as_array_mut(&mut self.0)[index]
    }
}

impl<R: NumericRegister> Zero for Vector<R> {
    /// Returns true if **all** elements in the vector are zero.
    #[inline(always)]
    fn is_zero(&self) -> bool {
        Mask::<R>(R::eq(self.0, R::ZERO)).all()
    }

    #[inline(always)]
    fn set_zero(&mut self) {
        self.0 = R::ZERO;
    }

    #[inline(always)]
    fn zero() -> Self {
        Self::ZERO
    }
}

impl<R: NumericRegister> One for Vector<R> {
    /// Returns true if **all** elements in the vector are one.
    #[inline(always)]
    fn is_one(&self) -> bool {
        Mask::<R>(R::eq(self.0, R::ONE)).all()
    }

    #[inline(always)]
    fn set_one(&mut self) {
        self.0 = R::ONE;
    }

    #[inline(always)]
    fn one() -> Self {
        Self::ONE
    }
}

impl<R: IntegerRegister> SaturatingAdd for Vector<R> {
    #[inline(always)]
    fn saturating_add(&self, v: &Self) -> Self {
        Self(R::saturating_add(self.0, v.0))
    }
}

impl<R: IntegerRegister> SaturatingSub for Vector<R> {
    #[inline(always)]
    fn saturating_sub(&self, v: &Self) -> Self {
        Self(R::saturating_sub(self.0, v.0))
    }
}

impl<R: IntegerRegister> Saturating for Vector<R> {
    #[inline(always)]
    fn saturating_add(self, v: Self) -> Self {
        Self(R::saturating_add(self.0, v.0))
    }

    #[inline(always)]
    fn saturating_sub(self, v: Self) -> Self {
        Self(R::saturating_sub(self.0, v.0))
    }
}

impl<R: IntegerRegister> WrappingAdd for Vector<R> {
    #[inline(always)]
    fn wrapping_add(&self, v: &Self) -> Self {
        Self(R::add(self.0, v.0))
    }
}

impl<R: IntegerRegister> WrappingSub for Vector<R> {
    #[inline(always)]
    fn wrapping_sub(&self, v: &Self) -> Self {
        Self(R::sub(self.0, v.0))
    }
}

impl<R: IntegerRegister> WrappingMul for Vector<R> {
    #[inline(always)]
    fn wrapping_mul(&self, v: &Self) -> Self {
        Self(R::mul(self.0, v.0))
    }
}

/*
#[rustfmt::skip]
macro_rules! impl_swizzle4 {
    (@ x) => { 0 };
    (@ y) => { 1 };
    (@ z) => { 2 };
    (@ w) => { 3 };

    (IMPL $a:ident $b:ident $c:ident $d:ident) => {paste::paste! {
        #[inline(always)]
        fn [<$a $b $c $d>](self) -> Self {
            const IMM8: i32 = MM_SHUFFLE!(
                impl_swizzle4!(@ $d),
                impl_swizzle4!(@ $c),
                impl_swizzle4!(@ $b),
                impl_swizzle4!(@ $a)
            );

            Self(R::permute::<IMM8>(self.0))
        }
    }};

    (DECL $(#[$meta:meta])* $a:ident $b:ident $c:ident $d:ident) => {paste::paste! {
        #[allow(missing_docs)]
        $(#[$meta])* fn [<$a $b $c $d>](self) -> Self;
    }};

    ($( $(#[$meta:meta])* [$a:ident $b:ident $c:ident $d:ident]),*) => {
        /// Only available for 4-lane vectors, this allows human-readable swizzle/permutations
        /// of the vector.
        pub trait Swizzle4 { $(impl_swizzle4!(DECL $(#[$meta])* $a $b $c $d);)* }

        /// Implements 4-lane swizzling for vectors.
        impl<R: PermuteRegister<Lanes = generic_array::typenum::consts::U4>> Swizzle4 for Vector<R> {
            $(impl_swizzle4!(IMPL $a $b $c $d);)*
        }
    }
}

#[rustfmt::skip]
macro_rules! impl_swizzle3 {
    (IMPL $a:ident $b:ident $c:ident) => {paste::paste! {
        #[inline(always)]
        fn [<$a $b $c>](self) -> Self {
            const IMM8: i32 = MM_SHUFFLE!(
                3, // 4th lane is unchanged
                impl_swizzle4!(@ $c),
                impl_swizzle4!(@ $b),
                impl_swizzle4!(@ $a)
            );

            Self(R::permute::<IMM8>(self.0))
        }
    }};

    (DECL $(#[$meta:meta])* $a:ident $b:ident $c:ident) => {paste::paste! {
        #[allow(missing_docs)]
        $(#[$meta])* fn [<$a $b $c>](self) -> Self;
    }};

    ($( $(#[$meta:meta])* [$a:ident $b:ident $c:ident]),*) => {
        /// Only available for "3-lane" (ignoring 4th lane) [`LinAlg3Register`] vectors,
        /// this allows human-readable swizzle/permutations of the vector. Permutations
        /// will ignore the 4th lane of the register, leaving it unchanged.
        pub trait Swizzle3 { $(impl_swizzle3!(DECL $(#[$meta])* $a $b $c);)* }

        /// Implements 3-lane swizzling for vectors support 3-lane linear algebra operations.
        impl<R: LinAlg3Register + PermuteRegister> Swizzle3 for Vector<R> {
            $(impl_swizzle3!(IMPL $a $b $c);)*
        }
    }
}

impl_swizzle3! {
    [x y z], [x x x], [x x y], [x x z], [x y x], [x y y], [x z x], [x z y], [x z z],
    [y x x], [y x y], [y x z], [y y x], [y y y], [y y z], [y z x], [y z y], [y z z],
    [z x x], [z x y], [z x z], [z y x], [z y y], [z y z], [z z x], [z z y], [z z z]
}

impl_swizzle4! {
    [x y z w], [x x x x], [x x x y], [x x x z], [x x x w], [x x y x], [x x y y], [x x y z],
    [x x y w], [x x z x], [x x z y], [x x z z], [x x z w], [x x w x], [x x w y], [x x w z],
    [x x w w], [x y x x], [x y x y], [x y x z], [x y x w], [x y y x], [x y y y], [x y y z],
    [x y y w], [x y z x], [x y z y], [x y z z], [x y w x], [x y w y], [x y w z], [x y w w],
    [x z x x], [x z x y], [x z x z], [x z x w], [x z y x], [x z y y], [x z y z], [x z y w],
    [x z z x], [x z z y], [x z z z], [x z z w], [x z w x], [x z w y], [x z w z], [x z w w],
    [x w x x], [x w x y], [x w x z], [x w x w], [x w y x], [x w y y], [x w y z], [x w y w],
    [x w z x], [x w z y], [x w z z], [x w z w], [x w w x], [x w w y], [x w w z], [x w w w],
    [y x x x], [y x x y], [y x x z], [y x x w], [y x y x], [y x y y], [y x y z], [y x y w],
    [y x z x], [y x z y], [y x z z], [y x z w], [y x w x], [y x w y], [y x w z], [y x w w],
    [y y x x], [y y x y], [y y x z], [y y x w], [y y y x], [y y y y], [y y y z], [y y y w],
    [y y z x], [y y z y], [y y z z], [y y z w], [y y w x], [y y w y], [y y w z], [y y w w],
    [y z x x], [y z x y], [y z x z], [y z x w], [y z y x], [y z y y], [y z y z], [y z y w],
    [y z z x], [y z z y], [y z z z], [y z z w], [y z w x], [y z w y], [y z w z], [y z w w],
    [y w x x], [y w x y], [y w x z], [y w x w], [y w y x], [y w y y], [y w y z], [y w y w],
    [y w z x], [y w z y], [y w z z], [y w z w], [y w w x], [y w w y], [y w w z], [y w w w],
    [z x x x], [z x x y], [z x x z], [z x x w], [z x y x], [z x y y], [z x y z], [z x y w],
    [z x z x], [z x z y], [z x z z], [z x z w], [z x w x], [z x w y], [z x w z], [z x w w],
    [z y x x], [z y x y], [z y x z], [z y x w], [z y y x], [z y y y], [z y y z], [z y y w],
    [z y z x], [z y z y], [z y z z], [z y z w], [z y w x], [z y w y], [z y w z], [z y w w],
    [z z x x], [z z x y], [z z x z], [z z x w], [z z y x], [z z y y], [z z y z], [z z y w],
    [z z z x], [z z z y], [z z z z], [z z z w], [z z w x], [z z w y], [z z w z], [z z w w],
    [z w x x], [z w x y], [z w x z], [z w x w], [z w y x], [z w y y], [z w y z], [z w y w],
    [z w z x], [z w z y], [z w z z], [z w z w], [z w w x], [z w w y], [z w w z], [z w w w],
    [w x x x], [w x x y], [w x x z], [w x x w], [w x y x], [w x y y], [w x y z], [w x y w],
    [w x z x], [w x z y], [w x z z], [w x z w], [w x w x], [w x w y], [w x w z], [w x w w],
    [w y x x], [w y x y], [w y x z], [w y x w], [w y y x], [w y y y], [w y y z], [w y y w],
    [w y z x], [w y z y], [w y z z], [w y z w], [w y w x], [w y w y], [w y w z], [w y w w],
    [w z x x], [w z x y], [w z x z], [w z x w], [w z y x], [w z y y], [w z y z], [w z y w],
    [w z z x], [w z z y], [w z z z], [w z z w], [w z w x], [w z w y], [w z w z], [w z w w],
    [w w x x], [w w x y], [w w x z], [w w x w], [w w y x], [w w y y], [w w y z], [w w y w],
    [w w z x], [w w z y], [w w z z], [w w z w], [w w w x], [w w w y], [w w w z], [w w w w]
}
 */

#[cfg(feature = "partial-ord")]
impl<R: PartialOrdRegister> PartialOrd for Vector<R> {
    /// Partial comparison between two vectors, returning `None` if
    /// the vectors are not fully ordered. Only returns `Some(Ordering)` if
    /// all lanes are less than, greater than, or equal.
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        let is_less = R::all(R::lt(self.0, other.0));
        let is_greater = R::all(R::gt(self.0, other.0));
        let is_equal = R::all(R::eq(self.0, other.0));

        match (is_less, is_greater, is_equal) {
            (true, false, false) => Some(core::cmp::Ordering::Less),
            (false, true, false) => Some(core::cmp::Ordering::Greater),
            (false, false, true) => Some(core::cmp::Ordering::Equal),
            _ => None,
        }
    }
}

// macro_rules! impl_unsigned_pow {
//     ($($t:ty),* $(,)?) => {$(
//         impl<R: NumericRegister> num_traits::Pow<$t> for Vector<R> {
//             type Output = Self;

//             #[inline(always)]
//             fn pow(self, mut e: $t) -> Self::Output {
//                 let mut res = Self::ONE;
//                 let mut x = self;

//                 while e != 0 {
//                     if e & 1 != 0 {
//                         res *= x;
//                     }

//                     x *= x;
//                     e >>= 1;
//                 }

//                 res
//             }
//         })*
//     };
// }

// impl_unsigned_pow!(u8, u16, u32, u64, usize);

#[cfg(feature = "rand")]
const _: () = {
    use generic_array::sequence::GenericSequence;
    use rand::{Fill, distr::Distribution};

    impl<R: Register> Distribution<Vector<R>> for rand::distr::Uniform<R::Element>
    where
        rand::distr::Uniform<R::Element>: Distribution<R::Element>,
        R::Element: rand::distr::uniform::SampleUniform,
    {
        #[inline(always)]
        fn sample<Rng: rand::Rng + ?Sized>(&self, rng: &mut Rng) -> Vector<R> {
            Vector::from_array(GenericArray::generate(|_| self.sample(rng)))
        }
    }

    macro_rules! impl_distr {
        ($($distr:ident),* $(,)?) => {$(
            impl<R: Register> Distribution<Vector<R>> for rand::distr::$distr
            where
                rand::distr::$distr: Distribution<R::Element>,
            {
                #[inline(always)]
                fn sample<Rng: rand::Rng + ?Sized>(&self, rng: &mut Rng) -> Vector<R> {
                    Vector::from_array(GenericArray::generate(|_| self.sample(rng)))
                }
            }
        )*};
    }

    impl_distr!(Open01, OpenClosed01, StandardUniform);

    impl<R: Register> Fill for Vector<R>
    where
        [R::Element]: Fill,
    {
        #[inline(always)]
        fn fill<Rng: rand::Rng + ?Sized>(&mut self, rng: &mut Rng) {
            Fill::fill(self.as_mut_slice(), rng);
        }
    }
};
