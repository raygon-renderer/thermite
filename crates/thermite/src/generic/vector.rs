use core::ops::{Add, Div, Mul, Neg, Not, Rem, Sub};

use super::*;

use crate::{
    generic::ops::{DivMasked, Square, SquareMasked},
    register::{
        BitCastRegister, BitshiftRegister, BitwiseRegister, CastMaskRegister, CastRegister, ConcatRegister, Element,
        ExtendRegister, FloatElement, FloatRegister, IndexableRegister, IntegerRegister, Lanes, LinAlg3Register,
        LinAlg4Register, NumericRegister, PartialOrdRegister, Register, SignedIntegerRegister, SignedRegister, Storage,
        SwizzleRegister, UnsignedIntegerRegister,
    },
};

impl<FROM, INTO> CastVector<Vector<FROM>> for Vector<INTO>
where
    FROM: Register + CastRegister<INTO>,
    INTO: Register + CastRegister<FROM>,
{
    #[inline(always)]
    fn cast_from(from: Vector<FROM>) -> Self {
        Vector::<INTO>::from(from)
    }

    fn cast_into(self) -> Vector<FROM> {
        Vector::<FROM>::from(self)
    }

    #[inline(always)]
    fn fast_cast_from(from: Vector<FROM>) -> Self {
        Vector::<INTO>::fast_from(from)
    }

    #[inline(always)]
    fn fast_cast_into(self) -> Vector<FROM> {
        Vector::<FROM>::fast_from(self)
    }
}

impl<FROM, INTO> BitCastVector<Vector<FROM>> for Vector<INTO>
where
    FROM: Register,
    INTO: Register + BitCastRegister<FROM>,
{
    #[inline(always)]
    fn from_bits(bits: Vector<FROM>) -> Self {
        Vector::<INTO>::from_bits(bits)
    }
}

impl<FROM, INTO> CastMask<Mask<FROM>> for Mask<INTO>
where
    FROM: Register,
    INTO: Register<Mask: CastMaskRegister<FROM::Mask>>,
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

impl<R: Register> GenericMask for Mask<R> {
    const FALSY: Self = Mask::<R>::FALSY;
    const TRUTHY: Self = Mask::<R>::TRUTHY;

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
    fn native_bitmask(&self) -> Option<u64> {
        self.native_bitmask()
    }

    #[inline(always)]
    fn bitmask(&self) -> BitArray<impl BitViewSized<Store = u32>> {
        self.bitmask()
    }

    #[inline(always)]
    fn ternlog<const IMM: i32>(a: Self, b: Self, c: Self) -> Self {
        Mask(<R::Mask as BitwiseRegister>::ternlog::<IMM>(a.0, b.0, c.0))
    }
}

#[rustfmt::skip] #[thermite_macros::vector_impl]
impl<R: Register> GenericVector for Vector<R> {
    type Element = R::Element;

    const EMPTY: Self = Vector(R::EMPTY);
    const LANES: usize = <R::Lanes as generic_array::typenum::Unsigned>::USIZE;
    const ISA: InstructionSet = R::ISA;

    type Lanes = R::Lanes;

    type Unsigned = Vector<R::Unsigned>;
    type Signed = Vector<R::Signed>;

    type Mask = Mask<R>;

    fn splat_const<C>() -> Self where C: SplatConst<Self::Element> {
        const { Self::splat_const(C::VALUE) }
    }

    #[masked] fn splat(value: Self::Element) -> Self { Vector(R::splat(value)) }

    fn single(value: Self::Element) -> Self { Vector(R::single(value)) }

    #[conditional] fn broadcast<const I: usize>(self) -> Self {}
    #[conditional] fn broadcastv(self, idx: usize) -> Self {}

    fn extract<const I: usize>(self) -> Self::Element { R::extract::<I>(self.0) }
    fn extractv(self, idx: usize) -> Self::Element { self.as_slice()[idx] }

    fn insert<const I: usize>(self, value: Self::Element) -> Self { Vector(R::insert::<I>(self.0, value)) }

    fn insertv(mut self, idx: usize, value: Self::Element) -> Self {
        let arr = self.as_mut_slice();
        arr[idx] = value;
        self
    }

    #[conditional] fn reverse(self) -> Self {}
    #[conditional] fn swap_bytes(self) -> Self {}

    // The arguments of these are reversed for the register
    fn z(self, mask: Self::Mask) -> Self { Vector(R::z(mask.0, self.0)) }
    fn nz(self, mask: Self::Mask) -> Self { Vector(R::nz(mask.0, self.0)) }

    const HAS_SIMPLE_UNPACK: bool = R::HAS_SIMPLE_UNPACK;

    fn unpack(self, other: Self) -> (Self, Self) {
        let (lo, hi) = R::unpack(self.0, other.0);
        (Vector(lo), Vector(hi))
    }

    fn map<F>(self, f: F) -> Self where F: Fn(Self::Element) -> Self::Element { Vector(R::map(self.0, f)) }
    fn fold<F>(self, init: Self::Element, f: F) -> Self::Element where F: Fn(Self::Element, Self::Element) -> Self::Element { R::fold(init, self.0, f) }
    fn reduce<F>(self, f: F) -> Self::Element where F: Fn(Self::Element, Self::Element) -> Self::Element { R::reduce(self.0, f) }

    #[masked] unsafe fn load(ptr: *const Self::Element) -> Self {}

    unsafe fn load_unaligned(ptr: *const Self::Element) -> Self { unsafe { Vector(R::load_unaligned(ptr)) } }
    unsafe fn load_streaming(ptr: *const Self::Element) -> Self { unsafe { Vector(R::load_stream(ptr)) } }

    unsafe fn store(self, ptr: *mut Self::Element) { unsafe { R::store(ptr, self.0) } }
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
impl<R: NumericRegister> NumericVector for Vector<R>
where
    R::Element: num_traits::Num,
{
    const ZERO: Self = Vector(R::ZERO);
    const ONE: Self = Vector(R::ONE);
    const TWO: Self = Vector(R::TWO);
    const MIN: Self = Vector(R::MIN);
    const MAX: Self = Vector(R::MAX);

    fn is_zero(self) -> Self::Mask { self.cmp_eq(Self::ZERO) }

    #[conditional] fn min(self, other: Self) -> Self {}
    #[conditional] fn max(self, other: Self) -> Self {}

    fn clamp(self, min: Self, max: Self) -> Self { self.min(max).max(min) }

    fn min_element(self) -> Self::Element { R::min_element(self.0) }
    fn max_element(self) -> Self::Element { R::max_element(self.0) }

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

impl<R: NumericRegister> NumVector for Vector<R> where R::Element: num_traits::Num + num_traits::NumCast {}

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

impl<R: SignedRegister> NumSignedVector for Vector<R> where R::Element: num_traits::Signed + num_traits::NumCast {}

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

    const HAS_NATIVE_LDEXP: bool = R::HAS_NATIVE_LDEXP;
    const HAS_NATIVE_FREXP: bool = R::HAS_NATIVE_FREXP;

    #[inline(always)] unsafe fn native_ldexp(self, exp: Self::SignedBits) -> Self {
        unsafe { Vector(R::native_ldexp(self.0, exp.0)) }
    }

    #[inline(always)] unsafe fn native_frexp(self) -> (Self, Self::SignedBits) {
        let (mantissa, exp) = unsafe { R::native_frexp(self.0) };
        (Vector(mantissa), Vector(exp))
    }

    #[inline(always)] fn total_order(self) -> Self::SignedBits { Vector(R::total_order(self.0)) }
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
