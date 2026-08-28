#![warn(missing_docs, clippy::missing_safety_doc)]

//! Vector type wrapping low-level registers with a vector-like interface.
//!
//! Most vector features are provided by the [`GenericVector`](crate::generic) traits,
//! but this is the underlying type that most vectors are based on, using low-level
//! registers for various architectures.

use super::ops::*;
use super::*;

use crate::{
    divider::{BranchfreeDivider, Denominator, Divider, vector::VectorDivider},
    mask::{CastMask, GenericSelectable, Mask},
    math::policy::Policy,
    register::{
        self, BitCastRegister, BitshiftRegister, BitwiseRegister, CastRegister, ConcatRegister, ExtendRegister,
        FloatRegister, IndexableRegister, IntegerRegister, LinAlg3Register, LinAlg4Register, NewRegister,
        NumericRegister, PartialOrdRegister, Register, SignedIntegerRegister, SignedRegister, Storage,
        UnsignedIntegerRegister,
    },
};

use core::ops::{Add, Div, Index, IndexMut, Mul};

use num_traits::{One, Saturating, SaturatingAdd, SaturatingSub, WrappingAdd, WrappingMul, WrappingSub, Zero};

use generic_array::GenericArray;

// pub mod streaming;
// pub mod unaligned;

/// SIMD Vector type.
///
/// This wraps a low-level register type and provides a vector-like interface, including
/// operator overloading and element-wise operations.
#[repr(transparent)]
pub struct Vector<R: Register>(#[doc(hidden)] pub Storage<R>);

#[thermite_macros::inline_always]
impl<R: Register> Clone for Vector<R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: Register> Copy for Vector<R> {}

/// `[Vector<R>; N]` -> `[Storage<R>; N]`
#[inline(always)]
const fn wrap_n<R: Register, const N: usize>(values: [Vector<R>; N]) -> [Storage<R>; N] {
    // SAFETY: Vector<R> is repr(transparent) around Storage<R>.
    unsafe { core::mem::transmute_copy(&values) }
}

/// The inverse of [`wrap_n`].
#[inline(always)]
const fn unwrap_n<R: Register, const N: usize>(values: [Storage<R>; N]) -> [Vector<R>; N] {
    // SAFETY: Vector<R> is repr(transparent) around Storage<R>.
    unsafe { core::mem::transmute_copy(&values) }
}

const _: () = {
    use core::fmt;

    impl<R: Register> fmt::Debug for Vector<R> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let mut t = f.debug_tuple("Vector");

            for v in R::as_slice(&self.0) {
                t.field(&v);
            }

            t.finish()
        }
    }
};

impl<R: Register> const_default::ConstDefault for Vector<R> {
    const DEFAULT: Self = Self::EMPTY;
}

#[thermite_macros::inline_always]
impl<R: Register> Default for Vector<R> {
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
}

#[thermite_macros::inline_always]
impl<FROM, INTO> CastVector<Vector<FROM>> for Vector<INTO>
where
    FROM: Register + CastRegister<INTO>,
    INTO: Register + CastRegister<FROM>,
{
    fn cast_from(from: Vector<FROM>) -> Self {
        if const { strict_saturates::<FROM>() } {
            Vector(<INTO as CastRegister<FROM>>::saturating_cast_from(from.0))
        } else {
            Vector(<INTO as CastRegister<FROM>>::cast_from(from.0))
        }
    }

    fn cast_into(self) -> Vector<FROM> {
        if const { strict_saturates::<INTO>() } {
            Vector(<FROM as CastRegister<INTO>>::saturating_cast_from(self.0))
        } else {
            Vector(<FROM as CastRegister<INTO>>::cast_from(self.0))
        }
    }

    fn fast_cast_into(self) -> Vector<FROM> {
        Vector(<FROM as CastRegister<INTO>>::fast_cast_from(self.0))
    }

    fn saturating_cast_from(from: Vector<FROM>) -> Self {
        Vector(<INTO as CastRegister<FROM>>::saturating_cast_from(from.0))
    }

    fn fast_cast_from(from: Vector<FROM>) -> Self {
        Vector(<INTO as CastRegister<FROM>>::fast_cast_from(from.0))
    }
}

/// The whole `strict_ieee754` cast redirect, in one place.
///
/// `cast` is only allowed to diverge from `as` where the SOURCE can hold a NaN
/// or a magnitude the destination cannot represent, which is to say where the
/// source is a float. `strict_ieee754` closes that gap by sending those casts to
/// the saturating lowering, which is `as`-exact on every input.
///
/// Keyed on the source element rather than the destination, and deliberately
/// not on "is the destination an integer": an integer source narrowing to a
/// smaller integer is _already_ exactly `as`, because `as` wraps there and so
/// does `cast_from`. Redirecting those would clamp where the language wraps,
/// turning the feature into a correctness regression. Float to float lands here
/// too and is harmless, since those pairs have no distinct saturating lowering
/// and the default sends them straight back to `cast_from`.
#[inline(always)]
const fn strict_saturates<R: Register>() -> bool {
    cfg!(feature = "strict_ieee754") && <R::Element as crate::element::Element>::IS_FLOAT
}

#[thermite_macros::inline_always]
impl<FROM, INTO> BitCastVector<Vector<FROM>> for Vector<INTO>
where
    FROM: Register,
    INTO: Register + BitCastRegister<FROM>,
{
    fn from_bits(bits: Vector<FROM>) -> Self {
        Vector(<INTO as BitCastRegister<FROM>>::from_bits(bits.0))
    }
}

#[thermite_macros::inline_always]
impl<S, R, FR, B> PackedFloatVector<S, Vector<FR>> for Vector<R>
where
    S: crate::element::float::spec::FloatSpec,
    B: CastRegister<R>,
    FR: FloatRegister<Element = f32, Lanes = R::Lanes, Bits = B>,
    R: register::PackedFloatRegister<S, FR> + CastRegister<B>,
{
    fn pack(values: Vector<FR>) -> Self {
        Vector(<R as register::PackedFloatRegister<S, FR>>::pack(values.0))
    }

    fn unpack(self) -> Vector<FR> {
        Vector(<R as register::PackedFloatRegister<S, FR>>::unpack(self.0))
    }
}

#[thermite_macros::inline_always]
impl<R, W> Sad16Vector<Vector<W>> for Vector<R>
where
    W: register::UnsignedIntegerRegister<Element = u16>,
    R: register::Sad16Register<W>,
{
    fn sad16(self, other: Self) -> Vector<W> {
        Vector(<R as register::Sad16Register<W>>::sad16(self.0, other.0))
    }
}

#[thermite_macros::inline_always]
impl<R, W> Sad32Vector<Vector<W>> for Vector<R>
where
    W: register::UnsignedIntegerRegister<Element = u32>,
    R: register::Sad32Register<W>,
{
    fn sad32(self, other: Self) -> Vector<W> {
        Vector(<R as register::Sad32Register<W>>::sad32(self.0, other.0))
    }

    fn sad32_accum(self, acc: Vector<W>, other: Self) -> Vector<W> {
        Vector(<R as register::Sad32Register<W>>::sad32_accum(acc.0, self.0, other.0))
    }
}

#[thermite_macros::inline_always]
impl<R, W> Sad64Vector<Vector<W>> for Vector<R>
where
    W: register::UnsignedIntegerRegister<Element = u64>,
    R: register::Sad64Register<W>,
{
    fn sad64(self, other: Self) -> Vector<W> {
        Vector(<R as register::Sad64Register<W>>::sad64(self.0, other.0))
    }

    fn sad64_accum(self, acc: Vector<W>, other: Self) -> Vector<W> {
        Vector(<R as register::Sad64Register<W>>::sad64_accum(acc.0, self.0, other.0))
    }
}

#[thermite_macros::inline_always]
impl<R> GenericSelectable for Vector<R>
where
    R: Register,
{
    type SelectableMask = Mask<R>;

    fn select<M>(mask: M, t: Self, f: Self) -> Self
    where
        Mask<R>: CastMask<M>,
    {
        Vector(R::blendv(Mask::mask_from(mask).0, f.0, t.0))
    }
}

impl<R: Register> crate::simd::HasIsa for Vector<R> {
    type Native = R::Native;

    const ISA: InstructionSet = R::ISA;
}

#[doc(hidden)]
pub struct SplatVectorImpl;
#[doc(hidden)]
pub struct NewVectorImpl;

impl<T, R: Register> VectorValue<T, Vector<R>> for SplatVectorImpl
where
    T: SplatConst<R::Element>,
{
    const VALUE: Vector<R> = const { Vector(register::reg_splat::<R>(T::VALUE)) };
}

impl<T, R: Register> VectorValue<T, Vector<R>> for NewVectorImpl
where
    T: NewConst<R::Element, R::Lanes>,
{
    const VALUE: Vector<R> = const {
        Vector(<<R as NewRegister<R::Element, R::Lanes, Storage<R>>>::New<T> as VectorValue<T, Storage<R>>>::VALUE)
    };
}

impl<R: Register> SplatVector<R::Element> for Vector<R> {
    type Splat<T: SplatConst<R::Element>> = SplatVectorImpl;
}

impl<R: Register> NewVector<R::Element, R::Lanes> for Vector<R> {
    type New<T: NewConst<R::Element, R::Lanes>> = NewVectorImpl;
}

#[thermite_macros::inline_always]
impl<R: Register> Interleave for Vector<R> {
    fn interleave(self, other: Self) -> (Self, Self) {
        let (a, b) = R::interleave(self.0, other.0);
        (Vector(a), Vector(b))
    }

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

    fn into_array(self) -> GenericArray<R::Element, R::Lanes> {
        // Spill through the store machinery rather than borrowing the storage as
        // an array, so a future runtime-length backend can write `lanes()`
        // elements into the (max-sized) buffer. The unaligned store is required:
        // GenericArray has element alignment, not register alignment.
        let mut arr: GenericArray<R::Element, R::Lanes> = unsafe { core::mem::zeroed() };
        // SAFETY: `arr` is exactly `Lanes` elements of `Element`.
        unsafe { R::store_unaligned(arr.as_mut_slice().as_mut_ptr(), self.0) };
        arr
    }

    #[masked] fn splat(value: Self::Element) -> Self { Vector(R::splat(value)) }

    fn single(value: Self::Element) -> Self { Vector(R::single(value)) }

    #[conditional] fn broadcast<const I: usize>(self) -> Self {}
    #[conditional] fn broadcastv(self, idx: usize) -> Self {}

    fn extract<const I: usize>(self) -> Self::Element { R::extract::<I>(self.0) }
    fn extractv(self, idx: usize) -> Self::Element { R::as_slice(&self.0)[idx] }
    fn last_element(self) -> Self::Element { R::last_element(self.0) }

    fn insert<const I: usize>(self, value: Self::Element) -> Self { Vector(R::insert::<I>(self.0, value)) }

    fn insertv(mut self, idx: usize, value: Self::Element) -> Self {
        R::as_mut_slice(&mut self.0)[idx] = value;
        self
    }

    fn permutev(self, indices: Self::Unsigned) -> Self { Vector(R::permutev(self.0, indices.0)) }
    fn swizzle(self, other: Self, indices: Self::Unsigned) -> Self { Vector(R::swizzle(self.0, other.0, indices.0)) }

    unsafe fn lookup_unchecked(values: &[Self::Element], indices: Self::Unsigned) -> Self {
        unsafe { Self(R::lookup(values, indices.0)) }
    }

    #[conditional] fn reverse(self) -> Self {}
    #[conditional] fn swap_bytes(self) -> Self {}

    fn compress(self, mask: Self::Mask) -> Self { Vector(R::compress(self.0, mask.0)) }
    fn compress_z(self, mask: Self::Mask) -> Self { Vector(R::compress_z(self.0, mask.0)) }
    fn compress_m(self, src: Self, mask: Self::Mask) -> Self { Vector(R::compress_m(src.0, mask.0, self.0)) }
    fn expand(self, mask: Self::Mask) -> Self { Vector(R::expand(self.0, mask.0)) }
    fn expand_z(self, mask: Self::Mask) -> Self { Vector(R::expand_z(self.0, mask.0)) }
    fn expand_m(self, src: Self, mask: Self::Mask) -> Self { Vector(R::expand_m(src.0, mask.0, self.0)) }

    // The `_n` family delegates straight to the register's shared-plan forms.
    // `Vector<R>` is `#[repr(transparent)]`, but the array conversions are
    // written as hand-rolled loops over `.0` rather than `array::map`/`from_fn`
    // (which do not inline in `target_feature` code) or a transmute. The loops
    // SROA away and cost nothing.
    fn compress_n<const N: usize>(values: [Self; N], mask: Self::Mask) -> [Self; N] { unwrap_n(R::compress_n::<N>(wrap_n(values), mask.0)) }
    fn compress_z_n<const N: usize>(values: [Self; N], mask: Self::Mask) -> [Self; N] { unwrap_n(R::compress_z_n::<N>(wrap_n(values), mask.0)) }
    fn expand_n<const N: usize>(values: [Self; N], mask: Self::Mask) -> [Self; N] { unwrap_n(R::expand_n::<N>(wrap_n(values), mask.0)) }
    fn expand_z_n<const N: usize>(values: [Self; N], mask: Self::Mask) -> [Self; N] { unwrap_n(R::expand_z_n::<N>(wrap_n(values), mask.0)) }
    fn align<const OFFSET: usize>(self, other: Self) -> Self { Vector(R::align::<OFFSET>(self.0, other.0)) }
    const HAS_NATIVE_ALIGN: bool = R::HAS_NATIVE_ALIGN;

    // The arguments of these are reversed for the register
    fn zz(self, mask: Self::Mask) -> Self { Vector(R::zz(mask.0, self.0)) }
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

    fn interleave_by<const GROUP: usize>(self, other: Self) -> (Self, Self) {
        let (a, b) = R::interleave_by::<GROUP>(self.0, other.0);
        (Vector(a), Vector(b))
    }

    fn deinterleave_by<const GROUP: usize>(self, other: Self) -> (Self, Self) {
        let (a, b) = R::deinterleave_by::<GROUP>(self.0, other.0);
        (Vector(a), Vector(b))
    }

    fn interleave_radix<const N: usize>(inputs: [Self; N]) -> [Self; N] { unwrap_n(R::interleave_radix::<N>(wrap_n(inputs))) }
    fn deinterleave_radix<const N: usize>(inputs: [Self; N]) -> [Self; N] { unwrap_n(R::deinterleave_radix::<N>(wrap_n(inputs))) }
    fn deinterleave_radix_by<const N: usize, const GROUP: usize>(inputs: [Self; N]) -> [Self; N] { unwrap_n(R::deinterleave_radix_by::<N, GROUP>(wrap_n(inputs))) }
    fn interleave_radix_by<const N: usize, const GROUP: usize>(inputs: [Self; N]) -> [Self; N] { unwrap_n(R::interleave_radix_by::<N, GROUP>(wrap_n(inputs))) }

    unsafe fn load_deinterleaved<const N: usize>(ptr: *const Self::Element) -> [Self; N] { unwrap_n(unsafe { R::load_deinterleaved::<N>(ptr) }) }
    unsafe fn store_interleaved<const N: usize>(ptr: *mut Self::Element, values: [Self; N]) { unsafe { R::store_interleaved::<N>(ptr, wrap_n(values)) } }

    // Overrides `GenericVector`'s lane-wise record defaults with the register
    // engine. The outer loops are hand-rolled for the same reason `wrap_n` is,
    // since `array::map` does not inline in `#[target_feature]` code.
    unsafe fn load_deinterleaved_arrays<const M: usize, const C: usize>(ptr: *const Self::Element) -> [[Self; C]; M] {
        let records = unsafe { R::load_deinterleaved_arrays::<M, C>(ptr) };

        let mut out = [[Vector(R::EMPTY); C]; M];
        let mut j = 0;
        while j < M {
            out[j] = unwrap_n(records[j]);
            j += 1;
        }
        out
    }

    unsafe fn store_interleaved_arrays<const M: usize, const C: usize>(ptr: *mut Self::Element, values: [[Self; C]; M]) {
        let mut records = [[R::EMPTY; C]; M];
        let mut j = 0;
        while j < M {
            records[j] = wrap_n(values[j]);
            j += 1;
        }
        unsafe { R::store_interleaved_arrays::<M, C>(ptr, records) }
    }

    unsafe fn load_deinterleaved_grouped<const M: usize, const TAIL: usize>(
        ptr: *const Self::Element,
    ) -> [StreamGroup<Self, TAIL>; M] {
        let groups = unsafe { R::load_deinterleaved_grouped::<M, TAIL>(ptr) };

        let mut out = [StreamGroup { head: Vector(R::EMPTY), tail: [Vector(R::EMPTY); TAIL] }; M];
        let mut j = 0;
        while j < M {
            out[j] = StreamGroup { head: Vector(groups[j].head), tail: unwrap_n(groups[j].tail) };
            j += 1;
        }
        out
    }

    unsafe fn store_interleaved_grouped<const M: usize, const TAIL: usize>(
        ptr: *mut Self::Element,
        values: [StreamGroup<Self, TAIL>; M],
    ) {
        let empty = StreamGroup { head: R::EMPTY, tail: [R::EMPTY; TAIL] };
        let mut regs = [empty; M];
        let mut j = 0;
        while j < M {
            regs[j] = StreamGroup { head: values[j].head.0, tail: wrap_n(values[j].tail) };
            j += 1;
        }
        unsafe { R::store_interleaved_grouped::<M, TAIL>(ptr, regs) }
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl<R, I> IndexableVector<Vector<I>> for Vector<R>
where
    R: IndexableRegister<I>,
    I: UnsignedIntegerRegister<Lanes = R::Lanes>,
{
    unsafe fn gather_ptr(ptr: *const Self::Element, indices: Vector<I>) -> Self {
        unsafe { Vector(R::gather(ptr, indices.0)) }
    }

    unsafe fn gather_ptr_m(src: Self, mask: Self::Mask, ptr: *const Self::Element, indices: Vector<I>) -> Self {
        unsafe { Vector(R::gather_m(src.0, mask.0, ptr, indices.0)) }
    }

    unsafe fn gather_ptr_z(mask: Self::Mask, ptr: *const Self::Element, indices: Vector<I>) -> Self {
        unsafe { Vector(R::gather_z(mask.0, ptr, indices.0)) }
    }

    unsafe fn scatter_ptr(value: Self, ptr: *mut Self::Element, indices: Vector<I>) {
        unsafe { R::scatter(value.0, ptr, indices.0) };
    }

    unsafe fn scatter_ptr_m(value: Self, mask: Self::Mask, ptr: *mut Self::Element, indices: Vector<I>) {
        unsafe { R::scatter_m(value.0, mask.0, ptr, indices.0) };
    }
}

#[rustfmt::skip] #[thermite_macros::vector_impl]
#[diagnostic::do_not_recommend]
impl<R: BitwiseRegister + Register> BitwiseVector for Vector<R> {
    const HAS_NATIVE_TERNLOG: bool = R::HAS_NATIVE_TERNLOG;

    #[conditional] fn ternlog<const IMM: i32>(a: Self, b: Self, c: Self) -> Self {}
    #[conditional] fn bilog<const IMM: i32>(a: Self, b: Self) -> Self {}
}

#[rustfmt::skip] #[thermite_macros::vector_impl]
#[diagnostic::do_not_recommend]
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

#[rustfmt::skip] #[thermite_macros::inline_always]
#[diagnostic::do_not_recommend]
impl<R: PartialOrdRegister> PartialOrdVector for Vector<R> {
    fn cmp_lt(self, other: Self) -> Self::Mask { Mask(R::lt(self.0, other.0)) }
    fn cmp_le(self, other: Self) -> Self::Mask { Mask(R::le(self.0, other.0)) }
    fn cmp_gt(self, other: Self) -> Self::Mask { Mask(R::gt(self.0, other.0)) }
    fn cmp_ge(self, other: Self) -> Self::Mask { Mask(R::ge(self.0, other.0)) }
    fn cmp_eq(self, other: Self) -> Self::Mask { Mask(R::eq(self.0, other.0)) }
    fn cmp_ne(self, other: Self) -> Self::Mask { Mask(R::ne(self.0, other.0)) }
}

#[rustfmt::skip] #[thermite_macros::vector_impl]
#[diagnostic::do_not_recommend]
impl<R: NumericRegister> NumericVector for Vector<R> {
    const ZERO: Self = Vector(R::ZERO);
    const ONE: Self = Vector(R::ONE);
    const TWO: Self = Vector(R::TWO);
    const MIN: Self = Vector(R::MIN);
    const MAX: Self = Vector(R::MAX);

    // Concrete vectors already have the `CastVector` relationship in both directions;
    // these just name it, so that generic code needing a float <-> integer conversion
    // does not have to carry a bound the composites cannot satisfy.
    fn to_signed_integer(self) -> Self::Signed {<Self::Signed as CastVector<Self>>::cast_from(self)}
    fn from_signed_integer(v: Self::Signed) -> Self {<Self::Signed as CastVector<Self>>::cast_into(v)}
    fn to_unsigned_integer(self) -> Self::Unsigned {<Self::Unsigned as CastVector<Self>>::cast_from(self)}
    fn from_unsigned_integer(v: Self::Unsigned) -> Self {<Self::Unsigned as CastVector<Self>>::cast_into(v)}
    fn fast_to_signed_integer(self) -> Self::Signed {<Self::Signed as CastVector<Self>>::fast_cast_from(self)}
    fn fast_to_unsigned_integer(self) -> Self::Unsigned {<Self::Unsigned as CastVector<Self>>::fast_cast_from(self)}

    fn is_zero(self) -> Self::Mask { self.cmp_eq(Self::ZERO) }

    fn is_all_zero(self) -> bool { R::is_all_zero(self.0) }

    #[conditional] fn min(self, other: Self) -> Self {}
    #[conditional] fn max(self, other: Self) -> Self {}

    // Explicit bodies: the delegation macro fills non-generic stubs, and these
    // carry a type parameter (same as the `native_*<P: Policy>` family below).
    fn sort_by<O: crate::sort::SortOrder>(self) -> Self { Vector(R::sort_by::<O>(self.0)) }
    fn bitonic_clean_by<O: crate::sort::SortOrder>(self) -> Self { Vector(R::bitonic_clean_by::<O>(self.0)) }

    fn prefix_sum(self) -> Self {}
    fn prefix_min(self) -> Self {}
    fn prefix_max(self) -> Self {}
    fn reverse_prefix_sum(self) -> Self {}
    fn reverse_prefix_min(self) -> Self {}
    fn reverse_prefix_max(self) -> Self {}

    fn clamp(self, min: Self, max: Self) -> Self { self.min(max).max(min) }

    fn min_element(self) -> Self::Element { R::min_element(self.0) }
    fn max_element(self) -> Self::Element { R::max_element(self.0) }
    fn min_max_element(self) -> (Self::Element, Self::Element) { R::min_max_element(self.0) }

    fn arg_minmax(self) -> (usize, usize) { R::arg_minmax(self.0) }

    #[conditional] fn scale(self, factor: Self::Element) -> Self {}

    fn pairwise_sum(lo: Self, hi: Self) -> Self {}
    fn relaxed_pairwise_sum(lo: Self, hi: Self) -> Self {}

    fn sum_elements(self) -> Self::Element { R::sum_elements(self.0) }
    fn prod_elements(self) -> Self::Element { R::prod_elements(self.0) }

    fn offset() -> Self { Vector(R::offset()) }
    fn indexed() -> Self { Vector(R::indexed()) }
}

#[thermite_macros::inline_always]
impl<R: NumericRegister> Square for Vector<R> {
    type Output = Self;

    fn square(self) -> Self {
        Vector(R::square(self.0))
    }
}

#[thermite_macros::inline_always]
impl<R: NumericRegister> SquareMasked<Mask<R>> for Vector<R> {
    fn square_c(self, mask: Mask<R>) -> Self {
        Vector(R::square_c(mask.0, self.0))
    }

    fn square_m(self, src: Self, mask: Mask<R>) -> Self {
        Vector(R::square_m(src.0, mask.0, self.0))
    }

    fn square_z(self, mask: Mask<R>) -> Self {
        Vector(R::square_z(mask.0, self.0))
    }
}

#[thermite_macros::inline_always]
impl<R: NumericRegister> core::iter::Sum for Vector<R> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Vector(R::ZERO), Add::add)
    }
}

#[thermite_macros::inline_always]
impl<R: NumericRegister> core::iter::Product for Vector<R> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Vector(R::ONE), Mul::mul)
    }
}

#[thermite_macros::inline_always]
impl<R: NumericRegister> num_traits::Bounded for Vector<R> {
    fn max_value() -> Self {
        Vector(R::MAX)
    }

    fn min_value() -> Self {
        Vector(R::MIN)
    }
}

#[rustfmt::skip] #[thermite_macros::vector_impl]
#[diagnostic::do_not_recommend]
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
#[diagnostic::do_not_recommend]
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
    #[conditional] fn trailing_ones(self) -> Self {}
    #[conditional] fn trailing_zeros(self) -> Self {}

    fn count_conflicts(self) -> Self {}
}

#[thermite_macros::inline_always]
impl<R: IntegerRegister> Div<Divider<R::Element>> for Vector<R> {
    type Output = Self;

    fn div(self, rhs: Divider<R::Element>) -> Self::Output {
        Self(R::div_branched(self.0, rhs))
    }
}

#[thermite_macros::inline_always]
impl<R: IntegerRegister> Div<BranchfreeDivider<R::Element>> for Vector<R> {
    type Output = Self;

    fn div(self, rhs: BranchfreeDivider<R::Element>) -> Self::Output {
        Self(R::div_branchfree(self.0, rhs))
    }
}

#[thermite_macros::inline_always]
impl<R: IntegerRegister> Div<VectorDivider<R>> for Vector<R> {
    type Output = Self;

    fn div(self, rhs: VectorDivider<R>) -> Self::Output {
        Self(R::divv_branchfree(self.0, rhs))
    }
}

#[thermite_macros::inline_always]
impl<R: IntegerRegister> DivMasked<Mask<R>, Divider<R::Element>> for Vector<R>
where
    R::Element: Denominator,
{
    fn div_c(self, mask: Mask<R>, rhs: Divider<R::Element>) -> Self::Output {
        Vector(R::div_branched_c(mask.0, self.0, rhs))
    }

    fn div_m(self, src: Self, mask: Mask<R>, rhs: Divider<R::Element>) -> Self::Output {
        Vector(R::div_branched_m(src.0, mask.0, self.0, rhs))
    }

    fn div_z(self, mask: Mask<R>, rhs: Divider<R::Element>) -> Self::Output {
        Vector(R::div_branched_z(mask.0, self.0, rhs))
    }
}

#[thermite_macros::inline_always]
impl<R: IntegerRegister> DivMasked<Mask<R>, BranchfreeDivider<R::Element>> for Vector<R>
where
    R::Element: Denominator,
{
    fn div_c(self, mask: Mask<R>, rhs: BranchfreeDivider<R::Element>) -> Self::Output {
        Vector(R::div_branchfree_c(mask.0, self.0, rhs))
    }

    fn div_m(self, src: Self, mask: Mask<R>, rhs: BranchfreeDivider<R::Element>) -> Self::Output {
        Vector(R::div_branchfree_m(src.0, mask.0, self.0, rhs))
    }

    fn div_z(self, mask: Mask<R>, rhs: BranchfreeDivider<R::Element>) -> Self::Output {
        Vector(R::div_branchfree_z(mask.0, self.0, rhs))
    }
}

#[thermite_macros::inline_always]
impl<R: IntegerRegister> DivMasked<Mask<R>, VectorDivider<R>> for Vector<R>
where
    R::Element: Denominator,
{
    fn div_c(self, mask: Mask<R>, rhs: VectorDivider<R>) -> Self::Output {
        Vector(R::divv_branchfree_c(mask.0, self.0, rhs))
    }

    fn div_m(self, src: Self, mask: Mask<R>, rhs: VectorDivider<R>) -> Self::Output {
        Vector(R::divv_branchfree_m(src.0, mask.0, self.0, rhs))
    }

    fn div_z(self, mask: Mask<R>, rhs: VectorDivider<R>) -> Self::Output {
        Vector(R::divv_branchfree_z(mask.0, self.0, rhs))
    }
}

#[rustfmt::skip] #[thermite_macros::vector_impl]
#[diagnostic::do_not_recommend]
impl<R: SignedIntegerRegister> SignedIntegerVector for Vector<R>
where
    R::Element: Denominator,
{
    #[conditional] fn srai<const I: i32>(self) -> Self {}
    #[conditional] fn sra(self, count: u32) -> Self {}
    #[conditional] fn srav(self, counts: Self::Unsigned) -> Self {}
    #[conditional] fn avg_floor(self, other: Self) -> Self {}
    #[conditional] fn avg_ceil(self, other: Self) -> Self {}
    #[conditional] fn mulhrs(self, other: Self) -> Self {}
}

#[rustfmt::skip] #[thermite_macros::vector_impl]
#[diagnostic::do_not_recommend]
impl<R: UnsignedIntegerRegister> UnsignedIntegerVector for Vector<R>
where
    R::Element: Denominator,
{
    fn is_power_of_two(self) -> Self::Mask { Mask(R::is_power_of_two(self.0)) }
    fn in_range(self, lo: Self, hi: Self) -> Self::Mask { Mask(R::in_range(self.0, lo.0, hi.0)) }

    #[conditional] fn next_power_of_two_m1(self) -> Self {}
    #[conditional] fn ilog2p1(self) -> Self {}
    #[conditional] fn parity(self) -> Self {}
    #[conditional] fn avg(self, other: Self) -> Self {}
    #[conditional] fn abs_diff(self, other: Self) -> Self {}

    fn morton<const N: usize>(values: [Self; N]) -> Self {
        Vector(R::morton::<N>(wrap_n(values)))
    }

    fn reverse_morton<const N: usize>(self) -> [Self; N] {
        unwrap_n(R::reverse_morton::<N>(self.0))
    }
}

#[rustfmt::skip] #[thermite_macros::vector_impl]
#[diagnostic::do_not_recommend]
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

#[rustfmt::skip] #[thermite_macros::inline_always]
impl<R: FloatRegister> FloatVectorWithBits for Vector<R> {
    type SignedBits = Vector<R::SignedBits>;
    type Bits = Vector<R::Bits>;

    const NATIVE_CAP: NativeCapability = R::NATIVE_CAP;

    unsafe fn native_ldexp(self, exp: Self::SignedBits) -> Self {
        unsafe { Vector(R::native_ldexp(self.0, exp.0)) }
    }

    unsafe fn native_frexp(self) -> (Self, Self::SignedBits) {
        let (mantissa, exp) = unsafe { R::native_frexp(self.0) };
        (Vector(mantissa), Vector(exp))
    }

    unsafe fn native_sin_cos<P: Policy>(self) -> (Self, Self) {
        let (sin, cos) = unsafe { R::native_sin_cos::<P>(self.0) };
        (Vector(sin), Vector(cos))
    }

    unsafe fn native_sin<P: Policy>(self) -> Self { unsafe { Vector(R::native_sin::<P>(self.0)) } }
    unsafe fn native_cos<P: Policy>(self) -> Self { unsafe { Vector(R::native_cos::<P>(self.0)) } }
    unsafe fn native_tan<P: Policy>(self) -> Self { unsafe { Vector(R::native_tan::<P>(self.0)) } }
    unsafe fn native_exp2<P: Policy>(self) -> Self { unsafe { Vector(R::native_exp2::<P>(self.0)) } }
    unsafe fn native_log2<P: Policy>(self) -> Self { unsafe { Vector(R::native_log2::<P>(self.0)) } }
    unsafe fn native_exp<P: Policy>(self) -> Self { unsafe { Vector(R::native_exp::<P>(self.0)) } }
    unsafe fn native_ln<P: Policy>(self) -> Self { unsafe { Vector(R::native_ln::<P>(self.0)) } }
    unsafe fn native_powf<P: Policy>(self, exp: Self) -> Self { unsafe { Vector(R::native_powf::<P>(self.0, exp.0)) } }

    fn total_order(self) -> Self::SignedBits { Vector(R::total_order(self.0)) }
    fn linear_order(self) -> Self::SignedBits { Vector(R::linear_order(self.0)) }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl<R: LinAlg3Register> LinAlg3Vector for Vector<R> {
    fn dot3(self, other: Self) -> Self::Element { R::dot3(self.0, other.0) }
    fn cross3<const FAST: bool>(self, other: Self) -> Self { Vector(R::cross3::<FAST>(self.0, other.0)) }
    fn refract(self, n: Self, eta: Self::Element) -> Self { Vector(R::refract(self.0, n.0, eta)) }
    fn zero4(self) -> Self { Vector(R::zero4(self.0)) }
    fn one4(self) -> Self { Vector(R::one4(self.0)) }
    fn min_element3(self) -> Self::Element { R::min_element3(self.0) }
    fn max_element3(self) -> Self::Element { R::max_element3(self.0) }
    fn sum_elements3(self) -> Self::Element { R::sum_elements3(self.0) }
    fn prod_elements3(self) -> Self::Element { R::prod_elements3(self.0) }

    fn mat3_transpose(m: &[Self; 3]) -> [Self; 3] {
        unwrap_n(R::mat3_transpose(unsafe { core::mem::transmute(m) }))
    }

    fn mat3_vec3_product<const COLUMN_MAJOR: bool>(self, m: &[Self; 3]) -> Self {
        // Single-vector convenience over the array form (`N == 1`).
        Self::mat3_vec3_product_array::<COLUMN_MAJOR, 1>(m, &[self])[0]
    }

    fn mat3_vec3_product_array<const COLUMN_MAJOR: bool, const N: usize>(
        m: &[Self; 3],
        vectors: &[Self; N],
    ) -> [Self; N] {
        // SAFETY: Vector<R> is repr(transparent) around Storage<R>.
        unwrap_n(R::mat3_vec3_product::<COLUMN_MAJOR, N>(
            unsafe { core::mem::transmute(m) },
            unsafe { core::mem::transmute(vectors) },
        ))
    }

    fn mat3_product<const COLUMN_MAJOR: bool>(lhs: &[Self; 3], rhs: &[Self; 3]) -> [Self; 3] {
        // SAFETY: Vector<R> is repr(transparent) around Storage<R>.
        unwrap_n(R::mat3_product::<COLUMN_MAJOR>(
            unsafe { core::mem::transmute(lhs) },
            unsafe { core::mem::transmute(rhs) },
        ))
    }

    fn mat3_det<const FAST: bool>(m: &[Self; 3]) -> Self::Element {
        // SAFETY: Vector<R> is repr(transparent) around Storage<R>.
        R::mat3_det::<FAST>(unsafe { core::mem::transmute(m) })
    }

    fn mat3_inverse_inplace<const FAST: bool>(m: &mut [Self; 3]) -> Self::Element {
        // SAFETY: Vector<R> is repr(transparent) around Storage<R>.
        R::mat3_inverse::<FAST>(unsafe { core::mem::transmute(m) })
    }

    fn mat3_normal<const DIVIDE: bool, const FAST: bool>(m: &[Self; 3]) -> [Self; 3] {
        // SAFETY: Vector<R> is repr(transparent) around Storage<R>.
        unwrap_n(R::mat3_normal::<DIVIDE, FAST>(unsafe { core::mem::transmute(m) }))
    }
}

#[thermite_macros::inline_always]
impl<R: LinAlg4Register> LinAlg4Vector for Vector<R> {
    fn dot4(self, other: Self) -> Self::Element {
        R::dot4(self.0, other.0)
    }

    fn quat4_product<const FAST: bool>(self, other: Self) -> Self {
        Vector(R::quat4_product::<FAST>(self.0, other.0))
    }

    fn quat4_vec3_product<const FAST: bool>(self, vec: Self) -> Self {
        Vector(R::quat4_vec3_product::<FAST>(self.0, vec.0))
    }

    fn quat_to_mat3<const COLUMN_MAJOR: bool>(self) -> [Self; 3] {
        unwrap_n(R::quat_to_mat3::<COLUMN_MAJOR>(self.0))
    }

    fn quat_to_mat4<const COLUMN_MAJOR: bool>(self) -> [Self; 4] {
        unwrap_n(R::quat_to_mat4::<COLUMN_MAJOR>(self.0))
    }

    fn mat4_transpose(m: &[Self; 4]) -> [Self; 4] {
        // SAFETY: transmute &[Vector<R>; 4] to &[Storage<R>; 4] is safe
        // because Vector<R> is repr(transparent) around Storage<R>
        unwrap_n(R::mat4_transpose(unsafe { core::mem::transmute(m) }))
    }

    fn mat4_vec4_product<const COLUMN_MAJOR: bool>(self, m: &[Self; 4]) -> Self {
        // Single-vector convenience over the array form (`N == 1`).
        Self::mat4_vec4_product_array::<COLUMN_MAJOR, 1>(m, &[self])[0]
    }

    fn mat4_vec3_product<const COLUMN_MAJOR: bool>(self, m: &[Self; 4]) -> Self {
        // Single-vector convenience over the array form (`N == 1`).
        Self::mat4_vec3_product_array::<COLUMN_MAJOR, 1>(m, &[self])[0]
    }

    fn mat4_vec3_product_array<const COLUMN_MAJOR: bool, const N: usize>(
        m: &[Self; 4],
        vectors: &[Self; N],
    ) -> [Self; N] {
        // SAFETY: Vector<R> is repr(transparent) around Storage<R>.
        unwrap_n(R::mat4_vec3_product::<COLUMN_MAJOR, N>(
            unsafe { core::mem::transmute(m) },
            unsafe { core::mem::transmute(vectors) },
        ))
    }

    fn mat4_point3_product<const COLUMN_MAJOR: bool>(self, m: &[Self; 4]) -> Self {
        // Single-vector convenience over the array form (`N == 1`).
        Self::mat4_point3_product_array::<COLUMN_MAJOR, 1>(m, &[self])[0]
    }

    fn mat4_point3_product_array<const COLUMN_MAJOR: bool, const N: usize>(
        m: &[Self; 4],
        vectors: &[Self; N],
    ) -> [Self; N] {
        // SAFETY: Vector<R> is repr(transparent) around Storage<R>.
        unwrap_n(R::mat4_point3_product::<COLUMN_MAJOR, N>(
            unsafe { core::mem::transmute(m) },
            unsafe { core::mem::transmute(vectors) },
        ))
    }

    fn mat4_product<const COLUMN_MAJOR: bool>(lhs: &[Self; 4], rhs: &[Self; 4]) -> [Self; 4] {
        // SAFETY: transmute &[Vector<R>; 4] to &[Storage<R>; 4] is safe
        // because Vector<R> is repr(transparent) around Storage<R>
        unwrap_n(R::mat4_product::<COLUMN_MAJOR>(
            unsafe { core::mem::transmute(lhs) }, //
            unsafe { core::mem::transmute(rhs) },
        ))
    }

    fn mat4_vec4_product_array<const COLUMN_MAJOR: bool, const N: usize>(
        m: &[Self; 4],
        vectors: &[Self; N],
    ) -> [Self; N] {
        // SAFETY: transmute &[Vector<R>; _] to &[Storage<R>; _] is safe because
        // Vector<R> is repr(transparent) around Storage<R>.
        unwrap_n(R::mat4_vec4_product::<COLUMN_MAJOR, N>(
            unsafe { core::mem::transmute(m) },
            unsafe { core::mem::transmute(vectors) },
        ))
    }

    fn mat4_det<const FAST: bool>(m: &[Self; 4]) -> Self::Element {
        // SAFETY: transmute &[Vector<R>; 4] to &[Storage<R>; 4] is safe
        // because Vector<R> is repr(transparent) around Storage<R>
        R::mat4_det::<FAST>(unsafe { core::mem::transmute(m) })
    }

    fn mat4_inverse_inplace<const FAST: bool>(m: &mut [Self; 4]) -> Self::Element {
        // SAFETY: transmute &[Vector<R>; 4] to &[Storage<R>; 4] is safe
        // because Vector<R> is repr(transparent) around Storage<R>
        R::mat4_inverse::<FAST>(unsafe { core::mem::transmute(m) })
    }
}

#[thermite_macros::inline_always]
impl<R: Register> VectorWithRegister<R> for Vector<R> {
    fn into_register(self) -> Storage<R> {
        self.0
    }

    fn from_register(reg: Storage<R>) -> Self {
        Vector(reg)
    }

    fn as_slice(&self) -> &[R::Element] {
        R::as_slice(&self.0)
    }

    fn as_mut_slice(&mut self) -> &mut [R::Element] {
        R::as_mut_slice(&mut self.0)
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

#[thermite_macros::inline_always]
impl<R: Register, B: Register> Extend<Mask<R>> for Mask<B>
where
    B::Mask: ExtendRegister<R::Mask>,
{
    fn extend(v: Mask<R>) -> Self {
        Mask(B::Mask::extend(v.0))
    }

    fn narrow(self) -> Mask<R> {
        Mask(B::Mask::narrow(self.0))
    }
}

#[thermite_macros::inline_always]
impl<R: Register, B: Register> Concat<Mask<R>> for Mask<B>
where
    B::Mask: ConcatRegister<R::Mask>,
{
    fn concat(lo: Mask<R>, hi: Mask<R>) -> Self {
        Mask(B::Mask::concat(lo.0, hi.0))
    }

    fn split(self) -> (Mask<R>, Mask<R>) {
        let (lo, hi) = B::Mask::split(self.0);
        (Mask(lo), Mask(hi))
    }
}

#[thermite_macros::inline_always]
impl<R: Register, B: Register> Extend<Vector<R>> for Vector<B>
where
    B: ExtendRegister<R>,
{
    fn extend(v: Vector<R>) -> Self {
        Vector(B::extend(v.0))
    }

    fn narrow(self) -> Vector<R> {
        Vector(B::narrow(self.0))
    }
}

#[thermite_macros::inline_always]
impl<R: Register, B: Register> Concat<Vector<R>> for Vector<B>
where
    B: ConcatRegister<R>,
{
    fn concat(lo: Vector<R>, hi: Vector<R>) -> Self {
        Vector(B::concat(lo.0, hi.0))
    }

    fn split(self) -> (Vector<R>, Vector<R>) {
        let (lo, hi) = B::split(self.0);
        (Vector(lo), Vector(hi))
    }
}

#[thermite_macros::inline_always]
impl<R: PartialOrdRegister> PartialEq for Vector<R> {
    /// Compare two vectors for equality, returning true only if all elements are equal.
    fn eq(&self, other: &Self) -> bool {
        Mask::<R>(R::eq(self.0, other.0)).all()
    }

    /// Compare two vectors for inequality, returning true if any element is not equal.
    #[allow(clippy::partialeq_ne_impl)] // sometimes might have better underlying implementation
    fn ne(&self, other: &Self) -> bool {
        Mask::<R>(R::ne(self.0, other.0)).any()
    }
}

#[thermite_macros::inline_always]
impl<R: Register> Index<usize> for Vector<R> {
    type Output = R::Element;

    fn index(&self, index: usize) -> &Self::Output {
        &R::as_slice(&self.0)[index]
    }
}

#[thermite_macros::inline_always]
impl<R: Register> IndexMut<usize> for Vector<R> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut R::as_mut_slice(&mut self.0)[index]
    }
}

#[thermite_macros::inline_always]
impl<R: NumericRegister> Zero for Vector<R> {
    /// Returns true if **all** elements in the vector are zero.
    fn is_zero(&self) -> bool {
        Mask::<R>(R::eq(self.0, R::ZERO)).all()
    }

    fn set_zero(&mut self) {
        self.0 = R::ZERO;
    }

    fn zero() -> Self {
        Self::ZERO
    }
}

#[thermite_macros::inline_always]
impl<R: NumericRegister> One for Vector<R> {
    /// Returns true if **all** elements in the vector are one.
    fn is_one(&self) -> bool {
        Mask::<R>(R::eq(self.0, R::ONE)).all()
    }

    fn set_one(&mut self) {
        self.0 = R::ONE;
    }

    fn one() -> Self {
        Self::ONE
    }
}

#[thermite_macros::inline_always]
impl<R: IntegerRegister> SaturatingAdd for Vector<R> {
    fn saturating_add(&self, v: &Self) -> Self {
        Self(R::saturating_add(self.0, v.0))
    }
}

#[thermite_macros::inline_always]
impl<R: IntegerRegister> SaturatingSub for Vector<R> {
    fn saturating_sub(&self, v: &Self) -> Self {
        Self(R::saturating_sub(self.0, v.0))
    }
}

#[thermite_macros::inline_always]
impl<R: IntegerRegister> Saturating for Vector<R> {
    fn saturating_add(self, v: Self) -> Self {
        Self(R::saturating_add(self.0, v.0))
    }

    fn saturating_sub(self, v: Self) -> Self {
        Self(R::saturating_sub(self.0, v.0))
    }
}

#[thermite_macros::inline_always]
impl<R: IntegerRegister> WrappingAdd for Vector<R> {
    fn wrapping_add(&self, v: &Self) -> Self {
        Self(R::add(self.0, v.0))
    }
}

#[thermite_macros::inline_always]
impl<R: IntegerRegister> WrappingSub for Vector<R> {
    fn wrapping_sub(&self, v: &Self) -> Self {
        Self(R::sub(self.0, v.0))
    }
}

#[thermite_macros::inline_always]
impl<R: IntegerRegister> WrappingMul for Vector<R> {
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
        let is_less = <R::Mask as MaskRegister>::all(R::lt(self.0, other.0));
        let is_greater = <R::Mask as MaskRegister>::all(R::gt(self.0, other.0));
        let is_equal = <R::Mask as MaskRegister>::all(R::eq(self.0, other.0));

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
            Vector(R::new(GenericArray::generate(|_| self.sample(rng))))
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
                    Vector(R::new(GenericArray::generate(|_| self.sample(rng))))
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
