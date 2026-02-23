#![warn(missing_docs, clippy::missing_safety_doc)]

use core::{marker::PhantomData, ops::Sub};

use crate::{
    BranchfreeDivider, Divider,
    divider::vector::VectorDivider,
    isa::InstructionSet,
    register::{ExtendRegister, IndexableRegister},
};

use super::{
    BitCastRegister, BitshiftRegister, BitwiseRegister, CastMaskRegister, CastRegister, CoreRegister, FloatRegister,
    IntegerRegister, Lanes, LinAlg3Register, LinAlg4Register, MaskRegister, NumericRegister, PartialOrdRegister,
    Register, SignedIntegerRegister, SignedRegister, Storage, SwizzleRegister, UnsignedIntegerRegister,
    ValidLinAlg3Length, ZeroUpper,
};

use generic_array::{
    GenericArray,
    functional::FunctionalSequence as _,
    sequence::GenericSequence,
    typenum::{self, Diff, Unsigned},
};

/// Assuming a 4-lane register, halves that.
pub type HalfRegister2<R> = ReducedRegister<R, typenum::U2>;

/// Wrapper around a real register to emulate reducing it by N number of lanes, where the upper lanes
/// are either ignored or zeroed out. This is arguably better than double-pumping scalars.
#[repr(transparent)]
pub struct ReducedRegister<R: CoreRegister, N>(pub(crate) Storage<R>, PhantomData<N>);

impl<R: CoreRegister, N> Clone for ReducedRegister<R, N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: CoreRegister, N> Copy for ReducedRegister<R, N> {}

/// Defines a Reducible superset of `CoreRegister`'s
pub trait CoreReducible<N: Unsigned>: CoreRegister<Lanes: Sub<N, Output: Lanes>> {}

/// Defines a Reducible superset of `Register`'s, which can be reduced to a smaller register type with fewer lanes.
pub trait Reducible<N: Unsigned>: CoreReducible<N> + Register {}

impl<T, N: Unsigned> CoreReducible<N> for T where T: CoreRegister<Lanes: Sub<N, Output: Lanes>> {}
impl<T, N: Unsigned> Reducible<N> for T where T: CoreReducible<N> + Register {}

impl<R: CoreRegister, N: Unsigned> ReducedRegister<R, N>
where
    R: CoreReducible<N>,
{
    #[inline(always)]
    pub const fn new(value: Storage<R>) -> Self {
        Self(value, PhantomData)
    }

    // e.g., for a 4-lane reduced to 3-lane, the bitmask of the 4-lane register would be 0b1111,
    // but the bitmask of the reduced 3-lane register should be 0b0111.
    const BITMASK: u64 = u64::MAX >> (64 - <Self as CoreRegister>::Lanes::USIZE);

    #[inline(always)]
    const fn min_idx(idx: usize) -> usize {
        // fast path if the number of lanes is a power of two, otherwise clamp to the max lane index
        if const { <Self as CoreRegister>::Lanes::USIZE.is_power_of_two() } {
            idx & (<Self as CoreRegister>::Lanes::USIZE - 1)
        } else if idx >= <Self as CoreRegister>::Lanes::USIZE {
            <Self as CoreRegister>::Lanes::USIZE - 1
        } else {
            idx
        }
    }

    #[inline(always)]
    fn mask() -> Storage<R::Mask> {
        <R::Mask as MaskRegister>::new_mask(GenericArray::generate(|i| i < <Self as CoreRegister>::Lanes::USIZE))
    }

    #[inline(always)]
    fn inv_mask() -> Storage<R::Mask> {
        <R::Mask as MaskRegister>::new_mask(GenericArray::generate(|i| i >= <Self as CoreRegister>::Lanes::USIZE))
    }

    #[inline(always)]
    fn pad_array<T: Copy>(value: GenericArray<T, <Self as CoreRegister>::Lanes>) -> GenericArray<T, R::Lanes> {
        unsafe {
            let mut padded: GenericArray<T, R::Lanes> = core::mem::zeroed();

            core::ptr::copy_nonoverlapping(
                value.as_ptr(),
                padded.as_mut_ptr(),
                <Self as CoreRegister>::Lanes::USIZE,
            );

            padded
        }
    }
}

impl<R: CoreRegister, N: Unsigned> CoreRegister for ReducedRegister<R, N>
where
    R: CoreReducible<N>,
{
    type Lanes = Diff<R::Lanes, N>;
    type Storage = Self;
    type Mask = ReducedRegister<R::Mask, N>;

    const IS_EMULATED: bool = R::IS_EMULATED;

    const ISA: InstructionSet = R::ISA;
    const EMPTY: Storage<Self> = Self(R::EMPTY, PhantomData);

    #[inline(always)]
    fn blendv(mask: Storage<Self::Mask>, on_false: Storage<Self>, on_true: Storage<Self>) -> Storage<Self> {
        Self(R::blendv(mask.0, on_false.0, on_true.0), PhantomData)
    }

    #[inline(always)]
    fn z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        Self(R::z(mask.0, value.0), PhantomData)
    }

    #[inline(always)]
    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        Self(R::nz(mask.0, value.0), PhantomData)
    }

    #[inline(always)]
    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        // We choose the shorter `Z::N` and `Self::Lanes`,
        // just in case Reduced was nested.
        if const { Z::N < <Self::Lanes as Unsigned>::USIZE } {
            Self(R::zeroupper_z::<Z>(value.0), PhantomData)
        } else {
            Self(R::zeroupper_z::<super::OwnLanes<Self>>(value.0), PhantomData)
        }
    }
}

#[rustfmt::skip] #[thermite_macros::reduced_impl]
impl<R: BitwiseRegister, N: Unsigned> BitwiseRegister for ReducedRegister<R, N> where R: CoreReducible<N> {
    #[conditional] fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn bitandnot(lhs: Storage<Self>, rhs:Storage<Self>) -> Storage<Self> {}
    #[conditional] fn not(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn ternlog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn bilog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {}
}

#[rustfmt::skip]
impl<R: MaskRegister, N: Unsigned> MaskRegister for ReducedRegister<R, N> where R: CoreReducible<N> {
    #[inline(always)]
    fn set(mask: Storage<Self>, lane: usize, value: bool) -> Storage<Self> {
        Self(R::set(mask.0, Self::min_idx(lane), value), PhantomData)
    }

    #[inline(always)]
    fn test(mask: Storage<Self>, lane: usize) -> bool {
        R::test(mask.0, Self::min_idx(lane))
    }

    const TRUTHY: Storage<Self> = Self(R::TRUTHY, PhantomData);
    const FALSY: Storage<Self> = Self(R::FALSY, PhantomData);

    #[inline(always)]
    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        Self(R::new_mask(Self::pad_array(value)), PhantomData)
    }

    #[inline(always)] fn all(value: Storage<Self>) -> bool { Self::native_bitmask(value) == Some(Self::BITMASK) }
    #[inline(always)] fn any(value: Storage<Self>) -> bool { Self::native_bitmask(value) != Some(0) }
    #[inline(always)] fn none(value: Storage<Self>) -> bool { Self::native_bitmask(value) == Some(0) }

    #[inline(always)]
    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        R::native_bitmask(value.0).map(|mask| mask & Self::BITMASK)
    }

    #[inline(always)]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let Some(native) = R::native_bitmask(value.0) else {
            unsafe { core::hint::unreachable_unchecked() }
        };

        use bitvec::slice::BitSlice;

        let bits = BitSlice::<u32>::from_slice(unsafe { core::mem::transmute::<&u64, &[u32; 2]>(&native) });

        view[..Self::Lanes::USIZE].copy_from_bitslice(&bits[..Self::Lanes::USIZE]);
    }
}

#[rustfmt::skip]
impl<R: Register, N: Unsigned> Register for ReducedRegister<R, N> where R: Reducible<N> {
    type Element = R::Element;

    const HAS_EQUAL_SIZE_MASK: bool = R::HAS_EQUAL_SIZE_MASK;

    #[inline(always)] fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> { ReducedRegister(R::from_mask(mask.0), PhantomData) }
    #[inline(always)] fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> { ReducedRegister(R::into_mask(value.0), PhantomData) }
    #[inline(always)] fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> { ReducedRegister(R::into_mask_unchecked(value.0), PhantomData) }
    #[inline(always)] fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> { ReducedRegister(R::msb_to_mask(value.0), PhantomData) }

    type Unsigned = ReducedRegister<R::Unsigned, N>;
    type Signed = ReducedRegister<R::Signed, N>;

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        Self(R::new(Self::pad_array(value)), PhantomData)
    }

    #[inline(always)] fn single(value: Self::Element) -> Storage<Self> { Self(R::single(value), PhantomData) }
    #[inline(always)] fn single_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> { Self(R::single_m(src.0, mask.0, value), PhantomData) }
    #[inline(always)] fn single_z(mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> { Self(R::single_z(mask.0, value), PhantomData) }

    #[inline(always)] fn splat(value: Self::Element) -> Storage<Self> { Self(R::splat(value), PhantomData) }
    #[inline(always)] fn splat_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> { Self(R::splat_m(src.0, mask.0, value), PhantomData) }
    #[inline(always)] fn splat_z(mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> { Self(R::splat_z(mask.0, value), PhantomData) }

    #[inline(always)]
    fn broadcast<const I: usize>(value: Storage<Self>) -> Storage<Self> {
        if const { I >= Self::Lanes::USIZE } { panic!("Broadcast Index out of bounds"); }
        Self(R::broadcast::<I>(value.0), PhantomData)
    }

    #[inline(always)]
    fn broadcast_c<const I: usize>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        if const { I >= Self::Lanes::USIZE } { panic!("Broadcast Index out of bounds"); }
        Self(R::broadcast_c::<I>(mask.0, value.0), PhantomData)
    }

    #[inline(always)]
    fn broadcast_m<const I: usize>(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
    ) -> Storage<Self> {
        if const { I >= Self::Lanes::USIZE } { panic!("Broadcast Index out of bounds"); }
        Self(R::broadcast_m::<I>(src.0, mask.0, value.0), PhantomData)
    }

    #[inline(always)]
    fn broadcast_z<const I: usize>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        if const { I >= Self::Lanes::USIZE } { panic!("Broadcast Index out of bounds"); }
        Self(R::broadcast_z::<I>(mask.0, value.0), PhantomData)
    }

    #[inline(always)] fn broadcastv(value: Storage<Self>, idx: usize) -> Storage<Self> { Self(R::broadcastv(value.0, Self::min_idx(idx)), PhantomData) }
    #[inline(always)] fn broadcastv_c(mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> { Self(R::broadcastv_c(mask.0, value.0, Self::min_idx(idx)), PhantomData) }
    #[inline(always)] fn broadcastv_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> { Self(R::broadcastv_m(src.0, mask.0, value.0, Self::min_idx(idx)), PhantomData) }
    #[inline(always)] fn broadcastv_z(mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> { Self(R::broadcastv_z(mask.0, value.0, Self::min_idx(idx)), PhantomData) }

    #[inline(always)]
    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        if const { N::USIZE == 0 } {
            unsafe { Self(R::load(ptr), PhantomData) }
        } else {
            unsafe { Self(R::load_z(Self::mask(), ptr), PhantomData) }
        }
    }

    #[inline(always)]
    unsafe fn load_m(src: Storage<Self>, mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        if const { N::USIZE == 0 } {
            unsafe { Self(R::load_m(src.0, mask.0, ptr), PhantomData) }
        } else {
            unsafe {
                Self(
                    R::load_m(src.0, <R::Mask as BitwiseRegister>::bitand(mask.0, Self::mask()), ptr),
                    PhantomData,
                )
            }
        }
    }

    #[inline(always)]
    unsafe fn load_z(mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        if const { N::USIZE == 0 } {
            unsafe { Self(R::load_z(mask.0, ptr), PhantomData) }
        } else {
            unsafe {
                Self(
                    R::load_z(<R::Mask as BitwiseRegister>::bitand(mask.0, Self::mask()), ptr),
                    PhantomData,
                )
            }
        }
    }

    #[inline(always)]
    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        // Otherwise this is a Reduced register, or some other hybrid
        let mut res = Self::EMPTY;

        // SAFETY: This is safe as long as the pointer is valid and of the correct length.
        for i in 0..<Self::Lanes as Unsigned>::USIZE {
            Self::as_array_mut(&mut res)[i] = unsafe { ptr.add(i).read_unaligned() };
        }

        res
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        if const { N::USIZE == 0 } {
            unsafe { R::store(ptr, value.0) }
        } else {
            unsafe { R::store_masked(ptr, Self::mask(), value.0) }
        }
    }

    #[inline(always)]
    unsafe fn store_masked(ptr: *mut Self::Element, mask: Storage<Self::Mask>, value: Storage<Self>) {
        // combine masked to ensure we don't write to the upper lanes,
        // even if the caller provides a mask with those lanes set.
        unsafe { R::store_masked(ptr, <R::Mask as BitwiseRegister>::bitand(mask.0, Self::mask()), value.0) }
    }

    #[inline(always)]
    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Storage<Self>) {
        // SAFETY: This is safe as long as the pointer is valid and of the correct length.
        for i in 0..<Self::Lanes as Unsigned>::USIZE {
            unsafe { ptr.add(i).write_unaligned(Self::as_array(&value)[i]) };
        }
    }

    const HAS_SIMPLE_UNPACK: bool = false;

    #[inline(always)]
    fn unpack(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        todo!()
    }

    #[inline(always)] fn swap_bytes(value: Storage<Self>) -> Storage<Self> { Self(R::swap_bytes(value.0), PhantomData) }
    #[inline(always)] fn swap_bytes_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> { Self(R::swap_bytes_c(mask.0, value.0), PhantomData) }
    #[inline(always)] fn swap_bytes_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> { Self(R::swap_bytes_m(src.0, mask.0, value.0), PhantomData) }
    #[inline(always)] fn swap_bytes_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> { Self(R::swap_bytes_z(mask.0, value.0), PhantomData) }
}

impl<IDX, R: IndexableRegister<IDX>, N: Unsigned> IndexableRegister<ReducedRegister<IDX, N>> for ReducedRegister<R, N>
where
    R: Reducible<N>,
    IDX: UnsignedIntegerRegister<Lanes = R::Lanes>,
{
    #[inline(always)]
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<ReducedRegister<IDX, N>>) -> Storage<Self> {
        unsafe { Self(R::gather_z(Self::mask(), ptr, indices.0), PhantomData) }
    }

    #[inline(always)]
    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<ReducedRegister<IDX, N>>,
    ) -> Storage<Self> {
        unsafe {
            let mask = <R::Mask as BitwiseRegister>::bitand(mask.0, Self::mask());
            Self(R::gather_m(src.0, mask, ptr, indices.0), PhantomData)
        }
    }

    #[inline(always)]
    unsafe fn gather_z(
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<ReducedRegister<IDX, N>>,
    ) -> Storage<Self> {
        unsafe {
            let mask = <R::Mask as BitwiseRegister>::bitand(mask.0, Self::mask());
            Self(R::gather_z(mask, ptr, indices.0), PhantomData)
        }
    }

    #[inline(always)]
    unsafe fn scatter(value: Storage<Self>, ptr: *mut Self::Element, indices: Storage<ReducedRegister<IDX, N>>) {
        unsafe { R::scatter_m(value.0, Self::mask(), ptr, indices.0) }
    }

    #[inline(always)]
    unsafe fn scatter_m(
        value: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *mut Self::Element,
        indices: Storage<ReducedRegister<IDX, N>>,
    ) {
        unsafe {
            let mask = <R::Mask as BitwiseRegister>::bitand(mask.0, Self::mask());
            R::scatter_m(value.0, mask, ptr, indices.0)
        }
    }
}

impl<R: SwizzleRegister, N: Unsigned> SwizzleRegister for ReducedRegister<R, N>
where
    R: Reducible<N>,
{
    const HAS_PERMUTEV: bool = R::HAS_PERMUTEV;

    #[inline(always)]
    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        Self(R::permutev(value.0, Self::pad_array(idxs)), PhantomData)
    }

    #[inline(always)]
    fn permutev_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
        idxs: GenericArray<u32, Self::Lanes>,
    ) -> Storage<Self> {
        Self(
            R::permutev_m(src.0, mask.0, value.0, Self::pad_array(idxs)),
            PhantomData,
        )
    }

    #[inline(always)]
    fn permutev_z(
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
        idxs: GenericArray<u32, Self::Lanes>,
    ) -> Storage<Self> {
        Self(R::permutev_z(mask.0, value.0, Self::pad_array(idxs)), PhantomData)
    }

    #[inline(always)]
    fn swizzle(a: Storage<Self>, b: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        Self(R::swizzle(a.0, b.0, Self::pad_array(idxs)), PhantomData)
    }

    #[inline(always)]
    fn swizzle_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        a: Storage<Self>,
        b: Storage<Self>,
        idxs: GenericArray<u32, Self::Lanes>,
    ) -> Storage<Self> {
        Self(
            R::swizzle_m(src.0, mask.0, a.0, b.0, Self::pad_array(idxs)),
            PhantomData,
        )
    }

    #[inline(always)]
    fn swizzle_z(
        mask: Storage<Self::Mask>,
        a: Storage<Self>,
        b: Storage<Self>,
        idxs: GenericArray<u32, Self::Lanes>,
    ) -> Storage<Self> {
        Self(R::swizzle_z(mask.0, a.0, b.0, Self::pad_array(idxs)), PhantomData)
    }
}

#[rustfmt::skip] #[thermite_macros::reduced_impl]
impl<R: BitshiftRegister, N: Unsigned> BitshiftRegister for ReducedRegister<R, N> where R: Reducible<N> {
    #[conditional] fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {}
    #[conditional] fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {}
    #[conditional] fn shli<const IMM: i32>(lhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn shri<const IMM: i32>(lhs: Storage<Self>) -> Storage<Self> {}

    // TODO: These is incorrect if reduced, since it shifts the entire register
    const HAS_WIDE_BYTE_SHIFTS: bool = R::HAS_WIDE_BYTE_SHIFTS;
    #[conditional] fn bshli<const IMM8: i32>(lhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn bshri<const IMM8: i32>(lhs: Storage<Self>) -> Storage<Self> {}

    const HAS_TRUE_SHIFTV: bool = R::HAS_TRUE_SHIFTV;

    #[conditional] fn shrv(value: Storage<Self>, count: Storage<Self::Unsigned>) -> Storage<Self> {}
    #[conditional] fn shlv(value: Storage<Self>, count: Storage<Self::Unsigned>) -> Storage<Self> {}

    #[conditional] fn rol(value: Storage<Self>, shift: u32) -> Storage<Self> {}
    #[conditional] fn ror(value: Storage<Self>, shift: u32) -> Storage<Self> {}

    #[conditional] fn roli<const IMM: i32>(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn rori<const IMM: i32>(value: Storage<Self>) -> Storage<Self> {}

    #[conditional] fn rolv(value: Storage<Self>, count: Storage<Self::Unsigned>) -> Storage<Self> {}
    #[conditional] fn rorv(value: Storage<Self>, count: Storage<Self::Unsigned>) -> Storage<Self> {}
}

impl<FROM, TO, N: Unsigned> CastRegister<ReducedRegister<FROM, N>> for ReducedRegister<TO, N>
where
    TO: CoreReducible<N> + CastRegister<FROM>,
    FROM: CoreReducible<N>,
{
    #[inline(always)]
    fn cast_from(value: Storage<ReducedRegister<FROM, N>>) -> Storage<Self> {
        Self(TO::cast_from(value.0), PhantomData)
    }

    #[inline(always)]
    fn fast_cast_from(value: Storage<ReducedRegister<FROM, N>>) -> Storage<Self> {
        Self(TO::fast_cast_from(value.0), PhantomData)
    }
}

impl<FROM, TO, N: Unsigned> BitCastRegister<ReducedRegister<FROM, N>> for ReducedRegister<TO, N>
where
    TO: CoreReducible<N> + BitCastRegister<FROM>,
    FROM: CoreReducible<N>,
{
    #[inline(always)]
    fn from_bits(value: Storage<ReducedRegister<FROM, N>>) -> Storage<Self> {
        Self(TO::from_bits(value.0), PhantomData)
    }
}

impl<FROM, TO, N: Unsigned> CastMaskRegister<ReducedRegister<FROM, N>> for ReducedRegister<TO, N>
where
    TO: CoreReducible<N> + CastMaskRegister<FROM>,
    FROM: CoreReducible<N>,
{
    #[inline(always)]
    fn mask_from(value: Storage<ReducedRegister<FROM, N>>) -> Storage<Self> {
        Self(TO::mask_from(value.0), PhantomData)
    }
}

impl<R: CoreRegister, N: Unsigned> ExtendRegister<ReducedRegister<R, N>> for ReducedRegister<R, N>
where
    R: CoreReducible<N>,
{
    #[inline(always)]
    fn extend(value: Storage<ReducedRegister<R, N>>) -> Storage<Self> {
        value
    }

    #[inline(always)]
    fn narrow(value: Storage<Self>) -> Storage<ReducedRegister<R, N>> {
        value
    }
}

impl<R: CoreRegister, N: Unsigned> ExtendRegister<ReducedRegister<R, N>> for R
where
    R: CoreReducible<N>,
{
    #[inline(always)]
    fn extend(value: Storage<ReducedRegister<R, N>>) -> Storage<Self> {
        // extend requires the upper lanes are clear
        Self::zeroupper(value.0)
    }

    #[inline(always)]
    fn narrow(value: Storage<Self>) -> Storage<ReducedRegister<R, N>> {
        ReducedRegister(value, PhantomData)
    }
}

#[rustfmt::skip] #[thermite_macros::reduced_impl]
impl<R: PartialOrdRegister, N: Unsigned> PartialOrdRegister for ReducedRegister<R, N> where R: Reducible<N> {
    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {}
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {}
    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {}
    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {}
    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {}
}

// pub trait SpecReducibleNumRegister: NumericRegister {
//     fn min_element(value: Storage<Self>) -> Self::Element;
//     fn max_element(value: Storage<Self>) -> Self::Element;
//     fn sum_elements(value: Storage<Self>) -> Self::Element;
//     fn prod_elements(value: Storage<Self>) -> Self::Element;
// }

#[rustfmt::skip] #[thermite_macros::reduced_impl]
impl<R: NumericRegister, N: Unsigned> NumericRegister for ReducedRegister<R, N> where R: Reducible<N> {
    const ZERO: Storage<Self> = Self(R::ZERO, PhantomData);
    const ONE: Storage<Self> = Self(R::ONE, PhantomData);
    const TWO: Storage<Self> = Self(R::TWO, PhantomData);

    const MIN: Storage<Self> = Self(R::MIN, PhantomData);
    const MAX: Storage<Self> = Self(R::MAX, PhantomData);

    #[conditional] fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn square(value: Storage<Self>) -> Storage<Self> {}

    #[conditional] fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}

    fn sort(value: Storage<Self>) -> Storage<Self> {
        if const { N::USIZE == 0 } {
            Self(R::sort(value.0), PhantomData)
        } else {
            // fallback to generic sort
            crate::backend::generic::polyfills::sort::sort_any::<Self>(value)
        }
    }

    fn offset() -> Storage<Self> {
        let mut offset = R::offset();

        for _ in 0..N::USIZE {
            offset = R::sub(offset, R::ONE);
        }

        Self(offset, PhantomData)
    }

    fn indexed() -> Storage<Self> { Self(R::indexed(), PhantomData) }

    fn min_element(value: Storage<Self>) -> Self::Element {
        if const { N::USIZE == 0 } {
            return R::min_element(value.0);
        }

        // conditionally broadcast lane 0 to upper lanes.
        R::min_element(R::broadcast_c::<0>(Self::inv_mask(), value.0))
    }

    fn max_element(value: Storage<Self>) -> Self::Element {
        if const { N::USIZE == 0 } {
            return R::max_element(value.0);
        }

        // conditionally broadcast lane 0 to upper lanes.
        R::max_element(R::broadcast_c::<0>(Self::inv_mask(), value.0))
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element {
        if const { N::USIZE == 0 } {
            return R::sum_elements(value.0);
        }

        R::sum_elements(R::blendv(Self::inv_mask(), value.0, R::ZERO))
    }

    fn prod_elements(value: Storage<Self>) -> Self::Element {
        if const { N::USIZE == 0 } {
            return R::prod_elements(value.0);
        }

        R::prod_elements(R::blendv(Self::inv_mask(), value.0, R::ONE))
    }
}

#[rustfmt::skip] #[thermite_macros::reduced_impl]
impl<R: SignedRegister, N: Unsigned> SignedRegister for ReducedRegister<R, N> where R: Reducible<N> {
    #[conditional] fn neg(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn abs(value: Storage<Self>) -> Storage<Self> {}

    fn signum(value: Storage<Self>) -> Storage<Self> {}

    #[conditional] fn copysign(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}

    const NEG_ONE: Storage<Self> = Self(R::NEG_ONE, PhantomData);
    const MIN_POSITIVE: Storage<Self> = Self(R::MIN_POSITIVE, PhantomData);

    fn is_negative(value: Storage<Self>) -> Storage<Self::Mask> {}
    fn is_positive(value: Storage<Self>) -> Storage<Self::Mask> {}

    fn select_negative(value: Storage<Self>, falsy: Storage<Self>, truthy: Storage<Self>) -> Storage<Self> {}
}

#[rustfmt::skip] #[thermite_macros::reduced_impl]
impl<R: IntegerRegister, N: Unsigned> IntegerRegister for ReducedRegister<R, N>
where
    R: Reducible<N>,
{
    #[conditional] fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}

    #[conditional] fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}

    #[conditional] fn div_branched(value: Storage<Self>, divider: Divider<Self::Element>) -> Storage<Self> {}
    #[conditional] fn div_branchfree(value: Storage<Self>, divider: BranchfreeDivider<Self::Element>) -> Storage<Self> {}

    fn divv_branchfree(value: Storage<Self>, dividers: VectorDivider<Self>) -> Storage<Self> {
        Self(R::divv_branchfree(value.0, VectorDivider {
            multipliers: crate::Vector(dividers.multipliers.0.0),
            shifts: crate::Vector(dividers.shifts.0.0),
        }), PhantomData)
    }

    fn divv_branchfree_c(mask: Storage<Self::Mask>, value: Storage<Self>, dividers: VectorDivider<Self>) -> Storage<Self> {
        Self(R::divv_branchfree_c(mask.0, value.0, VectorDivider {
            multipliers: crate::Vector(dividers.multipliers.0.0),
            shifts: crate::Vector(dividers.shifts.0.0),
        }), PhantomData)
    }

    fn divv_branchfree_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>, dividers: VectorDivider<Self>) -> Storage<Self> {
        Self(R::divv_branchfree_m(src.0, mask.0, value.0, VectorDivider {
            multipliers: crate::Vector(dividers.multipliers.0.0),
            shifts: crate::Vector(dividers.shifts.0.0),
        }), PhantomData)
    }

    fn divv_branchfree_z(mask: Storage<Self::Mask>, value: Storage<Self>, dividers: VectorDivider<Self>) -> Storage<Self> {
        Self(R::divv_branchfree_z(mask.0, value.0, VectorDivider {
            multipliers: crate::Vector(dividers.multipliers.0.0),
            shifts: crate::Vector(dividers.shifts.0.0),
        }), PhantomData)
    }

    const HAS_HARDWARE_POPCNT: bool = R::HAS_HARDWARE_POPCNT;

    #[conditional] fn count_ones(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn count_zeros(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn leading_zeros(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn leading_ones(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn trailing_ones(value: Storage<Self>) -> Storage<Self> {}
}

#[rustfmt::skip] #[thermite_macros::reduced_impl]
impl<R: UnsignedIntegerRegister, N: Unsigned> UnsignedIntegerRegister for ReducedRegister<R, N>
where
    R: Reducible<N>,
{
    #[conditional] fn ilog2p1(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn next_power_of_two_m1(mut value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn parity(mut value: Storage<Self>) -> Storage<Self> {}

    fn is_power_of_two(value: Storage<Self>) -> Storage<Self::Mask> {}
}

#[rustfmt::skip] #[thermite_macros::reduced_impl]
impl<R: SignedIntegerRegister, N: Unsigned> SignedIntegerRegister for ReducedRegister<R, N> where R: Reducible<N> {
    #[conditional] fn sra(value: Storage<Self>, shift: u32) -> Storage<Self> {}
    #[conditional] fn srai<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn srav(mut value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {}
}

#[rustfmt::skip] #[thermite_macros::reduced_impl]
impl<R: FloatRegister, N: Unsigned> FloatRegister for ReducedRegister<R, N>
where
    R: Reducible<N>,
{
    type Bits = ReducedRegister<R::Bits, N>;
    type SignedBits = ReducedRegister<R::SignedBits, N>;

    type ExtendedPrecision = ReducedRegister<R::ExtendedPrecision, N>;

    const HAS_TRUE_FMA: bool = R::HAS_TRUE_FMA;
    const HALF: Storage<Self> = Self(R::HALF, PhantomData);
    const NEG_ZERO: Storage<Self> = Self(R::NEG_ZERO, PhantomData);
    const INFINITY: Storage<Self> = Self(R::INFINITY, PhantomData);
    const NEG_INFINITY: Storage<Self> = Self(R::NEG_INFINITY, PhantomData);
    const NAN: Storage<Self> = Self(R::NAN, PhantomData);
    const EPSILON: Storage<Self> = Self(R::EPSILON, PhantomData);

    const EXP_MASK: Storage<Self::Bits> = ReducedRegister(R::EXP_MASK, PhantomData);

    const HAS_NATIVE_LDEXP: bool = R::HAS_NATIVE_LDEXP;
    const HAS_NATIVE_FREXP: bool = R::HAS_NATIVE_FREXP;

    unsafe fn block_autovectorization(value: &mut Storage<Self>) {
        unsafe { R::block_autovectorization(&mut value.0) }
    }

    unsafe fn native_ldexp(value: Storage<Self>, exp: Storage<Self::SignedBits>) -> Storage<Self> {
        unsafe { Self(R::native_ldexp(value.0, exp.0), PhantomData) }
    }

    unsafe fn native_frexp(value: Storage<Self>) -> (Storage<Self>, Storage<Self::SignedBits>) {
        let (val, exp) = unsafe { R::native_frexp(value.0) };
        (Self(val, PhantomData), ReducedRegister(exp, PhantomData))
    }

    fn total_order(value: Storage<Self>) -> Storage<Self::SignedBits> {}
    fn is_nan(value: Storage<Self>) -> Storage<Self::Mask> {}
    fn is_finite(value: Storage<Self>) -> Storage<Self::Mask> {}
    fn is_infinite(value: Storage<Self>) -> Storage<Self::Mask> {}
    fn is_normal(value: Storage<Self>) -> Storage<Self::Mask> {}
    fn is_subnormal(value: Storage<Self>) -> Storage<Self::Mask> {}
    fn is_zero_or_subnormal(value: Storage<Self>) -> Storage<Self::Mask> {}

    #[conditional] fn mul_adde(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn mul_sube(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn nmul_adde(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn nmul_sube(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn mul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn mul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn nmul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn nmul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {}

    #[conditional] fn sqrt(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn rcp(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn rsqrt(value: Storage<Self>) -> Storage<Self> {}

    const HAS_APPROX_RSQRT: bool = R::HAS_APPROX_RSQRT;
    const HAS_APPROX_RCP: bool = R::HAS_APPROX_RCP;

    #[conditional] fn floor(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn ceil(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn round(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn trunc(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn fract(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn mul_sign(value: Storage<Self>, sign: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn signed_zero(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn next_up(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn next_down(value: Storage<Self>) -> Storage<Self> {}
}

impl<R: LinAlg3Register, N: Unsigned> LinAlg3Register for ReducedRegister<R, N>
where
    R: Reducible<N>,
    Self::Lanes: ValidLinAlg3Length<Self>,
{
    #[inline(always)]
    fn dot3(lhs: Storage<Self>, rhs: Storage<Self>) -> Self::Element {
        R::dot3(lhs.0, rhs.0)
    }

    #[inline(always)]
    fn cross3<const DOP: bool>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self(R::cross3::<DOP>(lhs.0, rhs.0), PhantomData)
    }

    #[inline(always)]
    fn zero4(value: Storage<Self>) -> Storage<Self> {
        if const { Self::Lanes::USIZE == 3 } {
            value
        } else {
            Self(R::zero4(value.0), PhantomData)
        }
    }

    #[inline(always)]
    fn one4(value: Storage<Self>) -> Storage<Self> {
        if const { Self::Lanes::USIZE == 3 } {
            value
        } else {
            Self(R::one4(value.0), PhantomData)
        }
    }

    #[inline(always)]
    fn min_element3(value: Storage<Self>) -> Self::Element {
        R::min_element3(value.0)
    }

    #[inline(always)]
    fn max_element3(value: Storage<Self>) -> Self::Element {
        R::max_element3(value.0)
    }

    #[inline(always)]
    fn sum_elements3(value: Storage<Self>) -> Self::Element {
        R::sum_elements3(value.0)
    }

    #[inline(always)]
    fn prod_elements3(value: Storage<Self>) -> Self::Element {
        R::prod_elements3(value.0)
    }
}
