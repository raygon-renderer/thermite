#![warn(clippy::missing_safety_doc)]

use core::{marker::PhantomData, ops::Sub};

use crate::{
    BranchfreeDivider, Divider,
    divider::vector::VectorDivider,
    isa::InstructionSet,
    math::policy::Policy,
    register::{InterleaveRegister, NewRegister},
    swizzle::SwizzleIndices,
    vector::NewConst,
};

use super::{
    BitCastRegister, BitshiftRegister, BitwiseRegister, CastMaskRegister, CastRegister, CoreRegister, ExtendRegister,
    FloatRegister, IndexableRegister, IntegerRegister, Lanes, LinAlg3Register, MaskRegister, NativeCapability,
    NumericRegister, PartialOrdRegister, Register, SignedIntegerRegister, SignedRegister, Storage,
    UnsignedIntegerRegister, ValidLinAlg3Length, ZeroUpper,
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

impl<R: CoreRegister, N: Unsigned> core::fmt::Debug for ReducedRegister<R, N>
where
    R: CoreReducible<N>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Can't use `as_slice` here to trim the underlying register, so just zero the
        // trailing lanes for more ergonomics Debug views.
        let cleaned = R::zz(Self::mask(), self.0);
        f.debug_tuple("ReducedRegister").field(&cleaned).finish()
    }
}

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

    /// Consider f32x3: swizzle!(a, b, [0, 1, 3]) should be equivalent to swizzle!(a as f32x4, b as f32x4, [0, 1, 4]),
    /// since with f32x3 index 3 would be the first element of b, but with f32x4 it's still the first element of b,
    /// but we need to offset it by the difference in length, which is always N for this design.
    #[inline(always)]
    fn adjust_swizzle_idxs(idxs: GenericArray<u32, <Self as CoreRegister>::Lanes>) -> GenericArray<u32, R::Lanes> {
        let reduced_lanes = <Self as CoreRegister>::Lanes::U32;
        Self::pad_array(idxs.map(|idx| if idx >= reduced_lanes { idx + N::U32 } else { idx }))
    }
}

impl<R: CoreRegister, N: Unsigned> CoreRegister for ReducedRegister<R, N>
where
    R: CoreReducible<N>,
{
    type NativeIsa = R::NativeIsa;
    type Lanes = Diff<R::Lanes, N>;
    type Storage = Self;
    type Mask = ReducedRegister<R::Mask, N>;

    const IS_EMULATED: bool = R::IS_EMULATED;

    const ISA: InstructionSet = R::ISA;
    const EMPTY: Storage<Self> = Self(R::EMPTY, PhantomData);
    const HAS_EQUAL_SIZE_MASK: bool = R::HAS_EQUAL_SIZE_MASK;

    #[inline(always)]
    fn blendv(mask: Storage<Self::Mask>, on_false: Storage<Self>, on_true: Storage<Self>) -> Storage<Self> {
        Self(R::blendv(mask.0, on_false.0, on_true.0), PhantomData)
    }

    #[inline(always)]
    fn zz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        Self(R::zz(mask.0, value.0), PhantomData)
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

    #[inline(always)]
    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        Self(R::from_mask(mask.0), PhantomData)
    }
}

#[rustfmt::skip] #[thermite_macros::reduced_impl]
impl<R: BitwiseRegister, N: Unsigned> BitwiseRegister for ReducedRegister<R, N> where R: CoreReducible<N> {
    const HAS_NATIVE_TERNLOG: bool = R::HAS_NATIVE_TERNLOG;

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

    #[cfg(feature = "bitvec")]
    #[inline(always)]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let Some(native) = R::native_bitmask(value.0) else {
            unsafe { core::hint::unreachable_unchecked() }
        };

        use bitvec::slice::BitSlice;

        let bits = BitSlice::<u32>::from_slice(unsafe { core::mem::transmute::<&u64, &[u32; 2]>(&native) });

        view[..Self::Lanes::USIZE].copy_from_bitslice(&bits[..Self::Lanes::USIZE]);
    }

    // Drop bits belonging to the dead upper lanes before handing the word to the
    // wider register, mirroring what `native_bitmask` masks off on the way out.
    #[inline(always)]
    fn from_native_bitmask(bitmask: u64) -> Storage<Self> {
        Self(R::from_native_bitmask(bitmask & Self::BITMASK), PhantomData)
    }
}

impl<R: InterleaveRegister, N: Unsigned> InterleaveRegister for ReducedRegister<R, N>
where
    R: CoreReducible<N>,
{
    // Both methods exploit the fact that interleaving the L = M - N meaningful
    // lanes produces a 2L-element stream that is a *prefix* of the underlying
    // M-lane register's full 2M-element interleave stream. The upper lanes of
    // reduced registers are don't-cares, so the leftover full-width lanes can be
    // ignored (interleave) or overwritten (deinterleave).
    //
    // Only `InterleaveRegister` + flat lane storage are required, which keeps
    // this implementation valid for mask registers (which are not `Register`,
    // so no `as_slice`/swizzle access exists here).

    #[inline(always)]
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        const {
            assert!(
                size_of::<Storage<R>>().is_multiple_of(<R::Lanes as Unsigned>::USIZE),
                "ReducedRegister interleave requires flat lane storage (do not nest ReducedRegister)"
            );
        }

        // stream[0..L] is the low half of the full interleave, unchanged.
        let (full_lo, full_hi) = R::interleave(a.0, b.0);

        // stream[L..2L] is a *contiguous* lane window starting at lane L of the
        // (full_lo ++ full_hi) pair: spill the pair and take one unaligned load.
        // For the common 4 -> 3 lane (`f32x3A`) case this is the fast path:
        // native interleave + two stores + one `movups`-class load, no lane loop.
        let hi = unsafe {
            let buf = [full_lo, full_hi];
            let elem = const { size_of::<Storage<R>>() / <R::Lanes as Unsigned>::USIZE };
            let offset = <Self as CoreRegister>::Lanes::USIZE * elem;
            core::ptr::read_unaligned(buf.as_ptr().cast::<u8>().add(offset).cast::<Storage<R>>())
        };

        (Self(full_lo, PhantomData), Self(hi, PhantomData))
    }

    #[inline(always)]
    fn deinterleave(lo: Storage<Self>, hi: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        const {
            assert!(
                size_of::<Storage<R>>().is_multiple_of(<R::Lanes as Unsigned>::USIZE),
                "ReducedRegister deinterleave requires flat lane storage (do not nest ReducedRegister)"
            );
        }

        unsafe {
            let elem = const { size_of::<Storage<R>>() / <R::Lanes as Unsigned>::USIZE };
            let offset = <Self as CoreRegister>::Lanes::USIZE * elem;

            // Rebuild the contiguous 2L-element stream by storing `hi` *overlapping*
            // at lane offset L, overwriting `lo`'s ignored padding lanes.
            let mut buf = [lo.0, hi.0];
            let base = buf.as_mut_ptr().cast::<u8>();
            core::ptr::write_unaligned(base.add(offset).cast::<Storage<R>>(), hi.0);

            // Full-width deinterleave of the two M-lane windows: the evens/odds of
            // stream[0..2M] start with the L evens/odds of the valid 2L prefix,
            // which are exactly the original `a` and `b` (upper lanes are junk).
            let x = core::ptr::read(base.cast::<Storage<R>>());
            let y = core::ptr::read(base.add(size_of::<Storage<R>>()).cast::<Storage<R>>());

            let (a, b) = R::deinterleave(x, y);

            (Self(a, PhantomData), Self(b, PhantomData))
        }
    }
}

impl<R: Register, N: Unsigned> NewRegister<R::Element, <Self as CoreRegister>::Lanes, Storage<Self>>
    for ReducedRegister<R, N>
where
    R: CoreReducible<N>,
{
    type New<C: NewConst<R::Element, <Self as CoreRegister>::Lanes>> = ReducedNewConst<C, R, N>;
}

#[doc(hidden)]
pub struct ReducedNewConst<C, R, N>(PhantomData<(C, R, N)>);

struct ReducedPaddedConst<C, R, N>(PhantomData<(C, R, N)>);

impl<C, R: Register, N: Unsigned> NewConst<R::Element, R::Lanes> for ReducedPaddedConst<C, R, N>
where
    R: CoreReducible<N>,
    C: NewConst<R::Element, <ReducedRegister<R, N> as CoreRegister>::Lanes>,
{
    const VALUES: GenericArray<R::Element, R::Lanes> = const {
        let mut values: GenericArray<R::Element, R::Lanes> = unsafe { core::mem::zeroed() };
        let c_values = C::VALUES;
        let src = c_values.as_slice();
        let dst = values.as_mut_slice();

        let mut i = 0;
        while i < <ReducedRegister<R, N> as CoreRegister>::Lanes::USIZE {
            dst[i] = src[i];
            i += 1;
        }

        core::mem::forget(c_values);
        values
    };
}

impl<C, R: Register, N: Unsigned> crate::vector::VectorValue<C, Storage<ReducedRegister<R, N>>>
    for ReducedNewConst<C, R, N>
where
    R: CoreReducible<N>,
    C: NewConst<R::Element, <ReducedRegister<R, N> as CoreRegister>::Lanes>,
{
    const VALUE: Storage<ReducedRegister<R, N>> = {
        ReducedRegister(
            <<R as NewRegister<R::Element, R::Lanes, Storage<R>>>::New<ReducedPaddedConst<C, R, N>>
                as crate::vector::VectorValue<ReducedPaddedConst<C, R, N>, Storage<R>>>::VALUE,
            PhantomData,
        )
    };
}

#[rustfmt::skip]
impl<R: Register, N: Unsigned> Register for ReducedRegister<R, N> where R: Reducible<N> {
    type Element = R::Element;

    #[inline(always)] fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> { ReducedRegister(R::into_mask(value.0), PhantomData) }
    #[inline(always)] fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> { ReducedRegister(R::into_mask_unchecked(value.0), PhantomData) }
    #[inline(always)] fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> { ReducedRegister(R::msb_to_mask(value.0), PhantomData) }

    type Unsigned = ReducedRegister<R::Unsigned, N>;
    type Signed = ReducedRegister<R::Signed, N>;

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        Self(R::new(Self::pad_array(value)), PhantomData)
    }

    #[inline(always)] fn single(value: Self::Element) -> Storage<Self> { Self(R::single(value), PhantomData) }

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

    // Last LIVE lane, not the carrier's top (padding) lane.
    #[inline(always)] fn last_element(value: Storage<Self>) -> Self::Element { R::as_slice(&value.0)[Self::Lanes::USIZE - 1] }

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
        for i in 0..Self::lanes() {
            Self::as_mut_slice(&mut res)[i] = unsafe { ptr.add(i).read_unaligned() };
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
        for i in 0..Self::lanes() {
            unsafe { ptr.add(i).write_unaligned(Self::as_slice(&value)[i]) };
        }
    }

    #[inline(always)] fn swap_bytes(value: Storage<Self>) -> Storage<Self> { Self(R::swap_bytes(value.0), PhantomData) }
    #[inline(always)] fn swap_bytes_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> { Self(R::swap_bytes_c(mask.0, value.0), PhantomData) }
    #[inline(always)] fn swap_bytes_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> { Self(R::swap_bytes_m(src.0, mask.0, value.0), PhantomData) }
    #[inline(always)] fn swap_bytes_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> { Self(R::swap_bytes_z(mask.0, value.0), PhantomData) }

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
    fn permutev_const<I: SwizzleIndices<Self::Lanes>>(value: Storage<Self>) -> Storage<Self> {
        Self(
            // technically this just pads it, since the index should never be over the reduced lanes, but
            // the adjusted indices type is just convenient to reuse.
            R::permutev_const::<AdjustedIndices<Self::Lanes, R::Lanes, I>>(value.0),
            PhantomData,
        )
    }

    #[inline(always)]
    fn swizzle(a: Storage<Self>, b: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        Self(R::swizzle(a.0, b.0, Self::adjust_swizzle_idxs(idxs)), PhantomData)
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
            R::swizzle_m(src.0, mask.0, a.0, b.0, Self::adjust_swizzle_idxs(idxs)),
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
        Self(
            R::swizzle_z(mask.0, a.0, b.0, Self::adjust_swizzle_idxs(idxs)),
            PhantomData,
        )
    }

    #[inline(always)]
    fn swizzle_const<I: SwizzleIndices<Self::Lanes>>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        Self(
            R::swizzle_const::<AdjustedIndices<Self::Lanes, R::Lanes, I>>(a.0, b.0),
            PhantomData,
        )
    }
}

struct AdjustedIndices<N: Lanes, M: Lanes, I: SwizzleIndices<N>>(PhantomData<(N, M, I)>);

// M is always larger, so M - N is always the Reduced N, which is how much we need to shift indices that reference `b` by
impl<N: Lanes, M: Lanes, I: SwizzleIndices<N>> SwizzleIndices<M> for AdjustedIndices<N, M, I> {
    const INDICES: GenericArray<u32, M> = const {
        let mut indices: GenericArray<u32, M> = unsafe { core::mem::zeroed() };
        let old = I::INDICES;

        let new_idxs = indices.as_mut_slice();
        let old_idxs = old.as_slice();

        let mut i = 0;

        while i < N::USIZE {
            let idx = old_idxs[i];

            new_idxs[i] = if idx < N::U32 { idx } else { idx + (M::U32 - N::U32) };

            i += 1;
        }

        // copy over padding lanes exactly
        while i < M::USIZE {
            new_idxs[i] = i as u32;
            i += 1;
        }

        core::mem::forget(old);

        indices
    };
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

#[rustfmt::skip] #[thermite_macros::reduced_impl]
impl<R: BitshiftRegister, N: Unsigned> BitshiftRegister for ReducedRegister<R, N> where R: Reducible<N> {
    #[conditional] fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {}
    #[conditional] fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {}
    #[conditional] fn shli<const IMM: i32>(lhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn shri<const IMM: i32>(lhs: Storage<Self>) -> Storage<Self> {}

    const HAS_WIDE_BYTE_SHIFTS: bool = R::HAS_WIDE_BYTE_SHIFTS;

    // bshli is safe to delegate: bytes only move toward *higher* lanes, so the
    // padding-lane junk never enters the valid region (it only collects more junk).
    #[conditional] fn bshli<const IMM8: i32>(lhs: Storage<Self>) -> Storage<Self> {}

    // bshri must NOT be a plain delegation: a full-width byte shift pulls the
    // padding-lane junk *down* into the valid lanes. Zeroing the padding first
    // makes the full-width shift exactly match an L-lane-wide register (zeros
    // shift in from the top). No `#[conditional]` here: the masked variants then
    // come from the trait defaults, which are built on this corrected body
    // (the macro-generated variants would delegate to R's junk-leaking ones).
    fn bshri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        Self(R::bshri::<IMM8>(R::zz(Self::mask(), value.0)), PhantomData)
    }

    const HAS_TRUE_SHIFTV: bool = R::HAS_TRUE_SHIFTV;

    #[conditional] fn shrv(value: Storage<Self>, count: Storage<Self::Unsigned>) -> Storage<Self> {}
    #[conditional] fn shlv(value: Storage<Self>, count: Storage<Self::Unsigned>) -> Storage<Self> {}

    #[conditional] fn rol(value: Storage<Self>, shift: u32) -> Storage<Self> {}
    #[conditional] fn ror(value: Storage<Self>, shift: u32) -> Storage<Self> {}

    #[conditional] fn roli<const IMM: i32>(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn rori<const IMM: i32>(value: Storage<Self>) -> Storage<Self> {}

    #[conditional] fn rolv(value: Storage<Self>, count: Storage<Self::Unsigned>) -> Storage<Self> {}
    #[conditional] fn rorv(value: Storage<Self>, count: Storage<Self::Unsigned>) -> Storage<Self> {}

    // Per-lane bit reversal: padding-lane junk stays in its own lanes.
    #[conditional] fn reverse_bits(value: Storage<Self>) -> Storage<Self> {}
}

#[thermite_macros::inline_always]
impl<FROM, TO, N: Unsigned> CastRegister<ReducedRegister<FROM, N>> for ReducedRegister<TO, N>
where
    TO: CoreReducible<N> + CastRegister<FROM>,
    FROM: CoreReducible<N>,
{
    fn cast_from(value: Storage<ReducedRegister<FROM, N>>) -> Storage<Self> {
        Self(TO::cast_from(value.0), PhantomData)
    }

    fn fast_cast_from(value: Storage<ReducedRegister<FROM, N>>) -> Storage<Self> {
        Self(TO::fast_cast_from(value.0), PhantomData)
    }

    fn saturating_cast_from(value: Storage<ReducedRegister<FROM, N>>) -> Storage<Self> {
        Self(TO::saturating_cast_from(value.0), PhantomData)
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

/// Stamp the extend-from-scalar impl (`ExtendRegister<$elem>`) on `ReducedRegister`
/// for every scalar element type, satisfying the `Register: ExtendRegister<Self::Element>`
/// supertrait for every reduced width on every backend at once.
///
/// The inner register `R` stays *generic* -- that is what makes this cover the
/// `Simd3A`/`Simd3` associated-type defaults (`ReducedRegister<S::f32x4, U1>` and
/// friends), whose inner is an opaque projection that no concrete stamp could name.
///
/// The `FROM` type, on the other hand, MUST be a concrete scalar. Written with an
/// opaque `R::Element` as `FROM` it overlaps the extend-from-inner impl above
/// (`ExtendRegister<ReducedRegister<R, N>> for R`), since the compiler cannot prove
/// an associated type distinct from `ReducedRegister<_, _>`. A concrete `f32` is
/// provably distinct, so there is no overlap.
macro_rules! impl_reduced_extend_from_scalar {
    ($($elem:ty),* $(,)?) => {$(
        #[thermite_macros::inline_always]
        impl<R, N: Unsigned> ExtendRegister<$elem> for ReducedRegister<R, N>
        where
            R: Register<Element = $elem> + CoreReducible<N> + ExtendRegister<$elem>,
        {
            fn extend(value: Storage<$elem>) -> Storage<Self> {
                // The scalar lands in lane 0 with the rest zeroed, which is exactly
                // the reduced register's dead-upper-lane invariant.
                ReducedRegister(R::extend(value), PhantomData)
            }

            fn narrow(value: Storage<Self>) -> Storage<$elem> {
                R::narrow(value.0)
            }
        }
    )*};
}

// `usize` needs no entry: `FindUSize` aliases the `usizexN` slots onto `u32`/`u64`.
impl_reduced_extend_from_scalar!(f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);

#[rustfmt::skip] #[thermite_macros::reduced_impl]
impl<R: PartialOrdRegister, N: Unsigned> PartialOrdRegister for ReducedRegister<R, N> where R: Reducible<N> {
    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {}
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {}
    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {}
    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {}
    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {}
    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {}
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

    fn is_all_zero(value: Storage<Self>) -> bool {
        R::is_all_zero(Self::zeroupper(value).0)
    }

    #[conditional] fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn square(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn scale(value: Storage<Self>, scalar: Self::Element) -> Storage<Self> {}

    #[conditional] fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}

    fn sort_by<O: crate::sort::SortOrder>(value: Storage<Self>) -> Storage<Self> {
        if const { N::USIZE == 0 } {
            Self(R::sort_by::<O>(value.0), PhantomData)
        } else {
            // fallback to generic sort
            crate::backend::generic::polyfills::sort::sort_any::<Self, O>(value)
        }
    }

    // Paired with `sort_by` on purpose: with no dead lanes this register *is*
    // the wider one, so the same delegation applies, and a fast `sort_by` beside
    // a defaulted clean would make every cross-register merge silently quadratic
    // with all tests still green (the `sort_via_network!` no-drift rule).
    fn bitonic_clean_by<O: crate::sort::SortOrder>(value: Storage<Self>) -> Storage<Self> {
        if const { N::USIZE == 0 } {
            Self(R::bitonic_clean_by::<O>(value.0), PhantomData)
        } else {
            crate::backend::generic::polyfills::sort::sort_any::<Self, O>(value)
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

    fn pairwise_sum(lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self> {
        Self(R::pairwise_sum(lo.0, hi.0), PhantomData)
    }

    fn relaxed_pairwise_sum(lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self> {
        Self(R::relaxed_pairwise_sum(lo.0, hi.0), PhantomData)
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
    #[conditional] fn next_power_of_two_m1(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn parity(value: Storage<Self>) -> Storage<Self> {}

    fn avg(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {}
    fn abs_diff(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {}
    fn is_power_of_two(value: Storage<Self>) -> Storage<Self::Mask> {}

    // Hand-written: [Storage<Self>; D] is not splittable by reduced_impl.
    // Per-lane bit ops, so padding-lane junk stays in its own lanes.
    fn morton<const D: usize>(values: [Storage<Self>; D]) -> Storage<Self> {
        let mut inner = [R::ZERO; D];
        for k in 0..D {
            inner[k] = values[k].0;
        }
        Self(R::morton(inner), PhantomData)
    }

    fn reverse_morton<const D: usize>(code: Storage<Self>) -> [Storage<Self>; D] {
        let coords = R::reverse_morton::<D>(code.0);
        let mut out = [Self::ZERO; D];
        for k in 0..D {
            out[k] = Self(coords[k], PhantomData);
        }
        out
    }
}

#[rustfmt::skip] #[thermite_macros::reduced_impl]
impl<R: SignedIntegerRegister, N: Unsigned> SignedIntegerRegister for ReducedRegister<R, N> where R: Reducible<N> {
    #[conditional] fn sra(value: Storage<Self>, shift: u32) -> Storage<Self> {}
    #[conditional] fn srai<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn srav(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {}
    fn avg_floor(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {}
    fn avg_ceil(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {}
    fn mulhrs(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {}
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

    const NATIVE_CAP: NativeCapability = R::NATIVE_CAP;

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

    unsafe fn native_sin_cos<P: Policy>(value: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        let (s, c) = unsafe { R::native_sin_cos::<P>(value.0) };
        (Self(s, PhantomData), Self(c, PhantomData))
    }

    unsafe fn native_sin<P: Policy>(value: Storage<Self>) -> Storage<Self> {
        Self(unsafe { R::native_sin::<P>(value.0) }, PhantomData)
    }

    unsafe fn native_cos<P: Policy>(value: Storage<Self>) -> Storage<Self> {
        Self(unsafe { R::native_cos::<P>(value.0) }, PhantomData)
    }

    unsafe fn native_tan<P: Policy>(value: Storage<Self>) -> Storage<Self> {
        Self(unsafe { R::native_tan::<P>(value.0) }, PhantomData)
    }

    unsafe fn native_exp2<P: Policy>(value: Storage<Self>) -> Storage<Self> {
        Self(unsafe { R::native_exp2::<P>(value.0) }, PhantomData)
    }

    unsafe fn native_log2<P: Policy>(value: Storage<Self>) -> Storage<Self> {
        Self(unsafe { R::native_log2::<P>(value.0) }, PhantomData)
    }

    unsafe fn native_exp<P: Policy>(value: Storage<Self>) -> Storage<Self> {
        Self(unsafe { R::native_exp::<P>(value.0) }, PhantomData)
    }

    unsafe fn native_ln<P: Policy>(value: Storage<Self>) -> Storage<Self> {
        Self(unsafe { R::native_ln::<P>(value.0) }, PhantomData)
    }

    unsafe fn native_powf<P: Policy>(base: Storage<Self>, exp: Storage<Self>) -> Storage<Self> {
        Self(unsafe { R::native_powf::<P>(base.0, exp.0) }, PhantomData)
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

    fn mix(a: Storage<Self>, b: Storage<Self>, t: Storage<Self>) -> Storage<Self> {}
    // Live lanes are the LOW lanes of R, so R's even/odd parity matches.
    #[conditional] fn addsub(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn fmaddsub(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn fmsubadd(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {}
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
    fn cross3<const FAST: bool>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self(R::cross3::<FAST>(lhs.0, rhs.0), PhantomData)
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

    #[inline(always)]
    fn mat3_transpose(cols: &[Storage<Self>; 3]) -> [Storage<Self>; 3] {
        // Entirely delegate to the inner register's method
        unsafe { generic_array::const_transmute(R::mat3_transpose(core::mem::transmute(cols))) }
    }

    #[inline(always)]
    fn mat3_vec3_product<const COLUMN_MAJOR: bool, const M: usize>(
        cols: &[Storage<Self>; 3],
        vectors: &[Storage<Self>; M],
    ) -> [Storage<Self>; M] {
        // Entirely delegate to the inner register's method.
        unsafe {
            let raw =
                R::mat3_vec3_product::<COLUMN_MAJOR, M>(core::mem::transmute(cols), core::mem::transmute(vectors));
            // Storage<Self> is layout-compatible with Storage<R>.
            core::mem::transmute_copy::<[Storage<R>; M], [Storage<Self>; M]>(&raw)
        }
    }
}
