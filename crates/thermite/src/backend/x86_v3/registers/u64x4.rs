use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, BitwiseRegister, CastRegister, ConcatRegister, CoreRegister, ExtendRegister,
        IndexableRegister, IntegerRegister, InterleaveRegister, MaskElement, MaskRegister, NumericRegister,
        PartialOrdRegister, Register, SignedRegister, Storage, UnsignedIntegerRegister, array::ArrayRegister,
        empty_reg, reg,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct U64x4V3;

#[thermite_macros::inline_always]
impl CoreRegister for U64x4V3 {
    type NativeIsa = crate::backend::x86_v3::X86V3;
    type Lanes = typenum::U4;
    type Storage = arch::__m256i;
    type Mask = Self;

    const IS_EMULATED: bool = false;
    const EMPTY: Storage<Self> = empty_reg::<Self>();
    const HAS_EQUAL_SIZE_MASK: bool = true;

    fn blendv(mask: Storage<Self::Mask>, on_false: Storage<Self>, on_true: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_blendv_epi8(on_false, on_true, mask) }
    }

    fn zz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_and_si256(value, mask) }
    }

    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_andnot_si256(mask, value) }
    }

    fn zeroupper_z<Z: crate::register::ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        super::I64x4V3::zeroupper_z::<Z>(value)
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        mask
    }
}

#[thermite_macros::inline_always]
impl MaskRegister for U64x4V3 {
    const FALSY: Storage<Self> = reg::<Self, 4>([0; 4]);
    const TRUTHY: Storage<Self> = reg::<Self, 4>([!0; 4]);

    fn set(mut mask: Storage<Self::Mask>, lane: usize, value: bool) -> Storage<Self> {
        Self::as_mut_slice(&mut mask)[lane] = if value { MaskElement::TRUTHY } else { MaskElement::FALSY };
        mask
    }

    fn test(mask: Storage<Self::Mask>, lane: usize) -> bool {
        Self::as_slice(&mask)[lane].to_bool()
    }

    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm256_cvtboolx4_to_epi64_mask_v3(value) }
    }

    fn all(value: Storage<Self>) -> bool {
        unsafe { arch::_mm256_movemask_epi8(value) as u32 == 0xFFFF_FFFF }
    }

    fn any(value: Storage<Self>) -> bool {
        unsafe { arch::_mm256_movemask_epi8(value) != 0 }
    }

    fn none(value: Storage<Self>) -> bool {
        unsafe { arch::_mm256_movemask_epi8(value) == 0 }
    }

    fn from_native_bitmask(bitmask: u64) -> Storage<Self> {
        unsafe { arch::_mm256_movm_epi64x_v3(bitmask) }
    }

    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        Some(unsafe { arch::_mm256_movemask_pd(arch::_mm256_castsi256_pd(value)) as u64 })
    }

    #[cfg(feature = "bitvec")]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = unsafe { arch::_mm256_movemask_pd(arch::_mm256_castsi256_pd(value)) as u32 };
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for U64x4V3 {
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_xor_si256(lhs, rhs) }
    }

    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_and_si256(lhs, rhs) }
    }

    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_andnot_si256(lhs, rhs) }
    }

    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_or_si256(lhs, rhs) }
    }

    fn not(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_xor_si256(value, arch::_mm256_set1_epi8(-1)) }
    }
}

#[thermite_macros::inline_always]
impl ConcatRegister<super::U64x2V3> for U64x4V3 {
    fn concat(lo: Storage<super::U64x2V3>, hi: Storage<super::U64x2V3>) -> Storage<Self> {
        unsafe { arch::_mm256_setr_m128i(lo, hi) }
    }

    fn split(value: Storage<Self>) -> (Storage<super::U64x2V3>, Storage<super::U64x2V3>) {
        let lo = unsafe { arch::_mm256_castsi256_si128(value) };
        let hi = unsafe { arch::_mm256_extracti128_si256(value, 1) };

        (lo, hi)
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<super::U64x2V3> for U64x4V3 {
    fn extend(value: Storage<super::U64x2V3>) -> Storage<Self> {
        unsafe { arch::_mm256_zextsi128_si256(value) }
    }

    fn narrow(value: Storage<Self>) -> Storage<super::U64x2V3> {
        unsafe { arch::_mm256_castsi256_si128(value) }
    }
}

#[thermite_macros::inline_always]
impl Register for U64x4V3 {
    type Element = u64;

    type Signed = super::I64x4V3;
    type Unsigned = super::U64x4V3;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::ne(value, Self::ZERO)
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        value
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        super::I64x4V3::is_negative(value) // reuse signed implementation
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm256_loadu_si256(value.as_ptr() as *const _) }
    }

    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_setr_epi64x(value as i64, 0, 0, 0) }
    }

    impl_native_radix3!(arch::_mm256_interleave3_epi64, arch::_mm256_deinterleave3_epi64);

    impl_native_extract!(@epi64x4);

    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_set1_epi64x(value as i64) }
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_load_si256(ptr as *const _) }
    }

    unsafe fn load_m(src: Storage<Self>, mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        // use load_z + 2 bitwise ops to emulate load_m without blendv or scalar fallbacks
        unsafe { Self::bitor(Self::load_z(mask, ptr), Self::bitandnot(mask, src)) }
    }

    unsafe fn load_z(mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_maskload_epi64(ptr as *const _, mask) }
    }

    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_loadu_si256(ptr as *const _) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm256_store_si256(ptr as *mut _, value) }
    }

    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm256_storeu_si256(ptr as *mut _, value) }
    }

    unsafe fn load_stream(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_stream_load_si256(ptr as _) }
    }

    unsafe fn store_stream(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm256_stream_si256(ptr as _, value) }
    }

    unsafe fn lookup(values: &[Self::Element], indices: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { <Self::Signed as Register>::lookup(core::mem::transmute(values), indices) }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_permute4x64_epi64::<{ MM_SHUFFLE!(0, 1, 2, 3) }>(value) }
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_bswap_epi64x_v3(value) }
    }

    const HAS_PERMUTEV: bool = true;

    // Square transpose via the shared 256-bit family body (identical to `I64x4V3` -
    // both are raw `__m256i`): `(4, 1)` -> the 4x4 W=8 transpose. Self-inverse.
    fn deinterleave_radix_by<const N: usize, const GROUP: usize>(inputs: [Storage<Self>; N]) -> [Storage<Self>; N] {
        if const { N == 4 && GROUP == 1 } {
            unsafe { arch::radix_by_w64_si(inputs) }
        } else if const { arch::ladder_viable(N, GROUP, 8) } {
            // Certified ladder plan for any other pow-2 shape (see `polyfills::transpose256`).
            let plan = const { arch::ladder_search_elem(N, GROUP, 8) };
            unsafe { arch::ladder_radix_by_si::<N, true>(inputs, plan) }
        } else {
            crate::backend::generic::polyfills::deinterleave_radix_by_default::<Self, N, GROUP>(inputs)
        }
    }

    fn interleave_radix_by<const N: usize, const GROUP: usize>(inputs: [Storage<Self>; N]) -> [Storage<Self>; N] {
        if const { N == 4 && GROUP == 1 } {
            unsafe { arch::radix_by_w64_si(inputs) }
        } else if const { arch::ladder_viable(N, GROUP, 8) } {
            let plan = const { arch::ladder_search_elem(N, GROUP, 8) };
            unsafe { arch::ladder_radix_by_si::<N, false>(inputs, plan) }
        } else {
            crate::backend::generic::polyfills::interleave_radix_by_default::<Self, N, GROUP>(inputs)
        }
    }

    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        // Byte-identical to `I64x4V3::permutev` (both raw `__m256i`); see the
        // note there - `compress` sits on this, and the generic scalar
        // fallback it replaces was 6.9x per partition pass.
        super::I64x4V3::permutev(value, idxs)
    }

    compress_via_table!();

    // Byte-identical to the signed register (both raw `__m256i`); reuse it.
    // hand-written body below, so the flag is set here rather than by an
    // `impl_*_align*!` macro
    const HAS_NATIVE_ALIGN: bool = true;

    fn align<const OFFSET: usize>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        super::I64x4V3::align::<OFFSET>(a, b)
    }
}

#[thermite_macros::inline_always]
impl InterleaveRegister for U64x4V3 {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        super::I64x4V3::interleave(a, b) // reuse signed implementation
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        super::I64x4V3::deinterleave(a, b) // reuse signed implementation
    }
}

impl<I> IndexableRegister<I> for U64x4V3
where
    I: UnsignedIntegerRegister<Lanes = Self::Lanes>,
    super::I64x4V3: IndexableRegister<I>,
{
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<I>) -> Storage<Self> {
        unsafe { super::I64x4V3::gather(ptr as *const _, indices) }
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<I>,
    ) -> Storage<Self> {
        unsafe { super::I64x4V3::gather_m(src, mask, ptr as *const _, indices) }
    }

    unsafe fn gather_z(mask: Storage<Self::Mask>, ptr: *const Self::Element, indices: Storage<I>) -> Storage<Self> {
        unsafe { super::I64x4V3::gather_z(mask, ptr as *const _, indices) }
    }
}

#[thermite_macros::inline_always]
impl IndexableRegister<super::U32x8V3> for ArrayRegister<U64x4V3, 2> {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<super::U32x8V3>) -> Storage<Self> {
        let (lo, hi) = <super::U32x8V3>::split(indices);

        unsafe {
            let lo = arch::_mm256_i32gather_epi64::<8>(ptr as *const _, lo);
            let hi = arch::_mm256_i32gather_epi64::<8>(ptr as *const _, hi);

            ArrayRegister([lo, hi])
        }
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<super::U32x8V3>,
    ) -> Storage<Self> {
        let (lo, hi) = <super::U32x8V3>::split(indices);

        unsafe {
            let lo = arch::_mm256_mask_i32gather_epi64::<8>(src.0[0], ptr as *const _, lo, mask.0[0]);
            let hi = arch::_mm256_mask_i32gather_epi64::<8>(src.0[1], ptr as *const _, hi, mask.0[1]);

            ArrayRegister([lo, hi])
        }
    }
}

#[thermite_macros::inline_always]
impl BitshiftRegister for U64x4V3 {
    const HAS_TRUE_SHIFTV: bool = true;
    const HAS_WIDE_BYTE_SHIFTS: bool = false;

    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm256_sll_epi64(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm256_srl_epi64(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    fn shrv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm256_srlv_epi64(value, shifts) }
    }

    fn shlv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm256_sllv_epi64(value, shifts) }
    }

    fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_slli_epi64(value, IMM8) }
    }

    fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_srli_epi64(value, IMM8) }
    }
}

#[thermite_macros::inline_always]
impl PartialOrdRegister for U64x4V3 {
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmpgt_epu64x_v3(lhs, rhs) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmpeq_epi64(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl NumericRegister for U64x4V3 {
    sort_via_network!(4);

    const ZERO: Storage<Self> = reg::<Self, 4>([0; 4]);
    const ONE: Storage<Self> = reg::<Self, 4>([1; 4]);
    const TWO: Storage<Self> = reg::<Self, 4>([2; 4]);

    const MIN: Storage<Self> = reg::<Self, 4>([u64::MIN; 4]);
    const MAX: Storage<Self> = reg::<Self, 4>([u64::MAX; 4]);

    fn min_element(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi64_v3!(value; _mm_min_epu64x_v2 _mm_min_epu64x_v2) as u64
    }

    fn max_element(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi64_v3!(value; _mm_max_epu64x_v2 _mm_max_epu64x_v2) as u64
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi64_v3!(value; _mm_add_epi64 _mm_add_epi64) as u64
    }

    fn prod_elements(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi64_v3!(value; _mm_mullo_epi64x_v2 _mm_mullo_epi64x_v2) as u64
    }

    fn relaxed_pairwise_sum(lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self> {
        unsafe {
            // [a0,a1,a2,a3] hadd [b0,b1,b2,b3] -> [a0+a1, b0+b1, a2+a3, b2+b3] (relaxed)
            let lo_pd = arch::_mm256_castsi256_pd(lo);
            let hi_pd = arch::_mm256_castsi256_pd(hi);
            let even = arch::_mm256_shuffle_pd(lo_pd, hi_pd, 0b0000); // [a0,b0,a2,b2]
            let odd = arch::_mm256_shuffle_pd(lo_pd, hi_pd, 0b1111); // [a1,b1,a3,b3]
            arch::_mm256_add_epi64(arch::_mm256_castpd_si256(even), arch::_mm256_castpd_si256(odd))
        }
    }

    fn pairwise_sum(lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self> {
        let relaxed = Self::relaxed_pairwise_sum(lo, hi);
        unsafe {
            arch::_mm256_castpd_si256(arch::_mm256_permute4x64_pd(
                arch::_mm256_castsi256_pd(relaxed),
                0b11_01_10_00,
            ))
        }
    }

    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as typenum::Unsigned>::U64)
    }

    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as u64))
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_add_epi64(lhs, rhs) }
    }

    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_sub_epi64(lhs, rhs) }
    }

    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mullo_epi64x_v3(lhs, rhs) }
    }

    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a / b })
    }

    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a % b })
    }

    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_min_epu64x_v3(lhs, rhs) }
    }

    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_max_epu64x_v3(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl IntegerRegister for U64x4V3 {
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mullhi_epu64x_v3(lhs, rhs) }
    }

    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mullo_epi64x_v3(lhs, rhs) }
    }

    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::add(rhs, Self::min(lhs, Self::not(rhs)))
    }

    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::max(lhs, rhs), rhs)
    }

    fn wrapping_sum(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi64_v3!(value; _mm_add_epi64 _mm_add_epi64) as u64
    }

    fn wrapping_product(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi64_v3!(value; _mm_mullo_epi64x_v2 _mm_mullo_epi64x_v2) as u64
    }

    fn div_branched(value: Storage<Self>, divider: crate::Divider<Self::Element>) -> Storage<Self> {
        arch::div_epu::<Self>(value, divider.multiplier(), divider.shift())
    }

    fn div_branchfree(value: Storage<Self>, divider: crate::BranchfreeDivider<Self::Element>) -> Storage<Self> {
        arch::div_epu_bf::<Self>(value, divider.multiplier(), divider.shift())
    }

    fn divv_branchfree(value: Storage<Self>, dividers: crate::divider::vector::VectorDivider<Self>) -> Storage<Self> {
        arch::divv_epu_bf::<Self>(value, dividers.multipliers.0, dividers.shifts.0)
    }

    const HAS_HARDWARE_POPCNT: bool = false;

    fn count_ones(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_popcnt_epi64x_v3(value) }
    }

    fn count_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(Self::not(value))
    }

    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::splat(64), Self::ilog2p1(value))
    }

    fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {
        super::I64x4V3::trailing_zeros(value)
    }

    fn leading_ones(value: Storage<Self>) -> Storage<Self> {
        Self::leading_zeros(Self::not(value))
    }

    fn trailing_ones(value: Storage<Self>) -> Storage<Self> {
        Self::trailing_zeros(Self::not(value))
    }
}

#[thermite_macros::inline_always]
impl UnsignedIntegerRegister for U64x4V3 {
    /// 2D Morton via carry-less multiply (each 64-bit lane interleaves two 32-bit
    /// coords); every other `N` delegates to the generic shift/mask cascade.
    #[cfg(feature = "avx2-pclmul")]
    fn morton<const N: usize>(values: [Storage<Self>; N]) -> Storage<Self> {
        if const { N == 2 } {
            unsafe { arch::_mm256_morton2_epu64x_v3(values[0], values[1]) }
        } else {
            crate::backend::generic::polyfills::morton_cascade::<Self, N>(values)
        }
    }
}

impl CastRegister<ArrayRegister<U64x4V3, 2>> for super::U32x8V3 {
    fn cast_from(value: Storage<ArrayRegister<U64x4V3, 2>>) -> Storage<Self> {
        let (lo, hi) = <ArrayRegister<U64x4V3, 2> as ConcatRegister<U64x4V3>>::split(value);

        unsafe {
            let lo = arch::_mm256_cvtepi64_epi32_v3(lo);
            let hi = arch::_mm256_cvtepi64_epi32_v3(hi);

            arch::_mm256_setr_m128i(lo, hi)
        }
    }
}
