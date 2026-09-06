//! `u32x8` on AVX-512: the 256-bit unsigned-dword register under EVEX
//! (AVX512VL). The width port of [`u32x16`](super::u32x16). See that file
//! and [`i32x8`](super::i32x8) for the notes that are not width.

use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, BitwiseRegister, CastRegister, CoreRegister, IndexableRegister, IntegerRegister,
        InterleaveRegister, NumericRegister, PartialOrdRegister, Register, ShuffleRegister, Storage,
        UnsignedIntegerRegister, ZeroUpper, empty_reg, reg,
    },
};

use super::super::{Avx512Features, DefaultAvx512 as F};
use super::arch;
use super::kmask::KMask8;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct U32x8V4;

#[thermite_macros::inline_always]
impl CoreRegister for U32x8V4 {
    type Lanes = typenum::U8;
    type Storage = arch::__m256i;
    type Mask = KMask8;

    const IS_EMULATED: bool = false;

    const EMPTY: Storage<Self> = empty_reg::<Self>();

    const HAS_EQUAL_SIZE_MASK: bool = false;

    fn blendv(mask: Storage<Self::Mask>, on_false: Storage<Self>, on_true: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_blend_epi32(mask, on_false, on_true) }
    }

    fn zz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_mov_epi32(mask, value) }
    }

    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_mov_epi32(!mask, value) }
    }

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        if const { Z::N >= 8 } {
            value
        } else {
            unsafe { arch::_mm256_maskz_mov_epi32(const { ((1u32 << Z::N) - 1) as u8 }, value) }
        }
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        unsafe { arch::_mm256_movm_epi32(mask) }
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for U32x8V4 {
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_xor_si256(lhs, rhs) }
    }

    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_and_si256(lhs, rhs) }
    }

    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_andnot_si256(rhs, lhs) }
    }

    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_or_si256(lhs, rhs) }
    }

    fn not(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_ternarylogic_epi32::<{ crate::ternlog_imm!(!C) }>(value, value, value) }
    }

    const HAS_NATIVE_TERNLOG: bool = true;

    fn ternlog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_ternarylogic_epi32::<IMM>(a, b, c) }
    }

    fn bilog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        arch::bilog_ternlog::<Self, IMM>(a, b)
    }

    // --- masked variants -----------------------------------------------------

    masked_binary_v4! {
        bitxor => _mm256_mask_xor_epi32, _mm256_maskz_xor_epi32;
        bitand => _mm256_mask_and_epi32, _mm256_maskz_and_epi32;
        bitor => _mm256_mask_or_epi32, _mm256_maskz_or_epi32;
    }

    masked_andnot_v4! {
        bitandnot => _mm256_mask_andnot_epi32, _mm256_maskz_andnot_epi32;
    }

    fn not_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_ternarylogic_epi32::<{ crate::ternlog_imm!(!C) }>(value, mask, value, value) }
    }

    fn not_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_ternarylogic_epi32::<{ crate::ternlog_imm!(!C) }>(src, mask, value, value) }
    }

    fn not_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_ternarylogic_epi32::<{ crate::ternlog_imm!(!C) }>(mask, value, value, value) }
    }

    fn ternlog_c<const IMM: i32>(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_ternarylogic_epi32::<IMM>(a, mask, b, c) }
    }

    fn ternlog_m<const IMM: i32>(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_mov_epi32(src, mask, Self::ternlog::<IMM>(a, b, c)) }
    }

    fn ternlog_z<const IMM: i32>(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_ternarylogic_epi32::<IMM>(mask, a, b, c) }
    }

    fn bilog_c<const IMM: i32>(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        arch::bilog_ternlog_c::<Self, IMM>(mask, a, b)
    }

    fn bilog_m<const IMM: i32>(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        arch::bilog_ternlog_m::<Self, IMM>(src, mask, a, b)
    }

    fn bilog_z<const IMM: i32>(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        arch::bilog_ternlog_z::<Self, IMM>(mask, a, b)
    }
}

// Width ladder: ymm = two xmm (AVX forms), and zmm is the 2x wide register.
#[thermite_macros::inline_always]
impl crate::register::ConcatRegister<super::U32x4V4> for U32x8V4 {
    fn concat(lo: Storage<super::U32x4V4>, hi: Storage<super::U32x4V4>) -> Storage<Self> {
        unsafe { arch::_mm256_setr_m128i(lo, hi) }
    }

    fn split(value: Storage<Self>) -> (Storage<super::U32x4V4>, Storage<super::U32x4V4>) {
        let lo = unsafe { arch::_mm256_castsi256_si128(value) };
        let hi = unsafe { arch::_mm256_extracti128_si256::<1>(value) };
        (lo, hi)
    }
}

#[thermite_macros::inline_always]
impl crate::register::ExtendRegister<super::U32x4V4> for U32x8V4 {
    fn extend(value: Storage<super::U32x4V4>) -> Storage<Self> {
        unsafe { arch::_mm256_zextsi128_si256(value) }
    }

    fn narrow(value: Storage<Self>) -> Storage<super::U32x4V4> {
        unsafe { arch::_mm256_castsi256_si128(value) }
    }
}

#[thermite_macros::inline_always]
impl crate::register::WideRegister for U32x8V4 {
    type Wide = super::U32x16V4;
}

#[thermite_macros::inline_always]
impl Register for U32x8V4 {
    type Element = u32;

    type Signed = super::I32x8V4;
    type Unsigned = U32x8V4;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm256_test_epi32_mask(value, value) }
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::msb_to_mask(value)
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm256_movepi32_mask(value) }
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm256_loadu_si256(value.as_ptr() as *const _) }
    }

    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_zextsi128_si256(arch::_mm_cvtsi32_si128(value as i32)) }
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_set1_epi32(value as i32) }
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_load_si256(ptr as *const _) }
    }

    unsafe fn load_m(src: Storage<Self>, mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_mask_loadu_epi32(src, mask, ptr as *const _) }
    }

    unsafe fn load_z(mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_loadu_epi32(mask, ptr as *const _) }
    }

    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_loadu_si256(ptr as *const _) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm256_store_si256(ptr as *mut _, value) }
    }

    unsafe fn store_masked(ptr: *mut Self::Element, mask: Storage<Self::Mask>, value: Storage<Self>) {
        unsafe { arch::_mm256_mask_storeu_epi32(ptr as *mut _, mask, value) }
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
        if values.len() <= <Self::Lanes as Unsigned>::USIZE {
            let mut padded: GenericArray<Self::Element, Self::Lanes> = unsafe { core::mem::zeroed() };
            padded[..values.len()].copy_from_slice(values);

            unsafe { arch::_mm256_permutexvar_epi32(indices, Self::new(padded)) }
        } else {
            unsafe { <Self as IndexableRegister<Self::Unsigned>>::gather(values.as_ptr(), indices) }
        }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        unsafe {
            let idx = arch::_mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
            arch::_mm256_permutexvar_epi32(idx, value)
        }
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        unsafe {
            let pattern = arch::_mm256_broadcast_i32x4(arch::_mm_setr_epi8(
                3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
            ));
            arch::_mm256_shuffle_epi8(value, pattern)
        }
    }

    compress_expand_v4!(
        u8,
        _mm256_maskz_compress_epi32,
        _mm256_mask_expand_epi32,
        _mm256_maskz_expand_epi32
    );

    const HAS_PERMUTEV: bool = true;

    fn permutev(value: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm256_permutexvar_epi32(idxs, value) }
    }

    fn swizzle(a: Storage<Self>, b: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm256_permutex2var_epi32(a, idxs, b) }
    }

    // --- masked variants -----------------------------------------------------

    fn new_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: GenericArray<Self::Element, Self::Lanes>,
    ) -> Storage<Self> {
        unsafe { arch::_mm256_mask_loadu_epi32(src, mask, value.as_ptr() as *const _) }
    }

    fn new_z(mask: Storage<Self::Mask>, value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_loadu_epi32(mask, value.as_ptr() as *const _) }
    }

    fn splat_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_mask_set1_epi32(src, mask, value as i32) }
    }

    fn splat_z(mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_set1_epi32(mask, value as i32) }
    }

    fn broadcast<const I: usize>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_permutexvar_epi32(arch::_mm256_set1_epi32(I as i32), value) }
    }

    fn broadcast_c<const I: usize>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_permutexvar_epi32(value, mask, arch::_mm256_set1_epi32(I as i32), value) }
    }

    fn broadcast_m<const I: usize>(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
    ) -> Storage<Self> {
        unsafe { arch::_mm256_mask_permutexvar_epi32(src, mask, arch::_mm256_set1_epi32(I as i32), value) }
    }

    fn broadcast_z<const I: usize>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_permutexvar_epi32(mask, arch::_mm256_set1_epi32(I as i32), value) }
    }

    fn broadcastv(value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm256_permutexvar_epi32(arch::_mm256_set1_epi32(idx as i32), value) }
    }

    fn broadcastv_c(mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm256_mask_permutexvar_epi32(value, mask, arch::_mm256_set1_epi32(idx as i32), value) }
    }

    fn broadcastv_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm256_mask_permutexvar_epi32(src, mask, arch::_mm256_set1_epi32(idx as i32), value) }
    }

    fn broadcastv_z(mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_permutexvar_epi32(mask, arch::_mm256_set1_epi32(idx as i32), value) }
    }

    fn reverse_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe {
            let idx = arch::_mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
            arch::_mm256_mask_permutexvar_epi32(value, mask, idx, value)
        }
    }

    fn reverse_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe {
            let idx = arch::_mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
            arch::_mm256_mask_permutexvar_epi32(src, mask, idx, value)
        }
    }

    fn reverse_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe {
            let idx = arch::_mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
            arch::_mm256_maskz_permutexvar_epi32(mask, idx, value)
        }
    }

    masked_via_mov_v4! {
        mov = _mm256_mask_mov_epi32, movz = _mm256_maskz_mov_epi32;
        swap_bytes(value: Storage<Self>);
    }

    fn permutev_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm256_mask_permutexvar_epi32(src, mask, idxs, value) }
    }

    fn permutev_z(mask: Storage<Self::Mask>, value: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_permutexvar_epi32(mask, idxs, value) }
    }

    fn swizzle_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        a: Storage<Self>,
        b: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm256_mask_mov_epi32(src, mask, arch::_mm256_permutex2var_epi32(a, idxs, b)) }
    }

    fn swizzle_z(
        mask: Storage<Self::Mask>,
        a: Storage<Self>,
        b: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_permutex2var_epi32(mask, a, idxs, b) }
    }
}

#[thermite_macros::inline_always]
impl InterleaveRegister for U32x8V4 {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe {
            let lo = arch::_mm256_setr_epi32(0, 8, 1, 9, 2, 10, 3, 11);
            let hi = arch::_mm256_setr_epi32(4, 12, 5, 13, 6, 14, 7, 15);
            (
                arch::_mm256_permutex2var_epi32(a, lo, b),
                arch::_mm256_permutex2var_epi32(a, hi, b),
            )
        }
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe {
            let even = arch::_mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
            let odd = arch::_mm256_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15);
            (
                arch::_mm256_permutex2var_epi32(a, even, b),
                arch::_mm256_permutex2var_epi32(a, odd, b),
            )
        }
    }
}

#[thermite_macros::inline_always]
impl IndexableRegister<U32x8V4> for U32x8V4 {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<U32x8V4>) -> Storage<Self> {
        unsafe { arch::_mm256_i32gather_epi32::<4>(ptr as _, indices) }
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<U32x8V4>,
    ) -> Storage<Self> {
        unsafe { arch::_mm256_mmask_i32gather_epi32::<4>(src, mask, indices, ptr as _) }
    }

    unsafe fn scatter(value: Storage<Self>, ptr: *mut Self::Element, indices: Storage<U32x8V4>) {
        unsafe { arch::_mm256_i32scatter_epi32::<4>(ptr as _, indices, value) }
    }

    unsafe fn scatter_m(
        value: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *mut Self::Element,
        indices: Storage<U32x8V4>,
    ) {
        unsafe { arch::_mm256_mask_i32scatter_epi32::<4>(ptr as _, mask, indices, value) }
    }
}

// Eight 64-bit indices are one zmm (`u64x8`).
#[thermite_macros::inline_always]
impl IndexableRegister<super::U64x8V4> for U32x8V4 {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<super::U64x8V4>) -> Storage<Self> {
        unsafe { arch::_mm512_i64gather_epi32::<4>(indices, ptr as _) }
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<super::U64x8V4>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_i64gather_epi32::<4>(src, mask, indices, ptr as _) }
    }

    unsafe fn scatter(value: Storage<Self>, ptr: *mut Self::Element, indices: Storage<super::U64x8V4>) {
        unsafe { arch::_mm512_i64scatter_epi32::<4>(ptr as _, indices, value) }
    }

    unsafe fn scatter_m(
        value: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *mut Self::Element,
        indices: Storage<super::U64x8V4>,
    ) {
        unsafe { arch::_mm512_mask_i64scatter_epi32::<4>(ptr as _, mask, indices, value) }
    }
}

#[thermite_macros::inline_always]
impl BitshiftRegister for U32x8V4 {
    const HAS_TRUE_SHIFTV: bool = true;
    const HAS_WIDE_BYTE_SHIFTS: bool = false;

    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm256_sll_epi32(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm256_srl_epi32(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    fn shlv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm256_sllv_epi32(value, shifts) }
    }

    fn shrv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm256_srlv_epi32(value, shifts) }
    }

    fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_sll_epi32(value, arch::_mm_cvtsi32_si128(IMM8)) }
    }

    fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_srl_epi32(value, arch::_mm_cvtsi32_si128(IMM8)) }
    }

    fn rol(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm256_rolv_epi32(value, arch::_mm256_set1_epi32(shift as i32)) }
    }

    fn ror(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm256_rorv_epi32(value, arch::_mm256_set1_epi32(shift as i32)) }
    }

    fn roli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_rolv_epi32(value, arch::_mm256_set1_epi32(IMM8)) }
    }

    fn rori<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_rorv_epi32(value, arch::_mm256_set1_epi32(IMM8)) }
    }

    fn rolv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm256_rolv_epi32(value, shifts) }
    }

    fn rorv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm256_rorv_epi32(value, shifts) }
    }

    // --- masked variants -----------------------------------------------------

    masked_shift_v4! {
        shl => _mm256_mask_sll_epi32, _mm256_maskz_sll_epi32;
        shr => _mm256_mask_srl_epi32, _mm256_maskz_srl_epi32;
    }

    masked_shifti_v4! {
        shli => _mm256_mask_sll_epi32, _mm256_maskz_sll_epi32;
        shri => _mm256_mask_srl_epi32, _mm256_maskz_srl_epi32;
    }

    masked_binary_v4! {
        shlv(Storage<Self::Unsigned>) => _mm256_mask_sllv_epi32, _mm256_maskz_sllv_epi32;
        shrv(Storage<Self::Unsigned>) => _mm256_mask_srlv_epi32, _mm256_maskz_srlv_epi32;
        rolv(Storage<Self::Unsigned>) => _mm256_mask_rolv_epi32, _mm256_maskz_rolv_epi32;
        rorv(Storage<Self::Unsigned>) => _mm256_mask_rorv_epi32, _mm256_maskz_rorv_epi32;
    }

    fn rol_c(mask: Storage<Self::Mask>, value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm256_mask_rolv_epi32(value, mask, value, arch::_mm256_set1_epi32(shift as i32)) }
    }

    fn rol_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm256_mask_rolv_epi32(src, mask, value, arch::_mm256_set1_epi32(shift as i32)) }
    }

    fn rol_z(mask: Storage<Self::Mask>, value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_rolv_epi32(mask, value, arch::_mm256_set1_epi32(shift as i32)) }
    }

    fn ror_c(mask: Storage<Self::Mask>, value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm256_mask_rorv_epi32(value, mask, value, arch::_mm256_set1_epi32(shift as i32)) }
    }

    fn ror_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm256_mask_rorv_epi32(src, mask, value, arch::_mm256_set1_epi32(shift as i32)) }
    }

    fn ror_z(mask: Storage<Self::Mask>, value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_rorv_epi32(mask, value, arch::_mm256_set1_epi32(shift as i32)) }
    }

    fn roli_c<const IMM8: i32>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_rolv_epi32(value, mask, value, arch::_mm256_set1_epi32(IMM8)) }
    }

    fn roli_m<const IMM8: i32>(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_rolv_epi32(src, mask, value, arch::_mm256_set1_epi32(IMM8)) }
    }

    fn roli_z<const IMM8: i32>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_rolv_epi32(mask, value, arch::_mm256_set1_epi32(IMM8)) }
    }

    fn rori_c<const IMM8: i32>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_rorv_epi32(value, mask, value, arch::_mm256_set1_epi32(IMM8)) }
    }

    fn rori_m<const IMM8: i32>(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_rorv_epi32(src, mask, value, arch::_mm256_set1_epi32(IMM8)) }
    }

    fn rori_z<const IMM8: i32>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_rorv_epi32(mask, value, arch::_mm256_set1_epi32(IMM8)) }
    }

    masked_via_mov_v4! {
        mov = _mm256_mask_mov_epi32, movz = _mm256_maskz_mov_epi32;
        bshli<const IMM8: i32>(value: Storage<Self>);
        bshri<const IMM8: i32>(value: Storage<Self>);
        reverse_bits(value: Storage<Self>);
    }
}

#[thermite_macros::inline_always]
impl ShuffleRegister for U32x8V4 {
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe {
            arch::_mm256_castps_si256(arch::_mm256_shuffle_ps(
                arch::_mm256_castsi256_ps(lhs),
                arch::_mm256_castsi256_ps(rhs),
                IMM8,
            ))
        }
    }
}

#[thermite_macros::inline_always]
impl PartialOrdRegister for U32x8V4 {
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm256_cmpgt_epu32_mask(lhs, rhs) }
    }

    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm256_cmpge_epu32_mask(lhs, rhs) }
    }

    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm256_cmplt_epu32_mask(lhs, rhs) }
    }

    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm256_cmple_epu32_mask(lhs, rhs) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm256_cmpeq_epu32_mask(lhs, rhs) }
    }

    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm256_cmpneq_epu32_mask(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl NumericRegister for U32x8V4 {
    const ZERO: Storage<Self> = reg::<Self, 8>([0; 8]);
    const ONE: Storage<Self> = reg::<Self, 8>([1; 8]);
    const TWO: Storage<Self> = reg::<Self, 8>([2; 8]);

    const MIN: Storage<Self> = reg::<Self, 8>([u32::MIN; 8]);
    const MAX: Storage<Self> = reg::<Self, 8>([u32::MAX; 8]);

    fn min_element(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi32_v3!(value; _mm_min_epu32 _mm_min_epu32) as u32
    }

    fn max_element(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi32_v3!(value; _mm_max_epu32 _mm_max_epu32) as u32
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi32_v3!(value; _mm_add_epi32 _mm_add_epi32) as u32
    }

    fn prod_elements(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi32_v3!(value; _mm_mullo_epi32 _mm_mullo_epi32) as u32
    }

    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as Unsigned>::U32)
    }

    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as u32))
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_add_epi32(lhs, rhs) }
    }

    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_sub_epi32(lhs, rhs) }
    }

    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mullo_epi32(lhs, rhs) }
    }

    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a / b })
    }

    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a % b })
    }

    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_min_epu32(lhs, rhs) }
    }

    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_max_epu32(lhs, rhs) }
    }

    // --- masked variants -----------------------------------------------------

    masked_binary_v4! {
        add => _mm256_mask_add_epi32, _mm256_maskz_add_epi32;
        sub => _mm256_mask_sub_epi32, _mm256_maskz_sub_epi32;
        mul => _mm256_mask_mullo_epi32, _mm256_maskz_mullo_epi32;
        min => _mm256_mask_min_epu32, _mm256_maskz_min_epu32;
        max => _mm256_mask_max_epu32, _mm256_maskz_max_epu32;
    }

    masked_via_mov_v4! {
        mov = _mm256_mask_mov_epi32, movz = _mm256_maskz_mov_epi32;
        div(lhs: Storage<Self>, rhs: Storage<Self>);
        rem(lhs: Storage<Self>, rhs: Storage<Self>);
    }

    fn square_c(mask: Storage<Self::Mask>, lhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_mullo_epi32(lhs, mask, lhs, lhs) }
    }

    fn square_m(src: Storage<Self>, mask: Storage<Self::Mask>, lhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_mullo_epi32(src, mask, lhs, lhs) }
    }

    fn square_z(mask: Storage<Self::Mask>, lhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_mullo_epi32(mask, lhs, lhs) }
    }

    fn scale_c(mask: Storage<Self::Mask>, value: Storage<Self>, scalar: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_mask_mullo_epi32(value, mask, value, Self::splat(scalar)) }
    }

    fn scale_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
        scalar: Self::Element,
    ) -> Storage<Self> {
        unsafe { arch::_mm256_mask_mullo_epi32(src, mask, value, Self::splat(scalar)) }
    }

    fn scale_z(mask: Storage<Self::Mask>, value: Storage<Self>, scalar: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_mullo_epi32(mask, value, Self::splat(scalar)) }
    }
}

#[thermite_macros::inline_always]
impl IntegerRegister for U32x8V4 {
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe {
            let even_hi = arch::_mm256_srli_epi64::<32>(arch::_mm256_mul_epu32(lhs, rhs));
            let odd_hi = arch::_mm256_mul_epu32(arch::_mm256_srli_epi64::<32>(lhs), arch::_mm256_srli_epi64::<32>(rhs));
            arch::_mm256_mask_blend_epi32(0xAA, even_hi, odd_hi)
        }
    }

    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mullo_epi32(lhs, rhs) }
    }

    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::add(lhs, Self::min(rhs, Self::not(lhs)))
    }

    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::max(lhs, rhs), rhs)
    }

    fn wrapping_sum(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi32_v3!(value; _mm_add_epi32 _mm_add_epi32) as u32
    }

    fn wrapping_product(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi32_v3!(value; _mm_mullo_epi32 _mm_mullo_epi32) as u32
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

    const HAS_HARDWARE_POPCNT: bool = <F as Avx512Features>::AVX512VPOPCNTDQ;

    fn count_conflicts(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(unsafe { arch::_mm256_conflict_epi32(value) })
    }

    fn count_ones(value: Storage<Self>) -> Storage<Self> {
        if const { <F as Avx512Features>::AVX512VPOPCNTDQ } {
            unsafe { arch::_mm256_popcnt_epi32(value) }
        } else {
            unsafe { arch::_mm256_popcnt_epi32x_v3(value) }
        }
    }

    fn count_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(Self::not(value))
    }

    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_lzcnt_epi32(value) }
    }

    fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(Self::sub(Self::bitand(value, Self::sub(Self::ZERO, value)), Self::ONE))
    }

    fn leading_ones(value: Storage<Self>) -> Storage<Self> {
        Self::leading_zeros(Self::not(value))
    }

    fn trailing_ones(value: Storage<Self>) -> Storage<Self> {
        Self::trailing_zeros(Self::not(value))
    }

    // --- masked variants -----------------------------------------------------

    masked_binary_v4! {
        mullo => _mm256_mask_mullo_epi32, _mm256_maskz_mullo_epi32;
    }

    masked_via_mov_v4! {
        mov = _mm256_mask_mov_epi32, movz = _mm256_maskz_mov_epi32;
        mulhi(lhs: Storage<Self>, rhs: Storage<Self>);
        div_branched(value: Storage<Self>, divider: crate::Divider<Self::Element>);
        div_branchfree(value: Storage<Self>, divider: crate::BranchfreeDivider<Self::Element>);
        divv_branchfree(value: Storage<Self>, dividers: crate::divider::vector::VectorDivider<Self>);
    }

    fn saturating_add_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_add_epi32(lhs, mask, lhs, Self::min(rhs, Self::not(lhs))) }
    }

    fn saturating_add_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        lhs: Storage<Self>,
        rhs: Storage<Self>,
    ) -> Storage<Self> {
        unsafe { arch::_mm256_mask_add_epi32(src, mask, lhs, Self::min(rhs, Self::not(lhs))) }
    }

    fn saturating_add_z(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_add_epi32(mask, lhs, Self::min(rhs, Self::not(lhs))) }
    }

    fn saturating_sub_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_sub_epi32(lhs, mask, Self::max(lhs, rhs), rhs) }
    }

    fn saturating_sub_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        lhs: Storage<Self>,
        rhs: Storage<Self>,
    ) -> Storage<Self> {
        unsafe { arch::_mm256_mask_sub_epi32(src, mask, Self::max(lhs, rhs), rhs) }
    }

    fn saturating_sub_z(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_sub_epi32(mask, Self::max(lhs, rhs), rhs) }
    }

    masked_unary_v4! {
        count_ones => _mm256_mask_popcnt_epi32x_v4, _mm256_maskz_popcnt_epi32x_v4;
        leading_zeros => _mm256_mask_lzcnt_epi32, _mm256_maskz_lzcnt_epi32;
    }

    fn count_zeros_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_popcnt_epi32x_v4(value, mask, Self::not(value)) }
    }

    fn count_zeros_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_popcnt_epi32x_v4(src, mask, Self::not(value)) }
    }

    fn count_zeros_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_popcnt_epi32x_v4(mask, Self::not(value)) }
    }

    fn trailing_zeros_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let low = Self::sub(Self::bitand(value, Self::sub(Self::ZERO, value)), Self::ONE);
        unsafe { arch::_mm256_mask_popcnt_epi32x_v4(value, mask, low) }
    }

    fn trailing_zeros_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let low = Self::sub(Self::bitand(value, Self::sub(Self::ZERO, value)), Self::ONE);
        unsafe { arch::_mm256_mask_popcnt_epi32x_v4(src, mask, low) }
    }

    fn trailing_zeros_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let low = Self::sub(Self::bitand(value, Self::sub(Self::ZERO, value)), Self::ONE);
        unsafe { arch::_mm256_maskz_popcnt_epi32x_v4(mask, low) }
    }

    fn leading_ones_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_lzcnt_epi32(value, mask, Self::not(value)) }
    }

    fn leading_ones_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_lzcnt_epi32(src, mask, Self::not(value)) }
    }

    fn leading_ones_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_lzcnt_epi32(mask, Self::not(value)) }
    }

    fn trailing_ones_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let inv = Self::not(value);
        let low = Self::sub(Self::bitand(inv, Self::sub(Self::ZERO, inv)), Self::ONE);
        unsafe { arch::_mm256_mask_popcnt_epi32x_v4(value, mask, low) }
    }

    fn trailing_ones_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let inv = Self::not(value);
        let low = Self::sub(Self::bitand(inv, Self::sub(Self::ZERO, inv)), Self::ONE);
        unsafe { arch::_mm256_mask_popcnt_epi32x_v4(src, mask, low) }
    }

    fn trailing_ones_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let inv = Self::not(value);
        let low = Self::sub(Self::bitand(inv, Self::sub(Self::ZERO, inv)), Self::ONE);
        unsafe { arch::_mm256_maskz_popcnt_epi32x_v4(mask, low) }
    }
}

#[thermite_macros::inline_always]
impl UnsignedIntegerRegister for U32x8V4 {
    fn ilog2p1(value: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::splat(32), Self::leading_zeros(value))
    }

    fn abs_diff(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::max(a, b), Self::min(a, b))
    }

    /// 2D Morton via the same-width v3 PSHUFB nibble-LUT. Other `N` use the
    /// generic cascade.
    fn morton<const N: usize>(values: [Storage<Self>; N]) -> Storage<Self> {
        if const { N == 2 } {
            unsafe { arch::_mm256_morton2_epu32x_v3(values[0], values[1]) }
        } else {
            crate::backend::generic::polyfills::morton_cascade::<Self, N>(values)
        }
    }

    fn reverse_morton<const N: usize>(code: Storage<Self>) -> [Storage<Self>; N] {
        if const { N == 2 } {
            let odd = Self::shri::<1>(code);

            unsafe {
                crate::backend::generic::polyfills::morton_pack2::<Self, N>(
                    arch::_mm256_morton2_compress_epu32x_v3(code),
                    arch::_mm256_morton2_compress_epu32x_v3(odd),
                )
            }
        } else {
            crate::backend::generic::polyfills::reverse_morton_cascade::<Self, N>(code)
        }
    }

    // --- masked variants -----------------------------------------------------

    fn ilog2p1_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_sub_epi32(value, mask, Self::splat(32), Self::leading_zeros(value)) }
    }

    fn ilog2p1_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_sub_epi32(src, mask, Self::splat(32), Self::leading_zeros(value)) }
    }

    fn ilog2p1_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_sub_epi32(mask, Self::splat(32), Self::leading_zeros(value)) }
    }

    fn avg_c(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_sub_epi32(a, mask, Self::bitor(a, b), Self::shri::<1>(Self::bitxor(a, b))) }
    }

    fn avg_m(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_sub_epi32(src, mask, Self::bitor(a, b), Self::shri::<1>(Self::bitxor(a, b))) }
    }

    fn avg_z(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_sub_epi32(mask, Self::bitor(a, b), Self::shri::<1>(Self::bitxor(a, b))) }
    }

    fn abs_diff_c(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_sub_epi32(a, mask, Self::max(a, b), Self::min(a, b)) }
    }

    fn abs_diff_m(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mask_sub_epi32(src, mask, Self::max(a, b), Self::min(a, b)) }
    }

    fn abs_diff_z(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_maskz_sub_epi32(mask, Self::max(a, b), Self::min(a, b)) }
    }

    masked_via_mov_v4! {
        mov = _mm256_mask_mov_epi32, movz = _mm256_maskz_mov_epi32;
        next_power_of_two_m1(value: Storage<Self>);
        parity(value: Storage<Self>);
    }
}

// Widen/narrow legs against the native u64x8: single zmm converts.
#[thermite_macros::inline_always]
impl CastRegister<U32x8V4> for super::U64x8V4 {
    fn cast_from(value: Storage<U32x8V4>) -> Storage<Self> {
        unsafe { arch::_mm512_cvtepu32_epi64(value) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::U64x8V4> for U32x8V4 {
    fn cast_from(value: Storage<super::U64x8V4>) -> Storage<Self> {
        unsafe { arch::_mm512_cvtepi64_epi32(value) }
    }

    // vpmovusqd: a real unsigned saturating narrow.
    fn saturating_cast_from(value: Storage<super::U64x8V4>) -> Storage<Self> {
        unsafe { arch::_mm512_cvtusepi64_epi32(value) }
    }
}
