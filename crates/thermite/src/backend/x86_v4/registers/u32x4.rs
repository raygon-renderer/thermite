//! `u32x4` on AVX-512: the 128-bit unsigned-dword register under EVEX
//! (AVX512VL). Width port of [`u32x8`](super::u32x8). See
//! [`i32x4`](super::i32x4) for the xmm-specific spellings.

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
use super::kmask::KMask4;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct U32x4V4;

#[thermite_macros::inline_always]
impl CoreRegister for U32x4V4 {
    type Lanes = typenum::U4;
    type Storage = arch::__m128i;
    type Mask = KMask4;

    const IS_EMULATED: bool = false;

    const EMPTY: Storage<Self> = empty_reg::<Self>();

    const HAS_EQUAL_SIZE_MASK: bool = false;

    fn blendv(mask: Storage<Self::Mask>, on_false: Storage<Self>, on_true: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_blend_epi32(mask, on_false, on_true) }
    }

    fn zz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_mov_epi32(mask, value) }
    }

    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_mov_epi32(!mask, value) }
    }

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        if const { Z::N >= 4 } {
            value
        } else {
            unsafe { arch::_mm_maskz_mov_epi32(const { ((1u32 << Z::N) - 1) as u8 }, value) }
        }
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        unsafe { arch::_mm_movm_epi32(mask) }
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for U32x4V4 {
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_xor_si128(lhs, rhs) }
    }

    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_and_si128(lhs, rhs) }
    }

    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_andnot_si128(rhs, lhs) }
    }

    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_or_si128(lhs, rhs) }
    }

    fn not(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_ternarylogic_epi32::<{ crate::ternlog_imm!(!C) }>(value, value, value) }
    }

    const HAS_NATIVE_TERNLOG: bool = true;

    fn ternlog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_ternarylogic_epi32::<IMM>(a, b, c) }
    }

    fn bilog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        arch::bilog_ternlog::<Self, IMM>(a, b)
    }

    // --- masked variants -----------------------------------------------------

    masked_binary_v4! {
        bitxor => _mm_mask_xor_epi32, _mm_maskz_xor_epi32;
        bitand => _mm_mask_and_epi32, _mm_maskz_and_epi32;
        bitor => _mm_mask_or_epi32, _mm_maskz_or_epi32;
    }

    masked_andnot_v4! {
        bitandnot => _mm_mask_andnot_epi32, _mm_maskz_andnot_epi32;
    }

    fn not_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_ternarylogic_epi32::<{ crate::ternlog_imm!(!C) }>(value, mask, value, value) }
    }

    fn not_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_ternarylogic_epi32::<{ crate::ternlog_imm!(!C) }>(src, mask, value, value) }
    }

    fn not_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_ternarylogic_epi32::<{ crate::ternlog_imm!(!C) }>(mask, value, value, value) }
    }

    fn ternlog_c<const IMM: i32>(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_ternarylogic_epi32::<IMM>(a, mask, b, c) }
    }

    fn ternlog_m<const IMM: i32>(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_mov_epi32(src, mask, Self::ternlog::<IMM>(a, b, c)) }
    }

    fn ternlog_z<const IMM: i32>(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_ternarylogic_epi32::<IMM>(mask, a, b, c) }
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

#[thermite_macros::inline_always]
impl crate::register::WideRegister for U32x4V4 {
    type Wide = super::U32x8V4;
}

#[thermite_macros::inline_always]
impl Register for U32x4V4 {
    type Element = u32;

    type Signed = super::I32x4V4;
    type Unsigned = U32x4V4;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_test_epi32_mask(value, value) }
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::msb_to_mask(value)
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_movepi32_mask(value) }
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_loadu_si128(value.as_ptr() as *const _) }
    }

    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_cvtsi32_si128(value as i32) }
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_set1_epi32(value as i32) }
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_load_si128(ptr as *const _) }
    }

    unsafe fn load_m(src: Storage<Self>, mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_mask_loadu_epi32(src, mask, ptr as *const _) }
    }

    unsafe fn load_z(mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_maskz_loadu_epi32(mask, ptr as *const _) }
    }

    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_loadu_si128(ptr as *const _) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_store_si128(ptr as *mut _, value) }
    }

    unsafe fn store_masked(ptr: *mut Self::Element, mask: Storage<Self::Mask>, value: Storage<Self>) {
        unsafe { arch::_mm_mask_storeu_epi32(ptr as *mut _, mask, value) }
    }

    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_storeu_si128(ptr as *mut _, value) }
    }

    unsafe fn load_stream(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_stream_load_si128(ptr as _) }
    }

    unsafe fn store_stream(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_stream_si128(ptr as _, value) }
    }

    unsafe fn lookup(values: &[Self::Element], indices: Storage<Self::Unsigned>) -> Storage<Self> {
        if values.len() <= <Self::Lanes as Unsigned>::USIZE {
            let mut padded: GenericArray<Self::Element, Self::Lanes> = unsafe { core::mem::zeroed() };
            padded[..values.len()].copy_from_slice(values);

            Self::permutev(Self::new(padded), indices)
        } else {
            unsafe { <Self as IndexableRegister<Self::Unsigned>>::gather(values.as_ptr(), indices) }
        }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_shuffle_epi32::<0x1B>(value) }
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        unsafe {
            let pattern = arch::_mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
            arch::_mm_shuffle_epi8(value, pattern)
        }
    }

    compress_expand_v4!(
        u8,
        _mm_maskz_compress_epi32,
        _mm_mask_expand_epi32,
        _mm_maskz_expand_epi32
    );

    const HAS_PERMUTEV: bool = true;

    fn permutev(value: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_castps_si128(arch::_mm_permutevar_ps(arch::_mm_castsi128_ps(value), idxs)) }
    }

    fn swizzle(a: Storage<Self>, b: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_permutex2var_epi32(a, idxs, b) }
    }

    // --- masked variants -----------------------------------------------------

    fn new_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: GenericArray<Self::Element, Self::Lanes>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mask_loadu_epi32(src, mask, value.as_ptr() as *const _) }
    }

    fn new_z(mask: Storage<Self::Mask>, value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_loadu_epi32(mask, value.as_ptr() as *const _) }
    }

    fn splat_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_mask_set1_epi32(src, mask, value as i32) }
    }

    fn splat_z(mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_maskz_set1_epi32(mask, value as i32) }
    }

    fn broadcast<const I: usize>(value: Storage<Self>) -> Storage<Self> {
        Self::permutev(value, unsafe { arch::_mm_set1_epi32(I as i32) })
    }

    fn broadcast_c<const I: usize>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        Self::permutev_m(value, mask, value, unsafe { arch::_mm_set1_epi32(I as i32) })
    }

    fn broadcast_m<const I: usize>(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
    ) -> Storage<Self> {
        Self::permutev_m(src, mask, value, unsafe { arch::_mm_set1_epi32(I as i32) })
    }

    fn broadcast_z<const I: usize>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        Self::permutev_z(mask, value, unsafe { arch::_mm_set1_epi32(I as i32) })
    }

    fn broadcastv(value: Storage<Self>, idx: usize) -> Storage<Self> {
        Self::permutev(value, unsafe { arch::_mm_set1_epi32(idx as i32) })
    }

    fn broadcastv_c(mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        Self::permutev_m(value, mask, value, unsafe { arch::_mm_set1_epi32(idx as i32) })
    }

    fn broadcastv_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        Self::permutev_m(src, mask, value, unsafe { arch::_mm_set1_epi32(idx as i32) })
    }

    fn broadcastv_z(mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        Self::permutev_z(mask, value, unsafe { arch::_mm_set1_epi32(idx as i32) })
    }

    fn reverse_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_shuffle_epi32::<0x1B>(value, mask, value) }
    }

    fn reverse_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_shuffle_epi32::<0x1B>(src, mask, value) }
    }

    fn reverse_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_shuffle_epi32::<0x1B>(mask, value) }
    }

    masked_via_mov_v4! {
        mov = _mm_mask_mov_epi32, movz = _mm_maskz_mov_epi32;
        swap_bytes(value: Storage<Self>);
    }

    fn permutev_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe {
            arch::_mm_castps_si128(arch::_mm_mask_permutevar_ps(
                arch::_mm_castsi128_ps(src),
                mask,
                arch::_mm_castsi128_ps(value),
                idxs,
            ))
        }
    }

    fn permutev_z(mask: Storage<Self::Mask>, value: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_castps_si128(arch::_mm_maskz_permutevar_ps(mask, arch::_mm_castsi128_ps(value), idxs)) }
    }

    fn swizzle_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        a: Storage<Self>,
        b: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mask_mov_epi32(src, mask, arch::_mm_permutex2var_epi32(a, idxs, b)) }
    }

    fn swizzle_z(
        mask: Storage<Self::Mask>,
        a: Storage<Self>,
        b: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_maskz_permutex2var_epi32(mask, a, idxs, b) }
    }
}

#[thermite_macros::inline_always]
impl InterleaveRegister for U32x4V4 {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe { (arch::_mm_unpacklo_epi32(a, b), arch::_mm_unpackhi_epi32(a, b)) }
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe {
            let a = arch::_mm_castsi128_ps(a);
            let b = arch::_mm_castsi128_ps(b);
            (
                arch::_mm_castps_si128(arch::_mm_shuffle_ps(a, b, 0b10_00_10_00)),
                arch::_mm_castps_si128(arch::_mm_shuffle_ps(a, b, 0b11_01_11_01)),
            )
        }
    }
}

#[thermite_macros::inline_always]
impl IndexableRegister<U32x4V4> for U32x4V4 {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<U32x4V4>) -> Storage<Self> {
        unsafe { arch::_mm_i32gather_epi32::<4>(ptr as _, indices) }
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<U32x4V4>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mmask_i32gather_epi32::<4>(src, mask, indices, ptr as _) }
    }

    unsafe fn scatter(value: Storage<Self>, ptr: *mut Self::Element, indices: Storage<U32x4V4>) {
        unsafe { arch::_mm_i32scatter_epi32::<4>(ptr as _, indices, value) }
    }

    unsafe fn scatter_m(
        value: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *mut Self::Element,
        indices: Storage<U32x4V4>,
    ) {
        unsafe { arch::_mm_mask_i32scatter_epi32::<4>(ptr as _, mask, indices, value) }
    }
}

// Four 64-bit indices are one ymm (`u64x4`).
#[thermite_macros::inline_always]
impl IndexableRegister<super::U64x4V4> for U32x4V4 {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<super::U64x4V4>) -> Storage<Self> {
        unsafe { arch::_mm256_i64gather_epi32::<4>(ptr as _, indices) }
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<super::U64x4V4>,
    ) -> Storage<Self> {
        unsafe { arch::_mm256_mmask_i64gather_epi32::<4>(src, mask, indices, ptr as _) }
    }

    unsafe fn scatter(value: Storage<Self>, ptr: *mut Self::Element, indices: Storage<super::U64x4V4>) {
        unsafe { arch::_mm256_i64scatter_epi32::<4>(ptr as _, indices, value) }
    }

    unsafe fn scatter_m(
        value: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *mut Self::Element,
        indices: Storage<super::U64x4V4>,
    ) {
        unsafe { arch::_mm256_mask_i64scatter_epi32::<4>(ptr as _, mask, indices, value) }
    }
}

#[thermite_macros::inline_always]
impl BitshiftRegister for U32x4V4 {
    const HAS_TRUE_SHIFTV: bool = true;
    const HAS_WIDE_BYTE_SHIFTS: bool = true;

    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_sll_epi32(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_srl_epi32(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    fn shlv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_sllv_epi32(value, shifts) }
    }

    fn shrv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_srlv_epi32(value, shifts) }
    }

    fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sll_epi32(value, arch::_mm_cvtsi32_si128(IMM8)) }
    }

    fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_srl_epi32(value, arch::_mm_cvtsi32_si128(IMM8)) }
    }

    fn bshli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bslli_si128::<IMM8>(value) }
    }

    fn bshri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bsrli_si128::<IMM8>(value) }
    }

    fn rol(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_rolv_epi32(value, arch::_mm_set1_epi32(shift as i32)) }
    }

    fn ror(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_rorv_epi32(value, arch::_mm_set1_epi32(shift as i32)) }
    }

    fn roli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_rolv_epi32(value, arch::_mm_set1_epi32(IMM8)) }
    }

    fn rori<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_rorv_epi32(value, arch::_mm_set1_epi32(IMM8)) }
    }

    fn rolv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_rolv_epi32(value, shifts) }
    }

    fn rorv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_rorv_epi32(value, shifts) }
    }

    // --- masked variants -----------------------------------------------------

    masked_shift_v4! {
        shl => _mm_mask_sll_epi32, _mm_maskz_sll_epi32;
        shr => _mm_mask_srl_epi32, _mm_maskz_srl_epi32;
    }

    masked_shifti_v4! {
        shli => _mm_mask_sll_epi32, _mm_maskz_sll_epi32;
        shri => _mm_mask_srl_epi32, _mm_maskz_srl_epi32;
    }

    masked_binary_v4! {
        shlv(Storage<Self::Unsigned>) => _mm_mask_sllv_epi32, _mm_maskz_sllv_epi32;
        shrv(Storage<Self::Unsigned>) => _mm_mask_srlv_epi32, _mm_maskz_srlv_epi32;
        rolv(Storage<Self::Unsigned>) => _mm_mask_rolv_epi32, _mm_maskz_rolv_epi32;
        rorv(Storage<Self::Unsigned>) => _mm_mask_rorv_epi32, _mm_maskz_rorv_epi32;
    }

    fn rol_c(mask: Storage<Self::Mask>, value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_mask_rolv_epi32(value, mask, value, arch::_mm_set1_epi32(shift as i32)) }
    }

    fn rol_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_mask_rolv_epi32(src, mask, value, arch::_mm_set1_epi32(shift as i32)) }
    }

    fn rol_z(mask: Storage<Self::Mask>, value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_maskz_rolv_epi32(mask, value, arch::_mm_set1_epi32(shift as i32)) }
    }

    fn ror_c(mask: Storage<Self::Mask>, value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_mask_rorv_epi32(value, mask, value, arch::_mm_set1_epi32(shift as i32)) }
    }

    fn ror_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_mask_rorv_epi32(src, mask, value, arch::_mm_set1_epi32(shift as i32)) }
    }

    fn ror_z(mask: Storage<Self::Mask>, value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_maskz_rorv_epi32(mask, value, arch::_mm_set1_epi32(shift as i32)) }
    }

    fn roli_c<const IMM8: i32>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_rolv_epi32(value, mask, value, arch::_mm_set1_epi32(IMM8)) }
    }

    fn roli_m<const IMM8: i32>(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_rolv_epi32(src, mask, value, arch::_mm_set1_epi32(IMM8)) }
    }

    fn roli_z<const IMM8: i32>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_rolv_epi32(mask, value, arch::_mm_set1_epi32(IMM8)) }
    }

    fn rori_c<const IMM8: i32>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_rorv_epi32(value, mask, value, arch::_mm_set1_epi32(IMM8)) }
    }

    fn rori_m<const IMM8: i32>(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_rorv_epi32(src, mask, value, arch::_mm_set1_epi32(IMM8)) }
    }

    fn rori_z<const IMM8: i32>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_rorv_epi32(mask, value, arch::_mm_set1_epi32(IMM8)) }
    }

    masked_via_mov_v4! {
        mov = _mm_mask_mov_epi32, movz = _mm_maskz_mov_epi32;
        bshli<const IMM8: i32>(value: Storage<Self>);
        bshri<const IMM8: i32>(value: Storage<Self>);
        reverse_bits(value: Storage<Self>);
    }
}

#[thermite_macros::inline_always]
impl ShuffleRegister for U32x4V4 {
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe {
            arch::_mm_castps_si128(arch::_mm_shuffle_ps(
                arch::_mm_castsi128_ps(lhs),
                arch::_mm_castsi128_ps(rhs),
                IMM8,
            ))
        }
    }
}

#[thermite_macros::inline_always]
impl PartialOrdRegister for U32x4V4 {
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmpgt_epu32_mask(lhs, rhs) }
    }

    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmpge_epu32_mask(lhs, rhs) }
    }

    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmplt_epu32_mask(lhs, rhs) }
    }

    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmple_epu32_mask(lhs, rhs) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmpeq_epu32_mask(lhs, rhs) }
    }

    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmpneq_epu32_mask(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl NumericRegister for U32x4V4 {
    const ZERO: Storage<Self> = reg::<Self, 4>([0; 4]);
    const ONE: Storage<Self> = reg::<Self, 4>([1; 4]);
    const TWO: Storage<Self> = reg::<Self, 4>([2; 4]);

    const MIN: Storage<Self> = reg::<Self, 4>([u32::MIN; 4]);
    const MAX: Storage<Self> = reg::<Self, 4>([u32::MAX; 4]);

    fn min_element(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi32_v1!(value; _mm_min_epu32 _mm_min_epu32) as u32
    }

    fn max_element(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi32_v1!(value; _mm_max_epu32 _mm_max_epu32) as u32
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi32_v1!(value; _mm_add_epi32 _mm_add_epi32) as u32
    }

    fn prod_elements(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi32_v1!(value; _mm_mullo_epi32 _mm_mullo_epi32) as u32
    }

    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as Unsigned>::U32)
    }

    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as u32))
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_add_epi32(lhs, rhs) }
    }

    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sub_epi32(lhs, rhs) }
    }

    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mullo_epi32(lhs, rhs) }
    }

    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a / b })
    }

    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a % b })
    }

    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_min_epu32(lhs, rhs) }
    }

    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_max_epu32(lhs, rhs) }
    }

    // --- masked variants -----------------------------------------------------

    masked_binary_v4! {
        add => _mm_mask_add_epi32, _mm_maskz_add_epi32;
        sub => _mm_mask_sub_epi32, _mm_maskz_sub_epi32;
        mul => _mm_mask_mullo_epi32, _mm_maskz_mullo_epi32;
        min => _mm_mask_min_epu32, _mm_maskz_min_epu32;
        max => _mm_mask_max_epu32, _mm_maskz_max_epu32;
    }

    masked_via_mov_v4! {
        mov = _mm_mask_mov_epi32, movz = _mm_maskz_mov_epi32;
        div(lhs: Storage<Self>, rhs: Storage<Self>);
        rem(lhs: Storage<Self>, rhs: Storage<Self>);
    }

    fn square_c(mask: Storage<Self::Mask>, lhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_mullo_epi32(lhs, mask, lhs, lhs) }
    }

    fn square_m(src: Storage<Self>, mask: Storage<Self::Mask>, lhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_mullo_epi32(src, mask, lhs, lhs) }
    }

    fn square_z(mask: Storage<Self::Mask>, lhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_mullo_epi32(mask, lhs, lhs) }
    }

    fn scale_c(mask: Storage<Self::Mask>, value: Storage<Self>, scalar: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_mask_mullo_epi32(value, mask, value, Self::splat(scalar)) }
    }

    fn scale_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
        scalar: Self::Element,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mask_mullo_epi32(src, mask, value, Self::splat(scalar)) }
    }

    fn scale_z(mask: Storage<Self::Mask>, value: Storage<Self>, scalar: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_maskz_mullo_epi32(mask, value, Self::splat(scalar)) }
    }
}

#[thermite_macros::inline_always]
impl IntegerRegister for U32x4V4 {
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe {
            let even_hi = arch::_mm_srli_epi64::<32>(arch::_mm_mul_epu32(lhs, rhs));
            let odd_hi = arch::_mm_mul_epu32(arch::_mm_srli_epi64::<32>(lhs), arch::_mm_srli_epi64::<32>(rhs));
            arch::_mm_mask_blend_epi32(0b1010, even_hi, odd_hi)
        }
    }

    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mullo_epi32(lhs, rhs) }
    }

    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::add(lhs, Self::min(rhs, Self::not(lhs)))
    }

    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::max(lhs, rhs), rhs)
    }

    fn wrapping_sum(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi32_v1!(value; _mm_add_epi32 _mm_add_epi32) as u32
    }

    fn wrapping_product(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi32_v1!(value; _mm_mullo_epi32 _mm_mullo_epi32) as u32
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
        Self::count_ones(unsafe { arch::_mm_conflict_epi32(value) })
    }

    fn count_ones(value: Storage<Self>) -> Storage<Self> {
        if const { <F as Avx512Features>::AVX512VPOPCNTDQ } {
            unsafe { arch::_mm_popcnt_epi32(value) }
        } else {
            unsafe { arch::_mm_popcnt_epi32x_v2(value) }
        }
    }

    fn count_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(Self::not(value))
    }

    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_lzcnt_epi32(value) }
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
        mullo => _mm_mask_mullo_epi32, _mm_maskz_mullo_epi32;
    }

    masked_via_mov_v4! {
        mov = _mm_mask_mov_epi32, movz = _mm_maskz_mov_epi32;
        mulhi(lhs: Storage<Self>, rhs: Storage<Self>);
        div_branched(value: Storage<Self>, divider: crate::Divider<Self::Element>);
        div_branchfree(value: Storage<Self>, divider: crate::BranchfreeDivider<Self::Element>);
        divv_branchfree(value: Storage<Self>, dividers: crate::divider::vector::VectorDivider<Self>);
    }

    fn saturating_add_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_add_epi32(lhs, mask, lhs, Self::min(rhs, Self::not(lhs))) }
    }

    fn saturating_add_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        lhs: Storage<Self>,
        rhs: Storage<Self>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mask_add_epi32(src, mask, lhs, Self::min(rhs, Self::not(lhs))) }
    }

    fn saturating_add_z(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_add_epi32(mask, lhs, Self::min(rhs, Self::not(lhs))) }
    }

    fn saturating_sub_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_sub_epi32(lhs, mask, Self::max(lhs, rhs), rhs) }
    }

    fn saturating_sub_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        lhs: Storage<Self>,
        rhs: Storage<Self>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mask_sub_epi32(src, mask, Self::max(lhs, rhs), rhs) }
    }

    fn saturating_sub_z(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_sub_epi32(mask, Self::max(lhs, rhs), rhs) }
    }

    masked_unary_v4! {
        count_ones => _mm_mask_popcnt_epi32x_v4, _mm_maskz_popcnt_epi32x_v4;
        leading_zeros => _mm_mask_lzcnt_epi32, _mm_maskz_lzcnt_epi32;
    }

    fn count_zeros_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_popcnt_epi32x_v4(value, mask, Self::not(value)) }
    }

    fn count_zeros_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_popcnt_epi32x_v4(src, mask, Self::not(value)) }
    }

    fn count_zeros_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_popcnt_epi32x_v4(mask, Self::not(value)) }
    }

    fn trailing_zeros_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let low = Self::sub(Self::bitand(value, Self::sub(Self::ZERO, value)), Self::ONE);
        unsafe { arch::_mm_mask_popcnt_epi32x_v4(value, mask, low) }
    }

    fn trailing_zeros_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let low = Self::sub(Self::bitand(value, Self::sub(Self::ZERO, value)), Self::ONE);
        unsafe { arch::_mm_mask_popcnt_epi32x_v4(src, mask, low) }
    }

    fn trailing_zeros_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let low = Self::sub(Self::bitand(value, Self::sub(Self::ZERO, value)), Self::ONE);
        unsafe { arch::_mm_maskz_popcnt_epi32x_v4(mask, low) }
    }

    fn leading_ones_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_lzcnt_epi32(value, mask, Self::not(value)) }
    }

    fn leading_ones_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_lzcnt_epi32(src, mask, Self::not(value)) }
    }

    fn leading_ones_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_lzcnt_epi32(mask, Self::not(value)) }
    }

    fn trailing_ones_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let inv = Self::not(value);
        let low = Self::sub(Self::bitand(inv, Self::sub(Self::ZERO, inv)), Self::ONE);
        unsafe { arch::_mm_mask_popcnt_epi32x_v4(value, mask, low) }
    }

    fn trailing_ones_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let inv = Self::not(value);
        let low = Self::sub(Self::bitand(inv, Self::sub(Self::ZERO, inv)), Self::ONE);
        unsafe { arch::_mm_mask_popcnt_epi32x_v4(src, mask, low) }
    }

    fn trailing_ones_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let inv = Self::not(value);
        let low = Self::sub(Self::bitand(inv, Self::sub(Self::ZERO, inv)), Self::ONE);
        unsafe { arch::_mm_maskz_popcnt_epi32x_v4(mask, low) }
    }
}

#[thermite_macros::inline_always]
impl UnsignedIntegerRegister for U32x4V4 {
    fn ilog2p1(value: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::splat(32), Self::leading_zeros(value))
    }

    fn abs_diff(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::max(a, b), Self::min(a, b))
    }

    /// 2D Morton via the v2 PSHUFB nibble-LUT. Other `N` use the cascade.
    fn morton<const N: usize>(values: [Storage<Self>; N]) -> Storage<Self> {
        if const { N == 2 } {
            unsafe { arch::_mm_morton2_epu32x_v2(values[0], values[1]) }
        } else {
            crate::backend::generic::polyfills::morton_cascade::<Self, N>(values)
        }
    }

    fn reverse_morton<const N: usize>(code: Storage<Self>) -> [Storage<Self>; N] {
        if const { N == 2 } {
            let odd = Self::shri::<1>(code);

            unsafe {
                crate::backend::generic::polyfills::morton_pack2::<Self, N>(
                    arch::_mm_morton2_compress_epu32x_v2(code),
                    arch::_mm_morton2_compress_epu32x_v2(odd),
                )
            }
        } else {
            crate::backend::generic::polyfills::reverse_morton_cascade::<Self, N>(code)
        }
    }

    // --- masked variants -----------------------------------------------------

    fn ilog2p1_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_sub_epi32(value, mask, Self::splat(32), Self::leading_zeros(value)) }
    }

    fn ilog2p1_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_sub_epi32(src, mask, Self::splat(32), Self::leading_zeros(value)) }
    }

    fn ilog2p1_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_sub_epi32(mask, Self::splat(32), Self::leading_zeros(value)) }
    }

    fn avg_c(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_sub_epi32(a, mask, Self::bitor(a, b), Self::shri::<1>(Self::bitxor(a, b))) }
    }

    fn avg_m(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_sub_epi32(src, mask, Self::bitor(a, b), Self::shri::<1>(Self::bitxor(a, b))) }
    }

    fn avg_z(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_sub_epi32(mask, Self::bitor(a, b), Self::shri::<1>(Self::bitxor(a, b))) }
    }

    fn abs_diff_c(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_sub_epi32(a, mask, Self::max(a, b), Self::min(a, b)) }
    }

    fn abs_diff_m(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_sub_epi32(src, mask, Self::max(a, b), Self::min(a, b)) }
    }

    fn abs_diff_z(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_sub_epi32(mask, Self::max(a, b), Self::min(a, b)) }
    }

    masked_via_mov_v4! {
        mov = _mm_mask_mov_epi32, movz = _mm_maskz_mov_epi32;
        next_power_of_two_m1(value: Storage<Self>);
        parity(value: Storage<Self>);
    }
}

// Widen/narrow legs against the 256-bit u64x4: single converts.
#[thermite_macros::inline_always]
impl CastRegister<U32x4V4> for super::U64x4V4 {
    fn cast_from(value: Storage<U32x4V4>) -> Storage<Self> {
        unsafe { arch::_mm256_cvtepu32_epi64(value) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::U64x4V4> for U32x4V4 {
    fn cast_from(value: Storage<super::U64x4V4>) -> Storage<Self> {
        unsafe { arch::_mm256_cvtepi64_epi32(value) }
    }

    fn saturating_cast_from(value: Storage<super::U64x4V4>) -> Storage<Self> {
        unsafe { arch::_mm256_cvtusepi64_epi32(value) }
    }
}
