//! `i32x4` on AVX-512: the 128-bit signed-dword register under EVEX
//! (AVX512VL). Width port of [`i32x8`](super::i32x8). See
//! [`f32x4`](super::f32x4) for the xmm-specific spellings (no `vpermd` at
//! 128 bits: `vpermilps` through a float cast, `vpshufd imm` for the
//! reverse). Reductions use the v1 `_mm_reduce_epi32_v1!` fold. `count_ones`
//! falls back to the v2 LUT on tier 1.

use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, BitwiseRegister, CastRegister, CoreRegister, IndexableRegister, IntegerRegister,
        InterleaveRegister, NumericRegister, PartialOrdRegister, Register, ShuffleRegister, SignedIntegerRegister,
        SignedRegister, Storage, ZeroUpper, empty_reg, reg, reg_splat,
    },
};

use super::super::{Avx512Features, DefaultAvx512 as F};
use super::arch;
use super::kmask::KMask4;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct I32x4V4;

#[thermite_macros::inline_always]
impl CoreRegister for I32x4V4 {
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
impl BitwiseRegister for I32x4V4 {
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
impl crate::register::WideRegister for I32x4V4 {
    type Wide = super::I32x8V4;
}

#[thermite_macros::inline_always]
impl Register for I32x4V4 {
    type Element = i32;

    type Signed = I32x4V4;
    type Unsigned = super::U32x4V4;

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
        unsafe { arch::_mm_cvtsi32_si128(value) }
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_set1_epi32(value) }
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_load_si128(ptr as *const _) }
    }

    unsafe fn load_m(src: Storage<Self>, mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_mask_loadu_epi32(src, mask, ptr) }
    }

    unsafe fn load_z(mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_maskz_loadu_epi32(mask, ptr) }
    }

    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_loadu_si128(ptr as *const _) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_store_si128(ptr as *mut _, value) }
    }

    unsafe fn store_masked(ptr: *mut Self::Element, mask: Storage<Self::Mask>, value: Storage<Self>) {
        unsafe { arch::_mm_mask_storeu_epi32(ptr, mask, value) }
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

    // vpermilps through a float cast: the whole xmm is one lane.
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
        unsafe { arch::_mm_mask_loadu_epi32(src, mask, value.as_ptr()) }
    }

    fn new_z(mask: Storage<Self::Mask>, value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_loadu_epi32(mask, value.as_ptr()) }
    }

    fn splat_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_mask_set1_epi32(src, mask, value) }
    }

    fn splat_z(mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_maskz_set1_epi32(mask, value) }
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
impl InterleaveRegister for I32x4V4 {
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
impl IndexableRegister<super::U32x4V4> for I32x4V4 {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<super::U32x4V4>) -> Storage<Self> {
        unsafe { arch::_mm_i32gather_epi32::<4>(ptr, indices) }
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<super::U32x4V4>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mmask_i32gather_epi32::<4>(src, mask, indices, ptr) }
    }

    unsafe fn scatter(value: Storage<Self>, ptr: *mut Self::Element, indices: Storage<super::U32x4V4>) {
        unsafe { arch::_mm_i32scatter_epi32::<4>(ptr, indices, value) }
    }

    unsafe fn scatter_m(
        value: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *mut Self::Element,
        indices: Storage<super::U32x4V4>,
    ) {
        unsafe { arch::_mm_mask_i32scatter_epi32::<4>(ptr, mask, indices, value) }
    }
}

// Four 64-bit indices are one ymm (`u64x4`).
#[thermite_macros::inline_always]
impl IndexableRegister<super::U64x4V4> for I32x4V4 {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<super::U64x4V4>) -> Storage<Self> {
        unsafe { arch::_mm256_i64gather_epi32::<4>(ptr, indices) }
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<super::U64x4V4>,
    ) -> Storage<Self> {
        unsafe { arch::_mm256_mmask_i64gather_epi32::<4>(src, mask, indices, ptr) }
    }

    unsafe fn scatter(value: Storage<Self>, ptr: *mut Self::Element, indices: Storage<super::U64x4V4>) {
        unsafe { arch::_mm256_i64scatter_epi32::<4>(ptr, indices, value) }
    }

    unsafe fn scatter_m(
        value: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *mut Self::Element,
        indices: Storage<super::U64x4V4>,
    ) {
        unsafe { arch::_mm256_mask_i64scatter_epi32::<4>(ptr, mask, indices, value) }
    }
}

#[thermite_macros::inline_always]
impl BitshiftRegister for I32x4V4 {
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
impl ShuffleRegister for I32x4V4 {
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
impl PartialOrdRegister for I32x4V4 {
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmpgt_epi32_mask(lhs, rhs) }
    }

    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmpge_epi32_mask(lhs, rhs) }
    }

    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmplt_epi32_mask(lhs, rhs) }
    }

    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmple_epi32_mask(lhs, rhs) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmpeq_epi32_mask(lhs, rhs) }
    }

    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmpneq_epi32_mask(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl NumericRegister for I32x4V4 {
    const ZERO: Storage<Self> = reg::<Self, 4>([0; 4]);
    const ONE: Storage<Self> = reg::<Self, 4>([1; 4]);
    const TWO: Storage<Self> = reg::<Self, 4>([2; 4]);

    const MIN: Storage<Self> = reg::<Self, 4>([i32::MIN; 4]);
    const MAX: Storage<Self> = reg::<Self, 4>([i32::MAX; 4]);

    fn min_element(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi32_v1!(value; _mm_min_epi32 _mm_min_epi32)
    }

    fn max_element(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi32_v1!(value; _mm_max_epi32 _mm_max_epi32)
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi32_v1!(value; _mm_add_epi32 _mm_add_epi32)
    }

    fn prod_elements(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi32_v1!(value; _mm_mullo_epi32 _mm_mullo_epi32)
    }

    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as Unsigned>::I32)
    }

    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as i32))
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
        unsafe { arch::_mm_min_epi32(lhs, rhs) }
    }

    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_max_epi32(lhs, rhs) }
    }

    // --- masked variants -----------------------------------------------------

    masked_binary_v4! {
        add => _mm_mask_add_epi32, _mm_maskz_add_epi32;
        sub => _mm_mask_sub_epi32, _mm_maskz_sub_epi32;
        mul => _mm_mask_mullo_epi32, _mm_maskz_mullo_epi32;
        min => _mm_mask_min_epi32, _mm_maskz_min_epi32;
        max => _mm_mask_max_epi32, _mm_maskz_max_epi32;
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
impl SignedRegister for I32x4V4 {
    const NEG_ONE: Storage<Self> = reg::<Self, 4>([-1; 4]);
    const MIN_POSITIVE: Storage<Self> = reg_splat::<Self>(1);

    fn neg(value: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::ZERO, value)
    }

    fn is_negative(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::msb_to_mask(value)
    }

    fn abs(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_abs_epi32(value) }
    }

    fn copysign(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::neg_c(Self::msb_to_mask(Self::bitxor(lhs, rhs)), lhs)
    }

    fn signum(value: Storage<Self>) -> Storage<Self> {
        Self::min(Self::max(value, Self::NEG_ONE), Self::ONE)
    }

    // --- masked variants -----------------------------------------------------

    fn neg_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_sub_epi32(value, mask, Self::ZERO, value) }
    }

    fn neg_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_sub_epi32(src, mask, Self::ZERO, value) }
    }

    fn neg_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_sub_epi32(mask, Self::ZERO, value) }
    }

    masked_unary_v4! {
        abs => _mm_mask_abs_epi32, _mm_maskz_abs_epi32;
    }

    fn copysign_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::neg_c(mask & Self::msb_to_mask(Self::bitxor(lhs, rhs)), lhs)
    }

    fn copysign_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        lhs: Storage<Self>,
        rhs: Storage<Self>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mask_mov_epi32(src, mask, Self::copysign(lhs, rhs)) }
    }

    fn copysign_z(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_mov_epi32(mask, Self::copysign(lhs, rhs)) }
    }
}

#[thermite_macros::inline_always]
impl IntegerRegister for I32x4V4 {
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe {
            let even_hi = arch::_mm_srli_epi64::<32>(arch::_mm_mul_epi32(lhs, rhs));
            let odd_hi = arch::_mm_mul_epi32(arch::_mm_srli_epi64::<32>(lhs), arch::_mm_srli_epi64::<32>(rhs));
            arch::_mm_mask_blend_epi32(0b1010, even_hi, odd_hi)
        }
    }

    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mullo_epi32(lhs, rhs) }
    }

    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe {
            let sum = Self::add(lhs, rhs);
            let ov = arch::_mm_movepi32_mask(
                arch::_mm_ternarylogic_epi32::<{ crate::ternlog_imm!((A ^ C) & (B ^ C)) }>(lhs, rhs, sum),
            );
            let sat = Self::bitxor(Self::MAX, Self::srai::<31>(lhs));
            arch::_mm_mask_mov_epi32(sum, ov, sat)
        }
    }

    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe {
            let diff = Self::sub(lhs, rhs);
            let ov = arch::_mm_movepi32_mask(
                arch::_mm_ternarylogic_epi32::<{ crate::ternlog_imm!((A ^ B) & (A ^ C)) }>(lhs, rhs, diff),
            );
            let sat = Self::bitxor(Self::MAX, Self::srai::<31>(lhs));
            arch::_mm_mask_mov_epi32(diff, ov, sat)
        }
    }

    fn wrapping_sum(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi32_v1!(value; _mm_add_epi32 _mm_add_epi32)
    }

    fn wrapping_product(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi32_v1!(value; _mm_mullo_epi32 _mm_mullo_epi32)
    }

    fn div_branched(value: Storage<Self>, divider: crate::Divider<Self::Element>) -> Storage<Self> {
        arch::div_epi::<Self>(value, divider.multiplier(), divider.shift())
    }

    fn div_branchfree(value: Storage<Self>, divider: crate::BranchfreeDivider<Self::Element>) -> Storage<Self> {
        arch::div_epi_bf::<Self>(value, divider.multiplier(), divider.shift())
    }

    fn divv_branchfree(value: Storage<Self>, dividers: crate::divider::vector::VectorDivider<Self>) -> Storage<Self> {
        arch::divv_epi_bf::<Self>(value, dividers.multipliers.0, dividers.shifts.0)
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
        Self::count_ones(Self::sub(Self::bitand(value, Self::neg(value)), Self::ONE))
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
        saturating_add(lhs: Storage<Self>, rhs: Storage<Self>);
        saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>);
        div_branched(value: Storage<Self>, divider: crate::Divider<Self::Element>);
        div_branchfree(value: Storage<Self>, divider: crate::BranchfreeDivider<Self::Element>);
        divv_branchfree(value: Storage<Self>, dividers: crate::divider::vector::VectorDivider<Self>);
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
        let low = Self::sub(Self::bitand(value, Self::neg(value)), Self::ONE);
        unsafe { arch::_mm_mask_popcnt_epi32x_v4(value, mask, low) }
    }

    fn trailing_zeros_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let low = Self::sub(Self::bitand(value, Self::neg(value)), Self::ONE);
        unsafe { arch::_mm_mask_popcnt_epi32x_v4(src, mask, low) }
    }

    fn trailing_zeros_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let low = Self::sub(Self::bitand(value, Self::neg(value)), Self::ONE);
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
        let low = Self::sub(Self::bitand(inv, Self::neg(inv)), Self::ONE);
        unsafe { arch::_mm_mask_popcnt_epi32x_v4(value, mask, low) }
    }

    fn trailing_ones_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let inv = Self::not(value);
        let low = Self::sub(Self::bitand(inv, Self::neg(inv)), Self::ONE);
        unsafe { arch::_mm_mask_popcnt_epi32x_v4(src, mask, low) }
    }

    fn trailing_ones_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let inv = Self::not(value);
        let low = Self::sub(Self::bitand(inv, Self::neg(inv)), Self::ONE);
        unsafe { arch::_mm_maskz_popcnt_epi32x_v4(mask, low) }
    }
}

#[thermite_macros::inline_always]
impl SignedIntegerRegister for I32x4V4 {
    fn srai<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sra_epi32(value, arch::_mm_cvtsi32_si128(IMM8)) }
    }

    fn sra(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_sra_epi32(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    fn srav(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_srav_epi32(value, shifts) }
    }

    // --- masked variants -----------------------------------------------------

    masked_shift_v4! {
        sra => _mm_mask_sra_epi32, _mm_maskz_sra_epi32;
    }

    masked_shifti_v4! {
        srai => _mm_mask_sra_epi32, _mm_maskz_sra_epi32;
    }

    masked_binary_v4! {
        srav(Storage<Self::Unsigned>) => _mm_mask_srav_epi32, _mm_maskz_srav_epi32;
    }

    fn avg_floor_c(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_add_epi32(a, mask, Self::bitand(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }

    fn avg_floor_m(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_add_epi32(src, mask, Self::bitand(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }

    fn avg_floor_z(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_add_epi32(mask, Self::bitand(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }

    fn avg_ceil_c(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_sub_epi32(a, mask, Self::bitor(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }

    fn avg_ceil_m(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_sub_epi32(src, mask, Self::bitor(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }

    fn avg_ceil_z(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_sub_epi32(mask, Self::bitor(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }

    masked_via_mov_v4! {
        mov = _mm_mask_mov_epi32, movz = _mm_maskz_mov_epi32;
        mulhrs(a: Storage<Self>, b: Storage<Self>);
    }
}

// Widen/narrow legs against the 256-bit i64x4: single converts.
#[thermite_macros::inline_always]
impl CastRegister<I32x4V4> for super::I64x4V4 {
    fn cast_from(value: Storage<I32x4V4>) -> Storage<Self> {
        unsafe { arch::_mm256_cvtepi32_epi64(value) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::I64x4V4> for I32x4V4 {
    fn cast_from(value: Storage<super::I64x4V4>) -> Storage<Self> {
        unsafe { arch::_mm256_cvtepi64_epi32(value) }
    }

    fn saturating_cast_from(value: Storage<super::I64x4V4>) -> Storage<Self> {
        unsafe { arch::_mm256_cvtsepi64_epi32(value) }
    }
}
