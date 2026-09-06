//! `i64x2` on AVX-512: the 128-bit signed-qword register under EVEX
//! (AVX512VL). Width port of [`i64x4`](super::i64x4). See
//! [`f64x2`](super::f64x2) for the two-lane permute spellings. `KMask2` masks.
//! The v1 `_mm_reduce_epi64_v1!` fold feeds the native VL xmm ops.

use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, BitwiseRegister, CoreRegister, IndexableRegister, IntegerRegister, InterleaveRegister,
        NumericRegister, PartialOrdRegister, Register, SignedIntegerRegister, SignedRegister, Storage, ZeroUpper,
        empty_reg, reg, reg_splat,
    },
};

use super::super::{Avx512Features, DefaultAvx512 as F};
use super::arch;
use super::kmask::KMask2;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct I64x2V4;

#[thermite_macros::inline_always]
impl CoreRegister for I64x2V4 {
    type Lanes = typenum::U2;
    type Storage = arch::__m128i;
    type Mask = KMask2;

    const IS_EMULATED: bool = false;

    const EMPTY: Storage<Self> = empty_reg::<Self>();

    const HAS_EQUAL_SIZE_MASK: bool = false;

    fn blendv(mask: Storage<Self::Mask>, on_false: Storage<Self>, on_true: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_blend_epi64(mask, on_false, on_true) }
    }

    fn zz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_mov_epi64(mask, value) }
    }

    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_mov_epi64(!mask, value) }
    }

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        if const { Z::N >= 2 } {
            value
        } else {
            unsafe { arch::_mm_maskz_mov_epi64(const { ((1u32 << Z::N) - 1) as u8 }, value) }
        }
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        unsafe { arch::_mm_movm_epi64(mask) }
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for I64x2V4 {
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
        unsafe { arch::_mm_ternarylogic_epi64::<{ crate::ternlog_imm!(!C) }>(value, value, value) }
    }

    const HAS_NATIVE_TERNLOG: bool = true;

    fn ternlog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_ternarylogic_epi64::<IMM>(a, b, c) }
    }

    fn bilog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        arch::bilog_ternlog::<Self, IMM>(a, b)
    }

    // --- masked variants -----------------------------------------------------

    masked_binary_v4! {
        bitxor => _mm_mask_xor_epi64, _mm_maskz_xor_epi64;
        bitand => _mm_mask_and_epi64, _mm_maskz_and_epi64;
        bitor => _mm_mask_or_epi64, _mm_maskz_or_epi64;
    }

    masked_andnot_v4! {
        bitandnot => _mm_mask_andnot_epi64, _mm_maskz_andnot_epi64;
    }

    fn not_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_ternarylogic_epi64::<{ crate::ternlog_imm!(!C) }>(value, mask, value, value) }
    }

    fn not_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_ternarylogic_epi64::<{ crate::ternlog_imm!(!C) }>(src, mask, value, value) }
    }

    fn not_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_ternarylogic_epi64::<{ crate::ternlog_imm!(!C) }>(mask, value, value, value) }
    }

    fn ternlog_c<const IMM: i32>(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_ternarylogic_epi64::<IMM>(a, mask, b, c) }
    }

    fn ternlog_m<const IMM: i32>(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_mov_epi64(src, mask, Self::ternlog::<IMM>(a, b, c)) }
    }

    fn ternlog_z<const IMM: i32>(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_ternarylogic_epi64::<IMM>(mask, a, b, c) }
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

// Width ladder bottom rung: two scalars make an xmm, and ymm is the 2x wide register.
#[thermite_macros::inline_always]
impl crate::register::ConcatRegister<i64> for I64x2V4 {
    fn concat(lo: Storage<i64>, hi: Storage<i64>) -> Storage<Self> {
        unsafe { arch::_mm_set_epi64x(hi, lo) }
    }

    fn split(value: Storage<Self>) -> (Storage<i64>, Storage<i64>) {
        let lo = unsafe { arch::_mm_cvtsi128_si64(value) };
        let hi = unsafe { arch::_mm_extract_epi64::<1>(value) };
        (lo, hi)
    }
}

#[thermite_macros::inline_always]
impl crate::register::ExtendRegister<i64> for I64x2V4 {
    fn extend(value: Storage<i64>) -> Storage<Self> {
        unsafe { arch::_mm_cvtsi64x_si128(value) }
    }

    fn narrow(value: Storage<Self>) -> Storage<i64> {
        unsafe { arch::_mm_cvtsi128_si64(value) }
    }
}

#[thermite_macros::inline_always]
impl crate::register::WideRegister for I64x2V4 {
    type Wide = super::I64x4V4;
}

#[thermite_macros::inline_always]
impl Register for I64x2V4 {
    type Element = i64;

    type Signed = I64x2V4;
    type Unsigned = super::U64x2V4;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_test_epi64_mask(value, value) }
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::msb_to_mask(value)
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_movepi64_mask(value) }
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_loadu_si128(value.as_ptr() as *const _) }
    }

    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_set_epi64x(0, value) }
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_set1_epi64x(value) }
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_load_si128(ptr as *const _) }
    }

    unsafe fn load_m(src: Storage<Self>, mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_mask_loadu_epi64(src, mask, ptr) }
    }

    unsafe fn load_z(mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_maskz_loadu_epi64(mask, ptr) }
    }

    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_loadu_si128(ptr as *const _) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_store_si128(ptr as *mut _, value) }
    }

    unsafe fn store_masked(ptr: *mut Self::Element, mask: Storage<Self::Mask>, value: Storage<Self>) {
        unsafe { arch::_mm_mask_storeu_epi64(ptr, mask, value) }
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
        unsafe { arch::_mm_shuffle_epi32::<0x4E>(value) }
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        unsafe {
            let pattern = arch::_mm_setr_epi8(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8);
            arch::_mm_shuffle_epi8(value, pattern)
        }
    }

    compress_expand_v4!(
        u8,
        _mm_maskz_compress_epi64,
        _mm_mask_expand_epi64,
        _mm_maskz_expand_epi64
    );

    const HAS_PERMUTEV: bool = true;

    fn permutev(value: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_permutex2var_epi64(value, idxs, value) }
    }

    fn swizzle(a: Storage<Self>, b: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_permutex2var_epi64(a, idxs, b) }
    }

    // --- masked variants -----------------------------------------------------

    fn new_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: GenericArray<Self::Element, Self::Lanes>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mask_loadu_epi64(src, mask, value.as_ptr()) }
    }

    fn new_z(mask: Storage<Self::Mask>, value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_loadu_epi64(mask, value.as_ptr()) }
    }

    fn splat_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_mask_set1_epi64(src, mask, value) }
    }

    fn splat_z(mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_maskz_set1_epi64(mask, value) }
    }

    fn broadcast<const I: usize>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_permutex2var_epi64(value, arch::_mm_set1_epi64x(I as i64), value) }
    }

    fn broadcast_c<const I: usize>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_permutex2var_epi64(value, mask, arch::_mm_set1_epi64x(I as i64), value) }
    }

    fn broadcast_m<const I: usize>(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mask_mov_epi64(src, mask, Self::broadcast::<I>(value)) }
    }

    fn broadcast_z<const I: usize>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_permutex2var_epi64(mask, value, arch::_mm_set1_epi64x(I as i64), value) }
    }

    fn broadcastv(value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm_permutex2var_epi64(value, arch::_mm_set1_epi64x(idx as i64), value) }
    }

    fn broadcastv_c(mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm_mask_permutex2var_epi64(value, mask, arch::_mm_set1_epi64x(idx as i64), value) }
    }

    fn broadcastv_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm_mask_mov_epi64(src, mask, Self::broadcastv(value, idx)) }
    }

    fn broadcastv_z(mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm_maskz_permutex2var_epi64(mask, value, arch::_mm_set1_epi64x(idx as i64), value) }
    }

    fn reverse_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_shuffle_epi32::<0x4E>(value, mask, value) }
    }

    fn reverse_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_shuffle_epi32::<0x4E>(src, mask, value) }
    }

    fn reverse_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_shuffle_epi32::<0x4E>(mask, value) }
    }

    masked_via_mov_v4! {
        mov = _mm_mask_mov_epi64, movz = _mm_maskz_mov_epi64;
        swap_bytes(value: Storage<Self>);
    }

    fn permutev_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mask_mov_epi64(src, mask, arch::_mm_permutex2var_epi64(value, idxs, value)) }
    }

    fn permutev_z(mask: Storage<Self::Mask>, value: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_permutex2var_epi64(mask, value, idxs, value) }
    }

    fn swizzle_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        a: Storage<Self>,
        b: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mask_mov_epi64(src, mask, arch::_mm_permutex2var_epi64(a, idxs, b)) }
    }

    fn swizzle_z(
        mask: Storage<Self::Mask>,
        a: Storage<Self>,
        b: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_maskz_permutex2var_epi64(mask, a, idxs, b) }
    }
}

#[thermite_macros::inline_always]
impl InterleaveRegister for I64x2V4 {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe { (arch::_mm_unpacklo_epi64(a, b), arch::_mm_unpackhi_epi64(a, b)) }
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe { (arch::_mm_unpacklo_epi64(a, b), arch::_mm_unpackhi_epi64(a, b)) }
    }
}

#[thermite_macros::inline_always]
impl IndexableRegister<super::U64x2V4> for I64x2V4 {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<super::U64x2V4>) -> Storage<Self> {
        unsafe { arch::_mm_i64gather_epi64::<8>(ptr, indices) }
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<super::U64x2V4>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mmask_i64gather_epi64::<8>(src, mask, indices, ptr) }
    }

    unsafe fn scatter(value: Storage<Self>, ptr: *mut Self::Element, indices: Storage<super::U64x2V4>) {
        unsafe { arch::_mm_i64scatter_epi64::<8>(ptr, indices, value) }
    }

    unsafe fn scatter_m(
        value: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *mut Self::Element,
        indices: Storage<super::U64x2V4>,
    ) {
        unsafe { arch::_mm_mask_i64scatter_epi64::<8>(ptr, mask, indices, value) }
    }
}

#[thermite_macros::inline_always]
impl BitshiftRegister for I64x2V4 {
    const HAS_TRUE_SHIFTV: bool = true;
    const HAS_WIDE_BYTE_SHIFTS: bool = true;

    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_sll_epi64(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_srl_epi64(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    fn shlv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_sllv_epi64(value, shifts) }
    }

    fn shrv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_srlv_epi64(value, shifts) }
    }

    fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sll_epi64(value, arch::_mm_cvtsi32_si128(IMM8)) }
    }

    fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_srl_epi64(value, arch::_mm_cvtsi32_si128(IMM8)) }
    }

    fn bshli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bslli_si128::<IMM8>(value) }
    }

    fn bshri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bsrli_si128::<IMM8>(value) }
    }

    fn rol(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_rolv_epi64(value, arch::_mm_set1_epi64x(shift as i64)) }
    }

    fn ror(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_rorv_epi64(value, arch::_mm_set1_epi64x(shift as i64)) }
    }

    fn roli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_rolv_epi64(value, arch::_mm_set1_epi64x(IMM8 as i64)) }
    }

    fn rori<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_rorv_epi64(value, arch::_mm_set1_epi64x(IMM8 as i64)) }
    }

    fn rolv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_rolv_epi64(value, shifts) }
    }

    fn rorv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_rorv_epi64(value, shifts) }
    }

    // --- masked variants -----------------------------------------------------

    masked_shift_v4! {
        shl => _mm_mask_sll_epi64, _mm_maskz_sll_epi64;
        shr => _mm_mask_srl_epi64, _mm_maskz_srl_epi64;
    }

    masked_shifti_v4! {
        shli => _mm_mask_sll_epi64, _mm_maskz_sll_epi64;
        shri => _mm_mask_srl_epi64, _mm_maskz_srl_epi64;
    }

    masked_binary_v4! {
        shlv(Storage<Self::Unsigned>) => _mm_mask_sllv_epi64, _mm_maskz_sllv_epi64;
        shrv(Storage<Self::Unsigned>) => _mm_mask_srlv_epi64, _mm_maskz_srlv_epi64;
        rolv(Storage<Self::Unsigned>) => _mm_mask_rolv_epi64, _mm_maskz_rolv_epi64;
        rorv(Storage<Self::Unsigned>) => _mm_mask_rorv_epi64, _mm_maskz_rorv_epi64;
    }

    fn rol_c(mask: Storage<Self::Mask>, value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_mask_rolv_epi64(value, mask, value, arch::_mm_set1_epi64x(shift as i64)) }
    }

    fn rol_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_mask_rolv_epi64(src, mask, value, arch::_mm_set1_epi64x(shift as i64)) }
    }

    fn rol_z(mask: Storage<Self::Mask>, value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_maskz_rolv_epi64(mask, value, arch::_mm_set1_epi64x(shift as i64)) }
    }

    fn ror_c(mask: Storage<Self::Mask>, value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_mask_rorv_epi64(value, mask, value, arch::_mm_set1_epi64x(shift as i64)) }
    }

    fn ror_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_mask_rorv_epi64(src, mask, value, arch::_mm_set1_epi64x(shift as i64)) }
    }

    fn ror_z(mask: Storage<Self::Mask>, value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_maskz_rorv_epi64(mask, value, arch::_mm_set1_epi64x(shift as i64)) }
    }

    fn roli_c<const IMM8: i32>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_rolv_epi64(value, mask, value, arch::_mm_set1_epi64x(IMM8 as i64)) }
    }

    fn roli_m<const IMM8: i32>(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_rolv_epi64(src, mask, value, arch::_mm_set1_epi64x(IMM8 as i64)) }
    }

    fn roli_z<const IMM8: i32>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_rolv_epi64(mask, value, arch::_mm_set1_epi64x(IMM8 as i64)) }
    }

    fn rori_c<const IMM8: i32>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_rorv_epi64(value, mask, value, arch::_mm_set1_epi64x(IMM8 as i64)) }
    }

    fn rori_m<const IMM8: i32>(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_rorv_epi64(src, mask, value, arch::_mm_set1_epi64x(IMM8 as i64)) }
    }

    fn rori_z<const IMM8: i32>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_rorv_epi64(mask, value, arch::_mm_set1_epi64x(IMM8 as i64)) }
    }

    masked_via_mov_v4! {
        mov = _mm_mask_mov_epi64, movz = _mm_maskz_mov_epi64;
        bshli<const IMM8: i32>(value: Storage<Self>);
        bshri<const IMM8: i32>(value: Storage<Self>);
        reverse_bits(value: Storage<Self>);
    }
}

#[thermite_macros::inline_always]
impl PartialOrdRegister for I64x2V4 {
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmpgt_epi64_mask(lhs, rhs) }
    }

    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmpge_epi64_mask(lhs, rhs) }
    }

    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmplt_epi64_mask(lhs, rhs) }
    }

    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmple_epi64_mask(lhs, rhs) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmpeq_epi64_mask(lhs, rhs) }
    }

    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmpneq_epi64_mask(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl NumericRegister for I64x2V4 {
    const ZERO: Storage<Self> = reg::<Self, 2>([0; 2]);
    const ONE: Storage<Self> = reg::<Self, 2>([1; 2]);
    const TWO: Storage<Self> = reg::<Self, 2>([2; 2]);

    const MIN: Storage<Self> = reg::<Self, 2>([i64::MIN; 2]);
    const MAX: Storage<Self> = reg::<Self, 2>([i64::MAX; 2]);

    fn min_element(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi64_v1!(value; _mm_min_epi64)
    }

    fn max_element(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi64_v1!(value; _mm_max_epi64)
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi64_v1!(value; _mm_add_epi64)
    }

    fn prod_elements(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi64_v1!(value; _mm_mullo_epi64)
    }

    fn relaxed_pairwise_sum(lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self> {
        let (even, odd) = <Self as InterleaveRegister>::deinterleave(lo, hi);
        Self::add(even, odd)
    }

    fn pairwise_sum(lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self> {
        Self::relaxed_pairwise_sum(lo, hi)
    }

    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as Unsigned>::I64)
    }

    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as i64))
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_add_epi64(lhs, rhs) }
    }

    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sub_epi64(lhs, rhs) }
    }

    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mullo_epi64(lhs, rhs) }
    }

    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a / b })
    }

    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a % b })
    }

    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_min_epi64(lhs, rhs) }
    }

    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_max_epi64(lhs, rhs) }
    }

    // --- masked variants -----------------------------------------------------

    masked_binary_v4! {
        add => _mm_mask_add_epi64, _mm_maskz_add_epi64;
        sub => _mm_mask_sub_epi64, _mm_maskz_sub_epi64;
        mul => _mm_mask_mullo_epi64, _mm_maskz_mullo_epi64;
        min => _mm_mask_min_epi64, _mm_maskz_min_epi64;
        max => _mm_mask_max_epi64, _mm_maskz_max_epi64;
    }

    masked_via_mov_v4! {
        mov = _mm_mask_mov_epi64, movz = _mm_maskz_mov_epi64;
        div(lhs: Storage<Self>, rhs: Storage<Self>);
        rem(lhs: Storage<Self>, rhs: Storage<Self>);
    }

    fn square_c(mask: Storage<Self::Mask>, lhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_mullo_epi64(lhs, mask, lhs, lhs) }
    }

    fn square_m(src: Storage<Self>, mask: Storage<Self::Mask>, lhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_mullo_epi64(src, mask, lhs, lhs) }
    }

    fn square_z(mask: Storage<Self::Mask>, lhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_mullo_epi64(mask, lhs, lhs) }
    }

    fn scale_c(mask: Storage<Self::Mask>, value: Storage<Self>, scalar: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_mask_mullo_epi64(value, mask, value, Self::splat(scalar)) }
    }

    fn scale_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
        scalar: Self::Element,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mask_mullo_epi64(src, mask, value, Self::splat(scalar)) }
    }

    fn scale_z(mask: Storage<Self::Mask>, value: Storage<Self>, scalar: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_maskz_mullo_epi64(mask, value, Self::splat(scalar)) }
    }
}

#[thermite_macros::inline_always]
impl SignedRegister for I64x2V4 {
    const NEG_ONE: Storage<Self> = reg::<Self, 2>([-1; 2]);
    const MIN_POSITIVE: Storage<Self> = reg_splat::<Self>(1);

    fn neg(value: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::ZERO, value)
    }

    fn is_negative(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::msb_to_mask(value)
    }

    fn abs(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_abs_epi64(value) }
    }

    fn copysign(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::neg_c(Self::msb_to_mask(Self::bitxor(lhs, rhs)), lhs)
    }

    fn signum(value: Storage<Self>) -> Storage<Self> {
        Self::min(Self::max(value, Self::NEG_ONE), Self::ONE)
    }

    // --- masked variants -----------------------------------------------------

    fn neg_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_sub_epi64(value, mask, Self::ZERO, value) }
    }

    fn neg_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_sub_epi64(src, mask, Self::ZERO, value) }
    }

    fn neg_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_sub_epi64(mask, Self::ZERO, value) }
    }

    masked_unary_v4! {
        abs => _mm_mask_abs_epi64, _mm_maskz_abs_epi64;
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
        unsafe { arch::_mm_mask_mov_epi64(src, mask, Self::copysign(lhs, rhs)) }
    }

    fn copysign_z(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_mov_epi64(mask, Self::copysign(lhs, rhs)) }
    }
}

#[thermite_macros::inline_always]
impl IntegerRegister for I64x2V4 {
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe {
            let lomask = arch::_mm_set1_epi64x(0xFFFF_FFFF);

            let xh = arch::_mm_srli_epi64::<32>(lhs);
            let yh = arch::_mm_srli_epi64::<32>(rhs);

            let w0 = arch::_mm_mul_epu32(lhs, rhs);
            let w1 = arch::_mm_mul_epu32(lhs, yh);
            let w2 = arch::_mm_mul_epu32(xh, rhs);
            let w3 = arch::_mm_mul_epu32(xh, yh);

            let s1 = arch::_mm_add_epi64(w1, arch::_mm_srli_epi64::<32>(w0));
            let s1l = arch::_mm_and_si128(s1, lomask);
            let s1h = arch::_mm_srli_epi64::<32>(s1);
            let s2h = arch::_mm_srli_epi64::<32>(arch::_mm_add_epi64(w2, s1l));

            let hi = arch::_mm_add_epi64(arch::_mm_add_epi64(w3, s1h), s2h);

            let hi = arch::_mm_mask_sub_epi64(hi, arch::_mm_movepi64_mask(lhs), hi, rhs);
            arch::_mm_mask_sub_epi64(hi, arch::_mm_movepi64_mask(rhs), hi, lhs)
        }
    }

    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mullo_epi64(lhs, rhs) }
    }

    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe {
            let sum = Self::add(lhs, rhs);
            let ov = arch::_mm_movepi64_mask(
                arch::_mm_ternarylogic_epi64::<{ crate::ternlog_imm!((A ^ C) & (B ^ C)) }>(lhs, rhs, sum),
            );
            let sat = Self::bitxor(Self::MAX, Self::srai::<63>(lhs));
            arch::_mm_mask_mov_epi64(sum, ov, sat)
        }
    }

    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe {
            let diff = Self::sub(lhs, rhs);
            let ov = arch::_mm_movepi64_mask(
                arch::_mm_ternarylogic_epi64::<{ crate::ternlog_imm!((A ^ B) & (A ^ C)) }>(lhs, rhs, diff),
            );
            let sat = Self::bitxor(Self::MAX, Self::srai::<63>(lhs));
            arch::_mm_mask_mov_epi64(diff, ov, sat)
        }
    }

    fn wrapping_sum(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi64_v1!(value; _mm_add_epi64)
    }

    fn wrapping_product(value: Storage<Self>) -> Self::Element {
        _mm_reduce_epi64_v1!(value; _mm_mullo_epi64)
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
        Self::count_ones(unsafe { arch::_mm_conflict_epi64(value) })
    }

    fn count_ones(value: Storage<Self>) -> Storage<Self> {
        if const { <F as Avx512Features>::AVX512VPOPCNTDQ } {
            unsafe { arch::_mm_popcnt_epi64(value) }
        } else {
            unsafe { arch::_mm_popcnt_epi64x_v2(value) }
        }
    }

    fn count_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(Self::not(value))
    }

    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_lzcnt_epi64(value) }
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
        mullo => _mm_mask_mullo_epi64, _mm_maskz_mullo_epi64;
    }

    masked_via_mov_v4! {
        mov = _mm_mask_mov_epi64, movz = _mm_maskz_mov_epi64;
        mulhi(lhs: Storage<Self>, rhs: Storage<Self>);
        saturating_add(lhs: Storage<Self>, rhs: Storage<Self>);
        saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>);
        div_branched(value: Storage<Self>, divider: crate::Divider<Self::Element>);
        div_branchfree(value: Storage<Self>, divider: crate::BranchfreeDivider<Self::Element>);
        divv_branchfree(value: Storage<Self>, dividers: crate::divider::vector::VectorDivider<Self>);
    }

    masked_unary_v4! {
        count_ones => _mm_mask_popcnt_epi64x_v4, _mm_maskz_popcnt_epi64x_v4;
        leading_zeros => _mm_mask_lzcnt_epi64, _mm_maskz_lzcnt_epi64;
    }

    fn count_zeros_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_popcnt_epi64x_v4(value, mask, Self::not(value)) }
    }

    fn count_zeros_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_popcnt_epi64x_v4(src, mask, Self::not(value)) }
    }

    fn count_zeros_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_popcnt_epi64x_v4(mask, Self::not(value)) }
    }

    fn trailing_zeros_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let low = Self::sub(Self::bitand(value, Self::neg(value)), Self::ONE);
        unsafe { arch::_mm_mask_popcnt_epi64x_v4(value, mask, low) }
    }

    fn trailing_zeros_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let low = Self::sub(Self::bitand(value, Self::neg(value)), Self::ONE);
        unsafe { arch::_mm_mask_popcnt_epi64x_v4(src, mask, low) }
    }

    fn trailing_zeros_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let low = Self::sub(Self::bitand(value, Self::neg(value)), Self::ONE);
        unsafe { arch::_mm_maskz_popcnt_epi64x_v4(mask, low) }
    }

    fn leading_ones_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_lzcnt_epi64(value, mask, Self::not(value)) }
    }

    fn leading_ones_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_lzcnt_epi64(src, mask, Self::not(value)) }
    }

    fn leading_ones_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_lzcnt_epi64(mask, Self::not(value)) }
    }

    fn trailing_ones_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let inv = Self::not(value);
        let low = Self::sub(Self::bitand(inv, Self::neg(inv)), Self::ONE);
        unsafe { arch::_mm_mask_popcnt_epi64x_v4(value, mask, low) }
    }

    fn trailing_ones_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let inv = Self::not(value);
        let low = Self::sub(Self::bitand(inv, Self::neg(inv)), Self::ONE);
        unsafe { arch::_mm_mask_popcnt_epi64x_v4(src, mask, low) }
    }

    fn trailing_ones_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let inv = Self::not(value);
        let low = Self::sub(Self::bitand(inv, Self::neg(inv)), Self::ONE);
        unsafe { arch::_mm_maskz_popcnt_epi64x_v4(mask, low) }
    }
}

#[thermite_macros::inline_always]
impl SignedIntegerRegister for I64x2V4 {
    fn srai<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sra_epi64(value, arch::_mm_cvtsi32_si128(IMM8)) }
    }

    fn sra(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm_sra_epi64(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    fn srav(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_srav_epi64(value, shifts) }
    }

    // --- masked variants -----------------------------------------------------

    masked_shift_v4! {
        sra => _mm_mask_sra_epi64, _mm_maskz_sra_epi64;
    }

    masked_shifti_v4! {
        srai => _mm_mask_sra_epi64, _mm_maskz_sra_epi64;
    }

    masked_binary_v4! {
        srav(Storage<Self::Unsigned>) => _mm_mask_srav_epi64, _mm_maskz_srav_epi64;
    }

    fn avg_floor_c(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_add_epi64(a, mask, Self::bitand(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }

    fn avg_floor_m(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_add_epi64(src, mask, Self::bitand(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }

    fn avg_floor_z(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_add_epi64(mask, Self::bitand(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }

    fn avg_ceil_c(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_sub_epi64(a, mask, Self::bitor(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }

    fn avg_ceil_m(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_sub_epi64(src, mask, Self::bitor(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }

    fn avg_ceil_z(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_sub_epi64(mask, Self::bitor(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }

    masked_via_mov_v4! {
        mov = _mm_mask_mov_epi64, movz = _mm_maskz_mov_epi64;
        mulhrs(a: Storage<Self>, b: Storage<Self>);
    }
}
