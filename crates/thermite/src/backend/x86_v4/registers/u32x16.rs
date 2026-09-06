//! `u32x16` on AVX-512: the native 512-bit unsigned-dword register, the
//! unsigned sibling of i32x16.rs (see that file for the integer pattern and
//! f32x16.rs for the shared house rules).
//!
//! Unsigned-specific notes:
//!
//! - **Unsigned compares are first-class**: `vpcmpud` covers the whole
//!   relation set, so none of the v3 `_mm256_cmp*_epu32x_v3` polyfills
//!   survive. `min`/`max` are `vpminud`/`vpmaxud` as before.
//! - **Saturation is min/max plus the arithmetic**: `a +| b = a + min(b, !a)`,
//!   `a -| b = max(a, b) - b`. No compare, no mask register, and the final
//!   add/sub is where the `_c`/`_m`/`_z` variants put the caller's mask.
//! - `count_ones` forks on `F::AVX512VPOPCNTDQ` exactly like i32x16.rs, and
//!   `leading_zeros` is the single CD `vplzcntd`. That inverts the v3
//!   dependency: there `leading_zeros` was built from `ilog2p1`, here
//!   `ilog2p1` is `32 - lzcnt`.
//! - Morton stays on the v3 PSHUFB nibble-LUT, half-split across the two
//!   256-bit halves. A dedicated 512-bit body is an M4 item.

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
        UnsignedIntegerRegister, ZeroUpper, array::ArrayRegister, empty_reg, reg,
    },
};

use super::super::{Avx512Features, DefaultAvx512 as F};
use super::arch;
use super::kmask::KMask16;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct U32x16V4;

#[thermite_macros::inline_always]
impl CoreRegister for U32x16V4 {
    type Lanes = typenum::U16;
    type Storage = arch::__m512i;
    type Mask = KMask16;

    const IS_EMULATED: bool = false;

    const EMPTY: Storage<Self> = empty_reg::<Self>();

    const HAS_EQUAL_SIZE_MASK: bool = false;

    fn blendv(mask: Storage<Self::Mask>, on_false: Storage<Self>, on_true: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_blend_epi32(mask, on_false, on_true) }
    }

    fn zz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_mov_epi32(mask, value) }
    }

    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_mov_epi32(!mask, value) }
    }

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        if const { Z::N >= 16 } {
            value
        } else {
            unsafe { arch::_mm512_maskz_mov_epi32(const { ((1u32 << Z::N) - 1) as u16 }, value) }
        }
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        unsafe { arch::_mm512_movm_epi32(mask) }
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for U32x16V4 {
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_xor_si512(lhs, rhs) }
    }

    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_and_si512(lhs, rhs) }
    }

    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_andnot_si512(rhs, lhs) }
    }

    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_or_si512(lhs, rhs) }
    }

    fn not(value: Storage<Self>) -> Storage<Self> {
        // vpternlog imm 0x55 = NOT(a): one instruction, no constant load.
        unsafe { arch::_mm512_ternarylogic_epi32::<{ crate::ternlog_imm!(!C) }>(value, value, value) }
    }

    const HAS_NATIVE_TERNLOG: bool = true;

    fn ternlog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_ternarylogic_epi32::<IMM>(a, b, c) }
    }

    fn bilog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        arch::bilog_ternlog::<Self, IMM>(a, b)
    }

    // --- masked variants -----------------------------------------------------

    masked_binary_v4! {
        bitxor => _mm512_mask_xor_epi32, _mm512_maskz_xor_epi32;
        bitand => _mm512_mask_and_epi32, _mm512_maskz_and_epi32;
        bitor => _mm512_mask_or_epi32, _mm512_maskz_or_epi32;
    }

    masked_andnot_v4! {
        bitandnot => _mm512_mask_andnot_epi32, _mm512_maskz_andnot_epi32;
    }

    // vpternlog's merge form keeps operand A (`src` doubles as A), so the
    // NOT imm must read operand C: 0x55.
    fn not_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_ternarylogic_epi32::<{ crate::ternlog_imm!(!C) }>(value, mask, value, value) }
    }

    fn not_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_ternarylogic_epi32::<{ crate::ternlog_imm!(!C) }>(src, mask, value, value) }
    }

    fn not_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_ternarylogic_epi32::<{ crate::ternlog_imm!(!C) }>(mask, value, value, value) }
    }

    fn ternlog_c<const IMM: i32>(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_ternarylogic_epi32::<IMM>(a, mask, b, c) }
    }

    // The merge form's `src` is also operand A, so an arbitrary `src` costs
    // a separate merge move.
    fn ternlog_m<const IMM: i32>(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mov_epi32(src, mask, Self::ternlog::<IMM>(a, b, c)) }
    }

    fn ternlog_z<const IMM: i32>(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_ternarylogic_epi32::<IMM>(mask, a, b, c) }
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

// Width ladder: zmm = two ymm. `vinserti32x8` into a zero-extended low half.
#[thermite_macros::inline_always]
impl crate::register::ConcatRegister<super::U32x8V4> for U32x16V4 {
    fn concat(lo: Storage<super::U32x8V4>, hi: Storage<super::U32x8V4>) -> Storage<Self> {
        unsafe { arch::_mm512_inserti32x8::<1>(arch::_mm512_zextsi256_si512(lo), hi) }
    }

    fn split(value: Storage<Self>) -> (Storage<super::U32x8V4>, Storage<super::U32x8V4>) {
        let lo = unsafe { arch::_mm512_castsi512_si256(value) };
        let hi = unsafe { arch::_mm512_extracti32x8_epi32::<1>(value) };
        (lo, hi)
    }
}

#[thermite_macros::inline_always]
impl crate::register::ExtendRegister<super::U32x8V4> for U32x16V4 {
    fn extend(value: Storage<super::U32x8V4>) -> Storage<Self> {
        unsafe { arch::_mm512_zextsi256_si512(value) }
    }

    fn narrow(value: Storage<Self>) -> Storage<super::U32x8V4> {
        unsafe { arch::_mm512_castsi512_si256(value) }
    }
}

#[thermite_macros::inline_always]
impl Register for U32x16V4 {
    type Element = u32;

    type Signed = super::I32x16V4;
    type Unsigned = U32x16V4;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_test_epi32_mask(value, value) }
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::msb_to_mask(value)
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_movepi32_mask(value) }
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm512_loadu_si512(value.as_ptr() as *const _) }
    }

    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_zextsi128_si512(arch::_mm_cvtsi32_si128(value as i32)) }
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_set1_epi32(value as i32) }
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_load_si512(ptr as *const _) }
    }

    unsafe fn load_m(src: Storage<Self>, mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_mask_loadu_epi32(src, mask, ptr as *const _) }
    }

    unsafe fn load_z(mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_loadu_epi32(mask, ptr as *const _) }
    }

    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_loadu_si512(ptr as *const _) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm512_store_si512(ptr as *mut _, value) }
    }

    unsafe fn store_masked(ptr: *mut Self::Element, mask: Storage<Self::Mask>, value: Storage<Self>) {
        unsafe { arch::_mm512_mask_storeu_epi32(ptr as *mut _, mask, value) }
    }

    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm512_storeu_si512(ptr as *mut _, value) }
    }

    unsafe fn load_stream(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_stream_load_si512(ptr as _) }
    }

    unsafe fn store_stream(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm512_stream_si512(ptr as _, value) }
    }

    unsafe fn lookup(values: &[Self::Element], indices: Storage<Self::Unsigned>) -> Storage<Self> {
        if values.len() <= <Self::Lanes as Unsigned>::USIZE {
            let mut padded: GenericArray<Self::Element, Self::Lanes> = unsafe { core::mem::zeroed() };
            padded[..values.len()].copy_from_slice(values);

            unsafe { arch::_mm512_permutexvar_epi32(indices, Self::new(padded)) }
        } else {
            unsafe { <Self as IndexableRegister<Self::Unsigned>>::gather(values.as_ptr(), indices) }
        }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        unsafe {
            let idx = arch::_mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
            arch::_mm512_permutexvar_epi32(idx, value)
        }
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        unsafe {
            let pattern = arch::_mm512_broadcast_i32x4(arch::_mm_setr_epi8(
                3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
            ));
            arch::_mm512_shuffle_epi8(value, pattern)
        }
    }

    compress_expand_v4!(
        u16,
        _mm512_maskz_compress_epi32,
        _mm512_mask_expand_epi32,
        _mm512_maskz_expand_epi32
    );

    const HAS_PERMUTEV: bool = true;

    fn permutev(value: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm512_permutexvar_epi32(idxs, value) }
    }

    fn swizzle(a: Storage<Self>, b: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm512_permutex2var_epi32(a, idxs, b) }
    }

    // --- masked variants -----------------------------------------------------

    fn new_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: GenericArray<Self::Element, Self::Lanes>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_loadu_epi32(src, mask, value.as_ptr() as *const _) }
    }

    fn new_z(mask: Storage<Self::Mask>, value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_loadu_epi32(mask, value.as_ptr() as *const _) }
    }

    // vpbroadcastd from a GPR takes the mask directly.
    fn splat_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_mask_set1_epi32(src, mask, value as i32) }
    }

    fn splat_z(mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_set1_epi32(mask, value as i32) }
    }

    // Lane broadcast: one vpermd with a constant index, instead of the
    // default's extract-to-scalar round trip. The masked forms ride the
    // same instruction.
    fn broadcast<const I: usize>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_permutexvar_epi32(arch::_mm512_set1_epi32(I as i32), value) }
    }

    fn broadcast_c<const I: usize>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_permutexvar_epi32(value, mask, arch::_mm512_set1_epi32(I as i32), value) }
    }

    fn broadcast_m<const I: usize>(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_permutexvar_epi32(src, mask, arch::_mm512_set1_epi32(I as i32), value) }
    }

    fn broadcast_z<const I: usize>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_permutexvar_epi32(mask, arch::_mm512_set1_epi32(I as i32), value) }
    }

    fn broadcastv(value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm512_permutexvar_epi32(arch::_mm512_set1_epi32(idx as i32), value) }
    }

    fn broadcastv_c(mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm512_mask_permutexvar_epi32(value, mask, arch::_mm512_set1_epi32(idx as i32), value) }
    }

    fn broadcastv_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm512_mask_permutexvar_epi32(src, mask, arch::_mm512_set1_epi32(idx as i32), value) }
    }

    fn broadcastv_z(mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_permutexvar_epi32(mask, arch::_mm512_set1_epi32(idx as i32), value) }
    }

    fn reverse_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe {
            let idx = arch::_mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
            arch::_mm512_mask_permutexvar_epi32(value, mask, idx, value)
        }
    }

    fn reverse_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe {
            let idx = arch::_mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
            arch::_mm512_mask_permutexvar_epi32(src, mask, idx, value)
        }
    }

    fn reverse_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe {
            let idx = arch::_mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
            arch::_mm512_maskz_permutexvar_epi32(mask, idx, value)
        }
    }

    // vpshufb is masked per BYTE (`__mmask64`), so the lane mask cannot feed
    // it directly: shuffle, then one merge move.
    masked_via_mov_v4! {
        mov = _mm512_mask_mov_epi32, movz = _mm512_maskz_mov_epi32;
        swap_bytes(value: Storage<Self>);
    }

    fn permutev_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_permutexvar_epi32(src, mask, idxs, value) }
    }

    fn permutev_z(mask: Storage<Self::Mask>, value: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_permutexvar_epi32(mask, idxs, value) }
    }

    // vpermt2d merges only into `a` (vpermi2d into the index), neither of
    // which is an arbitrary `src`: permute, then one merge move. The zeroing
    // form is native.
    fn swizzle_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        a: Storage<Self>,
        b: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mov_epi32(src, mask, arch::_mm512_permutex2var_epi32(a, idxs, b)) }
    }

    fn swizzle_z(
        mask: Storage<Self::Mask>,
        a: Storage<Self>,
        b: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_permutex2var_epi32(mask, a, idxs, b) }
    }
}

#[thermite_macros::inline_always]
impl InterleaveRegister for U32x16V4 {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe {
            let lo = arch::_mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
            let hi = arch::_mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
            (
                arch::_mm512_permutex2var_epi32(a, lo, b),
                arch::_mm512_permutex2var_epi32(a, hi, b),
            )
        }
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe {
            let even = arch::_mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
            let odd = arch::_mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
            (
                arch::_mm512_permutex2var_epi32(a, even, b),
                arch::_mm512_permutex2var_epi32(a, odd, b),
            )
        }
    }
}

#[thermite_macros::inline_always]
impl IndexableRegister<U32x16V4> for U32x16V4 {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<U32x16V4>) -> Storage<Self> {
        unsafe { arch::_mm512_i32gather_epi32::<4>(indices, ptr as _) }
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<U32x16V4>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_i32gather_epi32::<4>(src, mask, indices, ptr as _) }
    }

    unsafe fn scatter(value: Storage<Self>, ptr: *mut Self::Element, indices: Storage<U32x16V4>) {
        unsafe { arch::_mm512_i32scatter_epi32::<4>(ptr as _, indices, value) }
    }

    unsafe fn scatter_m(
        value: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *mut Self::Element,
        indices: Storage<U32x16V4>,
    ) {
        unsafe { arch::_mm512_mask_i32scatter_epi32::<4>(ptr as _, mask, indices, value) }
    }
}

#[thermite_macros::inline_always]
impl IndexableRegister<ArrayRegister<super::U64x8V4, 2>> for U32x16V4 {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<ArrayRegister<super::U64x8V4, 2>>) -> Storage<Self> {
        let ArrayRegister([lo_idx, hi_idx]) = indices;

        unsafe {
            let lo = arch::_mm512_i64gather_epi32::<4>(lo_idx, ptr as _);
            let hi = arch::_mm512_i64gather_epi32::<4>(hi_idx, ptr as _);
            arch::_mm512_inserti32x8::<1>(arch::_mm512_zextsi256_si512(lo), hi)
        }
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<ArrayRegister<super::U64x8V4, 2>>,
    ) -> Storage<Self> {
        let ArrayRegister([lo_idx, hi_idx]) = indices;

        unsafe {
            let lo =
                arch::_mm512_mask_i64gather_epi32::<4>(arch::_mm512_castsi512_si256(src), mask as u8, lo_idx, ptr as _);
            let hi = arch::_mm512_mask_i64gather_epi32::<4>(
                arch::_mm512_extracti32x8_epi32::<1>(src),
                (mask >> 8) as u8,
                hi_idx,
                ptr as _,
            );
            arch::_mm512_inserti32x8::<1>(arch::_mm512_zextsi256_si512(lo), hi)
        }
    }
}

#[thermite_macros::inline_always]
impl BitshiftRegister for U32x16V4 {
    const HAS_TRUE_SHIFTV: bool = true;
    const HAS_WIDE_BYTE_SHIFTS: bool = false;

    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm512_sll_epi32(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm512_srl_epi32(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    fn shlv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm512_sllv_epi32(value, shifts) }
    }

    fn shrv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm512_srlv_epi32(value, shifts) }
    }

    fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_sll_epi32(value, arch::_mm_cvtsi32_si128(IMM8)) }
    }

    fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_srl_epi32(value, arch::_mm_cvtsi32_si128(IMM8)) }
    }

    // Rotates are native (vprolvd/vprorvd). The count is broadcast rather
    // than immediate for the same `IMM8 as u32` reason as the shifts. LLVM
    // folds a constant count to the vprold/vprord immediate form.
    fn rol(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm512_rolv_epi32(value, arch::_mm512_set1_epi32(shift as i32)) }
    }

    fn ror(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm512_rorv_epi32(value, arch::_mm512_set1_epi32(shift as i32)) }
    }

    fn roli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_rolv_epi32(value, arch::_mm512_set1_epi32(IMM8)) }
    }

    fn rori<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_rorv_epi32(value, arch::_mm512_set1_epi32(IMM8)) }
    }

    fn rolv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm512_rolv_epi32(value, shifts) }
    }

    fn rorv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm512_rorv_epi32(value, shifts) }
    }

    // --- masked variants -----------------------------------------------------

    masked_shift_v4! {
        shl => _mm512_mask_sll_epi32, _mm512_maskz_sll_epi32;
        shr => _mm512_mask_srl_epi32, _mm512_maskz_srl_epi32;
    }

    masked_shifti_v4! {
        shli => _mm512_mask_sll_epi32, _mm512_maskz_sll_epi32;
        shri => _mm512_mask_srl_epi32, _mm512_maskz_srl_epi32;
    }

    masked_binary_v4! {
        shlv(Storage<Self::Unsigned>) => _mm512_mask_sllv_epi32, _mm512_maskz_sllv_epi32;
        shrv(Storage<Self::Unsigned>) => _mm512_mask_srlv_epi32, _mm512_maskz_srlv_epi32;
        rolv(Storage<Self::Unsigned>) => _mm512_mask_rolv_epi32, _mm512_maskz_rolv_epi32;
        rorv(Storage<Self::Unsigned>) => _mm512_mask_rorv_epi32, _mm512_maskz_rorv_epi32;
    }

    fn rol_c(mask: Storage<Self::Mask>, value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm512_mask_rolv_epi32(value, mask, value, arch::_mm512_set1_epi32(shift as i32)) }
    }

    fn rol_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm512_mask_rolv_epi32(src, mask, value, arch::_mm512_set1_epi32(shift as i32)) }
    }

    fn rol_z(mask: Storage<Self::Mask>, value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_rolv_epi32(mask, value, arch::_mm512_set1_epi32(shift as i32)) }
    }

    fn ror_c(mask: Storage<Self::Mask>, value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm512_mask_rorv_epi32(value, mask, value, arch::_mm512_set1_epi32(shift as i32)) }
    }

    fn ror_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm512_mask_rorv_epi32(src, mask, value, arch::_mm512_set1_epi32(shift as i32)) }
    }

    fn ror_z(mask: Storage<Self::Mask>, value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_rorv_epi32(mask, value, arch::_mm512_set1_epi32(shift as i32)) }
    }

    fn roli_c<const IMM8: i32>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_rolv_epi32(value, mask, value, arch::_mm512_set1_epi32(IMM8)) }
    }

    fn roli_m<const IMM8: i32>(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_rolv_epi32(src, mask, value, arch::_mm512_set1_epi32(IMM8)) }
    }

    fn roli_z<const IMM8: i32>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_rolv_epi32(mask, value, arch::_mm512_set1_epi32(IMM8)) }
    }

    fn rori_c<const IMM8: i32>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_rorv_epi32(value, mask, value, arch::_mm512_set1_epi32(IMM8)) }
    }

    fn rori_m<const IMM8: i32>(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_rorv_epi32(src, mask, value, arch::_mm512_set1_epi32(IMM8)) }
    }

    fn rori_z<const IMM8: i32>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_rorv_epi32(mask, value, arch::_mm512_set1_epi32(IMM8)) }
    }

    // Whole-register byte shifts and the bit-reverse cascade have no masked
    // final step (the trait defaults end in a lane-wise loop / an OR of two
    // shifted halves): run them, then one merge move.
    masked_via_mov_v4! {
        mov = _mm512_mask_mov_epi32, movz = _mm512_maskz_mov_epi32;
        bshli<const IMM8: i32>(value: Storage<Self>);
        bshri<const IMM8: i32>(value: Storage<Self>);
        reverse_bits(value: Storage<Self>);
    }
}

#[thermite_macros::inline_always]
impl ShuffleRegister for U32x16V4 {
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe {
            arch::_mm512_castps_si512(arch::_mm512_shuffle_ps(
                arch::_mm512_castsi512_ps(lhs),
                arch::_mm512_castsi512_ps(rhs),
                IMM8,
            ))
        }
    }
}

#[thermite_macros::inline_always]
impl PartialOrdRegister for U32x16V4 {
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmpgt_epu32_mask(lhs, rhs) }
    }

    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmpge_epu32_mask(lhs, rhs) }
    }

    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmplt_epu32_mask(lhs, rhs) }
    }

    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmple_epu32_mask(lhs, rhs) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmpeq_epu32_mask(lhs, rhs) }
    }

    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmpneq_epu32_mask(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl NumericRegister for U32x16V4 {
    const ZERO: Storage<Self> = reg::<Self, 16>([0; 16]);
    const ONE: Storage<Self> = reg::<Self, 16>([1; 16]);
    const TWO: Storage<Self> = reg::<Self, 16>([2; 16]);

    const MIN: Storage<Self> = reg::<Self, 16>([u32::MIN; 16]);
    const MAX: Storage<Self> = reg::<Self, 16>([u32::MAX; 16]);

    fn min_element(value: Storage<Self>) -> Self::Element {
        unsafe { arch::_mm512_reduce_min_epu32(value) }
    }

    fn max_element(value: Storage<Self>) -> Self::Element {
        unsafe { arch::_mm512_reduce_max_epu32(value) }
    }

    // No unsigned add/mul reduction sequences: the epi32 forms are
    // bit-identical modulo the return type, so reinterpret the result.
    fn sum_elements(value: Storage<Self>) -> Self::Element {
        unsafe { arch::_mm512_reduce_add_epi32(value) as u32 }
    }

    fn prod_elements(value: Storage<Self>) -> Self::Element {
        unsafe { arch::_mm512_reduce_mul_epi32(value) as u32 }
    }

    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as Unsigned>::U32)
    }

    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as u32))
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_add_epi32(lhs, rhs) }
    }

    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_sub_epi32(lhs, rhs) }
    }

    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mullo_epi32(lhs, rhs) }
    }

    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a / b })
    }

    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a % b })
    }

    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_min_epu32(lhs, rhs) }
    }

    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_max_epu32(lhs, rhs) }
    }

    // --- masked variants -----------------------------------------------------
    // Embedded-mask forms: keep `lhs`/`src` where the mask is false, one
    // instruction, no blend.

    masked_binary_v4! {
        add => _mm512_mask_add_epi32, _mm512_maskz_add_epi32;
        sub => _mm512_mask_sub_epi32, _mm512_maskz_sub_epi32;
        mul => _mm512_mask_mullo_epi32, _mm512_maskz_mullo_epi32;
        min => _mm512_mask_min_epu32, _mm512_maskz_min_epu32;
        max => _mm512_mask_max_epu32, _mm512_maskz_max_epu32;
    }

    // Integer division is lane-wise scalar. Nothing to mask but the result.
    masked_via_mov_v4! {
        mov = _mm512_mask_mov_epi32, movz = _mm512_maskz_mov_epi32;
        div(lhs: Storage<Self>, rhs: Storage<Self>);
        rem(lhs: Storage<Self>, rhs: Storage<Self>);
    }

    fn square_c(mask: Storage<Self::Mask>, lhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mullo_epi32(lhs, mask, lhs, lhs) }
    }

    fn square_m(src: Storage<Self>, mask: Storage<Self::Mask>, lhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mullo_epi32(src, mask, lhs, lhs) }
    }

    fn square_z(mask: Storage<Self::Mask>, lhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_mullo_epi32(mask, lhs, lhs) }
    }

    fn scale_c(mask: Storage<Self::Mask>, value: Storage<Self>, scalar: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mullo_epi32(value, mask, value, Self::splat(scalar)) }
    }

    fn scale_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
        scalar: Self::Element,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mullo_epi32(src, mask, value, Self::splat(scalar)) }
    }

    fn scale_z(mask: Storage<Self::Mask>, value: Storage<Self>, scalar: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_mullo_epi32(mask, value, Self::splat(scalar)) }
    }
}

#[thermite_macros::inline_always]
impl IntegerRegister for U32x16V4 {
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        // Odd/even unsigned widening multiplies, then merge the two high-dword
        // sets: even products' hi bits land in the low dword of each qword
        // after the shift, odd products' hi bits are already in the high dword.
        unsafe {
            let even_hi = arch::_mm512_srli_epi64::<32>(arch::_mm512_mul_epu32(lhs, rhs));
            let odd_hi = arch::_mm512_mul_epu32(arch::_mm512_srli_epi64::<32>(lhs), arch::_mm512_srli_epi64::<32>(rhs));
            arch::_mm512_mask_blend_epi32(0xAAAA, even_hi, odd_hi)
        }
    }

    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mullo_epi32(lhs, rhs) }
    }

    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        // a +| b = a + min(b, MAX - a) = a + min(b, !a): vpternlog + vpminud +
        // vpaddd, no mask, and the final add is what the masked variants hook.
        Self::add(lhs, Self::min(rhs, Self::not(lhs)))
    }

    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        // a -| b = max(a, b) - b: two instructions with a native vpmaxud, no
        // mask or compare, and again the final subtract takes the mask.
        Self::sub(Self::max(lhs, rhs), rhs)
    }

    fn wrapping_sum(value: Storage<Self>) -> Self::Element {
        unsafe { arch::_mm512_reduce_add_epi32(value) as u32 }
    }

    fn wrapping_product(value: Storage<Self>) -> Self::Element {
        unsafe { arch::_mm512_reduce_mul_epi32(value) as u32 }
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
        Self::count_ones(unsafe { arch::_mm512_conflict_epi32(value) })
    }

    fn count_ones(value: Storage<Self>) -> Storage<Self> {
        if const { <F as Avx512Features>::AVX512VPOPCNTDQ } {
            unsafe { arch::_mm512_popcnt_epi32(value) }
        } else {
            // Floor fallback: full-width nibble-LUT port (vpshufb is BW).
            unsafe { arch::_mm512_popcnt_epi32x_v4(value) }
        }
    }

    fn count_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(Self::not(value))
    }

    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        // vplzcntd is CD, i.e. floor: single instruction, correct for 0 (-> 32).
        unsafe { arch::_mm512_lzcnt_epi32(value) }
    }

    fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {
        // popcount of the isolated-low-bit mask: (v & -v) - 1.
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
        mullo => _mm512_mask_mullo_epi32, _mm512_maskz_mullo_epi32;
    }

    // mulhi's final blend and the divider polyfills have no spare mask slot:
    // compute, then one merge move.
    masked_via_mov_v4! {
        mov = _mm512_mask_mov_epi32, movz = _mm512_maskz_mov_epi32;
        mulhi(lhs: Storage<Self>, rhs: Storage<Self>);
        div_branched(value: Storage<Self>, divider: crate::Divider<Self::Element>);
        div_branchfree(value: Storage<Self>, divider: crate::BranchfreeDivider<Self::Element>);
        divv_branchfree(value: Storage<Self>, dividers: crate::divider::vector::VectorDivider<Self>);
    }

    // Saturating add/sub end in a plain add/sub (see the base bodies), which
    // takes the mask.
    fn saturating_add_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_add_epi32(lhs, mask, lhs, Self::min(rhs, Self::not(lhs))) }
    }

    fn saturating_add_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        lhs: Storage<Self>,
        rhs: Storage<Self>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_add_epi32(src, mask, lhs, Self::min(rhs, Self::not(lhs))) }
    }

    fn saturating_add_z(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_add_epi32(mask, lhs, Self::min(rhs, Self::not(lhs))) }
    }

    fn saturating_sub_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_epi32(lhs, mask, Self::max(lhs, rhs), rhs) }
    }

    fn saturating_sub_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        lhs: Storage<Self>,
        rhs: Storage<Self>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_epi32(src, mask, Self::max(lhs, rhs), rhs) }
    }

    fn saturating_sub_z(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_sub_epi32(mask, Self::max(lhs, rhs), rhs) }
    }

    // The popcount family ends in `count_ones`, whose masked polyfill forks
    // on VPOPCNTDQ exactly like the unmasked one (`bits.rs`), and lzcnt is CD.
    masked_unary_v4! {
        count_ones => _mm512_mask_popcnt_epi32x_v4, _mm512_maskz_popcnt_epi32x_v4;
        leading_zeros => _mm512_mask_lzcnt_epi32, _mm512_maskz_lzcnt_epi32;
    }

    fn count_zeros_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_popcnt_epi32x_v4(value, mask, Self::not(value)) }
    }

    fn count_zeros_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_popcnt_epi32x_v4(src, mask, Self::not(value)) }
    }

    fn count_zeros_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_popcnt_epi32x_v4(mask, Self::not(value)) }
    }

    fn trailing_zeros_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let low = Self::sub(Self::bitand(value, Self::sub(Self::ZERO, value)), Self::ONE);
        unsafe { arch::_mm512_mask_popcnt_epi32x_v4(value, mask, low) }
    }

    fn trailing_zeros_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let low = Self::sub(Self::bitand(value, Self::sub(Self::ZERO, value)), Self::ONE);
        unsafe { arch::_mm512_mask_popcnt_epi32x_v4(src, mask, low) }
    }

    fn trailing_zeros_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let low = Self::sub(Self::bitand(value, Self::sub(Self::ZERO, value)), Self::ONE);
        unsafe { arch::_mm512_maskz_popcnt_epi32x_v4(mask, low) }
    }

    fn leading_ones_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_lzcnt_epi32(value, mask, Self::not(value)) }
    }

    fn leading_ones_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_lzcnt_epi32(src, mask, Self::not(value)) }
    }

    fn leading_ones_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_lzcnt_epi32(mask, Self::not(value)) }
    }

    fn trailing_ones_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let inv = Self::not(value);
        let low = Self::sub(Self::bitand(inv, Self::sub(Self::ZERO, inv)), Self::ONE);
        unsafe { arch::_mm512_mask_popcnt_epi32x_v4(value, mask, low) }
    }

    fn trailing_ones_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let inv = Self::not(value);
        let low = Self::sub(Self::bitand(inv, Self::sub(Self::ZERO, inv)), Self::ONE);
        unsafe { arch::_mm512_mask_popcnt_epi32x_v4(src, mask, low) }
    }

    fn trailing_ones_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let inv = Self::not(value);
        let low = Self::sub(Self::bitand(inv, Self::sub(Self::ZERO, inv)), Self::ONE);
        unsafe { arch::_mm512_maskz_popcnt_epi32x_v4(mask, low) }
    }
}

#[thermite_macros::inline_always]
impl UnsignedIntegerRegister for U32x16V4 {
    /// `32 - lzcnt`: the v3 dependency inverted, now that `vplzcntd` is a
    /// single floor instruction instead of a popcount cascade.
    fn ilog2p1(value: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::splat(32), Self::leading_zeros(value))
    }

    /// `|a - b| = max(a, b) - min(a, b)`: three instructions with native
    /// unsigned min/max, versus the default's two saturating subtracts (each
    /// already two ops here) plus an OR.
    fn abs_diff(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::max(a, b), Self::min(a, b))
    }

    // --- masked variants -----------------------------------------------------

    // ilog2p1, avg and abs_diff all end in a subtract, which takes the mask.
    fn ilog2p1_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_epi32(value, mask, Self::splat(32), Self::leading_zeros(value)) }
    }

    fn ilog2p1_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_epi32(src, mask, Self::splat(32), Self::leading_zeros(value)) }
    }

    fn ilog2p1_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_sub_epi32(mask, Self::splat(32), Self::leading_zeros(value)) }
    }

    fn avg_c(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_epi32(a, mask, Self::bitor(a, b), Self::shri::<1>(Self::bitxor(a, b))) }
    }

    fn avg_m(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_epi32(src, mask, Self::bitor(a, b), Self::shri::<1>(Self::bitxor(a, b))) }
    }

    fn avg_z(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_sub_epi32(mask, Self::bitor(a, b), Self::shri::<1>(Self::bitxor(a, b))) }
    }

    fn abs_diff_c(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_epi32(a, mask, Self::max(a, b), Self::min(a, b)) }
    }

    fn abs_diff_m(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_epi32(src, mask, Self::max(a, b), Self::min(a, b)) }
    }

    fn abs_diff_z(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_sub_epi32(mask, Self::max(a, b), Self::min(a, b)) }
    }

    // The OR-cascade and the parity fold have no masked final op beyond an
    // AND with ONE / an OR. One merge move after each.
    masked_via_mov_v4! {
        mov = _mm512_mask_mov_epi32, movz = _mm512_maskz_mov_epi32;
        next_power_of_two_m1(value: Storage<Self>);
        parity(value: Storage<Self>);
    }

    /// 2D Morton via the full-width PSHUFB nibble-LUT port. Every other `N`
    /// uses the generic cascade.
    fn morton<const N: usize>(values: [Storage<Self>; N]) -> Storage<Self> {
        if const { N == 2 } {
            unsafe { arch::_mm512_morton2_epu32x_v4(values[0], values[1]) }
        } else {
            crate::backend::generic::polyfills::morton_cascade::<Self, N>(values)
        }
    }

    /// 2D Morton decode via the full-width PSHUFB compress port. Other `N`
    /// use the cascade.
    fn reverse_morton<const N: usize>(code: Storage<Self>) -> [Storage<Self>; N] {
        if const { N == 2 } {
            let odd = Self::shri::<1>(code);

            unsafe {
                crate::backend::generic::polyfills::morton_pack2::<Self, N>(
                    arch::_mm512_morton2_compress_epu32x_v4(code),
                    arch::_mm512_morton2_compress_epu32x_v4(odd),
                )
            }
        } else {
            crate::backend::generic::polyfills::reverse_morton_cascade::<Self, N>(code)
        }
    }
}

// Widening cast for the cast matrix: u32x16 -> [u64x8; 2].
#[thermite_macros::inline_always]
impl CastRegister<U32x16V4> for ArrayRegister<super::U64x8V4, 2> {
    fn cast_from(value: Storage<U32x16V4>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm512_cvtepu32_epi64(arch::_mm512_castsi512_si256(value));
            let hi = arch::_mm512_cvtepu32_epi64(arch::_mm512_extracti32x8_epi32::<1>(value));
            ArrayRegister([lo, hi])
        }
    }
}
