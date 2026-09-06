//! `u64x8` on AVX-512: the native 512-bit unsigned-qword register, unsigned
//! sibling of `i64x8` (see f32x16.rs / i32x16.rs for the shared house rules).
//!
//! Unsigned-64 notes:
//!
//! - DQ is floor, so the whole 64-bit set that AVX2 lacked is native here:
//!   `vpcmpuq` (all six compares), `vpminuq`/`vpmaxuq`, `vpmullq`. None of the
//!   v3 `*_epu64x_v3` polyfill ladder survives.
//! - `mulhi` is still hand-built: there is no 64x64 -> high-64 multiply at any
//!   AVX-512 tier, so the 32-bit limb decomposition from the v3 divider
//!   polyfill is ported verbatim onto `vpmuludq`.
//! - Tier forks: `count_ones` reads `F::AVX512VPOPCNTDQ` (`vpopcntq`) with an
//!   explicit half-split fallback onto the inherited 256-bit polyfill.
//!   `morton::<2>` reads `F::VPCLMULQDQ` (`vpclmulqdq`, 512-bit carry-less
//!   multiply) and otherwise takes the generic cascade -- the v3 128-bit CLMUL
//!   path is behind the `avx2-pclmul` crate feature and its intrinsic import
//!   is not in scope on builds without it.

use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, BitwiseRegister, CastRegister, CoreRegister, IndexableRegister, IntegerRegister,
        InterleaveRegister, NumericRegister, PartialOrdRegister, Register, Storage, UnsignedIntegerRegister, ZeroUpper,
        array::ArrayRegister, empty_reg, reg,
    },
};

use super::super::{Avx512Features, DefaultAvx512 as F};
use super::arch;
use super::kmask::KMask8;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct U64x8V4;

#[thermite_macros::inline_always]
impl CoreRegister for U64x8V4 {
    type Lanes = typenum::U8;
    type Storage = arch::__m512i;
    type Mask = KMask8;

    const IS_EMULATED: bool = false;

    const EMPTY: Storage<Self> = empty_reg::<Self>();

    const HAS_EQUAL_SIZE_MASK: bool = false;

    fn blendv(mask: Storage<Self::Mask>, on_false: Storage<Self>, on_true: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_blend_epi64(mask, on_false, on_true) }
    }

    fn zz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_mov_epi64(mask, value) }
    }

    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_mov_epi64(!mask, value) }
    }

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        if const { Z::N >= 8 } {
            value
        } else {
            unsafe { arch::_mm512_maskz_mov_epi64(const { ((1u32 << Z::N) - 1) as u8 }, value) }
        }
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        unsafe { arch::_mm512_movm_epi64(mask) }
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for U64x8V4 {
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
        unsafe { arch::_mm512_ternarylogic_epi64::<{ crate::ternlog_imm!(!C) }>(value, value, value) }
    }

    const HAS_NATIVE_TERNLOG: bool = true;

    fn ternlog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_ternarylogic_epi64::<IMM>(a, b, c) }
    }

    fn bilog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        arch::bilog_ternlog::<Self, IMM>(a, b)
    }

    // --- masked variants -----------------------------------------------------

    masked_binary_v4! {
        bitxor => _mm512_mask_xor_epi64, _mm512_maskz_xor_epi64;
        bitand => _mm512_mask_and_epi64, _mm512_maskz_and_epi64;
        bitor => _mm512_mask_or_epi64, _mm512_maskz_or_epi64;
    }

    masked_andnot_v4! {
        bitandnot => _mm512_mask_andnot_epi64, _mm512_maskz_andnot_epi64;
    }

    // vpternlog's merge form keeps operand A (`src` doubles as A), so the
    // NOT imm must read operand C: 0x55.
    fn not_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_ternarylogic_epi64::<{ crate::ternlog_imm!(!C) }>(value, mask, value, value) }
    }

    fn not_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_ternarylogic_epi64::<{ crate::ternlog_imm!(!C) }>(src, mask, value, value) }
    }

    fn not_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_ternarylogic_epi64::<{ crate::ternlog_imm!(!C) }>(mask, value, value, value) }
    }

    fn ternlog_c<const IMM: i32>(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_ternarylogic_epi64::<IMM>(a, mask, b, c) }
    }

    // The merge form's `src` is also operand A, so an arbitrary `src` costs
    // a separate merge move.
    fn ternlog_m<const IMM: i32>(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mov_epi64(src, mask, Self::ternlog::<IMM>(a, b, c)) }
    }

    fn ternlog_z<const IMM: i32>(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_ternarylogic_epi64::<IMM>(mask, a, b, c) }
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

// Width ladder: zmm = two ymm. `vinserti64x4` into a zero-extended low half.
#[thermite_macros::inline_always]
impl crate::register::ConcatRegister<super::U64x4V4> for U64x8V4 {
    fn concat(lo: Storage<super::U64x4V4>, hi: Storage<super::U64x4V4>) -> Storage<Self> {
        unsafe { arch::_mm512_inserti64x4::<1>(arch::_mm512_zextsi256_si512(lo), hi) }
    }

    fn split(value: Storage<Self>) -> (Storage<super::U64x4V4>, Storage<super::U64x4V4>) {
        let lo = unsafe { arch::_mm512_castsi512_si256(value) };
        let hi = unsafe { arch::_mm512_extracti64x4_epi64::<1>(value) };
        (lo, hi)
    }
}

#[thermite_macros::inline_always]
impl crate::register::ExtendRegister<super::U64x4V4> for U64x8V4 {
    fn extend(value: Storage<super::U64x4V4>) -> Storage<Self> {
        unsafe { arch::_mm512_zextsi256_si512(value) }
    }

    fn narrow(value: Storage<Self>) -> Storage<super::U64x4V4> {
        unsafe { arch::_mm512_castsi512_si256(value) }
    }
}

#[thermite_macros::inline_always]
impl Register for U64x8V4 {
    type Element = u64;

    type Signed = super::I64x8V4;
    type Unsigned = U64x8V4;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_test_epi64_mask(value, value) }
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::msb_to_mask(value)
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_movepi64_mask(value) }
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm512_loadu_si512(value.as_ptr() as *const _) }
    }

    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_zextsi128_si512(arch::_mm_cvtsi64x_si128(value as i64)) }
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_set1_epi64(value as i64) }
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_load_si512(ptr as *const _) }
    }

    unsafe fn load_m(src: Storage<Self>, mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_mask_loadu_epi64(src, mask, ptr as *const _) }
    }

    unsafe fn load_z(mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_loadu_epi64(mask, ptr as *const _) }
    }

    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_loadu_si512(ptr as *const _) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm512_store_si512(ptr as *mut _, value) }
    }

    unsafe fn store_masked(ptr: *mut Self::Element, mask: Storage<Self::Mask>, value: Storage<Self>) {
        unsafe { arch::_mm512_mask_storeu_epi64(ptr as *mut _, mask, value) }
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

            unsafe { arch::_mm512_permutexvar_epi64(indices, Self::new(padded)) }
        } else {
            unsafe { <Self as IndexableRegister<Self>>::gather(values.as_ptr(), indices) }
        }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        unsafe {
            let idx = arch::_mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0);
            arch::_mm512_permutexvar_epi64(idx, value)
        }
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        unsafe {
            let pattern = arch::_mm512_broadcast_i32x4(arch::_mm_setr_epi8(
                7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8,
            ));
            arch::_mm512_shuffle_epi8(value, pattern)
        }
    }

    compress_expand_v4!(
        u8,
        _mm512_maskz_compress_epi64,
        _mm512_mask_expand_epi64,
        _mm512_maskz_expand_epi64
    );

    const HAS_PERMUTEV: bool = true;

    fn permutev(value: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        // The index register already holds qword lanes: vpermq direct.
        unsafe { arch::_mm512_permutexvar_epi64(idxs, value) }
    }

    fn swizzle(a: Storage<Self>, b: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm512_permutex2var_epi64(a, idxs, b) }
    }

    // --- masked variants -----------------------------------------------------

    fn new_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: GenericArray<Self::Element, Self::Lanes>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_loadu_epi64(src, mask, value.as_ptr() as *const _) }
    }

    fn new_z(mask: Storage<Self::Mask>, value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_loadu_epi64(mask, value.as_ptr() as *const _) }
    }

    // vpbroadcastq from a GPR takes the mask directly.
    fn splat_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_mask_set1_epi64(src, mask, value as i64) }
    }

    fn splat_z(mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_set1_epi64(mask, value as i64) }
    }

    // Lane broadcast: one vpermq with a constant index, instead of the
    // default's extract-to-scalar round trip. The masked forms ride the
    // same instruction.
    fn broadcast<const I: usize>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_permutexvar_epi64(arch::_mm512_set1_epi64(I as i64), value) }
    }

    fn broadcast_c<const I: usize>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_permutexvar_epi64(value, mask, arch::_mm512_set1_epi64(I as i64), value) }
    }

    fn broadcast_m<const I: usize>(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_permutexvar_epi64(src, mask, arch::_mm512_set1_epi64(I as i64), value) }
    }

    fn broadcast_z<const I: usize>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_permutexvar_epi64(mask, arch::_mm512_set1_epi64(I as i64), value) }
    }

    fn broadcastv(value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm512_permutexvar_epi64(arch::_mm512_set1_epi64(idx as i64), value) }
    }

    fn broadcastv_c(mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm512_mask_permutexvar_epi64(value, mask, arch::_mm512_set1_epi64(idx as i64), value) }
    }

    fn broadcastv_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm512_mask_permutexvar_epi64(src, mask, arch::_mm512_set1_epi64(idx as i64), value) }
    }

    fn broadcastv_z(mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_permutexvar_epi64(mask, arch::_mm512_set1_epi64(idx as i64), value) }
    }

    fn reverse_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe {
            let idx = arch::_mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0);
            arch::_mm512_mask_permutexvar_epi64(value, mask, idx, value)
        }
    }

    fn reverse_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe {
            let idx = arch::_mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0);
            arch::_mm512_mask_permutexvar_epi64(src, mask, idx, value)
        }
    }

    fn reverse_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe {
            let idx = arch::_mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0);
            arch::_mm512_maskz_permutexvar_epi64(mask, idx, value)
        }
    }

    // vpshufb is masked per BYTE (`__mmask64`), so the lane mask cannot feed
    // it directly: shuffle, then one merge move.
    masked_via_mov_v4! {
        mov = _mm512_mask_mov_epi64, movz = _mm512_maskz_mov_epi64;
        swap_bytes(value: Storage<Self>);
    }

    fn permutev_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_permutexvar_epi64(src, mask, idxs, value) }
    }

    fn permutev_z(mask: Storage<Self::Mask>, value: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_permutexvar_epi64(mask, idxs, value) }
    }

    // vpermt2q merges only into `a` (vpermi2q into the index), neither of
    // which is an arbitrary `src`: permute, then one merge move. The zeroing
    // form is native.
    fn swizzle_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        a: Storage<Self>,
        b: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mov_epi64(src, mask, arch::_mm512_permutex2var_epi64(a, idxs, b)) }
    }

    fn swizzle_z(
        mask: Storage<Self::Mask>,
        a: Storage<Self>,
        b: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_permutex2var_epi64(mask, a, idxs, b) }
    }
}

#[thermite_macros::inline_always]
impl InterleaveRegister for U64x8V4 {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe {
            let lo = arch::_mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
            let hi = arch::_mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
            (
                arch::_mm512_permutex2var_epi64(a, lo, b),
                arch::_mm512_permutex2var_epi64(a, hi, b),
            )
        }
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe {
            let even = arch::_mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
            let odd = arch::_mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
            (
                arch::_mm512_permutex2var_epi64(a, even, b),
                arch::_mm512_permutex2var_epi64(a, odd, b),
            )
        }
    }
}

#[thermite_macros::inline_always]
impl IndexableRegister<U64x8V4> for U64x8V4 {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<U64x8V4>) -> Storage<Self> {
        unsafe { arch::_mm512_i64gather_epi64::<8>(indices, ptr as _) }
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<U64x8V4>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_i64gather_epi64::<8>(src, mask, indices, ptr as _) }
    }

    unsafe fn scatter(value: Storage<Self>, ptr: *mut Self::Element, indices: Storage<U64x8V4>) {
        unsafe { arch::_mm512_i64scatter_epi64::<8>(ptr as _, indices, value) }
    }

    unsafe fn scatter_m(
        value: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *mut Self::Element,
        indices: Storage<U64x8V4>,
    ) {
        unsafe { arch::_mm512_mask_i64scatter_epi64::<8>(ptr as _, mask, indices, value) }
    }
}

// 16 dword indices feeding the 16-lane `[u64x8; 2]` grid slot: two
// `vpgatherqq`-class gathers off the two halves of the index register.
#[thermite_macros::inline_always]
impl IndexableRegister<super::U32x16V4> for ArrayRegister<U64x8V4, 2> {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<super::U32x16V4>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm512_i32gather_epi64::<8>(arch::_mm512_castsi512_si256(indices), ptr as _);
            let hi = arch::_mm512_i32gather_epi64::<8>(arch::_mm512_extracti32x8_epi32::<1>(indices), ptr as _);

            ArrayRegister([lo, hi])
        }
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<super::U32x16V4>,
    ) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm512_mask_i32gather_epi64::<8>(
                src.0[0],
                mask.0[0],
                arch::_mm512_castsi512_si256(indices),
                ptr as _,
            );
            let hi = arch::_mm512_mask_i32gather_epi64::<8>(
                src.0[1],
                mask.0[1],
                arch::_mm512_extracti32x8_epi32::<1>(indices),
                ptr as _,
            );

            ArrayRegister([lo, hi])
        }
    }

    unsafe fn scatter(value: Storage<Self>, ptr: *mut Self::Element, indices: Storage<super::U32x16V4>) {
        unsafe {
            arch::_mm512_i32scatter_epi64::<8>(ptr as _, arch::_mm512_castsi512_si256(indices), value.0[0]);
            arch::_mm512_i32scatter_epi64::<8>(ptr as _, arch::_mm512_extracti32x8_epi32::<1>(indices), value.0[1]);
        }
    }

    unsafe fn scatter_m(
        value: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *mut Self::Element,
        indices: Storage<super::U32x16V4>,
    ) {
        unsafe {
            arch::_mm512_mask_i32scatter_epi64::<8>(
                ptr as _,
                mask.0[0],
                arch::_mm512_castsi512_si256(indices),
                value.0[0],
            );
            arch::_mm512_mask_i32scatter_epi64::<8>(
                ptr as _,
                mask.0[1],
                arch::_mm512_extracti32x8_epi32::<1>(indices),
                value.0[1],
            );
        }
    }
}

// Eight 32-bit indices are one ymm (`u32x8`): vpgatherdq / vpscatterdq.
#[thermite_macros::inline_always]
impl IndexableRegister<super::U32x8V4> for U64x8V4 {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<super::U32x8V4>) -> Storage<Self> {
        unsafe { arch::_mm512_i32gather_epi64::<8>(indices, ptr as _) }
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<super::U32x8V4>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_i32gather_epi64::<8>(src, mask, indices, ptr as _) }
    }

    unsafe fn scatter(value: Storage<Self>, ptr: *mut Self::Element, indices: Storage<super::U32x8V4>) {
        unsafe { arch::_mm512_i32scatter_epi64::<8>(ptr as _, indices, value) }
    }

    unsafe fn scatter_m(
        value: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *mut Self::Element,
        indices: Storage<super::U32x8V4>,
    ) {
        unsafe { arch::_mm512_mask_i32scatter_epi64::<8>(ptr as _, mask, indices, value) }
    }
}

#[thermite_macros::inline_always]
impl BitshiftRegister for U64x8V4 {
    const HAS_TRUE_SHIFTV: bool = true;
    const HAS_WIDE_BYTE_SHIFTS: bool = false;

    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm512_sll_epi64(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm512_srl_epi64(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    fn shlv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm512_sllv_epi64(value, shifts) }
    }

    fn shrv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm512_srlv_epi64(value, shifts) }
    }

    fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_sll_epi64(value, arch::_mm_cvtsi32_si128(IMM8)) }
    }

    fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_srl_epi64(value, arch::_mm_cvtsi32_si128(IMM8)) }
    }

    // Rotates are native (vprolvq/vprorvq). The count is broadcast rather
    // than immediate for the same `IMM8 as u32` reason as the shifts. LLVM
    // folds a constant count to the vprolq/vprorq immediate form.
    fn rol(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm512_rolv_epi64(value, arch::_mm512_set1_epi64(shift as i64)) }
    }

    fn ror(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm512_rorv_epi64(value, arch::_mm512_set1_epi64(shift as i64)) }
    }

    fn roli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_rolv_epi64(value, arch::_mm512_set1_epi64(IMM8 as i64)) }
    }

    fn rori<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_rorv_epi64(value, arch::_mm512_set1_epi64(IMM8 as i64)) }
    }

    fn rolv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm512_rolv_epi64(value, shifts) }
    }

    fn rorv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm512_rorv_epi64(value, shifts) }
    }

    // --- masked variants -----------------------------------------------------

    masked_shift_v4! {
        shl => _mm512_mask_sll_epi64, _mm512_maskz_sll_epi64;
        shr => _mm512_mask_srl_epi64, _mm512_maskz_srl_epi64;
    }

    masked_shifti_v4! {
        shli => _mm512_mask_sll_epi64, _mm512_maskz_sll_epi64;
        shri => _mm512_mask_srl_epi64, _mm512_maskz_srl_epi64;
    }

    masked_binary_v4! {
        shlv(Storage<Self::Unsigned>) => _mm512_mask_sllv_epi64, _mm512_maskz_sllv_epi64;
        shrv(Storage<Self::Unsigned>) => _mm512_mask_srlv_epi64, _mm512_maskz_srlv_epi64;
        rolv(Storage<Self::Unsigned>) => _mm512_mask_rolv_epi64, _mm512_maskz_rolv_epi64;
        rorv(Storage<Self::Unsigned>) => _mm512_mask_rorv_epi64, _mm512_maskz_rorv_epi64;
    }

    fn rol_c(mask: Storage<Self::Mask>, value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm512_mask_rolv_epi64(value, mask, value, arch::_mm512_set1_epi64(shift as i64)) }
    }

    fn rol_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm512_mask_rolv_epi64(src, mask, value, arch::_mm512_set1_epi64(shift as i64)) }
    }

    fn rol_z(mask: Storage<Self::Mask>, value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_rolv_epi64(mask, value, arch::_mm512_set1_epi64(shift as i64)) }
    }

    fn ror_c(mask: Storage<Self::Mask>, value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm512_mask_rorv_epi64(value, mask, value, arch::_mm512_set1_epi64(shift as i64)) }
    }

    fn ror_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm512_mask_rorv_epi64(src, mask, value, arch::_mm512_set1_epi64(shift as i64)) }
    }

    fn ror_z(mask: Storage<Self::Mask>, value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_rorv_epi64(mask, value, arch::_mm512_set1_epi64(shift as i64)) }
    }

    fn roli_c<const IMM8: i32>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_rolv_epi64(value, mask, value, arch::_mm512_set1_epi64(IMM8 as i64)) }
    }

    fn roli_m<const IMM8: i32>(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_rolv_epi64(src, mask, value, arch::_mm512_set1_epi64(IMM8 as i64)) }
    }

    fn roli_z<const IMM8: i32>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_rolv_epi64(mask, value, arch::_mm512_set1_epi64(IMM8 as i64)) }
    }

    fn rori_c<const IMM8: i32>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_rorv_epi64(value, mask, value, arch::_mm512_set1_epi64(IMM8 as i64)) }
    }

    fn rori_m<const IMM8: i32>(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_rorv_epi64(src, mask, value, arch::_mm512_set1_epi64(IMM8 as i64)) }
    }

    fn rori_z<const IMM8: i32>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_rorv_epi64(mask, value, arch::_mm512_set1_epi64(IMM8 as i64)) }
    }

    // Whole-register byte shifts and the bit-reverse cascade have no masked
    // final step (the trait defaults end in a lane-wise loop / an OR of two
    // shifted halves): run them, then one merge move.
    masked_via_mov_v4! {
        mov = _mm512_mask_mov_epi64, movz = _mm512_maskz_mov_epi64;
        bshli<const IMM8: i32>(value: Storage<Self>);
        bshri<const IMM8: i32>(value: Storage<Self>);
        reverse_bits(value: Storage<Self>);
    }
}

#[thermite_macros::inline_always]
impl PartialOrdRegister for U64x8V4 {
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmpgt_epu64_mask(lhs, rhs) }
    }

    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmpge_epu64_mask(lhs, rhs) }
    }

    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmplt_epu64_mask(lhs, rhs) }
    }

    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmple_epu64_mask(lhs, rhs) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmpeq_epu64_mask(lhs, rhs) }
    }

    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmpneq_epu64_mask(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl NumericRegister for U64x8V4 {
    const ZERO: Storage<Self> = reg::<Self, 8>([0; 8]);
    const ONE: Storage<Self> = reg::<Self, 8>([1; 8]);
    const TWO: Storage<Self> = reg::<Self, 8>([2; 8]);

    const MIN: Storage<Self> = reg::<Self, 8>([u64::MIN; 8]);
    const MAX: Storage<Self> = reg::<Self, 8>([u64::MAX; 8]);

    fn min_element(value: Storage<Self>) -> Self::Element {
        unsafe { arch::_mm512_reduce_min_epu64(value) }
    }

    fn max_element(value: Storage<Self>) -> Self::Element {
        unsafe { arch::_mm512_reduce_max_epu64(value) }
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element {
        unsafe { arch::_mm512_reduce_add_epi64(value) as u64 }
    }

    fn prod_elements(value: Storage<Self>) -> Self::Element {
        unsafe { arch::_mm512_reduce_mul_epi64(value) as u64 }
    }

    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as Unsigned>::U64)
    }

    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as u64))
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_add_epi64(lhs, rhs) }
    }

    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_sub_epi64(lhs, rhs) }
    }

    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mullo_epi64(lhs, rhs) }
    }

    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a / b })
    }

    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a % b })
    }

    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_min_epu64(lhs, rhs) }
    }

    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_max_epu64(lhs, rhs) }
    }

    // --- masked variants -----------------------------------------------------
    // Embedded-mask forms: keep `lhs`/`src` where the mask is false, one
    // instruction, no blend.

    masked_binary_v4! {
        add => _mm512_mask_add_epi64, _mm512_maskz_add_epi64;
        sub => _mm512_mask_sub_epi64, _mm512_maskz_sub_epi64;
        mul => _mm512_mask_mullo_epi64, _mm512_maskz_mullo_epi64;
        min => _mm512_mask_min_epu64, _mm512_maskz_min_epu64;
        max => _mm512_mask_max_epu64, _mm512_maskz_max_epu64;
    }

    // Integer division is lane-wise scalar. Nothing to mask but the result.
    masked_via_mov_v4! {
        mov = _mm512_mask_mov_epi64, movz = _mm512_maskz_mov_epi64;
        div(lhs: Storage<Self>, rhs: Storage<Self>);
        rem(lhs: Storage<Self>, rhs: Storage<Self>);
    }

    fn square_c(mask: Storage<Self::Mask>, lhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mullo_epi64(lhs, mask, lhs, lhs) }
    }

    fn square_m(src: Storage<Self>, mask: Storage<Self::Mask>, lhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mullo_epi64(src, mask, lhs, lhs) }
    }

    fn square_z(mask: Storage<Self::Mask>, lhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_mullo_epi64(mask, lhs, lhs) }
    }

    fn scale_c(mask: Storage<Self::Mask>, value: Storage<Self>, scalar: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mullo_epi64(value, mask, value, Self::splat(scalar)) }
    }

    fn scale_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
        scalar: Self::Element,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mullo_epi64(src, mask, value, Self::splat(scalar)) }
    }

    fn scale_z(mask: Storage<Self::Mask>, value: Storage<Self>, scalar: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_mullo_epi64(mask, value, Self::splat(scalar)) }
    }
}

#[thermite_macros::inline_always]
impl IntegerRegister for U64x8V4 {
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        // No 64x64 -> high-64 multiply exists at any AVX-512 tier. Same 32-bit
        // limb decomposition as the v3 divider polyfill
        // (`_mm256_mullhi_epu64x_v3`), widened to zmm: four `vpmuludq` plus the
        // carry chain.
        unsafe {
            let lomask = arch::_mm512_set1_epi64(0xFFFF_FFFF);
            let xh = arch::_mm512_shuffle_epi32::<0b10_11_00_01>(lhs);
            let yh = arch::_mm512_shuffle_epi32::<0b10_11_00_01>(rhs);

            let w0 = arch::_mm512_mul_epu32(lhs, rhs);
            let w1 = arch::_mm512_mul_epu32(lhs, yh);
            let w2 = arch::_mm512_mul_epu32(xh, rhs);
            let w3 = arch::_mm512_mul_epu32(xh, yh);

            let w0h = arch::_mm512_srli_epi64::<32>(w0);
            let s1 = arch::_mm512_add_epi64(w1, w0h);
            let s1l = arch::_mm512_and_si512(s1, lomask);
            let s1h = arch::_mm512_srli_epi64::<32>(s1);
            let s2 = arch::_mm512_add_epi64(w2, s1l);
            let s2h = arch::_mm512_srli_epi64::<32>(s2);

            arch::_mm512_add_epi64(arch::_mm512_add_epi64(w3, s1h), s2h)
        }
    }

    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mullo_epi64(lhs, rhs) }
    }

    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        // a +| b = a + min(b, MAX - a) = a + min(b, !a): vpternlog + vpminuq +
        // vpaddq, no mask, and the final add is what the masked variants hook.
        Self::add(lhs, Self::min(rhs, Self::not(lhs)))
    }

    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        // Two instructions with a native `vpminuq`/`vpmaxuq`, and no mask or
        // constant at all: max(l, r) - r is 0 exactly where l < r.
        Self::sub(Self::max(lhs, rhs), rhs)
    }

    fn wrapping_sum(value: Storage<Self>) -> Self::Element {
        unsafe { arch::_mm512_reduce_add_epi64(value) as u64 }
    }

    fn wrapping_product(value: Storage<Self>) -> Self::Element {
        unsafe { arch::_mm512_reduce_mul_epi64(value) as u64 }
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
        Self::count_ones(unsafe { arch::_mm512_conflict_epi64(value) })
    }

    fn count_ones(value: Storage<Self>) -> Storage<Self> {
        if const { <F as Avx512Features>::AVX512VPOPCNTDQ } {
            unsafe { arch::_mm512_popcnt_epi64(value) }
        } else {
            // Floor fallback: full-width nibble-LUT port (vpsadbw sums the
            // byte counts straight into qword lanes).
            unsafe { arch::_mm512_popcnt_epi64x_v4(value) }
        }
    }

    fn count_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(Self::not(value))
    }

    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        // vplzcntq is CD, i.e. floor: single instruction, correct for 0 (-> 64).
        unsafe { arch::_mm512_lzcnt_epi64(value) }
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
        mullo => _mm512_mask_mullo_epi64, _mm512_maskz_mullo_epi64;
    }

    // mulhi's carry chain and the divider polyfills have no spare mask slot:
    // compute, then one merge move.
    masked_via_mov_v4! {
        mov = _mm512_mask_mov_epi64, movz = _mm512_maskz_mov_epi64;
        mulhi(lhs: Storage<Self>, rhs: Storage<Self>);
        div_branched(value: Storage<Self>, divider: crate::Divider<Self::Element>);
        div_branchfree(value: Storage<Self>, divider: crate::BranchfreeDivider<Self::Element>);
        divv_branchfree(value: Storage<Self>, dividers: crate::divider::vector::VectorDivider<Self>);
    }

    // Saturating add/sub end in a plain add/sub (see the base bodies), which
    // takes the mask.
    fn saturating_add_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_add_epi64(lhs, mask, lhs, Self::min(rhs, Self::not(lhs))) }
    }

    fn saturating_add_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        lhs: Storage<Self>,
        rhs: Storage<Self>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_add_epi64(src, mask, lhs, Self::min(rhs, Self::not(lhs))) }
    }

    fn saturating_add_z(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_add_epi64(mask, lhs, Self::min(rhs, Self::not(lhs))) }
    }

    fn saturating_sub_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_epi64(lhs, mask, Self::max(lhs, rhs), rhs) }
    }

    fn saturating_sub_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        lhs: Storage<Self>,
        rhs: Storage<Self>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_epi64(src, mask, Self::max(lhs, rhs), rhs) }
    }

    fn saturating_sub_z(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_sub_epi64(mask, Self::max(lhs, rhs), rhs) }
    }

    // The popcount family ends in `count_ones`, whose masked polyfill forks
    // on VPOPCNTDQ exactly like the unmasked one (`bits.rs`), and lzcnt is CD.
    masked_unary_v4! {
        count_ones => _mm512_mask_popcnt_epi64x_v4, _mm512_maskz_popcnt_epi64x_v4;
        leading_zeros => _mm512_mask_lzcnt_epi64, _mm512_maskz_lzcnt_epi64;
    }

    fn count_zeros_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_popcnt_epi64x_v4(value, mask, Self::not(value)) }
    }

    fn count_zeros_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_popcnt_epi64x_v4(src, mask, Self::not(value)) }
    }

    fn count_zeros_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_popcnt_epi64x_v4(mask, Self::not(value)) }
    }

    fn trailing_zeros_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let low = Self::sub(Self::bitand(value, Self::sub(Self::ZERO, value)), Self::ONE);
        unsafe { arch::_mm512_mask_popcnt_epi64x_v4(value, mask, low) }
    }

    fn trailing_zeros_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let low = Self::sub(Self::bitand(value, Self::sub(Self::ZERO, value)), Self::ONE);
        unsafe { arch::_mm512_mask_popcnt_epi64x_v4(src, mask, low) }
    }

    fn trailing_zeros_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let low = Self::sub(Self::bitand(value, Self::sub(Self::ZERO, value)), Self::ONE);
        unsafe { arch::_mm512_maskz_popcnt_epi64x_v4(mask, low) }
    }

    fn leading_ones_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_lzcnt_epi64(value, mask, Self::not(value)) }
    }

    fn leading_ones_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_lzcnt_epi64(src, mask, Self::not(value)) }
    }

    fn leading_ones_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_lzcnt_epi64(mask, Self::not(value)) }
    }

    fn trailing_ones_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let inv = Self::not(value);
        let low = Self::sub(Self::bitand(inv, Self::sub(Self::ZERO, inv)), Self::ONE);
        unsafe { arch::_mm512_mask_popcnt_epi64x_v4(value, mask, low) }
    }

    fn trailing_ones_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let inv = Self::not(value);
        let low = Self::sub(Self::bitand(inv, Self::sub(Self::ZERO, inv)), Self::ONE);
        unsafe { arch::_mm512_mask_popcnt_epi64x_v4(src, mask, low) }
    }

    fn trailing_ones_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let inv = Self::not(value);
        let low = Self::sub(Self::bitand(inv, Self::sub(Self::ZERO, inv)), Self::ONE);
        unsafe { arch::_mm512_maskz_popcnt_epi64x_v4(mask, low) }
    }
}

#[thermite_macros::inline_always]
impl UnsignedIntegerRegister for U64x8V4 {
    /// `64 - lzcnt`, now that `vplzcntq` is a single floor instruction.
    fn ilog2p1(value: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::splat(64), Self::leading_zeros(value))
    }

    /// `|a - b| = max(a, b) - min(a, b)`: three instructions with native
    /// unsigned min/max, versus the default's two saturating subtracts plus
    /// an OR.
    fn abs_diff(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        Self::sub(Self::max(a, b), Self::min(a, b))
    }

    // --- masked variants -----------------------------------------------------

    // ilog2p1, avg and abs_diff all end in a subtract, which takes the mask.
    fn ilog2p1_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_epi64(value, mask, Self::splat(64), Self::leading_zeros(value)) }
    }

    fn ilog2p1_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_epi64(src, mask, Self::splat(64), Self::leading_zeros(value)) }
    }

    fn ilog2p1_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_sub_epi64(mask, Self::splat(64), Self::leading_zeros(value)) }
    }

    fn avg_c(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_epi64(a, mask, Self::bitor(a, b), Self::shri::<1>(Self::bitxor(a, b))) }
    }

    fn avg_m(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_epi64(src, mask, Self::bitor(a, b), Self::shri::<1>(Self::bitxor(a, b))) }
    }

    fn avg_z(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_sub_epi64(mask, Self::bitor(a, b), Self::shri::<1>(Self::bitxor(a, b))) }
    }

    fn abs_diff_c(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_epi64(a, mask, Self::max(a, b), Self::min(a, b)) }
    }

    fn abs_diff_m(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_epi64(src, mask, Self::max(a, b), Self::min(a, b)) }
    }

    fn abs_diff_z(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_sub_epi64(mask, Self::max(a, b), Self::min(a, b)) }
    }

    // The OR-cascade and the parity fold have no masked final op beyond an
    // AND with ONE / an OR. One merge move after each.
    masked_via_mov_v4! {
        mov = _mm512_mask_mov_epi64, movz = _mm512_maskz_mov_epi64;
        next_power_of_two_m1(value: Storage<Self>);
        parity(value: Storage<Self>);
    }

    /// 2D Morton via carry-less multiply (each 64-bit lane interleaves two
    /// 32-bit coords). Every other `N` delegates to the generic shift/mask
    /// cascade.
    ///
    /// The CLMUL path needs `vpclmulqdq` (tier 2) for a 512-bit `vpclmulqdq`.
    /// The v3 128-bit fallback is not reachable from here: its intrinsic import
    /// (`_mm_clmulepi64_si128`) sits behind the `avx2-pclmul` crate feature, so
    /// the else arm takes the generic cascade rather than a half-split.
    fn morton<const N: usize>(values: [Storage<Self>; N]) -> Storage<Self> {
        if const { N == 2 && <F as Avx512Features>::VPCLMULQDQ } {
            unsafe {
                // Spread the low 32 bits of each lane by one (bit i -> bit 2i)
                // via carry-less self-multiply. `vpclmulqdq` is per-128-bit
                // lane, so imm 0x00/0x11 pick the low/high qword of each and
                // `vpunpcklqdq` re-packs the two 128-bit products.
                let lomask = arch::_mm512_set1_epi64(0xFFFF_FFFF);

                let x = arch::_mm512_and_si512(values[0], lomask);
                let y = arch::_mm512_and_si512(values[1], lomask);

                let xs = arch::_mm512_unpacklo_epi64(
                    arch::_mm512_clmulepi64_epi128::<0x00>(x, x),
                    arch::_mm512_clmulepi64_epi128::<0x11>(x, x),
                );
                let ys = arch::_mm512_unpacklo_epi64(
                    arch::_mm512_clmulepi64_epi128::<0x00>(y, y),
                    arch::_mm512_clmulepi64_epi128::<0x11>(y, y),
                );

                arch::_mm512_or_si512(xs, arch::_mm512_slli_epi64::<1>(ys))
            }
        } else {
            crate::backend::generic::polyfills::morton_cascade::<Self, N>(values)
        }
    }
}

// Narrowing cast for the cast matrix: [u64x8; 2] -> u32x16.
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<U64x8V4, 2>> for super::U32x16V4 {
    fn cast_from(value: Storage<ArrayRegister<U64x8V4, 2>>) -> Storage<Self> {
        let ArrayRegister([lo, hi]) = value;

        unsafe {
            let lo = arch::_mm512_cvtepi64_epi32(lo);
            let hi = arch::_mm512_cvtepi64_epi32(hi);

            arch::_mm512_inserti32x8::<1>(arch::_mm512_zextsi256_si512(lo), hi)
        }
    }

    // Unlike AVX2, there is a real unsigned saturating narrow here: no
    // clamp-then-truncate dance, just `vpmovusqd`.
    fn saturating_cast_from(value: Storage<ArrayRegister<U64x8V4, 2>>) -> Storage<Self> {
        let ArrayRegister([lo, hi]) = value;

        unsafe {
            let lo = arch::_mm512_cvtusepi64_epi32(lo);
            let hi = arch::_mm512_cvtusepi64_epi32(hi);

            arch::_mm512_inserti32x8::<1>(arch::_mm512_zextsi256_si512(lo), hi)
        }
    }
}
