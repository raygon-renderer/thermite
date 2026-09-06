//! `f32x16` on AVX-512: the native 512-bit float register, and the pattern
//! file for the v4 register style.
//!
//! v4 house style (vs the v3 files these are derived from):
//!
//! - **No `MaskRegister` impl**: `type Mask = KMask16` and the opmask type
//!   (kmask.rs) already implements the whole mask surface. The ~50 lines of
//!   per-file mask code in v3 have no v4 analogue.
//! - **Compares return opmasks natively** (`_mm512_cmp_ps_mask`), `blendv` is
//!   `mask_blend`, and EVERY `_c`/`_m`/`_z` variant is written out
//!   explicitly (the `masked_*_v4!` stampers in `x86_v4/macros.rs`, or by
//!   hand where the operand order matters) with the embedded-mask
//!   instruction instead of the derived op-then-blendv default. Composed
//!   ops push the mask into their final instruction. The few whose final
//!   step has no masked form (`vpshufb`'s byte mask, the two-source
//!   permute's fixed merge target, the next_up polyfill) say so and take
//!   one merge move.
//! - **Tier forks are `if const { F::FEATURE }`** with an explicit else arm
//!   (`use super::super::DefaultAvx512 as F` where needed). This file needs
//!   none: everything here is floor (F/CD/BW/DQ/VL).
//! - Defaults are acceptable where v3 needed bespoke code (`extract`,
//!   `align`, the radix/transpose ladders): the M4 sweep upgrades them with
//!   native encodings (`valignd`, `vpermt2ps` ladders) behind diff coverage.
//!   Do not set capability flags (`HAS_NATIVE_ALIGN`, ...) until the
//!   override lands.

use generic_array::{GenericArray, sequence::GenericSequence, typenum::Unsigned};

use crate::register::{
    BitwiseRegister, CastRegister, CoreRegister, FloatRegister, IndexableRegister, InterleaveRegister,
    NativeCapability, NumericRegister, PartialOrdRegister, Register, ShuffleRegister, SignedRegister, Storage,
    ZeroUpper, array::ArrayRegister, empty_reg, reg,
};

use super::arch;
use super::kmask::KMask16;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct F32x16V4;

#[thermite_macros::inline_always]
impl CoreRegister for F32x16V4 {
    type Lanes = generic_array::typenum::U16;
    type Storage = arch::__m512;
    type Mask = KMask16;

    const IS_EMULATED: bool = false;

    const EMPTY: Storage<Self> = empty_reg::<Self>();

    const HAS_EQUAL_SIZE_MASK: bool = false;

    fn blendv(mask: Storage<Self::Mask>, on_false: Storage<Self>, on_true: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_blend_ps(mask, on_false, on_true) }
    }

    fn zz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_mov_ps(mask, value) }
    }

    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_mov_ps(!mask, value) }
    }

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        if const { Z::N >= 16 } {
            value
        } else {
            // Zero-masked move with a constant prefix mask: one instruction
            // for every N, no special 128/256 cases needed.
            unsafe { arch::_mm512_maskz_mov_ps(const { ((1u32 << Z::N) - 1) as u16 }, value) }
        }
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        // vpmovm2d: opmask -> all-ones/all-zeros dword lanes (DQ, floor).
        unsafe { arch::_mm512_castsi512_ps(arch::_mm512_movm_epi32(mask)) }
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for F32x16V4 {
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_xor_ps(lhs, rhs) }
    }

    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_and_ps(lhs, rhs) }
    }

    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_andnot_ps(rhs, lhs) }
    }

    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_or_ps(lhs, rhs) }
    }

    fn not(value: Storage<Self>) -> Storage<Self> {
        // vpternlog imm 0x55 = NOT(a): one instruction, no constant load.
        unsafe {
            arch::_mm512_castsi512_ps(arch::_mm512_ternarylogic_epi32::<{ crate::ternlog_imm!(!C) }>(
                arch::_mm512_castps_si512(value),
                arch::_mm512_castps_si512(value),
                arch::_mm512_castps_si512(value),
            ))
        }
    }

    const HAS_NATIVE_TERNLOG: bool = true;

    fn ternlog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe {
            arch::_mm512_castsi512_ps(arch::_mm512_ternarylogic_epi32::<IMM>(
                arch::_mm512_castps_si512(a),
                arch::_mm512_castps_si512(b),
                arch::_mm512_castps_si512(c),
            ))
        }
    }

    fn bilog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        arch::bilog_ternlog::<Self, IMM>(a, b)
    }

    // --- masked variants -----------------------------------------------------
    // DQ (floor) has the float-typed forms of the bitwise ops, so no casts.

    masked_binary_v4! {
        bitxor => _mm512_mask_xor_ps, _mm512_maskz_xor_ps;
        bitand => _mm512_mask_and_ps, _mm512_maskz_and_ps;
        bitor => _mm512_mask_or_ps, _mm512_maskz_or_ps;
    }

    masked_andnot_v4! {
        bitandnot => _mm512_mask_andnot_ps, _mm512_maskz_andnot_ps;
    }

    // vpternlog's merge form keeps operand A (`src` doubles as A), so the
    // NOT imm must read operand C: 0x55.
    fn not_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let v = unsafe { arch::_mm512_castps_si512(value) };
        unsafe { arch::_mm512_castsi512_ps(arch::_mm512_mask_ternarylogic_epi32::<{ crate::ternlog_imm!(!C) }>(v, mask, v, v)) }
    }

    fn not_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let v = unsafe { arch::_mm512_castps_si512(value) };
        unsafe { arch::_mm512_castsi512_ps(arch::_mm512_mask_ternarylogic_epi32::<{ crate::ternlog_imm!(!C) }>(arch::_mm512_castps_si512(src), mask, v, v)) }
    }

    fn not_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let v = unsafe { arch::_mm512_castps_si512(value) };
        unsafe { arch::_mm512_castsi512_ps(arch::_mm512_maskz_ternarylogic_epi32::<{ crate::ternlog_imm!(!C) }>(mask, v, v, v)) }
    }

    fn ternlog_c<const IMM: i32>(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe {
            arch::_mm512_castsi512_ps(arch::_mm512_mask_ternarylogic_epi32::<IMM>(
                arch::_mm512_castps_si512(a),
                mask,
                arch::_mm512_castps_si512(b),
                arch::_mm512_castps_si512(c),
            ))
        }
    }

    // The merge form's `src` is also operand A, so an arbitrary `src` costs
    // a separate merge move.
    fn ternlog_m<const IMM: i32>(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mov_ps(src, mask, Self::ternlog::<IMM>(a, b, c)) }
    }

    fn ternlog_z<const IMM: i32>(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe {
            arch::_mm512_castsi512_ps(arch::_mm512_maskz_ternarylogic_epi32::<IMM>(
                mask,
                arch::_mm512_castps_si512(a),
                arch::_mm512_castps_si512(b),
                arch::_mm512_castps_si512(c),
            ))
        }
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

// Width ladder: zmm = two ymm. `vinsertf32x8` into a zero-extended low half.
#[thermite_macros::inline_always]
impl crate::register::ConcatRegister<super::F32x8V4> for F32x16V4 {
    fn concat(lo: Storage<super::F32x8V4>, hi: Storage<super::F32x8V4>) -> Storage<Self> {
        unsafe { arch::_mm512_insertf32x8::<1>(arch::_mm512_zextps256_ps512(lo), hi) }
    }

    fn split(value: Storage<Self>) -> (Storage<super::F32x8V4>, Storage<super::F32x8V4>) {
        let lo = unsafe { arch::_mm512_castps512_ps256(value) };
        let hi = unsafe { arch::_mm512_extractf32x8_ps::<1>(value) };
        (lo, hi)
    }
}

#[thermite_macros::inline_always]
impl crate::register::ExtendRegister<super::F32x8V4> for F32x16V4 {
    fn extend(value: Storage<super::F32x8V4>) -> Storage<Self> {
        unsafe { arch::_mm512_zextps256_ps512(value) }
    }

    fn narrow(value: Storage<Self>) -> Storage<super::F32x8V4> {
        unsafe { arch::_mm512_castps512_ps256(value) }
    }
}

#[thermite_macros::inline_always]
impl Register for F32x16V4 {
    type Element = f32;

    type Signed = super::I32x16V4;
    type Unsigned = super::U32x16V4;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        // k = lanes with any bit set (value != +0.0 bitwise), one vptestmd.
        unsafe { arch::_mm512_test_epi32_mask(arch::_mm512_castps_si512(value), arch::_mm512_castps_si512(value)) }
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::msb_to_mask(value)
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        // vpmovd2m (DQ): sign bit of each dword lane straight into k.
        unsafe { arch::_mm512_movepi32_mask(arch::_mm512_castps_si512(value)) }
    }

    fn new(value: GenericArray<f32, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm512_loadu_ps(value.as_ptr()) }
    }

    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_zextps128_ps512(arch::_mm_set_ss(value)) }
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_set1_ps(value) }
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_load_ps(ptr) }
    }

    unsafe fn load_m(src: Storage<Self>, mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_mask_loadu_ps(src, mask, ptr) }
    }

    unsafe fn load_z(mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_loadu_ps(mask, ptr) }
    }

    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_loadu_ps(ptr) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm512_store_ps(ptr, value) }
    }

    unsafe fn store_masked(ptr: *mut Self::Element, mask: Storage<Self::Mask>, value: Storage<Self>) {
        unsafe { arch::_mm512_mask_storeu_ps(ptr, mask, value) }
    }

    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm512_storeu_ps(ptr, value) }
    }

    unsafe fn load_stream(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_castsi512_ps(arch::_mm512_stream_load_si512(ptr as _)) }
    }

    unsafe fn store_stream(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm512_stream_ps(ptr, value) }
    }

    unsafe fn lookup(values: &[Self::Element], indices: Storage<Self::Unsigned>) -> Storage<Self> {
        if values.len() <= <Self::Lanes as Unsigned>::USIZE {
            let mut padded: GenericArray<Self::Element, Self::Lanes> = unsafe { core::mem::zeroed() };
            padded[..values.len()].copy_from_slice(values);

            unsafe { arch::_mm512_permutexvar_ps(indices, Self::new(padded)) }
        } else {
            unsafe { <Self as IndexableRegister<Self::Unsigned>>::gather(values.as_ptr(), indices) }
        }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        unsafe {
            let idx = arch::_mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
            arch::_mm512_permutexvar_ps(idx, value)
        }
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        // In-lane byte reversal: vpshufb (BW at 512-bit) with the 128-bit
        // bswap pattern broadcast to every block.
        unsafe {
            let pattern = arch::_mm512_broadcast_i32x4(arch::_mm_setr_epi8(
                3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12,
            ));
            arch::_mm512_castsi512_ps(arch::_mm512_shuffle_epi8(arch::_mm512_castps_si512(value), pattern))
        }
    }

    compress_expand_v4!(
        u16,
        _mm512_maskz_compress_ps,
        _mm512_mask_expand_ps,
        _mm512_maskz_expand_ps
    );

    const HAS_PERMUTEV: bool = true;

    fn permutev(value: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm512_permutexvar_ps(idxs, value) }
    }

    fn swizzle(a: Storage<Self>, b: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        // vpermt2ps: two-source variable permute is a single native
        // instruction (indices 0..15 pick from `a`, 16..31 from `b`), the
        // exact trait semantics, vs the 5-op emulation v3 needs.
        unsafe { arch::_mm512_permutex2var_ps(a, idxs, b) }
    }

    // --- masked variants -----------------------------------------------------

    fn new_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: GenericArray<f32, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_loadu_ps(src, mask, value.as_ptr()) }
    }

    fn new_z(mask: Storage<Self::Mask>, value: GenericArray<f32, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_loadu_ps(mask, value.as_ptr()) }
    }

    // vbroadcastss takes the mask directly.
    fn splat_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_mask_broadcastss_ps(src, mask, arch::_mm_set_ss(value)) }
    }

    fn splat_z(mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_broadcastss_ps(mask, arch::_mm_set_ss(value)) }
    }

    // Lane broadcast: one vpermps with a constant index, instead of the
    // default's extract-to-scalar round trip. The masked forms ride the
    // same instruction.
    fn broadcast<const I: usize>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_permutexvar_ps(arch::_mm512_set1_epi32(I as i32), value) }
    }

    fn broadcast_c<const I: usize>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_permutexvar_ps(value, mask, arch::_mm512_set1_epi32(I as i32), value) }
    }

    fn broadcast_m<const I: usize>(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_permutexvar_ps(src, mask, arch::_mm512_set1_epi32(I as i32), value) }
    }

    fn broadcast_z<const I: usize>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_permutexvar_ps(mask, arch::_mm512_set1_epi32(I as i32), value) }
    }

    fn broadcastv(value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm512_permutexvar_ps(arch::_mm512_set1_epi32(idx as i32), value) }
    }

    fn broadcastv_c(mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm512_mask_permutexvar_ps(value, mask, arch::_mm512_set1_epi32(idx as i32), value) }
    }

    fn broadcastv_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm512_mask_permutexvar_ps(src, mask, arch::_mm512_set1_epi32(idx as i32), value) }
    }

    fn broadcastv_z(mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_permutexvar_ps(mask, arch::_mm512_set1_epi32(idx as i32), value) }
    }

    fn reverse_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe {
            let idx = arch::_mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
            arch::_mm512_mask_permutexvar_ps(value, mask, idx, value)
        }
    }

    fn reverse_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe {
            let idx = arch::_mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
            arch::_mm512_mask_permutexvar_ps(src, mask, idx, value)
        }
    }

    fn reverse_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe {
            let idx = arch::_mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
            arch::_mm512_maskz_permutexvar_ps(mask, idx, value)
        }
    }

    // vpshufb is masked per BYTE (`__mmask64`), so the lane mask cannot feed
    // it directly: shuffle, then one merge move.
    masked_via_mov_v4! {
        mov = _mm512_mask_mov_ps, movz = _mm512_maskz_mov_ps;
        swap_bytes(value: Storage<Self>);
    }

    fn permutev_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_permutexvar_ps(src, mask, idxs, value) }
    }

    fn permutev_z(mask: Storage<Self::Mask>, value: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_permutexvar_ps(mask, idxs, value) }
    }

    // vpermt2ps merges only into `a` (vpermi2ps into the index), neither of
    // which is an arbitrary `src`: permute, then one merge move. The zeroing
    // form is native.
    fn swizzle_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        a: Storage<Self>,
        b: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mov_ps(src, mask, arch::_mm512_permutex2var_ps(a, idxs, b)) }
    }

    fn swizzle_z(
        mask: Storage<Self::Mask>,
        a: Storage<Self>,
        b: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_permutex2var_ps(mask, a, idxs, b) }
    }
}

#[thermite_macros::inline_always]
impl InterleaveRegister for F32x16V4 {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        // One vpermt2ps per output: no unpack + cross-lane fixup dance.
        unsafe {
            let lo = arch::_mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
            let hi = arch::_mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
            (
                arch::_mm512_permutex2var_ps(a, lo, b),
                arch::_mm512_permutex2var_ps(a, hi, b),
            )
        }
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe {
            let even = arch::_mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
            let odd = arch::_mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
            (
                arch::_mm512_permutex2var_ps(a, even, b),
                arch::_mm512_permutex2var_ps(a, odd, b),
            )
        }
    }
}

#[thermite_macros::inline_always]
impl IndexableRegister<super::U32x16V4> for F32x16V4 {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<super::U32x16V4>) -> Storage<Self> {
        unsafe { arch::_mm512_i32gather_ps::<4>(indices, ptr as _) }
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<super::U32x16V4>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_i32gather_ps::<4>(src, mask, indices, ptr as _) }
    }

    unsafe fn scatter(value: Storage<Self>, ptr: *mut Self::Element, indices: Storage<super::U32x16V4>) {
        unsafe { arch::_mm512_i32scatter_ps::<4>(ptr as _, indices, value) }
    }

    unsafe fn scatter_m(
        value: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *mut Self::Element,
        indices: Storage<super::U32x16V4>,
    ) {
        unsafe { arch::_mm512_mask_i32scatter_ps::<4>(ptr as _, mask, indices, value) }
    }
}

#[thermite_macros::inline_always]
impl IndexableRegister<ArrayRegister<super::U64x8V4, 2>> for F32x16V4 {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<ArrayRegister<super::U64x8V4, 2>>) -> Storage<Self> {
        let ArrayRegister([lo_idx, hi_idx]) = indices;

        unsafe {
            let lo = arch::_mm512_i64gather_ps::<4>(lo_idx, ptr as _);
            let hi = arch::_mm512_i64gather_ps::<4>(hi_idx, ptr as _);
            arch::_mm512_insertf32x8::<1>(arch::_mm512_zextps256_ps512(lo), hi)
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
                arch::_mm512_mask_i64gather_ps::<4>(arch::_mm512_castps512_ps256(src), mask as u8, lo_idx, ptr as _);
            let hi = arch::_mm512_mask_i64gather_ps::<4>(
                arch::_mm512_extractf32x8_ps::<1>(src),
                (mask >> 8) as u8,
                hi_idx,
                ptr as _,
            );
            arch::_mm512_insertf32x8::<1>(arch::_mm512_zextps256_ps512(lo), hi)
        }
    }
}

#[thermite_macros::inline_always]
impl ShuffleRegister for F32x16V4 {
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_shuffle_ps(lhs, rhs, IMM8) }
    }
}

#[thermite_macros::inline_always]
impl PartialOrdRegister for F32x16V4 {
    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmp_ps_mask(lhs, rhs, arch::_CMP_LT_OQ) }
    }

    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmp_ps_mask(lhs, rhs, arch::_CMP_LE_OQ) }
    }

    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmp_ps_mask(lhs, rhs, arch::_CMP_GT_OQ) }
    }

    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmp_ps_mask(lhs, rhs, arch::_CMP_GE_OQ) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmp_ps_mask(lhs, rhs, arch::_CMP_EQ_OQ) }
    }

    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmp_ps_mask(lhs, rhs, arch::_CMP_NEQ_UQ) }
    }
}

#[thermite_macros::inline_always]
impl NumericRegister for F32x16V4 {
    const ZERO: Storage<Self> = reg::<Self, 16>([0.0; 16]);
    const ONE: Storage<Self> = reg::<Self, 16>([1.0; 16]);
    const TWO: Storage<Self> = reg::<Self, 16>([2.0; 16]);

    const MIN: Storage<Self> = reg::<Self, 16>([f32::MIN; 16]);
    const MAX: Storage<Self> = reg::<Self, 16>([f32::MAX; 16]);

    fn min_element(value: Storage<Self>) -> Self::Element {
        unsafe { arch::_mm512_reduce_min_ps(value) }
    }

    fn max_element(value: Storage<Self>) -> Self::Element {
        unsafe { arch::_mm512_reduce_max_ps(value) }
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element {
        unsafe { arch::_mm512_reduce_add_ps(value) }
    }

    fn prod_elements(value: Storage<Self>) -> Self::Element {
        unsafe { arch::_mm512_reduce_mul_ps(value) }
    }

    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as Unsigned>::USIZE as f32)
    }

    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as f32))
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_add_ps(lhs, rhs) }
    }

    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_sub_ps(lhs, rhs) }
    }

    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mul_ps(lhs, rhs) }
    }

    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_div_ps(lhs, rhs) }
    }

    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::nmul_adde(Self::trunc(Self::div(lhs, rhs)), rhs, lhs)
    }

    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::fix_min::<Self>(lhs, rhs, unsafe { arch::_mm512_min_ps(lhs, rhs) })
    }

    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::fix_max::<Self>(lhs, rhs, unsafe { arch::_mm512_max_ps(lhs, rhs) })
    }

    // --- masked variants -----------------------------------------------------
    // Embedded-mask forms: keep `lhs`/`src` where the mask is false, one
    // instruction, no blend.

    masked_binary_v4! {
        add => _mm512_mask_add_ps, _mm512_maskz_add_ps;
        sub => _mm512_mask_sub_ps, _mm512_maskz_sub_ps;
        mul => _mm512_mask_mul_ps, _mm512_maskz_mul_ps;
        div => _mm512_mask_div_ps, _mm512_maskz_div_ps;
    }

    // Under strict IEEE the NaN/signed-zero fixups wrap the raw instruction,
    // so the mask has to apply to the fixed result.
    cfg_select! {
        feature = "strict_ieee754" => {
            masked_via_mov_v4! {
                mov = _mm512_mask_mov_ps, movz = _mm512_maskz_mov_ps;
                min(lhs: Storage<Self>, rhs: Storage<Self>);
                max(lhs: Storage<Self>, rhs: Storage<Self>);
            }
        }
        _ => {
            masked_binary_v4! {
                min => _mm512_mask_min_ps, _mm512_maskz_min_ps;
                max => _mm512_mask_max_ps, _mm512_maskz_max_ps;
            }
        }
    }

    // rem = lhs - trunc(lhs / rhs) * rhs: the final fnmadd takes the mask,
    // and its `mask3` form merges into the addend, which IS `lhs`.
    fn rem_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask3_fnmadd_ps(Self::trunc(Self::div(lhs, rhs)), rhs, lhs, mask) }
    }

    fn rem_m(src: Storage<Self>, mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mov_ps(src, mask, Self::rem(lhs, rhs)) }
    }

    fn rem_z(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_fnmadd_ps(mask, Self::trunc(Self::div(lhs, rhs)), rhs, lhs) }
    }

    fn square_c(mask: Storage<Self::Mask>, lhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mul_ps(lhs, mask, lhs, lhs) }
    }

    fn square_m(src: Storage<Self>, mask: Storage<Self::Mask>, lhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mul_ps(src, mask, lhs, lhs) }
    }

    fn square_z(mask: Storage<Self::Mask>, lhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_mul_ps(mask, lhs, lhs) }
    }

    fn scale_c(mask: Storage<Self::Mask>, value: Storage<Self>, scalar: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mul_ps(value, mask, value, Self::splat(scalar)) }
    }

    fn scale_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
        scalar: Self::Element,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mul_ps(src, mask, value, Self::splat(scalar)) }
    }

    fn scale_z(mask: Storage<Self::Mask>, value: Storage<Self>, scalar: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_mul_ps(mask, value, Self::splat(scalar)) }
    }
}

#[thermite_macros::inline_always]
impl SignedRegister for F32x16V4 {
    const NEG_ONE: Storage<Self> = reg::<Self, 16>([-1.0; 16]);
    const MIN_POSITIVE: Storage<Self> = reg::<Self, 16>([f32::MIN_POSITIVE; 16]);

    fn neg(value: Storage<Self>) -> Storage<Self> {
        Self::bitxor(value, Self::NEG_ZERO)
    }

    fn abs(value: Storage<Self>) -> Storage<Self> {
        Self::bitandnot(value, Self::NEG_ZERO)
    }

    fn copysign(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        // Native ternlog bit-select: sign from rhs, everything else from lhs.
        Self::ternlog::<0xCA>(Self::NEG_ZERO, rhs, lhs)
    }

    fn signum(value: Storage<Self>) -> Storage<Self> {
        let s = Self::bitor(Self::ONE, Self::bitand(value, Self::NEG_ZERO));
        #[cfg(feature = "strict_ieee754")]
        let s = Self::blendv(Self::is_nan(value), s, value);
        s
    }

    // --- masked variants -----------------------------------------------------

    // Embedded-mask xor / andnot against the sign constant: one instruction.
    fn neg_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_xor_ps(value, mask, value, Self::NEG_ZERO) }
    }

    fn neg_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_xor_ps(src, mask, value, Self::NEG_ZERO) }
    }

    fn neg_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_xor_ps(mask, value, Self::NEG_ZERO) }
    }

    fn abs_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_andnot_ps(value, mask, Self::NEG_ZERO, value) }
    }

    fn abs_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_andnot_ps(src, mask, Self::NEG_ZERO, value) }
    }

    fn abs_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_andnot_ps(mask, Self::NEG_ZERO, value) }
    }

    // The merge form keeps operand A, so put `lhs` first and select on C:
    // imm 0xD8 = `c ? b : a` (0xCA is `a ? b : c`).
    fn copysign_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe {
            arch::_mm512_castsi512_ps(arch::_mm512_mask_ternarylogic_epi32::<
                { crate::ternlog_imm!(C & B | !C & A) },
            >(
                arch::_mm512_castps_si512(lhs),
                mask,
                arch::_mm512_castps_si512(rhs),
                arch::_mm512_castps_si512(Self::NEG_ZERO),
            ))
        }
    }

    fn copysign_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        lhs: Storage<Self>,
        rhs: Storage<Self>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mov_ps(src, mask, Self::copysign(lhs, rhs)) }
    }

    fn copysign_z(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::ternlog_z::<0xCA>(mask, Self::NEG_ZERO, rhs, lhs)
    }
}

#[thermite_macros::inline_always]
impl FloatRegister for F32x16V4 {
    const HAS_NATIVE_FMA: tribool::Tribool = tribool::True;

    type Bits = super::U32x16V4;
    type SignedBits = super::I32x16V4;
    type ExtendedPrecision = ArrayRegister<super::F64x8V4, 2>;

    const HALF: Storage<Self> = reg::<Self, 16>([0.5; 16]);
    const NEG_ZERO: Storage<Self> = reg::<Self, 16>([-0.0; 16]);
    const EPSILON: Storage<Self> = reg::<Self, 16>([f32::EPSILON; 16]);
    const INFINITY: Storage<Self> = reg::<Self, 16>([f32::INFINITY; 16]);
    const NEG_INFINITY: Storage<Self> = reg::<Self, 16>([f32::NEG_INFINITY; 16]);
    const NAN: Storage<Self> = reg::<Self, 16>([f32::NAN; 16]);

    const EXP_MASK: Storage<Self::Bits> = reg::<Self::Bits, 16>([0x7F800000; 16]);

    fn mul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_fmadd_ps(lhs, rhs, acc) }
    }

    fn mul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_fmsub_ps(lhs, rhs, acc) }
    }

    fn nmul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_fnmadd_ps(lhs, rhs, acc) }
    }

    fn nmul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_fnmsub_ps(lhs, rhs, acc) }
    }

    fn mul_adde(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        Self::mul_add(lhs, rhs, acc)
    }

    fn mul_sube(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        Self::mul_sub(lhs, rhs, acc)
    }

    fn nmul_adde(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        Self::nmul_add(lhs, rhs, acc)
    }

    fn nmul_sube(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        Self::nmul_sub(lhs, rhs, acc)
    }

    // No 512-bit `addsub`: AVX-512 dropped it, and the trait default (xor the
    // alternating sign mask, then add) is the optimal lowering here anyway.

    fn fmaddsub(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_fmaddsub_ps(a, b, c) }
    }

    fn fmsubadd(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_fmsubadd_ps(a, b, c) }
    }

    fn sqrt(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_sqrt_ps(value) }
    }

    fn rsqrt(value: Storage<Self>) -> Storage<Self> {
        cfg_select! {
            feature = "strict_ieee754" => {
                Self::rcp(Self::sqrt(value))
            }
            _ => unsafe { arch::_mm512_rsqrt14_ps(value) }
        }
    }

    fn rcp(value: Storage<Self>) -> Storage<Self> {
        cfg_select! {
            feature = "strict_ieee754" => {
                Self::div(Self::ONE, value)
            }
            _ => unsafe { arch::_mm512_rcp14_ps(value) }
        }
    }

    // vrcp14/vrsqrt14: 2^-14 relative error, up from the pre-AVX512 12-bit
    // estimates. Newton-step counts in kernels keyed on element type still
    // hold (they assume the WORSE 12-bit seed), just with extra margin.
    const HAS_APPROX_RSQRT: bool = cfg!(not(feature = "strict_ieee754"));
    const HAS_APPROX_RCP: bool = cfg!(not(feature = "strict_ieee754"));

    // vrndscale imm[3:0]: bit 3 suppresses precision exceptions, bits 1:0 are
    // the rounding mode (00 nearest-even, 01 down, 10 up, 11 truncate).
    fn floor(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_roundscale_ps(value, 0x09) }
    }

    fn ceil(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_roundscale_ps(value, 0x0A) }
    }

    fn round(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_roundscale_ps(value, 0x08) }
    }

    fn trunc(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_roundscale_ps(value, 0x0B) }
    }

    fn next_up(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_nextupps_v4(value) }
    }

    fn next_down(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_nextdownps_v4(value) }
    }

    const NATIVE_CAP: NativeCapability = NativeCapability::NONE;

    // --- masked variants -----------------------------------------------------

    // vfmadd's merge form keeps the first multiplicand, which is exactly the
    // `_c` contract. The estimating `e` family is the same instruction here.
    masked_fma_v4! {
        mov = _mm512_mask_mov_ps;
        mul_add => _mm512_mask_fmadd_ps, _mm512_maskz_fmadd_ps;
        mul_sub => _mm512_mask_fmsub_ps, _mm512_maskz_fmsub_ps;
        nmul_add => _mm512_mask_fnmadd_ps, _mm512_maskz_fnmadd_ps;
        nmul_sub => _mm512_mask_fnmsub_ps, _mm512_maskz_fnmsub_ps;
        mul_adde => _mm512_mask_fmadd_ps, _mm512_maskz_fmadd_ps;
        mul_sube => _mm512_mask_fmsub_ps, _mm512_maskz_fmsub_ps;
        nmul_adde => _mm512_mask_fnmadd_ps, _mm512_maskz_fnmadd_ps;
        nmul_sube => _mm512_mask_fnmsub_ps, _mm512_maskz_fnmsub_ps;
        fmaddsub => _mm512_mask_fmaddsub_ps, _mm512_maskz_fmaddsub_ps;
        fmsubadd => _mm512_mask_fmsubadd_ps, _mm512_maskz_fmsubadd_ps;
    }

    // addsub = a + (b ^ ALT_NEG): the xor stays unmasked, the add takes it.
    fn addsub_c(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_add_ps(a, mask, a, Self::bitxor(b, Self::ALT_NEG)) }
    }

    fn addsub_m(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_add_ps(src, mask, a, Self::bitxor(b, Self::ALT_NEG)) }
    }

    fn addsub_z(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_add_ps(mask, a, Self::bitxor(b, Self::ALT_NEG)) }
    }

    masked_unary_v4! {
        sqrt => _mm512_mask_sqrt_ps, _mm512_maskz_sqrt_ps;
        floor => _mm512_mask_roundscale_ps, _mm512_maskz_roundscale_ps, 0x09;
        ceil => _mm512_mask_roundscale_ps, _mm512_maskz_roundscale_ps, 0x0A;
        round => _mm512_mask_roundscale_ps, _mm512_maskz_roundscale_ps, 0x08;
        trunc => _mm512_mask_roundscale_ps, _mm512_maskz_roundscale_ps, 0x0B;
    }

    // Same fork as `rcp`/`rsqrt`: exact division under strict IEEE (the
    // divide takes the mask, the sqrt underneath rsqrt runs unmasked), the
    // 14-bit estimates otherwise.
    cfg_select! {
        feature = "strict_ieee754" => {
            fn rcp_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::_mm512_mask_div_ps(value, mask, Self::ONE, value) }
            }

            fn rcp_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::_mm512_mask_div_ps(src, mask, Self::ONE, value) }
            }

            fn rcp_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::_mm512_maskz_div_ps(mask, Self::ONE, value) }
            }

            fn rsqrt_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::_mm512_mask_div_ps(value, mask, Self::ONE, Self::sqrt(value)) }
            }

            fn rsqrt_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::_mm512_mask_div_ps(src, mask, Self::ONE, Self::sqrt(value)) }
            }

            fn rsqrt_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::_mm512_maskz_div_ps(mask, Self::ONE, Self::sqrt(value)) }
            }
        }
        _ => {
            masked_unary_v4! {
                rcp => _mm512_mask_rcp14_ps, _mm512_maskz_rcp14_ps;
                rsqrt => _mm512_mask_rsqrt14_ps, _mm512_maskz_rsqrt14_ps;
            }
        }
    }

    // fract = value - trunc(value): the subtract takes the mask.
    fn fract_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_ps(value, mask, value, Self::trunc(value)) }
    }

    fn fract_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_ps(src, mask, value, Self::trunc(value)) }
    }

    fn fract_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_sub_ps(mask, value, Self::trunc(value)) }
    }

    // mul_sign = value ^ (sign & NEG_ZERO): one vpternlog, imm 0x78 =
    // `a ^ (b & c)`, merging into A = value.
    fn mul_sign_c(mask: Storage<Self::Mask>, value: Storage<Self>, sign: Storage<Self>) -> Storage<Self> {
        Self::ternlog_c::<0x78>(mask, value, sign, Self::NEG_ZERO)
    }

    fn mul_sign_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
        sign: Storage<Self>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mov_ps(src, mask, Self::mul_sign(value, sign)) }
    }

    fn mul_sign_z(mask: Storage<Self::Mask>, value: Storage<Self>, sign: Storage<Self>) -> Storage<Self> {
        Self::ternlog_z::<0x78>(mask, value, sign, Self::NEG_ZERO)
    }

    fn signed_zero_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_and_ps(value, mask, Self::NEG_ZERO, value) }
    }

    fn signed_zero_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_and_ps(src, mask, Self::NEG_ZERO, value) }
    }

    fn signed_zero_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_and_ps(mask, Self::NEG_ZERO, value) }
    }

    // The next_up/next_down polyfills end in a k-masked add, but the mask
    // there is the sign-split, not ours: run them, then one merge move.
    masked_via_mov_v4! {
        mov = _mm512_mask_mov_ps, movz = _mm512_maskz_mov_ps;
        next_up(value: Storage<Self>);
        next_down(value: Storage<Self>);
    }
}

// The widening leg of `ExtendedPrecision`: f32x16 -> [f64x8; 2].
#[thermite_macros::inline_always]
impl CastRegister<F32x16V4> for ArrayRegister<super::F64x8V4, 2> {
    fn cast_from(value: Storage<F32x16V4>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm512_cvtps_pd(arch::_mm512_castps512_ps256(value));
            let hi = arch::_mm512_cvtps_pd(arch::_mm512_extractf32x8_ps::<1>(value));
            ArrayRegister([lo, hi])
        }
    }
}

// And the narrowing leg back (required by `FloatRegister`'s
// `CastRegister<Self::ExtendedPrecision>` supertrait bound).
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<super::F64x8V4, 2>> for F32x16V4 {
    fn cast_from(value: Storage<ArrayRegister<super::F64x8V4, 2>>) -> Storage<Self> {
        let ArrayRegister([lo, hi]) = value;

        unsafe {
            arch::_mm512_insertf32x8::<1>(
                arch::_mm512_zextps256_ps512(arch::_mm512_cvtpd_ps(lo)),
                arch::_mm512_cvtpd_ps(hi),
            )
        }
    }
}
