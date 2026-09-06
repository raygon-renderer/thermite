//! `i64x8` on AVX-512: the native 512-bit signed-qword register.
//!
//! This is where x86's 64-bit integer support stops being a pile of polyfills.
//! See i32x16.rs for the integer pattern and f32x16.rs for the shared v4
//! house rules. Only the 64-bit-specific notes live here.
//!
//! What AVX-512 hands us natively that v3 had to emulate (all floor, F/CD/DQ):
//!
//! - **Compares**: `vpcmpq` gives gt/ge/lt/le/eq/ne straight into an opmask.
//!   v3 had `_mm256_cmpgt_epi64` and built everything else out of it.
//! - **min/max**: `vpminsq`/`vpmaxsq` (F) replace the v3 compare+blend pairs.
//! - **`vpmullq`** (DQ) is a real full-width low multiply, where v3 ran a three-term
//!   32-bit limb sum for `mul`/`mullo`.
//! - **Arithmetic shifts**: `vpsraq` in imm/scalar/variable forms. v3 had no
//!   64-bit `sra` at all and faked all three.
//! - **`vplzcntq`** (CD) for `leading_zeros`, where v3 went through `ilog2p1`.
//! - **`vpabsq`** for `abs`.
//!
//! Still emulated, because no tier has the instruction:
//!
//! - **`mulhi`**: there is no signed (or unsigned) 64x64 -> high-64 multiply
//!   anywhere in AVX-512. Same schoolbook 32-bit limb decomposition as v3's
//!   `_mm256_mullhi_epu64x_v3`, plus the signed fixup, but the fixup uses
//!   masked subtracts instead of sign-mask ANDs.
//! - **`saturating_add`/`saturating_sub`**: same ternlog overflow detection as
//!   i32x16 (imm 0x42 / 0x18) with `vpternlogq` + `vpmovq2m`.
//! - **`count_ones`** forks on `F::AVX512VPOPCNTDQ` (tier 2, `vpopcntq`) with
//!   a half-split onto the inherited 256-bit `psadbw` polyfill as the floor
//!   fallback. `HAS_HARDWARE_POPCNT` reports the same const so they cannot
//!   drift.

use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, BitwiseRegister, CastRegister, CoreRegister, IndexableRegister, IntegerRegister,
        InterleaveRegister, NumericRegister, PartialOrdRegister, Register, SignedIntegerRegister, SignedRegister,
        Storage, ZeroUpper, array::ArrayRegister, empty_reg, reg, reg_splat,
    },
};

use super::super::{Avx512Features, DefaultAvx512 as F};
use super::arch;
use super::kmask::KMask8;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct I64x8V4;

#[thermite_macros::inline_always]
impl CoreRegister for I64x8V4 {
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
impl BitwiseRegister for I64x8V4 {
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
impl crate::register::ConcatRegister<super::I64x4V4> for I64x8V4 {
    fn concat(lo: Storage<super::I64x4V4>, hi: Storage<super::I64x4V4>) -> Storage<Self> {
        unsafe { arch::_mm512_inserti64x4::<1>(arch::_mm512_zextsi256_si512(lo), hi) }
    }

    fn split(value: Storage<Self>) -> (Storage<super::I64x4V4>, Storage<super::I64x4V4>) {
        let lo = unsafe { arch::_mm512_castsi512_si256(value) };
        let hi = unsafe { arch::_mm512_extracti64x4_epi64::<1>(value) };
        (lo, hi)
    }
}

#[thermite_macros::inline_always]
impl crate::register::ExtendRegister<super::I64x4V4> for I64x8V4 {
    fn extend(value: Storage<super::I64x4V4>) -> Storage<Self> {
        unsafe { arch::_mm512_zextsi256_si512(value) }
    }

    fn narrow(value: Storage<Self>) -> Storage<super::I64x4V4> {
        unsafe { arch::_mm512_castsi512_si256(value) }
    }
}

#[thermite_macros::inline_always]
impl Register for I64x8V4 {
    type Element = i64;

    type Signed = I64x8V4;
    type Unsigned = super::U64x8V4;

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
        unsafe { arch::_mm512_zextsi128_si512(arch::_mm_set_epi64x(0, value)) }
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_set1_epi64(value) }
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_load_si512(ptr as *const _) }
    }

    unsafe fn load_m(src: Storage<Self>, mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_mask_loadu_epi64(src, mask, ptr) }
    }

    unsafe fn load_z(mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_loadu_epi64(mask, ptr) }
    }

    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_loadu_si512(ptr as *const _) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm512_store_si512(ptr as *mut _, value) }
    }

    unsafe fn store_masked(ptr: *mut Self::Element, mask: Storage<Self::Mask>, value: Storage<Self>) {
        unsafe { arch::_mm512_mask_storeu_epi64(ptr, mask, value) }
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
            unsafe { <Self as IndexableRegister<Self::Unsigned>>::gather(values.as_ptr(), indices) }
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
        unsafe { arch::_mm512_mask_loadu_epi64(src, mask, value.as_ptr()) }
    }

    fn new_z(mask: Storage<Self::Mask>, value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_loadu_epi64(mask, value.as_ptr()) }
    }

    // vpbroadcastq from a GPR takes the mask directly.
    fn splat_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_mask_set1_epi64(src, mask, value) }
    }

    fn splat_z(mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_set1_epi64(mask, value) }
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
impl InterleaveRegister for I64x8V4 {
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
impl IndexableRegister<super::U64x8V4> for I64x8V4 {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<super::U64x8V4>) -> Storage<Self> {
        unsafe { arch::_mm512_i64gather_epi64::<8>(indices, ptr as _) }
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<super::U64x8V4>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_i64gather_epi64::<8>(src, mask, indices, ptr as _) }
    }

    unsafe fn scatter(value: Storage<Self>, ptr: *mut Self::Element, indices: Storage<super::U64x8V4>) {
        unsafe { arch::_mm512_i64scatter_epi64::<8>(ptr as _, indices, value) }
    }

    unsafe fn scatter_m(
        value: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *mut Self::Element,
        indices: Storage<super::U64x8V4>,
    ) {
        unsafe { arch::_mm512_mask_i64scatter_epi64::<8>(ptr as _, mask, indices, value) }
    }
}

// Eight 32-bit indices are one ymm (`u32x8`): vpgatherdq / vpscatterdq.
// The x16 grid slot (two zmm): sixteen 32-bit indices in one zmm, split into
// the two ymm index halves of vpgatherdq / vpscatterdq.
#[thermite_macros::inline_always]
impl IndexableRegister<super::U32x16V4> for ArrayRegister<I64x8V4, 2> {
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

#[thermite_macros::inline_always]
impl IndexableRegister<super::U32x8V4> for I64x8V4 {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<super::U32x8V4>) -> Storage<Self> {
        unsafe { arch::_mm512_i32gather_epi64::<8>(indices, ptr) }
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<super::U32x8V4>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_i32gather_epi64::<8>(src, mask, indices, ptr) }
    }

    unsafe fn scatter(value: Storage<Self>, ptr: *mut Self::Element, indices: Storage<super::U32x8V4>) {
        unsafe { arch::_mm512_i32scatter_epi64::<8>(ptr, indices, value) }
    }

    unsafe fn scatter_m(
        value: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *mut Self::Element,
        indices: Storage<super::U32x8V4>,
    ) {
        unsafe { arch::_mm512_mask_i32scatter_epi64::<8>(ptr, mask, indices, value) }
    }
}

#[thermite_macros::inline_always]
impl BitshiftRegister for I64x8V4 {
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
impl PartialOrdRegister for I64x8V4 {
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmpgt_epi64_mask(lhs, rhs) }
    }

    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmpge_epi64_mask(lhs, rhs) }
    }

    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmplt_epi64_mask(lhs, rhs) }
    }

    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmple_epi64_mask(lhs, rhs) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmpeq_epi64_mask(lhs, rhs) }
    }

    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmpneq_epi64_mask(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl NumericRegister for I64x8V4 {
    const ZERO: Storage<Self> = reg::<Self, 8>([0; 8]);
    const ONE: Storage<Self> = reg::<Self, 8>([1; 8]);
    const TWO: Storage<Self> = reg::<Self, 8>([2; 8]);

    const MIN: Storage<Self> = reg::<Self, 8>([i64::MIN; 8]);
    const MAX: Storage<Self> = reg::<Self, 8>([i64::MAX; 8]);

    fn min_element(value: Storage<Self>) -> Self::Element {
        unsafe { arch::_mm512_reduce_min_epi64(value) }
    }

    fn max_element(value: Storage<Self>) -> Self::Element {
        unsafe { arch::_mm512_reduce_max_epi64(value) }
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element {
        unsafe { arch::_mm512_reduce_add_epi64(value) }
    }

    fn prod_elements(value: Storage<Self>) -> Self::Element {
        unsafe { arch::_mm512_reduce_mul_epi64(value) }
    }

    fn relaxed_pairwise_sum(lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self> {
        // `deinterleave` already lands the even/odd halves in exactly the
        // order `pairwise_sum` specifies, so relaxed and exact coincide:
        // two vpermt2q + one vpaddq, versus the scalar default.
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
        unsafe { arch::_mm512_add_epi64(lhs, rhs) }
    }

    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_sub_epi64(lhs, rhs) }
    }

    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        // vpmullq (DQ, floor): the three-term limb sum v3 needed is gone.
        unsafe { arch::_mm512_mullo_epi64(lhs, rhs) }
    }

    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a / b })
    }

    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a % b })
    }

    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_min_epi64(lhs, rhs) }
    }

    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_max_epi64(lhs, rhs) }
    }

    // --- masked variants -----------------------------------------------------
    // Embedded-mask forms: keep `lhs`/`src` where the mask is false, one
    // instruction, no blend.

    masked_binary_v4! {
        add => _mm512_mask_add_epi64, _mm512_maskz_add_epi64;
        sub => _mm512_mask_sub_epi64, _mm512_maskz_sub_epi64;
        mul => _mm512_mask_mullo_epi64, _mm512_maskz_mullo_epi64;
        min => _mm512_mask_min_epi64, _mm512_maskz_min_epi64;
        max => _mm512_mask_max_epi64, _mm512_maskz_max_epi64;
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
impl SignedRegister for I64x8V4 {
    const NEG_ONE: Storage<Self> = reg::<Self, 8>([-1; 8]);
    const MIN_POSITIVE: Storage<Self> = reg_splat::<Self>(1);

    fn neg(value: Storage<Self>) -> Storage<Self> {
        // No vpsign in EVEX-land. Plain subtract from zero.
        Self::sub(Self::ZERO, value)
    }

    fn is_negative(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::msb_to_mask(value)
    }

    fn abs(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_abs_epi64(value) }
    }

    fn copysign(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        // Negate exactly where the signs differ.
        Self::neg_c(Self::msb_to_mask(Self::bitxor(lhs, rhs)), lhs)
    }

    fn signum(value: Storage<Self>) -> Storage<Self> {
        // Three-valued clamp: min(max(v, -1), 1). Two instructions now that
        // vpminsq/vpmaxsq exist, where v3 needed a compare pair plus a subtract.
        Self::min(Self::max(value, Self::NEG_ONE), Self::ONE)
    }

    // --- masked variants -----------------------------------------------------

    // Embedded-mask subtract from zero: one instruction.
    fn neg_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_epi64(value, mask, Self::ZERO, value) }
    }

    fn neg_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_epi64(src, mask, Self::ZERO, value) }
    }

    fn neg_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_sub_epi64(mask, Self::ZERO, value) }
    }

    masked_unary_v4! {
        abs => _mm512_mask_abs_epi64, _mm512_maskz_abs_epi64;
    }

    // copysign = negate where the signs differ: the caller's mask ANDs into
    // the sign-difference mask (one `kand`) and the masked subtract stays.
    fn copysign_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::neg_c(mask & Self::msb_to_mask(Self::bitxor(lhs, rhs)), lhs)
    }

    fn copysign_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        lhs: Storage<Self>,
        rhs: Storage<Self>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mov_epi64(src, mask, Self::copysign(lhs, rhs)) }
    }

    fn copysign_z(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_mov_epi64(mask, Self::copysign(lhs, rhs)) }
    }
}

#[thermite_macros::inline_always]
impl IntegerRegister for I64x8V4 {
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        // There is no 64x64 -> high-64 multiply at any AVX-512 tier, so this
        // stays the v3 algorithm (`_mm256_mullhi_epu64x_v3`): schoolbook
        // 32-bit limbs via vpmuludq, then the signed correction
        // hi - (lhs < 0 ? rhs : 0) - (rhs < 0 ? lhs : 0). v3 built that
        // correction from sign-mask ANDs. Here it is two masked subtracts.
        unsafe {
            let lomask = arch::_mm512_set1_epi64(0xFFFF_FFFF);

            let xh = arch::_mm512_srli_epi64::<32>(lhs);
            let yh = arch::_mm512_srli_epi64::<32>(rhs);

            let w0 = arch::_mm512_mul_epu32(lhs, rhs);
            let w1 = arch::_mm512_mul_epu32(lhs, yh);
            let w2 = arch::_mm512_mul_epu32(xh, rhs);
            let w3 = arch::_mm512_mul_epu32(xh, yh);

            let s1 = arch::_mm512_add_epi64(w1, arch::_mm512_srli_epi64::<32>(w0));
            let s1l = arch::_mm512_and_si512(s1, lomask);
            let s1h = arch::_mm512_srli_epi64::<32>(s1);
            let s2h = arch::_mm512_srli_epi64::<32>(arch::_mm512_add_epi64(w2, s1l));

            let hi = arch::_mm512_add_epi64(arch::_mm512_add_epi64(w3, s1h), s2h);

            let hi = arch::_mm512_mask_sub_epi64(hi, arch::_mm512_movepi64_mask(lhs), hi, rhs);
            arch::_mm512_mask_sub_epi64(hi, arch::_mm512_movepi64_mask(rhs), hi, lhs)
        }
    }

    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mullo_epi64(lhs, rhs) }
    }

    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        // Signed overflow iff the operands agree in sign and the result does
        // not: msb of (~(l^r) & (l^sum)), which is ternlog imm 0x42 -> one
        // vpternlogq + vpmovq2m. Saturated value = MAX ^ (l >> 63)
        // (positive -> MAX, negative -> MIN), merged with an embedded mask.
        unsafe {
            let sum = Self::add(lhs, rhs);
            let ov = arch::_mm512_movepi64_mask(arch::_mm512_ternarylogic_epi64::<
                { crate::ternlog_imm!((A ^ C) & (B ^ C)) },
            >(lhs, rhs, sum));
            let sat = Self::bitxor(Self::MAX, Self::srai::<63>(lhs));
            arch::_mm512_mask_mov_epi64(sum, ov, sat)
        }
    }

    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        // Overflow iff the operands DISAGREE in sign and the result disagrees
        // with lhs: msb of ((l^r) & (l^diff)) = ternlog imm 0x18.
        unsafe {
            let diff = Self::sub(lhs, rhs);
            let ov = arch::_mm512_movepi64_mask(arch::_mm512_ternarylogic_epi64::<
                { crate::ternlog_imm!((A ^ B) & (A ^ C)) },
            >(lhs, rhs, diff));
            let sat = Self::bitxor(Self::MAX, Self::srai::<63>(lhs));
            arch::_mm512_mask_mov_epi64(diff, ov, sat)
        }
    }

    fn wrapping_sum(value: Storage<Self>) -> Self::Element {
        unsafe { arch::_mm512_reduce_add_epi64(value) }
    }

    fn wrapping_product(value: Storage<Self>) -> Self::Element {
        unsafe { arch::_mm512_reduce_mul_epi64(value) }
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
        mullo => _mm512_mask_mullo_epi64, _mm512_maskz_mullo_epi64;
    }

    // mulhi's signed fixup, the saturating fixup moves and the divider
    // polyfills have no spare mask slot: compute, then one merge move.
    masked_via_mov_v4! {
        mov = _mm512_mask_mov_epi64, movz = _mm512_maskz_mov_epi64;
        mulhi(lhs: Storage<Self>, rhs: Storage<Self>);
        saturating_add(lhs: Storage<Self>, rhs: Storage<Self>);
        saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>);
        div_branched(value: Storage<Self>, divider: crate::Divider<Self::Element>);
        div_branchfree(value: Storage<Self>, divider: crate::BranchfreeDivider<Self::Element>);
        divv_branchfree(value: Storage<Self>, dividers: crate::divider::vector::VectorDivider<Self>);
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
        let low = Self::sub(Self::bitand(value, Self::neg(value)), Self::ONE);
        unsafe { arch::_mm512_mask_popcnt_epi64x_v4(value, mask, low) }
    }

    fn trailing_zeros_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let low = Self::sub(Self::bitand(value, Self::neg(value)), Self::ONE);
        unsafe { arch::_mm512_mask_popcnt_epi64x_v4(src, mask, low) }
    }

    fn trailing_zeros_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let low = Self::sub(Self::bitand(value, Self::neg(value)), Self::ONE);
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
        let low = Self::sub(Self::bitand(inv, Self::neg(inv)), Self::ONE);
        unsafe { arch::_mm512_mask_popcnt_epi64x_v4(value, mask, low) }
    }

    fn trailing_ones_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let inv = Self::not(value);
        let low = Self::sub(Self::bitand(inv, Self::neg(inv)), Self::ONE);
        unsafe { arch::_mm512_mask_popcnt_epi64x_v4(src, mask, low) }
    }

    fn trailing_ones_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let inv = Self::not(value);
        let low = Self::sub(Self::bitand(inv, Self::neg(inv)), Self::ONE);
        unsafe { arch::_mm512_maskz_popcnt_epi64x_v4(mask, low) }
    }
}

#[thermite_macros::inline_always]
impl SignedIntegerRegister for I64x8V4 {
    fn srai<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_sra_epi64(value, arch::_mm_cvtsi32_si128(IMM8)) }
    }

    fn sra(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm512_sra_epi64(value, arch::_mm_cvtsi32_si128(shift as i32)) }
    }

    fn srav(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm512_srav_epi64(value, shifts) }
    }

    // --- masked variants -----------------------------------------------------

    masked_shift_v4! {
        sra => _mm512_mask_sra_epi64, _mm512_maskz_sra_epi64;
    }

    masked_shifti_v4! {
        srai => _mm512_mask_sra_epi64, _mm512_maskz_sra_epi64;
    }

    masked_binary_v4! {
        srav(Storage<Self::Unsigned>) => _mm512_mask_srav_epi64, _mm512_maskz_srav_epi64;
    }

    // avg_floor = (a & b) + ((a ^ b) >>a 1), avg_ceil = (a | b) - ((a ^ b) >>a 1):
    // the final add/sub takes the mask.
    fn avg_floor_c(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_add_epi64(a, mask, Self::bitand(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }

    fn avg_floor_m(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_add_epi64(src, mask, Self::bitand(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }

    fn avg_floor_z(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_add_epi64(mask, Self::bitand(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }

    fn avg_ceil_c(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_epi64(a, mask, Self::bitor(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }

    fn avg_ceil_m(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_epi64(src, mask, Self::bitor(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }

    fn avg_ceil_z(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_sub_epi64(mask, Self::bitor(a, b), Self::srai::<1>(Self::bitxor(a, b))) }
    }

    // mulhrs reconstructs from mulhi/mullo and ends in a rounding add whose
    // operands are both derived. One merge move after it.
    masked_via_mov_v4! {
        mov = _mm512_mask_mov_epi64, movz = _mm512_maskz_mov_epi64;
        mulhrs(a: Storage<Self>, b: Storage<Self>);
    }
}

// Narrowing cast for the cast matrix: [i64x8; 2] -> i32x16. The widening
// direction lives in i32x16.rs.
#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<I64x8V4, 2>> for super::I32x16V4 {
    fn cast_from(value: Storage<ArrayRegister<I64x8V4, 2>>) -> Storage<Self> {
        let ArrayRegister([lo, hi]) = value;

        unsafe {
            let lo = arch::_mm512_cvtepi64_epi32(lo);
            let hi = arch::_mm512_cvtepi64_epi32(hi);

            arch::_mm512_inserti32x8::<1>(arch::_mm512_zextsi256_si512(lo), hi)
        }
    }

    // vpmovsqd saturates in hardware, where v3 had to clamp and re-narrow.
    fn saturating_cast_from(value: Storage<ArrayRegister<I64x8V4, 2>>) -> Storage<Self> {
        let ArrayRegister([lo, hi]) = value;

        unsafe {
            let lo = arch::_mm512_cvtsepi64_epi32(lo);
            let hi = arch::_mm512_cvtsepi64_epi32(hi);

            arch::_mm512_inserti32x8::<1>(arch::_mm512_zextsi256_si512(lo), hi)
        }
    }
}
