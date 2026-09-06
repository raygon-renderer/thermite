//! `f64x8` on AVX-512: the native 512-bit double register.
//!
//! The double-precision sibling of [`f32x16`](super::f32x16). That file is the
//! pattern-setter and its module docs state the v4 house rules (no
//! `MaskRegister` impl, opmask compares, embedded-mask `_c`/`_z` overrides,
//! `if const { F::FEATURE }` tier forks, defaults left in place until the M4
//! sweep). Everything here is floor (F/CD/BW/DQ/VL), so this file needs no
//! tier fork.
//!
//! `f64` is the widest element thermite has, so `ExtendedPrecision = Self` and
//! there is no widening `CastRegister` leg at the bottom of the file (v3's
//! `f64x4.rs` makes the same call).

use generic_array::{GenericArray, sequence::GenericSequence, typenum::Unsigned};

use crate::register::{
    BitwiseRegister, CoreRegister, FloatRegister, IndexableRegister, InterleaveRegister, NativeCapability,
    NumericRegister, PartialOrdRegister, Register, ShuffleRegister, SignedRegister, Storage, ZeroUpper, empty_reg, reg,
};

use super::arch;
use super::kmask::KMask8;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct F64x8V4;

#[thermite_macros::inline_always]
impl CoreRegister for F64x8V4 {
    type Lanes = generic_array::typenum::U8;
    type Storage = arch::__m512d;
    type Mask = KMask8;

    const IS_EMULATED: bool = false;

    const EMPTY: Storage<Self> = empty_reg::<Self>();

    const HAS_EQUAL_SIZE_MASK: bool = false;

    fn blendv(mask: Storage<Self::Mask>, on_false: Storage<Self>, on_true: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_blend_pd(mask, on_false, on_true) }
    }

    fn zz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_mov_pd(mask, value) }
    }

    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_mov_pd(!mask, value) }
    }

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        if const { Z::N >= 8 } {
            value
        } else {
            // Zero-masked move with a constant prefix mask: one instruction
            // for every N, no special 128/256 cases needed.
            unsafe { arch::_mm512_maskz_mov_pd(const { ((1u32 << Z::N) - 1) as u8 }, value) }
        }
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        // vpmovm2q: opmask -> all-ones/all-zeros qword lanes (DQ, floor).
        unsafe { arch::_mm512_castsi512_pd(arch::_mm512_movm_epi64(mask)) }
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for F64x8V4 {
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_xor_pd(lhs, rhs) }
    }

    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_and_pd(lhs, rhs) }
    }

    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_andnot_pd(rhs, lhs) }
    }

    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_or_pd(lhs, rhs) }
    }

    fn not(value: Storage<Self>) -> Storage<Self> {
        // vpternlog imm 0x55 = NOT(a): one instruction, no constant load.
        unsafe {
            arch::_mm512_castsi512_pd(arch::_mm512_ternarylogic_epi64::<{ crate::ternlog_imm!(!C) }>(
                arch::_mm512_castpd_si512(value),
                arch::_mm512_castpd_si512(value),
                arch::_mm512_castpd_si512(value),
            ))
        }
    }

    const HAS_NATIVE_TERNLOG: bool = true;

    fn ternlog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe {
            arch::_mm512_castsi512_pd(arch::_mm512_ternarylogic_epi64::<IMM>(
                arch::_mm512_castpd_si512(a),
                arch::_mm512_castpd_si512(b),
                arch::_mm512_castpd_si512(c),
            ))
        }
    }

    fn bilog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        arch::bilog_ternlog::<Self, IMM>(a, b)
    }

    // --- masked variants -----------------------------------------------------
    // DQ (floor) has the double-typed forms of the bitwise ops, so no casts.

    masked_binary_v4! {
        bitxor => _mm512_mask_xor_pd, _mm512_maskz_xor_pd;
        bitand => _mm512_mask_and_pd, _mm512_maskz_and_pd;
        bitor => _mm512_mask_or_pd, _mm512_maskz_or_pd;
    }

    masked_andnot_v4! {
        bitandnot => _mm512_mask_andnot_pd, _mm512_maskz_andnot_pd;
    }

    // vpternlog's merge form keeps operand A (`src` doubles as A), so the
    // NOT imm must read operand C: 0x55.
    fn not_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let v = unsafe { arch::_mm512_castpd_si512(value) };
        unsafe { arch::_mm512_castsi512_pd(arch::_mm512_mask_ternarylogic_epi64::<{ crate::ternlog_imm!(!C) }>(v, mask, v, v)) }
    }

    fn not_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let v = unsafe { arch::_mm512_castpd_si512(value) };
        unsafe { arch::_mm512_castsi512_pd(arch::_mm512_mask_ternarylogic_epi64::<{ crate::ternlog_imm!(!C) }>(arch::_mm512_castpd_si512(src), mask, v, v)) }
    }

    fn not_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let v = unsafe { arch::_mm512_castpd_si512(value) };
        unsafe { arch::_mm512_castsi512_pd(arch::_mm512_maskz_ternarylogic_epi64::<{ crate::ternlog_imm!(!C) }>(mask, v, v, v)) }
    }

    fn ternlog_c<const IMM: i32>(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe {
            arch::_mm512_castsi512_pd(arch::_mm512_mask_ternarylogic_epi64::<IMM>(
                arch::_mm512_castpd_si512(a),
                mask,
                arch::_mm512_castpd_si512(b),
                arch::_mm512_castpd_si512(c),
            ))
        }
    }

    // The merge form's `src` is also operand A, so an arbitrary `src` costs
    // a separate merge move.
    fn ternlog_m<const IMM: i32>(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mov_pd(src, mask, Self::ternlog::<IMM>(a, b, c)) }
    }

    fn ternlog_z<const IMM: i32>(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe {
            arch::_mm512_castsi512_pd(arch::_mm512_maskz_ternarylogic_epi64::<IMM>(
                mask,
                arch::_mm512_castpd_si512(a),
                arch::_mm512_castpd_si512(b),
                arch::_mm512_castpd_si512(c),
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

// Width ladder: zmm = two ymm. `vinsertf64x4` into a zero-extended low half.
#[thermite_macros::inline_always]
impl crate::register::ConcatRegister<super::F64x4V4> for F64x8V4 {
    fn concat(lo: Storage<super::F64x4V4>, hi: Storage<super::F64x4V4>) -> Storage<Self> {
        unsafe { arch::_mm512_insertf64x4::<1>(arch::_mm512_zextpd256_pd512(lo), hi) }
    }

    fn split(value: Storage<Self>) -> (Storage<super::F64x4V4>, Storage<super::F64x4V4>) {
        let lo = unsafe { arch::_mm512_castpd512_pd256(value) };
        let hi = unsafe { arch::_mm512_extractf64x4_pd::<1>(value) };
        (lo, hi)
    }
}

#[thermite_macros::inline_always]
impl crate::register::ExtendRegister<super::F64x4V4> for F64x8V4 {
    fn extend(value: Storage<super::F64x4V4>) -> Storage<Self> {
        unsafe { arch::_mm512_zextpd256_pd512(value) }
    }

    fn narrow(value: Storage<Self>) -> Storage<super::F64x4V4> {
        unsafe { arch::_mm512_castpd512_pd256(value) }
    }
}

#[thermite_macros::inline_always]
impl Register for F64x8V4 {
    type Element = f64;

    type Signed = super::I64x8V4;
    type Unsigned = super::U64x8V4;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        // k = lanes with any bit set (value != +0.0 bitwise), one vptestmq.
        unsafe { arch::_mm512_test_epi64_mask(arch::_mm512_castpd_si512(value), arch::_mm512_castpd_si512(value)) }
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::msb_to_mask(value)
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        // vpmovq2m (DQ): sign bit of each qword lane straight into k.
        unsafe { arch::_mm512_movepi64_mask(arch::_mm512_castpd_si512(value)) }
    }

    fn new(value: GenericArray<f64, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm512_loadu_pd(value.as_ptr()) }
    }

    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_zextpd128_pd512(arch::_mm_set_sd(value)) }
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_set1_pd(value) }
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_load_pd(ptr) }
    }

    unsafe fn load_m(src: Storage<Self>, mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_mask_loadu_pd(src, mask, ptr) }
    }

    unsafe fn load_z(mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_loadu_pd(mask, ptr) }
    }

    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_loadu_pd(ptr) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm512_store_pd(ptr, value) }
    }

    unsafe fn store_masked(ptr: *mut Self::Element, mask: Storage<Self::Mask>, value: Storage<Self>) {
        unsafe { arch::_mm512_mask_storeu_pd(ptr, mask, value) }
    }

    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm512_storeu_pd(ptr, value) }
    }

    unsafe fn load_stream(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_castsi512_pd(arch::_mm512_stream_load_si512(ptr as _)) }
    }

    unsafe fn store_stream(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm512_stream_pd(ptr, value) }
    }

    unsafe fn lookup(values: &[Self::Element], indices: Storage<Self::Unsigned>) -> Storage<Self> {
        if values.len() <= <Self::Lanes as Unsigned>::USIZE {
            let mut padded: GenericArray<Self::Element, Self::Lanes> = unsafe { core::mem::zeroed() };
            padded[..values.len()].copy_from_slice(values);

            unsafe { arch::_mm512_permutexvar_pd(indices, Self::new(padded)) }
        } else {
            unsafe { <Self as IndexableRegister<Self::Unsigned>>::gather(values.as_ptr(), indices) }
        }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        unsafe {
            let idx = arch::_mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0);
            arch::_mm512_permutexvar_pd(idx, value)
        }
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        // In-lane byte reversal: vpshufb (BW at 512-bit) with the 128-bit
        // qword-bswap pattern broadcast to every block.
        unsafe {
            let pattern = arch::_mm512_broadcast_i32x4(arch::_mm_setr_epi8(
                7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8,
            ));
            arch::_mm512_castsi512_pd(arch::_mm512_shuffle_epi8(arch::_mm512_castpd_si512(value), pattern))
        }
    }

    compress_expand_v4!(
        u8,
        _mm512_maskz_compress_pd,
        _mm512_mask_expand_pd,
        _mm512_maskz_expand_pd
    );

    const HAS_PERMUTEV: bool = true;

    fn permutev(value: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        // The index register already holds the qword lanes `vpermpd`
        // addresses: one permute, no widen.
        unsafe { arch::_mm512_permutexvar_pd(idxs, value) }
    }

    fn swizzle(a: Storage<Self>, b: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        // vpermt2pd: two-source variable permute is a single native
        // instruction (indices 0..7 pick from `a`, 8..15 from `b`), the exact
        // trait semantics, vs the emulation v3 needs.
        unsafe { arch::_mm512_permutex2var_pd(a, idxs, b) }
    }

    // --- masked variants -----------------------------------------------------

    fn new_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: GenericArray<f64, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_loadu_pd(src, mask, value.as_ptr()) }
    }

    fn new_z(mask: Storage<Self::Mask>, value: GenericArray<f64, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_loadu_pd(mask, value.as_ptr()) }
    }

    // vbroadcastsd takes the mask directly.
    fn splat_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_mask_broadcastsd_pd(src, mask, arch::_mm_set_sd(value)) }
    }

    fn splat_z(mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_broadcastsd_pd(mask, arch::_mm_set_sd(value)) }
    }

    // Lane broadcast: one vpermpd with a constant index, instead of the
    // default's extract-to-scalar round trip. The masked forms ride the
    // same instruction.
    fn broadcast<const I: usize>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_permutexvar_pd(arch::_mm512_set1_epi64(I as i64), value) }
    }

    fn broadcast_c<const I: usize>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_permutexvar_pd(value, mask, arch::_mm512_set1_epi64(I as i64), value) }
    }

    fn broadcast_m<const I: usize>(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_permutexvar_pd(src, mask, arch::_mm512_set1_epi64(I as i64), value) }
    }

    fn broadcast_z<const I: usize>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_permutexvar_pd(mask, arch::_mm512_set1_epi64(I as i64), value) }
    }

    fn broadcastv(value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm512_permutexvar_pd(arch::_mm512_set1_epi64(idx as i64), value) }
    }

    fn broadcastv_c(mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm512_mask_permutexvar_pd(value, mask, arch::_mm512_set1_epi64(idx as i64), value) }
    }

    fn broadcastv_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm512_mask_permutexvar_pd(src, mask, arch::_mm512_set1_epi64(idx as i64), value) }
    }

    fn broadcastv_z(mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_permutexvar_pd(mask, arch::_mm512_set1_epi64(idx as i64), value) }
    }

    fn reverse_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe {
            let idx = arch::_mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0);
            arch::_mm512_mask_permutexvar_pd(value, mask, idx, value)
        }
    }

    fn reverse_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe {
            let idx = arch::_mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0);
            arch::_mm512_mask_permutexvar_pd(src, mask, idx, value)
        }
    }

    fn reverse_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe {
            let idx = arch::_mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0);
            arch::_mm512_maskz_permutexvar_pd(mask, idx, value)
        }
    }

    // vpshufb is masked per BYTE (`__mmask64`), so the lane mask cannot feed
    // it directly: shuffle, then one merge move.
    masked_via_mov_v4! {
        mov = _mm512_mask_mov_pd, movz = _mm512_maskz_mov_pd;
        swap_bytes(value: Storage<Self>);
    }

    fn permutev_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_permutexvar_pd(src, mask, idxs, value) }
    }

    fn permutev_z(mask: Storage<Self::Mask>, value: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_permutexvar_pd(mask, idxs, value) }
    }

    // vpermt2pd merges only into `a` (vpermi2pd into the index), neither of
    // which is an arbitrary `src`: permute, then one merge move. The zeroing
    // form is native.
    fn swizzle_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        a: Storage<Self>,
        b: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mov_pd(src, mask, arch::_mm512_permutex2var_pd(a, idxs, b)) }
    }

    fn swizzle_z(
        mask: Storage<Self::Mask>,
        a: Storage<Self>,
        b: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_permutex2var_pd(mask, a, idxs, b) }
    }
}

#[thermite_macros::inline_always]
impl InterleaveRegister for F64x8V4 {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        // One vpermt2pd per output: no unpack + cross-lane fixup dance.
        unsafe {
            let lo = arch::_mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
            let hi = arch::_mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
            (
                arch::_mm512_permutex2var_pd(a, lo, b),
                arch::_mm512_permutex2var_pd(a, hi, b),
            )
        }
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe {
            let even = arch::_mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
            let odd = arch::_mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
            (
                arch::_mm512_permutex2var_pd(a, even, b),
                arch::_mm512_permutex2var_pd(a, odd, b),
            )
        }
    }
}

#[thermite_macros::inline_always]
impl IndexableRegister<super::U64x8V4> for F64x8V4 {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<super::U64x8V4>) -> Storage<Self> {
        unsafe { arch::_mm512_i64gather_pd::<8>(indices, ptr as _) }
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<super::U64x8V4>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_i64gather_pd::<8>(src, mask, indices, ptr as _) }
    }

    unsafe fn scatter(value: Storage<Self>, ptr: *mut Self::Element, indices: Storage<super::U64x8V4>) {
        unsafe { arch::_mm512_i64scatter_pd::<8>(ptr as _, indices, value) }
    }

    unsafe fn scatter_m(
        value: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *mut Self::Element,
        indices: Storage<super::U64x8V4>,
    ) {
        unsafe { arch::_mm512_mask_i64scatter_pd::<8>(ptr as _, mask, indices, value) }
    }
}

// Eight 32-bit indices are one ymm (`u32x8`): vgatherdpd / vscatterdpd.
// The x16 grid slot (two zmm): sixteen 32-bit indices in one zmm, split into
// the two ymm index halves of vgatherdpd / vscatterdpd.
#[thermite_macros::inline_always]
impl IndexableRegister<super::U32x16V4> for crate::register::array::ArrayRegister<F64x8V4, 2> {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<super::U32x16V4>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm512_i32gather_pd::<8>(arch::_mm512_castsi512_si256(indices), ptr as _);
            let hi = arch::_mm512_i32gather_pd::<8>(arch::_mm512_extracti32x8_epi32::<1>(indices), ptr as _);

            crate::register::array::ArrayRegister([lo, hi])
        }
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<super::U32x16V4>,
    ) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm512_mask_i32gather_pd::<8>(
                src.0[0],
                mask.0[0],
                arch::_mm512_castsi512_si256(indices),
                ptr as _,
            );
            let hi = arch::_mm512_mask_i32gather_pd::<8>(
                src.0[1],
                mask.0[1],
                arch::_mm512_extracti32x8_epi32::<1>(indices),
                ptr as _,
            );

            crate::register::array::ArrayRegister([lo, hi])
        }
    }

    unsafe fn scatter(value: Storage<Self>, ptr: *mut Self::Element, indices: Storage<super::U32x16V4>) {
        unsafe {
            arch::_mm512_i32scatter_pd::<8>(ptr as _, arch::_mm512_castsi512_si256(indices), value.0[0]);
            arch::_mm512_i32scatter_pd::<8>(ptr as _, arch::_mm512_extracti32x8_epi32::<1>(indices), value.0[1]);
        }
    }

    unsafe fn scatter_m(
        value: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *mut Self::Element,
        indices: Storage<super::U32x16V4>,
    ) {
        unsafe {
            arch::_mm512_mask_i32scatter_pd::<8>(
                ptr as _,
                mask.0[0],
                arch::_mm512_castsi512_si256(indices),
                value.0[0],
            );
            arch::_mm512_mask_i32scatter_pd::<8>(
                ptr as _,
                mask.0[1],
                arch::_mm512_extracti32x8_epi32::<1>(indices),
                value.0[1],
            );
        }
    }
}

#[thermite_macros::inline_always]
impl IndexableRegister<super::U32x8V4> for F64x8V4 {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<super::U32x8V4>) -> Storage<Self> {
        unsafe { arch::_mm512_i32gather_pd::<8>(indices, ptr) }
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<super::U32x8V4>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_i32gather_pd::<8>(src, mask, indices, ptr) }
    }

    unsafe fn scatter(value: Storage<Self>, ptr: *mut Self::Element, indices: Storage<super::U32x8V4>) {
        unsafe { arch::_mm512_i32scatter_pd::<8>(ptr, indices, value) }
    }

    unsafe fn scatter_m(
        value: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *mut Self::Element,
        indices: Storage<super::U32x8V4>,
    ) {
        unsafe { arch::_mm512_mask_i32scatter_pd::<8>(ptr, mask, indices, value) }
    }
}

#[thermite_macros::inline_always]
impl ShuffleRegister for F64x8V4 {
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_shuffle_pd(lhs, rhs, IMM8) }
    }
}

#[thermite_macros::inline_always]
impl PartialOrdRegister for F64x8V4 {
    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmp_pd_mask(lhs, rhs, arch::_CMP_LT_OQ) }
    }

    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmp_pd_mask(lhs, rhs, arch::_CMP_LE_OQ) }
    }

    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmp_pd_mask(lhs, rhs, arch::_CMP_GT_OQ) }
    }

    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmp_pd_mask(lhs, rhs, arch::_CMP_GE_OQ) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmp_pd_mask(lhs, rhs, arch::_CMP_EQ_OQ) }
    }

    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm512_cmp_pd_mask(lhs, rhs, arch::_CMP_NEQ_UQ) }
    }
}

#[thermite_macros::inline_always]
impl NumericRegister for F64x8V4 {
    const ZERO: Storage<Self> = reg::<Self, 8>([0.0; 8]);
    const ONE: Storage<Self> = reg::<Self, 8>([1.0; 8]);
    const TWO: Storage<Self> = reg::<Self, 8>([2.0; 8]);

    const MIN: Storage<Self> = reg::<Self, 8>([f64::MIN; 8]);
    const MAX: Storage<Self> = reg::<Self, 8>([f64::MAX; 8]);

    fn min_element(value: Storage<Self>) -> Self::Element {
        unsafe { arch::_mm512_reduce_min_pd(value) }
    }

    fn max_element(value: Storage<Self>) -> Self::Element {
        unsafe { arch::_mm512_reduce_max_pd(value) }
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element {
        unsafe { arch::_mm512_reduce_add_pd(value) }
    }

    fn prod_elements(value: Storage<Self>) -> Self::Element {
        unsafe { arch::_mm512_reduce_mul_pd(value) }
    }

    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as Unsigned>::USIZE as f64)
    }

    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as f64))
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_add_pd(lhs, rhs) }
    }

    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_sub_pd(lhs, rhs) }
    }

    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mul_pd(lhs, rhs) }
    }

    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_div_pd(lhs, rhs) }
    }

    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::nmul_adde(Self::trunc(Self::div(lhs, rhs)), rhs, lhs)
    }

    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::fix_min::<Self>(lhs, rhs, unsafe { arch::_mm512_min_pd(lhs, rhs) })
    }

    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::fix_max::<Self>(lhs, rhs, unsafe { arch::_mm512_max_pd(lhs, rhs) })
    }

    // --- masked variants -----------------------------------------------------
    // Embedded-mask forms: keep `lhs`/`src` where the mask is false, one
    // instruction, no blend.

    masked_binary_v4! {
        add => _mm512_mask_add_pd, _mm512_maskz_add_pd;
        sub => _mm512_mask_sub_pd, _mm512_maskz_sub_pd;
        mul => _mm512_mask_mul_pd, _mm512_maskz_mul_pd;
        div => _mm512_mask_div_pd, _mm512_maskz_div_pd;
    }

    // Under strict IEEE the NaN/signed-zero fixups wrap the raw instruction,
    // so the mask has to apply to the fixed result.
    cfg_select! {
        feature = "strict_ieee754" => {
            masked_via_mov_v4! {
                mov = _mm512_mask_mov_pd, movz = _mm512_maskz_mov_pd;
                min(lhs: Storage<Self>, rhs: Storage<Self>);
                max(lhs: Storage<Self>, rhs: Storage<Self>);
            }
        }
        _ => {
            masked_binary_v4! {
                min => _mm512_mask_min_pd, _mm512_maskz_min_pd;
                max => _mm512_mask_max_pd, _mm512_maskz_max_pd;
            }
        }
    }

    // rem = lhs - trunc(lhs / rhs) * rhs: the final fnmadd takes the mask,
    // and its `mask3` form merges into the addend, which IS `lhs`.
    fn rem_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask3_fnmadd_pd(Self::trunc(Self::div(lhs, rhs)), rhs, lhs, mask) }
    }

    fn rem_m(src: Storage<Self>, mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mov_pd(src, mask, Self::rem(lhs, rhs)) }
    }

    fn rem_z(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_fnmadd_pd(mask, Self::trunc(Self::div(lhs, rhs)), rhs, lhs) }
    }

    fn square_c(mask: Storage<Self::Mask>, lhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mul_pd(lhs, mask, lhs, lhs) }
    }

    fn square_m(src: Storage<Self>, mask: Storage<Self::Mask>, lhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mul_pd(src, mask, lhs, lhs) }
    }

    fn square_z(mask: Storage<Self::Mask>, lhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_mul_pd(mask, lhs, lhs) }
    }

    fn scale_c(mask: Storage<Self::Mask>, value: Storage<Self>, scalar: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mul_pd(value, mask, value, Self::splat(scalar)) }
    }

    fn scale_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
        scalar: Self::Element,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mul_pd(src, mask, value, Self::splat(scalar)) }
    }

    fn scale_z(mask: Storage<Self::Mask>, value: Storage<Self>, scalar: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_mul_pd(mask, value, Self::splat(scalar)) }
    }
}

#[thermite_macros::inline_always]
impl SignedRegister for F64x8V4 {
    const NEG_ONE: Storage<Self> = reg::<Self, 8>([-1.0; 8]);
    const MIN_POSITIVE: Storage<Self> = reg::<Self, 8>([f64::MIN_POSITIVE; 8]);

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
        unsafe { arch::_mm512_mask_xor_pd(value, mask, value, Self::NEG_ZERO) }
    }

    fn neg_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_xor_pd(src, mask, value, Self::NEG_ZERO) }
    }

    fn neg_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_xor_pd(mask, value, Self::NEG_ZERO) }
    }

    fn abs_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_andnot_pd(value, mask, Self::NEG_ZERO, value) }
    }

    fn abs_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_andnot_pd(src, mask, Self::NEG_ZERO, value) }
    }

    fn abs_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_andnot_pd(mask, Self::NEG_ZERO, value) }
    }

    // The merge form keeps operand A, so put `lhs` first and select on C:
    // imm 0xD8 = `c ? b : a` (0xCA is `a ? b : c`).
    fn copysign_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe {
            arch::_mm512_castsi512_pd(arch::_mm512_mask_ternarylogic_epi64::<
                { crate::ternlog_imm!(C & B | !C & A) },
            >(
                arch::_mm512_castpd_si512(lhs),
                mask,
                arch::_mm512_castpd_si512(rhs),
                arch::_mm512_castpd_si512(Self::NEG_ZERO),
            ))
        }
    }

    fn copysign_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        lhs: Storage<Self>,
        rhs: Storage<Self>,
    ) -> Storage<Self> {
        unsafe { arch::_mm512_mask_mov_pd(src, mask, Self::copysign(lhs, rhs)) }
    }

    fn copysign_z(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::ternlog_z::<0xCA>(mask, Self::NEG_ZERO, rhs, lhs)
    }
}

#[thermite_macros::inline_always]
impl FloatRegister for F64x8V4 {
    const HAS_NATIVE_FMA: tribool::Tribool = tribool::True;

    type Bits = super::U64x8V4;
    type SignedBits = super::I64x8V4;
    type ExtendedPrecision = Self; // f64 is the highest precision available

    const HALF: Storage<Self> = reg::<Self, 8>([0.5; 8]);
    const NEG_ZERO: Storage<Self> = reg::<Self, 8>([-0.0; 8]);
    const EPSILON: Storage<Self> = reg::<Self, 8>([f64::EPSILON; 8]);
    const INFINITY: Storage<Self> = reg::<Self, 8>([f64::INFINITY; 8]);
    const NEG_INFINITY: Storage<Self> = reg::<Self, 8>([f64::NEG_INFINITY; 8]);
    const NAN: Storage<Self> = reg::<Self, 8>([f64::NAN; 8]);

    const EXP_MASK: Storage<Self::Bits> = reg::<Self::Bits, 8>([0x7FF0_0000_0000_0000; 8]);

    fn mul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_fmadd_pd(lhs, rhs, acc) }
    }

    fn mul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_fmsub_pd(lhs, rhs, acc) }
    }

    fn nmul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_fnmadd_pd(lhs, rhs, acc) }
    }

    fn nmul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_fnmsub_pd(lhs, rhs, acc) }
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
        unsafe { arch::_mm512_fmaddsub_pd(a, b, c) }
    }

    fn fmsubadd(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_fmsubadd_pd(a, b, c) }
    }

    fn sqrt(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_sqrt_pd(value) }
    }

    fn rsqrt(value: Storage<Self>) -> Storage<Self> {
        cfg_select! {
            feature = "strict_ieee754" => {
                Self::rcp(Self::sqrt(value))
            }
            _ => unsafe { arch::_mm512_rsqrt14_pd(value) }
        }
    }

    fn rcp(value: Storage<Self>) -> Storage<Self> {
        cfg_select! {
            feature = "strict_ieee754" => {
                Self::div(Self::ONE, value)
            }
            _ => unsafe { arch::_mm512_rcp14_pd(value) }
        }
    }

    // vrcp14/vrsqrt14: 2^-14 relative error, up from the pre-AVX512 12-bit
    // estimates. Newton-step counts in kernels keyed on element type still
    // hold (they assume the WORSE 12-bit seed), just with extra margin. Note
    // this is the first `f64` register in the tree with approximations at all.
    // The v3 `f64x4` reports `false` for both.
    const HAS_APPROX_RSQRT: bool = cfg!(not(feature = "strict_ieee754"));
    const HAS_APPROX_RCP: bool = cfg!(not(feature = "strict_ieee754"));

    // vrndscale imm[3:0]: bit 3 suppresses precision exceptions, bits 1:0 are
    // the rounding mode (00 nearest-even, 01 down, 10 up, 11 truncate).
    fn floor(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_roundscale_pd(value, 0x09) }
    }

    fn ceil(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_roundscale_pd(value, 0x0A) }
    }

    fn round(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_roundscale_pd(value, 0x08) }
    }

    fn trunc(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_roundscale_pd(value, 0x0B) }
    }

    fn next_up(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_nextuppd_v4(value) }
    }

    fn next_down(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_nextdownpd_v4(value) }
    }

    const NATIVE_CAP: NativeCapability = NativeCapability::NONE;

    // --- masked variants -----------------------------------------------------

    // vfmadd's merge form keeps the first multiplicand, which is exactly the
    // `_c` contract. The estimating `e` family is the same instruction here.
    masked_fma_v4! {
        mov = _mm512_mask_mov_pd;
        mul_add => _mm512_mask_fmadd_pd, _mm512_maskz_fmadd_pd;
        mul_sub => _mm512_mask_fmsub_pd, _mm512_maskz_fmsub_pd;
        nmul_add => _mm512_mask_fnmadd_pd, _mm512_maskz_fnmadd_pd;
        nmul_sub => _mm512_mask_fnmsub_pd, _mm512_maskz_fnmsub_pd;
        mul_adde => _mm512_mask_fmadd_pd, _mm512_maskz_fmadd_pd;
        mul_sube => _mm512_mask_fmsub_pd, _mm512_maskz_fmsub_pd;
        nmul_adde => _mm512_mask_fnmadd_pd, _mm512_maskz_fnmadd_pd;
        nmul_sube => _mm512_mask_fnmsub_pd, _mm512_maskz_fnmsub_pd;
        fmaddsub => _mm512_mask_fmaddsub_pd, _mm512_maskz_fmaddsub_pd;
        fmsubadd => _mm512_mask_fmsubadd_pd, _mm512_maskz_fmsubadd_pd;
    }

    // addsub = a + (b ^ ALT_NEG): the xor stays unmasked, the add takes it.
    fn addsub_c(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_add_pd(a, mask, a, Self::bitxor(b, Self::ALT_NEG)) }
    }

    fn addsub_m(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_add_pd(src, mask, a, Self::bitxor(b, Self::ALT_NEG)) }
    }

    fn addsub_z(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_add_pd(mask, a, Self::bitxor(b, Self::ALT_NEG)) }
    }

    masked_unary_v4! {
        sqrt => _mm512_mask_sqrt_pd, _mm512_maskz_sqrt_pd;
        floor => _mm512_mask_roundscale_pd, _mm512_maskz_roundscale_pd, 0x09;
        ceil => _mm512_mask_roundscale_pd, _mm512_maskz_roundscale_pd, 0x0A;
        round => _mm512_mask_roundscale_pd, _mm512_maskz_roundscale_pd, 0x08;
        trunc => _mm512_mask_roundscale_pd, _mm512_maskz_roundscale_pd, 0x0B;
    }

    // Same fork as `rcp`/`rsqrt`: exact division under strict IEEE (the
    // divide takes the mask, the sqrt underneath rsqrt runs unmasked), the
    // 14-bit estimates otherwise.
    cfg_select! {
        feature = "strict_ieee754" => {
            fn rcp_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::_mm512_mask_div_pd(value, mask, Self::ONE, value) }
            }

            fn rcp_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::_mm512_mask_div_pd(src, mask, Self::ONE, value) }
            }

            fn rcp_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::_mm512_maskz_div_pd(mask, Self::ONE, value) }
            }

            fn rsqrt_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::_mm512_mask_div_pd(value, mask, Self::ONE, Self::sqrt(value)) }
            }

            fn rsqrt_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::_mm512_mask_div_pd(src, mask, Self::ONE, Self::sqrt(value)) }
            }

            fn rsqrt_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::_mm512_maskz_div_pd(mask, Self::ONE, Self::sqrt(value)) }
            }
        }
        _ => {
            masked_unary_v4! {
                rcp => _mm512_mask_rcp14_pd, _mm512_maskz_rcp14_pd;
                rsqrt => _mm512_mask_rsqrt14_pd, _mm512_maskz_rsqrt14_pd;
            }
        }
    }

    // fract = value - trunc(value): the subtract takes the mask.
    fn fract_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_pd(value, mask, value, Self::trunc(value)) }
    }

    fn fract_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_sub_pd(src, mask, value, Self::trunc(value)) }
    }

    fn fract_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_sub_pd(mask, value, Self::trunc(value)) }
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
        unsafe { arch::_mm512_mask_mov_pd(src, mask, Self::mul_sign(value, sign)) }
    }

    fn mul_sign_z(mask: Storage<Self::Mask>, value: Storage<Self>, sign: Storage<Self>) -> Storage<Self> {
        Self::ternlog_z::<0x78>(mask, value, sign, Self::NEG_ZERO)
    }

    fn signed_zero_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_and_pd(value, mask, Self::NEG_ZERO, value) }
    }

    fn signed_zero_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_mask_and_pd(src, mask, Self::NEG_ZERO, value) }
    }

    fn signed_zero_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm512_maskz_and_pd(mask, Self::NEG_ZERO, value) }
    }

    // The next_up/next_down polyfills end in a k-masked add, but the mask
    // there is the sign-split, not ours: run them, then one merge move.
    masked_via_mov_v4! {
        mov = _mm512_mask_mov_pd, movz = _mm512_maskz_mov_pd;
        next_up(value: Storage<Self>);
        next_down(value: Storage<Self>);
    }
}
