//! `f32x4` on AVX-512: the 128-bit float register under EVEX (AVX512VL).
//! The narrowest rung of the [`f32x16`](super::f32x16) pattern. See that
//! file for the house rules and [`f32x8`](super::f32x8) for the width notes.
//!
//! xmm-specific spellings: there is no `vpermps` at 128 bits, so the
//! single-source permutes are `vpermilps` (`_mm_permutevar_ps`, whose masked
//! forms exist) and the fixed reverse is `vpermilps imm`. The two-source
//! `vpermt2ps` xmm form does exist and carries `swizzle`. Element reductions
//! use the v2 `_mm_reduce_ps_v2!` fold. `ExtendedPrecision` is the native
//! `f64x4` (ymm), one `vcvtps2pd`/`vcvtpd2ps` each way. 64-bit-index
//! gathers take the four indices in one ymm (`u64x4`).

use generic_array::{GenericArray, sequence::GenericSequence, typenum::Unsigned};

use crate::register::{
    BitwiseRegister, CastRegister, CoreRegister, FloatRegister, IndexableRegister, InterleaveRegister,
    NativeCapability, NumericRegister, PartialOrdRegister, Register, ShuffleRegister, SignedRegister, Storage,
    ZeroUpper, empty_reg, reg,
};

use super::arch;
use super::kmask::KMask4;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct F32x4V4;

#[thermite_macros::inline_always]
impl CoreRegister for F32x4V4 {
    type Lanes = generic_array::typenum::U4;
    type Storage = arch::__m128;
    type Mask = KMask4;

    const IS_EMULATED: bool = false;

    const EMPTY: Storage<Self> = empty_reg::<Self>();

    const HAS_EQUAL_SIZE_MASK: bool = false;

    fn blendv(mask: Storage<Self::Mask>, on_false: Storage<Self>, on_true: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_blend_ps(mask, on_false, on_true) }
    }

    fn zz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_mov_ps(mask, value) }
    }

    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_mov_ps(!mask, value) }
    }

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        if const { Z::N >= 4 } {
            value
        } else {
            unsafe { arch::_mm_maskz_mov_ps(const { ((1u32 << Z::N) - 1) as u8 }, value) }
        }
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        unsafe { arch::_mm_castsi128_ps(arch::_mm_movm_epi32(mask)) }
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for F32x4V4 {
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_xor_ps(lhs, rhs) }
    }

    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_and_ps(lhs, rhs) }
    }

    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_andnot_ps(rhs, lhs) }
    }

    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_or_ps(lhs, rhs) }
    }

    fn not(value: Storage<Self>) -> Storage<Self> {
        unsafe {
            arch::_mm_castsi128_ps(arch::_mm_ternarylogic_epi32::<{ crate::ternlog_imm!(!C) }>(
                arch::_mm_castps_si128(value),
                arch::_mm_castps_si128(value),
                arch::_mm_castps_si128(value),
            ))
        }
    }

    const HAS_NATIVE_TERNLOG: bool = true;

    fn ternlog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe {
            arch::_mm_castsi128_ps(arch::_mm_ternarylogic_epi32::<IMM>(
                arch::_mm_castps_si128(a),
                arch::_mm_castps_si128(b),
                arch::_mm_castps_si128(c),
            ))
        }
    }

    fn bilog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        arch::bilog_ternlog::<Self, IMM>(a, b)
    }

    // --- masked variants -----------------------------------------------------

    masked_binary_v4! {
        bitxor => _mm_mask_xor_ps, _mm_maskz_xor_ps;
        bitand => _mm_mask_and_ps, _mm_maskz_and_ps;
        bitor => _mm_mask_or_ps, _mm_maskz_or_ps;
    }

    masked_andnot_v4! {
        bitandnot => _mm_mask_andnot_ps, _mm_maskz_andnot_ps;
    }

    fn not_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let v = unsafe { arch::_mm_castps_si128(value) };
        unsafe { arch::_mm_castsi128_ps(arch::_mm_mask_ternarylogic_epi32::<{ crate::ternlog_imm!(!C) }>(v, mask, v, v)) }
    }

    fn not_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let v = unsafe { arch::_mm_castps_si128(value) };
        unsafe { arch::_mm_castsi128_ps(arch::_mm_mask_ternarylogic_epi32::<{ crate::ternlog_imm!(!C) }>(arch::_mm_castps_si128(src), mask, v, v)) }
    }

    fn not_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let v = unsafe { arch::_mm_castps_si128(value) };
        unsafe { arch::_mm_castsi128_ps(arch::_mm_maskz_ternarylogic_epi32::<{ crate::ternlog_imm!(!C) }>(mask, v, v, v)) }
    }

    fn ternlog_c<const IMM: i32>(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe {
            arch::_mm_castsi128_ps(arch::_mm_mask_ternarylogic_epi32::<IMM>(
                arch::_mm_castps_si128(a),
                mask,
                arch::_mm_castps_si128(b),
                arch::_mm_castps_si128(c),
            ))
        }
    }

    fn ternlog_m<const IMM: i32>(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_mov_ps(src, mask, Self::ternlog::<IMM>(a, b, c)) }
    }

    fn ternlog_z<const IMM: i32>(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe {
            arch::_mm_castsi128_ps(arch::_mm_maskz_ternarylogic_epi32::<IMM>(
                mask,
                arch::_mm_castps_si128(a),
                arch::_mm_castps_si128(b),
                arch::_mm_castps_si128(c),
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

#[thermite_macros::inline_always]
impl crate::register::WideRegister for F32x4V4 {
    type Wide = super::F32x8V4;
}

#[thermite_macros::inline_always]
impl Register for F32x4V4 {
    type Element = f32;

    type Signed = super::I32x4V4;
    type Unsigned = super::U32x4V4;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_test_epi32_mask(arch::_mm_castps_si128(value), arch::_mm_castps_si128(value)) }
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::msb_to_mask(value)
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_movepi32_mask(arch::_mm_castps_si128(value)) }
    }

    fn new(value: GenericArray<f32, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_loadu_ps(value.as_ptr()) }
    }

    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_set_ss(value) }
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_set1_ps(value) }
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_load_ps(ptr) }
    }

    unsafe fn load_m(src: Storage<Self>, mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_mask_loadu_ps(src, mask, ptr) }
    }

    unsafe fn load_z(mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_maskz_loadu_ps(mask, ptr) }
    }

    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_loadu_ps(ptr) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_store_ps(ptr, value) }
    }

    unsafe fn store_masked(ptr: *mut Self::Element, mask: Storage<Self::Mask>, value: Storage<Self>) {
        unsafe { arch::_mm_mask_storeu_ps(ptr, mask, value) }
    }

    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_storeu_ps(ptr, value) }
    }

    unsafe fn load_stream(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_castsi128_ps(arch::_mm_stream_load_si128(ptr as _)) }
    }

    unsafe fn store_stream(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_stream_ps(ptr, value) }
    }

    unsafe fn lookup(values: &[Self::Element], indices: Storage<Self::Unsigned>) -> Storage<Self> {
        if values.len() <= <Self::Lanes as Unsigned>::USIZE {
            let mut padded: GenericArray<Self::Element, Self::Lanes> = unsafe { core::mem::zeroed() };
            padded[..values.len()].copy_from_slice(values);

            unsafe { arch::_mm_permutevar_ps(Self::new(padded), indices) }
        } else {
            unsafe { <Self as IndexableRegister<Self::Unsigned>>::gather(values.as_ptr(), indices) }
        }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_permute_ps::<0x1B>(value) }
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        unsafe {
            let pattern = arch::_mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
            arch::_mm_castsi128_ps(arch::_mm_shuffle_epi8(arch::_mm_castps_si128(value), pattern))
        }
    }

    compress_expand_v4!(u8, _mm_maskz_compress_ps, _mm_mask_expand_ps, _mm_maskz_expand_ps);

    const HAS_PERMUTEV: bool = true;

    // vpermilps with a register index: the whole xmm is one 128-bit lane,
    // so this IS the full permute at this width.
    fn permutev(value: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_permutevar_ps(value, idxs) }
    }

    fn swizzle(a: Storage<Self>, b: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_permutex2var_ps(a, idxs, b) }
    }

    // --- masked variants -----------------------------------------------------

    fn new_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: GenericArray<f32, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_mask_loadu_ps(src, mask, value.as_ptr()) }
    }

    fn new_z(mask: Storage<Self::Mask>, value: GenericArray<f32, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_loadu_ps(mask, value.as_ptr()) }
    }

    fn splat_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_mask_broadcastss_ps(src, mask, arch::_mm_set_ss(value)) }
    }

    fn splat_z(mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_maskz_broadcastss_ps(mask, arch::_mm_set_ss(value)) }
    }

    fn broadcast<const I: usize>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_permutevar_ps(value, arch::_mm_set1_epi32(I as i32)) }
    }

    fn broadcast_c<const I: usize>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_permutevar_ps(value, mask, value, arch::_mm_set1_epi32(I as i32)) }
    }

    fn broadcast_m<const I: usize>(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mask_permutevar_ps(src, mask, value, arch::_mm_set1_epi32(I as i32)) }
    }

    fn broadcast_z<const I: usize>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_permutevar_ps(mask, value, arch::_mm_set1_epi32(I as i32)) }
    }

    fn broadcastv(value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm_permutevar_ps(value, arch::_mm_set1_epi32(idx as i32)) }
    }

    fn broadcastv_c(mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm_mask_permutevar_ps(value, mask, value, arch::_mm_set1_epi32(idx as i32)) }
    }

    fn broadcastv_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm_mask_permutevar_ps(src, mask, value, arch::_mm_set1_epi32(idx as i32)) }
    }

    fn broadcastv_z(mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm_maskz_permutevar_ps(mask, value, arch::_mm_set1_epi32(idx as i32)) }
    }

    fn reverse_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_permute_ps::<0x1B>(value, mask, value) }
    }

    fn reverse_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_permute_ps::<0x1B>(src, mask, value) }
    }

    fn reverse_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_permute_ps::<0x1B>(mask, value) }
    }

    masked_via_mov_v4! {
        mov = _mm_mask_mov_ps, movz = _mm_maskz_mov_ps;
        swap_bytes(value: Storage<Self>);
    }

    fn permutev_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mask_permutevar_ps(src, mask, value, idxs) }
    }

    fn permutev_z(mask: Storage<Self::Mask>, value: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_permutevar_ps(mask, value, idxs) }
    }

    fn swizzle_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        a: Storage<Self>,
        b: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mask_mov_ps(src, mask, arch::_mm_permutex2var_ps(a, idxs, b)) }
    }

    fn swizzle_z(
        mask: Storage<Self::Mask>,
        a: Storage<Self>,
        b: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_maskz_permutex2var_ps(mask, a, idxs, b) }
    }
}

#[thermite_macros::inline_always]
impl InterleaveRegister for F32x4V4 {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe { (arch::_mm_unpacklo_ps(a, b), arch::_mm_unpackhi_ps(a, b)) }
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe {
            (
                arch::_mm_shuffle_ps(a, b, 0b10_00_10_00),
                arch::_mm_shuffle_ps(a, b, 0b11_01_11_01),
            )
        }
    }
}

#[thermite_macros::inline_always]
impl IndexableRegister<super::U32x4V4> for F32x4V4 {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<super::U32x4V4>) -> Storage<Self> {
        unsafe { arch::_mm_i32gather_ps::<4>(ptr, indices) }
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<super::U32x4V4>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mmask_i32gather_ps::<4>(src, mask, indices, ptr) }
    }

    unsafe fn scatter(value: Storage<Self>, ptr: *mut Self::Element, indices: Storage<super::U32x4V4>) {
        unsafe { arch::_mm_i32scatter_ps::<4>(ptr, indices, value) }
    }

    unsafe fn scatter_m(
        value: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *mut Self::Element,
        indices: Storage<super::U32x4V4>,
    ) {
        unsafe { arch::_mm_mask_i32scatter_ps::<4>(ptr, mask, indices, value) }
    }
}

// Four 64-bit indices are one ymm (`u64x4`).
#[thermite_macros::inline_always]
impl IndexableRegister<super::U64x4V4> for F32x4V4 {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<super::U64x4V4>) -> Storage<Self> {
        unsafe { arch::_mm256_i64gather_ps::<4>(ptr, indices) }
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<super::U64x4V4>,
    ) -> Storage<Self> {
        unsafe { arch::_mm256_mmask_i64gather_ps::<4>(src, mask, indices, ptr) }
    }

    unsafe fn scatter(value: Storage<Self>, ptr: *mut Self::Element, indices: Storage<super::U64x4V4>) {
        unsafe { arch::_mm256_i64scatter_ps::<4>(ptr, indices, value) }
    }

    unsafe fn scatter_m(
        value: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *mut Self::Element,
        indices: Storage<super::U64x4V4>,
    ) {
        unsafe { arch::_mm256_mask_i64scatter_ps::<4>(ptr, mask, indices, value) }
    }
}

#[thermite_macros::inline_always]
impl ShuffleRegister for F32x4V4 {
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_shuffle_ps(lhs, rhs, IMM8) }
    }
}

#[thermite_macros::inline_always]
impl PartialOrdRegister for F32x4V4 {
    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmp_ps_mask(lhs, rhs, arch::_CMP_LT_OQ) }
    }

    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmp_ps_mask(lhs, rhs, arch::_CMP_LE_OQ) }
    }

    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmp_ps_mask(lhs, rhs, arch::_CMP_GT_OQ) }
    }

    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmp_ps_mask(lhs, rhs, arch::_CMP_GE_OQ) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmp_ps_mask(lhs, rhs, arch::_CMP_EQ_OQ) }
    }

    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmp_ps_mask(lhs, rhs, arch::_CMP_NEQ_UQ) }
    }
}

#[thermite_macros::inline_always]
impl NumericRegister for F32x4V4 {
    const ZERO: Storage<Self> = reg::<Self, 4>([0.0; 4]);
    const ONE: Storage<Self> = reg::<Self, 4>([1.0; 4]);
    const TWO: Storage<Self> = reg::<Self, 4>([2.0; 4]);

    const MIN: Storage<Self> = reg::<Self, 4>([f32::MIN; 4]);
    const MAX: Storage<Self> = reg::<Self, 4>([f32::MAX; 4]);

    fn min_element(value: Storage<Self>) -> Self::Element {
        _mm_reduce_ps_v2!(value; _mm_min_ps _mm_min_ss)
    }

    fn max_element(value: Storage<Self>) -> Self::Element {
        _mm_reduce_ps_v2!(value; _mm_max_ps _mm_max_ss)
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element {
        _mm_reduce_ps_v2!(value; _mm_add_ps _mm_add_ss)
    }

    fn prod_elements(value: Storage<Self>) -> Self::Element {
        _mm_reduce_ps_v2!(value; _mm_mul_ps _mm_mul_ss)
    }

    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as Unsigned>::USIZE as f32)
    }

    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as f32))
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_add_ps(lhs, rhs) }
    }

    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sub_ps(lhs, rhs) }
    }

    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mul_ps(lhs, rhs) }
    }

    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_div_ps(lhs, rhs) }
    }

    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::nmul_adde(Self::trunc(Self::div(lhs, rhs)), rhs, lhs)
    }

    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::fix_min::<Self>(lhs, rhs, unsafe { arch::_mm_min_ps(lhs, rhs) })
    }

    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::fix_max::<Self>(lhs, rhs, unsafe { arch::_mm_max_ps(lhs, rhs) })
    }

    // --- masked variants -----------------------------------------------------

    masked_binary_v4! {
        add => _mm_mask_add_ps, _mm_maskz_add_ps;
        sub => _mm_mask_sub_ps, _mm_maskz_sub_ps;
        mul => _mm_mask_mul_ps, _mm_maskz_mul_ps;
        div => _mm_mask_div_ps, _mm_maskz_div_ps;
    }

    cfg_select! {
        feature = "strict_ieee754" => {
            masked_via_mov_v4! {
                mov = _mm_mask_mov_ps, movz = _mm_maskz_mov_ps;
                min(lhs: Storage<Self>, rhs: Storage<Self>);
                max(lhs: Storage<Self>, rhs: Storage<Self>);
            }
        }
        _ => {
            masked_binary_v4! {
                min => _mm_mask_min_ps, _mm_maskz_min_ps;
                max => _mm_mask_max_ps, _mm_maskz_max_ps;
            }
        }
    }

    fn rem_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask3_fnmadd_ps(Self::trunc(Self::div(lhs, rhs)), rhs, lhs, mask) }
    }

    fn rem_m(src: Storage<Self>, mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_mov_ps(src, mask, Self::rem(lhs, rhs)) }
    }

    fn rem_z(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_fnmadd_ps(mask, Self::trunc(Self::div(lhs, rhs)), rhs, lhs) }
    }

    fn square_c(mask: Storage<Self::Mask>, lhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_mul_ps(lhs, mask, lhs, lhs) }
    }

    fn square_m(src: Storage<Self>, mask: Storage<Self::Mask>, lhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_mul_ps(src, mask, lhs, lhs) }
    }

    fn square_z(mask: Storage<Self::Mask>, lhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_mul_ps(mask, lhs, lhs) }
    }

    fn scale_c(mask: Storage<Self::Mask>, value: Storage<Self>, scalar: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_mask_mul_ps(value, mask, value, Self::splat(scalar)) }
    }

    fn scale_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
        scalar: Self::Element,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mask_mul_ps(src, mask, value, Self::splat(scalar)) }
    }

    fn scale_z(mask: Storage<Self::Mask>, value: Storage<Self>, scalar: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_maskz_mul_ps(mask, value, Self::splat(scalar)) }
    }
}

#[thermite_macros::inline_always]
impl SignedRegister for F32x4V4 {
    const NEG_ONE: Storage<Self> = reg::<Self, 4>([-1.0; 4]);
    const MIN_POSITIVE: Storage<Self> = reg::<Self, 4>([f32::MIN_POSITIVE; 4]);

    fn neg(value: Storage<Self>) -> Storage<Self> {
        Self::bitxor(value, Self::NEG_ZERO)
    }

    fn abs(value: Storage<Self>) -> Storage<Self> {
        Self::bitandnot(value, Self::NEG_ZERO)
    }

    fn copysign(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::ternlog::<0xCA>(Self::NEG_ZERO, rhs, lhs)
    }

    fn signum(value: Storage<Self>) -> Storage<Self> {
        let s = Self::bitor(Self::ONE, Self::bitand(value, Self::NEG_ZERO));
        #[cfg(feature = "strict_ieee754")]
        let s = Self::blendv(Self::is_nan(value), s, value);
        s
    }

    // --- masked variants -----------------------------------------------------

    fn neg_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_xor_ps(value, mask, value, Self::NEG_ZERO) }
    }

    fn neg_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_xor_ps(src, mask, value, Self::NEG_ZERO) }
    }

    fn neg_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_xor_ps(mask, value, Self::NEG_ZERO) }
    }

    fn abs_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_andnot_ps(value, mask, Self::NEG_ZERO, value) }
    }

    fn abs_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_andnot_ps(src, mask, Self::NEG_ZERO, value) }
    }

    fn abs_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_andnot_ps(mask, Self::NEG_ZERO, value) }
    }

    fn copysign_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe {
            arch::_mm_castsi128_ps(arch::_mm_mask_ternarylogic_epi32::<
                { crate::ternlog_imm!(C & B | !C & A) },
            >(
                arch::_mm_castps_si128(lhs),
                mask,
                arch::_mm_castps_si128(rhs),
                arch::_mm_castps_si128(Self::NEG_ZERO),
            ))
        }
    }

    fn copysign_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        lhs: Storage<Self>,
        rhs: Storage<Self>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mask_mov_ps(src, mask, Self::copysign(lhs, rhs)) }
    }

    fn copysign_z(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::ternlog_z::<0xCA>(mask, Self::NEG_ZERO, rhs, lhs)
    }
}

#[thermite_macros::inline_always]
impl FloatRegister for F32x4V4 {
    const HAS_NATIVE_FMA: tribool::Tribool = tribool::True;

    type Bits = super::U32x4V4;
    type SignedBits = super::I32x4V4;
    type ExtendedPrecision = super::F64x4V4;

    const HALF: Storage<Self> = reg::<Self, 4>([0.5; 4]);
    const NEG_ZERO: Storage<Self> = reg::<Self, 4>([-0.0; 4]);
    const EPSILON: Storage<Self> = reg::<Self, 4>([f32::EPSILON; 4]);
    const INFINITY: Storage<Self> = reg::<Self, 4>([f32::INFINITY; 4]);
    const NEG_INFINITY: Storage<Self> = reg::<Self, 4>([f32::NEG_INFINITY; 4]);
    const NAN: Storage<Self> = reg::<Self, 4>([f32::NAN; 4]);

    const EXP_MASK: Storage<Self::Bits> = reg::<Self::Bits, 4>([0x7F800000; 4]);

    fn mul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_fmadd_ps(lhs, rhs, acc) }
    }

    fn mul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_fmsub_ps(lhs, rhs, acc) }
    }

    fn nmul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_fnmadd_ps(lhs, rhs, acc) }
    }

    fn nmul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_fnmsub_ps(lhs, rhs, acc) }
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

    fn addsub(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_addsub_ps(a, b) }
    }

    fn fmaddsub(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_fmaddsub_ps(a, b, c) }
    }

    fn fmsubadd(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_fmsubadd_ps(a, b, c) }
    }

    fn sqrt(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sqrt_ps(value) }
    }

    fn rsqrt(value: Storage<Self>) -> Storage<Self> {
        cfg_select! {
            feature = "strict_ieee754" => {
                Self::rcp(Self::sqrt(value))
            }
            _ => unsafe { arch::_mm_rsqrt14_ps(value) }
        }
    }

    fn rcp(value: Storage<Self>) -> Storage<Self> {
        cfg_select! {
            feature = "strict_ieee754" => {
                Self::div(Self::ONE, value)
            }
            _ => unsafe { arch::_mm_rcp14_ps(value) }
        }
    }

    const HAS_APPROX_RSQRT: bool = cfg!(not(feature = "strict_ieee754"));
    const HAS_APPROX_RCP: bool = cfg!(not(feature = "strict_ieee754"));

    fn floor(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_roundscale_ps(value, 0x09) }
    }

    fn ceil(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_roundscale_ps(value, 0x0A) }
    }

    fn round(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_roundscale_ps(value, 0x08) }
    }

    fn trunc(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_roundscale_ps(value, 0x0B) }
    }

    fn next_up(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_nextupps_v4(value) }
    }

    fn next_down(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_nextdownps_v4(value) }
    }

    const NATIVE_CAP: NativeCapability = NativeCapability::NONE;

    // --- masked variants -----------------------------------------------------

    masked_fma_v4! {
        mov = _mm_mask_mov_ps;
        mul_add => _mm_mask_fmadd_ps, _mm_maskz_fmadd_ps;
        mul_sub => _mm_mask_fmsub_ps, _mm_maskz_fmsub_ps;
        nmul_add => _mm_mask_fnmadd_ps, _mm_maskz_fnmadd_ps;
        nmul_sub => _mm_mask_fnmsub_ps, _mm_maskz_fnmsub_ps;
        mul_adde => _mm_mask_fmadd_ps, _mm_maskz_fmadd_ps;
        mul_sube => _mm_mask_fmsub_ps, _mm_maskz_fmsub_ps;
        nmul_adde => _mm_mask_fnmadd_ps, _mm_maskz_fnmadd_ps;
        nmul_sube => _mm_mask_fnmsub_ps, _mm_maskz_fnmsub_ps;
        fmaddsub => _mm_mask_fmaddsub_ps, _mm_maskz_fmaddsub_ps;
        fmsubadd => _mm_mask_fmsubadd_ps, _mm_maskz_fmsubadd_ps;
    }

    fn addsub_c(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_add_ps(a, mask, a, Self::bitxor(b, Self::ALT_NEG)) }
    }

    fn addsub_m(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_add_ps(src, mask, a, Self::bitxor(b, Self::ALT_NEG)) }
    }

    fn addsub_z(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_add_ps(mask, a, Self::bitxor(b, Self::ALT_NEG)) }
    }

    masked_unary_v4! {
        sqrt => _mm_mask_sqrt_ps, _mm_maskz_sqrt_ps;
        floor => _mm_mask_roundscale_ps, _mm_maskz_roundscale_ps, 0x09;
        ceil => _mm_mask_roundscale_ps, _mm_maskz_roundscale_ps, 0x0A;
        round => _mm_mask_roundscale_ps, _mm_maskz_roundscale_ps, 0x08;
        trunc => _mm_mask_roundscale_ps, _mm_maskz_roundscale_ps, 0x0B;
    }

    cfg_select! {
        feature = "strict_ieee754" => {
            fn rcp_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::_mm_mask_div_ps(value, mask, Self::ONE, value) }
            }

            fn rcp_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::_mm_mask_div_ps(src, mask, Self::ONE, value) }
            }

            fn rcp_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::_mm_maskz_div_ps(mask, Self::ONE, value) }
            }

            fn rsqrt_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::_mm_mask_div_ps(value, mask, Self::ONE, Self::sqrt(value)) }
            }

            fn rsqrt_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::_mm_mask_div_ps(src, mask, Self::ONE, Self::sqrt(value)) }
            }

            fn rsqrt_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::_mm_maskz_div_ps(mask, Self::ONE, Self::sqrt(value)) }
            }
        }
        _ => {
            masked_unary_v4! {
                rcp => _mm_mask_rcp14_ps, _mm_maskz_rcp14_ps;
                rsqrt => _mm_mask_rsqrt14_ps, _mm_maskz_rsqrt14_ps;
            }
        }
    }

    fn fract_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_sub_ps(value, mask, value, Self::trunc(value)) }
    }

    fn fract_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_sub_ps(src, mask, value, Self::trunc(value)) }
    }

    fn fract_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_sub_ps(mask, value, Self::trunc(value)) }
    }

    fn mul_sign_c(mask: Storage<Self::Mask>, value: Storage<Self>, sign: Storage<Self>) -> Storage<Self> {
        Self::ternlog_c::<0x78>(mask, value, sign, Self::NEG_ZERO)
    }

    fn mul_sign_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
        sign: Storage<Self>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mask_mov_ps(src, mask, Self::mul_sign(value, sign)) }
    }

    fn mul_sign_z(mask: Storage<Self::Mask>, value: Storage<Self>, sign: Storage<Self>) -> Storage<Self> {
        Self::ternlog_z::<0x78>(mask, value, sign, Self::NEG_ZERO)
    }

    fn signed_zero_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_and_ps(value, mask, Self::NEG_ZERO, value) }
    }

    fn signed_zero_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_and_ps(src, mask, Self::NEG_ZERO, value) }
    }

    fn signed_zero_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_and_ps(mask, Self::NEG_ZERO, value) }
    }

    masked_via_mov_v4! {
        mov = _mm_mask_mov_ps, movz = _mm_maskz_mov_ps;
        next_up(value: Storage<Self>);
        next_down(value: Storage<Self>);
    }
}

// ExtendedPrecision legs: f32x4 (xmm) <-> f64x4 (ymm), one convert each way.
#[thermite_macros::inline_always]
impl CastRegister<F32x4V4> for super::F64x4V4 {
    fn cast_from(value: Storage<F32x4V4>) -> Storage<Self> {
        unsafe { arch::_mm256_cvtps_pd(value) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::F64x4V4> for F32x4V4 {
    fn cast_from(value: Storage<super::F64x4V4>) -> Storage<Self> {
        unsafe { arch::_mm256_cvtpd_ps(value) }
    }
}

// --- 3/4-lane linear algebra: the required reductions plus the masked
// lane-4 fills. The wide mat4 product overrides are an M4 item. ---
#[thermite_macros::inline_always]
impl crate::register::LinAlg3Register for F32x4V4 {
    fn zero4(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_mov_ps(0b0111, value) }
    }

    fn one4(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_mov_ps(value, 0b1000, arch::_mm_set1_ps(1.0)) }
    }

    fn min_element3(value: Storage<Self>) -> Self::Element {
        _mm_reduce_ps3_v1!(value; _mm_min_ss)
    }

    fn max_element3(value: Storage<Self>) -> Self::Element {
        _mm_reduce_ps3_v1!(value; _mm_max_ss)
    }

    fn sum_elements3(value: Storage<Self>) -> Self::Element {
        _mm_reduce_ps3_v1!(value; _mm_add_ss)
    }

    fn prod_elements3(value: Storage<Self>) -> Self::Element {
        _mm_reduce_ps3_v1!(value; _mm_mul_ss)
    }
}

impl crate::register::LinAlg4Register for F32x4V4 {}
