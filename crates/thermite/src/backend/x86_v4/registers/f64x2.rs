//! `f64x2` on AVX-512: the 128-bit double register under EVEX (AVX512VL).
//! Width port of [`f64x4`](super::f64x4). See [`f32x4`](super::f32x4) for
//! the xmm-specific spellings. The two-lane permutes are the `vpermt2pd`
//! xmm form with both sources the same register (index bit 0 picks the
//! lane). The fixed reverse is `vpermilpd imm`. `KMask2` masks, the v1
//! `_mm_reduce_pd_v1!` fold. Masked splat is `vmovddup`.

use generic_array::{GenericArray, sequence::GenericSequence, typenum::Unsigned};

use crate::register::{
    BitwiseRegister, CoreRegister, FloatRegister, IndexableRegister, InterleaveRegister, NativeCapability,
    NumericRegister, PartialOrdRegister, Register, ShuffleRegister, SignedRegister, Storage, ZeroUpper, empty_reg, reg,
};

use super::arch;
use super::kmask::KMask2;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct F64x2V4;

#[thermite_macros::inline_always]
impl CoreRegister for F64x2V4 {
    type Lanes = generic_array::typenum::U2;
    type Storage = arch::__m128d;
    type Mask = KMask2;

    const IS_EMULATED: bool = false;

    const EMPTY: Storage<Self> = empty_reg::<Self>();

    const HAS_EQUAL_SIZE_MASK: bool = false;

    fn blendv(mask: Storage<Self::Mask>, on_false: Storage<Self>, on_true: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_blend_pd(mask, on_false, on_true) }
    }

    fn zz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_mov_pd(mask, value) }
    }

    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_mov_pd(!mask, value) }
    }

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        if const { Z::N >= 2 } {
            value
        } else {
            unsafe { arch::_mm_maskz_mov_pd(const { ((1u32 << Z::N) - 1) as u8 }, value) }
        }
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        unsafe { arch::_mm_castsi128_pd(arch::_mm_movm_epi64(mask)) }
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for F64x2V4 {
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_xor_pd(lhs, rhs) }
    }

    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_and_pd(lhs, rhs) }
    }

    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_andnot_pd(rhs, lhs) }
    }

    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_or_pd(lhs, rhs) }
    }

    fn not(value: Storage<Self>) -> Storage<Self> {
        unsafe {
            arch::_mm_castsi128_pd(arch::_mm_ternarylogic_epi64::<{ crate::ternlog_imm!(!C) }>(
                arch::_mm_castpd_si128(value),
                arch::_mm_castpd_si128(value),
                arch::_mm_castpd_si128(value),
            ))
        }
    }

    const HAS_NATIVE_TERNLOG: bool = true;

    fn ternlog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe {
            arch::_mm_castsi128_pd(arch::_mm_ternarylogic_epi64::<IMM>(
                arch::_mm_castpd_si128(a),
                arch::_mm_castpd_si128(b),
                arch::_mm_castpd_si128(c),
            ))
        }
    }

    fn bilog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        arch::bilog_ternlog::<Self, IMM>(a, b)
    }

    // --- masked variants -----------------------------------------------------

    masked_binary_v4! {
        bitxor => _mm_mask_xor_pd, _mm_maskz_xor_pd;
        bitand => _mm_mask_and_pd, _mm_maskz_and_pd;
        bitor => _mm_mask_or_pd, _mm_maskz_or_pd;
    }

    masked_andnot_v4! {
        bitandnot => _mm_mask_andnot_pd, _mm_maskz_andnot_pd;
    }

    fn not_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let v = unsafe { arch::_mm_castpd_si128(value) };
        unsafe { arch::_mm_castsi128_pd(arch::_mm_mask_ternarylogic_epi64::<{ crate::ternlog_imm!(!C) }>(v, mask, v, v)) }
    }

    fn not_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let v = unsafe { arch::_mm_castpd_si128(value) };
        unsafe { arch::_mm_castsi128_pd(arch::_mm_mask_ternarylogic_epi64::<{ crate::ternlog_imm!(!C) }>(arch::_mm_castpd_si128(src), mask, v, v)) }
    }

    fn not_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let v = unsafe { arch::_mm_castpd_si128(value) };
        unsafe { arch::_mm_castsi128_pd(arch::_mm_maskz_ternarylogic_epi64::<{ crate::ternlog_imm!(!C) }>(mask, v, v, v)) }
    }

    fn ternlog_c<const IMM: i32>(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe {
            arch::_mm_castsi128_pd(arch::_mm_mask_ternarylogic_epi64::<IMM>(
                arch::_mm_castpd_si128(a),
                mask,
                arch::_mm_castpd_si128(b),
                arch::_mm_castpd_si128(c),
            ))
        }
    }

    fn ternlog_m<const IMM: i32>(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_mov_pd(src, mask, Self::ternlog::<IMM>(a, b, c)) }
    }

    fn ternlog_z<const IMM: i32>(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe {
            arch::_mm_castsi128_pd(arch::_mm_maskz_ternarylogic_epi64::<IMM>(
                mask,
                arch::_mm_castpd_si128(a),
                arch::_mm_castpd_si128(b),
                arch::_mm_castpd_si128(c),
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

// Width ladder bottom rung: two scalars make an xmm, and ymm is the 2x wide register.
#[thermite_macros::inline_always]
impl crate::register::ConcatRegister<f64> for F64x2V4 {
    fn concat(lo: Storage<f64>, hi: Storage<f64>) -> Storage<Self> {
        unsafe { arch::_mm_setr_pd(lo, hi) }
    }

    fn split(value: Storage<Self>) -> (Storage<f64>, Storage<f64>) {
        let lo = unsafe { arch::_mm_cvtsd_f64(value) };
        let hi = unsafe { arch::_mm_cvtsd_f64(arch::_mm_unpackhi_pd(value, value)) };
        (lo, hi)
    }
}

#[thermite_macros::inline_always]
impl crate::register::ExtendRegister<f64> for F64x2V4 {
    fn extend(value: Storage<f64>) -> Storage<Self> {
        unsafe { arch::_mm_set_sd(value) }
    }

    fn narrow(value: Storage<Self>) -> Storage<f64> {
        unsafe { arch::_mm_cvtsd_f64(value) }
    }
}

#[thermite_macros::inline_always]
impl crate::register::WideRegister for F64x2V4 {
    type Wide = super::F64x4V4;
}

#[thermite_macros::inline_always]
impl Register for F64x2V4 {
    type Element = f64;

    type Signed = super::I64x2V4;
    type Unsigned = super::U64x2V4;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_test_epi64_mask(arch::_mm_castpd_si128(value), arch::_mm_castpd_si128(value)) }
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::msb_to_mask(value)
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_movepi64_mask(arch::_mm_castpd_si128(value)) }
    }

    fn new(value: GenericArray<f64, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_loadu_pd(value.as_ptr()) }
    }

    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_set_sd(value) }
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_set1_pd(value) }
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_load_pd(ptr) }
    }

    unsafe fn load_m(src: Storage<Self>, mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_mask_loadu_pd(src, mask, ptr) }
    }

    unsafe fn load_z(mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_maskz_loadu_pd(mask, ptr) }
    }

    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_loadu_pd(ptr) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_store_pd(ptr, value) }
    }

    unsafe fn store_masked(ptr: *mut Self::Element, mask: Storage<Self::Mask>, value: Storage<Self>) {
        unsafe { arch::_mm_mask_storeu_pd(ptr, mask, value) }
    }

    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_storeu_pd(ptr, value) }
    }

    unsafe fn load_stream(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_castsi128_pd(arch::_mm_stream_load_si128(ptr as _)) }
    }

    unsafe fn store_stream(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_stream_pd(ptr, value) }
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
        unsafe { arch::_mm_permute_pd::<0b01>(value) }
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        unsafe {
            let pattern = arch::_mm_setr_epi8(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8);
            arch::_mm_castsi128_pd(arch::_mm_shuffle_epi8(arch::_mm_castpd_si128(value), pattern))
        }
    }

    compress_expand_v4!(u8, _mm_maskz_compress_pd, _mm_mask_expand_pd, _mm_maskz_expand_pd);

    const HAS_PERMUTEV: bool = true;

    // vpermt2pd with both sources `value`: index bit 0 selects the lane.
    fn permutev(value: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_permutex2var_pd(value, idxs, value) }
    }

    fn swizzle(a: Storage<Self>, b: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_permutex2var_pd(a, idxs, b) }
    }

    // --- masked variants -----------------------------------------------------

    fn new_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: GenericArray<f64, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_mask_loadu_pd(src, mask, value.as_ptr()) }
    }

    fn new_z(mask: Storage<Self::Mask>, value: GenericArray<f64, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_loadu_pd(mask, value.as_ptr()) }
    }

    fn splat_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_mask_movedup_pd(src, mask, arch::_mm_set_sd(value)) }
    }

    fn splat_z(mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_maskz_movedup_pd(mask, arch::_mm_set_sd(value)) }
    }

    fn broadcast<const I: usize>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_permutex2var_pd(value, arch::_mm_set1_epi64x(I as i64), value) }
    }

    fn broadcast_c<const I: usize>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_permutex2var_pd(value, mask, arch::_mm_set1_epi64x(I as i64), value) }
    }

    fn broadcast_m<const I: usize>(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mask_mov_pd(src, mask, Self::broadcast::<I>(value)) }
    }

    fn broadcast_z<const I: usize>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_permutex2var_pd(mask, value, arch::_mm_set1_epi64x(I as i64), value) }
    }

    fn broadcastv(value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm_permutex2var_pd(value, arch::_mm_set1_epi64x(idx as i64), value) }
    }

    fn broadcastv_c(mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm_mask_permutex2var_pd(value, mask, arch::_mm_set1_epi64x(idx as i64), value) }
    }

    fn broadcastv_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm_mask_mov_pd(src, mask, Self::broadcastv(value, idx)) }
    }

    fn broadcastv_z(mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        unsafe { arch::_mm_maskz_permutex2var_pd(mask, value, arch::_mm_set1_epi64x(idx as i64), value) }
    }

    fn reverse_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_permute_pd::<0b01>(value, mask, value) }
    }

    fn reverse_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_permute_pd::<0b01>(src, mask, value) }
    }

    fn reverse_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_permute_pd::<0b01>(mask, value) }
    }

    masked_via_mov_v4! {
        mov = _mm_mask_mov_pd, movz = _mm_maskz_mov_pd;
        swap_bytes(value: Storage<Self>);
    }

    fn permutev_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mask_mov_pd(src, mask, arch::_mm_permutex2var_pd(value, idxs, value)) }
    }

    fn permutev_z(mask: Storage<Self::Mask>, value: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_permutex2var_pd(mask, value, idxs, value) }
    }

    fn swizzle_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        a: Storage<Self>,
        b: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mask_mov_pd(src, mask, arch::_mm_permutex2var_pd(a, idxs, b)) }
    }

    fn swizzle_z(
        mask: Storage<Self::Mask>,
        a: Storage<Self>,
        b: Storage<Self>,
        idxs: Storage<Self::Unsigned>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_maskz_permutex2var_pd(mask, a, idxs, b) }
    }
}

#[thermite_macros::inline_always]
impl InterleaveRegister for F64x2V4 {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe { (arch::_mm_unpacklo_pd(a, b), arch::_mm_unpackhi_pd(a, b)) }
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe { (arch::_mm_unpacklo_pd(a, b), arch::_mm_unpackhi_pd(a, b)) }
    }
}

#[thermite_macros::inline_always]
impl IndexableRegister<super::U64x2V4> for F64x2V4 {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<super::U64x2V4>) -> Storage<Self> {
        unsafe { arch::_mm_i64gather_pd::<8>(ptr, indices) }
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<super::U64x2V4>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mmask_i64gather_pd::<8>(src, mask, indices, ptr) }
    }

    unsafe fn scatter(value: Storage<Self>, ptr: *mut Self::Element, indices: Storage<super::U64x2V4>) {
        unsafe { arch::_mm_i64scatter_pd::<8>(ptr, indices, value) }
    }

    unsafe fn scatter_m(
        value: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *mut Self::Element,
        indices: Storage<super::U64x2V4>,
    ) {
        unsafe { arch::_mm_mask_i64scatter_pd::<8>(ptr, mask, indices, value) }
    }
}

#[thermite_macros::inline_always]
impl ShuffleRegister for F64x2V4 {
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_shuffle_pd(lhs, rhs, IMM8) }
    }
}

#[thermite_macros::inline_always]
impl PartialOrdRegister for F64x2V4 {
    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmp_pd_mask(lhs, rhs, arch::_CMP_LT_OQ) }
    }

    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmp_pd_mask(lhs, rhs, arch::_CMP_LE_OQ) }
    }

    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmp_pd_mask(lhs, rhs, arch::_CMP_GT_OQ) }
    }

    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmp_pd_mask(lhs, rhs, arch::_CMP_GE_OQ) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmp_pd_mask(lhs, rhs, arch::_CMP_EQ_OQ) }
    }

    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm_cmp_pd_mask(lhs, rhs, arch::_CMP_NEQ_UQ) }
    }
}

#[thermite_macros::inline_always]
impl NumericRegister for F64x2V4 {
    const ZERO: Storage<Self> = reg::<Self, 2>([0.0; 2]);
    const ONE: Storage<Self> = reg::<Self, 2>([1.0; 2]);
    const TWO: Storage<Self> = reg::<Self, 2>([2.0; 2]);

    const MIN: Storage<Self> = reg::<Self, 2>([f64::MIN; 2]);
    const MAX: Storage<Self> = reg::<Self, 2>([f64::MAX; 2]);

    fn min_element(value: Storage<Self>) -> Self::Element {
        _mm_reduce_pd_v1!(value; _mm_min_pd _mm_min_sd)
    }

    fn max_element(value: Storage<Self>) -> Self::Element {
        _mm_reduce_pd_v1!(value; _mm_max_pd _mm_max_sd)
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element {
        _mm_reduce_pd_v1!(value; _mm_add_pd _mm_add_sd)
    }

    fn prod_elements(value: Storage<Self>) -> Self::Element {
        _mm_reduce_pd_v1!(value; _mm_mul_pd _mm_mul_sd)
    }

    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as Unsigned>::USIZE as f64)
    }

    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as f64))
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_add_pd(lhs, rhs) }
    }

    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sub_pd(lhs, rhs) }
    }

    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mul_pd(lhs, rhs) }
    }

    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_div_pd(lhs, rhs) }
    }

    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::nmul_adde(Self::trunc(Self::div(lhs, rhs)), rhs, lhs)
    }

    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::fix_min::<Self>(lhs, rhs, unsafe { arch::_mm_min_pd(lhs, rhs) })
    }

    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::fix_max::<Self>(lhs, rhs, unsafe { arch::_mm_max_pd(lhs, rhs) })
    }

    // --- masked variants -----------------------------------------------------

    masked_binary_v4! {
        add => _mm_mask_add_pd, _mm_maskz_add_pd;
        sub => _mm_mask_sub_pd, _mm_maskz_sub_pd;
        mul => _mm_mask_mul_pd, _mm_maskz_mul_pd;
        div => _mm_mask_div_pd, _mm_maskz_div_pd;
    }

    cfg_select! {
        feature = "strict_ieee754" => {
            masked_via_mov_v4! {
                mov = _mm_mask_mov_pd, movz = _mm_maskz_mov_pd;
                min(lhs: Storage<Self>, rhs: Storage<Self>);
                max(lhs: Storage<Self>, rhs: Storage<Self>);
            }
        }
        _ => {
            masked_binary_v4! {
                min => _mm_mask_min_pd, _mm_maskz_min_pd;
                max => _mm_mask_max_pd, _mm_maskz_max_pd;
            }
        }
    }

    fn rem_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask3_fnmadd_pd(Self::trunc(Self::div(lhs, rhs)), rhs, lhs, mask) }
    }

    fn rem_m(src: Storage<Self>, mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_mov_pd(src, mask, Self::rem(lhs, rhs)) }
    }

    fn rem_z(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_fnmadd_pd(mask, Self::trunc(Self::div(lhs, rhs)), rhs, lhs) }
    }

    fn square_c(mask: Storage<Self::Mask>, lhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_mul_pd(lhs, mask, lhs, lhs) }
    }

    fn square_m(src: Storage<Self>, mask: Storage<Self::Mask>, lhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_mul_pd(src, mask, lhs, lhs) }
    }

    fn square_z(mask: Storage<Self::Mask>, lhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_mul_pd(mask, lhs, lhs) }
    }

    fn scale_c(mask: Storage<Self::Mask>, value: Storage<Self>, scalar: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_mask_mul_pd(value, mask, value, Self::splat(scalar)) }
    }

    fn scale_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        value: Storage<Self>,
        scalar: Self::Element,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mask_mul_pd(src, mask, value, Self::splat(scalar)) }
    }

    fn scale_z(mask: Storage<Self::Mask>, value: Storage<Self>, scalar: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_maskz_mul_pd(mask, value, Self::splat(scalar)) }
    }
}

#[thermite_macros::inline_always]
impl SignedRegister for F64x2V4 {
    const NEG_ONE: Storage<Self> = reg::<Self, 2>([-1.0; 2]);
    const MIN_POSITIVE: Storage<Self> = reg::<Self, 2>([f64::MIN_POSITIVE; 2]);

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
        unsafe { arch::_mm_mask_xor_pd(value, mask, value, Self::NEG_ZERO) }
    }

    fn neg_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_xor_pd(src, mask, value, Self::NEG_ZERO) }
    }

    fn neg_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_xor_pd(mask, value, Self::NEG_ZERO) }
    }

    fn abs_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_andnot_pd(value, mask, Self::NEG_ZERO, value) }
    }

    fn abs_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_andnot_pd(src, mask, Self::NEG_ZERO, value) }
    }

    fn abs_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_andnot_pd(mask, Self::NEG_ZERO, value) }
    }

    fn copysign_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe {
            arch::_mm_castsi128_pd(arch::_mm_mask_ternarylogic_epi64::<
                { crate::ternlog_imm!(C & B | !C & A) },
            >(
                arch::_mm_castpd_si128(lhs),
                mask,
                arch::_mm_castpd_si128(rhs),
                arch::_mm_castpd_si128(Self::NEG_ZERO),
            ))
        }
    }

    fn copysign_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        lhs: Storage<Self>,
        rhs: Storage<Self>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mask_mov_pd(src, mask, Self::copysign(lhs, rhs)) }
    }

    fn copysign_z(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::ternlog_z::<0xCA>(mask, Self::NEG_ZERO, rhs, lhs)
    }
}

#[thermite_macros::inline_always]
impl FloatRegister for F64x2V4 {
    const HAS_NATIVE_FMA: tribool::Tribool = tribool::True;

    type Bits = super::U64x2V4;
    type SignedBits = super::I64x2V4;
    type ExtendedPrecision = Self;

    const HALF: Storage<Self> = reg::<Self, 2>([0.5; 2]);
    const NEG_ZERO: Storage<Self> = reg::<Self, 2>([-0.0; 2]);
    const EPSILON: Storage<Self> = reg::<Self, 2>([f64::EPSILON; 2]);
    const INFINITY: Storage<Self> = reg::<Self, 2>([f64::INFINITY; 2]);
    const NEG_INFINITY: Storage<Self> = reg::<Self, 2>([f64::NEG_INFINITY; 2]);
    const NAN: Storage<Self> = reg::<Self, 2>([f64::NAN; 2]);

    const EXP_MASK: Storage<Self::Bits> = reg::<Self::Bits, 2>([0x7FF0_0000_0000_0000; 2]);

    fn mul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_fmadd_pd(lhs, rhs, acc) }
    }

    fn mul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_fmsub_pd(lhs, rhs, acc) }
    }

    fn nmul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_fnmadd_pd(lhs, rhs, acc) }
    }

    fn nmul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_fnmsub_pd(lhs, rhs, acc) }
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
        unsafe { arch::_mm_addsub_pd(a, b) }
    }

    fn fmaddsub(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_fmaddsub_pd(a, b, c) }
    }

    fn fmsubadd(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_fmsubadd_pd(a, b, c) }
    }

    fn sqrt(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sqrt_pd(value) }
    }

    fn rsqrt(value: Storage<Self>) -> Storage<Self> {
        cfg_select! {
            feature = "strict_ieee754" => {
                Self::rcp(Self::sqrt(value))
            }
            _ => unsafe { arch::_mm_rsqrt14_pd(value) }
        }
    }

    fn rcp(value: Storage<Self>) -> Storage<Self> {
        cfg_select! {
            feature = "strict_ieee754" => {
                Self::div(Self::ONE, value)
            }
            _ => unsafe { arch::_mm_rcp14_pd(value) }
        }
    }

    const HAS_APPROX_RSQRT: bool = cfg!(not(feature = "strict_ieee754"));
    const HAS_APPROX_RCP: bool = cfg!(not(feature = "strict_ieee754"));

    fn floor(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_roundscale_pd(value, 0x09) }
    }

    fn ceil(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_roundscale_pd(value, 0x0A) }
    }

    fn round(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_roundscale_pd(value, 0x08) }
    }

    fn trunc(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_roundscale_pd(value, 0x0B) }
    }

    fn next_up(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_nextuppd_v4(value) }
    }

    fn next_down(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_nextdownpd_v4(value) }
    }

    const NATIVE_CAP: NativeCapability = NativeCapability::NONE;

    // --- masked variants -----------------------------------------------------

    masked_fma_v4! {
        mov = _mm_mask_mov_pd;
        mul_add => _mm_mask_fmadd_pd, _mm_maskz_fmadd_pd;
        mul_sub => _mm_mask_fmsub_pd, _mm_maskz_fmsub_pd;
        nmul_add => _mm_mask_fnmadd_pd, _mm_maskz_fnmadd_pd;
        nmul_sub => _mm_mask_fnmsub_pd, _mm_maskz_fnmsub_pd;
        mul_adde => _mm_mask_fmadd_pd, _mm_maskz_fmadd_pd;
        mul_sube => _mm_mask_fmsub_pd, _mm_maskz_fmsub_pd;
        nmul_adde => _mm_mask_fnmadd_pd, _mm_maskz_fnmadd_pd;
        nmul_sube => _mm_mask_fnmsub_pd, _mm_maskz_fnmsub_pd;
        fmaddsub => _mm_mask_fmaddsub_pd, _mm_maskz_fmaddsub_pd;
        fmsubadd => _mm_mask_fmsubadd_pd, _mm_maskz_fmsubadd_pd;
    }

    fn addsub_c(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_add_pd(a, mask, a, Self::bitxor(b, Self::ALT_NEG)) }
    }

    fn addsub_m(src: Storage<Self>, mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_add_pd(src, mask, a, Self::bitxor(b, Self::ALT_NEG)) }
    }

    fn addsub_z(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_add_pd(mask, a, Self::bitxor(b, Self::ALT_NEG)) }
    }

    masked_unary_v4! {
        sqrt => _mm_mask_sqrt_pd, _mm_maskz_sqrt_pd;
        floor => _mm_mask_roundscale_pd, _mm_maskz_roundscale_pd, 0x09;
        ceil => _mm_mask_roundscale_pd, _mm_maskz_roundscale_pd, 0x0A;
        round => _mm_mask_roundscale_pd, _mm_maskz_roundscale_pd, 0x08;
        trunc => _mm_mask_roundscale_pd, _mm_maskz_roundscale_pd, 0x0B;
    }

    cfg_select! {
        feature = "strict_ieee754" => {
            fn rcp_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::_mm_mask_div_pd(value, mask, Self::ONE, value) }
            }

            fn rcp_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::_mm_mask_div_pd(src, mask, Self::ONE, value) }
            }

            fn rcp_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::_mm_maskz_div_pd(mask, Self::ONE, value) }
            }

            fn rsqrt_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::_mm_mask_div_pd(value, mask, Self::ONE, Self::sqrt(value)) }
            }

            fn rsqrt_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::_mm_mask_div_pd(src, mask, Self::ONE, Self::sqrt(value)) }
            }

            fn rsqrt_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::_mm_maskz_div_pd(mask, Self::ONE, Self::sqrt(value)) }
            }
        }
        _ => {
            masked_unary_v4! {
                rcp => _mm_mask_rcp14_pd, _mm_maskz_rcp14_pd;
                rsqrt => _mm_mask_rsqrt14_pd, _mm_maskz_rsqrt14_pd;
            }
        }
    }

    fn fract_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_sub_pd(value, mask, value, Self::trunc(value)) }
    }

    fn fract_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_sub_pd(src, mask, value, Self::trunc(value)) }
    }

    fn fract_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_sub_pd(mask, value, Self::trunc(value)) }
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
        unsafe { arch::_mm_mask_mov_pd(src, mask, Self::mul_sign(value, sign)) }
    }

    fn mul_sign_z(mask: Storage<Self::Mask>, value: Storage<Self>, sign: Storage<Self>) -> Storage<Self> {
        Self::ternlog_z::<0x78>(mask, value, sign, Self::NEG_ZERO)
    }

    fn signed_zero_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_and_pd(value, mask, Self::NEG_ZERO, value) }
    }

    fn signed_zero_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mask_and_pd(src, mask, Self::NEG_ZERO, value) }
    }

    fn signed_zero_z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_maskz_and_pd(mask, Self::NEG_ZERO, value) }
    }

    masked_via_mov_v4! {
        mov = _mm_mask_mov_pd, movz = _mm_maskz_mov_pd;
        next_up(value: Storage<Self>);
        next_down(value: Storage<Self>);
    }
}
