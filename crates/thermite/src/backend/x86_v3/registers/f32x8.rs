use generic_array::{GenericArray, sequence::GenericSequence, typenum::Unsigned};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, BitwiseRegister, CastRegister, ConcatRegister, CoreRegister, Element, ExtendRegister,
        FloatRegister, IndexableRegister, InterleaveRegister, MaskElement, MaskRegister, NativeCapability,
        NumericRegister, PartialOrdRegister, PermuteRegister, Register, ShuffleRegister, SignedRegister, Storage,
        ZeroUpper, array::ArrayRegister, empty_reg, reg,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct F32x8V3;

/// Interleaved-complex 4x4 transpose (`(de)interleave_radix_by::<4, 2>`): the `W = 8`
/// bytes (64-bit effective element) square transpose. A thin adapter over the shared
/// [`transpose256_w64`](arch::transpose256_w64) - `f32` pairs cast to the pd domain and
/// back, both casts free. See that function for why the body is register-type-agnostic.
#[inline(always)]
fn transpose_4x4_pairs(i: [arch::__m256; 4]) -> [arch::__m256; 4] {
    unsafe {
        let t = arch::transpose256_w64([
            arch::_mm256_castps_pd(i[0]),
            arch::_mm256_castps_pd(i[1]),
            arch::_mm256_castps_pd(i[2]),
            arch::_mm256_castps_pd(i[3]),
        ]);
        [
            arch::_mm256_castpd_ps(t[0]),
            arch::_mm256_castpd_ps(t[1]),
            arch::_mm256_castpd_ps(t[2]),
            arch::_mm256_castpd_ps(t[3]),
        ]
    }
}

#[thermite_macros::inline_always]
impl CoreRegister for F32x8V3 {
    type Lanes = generic_array::typenum::U8;
    type Storage = arch::__m256;
    type Mask = Self;

    const IS_EMULATED: bool = false;

    const ISA: InstructionSet = InstructionSet::X86V3;

    const EMPTY: Storage<Self> = empty_reg::<Self>();

    const HAS_EQUAL_SIZE_MASK: bool = true;

    fn blendv(mask: Storage<Self::Mask>, on_false: Storage<Self>, on_true: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_blendv_ps(on_false, on_true, mask) }
    }

    fn zz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_and_ps(value, mask) }
    }

    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_andnot_ps(mask, value) }
    }

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        if const { Z::N >= 8 } {
            value
        } else if const { Z::N == 4 } {
            unsafe { arch::_mm256_zextps128_ps256(arch::_mm256_castps256_ps128(value)) }
        } else {
            unsafe {
                arch::_mm256_and_ps(
                    value,
                    arch::_mm256_castsi256_ps(arch::_mm256_zeroupper_mask_epi32::<Z>()),
                )
            }
        }
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        mask
    }
}

#[thermite_macros::inline_always]
impl MaskRegister for F32x8V3 {
    const FALSY: Storage<Self> = reg::<Self, 8>([f32::from_bits(0); 8]);
    const TRUTHY: Storage<Self> = reg::<Self, 8>([f32::from_bits(!0); 8]);

    fn set(mut mask: Storage<Self::Mask>, lane: usize, value: bool) -> Storage<Self> {
        Self::as_mut_slice(&mut mask)[lane] = if value { MaskElement::TRUTHY } else { MaskElement::FALSY };
        mask
    }

    fn test(mask: Storage<Self::Mask>, lane: usize) -> bool {
        Self::as_slice(&mask)[lane].to_bool()
    }

    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm256_castsi256_ps(arch::_mm256_cvtboolx8_to_epi32_mask_v3(value)) }
    }

    fn all(value: Storage<Self>) -> bool {
        unsafe { arch::_mm256_movemask_ps(value) == 0xff }
    }

    fn any(value: Storage<Self>) -> bool {
        unsafe { arch::_mm256_movemask_ps(value) != 0 }
    }

    fn none(value: Storage<Self>) -> bool {
        unsafe { arch::_mm256_movemask_ps(value) == 0 }
    }

    fn count_set<const N: usize>(values: [Storage<Self>; N]) -> usize {
        unsafe { arch::_mm256_count_mask_ps_v3(values) }
    }

    fn from_native_bitmask(bitmask: u64) -> Storage<Self> {
        unsafe { arch::_mm256_castsi256_ps(arch::_mm256_movm_epi32x_v3(bitmask)) }
    }

    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        Some(unsafe { arch::_mm256_movemask_ps(value) as u64 })
    }

    #[cfg(feature = "bitvec")]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = unsafe { arch::_mm256_movemask_ps(value) as u32 };
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for F32x8V3 {
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_xor_ps(lhs, rhs) }
    }

    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_and_ps(lhs, rhs) }
    }

    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_andnot_ps(lhs, rhs) }
    }

    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_or_ps(lhs, rhs) }
    }

    fn not(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_xor_ps(value, arch::_mm256_set1_ps(f32::from_bits(!0))) }
    }
}

#[thermite_macros::inline_always]
impl ConcatRegister<super::F32x4V3> for F32x8V3 {
    fn concat(lo: Storage<super::F32x4V3>, hi: Storage<super::F32x4V3>) -> Storage<Self> {
        unsafe { arch::_mm256_setr_m128(lo, hi) }
    }

    fn split(value: Storage<Self>) -> (Storage<super::F32x4V3>, Storage<super::F32x4V3>) {
        let lo = unsafe { arch::_mm256_castps256_ps128(value) };
        let hi = unsafe { arch::_mm256_extractf128_ps(value, 1) };
        (lo, hi)
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<super::F32x4V3> for F32x8V3 {
    fn extend(value: Storage<super::F32x4V3>) -> Storage<Self> {
        unsafe { arch::_mm256_zextps128_ps256(value) }
    }

    fn narrow(value: Storage<Self>) -> Storage<super::F32x4V3> {
        unsafe { arch::_mm256_castps256_ps128(value) }
    }
}

#[thermite_macros::inline_always]
impl Register for F32x8V3 {
    type Element = f32;

    type Signed = super::I32x8V3;
    type Unsigned = super::U32x8V3;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe {
            // value != 0.0
            arch::_mm256_castsi256_ps(arch::_mm256_xor_si256(
                arch::_mm256_set1_epi8(-1),
                arch::_mm256_cmpeq_epi32(arch::_mm256_castps_si256(value), arch::_mm256_setzero_si256()),
            ))
        }
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        value
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        value // floats support msb masks directly
    }

    fn new(value: generic_array::GenericArray<f32, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm256_loadu_ps(value.as_ptr()) }
    }

    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_setr_ps(value, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) }
    }

    impl_native_radix3!(arch::_mm256_interleave3_ps, arch::_mm256_deinterleave3_ps);

    impl_native_extract!(@ps256);

    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_set1_ps(value) }
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_load_ps(ptr) }
    }

    unsafe fn load_m(src: Storage<Self>, mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        // use load_z + 2 bitwise ops to emulate load_m without blendv or scalar fallbacks
        unsafe { Self::bitor(Self::load_z(mask, ptr), Self::bitandnot(mask, src)) }
    }

    unsafe fn load_z(mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_maskload_ps(ptr, arch::_mm256_castps_si256(mask)) }
    }

    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_loadu_ps(ptr) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm256_store_ps(ptr, value) }
    }

    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm256_storeu_ps(ptr, value) }
    }

    unsafe fn load_stream(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_castsi256_ps(arch::_mm256_stream_load_si256(ptr as _)) }
    }

    unsafe fn store_stream(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm256_stream_ps(ptr, value) }
    }

    unsafe fn lookup(values: &[Self::Element], indices: Storage<Self::Unsigned>) -> Storage<Self> {
        if values.len() <= <Self::Lanes as Unsigned>::USIZE {
            let mut padded: GenericArray<Self::Element, Self::Lanes> = unsafe { core::mem::zeroed() };
            padded[..values.len()].copy_from_slice(values);

            unsafe { arch::_mm256_permutevar8x32_ps(Self::new(padded), indices) }
        } else {
            unsafe { <Self as IndexableRegister<Self::Unsigned>>::gather(values.as_ptr(), indices) }
        }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        // TODO: Improve this
        let (lo, hi) = Self::split(value);
        Self::concat(super::F32x4V3::reverse(hi), super::F32x4V3::reverse(lo))
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_bswap_psx_v3(value) }
    }

    compress_via_table!();

    const HAS_PERMUTEV: bool = true;

    impl_float_align_via_bits!(super::U32x8V3);

    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm256_permutevar8x32_ps(value, core::mem::transmute(idxs)) }
    }

    fn interleave_by<const GROUP: usize>(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        if const { GROUP == 2 } {
            unsafe {
                // Pair granularity = `f64`-lane interleave: adjacent `f32` pairs (one complex each)
                // move as a unit. Same structure as `interleave`, on the `pd` reinterpretation.
                let (a, b) = (arch::_mm256_castps_pd(a), arch::_mm256_castps_pd(b));
                let u_lo = arch::_mm256_unpacklo_pd(a, b);
                let u_hi = arch::_mm256_unpackhi_pd(a, b);
                let res_lo = arch::_mm256_permute2f128_pd(u_lo, u_hi, 0x20);
                let res_hi = arch::_mm256_permute2f128_pd(u_lo, u_hi, 0x31);
                (arch::_mm256_castpd_ps(res_lo), arch::_mm256_castpd_ps(res_hi))
            }
        } else {
            crate::backend::generic::polyfills::interleave_by_default::<Self, GROUP>(a, b)
        }
    }

    fn deinterleave_by<const GROUP: usize>(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        if const { GROUP == 2 } {
            unsafe {
                let (a, b) = (arch::_mm256_castps_pd(a), arch::_mm256_castps_pd(b));
                let t0 = arch::_mm256_permute2f128_pd(a, b, 0x20);
                let t1 = arch::_mm256_permute2f128_pd(a, b, 0x31);
                let o0 = arch::_mm256_unpacklo_pd(t0, t1);
                let o1 = arch::_mm256_unpackhi_pd(t0, t1);
                (arch::_mm256_castpd_ps(o0), arch::_mm256_castpd_ps(o1))
            }
        } else {
            crate::backend::generic::polyfills::deinterleave_by_default::<Self, GROUP>(a, b)
        }
    }

    // The `(N, GROUP) == (4, 2)` square case is the interleaved-complex 4x4
    // transpose: four `f64`-lane unpacks + four `permute2f128` = 8 ops (vs the
    // ~12 a 2-round `interleave_by::<2>` network folds to). It is its own inverse,
    // so `interleave_radix_by` reuses the same body. Every other shape defers to the
    // generic default.
    fn deinterleave_radix_by<const N: usize, const GROUP: usize>(inputs: [Storage<Self>; N]) -> [Storage<Self>; N] {
        if const { N == 4 && GROUP == 2 } {
            // SAFETY: `N == 4` on this arm, so indices 0..4 are in bounds.
            unsafe {
                let t = transpose_4x4_pairs([
                    *inputs.get_unchecked(0),
                    *inputs.get_unchecked(1),
                    *inputs.get_unchecked(2),
                    *inputs.get_unchecked(3),
                ]);
                let mut out = [Self::EMPTY; N];
                *out.get_unchecked_mut(0) = t[0];
                *out.get_unchecked_mut(1) = t[1];
                *out.get_unchecked_mut(2) = t[2];
                *out.get_unchecked_mut(3) = t[3];
                out
            }
        } else if const { N == 8 && GROUP == 1 } {
            // The full 8x8 f32 transpose (the square N==LANES, GROUP==1 case).
            // SAFETY: `N == 8` on this arm, so indices 0..8 are in bounds.
            unsafe {
                let t = arch::transpose256_w32([
                    *inputs.get_unchecked(0),
                    *inputs.get_unchecked(1),
                    *inputs.get_unchecked(2),
                    *inputs.get_unchecked(3),
                    *inputs.get_unchecked(4),
                    *inputs.get_unchecked(5),
                    *inputs.get_unchecked(6),
                    *inputs.get_unchecked(7),
                ]);
                let mut out = [Self::EMPTY; N];
                let mut k = 0;
                while k < 8 {
                    *out.get_unchecked_mut(k) = t[k];
                    k += 1;
                }
                out
            }
        } else if const { arch::ladder_viable(N, GROUP, 4) } {
            // Any other pow-2 shape the certified ladder engine covers (see
            // `polyfills::transpose256`): min-round plan found by compile-time search.
            let plan = const { arch::ladder_search_elem(N, GROUP, 4) };
            unsafe { arch::ladder_radix_by_ps::<N, true>(inputs, plan) }
        } else {
            crate::backend::generic::polyfills::deinterleave_radix_by_default::<Self, N, GROUP>(inputs)
        }
    }

    fn interleave_radix_by<const N: usize, const GROUP: usize>(inputs: [Storage<Self>; N]) -> [Storage<Self>; N] {
        if const { N == 4 && GROUP == 2 } {
            // The pair transpose is its own inverse - reuse the same 8-op sequence.
            // SAFETY: `N == 4` on this arm, so indices 0..4 are in bounds.
            unsafe {
                let t = transpose_4x4_pairs([
                    *inputs.get_unchecked(0),
                    *inputs.get_unchecked(1),
                    *inputs.get_unchecked(2),
                    *inputs.get_unchecked(3),
                ]);
                let mut out = [Self::EMPTY; N];
                *out.get_unchecked_mut(0) = t[0];
                *out.get_unchecked_mut(1) = t[1];
                *out.get_unchecked_mut(2) = t[2];
                *out.get_unchecked_mut(3) = t[3];
                out
            }
        } else if const { N == 8 && GROUP == 1 } {
            // The 8x8 transpose is its own inverse - reuse the shared `transpose256_w32`.
            // SAFETY: `N == 8` on this arm, so indices 0..8 are in bounds.
            unsafe {
                let t = arch::transpose256_w32([
                    *inputs.get_unchecked(0),
                    *inputs.get_unchecked(1),
                    *inputs.get_unchecked(2),
                    *inputs.get_unchecked(3),
                    *inputs.get_unchecked(4),
                    *inputs.get_unchecked(5),
                    *inputs.get_unchecked(6),
                    *inputs.get_unchecked(7),
                ]);
                let mut out = [Self::EMPTY; N];
                let mut k = 0;
                while k < 8 {
                    *out.get_unchecked_mut(k) = t[k];
                    k += 1;
                }
                out
            }
        } else if const { arch::ladder_viable(N, GROUP, 4) } {
            // The certified ladder plan run in the inverse direction (DEINT = false).
            let plan = const { arch::ladder_search_elem(N, GROUP, 4) };
            unsafe { arch::ladder_radix_by_ps::<N, false>(inputs, plan) }
        } else {
            crate::backend::generic::polyfills::interleave_radix_by_default::<Self, N, GROUP>(inputs)
        }
    }

    fn swizzle(a: Storage<Self>, b: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        unsafe {
            let idxs: arch::__m256i = core::mem::transmute(idxs);

            let blend = arch::_mm256_cmpgt_epi32(idxs, arch::_mm256_set1_epi32(7));
            let a_idxs = arch::_mm256_and_si256(idxs, arch::_mm256_set1_epi32(0b111));
            let b_idxs = arch::_mm256_sub_epi32(idxs, arch::_mm256_set1_epi32(8));

            let tmp_a = arch::_mm256_permutevar8x32_ps(a, a_idxs);
            let tmp_b = arch::_mm256_permutevar8x32_ps(b, b_idxs);

            arch::_mm256_blendv_ps(tmp_a, tmp_b, arch::_mm256_castsi256_ps(blend))
        }
    }
}

#[thermite_macros::inline_always]
impl InterleaveRegister for F32x8V3 {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe {
            // 1. In-lane interleaves
            let u_lo = arch::_mm256_unpacklo_ps(a, b);
            let u_hi = arch::_mm256_unpackhi_ps(a, b);

            // 2. Cross-lane permutations
            let res_lo = arch::_mm256_permute2f128_ps(u_lo, u_hi, 0x20);
            let res_hi = arch::_mm256_permute2f128_ps(u_lo, u_hi, 0x31);

            (res_lo, res_hi)
        }
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe {
            let t0 = arch::_mm256_permute2f128_ps(a, b, 0x20);
            let t1 = arch::_mm256_permute2f128_ps(a, b, 0x31);

            let a = arch::_mm256_shuffle_ps(t0, t1, 0x88);
            let b = arch::_mm256_shuffle_ps(t0, t1, 0xDD);

            (a, b)
        }
    }

}

#[thermite_macros::inline_always]
impl IndexableRegister<super::U32x8V3> for F32x8V3 {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<super::U32x8V3>) -> Storage<Self> {
        unsafe { arch::_mm256_i32gather_ps::<4>(ptr, indices) }
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<super::U32x8V3>,
    ) -> Storage<Self> {
        unsafe { arch::_mm256_mask_i32gather_ps::<4>(src, ptr, indices, mask) }
    }
}

#[thermite_macros::inline_always]
impl IndexableRegister<ArrayRegister<super::U64x4V3, 2>> for F32x8V3 {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<ArrayRegister<super::U64x4V3, 2>>) -> Storage<Self> {
        let (lo_idx, hi_idx) = <ArrayRegister<super::U64x4V3, 2> as ConcatRegister<super::U64x4V3>>::split(indices);

        unsafe {
            Self::concat(
                arch::_mm256_i64gather_ps::<4>(ptr, lo_idx),
                arch::_mm256_i64gather_ps::<4>(ptr, hi_idx),
            )
        }
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<ArrayRegister<super::U64x4V3, 2>>,
    ) -> Storage<Self> {
        let (lo_src, hi_src) = Self::split(src);
        let (lo_mask, hi_mask) = Self::split(mask);
        let (lo_idx, hi_idx) = <ArrayRegister<super::U64x4V3, 2> as ConcatRegister<super::U64x4V3>>::split(indices);

        unsafe {
            Self::concat(
                arch::_mm256_mask_i64gather_ps::<4>(lo_src, ptr, lo_idx, lo_mask),
                arch::_mm256_mask_i64gather_ps::<4>(hi_src, ptr, hi_idx, hi_mask),
            )
        }
    }
}

#[thermite_macros::inline_always]
impl ShuffleRegister for F32x8V3 {
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_shuffle_ps(lhs, rhs, IMM8) }
    }
}

#[thermite_macros::inline_always]
impl PermuteRegister for F32x8V3 {
    fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_permutevar8x32_ps(value, const { super::shuffle_to_m256i(IMM8) }) }
    }
}

#[thermite_macros::inline_always]
impl PartialOrdRegister for F32x8V3 {
    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmp_ps(lhs, rhs, arch::_CMP_LT_OQ) }
    }

    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmp_ps(lhs, rhs, arch::_CMP_LE_OQ) }
    }

    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmp_ps(lhs, rhs, arch::_CMP_GT_OQ) }
    }

    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmp_ps(lhs, rhs, arch::_CMP_GE_OQ) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmp_ps(lhs, rhs, arch::_CMP_EQ_OQ) }
    }

    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmp_ps(lhs, rhs, arch::_CMP_NEQ_UQ) }
    }
}

#[thermite_macros::inline_always]
impl NumericRegister for F32x8V3 {
    sort_via_network!(8);

    const ZERO: Storage<Self> = reg::<Self, 8>([0.0; 8]);
    const ONE: Storage<Self> = reg::<Self, 8>([1.0; 8]);
    const TWO: Storage<Self> = reg::<Self, 8>([2.0; 8]);

    const MIN: Storage<Self> = reg::<Self, 8>([f32::MIN; 8]);
    const MAX: Storage<Self> = reg::<Self, 8>([f32::MAX; 8]);

    fn min_element(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_ps_v3!(value; _mm_min_ps _mm_min_ss)
    }

    fn max_element(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_ps_v3!(value; _mm_max_ps _mm_max_ss)
    }

    fn min_max_element(value: Storage<Self>) -> (Self::Element, Self::Element) {
        _mm256_reduce2_ps_v3!(value; _mm_min_ps _mm_min_ss, _mm_max_ps _mm_max_ss)
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_ps_v3!(value; _mm_add_ps _mm_add_ss)
    }

    fn prod_elements(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_ps_v3!(value; _mm_mul_ps _mm_mul_ss)
    }

    fn relaxed_pairwise_sum(lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_hadd_ps(lo, hi) }
    }

    fn pairwise_sum(lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self> {
        // hadd gives [a01,a23,b01,b23,a45,a67,b45,b67]; vpermq [0,2,1,3] -> strict
        let relaxed = Self::relaxed_pairwise_sum(lo, hi);
        unsafe {
            arch::_mm256_castpd_ps(arch::_mm256_permute4x64_pd(
                arch::_mm256_castps_pd(relaxed),
                0b11_01_10_00,
            ))
        }
    }

    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as Unsigned>::USIZE as f32)
    }

    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as f32))
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_add_ps(lhs, rhs) }
    }

    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_sub_ps(lhs, rhs) }
    }

    fn add_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_add_ps(lhs, arch::_mm256_and_ps(rhs, mask)) }
    }

    fn sub_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_sub_ps(lhs, arch::_mm256_and_ps(rhs, mask)) }
    }

    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mul_ps(lhs, rhs) }
    }

    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_div_ps(lhs, rhs) }
    }

    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        // https://stackoverflow.com/a/26342944/2083075
        Self::nmul_adde(Self::trunc(Self::div(lhs, rhs)), rhs, lhs)
    }

    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::fix_min::<Self>(lhs, rhs, unsafe { arch::_mm256_min_ps(lhs, rhs) })
    }

    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::fix_max::<Self>(lhs, rhs, unsafe { arch::_mm256_max_ps(lhs, rhs) })
    }
}

#[thermite_macros::inline_always]
impl SignedRegister for F32x8V3 {
    const NEG_ONE: Storage<Self> = reg::<Self, 8>([-1.0; 8]);
    const MIN_POSITIVE: Storage<Self> = reg::<Self, 8>([f32::MIN_POSITIVE; 8]);

    fn neg(value: Storage<Self>) -> Storage<Self> {
        Self::bitxor(value, Self::NEG_ZERO)
    }

    fn abs(value: Storage<Self>) -> Storage<Self> {
        Self::bitandnot(Self::NEG_ZERO, value)
    }

    fn copysign(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        // take everything but the sign from lhs, and copy the sign from rhs
        Self::bitor(Self::bitandnot(Self::NEG_ZERO, lhs), Self::bitand(Self::NEG_ZERO, rhs))
    }

    fn signum(value: Storage<Self>) -> Storage<Self> {
        let s = Self::bitor(Self::ONE, Self::bitand(value, Self::NEG_ZERO));
        #[cfg(feature = "strict_ieee754")]
        let s = Self::blendv(Self::is_nan(value), s, value);
        s
    }

    fn neg_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        Self::bitxor(value, Self::bitand(Self::NEG_ZERO, mask))
    }
}

#[thermite_macros::inline_always]
impl FloatRegister for F32x8V3 {
    const HAS_TRUE_FMA: bool = true;

    type Bits = super::U32x8V3;
    type SignedBits = super::I32x8V3;
    type ExtendedPrecision = ArrayRegister<super::F64x4V3, 2>;

    const HALF: Storage<Self> = reg::<Self, 8>([0.5; 8]);
    const NEG_ZERO: Storage<Self> = reg::<Self, 8>([-0.0; 8]);
    const EPSILON: Storage<Self> = reg::<Self, 8>([f32::EPSILON; 8]);
    const INFINITY: Storage<Self> = reg::<Self, 8>([f32::INFINITY; 8]);
    const NEG_INFINITY: Storage<Self> = reg::<Self, 8>([f32::NEG_INFINITY; 8]);
    const NAN: Storage<Self> = reg::<Self, 8>([f32::NAN; 8]);

    const EXP_MASK: Storage<Self::Bits> = reg::<Self::Bits, 8>([0x7F800000; 8]);

    fn mul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_fmadd_ps(lhs, rhs, acc) }
    }

    fn mul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_fmsub_ps(lhs, rhs, acc) }
    }

    fn nmul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_fnmadd_ps(lhs, rhs, acc) }
    }

    fn nmul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_fnmsub_ps(lhs, rhs, acc) }
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
        unsafe { arch::_mm256_addsub_ps(a, b) }
    }

    fn fmaddsub(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_fmaddsub_ps(a, b, c) }
    }

    fn fmsubadd(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_fmsubadd_ps(a, b, c) }
    }

    fn sqrt(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_sqrt_ps(value) }
    }

    fn rsqrt(value: Storage<Self>) -> Storage<Self> {
        cfg_select! {
            feature = "strict_ieee754" => {
                Self::rcp(Self::sqrt(value))
            }
            _ => unsafe { arch::_mm256_rsqrt_ps(value) }
        }
    }

    fn rcp(value: Storage<Self>) -> Storage<Self> {
        cfg_select! {
            feature = "strict_ieee754" => {
                Self::div(Self::ONE, value)
            }
            _ => unsafe { arch::_mm256_rcp_ps(value) }
        }
    }

    const HAS_APPROX_RSQRT: bool = cfg!(not(feature = "strict_ieee754"));
    const HAS_APPROX_RCP: bool = cfg!(not(feature = "strict_ieee754"));

    fn floor(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_floor_ps(value) }
    }

    fn ceil(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_ceil_ps(value) }
    }

    fn round(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_round_ps(value, arch::_MM_FROUND_TO_NEAREST_INT | arch::_MM_FROUND_NO_EXC) }
    }

    fn trunc(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_round_ps(value, arch::_MM_FROUND_TO_ZERO | arch::_MM_FROUND_NO_EXC) }
    }

    fn next_up(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_nextupps_v3(value) }
    }

    fn next_down(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_nextdownps_v3(value) }
    }

    const NATIVE_CAP: NativeCapability = NativeCapability::NONE;
}

#[thermite_macros::inline_always]
impl CastRegister<F32x8V3> for ArrayRegister<super::F64x4V3, 2> {
    fn cast_from(value: Storage<F32x8V3>) -> Storage<Self> {
        let (lo, hi) = F32x8V3::split(value);

        unsafe {
            let lo = arch::_mm256_cvtps_pd(lo);
            let hi = arch::_mm256_cvtps_pd(hi);

            ArrayRegister([lo, hi])
        }
    }
}
