use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, BitwiseRegister, CastRegister, ConcatRegister, CoreRegister, Element, ExtendRegister,
        FloatRegister, IndexableRegister, InterleaveRegister, LinAlg3Register, LinAlg4Register, MaskElement,
        MaskRegister, NativeCapability, NumericRegister, PartialOrdRegister, PermuteRegister, Register,
        ShuffleRegister, SignedRegister, Storage, ZeroUpper, array::ArrayRegister, empty_reg, reg,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct F64x4V3;

/// Plain 4x4 `f64` transpose (`(de)interleave_radix_by::<4, 1>` on [`F64x4V3`]): the
/// `W = 8` bytes square transpose. `f64x4`'s storage is already the pd domain the shared
/// [`transpose256_w64`](arch::transpose256_w64) is spelled in, so this is a direct call -
/// no cast. Same 8-op body every 64-bit-element 256-bit register (i64x4/u64x4, f32x8
/// `(4,2)`) reuses. Its own inverse, so interleave reuses it.
#[inline(always)]
fn transpose_4x4_f64(i: [arch::__m256d; 4]) -> [arch::__m256d; 4] {
    unsafe { arch::transpose256_w64(i) }
}

#[thermite_macros::inline_always]
impl CoreRegister for F64x4V3 {
    type Lanes = typenum::U4;
    type Storage = arch::__m256d;
    type Mask = Self;

    const IS_EMULATED: bool = false;

    const ISA: InstructionSet = InstructionSet::X86V3;

    const EMPTY: Storage<Self> = empty_reg::<Self>();

    const HAS_EQUAL_SIZE_MASK: bool = true;

    fn blendv(mask: Storage<Self::Mask>, on_false: Storage<Self>, on_true: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_blendv_pd(on_false, on_true, mask) }
    }

    fn zz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_and_pd(value, mask) }
    }

    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_andnot_pd(mask, value) }
    }

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        if const { Z::N >= 4 } {
            value
        } else if const { Z::N == 2 } {
            unsafe { arch::_mm256_zextpd128_pd256(arch::_mm256_castpd256_pd128(value)) }
        } else {
            unsafe {
                arch::_mm256_and_pd(
                    value,
                    arch::_mm256_castsi256_pd(arch::_mm256_zeroupper_mask_epi64::<Z>()),
                )
            }
        }
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        mask
    }
}

#[thermite_macros::inline_always]
impl MaskRegister for F64x4V3 {
    const FALSY: Storage<Self> = reg::<Self, 4>([f64::from_bits(0); 4]);
    const TRUTHY: Storage<Self> = reg::<Self, 4>([f64::from_bits(!0); 4]);

    fn set(mut mask: Storage<Self::Mask>, lane: usize, value: bool) -> Storage<Self> {
        Self::as_mut_slice(&mut mask)[lane] = if value { MaskElement::TRUTHY } else { MaskElement::FALSY };
        mask
    }

    fn test(mask: Storage<Self::Mask>, lane: usize) -> bool {
        Self::as_slice(&mask)[lane].to_bool()
    }

    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm256_castsi256_pd(arch::_mm256_cvtboolx4_to_epi64_mask_v3(value)) }
    }

    fn all(value: Storage<Self>) -> bool {
        unsafe { arch::_mm256_movemask_pd(value) == 0b1111 }
    }

    fn any(value: Storage<Self>) -> bool {
        unsafe { arch::_mm256_movemask_pd(value) != 0 }
    }

    fn none(value: Storage<Self>) -> bool {
        unsafe { arch::_mm256_movemask_pd(value) == 0 }
    }

    fn from_native_bitmask(bitmask: u64) -> Storage<Self> {
        unsafe { arch::_mm256_castsi256_pd(arch::_mm256_movm_epi64x_v3(bitmask)) }
    }

    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        Some(unsafe { arch::_mm256_movemask_pd(value) as u64 })
    }

    #[cfg(feature = "bitvec")]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = unsafe { arch::_mm256_movemask_pd(value) as u32 };
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for F64x4V3 {
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_xor_pd(lhs, rhs) }
    }

    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_and_pd(lhs, rhs) }
    }

    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_andnot_pd(lhs, rhs) }
    }

    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_or_pd(lhs, rhs) }
    }

    fn not(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_xor_pd(value, arch::_mm256_set1_pd(f64::from_bits(!0))) }
    }
}

#[thermite_macros::inline_always]
impl ConcatRegister<super::F64x2V3> for F64x4V3 {
    fn concat(lo: Storage<super::F64x2V3>, hi: Storage<super::F64x2V3>) -> Storage<Self> {
        unsafe { arch::_mm256_setr_m128d(lo, hi) }
    }

    fn split(value: Storage<Self>) -> (Storage<super::F64x2V3>, Storage<super::F64x2V3>) {
        let lo = unsafe { arch::_mm256_castpd256_pd128(value) };
        let hi = unsafe { arch::_mm256_extractf128_pd(value, 1) };
        (lo, hi)
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<super::F64x2V3> for F64x4V3 {
    fn extend(value: Storage<super::F64x2V3>) -> Storage<Self> {
        unsafe { arch::_mm256_zextpd128_pd256(value) }
    }

    fn narrow(value: Storage<Self>) -> Storage<super::F64x2V3> {
        unsafe { arch::_mm256_castpd256_pd128(value) }
    }
}

#[thermite_macros::inline_always]
impl Register for F64x4V3 {
    type Element = f64;

    type Signed = super::I64x4V3;
    type Unsigned = super::U64x4V3;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe {
            // value != 0.0
            arch::_mm256_castsi256_pd(arch::_mm256_xor_si256(
                arch::_mm256_set1_epi8(-1),
                arch::_mm256_cmpeq_epi64(arch::_mm256_castpd_si256(value), arch::_mm256_setzero_si256()),
            ))
        }
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        value
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        value // floats support msb masks directly
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm256_loadu_pd(value.as_ptr() as *const _) }
    }

    fn single(value: Self::Element) -> crate::register::Storage<Self> {
        unsafe { arch::_mm256_setr_pd(value, 0.0, 0.0, 0.0) }
    }

    impl_native_radix3!(arch::_mm256_interleave3_pd, arch::_mm256_deinterleave3_pd);

    fn interleave_by<const GROUP: usize>(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        if const { GROUP == 2 } {
            // Pair granularity on `f64x4` is a 128-bit-lane (whole-pair) interleave: `lo` is
            // `[a.P0, b.P0]`, `hi` is `[a.P1, b.P1]` - two `permute2f128`s, no `unpck` needed.
            unsafe {
                (
                    arch::_mm256_permute2f128_pd(a, b, 0x20),
                    arch::_mm256_permute2f128_pd(a, b, 0x31),
                )
            }
        } else {
            crate::backend::generic::polyfills::interleave_by_default::<Self, GROUP>(a, b)
        }
    }

    fn deinterleave_by<const GROUP: usize>(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        if const { GROUP == 2 } {
            // Its own inverse: the 2x2 128-bit block transpose. `a = [lo.P0, hi.P0]`,
            // `b = [lo.P1, hi.P1]`.
            unsafe {
                (
                    arch::_mm256_permute2f128_pd(a, b, 0x20),
                    arch::_mm256_permute2f128_pd(a, b, 0x31),
                )
            }
        } else {
            crate::backend::generic::polyfills::deinterleave_by_default::<Self, GROUP>(a, b)
        }
    }

    // The `(N, GROUP) == (4, 1)` square case is the 4x4 `f64` transpose (8 ops,
    // its own inverse); `interleave_radix_by` reuses the same body. Everything else
    // defers to the generic default (which forwards `GROUP == 1` back to the native
    // `deinterleave_radix`/radix-3 paths).
    fn deinterleave_radix_by<const N: usize, const GROUP: usize>(inputs: [Storage<Self>; N]) -> [Storage<Self>; N] {
        if const { N == 4 && GROUP == 1 } {
            // SAFETY: `N == 4` on this arm, so indices 0..4 are in bounds.
            unsafe {
                let t = transpose_4x4_f64([
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
        } else if const { arch::ladder_viable(N, GROUP, 8) } {
            // Certified ladder plan for any other pow-2 shape (see `polyfills::transpose256`);
            // f64 declares its 8-byte elements so GROUP converts to 32-bit slots.
            let plan = const { arch::ladder_search_elem(N, GROUP, 8) };
            unsafe { arch::ladder_radix_by_pd::<N, true>(inputs, plan) }
        } else {
            crate::backend::generic::polyfills::deinterleave_radix_by_default::<Self, N, GROUP>(inputs)
        }
    }

    fn interleave_radix_by<const N: usize, const GROUP: usize>(inputs: [Storage<Self>; N]) -> [Storage<Self>; N] {
        if const { N == 4 && GROUP == 1 } {
            // SAFETY: `N == 4` on this arm, so indices 0..4 are in bounds.
            unsafe {
                let t = transpose_4x4_f64([
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
        } else if const { arch::ladder_viable(N, GROUP, 8) } {
            let plan = const { arch::ladder_search_elem(N, GROUP, 8) };
            unsafe { arch::ladder_radix_by_pd::<N, false>(inputs, plan) }
        } else {
            crate::backend::generic::polyfills::interleave_radix_by_default::<Self, N, GROUP>(inputs)
        }
    }

    impl_native_extract!(@pd256);

    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_set1_pd(value) }
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_load_pd(ptr) }
    }

    unsafe fn load_m(src: Storage<Self>, mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        // use load_z + 2 bitwise ops to emulate load_m without blendv or scalar fallbacks
        unsafe { Self::bitor(Self::load_z(mask, ptr), Self::bitandnot(mask, src)) }
    }

    unsafe fn load_z(mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_maskload_pd(ptr, arch::_mm256_castpd_si256(mask)) }
    }

    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_loadu_pd(ptr) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm256_store_pd(ptr, value) }
    }

    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm256_storeu_pd(ptr, value) }
    }

    unsafe fn load_stream(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_castsi256_pd(arch::_mm256_stream_load_si256(ptr as _)) }
    }

    unsafe fn store_stream(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm256_stream_pd(ptr, value) }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_permute4x64_pd::<{ MM_SHUFFLE!(0, 1, 2, 3) }>(value) }
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_bswap_pdx_v3(value) }
    }

    const HAS_PERMUTEV: bool = true;

    impl_float_align_via_bits!(super::U64x4V3);

    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        unsafe {
            let idxs: arch::__m128i = core::mem::transmute(idxs); // [i0, i1, i2, i3]
            let even = arch::_mm_slli_epi32(idxs, 1); // [2i0, 2i1, 2i2, 2i3]
            let odd = arch::_mm_add_epi32(even, arch::_mm_set1_epi32(1)); // [2i0+1, ...]
            // interleave -> [2i0,2i0+1, 2i1,2i1+1 | 2i2,2i2+1, 2i3,2i3+1]
            let idx8 = arch::_mm256_set_m128i(arch::_mm_unpackhi_epi32(even, odd), arch::_mm_unpacklo_epi32(even, odd));
            arch::_mm256_castps_pd(arch::_mm256_permutevar8x32_ps(arch::_mm256_castpd_ps(value), idx8))
        }
    }

    compress_via_table!();
}

#[thermite_macros::inline_always]
impl InterleaveRegister for F64x4V3 {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe {
            let u_lo = arch::_mm256_unpacklo_pd(a, b);
            let u_hi = arch::_mm256_unpackhi_pd(a, b);

            let res_lo = arch::_mm256_permute2f128_pd(u_lo, u_hi, 0x20);
            let res_hi = arch::_mm256_permute2f128_pd(u_lo, u_hi, 0x31);

            (res_lo, res_hi)
        }
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe {
            let t0 = arch::_mm256_permute2f128_pd(a, b, 0x20);
            let t1 = arch::_mm256_permute2f128_pd(a, b, 0x31);

            let a = arch::_mm256_unpacklo_pd(t0, t1);
            let b = arch::_mm256_unpackhi_pd(t0, t1);

            (a, b)
        }
    }
}

#[thermite_macros::inline_always]
impl IndexableRegister<super::U32x4V3> for F64x4V3 {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<super::U32x4V3>) -> Storage<Self> {
        unsafe { arch::_mm256_i32gather_pd::<8>(ptr as *const _, indices) }
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<super::U32x4V3>,
    ) -> Storage<Self> {
        unsafe { arch::_mm256_mask_i32gather_pd::<8>(src, ptr as *const _, indices, mask) }
    }
}

#[thermite_macros::inline_always]
impl IndexableRegister<super::U64x4V3> for F64x4V3 {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<super::U64x4V3>) -> Storage<Self> {
        unsafe { arch::_mm256_i64gather_pd::<8>(ptr, indices) }
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<super::U64x4V3>,
    ) -> Storage<Self> {
        unsafe { arch::_mm256_mask_i64gather_pd::<8>(src, ptr, indices, mask) }
    }
}

#[thermite_macros::inline_always]
impl IndexableRegister<super::U32x8V3> for ArrayRegister<F64x4V3, 2> {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<super::U32x8V3>) -> Storage<Self> {
        let (lo, hi) = <super::U32x8V3>::split(indices);

        unsafe {
            let lo = arch::_mm256_i32gather_pd::<8>(ptr as *const _, lo);
            let hi = arch::_mm256_i32gather_pd::<8>(ptr as *const _, hi);

            ArrayRegister([lo, hi])
        }
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<super::U32x8V3>,
    ) -> Storage<Self> {
        let (lo, hi) = <super::U32x8V3>::split(indices);

        unsafe {
            let lo = arch::_mm256_mask_i32gather_pd::<8>(src.0[0], ptr as *const _, lo, mask.0[0]);
            let hi = arch::_mm256_mask_i32gather_pd::<8>(src.0[1], ptr as *const _, hi, mask.0[1]);

            ArrayRegister([lo, hi])
        }
    }
}

#[thermite_macros::inline_always]
impl ShuffleRegister for F64x4V3 {
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_shuffle_pd(lhs, rhs, IMM8) }
    }
}

#[thermite_macros::inline_always]
impl PermuteRegister for F64x4V3 {
    fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_permute4x64_pd(value, IMM8) }
    }
}

#[thermite_macros::inline_always]
impl PartialOrdRegister for F64x4V3 {
    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmp_pd(lhs, rhs, arch::_CMP_LT_OQ) }
    }

    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmp_pd(lhs, rhs, arch::_CMP_LE_OQ) }
    }

    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmp_pd(lhs, rhs, arch::_CMP_GT_OQ) }
    }

    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmp_pd(lhs, rhs, arch::_CMP_GE_OQ) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmp_pd(lhs, rhs, arch::_CMP_EQ_OQ) }
    }

    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmp_pd(lhs, rhs, arch::_CMP_NEQ_UQ) }
    }
}

#[thermite_macros::inline_always]
impl NumericRegister for F64x4V3 {
    const ZERO: Storage<Self> = reg::<Self, 4>([0.0; 4]);
    const ONE: Storage<Self> = reg::<Self, 4>([1.0; 4]);
    const TWO: Storage<Self> = reg::<Self, 4>([2.0; 4]);

    const MIN: Storage<Self> = reg::<Self, 4>([f64::MIN; 4]);
    const MAX: Storage<Self> = reg::<Self, 4>([f64::MAX; 4]);

    fn min_element(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_pd_v3!(value; _mm_min_pd _mm_min_sd)
    }

    fn max_element(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_pd_v3!(value; _mm_max_pd _mm_max_sd)
    }

    fn min_max_element(value: Storage<Self>) -> (Self::Element, Self::Element) {
        _mm256_reduce2_pd_v3!(value; _mm_min_pd _mm_min_sd, _mm_max_pd _mm_max_sd)
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_pd_v3!(value; _mm_add_pd _mm_add_sd)
    }

    fn prod_elements(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_pd_v3!(value; _mm_mul_pd _mm_mul_sd)
    }

    fn relaxed_pairwise_sum(lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_hadd_pd(lo, hi) }
    }

    fn pairwise_sum(lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self> {
        // hadd gives [a0+a1,b0+b1,a2+a3,b2+b3]; vpermq [0,2,1,3] -> strict
        let relaxed = Self::relaxed_pairwise_sum(lo, hi);
        unsafe { arch::_mm256_permute4x64_pd(relaxed, 0b11_01_10_00) }
    }

    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as typenum::Unsigned>::USIZE as f64)
    }

    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as f64))
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_add_pd(lhs, rhs) }
    }

    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_sub_pd(lhs, rhs) }
    }

    fn add_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_add_pd(lhs, arch::_mm256_and_pd(rhs, mask)) }
    }

    fn sub_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_sub_pd(lhs, arch::_mm256_and_pd(rhs, mask)) }
    }

    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mul_pd(lhs, rhs) }
    }

    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_div_pd(lhs, rhs) }
    }

    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        // https://stackoverflow.com/a/26342944/2083075
        Self::nmul_adde(Self::trunc(Self::div(lhs, rhs)), rhs, lhs)
    }

    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::fix_min::<Self>(lhs, rhs, unsafe { arch::_mm256_min_pd(lhs, rhs) })
    }

    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::fix_max::<Self>(lhs, rhs, unsafe { arch::_mm256_max_pd(lhs, rhs) })
    }
}

#[thermite_macros::inline_always]
impl SignedRegister for F64x4V3 {
    const NEG_ONE: Storage<Self> = reg::<Self, 4>([-1.0; 4]);
    const MIN_POSITIVE: Storage<Self> = reg::<Self, 4>([f64::MIN_POSITIVE; 4]);

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
impl FloatRegister for F64x4V3 {
    const HAS_TRUE_FMA: bool = true;

    type Bits = super::U64x4V3;
    type SignedBits = super::I64x4V3;
    type ExtendedPrecision = Self; // f64 is the highest precision available

    const HALF: Storage<Self> = reg::<Self, 4>([0.5; 4]);
    const NEG_ZERO: Storage<Self> = reg::<Self, 4>([-0.0; 4]);
    const EPSILON: Storage<Self> = reg::<Self, 4>([f64::EPSILON; 4]);
    const INFINITY: Storage<Self> = reg::<Self, 4>([f64::INFINITY; 4]);
    const NEG_INFINITY: Storage<Self> = reg::<Self, 4>([f64::NEG_INFINITY; 4]);
    const NAN: Storage<Self> = reg::<Self, 4>([f64::NAN; 4]);

    const EXP_MASK: Storage<Self::Bits> = reg::<Self::Bits, 4>([0x7FF0_0000_0000_0000; 4]);

    fn mul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_fmadd_pd(lhs, rhs, acc) }
    }

    fn mul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_fmsub_pd(lhs, rhs, acc) }
    }

    fn nmul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_fnmadd_pd(lhs, rhs, acc) }
    }

    fn nmul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_fnmsub_pd(lhs, rhs, acc) }
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
        unsafe { arch::_mm256_addsub_pd(a, b) }
    }

    fn fmaddsub(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_fmaddsub_pd(a, b, c) }
    }

    fn fmsubadd(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_fmsubadd_pd(a, b, c) }
    }

    fn sqrt(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_sqrt_pd(value) }
    }

    const HAS_APPROX_RCP: bool = false;
    const HAS_APPROX_RSQRT: bool = false;

    fn floor(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_floor_pd(value) }
    }

    fn ceil(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_ceil_pd(value) }
    }

    fn round(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_round_pd(value, arch::_MM_FROUND_TO_NEAREST_INT | arch::_MM_FROUND_NO_EXC) }
    }

    fn trunc(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_round_pd(value, arch::_MM_FROUND_TO_ZERO | arch::_MM_FROUND_NO_EXC) }
    }

    fn next_up(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_nextuppd_v3(value) }
    }

    fn next_down(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_nextdownpd_v3(value) }
    }

    const NATIVE_CAP: NativeCapability = NativeCapability::NONE;
}

#[thermite_macros::inline_always]
impl LinAlg4Register for F64x4V3 {
    // Use 2x128-bit registers on AVX2 f64x4 to avoid lane-crossing shuffles
    // that are significantly slower than just using two xmm registers.
    fn mat4_inverse(m: &mut [Storage<Self>; 4]) -> f64 {
        type Paired = ArrayRegister<super::F64x2V3, 2>;

        let mut pm: [Storage<Paired>; 4] = [Paired::EMPTY; 4];

        let mut i = 0;
        while i < 4 {
            let (lo, hi) = Self::split(m[i]);
            pm[i] = <Paired as ConcatRegister<super::F64x2V3>>::concat(lo, hi);
            i += 1;
        }

        // On a singular matrix the paired inverse leaves `pm` untouched, so
        // recombining it unconditionally yields the original (un-clobbered) matrix.
        let det = <Paired as LinAlg4Register>::mat4_inverse(&mut pm);

        let mut i = 0;
        while i < 4 {
            let (lo, hi) = <Paired as ConcatRegister<super::F64x2V3>>::split(pm[i]);
            m[i] = Self::concat(lo, hi);
            i += 1;
        }

        det
    }

    fn mat4_det(cols: &[Storage<Self>; 4]) -> f64 {
        // Same paired-128 split as the inverse; no recombine needed.
        type Paired = ArrayRegister<super::F64x2V3, 2>;

        let mut pm: [Storage<Paired>; 4] = [Paired::EMPTY; 4];

        let mut i = 0;
        while i < 4 {
            let (lo, hi) = Self::split(cols[i]);
            pm[i] = <Paired as ConcatRegister<super::F64x2V3>>::concat(lo, hi);
            i += 1;
        }

        <Paired as LinAlg4Register>::mat4_det(&pm)
    }
}

#[thermite_macros::inline_always]
impl LinAlg3Register for F64x4V3 {
    fn dot3(lhs: Storage<Self>, rhs: Storage<Self>) -> f64 {
        // Borrowed from glam
        unsafe {
            let x2_y2_z2_w2 = arch::_mm256_mul_pd(lhs, rhs);
            let y2_0_0_0 = arch::_mm256_permute4x64_pd(x2_y2_z2_w2, 0b00_00_00_01);
            let z2_0_0_0 = arch::_mm256_permute4x64_pd(x2_y2_z2_w2, 0b00_00_00_10);
            let x2y2_0_0_0 = arch::_mm256_add_pd(x2_y2_z2_w2, y2_0_0_0);
            arch::_mm256_cvtsd_f64(arch::_mm256_add_pd(x2y2_0_0_0, z2_0_0_0))
        }
    }

    fn zero4(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_blend_pd(value, arch::_mm256_set1_pd(0.0), 0b1000) }
    }

    fn one4(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_blend_pd(value, arch::_mm256_set1_pd(1.0), 0b1000) }
    }

    fn min_element3(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_pd3_v3!(value; _mm_min_sd)
    }

    fn max_element3(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_pd3_v3!(value; _mm_max_sd)
    }

    fn sum_elements3(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_pd3_v3!(value; _mm_add_sd)
    }

    fn prod_elements3(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_pd3_v3!(value; _mm_mul_sd)
    }
}

#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<F64x4V3, 2>> for super::F32x8V3 {
    fn cast_from(value: Storage<ArrayRegister<F64x4V3, 2>>) -> Storage<Self> {
        let (lo, hi) = <ArrayRegister<F64x4V3, 2> as ConcatRegister<F64x4V3>>::split(value);

        unsafe {
            let lo = arch::_mm256_cvtpd_ps(lo);
            let hi = arch::_mm256_cvtpd_ps(hi);

            Self::concat(lo, hi)
        }
    }
}
