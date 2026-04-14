use generic_array::{GenericArray, sequence::GenericSequence, typenum::Unsigned};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, BitwiseRegister, CastRegister, ConcatRegister, CoreRegister, Element, ExtendRegister,
        FloatRegister, IndexableRegister, InterleaveRegister, MaskElement, MaskRegister, NativeCapability,
        NumericRegister, PartialOrdRegister, PermuteRegister, Register, ShuffleRegister, SignedRegister, Storage,
        SwizzleRegister, ZeroUpper, array::ArrayRegister, empty_reg, reg,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct F32x8V3;

impl CoreRegister for F32x8V3 {
    type Lanes = generic_array::typenum::U8;
    type Storage = arch::__m256;
    type Mask = Self;

    const IS_EMULATED: bool = false;

    const ISA: InstructionSet = InstructionSet::X86V3;

    const EMPTY: Storage<Self> = empty_reg::<Self>();

    const HAS_EQUAL_SIZE_MASK: bool = true;

    #[inline(always)]
    fn blendv(mask: Storage<Self::Mask>, on_false: Storage<Self>, on_true: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_blendv_ps(on_false, on_true, mask) }
    }

    #[inline(always)]
    fn z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_and_ps(value, mask) }
    }

    #[inline(always)]
    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_andnot_ps(mask, value) }
    }

    #[inline(always)]
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
}

impl MaskRegister for F32x8V3 {
    const FALSY: Storage<Self> = reg::<Self, 8>([f32::from_bits(0); 8]);
    const TRUTHY: Storage<Self> = reg::<Self, 8>([f32::from_bits(!0); 8]);

    #[inline(always)]
    fn set(mut mask: Storage<Self::Mask>, lane: usize, value: bool) -> Storage<Self> {
        Self::as_array_mut(&mut mask)[lane] = if value { MaskElement::TRUTHY } else { MaskElement::FALSY };
        mask
    }

    #[inline(always)]
    fn test(mask: Storage<Self::Mask>, lane: usize) -> bool {
        Self::as_array(&mask)[lane].to_bool()
    }

    #[inline(always)]
    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm256_castsi256_ps(arch::_mm256_cvtboolx8_to_epi32_mask_v3(value)) }
    }

    #[inline(always)]
    fn all(value: Storage<Self>) -> bool {
        unsafe { arch::_mm256_movemask_ps(value) == 0xff }
    }

    #[inline(always)]
    fn any(value: Storage<Self>) -> bool {
        unsafe { arch::_mm256_movemask_ps(value) != 0 }
    }

    #[inline(always)]
    fn none(value: Storage<Self>) -> bool {
        unsafe { arch::_mm256_movemask_ps(value) == 0 }
    }

    #[inline(always)]
    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        Some(unsafe { arch::_mm256_movemask_ps(value) as u64 })
    }

    #[inline(always)]
    #[cfg(feature = "bitvec")]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = unsafe { arch::_mm256_movemask_ps(value) as u32 };
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

#[rustfmt::skip] #[thermite_macros::bitand_z]
impl BitwiseRegister for F32x8V3 {
    #[masked] fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_xor_ps(lhs, rhs) }
    }

    #[masked] fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_and_ps(lhs, rhs) }
    }

    #[masked] fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_andnot_ps(lhs, rhs) }
    }

    #[masked] fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_or_ps(lhs, rhs) }
    }

    #[masked] fn not(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_xor_ps(value, arch::_mm256_set1_ps(f32::from_bits(!0))) }
    }
}

impl ConcatRegister<super::F32x4V3> for F32x8V3 {
    #[inline(always)]
    fn concat(lo: Storage<super::F32x4V3>, hi: Storage<super::F32x4V3>) -> Storage<Self> {
        unsafe { arch::_mm256_setr_m128(lo, hi) }
    }

    #[inline(always)]
    fn split(value: Storage<Self>) -> (Storage<super::F32x4V3>, Storage<super::F32x4V3>) {
        let lo = unsafe { arch::_mm256_castps256_ps128(value) };
        let hi = unsafe { arch::_mm256_extractf128_ps(value, 1) };
        (lo, hi)
    }
}

impl ExtendRegister<super::F32x4V3> for F32x8V3 {
    #[inline(always)]
    fn extend(value: Storage<super::F32x4V3>) -> Storage<Self> {
        unsafe { arch::_mm256_zextps128_ps256(value) }
    }

    #[inline(always)]
    fn narrow(value: Storage<Self>) -> Storage<super::F32x4V3> {
        unsafe { arch::_mm256_castps256_ps128(value) }
    }
}

impl Register for F32x8V3 {
    type Element = f32;

    type Signed = super::I32x8V3;
    type Unsigned = super::U32x8V3;

    #[inline(always)]
    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe {
            // value != 0.0
            arch::_mm256_castsi256_ps(arch::_mm256_xor_si256(
                arch::_mm256_set1_epi8(-1),
                arch::_mm256_cmpeq_epi32(arch::_mm256_castps_si256(value), arch::_mm256_setzero_si256()),
            ))
        }
    }

    #[inline(always)]
    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        value
    }

    #[inline(always)]
    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        mask
    }

    #[inline(always)]
    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        value // floats support msb masks directly
    }

    #[inline(always)]
    fn new(value: generic_array::GenericArray<f32, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm256_loadu_ps(value.as_ptr()) }
    }

    #[inline(always)]
    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_setr_ps(value, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) }
    }

    #[inline(always)]
    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_set1_ps(value) }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_load_ps(ptr) }
    }

    #[inline(always)]
    unsafe fn load_m(src: Storage<Self>, mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        // use load_z + 2 bitwise ops to emulate load_m without blendv or scalar fallbacks
        unsafe { Self::bitor(Self::load_z(mask, ptr), Self::bitandnot(mask, src)) }
    }

    #[inline(always)]
    unsafe fn load_z(mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_maskload_ps(ptr, arch::_mm256_castps_si256(mask)) }
    }

    #[inline(always)]
    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_loadu_ps(ptr) }
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm256_store_ps(ptr, value) }
    }

    #[inline(always)]
    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm256_storeu_ps(ptr, value) }
    }

    #[inline(always)]
    unsafe fn load_stream(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_castsi256_ps(arch::_mm256_stream_load_si256(ptr as _)) }
    }

    #[inline(always)]
    unsafe fn store_stream(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm256_stream_ps(ptr, value) }
    }

    #[inline(always)]
    unsafe fn lookup(values: &[Self::Element], indices: Storage<Self::Unsigned>) -> Storage<Self> {
        if values.len() <= <Self::Lanes as Unsigned>::USIZE {
            let mut padded: GenericArray<Self::Element, Self::Lanes> = unsafe { core::mem::zeroed() };
            padded[..values.len()].copy_from_slice(values);

            unsafe { arch::_mm256_permutevar8x32_ps(Self::new(padded), indices) }
        } else {
            unsafe { <Self as IndexableRegister<Self::Unsigned>>::gather(values.as_ptr(), indices) }
        }
    }

    #[inline(always)]
    fn reverse(value: Storage<Self>) -> Storage<Self> {
        // TODO: Improve this
        let (lo, hi) = Self::split(value);
        Self::concat(super::F32x4V3::reverse(hi), super::F32x4V3::reverse(lo))
    }

    #[inline(always)]
    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_bswap_psx_v3(value) }
    }
}

impl InterleaveRegister for F32x8V3 {
    #[inline(always)]
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

    #[inline(always)]
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

impl IndexableRegister<super::U32x8V3> for F32x8V3 {
    #[inline(always)]
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<super::U32x8V3>) -> Storage<Self> {
        unsafe { arch::_mm256_i32gather_ps::<4>(ptr, indices) }
    }

    #[inline(always)]
    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<super::U32x8V3>,
    ) -> Storage<Self> {
        unsafe { arch::_mm256_mask_i32gather_ps::<4>(src, ptr, indices, mask) }
    }
}

impl IndexableRegister<ArrayRegister<super::U64x4V3, 2>> for F32x8V3 {
    #[inline(always)]
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<ArrayRegister<super::U64x4V3, 2>>) -> Storage<Self> {
        let (lo_idx, hi_idx) = <ArrayRegister<super::U64x4V3, 2> as ConcatRegister<super::U64x4V3>>::split(indices);

        unsafe {
            Self::concat(
                arch::_mm256_i64gather_ps::<4>(ptr, lo_idx),
                arch::_mm256_i64gather_ps::<4>(ptr, hi_idx),
            )
        }
    }

    #[inline(always)]
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

impl ShuffleRegister for F32x8V3 {
    #[inline(always)]
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_shuffle_ps(lhs, rhs, IMM8) }
    }
}

impl PermuteRegister for F32x8V3 {
    #[inline(always)]
    fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_permutevar8x32_ps(value, const { super::shuffle_to_m256i(IMM8) }) }
    }
}

impl SwizzleRegister for F32x8V3 {
    const HAS_PERMUTEV: bool = true;

    #[inline(always)]
    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm256_permutevar8x32_ps(value, core::mem::transmute(idxs)) }
    }

    #[inline(always)]
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

impl PartialOrdRegister for F32x8V3 {
    #[inline(always)]
    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmp_ps(lhs, rhs, arch::_CMP_LT_OQ) }
    }

    #[inline(always)]
    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmp_ps(lhs, rhs, arch::_CMP_LE_OQ) }
    }

    #[inline(always)]
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmp_ps(lhs, rhs, arch::_CMP_GT_OQ) }
    }

    #[inline(always)]
    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmp_ps(lhs, rhs, arch::_CMP_GE_OQ) }
    }

    #[inline(always)]
    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmp_ps(lhs, rhs, arch::_CMP_EQ_OQ) }
    }

    #[inline(always)]
    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmp_ps(lhs, rhs, arch::_CMP_NEQ_OQ) }
    }
}

#[thermite_macros::bitand_z]
impl NumericRegister for F32x8V3 {
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

    fn sum_elements(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_ps_v3!(value; _mm_add_ps _mm_add_ss)
    }

    fn prod_elements(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_ps_v3!(value; _mm_mul_ps _mm_mul_ss)
    }

    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as Unsigned>::USIZE as f32)
    }

    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as f32))
    }

    #[masked]
    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_add_ps(lhs, rhs) }
    }

    #[masked]
    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_sub_ps(lhs, rhs) }
    }

    fn add_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_add_ps(lhs, arch::_mm256_and_ps(rhs, mask)) }
    }

    fn sub_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_sub_ps(lhs, arch::_mm256_and_ps(rhs, mask)) }
    }

    #[masked]
    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mul_ps(lhs, rhs) }
    }

    #[masked]
    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_div_ps(lhs, rhs) }
    }

    #[masked]
    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        // https://stackoverflow.com/a/26342944/2083075
        Self::nmul_adde(Self::trunc(Self::div(lhs, rhs)), rhs, lhs)
    }

    #[masked]
    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::fix_min::<Self>(lhs, rhs, unsafe { arch::_mm256_min_ps(lhs, rhs) })
    }

    #[masked]
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::fix_max::<Self>(lhs, rhs, unsafe { arch::_mm256_max_ps(lhs, rhs) })
    }
}

#[thermite_macros::bitand_z]
impl SignedRegister for F32x8V3 {
    const NEG_ONE: Storage<Self> = reg::<Self, 8>([-1.0; 8]);
    const MIN_POSITIVE: Storage<Self> = reg::<Self, 8>([f32::MIN_POSITIVE; 8]);

    #[masked]
    fn neg(value: Storage<Self>) -> Storage<Self> {
        Self::bitxor(value, Self::NEG_ZERO)
    }

    #[masked]
    fn abs(value: Storage<Self>) -> Storage<Self> {
        Self::bitandnot(Self::NEG_ZERO, value)
    }

    #[masked]
    fn copysign(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        // take everything but the sign from lhs, and copy the sign from rhs
        Self::bitor(Self::bitandnot(Self::NEG_ZERO, lhs), Self::bitand(Self::NEG_ZERO, rhs))
    }

    fn signum(value: Storage<Self>) -> Storage<Self> {
        Self::bitor(Self::ONE, Self::bitand(value, Self::NEG_ZERO))
    }

    fn neg_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        Self::bitxor(value, Self::bitand(Self::NEG_ZERO, mask))
    }
}

#[thermite_macros::bitand_z]
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

    #[masked]
    fn mul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_fmadd_ps(lhs, rhs, acc) }
    }

    #[masked]
    fn mul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_fmsub_ps(lhs, rhs, acc) }
    }

    #[masked]
    fn nmul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_fnmadd_ps(lhs, rhs, acc) }
    }

    #[masked]
    fn nmul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_fnmsub_ps(lhs, rhs, acc) }
    }

    #[masked]
    fn mul_adde(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        Self::mul_add(lhs, rhs, acc)
    }

    #[masked]
    fn mul_sube(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        Self::mul_sub(lhs, rhs, acc)
    }

    #[masked]
    fn nmul_adde(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        Self::nmul_add(lhs, rhs, acc)
    }

    #[masked]
    fn nmul_sube(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        Self::nmul_sub(lhs, rhs, acc)
    }

    #[masked]
    fn sqrt(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_sqrt_ps(value) }
    }

    #[masked]
    fn rsqrt(value: Storage<Self>) -> Storage<Self> {
        if const { cfg!(feature = "strict_ieee754") } {
            return Self::rcp(Self::sqrt(value));
        }

        unsafe { arch::_mm256_rsqrt_ps(value) }
    }

    #[masked]
    fn rcp(value: Storage<Self>) -> Storage<Self> {
        if const { cfg!(feature = "strict_ieee754") } {
            return Self::div(Self::ONE, value);
        }

        unsafe { arch::_mm256_rcp_ps(value) }
    }

    const HAS_APPROX_RSQRT: bool = cfg!(not(feature = "strict_ieee754"));
    const HAS_APPROX_RCP: bool = cfg!(not(feature = "strict_ieee754"));

    #[masked]
    fn floor(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_floor_ps(value) }
    }

    #[masked]
    fn ceil(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_ceil_ps(value) }
    }

    #[masked]
    fn round(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_round_ps(value, arch::_MM_FROUND_TO_NEAREST_INT | arch::_MM_FROUND_NO_EXC) }
    }

    #[masked]
    fn trunc(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_round_ps(value, arch::_MM_FROUND_TO_ZERO | arch::_MM_FROUND_NO_EXC) }
    }

    #[masked]
    fn next_up(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_nextupps_v3(value) }
    }

    #[masked]
    fn next_down(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_nextdownps_v3(value) }
    }

    const NATIVE_CAP: NativeCapability = NativeCapability::NONE;
}

impl CastRegister<F32x8V3> for ArrayRegister<super::F64x4V3, 2> {
    #[inline(always)]
    fn cast_from(value: Storage<F32x8V3>) -> Storage<Self> {
        let (lo, hi) = F32x8V3::split(value);

        unsafe {
            let lo = arch::_mm256_cvtps_pd(lo);
            let hi = arch::_mm256_cvtps_pd(hi);

            ArrayRegister([lo, hi])
        }
    }
}
