use generic_array::{GenericArray, sequence::GenericSequence, typenum::Unsigned};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, CastRegister, FloatRegister, MaskRegister, NumericRegister, PartialOrdRegister,
        PermuteRegister, Register, ShuffleRegister, SignedRegister, Storage, SwizzleRegister, dp::DoublePumpRegister,
        empty_reg, reg,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct F32x8V3;

impl Register for F32x8V3 {
    type Lanes = generic_array::typenum::U8;

    type Element = f32;
    type Storage = arch::__m256;
    type HalfRegister = super::f32x4::F32x4V3;
    type DoubleRegister = DoublePumpRegister<Self>;

    const ISA: InstructionSet = InstructionSet::X86V3;

    type USize = super::U32x8V3;
    type ISize = super::I32x8V3;

    const EMPTY: Storage<Self> = empty_reg::<Self>();

    #[inline(always)]
    fn join(lo: Storage<Self::HalfRegister>, hi: Storage<Self::HalfRegister>) -> Storage<Self>
    where
        Self::HalfRegister: Register,
    {
        unsafe { arch::_mm256_setr_m128(lo, hi) }
    }

    #[inline(always)]
    fn split(value: Storage<Self>) -> (Storage<Self::HalfRegister>, Storage<Self::HalfRegister>)
    where
        Self::HalfRegister: Register,
    {
        let lo = unsafe { arch::_mm256_castps256_ps128(value) };
        let hi = unsafe { arch::_mm256_extractf128_ps(value, 1) };
        (lo, hi)
    }

    #[inline(always)]
    fn new(value: generic_array::GenericArray<f32, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm256_loadu_ps(value.as_ptr()) }
    }

    #[inline(always)]
    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_setr_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, value) }
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
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_xor_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_and_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_andnot_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_or_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn not(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_xor_ps(value, arch::_mm256_set1_ps(f32::from_bits(!0))) }
    }

    #[inline(always)]
    fn blendv(mask: Storage<Self>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_blendv_ps(lhs, rhs, mask) }
    }

    const HAS_MSB_BLENDV: bool = true;

    #[inline(always)]
    fn reverse(value: Storage<Self>) -> Storage<Self> {
        let (lo, hi) = Self::split(value);
        Self::join(Self::HalfRegister::reverse(hi), Self::HalfRegister::reverse(lo))
    }

    #[inline(always)]
    fn unpack(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe {
            // 1. Unpack: Generate the ABAB pattern (but swizzled lanes)
            // Latency: ~1 cycle
            let v0 = arch::_mm256_unpacklo_ps(a, b);
            let v1 = arch::_mm256_unpackhi_ps(a, b);

            // 2. Permute: Fix the lane ordering
            // Latency: ~3 cycles
            let real_lo = arch::_mm256_permute2f128_ps(v0, v1, 0x20);
            let real_hi = arch::_mm256_permute2f128_ps(v0, v1, 0x31);

            (real_lo, real_hi)
        }
    }

    #[inline(always)]
    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_bswap_psx_v3(value) }
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

impl MaskRegister for F32x8V3 {
    const FALSY: Storage<Self> = reg::<Self, 8>([f32::from_bits(0); 8]);
    const TRUTHY: Storage<Self> = reg::<Self, 8>([f32::from_bits(!0); 8]);

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
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = unsafe { arch::_mm256_movemask_ps(value) as u32 };
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
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

impl NumericRegister for F32x8V3 {
    const ZERO: Storage<Self> = reg::<Self, 8>([0.0; 8]);
    const ONE: Storage<Self> = reg::<Self, 8>([1.0; 8]);
    const TWO: Storage<Self> = reg::<Self, 8>([2.0; 8]);

    const MIN: Storage<Self> = reg::<Self, 8>([f32::MIN; 8]);
    const MAX: Storage<Self> = reg::<Self, 8>([f32::MAX; 8]);

    #[inline(always)]
    fn min_element(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_ps_v3!(value; _mm_min_ps _mm_min_ss)
    }

    #[inline(always)]
    fn max_element(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_ps_v3!(value; _mm_max_ps _mm_max_ss)
    }

    #[inline(always)]
    fn sum_elements(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_ps_v3!(value; _mm_add_ps _mm_add_ss)
    }

    #[inline(always)]
    fn prod_elements(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_ps_v3!(value; _mm_mul_ps _mm_mul_ss)
    }

    #[inline(always)]
    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as Unsigned>::USIZE as f32)
    }

    #[inline(always)]
    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as f32))
    }

    #[inline(always)]
    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_add_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_sub_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mul_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_div_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        // https://stackoverflow.com/a/26342944/2083075
        Self::nmul_adde(Self::trunc(Self::div(lhs, rhs)), rhs, lhs)
    }

    #[inline(always)]
    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_min_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_max_ps(lhs, rhs) }
    }
}

impl SignedRegister for F32x8V3 {
    const NEG_ONE: Storage<Self> = reg::<Self, 8>([-1.0; 8]);
    const MIN_POSITIVE: Storage<Self> = reg::<Self, 8>([f32::MIN_POSITIVE; 8]);

    #[inline(always)]
    fn neg(value: Storage<Self>) -> Storage<Self> {
        Self::bitxor(value, Self::NEG_ZERO)
    }

    #[inline(always)]
    fn abs(value: Storage<Self>) -> Storage<Self> {
        Self::bitandnot(Self::NEG_ZERO, value)
    }

    #[inline(always)]
    fn copysign(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        // take everything but the sign from lhs, and copy the sign from rhs
        Self::bitor(Self::bitandnot(Self::NEG_ZERO, lhs), Self::bitand(Self::NEG_ZERO, rhs))
    }

    #[inline(always)]
    fn signum(value: Storage<Self>) -> Storage<Self> {
        Self::bitor(Self::ONE, Self::bitand(value, Self::NEG_ZERO))
    }

    #[inline(always)]
    fn conditional_negate(value: Storage<Self>, mask: Storage<Self>) -> Storage<Self> {
        Self::bitxor(value, Self::bitand(Self::NEG_ZERO, mask))
    }
}

impl FloatRegister for F32x8V3 {
    const HAS_TRUE_FMA: bool = true;

    type Bits = super::U32x8V3;
    type Signed = super::I32x8V3;
    type ExtendedPrecision = DoublePumpRegister<super::F64x4V3>;

    const HALF: Storage<Self> = reg::<Self, 8>([0.5; 8]);
    const NEG_ZERO: Storage<Self> = reg::<Self, 8>([-0.0; 8]);
    const EPSILON: Storage<Self> = reg::<Self, 8>([f32::EPSILON; 8]);
    const INFINITY: Storage<Self> = reg::<Self, 8>([f32::INFINITY; 8]);
    const NEG_INFINITY: Storage<Self> = reg::<Self, 8>([f32::NEG_INFINITY; 8]);
    const NAN: Storage<Self> = reg::<Self, 8>([f32::NAN; 8]);

    const EXP_MASK: Storage<Self::Bits> = reg::<Self::Bits, 8>([0x7F800000; 8]);

    #[inline(always)]
    fn mul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_fmadd_ps(lhs, rhs, acc) }
    }

    #[inline(always)]
    fn mul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_fmsub_ps(lhs, rhs, acc) }
    }

    #[inline(always)]
    fn nmul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_fnmadd_ps(lhs, rhs, acc) }
    }

    #[inline(always)]
    fn nmul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_fnmsub_ps(lhs, rhs, acc) }
    }

    #[inline(always)]
    fn mul_adde(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        Self::mul_add(lhs, rhs, acc)
    }

    #[inline(always)]
    fn mul_sube(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        Self::mul_sub(lhs, rhs, acc)
    }

    #[inline(always)]
    fn nmul_adde(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        Self::nmul_add(lhs, rhs, acc)
    }

    #[inline(always)]
    fn nmul_sube(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        Self::nmul_sub(lhs, rhs, acc)
    }

    #[inline(always)]
    fn sqrt(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_sqrt_ps(value) }
    }

    #[inline(always)]
    fn rsqrt(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_rsqrt_ps(value) }
    }

    #[inline(always)]
    fn rcp(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_rcp_ps(value) }
    }

    const HAS_APPROX_RSQRT: bool = true;
    const HAS_APPROX_RCP: bool = true;

    #[inline(always)]
    fn floor(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_floor_ps(value) }
    }

    #[inline(always)]
    fn ceil(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_ceil_ps(value) }
    }

    #[inline(always)]
    fn round(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_round_ps(value, arch::_MM_FROUND_TO_NEAREST_INT | arch::_MM_FROUND_NO_EXC) }
    }

    #[inline(always)]
    fn trunc(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_round_ps(value, arch::_MM_FROUND_TO_ZERO | arch::_MM_FROUND_NO_EXC) }
    }

    #[inline(always)]
    fn next_up(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_nextupps_v3(value) }
    }

    #[inline(always)]
    fn next_down(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_nextdownps_v3(value) }
    }
}

impl CastRegister<F32x8V3> for DoublePumpRegister<super::F64x4V3> {
    #[inline(always)]
    fn cast_from(value: Storage<F32x8V3>) -> Storage<Self> {
        let (lo, hi) = F32x8V3::split(value);

        unsafe {
            let lo = arch::_mm256_cvtps_pd(lo);
            let hi = arch::_mm256_cvtps_pd(hi);

            DoublePumpRegister::join(lo, hi)
        }
    }
}
