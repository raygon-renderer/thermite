use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, CastRegister, CoreRegister, FloatRegister, LinAlg3Register, LinAlg4Register, NumericRegister,
        PartialOrdRegister, PermuteRegister, Register, ShuffleRegister, SignedRegister, Storage, SwizzleRegister,
        dp::DoublePumpRegister, empty_reg, reg,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct F64x4V3;

impl CoreRegister for F64x4V3 {
    type Lanes = typenum::U4;
    type Element = f64;
    type Storage = arch::__m256d;
}

impl Register for F64x4V3 {
    type HalfRegister = super::F64x2V3;
    type DoubleRegister = DoublePumpRegister<Self>;

    const IS_EMULATED: bool = false;

    const ISA: InstructionSet = InstructionSet::X86V3;

    type ISize = super::I64x4V3;
    type USize = super::U64x4V3;

    const EMPTY: Storage<Self> = empty_reg::<Self>();

    #[inline(always)]
    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm256_loadu_pd(value.as_ptr() as *const _) }
    }

    #[inline(always)]
    fn single(value: Self::Element) -> crate::register::Storage<Self> {
        unsafe { arch::_mm256_setr_pd(value, 0.0, 0.0, 0.0) }
    }

    #[inline(always)]
    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_set1_pd(value) }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_load_pd(ptr) }
    }

    #[inline(always)]
    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_loadu_pd(ptr) }
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm256_store_pd(ptr, value) }
    }

    #[inline(always)]
    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm256_storeu_pd(ptr, value) }
    }

    #[inline(always)]
    unsafe fn load_stream(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_castsi256_pd(arch::_mm256_stream_load_si256(ptr as _)) }
    }

    #[inline(always)]
    unsafe fn store_stream(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm256_stream_pd(ptr, value) }
    }

    const FALSY: Storage<Self> = reg::<Self, 4>([f64::from_bits(0); 4]);
    const TRUTHY: Storage<Self> = reg::<Self, 4>([f64::from_bits(!0); 4]);

    #[inline(always)]
    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm256_castsi256_pd(arch::_mm256_cvtboolx4_to_epi64_mask_v3(value)) }
    }

    #[inline(always)]
    fn all(value: Storage<Self>) -> bool {
        unsafe { arch::_mm256_movemask_pd(value) == 0b1111 }
    }

    #[inline(always)]
    fn any(value: Storage<Self>) -> bool {
        unsafe { arch::_mm256_movemask_pd(value) != 0 }
    }

    #[inline(always)]
    fn none(value: Storage<Self>) -> bool {
        unsafe { arch::_mm256_movemask_pd(value) == 0 }
    }

    #[inline(always)]
    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        Some(unsafe { arch::_mm256_movemask_pd(value) as u64 })
    }

    #[inline(always)]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = unsafe { arch::_mm256_movemask_pd(value) as u32 };
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }

    #[inline(always)]
    fn join(lo: Storage<Self::HalfRegister>, hi: Storage<Self::HalfRegister>) -> Storage<Self>
    where
        Self::HalfRegister: Register,
    {
        unsafe { arch::_mm256_setr_m128d(lo, hi) }
    }

    #[inline(always)]
    fn split(value: Storage<Self>) -> (Storage<Self::HalfRegister>, Storage<Self::HalfRegister>)
    where
        Self::HalfRegister: Register,
    {
        let lo = unsafe { arch::_mm256_castpd256_pd128(value) };
        let hi = unsafe { arch::_mm256_extractf128_pd(value, 1) };
        (lo, hi)
    }

    #[inline(always)]
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_xor_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_and_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_andnot_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_or_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn not(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_xor_pd(value, arch::_mm256_set1_pd(f64::from_bits(!0))) }
    }

    #[inline(always)]
    fn blendv(mask: Storage<Self>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_blendv_pd(lhs, rhs, mask) }
    }

    const HAS_MSB_BLENDV: bool = true;

    #[inline(always)]
    fn reverse(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_permute4x64_pd::<{ MM_SHUFFLE!(0, 1, 2, 3) }>(value) }
    }

    const HAS_SIMPLE_UNPACK: bool = false;

    #[inline(always)]
    fn unpack(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe {
            // 1. Local Interleave
            // Use shuffle_pd (0x0/0xF) instead of unpacklo/hi for that 0.5 CPI throughput on some CPUs
            let t1 = arch::_mm256_shuffle_pd(a, b, 0x0); // "unpacklo" equivalent
            let t2 = arch::_mm256_shuffle_pd(a, b, 0xF); // "unpackhi" equivalent

            // 2. Lane Fix, may run in parallel
            let res_a = arch::_mm256_permute4x64_pd(t1, 0xD8);
            let res_b = arch::_mm256_permute4x64_pd(t2, 0xD8);

            (res_a, res_b)
        }
    }

    #[inline(always)]
    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_bswap_pdx_v3(value) }
    }
}

impl ShuffleRegister for F64x4V3 {
    #[inline(always)]
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_shuffle_pd(lhs, rhs, IMM8) }
    }
}

impl PermuteRegister for F64x4V3 {
    #[inline(always)]
    fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_permute4x64_pd(value, IMM8) }
    }
}

impl SwizzleRegister for F64x4V3 {
    const HAS_PERMUTEV: bool = true;

    #[inline(always)]
    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        unsafe {
            let idxs: arch::__m128i = core::mem::transmute(idxs);
            arch::_mm256_permutevar_pd(value, arch::_mm256_cvtepu32_epi64(idxs))
        }
    }
}

impl PartialOrdRegister for F64x4V3 {
    #[inline(always)]
    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmp_pd(lhs, rhs, arch::_CMP_LT_OQ) }
    }

    #[inline(always)]
    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmp_pd(lhs, rhs, arch::_CMP_LE_OQ) }
    }

    #[inline(always)]
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmp_pd(lhs, rhs, arch::_CMP_GT_OQ) }
    }

    #[inline(always)]
    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmp_pd(lhs, rhs, arch::_CMP_GE_OQ) }
    }

    #[inline(always)]
    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmp_pd(lhs, rhs, arch::_CMP_EQ_OQ) }
    }

    #[inline(always)]
    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmp_pd(lhs, rhs, arch::_CMP_NEQ_OQ) }
    }
}

impl NumericRegister for F64x4V3 {
    const ZERO: Storage<Self> = reg::<Self, 4>([0.0; 4]);
    const ONE: Storage<Self> = reg::<Self, 4>([1.0; 4]);
    const TWO: Storage<Self> = reg::<Self, 4>([2.0; 4]);

    const MIN: Storage<Self> = reg::<Self, 4>([f64::MIN; 4]);
    const MAX: Storage<Self> = reg::<Self, 4>([f64::MAX; 4]);

    #[inline(always)]
    fn min_element(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_pd_v3!(value; _mm_min_pd _mm_min_sd)
    }

    #[inline(always)]
    fn max_element(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_pd_v3!(value; _mm_max_pd _mm_max_sd)
    }

    #[inline(always)]
    fn sum_elements(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_pd_v3!(value; _mm_add_pd _mm_add_sd)
    }

    #[inline(always)]
    fn prod_elements(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_pd_v3!(value; _mm_mul_pd _mm_mul_sd)
    }

    #[inline(always)]
    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as typenum::Unsigned>::USIZE as f64)
    }

    #[inline(always)]
    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as f64))
    }

    #[inline(always)]
    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_add_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_sub_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mul_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_div_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        // https://stackoverflow.com/a/26342944/2083075
        Self::nmul_adde(Self::trunc(Self::div(lhs, rhs)), rhs, lhs)
    }

    #[inline(always)]
    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_min_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_max_pd(lhs, rhs) }
    }
}

impl SignedRegister for F64x4V3 {
    const NEG_ONE: Storage<Self> = reg::<Self, 4>([-1.0; 4]);
    const MIN_POSITIVE: Storage<Self> = reg::<Self, 4>([f64::MIN_POSITIVE; 4]);

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

impl FloatRegister for F64x4V3 {
    const HAS_TRUE_FMA: bool = true;

    type Bits = super::U64x4V3;
    type Signed = super::I64x4V3;
    type ExtendedPrecision = Self; // f64 is the highest precision available

    const HALF: Storage<Self> = reg::<Self, 4>([0.5; 4]);
    const NEG_ZERO: Storage<Self> = reg::<Self, 4>([-0.0; 4]);
    const EPSILON: Storage<Self> = reg::<Self, 4>([f64::EPSILON; 4]);
    const INFINITY: Storage<Self> = reg::<Self, 4>([f64::INFINITY; 4]);
    const NEG_INFINITY: Storage<Self> = reg::<Self, 4>([f64::NEG_INFINITY; 4]);
    const NAN: Storage<Self> = reg::<Self, 4>([f64::NAN; 4]);

    const EXP_MASK: crate::register::Storage<Self::Bits> = reg::<Self::Bits, 4>([0x7FF0_0000_0000_0000; 4]);

    #[inline(always)]
    fn mul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_fmadd_pd(lhs, rhs, acc) }
    }

    #[inline(always)]
    fn mul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_fmsub_pd(lhs, rhs, acc) }
    }

    #[inline(always)]
    fn nmul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_fnmadd_pd(lhs, rhs, acc) }
    }

    #[inline(always)]
    fn nmul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_fnmsub_pd(lhs, rhs, acc) }
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
        unsafe { arch::_mm256_sqrt_pd(value) }
    }

    const HAS_APPROX_RCP: bool = false;
    const HAS_APPROX_RSQRT: bool = false;

    #[inline(always)]
    fn floor(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_floor_pd(value) }
    }

    #[inline(always)]
    fn ceil(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_ceil_pd(value) }
    }

    #[inline(always)]
    fn round(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_round_pd(value, arch::_MM_FROUND_TO_NEAREST_INT | arch::_MM_FROUND_NO_EXC) }
    }

    #[inline(always)]
    fn trunc(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_round_pd(value, arch::_MM_FROUND_TO_ZERO | arch::_MM_FROUND_NO_EXC) }
    }

    #[inline(always)]
    fn next_up(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_nextuppd_v3(value) }
    }

    #[inline(always)]
    fn next_down(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_nextdownpd_v3(value) }
    }

    const HAS_NATIVE_LDEXP: bool = false;
    const HAS_NATIVE_FREXP: bool = false;
}

macro_rules! s {
    ($ty:ty: $v:expr, [$a:literal, $b:literal, $c:literal, $d:literal]) => {
        unsafe { arch::_mm256_permute4x64_pd::<{ MM_SHUFFLE!($a, $b, $c, $d) }>($v) }
    };
    ($ty:ty: $v1:expr, $v2:expr, [$a:literal, $b:literal, $c:literal, $d:literal]) => {
        Self::swizzle($v1, $v2, const { GenericArray::from_array([$a, $b, $c, $d]) })
    };
}

impl LinAlg4Register for F64x4V3 {
    #[inline(always)]
    fn mat4_inverse(m: &mut [Storage<Self>; 4]) -> bool {
        // dedicated x86-v3 implementation that takes
        // advantage of `_mm256_permute4x64_pd`, though
        // generic swizzling is still used for some operations.
        impl_mat4_inverse!(m, s)
    }
}

impl LinAlg3Register for F64x4V3 {
    #[inline(always)]
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

    #[inline(always)]
    fn zero4(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_blend_pd(value, arch::_mm256_set1_pd(0.0), 0b1000) }
    }

    #[inline(always)]
    fn one4(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_blend_pd(value, arch::_mm256_set1_pd(1.0), 0b1000) }
    }

    #[inline(always)]
    fn min_element3(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_pd3_v3!(value; _mm_min_sd)
    }

    #[inline(always)]
    fn max_element3(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_pd3_v3!(value; _mm_max_sd)
    }

    #[inline(always)]
    fn sum_elements3(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_pd3_v3!(value; _mm_add_sd)
    }

    #[inline(always)]
    fn prod_elements3(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_pd3_v3!(value; _mm_mul_sd)
    }
}

impl CastRegister<DoublePumpRegister<F64x4V3>> for super::F32x8V3 {
    #[inline(always)]
    fn cast_from(value: Storage<DoublePumpRegister<F64x4V3>>) -> Storage<Self> {
        let (lo, hi) = <DoublePumpRegister<F64x4V3> as Register>::split(value);

        unsafe {
            let lo = arch::_mm256_cvtpd_ps(lo);
            let hi = arch::_mm256_cvtpd_ps(hi);

            arch::_mm256_setr_m128(lo, hi)
        }
    }
}
