use generic_array::{GenericArray, sequence::GenericSequence, typenum};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, CastRegister, FloatRegister, LinAlg3Register, MaskRegister, NumericRegister,
        PartialOrdRegister, PermuteRegister, Register, ShuffleRegister, SignedRegister, SwizzleRegister,
        dp::DoublePumpRegister, empty_reg, reg,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
pub struct F64x4V3;

impl Register for F64x4V3 {
    type Lanes = typenum::U4;

    type Element = f64;
    type Storage = arch::__m256d;
    type HalfRegister = super::F64x2V3;
    type DoubleRegister = DoublePumpRegister<Self>;

    const ISA: InstructionSet = InstructionSet::X86V3;

    type SCOUNT = super::I64x4V3;
    type UCOUNT = super::U64x4V3;

    const EMPTY: Self::Storage = empty_reg::<Self>();

    #[inline(always)]
    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Self::Storage {
        unsafe { arch::_mm256_loadu_pd(value.as_ptr() as *const _) }
    }

    #[inline(always)]
    fn splat(value: Self::Element) -> Self::Storage {
        unsafe { arch::_mm256_set1_pd(value) }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const Self::Element) -> Self::Storage {
        unsafe { arch::_mm256_load_pd(ptr) }
    }

    #[inline(always)]
    unsafe fn load_unaligned(ptr: *const Self::Element) -> Self::Storage {
        unsafe { arch::_mm256_loadu_pd(ptr) }
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut Self::Element, value: Self::Storage) {
        unsafe { arch::_mm256_store_pd(ptr, value) }
    }

    #[inline(always)]
    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Self::Storage) {
        unsafe { arch::_mm256_storeu_pd(ptr, value) }
    }

    #[inline(always)]
    fn join(
        lo: <Self::HalfRegister as Register>::Storage,
        hi: <Self::HalfRegister as Register>::Storage,
    ) -> Self::Storage
    where
        Self::HalfRegister: Register,
    {
        unsafe { arch::_mm256_setr_m128d(lo, hi) }
    }

    #[inline(always)]
    fn split(
        value: Self::Storage,
    ) -> (
        <Self::HalfRegister as Register>::Storage,
        <Self::HalfRegister as Register>::Storage,
    )
    where
        Self::HalfRegister: Register,
    {
        let lo = unsafe { arch::_mm256_castpd256_pd128(value) };
        let hi = unsafe { arch::_mm256_extractf128_pd(value, 1) };
        (lo, hi)
    }

    #[inline(always)]
    fn bitxor(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_xor_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn bitand(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_and_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn bitandnot(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_andnot_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn bitor(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_or_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn not(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_xor_pd(value, arch::_mm256_set1_pd(f64::from_bits(!0))) }
    }

    #[inline(always)]
    fn blendv(mask: Self::Storage, lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_blendv_pd(lhs, rhs, mask) }
    }

    const HAS_MSB_BLENDV: bool = true;

    #[inline(always)]
    fn reverse(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_permute4x64_pd::<{ MM_SHUFFLE!(0, 1, 2, 3) }>(value) }
    }
}

impl ShuffleRegister for F64x4V3 {
    #[inline(always)]
    fn shuffle<const IMM8: i32>(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_shuffle_pd(lhs, rhs, IMM8) }
    }
}

impl PermuteRegister for F64x4V3 {
    #[inline(always)]
    fn permute<const IMM8: i32>(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_permute_pd(value, IMM8) }
    }
}

impl SwizzleRegister for F64x4V3 {
    #[inline(always)]
    fn permutev(value: Self::Storage, idxs: GenericArray<u32, Self::Lanes>) -> Self::Storage {
        unsafe {
            let idxs: arch::__m128i = core::mem::transmute(idxs);
            arch::_mm256_permutevar_pd(value, arch::_mm256_cvtepu32_epi64(idxs))
        }
    }
}

impl MaskRegister for F64x4V3 {
    const FALSY: Self::Storage = reg::<Self, 4>([f64::from_bits(0); 4]);
    const TRUTHY: Self::Storage = reg::<Self, 4>([f64::from_bits(!0); 4]);

    #[inline(always)]
    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Self::Storage {
        unsafe { arch::_mm256_castsi256_pd(arch::_mm256_cvtboolx4_to_epi64_mask_v3(value)) }
    }

    #[inline(always)]
    fn all(value: Self::Storage) -> bool {
        unsafe { arch::_mm256_movemask_pd(value) == 0b1111 }
    }

    #[inline(always)]
    fn any(value: Self::Storage) -> bool {
        unsafe { arch::_mm256_movemask_pd(value) != 0 }
    }

    #[inline(always)]
    fn none(value: Self::Storage) -> bool {
        unsafe { arch::_mm256_movemask_pd(value) == 0 }
    }
}

impl PartialOrdRegister for F64x4V3 {
    #[inline(always)]
    fn lt(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_cmp_pd(lhs, rhs, arch::_CMP_LT_OQ) }
    }

    #[inline(always)]
    fn le(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_cmp_pd(lhs, rhs, arch::_CMP_LE_OQ) }
    }

    #[inline(always)]
    fn gt(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_cmp_pd(lhs, rhs, arch::_CMP_GT_OQ) }
    }

    #[inline(always)]
    fn ge(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_cmp_pd(lhs, rhs, arch::_CMP_GE_OQ) }
    }

    #[inline(always)]
    fn eq(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_cmp_pd(lhs, rhs, arch::_CMP_EQ_OQ) }
    }

    #[inline(always)]
    fn ne(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_cmp_pd(lhs, rhs, arch::_CMP_NEQ_OQ) }
    }
}

impl NumericRegister for F64x4V3 {
    const ZERO: Self::Storage = reg::<Self, 4>([0.0; 4]);
    const ONE: Self::Storage = reg::<Self, 4>([1.0; 4]);
    const TWO: Self::Storage = reg::<Self, 4>([2.0; 4]);

    const MIN: Self::Storage = reg::<Self, 4>([f64::MIN; 4]);
    const MAX: Self::Storage = reg::<Self, 4>([f64::MAX; 4]);

    #[inline(always)]
    fn min_element(value: Self::Storage) -> Self::Element {
        _mm256_reduce_pd_v3!(value; _mm_min_pd _mm_min_sd)
    }

    #[inline(always)]
    fn max_element(value: Self::Storage) -> Self::Element {
        _mm256_reduce_pd_v3!(value; _mm_max_pd _mm_max_sd)
    }

    #[inline(always)]
    fn sum_elements(value: Self::Storage) -> Self::Element {
        _mm256_reduce_pd_v3!(value; _mm_add_pd _mm_add_sd)
    }

    #[inline(always)]
    fn prod_elements(value: Self::Storage) -> Self::Element {
        _mm256_reduce_pd_v3!(value; _mm_mul_pd _mm_mul_sd)
    }

    #[inline(always)]
    fn offset() -> Self::Storage {
        Self::splat(<Self::Lanes as typenum::Unsigned>::USIZE as f64)
    }

    #[inline(always)]
    fn indexed() -> Self::Storage {
        Self::new(GenericArray::generate(|i| i as f64))
    }

    #[inline(always)]
    fn add(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_add_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn sub(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_sub_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn mul(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_mul_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn div(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_div_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn rem(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        // https://stackoverflow.com/a/26342944/2083075
        Self::nmul_adde(Self::trunc(Self::div(lhs, rhs)), rhs, lhs)
    }

    #[inline(always)]
    fn min(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_min_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn max(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_max_pd(lhs, rhs) }
    }
}

impl SignedRegister for F64x4V3 {
    const NEG_ONE: Self::Storage = reg::<Self, 4>([-1.0; 4]);
    const MIN_POSITIVE: Self::Storage = reg::<Self, 4>([f64::MIN_POSITIVE; 4]);

    #[inline(always)]
    fn neg(value: Self::Storage) -> Self::Storage {
        Self::bitxor(value, Self::NEG_ZERO)
    }

    #[inline(always)]
    fn abs(value: Self::Storage) -> Self::Storage {
        Self::bitandnot(Self::NEG_ZERO, value)
    }

    #[inline(always)]
    fn copysign(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        // take everything but the sign from lhs, and copy the sign from rhs
        Self::bitor(Self::bitandnot(Self::NEG_ZERO, lhs), Self::bitand(Self::NEG_ZERO, rhs))
    }

    #[inline(always)]
    fn signum(value: Self::Storage) -> Self::Storage {
        Self::bitor(Self::ONE, Self::bitand(value, Self::NEG_ZERO))
    }

    #[inline(always)]
    fn conditional_negate(value: Self::Storage, mask: Self::Storage) -> Self::Storage {
        Self::bitxor(value, Self::bitand(Self::NEG_ZERO, mask))
    }
}

impl FloatRegister for F64x4V3 {
    const HAS_TRUE_FMA: bool = true;

    type Bits = super::U64x4V3;
    type Signed = super::I64x4V3;
    type ExtendedPrecision = Self; // f64 is the highest precision available

    const HALF: Self::Storage = reg::<Self, 4>([0.5; 4]);
    const NEG_ZERO: Self::Storage = reg::<Self, 4>([-0.0; 4]);
    const EPSILON: Self::Storage = reg::<Self, 4>([f64::EPSILON; 4]);
    const INFINITY: Self::Storage = reg::<Self, 4>([f64::INFINITY; 4]);
    const NEG_INFINITY: Self::Storage = reg::<Self, 4>([f64::NEG_INFINITY; 4]);
    const NAN: Self::Storage = reg::<Self, 4>([f64::NAN; 4]);

    const EXP_MASK: crate::register::Storage<Self::Bits> = reg::<Self::Bits, 4>([0x7FF0_0000_0000_0000; 4]);

    #[inline(always)]
    fn mul_add(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_fmadd_pd(lhs, rhs, acc) }
    }

    #[inline(always)]
    fn mul_sub(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_fmsub_pd(lhs, rhs, acc) }
    }

    #[inline(always)]
    fn nmul_add(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_fnmadd_pd(lhs, rhs, acc) }
    }

    #[inline(always)]
    fn nmul_sub(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_fnmsub_pd(lhs, rhs, acc) }
    }

    #[inline(always)]
    fn mul_adde(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        Self::mul_add(lhs, rhs, acc)
    }

    #[inline(always)]
    fn mul_sube(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        Self::mul_sub(lhs, rhs, acc)
    }

    #[inline(always)]
    fn nmul_adde(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        Self::nmul_add(lhs, rhs, acc)
    }

    #[inline(always)]
    fn nmul_sube(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        Self::nmul_sub(lhs, rhs, acc)
    }

    #[inline(always)]
    fn sqrt(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_sqrt_pd(value) }
    }

    const HAS_APPROX_RCP: bool = false;
    const HAS_APPROX_RSQRT: bool = false;

    #[inline(always)]
    fn floor(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_floor_pd(value) }
    }

    #[inline(always)]
    fn ceil(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_ceil_pd(value) }
    }

    #[inline(always)]
    fn round(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_round_pd(value, arch::_MM_FROUND_TO_NEAREST_INT | arch::_MM_FROUND_NO_EXC) }
    }

    #[inline(always)]
    fn trunc(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_round_pd(value, arch::_MM_FROUND_TO_ZERO | arch::_MM_FROUND_NO_EXC) }
    }

    #[inline(always)]
    fn next_up(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_nextuppd_v3(value) }
    }

    #[inline(always)]
    fn next_down(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_nextdownpd_v3(value) }
    }
}

impl LinAlg3Register for F64x4V3 {
    #[inline(always)]
    fn dot3(lhs: Self::Storage, rhs: Self::Storage) -> f64 {
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
    fn cross3(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        // Borrowed from glam
        unsafe {
            // x  <-  a.y*b.z - a.z*b.y
            // y  <-  a.z*b.x - a.x*b.z
            // z  <-  a.x*b.y - a.y*b.x
            // We can save a shuffle by grouping it in this wacky order:
            // (self.zxy() * rhs - self * rhs.zxy()).zxy()
            let lhszxy = arch::_mm256_permute4x64_pd(lhs, 0b11_01_00_10);
            let rhszxy = arch::_mm256_permute4x64_pd(rhs, 0b11_01_00_10);
            let lhszxy_rhs = arch::_mm256_mul_pd(lhszxy, rhs);
            let rhszxy_lhs = arch::_mm256_mul_pd(rhszxy, lhs);
            let sub = arch::_mm256_sub_pd(lhszxy_rhs, rhszxy_lhs);
            arch::_mm256_permute4x64_pd(sub, 0b11_01_00_10)
        }
    }

    #[inline(always)]
    fn zero4(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_blend_pd(value, arch::_mm256_set1_pd(0.0), 0b1000) }
    }

    #[inline(always)]
    fn one4(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm256_blend_pd(value, arch::_mm256_set1_pd(1.0), 0b1000) }
    }

    #[inline(always)]
    fn min_element3(value: Self::Storage) -> Self::Element {
        _mm256_reduce_pd3_v3!(value; _mm_min_sd)
    }

    #[inline(always)]
    fn max_element3(value: Self::Storage) -> Self::Element {
        _mm256_reduce_pd3_v3!(value; _mm_max_sd)
    }

    #[inline(always)]
    fn sum_elements3(value: Self::Storage) -> Self::Element {
        _mm256_reduce_pd3_v3!(value; _mm_add_sd)
    }

    #[inline(always)]
    fn prod_elements3(value: Self::Storage) -> Self::Element {
        _mm256_reduce_pd3_v3!(value; _mm_mul_sd)
    }
}

impl CastRegister<DoublePumpRegister<F64x4V3>> for super::F32x8V3 {
    #[inline(always)]
    fn cast_from(value: <DoublePumpRegister<F64x4V3> as Register>::Storage) -> Self::Storage {
        let (lo, hi) = <DoublePumpRegister<F64x4V3> as Register>::split(value);

        unsafe {
            let lo = arch::_mm256_cvtpd_ps(lo);
            let hi = arch::_mm256_cvtpd_ps(hi);

            arch::_mm256_setr_m128(lo, hi)
        }
    }
}
