use generic_array::{GenericArray, sequence::GenericSequence, typenum};

use crate::register::{
    FloatRegister, MaskRegister, NumericRegister, PartialOrdRegister, PermuteRegister, Register, ShiftRegister,
    ShuffleRegister, SignedRegister, SwizzleRegister, empty_reg, reg,
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
pub struct F64x2V3;

impl Register for F64x2V3 {
    type Lanes = typenum::U2;

    type Element = f64;
    type Storage = arch::__m128d;
    type HalfRegister = ();
    type DoubleRegister = super::F64x4V3;

    type SCOUNT = super::I64x2V3;
    type UCOUNT = super::U64x2V3;

    const EMPTY: Self::Storage = empty_reg::<Self>();

    #[inline(always)]
    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Self::Storage {
        unsafe { arch::_mm_loadu_pd(value.as_ptr() as *const _) }
    }

    #[inline(always)]
    fn splat(value: Self::Element) -> Self::Storage {
        unsafe { arch::_mm_set1_pd(value) }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const Self::Element) -> Self::Storage {
        unsafe { arch::_mm_load_pd(ptr) }
    }

    #[inline(always)]
    unsafe fn load_unaligned(ptr: *const Self::Element) -> Self::Storage {
        unsafe { arch::_mm_loadu_pd(ptr) }
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut Self::Element, value: Self::Storage) {
        unsafe { arch::_mm_store_pd(ptr, value) }
    }

    #[inline(always)]
    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Self::Storage) {
        unsafe { arch::_mm_storeu_pd(ptr, value) }
    }

    #[inline(always)]
    fn bitxor(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_xor_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn bitand(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_and_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn bitandnot(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_andnot_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn bitor(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_or_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn not(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_xor_pd(value, arch::_mm_set1_pd(f64::from_bits(!0))) }
    }

    #[inline(always)]
    fn blendv(mask: Self::Storage, lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_blendv_pd(lhs, rhs, mask) }
    }

    const HAS_MSB_BLENDV: bool = true;

    #[inline(always)]
    fn reverse(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_permute_pd(value, 0b01) }
    }

    #[inline(always)]
    fn reduce<F>(value: Self::Storage, f: F) -> Self::Element
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element,
    {
        let arr = Self::as_array(&value);

        f(arr[0], arr[1])
    }
}

impl ShuffleRegister for F64x2V3 {
    #[inline(always)]
    fn shuffle<const IMM8: i32>(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_shuffle_pd(lhs, rhs, IMM8) }
    }
}

impl PermuteRegister for F64x2V3 {
    #[inline(always)]
    fn permute<const IMM8: i32>(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_permute_pd(value, IMM8) }
    }
}

impl MaskRegister for F64x2V3 {
    const FALSY: Self::Storage = reg::<Self, 2>([f64::from_bits(0); 2]);
    const TRUTHY: Self::Storage = reg::<Self, 2>([f64::from_bits(!0); 2]);

    #[inline(always)]
    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Self::Storage {
        unsafe { arch::_mm_castsi128_pd(arch::_mm_cvtboolx2_to_epi64_mask_v2(value)) }
    }

    #[inline(always)]
    fn all(value: Self::Storage) -> bool {
        unsafe { arch::_mm_movemask_pd(value) == 0b11 }
    }

    #[inline(always)]
    fn any(value: Self::Storage) -> bool {
        unsafe { arch::_mm_movemask_pd(value) != 0 }
    }

    #[inline(always)]
    fn none(value: Self::Storage) -> bool {
        unsafe { arch::_mm_movemask_pd(value) == 0 }
    }
}

impl PartialOrdRegister for F64x2V3 {
    #[inline(always)]
    fn lt(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_cmp_pd(lhs, rhs, arch::_CMP_LT_OQ) }
    }

    #[inline(always)]
    fn le(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_cmp_pd(lhs, rhs, arch::_CMP_LE_OQ) }
    }

    #[inline(always)]
    fn gt(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_cmp_pd(lhs, rhs, arch::_CMP_GT_OQ) }
    }

    #[inline(always)]
    fn ge(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_cmp_pd(lhs, rhs, arch::_CMP_GE_OQ) }
    }

    #[inline(always)]
    fn eq(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_cmp_pd(lhs, rhs, arch::_CMP_EQ_OQ) }
    }

    #[inline(always)]
    fn ne(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_cmp_pd(lhs, rhs, arch::_CMP_NEQ_OQ) }
    }
}

impl NumericRegister for F64x2V3 {
    const ZERO: Self::Storage = reg::<Self, 2>([0.0; 2]);
    const ONE: Self::Storage = reg::<Self, 2>([1.0; 2]);
    const TWO: Self::Storage = reg::<Self, 2>([2.0; 2]);

    const MIN: Self::Storage = reg::<Self, 2>([f64::MIN; 2]);
    const MAX: Self::Storage = reg::<Self, 2>([f64::MAX; 2]);

    #[inline(always)]
    fn min_element(value: Self::Storage) -> Self::Element {
        _mm_reduce_pd!(value; _mm_min_pd _mm_min_sd)
    }

    #[inline(always)]
    fn max_element(value: Self::Storage) -> Self::Element {
        _mm_reduce_pd!(value; _mm_max_pd _mm_max_sd)
    }

    #[inline(always)]
    fn sum_elements(value: Self::Storage) -> Self::Element {
        _mm_reduce_pd!(value; _mm_add_pd _mm_add_sd)
    }

    #[inline(always)]
    fn prod_elements(value: Self::Storage) -> Self::Element {
        _mm_reduce_pd!(value; _mm_mul_pd _mm_mul_sd)
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
        unsafe { arch::_mm_add_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn sub(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_sub_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn mul(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_mul_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn div(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_div_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn rem(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        // https://stackoverflow.com/a/26342944/2083075
        Self::nmul_adde(Self::trunc(Self::div(lhs, rhs)), rhs, lhs)
    }

    #[inline(always)]
    fn min(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_min_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn max(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_max_pd(lhs, rhs) }
    }
}

impl SignedRegister for F64x2V3 {
    const NEG_ONE: Self::Storage = reg::<Self, 2>([-1.0; 2]);

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

impl FloatRegister for F64x2V3 {
    const HAS_TRUE_FMA: bool = true;

    type Bits = super::U64x2V3;
    type Signed = super::I64x2V3;
    type ExtendedPrecision = Self; // f64 is the highest precision available

    const HALF: Self::Storage = reg::<Self, 2>([0.5; 2]);
    const NEG_ZERO: Self::Storage = reg::<Self, 2>([-0.0; 2]);
    const EPSILON: Self::Storage = reg::<Self, 2>([f64::EPSILON; 2]);
    const INFINITY: Self::Storage = reg::<Self, 2>([f64::INFINITY; 2]);
    const NEG_INFINITY: Self::Storage = reg::<Self, 2>([f64::NEG_INFINITY; 2]);
    const NAN: Self::Storage = reg::<Self, 2>([f64::NAN; 2]);

    const EXP_MASK: crate::register::Storage<Self::Bits> = reg::<Self::Bits, 2>([0x7FF0_0000_0000_0000; 2]);

    #[inline(always)]
    fn mul_add(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_fmadd_pd(lhs, rhs, acc) }
    }

    #[inline(always)]
    fn mul_sub(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_fmsub_pd(lhs, rhs, acc) }
    }

    #[inline(always)]
    fn nmul_add(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_fnmadd_pd(lhs, rhs, acc) }
    }

    #[inline(always)]
    fn nmul_sub(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_fnmsub_pd(lhs, rhs, acc) }
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
        unsafe { arch::_mm_sqrt_pd(value) }
    }

    const HAS_APPROX_RSQRT: bool = false;
    const HAS_APPROX_RCP: bool = false;

    #[inline(always)]
    fn floor(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_floor_pd(value) }
    }

    #[inline(always)]
    fn ceil(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_ceil_pd(value) }
    }

    #[inline(always)]
    fn round(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_round_pd(value, arch::_MM_FROUND_TO_NEAREST_INT | arch::_MM_FROUND_NO_EXC) }
    }

    #[inline(always)]
    fn trunc(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_round_pd(value, arch::_MM_FROUND_TO_ZERO | arch::_MM_FROUND_NO_EXC) }
    }

    #[inline(always)]
    fn next_up(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_nextuppd_v2(value) }
    }

    #[inline(always)]
    fn next_down(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_nextdownpd_v2(value) }
    }
}
