use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    backend::scalar::Scalar,
    isa::InstructionSet,
    register::{
        BitshiftRegister, BitwiseRegister, CastRegister, CoreRegister, Element, FloatRegister, MaskRegister,
        NumericRegister, PartialOrdRegister, PermuteRegister, Register, ShuffleRegister, SignedRegister, Storage,
        SwizzleRegister, dp::DoublePumpRegister, empty_reg, reg,
    },
    simd::Simd,
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct F64x2V2;

impl CoreRegister for F64x2V2 {
    type Lanes = typenum::U2;
    type Storage = arch::__m128d;
    type Mask = Self;

    const IS_EMULATED: bool = false;

    const ISA: InstructionSet = InstructionSet::X86V2;

    const EMPTY: Storage<Self> = empty_reg::<Self>();

    #[inline(always)]
    fn blendv(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_blendv_pd(lhs, rhs, mask) }
    }

    #[inline(always)]
    fn z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_and_pd(value, mask) }
    }

    #[inline(always)]
    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_andnot_pd(mask, value) }
    }
}

impl MaskRegister for F64x2V2 {
    #[inline(always)]
    fn set(mut mask: Storage<Self>, lane: usize, value: bool) -> Storage<Self> {
        Self::as_array_mut(&mut mask)[lane] = if value { Element::TRUTHY } else { Element::FALSY };
        mask
    }

    #[inline(always)]
    fn test(mask: Storage<Self>, lane: usize) -> bool {
        Self::as_array(&mask)[lane].to_bool()
    }

    const FALSY: Storage<Self> = reg::<Self, 2>([f64::from_bits(0); 2]);
    const TRUTHY: Storage<Self> = reg::<Self, 2>([f64::from_bits(!0); 2]);

    #[inline(always)]
    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_castsi128_pd(arch::_mm_cvtboolx2_to_epi64_mask_v2(value)) }
    }

    #[inline(always)]
    fn all(value: Storage<Self>) -> bool {
        unsafe { arch::_mm_movemask_pd(value) == 0b11 }
    }

    #[inline(always)]
    fn any(value: Storage<Self>) -> bool {
        unsafe { arch::_mm_movemask_pd(value) != 0 }
    }

    #[inline(always)]
    fn none(value: Storage<Self>) -> bool {
        unsafe { arch::_mm_movemask_pd(value) == 0 }
    }

    #[inline(always)]
    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        Some(unsafe { arch::_mm_movemask_pd(value) as u64 })
    }

    #[inline(always)]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = unsafe { arch::_mm_movemask_pd(value) as u32 };
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

#[thermite_macros::bitand_z]
impl BitwiseRegister for F64x2V2 {
    #[inline(always)]
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_xor_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_and_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_andnot_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_or_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn not(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_xor_pd(value, arch::_mm_set1_pd(f64::from_bits(!0))) }
    }
}

impl Register for F64x2V2 {
    type HalfRegister = f64; // Scalar register
    type DoubleRegister = DoublePumpRegister<Self>;

    type Element = f64;

    type ISize = super::I64x2V2;
    type USize = super::U64x2V2;

    const HAS_EQUAL_SIZE_MASK: bool = true;

    #[inline(always)]
    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe {
            // value != 0.0
            arch::_mm_castsi128_pd(arch::_mm_xor_si128(
                arch::_mm_set1_epi8(-1),
                arch::_mm_cmpeq_epi64(arch::_mm_castpd_si128(value), arch::_mm_setzero_si128()),
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
    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_loadu_pd(value.as_ptr() as *const _) }
    }

    #[inline(always)]
    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_set_sd(value) }
    }

    #[inline(always)]
    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_set1_pd(value) }
    }

    #[inline(always)]
    fn split(value: Storage<Self>) -> (Storage<Self::HalfRegister>, Storage<Self::HalfRegister>)
    where
        Self::HalfRegister: Register,
    {
        unsafe {
            let mut arr = [0f64; 2];
            Self::store_unaligned(arr.as_mut_ptr(), value);
            (arr[0], arr[1])
        }
    }

    #[inline(always)]
    fn join(lo: Storage<Self::HalfRegister>, hi: Storage<Self::HalfRegister>) -> Storage<Self>
    where
        Self::HalfRegister: Register,
    {
        unsafe { arch::_mm_setr_pd(lo, hi) }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_load_pd(ptr) }
    }

    #[inline(always)]
    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_loadu_pd(ptr) }
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_store_pd(ptr, value) }
    }

    #[inline(always)]
    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_storeu_pd(ptr, value) }
    }

    #[inline(always)]
    unsafe fn load_stream(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_castsi128_pd(arch::_mm_stream_load_si128(ptr as _)) }
    }

    #[inline(always)]
    unsafe fn store_stream(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_stream_pd(ptr, value) }
    }

    #[inline(always)]
    fn reverse(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_shuffle_pd(value, value, 0b01) }
    }

    const HAS_SIMPLE_UNPACK: bool = true;

    #[inline(always)]
    fn unpack(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe { (arch::_mm_unpacklo_pd(a, b), arch::_mm_unpackhi_pd(a, b)) }
    }

    #[inline(always)]
    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bswap_pdx_v2(value) }
    }

    #[inline(always)]
    fn reduce<F>(value: Storage<Self>, f: F) -> Self::Element
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element,
    {
        let arr = Self::as_array(&value);

        f(arr[0], arr[1])
    }
}

impl ShuffleRegister for F64x2V2 {
    #[inline(always)]
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_shuffle_pd(lhs, rhs, IMM8) }
    }
}

impl PermuteRegister for F64x2V2 {
    #[inline(always)]
    fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_shuffle_pd(value, value, IMM8) }
    }
}

impl SwizzleRegister for F64x2V2 {
    const HAS_PERMUTEV: bool = false;
}

impl PartialOrdRegister for F64x2V2 {
    #[inline(always)]
    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmplt_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmple_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpgt_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpge_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpeq_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpneq_pd(lhs, rhs) }
    }
}

#[thermite_macros::bitand_z]
impl NumericRegister for F64x2V2 {
    const ZERO: Storage<Self> = reg::<Self, 2>([0.0; 2]);
    const ONE: Storage<Self> = reg::<Self, 2>([1.0; 2]);
    const TWO: Storage<Self> = reg::<Self, 2>([2.0; 2]);

    const MIN: Storage<Self> = reg::<Self, 2>([f64::MIN; 2]);
    const MAX: Storage<Self> = reg::<Self, 2>([f64::MAX; 2]);

    #[skip_masked]
    #[inline(always)]
    fn min_element(value: Storage<Self>) -> Self::Element {
        _mm_reduce_pd_v1!(value; _mm_min_pd _mm_min_sd)
    }

    #[skip_masked]
    #[inline(always)]
    fn max_element(value: Storage<Self>) -> Self::Element {
        _mm_reduce_pd_v1!(value; _mm_max_pd _mm_max_sd)
    }

    #[skip_masked]
    #[inline(always)]
    fn sum_elements(value: Storage<Self>) -> Self::Element {
        _mm_reduce_pd_v1!(value; _mm_add_pd _mm_add_sd)
    }

    #[skip_masked]
    #[inline(always)]
    fn prod_elements(value: Storage<Self>) -> Self::Element {
        _mm_reduce_pd_v1!(value; _mm_mul_pd _mm_mul_sd)
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
        unsafe { arch::_mm_add_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sub_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mul_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_div_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        // https://stackoverflow.com/a/26342944/2083075
        Self::nmul_adde(Self::trunc(Self::div(lhs, rhs)), rhs, lhs)
    }

    #[inline(always)]
    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_min_pd(lhs, rhs) }
    }

    #[inline(always)]
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_max_pd(lhs, rhs) }
    }
}

impl SignedRegister for F64x2V2 {
    const NEG_ONE: Storage<Self> = reg::<Self, 2>([-1.0; 2]);
    const MIN_POSITIVE: Storage<Self> = reg::<Self, 2>([f64::MIN_POSITIVE; 2]);

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

impl FloatRegister for F64x2V2 {
    const HAS_TRUE_FMA: bool = false;

    type Bits = super::U64x2V2;
    type Signed = super::I64x2V2;
    type ExtendedPrecision = Self; // f64 is the highest precision available

    const HALF: Storage<Self> = reg::<Self, 2>([0.5; 2]);
    const NEG_ZERO: Storage<Self> = reg::<Self, 2>([-0.0; 2]);
    const EPSILON: Storage<Self> = reg::<Self, 2>([f64::EPSILON; 2]);
    const INFINITY: Storage<Self> = reg::<Self, 2>([f64::INFINITY; 2]);
    const NEG_INFINITY: Storage<Self> = reg::<Self, 2>([f64::NEG_INFINITY; 2]);
    const NAN: Storage<Self> = reg::<Self, 2>([f64::NAN; 2]);

    const EXP_MASK: crate::register::Storage<Self::Bits> = reg::<Self::Bits, 2>([0x7FF0_0000_0000_0000; 2]);

    #[cfg(not(feature = "disable_fma_emulation"))]
    #[inline(always)]
    fn mul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_fmadd_pdx_v1(lhs, rhs, acc) }
    }

    #[cfg(not(feature = "disable_fma_emulation"))]
    #[inline(always)]
    fn mul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_fmadd_pdx_v1(lhs, rhs, Self::neg(acc)) }
    }

    #[cfg(not(feature = "disable_fma_emulation"))]
    #[inline(always)]
    fn nmul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_fmadd_pdx_v1(Self::neg(lhs), rhs, acc) }
    }

    #[cfg(not(feature = "disable_fma_emulation"))]
    #[inline(always)]
    fn nmul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_fmadd_pdx_v1(Self::neg(lhs), rhs, Self::neg(acc)) }
    }

    #[inline(always)]
    fn sqrt(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sqrt_pd(value) }
    }

    const HAS_APPROX_RSQRT: bool = false;
    const HAS_APPROX_RCP: bool = false;

    #[inline(always)]
    fn floor(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_floor_pd(value) }
    }

    #[inline(always)]
    fn ceil(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_ceil_pd(value) }
    }

    #[inline(always)]
    fn round(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_round_pd(value, arch::_MM_FROUND_TO_NEAREST_INT | arch::_MM_FROUND_NO_EXC) }
    }

    #[inline(always)]
    fn trunc(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_round_pd(value, arch::_MM_FROUND_TO_ZERO | arch::_MM_FROUND_NO_EXC) }
    }

    const HAS_NATIVE_LDEXP: bool = false;
    const HAS_NATIVE_FREXP: bool = false;
}

impl CastRegister<DoublePumpRegister<F64x2V2>> for super::F32x4V2 {
    #[inline(always)]
    fn cast_from(value: Storage<DoublePumpRegister<F64x2V2>>) -> Storage<Self> {
        let (lo, hi) = <DoublePumpRegister<F64x2V2> as Register>::split(value);

        unsafe {
            let lo = arch::_mm_cvtpd_ps(lo);
            let hi = arch::_mm_cvtpd_ps(hi);

            arch::_mm_movelh_ps(lo, hi)
        }
    }
}

impl CastRegister<<Scalar as Simd>::f32x2> for F64x2V2 {
    #[inline(always)]
    fn cast_from(value: Storage<<Scalar as Simd>::f32x2>) -> Storage<Self> {
        unsafe { arch::_mm_cvtps_pd(arch::_mm_setr_ps(value.0, value.1, 0.0, 0.0)) }
    }
}

impl CastRegister<F64x2V2> for <Scalar as Simd>::f32x2 {
    #[inline(always)]
    fn cast_from(value: Storage<F64x2V2>) -> Storage<<Scalar as Simd>::f32x2> {
        unsafe {
            let ps = arch::_mm_cvtpd_ps(value);

            DoublePumpRegister(
                arch::_mm_cvtss_f32(ps),
                f32::from_bits(arch::_mm_extract_ps::<1>(ps) as u32),
            )
        }
    }
}
