use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    backend::scalar::Scalar,
    isa::InstructionSet,
    register::{
        BitshiftRegister, BitwiseRegister, CastRegister, ConcatRegister, CoreRegister, Element, ExtendRegister,
        FloatRegister, IndexableRegister, MaskElement, MaskRegister, NativeCapability, NumericRegister,
        PartialOrdRegister, PermuteRegister, Register, ShuffleRegister, SignedRegister, Storage, SwizzleRegister,
        WideRegister, ZeroUpper, dp::DoublePumpRegister, empty_reg, reg,
    },
    simd::Simd,
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct F64x2V3;

impl CoreRegister for F64x2V3 {
    type Lanes = typenum::U2;
    type Storage = arch::__m128d;
    type Mask = Self;

    const IS_EMULATED: bool = false;

    const ISA: InstructionSet = InstructionSet::X86V3;

    const EMPTY: Storage<Self> = empty_reg::<Self>();

    #[inline(always)]
    fn blendv(mask: Storage<Self::Mask>, on_false: Storage<Self>, on_true: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_blendv_pd(on_false, on_true, mask) }
    }

    #[inline(always)]
    fn z(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_and_pd(value, mask) }
    }

    #[inline(always)]
    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_andnot_pd(mask, value) }
    }

    #[inline(always)]
    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        if const { Z::N >= 2 } {
            value
        } else if const { Z::N == 1 } {
            unsafe { arch::_mm_castsi128_pd(arch::_mm_move_epi64(arch::_mm_castpd_si128(value))) }
        } else {
            Self::EMPTY // N == 0, so zero everything
        }
    }
}

impl MaskRegister for F64x2V3 {
    const FALSY: Storage<Self> = reg::<Self, 2>([f64::from_bits(0); 2]);
    const TRUTHY: Storage<Self> = reg::<Self, 2>([f64::from_bits(!0); 2]);

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

#[rustfmt::skip] #[thermite_macros::bitand_z]
impl BitwiseRegister for F64x2V3 {
    #[masked] fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_xor_pd(lhs, rhs) }
    }

    #[masked] fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_and_pd(lhs, rhs) }
    }

    #[masked] fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_andnot_pd(lhs, rhs) }
    }

    #[masked] fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_or_pd(lhs, rhs) }
    }

    #[masked] fn not(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_xor_pd(value, arch::_mm_set1_pd(f64::from_bits(!0))) }
    }
}

impl ConcatRegister<f64> for F64x2V3 {
    #[inline(always)]
    fn concat(lo: Storage<f64>, hi: Storage<f64>) -> Storage<Self> {
        unsafe { arch::_mm_setr_pd(lo, hi) }
    }

    #[inline(always)]
    fn split(value: Storage<Self>) -> (Storage<f64>, Storage<f64>) {
        unsafe {
            let mut arr = [0f64; 2];
            Self::store_unaligned(arr.as_mut_ptr(), value);
            (arr[0], arr[1])
        }
    }
}

impl ExtendRegister<f64> for F64x2V3 {
    #[inline(always)]
    fn extend(value: Storage<f64>) -> Storage<Self> {
        unsafe { arch::_mm_setr_pd(value, 0.0) }
    }

    #[inline(always)]
    fn narrow(value: Storage<Self>) -> Storage<f64> {
        unsafe { arch::_mm_cvtsd_f64(value) }
    }
}

impl WideRegister for F64x2V3 {
    type Wide = super::F64x4V3;
}

impl Register for F64x2V3 {
    type Element = f64;

    type Signed = super::I64x2V3;
    type Unsigned = super::U64x2V3;

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
    fn single(value: Self::Element) -> crate::register::Storage<Self> {
        unsafe { arch::_mm_set_sd(value) }
    }

    #[inline(always)]
    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_set1_pd(value) }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_load_pd(ptr) }
    }

    #[inline(always)]
    unsafe fn load_m(src: Storage<Self>, mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        // use load_z + 2 bitwise ops to emulate load_m without blendv or scalar fallbacks
        unsafe { Self::bitor(Self::load_z(mask, ptr), Self::bitandnot(mask, src)) }
    }

    #[inline(always)]
    unsafe fn load_z(mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_maskload_pd(ptr, arch::_mm_castpd_si128(mask)) }
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
        unsafe { arch::_mm_permute_pd(value, 0b01) }
    }

    #[inline(always)]
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe { (arch::_mm_unpacklo_pd(a, b), arch::_mm_unpackhi_pd(a, b)) }
    }

    #[inline(always)]
    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
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

impl IndexableRegister<super::U64x2V3> for F64x2V3 {
    #[inline(always)]
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<super::U64x2V3>) -> Storage<Self> {
        unsafe { arch::_mm_i64gather_pd::<8>(ptr, indices) }
    }

    #[inline(always)]
    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<super::U64x2V3>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mask_i64gather_pd::<8>(src, ptr, indices, mask) }
    }
}

impl IndexableRegister<super::U32x2V3> for F64x2V3 {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<super::U32x2V3>) -> Storage<Self> {
        unsafe { arch::_mm_i32gather_pd::<8>(ptr, indices.0) }
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<super::U32x2V3>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mask_i32gather_pd::<8>(src, ptr, indices.0, mask) }
    }
}

impl ShuffleRegister for F64x2V3 {
    #[inline(always)]
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_shuffle_pd(lhs, rhs, IMM8) }
    }
}

impl PermuteRegister for F64x2V3 {
    #[inline(always)]
    fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_permute_pd(value, IMM8) }
    }
}

impl SwizzleRegister for F64x2V3 {
    const HAS_PERMUTEV: bool = false;
}

impl PartialOrdRegister for F64x2V3 {
    #[inline(always)]
    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmp_pd(lhs, rhs, arch::_CMP_LT_OQ) }
    }

    #[inline(always)]
    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmp_pd(lhs, rhs, arch::_CMP_LE_OQ) }
    }

    #[inline(always)]
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmp_pd(lhs, rhs, arch::_CMP_GT_OQ) }
    }

    #[inline(always)]
    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmp_pd(lhs, rhs, arch::_CMP_GE_OQ) }
    }

    #[inline(always)]
    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmp_pd(lhs, rhs, arch::_CMP_EQ_OQ) }
    }

    #[inline(always)]
    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmp_pd(lhs, rhs, arch::_CMP_NEQ_OQ) }
    }
}

#[thermite_macros::bitand_z]
impl NumericRegister for F64x2V3 {
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
        Self::splat(<Self::Lanes as typenum::Unsigned>::USIZE as f64)
    }

    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as f64))
    }

    #[masked]
    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_add_pd(lhs, rhs) }
    }

    #[masked]
    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sub_pd(lhs, rhs) }
    }

    fn add_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_add_pd(lhs, arch::_mm_and_pd(rhs, mask)) }
    }

    fn sub_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sub_pd(lhs, arch::_mm_and_pd(rhs, mask)) }
    }

    #[masked]
    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mul_pd(lhs, rhs) }
    }

    #[masked]
    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_div_pd(lhs, rhs) }
    }

    #[masked]
    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        // https://stackoverflow.com/a/26342944/2083075
        Self::nmul_adde(Self::trunc(Self::div(lhs, rhs)), rhs, lhs)
    }

    #[masked]
    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::fix_min::<Self>(lhs, rhs, unsafe { arch::_mm_min_pd(lhs, rhs) })
    }

    #[masked]
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::fix_max::<Self>(lhs, rhs, unsafe { arch::_mm_max_pd(lhs, rhs) })
    }
}

#[thermite_macros::bitand_z]
impl SignedRegister for F64x2V3 {
    const NEG_ONE: Storage<Self> = reg::<Self, 2>([-1.0; 2]);
    const MIN_POSITIVE: Storage<Self> = reg::<Self, 2>([f64::MIN_POSITIVE; 2]);

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
impl FloatRegister for F64x2V3 {
    const HAS_TRUE_FMA: bool = true;

    type Bits = super::U64x2V3;
    type SignedBits = super::I64x2V3;
    type ExtendedPrecision = Self; // f64 is the highest precision available

    const HALF: Storage<Self> = reg::<Self, 2>([0.5; 2]);
    const NEG_ZERO: Storage<Self> = reg::<Self, 2>([-0.0; 2]);
    const EPSILON: Storage<Self> = reg::<Self, 2>([f64::EPSILON; 2]);
    const INFINITY: Storage<Self> = reg::<Self, 2>([f64::INFINITY; 2]);
    const NEG_INFINITY: Storage<Self> = reg::<Self, 2>([f64::NEG_INFINITY; 2]);
    const NAN: Storage<Self> = reg::<Self, 2>([f64::NAN; 2]);

    const EXP_MASK: crate::register::Storage<Self::Bits> = reg::<Self::Bits, 2>([0x7FF0_0000_0000_0000; 2]);

    #[masked]
    fn mul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_fmadd_pd(lhs, rhs, acc) }
    }

    #[masked]
    fn mul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_fmsub_pd(lhs, rhs, acc) }
    }

    #[masked]
    fn nmul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_fnmadd_pd(lhs, rhs, acc) }
    }

    #[masked]
    fn nmul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_fnmsub_pd(lhs, rhs, acc) }
    }

    #[masked]
    fn sqrt(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sqrt_pd(value) }
    }

    const HAS_APPROX_RSQRT: bool = false;
    const HAS_APPROX_RCP: bool = false;

    #[masked]
    fn floor(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_floor_pd(value) }
    }

    #[masked]
    fn ceil(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_ceil_pd(value) }
    }

    #[masked]
    fn round(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_round_pd(value, arch::_MM_FROUND_TO_NEAREST_INT | arch::_MM_FROUND_NO_EXC) }
    }

    #[masked]
    fn trunc(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_round_pd(value, arch::_MM_FROUND_TO_ZERO | arch::_MM_FROUND_NO_EXC) }
    }

    const NATIVE_CAP: NativeCapability = NativeCapability::NONE;
}

impl CastRegister<<Scalar as Simd>::f32x2> for F64x2V3 {
    #[inline(always)]
    fn cast_from(value: Storage<<Scalar as Simd>::f32x2>) -> Storage<Self> {
        unsafe { arch::_mm_cvtps_pd(arch::_mm_setr_ps(value.0, value.1, 0.0, 0.0)) }
    }
}

impl CastRegister<F64x2V3> for <Scalar as Simd>::f32x2 {
    #[inline(always)]
    fn cast_from(value: Storage<F64x2V3>) -> Storage<<Scalar as Simd>::f32x2> {
        unsafe {
            let ps = arch::_mm_cvtpd_ps(value);

            DoublePumpRegister::concat(
                arch::_mm_cvtss_f32(ps),
                f32::from_bits(arch::_mm_extract_ps::<1>(ps) as u32),
            )
        }
    }
}
