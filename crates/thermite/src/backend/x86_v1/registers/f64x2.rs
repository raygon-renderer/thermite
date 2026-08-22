use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    backend::scalar::Scalar,
    isa::InstructionSet,
    register::{
        BitwiseRegister, CastRegister, ConcatRegister, CoreRegister, ExtendRegister, FloatRegister, InterleaveRegister,
        MaskElement, MaskRegister, NativeCapability, NumericRegister, PartialOrdRegister, PermuteRegister, Register,
        ShuffleRegister, SignedRegister, Storage, ZeroUpper, array::ArrayRegister, empty_reg, reg,
    },
    simd::Simd,
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct F64x2V1;

#[thermite_macros::inline_always]
impl ConcatRegister<f64> for F64x2V1 {
    fn concat(lo: Storage<f64>, hi: Storage<f64>) -> Storage<Self> {
        unsafe { arch::_mm_setr_pd(lo, hi) }
    }

    fn split(value: Storage<Self>) -> (Storage<f64>, Storage<f64>) {
        unsafe {
            let mut arr = [0f64; 2];
            Self::store_unaligned(arr.as_mut_ptr(), value);
            (arr[0], arr[1])
        }
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<f64> for F64x2V1 {
    fn extend(value: Storage<f64>) -> Storage<Self> {
        unsafe { arch::_mm_setr_pd(value, 0.0) }
    }

    fn narrow(value: Storage<Self>) -> Storage<f64> {
        unsafe { arch::_mm_cvtsd_f64(value) }
    }
}

#[thermite_macros::inline_always]
impl CoreRegister for F64x2V1 {
    type Lanes = typenum::U2;
    type Storage = arch::__m128d;
    type Mask = Self;

    const IS_EMULATED: bool = false;
    const EMPTY: Storage<Self> = empty_reg::<Self>();
    const HAS_EQUAL_SIZE_MASK: bool = true;

    fn blendv(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_blendv_pdx_v1(lhs, rhs, mask) }
    }

    fn zz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_and_pd(value, mask) }
    }

    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_andnot_pd(mask, value) }
    }

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        if const { Z::N >= 2 } {
            value
        } else if const { Z::N == 1 } {
            unsafe { arch::_mm_castsi128_pd(arch::_mm_move_epi64(arch::_mm_castpd_si128(value))) }
        } else {
            Self::EMPTY // N == 0, so zero everything
        }
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        mask
    }
}

#[thermite_macros::inline_always]
impl MaskRegister for F64x2V1 {
    fn set(mut mask: Storage<Self::Mask>, lane: usize, value: bool) -> Storage<Self> {
        Self::as_mut_slice(&mut mask)[lane] = if value { MaskElement::TRUTHY } else { MaskElement::FALSY };
        mask
    }

    fn test(mask: Storage<Self::Mask>, lane: usize) -> bool {
        Self::as_slice(&mask)[lane].to_bool()
    }

    const FALSY: Storage<Self> = reg::<Self, 2>([f64::from_bits(0); 2]);
    const TRUTHY: Storage<Self> = reg::<Self, 2>([f64::from_bits(!0); 2]);

    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_castsi128_pd(arch::_mm_cvtboolx2_to_epi64_mask_v1(value)) }
    }

    fn all(value: Storage<Self>) -> bool {
        unsafe { arch::_mm_movemask_pd(value) == 0b11 }
    }

    fn any(value: Storage<Self>) -> bool {
        unsafe { arch::_mm_movemask_pd(value) != 0 }
    }

    fn none(value: Storage<Self>) -> bool {
        unsafe { arch::_mm_movemask_pd(value) == 0 }
    }

    fn from_native_bitmask(bitmask: u64) -> Storage<Self> {
        unsafe { arch::_mm_castsi128_pd(arch::_mm_movm_epi64x_v1(bitmask)) }
    }

    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        Some(unsafe { arch::_mm_movemask_pd(value) as u64 })
    }

    #[cfg(feature = "bitvec")]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = unsafe { arch::_mm_movemask_pd(value) as u32 };
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for F64x2V1 {
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_xor_pd(lhs, rhs) }
    }

    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_and_pd(lhs, rhs) }
    }

    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_andnot_pd(lhs, rhs) }
    }

    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_or_pd(lhs, rhs) }
    }

    fn not(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_xor_pd(value, arch::_mm_set1_pd(f64::from_bits(!0))) }
    }
}

#[thermite_macros::inline_always]
impl InterleaveRegister for F64x2V1 {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe { (arch::_mm_unpacklo_pd(a, b), arch::_mm_unpackhi_pd(a, b)) }
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe { (arch::_mm_unpacklo_pd(a, b), arch::_mm_unpackhi_pd(a, b)) }
    }
}

#[thermite_macros::inline_always]
impl Register for F64x2V1 {
    type Element = f64;

    type Signed = super::I64x2V1;
    type Unsigned = super::U64x2V1;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe {
            // value != 0.0
            arch::_mm_castsi128_pd(arch::_mm_xor_si128(
                arch::_mm_set1_epi8(-1),
                arch::_mm_cmpeq_epi64x_v1(arch::_mm_castpd_si128(value), arch::_mm_setzero_si128()),
            ))
        }
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        value
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        // unlike v2/v3, whose `blendv_pd` reads only the MSB of each lane, the
        // v1 blendv is a full bitwise select - the sign bit must be smeared
        // across the whole lane to form a real mask
        unsafe { arch::_mm_castsi128_pd(arch::_mm_signbits_epi64x_v1(arch::_mm_castpd_si128(value))) }
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_loadu_pd(value.as_ptr() as *const _) }
    }

    impl_native_extract!(@pd128);

    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_set_sd(value) }
    }

    impl_native_radix3!(arch::_mm_interleave3_pd, arch::_mm_deinterleave3_pd);

    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_set1_pd(value) }
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_load_pd(ptr) }
    }

    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_loadu_pd(ptr) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_store_pd(ptr, value) }
    }

    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_storeu_pd(ptr, value) }
    }

    unsafe fn load_stream(ptr: *const Self::Element) -> Storage<Self> {
        // no non-temporal load before SSE4.1 (`movntdqa`); a regular aligned load is the best SSE2 can do
        unsafe { arch::_mm_load_pd(ptr) }
    }

    unsafe fn store_stream(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_stream_pd(ptr, value) }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_shuffle_pd(value, value, 0b01) }
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bswap_pdx_v1(value) }
    }

    fn reduce<F>(value: Storage<Self>, f: F) -> Self::Element
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element,
    {
        let arr = Self::as_slice(&value);

        f(arr[0], arr[1])
    }

    const HAS_PERMUTEV: bool = false;

    impl_float_align_via_bits!(super::U64x2V1);
}

#[thermite_macros::inline_always]
impl ShuffleRegister for F64x2V1 {
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_shuffle_pd(lhs, rhs, IMM8) }
    }
}

#[thermite_macros::inline_always]
impl PermuteRegister for F64x2V1 {
    fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_shuffle_pd(value, value, IMM8) }
    }
}

#[thermite_macros::inline_always]
impl PartialOrdRegister for F64x2V1 {
    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmplt_pd(lhs, rhs) }
    }

    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmple_pd(lhs, rhs) }
    }

    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpgt_pd(lhs, rhs) }
    }

    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpge_pd(lhs, rhs) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpeq_pd(lhs, rhs) }
    }

    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpneq_pd(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl NumericRegister for F64x2V1 {
    sort_via_network!(2);

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

    fn min_max_element(value: Storage<Self>) -> (Self::Element, Self::Element) {
        _mm_reduce2_pd_v1!(value; _mm_min_pd _mm_min_sd, _mm_max_pd _mm_max_sd)
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element {
        _mm_reduce_pd_v1!(value; _mm_add_pd _mm_add_sd)
    }

    fn prod_elements(value: Storage<Self>) -> Self::Element {
        _mm_reduce_pd_v1!(value; _mm_mul_pd _mm_mul_sd)
    }

    fn pairwise_sum(lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self> {
        _mm_pairwise_sum_pd_v1!(lo, hi)
    }

    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as typenum::Unsigned>::USIZE as f64)
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

    fn add_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_add_pd(lhs, arch::_mm_and_pd(rhs, mask)) }
    }

    fn sub_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sub_pd(lhs, arch::_mm_and_pd(rhs, mask)) }
    }

    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mul_pd(lhs, rhs) }
    }

    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_div_pd(lhs, rhs) }
    }

    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        // https://stackoverflow.com/a/26342944/2083075
        Self::nmul_adde(Self::trunc(Self::div(lhs, rhs)), rhs, lhs)
    }

    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::fix_min::<Self>(lhs, rhs, unsafe { arch::_mm_min_pd(lhs, rhs) })
    }

    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::fix_max::<Self>(lhs, rhs, unsafe { arch::_mm_max_pd(lhs, rhs) })
    }
}

#[thermite_macros::inline_always]
impl SignedRegister for F64x2V1 {
    const NEG_ONE: Storage<Self> = reg::<Self, 2>([-1.0; 2]);
    const MIN_POSITIVE: Storage<Self> = reg::<Self, 2>([f64::MIN_POSITIVE; 2]);

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
impl FloatRegister for F64x2V1 {
    const HAS_TRUE_FMA: bool = false;

    type Bits = super::U64x2V1;
    type SignedBits = super::I64x2V1;
    type ExtendedPrecision = Self; // f64 is the highest precision available

    const HALF: Storage<Self> = reg::<Self, 2>([0.5; 2]);
    const NEG_ZERO: Storage<Self> = reg::<Self, 2>([-0.0; 2]);
    const EPSILON: Storage<Self> = reg::<Self, 2>([f64::EPSILON; 2]);
    const INFINITY: Storage<Self> = reg::<Self, 2>([f64::INFINITY; 2]);
    const NEG_INFINITY: Storage<Self> = reg::<Self, 2>([f64::NEG_INFINITY; 2]);
    const NAN: Storage<Self> = reg::<Self, 2>([f64::NAN; 2]);

    const EXP_MASK: crate::register::Storage<Self::Bits> = reg::<Self::Bits, 2>([0x7FF0_0000_0000_0000; 2]);

    // Correctly rounded (bit-identical to hardware FMA): BM 2008 round-to-odd,
    // see `fmadd_ro`. Negations are exact, so the three variants compose from it.
    fn mul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::fmadd_ro::<Self>(lhs, rhs, acc)
    }

    fn mul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::fmadd_ro::<Self>(lhs, rhs, Self::neg(acc))
    }

    fn nmul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::fmadd_ro::<Self>(Self::neg(lhs), rhs, acc)
    }

    fn nmul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::fmadd_ro::<Self>(Self::neg(lhs), rhs, Self::neg(acc))
    }

    fn sqrt(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sqrt_pd(value) }
    }

    const HAS_APPROX_RSQRT: bool = false;
    const HAS_APPROX_RCP: bool = false;

    fn floor(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_floor_pdx_v1(value) }
    }

    fn ceil(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_ceil_pdx_v1(value) }
    }

    fn round(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_round_pdx_v1(value) }
    }

    fn trunc(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_trunc_pdx_v1(value) }
    }

    const NATIVE_CAP: NativeCapability = NativeCapability::NONE;
}

#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<F64x2V1, 2>> for super::F32x4V1 {
    fn cast_from(value: Storage<ArrayRegister<F64x2V1, 2>>) -> Storage<Self> {
        let (lo, hi) = <ArrayRegister<F64x2V1, 2> as ConcatRegister<F64x2V1>>::split(value);

        unsafe {
            let lo = arch::_mm_cvtpd_ps(lo);
            let hi = arch::_mm_cvtpd_ps(hi);

            arch::_mm_movelh_ps(lo, hi)
        }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<<Scalar as Simd>::f32x2> for F64x2V1 {
    fn cast_from(value: Storage<<Scalar as Simd>::f32x2>) -> Storage<Self> {
        unsafe { arch::_mm_cvtps_pd(arch::_mm_setr_ps(value.0[0], value.0[1], 0.0, 0.0)) }
    }
}

#[thermite_macros::inline_always]
impl CastRegister<F64x2V1> for <Scalar as Simd>::f32x2 {
    fn cast_from(value: Storage<F64x2V1>) -> Storage<<Scalar as Simd>::f32x2> {
        unsafe {
            let ps = arch::_mm_cvtpd_ps(value);

            ArrayRegister([
                arch::_mm_cvtss_f32(ps),
                f32::from_bits(arch::_mm_extract_psx_v1::<1>(ps) as u32),
            ])
        }
    }
}
