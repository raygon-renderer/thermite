use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitsRegister, BitshiftRegister, CastRegister, FloatRegister, LinAlg3Register, MaskRegister, NumericRegister,
        PartialOrdRegister, PermuteRegister, Register, ShuffleRegister, SignedRegister, Storage, SwizzleRegister,
        dp::DoublePumpRegister,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct F64x2Wasm32;

impl Register for F64x2Wasm32 {
    type Lanes = typenum::U2;

    type Element = f64;
    type Storage = arch::v128;
    type HalfRegister = ();
    type DoubleRegister = DoublePumpRegister<Self>;

    const ISA: InstructionSet = InstructionSet::WASM32;

    type ISize = super::I64x2Wasm32;
    type USize = super::U64x2Wasm32;

    const EMPTY: Storage<Self> = arch::f64x2(0.0, 0.0);

    #[inline(always)]
    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        arch::f64x2(value[0], value[1])
    }

    #[inline(always)]
    fn single(value: Self::Element) -> Storage<Self> {
        arch::f64x2(value, 0.0)
    }

    #[inline(always)]
    fn splat(value: Self::Element) -> Storage<Self> {
        arch::f64x2_splat(value)
    }

    fn extract<const I: usize>(value: Storage<Self>) -> Self::Element {
        arch::f64x2_extract_lane::<I>(value)
    }

    #[inline(always)]
    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::v128_load(ptr as *const _) }
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::v128_store(ptr as *mut _, value) }
    }

    #[inline(always)]
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_xor(lhs, rhs)
    }

    #[inline(always)]
    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_and(lhs, rhs)
    }

    #[inline(always)]
    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_or(lhs, rhs)
    }

    #[inline(always)]
    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_andnot(rhs, lhs) // NOTE: arguments are reversed
    }

    #[inline(always)]
    fn not(value: Storage<Self>) -> Storage<Self> {
        arch::v128_not(value)
    }

    #[inline(always)]
    fn blendv(mask: Storage<Self>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_laneselect(rhs, lhs, mask)
    }

    const HAS_MSB_BLENDV: bool = false;

    #[inline(always)]
    fn reverse(mut value: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_swizzle(value, arch::x2indices(1, 0))
    }

    #[inline(always)]
    fn unpack(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        let low = Self::swizzle(a, b, GenericArray::from_array([0, 2]));
        let high = Self::swizzle(a, b, GenericArray::from_array([1, 3]));

        (low, high)
    }

    #[inline(always)]
    #[rustfmt::skip]
    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        // within each 64-bit lane, swap the bytes
        arch::u8x16_relaxed_swizzle(value, arch::u8x16(
            7, 6, 5, 4,3, 2, 1, 0,
            15, 14, 13, 12, 11, 10, 9, 8,
        ))
    }
}

impl MaskRegister for F64x2Wasm32 {
    const FALSY: Storage<Self> = arch::f64x2(f64::from_bits(0), f64::from_bits(0));
    const TRUTHY: Storage<Self> = arch::f64x2(f64::from_bits(!0), f64::from_bits(!0));

    #[inline(always)]
    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        arch::bx2_to_i64x2x(value)
    }

    #[inline(always)]
    fn all(value: Storage<Self>) -> bool {
        arch::i64x2_all_true(value)
    }

    #[inline(always)]
    fn any(value: Storage<Self>) -> bool {
        arch::v128_any_true(value)
    }

    #[inline(always)]
    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        Some(arch::i64x2_bitmask(value) as u64)
    }

    #[inline(always)]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = arch::i64x2_bitmask(value) as u32;
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

impl ShuffleRegister for F64x2Wasm32 {
    #[inline(always)]
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::blendv(const { arch::imm8x2_to_mask::<IMM8>() }, lhs, rhs)
    }
}

impl PermuteRegister for F64x2Wasm32 {
    #[inline(always)]
    fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_swizzle(value, const { arch::imm8x2_to_indices::<IMM8>() })
    }
}

impl SwizzleRegister for F64x2Wasm32 {
    const HAS_PERMUTEV: bool = true;

    #[inline(always)]
    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        arch::u8x16_relaxed_swizzle(value, arch::x2indices(idxs[0] as u8, idxs[1] as u8))
    }
}

#[rustfmt::skip]
impl PartialOrdRegister for F64x2Wasm32 {
    #[inline(always)] fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::f64x2_ge(lhs, rhs) }
    #[inline(always)] fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::f64x2_lt(lhs, rhs) }
    #[inline(always)] fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::f64x2_le(lhs, rhs) }
    #[inline(always)] fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::f64x2_ne(lhs, rhs) }
    #[inline(always)] fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::f64x2_gt(lhs, rhs) }
    #[inline(always)] fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::f64x2_eq(lhs, rhs) }
}

impl NumericRegister for F64x2Wasm32 {
    const ZERO: Storage<Self> = arch::f64x2(0.0, 0.0);
    const ONE: Storage<Self> = arch::f64x2(1.0, 1.0);
    const TWO: Storage<Self> = arch::f64x2(2.0, 2.0);

    const MIN: Storage<Self> = arch::f64x2(f64::MIN, f64::MIN);
    const MAX: Storage<Self> = arch::f64x2(f64::MAX, f64::MAX);

    #[inline(always)]
    fn min_element(value: Storage<Self>) -> Self::Element {
        reduce_64x2!(f value; f64x2_relaxed_min f64x2_relaxed_min)
    }

    #[inline(always)]
    fn max_element(value: Storage<Self>) -> Self::Element {
        reduce_64x2!(f value; f64x2_relaxed_max f64x2_relaxed_max)
    }

    #[inline(always)]
    fn sum_elements(value: Storage<Self>) -> Self::Element {
        reduce_64x2!(f value; f64x2_add f64x2_add)
    }

    #[inline(always)]
    fn prod_elements(value: Storage<Self>) -> Self::Element {
        reduce_64x2!(f value; f64x2_mul f64x2_mul)
    }

    #[inline(always)]
    fn offset() -> Storage<Self> {
        arch::f64x2_splat(<Self::Lanes as Unsigned>::USIZE as f64)
    }

    #[inline(always)]
    fn indexed() -> Storage<Self> {
        arch::f64x2(0.0, 1.0)
    }

    #[inline(always)]
    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::f64x2_add(lhs, rhs)
    }

    #[inline(always)]
    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::f64x2_sub(lhs, rhs)
    }

    #[inline(always)]
    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::f64x2_mul(lhs, rhs)
    }

    #[inline(always)]
    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::f64x2_div(lhs, rhs)
    }

    #[inline(always)]
    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        // https://stackoverflow.com/a/26342944/2083075
        Self::nmul_adde(Self::trunc(Self::div(lhs, rhs)), rhs, lhs)
    }

    #[inline(always)]
    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::f64x2_relaxed_min(lhs, rhs)
    }

    #[inline(always)]
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::f64x2_relaxed_max(lhs, rhs)
    }
}

impl SignedRegister for F64x2Wasm32 {
    const NEG_ONE: Storage<Self> = arch::f64x2(-1.0, -1.0);
    const MIN_POSITIVE: Storage<Self> = arch::f64x2(f64::MIN_POSITIVE, f64::MIN_POSITIVE);

    #[inline(always)]
    fn neg(value: Storage<Self>) -> Storage<Self> {
        arch::f64x2_neg(value)
    }

    #[inline(always)]
    fn abs(value: Storage<Self>) -> Storage<Self> {
        arch::f64x2_abs(value)
    }

    #[inline(always)]
    fn copysign(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
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

impl FloatRegister for F64x2Wasm32 {
    // No way to know if FMA is supported at compile time
    const HAS_TRUE_FMA: bool = false;

    type Bits = super::U64x2Wasm32;
    type Signed = super::I64x2Wasm32;
    type ExtendedPrecision = Self;

    const HALF: Storage<Self> = arch::f64x2(0.5, 0.5);
    const NEG_ZERO: Storage<Self> = arch::f64x2(-0.0, -0.0);
    const EPSILON: Storage<Self> = arch::f64x2(f64::EPSILON, f64::EPSILON);
    const INFINITY: Storage<Self> = arch::f64x2(f64::INFINITY, f64::INFINITY);
    const NEG_INFINITY: Storage<Self> = arch::f64x2(f64::NEG_INFINITY, f64::NEG_INFINITY);
    const NAN: Storage<Self> = arch::f64x2(f64::NAN, f64::NAN);

    const EXP_MASK: Storage<Self::Bits> = arch::u64x2(0x7FF0_0000_0000_0000, 0x7FF0_0000_0000_0000);

    // These MUST use a polyfill implementation to ensure consistent behavior across platforms

    #[inline(always)]
    fn mul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::f64x2_maddx(lhs, rhs, acc)
    }

    #[inline(always)]
    fn mul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::f64x2_maddx(lhs, rhs, arch::f64x2_neg(acc))
    }

    #[inline(always)]
    fn nmul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::f64x2_maddx(arch::f64x2_neg(lhs), rhs, acc)
    }

    #[inline(always)]
    fn nmul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::f64x2_maddx(arch::f64x2_neg(lhs), rhs, arch::f64x2_neg(acc))
    }

    // These, however, can use the native relaxed FMA instructions, which
    // may or may not be true FMA depending on the platform.

    #[inline(always)]
    fn mul_adde(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::f64x2_relaxed_madd(lhs, rhs, acc)
    }

    #[inline(always)]
    fn mul_sube(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::f64x2_relaxed_madd(lhs, rhs, arch::f64x2_neg(acc))
    }

    #[inline(always)]
    fn nmul_adde(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::f64x2_relaxed_nmadd(lhs, rhs, acc)
    }

    #[inline(always)]
    fn nmul_sube(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::f64x2_relaxed_nmadd(lhs, rhs, arch::f64x2_neg(acc))
    }

    #[inline(always)]
    fn sqrt(value: Storage<Self>) -> Storage<Self> {
        arch::f64x2_sqrt(value)
    }

    const HAS_APPROX_RCP: bool = false;
    const HAS_APPROX_RSQRT: bool = false;

    #[inline(always)]
    fn floor(value: Storage<Self>) -> Storage<Self> {
        arch::f64x2_floor(value)
    }

    #[inline(always)]
    fn ceil(value: Storage<Self>) -> Storage<Self> {
        arch::f64x2_ceil(value)
    }

    #[inline(always)]
    fn round(value: Storage<Self>) -> Storage<Self> {
        arch::f64x2_nearest(value)
    }

    #[inline(always)]
    fn trunc(value: Storage<Self>) -> Storage<Self> {
        arch::f64x2_trunc(value)
    }
}

impl CastRegister<DoublePumpRegister<F64x2Wasm32>> for super::F32x4Wasm32 {
    #[inline(always)]
    fn cast_from(value: Storage<DoublePumpRegister<F64x2Wasm32>>) -> Storage<Self> {
        let lo_demoted = arch::f32x4_demote_f64x2_zero(value.0);
        let hi_demoted = arch::f32x4_demote_f64x2_zero(value.1);

        arch::i32x4_shuffle::<0, 1, 4, 5>(lo_demoted, hi_demoted)
    }
}

impl CastRegister<super::U64x2Wasm32> for F64x2Wasm32 {
    #[inline(always)]
    fn cast_from(value: Storage<super::U64x2Wasm32>) -> Storage<Self> {
        arch::convert_u64x2_to_f64x2(value)
    }

    #[inline(always)]
    fn fast_cast_from(value: Storage<super::U64x2Wasm32>) -> Storage<Self> {
        arch::convert_epu64_pd_limited::<Self>(value)
    }
}

impl CastRegister<super::I64x2Wasm32> for F64x2Wasm32 {
    #[inline(always)]
    fn cast_from(value: Storage<super::I64x2Wasm32>) -> Storage<Self> {
        arch::convert_i64x2_to_f64x2(value)
    }

    #[inline(always)]
    fn fast_cast_from(value: Storage<super::I64x2Wasm32>) -> Storage<Self> {
        arch::convert_epi64_pd_limited::<Self>(value)
    }
}

impl CastRegister<F64x2Wasm32> for super::U64x2Wasm32 {
    #[inline(always)]
    fn cast_from(value: Storage<F64x2Wasm32>) -> Storage<Self> {
        arch::u64x2(
            arch::f64x2_extract_lane::<0>(value) as u64,
            arch::f64x2_extract_lane::<1>(value) as u64,
        )
    }

    #[inline(always)]
    fn fast_cast_from(value: Storage<F64x2Wasm32>) -> Storage<Self> {
        arch::convert_pd_epu64_limited::<F64x2Wasm32>(value)
    }
}

impl CastRegister<F64x2Wasm32> for super::I64x2Wasm32 {
    #[inline(always)]
    fn cast_from(value: Storage<F64x2Wasm32>) -> Storage<Self> {
        arch::i64x2(
            arch::f64x2_extract_lane::<0>(value) as i64,
            arch::f64x2_extract_lane::<1>(value) as i64,
        )
    }

    #[inline(always)]
    fn fast_cast_from(value: Storage<F64x2Wasm32>) -> Storage<Self> {
        arch::convert_pd_epi64_limited::<F64x2Wasm32>(value)
    }
}
