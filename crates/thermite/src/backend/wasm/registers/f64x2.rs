use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitCastRegister, BitshiftRegister, BitwiseRegister, CastRegister, ConcatRegister, CoreRegister, ExtendRegister,
        FloatRegister, InterleaveRegister, LinAlg3Register, MaskElement, MaskRegister, NativeCapability,
        NumericRegister, PartialOrdRegister, PermuteRegister, Register, ShuffleRegister, SignedRegister, Storage,
        SwizzleRegister, ZeroUpper, array::ArrayRegister, empty_reg,
    },
    swizzle::SwizzleIndices,
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct F64x2Wasm;

#[thermite_macros::inline_always]
impl CoreRegister for F64x2Wasm {
    type Lanes = typenum::U2;
    type Storage = arch::v128;
    type Mask = Self;

    const IS_EMULATED: bool = false;
    const ISA: InstructionSet = arch::ISA;
    const HAS_EQUAL_SIZE_MASK: bool = true;
    const EMPTY: Storage<Self> = arch::f64x2(0.0, 0.0);

    fn blendv(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_laneselect(rhs, lhs, mask)
    }

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        if const { Z::N >= 2 } {
            value
        } else if const { Z::N == 1 } {
            arch::v128_and(value, arch::u64x2(!0, 0))
        } else {
            Self::EMPTY
        }
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        mask
    }
}

#[thermite_macros::inline_always]
impl BitwiseRegister for F64x2Wasm {
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_xor(lhs, rhs)
    }

    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_and(lhs, rhs)
    }

    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_andnot(rhs, lhs) // NOTE: WASM andnot has operands reversed vs. the trait
    }

    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::v128_or(lhs, rhs)
    }

    fn not(value: Storage<Self>) -> Storage<Self> {
        arch::v128_not(value)
    }
}

impl InterleaveRegister for F64x2Wasm {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        let low = arch::i64x2_shuffle::<0, 2>(a, b);
        let high = arch::i64x2_shuffle::<1, 3>(a, b);
        (low, high)
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        let evens = arch::i64x2_shuffle::<0, 2>(a, b);
        let odds = arch::i64x2_shuffle::<1, 3>(a, b);
        (evens, odds)
    }
}

#[thermite_macros::inline_always]
impl MaskRegister for F64x2Wasm {
    const FALSY: Storage<Self> = arch::f64x2(f64::from_bits(0), f64::from_bits(0));
    const TRUTHY: Storage<Self> = arch::f64x2(f64::from_bits(!0), f64::from_bits(!0));

    fn set(mut mask: Storage<Self>, lane: usize, value: bool) -> Storage<Self> {
        Self::as_array_mut(&mut mask)[lane] = if value { MaskElement::TRUTHY } else { MaskElement::FALSY };
        mask
    }

    fn test(mask: Storage<Self>, lane: usize) -> bool {
        Self::as_array(&mask)[lane].to_bool()
    }

    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        arch::bx2_to_i64x2x(value)
    }

    fn all(value: Storage<Self>) -> bool {
        arch::i64x2_all_true(value)
    }

    fn any(value: Storage<Self>) -> bool {
        arch::v128_any_true(value)
    }

    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        Some(arch::i64x2_bitmask(value) as u64)
    }

    #[cfg(feature = "bitvec")]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = arch::i64x2_bitmask(value) as u32;
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

#[thermite_macros::inline_always]
impl Register for F64x2Wasm {
    type Element = f64;
    type Signed = super::I64x2Wasm;
    type Unsigned = super::U64x2Wasm;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        arch::f64x2_ne(value, Self::ZERO)
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        value
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        // `blendv` is `u8x16_relaxed_laneselect` (per-byte high-bit select), so it needs a
        // fully-smeared per-lane mask, not just the sign bit. Arithmetic-shift smears the sign
        // across the whole 64-bit lane.
        arch::i64x2_shr(value, 63)
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        arch::f64x2(value[0], value[1])
    }

    fn single(value: Self::Element) -> Storage<Self> {
        arch::f64x2(value, 0.0)
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        arch::f64x2_splat(value)
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::v128_load(ptr as *const _) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::v128_store(ptr as *mut _, value) }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_swizzle(value, arch::x2indices(1, 0))
    }

    #[rustfmt::skip]
    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        // within each 64-bit lane, swap the bytes
        arch::u8x16_relaxed_swizzle(value, arch::u8x16(
            7, 6, 5, 4, 3, 2, 1, 0,
            15, 14, 13, 12, 11, 10, 9, 8,
        ))
    }

    fn extract<const I: usize>(value: Storage<Self>) -> Self::Element {
        arch::f64x2_extract_lane::<I>(value)
    }

    fn insert<const I: usize>(value: Storage<Self>, element: Self::Element) -> Storage<Self> {
        arch::f64x2_replace_lane::<I>(value, element)
    }
}

#[thermite_macros::inline_always]
impl ShuffleRegister for F64x2Wasm {
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::blendv(const { arch::imm8x2_to_mask::<IMM8>() }, lhs, rhs)
    }
}

#[thermite_macros::inline_always]
impl PermuteRegister for F64x2Wasm {
    fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_swizzle(value, const { arch::imm8x2_to_indices::<IMM8>() })
    }
}

#[thermite_macros::inline_always]
impl SwizzleRegister for F64x2Wasm {
    const HAS_PERMUTEV: bool = true;

    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        arch::u8x16_relaxed_swizzle(value, arch::x2indices(idxs[0] as u8, idxs[1] as u8))
    }

    /// Const-index permute: emits a single `i64x2.shuffle`. Only 4 possible
    /// index combinations for 2 lanes, so this match collapses to one arm at
    /// monomorphization time without needing `generic_const_exprs`.
    fn permutev_const<I: SwizzleIndices<Self::Lanes>>(value: Storage<Self>) -> Storage<Self> {
        // Mask the indices to the 2 in-register lanes so an out-of-range index wraps instead of
        // falling through to `unreachable!()` (which is UB in release).
        match (I::INDICES[0] & 1, I::INDICES[1] & 1) {
            (0, 0) => arch::i64x2_shuffle::<0, 0>(value, value),
            (0, 1) => arch::i64x2_shuffle::<0, 1>(value, value),
            (1, 0) => arch::i64x2_shuffle::<1, 0>(value, value),
            (1, 1) => arch::i64x2_shuffle::<1, 1>(value, value),
            _ => unreachable!(),
        }
    }

    /// Const-index two-source swizzle: emits a single `i64x2.shuffle`.
    /// Indices 0-1 select from `a`, 2-3 from `b`. Only 16 cases for 2-lane registers.
    #[rustfmt::skip]
    fn swizzle_const<I: SwizzleIndices<Self::Lanes>>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        // Mask to the 4 source lanes (2 from `a`, 2 from `b`) so out-of-range indices wrap.
        match (I::INDICES[0] & 3, I::INDICES[1] & 3) {
            (0, 0) => arch::i64x2_shuffle::<0, 0>(a, b),
            (0, 1) => arch::i64x2_shuffle::<0, 1>(a, b),
            (0, 2) => arch::i64x2_shuffle::<0, 2>(a, b),
            (0, 3) => arch::i64x2_shuffle::<0, 3>(a, b),
            (1, 0) => arch::i64x2_shuffle::<1, 0>(a, b),
            (1, 1) => arch::i64x2_shuffle::<1, 1>(a, b),
            (1, 2) => arch::i64x2_shuffle::<1, 2>(a, b),
            (1, 3) => arch::i64x2_shuffle::<1, 3>(a, b),
            (2, 0) => arch::i64x2_shuffle::<2, 0>(a, b),
            (2, 1) => arch::i64x2_shuffle::<2, 1>(a, b),
            (2, 2) => arch::i64x2_shuffle::<2, 2>(a, b),
            (2, 3) => arch::i64x2_shuffle::<2, 3>(a, b),
            (3, 0) => arch::i64x2_shuffle::<3, 0>(a, b),
            (3, 1) => arch::i64x2_shuffle::<3, 1>(a, b),
            (3, 2) => arch::i64x2_shuffle::<3, 2>(a, b),
            (3, 3) => arch::i64x2_shuffle::<3, 3>(a, b),
            _ => unreachable!(),
        }
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl PartialOrdRegister for F64x2Wasm {
    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::f64x2_ge(lhs, rhs) }
    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::f64x2_lt(lhs, rhs) }
    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::f64x2_le(lhs, rhs) }
    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::f64x2_ne(lhs, rhs) }
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::f64x2_gt(lhs, rhs) }
    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::f64x2_eq(lhs, rhs) }
}

#[thermite_macros::inline_always]
impl NumericRegister for F64x2Wasm {
    const ZERO: Storage<Self> = arch::f64x2(0.0, 0.0);
    const ONE: Storage<Self> = arch::f64x2(1.0, 1.0);
    const TWO: Storage<Self> = arch::f64x2(2.0, 2.0);

    const MIN: Storage<Self> = arch::f64x2(f64::MIN, f64::MIN);
    const MAX: Storage<Self> = arch::f64x2(f64::MAX, f64::MAX);

    fn min_element(value: Storage<Self>) -> Self::Element {
        cfg_select! {
            feature = "strict_ieee754" => {
                reduce_64x2!(f value; f64x2_min f64x2_min)
            }
            _ => reduce_64x2!(f value; f64x2_relaxed_min f64x2_relaxed_min),
        }
    }

    fn max_element(value: Storage<Self>) -> Self::Element {
        cfg_select! {
            feature = "strict_ieee754" => {
                reduce_64x2!(f value; f64x2_max f64x2_max)
            }
            _ => reduce_64x2!(f value; f64x2_relaxed_max f64x2_relaxed_max),
        }
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element {
        reduce_64x2!(f value; f64x2_add f64x2_add)
    }

    fn prod_elements(value: Storage<Self>) -> Self::Element {
        reduce_64x2!(f value; f64x2_mul f64x2_mul)
    }

    fn pairwise_sum(lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self> {
        let even = arch::i64x2_shuffle::<0, 2>(lo, hi); // [a0, b0]
        let odd = arch::i64x2_shuffle::<1, 3>(lo, hi); // [a1, b1]
        arch::f64x2_add(even, odd)
    }

    fn offset() -> Storage<Self> {
        arch::f64x2_splat(<Self::Lanes as Unsigned>::USIZE as f64)
    }

    fn indexed() -> Storage<Self> {
        arch::f64x2(0.0, 1.0)
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::f64x2_add(lhs, rhs)
    }

    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::f64x2_sub(lhs, rhs)
    }

    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::f64x2_mul(lhs, rhs)
    }

    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::f64x2_div(lhs, rhs)
    }

    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        // https://stackoverflow.com/a/26342944/2083075
        Self::nmul_adde(Self::trunc(Self::div(lhs, rhs)), rhs, lhs)
    }

    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        cfg_select! {
            feature = "strict_ieee754" => {
                arch::f64x2_min(lhs, rhs)
            }
            _ => arch::f64x2_relaxed_min(lhs, rhs),
        }
    }

    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        cfg_select! {
            feature = "strict_ieee754" => {
                arch::f64x2_max(lhs, rhs)
            }
            _ => arch::f64x2_relaxed_max(lhs, rhs),
        }
    }
}

#[thermite_macros::inline_always]
impl SignedRegister for F64x2Wasm {
    const NEG_ONE: Storage<Self> = arch::f64x2(-1.0, -1.0);
    const MIN_POSITIVE: Storage<Self> = arch::f64x2(f64::MIN_POSITIVE, f64::MIN_POSITIVE);

    fn neg(value: Storage<Self>) -> Storage<Self> {
        arch::f64x2_neg(value)
    }

    fn abs(value: Storage<Self>) -> Storage<Self> {
        arch::f64x2_abs(value)
    }

    fn copysign(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::bitor(Self::bitandnot(Self::NEG_ZERO, lhs), Self::bitand(Self::NEG_ZERO, rhs))
    }

    fn signum(value: Storage<Self>) -> Storage<Self> {
        Self::bitor(Self::ONE, Self::bitand(value, Self::NEG_ZERO))
    }

    fn neg_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        Self::bitxor(value, Self::bitand(Self::NEG_ZERO, mask))
    }
}

#[thermite_macros::inline_always]
impl FloatRegister for F64x2Wasm {
    // No way to know if FMA is supported at compile time
    const HAS_TRUE_FMA: bool = false;

    type Bits = super::U64x2Wasm;
    type SignedBits = super::I64x2Wasm;
    type ExtendedPrecision = Self;

    const HALF: Storage<Self> = arch::f64x2(0.5, 0.5);
    const NEG_ZERO: Storage<Self> = arch::f64x2(-0.0, -0.0);
    const EPSILON: Storage<Self> = arch::f64x2(f64::EPSILON, f64::EPSILON);
    const INFINITY: Storage<Self> = arch::f64x2(f64::INFINITY, f64::INFINITY);
    const NEG_INFINITY: Storage<Self> = arch::f64x2(f64::NEG_INFINITY, f64::NEG_INFINITY);
    const NAN: Storage<Self> = arch::f64x2(f64::NAN, f64::NAN);

    const EXP_MASK: Storage<Self::Bits> = arch::u64x2(0x7FF0_0000_0000_0000, 0x7FF0_0000_0000_0000);

    // These MUST use a polyfill implementation to ensure consistent behavior across platforms

    fn mul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::f64x2_maddx(lhs, rhs, acc)
    }

    fn mul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::f64x2_maddx(lhs, rhs, arch::f64x2_neg(acc))
    }

    fn nmul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::f64x2_maddx(arch::f64x2_neg(lhs), rhs, acc)
    }

    fn nmul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::f64x2_maddx(arch::f64x2_neg(lhs), rhs, arch::f64x2_neg(acc))
    }

    // These, however, can use the native relaxed FMA instructions, which
    // may or may not be true FMA depending on the platform.

    fn mul_adde(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::f64x2_relaxed_madd(lhs, rhs, acc)
    }

    fn mul_sube(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::f64x2_relaxed_madd(lhs, rhs, arch::f64x2_neg(acc))
    }

    fn nmul_adde(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::f64x2_relaxed_nmadd(lhs, rhs, acc)
    }

    fn nmul_sube(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::f64x2_relaxed_nmadd(lhs, rhs, arch::f64x2_neg(acc))
    }

    fn sqrt(value: Storage<Self>) -> Storage<Self> {
        arch::f64x2_sqrt(value)
    }

    const HAS_APPROX_RCP: bool = false;
    const HAS_APPROX_RSQRT: bool = false;

    fn floor(value: Storage<Self>) -> Storage<Self> {
        arch::f64x2_floor(value)
    }

    fn ceil(value: Storage<Self>) -> Storage<Self> {
        arch::f64x2_ceil(value)
    }

    fn round(value: Storage<Self>) -> Storage<Self> {
        arch::f64x2_nearest(value)
    }

    fn trunc(value: Storage<Self>) -> Storage<Self> {
        arch::f64x2_trunc(value)
    }

    const NATIVE_CAP: NativeCapability = NativeCapability::NONE;
}

#[thermite_macros::inline_always]
impl CastRegister<ArrayRegister<F64x2Wasm, 2>> for super::F32x4Wasm {
    fn cast_from(value: Storage<ArrayRegister<F64x2Wasm, 2>>) -> Storage<Self> {
        let lo_demoted = arch::f32x4_demote_f64x2_zero(value.0[0]);
        let hi_demoted = arch::f32x4_demote_f64x2_zero(value.0[1]);
        arch::i32x4_shuffle::<0, 1, 4, 5>(lo_demoted, hi_demoted)
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::U64x2Wasm> for F64x2Wasm {
    fn cast_from(value: Storage<super::U64x2Wasm>) -> Storage<Self> {
        arch::convert_u64x2_to_f64x2(value)
    }

    fn fast_cast_from(value: Storage<super::U64x2Wasm>) -> Storage<Self> {
        arch::convert_epu64_pd_limited::<Self>(value)
    }
}

#[thermite_macros::inline_always]
impl CastRegister<super::I64x2Wasm> for F64x2Wasm {
    fn cast_from(value: Storage<super::I64x2Wasm>) -> Storage<Self> {
        arch::convert_i64x2_to_f64x2(value)
    }

    fn fast_cast_from(value: Storage<super::I64x2Wasm>) -> Storage<Self> {
        arch::convert_epi64_pd_limited::<Self>(value)
    }
}

#[thermite_macros::inline_always]
impl CastRegister<F64x2Wasm> for super::U64x2Wasm {
    fn cast_from(value: Storage<F64x2Wasm>) -> Storage<Self> {
        arch::u64x2(
            arch::f64x2_extract_lane::<0>(value) as u64,
            arch::f64x2_extract_lane::<1>(value) as u64,
        )
    }

    fn fast_cast_from(value: Storage<F64x2Wasm>) -> Storage<Self> {
        arch::convert_pd_epu64_limited::<F64x2Wasm>(value)
    }
}

#[thermite_macros::inline_always]
impl CastRegister<F64x2Wasm> for super::I64x2Wasm {
    fn cast_from(value: Storage<F64x2Wasm>) -> Storage<Self> {
        arch::i64x2(
            arch::f64x2_extract_lane::<0>(value) as i64,
            arch::f64x2_extract_lane::<1>(value) as i64,
        )
    }

    fn fast_cast_from(value: Storage<F64x2Wasm>) -> Storage<Self> {
        arch::convert_pd_epi64_limited::<F64x2Wasm>(value)
    }
}

#[thermite_macros::inline_always]
impl ConcatRegister<f64> for F64x2Wasm {
    fn concat(lo: Storage<f64>, hi: Storage<f64>) -> Storage<Self> {
        arch::f64x2(lo, hi)
    }

    fn split(value: Storage<Self>) -> (Storage<f64>, Storage<f64>) {
        (
            arch::f64x2_extract_lane::<0>(value),
            arch::f64x2_extract_lane::<1>(value),
        )
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<f64> for F64x2Wasm {
    fn extend(value: Storage<f64>) -> Storage<Self> {
        arch::f64x2(value, 0.0)
    }

    fn narrow(value: Storage<Self>) -> Storage<f64> {
        arch::f64x2_extract_lane::<0>(value)
    }
}
