use generic_array::{
    GenericArray,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitwiseRegister, CastRegister, CoreRegister, FloatRegister, InterleaveRegister, LinAlg3Register,
        LinAlg4Register, MaskElement, MaskRegister, NativeCapability, NumericRegister, PartialOrdRegister,
        PermuteRegister, Register, ShuffleRegister, SignedRegister, Storage, ZeroUpper, array::ArrayRegister, reg,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct F32x4Wasm;

#[thermite_macros::inline_always]
impl CoreRegister for F32x4Wasm {
    type Lanes = typenum::U4;
    type Storage = arch::v128;
    type Mask = Self;

    const IS_EMULATED: bool = false;
    const HAS_EQUAL_SIZE_MASK: bool = true;
    const EMPTY: Storage<Self> = arch::f32x4(0.0, 0.0, 0.0, 0.0);

    fn blendv(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_laneselect(rhs, lhs, mask)
    }

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        if const { Z::N >= 4 } {
            value
        } else if const { Z::N == 3 } {
            arch::v128_and(value, arch::u32x4(!0, !0, !0, 0))
        } else if const { Z::N == 2 } {
            arch::v128_and(value, arch::u32x4(!0, !0, 0, 0))
        } else if const { Z::N == 1 } {
            arch::v128_and(value, arch::u32x4(!0, 0, 0, 0))
        } else {
            Self::EMPTY
        }
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        mask
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for F32x4Wasm {
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

#[thermite_macros::inline_always]
impl InterleaveRegister for F32x4Wasm {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        let low = arch::i32x4_shuffle::<0, 4, 1, 5>(a, b);
        let high = arch::i32x4_shuffle::<2, 6, 3, 7>(a, b);
        (low, high)
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        let evens = arch::i32x4_shuffle::<0, 2, 4, 6>(a, b);
        let odds = arch::i32x4_shuffle::<1, 3, 5, 7>(a, b);
        (evens, odds)
    }
}

#[thermite_macros::inline_always]
impl MaskRegister for F32x4Wasm {
    const FALSY: Storage<Self> = arch::f32x4(
        f32::from_bits(0),
        f32::from_bits(0),
        f32::from_bits(0),
        f32::from_bits(0),
    );

    const TRUTHY: Storage<Self> = arch::f32x4(
        f32::from_bits(!0),
        f32::from_bits(!0),
        f32::from_bits(!0),
        f32::from_bits(!0),
    );

    fn set(mut mask: Storage<Self>, lane: usize, value: bool) -> Storage<Self> {
        Self::as_mut_slice(&mut mask)[lane] = if value { MaskElement::TRUTHY } else { MaskElement::FALSY };
        mask
    }

    fn test(mask: Storage<Self>, lane: usize) -> bool {
        Self::as_slice(&mask)[lane].to_bool()
    }

    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        arch::bx4_to_i32x4x(value)
    }

    fn all(value: Storage<Self>) -> bool {
        arch::i32x4_all_true(value)
    }

    fn any(value: Storage<Self>) -> bool {
        arch::v128_any_true(value)
    }

    fn from_native_bitmask(bitmask: u64) -> Storage<Self> {
        arch::bitmask_to_i32x4x(bitmask)
    }

    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        Some(arch::i32x4_bitmask(value) as u64)
    }

    #[cfg(feature = "bitvec")]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = arch::i32x4_bitmask(value) as u32;
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

#[thermite_macros::inline_always]
impl Register for F32x4Wasm {
    type Element = f32;
    type Signed = super::I32x4Wasm;
    type Unsigned = super::U32x4Wasm;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        arch::f32x4_ne(value, Self::ZERO)
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        value
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        // `blendv` is `u8x16_relaxed_laneselect`, which selects per *byte* on each byte's high
        // bit - so it needs a fully-smeared per-lane mask, not just the sign bit (unlike x86's
        // `blendv_ps`, which checks only the lane MSB). Arithmetic-shift smears the sign across
        // the whole 32-bit lane.
        arch::i32x4_shr(value, 31)
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        arch::f32x4(value[0], value[1], value[2], value[3])
    }

    fn single(value: Self::Element) -> Storage<Self> {
        arch::f32x4(value, 0.0, 0.0, 0.0)
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        arch::f32x4_splat(value)
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::v128_load(ptr as *const _) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::v128_store(ptr as *mut _, value) }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_swizzle(value, arch::x4indices(3, 2, 1, 0))
    }

    #[rustfmt::skip]
    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        // within each 32-bit lane, swap the bytes
        arch::u8x16_relaxed_swizzle(value, arch::u8x16(
            3, 2, 1, 0,
            7, 6, 5, 4,
            11, 10, 9, 8,
            15, 14, 13, 12,
        ))
    }

    fn extract<const I: usize>(value: Storage<Self>) -> Self::Element {
        arch::f32x4_extract_lane::<I>(value)
    }

    fn insert<const I: usize>(value: Storage<Self>, element: Self::Element) -> Storage<Self> {
        arch::f32x4_replace_lane::<I>(value, element)
    }

    const HAS_PERMUTEV: bool = true;

    impl_wasm_align_shuffle!();

    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        arch::u8x16_relaxed_swizzle(
            value,
            arch::wasm_lane_table_dyn::<4>(unsafe { core::mem::transmute(idxs) }),
        )
    }

    compress_via_table!();
}

#[thermite_macros::inline_always]
impl ShuffleRegister for F32x4Wasm {
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::blendv(const { arch::imm8x4_to_mask::<IMM8>() }, lhs, rhs)
    }
}

#[thermite_macros::inline_always]
impl PermuteRegister for F32x4Wasm {
    fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_swizzle(value, const { arch::imm8x4_to_indices::<IMM8>() })
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl PartialOrdRegister for F32x4Wasm {
    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::f32x4_ge(lhs, rhs) }
    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::f32x4_lt(lhs, rhs) }
    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::f32x4_le(lhs, rhs) }
    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::f32x4_ne(lhs, rhs) }
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::f32x4_gt(lhs, rhs) }
    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::f32x4_eq(lhs, rhs) }
}

#[thermite_macros::inline_always]
impl NumericRegister for F32x4Wasm {
    sort_via_network!(4);

    const ZERO: Storage<Self> = arch::f32x4(0.0, 0.0, 0.0, 0.0);
    const ONE: Storage<Self> = arch::f32x4(1.0, 1.0, 1.0, 1.0);
    const TWO: Storage<Self> = arch::f32x4(2.0, 2.0, 2.0, 2.0);

    const MIN: Storage<Self> = arch::f32x4(f32::MIN, f32::MIN, f32::MIN, f32::MIN);
    const MAX: Storage<Self> = arch::f32x4(f32::MAX, f32::MAX, f32::MAX, f32::MAX);

    fn min_element(value: Storage<Self>) -> Self::Element {
        cfg_select! {
            feature = "strict_ieee754" => {
                reduce_32x4!(f value; f32x4_min f32x4_min)
            }
            _ => reduce_32x4!(f value; f32x4_relaxed_min f32x4_relaxed_min),
        }
    }

    fn max_element(value: Storage<Self>) -> Self::Element {
        cfg_select! {
            feature = "strict_ieee754" => {
                reduce_32x4!(f value; f32x4_max f32x4_max)
            }
            _ => reduce_32x4!(f value; f32x4_relaxed_max f32x4_relaxed_max),
        }
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element {
        reduce_32x4!(f value; f32x4_add f32x4_add)
    }

    fn prod_elements(value: Storage<Self>) -> Self::Element {
        reduce_32x4!(f value; f32x4_mul f32x4_mul)
    }

    fn pairwise_sum(lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self> {
        let even = arch::i32x4_shuffle::<0, 2, 4, 6>(lo, hi); // [a0,a2,b0,b2]
        let odd = arch::i32x4_shuffle::<1, 3, 5, 7>(lo, hi); // [a1,a3,b1,b3]
        arch::f32x4_add(even, odd)
    }

    fn offset() -> Storage<Self> {
        arch::f32x4_splat(<Self::Lanes as Unsigned>::USIZE as f32)
    }

    fn indexed() -> Storage<Self> {
        arch::f32x4(0.0, 1.0, 2.0, 3.0)
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::f32x4_add(lhs, rhs)
    }

    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::f32x4_sub(lhs, rhs)
    }

    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::f32x4_mul(lhs, rhs)
    }

    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::f32x4_div(lhs, rhs)
    }

    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        // https://stackoverflow.com/a/26342944/2083075
        Self::nmul_adde(Self::trunc(Self::div(lhs, rhs)), rhs, lhs)
    }

    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        cfg_select! {
            feature = "strict_ieee754" => {
                arch::f32x4_min(lhs, rhs)
            }
            _ => arch::f32x4_relaxed_min(lhs, rhs),
        }
    }

    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        cfg_select! {
            feature = "strict_ieee754" => {
                arch::f32x4_max(lhs, rhs)
            }
            _ => arch::f32x4_relaxed_max(lhs, rhs),
        }
    }
}

#[thermite_macros::inline_always]
impl SignedRegister for F32x4Wasm {
    const NEG_ONE: Storage<Self> = arch::f32x4(-1.0, -1.0, -1.0, -1.0);
    const MIN_POSITIVE: Storage<Self> = arch::f32x4(
        f32::MIN_POSITIVE,
        f32::MIN_POSITIVE,
        f32::MIN_POSITIVE,
        f32::MIN_POSITIVE,
    );

    fn neg(value: Storage<Self>) -> Storage<Self> {
        arch::f32x4_neg(value)
    }

    fn abs(value: Storage<Self>) -> Storage<Self> {
        arch::f32x4_abs(value)
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
impl FloatRegister for F32x4Wasm {
    // No way to know if FMA is supported at compile time
    const HAS_TRUE_FMA: bool = false;

    type Bits = super::U32x4Wasm;
    type SignedBits = super::I32x4Wasm;
    type ExtendedPrecision = ArrayRegister<super::F64x2Wasm, 2>;

    const HALF: Storage<Self> = arch::f32x4(0.5, 0.5, 0.5, 0.5);
    const NEG_ZERO: Storage<Self> = arch::f32x4(-0.0, -0.0, -0.0, -0.0);
    const EPSILON: Storage<Self> = arch::f32x4(f32::EPSILON, f32::EPSILON, f32::EPSILON, f32::EPSILON);
    const INFINITY: Storage<Self> = arch::f32x4(f32::INFINITY, f32::INFINITY, f32::INFINITY, f32::INFINITY);
    const NEG_INFINITY: Storage<Self> = arch::f32x4(
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    );
    const NAN: Storage<Self> = arch::f32x4(f32::NAN, f32::NAN, f32::NAN, f32::NAN);

    const EXP_MASK: Storage<Self::Bits> = arch::u32x4(0x7F800000, 0x7F800000, 0x7F800000, 0x7F800000);

    // Correctly rounded (bit-identical to hardware FMA) on every engine: the
    // relaxed madd instruction when a one-time canary proves it is a genuine
    // fused FMA (single instruction), else the widen + round-to-odd emulation.
    // Both branches produce identical bits, see `f32x4_fmadd_auto`.

    fn mul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::f32x4_fmadd_auto(lhs, rhs, acc)
    }

    fn mul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::f32x4_fmadd_auto(lhs, rhs, arch::f32x4_neg(acc))
    }

    fn nmul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::f32x4_fnmadd_auto(lhs, rhs, acc)
    }

    fn nmul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::f32x4_fnmadd_auto(lhs, rhs, arch::f32x4_neg(acc))
    }

    // These, however, can use the native relaxed FMA instructions, which
    // may or may not be true FMA depending on the platform.

    fn mul_adde(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::f32x4_relaxed_madd(lhs, rhs, acc)
    }

    fn mul_sube(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::f32x4_relaxed_madd(lhs, rhs, arch::f32x4_neg(acc))
    }

    fn nmul_adde(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::f32x4_relaxed_nmadd(lhs, rhs, acc)
    }

    fn nmul_sube(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::f32x4_relaxed_nmadd(lhs, rhs, arch::f32x4_neg(acc))
    }

    fn sqrt(value: Storage<Self>) -> Storage<Self> {
        arch::f32x4_sqrt(value)
    }

    const HAS_APPROX_RCP: bool = false;
    const HAS_APPROX_RSQRT: bool = false;

    fn floor(value: Storage<Self>) -> Storage<Self> {
        arch::f32x4_floor(value)
    }

    fn ceil(value: Storage<Self>) -> Storage<Self> {
        arch::f32x4_ceil(value)
    }

    fn round(value: Storage<Self>) -> Storage<Self> {
        arch::f32x4_nearest(value)
    }

    fn trunc(value: Storage<Self>) -> Storage<Self> {
        arch::f32x4_trunc(value)
    }

    const NATIVE_CAP: NativeCapability = NativeCapability::NONE;
}

#[thermite_macros::inline_always]
impl CastRegister<F32x4Wasm> for ArrayRegister<super::F64x2Wasm, 2> {
    fn cast_from(value: Storage<F32x4Wasm>) -> Storage<Self> {
        // promote each pair of f32 to f64
        let lo = arch::f64x2_promote_low_f32x4(value);
        let hi = arch::f64x2_promote_low_f32x4(arch::i32x4_shuffle::<2, 3, 2, 3>(value, value));
        ArrayRegister([lo, hi])
    }
}

#[thermite_macros::inline_always]
impl LinAlg3Register for F32x4Wasm {
    fn min_element3(value: Storage<Self>) -> Self::Element {
        // Replace lane 3 with +∞ (identity for min) so the 4-lane reduction ignores it.
        let v = arch::f32x4_replace_lane::<3>(value, f32::INFINITY);
        cfg_select! {
            feature = "strict_ieee754" => {
                reduce_32x4!(f v; f32x4_min f32x4_min)
            }
            _ => reduce_32x4!(f v; f32x4_relaxed_min f32x4_relaxed_min),
        }
    }

    fn max_element3(value: Storage<Self>) -> Self::Element {
        // Replace lane 3 with -∞ (identity for max) so the 4-lane reduction ignores it.
        let v = arch::f32x4_replace_lane::<3>(value, f32::NEG_INFINITY);
        cfg_select! {
            feature = "strict_ieee754" => {
                reduce_32x4!(f v; f32x4_max f32x4_max)
            }
            _ => reduce_32x4!(f v; f32x4_relaxed_max f32x4_relaxed_max)
        }
    }

    fn sum_elements3(value: Storage<Self>) -> Self::Element {
        // Replace lane 3 with 0 (additive identity) so the 4-lane reduction ignores it.
        let v = arch::f32x4_replace_lane::<3>(value, 0.0);
        reduce_32x4!(f v; f32x4_add f32x4_add)
    }

    fn prod_elements3(value: Storage<Self>) -> Self::Element {
        // Replace lane 3 with 1 (multiplicative identity) so the 4-lane reduction ignores it.
        let v = arch::f32x4_replace_lane::<3>(value, 1.0);
        reduce_32x4!(f v; f32x4_mul f32x4_mul)
    }
}

macro_rules! s {
    ($ty:ty: $v:expr, [$a:literal, $b:literal, $c:literal, $d:literal]) => {
        arch::i32x4_shuffle::<$a, $b, $c, $d>($v, $v)
    };
    ($ty:ty: $v1:expr, $v2:expr, [$a:literal, $b:literal, $c:literal, $d:literal]) => {
        arch::i32x4_shuffle::<$a, $b, $c, $d>($v1, $v2)
    };
}

#[thermite_macros::inline_always]
impl LinAlg4Register for F32x4Wasm {
    // dedicated WASM implementation that takes advantage of `i32x4_shuffle` directly.
    fn mat4_adjugate<const FAST: bool>(m: &[Storage<Self>; 4]) -> ([Storage<Self>; 4], Self::Element) {
        impl_mat4_inverse!(m, s)
    }

    fn mat4_det<const FAST: bool>(m: &[Storage<Self>; 4]) -> Self::Element {
        impl_mat4_inverse!(DET_ONLY m, s)
    }
}
