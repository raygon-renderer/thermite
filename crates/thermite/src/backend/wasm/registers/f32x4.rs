use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitsRegister, BitshiftRegister, CastRegister, FloatRegister, LinAlg3Register, LinAlg4Register, MaskRegister,
        NumericRegister, PartialOrdRegister, PermuteRegister, Register, ShuffleRegister, SignedRegister, Storage,
        SwizzleRegister, dp::DoublePumpRegister,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct F32x4Wasm;

impl Register for F32x4Wasm {
    type Lanes = typenum::U4;

    type Element = f32;
    type Storage = arch::v128;
    type HalfRegister = ();
    type DoubleRegister = DoublePumpRegister<Self>;

    const IS_EMULATED: bool = false;

    const ISA: InstructionSet = arch::ISA;

    type ISize = super::I32x4Wasm;
    type USize = super::U32x4Wasm;

    const EMPTY: Storage<Self> = arch::f32x4(0.0, 0.0, 0.0, 0.0);

    #[inline(always)]
    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        arch::f32x4(value[0], value[1], value[2], value[3])
    }

    #[inline(always)]
    fn single(value: Self::Element) -> Storage<Self> {
        arch::f32x4(value, 0.0, 0.0, 0.0)
    }

    #[inline(always)]
    fn splat(value: Self::Element) -> Storage<Self> {
        arch::f32x4_splat(value)
    }

    fn extract<const I: usize>(value: Storage<Self>) -> Self::Element {
        arch::f32x4_extract_lane::<I>(value)
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
    fn blendv(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_laneselect(rhs, lhs, mask)
    }

    const HAS_MSB_BLENDV: bool = false;

    #[inline(always)]
    fn reverse(mut value: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_swizzle(value, arch::x4indices(3, 2, 1, 0))
    }

    #[inline(always)]
    fn unpack(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        let low = arch::i32x4_shuffle::<0, 4, 1, 5>(a, b);
        let high = arch::i32x4_shuffle::<2, 6, 3, 7>(a, b);

        (low, high)
    }

    #[inline(always)]
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
}

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

    #[inline(always)]
    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        arch::bx4_to_i32x4x(value)
    }

    #[inline(always)]
    fn all(value: Storage<Self>) -> bool {
        arch::i32x4_all_true(value)
    }

    #[inline(always)]
    fn any(value: Storage<Self>) -> bool {
        arch::v128_any_true(value)
    }

    #[inline(always)]
    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        Some(arch::i32x4_bitmask(value) as u64)
    }

    #[inline(always)]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = arch::i32x4_bitmask(value) as u32;
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

impl ShuffleRegister for F32x4Wasm {
    #[inline(always)]
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::blendv(const { arch::imm8x4_to_mask::<IMM8>() }, lhs, rhs)
    }
}

impl PermuteRegister for F32x4Wasm {
    #[inline(always)]
    fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        arch::u8x16_relaxed_swizzle(value, const { arch::imm8x4_to_indices::<IMM8>() })
    }
}

impl SwizzleRegister for F32x4Wasm {
    const HAS_PERMUTEV: bool = true;

    #[inline(always)]
    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        arch::u8x16_relaxed_swizzle(
            value,
            arch::x4indices(idxs[0] as u8, idxs[1] as u8, idxs[2] as u8, idxs[3] as u8),
        )
    }
}

#[rustfmt::skip]
impl PartialOrdRegister for F32x4Wasm {
    #[inline(always)] fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::f32x4_ge(lhs, rhs) }
    #[inline(always)] fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::f32x4_lt(lhs, rhs) }
    #[inline(always)] fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::f32x4_le(lhs, rhs) }
    #[inline(always)] fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::f32x4_ne(lhs, rhs) }
    #[inline(always)] fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::f32x4_gt(lhs, rhs) }
    #[inline(always)] fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { arch::f32x4_eq(lhs, rhs) }
}

impl NumericRegister for F32x4Wasm {
    const ZERO: Storage<Self> = arch::f32x4(0.0, 0.0, 0.0, 0.0);
    const ONE: Storage<Self> = arch::f32x4(1.0, 1.0, 1.0, 1.0);
    const TWO: Storage<Self> = arch::f32x4(2.0, 2.0, 2.0, 2.0);

    const MIN: Storage<Self> = arch::f32x4(f32::MIN, f32::MIN, f32::MIN, f32::MIN);
    const MAX: Storage<Self> = arch::f32x4(f32::MAX, f32::MAX, f32::MAX, f32::MAX);

    #[inline(always)]
    fn min_element(value: Storage<Self>) -> Self::Element {
        reduce_32x4!(f value; f32x4_relaxed_min f32x4_relaxed_min)
    }

    #[inline(always)]
    fn max_element(value: Storage<Self>) -> Self::Element {
        reduce_32x4!(f value; f32x4_relaxed_max f32x4_relaxed_max)
    }

    #[inline(always)]
    fn sum_elements(value: Storage<Self>) -> Self::Element {
        reduce_32x4!(f value; f32x4_add f32x4_add)
    }

    #[inline(always)]
    fn prod_elements(value: Storage<Self>) -> Self::Element {
        reduce_32x4!(f value; f32x4_mul f32x4_mul)
    }

    #[inline(always)]
    fn offset() -> Storage<Self> {
        arch::f32x4_splat(<Self::Lanes as Unsigned>::USIZE as f32)
    }

    #[inline(always)]
    fn indexed() -> Storage<Self> {
        arch::f32x4(0.0, 1.0, 2.0, 3.0)
    }

    #[inline(always)]
    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::f32x4_add(lhs, rhs)
    }

    #[inline(always)]
    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::f32x4_sub(lhs, rhs)
    }

    #[inline(always)]
    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::f32x4_mul(lhs, rhs)
    }

    #[inline(always)]
    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::f32x4_div(lhs, rhs)
    }

    #[inline(always)]
    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        // https://stackoverflow.com/a/26342944/2083075
        Self::nmul_adde(Self::trunc(Self::div(lhs, rhs)), rhs, lhs)
    }

    #[inline(always)]
    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::f32x4_relaxed_min(lhs, rhs)
    }

    #[inline(always)]
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::f32x4_relaxed_max(lhs, rhs)
    }
}

impl SignedRegister for F32x4Wasm {
    const NEG_ONE: Storage<Self> = arch::f32x4(-1.0, -1.0, -1.0, -1.0);
    const MIN_POSITIVE: Storage<Self> = arch::f32x4(
        f32::MIN_POSITIVE,
        f32::MIN_POSITIVE,
        f32::MIN_POSITIVE,
        f32::MIN_POSITIVE,
    );

    #[inline(always)]
    fn neg(value: Storage<Self>) -> Storage<Self> {
        arch::f32x4_neg(value)
    }

    #[inline(always)]
    fn abs(value: Storage<Self>) -> Storage<Self> {
        arch::f32x4_abs(value)
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
    fn conditional_negate(value: Storage<Self>, mask: Storage<Self::Mask>) -> Storage<Self> {
        Self::bitxor(value, Self::bitand(Self::NEG_ZERO, mask))
    }
}

impl FloatRegister for F32x4Wasm {
    // No way to know if FMA is supported at compile time
    const HAS_TRUE_FMA: bool = false;

    type Bits = super::U32x4Wasm;
    type Signed = super::I32x4Wasm;
    type ExtendedPrecision = DoublePumpRegister<super::F64x2Wasm>;

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

    // These MUST use a polyfill implementation to ensure consistent behavior across platforms

    #[inline(always)]
    fn mul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::f32x4_maddx(lhs, rhs, acc)
    }

    #[inline(always)]
    fn mul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::f32x4_maddx(lhs, rhs, arch::f32x4_neg(acc))
    }

    #[inline(always)]
    fn nmul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::f32x4_maddx(arch::f32x4_neg(lhs), rhs, acc)
    }

    #[inline(always)]
    fn nmul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::f32x4_maddx(arch::f32x4_neg(lhs), rhs, arch::f32x4_neg(acc))
    }

    // These, however, can use the native relaxed FMA instructions, which
    // may or may not be true FMA depending on the platform.

    #[inline(always)]
    fn mul_adde(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::f32x4_relaxed_madd(lhs, rhs, acc)
    }

    #[inline(always)]
    fn mul_sube(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::f32x4_relaxed_madd(lhs, rhs, arch::f32x4_neg(acc))
    }

    #[inline(always)]
    fn nmul_adde(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::f32x4_relaxed_nmadd(lhs, rhs, acc)
    }

    #[inline(always)]
    fn nmul_sube(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        arch::f32x4_relaxed_nmadd(lhs, rhs, arch::f32x4_neg(acc))
    }

    #[inline(always)]
    fn sqrt(value: Storage<Self>) -> Storage<Self> {
        arch::f32x4_sqrt(value)
    }

    const HAS_APPROX_RCP: bool = false;
    const HAS_APPROX_RSQRT: bool = false;

    #[inline(always)]
    fn floor(value: Storage<Self>) -> Storage<Self> {
        arch::f32x4_floor(value)
    }

    #[inline(always)]
    fn ceil(value: Storage<Self>) -> Storage<Self> {
        arch::f32x4_ceil(value)
    }

    #[inline(always)]
    fn round(value: Storage<Self>) -> Storage<Self> {
        arch::f32x4_nearest(value)
    }

    #[inline(always)]
    fn trunc(value: Storage<Self>) -> Storage<Self> {
        arch::f32x4_trunc(value)
    }

    const HAS_NATIVE_LDEXP: bool = false;
    const HAS_NATIVE_FREXP: bool = false;
}

impl CastRegister<F32x4Wasm> for DoublePumpRegister<super::F64x2Wasm> {
    #[inline(always)]
    fn cast_from(value: Storage<F32x4Wasm>) -> Storage<Self> {
        // promote each pair of f32 to f64
        let lo = arch::f64x2_promote_low_f32x4(value);
        let hi = arch::f64x2_promote_low_f32x4(arch::i32x4_shuffle::<2, 3, 2, 3>(value, value));

        DoublePumpRegister(lo, hi)
    }
}

impl LinAlg3Register for F32x4Wasm {
    fn min_element3(value: Storage<Self>) -> Self::Element {
        todo!()
    }

    fn max_element3(value: Storage<Self>) -> Self::Element {
        todo!()
    }

    fn sum_elements3(value: Storage<Self>) -> Self::Element {
        todo!()
    }

    fn prod_elements3(value: Storage<Self>) -> Self::Element {
        todo!()
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

impl LinAlg4Register for F32x4Wasm {
    #[inline(always)]
    fn mat4_inverse(m: &mut [Storage<Self>; 4]) -> bool {
        // dedicated WASM implementation that takes
        // advantage of `i32x4_shuffle` directly.
        impl_mat4_inverse!(m, s)
    }
}
