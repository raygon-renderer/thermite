use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitsRegister, BitshiftRegister, FloatRegister, LinAlg3Register, MaskRegister, NumericRegister,
        PartialOrdRegister, PermuteRegister, Register, ShuffleRegister, SignedRegister, Storage, SwizzleRegister,
        empty_reg, reg,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
pub struct F32x4V3;

impl Register for F32x4V3 {
    type Lanes = typenum::U4;

    type Element = f32;
    type Storage = arch::__m128;
    type HalfRegister = ();
    type DoubleRegister = super::F32x8V3;

    const ISA: InstructionSet = InstructionSet::X86V3;

    type UCOUNT = super::U32x4V3;
    type SCOUNT = super::I32x4V3;

    const EMPTY: Self::Storage = empty_reg::<Self>();

    #[inline(always)]
    fn new(value: GenericArray<f32, Self::Lanes>) -> Self::Storage {
        unsafe { arch::_mm_loadu_ps(value.as_ptr()) }
    }

    #[inline(always)]
    fn splat(value: Self::Element) -> Self::Storage {
        unsafe { arch::_mm_set1_ps(value) }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const Self::Element) -> Self::Storage {
        unsafe { arch::_mm_load_ps(ptr) }
    }

    #[inline(always)]
    unsafe fn load_unaligned(ptr: *const Self::Element) -> Self::Storage {
        unsafe { arch::_mm_loadu_ps(ptr) }
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut Self::Element, value: Self::Storage) {
        unsafe { arch::_mm_store_ps(ptr, value) }
    }

    #[inline(always)]
    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Self::Storage) {
        unsafe { arch::_mm_storeu_ps(ptr, value) }
    }

    #[inline(always)]
    fn bitxor(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_xor_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn bitand(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_and_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn bitandnot(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_andnot_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn bitor(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_or_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn not(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_xor_ps(value, arch::_mm_set1_ps(f32::from_bits(!0))) }
    }

    #[inline(always)]
    fn blendv(mask: Self::Storage, lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_blendv_ps(lhs, rhs, mask) }
    }

    const HAS_MSB_BLENDV: bool = true;

    #[inline(always)]
    fn reverse(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_permute_ps(value, 0b11_01_10_00) }
    }
}

impl ShuffleRegister for F32x4V3 {
    #[inline(always)]
    fn shuffle<const IMM8: i32>(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_shuffle_ps(lhs, rhs, IMM8) }
    }
}

impl PermuteRegister for F32x4V3 {
    #[inline(always)]
    fn permute<const IMM8: i32>(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_shuffle_ps(value, value, IMM8) }
    }
}

impl SwizzleRegister for F32x4V3 {
    #[inline(always)]
    fn permutev(value: Self::Storage, idxs: GenericArray<u32, Self::Lanes>) -> Self::Storage {
        unsafe { arch::_mm_permutevar_ps(value, core::mem::transmute(idxs)) }
    }

    #[inline(always)]
    fn swizzle(a: Self::Storage, b: Self::Storage, idxs: GenericArray<u32, Self::Lanes>) -> Self::Storage {
        unsafe {
            let idxs: arch::__m128i = core::mem::transmute(idxs);

            let four = arch::_mm_set1_epi32(4);

            // NOTE: Because of lt, this is reversed
            let blend = arch::_mm_cmplt_epi32(idxs, four);
            let a_idxs = arch::_mm_and_si128(idxs, arch::_mm_set1_epi32(0b11));
            let b_idxs = arch::_mm_sub_epi32(idxs, four);

            let tmp_a = arch::_mm_permutevar_ps(a, a_idxs);
            let tmp_b = arch::_mm_permutevar_ps(b, b_idxs);

            // NOTE: Again, reversed
            arch::_mm_blendv_ps(tmp_b, tmp_a, arch::_mm_castsi128_ps(blend))
        }
    }
}

impl MaskRegister for F32x4V3 {
    const FALSY: Self::Storage = reg::<Self, 4>([0.0; 4]);
    const TRUTHY: Self::Storage = reg::<Self, 4>([f32::from_bits(!0); 4]);

    #[inline(always)]
    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Self::Storage {
        unsafe { arch::_mm_castsi128_ps(arch::_mm_cvtboolx4_to_epi32_mask_v2(value)) }
    }

    #[inline(always)]
    fn all(value: Self::Storage) -> bool {
        unsafe { arch::_mm_movemask_ps(value) == 0b1111 }
    }

    #[inline(always)]
    fn any(value: Self::Storage) -> bool {
        unsafe { arch::_mm_movemask_ps(value) != 0 }
    }

    #[inline(always)]
    fn none(value: Self::Storage) -> bool {
        unsafe { arch::_mm_movemask_ps(value) == 0 }
    }
}

impl PartialOrdRegister for F32x4V3 {
    #[inline(always)]
    fn lt(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_cmplt_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn le(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_cmple_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn gt(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_cmpgt_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn ge(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_cmpge_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn eq(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_cmpeq_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn ne(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_cmpneq_ps(lhs, rhs) }
    }
}

impl NumericRegister for F32x4V3 {
    const ZERO: Self::Storage = reg::<Self, 4>([0.0; 4]);
    const ONE: Self::Storage = reg::<Self, 4>([1.0; 4]);
    const TWO: Self::Storage = reg::<Self, 4>([2.0; 4]);

    const MIN: Self::Storage = reg::<Self, 4>([f32::MIN; 4]);
    const MAX: Self::Storage = reg::<Self, 4>([f32::MAX; 4]);

    #[inline(always)]
    fn min_element(value: Self::Storage) -> Self::Element {
        _mm_reduce_ps_v2!(value; _mm_min_ps _mm_min_ss)
    }

    #[inline(always)]
    fn max_element(value: Self::Storage) -> Self::Element {
        _mm_reduce_ps_v2!(value; _mm_max_ps _mm_max_ss)
    }

    #[inline(always)]
    fn sum_elements(value: Self::Storage) -> Self::Element {
        _mm_reduce_ps_v2!(value; _mm_add_ps _mm_add_ss)
    }

    #[inline(always)]
    fn prod_elements(value: Self::Storage) -> Self::Element {
        _mm_reduce_ps_v2!(value; _mm_mul_ps _mm_mul_ss)
    }

    #[inline(always)]
    fn offset() -> Self::Storage {
        Self::splat(<Self::Lanes as Unsigned>::USIZE as f32)
    }

    #[inline(always)]
    fn indexed() -> Self::Storage {
        Self::new(GenericArray::generate(|i| i as f32))
    }

    #[inline(always)]
    fn add(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_add_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn sub(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_sub_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn mul(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_mul_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn div(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_div_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn rem(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        // https://stackoverflow.com/a/26342944/2083075
        Self::nmul_adde(Self::trunc(Self::div(lhs, rhs)), rhs, lhs)
    }

    #[inline(always)]
    fn min(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_min_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn max(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_max_ps(lhs, rhs) }
    }
}

impl SignedRegister for F32x4V3 {
    const NEG_ONE: Self::Storage = reg::<Self, 4>([-1.0; 4]);
    const MIN_POSITIVE: Self::Storage = reg::<Self, 4>([f32::MIN_POSITIVE; 4]);

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

impl FloatRegister for F32x4V3 {
    const HAS_TRUE_FMA: bool = true;

    type Bits = super::U32x4V3;
    type Signed = super::I32x4V3;
    type ExtendedPrecision = super::F64x4V3;

    const HALF: Self::Storage = reg::<Self, 4>([0.5; 4]);
    const NEG_ZERO: Self::Storage = reg::<Self, 4>([-0.0; 4]);
    const EPSILON: Self::Storage = reg::<Self, 4>([f32::EPSILON; 4]);
    const INFINITY: Self::Storage = reg::<Self, 4>([f32::INFINITY; 4]);
    const NEG_INFINITY: Self::Storage = reg::<Self, 4>([f32::NEG_INFINITY; 4]);
    const NAN: Self::Storage = reg::<Self, 4>([f32::NAN; 4]);

    const EXP_MASK: Storage<Self::Bits> = reg::<Self::Bits, 4>([0x7F800000; 4]);

    #[inline(always)]
    fn mul_add(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_fmadd_ps(lhs, rhs, acc) }
    }

    #[inline(always)]
    fn mul_sub(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_fmsub_ps(lhs, rhs, acc) }
    }

    #[inline(always)]
    fn nmul_add(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_fnmadd_ps(lhs, rhs, acc) }
    }

    #[inline(always)]
    fn nmul_sub(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_fnmsub_ps(lhs, rhs, acc) }
    }

    // #[inline(always)]
    // fn mul_adde(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
    //     Self::mul_add(lhs, rhs, acc)
    // }

    // #[inline(always)]
    // fn mul_sube(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
    //     Self::mul_sub(lhs, rhs, acc)
    // }

    // #[inline(always)]
    // fn nmul_adde(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
    //     Self::nmul_add(lhs, rhs, acc)
    // }

    // #[inline(always)]
    // fn nmul_sube(lhs: Self::Storage, rhs: Self::Storage, acc: Self::Storage) -> Self::Storage {
    //     Self::nmul_sub(lhs, rhs, acc)
    // }

    #[inline(always)]
    fn sqrt(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_sqrt_ps(value) }
    }

    #[inline(always)]
    fn rsqrt(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_rsqrt_ps(value) }
    }

    #[inline(always)]
    fn rcp(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_rcp_ps(value) }
    }

    const HAS_APPROX_RSQRT: bool = true;
    const HAS_APPROX_RCP: bool = true;

    #[inline(always)]
    fn floor(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_floor_ps(value) }
    }

    #[inline(always)]
    fn ceil(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_ceil_ps(value) }
    }

    #[inline(always)]
    fn round(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_round_ps(value, arch::_MM_FROUND_TO_NEAREST_INT | arch::_MM_FROUND_NO_EXC) }
    }

    #[inline(always)]
    fn trunc(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_round_ps(value, arch::_MM_FROUND_TO_ZERO | arch::_MM_FROUND_NO_EXC) }
    }

    #[inline(always)]
    fn next_up(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_nextupps_v2(value) }
    }

    #[inline(always)]
    fn next_down(value: Self::Storage) -> Self::Storage {
        unsafe { arch::_mm_nextdownps_v2(value) }
    }
}

// Just use the SSE4.1 implementation
impl LinAlg3Register for F32x4V3 {
    #[inline(always)]
    fn dot3(lhs: Self::Storage, rhs: Self::Storage) -> f32 {
        unsafe { arch::dot3_v2(lhs, rhs) }
    }

    #[inline(always)]
    fn cross3(lhs: Self::Storage, rhs: Self::Storage) -> Self::Storage {
        unsafe { arch::cross3_v2(lhs, rhs) }
    }

    #[inline(always)]
    fn zero4(value: Self::Storage) -> Self::Storage {
        unsafe { arch::zero4_v2(value) }
    }

    #[inline(always)]
    fn one4(value: Self::Storage) -> Self::Storage {
        unsafe { arch::one4_v2(value) }
    }

    #[inline(always)]
    fn min_element3(value: Self::Storage) -> Self::Element {
        _mm_reduce_ps3_v1!(value; _mm_min_ss)
    }

    #[inline(always)]
    fn max_element3(value: Self::Storage) -> Self::Element {
        _mm_reduce_ps3_v1!(value; _mm_max_ss)
    }

    #[inline(always)]
    fn sum_elements3(value: Self::Storage) -> Self::Element {
        _mm_reduce_ps3_v1!(value; _mm_add_ss)
    }

    #[inline(always)]
    fn prod_elements3(value: Self::Storage) -> Self::Element {
        _mm_reduce_ps3_v1!(value; _mm_mul_ss)
    }
}
