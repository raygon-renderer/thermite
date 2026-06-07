use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    backend::scalar::Scalar,
    isa::InstructionSet,
    register::{
        BitCastRegister, BitshiftRegister, BitwiseRegister, BlendRegister, CastRegister, ConcatRegister, CoreRegister,
        Element, ExtendRegister, FloatRegister, InterleaveRegister, LinAlg3Register, LinAlg4Register, MaskElement,
        MaskRegister, NativeCapability, NumericRegister, PartialOrdRegister, PermuteRegister, Register,
        ShuffleRegister, SignedRegister, Storage, SwizzleRegister, ZeroUpper, array::ArrayRegister, empty_reg, reg,
    },
    simd::Simd,
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct F32x4V2;

#[thermite_macros::inline_always]
impl CoreRegister for F32x4V2 {
    type Lanes = typenum::U4;
    type Storage = arch::__m128;
    type Mask = Self;

    const IS_EMULATED: bool = false;
    const ISA: InstructionSet = InstructionSet::X86V2;
    const EMPTY: Storage<Self> = empty_reg::<Self>();
    const HAS_EQUAL_SIZE_MASK: bool = true;

    fn blendv(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_blendv_ps(lhs, rhs, mask) }
    }

    fn zz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_and_ps(value, mask) }
    }

    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_andnot_ps(mask, value) }
    }

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        if const { Z::N >= 4 } {
            value
        } else if const { Z::N == 2 } {
            unsafe { arch::_mm_castsi128_ps(arch::_mm_move_epi64(arch::_mm_castps_si128(value))) }
        } else {
            unsafe { arch::_mm_and_ps(value, arch::_mm_castsi128_ps(arch::_mm_zeroupper_mask_epi32::<Z>())) }
        }
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        mask
    }
}

#[thermite_macros::inline_always]
impl MaskRegister for F32x4V2 {
    const FALSY: Storage<Self> = reg::<Self, 4>([0.0; 4]);
    const TRUTHY: Storage<Self> = reg::<Self, 4>([f32::from_bits(!0); 4]);

    fn set(mut mask: Storage<Self::Mask>, lane: usize, value: bool) -> Storage<Self> {
        Self::as_array_mut(&mut mask)[lane] = if value { MaskElement::TRUTHY } else { MaskElement::FALSY };
        mask
    }

    fn test(mask: Storage<Self::Mask>, lane: usize) -> bool {
        Self::as_array(&mask)[lane].to_bool()
    }

    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_castsi128_ps(arch::_mm_cvtboolx4_to_epi32_mask_v2(value)) }
    }

    fn all(value: Storage<Self>) -> bool {
        unsafe { arch::_mm_movemask_ps(value) == 0b1111 }
    }

    fn any(value: Storage<Self>) -> bool {
        unsafe { arch::_mm_movemask_ps(value) != 0 }
    }

    fn none(value: Storage<Self>) -> bool {
        unsafe { arch::_mm_movemask_ps(value) == 0 }
    }

    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        Some(unsafe { arch::_mm_movemask_ps(value) as u64 })
    }

    #[cfg(feature = "bitvec")]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = unsafe { arch::_mm_movemask_ps(value) as u32 };
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for F32x4V2 {
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_xor_ps(lhs, rhs) }
    }

    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_and_ps(lhs, rhs) }
    }

    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_andnot_ps(lhs, rhs) }
    }

    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_or_ps(lhs, rhs) }
    }

    fn not(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_xor_ps(value, arch::_mm_set1_ps(f32::from_bits(!0))) }
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<f32> for F32x4V2 {
    fn extend(value: Storage<f32>) -> Storage<Self> {
        unsafe { arch::_mm_setr_ps(value, 0.0, 0.0, 0.0) }
    }

    fn narrow(value: Storage<Self>) -> Storage<f32> {
        unsafe { arch::_mm_cvtss_f32(value) }
    }
}

#[thermite_macros::inline_always]
impl InterleaveRegister for F32x4V2 {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe { (arch::_mm_unpacklo_ps(a, b), arch::_mm_unpackhi_ps(a, b)) }
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe {
            let lo = arch::_mm_shuffle_ps(a, b, 0x88);
            let hi = arch::_mm_shuffle_ps(a, b, 0xDD);

            (lo, hi)
        }
    }
}

#[thermite_macros::inline_always]
impl Register for F32x4V2 {
    type Element = f32;

    type Signed = super::I32x4V2;
    type Unsigned = super::U32x4V2;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe {
            // value != 0.0
            arch::_mm_castsi128_ps(arch::_mm_xor_si128(
                arch::_mm_set1_epi8(-1),
                arch::_mm_cmpeq_epi32(arch::_mm_castps_si128(value), arch::_mm_setzero_si128()),
            ))
        }
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        value
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        value // floats support msb masks directly
    }

    fn new(value: GenericArray<f32, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_loadu_ps(value.as_ptr()) }
    }

    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_set_ss(value) }
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_set1_ps(value) }
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_load_ps(ptr) }
    }

    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_loadu_ps(ptr) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_store_ps(ptr, value) }
    }

    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_storeu_ps(ptr, value) }
    }

    unsafe fn load_stream(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_castsi128_ps(arch::_mm_stream_load_si128(ptr as _)) }
    }

    unsafe fn store_stream(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_stream_ps(ptr, value) }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_shuffle_ps::<{ MM_SHUFFLE!(0, 1, 2, 3) }>(value, value) }
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bswap_psx_v2(value) }
    }
}

#[thermite_macros::inline_always]
impl ShuffleRegister for F32x4V2 {
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_shuffle_ps(lhs, rhs, IMM8) }
    }
}

#[thermite_macros::inline_always]
impl PermuteRegister for F32x4V2 {
    fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_shuffle_ps(value, value, IMM8) }
    }
}

#[thermite_macros::inline_always]
impl BlendRegister for F32x4V2 {
    fn blend<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_blend_ps::<IMM8>(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl SwizzleRegister for F32x4V2 {
    const HAS_PERMUTEV: bool = true;

    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_permutevar_ps_v2(value, core::mem::transmute(idxs)) }
    }

    fn swizzle(a: Storage<Self>, b: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        unsafe {
            let idxs: arch::__m128i = core::mem::transmute(idxs);

            let four = arch::_mm_set1_epi32(4);

            // NOTE: Because of lt, this is reversed
            let blend = arch::_mm_cmplt_epi32(idxs, four);
            let a_idxs = arch::_mm_and_si128(idxs, arch::_mm_set1_epi32(0b11));
            let b_idxs = arch::_mm_sub_epi32(idxs, four);

            let tmp_a = arch::_mm_permutevar_ps_v2(a, a_idxs);
            let tmp_b = arch::_mm_permutevar_ps_v2(b, b_idxs);

            // NOTE: Again, reversed
            arch::_mm_blendv_ps(tmp_b, tmp_a, arch::_mm_castsi128_ps(blend))
        }
    }
}

#[thermite_macros::inline_always]
impl PartialOrdRegister for F32x4V2 {
    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmplt_ps(lhs, rhs) }
    }

    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmple_ps(lhs, rhs) }
    }

    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpgt_ps(lhs, rhs) }
    }

    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpge_ps(lhs, rhs) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpeq_ps(lhs, rhs) }
    }

    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpneq_ps(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl NumericRegister for F32x4V2 {
    const ZERO: Storage<Self> = reg::<Self, 4>([0.0; 4]);
    const ONE: Storage<Self> = reg::<Self, 4>([1.0; 4]);
    const TWO: Storage<Self> = reg::<Self, 4>([2.0; 4]);

    const MIN: Storage<Self> = reg::<Self, 4>([f32::MIN; 4]);
    const MAX: Storage<Self> = reg::<Self, 4>([f32::MAX; 4]);

    fn min_element(value: Storage<Self>) -> Self::Element {
        _mm_reduce_ps_v2!(value; _mm_min_ps _mm_min_ss)
    }

    fn max_element(value: Storage<Self>) -> Self::Element {
        _mm_reduce_ps_v2!(value; _mm_max_ps _mm_max_ss)
    }

    fn min_max_element(value: Storage<Self>) -> (Self::Element, Self::Element) {
        _mm_reduce2_ps_v2!(value; _mm_min_ps _mm_min_ss, _mm_max_ps _mm_max_ss)
    }

    fn sum_elements(value: Storage<Self>) -> Self::Element {
        _mm_reduce_ps_v2!(value; _mm_add_ps _mm_add_ss)
    }

    fn prod_elements(value: Storage<Self>) -> Self::Element {
        _mm_reduce_ps_v2!(value; _mm_mul_ps _mm_mul_ss)
    }

    fn pairwise_sum(lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_hadd_ps(lo, hi) }
    }

    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as Unsigned>::USIZE as f32)
    }

    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as f32))
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_add_ps(lhs, rhs) }
    }

    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sub_ps(lhs, rhs) }
    }

    fn add_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_add_ps(lhs, arch::_mm_and_ps(rhs, mask)) }
    }

    fn sub_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sub_ps(lhs, arch::_mm_and_ps(rhs, mask)) }
    }

    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mul_ps(lhs, rhs) }
    }

    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_div_ps(lhs, rhs) }
    }

    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        // https://stackoverflow.com/a/26342944/2083075
        Self::nmul_adde(Self::trunc(Self::div(lhs, rhs)), rhs, lhs)
    }

    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::fix_min::<Self>(lhs, rhs, unsafe { arch::_mm_min_ps(lhs, rhs) })
    }

    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        arch::fix_max::<Self>(lhs, rhs, unsafe { arch::_mm_max_ps(lhs, rhs) })
    }

    fn sort(value: Storage<Self>) -> Storage<Self> {
        arch::sort_4::<Self>(value)
    }
}

#[thermite_macros::inline_always]
impl SignedRegister for F32x4V2 {
    const NEG_ONE: Storage<Self> = reg::<Self, 4>([-1.0; 4]);
    const MIN_POSITIVE: Storage<Self> = reg::<Self, 4>([f32::MIN_POSITIVE; 4]);

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
impl FloatRegister for F32x4V2 {
    const HAS_TRUE_FMA: bool = false;

    type Bits = super::U32x4V2;
    type SignedBits = super::I32x4V2;
    type ExtendedPrecision = ArrayRegister<super::F64x2V2, 2>;

    const HALF: Storage<Self> = reg::<Self, 4>([0.5; 4]);
    const NEG_ZERO: Storage<Self> = reg::<Self, 4>([-0.0; 4]);
    const EPSILON: Storage<Self> = reg::<Self, 4>([f32::EPSILON; 4]);
    const INFINITY: Storage<Self> = reg::<Self, 4>([f32::INFINITY; 4]);
    const NEG_INFINITY: Storage<Self> = reg::<Self, 4>([f32::NEG_INFINITY; 4]);
    const NAN: Storage<Self> = reg::<Self, 4>([f32::NAN; 4]);

    const EXP_MASK: Storage<Self::Bits> = reg::<Self::Bits, 4>([0x7F800000; 4]);

    #[cfg(not(feature = "disable_fast_fma"))]
    fn mul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_fmadd_psx_v1(lhs, rhs, acc) }
    }

    #[cfg(not(feature = "disable_fast_fma"))]
    fn mul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_fmadd_psx_v1(lhs, rhs, Self::neg(acc)) }
    }

    #[cfg(not(feature = "disable_fast_fma"))]
    fn nmul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_fmadd_psx_v1(Self::neg(lhs), rhs, acc) }
    }

    #[cfg(not(feature = "disable_fast_fma"))]
    fn nmul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_fmadd_psx_v1(Self::neg(lhs), rhs, Self::neg(acc)) }
    }

    fn sqrt(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sqrt_ps(value) }
    }

    fn rsqrt(value: Storage<Self>) -> Storage<Self> {
        cfg_select! {
            feature = "strict_ieee754" => {
                Self::rcp(Self::sqrt(value))
            }
            _ => unsafe { arch::_mm_rsqrt_ps(value) }
        }
    }

    fn rcp(value: Storage<Self>) -> Storage<Self> {
        cfg_select! {
            feature = "strict_ieee754" => {
                Self::div(Self::ONE, value)
            }
            _ => unsafe { arch::_mm_rcp_ps(value) }
        }
    }

    const HAS_APPROX_RSQRT: bool = cfg!(not(feature = "strict_ieee754"));
    const HAS_APPROX_RCP: bool = cfg!(not(feature = "strict_ieee754"));

    fn floor(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_floor_ps(value) }
    }

    fn ceil(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_ceil_ps(value) }
    }

    fn round(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_round_ps(value, arch::_MM_FROUND_TO_NEAREST_INT | arch::_MM_FROUND_NO_EXC) }
    }

    fn trunc(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_round_ps(value, arch::_MM_FROUND_TO_ZERO | arch::_MM_FROUND_NO_EXC) }
    }

    const NATIVE_CAP: NativeCapability = NativeCapability::NONE;
}

macro_rules! s {
    // Indices are in lane order (`out[i] = src(idx[i])`), matching `swizzle_const`.
    // `MM_SHUFFLE_R!` packs lane 0 into the low bits, which is what `_mm_shuffle_ps`
    // reads first - using the conventional (high-first) `MM_SHUFFLE!` here reverses
    // the lanes and corrupts asymmetric patterns.
    ($ty:ty: $v:expr, [$a:literal, $b:literal, $c:literal, $d:literal]) => {
        unsafe { arch::_mm_shuffle_ps::<{ MM_SHUFFLE_R!($a, $b, $c, $d) }>($v, $v) }
    };
    // Two-input form: `_mm_shuffle_ps` always takes lanes 0,1 from `$v1` and 2,3
    // from `$v2`, so callers must use the `[lo from v1, hi from v2]` split (as
    // `impl_mat4_inverse!` does). The `& 3` maps the `4..=7` v2 indices into v2's lanes.
    ($ty:ty: $v1:expr, $v2:expr, [$a:literal, $b:literal, $c:literal, $d:literal]) => {
        unsafe { arch::_mm_shuffle_ps::<{ MM_SHUFFLE_R!($a, $b, $c & 3, $d & 3) }>($v1, $v2) }
    };
}

#[thermite_macros::inline_always]
impl LinAlg4Register for F32x4V2 {
    fn mat4_inverse<const DET_ONLY: bool>(m: &mut [Storage<Self>; 4], det: &mut Self::Element) -> bool {
        // dedicated x86-v2/v1 implementation that takes
        // advantage of `_mm_shuffle_ps` directly.
        impl_mat4_inverse!(m, det, s, DET_ONLY)
    }
}

#[thermite_macros::inline_always]
impl LinAlg3Register for F32x4V2 {
    fn dot3(lhs: Storage<Self>, rhs: Storage<Self>) -> f32 {
        unsafe { arch::dot3_v1(lhs, rhs) }
    }

    //
    // fn cross3(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
    //     unsafe { arch::cross3_v1(lhs, rhs) }
    // }

    fn zero4(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::zero4_v2(value) }
    }

    fn one4(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::one4_v2(value) }
    }

    fn min_element3(value: Storage<Self>) -> Self::Element {
        _mm_reduce_ps3_v1!(value; _mm_min_ss)
    }

    fn max_element3(value: Storage<Self>) -> Self::Element {
        _mm_reduce_ps3_v1!(value; _mm_max_ss)
    }

    fn sum_elements3(value: Storage<Self>) -> Self::Element {
        _mm_reduce_ps3_v1!(value; _mm_add_ss)
    }

    fn prod_elements3(value: Storage<Self>) -> Self::Element {
        _mm_reduce_ps3_v1!(value; _mm_mul_ss)
    }
}

#[thermite_macros::inline_always]
impl CastRegister<F32x4V2> for ArrayRegister<super::F64x2V2, 2> {
    fn cast_from(value: Storage<F32x4V2>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm_cvtps_pd(value);
            let hi = arch::_mm_cvtps_pd(arch::_mm_movehl_ps(value, value));

            ArrayRegister([lo, hi])
        }
    }
}
