use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    backend::scalar::Scalar,
    isa::InstructionSet,
    register::{
        BitsRegister, BitshiftRegister, BlendRegister, CastRegister, CoreRegister, FloatRegister, LinAlg3Register,
        LinAlg4Register, NumericRegister, PartialOrdRegister, PermuteRegister, Register, ShuffleRegister,
        SignedRegister, Storage, SwizzleRegister, dp::DoublePumpRegister, empty_reg, reg,
    },
    simd::Simd,
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct F32x4V2;

impl CoreRegister for F32x4V2 {
    type Lanes = typenum::U4;
    type Element = f32;
    type Storage = arch::__m128;
}

impl Register for F32x4V2 {
    type HalfRegister = <Scalar as Simd>::f32x2;
    type DoubleRegister = DoublePumpRegister<Self>;

    const IS_EMULATED: bool = false;

    const ISA: InstructionSet = InstructionSet::X86V2;

    type USize = super::U32x4V2;
    type ISize = super::I32x4V2;

    const EMPTY: Storage<Self> = empty_reg::<Self>();

    #[inline(always)]
    fn new(value: GenericArray<f32, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_loadu_ps(value.as_ptr()) }
    }

    #[inline(always)]
    fn single(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_set_ss(value) }
    }

    #[inline(always)]
    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_set1_ps(value) }
    }

    #[inline(always)]
    fn split(value: Storage<Self>) -> (Storage<Self::HalfRegister>, Storage<Self::HalfRegister>)
    where
        Self::HalfRegister: Register,
    {
        unsafe {
            let mut arr = [0f32; 4];
            Self::store_unaligned(arr.as_mut_ptr(), value);
            (DoublePumpRegister(arr[0], arr[1]), DoublePumpRegister(arr[2], arr[3]))
        }
    }

    #[inline(always)]
    fn join(lo: Storage<Self::HalfRegister>, hi: Storage<Self::HalfRegister>) -> Storage<Self>
    where
        Self::HalfRegister: Register,
    {
        unsafe { arch::_mm_setr_ps(lo.0, lo.1, hi.0, hi.1) }
    }

    #[inline(always)]
    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_load_ps(ptr) }
    }

    #[inline(always)]
    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_loadu_ps(ptr) }
    }

    #[inline(always)]
    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_store_ps(ptr, value) }
    }

    #[inline(always)]
    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_storeu_ps(ptr, value) }
    }

    #[inline(always)]
    unsafe fn load_stream(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_castsi128_ps(arch::_mm_stream_load_si128(ptr as _)) }
    }

    #[inline(always)]
    unsafe fn store_stream(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm_stream_ps(ptr, value) }
    }

    const FALSY: Storage<Self> = reg::<Self, 4>([0.0; 4]);
    const TRUTHY: Storage<Self> = reg::<Self, 4>([f32::from_bits(!0); 4]);

    #[inline(always)]
    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_castsi128_ps(arch::_mm_cvtboolx4_to_epi32_mask_v2(value)) }
    }

    #[inline(always)]
    fn all(value: Storage<Self>) -> bool {
        unsafe { arch::_mm_movemask_ps(value) == 0b1111 }
    }

    #[inline(always)]
    fn any(value: Storage<Self>) -> bool {
        unsafe { arch::_mm_movemask_ps(value) != 0 }
    }

    #[inline(always)]
    fn none(value: Storage<Self>) -> bool {
        unsafe { arch::_mm_movemask_ps(value) == 0 }
    }

    #[inline(always)]
    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        Some(unsafe { arch::_mm_movemask_ps(value) as u64 })
    }

    #[inline(always)]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = unsafe { arch::_mm_movemask_ps(value) as u32 };
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }

    #[inline(always)]
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_xor_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_and_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_andnot_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_or_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn not(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_xor_ps(value, arch::_mm_set1_ps(f32::from_bits(!0))) }
    }

    #[inline(always)]
    fn blendv(mask: Storage<Self>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_blendv_ps(lhs, rhs, mask) }
    }

    const HAS_MSB_BLENDV: bool = true;

    #[inline(always)]
    fn reverse(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_shuffle_ps(value, value, 0b11_01_10_00) }
    }

    const HAS_SIMPLE_UNPACK: bool = true;

    #[inline(always)]
    fn unpack(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe { (arch::_mm_unpacklo_ps(a, b), arch::_mm_unpackhi_ps(a, b)) }
    }

    #[inline(always)]
    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bswap_psx_v2(value) }
    }
}

impl ShuffleRegister for F32x4V2 {
    #[inline(always)]
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_shuffle_ps(lhs, rhs, IMM8) }
    }
}

impl PermuteRegister for F32x4V2 {
    #[inline(always)]
    fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_shuffle_ps(value, value, IMM8) }
    }
}

impl BlendRegister for F32x4V2 {
    #[inline(always)]
    fn blend<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_blend_ps::<IMM8>(lhs, rhs) }
    }
}

impl SwizzleRegister for F32x4V2 {
    const HAS_PERMUTEV: bool = true;

    #[inline(always)]
    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_permutevar_ps_v2(value, core::mem::transmute(idxs)) }
    }

    #[inline(always)]
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

impl PartialOrdRegister for F32x4V2 {
    #[inline(always)]
    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmplt_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmple_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpgt_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpge_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpeq_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_cmpneq_ps(lhs, rhs) }
    }
}

impl NumericRegister for F32x4V2 {
    const ZERO: Storage<Self> = reg::<Self, 4>([0.0; 4]);
    const ONE: Storage<Self> = reg::<Self, 4>([1.0; 4]);
    const TWO: Storage<Self> = reg::<Self, 4>([2.0; 4]);

    const MIN: Storage<Self> = reg::<Self, 4>([f32::MIN; 4]);
    const MAX: Storage<Self> = reg::<Self, 4>([f32::MAX; 4]);

    #[inline(always)]
    fn min_element(value: Storage<Self>) -> Self::Element {
        _mm_reduce_ps_v2!(value; _mm_min_ps _mm_min_ss)
    }

    #[inline(always)]
    fn max_element(value: Storage<Self>) -> Self::Element {
        _mm_reduce_ps_v2!(value; _mm_max_ps _mm_max_ss)
    }

    #[inline(always)]
    fn sum_elements(value: Storage<Self>) -> Self::Element {
        _mm_reduce_ps_v2!(value; _mm_add_ps _mm_add_ss)
    }

    #[inline(always)]
    fn prod_elements(value: Storage<Self>) -> Self::Element {
        _mm_reduce_ps_v2!(value; _mm_mul_ps _mm_mul_ss)
    }

    #[inline(always)]
    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as Unsigned>::USIZE as f32)
    }

    #[inline(always)]
    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as f32))
    }

    #[inline(always)]
    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_add_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sub_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_mul_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_div_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        // https://stackoverflow.com/a/26342944/2083075
        Self::nmul_adde(Self::trunc(Self::div(lhs, rhs)), rhs, lhs)
    }

    #[inline(always)]
    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_min_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_max_ps(lhs, rhs) }
    }

    #[inline(always)]
    fn sort(value: Storage<Self>) -> Storage<Self> {
        arch::sort_4::<Self>(value)
    }
}

impl SignedRegister for F32x4V2 {
    const NEG_ONE: Storage<Self> = reg::<Self, 4>([-1.0; 4]);
    const MIN_POSITIVE: Storage<Self> = reg::<Self, 4>([f32::MIN_POSITIVE; 4]);

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

impl FloatRegister for F32x4V2 {
    const HAS_TRUE_FMA: bool = false;

    type Bits = super::U32x4V2;
    type Signed = super::I32x4V2;
    type ExtendedPrecision = DoublePumpRegister<super::F64x2V2>;

    const HALF: Storage<Self> = reg::<Self, 4>([0.5; 4]);
    const NEG_ZERO: Storage<Self> = reg::<Self, 4>([-0.0; 4]);
    const EPSILON: Storage<Self> = reg::<Self, 4>([f32::EPSILON; 4]);
    const INFINITY: Storage<Self> = reg::<Self, 4>([f32::INFINITY; 4]);
    const NEG_INFINITY: Storage<Self> = reg::<Self, 4>([f32::NEG_INFINITY; 4]);
    const NAN: Storage<Self> = reg::<Self, 4>([f32::NAN; 4]);

    const EXP_MASK: Storage<Self::Bits> = reg::<Self::Bits, 4>([0x7F800000; 4]);

    #[cfg(not(feature = "disable_fma_emulation"))]
    #[inline(always)]
    fn mul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_fmadd_psx_v1(lhs, rhs, acc) }
    }

    #[cfg(not(feature = "disable_fma_emulation"))]
    #[inline(always)]
    fn mul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_fmadd_psx_v1(lhs, rhs, Self::neg(acc)) }
    }

    #[cfg(not(feature = "disable_fma_emulation"))]
    #[inline(always)]
    fn nmul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_fmadd_psx_v1(Self::neg(lhs), rhs, acc) }
    }

    #[cfg(not(feature = "disable_fma_emulation"))]
    #[inline(always)]
    fn nmul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_fmadd_psx_v1(Self::neg(lhs), rhs, Self::neg(acc)) }
    }

    #[inline(always)]
    fn sqrt(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_sqrt_ps(value) }
    }

    #[inline(always)]
    fn rsqrt(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_rsqrt_ps(value) }
    }

    #[inline(always)]
    fn rcp(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_rcp_ps(value) }
    }

    const HAS_APPROX_RSQRT: bool = true;
    const HAS_APPROX_RCP: bool = true;

    #[inline(always)]
    fn floor(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_floor_ps(value) }
    }

    #[inline(always)]
    fn ceil(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_ceil_ps(value) }
    }

    #[inline(always)]
    fn round(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_round_ps(value, arch::_MM_FROUND_TO_NEAREST_INT | arch::_MM_FROUND_NO_EXC) }
    }

    #[inline(always)]
    fn trunc(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_round_ps(value, arch::_MM_FROUND_TO_ZERO | arch::_MM_FROUND_NO_EXC) }
    }

    const HAS_NATIVE_LDEXP: bool = false;
    const HAS_NATIVE_FREXP: bool = false;
}

macro_rules! s {
    ($ty:ty: $v:expr, [$a:literal, $b:literal, $c:literal, $d:literal]) => {
        unsafe { arch::_mm_shuffle_ps::<{ MM_SHUFFLE!($a, $b, $c, $d) }>($v, $v) }
    };
    ($ty:ty: $v1:expr, $v2:expr, [$a:literal, $b:literal, $c:literal, $d:literal]) => {
        unsafe { arch::_mm_shuffle_ps::<{ MM_SHUFFLE!($a, $b, $c, $d) }>($v1, $v2) }
    };
}

impl LinAlg4Register for F32x4V2 {
    #[inline(always)]
    fn mat4_inverse(m: &mut [Storage<Self>; 4]) -> bool {
        // dedicated x86-v2/v1 implementation that takes
        // advantage of `_mm_shuffle_ps` directly.
        impl_mat4_inverse!(m, s)
    }
}

impl LinAlg3Register for F32x4V2 {
    #[inline(always)]
    fn dot3(lhs: Storage<Self>, rhs: Storage<Self>) -> f32 {
        unsafe { arch::dot3_v1(lhs, rhs) }
    }

    // #[inline(always)]
    // fn cross3(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
    //     unsafe { arch::cross3_v1(lhs, rhs) }
    // }

    #[inline(always)]
    fn zero4(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::zero4_v2(value) }
    }

    #[inline(always)]
    fn one4(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::one4_v2(value) }
    }

    #[inline(always)]
    fn min_element3(value: Storage<Self>) -> Self::Element {
        _mm_reduce_ps3_v1!(value; _mm_min_ss)
    }

    #[inline(always)]
    fn max_element3(value: Storage<Self>) -> Self::Element {
        _mm_reduce_ps3_v1!(value; _mm_max_ss)
    }

    #[inline(always)]
    fn sum_elements3(value: Storage<Self>) -> Self::Element {
        _mm_reduce_ps3_v1!(value; _mm_add_ss)
    }

    #[inline(always)]
    fn prod_elements3(value: Storage<Self>) -> Self::Element {
        _mm_reduce_ps3_v1!(value; _mm_mul_ss)
    }
}

impl CastRegister<F32x4V2> for DoublePumpRegister<super::F64x2V2> {
    #[inline(always)]
    fn cast_from(value: Storage<F32x4V2>) -> Storage<Self> {
        unsafe {
            let lo = arch::_mm_cvtps_pd(value);
            let hi = arch::_mm_cvtps_pd(arch::_mm_movehl_ps(value, value));

            DoublePumpRegister::join(lo, hi)
        }
    }
}
