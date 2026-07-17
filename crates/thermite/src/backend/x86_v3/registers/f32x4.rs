use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    backend::scalar::Scalar,
    isa::InstructionSet,
    register::{
        BitCastRegister, BitshiftRegister, BitwiseRegister, BlendRegister, ConcatRegister, CoreRegister, Element,
        FloatRegister, IndexableRegister, InterleaveRegister, LinAlg3Register, LinAlg4Register, MaskElement,
        MaskRegister, NativeCapability, NumericRegister, PartialOrdRegister, PermuteRegister, Register,
        ShuffleRegister, SignedRegister, Storage, WideRegister, ZeroUpper, empty_reg, reg,
    },
    simd::Simd,
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct F32x4V3;

#[thermite_macros::inline_always]
impl CoreRegister for F32x4V3 {
    type Lanes = typenum::U4;
    type Storage = arch::__m128;
    type Mask = Self;

    const IS_EMULATED: bool = false;

    const ISA: InstructionSet = InstructionSet::X86V3;

    const EMPTY: Storage<Self> = empty_reg::<Self>();

    const HAS_EQUAL_SIZE_MASK: bool = true;

    fn blendv(mask: Storage<Self::Mask>, on_false: Storage<Self>, on_true: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_blendv_ps(on_false, on_true, mask) }
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
impl MaskRegister for F32x4V3 {
    const FALSY: Storage<Self> = reg::<Self, 4>([0.0; 4]);
    const TRUTHY: Storage<Self> = reg::<Self, 4>([f32::from_bits(!0); 4]);

    fn set(mut mask: Storage<Self::Mask>, lane: usize, value: bool) -> Storage<Self> {
        Self::as_mut_slice(&mut mask)[lane] = if value { MaskElement::TRUTHY } else { MaskElement::FALSY };
        mask
    }

    fn test(mask: Storage<Self::Mask>, lane: usize) -> bool {
        Self::as_slice(&mask)[lane].to_bool()
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
impl BitwiseRegister for F32x4V3 {
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
impl WideRegister for F32x4V3 {
    type Wide = super::F32x8V3;
}

#[thermite_macros::inline_always]
impl Register for F32x4V3 {
    type Element = f32;

    type Signed = super::I32x4V3;
    type Unsigned = super::U32x4V3;

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

    impl_native_radix3!(arch::_mm_interleave3_ps, arch::_mm_deinterleave3_ps);

    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_set1_ps(value) }
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_load_ps(ptr) }
    }

    unsafe fn load_m(src: Storage<Self>, mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        // use load_z + 2 bitwise ops to emulate load_m without blendv or scalar fallbacks
        unsafe { Self::bitor(Self::load_z(mask, ptr), Self::bitandnot(mask, src)) }
    }

    unsafe fn load_z(mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm_maskload_ps(ptr, arch::_mm_castps_si128(mask)) }
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

    unsafe fn lookup(values: &[Self::Element], indices: Storage<Self::Unsigned>) -> Storage<Self> {
        if values.len() <= 8 {
            let mut padded = [0f32; 8];
            padded[..values.len()].copy_from_slice(values);

            unsafe {
                let table = arch::_mm256_loadu_ps(padded.as_ptr());
                let idx = arch::_mm256_castsi128_si256(indices);
                let result = arch::_mm256_permutevar8x32_ps(table, idx);
                arch::_mm256_castps256_ps128(result)
            }
        } else {
            unsafe { <Self as IndexableRegister<Self::Unsigned>>::gather(values.as_ptr(), indices) }
        }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_permute_ps::<{ MM_SHUFFLE!(0, 1, 2, 3) }>(value) }
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_bswap_psx_v2(value) }
    }

    const HAS_PERMUTEV: bool = true;

    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm_permutevar_ps(value, core::mem::transmute(idxs)) }
    }

    fn swizzle(a: Storage<Self>, b: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
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

    compress_via_table!();
}

#[thermite_macros::inline_always]
impl InterleaveRegister for F32x4V3 {
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
impl IndexableRegister<super::U32x4V3> for F32x4V3 {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<super::U32x4V3>) -> Storage<Self> {
        unsafe { arch::_mm_i32gather_ps::<4>(ptr, indices) }
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<super::U32x4V3>,
    ) -> Storage<Self> {
        unsafe { arch::_mm_mask_i32gather_ps::<4>(src, ptr, indices, mask) }
    }
}

#[thermite_macros::inline_always]
impl IndexableRegister<super::U64x4V3> for F32x4V3 {
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<super::U64x4V3>) -> Storage<Self> {
        unsafe { arch::_mm256_i64gather_ps::<4>(ptr, indices) }
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<super::U64x4V3>,
    ) -> Storage<Self> {
        unsafe { arch::_mm256_mask_i64gather_ps::<4>(src, ptr, indices, mask) }
    }
}

#[thermite_macros::inline_always]
impl ShuffleRegister for F32x4V3 {
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_shuffle_ps(lhs, rhs, IMM8) }
    }
}

#[thermite_macros::inline_always]
impl PermuteRegister for F32x4V3 {
    fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_permute_ps(value, IMM8) }
    }
}

#[thermite_macros::inline_always]
impl BlendRegister for F32x4V3 {
    fn blend<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_blend_ps::<IMM8>(lhs, rhs) }
    }
}

#[rustfmt::skip]
#[thermite_macros::inline_always]
impl PartialOrdRegister for F32x4V3 {
    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { unsafe { arch::_mm_cmplt_ps(lhs, rhs) } }
    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { unsafe { arch::_mm_cmple_ps(lhs, rhs) } }
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { unsafe { arch::_mm_cmpgt_ps(lhs, rhs) } }
    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { unsafe { arch::_mm_cmpge_ps(lhs, rhs) } }
    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { unsafe { arch::_mm_cmpeq_ps(lhs, rhs) } }
    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { unsafe { arch::_mm_cmpneq_ps(lhs, rhs) } }
}

#[thermite_macros::inline_always]
impl NumericRegister for F32x4V3 {
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
impl SignedRegister for F32x4V3 {
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
impl FloatRegister for F32x4V3 {
    const HAS_TRUE_FMA: bool = true;

    type Bits = super::U32x4V3;
    type SignedBits = super::I32x4V3;
    type ExtendedPrecision = super::F64x4V3;

    const HALF: Storage<Self> = reg::<Self, 4>([0.5; 4]);
    const NEG_ZERO: Storage<Self> = reg::<Self, 4>([-0.0; 4]);
    const EPSILON: Storage<Self> = reg::<Self, 4>([f32::EPSILON; 4]);
    const INFINITY: Storage<Self> = reg::<Self, 4>([f32::INFINITY; 4]);
    const NEG_INFINITY: Storage<Self> = reg::<Self, 4>([f32::NEG_INFINITY; 4]);
    const NAN: Storage<Self> = reg::<Self, 4>([f32::NAN; 4]);

    const EXP_MASK: Storage<Self::Bits> = reg::<Self::Bits, 4>([0x7F800000; 4]);

    fn mul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_fmadd_ps(lhs, rhs, acc) }
    }

    fn mul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_fmsub_ps(lhs, rhs, acc) }
    }

    fn nmul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_fnmadd_ps(lhs, rhs, acc) }
    }

    fn nmul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_fnmsub_ps(lhs, rhs, acc) }
    }

    //
    // fn mul_adde(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
    //     Self::mul_add(lhs, rhs, acc)
    // }

    //
    // fn mul_sube(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
    //     Self::mul_sub(lhs, rhs, acc)
    // }

    //
    // fn nmul_adde(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
    //     Self::nmul_add(lhs, rhs, acc)
    // }

    //
    // fn nmul_sube(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
    //     Self::nmul_sub(lhs, rhs, acc)
    // }

    fn addsub(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_addsub_ps(a, b) }
    }

    fn fmaddsub(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_fmaddsub_ps(a, b, c) }
    }

    fn fmsubadd(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm_fmsubadd_ps(a, b, c) }
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
    // `MM_SHUFFLE_R!` packs lane 0 into the low bits (what `_mm_permute_ps` /
    // `_mm_shuffle_ps` read first); the conventional `MM_SHUFFLE!` reverses lanes.
    ($ty:ty: $v:expr, [$a:literal, $b:literal, $c:literal, $d:literal]) => {
        unsafe { arch::_mm_permute_ps::<{ MM_SHUFFLE_R!($a, $b, $c, $d) }>($v) }
    };
    // Two-input form takes lanes 0,1 from `$v1` and 2,3 from `$v2`; callers must use
    // the `[lo from v1, hi from v2]` split. `& 3` maps the v2 indices (4..=7) into v2.
    ($ty:ty: $v1:expr, $v2:expr, [$a:literal, $b:literal, $c:literal, $d:literal]) => {
        unsafe { arch::_mm_shuffle_ps::<{ MM_SHUFFLE_R!($a, $b, $c & 3, $d & 3) }>($v1, $v2) }
    };
}

#[thermite_macros::inline_always]
impl LinAlg4Register for F32x4V3 {
    fn mat4_product<const COLUMN_MAJOR: bool>(
        lhs: &[Storage<Self>; 4],
        rhs: &[Storage<Self>; 4],
    ) -> [Storage<Self>; 4] {
        Self::mat4_product_wide::<COLUMN_MAJOR>(lhs, rhs)
    }

    fn mat4_vec3_product<const COLUMN_MAJOR: bool, const N: usize>(
        cols: &[Storage<Self>; 4],
        vectors: &[Storage<Self>; N],
    ) -> [Storage<Self>; N] {
        type W = super::F32x8V3;

        let m = if const { COLUMN_MAJOR } {
            *cols
        } else {
            Self::mat4_transpose(cols)
        };

        // Duplicate the three basis columns into both halves (hoisted).
        let a0 = W::concat(m[0], m[0]);
        let a1 = W::concat(m[1], m[1]);
        let a2 = W::concat(m[2], m[2]);

        let mut out = [Self::EMPTY; N];

        // Two vectors per pass: [M*v_i | M*v_{i+1}] (3 terms, no fold).
        let mut i = 0;
        while i + 1 < N {
            let (va, vb) = (vectors[i], vectors[i + 1]);
            let x = W::concat(Self::broadcast::<0>(va), Self::broadcast::<0>(vb));
            let y = W::concat(Self::broadcast::<1>(va), Self::broadcast::<1>(vb));
            let z = W::concat(Self::broadcast::<2>(va), Self::broadcast::<2>(vb));
            let prod = W::mul_adde(a2, z, W::mul_adde(a1, y, W::mul(a0, x)));
            let (ra, rb) = W::split(prod);
            out[i] = ra;
            out[i + 1] = rb;
            i += 2;
        }

        if const { N % 2 == 1 } {
            out[N - 1] = Self::mat4_vec3_product_wide(&m, vectors[N - 1]);
        }

        out
    }

    fn mat4_vec4_product<const COLUMN_MAJOR: bool, const N: usize>(
        cols: &[Storage<Self>; 4],
        vectors: &[Storage<Self>; N],
    ) -> [Storage<Self>; N] {
        type W = super::F32x8V3;

        let m = if const { COLUMN_MAJOR } {
            *cols
        } else {
            Self::mat4_transpose(cols)
        };

        // Duplicate each basis column into both 128-bit halves; hoisted across
        // every pair (4 ymm, loop-invariant).
        let a0 = W::concat(m[0], m[0]);
        let a1 = W::concat(m[1], m[1]);
        let a2 = W::concat(m[2], m[2]);
        let a3 = W::concat(m[3], m[3]);

        let mut out = [Self::EMPTY; N];

        // Process two vectors per wide pass: [M*v_i | M*v_{i+1}], no per-vector fold.
        let mut i = 0;
        while i + 1 < N {
            let (va, vb) = (vectors[i], vectors[i + 1]);
            // coefficients: lane k of each vector broadcast within its own 128-bit half
            let x = W::concat(Self::broadcast::<0>(va), Self::broadcast::<0>(vb));
            let y = W::concat(Self::broadcast::<1>(va), Self::broadcast::<1>(vb));
            let z = W::concat(Self::broadcast::<2>(va), Self::broadcast::<2>(vb));
            let w = W::concat(Self::broadcast::<3>(va), Self::broadcast::<3>(vb));

            let prod = W::add(W::mul_adde(a1, y, W::mul(a0, x)), W::mul_adde(a3, w, W::mul(a2, z)));

            let (ra, rb) = W::split(prod);
            out[i] = ra;
            out[i + 1] = rb;
            i += 2;
        }

        // Odd tail: one leftover vector via the single-vector wide path.
        if const { N % 2 == 1 } {
            out[N - 1] = Self::mat4_vec4_product_wide(&m, vectors[N - 1]);
        }

        out
    }

    // dedicated x86-v3 implementation that takes advantage of `_mm_permute_ps`/`_mm_shuffle_ps`.
    fn mat4_inverse(m: &mut [Storage<Self>; 4]) -> Self::Element {
        impl_mat4_inverse!(m, s)
    }

    fn mat4_det(m: &[Storage<Self>; 4]) -> Self::Element {
        impl_mat4_inverse!(DET_ONLY m, s)
    }
}

// Just use the SSE4.1 implementation
#[thermite_macros::inline_always]
impl LinAlg3Register for F32x4V3 {
    fn dot3(lhs: Storage<Self>, rhs: Storage<Self>) -> f32 {
        unsafe { arch::dot3_v1(lhs, rhs) }
    }

    fn zero4(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::zero4_v2(value) }
    }

    fn one4(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::one4_v2(value) }
    }

    fn mat3_vec3_product<const COLUMN_MAJOR: bool, const N: usize>(
        cols: &[Storage<Self>; 3],
        vectors: &[Storage<Self>; N],
    ) -> [Storage<Self>; N] {
        let mut out = [Self::EMPTY; N];

        if const { COLUMN_MAJOR } {
            type W = super::F32x8V3;

            // Duplicate the three columns into both halves (hoisted).
            let a0 = W::concat(cols[0], cols[0]);
            let a1 = W::concat(cols[1], cols[1]);
            let a2 = W::concat(cols[2], cols[2]);

            // Two vectors per pass: [M*v_i | M*v_{i+1}] (3 terms, no fold).
            let mut i = 0;
            while i + 1 < N {
                let (va, vb) = (vectors[i], vectors[i + 1]);
                let x = W::concat(Self::broadcast::<0>(va), Self::broadcast::<0>(vb));
                let y = W::concat(Self::broadcast::<1>(va), Self::broadcast::<1>(vb));
                let z = W::concat(Self::broadcast::<2>(va), Self::broadcast::<2>(vb));
                let prod = W::mul_adde(a2, z, W::mul_adde(a1, y, W::mul(a0, x)));
                let (ra, rb) = W::split(prod);
                out[i] = ra;
                out[i + 1] = rb;
                i += 2;
            }

            if const { N % 2 == 1 } {
                out[N - 1] = Self::mat3_vec3_product_wide(cols, vectors[N - 1]);
            }
        } else {
            // row-major: per-row dot3
            let mut i = 0;
            while i < N {
                let v = vectors[i];
                let mut r = Self::EMPTY;
                r = Self::insert::<0>(r, Self::dot3(cols[0], v));
                r = Self::insert::<1>(r, Self::dot3(cols[1], v));
                out[i] = Self::insert::<2>(r, Self::dot3(cols[2], v));
                i += 1;
            }
        }

        out
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
