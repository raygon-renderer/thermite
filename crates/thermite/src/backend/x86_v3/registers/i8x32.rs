//! Native 256-bit signed 8-bit register for x86-v3 (AVX2). This is the native-width 8-bit
//! register (`Native8Width = U32`). x86 has no native 8-bit shift or multiply, so those use
//! AVX2 polyfills (16-bit ops + masking); everything else is native AVX2.

use generic_array::{
    GenericArray,
    sequence::GenericSequence,
    typenum::{self, Unsigned},
};

use crate::{
    isa::InstructionSet,
    register::{
        BitshiftRegister, BitwiseRegister, CoreRegister, ExtendRegister, IntegerRegister, InterleaveRegister,
        MaskElement, MaskRegister, NumericRegister, PartialOrdRegister, Register, SignedIntegerRegister,
        SignedRegister, Storage, ZeroUpper, empty_reg, reg, reg_splat,
    },
};

use super::arch;

#[cfg_attr(not(feature = "document_registers"), doc(hidden))]
#[derive(Debug, Clone, Copy, Hash)]
pub struct I8x32V3;

#[thermite_macros::inline_always]
impl CoreRegister for I8x32V3 {
    type NativeIsa = crate::backend::x86_v3::X86V3;
    type Lanes = typenum::U32;
    type Storage = arch::__m256i;
    type Mask = Self;

    const IS_EMULATED: bool = false;
    const EMPTY: Storage<Self> = empty_reg::<Self>();
    const HAS_EQUAL_SIZE_MASK: bool = true;

    fn blendv(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_blendv_epi8(lhs, rhs, mask) }
    }

    fn zz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_and_si256(value, mask) }
    }

    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_andnot_si256(mask, value) }
    }

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        if const { Z::N >= 32 } {
            value
        } else {
            let mut arr = [0i8; 32];
            unsafe { arch::_mm256_storeu_si256(arr.as_mut_ptr() as *mut _, value) };
            let mut i = const { 32 - Z::N };
            while i < 32 {
                arr[i] = 0;
                i += 1;
            }
            unsafe { arch::_mm256_loadu_si256(arr.as_ptr() as *const _) }
        }
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
        mask
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<i8> for I8x32V3 {
    fn extend(value: Storage<i8>) -> Storage<Self> {
        let mut arr = [0i8; 32];
        arr[0] = value;
        unsafe { arch::_mm256_loadu_si256(arr.as_ptr() as *const _) }
    }

    fn narrow(value: Storage<Self>) -> Storage<i8> {
        Self::as_slice(&value)[0]
    }
}

#[thermite_macros::inline_always]
impl MaskRegister for I8x32V3 {
    const FALSY: Storage<Self> = reg::<Self, 32>([0; 32]);
    const TRUTHY: Storage<Self> = reg::<Self, 32>([-1; 32]);

    fn set(mut mask: Storage<Self::Mask>, lane: usize, value: bool) -> Storage<Self> {
        Self::as_mut_slice(&mut mask)[lane] = if value { MaskElement::TRUTHY } else { MaskElement::FALSY };
        mask
    }

    fn test(mask: Storage<Self::Mask>, lane: usize) -> bool {
        Self::as_slice(&mask)[lane].to_bool()
    }

    fn all(value: Storage<Self>) -> bool {
        unsafe { arch::_mm256_movemask_epi8(value) as u32 == 0xFFFF_FFFF }
    }

    fn any(value: Storage<Self>) -> bool {
        unsafe { arch::_mm256_movemask_epi8(value) != 0 }
    }

    fn none(value: Storage<Self>) -> bool {
        unsafe { arch::_mm256_movemask_epi8(value) == 0 }
    }

    fn from_native_bitmask(bitmask: u64) -> Storage<Self> {
        unsafe { arch::_mm256_movm_epi8x_v3(bitmask) }
    }

    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        // One bit per byte lane, directly - 32 bits, no packing/permute fixup needed.
        Some(unsafe { (arch::_mm256_movemask_epi8(value) as u32) as u64 })
    }

    #[cfg(feature = "bitvec")]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        let mask = Self::native_bitmask(value).unwrap() as u32;
        let mask = bitvec::slice::BitSlice::from_slice(core::slice::from_ref(&mask));
        view.copy_from_bitslice(&mask[..Self::Lanes::USIZE]);
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitwiseRegister for I8x32V3 {
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_xor_si256(lhs, rhs) }
    }
    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_and_si256(lhs, rhs) }
    }
    fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_andnot_si256(lhs, rhs) }
    }
    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_or_si256(lhs, rhs) }
    }
    fn not(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_xor_si256(value, arch::_mm256_set1_epi8(-1)) }
    }
}

#[thermite_macros::inline_always]
impl Register for I8x32V3 {
    type Element = i8;

    type Signed = super::I8x32V3;
    type Unsigned = super::U8x32V3;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::ne(value, Self::ZERO)
    }

    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {
        value
    }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        unsafe { arch::_mm256_cmpgt_epi8(arch::_mm256_setzero_si256(), value) }
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        unsafe { arch::_mm256_loadu_si256(value.as_ptr() as *const _) }
    }

    fn single(value: Self::Element) -> Storage<Self> {
        let mut arr = [0i8; 32];
        arr[0] = value;
        unsafe { arch::_mm256_loadu_si256(arr.as_ptr() as *const _) }
    }

    impl_native_extract!(@epi8x32);

    fn splat(value: Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_set1_epi8(value) }
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_load_si256(ptr as *const _) }
    }

    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_loadu_si256(ptr as *const _) }
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm256_store_si256(ptr as *mut _, value) }
    }

    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm256_storeu_si256(ptr as *mut _, value) }
    }

    unsafe fn load_stream(ptr: *const Self::Element) -> Storage<Self> {
        unsafe { arch::_mm256_stream_load_si256(ptr as _) }
    }

    unsafe fn store_stream(ptr: *mut Self::Element, value: Storage<Self>) {
        unsafe { arch::_mm256_stream_si256(ptr as _, value) }
    }

    fn reverse(value: Storage<Self>) -> Storage<Self> {
        // Reverse bytes within each 128-bit lane, then swap the two lanes.
        unsafe {
            let rev = arch::_mm256_shuffle_epi8(
                value,
                arch::_mm256_setr_epi8(
                    15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, // lane 0
                    15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, // lane 1
                ),
            );
            arch::_mm256_permute2x128_si256(rev, rev, 0x01)
        }
    }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        // One byte per lane, so byte-swap within a lane is the identity.
        value
    }

    const HAS_PERMUTEV: bool = true;

    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        unsafe {
            let p = idxs.as_ptr() as *const arch::__m256i;
            arch::_mm256_permutev_epi8x_v3(
                value,
                arch::_mm256_loadu_si256(p),
                arch::_mm256_loadu_si256(p.add(1)),
                arch::_mm256_loadu_si256(p.add(2)),
                arch::_mm256_loadu_si256(p.add(3)),
            )
        }
    }

    compress_via_wide!();

    impl_byte_align_alignr256!();
}

#[thermite_macros::inline_always]
impl InterleaveRegister for I8x32V3 {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        unsafe {
            let u_lo = arch::_mm256_unpacklo_epi8(a, b);
            let u_hi = arch::_mm256_unpackhi_epi8(a, b);
            let res_lo = arch::_mm256_permute2x128_si256(u_lo, u_hi, 0x20);
            let res_hi = arch::_mm256_permute2x128_si256(u_lo, u_hi, 0x31);
            (res_lo, res_hi)
        }
    }

    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        // Within each 128-bit lane, gather even bytes to the low 64 bits and odd bytes to the
        // high 64 bits (one `pshufb` per input). `unpacklo/hi_epi64` then merges the inputs'
        // even/odd groups, and `permute4x64` fixes the 128-bit-lane ordering.
        unsafe {
            let shuf = arch::_mm256_setr_epi8(
                0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, // lane 0
                0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, // lane 1
            );
            let a_s = arch::_mm256_shuffle_epi8(a, shuf);
            let b_s = arch::_mm256_shuffle_epi8(b, shuf);

            let even_pre = arch::_mm256_unpacklo_epi64(a_s, b_s);
            let odd_pre = arch::_mm256_unpackhi_epi64(a_s, b_s);

            (
                arch::_mm256_permute4x64_epi64(even_pre, 0b11_01_10_00),
                arch::_mm256_permute4x64_epi64(odd_pre, 0b11_01_10_00),
            )
        }
    }
}

#[rustfmt::skip] #[thermite_macros::inline_always]
impl BitshiftRegister for I8x32V3 {
    const HAS_TRUE_SHIFTV: bool = false;
    const HAS_WIDE_BYTE_SHIFTS: bool = true;

    fn bshli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_bslli_epi128(value, IMM8) }
    }
    fn bshri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_bsrli_epi128(value, IMM8) }
    }
    fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm256_sll_epi8x_v3(value, shift) }
    }
    fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm256_srl_epi8x_v3(value, shift) }
    }
    fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_slli_epi8x_v3::<IMM8>(value) }
    }
    fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_srli_epi8x_v3::<IMM8>(value) }
    }
    fn shlv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm256_sllv_epi8x_v3(value, shifts) }
    }
    fn shrv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm256_srlv_epi8x_v3(value, shifts) }
    }
}

#[thermite_macros::inline_always]
impl PartialOrdRegister for I8x32V3 {
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmpgt_epi8(lhs, rhs) }
    }

    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_cmpeq_epi8(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl NumericRegister for I8x32V3 {
    const ZERO: Storage<Self> = reg::<Self, 32>([0; 32]);
    const ONE: Storage<Self> = reg::<Self, 32>([1; 32]);
    const TWO: Storage<Self> = reg::<Self, 32>([2; 32]);

    const MIN: Storage<Self> = reg::<Self, 32>([i8::MIN; 32]);
    const MAX: Storage<Self> = reg::<Self, 32>([i8::MAX; 32]);

    fn min_element(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi8_v3!(value; _mm_min_epi8)
    }
    fn max_element(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi8_v3!(value; _mm_max_epi8)
    }
    fn sum_elements(value: Storage<Self>) -> Self::Element {
        _mm256_reduce_epi8_v3!(value; _mm_add_epi8)
    }
    fn prod_elements(value: Storage<Self>) -> Self::Element {
        // No `_mm_mullo_epi8`; reduce via scalar fold (rarely used at byte width).
        Self::as_slice(&value).iter().copied().fold(1i8, i8::wrapping_mul)
    }

    fn offset() -> Storage<Self> {
        Self::splat(<Self::Lanes as typenum::Unsigned>::I8)
    }

    fn indexed() -> Storage<Self> {
        Self::new(GenericArray::generate(|i| i as i8))
    }

    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_add_epi8(lhs, rhs) }
    }
    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_sub_epi8(lhs, rhs) }
    }
    fn add_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_add_epi8(lhs, arch::_mm256_and_si256(rhs, mask)) }
    }
    fn sub_c(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_sub_epi8(lhs, arch::_mm256_and_si256(rhs, mask)) }
    }
    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mullo_epi8x_v3(lhs, rhs) }
    }
    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a.wrapping_div(b) })
    }
    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        Self::zip(lhs, rhs, |a, b| if b == 0 { 0 } else { a.wrapping_rem(b) })
    }
    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_min_epi8(lhs, rhs) }
    }
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_max_epi8(lhs, rhs) }
    }
}

#[thermite_macros::inline_always]
impl SignedRegister for I8x32V3 {
    const NEG_ONE: Storage<Self> = reg::<Self, 32>([-1; 32]);
    const MIN_POSITIVE: Storage<Self> = reg_splat::<Self>(1);

    fn neg(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_sign_epi8(value, Self::NEG_ONE) }
    }
    fn is_negative(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::msb_to_mask(value)
    }
    fn is_positive(value: Storage<Self>) -> Storage<Self::Mask> {
        Self::not(Self::is_negative(value))
    }
    fn abs(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_abs_epi8(value) }
    }
    fn signum(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_sign_epi8(arch::_mm256_set1_epi8(1), value) }
    }
}

#[thermite_macros::inline_always]
impl IntegerRegister for I8x32V3 {
    fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mulhi_epi8x_v3(lhs, rhs) }
    }
    fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_mullo_epi8x_v3(lhs, rhs) }
    }
    fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_adds_epi8(lhs, rhs) }
    }
    fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_subs_epi8(lhs, rhs) }
    }

    fn div_branched(value: Storage<Self>, divider: crate::Divider<Self::Element>) -> Storage<Self> {
        arch::div_epi::<Self>(value, divider.multiplier(), divider.shift())
    }
    fn div_branchfree(value: Storage<Self>, divider: crate::BranchfreeDivider<Self::Element>) -> Storage<Self> {
        arch::div_epi_bf::<Self>(value, divider.multiplier(), divider.shift())
    }
    fn divv_branchfree(value: Storage<Self>, dividers: crate::divider::vector::VectorDivider<Self>) -> Storage<Self> {
        arch::divv_epi_bf::<Self>(value, dividers.multipliers.0, dividers.shifts.0)
    }

    const HAS_HARDWARE_POPCNT: bool = false;

    fn count_ones(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_popcnt_epi8x_v3(value) }
    }
    fn count_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(Self::not(value))
    }
    fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
        super::U8x32V3::leading_zeros(value)
    }
    fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {
        Self::count_ones(Self::sub(Self::bitand(value, Self::neg(value)), Self::ONE))
    }
    fn leading_ones(value: Storage<Self>) -> Storage<Self> {
        Self::leading_zeros(Self::not(value))
    }
    fn trailing_ones(value: Storage<Self>) -> Storage<Self> {
        Self::trailing_zeros(Self::not(value))
    }
}

#[thermite_macros::inline_always]
impl SignedIntegerRegister for I8x32V3 {
    fn srai<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {
        unsafe { arch::_mm256_srai_epi8x_v3::<IMM8>(value) }
    }
    fn sra(value: Storage<Self>, shift: u32) -> Storage<Self> {
        unsafe { arch::_mm256_sra_epi8x_v3(value, shift) }
    }
    fn srav(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
        unsafe { arch::_mm256_srav_epi8x_v3(value, shifts) }
    }
}
