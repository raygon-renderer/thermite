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
        Element, ExtendRegister, FloatRegister, IndexableRegister, IntegerRegister, InterleaveRegister,
        LinAlg3Register, LinAlg4Register, MaskElement, MaskRegister, NativeCapability, NumericRegister,
        PartialOrdRegister, PermuteRegister, Register, ShuffleRegister, SignedIntegerRegister, SignedRegister, Storage,
        SwizzleIndices, SwizzleRegister, WideRegister, ZeroUpper, empty_reg, reg,
    },
    simd::Simd,
};

use super::*;

macro_rules! decl_i32xN {
    ($name:ident x $N:literal { $($i:ident : $idx:literal),* }) => {paste::paste! {
        #[cfg_attr(target_arch = "spirv", rust_gpu::vector::v1)]
        #[derive(Debug, Clone, Copy, const_default::ConstDefault, PartialEq, PartialOrd)]
        pub struct $name {
            $(pub $i: i32,)*
        }

        impl CoreRegister for $name {
            type Lanes = typenum::[<U $N>];
            type Storage = Self;
            type Mask = super::[<Mx $N>];

            const IS_EMULATED: bool = false;
            const ISA: InstructionSet = InstructionSet::SPIRV;
            const EMPTY: Self = <Self as const_default::ConstDefault>::DEFAULT;
            const HAS_EQUAL_SIZE_MASK: bool = false;

            #[inline(always)]
            fn blendv(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Self {
                Self { $($i: <i32 as CoreRegister>::blendv(mask.$i, a.$i, b.$i),)* }
            }

            #[inline(always)]
            fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
                Self { $($i: if const { Z::N > $idx } { value.$i } else { 0 },)* }
            }
        }

        // Integer types support bitwise ops directly — no need to bitcast through u32.
        impl BitwiseRegister for $name {
            #[inline(always)]
            fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                unsafe { arch::op_opbitwisexor::<Self>(lhs, rhs) }
            }
            #[inline(always)]
            fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                unsafe { arch::op_opbitwiseand::<Self>(lhs, rhs) }
            }
            #[inline(always)]
            fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                unsafe { arch::op_opbitwiseor::<Self>(lhs, rhs) }
            }
            #[inline(always)]
            fn not(value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::op_opnot::<Self>(value) }
            }
        }

        impl InterleaveRegister for $name {
            #[inline(always)]
            fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
                unsafe { arch::[<spirv_interleave $N>](a, b) }
            }
            #[inline(always)]
            fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
                unsafe { arch::[<spirv_deinterleave $N>](a, b) }
            }
        }

        impl Register for $name {
            type Element = i32;
            type Signed   = Self;
            type Unsigned = super::[<U32x $N>];

            // true -> -1i32 (all-ones), false -> 0
            #[inline(always)]
            fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
                unsafe { arch::op_opselect::<Self, super::[<Mx $N>]>(mask, Self::splat(-1), Self::EMPTY) }
            }

            // Non-zero -> true, zero -> false.
            #[inline(always)]
            fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
                unsafe { arch::op_opinotequal::<super::[<Mx $N>], Self>(value, Self::EMPTY) }
            }

            // MSB set ↔ negative signed value.
            #[inline(always)]
            fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
                unsafe { arch::op_opslessthan::<super::[<Mx $N>], Self>(value, Self::EMPTY) }
            }

            #[inline(always)]
            fn new(value: GenericArray<i32, Self::Lanes>) -> Storage<Self> {
                Self { $($i: value[$idx],)* }
            }

            // Place the scalar in lane 0, zero all other lanes.
            #[inline(always)]
            fn single(value: i32) -> Storage<Self> {
                Self { $($i: if const { $idx == 0 } { value } else { 0 },)* }
            }

            #[inline(always)]
            fn splat(value: i32) -> Storage<Self> {
                Self { $($i: value,)* }
            }

            // Byte-swap each i32 lane: bitcast to u32, swap, bitcast back.
            #[inline(always)]
            fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
                let u = unsafe { arch::op_opbitcast::<super::[<U32x $N>], Self>(value) };
                let swapped = <super::[<U32x $N>] as Register>::swap_bytes(u);
                unsafe { arch::op_opbitcast::<Self, super::[<U32x $N>]>(swapped) }
            }

            #[inline(always)]
            fn extract<const I: usize>(value: Storage<Self>) -> i32 {
                unsafe { arch::op_opvectorextractdynamic::<i32, Self, usize>(value, I) }
            }

            #[inline(always)]
            fn insert<const I: usize>(value: Storage<Self>, element: i32) -> Storage<Self> {
                unsafe { arch::op_opvectorinsertdynamic::<Self, i32, usize>(value, element, I) }
            }

            // Single OpVectorShuffle with every output lane pointing to I.
            #[inline(always)]
            fn broadcast<const I: usize>(value: Storage<Self>) -> Storage<Self> {
                let mut result = Self::EMPTY;
                unsafe {
                    core::arch::asm!(
                        "%v      = OpLoad typeof*{v} {v}",
                        concat!("%result = OpVectorShuffle typeof*{result} %v %v", $(concat!(" {i", stringify!($idx), "}")),+),
                        "OpStore {result} %result",
                        v      = in(reg) &value,
                        result = in(reg) &mut result,
                        $([<i $idx>] = const I,)+
                    );
                }
                result
            }

            // Runtime-index version: extract the chosen lane, then splat.
            #[inline(always)]
            fn broadcastv(value: Storage<Self>, idx: usize) -> Storage<Self> {
                let elem = unsafe { arch::op_opvectorextractdynamic::<i32, Self, usize>(value, idx) };
                Self::splat(elem)
            }

            #[inline(always)]
            fn map<F>(value: Storage<Self>, mut f: F) -> Storage<Self>
            where F: FnMut(i32) -> i32 {
                Self { $($i: f(value.$i),)* }
            }

            #[inline(always)]
            fn zip<F>(lhs: Storage<Self>, rhs: Storage<Self>, f: F) -> Storage<Self>
            where F: Fn(i32, i32) -> i32 {
                Self { $($i: f(lhs.$i, rhs.$i),)* }
            }

            #[inline(always)]
            fn fold<F>(first: i32, value: Storage<Self>, f: F) -> i32
            where F: Fn(i32, i32) -> i32 {
                let acc = first;
                $(let acc = f(acc, value.$i);)*
                acc
            }

            #[inline(always)]
            fn reduce<F>(value: Storage<Self>, f: F) -> i32
            where F: Fn(i32, i32) -> i32 {
                let acc = Self::extract::<0>(value);
                $(let acc = if const { $idx > 0 } { f(acc, value.$i) } else { acc };)*
                acc
            }

            // Single OpVectorShuffle with descending indices.
            #[inline(always)]
            fn reverse(value: Storage<Self>) -> Storage<Self> {
                let mut result = Self::EMPTY;
                unsafe {
                    core::arch::asm!(
                        "%v      = OpLoad typeof*{v} {v}",
                        concat!("%result = OpVectorShuffle typeof*{result} %v %v", $(concat!(" {r", stringify!($idx), "}")),+),
                        "OpStore {result} %result",
                        v      = in(reg) &value,
                        result = in(reg) &mut result,
                        $([<r $idx>] = const { $N - 1 - $idx },)+
                    );
                }
                result
            }
        }

        impl NumericRegister for $name {
            const ZERO: Self = Self { $($i: 0,)* };
            const ONE:  Self = Self { $($i: 1,)* };
            const TWO:  Self = Self { $($i: 2,)* };
            const MIN:  Self = Self { $($i: i32::MIN,)* };
            const MAX:  Self = Self { $($i: i32::MAX,)* };

            #[inline(always)] fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                unsafe { arch::op_opiadd::<Self>(lhs, rhs) }
            }
            #[inline(always)] fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                unsafe { arch::op_opisub::<Self>(lhs, rhs) }
            }
            #[inline(always)] fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                unsafe { arch::op_opimul::<Self>(lhs, rhs) }
            }
            // Signed truncating division (matches Rust's /).
            #[inline(always)] fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                unsafe { arch::op_opsdiv::<Self>(lhs, rhs) }
            }
            // Signed truncating remainder (matches Rust's %).
            #[inline(always)] fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                unsafe { arch::op_opsrem::<Self>(lhs, rhs) }
            }

            #[inline(always)] fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                unsafe { arch::glsl_op2::<Self, Self, Self, { arch::glsl::S_MIN }, false>(lhs, rhs) }
            }
            #[inline(always)] fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                unsafe { arch::glsl_op2::<Self, Self, Self, { arch::glsl::S_MAX }, false>(lhs, rhs) }
            }

            // Scalar reductions via direct field access — no permutes needed on GPU.
            #[inline(always)] fn min_element(value: Storage<Self>) -> i32 {
                Self::reduce(value, |a, b| if a < b { a } else { b })
            }
            #[inline(always)] fn max_element(value: Storage<Self>) -> i32 {
                Self::reduce(value, |a, b| if a > b { a } else { b })
            }
            #[inline(always)] fn sum_elements(value: Storage<Self>) -> i32 {
                Self::reduce(value, |a, b| a.wrapping_add(b))
            }
            #[inline(always)] fn prod_elements(value: Storage<Self>) -> i32 {
                Self::reduce(value, |a, b| a.wrapping_mul(b))
            }

            #[inline(always)] fn offset() -> Storage<Self> { Self::splat($N as i32) }
            #[inline(always)] fn indexed() -> Storage<Self> { Self { $($i: $idx as i32,)* } }
        }

        impl PartialOrdRegister for $name {
            #[inline(always)]
            fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
                unsafe { arch::op_opiequal::<super::[<Mx $N>], Self>(lhs, rhs) }
            }
            #[inline(always)]
            fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
                unsafe { arch::op_opinotequal::<super::[<Mx $N>], Self>(lhs, rhs) }
            }
            #[inline(always)]
            fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
                unsafe { arch::op_opslessthan::<super::[<Mx $N>], Self>(lhs, rhs) }
            }
            #[inline(always)]
            fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
                unsafe { arch::op_opsgreaterthan::<super::[<Mx $N>], Self>(lhs, rhs) }
            }
            #[inline(always)]
            fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
                unsafe { arch::op_opslessthanequal::<super::[<Mx $N>], Self>(lhs, rhs) }
            }
            #[inline(always)]
            fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
                unsafe { arch::op_opsgreaterthanequal::<super::[<Mx $N>], Self>(lhs, rhs) }
            }
        }

        impl CastRegister<$name> for $name {
            #[inline(always)]
            fn cast_from(value: Storage<Self>) -> Storage<Self> { value }
        }

        impl BitCastRegister<$name> for $name {
            #[inline(always)]
            fn from_bits(value: Storage<Self>) -> Storage<Self> { value }
        }

        impl SwizzleRegister for $name {
            // No single SPIR-V instruction for runtime-index permute; scalar fallback is used.
            const HAS_PERMUTEV: bool = false;

            #[inline(always)]
            fn permutev_const<I: SwizzleIndices<Self::Lanes>>(value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::[<spirv_permute $N>]::<Self, I>(value) }
            }

            #[inline(always)]
            fn swizzle_const<I: SwizzleIndices<Self::Lanes>>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
                unsafe { arch::[<spirv_swizzle $N>]::<Self, I>(a, b) }
            }
        }

        impl BitshiftRegister for $name {
            // SPIR-V has no byte-wide register shifts (no equivalent to _mm_bslli_si128).
            const HAS_WIDE_BYTE_SHIFTS: bool = false;
            // OpShiftLeftLogical / OpShiftRightLogical accept vector shift amounts natively.
            const HAS_TRUE_SHIFTV: bool = true;

            // Scalar shift: splat amount to U32xN, then emit vector shift op.
            #[inline(always)]
            fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
                let shifts = <super::[<U32x $N>]>::splat(shift);
                unsafe { arch::op_opshiftleftlogical::<Self, super::[<U32x $N>]>(value, shifts) }
            }
            #[inline(always)]
            fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
                let shifts = <super::[<U32x $N>]>::splat(shift);
                unsafe { arch::op_opshiftrightlogical::<Self, super::[<U32x $N>]>(value, shifts) }
            }
            // Variable shifts: shift amounts already in U32xN form.
            #[inline(always)]
            fn shlv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
                unsafe { arch::op_opshiftleftlogical::<Self, super::[<U32x $N>]>(value, shifts) }
            }
            #[inline(always)]
            fn shrv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
                unsafe { arch::op_opshiftrightlogical::<Self, super::[<U32x $N>]>(value, shifts) }
            }
            // OpBitReverse: native SPIR-V instruction — override the default swap+shift chain.
            #[inline(always)]
            fn reverse_bits(value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::op_opbitreverse::<Self>(value) }
            }
        }

        impl SignedRegister for $name {
            const NEG_ONE:      Self = Self { $($i: -1,)* };
            const MIN_POSITIVE: Self = Self { $($i: 1,)* };

            #[inline(always)]
            fn neg(value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::op_opsnegate::<Self>(value) }
            }

            #[inline(always)]
            fn abs(value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::glsl_op1::<Self, Self, { arch::glsl::S_ABS }, false>(value) }
            }

            // MSB set ↔ negative for two's-complement integers; same as msb_to_mask.
            #[inline(always)]
            fn is_negative(value: Storage<Self>) -> Storage<Self::Mask> {
                unsafe { arch::op_opslessthan::<super::[<Mx $N>], Self>(value, Self::EMPTY) }
            }

            // Integer copysign: abs(lhs) with the sign of rhs.
            // Edge case: lhs = i32::MIN -> abs overflows to i32::MIN; behavior matches scalar.
            #[inline(always)]
            fn copysign(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                let abs_lhs  = Self::abs(lhs);
                let neg_abs  = Self::neg(abs_lhs);
                let rhs_neg  = Self::is_negative(rhs);
                Self::blendv(rhs_neg, abs_lhs, neg_abs)
            }
        }

        impl IntegerRegister for $name {
            // High 32 bits of the 64-bit signed product via OpSMulExtended.
            #[inline(always)]
            fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                let (_, hi) = unsafe { arch::spirv_smul_extended(lhs, rhs) };
                hi
            }

            // Low 32 bits of the product — same instruction as mul.
            #[inline(always)]
            fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                unsafe { arch::op_opimul::<Self>(lhs, rhs) }
            }

            // Signed saturating add: clamp to [i32::MIN, i32::MAX] on overflow.
            // Overflow when (lhs ^ sum) & (rhs ^ sum) < 0 (inputs have same sign, output differs).
            #[inline(always)]
            fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                let sum       = Self::add(lhs, rhs);
                let overflow  = Self::bitand(Self::bitxor(lhs, sum), Self::bitxor(rhs, sum));
                let ovf_mask  = Self::is_negative(overflow);
                // Clamp to MIN when lhs < 0 (negative overflow), MAX when lhs ≥ 0 (positive overflow).
                let clamped   = Self::blendv(Self::is_negative(lhs), Self::MAX, Self::MIN);
                Self::blendv(ovf_mask, sum, clamped)
            }

            // Signed saturating sub: clamp to [i32::MIN, i32::MAX] on underflow/overflow.
            // Overflow when (lhs ^ rhs) & (lhs ^ diff) < 0 (inputs differ in sign, result matches rhs).
            #[inline(always)]
            fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                let diff      = Self::sub(lhs, rhs);
                let overflow  = Self::bitand(Self::bitxor(lhs, rhs), Self::bitxor(lhs, diff));
                let ovf_mask  = Self::is_negative(overflow);
                let clamped   = Self::blendv(Self::is_negative(lhs), Self::MAX, Self::MIN);
                Self::blendv(ovf_mask, diff, clamped)
            }

            // OpBitCount is always available in SPIR-V — no software fallback needed.
            const HAS_HARDWARE_POPCNT: bool = true;

            #[inline(always)]
            fn count_ones(value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::op_opbitcount::<Self>(value) }
            }

            // GLSLstd450 FindUMsb: position of the highest set bit (0–31), or undefined for 0.
            // Blend in 32 for the zero-input case (spec says result is undefined for 0).
            #[inline(always)]
            fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
                // Treat the bit pattern as unsigned for FindUMsb.
                let value_u = unsafe { arch::op_opbitcast::<super::[<U32x $N>], Self>(value) };
                let msb     = unsafe {
                    arch::glsl_op1::<super::[<U32x $N>], super::[<U32x $N>], { arch::glsl::FIND_U_MSB }, false>(value_u)
                };
                let msb_i = unsafe { arch::op_opbitcast::<Self, super::[<U32x $N>]>(msb) };
                let lz    = Self::sub(Self::splat(31), msb_i);
                let is_zero = <Self as PartialOrdRegister>::eq(value, Self::ZERO);
                Self::blendv(is_zero, lz, Self::splat(32))
            }

            // GLSLstd450 FindILsb: position of the lowest set bit (0–31), or -1 for 0.
            // Blend in 32 for the zero-input case.
            #[inline(always)]
            fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {
                let lsb     = unsafe { arch::glsl_op1::<Self, Self, { arch::glsl::FIND_I_LSB }, false>(value) };
                let is_zero = <Self as PartialOrdRegister>::eq(value, Self::ZERO);
                Self::blendv(is_zero, lsb, Self::splat(32))
            }

            // Magic-number division using the generic polyfill.
            // GPU has native OpSDiv, but precomputed multipliers avoid repeated divisions
            // for constant divisors (common in graphics shaders).
            #[inline(always)]
            fn div_branched(
                value: Storage<Self>,
                divider: crate::Divider<Self::Element>,
            ) -> Storage<Self> {
                crate::backend::generic::polyfills::div_epi::<Self>(
                    value, divider.multiplier(), divider.shift(),
                )
            }

            #[inline(always)]
            fn div_branchfree(
                value: Storage<Self>,
                divider: crate::BranchfreeDivider<Self::Element>,
            ) -> Storage<Self> {
                crate::backend::generic::polyfills::div_epi_bf::<Self>(
                    value, divider.multiplier(), divider.shift(),
                )
            }

            #[inline(always)]
            fn divv_branchfree(
                value: Storage<Self>,
                dividers: crate::divider::vector::VectorDivider<Self>,
            ) -> Storage<Self> {
                crate::backend::generic::polyfills::divv_epi_bf::<Self>(
                    value, dividers.multipliers.0, dividers.shifts.0,
                )
            }
        }

        impl SignedIntegerRegister for $name {
            // Arithmetic right shift — OpShiftRightArithmetic preserves the sign bit.
            #[inline(always)]
            fn sra(value: Storage<Self>, shift: u32) -> Storage<Self> {
                let shifts = <super::[<U32x $N>]>::splat(shift);
                unsafe { arch::op_opshiftrightarithmetic::<Self, super::[<U32x $N>]>(value, shifts) }
            }

            // Variable arithmetic right shift — override the scalar fallback.
            #[inline(always)]
            fn srav(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
                unsafe { arch::op_opshiftrightarithmetic::<Self, super::[<U32x $N>]>(value, shifts) }
            }
        }
    }};
}

decl_i32xN!(I32x2 x 2 { x:0, y:1 });
decl_i32xN!(I32x3 x 3 { x:0, y:1, z:2 });
decl_i32xN!(I32x4 x 4 { x:0, y:1, z:2, w:3 });

// Cross-type casts between I32xN and U32xN.
// Both OpBitcast (bit-for-bit reinterpret) and CastRegister (bit-for-bit, same width) reduce to
// OpBitcast on SPIR-V since no extension or truncation occurs at equal width.
macro_rules! impl_i32_casts {
    ($i:ident <=> $u:ident) => {
        impl CastRegister<$u> for $i {
            #[inline(always)]
            fn cast_from(value: $u) -> $i {
                unsafe { arch::op_opbitcast::<$i, $u>(value) }
            }
        }

        impl CastRegister<$i> for $u {
            #[inline(always)]
            fn cast_from(value: $i) -> $u {
                unsafe { arch::op_opbitcast::<$u, $i>(value) }
            }
        }

        impl BitCastRegister<$u> for $i {
            #[inline(always)]
            fn from_bits(value: $u) -> $i {
                unsafe { arch::op_opbitcast::<$i, $u>(value) }
            }
        }

        impl BitCastRegister<$i> for $u {
            #[inline(always)]
            fn from_bits(value: $i) -> $u {
                unsafe { arch::op_opbitcast::<$u, $i>(value) }
            }
        }
    };
}

impl_i32_casts!(I32x2 <=> U32x2);
impl_i32_casts!(I32x3 <=> U32x3);
impl_i32_casts!(I32x4 <=> U32x4);

// ExtendRegister: place scalar in lane 0, zero the rest.
impl ExtendRegister<i32> for I32x2 {
    #[inline(always)]
    fn extend(value: i32) -> I32x2 {
        I32x2::single(value)
    }
    #[inline(always)]
    fn narrow(value: I32x2) -> i32 {
        I32x2::extract::<0>(value)
    }
}

impl ExtendRegister<i32> for I32x3 {
    #[inline(always)]
    fn extend(value: i32) -> I32x3 {
        I32x3::single(value)
    }
    #[inline(always)]
    fn narrow(value: I32x3) -> i32 {
        I32x3::extract::<0>(value)
    }
}

impl ExtendRegister<i32> for I32x4 {
    #[inline(always)]
    fn extend(value: i32) -> I32x4 {
        I32x4::single(value)
    }
    #[inline(always)]
    fn narrow(value: I32x4) -> i32 {
        I32x4::extract::<0>(value)
    }
}

// ConcatRegister: combine two N-lane vectors into a 2N-lane vector, or split.
impl ConcatRegister<i32> for I32x2 {
    #[inline(always)]
    fn concat(lo: i32, hi: i32) -> I32x2 {
        I32x2 { x: lo, y: hi }
    }
    #[inline(always)]
    fn split(value: I32x2) -> (i32, i32) {
        (value.x, value.y)
    }
}

// I32x2 can be widened to I32x4.
impl WideRegister for I32x2 {
    type Wide = I32x4;
}

impl ConcatRegister<I32x2> for I32x4 {
    #[inline(always)]
    fn concat(lo: I32x2, hi: I32x2) -> I32x4 {
        I32x4 {
            x: lo.x,
            y: lo.y,
            z: hi.x,
            w: hi.y,
        }
    }
    #[inline(always)]
    fn split(value: I32x4) -> (I32x2, I32x2) {
        (I32x2 { x: value.x, y: value.y }, I32x2 { x: value.z, y: value.w })
    }
}

// ExtendRegister impls for widening to larger vectors.
impl ExtendRegister<I32x2> for I32x3 {
    #[inline(always)]
    fn extend(value: I32x2) -> I32x3 {
        I32x3 {
            x: value.x,
            y: value.y,
            z: 0,
        }
    }
    #[inline(always)]
    fn narrow(value: I32x3) -> I32x2 {
        I32x2 { x: value.x, y: value.y }
    }
}

impl ExtendRegister<I32x2> for I32x4 {
    #[inline(always)]
    fn extend(value: I32x2) -> I32x4 {
        I32x4 {
            x: value.x,
            y: value.y,
            z: 0,
            w: 0,
        }
    }
    #[inline(always)]
    fn narrow(value: I32x4) -> I32x2 {
        I32x2 { x: value.x, y: value.y }
    }
}

impl ExtendRegister<I32x3> for I32x4 {
    #[inline(always)]
    fn extend(value: I32x3) -> I32x4 {
        I32x4 {
            x: value.x,
            y: value.y,
            z: value.z,
            w: 0,
        }
    }
    #[inline(always)]
    fn narrow(value: I32x4) -> I32x3 {
        I32x3 {
            x: value.x,
            y: value.y,
            z: value.z,
        }
    }
}
