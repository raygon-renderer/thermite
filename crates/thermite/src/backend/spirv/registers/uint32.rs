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
        PartialOrdRegister, PermuteRegister, Register, ShuffleRegister, SignedRegister, Storage, SwizzleIndices,
        SwizzleRegister, UnsignedIntegerRegister, WideRegister, ZeroUpper, empty_reg, reg,
    },
    simd::Simd,
};

use super::*;

macro_rules! decl_u32xN {
    ($name:ident x $N:literal { $($i:ident : $idx:literal),* }) => {paste::paste! {
        #[cfg_attr(target_arch = "spirv", rust_gpu::vector::v1)]
        #[derive(Debug, Clone, Copy, const_default::ConstDefault, PartialEq, PartialOrd)]
        pub struct $name {
            $(pub $i: u32,)*
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
                Self { $($i: <u32 as CoreRegister>::blendv(mask.$i, a.$i, b.$i),)* }
            }

            #[inline(always)]
            fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
                Self { $($i: if const { Z::N > $idx } { value.$i } else { 0 },)* }
            }
        }

        // Integer types support bitwise ops directly — no need to bitcast through another type.
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
            type Element = u32;
            type Signed   = super::[<I32x $N>];
            type Unsigned = Self;

            // true -> 0xFFFF_FFFFu32 (all-ones), false -> 0
            #[inline(always)]
            fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
                unsafe { arch::op_opselect::<Self, super::[<Mx $N>]>(mask, Self::splat(u32::MAX), Self::EMPTY) }
            }

            // Non-zero -> true, zero -> false.
            #[inline(always)]
            fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
                unsafe { arch::op_opinotequal::<super::[<Mx $N>], Self>(value, Self::EMPTY) }
            }

            // Treat bit 31 as a sign bit: bitcast to I32xN and check for negative.
            #[inline(always)]
            fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
                let signed = unsafe { arch::op_opbitcast::<super::[<I32x $N>], Self>(value) };
                unsafe { arch::op_opslessthan::<super::[<Mx $N>], super::[<I32x $N>]>(signed, <super::[<I32x $N>]>::EMPTY) }
            }

            #[inline(always)]
            fn new(value: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
                Self { $($i: value[$idx],)* }
            }

            // Place the scalar in lane 0, zero all other lanes.
            #[inline(always)]
            fn single(value: u32) -> Storage<Self> {
                Self { $($i: if const { $idx == 0 } { value } else { 0 },)* }
            }

            #[inline(always)]
            fn splat(value: u32) -> Storage<Self> {
                Self { $($i: value,)* }
            }

            // Reverse byte order in each 32-bit lane using bitwise ops and shifts.
            // ((x & 0x000000FF) << 24) | ((x & 0x0000FF00) << 8)
            // | ((x & 0x00FF0000) >> 8) | ((x & 0xFF000000) >> 24)
            #[inline(always)]
            fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
                let sh24 = Self::splat(24u32);
                let sh8  = Self::splat(8u32);
                let b0 = unsafe { arch::op_opshiftleftlogical::<Self, Self>(
                    Self::bitand(value, Self::splat(0x000000FFu32)), sh24) };
                let b1 = unsafe { arch::op_opshiftleftlogical::<Self, Self>(
                    Self::bitand(value, Self::splat(0x0000FF00u32)), sh8) };
                let b2 = unsafe { arch::op_opshiftrightlogical::<Self, Self>(
                    Self::bitand(value, Self::splat(0x00FF0000u32)), sh8) };
                let b3 = unsafe { arch::op_opshiftrightlogical::<Self, Self>(
                    Self::bitand(value, Self::splat(0xFF000000u32)), sh24) };
                Self::bitor(Self::bitor(b0, b1), Self::bitor(b2, b3))
            }

            #[inline(always)]
            fn extract<const I: usize>(value: Storage<Self>) -> u32 {
                unsafe { arch::op_opvectorextractdynamic::<u32, Self, usize>(value, I) }
            }

            #[inline(always)]
            fn insert<const I: usize>(value: Storage<Self>, element: u32) -> Storage<Self> {
                unsafe { arch::op_opvectorinsertdynamic::<Self, u32, usize>(value, element, I) }
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
                let elem = unsafe { arch::op_opvectorextractdynamic::<u32, Self, usize>(value, idx) };
                Self::splat(elem)
            }

            #[inline(always)]
            fn map<F>(value: Storage<Self>, mut f: F) -> Storage<Self>
            where F: FnMut(u32) -> u32 {
                Self { $($i: f(value.$i),)* }
            }

            #[inline(always)]
            fn zip<F>(lhs: Storage<Self>, rhs: Storage<Self>, f: F) -> Storage<Self>
            where F: Fn(u32, u32) -> u32 {
                Self { $($i: f(lhs.$i, rhs.$i),)* }
            }

            #[inline(always)]
            fn fold<F>(first: u32, value: Storage<Self>, f: F) -> u32
            where F: Fn(u32, u32) -> u32 {
                let acc = first;
                $(let acc = f(acc, value.$i);)*
                acc
            }

            #[inline(always)]
            fn reduce<F>(value: Storage<Self>, f: F) -> u32
            where F: Fn(u32, u32) -> u32 {
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
            const MIN:  Self = Self { $($i: u32::MIN,)* };
            const MAX:  Self = Self { $($i: u32::MAX,)* };

            #[inline(always)] fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                unsafe { arch::op_opiadd::<Self>(lhs, rhs) }
            }
            #[inline(always)] fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                unsafe { arch::op_opisub::<Self>(lhs, rhs) }
            }
            #[inline(always)] fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                unsafe { arch::op_opimul::<Self>(lhs, rhs) }
            }
            // Unsigned truncating division (matches Rust's /).
            #[inline(always)] fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                unsafe { arch::op_opudiv::<Self>(lhs, rhs) }
            }
            // Unsigned remainder (matches Rust's %).
            #[inline(always)] fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                unsafe { arch::op_opumod::<Self>(lhs, rhs) }
            }

            #[inline(always)] fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                unsafe { arch::glsl_op2::<Self, Self, Self, { arch::glsl::U_MIN }, false>(lhs, rhs) }
            }
            #[inline(always)] fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                unsafe { arch::glsl_op2::<Self, Self, Self, { arch::glsl::U_MAX }, false>(lhs, rhs) }
            }

            // Scalar reductions via direct field access — no permutes needed on GPU.
            #[inline(always)] fn min_element(value: Storage<Self>) -> u32 {
                Self::reduce(value, |a, b| if a < b { a } else { b })
            }
            #[inline(always)] fn max_element(value: Storage<Self>) -> u32 {
                Self::reduce(value, |a, b| if a > b { a } else { b })
            }
            #[inline(always)] fn sum_elements(value: Storage<Self>) -> u32 {
                Self::reduce(value, |a, b| a.wrapping_add(b))
            }
            #[inline(always)] fn prod_elements(value: Storage<Self>) -> u32 {
                Self::reduce(value, |a, b| a.wrapping_mul(b))
            }

            #[inline(always)] fn offset() -> Storage<Self> { Self::splat($N as u32) }
            #[inline(always)] fn indexed() -> Storage<Self> { Self { $($i: $idx as u32,)* } }
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
                unsafe { arch::op_opulessthan::<super::[<Mx $N>], Self>(lhs, rhs) }
            }
            #[inline(always)]
            fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
                unsafe { arch::op_opugreaterthan::<super::[<Mx $N>], Self>(lhs, rhs) }
            }
            #[inline(always)]
            fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
                unsafe { arch::op_opulessthanequal::<super::[<Mx $N>], Self>(lhs, rhs) }
            }
            #[inline(always)]
            fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
                unsafe { arch::op_opugreaterthanequal::<super::[<Mx $N>], Self>(lhs, rhs) }
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
            // SPIR-V has no byte-wide register shifts.
            const HAS_WIDE_BYTE_SHIFTS: bool = false;
            // OpShiftLeftLogical / OpShiftRightLogical accept vector shift amounts natively.
            const HAS_TRUE_SHIFTV: bool = true;

            // Scalar shift: splat shift amount to Self (U32xN), then emit vector shift op.
            // For unsigned types, the shift operand type is Self (same as value type).
            #[inline(always)]
            fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {
                let shifts = Self::splat(shift);
                unsafe { arch::op_opshiftleftlogical::<Self, Self>(value, shifts) }
            }
            #[inline(always)]
            fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {
                let shifts = Self::splat(shift);
                unsafe { arch::op_opshiftrightlogical::<Self, Self>(value, shifts) }
            }
            // Variable shifts: shift amounts already in Self (U32xN) form.
            #[inline(always)]
            fn shlv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
                unsafe { arch::op_opshiftleftlogical::<Self, Self>(value, shifts) }
            }
            #[inline(always)]
            fn shrv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {
                unsafe { arch::op_opshiftrightlogical::<Self, Self>(value, shifts) }
            }
            // OpBitReverse: native SPIR-V instruction — override the default swap+shift chain.
            #[inline(always)]
            fn reverse_bits(value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::op_opbitreverse::<Self>(value) }
            }
        }

        impl IntegerRegister for $name {
            // High 32 bits of the 64-bit unsigned product via OpUMulExtended.
            #[inline(always)]
            fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                let (_, hi) = unsafe { arch::spirv_umul_extended(lhs, rhs) };
                hi
            }

            // Low 32 bits of the product — same instruction as mul.
            #[inline(always)]
            fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                unsafe { arch::op_opimul::<Self>(lhs, rhs) }
            }

            // Unsigned saturating add: clamp to u32::MAX on carry.
            // OpIAddCarry provides the carry bit at no extra cost.
            #[inline(always)]
            fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                let (sum, carry) = unsafe { arch::spirv_iadd_carry(lhs, rhs) };
                let overflow = <Self as PartialOrdRegister>::ne(carry, Self::ZERO);
                Self::blendv(overflow, sum, Self::MAX)
            }

            // Unsigned saturating sub: clamp to 0 on borrow.
            // OpISubBorrow provides the borrow bit at no extra cost.
            #[inline(always)]
            fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                let (diff, borrow) = unsafe { arch::spirv_isub_borrow(lhs, rhs) };
                let underflow = <Self as PartialOrdRegister>::ne(borrow, Self::ZERO);
                Self::blendv(underflow, diff, Self::ZERO)
            }

            // OpBitCount is always available in SPIR-V.
            const HAS_HARDWARE_POPCNT: bool = true;

            #[inline(always)]
            fn count_ones(value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::op_opbitcount::<Self>(value) }
            }

            // GLSLstd450 FindUMsb: position of the highest set bit (0–31), or undefined for 0.
            // Blend in 32 for the zero case.
            #[inline(always)]
            fn leading_zeros(value: Storage<Self>) -> Storage<Self> {
                let msb = unsafe { arch::glsl_op1::<Self, Self, { arch::glsl::FIND_U_MSB }, false>(value) };
                // msb is in 0..=31 for non-zero. For zero the result is implementation-defined.
                // Compute 31 - msb in signed arithmetic to handle the -1 (0xFFFF_FFFF) case.
                let msb_i = unsafe { arch::op_opbitcast::<super::[<I32x $N>], Self>(msb) };
                let lz_i  = <super::[<I32x $N>]>::sub(<super::[<I32x $N>]>::splat(31), msb_i);
                let lz    = unsafe { arch::op_opbitcast::<Self, super::[<I32x $N>]>(lz_i) };
                let is_zero = <Self as PartialOrdRegister>::eq(value, Self::ZERO);
                Self::blendv(is_zero, lz, Self::splat(32u32))
            }

            // GLSLstd450 FindILsb: position of the lowest set bit (0–31), or -1 for 0.
            #[inline(always)]
            fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {
                // FindILsb result in U32xN space: 0..=31 for non-zero, 0xFFFF_FFFF for 0.
                let lsb = unsafe { arch::glsl_op1::<Self, Self, { arch::glsl::FIND_I_LSB }, false>(value) };
                let is_zero = <Self as PartialOrdRegister>::eq(value, Self::ZERO);
                Self::blendv(is_zero, lsb, Self::splat(32u32))
            }

            // Magic-number division using the generic polyfill (precomputed multipliers).
            #[inline(always)]
            fn div_branched(
                value: Storage<Self>,
                divider: crate::Divider<Self::Element>,
            ) -> Storage<Self> {
                crate::backend::generic::polyfills::div_epu::<Self>(
                    value, divider.multiplier(), divider.shift(),
                )
            }

            #[inline(always)]
            fn div_branchfree(
                value: Storage<Self>,
                divider: crate::BranchfreeDivider<Self::Element>,
            ) -> Storage<Self> {
                crate::backend::generic::polyfills::div_epu_bf::<Self>(
                    value, divider.multiplier(), divider.shift(),
                )
            }

            #[inline(always)]
            fn divv_branchfree(
                value: Storage<Self>,
                dividers: crate::divider::vector::VectorDivider<Self>,
            ) -> Storage<Self> {
                crate::backend::generic::polyfills::divv_epu_bf::<Self>(
                    value, dividers.multipliers.0, dividers.shifts.0,
                )
            }
        }

        impl UnsignedIntegerRegister for $name {
            // GLSLstd450 FindUMsb returns floor(log2(x)) for x > 0, and undefined for x = 0.
            // Implementations typically return 0xFFFF_FFFF (-1) for x = 0.
            // ilog2p1(x) = floor(log2(x)) + 1 for x > 0, 0 for x = 0.
            // 0xFFFF_FFFF + 1 wraps to 0 in u32, so the zero case resolves correctly
            // assuming FindUMsb(0) = 0xFFFF_FFFF (common GPU behavior).
            #[inline(always)]
            fn ilog2p1(value: Storage<Self>) -> Storage<Self> {
                let msb = unsafe { arch::glsl_op1::<Self, Self, { arch::glsl::FIND_U_MSB }, false>(value) };
                Self::add(msb, Self::ONE)
            }
        }
    }};
}

decl_u32xN!(U32x2 x 2 { x:0, y:1 });
decl_u32xN!(U32x3 x 3 { x:0, y:1, z:2 });
decl_u32xN!(U32x4 x 4 { x:0, y:1, z:2, w:3 });

// ExtendRegister: place scalar in lane 0, zero the rest.
impl ExtendRegister<u32> for U32x2 {
    #[inline(always)]
    fn extend(value: u32) -> U32x2 {
        U32x2::single(value)
    }
    #[inline(always)]
    fn narrow(value: U32x2) -> u32 {
        U32x2::extract::<0>(value)
    }
}

impl ExtendRegister<u32> for U32x3 {
    #[inline(always)]
    fn extend(value: u32) -> U32x3 {
        U32x3::single(value)
    }
    #[inline(always)]
    fn narrow(value: U32x3) -> u32 {
        U32x3::extract::<0>(value)
    }
}

impl ExtendRegister<u32> for U32x4 {
    #[inline(always)]
    fn extend(value: u32) -> U32x4 {
        U32x4::single(value)
    }
    #[inline(always)]
    fn narrow(value: U32x4) -> u32 {
        U32x4::extract::<0>(value)
    }
}

// ConcatRegister: combine two N-lane vectors into a 2N-lane vector, or split.
impl ConcatRegister<u32> for U32x2 {
    #[inline(always)]
    fn concat(lo: u32, hi: u32) -> U32x2 {
        U32x2 { x: lo, y: hi }
    }
    #[inline(always)]
    fn split(value: U32x2) -> (u32, u32) {
        (value.x, value.y)
    }
}

// U32x2 can be widened to U32x4.
impl WideRegister for U32x2 {
    type Wide = U32x4;
}

impl ConcatRegister<U32x2> for U32x4 {
    #[inline(always)]
    fn concat(lo: U32x2, hi: U32x2) -> U32x4 {
        U32x4 {
            x: lo.x,
            y: lo.y,
            z: hi.x,
            w: hi.y,
        }
    }
    #[inline(always)]
    fn split(value: U32x4) -> (U32x2, U32x2) {
        (U32x2 { x: value.x, y: value.y }, U32x2 { x: value.z, y: value.w })
    }
}

// ExtendRegister impls for widening to larger vectors.
impl ExtendRegister<U32x2> for U32x3 {
    #[inline(always)]
    fn extend(value: U32x2) -> U32x3 {
        U32x3 {
            x: value.x,
            y: value.y,
            z: 0,
        }
    }
    #[inline(always)]
    fn narrow(value: U32x3) -> U32x2 {
        U32x2 { x: value.x, y: value.y }
    }
}

impl ExtendRegister<U32x2> for U32x4 {
    #[inline(always)]
    fn extend(value: U32x2) -> U32x4 {
        U32x4 {
            x: value.x,
            y: value.y,
            z: 0,
            w: 0,
        }
    }
    #[inline(always)]
    fn narrow(value: U32x4) -> U32x2 {
        U32x2 { x: value.x, y: value.y }
    }
}

impl ExtendRegister<U32x3> for U32x4 {
    #[inline(always)]
    fn extend(value: U32x3) -> U32x4 {
        U32x4 {
            x: value.x,
            y: value.y,
            z: value.z,
            w: 0,
        }
    }
    #[inline(always)]
    fn narrow(value: U32x4) -> U32x3 {
        U32x3 {
            x: value.x,
            y: value.y,
            z: value.z,
        }
    }
}
