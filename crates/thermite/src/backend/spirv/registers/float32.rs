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
        Element, ExtendRegister, FloatRegister, IndexableRegister, InterleaveRegister, LinAlg3Register,
        LinAlg4Register, MaskElement, MaskRegister, NativeCapability, NumericRegister, PartialOrdRegister,
        PermuteRegister, Register, ShuffleRegister, SignedRegister, Storage, SwizzleIndices, WideRegister, ZeroUpper,
        empty_reg, reg,
    },
    simd::Simd,
};

use crate::math::policy::{Policy, PrecisionPolicy};

use super::*;

macro_rules! decl_f32xN {
    ($name:ident x $N:literal { $($f:ident : $idx:literal),* }) => {paste::paste! {
        #[cfg_attr(target_arch = "spirv", rust_gpu::vector::v1)]
        #[derive(Debug, Clone, Copy, const_default::ConstDefault, PartialEq, PartialOrd)]
        pub struct $name {
            $(pub $f: f32,)*
        }

        #[cfg_attr(target_arch = "spirv", spirv_std_macros::spirv(matrix))]
        #[derive(Clone, Copy, Debug, const_default::ConstDefault)]
        pub struct [<F32x $N x $N>] {
            $(pub $f: $name,)*
        }

        impl [<F32x $N x $N>] {
            /// Computes the matrix inverse using `GLSLstd450 MatrixInverse`.
            ///
            /// # Safety
            /// The matrix must be invertible. Behavior is undefined for singular or
            /// near-singular matrices; GLSL `MatrixInverse` provides no error feedback.
            /// Use [`LinAlg4Register::mat4_inverse`] if a determinant check is needed.
            #[inline(always)]
            pub unsafe fn inverse(self) -> Self {
                unsafe { arch::glsl_op1::<Self, Self, { arch::glsl::MATRIX_INVERSE }, false>(self) }
            }
        }

        impl crate::simd::HasIsa for $name {
            type Native = crate::backend::spirv::SPIRV;
        }

        #[thermite_macros::inline_always]
        impl CoreRegister for $name {
            type Lanes = typenum::[<U $N>];
            type Storage = Self;
            type Mask = super::[<Mx $N>];

            const IS_EMULATED: bool = false;
            const EMPTY: Self = <Self as const_default::ConstDefault>::DEFAULT;
            const HAS_EQUAL_SIZE_MASK: bool = false;

            fn blendv(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Self {
                // OpSelect on full vector types: b where mask lane is true, a otherwise.
                // Avoids per-lane OpCompositeExtract + scalar OpSelect + OpCompositeConstruct.
                unsafe { arch::op_opselect::<Self, Storage<Self::Mask>>(mask, b, a) }
            }

            fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
                Self { $($f: if const { Z::N > $idx } { value.$f } else { 0.0 },)* }
            }

            fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {
                let truthy = Self { $($f: f32::from_bits(!0),)* };
                unsafe { arch::op_opselect::<Self, super::[<Mx $N>]>(mask, truthy, Self::EMPTY) }
            }
        }

        #[thermite_macros::inline_always]
        impl InterleaveRegister for $name {
            fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
                unsafe { arch::[<spirv_interleave $N>](a, b) }
            }
            fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
                unsafe { arch::[<spirv_deinterleave $N>](a, b) }
            }
        }

        // SPIR-V has no bitwise ops on float types, so we round-trip through the
        // same-width uint peer. OpBitcast is a zero-cost type reinterpretation.
        // `type_u` is used only as a type source for `typeof*` in the asm block.
        #[thermite_macros::inline_always]
        impl BitwiseRegister for $name {
            fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                let mut result = Self::EMPTY;
                let type_u = <super::[<U32x $N>] as const_default::ConstDefault>::DEFAULT;

                unsafe {
                    core::arch::asm!(
                        "%lhs    = OpLoad typeof*{lhs} {lhs}",
                        "%rhs    = OpLoad typeof*{rhs} {rhs}",
                        "%lhs_u  = OpBitcast typeof*{type_u} %lhs",
                        "%rhs_u  = OpBitcast typeof*{type_u} %rhs",
                        "%r_u    = OpBitwiseXor typeof*{type_u} %lhs_u %rhs_u",
                        "%result = OpBitcast typeof*{result} %r_u",
                        "OpStore {result} %result",
                        lhs    = in(reg) &lhs,
                        rhs    = in(reg) &rhs,
                        result = in(reg) &mut result,
                        type_u = in(reg) &type_u,
                    );
                }

                result
            }

            fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                let mut result = Self::EMPTY;
                let type_u = <super::[<U32x $N>] as const_default::ConstDefault>::DEFAULT;

                unsafe {
                    core::arch::asm!(
                        "%lhs    = OpLoad typeof*{lhs} {lhs}",
                        "%rhs    = OpLoad typeof*{rhs} {rhs}",
                        "%lhs_u  = OpBitcast typeof*{type_u} %lhs",
                        "%rhs_u  = OpBitcast typeof*{type_u} %rhs",
                        "%r_u    = OpBitwiseAnd typeof*{type_u} %lhs_u %rhs_u",
                        "%result = OpBitcast typeof*{result} %r_u",
                        "OpStore {result} %result",
                        lhs    = in(reg) &lhs,
                        rhs    = in(reg) &rhs,
                        result = in(reg) &mut result,
                        type_u = in(reg) &type_u,
                    );
                }

                result
            }

            fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                let mut result = Self::EMPTY;
                let type_u = <super::[<U32x $N>] as const_default::ConstDefault>::DEFAULT;

                unsafe {
                    core::arch::asm!(
                        "%lhs    = OpLoad typeof*{lhs} {lhs}",
                        "%rhs    = OpLoad typeof*{rhs} {rhs}",
                        "%lhs_u  = OpBitcast typeof*{type_u} %lhs",
                        "%rhs_u  = OpBitcast typeof*{type_u} %rhs",
                        "%r_u    = OpBitwiseOr typeof*{type_u} %lhs_u %rhs_u",
                        "%result = OpBitcast typeof*{result} %r_u",
                        "OpStore {result} %result",
                        lhs    = in(reg) &lhs,
                        rhs    = in(reg) &rhs,
                        result = in(reg) &mut result,
                        type_u = in(reg) &type_u,
                    );
                }

                result
            }

            fn not(value: Storage<Self>) -> Storage<Self> {
                let mut result = Self::EMPTY;
                let type_u = <super::[<U32x $N>] as const_default::ConstDefault>::DEFAULT;

                unsafe {
                    core::arch::asm!(
                        "%value  = OpLoad typeof*{value} {value}",
                        "%val_u  = OpBitcast typeof*{type_u} %value",
                        "%r_u    = OpNot typeof*{type_u} %val_u",
                        "%result = OpBitcast typeof*{result} %r_u",
                        "OpStore {result} %result",
                        value  = in(reg) &value,
                        result = in(reg) &mut result,
                        type_u = in(reg) &type_u,
                    );
                }

                result
            }
        }

        #[thermite_macros::inline_always]
        impl Register for $name {
            type Element = f32;
            type Signed   = super::[<I32x $N>];
            type Unsigned = super::[<U32x $N>];

            // A float register is truthy if any bit is set; use unordered != 0.0
            // so NaN lanes (all-ones) also become true.
            fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {
                unsafe { arch::op_opfunordnotequal::<super::[<Mx $N>], Self>(value, Self::EMPTY) }
            }

            fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
                cfg_select! {
                    target_feature = "Kernel" => {
                        unsafe { arch::op_opsignbitset::<super::[<Mx $N>], Self>(value) }
                    }
                    _ => {
                        let bits: Storage<<Self as FloatRegister>::Bits> = <<Self as FloatRegister>::Bits as BitCastRegister<Self>>::from_bits(value);
                        <<Self as FloatRegister>::Bits as PartialOrdRegister>::lt(bits, <Self as FloatRegister>::Bits::ZERO)
                    }
                }
            }

            fn new(value: GenericArray<f32, Self::Lanes>) -> Storage<Self> {
                Self { $($f: value[$idx],)* }
            }

            // Place the scalar in lane 0, zero all other lanes.
            fn single(value: f32) -> Storage<Self> {
                Self { $($f: if const { $idx == 0 } { value } else { 0.0 },)* }
            }

            fn splat(value: f32) -> Storage<Self> {
                Self { $($f: value,)* }
            }

            // Byte-swap each f32 lane: bitcast to uint, swap, bitcast back.
            fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
                let u = unsafe { arch::op_opbitcast::<super::[<U32x $N>], Self>(value) };
                let swapped = <super::[<U32x $N>] as Register>::swap_bytes(u);
                unsafe { arch::op_opbitcast::<Self, super::[<U32x $N>]>(swapped) }
            }

            fn extract<const I: usize>(value: Storage<Self>) -> f32 {
                unsafe { arch::op_opvectorextractdynamic::<f32, Self, usize>(value, I) }
            }

            fn insert<const I: usize>(value: Storage<Self>, element: f32) -> Storage<Self> {
                unsafe { arch::op_opvectorinsertdynamic::<Self, f32, usize>(value, element, I) }
            }

            // Single OpVectorShuffle with every output lane pointing to I.
            // Each slot gets its own operand name ({i0}, {i1}, ...) driven by $idx so
            // the macro repetition has a fragment to expand on; all bind to const I.
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
            fn broadcastv(value: Storage<Self>, idx: usize) -> Storage<Self> {
                let elem = unsafe { arch::op_opvectorextractdynamic::<f32, Self, usize>(value, idx) };
                Self::splat(elem)
            }

            fn map<F>(value: Storage<Self>, mut f: F) -> Storage<Self>
            where F: FnMut(f32) -> f32 {
                Self { $($f: f(value.$f),)* }
            }

            fn zip<F>(lhs: Storage<Self>, rhs: Storage<Self>, f: F) -> Storage<Self>
            where F: Fn(f32, f32) -> f32 {
                Self { $($f: f(lhs.$f, rhs.$f),)* }
            }

            fn fold<F>(first: f32, value: Storage<Self>, f: F) -> f32
            where F: Fn(f32, f32) -> f32 {
                let acc = first;
                $(let acc = f(acc, value.$f);)*
                acc
            }

            fn reduce<F>(value: Storage<Self>, f: F) -> f32
            where F: Fn(f32, f32) -> f32 {
                // Seed with lane 0; if const skips the redundant fold of lane 0.
                let acc = Self::extract::<0>(value);
                $(let acc = if const { $idx > 0 } { f(acc, value.$f) } else { acc };)*
                acc
            }

            // Single OpVectorShuffle with indices in descending order.
            // Each slot {r0}, {r1}, ... binds to const { $N - 1 - $idx }.
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

            // No single SPIR-V instruction for runtime-index permute; scalar fallback is used.
            const HAS_PERMUTEV: bool = false;

            // Compile-time permute: single OpVectorShuffle with literal indices.
            fn permutev_const<I: SwizzleIndices<Self::Lanes>>(value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::[<spirv_permute $N>]::<Self, I>(value) }
            }

            // Compile-time two-source swizzle: single OpVectorShuffle with literal indices.
            fn swizzle_const<I: SwizzleIndices<Self::Lanes>>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
                unsafe { arch::[<spirv_swizzle $N>]::<Self, I>(a, b) }
            }
        }

        #[thermite_macros::inline_always]
        impl NumericRegister for $name {
            const ZERO: Self = Self { $($f: 0.0,)* };
            const ONE:  Self = Self { $($f: 1.0,)* };
            const TWO:  Self = Self { $($f: 2.0,)* };
            const MIN:  Self = Self { $($f: f32::MIN,)* };
            const MAX:  Self = Self { $($f: f32::MAX,)* };

            fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                unsafe { arch::op_opfadd::<Self>(lhs, rhs) }
            }
            fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                unsafe { arch::op_opfsub::<Self>(lhs, rhs) }
            }
            fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                unsafe { arch::op_opfmul::<Self>(lhs, rhs) }
            }
            fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                unsafe { arch::op_opfdiv::<Self>(lhs, rhs) }
            }
            // OpFRem: truncating remainder (matches Rust's %)
            fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                unsafe { arch::op_opfrem::<Self>(lhs, rhs) }
            }

            fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                unsafe { arch::glsl_op2::<Self, Self, Self, { arch::glsl::F_MIN }, false>(lhs, rhs) }
            }
            fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                unsafe { arch::glsl_op2::<Self, Self, Self, { arch::glsl::F_MAX }, false>(lhs, rhs) }
            }

            fn scale(value: Storage<Self>, factor: Self::Element) -> Storage<Self> {
                unsafe { arch::op_opvectortimesscalar::<Storage<Self>, f32>(value, factor) }
            }

            // Scalar reductions via our unrolled field-level reduce/fold.
            fn min_element(value: Storage<Self>) -> f32 {
                Self::reduce(value, |a, b| if a < b { a } else { b })
            }
            fn max_element(value: Storage<Self>) -> f32 {
                Self::reduce(value, |a, b| if a > b { a } else { b })
            }
            fn sum_elements(value: Storage<Self>) -> f32 {
                Self::reduce(value, |a, b| a + b)
            }
            fn prod_elements(value: Storage<Self>) -> f32 {
                Self::reduce(value, |a, b| a * b)
            }
            fn pairwise_sum(lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self> {
                Self::pairwise_sum_impl(lo, hi)
            }

            // Lane count as a splat; 0.0, 1.0, ... per lane.
            fn offset() -> Storage<Self> { Self::splat($N as f32) }
            fn indexed() -> Storage<Self> { Self { $($f: $idx as f32,)* } }
        }

        #[thermite_macros::inline_always]
        impl PartialOrdRegister for $name {
            fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
                unsafe { arch::op_opfordequal::<super::[<Mx $N>], Self>(lhs, rhs) }
            }
            fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
                unsafe { arch::op_opfordgreaterthan::<super::[<Mx $N>], Self>(lhs, rhs) }
            }
            fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
                unsafe { arch::op_opfordgreaterthanequal::<super::[<Mx $N>], Self>(lhs, rhs) }
            }
            fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
                unsafe { arch::op_opfordlessthan::<super::[<Mx $N>], Self>(lhs, rhs) }
            }
            fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
                unsafe { arch::op_opfordlessthanequal::<super::[<Mx $N>], Self>(lhs, rhs) }
            }
            fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> {
                unsafe { arch::op_opfordnotequal::<super::[<Mx $N>], Self>(lhs, rhs) }
            }
        }

        #[thermite_macros::inline_always]
        impl CastRegister<$name> for $name {
            fn cast_from(value: Storage<Self>) -> Storage<Self> { value }
        }

        impl BitCastRegister<$name> for $name {
            fn from_bits(value: Storage<Self>) -> Storage<Self> { value }
        }

        #[thermite_macros::inline_always]
        impl SignedRegister for $name {
            const NEG_ONE:      Self = Self { $($f: -1.0,)* };
            const MIN_POSITIVE: Self = Self { $($f: f32::MIN_POSITIVE,)* };

            fn neg(value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::op_opfnegate::<Self>(value) }
            }

            fn abs(value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::glsl_op1::<Self, Self, { arch::glsl::F_ABS }, false>(value) }
            }

            // Bitwise copysign avoids the abs+blend+neg chain the default uses.
            // IEEE 754: result = (|lhs| bits) | (sign bit of rhs)
            fn copysign(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
                let lhs_u = unsafe { arch::op_opbitcast::<super::[<U32x $N>], Self>(lhs) };
                let rhs_u = unsafe { arch::op_opbitcast::<super::[<U32x $N>], Self>(rhs) };
                let abs_mask  = <super::[<U32x $N>] as Register>::splat(0x7FFF_FFFFu32);
                let sign_mask = <super::[<U32x $N>] as Register>::splat(0x8000_0000u32);
                let abs_bits = <super::[<U32x $N>] as BitwiseRegister>::bitand(lhs_u, abs_mask);
                let sign     = <super::[<U32x $N>] as BitwiseRegister>::bitand(rhs_u, sign_mask);
                let result_u = <super::[<U32x $N>] as BitwiseRegister>::bitor(abs_bits, sign);
                unsafe { arch::op_opbitcast::<Self, super::[<U32x $N>]>(result_u) }
            }

            fn is_negative(value: Storage<Self>) -> Storage<Self::Mask> {
                unsafe { cfg_select! {
                    target_feature = "Kernel" => {
                        arch::op_opsignbitset::<super::[<Mx $N>], Self>(value)
                    }
                    _ => arch::op_opfordlessthan::<super::[<Mx $N>], Self>(value, Self::ZERO),
                } }
            }
        }

        #[thermite_macros::inline_always]
        impl FloatRegister for $name {
            type Bits          = super::[<U32x $N>];
            type SignedBits    = super::[<I32x $N>];
            // No F64xN registers yet; use Self as the safe fallback.
            type ExtendedPrecision = $name;

            // GLSL Fma is a true fused multiply-add on all GPU hardware.
            const HAS_TRUE_FMA: bool = true;

            const HALF:         Self = Self { $($f: 0.5,)* };
            const NEG_ZERO:     Self = Self { $($f: -0.0,)* };
            const INFINITY:     Self = Self { $($f: f32::INFINITY,)* };
            const NEG_INFINITY: Self = Self { $($f: f32::NEG_INFINITY,)* };
            const NAN:          Self = Self { $($f: f32::NAN,)* };
            const EPSILON:      Self = Self { $($f: f32::EPSILON,)* };

            // All-ones exponent, mantissa = 0 -> ±Infinity pattern for f32.
            const EXP_MASK: super::[<U32x $N>] = super::[<U32x $N>] { $($f: 0x7F80_0000u32,)* };

            // GLSL InverseSqrt is a native hardware instruction on GPU.
            const HAS_APPROX_RSQRT: bool = true;
            const HAS_APPROX_RCP:   bool = false;

            // GLSL.std.450 provides all these transcendentals natively.
            const NATIVE_CAP: NativeCapability = NativeCapability(
                NativeCapability::LDEXP
                | NativeCapability::FREXP
                | NativeCapability::SIN
                | NativeCapability::COS
                | NativeCapability::TAN
                | NativeCapability::EXP2
                | NativeCapability::LOG2
                | NativeCapability::EXP
                | NativeCapability::LN
                | NativeCapability::POWF,
            );

            // True FMA: lhs * rhs + acc
            fn mul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
                unsafe { arch::glsl_op3::<Self, Self, Self, Self, { arch::glsl::FMA }, false>(lhs, rhs, acc) }
            }
            // -(lhs * rhs) + acc = FMA(-lhs, rhs, acc)
            fn nmul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
                unsafe { arch::glsl_op3::<Self, Self, Self, Self, { arch::glsl::FMA }, false>(Self::neg(lhs), rhs, acc) }
            }
            // lhs * rhs - acc = FMA(lhs, rhs, -acc)
            fn mul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
                unsafe { arch::glsl_op3::<Self, Self, Self, Self, { arch::glsl::FMA }, false>(lhs, rhs, Self::neg(acc)) }
            }
            // -(lhs * rhs) - acc = FMA(-lhs, rhs, -acc)
            fn nmul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {
                unsafe { arch::glsl_op3::<Self, Self, Self, Self, { arch::glsl::FMA }, false>(Self::neg(lhs), rhs, Self::neg(acc)) }
            }

            fn sqrt(value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::glsl_op1::<Self, Self, { arch::glsl::SQRT }, false>(value) }
            }

            // GLSL InverseSqrt: 1/sqrt(x), native GPU instruction.
            fn rsqrt(value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::glsl_op1::<Self, Self, { arch::glsl::INVERSE_SQRT }, false>(value) }
            }

            fn floor(value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::glsl_op1::<Self, Self, { arch::glsl::FLOOR }, false>(value) }
            }

            fn ceil(value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::glsl_op1::<Self, Self, { arch::glsl::CEIL }, false>(value) }
            }

            // GLSLstd450 Round: nearest, tie-breaking is implementation-defined per GPU vendor.
            // Rust's f32::round() rounds ties away from zero; if strict banker's rounding is
            // ever needed, swap to arch::glsl::ROUND_EVEN (opcode 2).
            fn round(value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::glsl_op1::<Self, Self, { arch::glsl::ROUND }, false>(value) }
            }

            fn trunc(value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::glsl_op1::<Self, Self, { arch::glsl::TRUNC }, false>(value) }
            }

            // GLSL Fract is native; avoids the sub(v, trunc(v)) default chain.
            fn fract(value: Storage<Self>) -> Storage<Self> {
                unsafe { arch::glsl_op1::<Self, Self, { arch::glsl::FRACT }, false>(value) }
            }

            // GLSL FMix: a + (b - a) * t, with hardware guarantees. Beats the FMA fallback.
            fn mix(a: Storage<Self>, b: Storage<Self>, t: Storage<Self>) -> Storage<Self> {
                unsafe { arch::glsl_op3::<Self, Self, Self, Self, { arch::glsl::F_MIX }, false>(a, b, t) }
            }

            // TODO!!!! Check if we can put the condition in the generic!
            // do not let me forget about this once integer registers are complete
            // and we can test compiles.

            unsafe fn native_sin<P: Policy>(value: Storage<Self>) -> Storage<Self> {
                if const { P::POLICY.precision.lt(PrecisionPolicy::Average) } {
                    unsafe { arch::glsl_op1::<Self, Self, { arch::glsl::SIN }, true>(value) }
                } else {
                    unsafe { arch::glsl_op1::<Self, Self, { arch::glsl::SIN }, false>(value) }
                }
            }

            unsafe fn native_cos<P: Policy>(value: Storage<Self>) -> Storage<Self> {
                if const { P::POLICY.precision.lt(PrecisionPolicy::Average) } {
                    unsafe { arch::glsl_op1::<Self, Self, { arch::glsl::COS }, true>(value) }
                } else {
                    unsafe { arch::glsl_op1::<Self, Self, { arch::glsl::COS }, false>(value) }
                }
            }

            // Emit both with a single call each rather than sharing a combined instruction;
            // the GPU scheduler can still issue them in parallel.
            unsafe fn native_sin_cos<P: Policy>(value: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
                unsafe { (Self::native_sin::<P>(value), Self::native_cos::<P>(value)) }
            }

            unsafe fn native_tan<P: Policy>(value: Storage<Self>) -> Storage<Self> {
                if const { P::POLICY.precision.lt(PrecisionPolicy::Average) } {
                    unsafe { arch::glsl_op1::<Self, Self, { arch::glsl::TAN }, true>(value) }
                } else {
                    unsafe { arch::glsl_op1::<Self, Self, { arch::glsl::TAN }, false>(value) }
                }
            }

            unsafe fn native_exp2<P: Policy>(value: Storage<Self>) -> Storage<Self> {
                if const { P::POLICY.precision.lt(PrecisionPolicy::Average) } {
                    unsafe { arch::glsl_op1::<Self, Self, { arch::glsl::EXP2 }, true>(value) }
                } else {
                    unsafe { arch::glsl_op1::<Self, Self, { arch::glsl::EXP2 }, false>(value) }
                }
            }

            unsafe fn native_log2<P: Policy>(value: Storage<Self>) -> Storage<Self> {
                if const { P::POLICY.precision.lt(PrecisionPolicy::Average) } {
                    unsafe { arch::glsl_op1::<Self, Self, { arch::glsl::LOG2 }, true>(value) }
                } else {
                    unsafe { arch::glsl_op1::<Self, Self, { arch::glsl::LOG2 }, false>(value) }
                }
            }

            unsafe fn native_exp<P: Policy>(value: Storage<Self>) -> Storage<Self> {
                if const { P::POLICY.precision.lt(PrecisionPolicy::Average) } {
                    unsafe { arch::glsl_op1::<Self, Self, { arch::glsl::EXP }, true>(value) }
                } else {
                    unsafe { arch::glsl_op1::<Self, Self, { arch::glsl::EXP }, false>(value) }
                }
            }

            // GLSL LOG is the natural logarithm (ln).
            unsafe fn native_ln<P: Policy>(value: Storage<Self>) -> Storage<Self> {
                if const { P::POLICY.precision.lt(PrecisionPolicy::Average) } {
                    unsafe { arch::glsl_op1::<Self, Self, { arch::glsl::LOG }, true>(value) }
                } else {
                    unsafe { arch::glsl_op1::<Self, Self, { arch::glsl::LOG }, false>(value) }
                }
            }

            unsafe fn native_powf<P: Policy>(base: Storage<Self>, exp: Storage<Self>) -> Storage<Self> {
                if const { P::POLICY.precision.lt(PrecisionPolicy::Average) } {
                    unsafe { arch::glsl_op2::<Self, Self, Self, { arch::glsl::POW }, true>(base, exp) }
                } else {
                    unsafe { arch::glsl_op2::<Self, Self, Self, { arch::glsl::POW }, false>(base, exp) }
                }
            }

            // GLSL Ldexp: x * 2^exp, with int exponent lanes.
            unsafe fn native_ldexp(value: Storage<Self>, exp: Storage<Self::SignedBits>) -> Storage<Self> {
                unsafe { arch::glsl_op2::<Self, Self, super::[<I32x $N>], { arch::glsl::LDEXP }, false>(value, exp) }
            }

            // GLSL FrexpStruct: splits x into (significand in [0.5, 1), exponent).
            unsafe fn native_frexp(value: Storage<Self>) -> (Storage<Self>, Storage<Self::SignedBits>) {
                unsafe { arch::glsl_frexp::<Self, super::[<I32x $N>]>(value) }
            }

            // Native SPIR-V classification ops; cheaper than the bitwise-trick defaults on GPU.
            fn is_nan(value: Storage<Self>) -> Storage<Self::Mask> {
                unsafe { arch::op_opisnan::<super::[<Mx $N>], Self>(value) }
            }

            fn is_infinite(value: Storage<Self>) -> Storage<Self::Mask> {
                unsafe { arch::op_opisinf::<super::[<Mx $N>], Self>(value) }
            }

            fn is_finite(value: Storage<Self>) -> Storage<Self::Mask> {
                unsafe { arch::op_opisfinite::<super::[<Mx $N>], Self>(value) }
            }

            fn is_normal(value: Storage<Self>) -> Storage<Self::Mask> {
                unsafe { arch::op_opisnormal::<super::[<Mx $N>], Self>(value) }
            }
        }
    }};
}

decl_f32xN!(F32x2 x 2 { x:0, y:1 });
decl_f32xN!(F32x3 x 3 { x:0, y:1, z:2 });
decl_f32xN!(F32x4 x 4 { x:0, y:1, z:2, w:3 });

impl F32x2 {
    #[inline(always)]
    fn pairwise_sum_impl(lo: Self, hi: Self) -> Self {
        Self {
            x: lo.x + lo.y,
            y: hi.x + hi.y,
        }
    }
}
impl F32x3 {
    #[inline(always)]
    fn pairwise_sum_impl(lo: Self, hi: Self) -> Self {
        Self {
            x: lo.x + lo.y,
            y: hi.x + hi.y,
            z: lo.z + hi.z,
        }
    }
}
impl F32x4 {
    #[inline(always)]
    fn pairwise_sum_impl(lo: Self, hi: Self) -> Self {
        Self {
            x: lo.x + lo.y,
            y: lo.z + lo.w,
            z: hi.x + hi.y,
            w: hi.z + hi.w,
        }
    }
}

macro_rules! impl_f32_casts {
    ($f:ident <=> $i:ident, $u:ident) => {
        #[thermite_macros::inline_always]
        impl CastRegister<$i> for $f {
            fn cast_from(value: $i) -> $f {
                unsafe { arch::op_opconvertstof::<$f, $i>(value) }
            }
        }

        #[thermite_macros::inline_always]
        impl CastRegister<$u> for $f {
            fn cast_from(value: $u) -> $f {
                unsafe { arch::op_opconvertutof::<$f, $u>(value) }
            }
        }

        #[thermite_macros::inline_always]
        impl CastRegister<$f> for $i {
            fn cast_from(value: $f) -> $i {
                unsafe { arch::op_opconvertftos::<$i, $f>(value) }
            }
        }

        #[thermite_macros::inline_always]
        impl CastRegister<$f> for $u {
            fn cast_from(value: $f) -> $u {
                unsafe { arch::op_opconvertftou::<$u, $f>(value) }
            }
        }

        #[thermite_macros::inline_always]
        impl BitCastRegister<$i> for $f {
            fn from_bits(value: $i) -> $f {
                unsafe { arch::op_opbitcast::<$f, $i>(value) }
            }
        }

        #[thermite_macros::inline_always]
        impl BitCastRegister<$u> for $f {
            fn from_bits(value: $u) -> $f {
                unsafe { arch::op_opbitcast::<$f, $u>(value) }
            }
        }

        #[thermite_macros::inline_always]
        impl BitCastRegister<$f> for $i {
            fn from_bits(value: $f) -> $i {
                unsafe { arch::op_opbitcast::<$i, $f>(value) }
            }
        }

        #[thermite_macros::inline_always]
        impl BitCastRegister<$f> for $u {
            fn from_bits(value: $f) -> $u {
                unsafe { arch::op_opbitcast::<$u, $f>(value) }
            }
        }
    };
}

impl_f32_casts!(F32x2 <=> I32x2, U32x2);
impl_f32_casts!(F32x3 <=> I32x3, U32x3);
impl_f32_casts!(F32x4 <=> I32x4, U32x4);

#[thermite_macros::inline_always]
impl ExtendRegister<f32> for F32x2 {
    fn extend(value: f32) -> F32x2 {
        F32x2::single(value)
    }

    fn narrow(value: F32x2) -> f32 {
        F32x2::extract::<0>(value)
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<f32> for F32x3 {
    fn extend(value: f32) -> F32x3 {
        F32x3::single(value)
    }

    fn narrow(value: F32x3) -> f32 {
        F32x3::extract::<0>(value)
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<f32> for F32x4 {
    fn extend(value: f32) -> F32x4 {
        F32x4::single(value)
    }

    fn narrow(value: F32x4) -> f32 {
        F32x4::extract::<0>(value)
    }
}

#[thermite_macros::inline_always]
impl ConcatRegister<f32> for F32x2 {
    fn concat(lo: f32, hi: f32) -> F32x2 {
        F32x2 { x: lo, y: hi }
    }

    fn split(value: F32x2) -> (f32, f32) {
        (value.x, value.y)
    }
}

#[thermite_macros::inline_always]
impl WideRegister for F32x2 {
    type Wide = F32x4;
}

impl ConcatRegister<F32x2> for F32x4 {
    fn concat(lo: F32x2, hi: F32x2) -> F32x4 {
        F32x4 {
            x: lo.x,
            y: lo.y,
            z: hi.x,
            w: hi.y,
        }
    }

    fn split(value: F32x4) -> (F32x2, F32x2) {
        (F32x2 { x: value.x, y: value.y }, F32x2 { x: value.z, y: value.w })
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<F32x2> for F32x3 {
    fn extend(value: F32x2) -> F32x3 {
        F32x3 {
            x: value.x,
            y: value.y,
            z: 0.0,
        }
    }

    fn narrow(value: F32x3) -> F32x2 {
        F32x2 { x: value.x, y: value.y }
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<F32x2> for F32x4 {
    fn extend(value: F32x2) -> F32x4 {
        F32x4 {
            x: value.x,
            y: value.y,
            z: 0.0,
            w: 0.0,
        }
    }

    fn narrow(value: F32x4) -> F32x2 {
        F32x2 { x: value.x, y: value.y }
    }
}

#[thermite_macros::inline_always]
impl ExtendRegister<F32x3> for F32x4 {
    fn extend(value: F32x3) -> F32x4 {
        F32x4 {
            x: value.x,
            y: value.y,
            z: value.z,
            w: 0.0,
        }
    }

    fn narrow(value: F32x4) -> F32x3 {
        F32x3 {
            x: value.x,
            y: value.y,
            z: value.z,
        }
    }
}

// Scalar difference-of-products: (a*b) - (c*d) with error compensation via FMA.
// Used by both F32x3 and F32x4 cross3 `FAST = false` paths. GPUs execute lanes
// independently, so direct element access is always preferable to permutes.
#[inline(always)]
fn cross_dop(a: f32, b: f32, c: f32, d: f32) -> f32 {
    let cd = c * d;
    <f32 as FloatRegister>::mul_sub(a, b, cd) + <f32 as FloatRegister>::nmul_add(c, d, cd)
}

// LinAlg3Register impls for F32x3 and F32x4.
//
// Both override cross3 with scalar per-component implementations.
// F32x3: uses GLSLstd450 Cross for the `FAST = true` path.
// F32x4: uses the plain scalar formula for the `FAST = true` path (w zeroed).
macro_rules! impl_spirv_linalg3 {
    // Element-only reduction methods, shared between F32x3 and F32x4.
    // F32x3: all three lanes are the full register.
    // F32x4: w lane is ignored (only x, y, z are summed/compared).
    (@reductions) => {
        #[inline(always)]
        fn min_element3(value: Storage<Self>) -> f32 {
            f32::min(f32::min(value.x, value.y), value.z)
        }

        #[inline(always)]
        fn max_element3(value: Storage<Self>) -> f32 {
            f32::max(f32::max(value.x, value.y), value.z)
        }

        #[inline(always)]
        fn sum_elements3(value: Storage<Self>) -> f32 {
            value.x + value.y + value.z
        }

        #[inline(always)]
        fn prod_elements3(value: Storage<Self>) -> f32 {
            value.x * value.y * value.z
        }

        #[inline(always)]
        fn dot3(lhs: Storage<Self>, rhs: Storage<Self>) -> f32 {
            <f32 as FloatRegister>::mul_add(
                lhs.x,
                rhs.x,
                <f32 as FloatRegister>::mul_add(lhs.y, rhs.y, lhs.z * rhs.z),
            )
        }
    };
}

#[thermite_macros::inline_always]
impl LinAlg3Register for F32x3 {
    impl_spirv_linalg3!(@reductions);

    fn cross3<const FAST: bool>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        if !FAST {
            F32x3 {
                x: cross_dop(lhs.y, rhs.z, lhs.z, rhs.y),
                y: cross_dop(lhs.z, rhs.x, lhs.x, rhs.z),
                z: cross_dop(lhs.x, rhs.y, lhs.y, rhs.x),
            }
        } else {
            // GLSLstd450 Cross: single native instruction for 3-component vectors.
            unsafe { arch::glsl_op2::<Self, Self, Self, { arch::glsl::CROSS }, false>(lhs, rhs) }
        }
    }
}

#[thermite_macros::inline_always]
impl LinAlg3Register for F32x4 {
    impl_spirv_linalg3!(@reductions);

    fn cross3<const FAST: bool>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        if !FAST {
            F32x4 {
                x: cross_dop(lhs.y, rhs.z, lhs.z, rhs.y),
                y: cross_dop(lhs.z, rhs.x, lhs.x, rhs.z),
                z: cross_dop(lhs.x, rhs.y, lhs.y, rhs.x),
                w: 0.0,
            }
        } else {
            F32x4 {
                x: <f32 as FloatRegister>::mul_sub(lhs.y, rhs.z, lhs.z * rhs.y),
                y: <f32 as FloatRegister>::mul_sub(lhs.z, rhs.x, lhs.x * rhs.z),
                z: <f32 as FloatRegister>::mul_sub(lhs.x, rhs.y, lhs.y * rhs.x),
                w: 0.0,
            }
        }
    }
}

// LinAlg4Register for F32x4.
//
// Matrix ops use native SPIR-V/GLSL instructions via #[spirv(matrix)] structs.
// Quaternion ops are rewritten as scalar FMA chains - the SIMD default uses broadcasts
// and sign-XOR shuffles that are wasteful on GPU SIMT where each lane is independent.
#[rustfmt::skip] #[thermite_macros::inline_always]
impl LinAlg4Register for F32x4 {
    fn dot4(lhs: Storage<Self>, rhs: Storage<Self>) -> Self::Element {
        <f32 as FloatRegister>::mul_add(lhs.x, rhs.x, <f32 as FloatRegister>::mul_add(lhs.y, rhs.y, <f32 as FloatRegister>::mul_add(lhs.z, rhs.z, lhs.w * rhs.w)))
    }

    // Scalar FMA chain for quaternion product lhs * rhs (components: x, y, z, w).
    // Each output component is a 4-term signed dot product, evaluated with chained FMAs
    // to minimize intermediate rounding without the sign-XOR shuffle the SIMD default uses.
    //
    //   result.x = +lhs.w*rhs.x + lhs.x*rhs.w + lhs.y*rhs.z - lhs.z*rhs.y
    //   result.y = +lhs.w*rhs.y - lhs.x*rhs.z + lhs.y*rhs.w + lhs.z*rhs.x
    //   result.z = +lhs.w*rhs.z + lhs.x*rhs.y - lhs.y*rhs.x + lhs.z*rhs.w
    //   result.w = +lhs.w*rhs.w - lhs.x*rhs.x - lhs.y*rhs.y - lhs.z*rhs.z
    //
    // `FAST` is ignored.
    fn quat4_product<const FAST: bool>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        use FloatRegister as FR;
        F32x4 {
            x: <f32 as FR>::mul_add(lhs.w, rhs.x, <f32 as FR>::mul_add(lhs.x, rhs.w, <f32 as FR>::mul_sub( lhs.y, rhs.z,  lhs.z * rhs.y))),
            y: <f32 as FR>::mul_add(lhs.w, rhs.y, <f32 as FR>::nmul_add(lhs.x, rhs.z, <f32 as FR>::mul_add( lhs.y, rhs.w,  lhs.z * rhs.x))),
            z: <f32 as FR>::mul_add(lhs.w, rhs.z, <f32 as FR>::mul_add( lhs.x, rhs.y, <f32 as FR>::nmul_add(lhs.y, rhs.x,  lhs.z * rhs.w))),
            w: <f32 as FR>::mul_add(lhs.w, rhs.w, <f32 as FR>::nmul_add(lhs.x, rhs.x, <f32 as FR>::nmul_sub(lhs.y, rhs.y,  lhs.z * rhs.z))),
        }
    }

    // Giesen fast quat-vec3 rotate: v + 2w(q x v) + 2(q x (q x v)).
    // Replaces the SIMD default which broadcasts w into a full register before multiplying.
    fn quat4_vec3_product<const FAST: bool>(q: Storage<Self>, v: Storage<Self>) -> Storage<Self> {
        let w = q.w;
        // t = 2 * cross(q, v)
        let t = Self::cross3::<FAST>(q, v);
        let t = F32x4 { x: t.x + t.x, y: t.y + t.y, z: t.z + t.z, w: 0.0 };
        // result = v + w*t + cross(q, t)
        let ct = Self::cross3::<FAST>(q, t);
        F32x4 {
            x: v.x + <f32 as FloatRegister>::mul_add(w, t.x, ct.x),
            y: v.y + <f32 as FloatRegister>::mul_add(w, t.y, ct.y),
            z: v.z + <f32 as FloatRegister>::mul_add(w, t.z, ct.z),
            w: v.w,
        }
    }

    // OpTranspose: single hardware instruction, replaces the 4-interleave default.
    fn mat4_transpose(m: &[Storage<Self>; 4]) -> [Storage<Self>; 4] {
        let mat = F32x4x4 { x: m[0], y: m[1], z: m[2], w: m[3] };
        let result = unsafe { arch::op_optranspose::<F32x4x4>(mat) };
        [result.x, result.y, result.z, result.w]
    }

    // Column-major:    OpMatrixTimesVector(M, v)  =  M * v
    // Row-major:       OpVectorTimesMatrix(v, M)  =  M * v  when M stores rows as columns,
    //                  since result[j] = dot(v, col_j(M)) = dot(v, row_j)
    fn mat4_vec4_product<const COLUMN_MAJOR: bool, const N: usize>(
        cols: &[Storage<Self>; 4],
        vectors: &[Storage<Self>; N],
    ) -> [Storage<Self>; N] {
        // SIMT: each invocation is one lane, so just apply the native matrix op
        // to every vector (the matrix is built once).
        let mat = F32x4x4 { x: cols[0], y: cols[1], z: cols[2], w: cols[3] };
        let mut out = [Self::EMPTY; N];
        let mut i = 0;
        while i < N {
            out[i] = if const { COLUMN_MAJOR } {
                unsafe { arch::op_opmatrixtimesvector::<F32x4, F32x4x4>(mat, vectors[i]) }
            } else {
                unsafe { arch::op_opvectortimesmatrix::<F32x4, F32x4x4>(vectors[i], mat) }
            };
            i += 1;
        }
        out
    }

    // Swap lhs/rhs for row-major: OpMatrixTimesMatrix(rhs_stored, lhs_stored)
    //   = rhs^T * lhs^T  (row-stored = transposed)  =  (lhs * rhs)^T
    // The result cols are the rows of (lhs*rhs), which is the correct row-major output.
    fn mat4_product<const COLUMN_MAJOR: bool>(
        lhs: &[Storage<Self>; 4],
        rhs: &[Storage<Self>; 4],
    ) -> [Storage<Self>; 4] {
        let (lhs, rhs) = if const { COLUMN_MAJOR } { (lhs, rhs) } else { (rhs, lhs) };
        let lhs_mat = F32x4x4 { x: lhs[0], y: lhs[1], z: lhs[2], w: lhs[3] };
        let rhs_mat = F32x4x4 { x: rhs[0], y: rhs[1], z: rhs[2], w: rhs[3] };
        let result = unsafe { arch::op_opmatrixtimesmatrix::<F32x4x4>(lhs_mat, rhs_mat) };
        [result.x, result.y, result.z, result.w]
    }

    // GLSL Determinant + MatrixInverse in one asm block (via arch::glsl_determinant_and_inverse).
    // Returns the determinant and leaves `m` untouched for an exactly-singular
    // matrix (GLSL MatrixInverse is undefined there); the caller inspects the det.
    // `FAST` is inert here: GLSL `MatrixInverse` is one opaque intrinsic,
    // so there is no difference-of-products for it to select a lowering for.
    fn mat4_inverse<const FAST: bool>(m: &mut [Storage<Self>; 4]) -> Self::Element {
        let mat = F32x4x4 { x: m[0], y: m[1], z: m[2], w: m[3] };
        let (d, result): (f32, F32x4x4) = unsafe { arch::glsl_determinant_and_inverse(mat) };
        if crate::likely(d != 0.0) {
            m[0] = result.x;
            m[1] = result.y;
            m[2] = result.z;
            m[3] = result.w;
        }
        d
    }

    fn mat4_det<const FAST: bool>(cols: &[Storage<Self>; 4]) -> Self::Element {
        let mat = F32x4x4 { x: cols[0], y: cols[1], z: cols[2], w: cols[3] };
        unsafe { arch::glsl_determinant(mat) }
    }
}
