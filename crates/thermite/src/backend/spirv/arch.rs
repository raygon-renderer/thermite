use crate::register::SwizzleIndices;
use const_default::ConstDefault;
use generic_array::typenum;

/// Generate a typed wrapper for a core SPIR-V instruction.
///
/// Syntax:
/// ```ignore
/// spirv_op!(OpName[T0, T1, ...](arg0: T0, arg1: T1, ...) -> TOut);
/// ```
///
/// Type params are declared explicitly in `[...]` to avoid duplication issues when multiple
/// args share the same type. `TOut` must always appear in `[...]` and be `Copy + ConstDefault`.
///
/// Examples:
/// ```ignore
/// spirv_op!(OpFNegate[T](a: T) -> T);
/// spirv_op!(OpFAdd[T](a: T, b: T) -> T);
/// spirv_op!(OpConvertFToS[TIn, TOut](a: TIn) -> TOut);
/// spirv_op!(OpSelect[TM, T](mask: TM, a: T, b: T) -> T);
/// ```
macro_rules! spirv_op {
    // Single output: -> TOut
    ($Op:ident [ $TOut:ident $(, $TExtra:ident)* ] ( $($arg:ident : $T:ident),+ )) => {
        paste::paste! {
            #[inline(always)]
            pub unsafe fn [<op_ $Op:lower>]<$TOut: Copy + ConstDefault $(, $TExtra: Copy)*>(
                $($arg: $T,)+
            ) -> $TOut {
                let mut result = $TOut::DEFAULT;
                unsafe {
                    core::arch::asm!(
                        $(concat!("%", stringify!($arg), " = OpLoad typeof*{", stringify!($arg), "} {", stringify!($arg), "}"),)+
                        concat!("%result = ", stringify!($Op), " typeof*{result}", $(" %", stringify!($arg),)+ ""),
                        "OpStore {result} %result",
                        $($arg = in(reg) &$arg,)+
                        result = in(reg) &mut result,
                    );
                }
                result
            }
        }
    };
}

// Core SPIR-V ops used by Thermite's register traits.
// Add entries here as register implementations need them.
// Instruction names must match the SPIR-V spec exactly, since rust-gpu resolves them by name.
spirv_op!(OpFNegate[T](a: T));
spirv_op!(OpSNegate[T](a: T));
spirv_op!(OpNot[T](a: T));
spirv_op!(OpBitwiseAnd[T](a: T, b: T));
spirv_op!(OpBitwiseOr[T](a: T, b: T));
spirv_op!(OpBitwiseXor[T](a: T, b: T));
spirv_op!(OpShiftRightLogical[T, TS](a: T, b: TS));
spirv_op!(OpShiftRightArithmetic[T, TS](a: T, b: TS));
spirv_op!(OpShiftLeftLogical[T, TS](a: T, b: TS));
spirv_op!(OpIAdd[T](a: T, b: T));
spirv_op!(OpISub[T](a: T, b: T));
spirv_op!(OpIMul[T](a: T, b: T));
spirv_op!(OpFAdd[T](a: T, b: T));
spirv_op!(OpFSub[T](a: T, b: T));
spirv_op!(OpFMul[T](a: T, b: T));
spirv_op!(OpFDiv[T](a: T, b: T));
spirv_op!(OpFRem[T](a: T, b: T));
spirv_op!(OpFMod[T](a: T, b: T));
spirv_op!(OpSelect[T, TMask](mask: TMask, a: T, b: T));
spirv_op!(OpDot[TOut, T](a: T, b: T));
spirv_op!(OpConvertFToS[TOut, TIn](a: TIn));
spirv_op!(OpConvertFToU[TOut, TIn](a: TIn));
spirv_op!(OpConvertSToF[TOut, TIn](a: TIn));
spirv_op!(OpConvertUToF[TOut, TIn](a: TIn));
spirv_op!(OpFConvert[TOut, TIn](a: TIn));
spirv_op!(OpSConvert[TOut, TIn](a: TIn));
spirv_op!(OpUConvert[TOut, TIn](a: TIn));
spirv_op!(OpBitcast[TOut, TIn](a: TIn));

// Integer arithmetic
spirv_op!(OpSDiv[T](a: T, b: T));
spirv_op!(OpUDiv[T](a: T, b: T));
spirv_op!(OpSRem[T](a: T, b: T));
spirv_op!(OpSMod[T](a: T, b: T));
spirv_op!(OpUMod[T](a: T, b: T));

// Float comparisons - ordered (returns bool/bvecN; false if either operand is NaN)
spirv_op!(OpFOrdEqual[TOut, TIn](a: TIn, b: TIn));
spirv_op!(OpFOrdNotEqual[TOut, TIn](a: TIn, b: TIn));
spirv_op!(OpFOrdLessThan[TOut, TIn](a: TIn, b: TIn));
spirv_op!(OpFOrdGreaterThan[TOut, TIn](a: TIn, b: TIn));
spirv_op!(OpFOrdLessThanEqual[TOut, TIn](a: TIn, b: TIn));
spirv_op!(OpFOrdGreaterThanEqual[TOut, TIn](a: TIn, b: TIn));

// Float comparisons - unordered (returns true if either operand is NaN)
spirv_op!(OpFUnordEqual[TOut, TIn](a: TIn, b: TIn));
spirv_op!(OpFUnordNotEqual[TOut, TIn](a: TIn, b: TIn));
spirv_op!(OpFUnordLessThan[TOut, TIn](a: TIn, b: TIn));
spirv_op!(OpFUnordGreaterThan[TOut, TIn](a: TIn, b: TIn));
spirv_op!(OpFUnordLessThanEqual[TOut, TIn](a: TIn, b: TIn));
spirv_op!(OpFUnordGreaterThanEqual[TOut, TIn](a: TIn, b: TIn));

// Integer comparisons
spirv_op!(OpIEqual[TOut, TIn](a: TIn, b: TIn));
spirv_op!(OpINotEqual[TOut, TIn](a: TIn, b: TIn));
spirv_op!(OpSLessThan[TOut, TIn](a: TIn, b: TIn));
spirv_op!(OpSGreaterThan[TOut, TIn](a: TIn, b: TIn));
spirv_op!(OpSLessThanEqual[TOut, TIn](a: TIn, b: TIn));
spirv_op!(OpSGreaterThanEqual[TOut, TIn](a: TIn, b: TIn));
spirv_op!(OpULessThan[TOut, TIn](a: TIn, b: TIn));
spirv_op!(OpUGreaterThan[TOut, TIn](a: TIn, b: TIn));
spirv_op!(OpULessThanEqual[TOut, TIn](a: TIn, b: TIn));
spirv_op!(OpUGreaterThanEqual[TOut, TIn](a: TIn, b: TIn));

// Logical ops (operate on bool scalars or bool vectors - distinct from bitwise integer ops)
spirv_op!(OpLogicalNot[T](a: T));
spirv_op!(OpLogicalAnd[T](a: T, b: T));
spirv_op!(OpLogicalOr[T](a: T, b: T));
spirv_op!(OpLogicalEqual[T](a: T, b: T));
spirv_op!(OpLogicalNotEqual[T](a: T, b: T));

// Reduction ops - scalar bool result from a bool vector
spirv_op!(OpAny[TOut, TIn](a: TIn));
spirv_op!(OpAll[TOut, TIn](a: TIn));

// Dynamic lane access - runtime-index extract/insert on any vector type.
// OpVectorExtractDynamic: extract a single component at a runtime index.
// OpVectorInsertDynamic: produce a new vector with one component replaced.
spirv_op!(OpVectorExtractDynamic[TOut, TVec, TIdx](vec: TVec, idx: TIdx));
spirv_op!(OpVectorInsertDynamic[TVec, TScalar, TIdx](vec: TVec, scalar: TScalar, idx: TIdx));

// Vector/scalar mixed arithmetic
// Multiplies a vector by a scalar float without needing a splat.
spirv_op!(OpVectorTimesScalar[TV, TS](v: TV, s: TS));

// Bit counting - maps to IntegerRegister::leading_zeros / trailing_zeros / count_ones
// Note: these are core SPIR-V ops, not GLSLstd450 (GLSLstd450 has FindILsb/FindUMsb for a subset)
spirv_op!(OpBitCount[T](a: T));
spirv_op!(OpBitReverse[T](a: T));

// Float classification - native SPIR-V equivalents of FloatRegister::is_nan/is_infinite/is_normal.
// On GPU these may be cheaper than the bitwise-trick implementations used on x86.
// OpSignBitSet is the direct MSB read used by select_negative/msb_to_mask.
spirv_op!(OpIsNan[TOut, TIn](a: TIn));
spirv_op!(OpIsInf[TOut, TIn](a: TIn));
spirv_op!(OpIsFinite[TOut, TIn](a: TIn));
spirv_op!(OpIsNormal[TOut, TIn](a: TIn));
spirv_op!(OpSignBitSet[TOut, TIn](a: TIn));

// Ordered/unordered pair-checks - true if both operands are non-NaN / at least one is NaN.
// Cheaper than two OpIsNan + logical combine when you need the pair test directly.
spirv_op!(OpOrdered[TOut, TIn](a: TIn, b: TIn));
spirv_op!(OpUnordered[TOut, TIn](a: TIn, b: TIn));

// Saturating integer conversion - maps to CastRegister with saturation semantics.
// x86 equivalent: PACKSS / PACKUS. Useful for quantization and fixed-point workflows.
spirv_op!(OpSatConvertSToU[TOut, TIn](a: TIn));
spirv_op!(OpSatConvertUToS[TOut, TIn](a: TIn));

// Quantize f32 to f16 precision while remaining in f32 storage.
// Equivalent to `(f32)(f16)x` - rounds mantissa and flushes out-of-range to inf.
// Useful for mixed-precision / quantization workflows on both GPU and CPU (via software).
spirv_op!(OpQuantizeToF16[T](a: T));

// Matrix operations - operands must be #[spirv(matrix)]-annotated structs (OpTypeMatrix).
// TV is the vector element type, TM is the square matrix type.
spirv_op!(OpTranspose[TM](a: TM));
spirv_op!(OpMatrixTimesVector[TV, TM](mat: TM, vec: TV));
spirv_op!(OpVectorTimesMatrix[TV, TM](vec: TV, mat: TM));
spirv_op!(OpMatrixTimesMatrix[TM](lhs: TM, rhs: TM));
spirv_op!(OpMatrixTimesScalar[TM, TS](mat: TM, scalar: TS));

// Widening multiply - produce a (lo, hi) pair covering the full 2N-bit result.
// x86 equivalent: MULX / PMULHW+PMULLW pair. Maps to IntegerRegister::mulhi / mullo pair.
// These return an OpTypeStruct{lo, hi} so they need concrete wrapper functions below.

/// `OpUMulExtended` - unsigned widening multiply returning `(lo, hi)`.
/// Both outputs have the same type as the inputs.
#[inline(always)]
pub unsafe fn spirv_umul_extended<T: Copy + ConstDefault>(a: T, b: T) -> (T, T) {
    let mut lo = T::DEFAULT;
    let mut hi = T::DEFAULT;
    unsafe {
        core::arch::asm!(
            "%a      = OpLoad typeof*{a} {a}",
            "%b      = OpLoad typeof*{b} {b}",
            "%_pair  = OpUMulExtended typeof*{lo} %a %b",
            "%lo     = OpCompositeExtract typeof*{lo} %_pair 0",
            "%hi     = OpCompositeExtract typeof*{hi} %_pair 1",
            "OpStore {lo} %lo",
            "OpStore {hi} %hi",
            a  = in(reg) &a,
            b  = in(reg) &b,
            lo = in(reg) &mut lo,
            hi = in(reg) &mut hi,
        );
    }
    (lo, hi)
}

/// `OpSMulExtended` - signed widening multiply returning `(lo, hi)`.
/// Both outputs have the same type as the inputs.
#[inline(always)]
pub unsafe fn spirv_smul_extended<T: Copy + ConstDefault>(a: T, b: T) -> (T, T) {
    let mut lo = T::DEFAULT;
    let mut hi = T::DEFAULT;
    unsafe {
        core::arch::asm!(
            "%a      = OpLoad typeof*{a} {a}",
            "%b      = OpLoad typeof*{b} {b}",
            "%_pair  = OpSMulExtended typeof*{lo} %a %b",
            "%lo     = OpCompositeExtract typeof*{lo} %_pair 0",
            "%hi     = OpCompositeExtract typeof*{hi} %_pair 1",
            "OpStore {lo} %lo",
            "OpStore {hi} %hi",
            a  = in(reg) &a,
            b  = in(reg) &b,
            lo = in(reg) &mut lo,
            hi = in(reg) &mut hi,
        );
    }
    (lo, hi)
}

/// `OpIAddCarry` - add with carry out. Returns `(sum, carry)` where carry is 0 or 1.
/// x86 equivalent: `ADDC`. Useful for multi-precision integer arithmetic.
#[inline(always)]
pub unsafe fn spirv_iadd_carry<T: Copy + ConstDefault>(a: T, b: T) -> (T, T) {
    let mut sum = T::DEFAULT;
    let mut carry = T::DEFAULT;
    unsafe {
        core::arch::asm!(
            "%a      = OpLoad typeof*{a} {a}",
            "%b      = OpLoad typeof*{b} {b}",
            "%_pair  = OpIAddCarry typeof*{sum} %a %b",
            "%sum    = OpCompositeExtract typeof*{sum}   %_pair 0",
            "%carry  = OpCompositeExtract typeof*{carry} %_pair 1",
            "OpStore {sum}   %sum",
            "OpStore {carry} %carry",
            a     = in(reg) &a,
            b     = in(reg) &b,
            sum   = in(reg) &mut sum,
            carry = in(reg) &mut carry,
        );
    }
    (sum, carry)
}

/// `OpISubBorrow` - subtract with borrow out. Returns `(difference, borrow)` where borrow is 0 or 1.
/// x86 equivalent: `SUBB`. Useful for multi-precision integer arithmetic.
#[inline(always)]
pub unsafe fn spirv_isub_borrow<T: Copy + ConstDefault>(a: T, b: T) -> (T, T) {
    let mut diff = T::DEFAULT;
    let mut borrow = T::DEFAULT;
    unsafe {
        core::arch::asm!(
            "%a      = OpLoad typeof*{a} {a}",
            "%b      = OpLoad typeof*{b} {b}",
            "%_pair  = OpISubBorrow typeof*{diff} %a %b",
            "%diff   = OpCompositeExtract typeof*{diff}   %_pair 0",
            "%borrow = OpCompositeExtract typeof*{borrow} %_pair 1",
            "OpStore {diff}   %diff",
            "OpStore {borrow} %borrow",
            a      = in(reg) &a,
            b      = in(reg) &b,
            diff   = in(reg) &mut diff,
            borrow = in(reg) &mut borrow,
        );
    }
    (diff, borrow)
}

/// `GLSLstd450 ModfStruct` - splits `x` into fractional and integer parts.
/// Returns `(fract, whole)` where both have the same type as the input.
/// Works on scalars and `#[rust_gpu::vector::v1]` vectors.
#[inline(always)]
pub unsafe fn glsl_modf<T: Copy + ConstDefault>(x: T) -> (T, T) {
    let mut fract = T::DEFAULT;
    let mut whole = T::DEFAULT;
    unsafe {
        core::arch::asm!(
            "%glsl    = OpExtInstImport \"GLSL.std.450\"",
            "%x       = OpLoad typeof*{x} {x}",
            "%_struct = OpExtInst typeof*{fract} %glsl 36 %x",
            "%fract   = OpCompositeExtract typeof*{fract} %_struct 0",
            "%whole   = OpCompositeExtract typeof*{whole} %_struct 1",
            "OpStore {fract} %fract",
            "OpStore {whole} %whole",
            x     = in(reg) &x,
            fract = in(reg) &mut fract,
            whole = in(reg) &mut whole,
        );
    }
    (fract, whole)
}

/// `GLSLstd450 FrexpStruct` - splits `x` into significand and exponent.
/// Returns `(significand, exponent)`. `TF` is the float type, `TI` is the
/// matching signed integer type (e.g. `f32`/`i32`, `F32x4`/`I32x4`).
#[inline(always)]
pub unsafe fn glsl_frexp<TF: Copy + ConstDefault, TI: Copy + ConstDefault>(x: TF) -> (TF, TI) {
    let mut significand = TF::DEFAULT;
    let mut exponent = TI::DEFAULT;

    unsafe {
        core::arch::asm!(
            "%glsl        = OpExtInstImport \"GLSL.std.450\"",
            "%x           = OpLoad typeof*{x} {x}",
            "%_struct     = OpExtInst typeof*{significand} %glsl 52 %x",
            "%significand = OpCompositeExtract typeof*{significand} %_struct 0",
            "%exponent    = OpCompositeExtract typeof*{exponent}    %_struct 1",
            "OpStore {significand} %significand",
            "OpStore {exponent}    %exponent",
            x           = in(reg) &x,
            significand = in(reg) &mut significand,
            exponent    = in(reg) &mut exponent,
        );
    }

    (significand, exponent)
}

/// `GLSLstd450 Determinant` - compute the scalar determinant of a square matrix.
/// `TM` is the `#[spirv(matrix)]` matrix type; `TF` is the scalar float result type.
/// Unlike `glsl_op1`, this allows the input and output types to differ.
#[inline(always)]
pub unsafe fn glsl_determinant<TM: Copy, TF: Copy + ConstDefault>(mat: TM) -> TF {
    let mut result = TF::DEFAULT;
    unsafe {
        core::arch::asm!(
            "%glsl   = OpExtInstImport \"GLSL.std.450\"",
            "%mat    = OpLoad typeof*{mat} {mat}",
            "%result = OpExtInst typeof*{result} %glsl {op} %mat",
            "OpStore {result} %result",
            mat    = in(reg) &mat,
            result = in(reg) &mut result,
            op     = const glsl::DETERMINANT,
        );
    }
    result
}

/// `GLSLstd450 Determinant + MatrixInverse` - compute both in one asm block.
/// Returns `(determinant, inverse_matrix)`. Using a single block avoids loading the matrix twice.
/// The caller must check the determinant before trusting the inverse.
#[inline(always)]
pub unsafe fn glsl_determinant_and_inverse<TM: Copy + ConstDefault, TF: Copy + ConstDefault>(mat: TM) -> (TF, TM) {
    let mut det = TF::DEFAULT;
    let mut inv = TM::DEFAULT;
    unsafe {
        core::arch::asm!(
            "%glsl   = OpExtInstImport \"GLSL.std.450\"",
            "%mat    = OpLoad typeof*{mat} {mat}",
            "%det    = OpExtInst typeof*{det} %glsl {det_op} %mat",
            "%inv    = OpExtInst typeof*{inv} %glsl {inv_op} %mat",
            "OpStore {det} %det",
            "OpStore {inv} %inv",
            mat     = in(reg) &mat,
            det     = in(reg) &mut det,
            inv     = in(reg) &mut inv,
            det_op  = const glsl::DETERMINANT,
            inv_op  = const glsl::MATRIX_INVERSE,
        );
    }
    (det, inv)
}

/// Generate a GLSLstd450 extended-instruction wrapper for any number of same-typed arguments.
///
/// Each generated function takes N arguments of type `T: Copy + ConstDefault`, one const generic
/// `OP: u32` (the GLSLstd450 opcode), and one const generic `RP: bool`. When `RP` is `true` an
/// `OpDecorate %result RelaxedPrecision` is emitted on the result SSA value, allowing the driver
/// to use reduced-precision arithmetic. The zero-argument form (`glsl_op0`) is also supported.
macro_rules! glsl_op {
    // Internal helper: emit the full asm! block.
    // [$($arg)*]    - input argument identifiers
    // [$($extra)*]  - zero or more extra SPIR-V instruction strings inserted before OpStore
    //                 (used to conditionally inject OpDecorate for relaxed precision)
    (@emit [$($arg:ident)*] [$($extra:literal)*]) => {{
        let mut result = T::DEFAULT;

        core::arch::asm!(
            "%glsl = OpExtInstImport \"GLSL.std.450\"",
            $(concat!("%", stringify!($arg), " = OpLoad typeof*{", stringify!($arg), "} {", stringify!($arg), "}"),)*
            $($extra,)*
            concat!("%result = OpExtInst typeof*{result} %glsl {op}", $(" %", stringify!($arg)),*),
            "OpStore {result} %result",
            $($arg = in(reg) &$arg,)*
            result = in(reg) &mut result,
            op = const OP,
        );

        result
    }};

    ($vis:vis unsafe fn $fn_name:ident($($arg:ident: $ty:ident),*) -> T) => {
        #[inline(always)]
        $vis unsafe fn $fn_name<T: Copy + ConstDefault, $($ty: Copy + ConstDefault,)* const OP: u32, const RP: bool>($($arg: $ty),*) -> T {
            // `RP` is a const generic bool, so the compiler eliminates the dead branch during
            // monomorphization. The decoration must live in the same asm! block as the
            // OpExtInst that defines %result, so two nearly-identical expansions are
            // unavoidable, and the @emit helper keeps the shared logic in one place.
            if RP {
                unsafe { glsl_op!(@emit [$($arg)*] ["OpDecorate %result RelaxedPrecision"]) }
            } else {
                unsafe { glsl_op!(@emit [$($arg)*] []) }
            }
        }
    };
}

glsl_op!(pub unsafe fn glsl_op1(a: A) -> T);
glsl_op!(pub unsafe fn glsl_op2(a: A, b: B) -> T);
glsl_op!(pub unsafe fn glsl_op3(a: A, b: B, c: C) -> T);

// GLSLstd450 opcode constants - values verified against Khronos GLSL.std.450.h
pub mod glsl {
    // Rounding
    pub const ROUND: u32 = 1;
    pub const ROUND_EVEN: u32 = 2;
    pub const TRUNC: u32 = 3;
    pub const FLOOR: u32 = 8;
    pub const CEIL: u32 = 9;
    pub const FRACT: u32 = 10;
    // Abs / sign
    pub const F_ABS: u32 = 4;
    pub const S_ABS: u32 = 5;
    pub const F_SIGN: u32 = 6;
    pub const S_SIGN: u32 = 7;
    // Angle
    pub const RADIANS: u32 = 11;
    pub const DEGREES: u32 = 12;
    // Trig
    pub const SIN: u32 = 13;
    pub const COS: u32 = 14;
    pub const TAN: u32 = 15;
    pub const ASIN: u32 = 16;
    pub const ACOS: u32 = 17;
    pub const ATAN: u32 = 18;
    pub const SINH: u32 = 19;
    pub const COSH: u32 = 20;
    pub const TANH: u32 = 21;
    pub const ASINH: u32 = 22;
    pub const ACOSH: u32 = 23;
    pub const ATANH: u32 = 24;
    pub const ATAN2: u32 = 25;
    // Exponential / logarithmic
    pub const POW: u32 = 26;
    pub const EXP: u32 = 27;
    pub const LOG: u32 = 28;
    pub const EXP2: u32 = 29;
    pub const LOG2: u32 = 30;
    pub const SQRT: u32 = 31;
    pub const INVERSE_SQRT: u32 = 32;
    // Linear algebra (vector/matrix - glsl_op1/2/3 not suitable for these)
    pub const DETERMINANT: u32 = 33;
    pub const MATRIX_INVERSE: u32 = 34;
    // Float decomposition (need OpVariable for out-param variants - use Struct variants)
    pub const MODF_STRUCT: u32 = 36;
    pub const FREXP_STRUCT: u32 = 52;
    pub const LDEXP: u32 = 53;
    // Min / max / clamp
    pub const F_MIN: u32 = 37;
    pub const U_MIN: u32 = 38;
    pub const S_MIN: u32 = 39;
    pub const F_MAX: u32 = 40;
    pub const U_MAX: u32 = 41;
    pub const S_MAX: u32 = 42;
    pub const F_CLAMP: u32 = 43;
    pub const U_CLAMP: u32 = 44;
    pub const S_CLAMP: u32 = 45;
    // NaN-propagating min/max/clamp (prefer these over F_MIN/F_MAX for IEEE correctness)
    pub const N_MIN: u32 = 79;
    pub const N_MAX: u32 = 80;
    pub const N_CLAMP: u32 = 81;
    // Interpolation
    pub const F_MIX: u32 = 46;
    pub const STEP: u32 = 48;
    pub const SMOOTH_STEP: u32 = 49;
    // FMA
    pub const FMA: u32 = 50;
    // Geometry (vector operands - work with glsl_op1/2/3 for vec types)
    pub const LENGTH: u32 = 66;
    pub const DISTANCE: u32 = 67;
    pub const CROSS: u32 = 68;
    pub const NORMALIZE: u32 = 69;
    pub const FACE_FORWARD: u32 = 70;
    pub const REFLECT: u32 = 71;
    pub const REFRACT: u32 = 72;
    // Bit ops
    pub const FIND_I_LSB: u32 = 73;
    pub const FIND_S_MSB: u32 = 74;
    pub const FIND_U_MSB: u32 = 75;
    // Pack / unpack (scalar only - produce/consume u32)
    pub const PACK_SNORM_4X8: u32 = 54;
    pub const PACK_UNORM_4X8: u32 = 55;
    pub const PACK_SNORM_2X16: u32 = 56;
    pub const PACK_UNORM_2X16: u32 = 57;
    pub const PACK_HALF_2X16: u32 = 58;
    pub const PACK_DOUBLE_2X32: u32 = 59;
    pub const UNPACK_SNORM_2X16: u32 = 60;
    pub const UNPACK_UNORM_2X16: u32 = 61;
    pub const UNPACK_HALF_2X16: u32 = 62;
    pub const UNPACK_SNORM_4X8: u32 = 63;
    pub const UNPACK_UNORM_4X8: u32 = 64;
    pub const UNPACK_DOUBLE_2X32: u32 = 65;
}

// Internal helper - emits OpVectorShuffle with fixed literal indices (no SwizzleIndices needed).
// Used for interleave/deinterleave where the shuffle pattern is known at compile time.
macro_rules! spirv_shuffle_fixed {
    ($result:ident, $a:ident, $b:ident, [$($idx:literal),+]) => {
        core::arch::asm!(
            concat!("%a = OpLoad typeof*{a} {a}"),
            concat!("%b = OpLoad typeof*{b} {b}"),
            concat!("%result = OpVectorShuffle typeof*{result} %a %b", $(" ", stringify!($idx)),+),
            "OpStore {result} %result",
            a      = in(reg) &$a,
            b      = in(reg) &$b,
            result = in(reg) &mut $result,
        );
    };
}

// pub(super) so register submodules can use it
pub(super) use spirv_shuffle_fixed;

// Internal helper - emits OpVectorShuffle with literal indices from a SwizzleIndices impl.
// $src1 and $src2 are the two SPIR-V source operands (pass the same ident twice for permute).
// $lane_indices is a comma-separated list of integer literals matching the lane count,
// used both to index I::INDICES::as_slice() and to build the asm index placeholder names.
macro_rules! spirv_shuffle_impl {
    ($result:ident, $src1:ident, $src2:ident, [$($n:literal),+]) => {
        paste::paste! {
            core::arch::asm!(
                concat!("%", stringify!($src1), " = OpLoad typeof*{", stringify!($src1), "} {", stringify!($src1), "}"),
                concat!("%", stringify!($src2), " = OpLoad typeof*{", stringify!($src2), "} {", stringify!($src2), "}"),
                concat!(
                    "%result = OpVectorShuffle typeof*{result} %",
                    stringify!($src1), " %", stringify!($src2),
                    $(" {", stringify!([<i $n>]), "}"),+
                ),
                "OpStore {result} %result",
                $src1  = in(reg) &$src1,
                $src2  = in(reg) &$src2,
                result = in(reg) &mut $result,
                $([<i $n>] = const I::INDICES.as_slice()[$n],)+
            );
        }
    };

    // Permute: both source operands are the same register, so only one OpLoad is needed.
    (permute: $result:ident, $src:ident, [$($n:literal),+]) => {
        paste::paste! {
            core::arch::asm!(
                concat!("%src    = OpLoad typeof*{", stringify!($src), "} {", stringify!($src), "}"),
                concat!(
                    "%result = OpVectorShuffle typeof*{result} %src %src",
                    $(" {", stringify!([<i $n>]), "}"),+
                ),
                "OpStore {result} %result",
                $src   = in(reg) &$src,
                result = in(reg) &mut $result,
                $([<i $n>] = const I::INDICES.as_slice()[$n],)+
            );
        }
    };
}

// Interleave helpers - fixed-index OpVectorShuffle pairs for 2/3/4-lane vectors.
// Indices: a = 0..N, b = N..2N.
//
// interleave_lo: zip even lanes (a0,b0, a1,b1, ...)
// interleave_hi: zip odd  lanes (a1,b1, a2,b2, ...) - or upper half for 4-lane
// deinterleave_even: extract even-indexed lanes from the interleaved pair -> a
// deinterleave_odd:  extract odd-indexed  lanes from the interleaved pair -> b

/// Interleave two 2-lane vectors: `([a0, b0], [a1, b1])`
#[inline(always)]
pub unsafe fn spirv_interleave2<TV: Copy + ConstDefault>(a: TV, b: TV) -> (TV, TV) {
    let mut lo = TV::DEFAULT;
    let mut hi = TV::DEFAULT;
    unsafe {
        spirv_shuffle_fixed!(lo, a, b, [0, 2]);
        spirv_shuffle_fixed!(hi, a, b, [1, 3]);
    }
    (lo, hi)
}

/// Deinterleave two 2-lane vectors: `([a0, a1], [b0, b1])`
#[inline(always)]
pub unsafe fn spirv_deinterleave2<TV: Copy + ConstDefault>(a: TV, b: TV) -> (TV, TV) {
    let mut evens = TV::DEFAULT;
    let mut odds = TV::DEFAULT;
    unsafe {
        spirv_shuffle_fixed!(evens, a, b, [0, 1]);
        spirv_shuffle_fixed!(odds, a, b, [2, 3]);
    }
    (evens, odds)
}

/// Interleave two 3-lane vectors: `([a0, b0, a1], [b1, a2, b2])`
#[inline(always)]
pub unsafe fn spirv_interleave3<TV: Copy + ConstDefault>(a: TV, b: TV) -> (TV, TV) {
    let mut lo = TV::DEFAULT;
    let mut hi = TV::DEFAULT;
    unsafe {
        spirv_shuffle_fixed!(lo, a, b, [0, 3, 1]);
        spirv_shuffle_fixed!(hi, a, b, [4, 2, 5]);
    }
    (lo, hi)
}

/// Deinterleave two 3-lane vectors: `([a0, a1, a2], [b0, b1, b2])`
/// Inverse of `spirv_interleave3`: given `lo=[a0,b0,a1]`, `hi=[b1,a2,b2]`,
/// recover even stream at indices [0,2,4] and odd stream at [1,3,5].
#[inline(always)]
pub unsafe fn spirv_deinterleave3<TV: Copy + ConstDefault>(a: TV, b: TV) -> (TV, TV) {
    let mut evens = TV::DEFAULT;
    let mut odds = TV::DEFAULT;
    unsafe {
        spirv_shuffle_fixed!(evens, a, b, [0, 2, 4]);
        spirv_shuffle_fixed!(odds, a, b, [1, 3, 5]);
    }
    (evens, odds)
}

/// Interleave two 4-lane vectors: `([a0, b0, a1, b1], [a2, b2, a3, b3])`
#[inline(always)]
pub unsafe fn spirv_interleave4<TV: Copy + ConstDefault>(a: TV, b: TV) -> (TV, TV) {
    let mut lo = TV::DEFAULT;
    let mut hi = TV::DEFAULT;
    unsafe {
        spirv_shuffle_fixed!(lo, a, b, [0, 4, 1, 5]);
        spirv_shuffle_fixed!(hi, a, b, [2, 6, 3, 7]);
    }
    (lo, hi)
}

/// Deinterleave two 4-lane vectors: `([a0, a2, b0, b2], [a1, a3, b1, b3])`
#[inline(always)]
pub unsafe fn spirv_deinterleave4<TV: Copy + ConstDefault>(a: TV, b: TV) -> (TV, TV) {
    let mut evens = TV::DEFAULT;
    let mut odds = TV::DEFAULT;
    unsafe {
        spirv_shuffle_fixed!(evens, a, b, [0, 2, 4, 6]);
        spirv_shuffle_fixed!(odds, a, b, [1, 3, 5, 7]);
    }
    (evens, odds)
}

/// `OpVectorShuffle` permute for 2-lane vectors.
/// Indices are into `a` only and must be in `0..2`.
#[inline(always)]
pub unsafe fn spirv_permute2<TV: Copy + ConstDefault, I: SwizzleIndices<typenum::U2>>(a: TV) -> TV {
    let mut result = TV::DEFAULT;
    unsafe {
        spirv_shuffle_impl!(permute: result, a, [0, 1]);
    }
    result
}

/// `OpVectorShuffle` permute for 3-lane vectors.
/// Indices are into `a` only and must be in `0..3`.
#[inline(always)]
pub unsafe fn spirv_permute3<TV: Copy + ConstDefault, I: SwizzleIndices<typenum::U3>>(a: TV) -> TV {
    let mut result = TV::DEFAULT;
    unsafe {
        spirv_shuffle_impl!(permute: result, a, [0, 1, 2]);
    }
    result
}

/// `OpVectorShuffle` permute for 4-lane vectors.
/// Indices are into `a` only and must be in `0..4`.
#[inline(always)]
pub unsafe fn spirv_permute4<TV: Copy + ConstDefault, I: SwizzleIndices<typenum::U4>>(a: TV) -> TV {
    let mut result = TV::DEFAULT;
    unsafe {
        spirv_shuffle_impl!(permute: result, a, [0, 1, 2, 3]);
    }
    result
}

/// `OpVectorShuffle` swizzle for 2-lane vectors.
/// Indices span both `a` (0..2) and `b` (2..4).
#[inline(always)]
pub unsafe fn spirv_swizzle2<TV: Copy + ConstDefault, I: SwizzleIndices<typenum::U2>>(a: TV, b: TV) -> TV {
    let mut result = TV::DEFAULT;
    unsafe {
        spirv_shuffle_impl!(result, a, b, [0, 1]);
    }
    result
}

/// `OpVectorShuffle` swizzle for 3-lane vectors.
/// Indices span both `a` (0..3) and `b` (3..6).
#[inline(always)]
pub unsafe fn spirv_swizzle3<TV: Copy + ConstDefault, I: SwizzleIndices<typenum::U3>>(a: TV, b: TV) -> TV {
    let mut result = TV::DEFAULT;
    unsafe {
        spirv_shuffle_impl!(result, a, b, [0, 1, 2]);
    }
    result
}

/// `OpVectorShuffle` swizzle for 4-lane vectors.
/// Indices span both `a` (0..4) and `b` (4..8).
#[inline(always)]
pub unsafe fn spirv_swizzle4<TV: Copy + ConstDefault, I: SwizzleIndices<typenum::U4>>(a: TV, b: TV) -> TV {
    let mut result = TV::DEFAULT;
    unsafe {
        spirv_shuffle_impl!(result, a, b, [0, 1, 2, 3]);
    }
    result
}
