use generic_array::{
    GenericArray,
    typenum::{self},
};

use crate::register::{
    BitwiseRegister, CoreRegister, FloatElement, FloatRegister, IndexableRegister, InterleaveRegister, MaskElement,
    NativeCapability, NumericRegister, PartialOrdRegister, PermuteRegister, Register, ShuffleRegister, SignedRegister,
    Storage, UnsignedIntegerRegister, ZeroUpper,
};

use crate::element::float::algebraic::AlgebraicFloat;
use crate::isa::InstructionSet;
use crate::vector::ops::MulAddExt;

#[rustfmt::skip]
macro_rules! decl_float_scalar { ($f:ty $(: $s:ident)? => $width:literal) => {paste::paste! {

#[thermite_macros::inline_always]
impl CoreRegister for [<f $width>] {
    type Lanes = typenum::U1;
    type Storage = [<f $width>];
    type Mask = bool;

    const IS_EMULATED: bool = false;
    const EMPTY: Storage<Self> = 0.0;
    const HAS_EQUAL_SIZE_MASK: bool = false;

    fn blendv(mask: Storage<Self::Mask>, lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        core::hint::select_unpredictable(mask, rhs, lhs)
    }

    fn zz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        core::hint::select_unpredictable(mask, value, 0.0)
    }

    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        core::hint::select_unpredictable(mask, 0.0, value)
    }

    fn zeroupper_z<Z: ZeroUpper>(value: Storage<Self>) -> Storage<Self> {
        if const { Z::N >= 1 } { value } else { Self::EMPTY } // if N == 0 zero everything
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> { Self::from_bool(mask) }
}

#[thermite_macros::inline_always]
impl BitwiseRegister for [<f $width>] {
    fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        $f::from_bits(lhs.to_bits() ^ rhs.to_bits())
    }

    fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        $f::from_bits(lhs.to_bits() & rhs.to_bits())
    }

    fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        $f::from_bits(lhs.to_bits() | rhs.to_bits())
    }

    fn not(value: Storage<Self>) -> Storage<Self> {
        $f::from_bits(!value.to_bits())
    }
}

#[rustfmt::skip]
#[thermite_macros::inline_always]
impl InterleaveRegister for [<f $width>] {
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) { (a, b) }
    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) { (a, b) }
}

#[thermite_macros::inline_always]
impl Register for [<f $width>] {
    type Element = [<f $width>];

    type Signed = [<i $width>];
    type Unsigned = [<u $width>];

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> { value.to_bool() }

    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {
        // scalars don't use MSB for mask conversion, but we can use NEG_ZERO to extract the sign bit
        // and convert to a mask
        Self::into_mask(Self::bitand(value, Self::NEG_ZERO))
    }

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> { value[0] }
    fn single(value: Self::Element) -> Storage<Self> { value }
    fn splat(value: Self::Element) -> Storage<Self> { value }
    fn broadcast<const I: usize>(value: Storage<Self>) -> Storage<Self> { value }
    fn reverse(value: Storage<Self>) -> Storage<Self> { value }

    fn swap_bytes(value: Storage<Self>) -> Storage<Self> {
        $f::from_bits(value.to_bits().swap_bytes())
    }

    const HAS_PERMUTEV: bool = false;

    fn permutev(value: Storage<Self>, _idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        value
    }

    fn swizzle(a: Storage<Self>, b: Storage<Self>, idxs: Storage<Self::Unsigned>) -> Storage<Self> {
        if idxs & 0b1 == 0 { a } else { b }
    }
}

#[thermite_macros::inline_always]
impl<I> IndexableRegister<I> for [<f $width>]
where
    I: UnsignedIntegerRegister<Lanes = Self::Lanes>,
{
}

#[thermite_macros::inline_always]
impl ShuffleRegister for [<f $width>] {
    fn shuffle<const IMM8: i32>(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        if IMM8 & 0b01 == 0 { lhs } else { rhs }
    }
}

#[thermite_macros::inline_always]
impl PermuteRegister for [<f $width>] {
    fn permute<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> { value }
}

#[thermite_macros::inline_always]
impl PartialOrdRegister for [<f $width>] {
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> { lhs > rhs }
    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> { lhs == rhs }
    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> { lhs >= rhs }
    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> { lhs < rhs }
    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> { lhs <= rhs }
    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> { lhs != rhs }
}

#[thermite_macros::inline_always]
impl NumericRegister for [<f $width>] {
    const ZERO: Storage<Self> = 0.0;
    const ONE: Storage<Self> = 1.0;
    const TWO: Storage<Self> = 2.0;

    const MIN: Storage<Self> = $f::MIN;
    const MAX: Storage<Self> = $f::MAX;

    fn min_element(value: Storage<Self>) -> Self::Element { value }
    fn max_element(value: Storage<Self>) -> Self::Element { value }
    fn sum_elements(value: Storage<Self>) -> Self::Element { value }
    fn prod_elements(value: Storage<Self>) -> Self::Element { value }
    fn pairwise_sum(lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self> { lo.alg_add(hi) }
    fn offset() -> Storage<Self> { 1.0 }
    fn indexed() -> Storage<Self> { 0.0 }

    // Strict IEEE-754 by default; reassociable under `algebraic-scalar`, which is
    // what lets LLVM vectorize a loop over these. Every emulated register built
    // out of scalar lanes (`ArrayRegister<f32, N>` and friends) delegates here,
    // so this is the single point of control.
    fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs.alg_add(rhs) }
    fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs.alg_sub(rhs) }
    fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs.alg_mul(rhs) }
    fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs.alg_div(rhs) }
    fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs.alg_rem(rhs) }
    fn sort(value: Storage<Self>) -> Storage<Self> { value } // no-op for scalar

    // Strict IEEE-754: the scalar backend is the differential-test oracle, so it
    // must define the same tie/NaN semantics the SIMD backends' `fix_min`/`fix_max`
    // implement: min/max(x, NaN) = x (either operand order), min(-0, +0) = -0 and
    // max(-0, +0) = +0 regardless of operand order.
    #[cfg(feature = "strict_ieee754")]
    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        if rhs != rhs {
            lhs
        } else if lhs == rhs {
            <$f>::from_bits(lhs.to_bits() | rhs.to_bits())
        } else if lhs < rhs {
            lhs
        } else {
            rhs
        }
    }
    #[cfg(feature = "strict_ieee754")]
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {
        if rhs != rhs {
            lhs
        } else if lhs == rhs {
            <$f>::from_bits(lhs.to_bits() & rhs.to_bits())
        } else if lhs < rhs {
            rhs
        } else {
            lhs
        }
    }

    #[cfg(all(not(feature = "strict_ieee754"), any(target_arch = "x86", target_arch = "x86_64", target_arch = "arm", target_arch = "aarch64")))]
    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { core::hint::select_unpredictable(lhs < rhs, lhs, rhs) }
    #[cfg(all(not(feature = "strict_ieee754"), any(target_arch = "x86", target_arch = "x86_64", target_arch = "arm", target_arch = "aarch64")))]
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { core::hint::select_unpredictable(lhs < rhs, rhs, lhs) }

    #[cfg(not(any(feature = "strict_ieee754", target_arch = "x86", target_arch = "x86_64", target_arch = "arm", target_arch = "aarch64")))]
    fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { if lhs < rhs { lhs } else { rhs } }
    #[cfg(not(any(feature = "strict_ieee754", target_arch = "x86", target_arch = "x86_64", target_arch = "arm", target_arch = "aarch64")))]
    fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { if lhs < rhs { rhs } else { lhs } }
}

#[thermite_macros::inline_always]
impl SignedRegister for [<f $width>] {
    const NEG_ONE: Storage<Self> = -1.0;
    const MIN_POSITIVE: Storage<Self> = <$f>::MIN_POSITIVE;

    fn neg(value: Storage<Self>) -> Storage<Self> { -value }
    fn abs(value: Storage<Self>) -> Storage<Self> { value.abs() }
    fn signum(value: Storage<Self>) -> Storage<Self> { value.signum() }
    fn copysign(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { lhs.copysign(rhs) }

    fn neg_c(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        if mask { -value } else { value }
    }
}

#[thermite_macros::inline_always]
impl FloatRegister for [<f $width>] {
    type Bits = [<u $width>];
    type SignedBits = [<i $width>];
    type ExtendedPrecision = f64;

    const HAS_NATIVE_FMA: tribool::Tribool = crate::element::float::arch::HAS_NATIVE_FMA;

    const HALF: Storage<Self> = 0.5;
    const NEG_ZERO: Storage<Self> = -0.0;
    const INFINITY: Storage<Self> = $f::INFINITY;
    const NEG_INFINITY: Storage<Self> = $f::NEG_INFINITY;
    const NAN: Storage<Self> = $f::NAN;
    const EPSILON: Storage<Self> = $f::EPSILON;

    const EXP_MASK: Storage<Self::Bits> = $f::INFINITY.to_bits(); // all exponent bits set

    const HAS_APPROX_RSQRT: bool = false;
    const HAS_APPROX_RCP: bool = false;

    // --- error-free transformations, forced strict ------------------------------
    //
    // The `FloatRegister` defaults are written out of `Self::add`/`sub`/`mul`, which here
    // are `alg_*` and reassociable under `algebraic-scalar`. LLVM then folds
    // `(a - (s - v)) + (b - v)` to zero and every error term silently vanishes. These
    // overrides use the plain element operators, which stay strict. Ordinary arithmetic
    // stays algebraic so loops over this backend still vectorize.
    //
    // `ArrayRegister<f32, N>` and friends delegate here, so these cover those widths too.
    fn two_sum<const FAST: bool>(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        let s = a + b;

        if const { FAST } {
            return (s, b - (s - a));
        }

        let v = s - a;
        (s, (a - (s - v)) + (b - v))
    }

    fn two_diff<const FAST: bool>(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        let s = a - b;

        if const { FAST } {
            return (s, (a - s) - b);
        }

        let v = s - a;
        (s, (a - (s - v)) - (b + v))
    }

    fn veltkamp_split(a: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        let c = a * Self::VELTKAMP_SPLITTER;
        let hi = c - (c - a);
        (hi, a - hi)
    }

    fn rebalance_for_split(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        // One lane, so this branches where the vector form selects.
        if Self::abs(a) > Self::VELTKAMP_SPLIT_THRESH {
            (a * Self::VELTKAMP_SPLIT_DOWN, b * Self::VELTKAMP_SPLIT_UP)
        } else if Self::abs(b) > Self::VELTKAMP_SPLIT_THRESH {
            (a * Self::VELTKAMP_SPLIT_UP, b * Self::VELTKAMP_SPLIT_DOWN)
        } else {
            (a, b)
        }
    }

    fn two_prod<const SQUARE: bool>(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        let b = if const { SQUARE } { a } else { b };

        let p = a * b;

        if matches!(<Self as FloatRegister>::HAS_NATIVE_FMA, tribool::True) {
            return (p, MulAddExt::mul_sub(a, b, p));
        }

        if const { SQUARE } {
            let (hi, lo) = Self::veltkamp_split(a);
            let cross = hi * lo;

            return (p, ((hi * hi - p) + (cross + cross)) + lo * lo);
        }

        let (sa, sb) = Self::rebalance_for_split(a, b);
        let (a_hi, a_lo) = Self::veltkamp_split(sa);
        let (b_hi, b_lo) = Self::veltkamp_split(sb);

        (p, ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo)
    }

    // Mostly used for its high word. `Self::div` here is `alg_div`, whose `arcp` rewrites
    // `x / c` for a constant `c` into `x * RN(1/c)` (two roundings, up to 1.204 ulp
    // measured). Plain `/` is strict, so `two_quot(a, b).0` is a free correctly-rounded
    // division.
    fn two_quot(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        let q = a / b;

        if matches!(<Self as FloatRegister>::HAS_NATIVE_FMA, tribool::True) {
            return (q, MulAddExt::nmul_add(q, b, a));
        }

        let (p, e) = Self::two_prod::<false>(q, b);

        (q, (a - p) - e)
    }

    fn mul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> { MulAddExt::mul_add(lhs, rhs, acc) }
    fn mul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> { MulAddExt::mul_sub(lhs, rhs, acc) }
    fn nmul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> { MulAddExt::nmul_add(lhs, rhs, acc) }
    fn nmul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> { MulAddExt::nmul_sub(lhs, rhs, acc) }

    // The `_e` variants delegate to the element layer instead of the `FloatRegister`
    // defaults, whose unfused arm is `alg_mul`/`alg_add`. A reassociable accumulate
    // re-brackets a Cody-Waite reduction, `((x - m1) - m2) - m3` into `x - (m1+m2+m3)`,
    // rounding the split constant back to one word: measured 122705 ulp on `sin(1e5)`,
    // 76 ulp on `exp(-302)`. A strict multiply alone fixes `exp` (two terms, LLVM declined
    // the rewrite) but not `sin` (three or four terms); the accumulate is what matters.
    //
    // Cost: an FMA-shaped reduction loop no longer reassociates on this backend, so it is
    // harder to autovectorize. Accepted, since a true FMA cannot be algebraic anyway.
    // Plain `add`/`sub`/`mul` loops still reassociate freely.
    fn mul_adde(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> { MulAddExt::mul_adde(lhs, rhs, acc) }
    fn mul_sube(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> { MulAddExt::mul_sube(lhs, rhs, acc) }
    fn nmul_adde(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> { MulAddExt::nmul_adde(lhs, rhs, acc) }
    fn nmul_sube(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> { MulAddExt::nmul_sube(lhs, rhs, acc) }

    fn sqrt(value: Storage<Self>) -> Storage<Self> { FloatElement::sqrt(value) }
    fn floor(value: Storage<Self>) -> Storage<Self> { FloatElement::floor(value) }
    fn ceil(value: Storage<Self>) -> Storage<Self> { FloatElement::ceil(value) }
    fn round(value: Storage<Self>) -> Storage<Self> { FloatElement::round(value) }
    fn trunc(value: Storage<Self>) -> Storage<Self> { FloatElement::trunc(value) }
    fn fract(value: Storage<Self>) -> Storage<Self> { FloatElement::fract(value) }
    fn next_up(value: Storage<Self>) -> Storage<Self> { FloatElement::next_up(value) }
    fn next_down(value: Storage<Self>) -> Storage<Self> { FloatElement::next_down(value) }

    // TODO: maybe at some point?
    const NATIVE_CAP: NativeCapability = NativeCapability::NONE;

    unsafe fn block_autovectorization(value: &mut Storage<Self>) {
        unsafe {
            // x86_64: Use "xmm_reg" to keep it in the float/vector registers.
            // aarch64: Use "vreg" (or "reg" often works as floats are standard).
            #[cfg(target_arch = "x86_64")]
            core::arch::asm!(
                "/* {0} */",
                inout(xmm_reg) *value, // tied operand: reads xmmN, writes xmmN
                options(nomem, nostack, preserves_flags)
            );

            // `:v` names the whole vector register; the template is only a
            // comment (this is a pure compiler barrier), so the formatting is
            // cosmetic, and the modifier just silences `asm_sub_register`.
            #[cfg(target_arch = "aarch64")]
            core::arch::asm!(
                "/* {0:v} */",
                inout(vreg) *value,
                options(nomem, nostack, preserves_flags)
            );
        }
    }
}

}}} // end macro

decl_float_scalar!(f32 => 32);
decl_float_scalar!(f64 => 64);
