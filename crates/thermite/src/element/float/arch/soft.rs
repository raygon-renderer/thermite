//! Fallback rung: whatever the rungs above could not prove is a single instruction.
//!
//! `sqrt` and (with `std`) `fma` go to the standard library, which lowers them to
//! generic LLVM intrinsics, which are selected with the enclosing function's target
//! features, so inside a `#[thermite::dispatch]` trampoline they can still become one
//! instruction where a `cfg!`-driven rung would have emitted a call. `sqrt` without
//! `std` goes to `libm`.
//!
//! `fma` without `std` does not: it calls the round-to-odd emulation ([`fmadd_ro`] /
//! [`fmadd_widen_ro`]) on the 1-lane scalar registers instead, which is correctly
//! rounded (bit-identical to a hardware FMA) for every input. `libm`'s f32 chain
//! carries a live subnormal-rounding bug (compiler-builtins#1262) and its f64 path is
//! unverified against hardware, while the emulation is proven and hardware-tested
//! (`tests/fma_exact.rs`), so `libm` is out of the FMA family entirely. Enabling
//! `std` can therefore change the scalar `mul_add` lowering on non-FMA hosts. That
//! is an accepted cost: `mul_add` through `llvm.fma` is the only dispatch-aware
//! scalar FMA available on stable, and the `nightly` rung cannot be required.
//!
//! [`fmadd_ro`]: crate::backend::generic::polyfills::fmadd_ro
//! [`fmadd_widen_ro`]: crate::backend::generic::polyfills::fmadd_widen_ro
//!
//! The four rounding operations are different: they are pure bit manipulation, and
//! `libm`'s versions are unreachable for the optimizer. Its *generic* inner functions are
//! `#[inline]`, but the concrete `libm::floor` wrappers are not, so every call crosses a
//! crate boundary as a real `call`, an optimization barrier that blocks constant folding
//! and CSE and forces caller-saved spills around it. Measured: `floor(x) + floor(x)`
//! compiles to two calls plus 80 bytes of stack, versus one inlined body and an `addsd`.
//!
//! So they are reimplemented here, using the same algorithms `libm` does (from musl,
//! MIT): integer significand masking for `floor`/`ceil`/`trunc`, and the magic-constant
//! round trip for ties-to-even. Measured against a branchless magic-number `floor`
//! (llvm-mca, znver3): 13 vs 30 cycles of latency, and, decisively, this version runs
//! on the integer ALUs while the branchless one puts 3.00 uOps on each of FP0-FP3, the
//! exact ports the surrounding SIMD work needs.
//!
//! This rung is never trusted for [`HAS_NATIVE_FMA`](super::HAS_NATIVE_FMA), which stays
//! baseline-gated.
//!
//! # `outline_scalar_math`
//!
//! Inlining these costs code size at every call site. Enabling the
//! `outline_scalar_math` feature emits each one out of line instead, trading the folding
//! and CSE back for size.

use core::cfg_select;

macro_rules! impl_fallback {
    ($t:ty, $bits:ty, $ibits:ty, $sqrt:ident, $floor:ident, $ceil:ident, $trunc:ident,
     $round:ident, $fma:ident, [$l_sqrt:ident, $ro_fma:ident]) => {
        /// Bits of significand.
        const SIG_BITS: u32 = <$t>::MANTISSA_DIGITS - 1;
        const SIG_MASK: $bits = (1 << SIG_BITS) - 1;
        const SIGN_MASK: $bits = 1 << (<$bits>::BITS - 1);
        const EXP_MASK: $bits = !(SIG_MASK | SIGN_MASK);
        const EXP_BIAS: i32 = (EXP_MASK >> SIG_BITS) as i32 / 2;

        /// Unbiased exponent. `SIG_BITS` or more means there is no fractional part, which
        /// also covers infinities and NaNs.
        #[inline(always)]
        fn exp_unbiased(bits: $bits) -> i32 {
            ((bits & EXP_MASK) >> SIG_BITS) as i32 - EXP_BIAS
        }

        #[inline(always)]
        pub fn $sqrt(x: $t) -> $t {
            cfg_select! { feature = "std" => x.sqrt(), _ => libm::$l_sqrt(x) }
        }

        #[inline(always)]
        pub fn $fma(x: $t, y: $t, z: $t) -> $t {
            cfg_select! {
                feature = "std" => <$t>::mul_add(x, y, z),
                _ => crate::backend::generic::polyfills::$ro_fma::<$t>(x, y, z),
            }
        }

        #[cfg_attr(not(feature = "outline_scalar_math"), inline(always))]
        #[cfg_attr(feature = "outline_scalar_math", inline(never))]
        pub fn $floor(x: $t) -> $t {
            let mut ix = x.to_bits();
            let e = exp_unbiased(ix);

            if e >= SIG_BITS as i32 {
                return x; // no fractional part, or non-finite
            }

            if e >= 0 {
                // |x| >= 1: mask off the low `SIG_BITS - e` significand bits, biasing
                // negatives up first so the truncation lands away from zero.
                let m = SIG_MASK >> e as u32;
                if ix & m == 0 {
                    return x;
                }
                if ix & SIGN_MASK != 0 {
                    ix += m;
                }
                <$t>::from_bits(ix & !m)
            } else if ix & SIGN_MASK == 0 {
                0.0 // 0 <= x < 1
            } else if ix << 1 != 0 {
                -1.0 // -1 < x < 0
            } else {
                x // -0.0 is unchanged
            }
        }

        #[cfg_attr(not(feature = "outline_scalar_math"), inline(always))]
        #[cfg_attr(feature = "outline_scalar_math", inline(never))]
        pub fn $ceil(x: $t) -> $t {
            let mut ix = x.to_bits();
            let e = exp_unbiased(ix);

            if e >= SIG_BITS as i32 {
                return x;
            }

            if e >= 0 {
                let m = SIG_MASK >> e as u32;
                if ix & m == 0 {
                    return x;
                }
                if ix & SIGN_MASK == 0 {
                    ix += m;
                }
                <$t>::from_bits(ix & !m)
            } else if ix & SIGN_MASK != 0 {
                -0.0 // -1 < x <= -0
            } else if ix << 1 != 0 {
                1.0 // 0 < x < 1
            } else {
                x // +0.0 is unchanged
            }
        }

        #[cfg_attr(not(feature = "outline_scalar_math"), inline(always))]
        #[cfg_attr(feature = "outline_scalar_math", inline(never))]
        pub fn $trunc(x: $t) -> $t {
            let ix = x.to_bits();
            let e = exp_unbiased(ix);

            if e >= SIG_BITS as i32 {
                return x;
            }

            // A negative exponent means |x| < 1, so everything but the sign goes.
            let mask = if e < 0 { SIGN_MASK } else { !(SIG_MASK >> e as u32) };

            if ix & !mask == 0 {
                return x;
            }

            <$t>::from_bits(ix & mask)
        }

        // Ties to even, matching the vector backends. Adding and subtracting 2^SIG_BITS
        // forces the fractional bits out through the active rounding mode, which Rust
        // fixes at nearest-ties-even; the guard keeps values that already have no
        // fractional part (and non-finites) untouched.
        #[cfg_attr(not(feature = "outline_scalar_math"), inline(always))]
        #[cfg_attr(feature = "outline_scalar_math", inline(never))]
        pub fn $round(x: $t) -> $t {
            const MAGIC: $t = (1 as $ibits << SIG_BITS) as $t;

            let ix = x.to_bits();
            if exp_unbiased(ix) >= SIG_BITS as i32 {
                return x;
            }

            let a = <$t>::from_bits(ix & !SIGN_MASK);
            let r = (a + MAGIC) - MAGIC;

            <$t>::from_bits(r.to_bits() | (ix & SIGN_MASK)) // copysign(r, x)
        }
    };
}

mod f64_impl {
    use super::*;
    impl_fallback!(f64, u64, i64, sqrt, floor, ceil, trunc, round, fma, [sqrt, fmadd_ro]);
}

mod f32_impl {
    use super::*;
    impl_fallback!(
        f32,
        u32,
        i32,
        sqrtf,
        floorf,
        ceilf,
        truncf,
        roundf,
        fmaf,
        [sqrtf, fmadd_widen_ro]
    );
}

pub use f32_impl::{ceilf, floorf, fmaf, roundf, sqrtf, truncf};
pub use f64_impl::{ceil, floor, fma, round, sqrt, trunc};
