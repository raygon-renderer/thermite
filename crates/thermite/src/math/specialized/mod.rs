#![allow(clippy::excessive_precision, clippy::approx_constant)]

//! Element-parameterized implementations behind the public math traits.
//!
//! The traits in this module carry the actual
//! algorithms (polynomial approximations, range reductions, Newton iterations,
//! etc.) for `sin`, `exp`, `ln`, and the rest. Each is generic over an element
//! type `E`, and the corresponding public trait in [`crate::math`]
//! ([`CoreMath`](crate::math::CoreMath),
//! [`TranscendentalMath`](crate::math::TranscendentalMath), ...) is a thin shim
//! that simply forwards to the specialized method for its element type.
//!
//! # Why the element is the unit of specialization
//!
//! Splitting the implementation out by *element type* - rather than by vector or
//! backend - is what lets the math library extend to composite number systems.
//! A vector's element is not required to be a primitive `f32`/`f64`: it can
//! itself be a structured value, and a math implementation written against that
//! element flows through the exact same public traits.
//!
//! The motivating case is compensated (double-double) arithmetic: a
//! `Compensated` vector's "element" is itself a `Compensated` value, so
//! implementing the specialized traits for that element gives every
//! `Compensated` vector full transcendental support with no changes to generic
//! callers. The same pattern is intended for `Complex` (in `thermite-complex`) and dual/hyperdual
//! numbers as those land - implement the specialized math for the new element
//! type and the entire public math API lights up for it automatically.
//!
//! Most code should never name these traits directly; bound on the public
//! `*Math` traits instead. They are documented here for implementors adding a
//! new element type.

use core::marker::PhantomData;

use crate::{
    element::{FloatElement, FloatElementWithBits},
    mask::*,
    math::{
        CoreMathWithPolicy, FloatConsts, RealMathWithPolicy, TranscendentalMathWithPolicy, algorithms,
        policy::policies::{ExtraPrecision, LessPrecision},
    },
    register::NativeCapability,
    vector::{ops::BitAndNot, *},
};

// use super::MathWithPolicy;
use super::policy::{DenormalBehavior, Policy, PrecisionPolicy};

mod generic;

impl<E, V> SpecializedFloatMath<E> for V
where
    E: FloatElement,
    V: FloatVectorWithBits<Element = E>,
{
}

pub trait SpecializedFloatMath<E: FloatElementWithBits>: FloatVectorWithBits<Element = E> {
    #[inline(always)]
    fn ldexp<P: Policy>(self, exp: Self::SignedBits) -> Self {
        if const { Self::NATIVE_CAP.has(NativeCapability::LDEXP) } {
            return unsafe { Self::native_ldexp(self, exp) };
        }

        // constants
        let mantissa_bits = <Self::Element as FloatElementWithBits>::MANTISSA_BITS;
        let exp_lsb_mask: Self::Bits = crate::const_splat!(<Self> = <S: FloatVectorWithBits>
            <S::Bits as GenericVector>::Element: <S::Element as FloatElementWithBits>::EXP_LSB_MASK);
        let sign_mantissa_mask: Self::Bits = crate::const_splat!(<Self> = <S: FloatVectorWithBits>
            <S::Bits as GenericVector>::Element: <S::Element as FloatElementWithBits>::SIGN_MANTISSA_MASK);
        let max_biased_exp: Self::SignedBits = crate::const_splat!(<Self> = <S: FloatVectorWithBits>
            <S::SignedBits as GenericVector>::Element: <S::Element as FloatElementWithBits>::MAX_BIASED_EXP);
        let exp_bias: Self::SignedBits = crate::const_splat!(<Self> = <S: FloatVectorWithBits>
            <S::SignedBits as GenericVector>::Element: <S::Element as FloatElementWithBits>::EXP_BIAS);

        // special handling for denormals when we want to preserve them, since the normal path would flush them to zero
        if const {
            matches!(P::POLICY.denormal_behavior, DenormalBehavior::Preserve if <Self::Element as FloatElement>::HAS_SUBNORMALS)
        } {
            // libm/musl-style product chain, branchless: peel off up to three
            // power-of-two factors. The chunk bounds are chosen so that
            // (a) every factor is a normal float, and (b) on the negative side
            // (`exp_min + sig_total`: -102 for f32, -969 for f64) the running
            // product cannot land subnormal until the FINAL multiply - so IEEE
            // gradual underflow rounds exactly once. Three chunks cover the
            // full useful range (max finite down past the smallest subnormal
            // and back). In the common case the first chunk absorbs the whole
            // exponent and the remaining two factors are an exact `* 1.0`.
            //
            // Wrap-free for extreme requests like `ldexp(x, i32::MIN)`: each
            // step subtracts a same-sign clamped chunk, moving `exp`
            // monotonically toward zero.
            let mant_p2 = Self::SignedBits::splat(unsafe {
                <E as FloatElementWithBits>::SignedBits::try_from(E::MANTISSA_BITS + 2).unwrap_unchecked()
            });
            let chunk_neg = mant_p2 - exp_bias; // exp_min + sig_total
            let chunk_pos = exp_bias;

            let mut exp = exp;
            let mut result = self;

            let mut i = 0;
            while i < 3 {
                let k = exp.max(chunk_neg).min(chunk_pos);
                result *= Self::from_bits((k + exp_bias) << mantissa_bits);
                exp -= k;
                i += 1;
            }

            if const { P::POLICY.check_overflow } {
                // preserve the exact input NaN payload rather than whatever the
                // multiplies quiet it into
                result = self.is_nan().select(self, result);
            }

            return result;
        }

        let bits: Self::Bits = self.into_bits();

        let biased_exp = Self::SignedBits::from_bits((bits >> mantissa_bits) & exp_lsb_mask);

        let mut exp = exp;

        if const { P::POLICY.check_overflow } {
            // Saturate the requested shift so `biased_exp + exp` cannot wrap
            // (e.g. `ldexp(1.0, i32::MAX)`); +-4*bias is already far past the
            // overflow/underflow thresholds, so saturation doesn't change results.
            let exp_limit = exp_bias.shli::<2>();
            exp = exp.max(-exp_limit).min(exp_limit);
        }

        // the true (unclamped) new biased exponent; wrap-free thanks to the saturation above
        let new_exp = biased_exp + exp;

        if const { !P::POLICY.check_overflow } {
            // garbage in, garbage out per the policy contract: assemble and return
            let sign_mantissa = Self::SignedBits::from_bits(bits & sign_mantissa_mask);
            return Self::from_bits((new_exp << mantissa_bits) | sign_mantissa);
        }

        // Checked tail. Both forms clamp the biased exponent - which already
        // lands on the right FIELD value for the special cases (0 on underflow,
        // MAX_BIASED_EXP on overflow) - and then fix up the specials:
        // - out of range: zero the mantissa, so the clamped field reads as a
        //   signed zero / signed infinity (never the NaN a mantissa-preserving
        //   clamp would encode);
        // - zero/subnormal input (flushed on this path): zero the exponent
        //   field too, giving a signed zero regardless of the shift;
        // - non-finite input: force the exponent field to all-ones and keep the
        //   mantissa, so inf and NaN pass through (incl. `ldexp(inf, -k)`).
        //
        // Which lowering is cheaper depends on the hardware: the bit assembly
        // is 4 ternlogs plus mask fixups (one instruction each with AVX-512),
        // while without native ternary logic each ternlog expands to a DNF
        // chain and the blend form wins - measured on znver3 with llvm-mca,
        // 3.8 cyc/iter for blends against 5.8 for ternlogs.
        let clamped_exp = new_exp.max(Self::SignedBits::ZERO).min(max_biased_exp);

        let zero_sub = biased_exp.cmp_eq(Self::SignedBits::ZERO);

        if const { <Self::SignedBits as BitwiseVector>::HAS_NATIVE_TERNLOG } {
            let out_of_range = new_exp.cmp_le(Self::SignedBits::ZERO) | new_exp.cmp_ge(max_biased_exp);
            let non_finite = biased_exp.cmp_eq(max_biased_exp);

            let keep_mantissa =
                GenericMask::ternlog::<{ crate::ternlog_imm!(!(A | B) | C) }>(out_of_range, zero_sub, non_finite);

            let ibits = Self::SignedBits::from_bits(bits);
            let sign_bit = Self::SignedBits::from_bits(<Self as FloatVector>::NEG_ZERO);
            let sign_mantissa = Self::SignedBits::from_bits(sign_mantissa_mask);
            let exp_field = max_biased_exp << mantissa_bits;

            // ibits & mantissa-field mask (= sign_mantissa & !sign), gated by keep_mantissa
            let mantissa =
                Self::SignedBits::ternlog::<{ crate::ternlog_imm!(A & B & !C) }>(ibits, sign_mantissa, sign_bit)
                    .zz(keep_mantissa);

            // clamped exponent field | forced all-ones field | mantissa
            let field = Self::SignedBits::ternlog::<{ crate::ternlog_imm!(A | B | C) }>(
                (clamped_exp << mantissa_bits).nz(zero_sub),
                exp_field.zz(non_finite),
                mantissa,
            );

            // (bits & SIGN) | field
            return Self::from_bits(Self::SignedBits::ternlog::<{ crate::ternlog_imm!((A & B) | C) }>(
                ibits, sign_bit, field,
            ));
        }

        let sign_mantissa = Self::SignedBits::from_bits(bits & sign_mantissa_mask);

        let mut result = Self::from_bits((clamped_exp << mantissa_bits) | sign_mantissa);

        let overflow = new_exp.cmp_ge(max_biased_exp);

        // a zero/subnormal input flushes to a signed zero for ANY
        // shift, so it must be applied after (and therefore win over) the
        // overflow select - `ldexp(0.0, 300)` is 0.0, not infinity.
        result = overflow
            .cast::<Self::Mask>()
            .select(Self::INFINITY.copysign(self), result);
        result = (new_exp.cmp_le(Self::SignedBits::ZERO) | zero_sub)
            .cast::<Self::Mask>()
            .select(Self::ZERO.copysign(self), result);
        result = biased_exp
            .cmp_eq(max_biased_exp)
            .cast::<Self::Mask>()
            .select(self, result);

        result
    }

    #[inline(always)]
    fn frexp<P: Policy>(self) -> (Self, Self::SignedBits) {
        if const { Self::NATIVE_CAP.has(NativeCapability::FREXP) } {
            return unsafe { Self::native_frexp(self) };
        }

        let exp_lsb_mask: Self::Bits = crate::const_splat!(<Self> = <S: FloatVectorWithBits>
            <S::Bits as GenericVector>::Element: <S::Element as FloatElementWithBits>::EXP_LSB_MASK);
        let frexp_bias_offset: Self::SignedBits = crate::const_splat!(<Self> = <S: FloatVectorWithBits>
            <S::SignedBits as GenericVector>::Element: <S::Element as FloatElementWithBits>::FREXP_BIAS_OFFSET);
        let sign_mantissa_mask: Self::Bits = crate::const_splat!(<Self> = <S: FloatVectorWithBits>
            <S::Bits as GenericVector>::Element: <S::Element as FloatElementWithBits>::SIGN_MANTISSA_MASK);
        let half_exp_bits: Self::Bits = crate::const_splat!(<Self> = <S: FloatVectorWithBits>
            <S::Bits as GenericVector>::Element: <S::Element as FloatElementWithBits>::HALF_EXP_BITS);

        let bits: Self::Bits = self.into_bits();

        // Exponent field as given: zero means the input is a zero or a
        // subnormal. This mask is needed for the validity guard below anyway,
        // so reusing it as the renormalization predicate is free.
        let biased_exp = Self::SignedBits::from_bits((bits >> E::MANTISSA_BITS) & exp_lsb_mask);
        let zero_exp = biased_exp.cmp_eq(Self::SignedBits::ZERO);

        // subtract bias to get actual exponent
        let mut exp: Self::SignedBits = biased_exp - frexp_bias_offset;

        // extract sign and mantissa, then give it the correct exponent
        // let mut fraction = (bits & sign_mantissa_mask) | half_exp_bits;
        let mut fraction =
            Self::Bits::ternlog::<{ crate::ternlog_imm!((A & B) | C) }>(bits, sign_mantissa_mask, half_exp_bits);

        // A subnormal carries no implicit leading one, so its exponent field
        // means nothing until the value is renormalized - skip this and
        // `frexp(1e-40f32)` answers `(1e-40, 0)`, silently breaking the
        // `0.5 <= |frac| < 1` postcondition while still satisfying
        // `x == frac * 2^exp`.
        //
        // Renormalizing costs a multiply, a second extraction and two selects.
        // Under a flushing policy subnormals are rare by assumption, so that
        // work sits behind a branch and the common path stays exactly as cheap
        // as it was (llvm-mca, znver3: 2.2 cyc/iter, against 4.0 if the fixup
        // runs unconditionally). Under `Preserve` they are expected instead, so
        // the branch would only mispredict - run it straight-line there, as
        // this kernel always used to.
        //
        // The branch is the right call well past "rare". Measured with rdtsc
        // over 512 KiB of random input (TSC cycles per f32x8, subnormals placed
        // in a random lane at the stated per-VECTOR rate):
        //
        //     P(subnormal)     0%     1%     5%    10%    25%    50%   100%
        //     branchy        3.14   3.84   6.33   8.13  12.98  20.59  13.46
        //     branchless     5.20   5.52   6.86   7.99  10.31  12.35  12.47
        //
        // Crossover is ~8-9% of vectors, i.e. ~1 element in 90. Past that the
        // mispredicts dominate, peaking at 50% where the branch is maximally
        // unpredictable (+67%); at 100% it is predictable again and the cost
        // falls back. Callers who genuinely expect dense subnormals under a
        // flushing policy should ask for `AvoidBranching<P, true>` - or, more
        // likely, they wanted `PreserveDenormals<P>` all along.
        // `Ignore` promises the hardware is running with denormals disabled
        // (DAZ/FTZ), so one can never arrive here - skip the fixup and even its
        // test entirely, exactly as `flush_denormals` does for that policy.
        if const {
            <Self::Element as FloatElement>::HAS_SUBNORMALS
                && !matches!(P::POLICY.denormal_behavior, DenormalBehavior::Ignore)
        } {
            if const { P::POLICY.avoid_branching || matches!(P::POLICY.denormal_behavior, DenormalBehavior::Preserve) }
                || crate::unlikely(zero_exp.any())
            {
                let exp_bias: Self::SignedBits = crate::const_splat!(<Self> = <S: FloatVectorWithBits>
                    <S::SignedBits as GenericVector>::Element: <S::Element as FloatElementWithBits>::EXP_BIAS);

                let shift_amount = Self::SignedBits::splat(unsafe {
                    <E as FloatElementWithBits>::SignedBits::try_from(E::MANTISSA_BITS + 1).unwrap_unchecked()
                });

                let normalizer = Self::from_bits((exp_bias + shift_amount) << E::MANTISSA_BITS);

                // scale the zero-exponent lanes up into the normal range, then
                // redo the extraction for them and undo the scale in `exp`
                let scaled: Self::Bits = self.mul_c(zero_exp.cast(), normalizer).into_bits();
                let scaled_exp = Self::SignedBits::from_bits((scaled >> E::MANTISSA_BITS) & exp_lsb_mask);

                exp = zero_exp.select(scaled_exp - frexp_bias_offset - shift_amount, exp);
                fraction = zero_exp.cast::<Self::Mask>().select(
                    Self::Bits::ternlog::<{ crate::ternlog_imm!((A & B) | C) }>(
                        scaled,
                        sign_mantissa_mask,
                        half_exp_bits,
                    ),
                    fraction,
                );
            }
        }

        if const { P::POLICY.check_overflow } {
            // `+-0` must come back as `(+-0, 0)`, and inf/NaN pass through. The
            // zero test is on the float rather than the exponent field: after
            // the fixup above a subnormal is a legitimate result, so only a
            // true zero should be rejected here.
            let valid = self.is_finite() & self.cmp_ne(Self::ZERO);

            exp = exp.zz(valid.cast());
            fraction = valid.select(fraction, bits);
        }

        (Self::from_bits(fraction), exp)
    }

    #[inline(always)]
    fn flush_denormals<P: Policy>(self) -> Self {
        if const {
            matches!(
                P::POLICY.denormal_behavior,
                DenormalBehavior::Preserve | DenormalBehavior::Ignore
            ) || !<Self::Element as FloatElement>::HAS_SUBNORMALS
        } {
            return self;
        }

        if const { matches!(P::POLICY.denormal_behavior, DenormalBehavior::Crush) } {
            let denormal_trick: Self::Bits = crate::const_splat!(
                <Self> = <S: FloatVectorWithBits>
                <S::Bits as GenericVector>::Element: <S::Element as FloatElementWithBits>::DENORMAL_TRICK
            );

            let dt = Self::from_bits(denormal_trick);

            return dt - (dt - self);
        }

        let abs_bits = Self::SignedBits::from_bits(self.abs());

        let max_subnormal: Self::Bits = crate::const_splat!(
            <Self> = <S: FloatVectorWithBits>
            <S::Bits as GenericVector>::Element: <S::Element as FloatElementWithBits>::MAX_SUBNORMAL
        );

        let max_subnormal_signed: Self::SignedBits = Self::SignedBits::from_bits(max_subnormal);

        // zero self if subnormal (when cmp_gt is false)
        //
        // NOTE: Use a Signed comparison here, since that's faster than unsigned comparisons on most archs,
        // and we know that abs_bits is considered positive as an integer since the msb is zero.
        let mut res = Self::from_bits(self.zz(abs_bits.cmp_gt(max_subnormal_signed).cast()));

        // we should preserve -0.0 for greater precision policies
        if const { P::POLICY.precision.gt(PrecisionPolicy::Average) && <Self::Element as FloatElement>::HAS_SIGNED_ZERO }
        {
            // get the sign by xor-ing the non-sign bits, leaving only the sign
            let sign = Self::SignedBits::from_bits(self) ^ abs_bits;

            res |= Self::from_bits(sign); // add back sign
        }

        res
    }
}

/// `AsFloatVectorWithBitsKernel` that calls `flush_denormals` on each input with the given policy.
pub struct FlushDenormals<P: Policy>(PhantomData<P>);

impl<P: Policy> FlushDenormals<P> {
    #[inline(always)]
    pub fn flush_denormals<V: FloatVector, const N: usize>(values: [V; N]) -> Option<[V; N]> {
        V::with_bits(values, FlushDenormals::<P>(PhantomData))
    }
}

impl<P: Policy, const N: usize, V: FloatVector> AsFloatVectorWithBitsKernel<V, N> for FlushDenormals<P> {
    type Output = [V; N];

    #[inline(always)]
    fn with_bits<
        W: FloatVectorWithBits<
                Element = <V>::Element,
                Lanes = <V>::Lanes,
                Mask = <V>::Mask,
                Signed = <V>::Signed,
                Unsigned = <V>::Unsigned,
                ExtendedPrecision = <V as FloatVector>::ExtendedPrecision,
            > + CastVector<V>,
    >(
        self,
        v: [W; N],
    ) -> Self::Output {
        v.map(|v| W::cast_into(v.flush_denormals::<P>()))
    }
}

pub trait SpecializedCoreMath<E>: FloatVector<Element = E> {
    #[inline(always)]
    fn poly<P: Policy, const N: usize>(self, coeffs: &[E; N]) -> Self {
        let x = self;

        if const {
            !P::POLICY.unroll_loops
                || P::POLICY.precision.ge(PrecisionPolicy::Best)
                || !Self::ISA.has_instruction_level_parallelism()
        } {
            // SPIR-V is terrible at unrolling loops like this,
            // so we'll just do it ourselves.
            #[cfg(all(feature = "spirv", target_arch = "spirv"))]
            {
                use crunchy::unroll;

                let mut res = Self::splat(coeffs[N - 1]);

                macro_rules! unroll_poly {
                    ($($len:tt),*) => {
                        $(if const { N == $len } {
                            unroll! {
                                for i in 1..$len {
                                    res = res.mul_adde(x, Self::splat(coeffs[
                                        const { if $len > i + 1 { $len - 1 - i } else { 0 } }
                                    ]));
                                }
                            }
                        } else )* {
                            let mut i = const { N - 1 };
                            while i > 0 {
                                i -= 1;
                                unsafe { core::hint::assert_unchecked(i < N) };
                                res = res.mul_adde(x, Self::splat(coeffs[i]));
                            }
                        }
                    };
                }

                unroll_poly!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

                return res;
            }

            // basic Horner's method that's both compact and accurate, even without FMA
            let mut res = Self::splat(coeffs[N - 1]);
            for &c in coeffs.iter().rev().skip(1) {
                res = res.mul_adde(x, Self::splat(c));
            }
            return res;
        }

        // NumVector provides the num_traits::MulAdd implementation needed for fast_polynomial
        let res = fast_polynomial::poly_f_n::<_, _, N>(crate::vector::NumVector(x), |i| unsafe {
            crate::vector::NumVector(Self::splat(*coeffs.get_unchecked(i)))
        });

        res.0
    }

    #[inline(always)]
    fn poly_rev<P: Policy, const N: usize>(self, coeffs: &[E; N]) -> Self {
        let x = self;

        if const {
            !P::POLICY.unroll_loops
                || P::POLICY.precision.ge(PrecisionPolicy::Best)
                || !Self::ISA.has_instruction_level_parallelism()
        } {
            #[cfg(all(feature = "spirv", target_arch = "spirv"))]
            {
                use crunchy::unroll;

                let mut res = Self::splat(coeffs[0]);

                macro_rules! unroll_poly {
                    ($($len:tt),*) => {
                        $(if const { N == $len } {
                            unroll! {
                                for i in 1..$len {
                                    res = res.mul_adde(x, Self::splat(coeffs[i]));
                                }
                            }
                        } else )* {
                            let mut i = 1usize;
                            while i < N {
                                unsafe { core::hint::assert_unchecked(i < N) };
                                res = res.mul_adde(x, Self::splat(coeffs[i]));
                                i += 1;
                            }
                        }
                    };
                }

                unroll_poly!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

                return res;
            }

            // basic Horner's method that's both compact and accurate, even without FMA
            let mut res = Self::splat(coeffs[0]);
            for &c in coeffs.iter().skip(1) {
                res = res.mul_adde(x, Self::splat(c));
            }
            return res;
        }

        let res = fast_polynomial::poly_f_n::<_, _, N>(crate::vector::NumVector(x), |i| unsafe {
            crate::vector::NumVector(Self::splat(*coeffs.get_unchecked(N - 1 - i)))
        });

        res.0
    }

    #[inline(always)]
    fn poly_rational<P: Policy, const N: usize, const D: usize>(
        self,
        numerator: &[E; N],
        denominator: &[E; D],
    ) -> Self {
        let x = self;

        if const { P::POLICY.precision.le(PrecisionPolicy::Average) } {
            let n = Self::poly::<P, N>(x, numerator);
            let d = Self::poly::<P, D>(x, denominator);

            return n.approx_div_p::<P>(d);
        }

        let invert = x.cmp_gt(Self::ONE);

        let mut n0 = Self::EMPTY;
        let mut n1 = Self::EMPTY;
        let mut d0 = Self::EMPTY;
        let mut d1 = Self::EMPTY;

        if const { P::POLICY.avoid_branching } || !invert.all() {
            n0 = Self::poly::<P, N>(x, numerator);
            d0 = Self::poly::<P, D>(x, denominator);
        }

        let mut z = Self::EMPTY;

        if const { P::POLICY.avoid_branching } || invert.any() {
            z = Self::reciprocal::<P>(x);
            n1 = Self::poly_rev::<P, N>(z, numerator);
            d1 = Self::poly_rev::<P, D>(z, denominator);
        }

        let n = invert.select(n1, n0);
        let d = invert.select(d1, d0);

        let res = n.approx_div_p::<P>(d);

        // no correction needed if same degree
        if const { N == D } {
            return res;
        }

        if const { P::POLICY.avoid_branching } || invert.any() {
            // when the degree of the numerator and denominator are different, we need to correct
            // the result by shifting over the difference in degrees
            let (mut u, mut e) = if N < D { (z, D - N) } else { (x, N - D) };

            let mut corrected = res;

            // `res = res * powi(u, e)` assuming e > 0
            // because e > 0 we can jump straight into the loop without a pre-check,
            // and avoid an extra square of u at the end
            loop {
                if e & 1 != 0 {
                    corrected *= u;
                }

                e >>= 1;

                if e == 0 {
                    // correction isn't actually needed for non-inverted case
                    return invert.select(corrected, res);
                }

                u = u.square();
            }
        }

        res
    }

    #[inline(always)]
    fn reciprocal<P: Policy>(self) -> Self {
        if const { Self::HAS_APPROX_RCP && P::POLICY.precision.ge(PrecisionPolicy::Best) } {
            return Self::ONE / self;
        }

        let mut y = self.rcp();

        // if we have approximate reciprocal and want better precision
        if const { Self::HAS_APPROX_RCP && P::POLICY.precision.gt(PrecisionPolicy::Worst) } {
            // one iteration of Newton's method
            y = y * self.nmul_adde(y, Self::TWO);
        }

        y
    }

    #[inline(always)]
    fn approx_div<P: Policy>(self, rhs: Self) -> Self {
        if const { Self::HAS_APPROX_RCP && P::POLICY.precision.gt(PrecisionPolicy::Worst) } {
            return self / rhs;
        }

        self * rhs.rcp()
    }

    #[inline(always)]
    fn reciprocal_adde<P: Policy>(self, a: Self) -> Self {
        if const { Self::HAS_APPROX_RCP && P::POLICY.precision.ge(PrecisionPolicy::Best) } {
            return Self::ONE / self + a;
        }

        let mut y = self.rcp();

        if const { Self::HAS_APPROX_RCP && P::POLICY.precision.gt(PrecisionPolicy::Worst) } {
            // one iteration of Newton's method
            y = y.mul_adde(self.nmul_adde(y, Self::TWO), a);
        } else {
            y += a;
        }

        y
    }

    fn inverse_sqrt<P: Policy>(self) -> Self;

    // TODO: Look into better algorithms than doubling
    #[inline(always)]
    fn powi<P: Policy>(self, e: i32) -> Self {
        let mut x = self;
        let mut res = Self::ONE;

        let mut e = if e < 0 {
            x = Self::reciprocal::<P>(x);

            e.wrapping_neg() as u32
        } else {
            e as u32
        };

        while e != 0 {
            if e & 1 != 0 {
                res *= x;
            }

            x = x.square();
            e >>= 1;
        }

        res
    }

    #[inline(always)]
    fn powic<P: Policy, const N: i32>(self) -> Self {
        self.powi_p::<P>(N)
    }

    #[inline(always)]
    fn powiv<P: Policy>(self, mut e: Self::Signed) -> Self {
        let mut x = self;
        let mut res = Self::ONE;

        x = e.is_negative().select(Self::reciprocal::<P>(x), x);
        e = e.abs();

        loop {
            let nx = res * x;

            res = (e & Self::Signed::ONE).is_zero().select(res, nx);

            e >>= 1;

            if e.is_all_zero() {
                return res;
            }

            x = x.square();
        }
    }
}

pub trait SpecializedTranscendentalMath<E>: SpecializedCoreMath<E> {
    fn sin_cos<P: Policy>(self) -> (Self, Self);

    #[inline(always)]
    fn sin<P: Policy>(self) -> Self {
        Self::sin_cos::<P>(self).0
    }

    #[inline(always)]
    fn cos<P: Policy>(self) -> Self {
        Self::sin_cos::<P>(self).1
    }

    #[inline(always)]
    fn tan<P: Policy>(self) -> Self {
        let (s, c) = Self::sin_cos::<P>(self);
        s / c
    }

    #[inline(always)]
    fn sincos_pi<P: Policy>(self) -> (Self, Self) {
        Self::sin_cos::<P>(self * Self::PI)
    }

    #[inline(always)]
    fn sin_pi<P: Policy>(self) -> Self {
        Self::sincos_pi::<P>(self).0
    }

    #[inline(always)]
    fn cos_pi<P: Policy>(self) -> Self {
        Self::sincos_pi::<P>(self).1
    }

    #[inline(always)]
    fn tan_pi<P: Policy>(self) -> Self {
        let (s, c) = Self::sincos_pi::<P>(self);
        s.approx_div_p::<P>(c)
    }

    fn sinc<P: Policy>(self) -> Self;

    #[inline(always)]
    fn sinc_pi<P: Policy>(self) -> Self {
        Self::sinc::<P>(self * Self::PI)
    }

    fn sinh_cosh<P: Policy>(self) -> (Self, Self);

    #[inline(always)]
    fn sinh<P: Policy>(self) -> Self {
        Self::sinh_cosh::<P>(self).0
    }

    #[inline(always)]
    fn cosh<P: Policy>(self) -> Self {
        Self::sinh_cosh::<P>(self).1
    }

    fn tanh<P: Policy>(self) -> Self;

    fn asin<P: Policy>(self) -> Self;
    fn acos<P: Policy>(self) -> Self;
    fn atan<P: Policy>(self) -> Self;

    fn asinh<P: Policy>(self) -> Self;
    fn acosh<P: Policy>(self) -> Self;
    fn atanh<P: Policy>(self) -> Self;

    fn exp<P: Policy>(self) -> Self;
    fn exph<P: Policy>(self) -> Self;
    fn exp2<P: Policy>(self) -> Self;
    fn exp10<P: Policy>(self) -> Self;
    fn exp_m1<P: Policy>(self) -> Self;
    fn exp2_m1<P: Policy>(self) -> Self;
    fn exp10_m1<P: Policy>(self) -> Self;

    fn powf<P: Policy>(self, e: Self) -> Self;
    fn cbrt<P: Policy>(self) -> Self;

    #[inline(always)]
    fn sqrt1pm1<P: Policy>(self) -> Self {
        // sqrt(1 + x) - 1 = x / (sqrt(1 + x) + 1); no cancellation near x = 0.
        let s = (self + Self::ONE).sqrt();
        let mut r = Self::approx_div::<P>(self, s + Self::ONE);

        if const { P::POLICY.check_overflow } {
            // x = +inf would give inf/inf = NaN; the naive form is correct for the
            // non-finite inputs (and only those need the fallback).
            r = self.is_finite().select(r, s - Self::ONE);
        }

        r
    }

    #[inline(always)]
    fn compound<P: Policy>(self, n: Self) -> Self {
        // (1 + x)^n = exp(n * ln(1 + x)); routing through ln_1p keeps it accurate for small x.
        let l = Self::ln_1p::<P>(self);
        let p = n * l;

        // The Dekker residual below is only a residual if the product is single rounded,
        // which without FMA hardware would mean emulated FMA, never used in these
        // kernels. Non-FMA backends keep the uncorrected form (mean ~3.5 vs ~1.8 ulp
        // over a dense sweep) rather than pay the emulation.
        if const { P::POLICY.precision.le(PrecisionPolicy::Average) || !Self::HAS_TRUE_FMA } {
            return Self::exp::<P>(p);
        }

        // Any absolute error in the exponent is relative error in the result, and at a
        // large |n * l| the product's own rounding dominates everything: half an ulp of
        // p = 488 (x = 0.05, n = 1e4) is 2.8e-14, i.e. ~250 ulp of the answer. The Dekker
        // residual recovers that term exactly (a single fused instruction on this
        // hardware), and first-order correction is all it needs, since
        // e^(p + lo) = e^p * (1 + lo + O(lo^2)) with lo^2 < 1e-27 relative. What remains
        // is n * (the single-width error of `ln_1p` itself). Shrinking that needs a
        // double-double log core.
        let p_lo = n.mul_sube(l, p);
        let e = Self::exp::<P>(p);

        // The correction is only meaningful (and only safe) on a finite result: a
        // legitimate overflow gives e = inf, where `p_lo * inf + inf` is NaN for a
        // negative residual.
        e.is_finite().select(p_lo.mul_adde(e, e), e)
    }

    #[inline(always)]
    fn powf_m1<P: Policy>(self, e: Self) -> Self {
        // x^e - 1 = expm1(e * ln(x)); avoids the outer cancellation of pow(x, e) - 1.
        let l = Self::ln::<P>(self);
        let p = e * l;

        // As in `compound`, the residual needs a real FMA. Non-FMA backends keep the
        // uncorrected form rather than pay emulation.
        if const { P::POLICY.precision.le(PrecisionPolicy::Average) || !Self::HAS_TRUE_FMA } {
            return Self::exp_m1::<P>(p);
        }

        // Same Dekker product-residual as `compound` (see there): expm1(p + lo) =
        // expm1(p) + e^p * lo = r + (r + 1) * lo to first order.
        let p_lo = e.mul_sube(l, p);
        let r = Self::exp_m1::<P>(p);

        // As in `compound`: no correction on an overflowed (infinite) result.
        r.is_finite().select(p_lo.mul_adde(r + Self::ONE, r), r)
    }

    #[inline(always)]
    fn haversin<P: Policy>(self) -> Self {
        // (1 - cos(x)) / 2 = sin^2(x/2); no cancellation near x = 0.
        let s = Self::sin::<P>(self * Self::HALF);
        s * s
    }

    #[inline(always)]
    fn versin<P: Policy>(self) -> Self {
        // 1 - cos(x) = 2 sin^2(x/2)
        let h = Self::haversin::<P>(self);
        h + h
    }

    #[inline(always)]
    fn cos_m1<P: Policy>(self) -> Self {
        // cos(x) - 1 = -(1 - cos(x))
        -Self::versin::<P>(self)
    }

    #[inline(always)]
    fn nth_root<P: Policy, const N: usize>(self) -> Self {
        let mut x = self;

        match N {
            0 => Self::NAN, // undefined
            1 => x,
            2 => x.sqrt(),
            3 => x.cbrt_p::<P>(),

            // 4th root is just two square roots, and for regular precision policies this is usually faster than a dedicated 4th root method
            4 if const { P::POLICY.precision.le(PrecisionPolicy::Average) } => x.sqrt().sqrt(),

            _ => {
                let mut is_neg = GenericMask::FALSY;

                // for odd powers, work with absolute value and restore sign later
                if const { N & 1 == 1 } {
                    is_neg = x.is_negative();
                    x = x.abs(); // abs is faster than neg_c, just 1 AND
                }

                // initial guess using reduced precision
                let y = x.powf_p::<LessPrecision<P>>(Self::splat(E::from_ratio(1, N as crate::LargeInt)));

                // One iteration of Halley's method for nth root
                let y_n = y.powi_p::<P>(N as i32);

                let np1 = Self::splat(E::from_int((N + 1) as crate::LargeInt));
                let nm1 = Self::splat(E::from_int((N - 1) as crate::LargeInt));

                // Dimensionless form of the correction: q = y^N / x is ~1 whatever the
                // magnitude of x, so t ~ guess_error / 2N and nothing here can overflow
                // or underflow. The textbook `y * (x - y^N) / ((N+1) y^N + (N-1) x)` has
                // an O(x^{(N+1)/N}) numerator: for N = 5 it overflowed past x ~ 1e269
                // (returning sign-garbage infinities) and underflowed to zero below
                // x ~ 1e-250, silently dropping the refinement there. The extra
                // division's rounding only perturbs t by ~ulp, an O(ulp/2N) relative
                // effect on y, far below the final rounding.
                let q = y_n / x;
                let t = (Self::ONE - q) / q.mul_adde(np1, nm1);

                // y += 2*y*t
                let mut y2 = t.mul_adde(y + y, y);

                if const { P::POLICY.check_overflow } {
                    // x = 0 makes q = 0/0 and x = inf makes 1 - inf/inf: NaN in t,
                    // while the uncorrected guess is already exact for both. NaN
                    // inputs still pass through (the guess is NaN too).
                    y2 = (x.cmp_eq(Self::ZERO) | x.is_infinite()).select(y, y2);
                }

                if const { N & 1 == 1 } {
                    y2 = y2.neg_c(is_neg);
                }

                y2
            }
        }
    }

    fn ln<P: Policy>(self) -> Self;
    fn ln_1p<P: Policy>(self) -> Self;
    fn log2<P: Policy>(self) -> Self;
    fn log10<P: Policy>(self) -> Self;

    // log_b(1 + x) = ln(1 + x) / ln(b) = ln_1p(x) * log_b(e). Routing through the
    // cancellation-safe ln_1p keeps the near-zero accuracy; scaling by a constant
    // preserves the relative error.
    #[inline(always)]
    fn log2_p1<P: Policy>(self) -> Self {
        Self::ln_1p::<P>(self) * Self::LOG2_E
    }

    #[inline(always)]
    fn log10_p1<P: Policy>(self) -> Self {
        Self::ln_1p::<P>(self) * Self::LOG10_E
    }

    fn log_n<P: Policy, const N: usize>(self) -> Self;

    #[inline(always)]
    fn log<P: Policy>(self, base: Self) -> Self {
        Self::ln::<P>(self) / Self::ln::<P>(base)
    }

    /// ln(1 - e^(-x))
    #[inline(always)]
    fn ln1m_expnx<P: Policy>(self) -> Self {
        Self::ln::<P>(Self::ONE - Self::exp::<P>(-self))
    }

    fn ln1m_expnx_ext<P: Policy>(self, lnx: Self) -> Self;
}

#[inline(always)]
fn hypot_n_impl<E, V, P, const N: usize, const INV: bool>(mut values: [V; N]) -> V
where
    E: FloatElement,
    V: SpecializedSpatialMath<E>,
    P: Policy,
{
    #[cfg(not(target_arch = "spirv"))]
    if let Some(new_values) = FlushDenormals::<P>::flush_denormals(values) {
        values = new_values;
    }

    if const { N == 0 } {
        if INV {
            return V::INFINITY; // 1/0 == infinity
        }

        return V::ZERO;
    }

    if const { N == 1 } {
        let mut res = values[0].abs(); // sqrt(x^2) == abs(x)

        if INV {
            res = res.reciprocal_p::<P>();
        }

        return res;
    }

    // special case N=2 which saves a couple instructions
    if const { N == 2 } {
        let x = values[0];
        let y = values[1];

        return if const { P::POLICY.precision.le(PrecisionPolicy::Worst) } {
            // Use the worst precision method, which is usually faster
            let res = x.mul_adde(x, y.square());

            return if INV { res.inverse_sqrt_p::<P>() } else { res.sqrt() };
        } else {
            // Use a more precise method
            let x = x.abs();
            let y = y.abs();

            let max = x.max(y);
            let min = x.min(y);

            // guard the all-zero input: max == 0 would make min/max = 0/0 = NaN.
            // Dividing by 1 instead yields t = 0, so the norm is 0 (and the
            // inverse norm is +inf), matching the general N-ary path below.
            let t = min / max.cmp_eq(V::ZERO).select(V::ONE, max);

            let s = t.mul_adde(t, V::ONE); // 1 + t^2

            let mut res;
            if INV {
                res = s.inverse_sqrt_p::<P>() / max;

                if const { P::POLICY.check_overflow } {
                    res = max.is_infinite().select(V::ZERO, res);
                }
            } else {
                res = max * s.sqrt();

                if const { P::POLICY.check_overflow } {
                    res = max.is_infinite().select(max, res);
                }
            }

            res
        };
    }

    if const { P::POLICY.precision.le(PrecisionPolicy::Worst) } {
        // square each value in place, zero dependencies
        for value in values.iter_mut() {
            *value *= *value;
        }

        crate::math::algorithms::reduce_in_place(&mut values, |a, b| a + b);

        return if INV {
            values[0].inverse_sqrt_p::<P>()
        } else {
            values[0].sqrt()
        };
    }

    // high-precision path

    // take absolute value of each element in place, zero dependencies,
    // since we're squaring anyway this doesn't lose any information
    for x in &mut values {
        *x = x.abs();
    }

    let max_abs = crate::math::algorithms::reduce_array(values, |a, b| a.max(b));
    let is_zero = max_abs.cmp_eq(V::ZERO);

    let scale = is_zero.select(V::ONE, max_abs.reciprocal_p::<P>());

    for x in &mut values {
        *x *= scale; // scale to prevent overflow
        *x = x.square(); // square in place
    }

    // sum squares in place
    crate::math::algorithms::reduce_in_place(&mut values, |a, b| a + b);

    let mut res;

    if INV {
        res = scale * values[0].inverse_sqrt_p::<P>();

        if const { P::POLICY.check_overflow } {
            res = max_abs.is_infinite().select(V::ZERO, res);
        }
    } else {
        res = max_abs * values[0].sqrt();

        if const { P::POLICY.check_overflow } {
            res = max_abs.is_infinite().select(max_abs, res);
        }
    }

    res
}

pub trait SpecializedSpatialMath<E>: SpecializedCoreMath<E> {
    // type Scalar: SpecializedRealMath<E>;

    #[inline(always)]
    fn hypot<P: Policy>(self, y: Self) -> Self {
        Self::hypot_n::<P, 2>([self, y])
    }

    #[inline(always)]
    fn hypot_n<P: Policy, const N: usize>(values: [Self; N]) -> Self {
        hypot_n_impl::<E, Self, P, N, false>(values)
    }

    #[inline(always)]
    fn inv_hypot_n<P: Policy, const N: usize>(values: [Self; N]) -> Self {
        hypot_n_impl::<E, Self, P, N, true>(values)
    }

    fn l1_norm<P: Policy>(self) -> Self;

    #[inline(always)]
    fn l2_norm<P: Policy>(self) -> Self {
        Self::l2_norm_squared::<P>(self).sqrt()
    }

    fn l2_norm_squared<P: Policy>(self) -> Self;
}

pub trait SpecializedRealMath<E>: SpecializedTranscendentalMath<E> + SpecializedSpatialMath<E> {
    #[inline(always)]
    fn tolerance<P: Policy>() -> Self {
        Self::splat(Self::Element::from_int(P::POLICY.precision.tolerance()) * Self::Element::EPSILON)
    }

    #[inline(always)]
    fn to_degrees<P: Policy>(self) -> Self {
        self * Self::FRAC_180_PI
    }

    #[inline(always)]
    fn to_radians<P: Policy>(self) -> Self {
        self * Self::FRAC_PI_180
    }

    #[inline(always)]
    fn wrap_angle<P: Policy>(self) -> Self {
        // self - floor((self + π) / 2π) * 2π
        (-Self::TAU).mul_adde(((self + Self::PI) * (Self::FRAC_1_PI * Self::HALF)).floor(), self)
    }

    #[inline(always)]
    fn angle_diff<P: Policy>(self, other: Self) -> Self {
        (self - other).wrap_angle_p::<P>()
    }

    fn atan2<P: Policy>(self, x: Self) -> Self;

    #[inline(always)]
    fn step<P: Policy>(self, t: Self) -> Self {
        // use z() masked zeroing to avoid branching or select
        Self::ONE.zz(self.cmp_ge(t))
    }

    #[inline(always)]
    fn lerp<P: Policy>(self, a: Self, b: Self) -> Self {
        self.mix(a, b)
    }

    #[inline(always)]
    fn rescale<P: Policy>(self, in_min: Self, in_max: Self, out_min: Self, out_max: Self) -> Self {
        let in_range = in_max - in_min;

        let mut t = self - in_min;

        t = if const { P::POLICY.precision.le(PrecisionPolicy::Worst) } {
            t * in_range.rcp()
        } else {
            t / in_range
        };

        Self::lerp::<P>(t, out_min, out_max)
    }

    #[inline(always)]
    fn logaddexp<P: Policy>(self, other: Self) -> Self {
        // max(a, b) + ln(1 + exp(-|a - b|)): stable against overflow for large a, b.
        let m = self.max(other);
        let d = (self - other).abs();
        let mut r = m + Self::ln_1p::<P>(Self::exp::<P>(-d));

        if const { P::POLICY.check_overflow } {
            // a == b == +-inf makes a - b NaN; the answer is that infinity (= m).
            r = d.is_nan().select(m, r);
        }

        r
    }

    #[inline(always)]
    fn logsumexp_n<P: Policy, const N: usize>(mut values: [Self; N]) -> Self {
        // The empty sum is 0, and ln(0) = -inf: the identity element of logaddexp,
        // so folding logsumexp_n over any partition of the inputs agrees.
        if const { N == 0 } {
            return Self::NEG_INFINITY;
        }

        if const { N == 1 } {
            return values[0];
        }

        // The pairwise form is a max, a subtract and one exp. The general path below
        // cannot beat that, and `logaddexp` already handles its own edge cases.
        if const { N == 2 } {
            return Self::logaddexp::<P>(values[0], values[1]);
        }

        // Tree-reduced max, and then a tree-reduced sum below: the max gates every
        // exp and the sum gates the final log, so both reductions sit on the
        // critical path, where O(log N) dependency depth beats a running fold's
        // O(N). (`reduce_array` copies, so `values` is still intact after this.)
        let m = algorithms::reduce_array(values, |a, b| a.max(b));

        let mut r;

        if const { P::POLICY.precision.le(PrecisionPolicy::Worst) } {
            let mut i = 0;
            while i < N {
                values[i] = Self::exp::<P>(values[i] - m);
                i += 1;
            }

            algorithms::reduce_in_place(&mut values, |a, b| a + b);

            r = m + Self::ln::<P>(values[0]);
        } else {
            // Sum every term _except_ the dominant one, so the total can go into
            // `ln_1p`. The one term at the max contributes exactly exp(0) = 1, which is
            // the term `ln_1p` supplies exactly. Summing it in and taking `ln` instead
            // would round `1 + s` before the log ever saw `s`, and in the usual
            // log-domain case (one weight dominating the rest) that rounding is the whole
            // answer.
            //
            // `used` is what keeps ties honest: duplicates of the max must still
            // contribute, so exactly the first lane-wise occurrence is dropped. It is a
            // serial chain across the loop, but of single-cycle bitwise ops, so the exps
            // around it stay independent.
            let mut used = <Self::Mask as GenericMask>::FALSY;

            let mut i = 0;
            while i < N {
                let d = values[i] - m;
                let dominant = d.cmp_eq(Self::ZERO).bitandnot(used);

                used |= dominant;
                values[i] = Self::exp::<P>(d).nz(dominant);

                i += 1;
            }

            algorithms::reduce_in_place(&mut values, |a, b| a + b);

            r = m + Self::ln_1p::<P>(values[0]);
        }

        if const { P::POLICY.check_overflow } {
            // An infinite (or NaN) max is the answer: every difference against it is
            // NaN otherwise. All-(-inf) inputs are the log-domain zero and must stay
            // -inf rather than becoming NaN, which is what makes this worth a select.
            r = m.is_finite().select(r, m);
        }

        r
    }

    #[inline(always)]
    fn logsubexp<P: Policy>(self, other: Self) -> Self {
        // ln(e^a - e^b) = a + ln(1 - e^-(a - b)). Nothing here needs a max: the gap has
        // to be positive for the result to exist at all, so the subtraction is already
        // the stable one.
        let d = self - other;

        // `ln(1 - e^-d)` is exactly `ln1m_expnx` of the gap, and the element ladders carry
        // the regime handling. At Average and above both f32 and f64 use the shared
        // two-branch Maechler kernel (`generic::ln1m_expnx_internal`, whose docs cover why
        // no single expression survives both ends of the gap). The lower tiers keep their
        // cheap forms (f32: the clamped rational approximation, f64: the naive
        // expression) with the accuracy losses those tiers accept.
        let mut r = self + Self::ln1m_expnx::<P>(d);

        if const { P::POLICY.check_overflow } {
            // a == b == -inf is the log-domain 0 - 0: the difference is NaN, but the
            // answer is 0, i.e. -inf. (a == b == +inf correctly stays NaN, since
            // inf - inf is not defined.)
            r = (d.is_nan() & self.cmp_eq(Self::NEG_INFINITY)).select(Self::NEG_INFINITY, r);
        }

        r
    }

    #[inline(always)]
    fn smoothstep<P: Policy, const N: usize>(self, edges: Option<(Self, Self)>) -> Self {
        let mut t = self;

        #[cfg(not(target_arch = "spirv"))]
        if let Some(new_t) = FlushDenormals::<P>::flush_denormals([t]) {
            t = new_t[0];
        }

        if let Some((a, b)) = edges {
            let xa = t - a;
            let ba = b - a;

            t = if const { P::POLICY.precision.le(PrecisionPolicy::Worst) } {
                xa * ba.rcp()
            } else {
                xa / ba
            };
        }

        if const { P::POLICY.check_overflow } {
            t = t.clamp(Self::ZERO, Self::ONE);
        }

        match N {
            // t was already scaled to between the edges
            0 => Self::step::<P>(t, Self::HALF),
            1 => t, // linear
            _ => {
                let coeffs = const { Smoothstep::<N>::COEFFICIENTS };
                let mut y = Self::splat(E::from_int(coeffs[0]));

                let mut i = 1usize;
                while i < N {
                    #[cfg(all(feature = "spirv", target_arch = "spirv"))]
                    let c = coeffs[i];
                    #[cfg(not(all(feature = "spirv", target_arch = "spirv")))]
                    let c = unsafe { *coeffs.get_unchecked(i) };
                    y = y.mul_adde(t, Self::splat(E::from_int(c)));
                    i += 1;
                }

                y * t.powi_p::<P>(N as i32)
            }
        }
    }

    #[inline(always)]
    fn smoothstep_derivative<P: Policy, const N: usize>(self, edges: Option<(Self, Self)>) -> Self {
        let mut t = self;
        let mut dt_dx = Self::ONE;

        #[cfg(not(target_arch = "spirv"))]
        if let Some(new_t) = FlushDenormals::<P>::flush_denormals([t]) {
            t = new_t[0];
        }

        if let Some((a, b)) = edges {
            let xa = t - a;
            let ba = b - a;

            (dt_dx, t) = if const { P::POLICY.precision.le(PrecisionPolicy::Worst) } {
                let bar = ba.rcp();

                (bar, xa * bar)
            } else {
                (ba.reciprocal_p::<P>(), xa / ba)
            };
        }

        match N {
            // derivative of step function is infinite at 0.5, so-called Dirac delta function
            0 => t.cmp_eq(Self::HALF).select(Self::INFINITY, Self::ZERO),
            1 => dt_dx,
            _ => {
                if const { P::POLICY.check_overflow } {
                    t = t.clamp(Self::ZERO, Self::ONE);
                }

                let coeffs = const { Smoothstep::<N>::COEFFICIENTS };
                let mut y = Self::splat(E::from_int(coeffs[0] * (2 * N - 1) as crate::LargeInt));
                let mut k = 1usize;

                while k < N {
                    #[cfg(all(feature = "spirv", target_arch = "spirv"))]
                    let c = coeffs[k];
                    #[cfg(not(all(feature = "spirv", target_arch = "spirv")))]
                    let c = unsafe { *coeffs.get_unchecked(k) };
                    // order - k for derivative coefficient
                    y = y.mul_adde(t, Self::splat(E::from_int(c * (2 * N - k - 1) as crate::LargeInt)));
                    k += 1;
                }

                y * dt_dx * t.powi_p::<P>((N - 1) as i32)
            }
        }
    }

    #[inline(always)]
    fn inverse_smoothstep<P: Policy, const N: usize>(mut y: Self, edges: Option<(Self, Self)>) -> Self {
        let mut ba = Self::ONE;
        let mut bar = Self::ONE;
        let mut bar_a = Self::ONE; // (b - a) * a

        #[cfg(not(target_arch = "spirv"))]
        if let Some(new_y) = FlushDenormals::<P>::flush_denormals([y]) {
            y = new_y[0];
        }

        //                             // Initial guess: y - 2y * (1 - y) * (y - 0.5)
        // While we have a good initial guess for the inverse, S-curves are most stable at the
        // midpoint, so start there. Converges much faster this way.
        let mut x0 = Self::HALF; //(y + y).nmul_adde((Self::ONE - y) * (y - Self::HALF), y);

        if let Some((a, b)) = edges {
            ba = b - a;

            if const { P::POLICY.precision.le(PrecisionPolicy::Worst) } {
                bar = ba.rcp();
                bar_a = bar * a;
            } else {
                bar = ba.reciprocal_p::<P>();
                bar_a = a / ba;
            }

            match N {
                0 => return y.step_p::<P>(Self::HALF).mul_adde(ba, a),
                1 => return y.mul_adde(ba, a),

                // scale the initial guess to fit the edges
                _ => x0 = x0.mul_adde(ba, a),
            }
        }

        match N {
            0 => return y.step_p::<P>(Self::HALF),
            1 => return y,

            // N=2 has a closed-form solution
            2 => {
                let mut t = y.nmul_adde(Self::TWO, Self::ONE).asin_p::<P>();

                if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
                    t *= Self::splat(E::from_ratio(1, 3)); // multiply by 1/3 for medium precision
                } else {
                    // exact division for higher precisions
                    t /= Self::splat(E::from_int(3));
                }

                t = Self::HALF - t.sin_p::<P>();

                if let Some((a, _)) = edges {
                    // rescale to original edges
                    t = t.mul_adde(ba, a);
                }

                return t;
            }
            _ => {}
        }

        let bounds = edges.or(Some((Self::ZERO, Self::ONE)));

        #[rustfmt::skip]
        let (v, _converged) = algorithms::newtons_method::<Self, P, _>(x0, Self::tolerance::<P>(), bounds, #[inline(always)] move |x: Self| {
            let mut t = x;
            let dt_dx = bar;

            if edges.is_some() {
                // adjust by precalculated scales
                t = t.mul_sube(bar, bar_a);
            }

            // This closure is only reached for N >= 3, but it is still monomorphized
            // (and its const-generic arithmetic const-evaluated) for N = 0/1, where
            // `N - 1` / `2*N - 1` would underflow `usize` at compile time.
            let xn1 = t.powi_p::<P>(N as i32 - 1);

            let coeffs = const { Smoothstep::<N>::COEFFICIENTS };

            let mut fx = Self::splat(E::from_int(coeffs[0]));
            let mut fpx = Self::splat(E::from_int(coeffs[0] * (2 * N).saturating_sub(1) as crate::LargeInt));

            let mut k = 1usize;

            while k < N {
                #[cfg(all(feature = "spirv", target_arch = "spirv"))]
                let c = coeffs[k];
                #[cfg(not(all(feature = "spirv", target_arch = "spirv")))]
                let c = unsafe { *coeffs.get_unchecked(k) };

                fx = fx.mul_adde(t, Self::splat(E::from_int(c)));
                fpx = fpx.mul_adde(t, Self::splat(E::from_int(c * (2 * N - k - 1) as crate::LargeInt)));

                k += 1;
            }

            (t.mul_sube(xn1 * fx, y), (fpx * dt_dx * xn1).min(Self::HALF))
        });

        v
    }

    #[inline(always)]
    fn smooth_interpolator<P: Policy>(x: Self, edges: Option<(Self, Self)>, k: Self) -> Self {
        let mut t = x;

        #[cfg(not(target_arch = "spirv"))]
        if let Some(new_t) = FlushDenormals::<P>::flush_denormals([t]) {
            t = new_t[0];
        }

        if let Some((a, b)) = edges {
            // rescale t to [0, 1]
            t = (t - a) / (b - a);
        }

        let kt = k * t;

        // (2x-1) / (kx^2-kx)
        let e = t.mul_sube(Self::TWO, Self::ONE) / kt.mul_sube(t, kt);

        // exp(e) + 1
        let d = e.exp_p::<P>() + Self::ONE;

        // 1/(exp(e) + 1), it's important this is done in extra precision
        let mut res = d.reciprocal_p::<ExtraPrecision<P>>();

        let overflow = e.is_infinite();

        // If the denominator is small enough, it could cause overflow,
        // however that only really happens when t is very close to 0 or 1,
        // or when k is very small. So approximate it with a step function.
        if const { P::POLICY.avoid_branching } || crate::unlikely(overflow.any()) {
            res = overflow.select(t.step_p::<P>(Self::HALF), res);
        }

        // these are important since the Exp formulation is discontinuous at 0 and 1,
        // and this maintains the asymptotes when t is outside the range [0, 1]
        res = t.cmp_ge(Self::ONE).select(Self::ONE, res);
        res = t.cmp_le(Self::ZERO).select(Self::ZERO, res);

        res
    }

    #[inline(always)]
    fn smooth_interpolator_inverse<P: Policy>(y: Self, edges: Option<(Self, Self)>, k: Self) -> Self {
        // k ln(1/y - 1)
        let l = k * (y.reciprocal_p::<P>() - Self::ONE).ln_p::<P>();

        // ((l + 2) - sqrt(l^2 + 4)) / 2l
        let a = l + Self::TWO;
        let b = l.mul_adde(l, Self::splat(E::from_int(4))).sqrt();
        let mut t = (a - b) / (Self::TWO * l);

        // handle out-of-bounds inputs
        t = y.cmp_ge(Self::ONE).select(Self::ONE, t);
        t = y.cmp_le(Self::ZERO).select(Self::ZERO, t);

        if let Some((a, b)) = edges {
            // rescale t to the original edges
            t = t.mul_adde(b - a, a);
        }

        t
    }
}

// /// Provides specialized math routines for the given element type.
// ///
// /// Vectors implementing this will automatically have the [`Math`] and [`MathWithPolicy`] traits
// /// implemented for them.
// pub trait SpecializedMath<E>: SpecializedCoreMath<E> {
//     fn erf<P: Policy>(self) -> Self;

//     #[inline(always)]
//     fn erfc<P: Policy>(self) -> Self {
//         Self::ONE - Self::erf::<P>(self) // erfc(x) = 1 - erf(x), fallback implementation
//     }

//     fn erfinv<P: Policy>(self) -> Self;
// }

pub(crate) mod pd;
pub(crate) mod ps;

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum ExpMode {
    Exp = 0,
    Expm1,
    Exph,
    Pow2,
    Pow2m1,
    Pow10,
    Pow10m1,
}

const EXP_MODE_EXP: u8 = ExpMode::Exp as u8;
const EXP_MODE_EXPM1: u8 = ExpMode::Expm1 as u8;
const EXP_MODE_EXPH: u8 = ExpMode::Exph as u8;
const EXP_MODE_POW2: u8 = ExpMode::Pow2 as u8;
const EXP_MODE_POW2M1: u8 = ExpMode::Pow2m1 as u8;
const EXP_MODE_POW10: u8 = ExpMode::Pow10 as u8;
const EXP_MODE_POW10M1: u8 = ExpMode::Pow10m1 as u8;

const fn binomial(a: i32, b: i32) -> crate::LargeInt {
    if b <= 0 {
        return 1;
    }

    let mut res: crate::LargeInt = 1;
    let mut i = 0;

    while i < b {
        let n: crate::LargeInt = res * (a - i) as crate::LargeInt;
        res = n / (i + 1) as crate::LargeInt;

        i += 1;
    }

    res
}

// const fn const_powi(base: i64, exp: i32) -> i64 {
//     let mut result: i64 = 1;
//     let mut b = base;
//     let mut e = exp;

//     while e > 0 {
//         if e & 1 != 0 {
//             result = result * b;
//         }

//         e >>= 1;

//         if e == 0 {
//             break;
//         }

//         b *= b;
//     }

//     result
// }

// /// Returns (numerator, denominator) of the n-th generalized harmonic number of order m
// const fn generalized_harmonic(n: i32, m: i32) -> (i64, i64) {
//     let mut n = 0;
//     let mut d = 0;

//     let mut k = 1;

//     while k <= n {
//         d += const_powi(k, m);
//         n += 1;
//     }

//     (n, d)
// }

pub struct Smoothstep<const N: usize>(PhantomData<[crate::LargeInt; N]>);

impl<const N: usize> Smoothstep<N> {
    // ensure these coefficients are generated at compile time
    pub const COEFFICIENTS: [crate::LargeInt; N] = const {
        let mut coeffs = [0; N];
        // `N as i32 - 1` (not `(N - 1) as i32`) so the N=0 case - an empty coeff
        // array whose loop never runs, so `n` is unused - doesn't underflow `usize`
        // at compile time. This lets `smoothstep`/`inverse_smoothstep::<0>` compile.
        let n = N as i32 - 1;

        let mut k = 0;
        while k < N {
            let c = binomial(-1 - n, k as i32) * binomial(n + n + 1, n - k as i32);

            // store in reverse order for easier polynomial evaluation
            coeffs[N - k - 1] = c;
            k += 1;
        }

        coeffs
    };
}
