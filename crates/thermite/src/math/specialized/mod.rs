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
//! # The element as the unit of specialization
//!
//! Splitting the implementation out by *element type* (rather than by vector or
//! backend) is what lets the math library extend to composite number systems.
//! A vector's element is not required to be a primitive `f32`/`f64`: it can
//! itself be a structured value, and a math implementation written against that
//! element flows through the exact same public traits.
//!
//! The motivating case is compensated (double-double) arithmetic: a
//! `Compensated` vector's "element" is itself a `Compensated` value, so
//! implementing the specialized traits for that element gives every
//! `Compensated` vector full transcendental support with no changes to generic
//! callers. The same pattern is intended for `Complex` (in `thermite-complex`) and dual/hyperdual
//! numbers as those land: implement the specialized math for the new element
//! type and the entire public math API lights up for it automatically.
//!
//! Most code should never name these traits directly. Bound on the public
//! `*Math` traits instead. They are documented here for implementors adding a
//! new element type.

use core::marker::PhantomData;

use crate::{
    element::{FloatElement, FloatElementWithBits},
    mask::*,
    math::{
        CoreMathWithPolicy, FloatConsts, PrimalProjection, RealMathWithPolicy, TranscendentalMathWithPolicy,
        algorithms,
        policy::policies::{ExtraPrecision, LessPrecision},
    },
    register::NativeCapability,
    vector::{ops::BitAndNot, *},
};

// use super::MathWithPolicy;
use super::policy::{DenormalBehavior, Policy, PrecisionPolicy};

use generic_array::{ArrayLength, GenericArray, typenum::Unsigned};

mod generic;

/// Scalarized `libm` machinery for the [`Reference`](PrecisionPolicy::Reference) tier.
///
/// Public because the sibling crates wire their own reference arms with it
/// (`thermite-special` maps `erf`/`tgamma`/`lgamma` onto libm the same way), and
/// `#[doc(hidden)]` because that is the only audience. Depend on the tier's contract,
/// not on these helpers.
#[doc(hidden)]
pub mod reference;

use crate::math::policy::policies::MediumPrecision;

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
            // product cannot land subnormal until the FINAL multiply, so IEEE
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

        // the true (unclamped) new biased exponent, wrap-free thanks to the saturation above
        let new_exp = biased_exp + exp;

        if const { !P::POLICY.check_overflow } {
            // garbage in, garbage out per the policy contract: assemble and return
            let sign_mantissa = Self::SignedBits::from_bits(bits & sign_mantissa_mask);
            return Self::from_bits((new_exp << mantissa_bits) | sign_mantissa);
        }

        // Checked tail. Both forms clamp the biased exponent, which already
        // lands on the right FIELD value for the special cases (0 on underflow,
        // MAX_BIASED_EXP on overflow), and then fix up the specials:
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
        // chain and the blend form wins. Measured on znver3 with llvm-mca,
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
        // overflow select: `ldexp(0.0, 300)` is 0.0, not infinity.
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
        // means nothing until the value is renormalized. Skip this and
        // `frexp(1e-40f32)` answers `(1e-40, 0)`, silently breaking the
        // `0.5 <= |frac| < 1` postcondition while still satisfying
        // `x == frac * 2^exp`.
        //
        // Renormalizing costs a multiply, a second extraction and two selects.
        // Under a flushing policy subnormals are rare by assumption, so that
        // work sits behind a branch and the common path stays exactly as cheap
        // as it was (llvm-mca, znver3: 2.2 cyc/iter, against 4.0 if the fixup
        // runs unconditionally). Under `Preserve` they are expected instead, so
        // the branch would only mispredict, so run it straight-line there.
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
        // unpredictable (+67%). At 100% it is predictable again and the cost
        // falls back. Callers who genuinely expect dense subnormals under a
        // flushing policy should ask for `AvoidBranching<P, true>`, or, more
        // likely, they wanted `PreserveDenormals<P>` all along.
        // `Ignore` promises the hardware is running with denormals disabled
        // (DAZ/FTZ), so one can never arrive here. Skip the fixup and even its
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

/// Compensated Horner evaluation, Graillat-Langlois-Louvet 2005.
///
/// Runs an ordinary Horner recurrence and, alongside it, an exact accumulation of every
/// rounding error the recurrence commits. The result is about what a doubled-precision
/// Horner would give, which makes it **insensitive to the conditioning of the polynomial**,
/// the reason it is here.
///
/// `REV` picks the coefficient order: `false` is constant-term-first (`poly_n`), `true` is
/// leading-term-first (`poly_rev_n`). It is a const parameter so the index arithmetic folds.
///
/// # Why this is opt-in and not a precision tier
///
/// About **10 operations per term against 1 FMA**: `two_product` is 2, `two_sum` is 6, and
/// the error accumulator 2. No standard policy enables it. A call site that wants it asks
/// with `UseCompensation<P, true>`, which is how the Bessel rationals reach it.
///
/// # Two arms, chosen by `HAS_NATIVE_FMA`
///
/// The product half of the compensation needs a **correctly rounded** FMA for `pi` to be the
/// exact product error. Where the hardware fuses, that is one instruction and this is the full
/// Graillat-Langlois-Louvet scheme: both error sources captured, the condition number entering
/// **squared**, behavior equivalent to doubled precision.
///
/// Where it does not fuse (and on `Indeterminate`, i.e. wasm, which cannot promise it) the
/// product half is **dropped** and only the sums are compensated. Thermite's `mul_add` is
/// correctly rounded on every backend, so the full scheme would still be _exact_ there. It is
/// simply not worth it. The emulated correctly rounded FMA is a stronger guarantee than the
/// product error this needs, and measured **27x on the 1-lane f64 seed** and about **4x on
/// f64x4** for roughly 2x of accuracy.
///
/// The downgrade is real and is not hidden: with the product errors uncompensated they are
/// still amplified by the full condition number, so the bound improves only from
/// `gamma_{2n}` to `gamma_n` (about a factor of two, not an order of magnitude). The
/// compensation stops being conditioning-proof and becomes a constant-factor improvement. It
/// is kept because the same ~9 operations deliver that factor with no FMA anywhere, which is
/// strictly better than the alternative of not compensating at all.
#[inline(always)]
fn compensated_horner<V, E, const N: usize, const REV: bool>(x: V, coeffs: &[E; N]) -> V
where
    E: Copy,
    V: FloatVector<Element = E>,
{
    // Leading coefficient: last slot when constant-first, first slot when leading-first.
    let mut s = V::splat(coeffs[if REV { 0 } else { N - 1 }]);
    let mut e = V::ZERO;

    let mut i = 1usize;
    while i < N {
        let c = V::splat(coeffs[if REV { i } else { N - 1 - i }]);

        let p = s * x;

        // two_sum(p, c): t is the rounded sum, sigma the exact error (Knuth, 6 ops: the
        // operands are not ordered by magnitude, so the cheap fast_two_sum is not valid).
        // This half needs no FMA at all, only adds and subtracts.
        let t = p + c;
        let b = t - p;
        let sigma = (p - (t - b)) + (c - b);

        // two_product(s, x): `pi` is the EXACT error of the rounded product `p`, and is
        // available in one instruction only where the hardware fuses.
        //
        // Where it does not, `mul_add` lowers to the emulated correctly rounded FMA, which is
        // strictly more work than this needs: correct rounding of a SUM is a stronger
        // guarantee than the product error we are extracting, and we throw the rest away.
        // Measured, that path cost 27x on the 1-lane f64 seed and about 4x on f64x4, against
        // roughly 2x of accuracy. So off-FMA this degrades to compensating the SUMS only.
        //
        // That is a real downgrade, and an honest one: capturing both errors makes the
        // condition number enter SQUARED, which is what makes full compensation behave like
        // doubled precision. Sums alone leaves the surviving product errors amplified by the
        // full condition number, so the bound only improves from `gamma_{2n}` to `gamma_n`.
        // That is a factor of about two, not an order of magnitude. It is still worth
        // having: the same ~9 operations buy that factor without an FMA anywhere.
        //
        // `Indeterminate` (wasm, where the engine may or may not fuse a relaxed madd) takes the
        // cheap arm as well. It cannot PROMISE fusion, and `pi` is only the exact product error
        // under a genuine FMA. A `mul_add` that silently lowers to multiply-then-add makes the
        // compensation compensate for the wrong thing.
        let inc = if const { matches!(V::HAS_NATIVE_FMA, tribool::True) } {
            s.mul_add(x, -p) + sigma
        } else {
            sigma
        };

        // The error terms ride the same recurrence as the value. `mul_adde` and not `mul_add`:
        // `e` is already a correction of relative size ~eps, so its own rounding is second
        // order, and insisting on a correctly rounded FMA here would drag the emulated path
        // back in on exactly the backends the branch above just rescued.
        e = e.mul_adde(x, inc);
        s = t;

        i += 1;
    }

    s + e
}

pub trait SpecializedCoreMath<E>: FloatVector<Element = E> + PrimalProjection {
    /// Backing definition of [`CoreMath::poly_n_primal`](crate::math::CoreMath::poly_n_primal).
    ///
    /// Horner over primal coefficients. The multiply stays in `Self` (both operands
    /// genuinely vary), but the addend is a constant with no augmentation, so the
    /// step is `mul_add_primal` rather than a full `mul_adde` against a lifted zero.
    /// A type that is its own primal inherits `mul_add_primal = mul_adde`, so this
    /// compiles to exactly [`poly`](Self::poly) there.
    #[inline(always)]
    fn poly_n_primal<P: Policy, N: ArrayLength>(self, coeffs: &GenericArray<Self::Primal, N>) -> Self {
        let x = self;

        let n = const { N::USIZE };

        let mut res = Self::from_primal(coeffs[n - 1]);
        let mut i = n - 1;
        while i > 0 {
            i -= 1;
            unsafe { core::hint::assert_unchecked(i < n) };
            res = res.mul_add_primal::<P>(x, coeffs[i]);
        }
        res
    }

    /// Backing definition of [`CoreMath::poly_rev_n_primal`](crate::math::CoreMath::poly_rev_n_primal).
    ///
    /// [`poly_n_primal`](Self::poly_n_primal) with the coefficients in descending order.
    /// The same primal-Horner step, walked forwards.
    #[inline(always)]
    fn poly_rev_n_primal<P: Policy, N: ArrayLength>(self, coeffs: &GenericArray<Self::Primal, N>) -> Self {
        let x = self;

        let n = const { N::USIZE };

        let mut res = Self::from_primal(coeffs[0]);
        let mut i = 1usize;
        while i < n {
            unsafe { core::hint::assert_unchecked(i < n) };
            res = res.mul_add_primal::<P>(x, coeffs[i]);
            i += 1;
        }
        res
    }

    /// Backing definition of [`CoreMath::poly_primal`](crate::math::CoreMath::poly_primal).
    ///
    /// The same primal-Horner walk as [`poly_n_primal`](Self::poly_n_primal) over a
    /// runtime length. Unlike the slice reductions, a polynomial is not folded over
    /// chunks, since Horner carries `x^k` through every step, so this is its own loop.
    #[inline(always)]
    fn poly_primal<P: Policy>(self, coeffs: &[Self::Primal]) -> Self {
        let x = self;

        let Some((&last, rest)) = coeffs.split_last() else {
            return Self::ZERO;
        };

        let mut res = Self::from_primal(last);
        for &c in rest.iter().rev() {
            res = res.mul_add_primal::<P>(x, c);
        }

        res
    }

    /// Backing definition of [`CoreMath::poly_rev_primal`](crate::math::CoreMath::poly_rev_primal).
    ///
    /// [`poly_primal`](Self::poly_primal) with the coefficients descending.
    #[inline(always)]
    fn poly_rev_primal<P: Policy>(self, coeffs: &[Self::Primal]) -> Self {
        let x = self;

        let Some((&first, rest)) = coeffs.split_first() else {
            return Self::ZERO;
        };

        let mut res = Self::from_primal(first);
        for &c in rest {
            res = res.mul_add_primal::<P>(x, c);
        }

        res
    }

    /// One Horner step against a primal addend: `self * m + a`.
    ///
    /// The single point where a composite says how to add an unaugmented constant, so
    /// [`poly_n_primal`](Self::poly_n_primal) and anything else built on it inherit the
    /// saving from one override rather than reimplementing the evaluator. The default
    /// is correct for every type. It just lifts, which is free only when `Self` is its
    /// own primal.
    #[inline(always)]
    fn mul_add_primal<P: Policy>(self, m: Self, a: Self::Primal) -> Self {
        self.mul_adde(m, Self::from_primal(a))
    }

    /// `-(self * m) + a`, the negated twin of [`mul_add_primal`](Self::mul_add_primal).
    ///
    /// Exists because the fused complex Horner step spends one of its two FMAs negated
    /// (`re*zr - im*zi + c`), so a composite that overrides only the positive form still
    /// pays the augmented add on every other term.
    ///
    /// The default negates a multiplicand and defers, which is correct for every type
    /// and free wherever the negation folds into the multiply. Override it alongside
    /// `mul_add_primal` when the fused form is worth spelling out. There are no `_sub`
    /// twins: nothing needs them yet.
    #[inline(always)]
    fn nmul_add_primal<P: Policy>(self, m: Self, a: Self::Primal) -> Self {
        (-self).mul_add_primal::<P>(m, a)
    }

    #[inline(always)]
    fn difference_of_products<P: Policy>(self, b: Self, c: Self, d: Self) -> Self {
        let (a, cd) = (self, c * d);

        if const { !matches!(Self::HAS_NATIVE_FMA, tribool::True) } {
            a * b - cd
        } else if const { P::POLICY.precision.lt(PrecisionPolicy::Average) } {
            a.mul_sub(b, cd)
        } else {
            a.mul_sub(b, cd) + c.nmul_add(d, cd) // value + error
        }
    }

    #[inline(always)]
    fn sum_of_products<P: Policy>(self, b: Self, c: Self, d: Self) -> Self {
        let (a, cd) = (self, c * d);

        if const { !matches!(Self::HAS_NATIVE_FMA, tribool::True) } {
            a * b + cd
        } else if const { P::POLICY.precision.lt(PrecisionPolicy::Average) } {
            a.mul_add(b, cd)
        } else {
            a.mul_add(b, cd) - c.nmul_add(d, cd)
        }
    }

    #[inline(always)]
    fn poly<P: Policy>(self, coeffs: &[E]) -> Self {
        if const {
            !P::POLICY.unroll_loops
                || P::POLICY.precision.ge(PrecisionPolicy::Best)
                || !Self::ISA.has_instruction_level_parallelism()
        } {
            if crate::unlikely(coeffs.is_empty()) {
                return Self::ZERO;
            }

            let mut res = Self::splat(coeffs[coeffs.len() - 1]);
            for &c in coeffs.iter().rev().skip(1) {
                res = res.mul_adde(self, Self::splat(c));
            }
            return res;
        }

        // NumVector provides the num_traits::MulAdd implementation needed for fast_polynomial
        let res = fast_polynomial::poly_f::<_, _>(crate::vector::NumVector(self), coeffs.len(), |i| unsafe {
            crate::vector::NumVector(Self::splat(*coeffs.get_unchecked(i)))
        });

        res.0
    }

    #[inline(always)]
    fn poly_rev<P: Policy>(self, coeffs: &[E]) -> Self {
        if const {
            !P::POLICY.unroll_loops
                || P::POLICY.precision.ge(PrecisionPolicy::Best)
                || !Self::ISA.has_instruction_level_parallelism()
        } {
            if crate::unlikely(coeffs.is_empty()) {
                return Self::ZERO;
            }

            let mut res = Self::splat(coeffs[0]);
            for &c in coeffs.iter().skip(1) {
                res = res.mul_adde(self, Self::splat(c));
            }
            return res;
        }

        // NumVector provides the num_traits::MulAdd implementation needed for fast_polynomial
        let res = fast_polynomial::poly_f::<_, _>(crate::vector::NumVector(self), coeffs.len(), |i| unsafe {
            crate::vector::NumVector(Self::splat(*coeffs.get_unchecked(coeffs.len() - 1 - i)))
        });

        res.0
    }

    #[inline(always)]
    fn poly_n<P: Policy, const N: usize>(self, coeffs: &[E; N]) -> Self {
        let x = self;

        // Opt-in only: no standard policy sets `use_compensation`, so this arm exists for
        // call sites that ask with `UseCompensation<P, true>`.
        if const { P::POLICY.use_compensation } {
            return compensated_horner::<Self, E, N, false>(x, coeffs);
        }

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
    fn poly_rev_n<P: Policy, const N: usize>(self, coeffs: &[E; N]) -> Self {
        let x = self;

        // See `poly_n`. `poly_rational_n`'s reciprocal branch evaluates through here, so
        // leaving it out would silently keep the old accuracy for every `x > 1`.
        if const { P::POLICY.use_compensation } {
            return compensated_horner::<Self, E, N, true>(x, coeffs);
        }

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
    fn poly_rational_n<P: Policy, const N: usize, const D: usize>(
        self,
        numerator: &[E; N],
        denominator: &[E; D],
    ) -> Self {
        let x = self;

        if const { P::POLICY.precision.le(PrecisionPolicy::Average) } {
            let n = Self::poly_n::<P, N>(x, numerator);
            let d = Self::poly_n::<P, D>(x, denominator);

            return n.approx_div_p::<P>(d);
        }

        let invert = x.cmp_gt(Self::ONE);

        let mut n0 = Self::EMPTY;
        let mut n1 = Self::EMPTY;
        let mut d0 = Self::EMPTY;
        let mut d1 = Self::EMPTY;

        if const { P::POLICY.avoid_branching } || !invert.all() {
            n0 = Self::poly_n::<P, N>(x, numerator);
            d0 = Self::poly_n::<P, D>(x, denominator);
        }

        let mut z = Self::EMPTY;

        if const { P::POLICY.avoid_branching } || invert.any() {
            z = Self::approx_reciprocal::<P>(x);
            n1 = Self::poly_rev_n::<P, N>(z, numerator);
            d1 = Self::poly_rev_n::<P, D>(z, denominator);
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
    fn approx_reciprocal<P: Policy>(self) -> Self {
        // Two policy reasons to spend a real division: `Best` wants full precision, and
        // `Preserve` cannot use the estimate at all. `rcpps`/`rsqrtps` treat a denormal
        // OPERAND as zero in hardware regardless of MXCSR, so the estimate returns `inf`
        // for a subnormal input and the Newton step below turns that into `-inf`.
        //
        // No `HAS_APPROX_RCP` here: when the backend has no estimate, `rcp()` IS
        // `Self::ONE / self`, so the path below already lands on this answer.
        if const {
            P::POLICY.precision.ge(PrecisionPolicy::Best)
                || matches!(P::POLICY.denormal_behavior, DenormalBehavior::Preserve)
        } {
            return Self::ONE / self;
        }

        let mut y = self.rcp();

        // The capability DOES gate the refinement: with no estimate `y` is already exact,
        // and a Newton step on an exact value is pure cost.
        if const { Self::HAS_APPROX_RCP && P::POLICY.precision.gt(PrecisionPolicy::Worst) } {
            // One iteration of Newton's method. It is invalid at `self = 0` and
            // `self = inf`, where the estimate was already exactly right and the step
            // turns it into `inf * NaN`. That is DELIBERATELY left unguarded: this path
            // only exists on the fast tiers (the registers set `HAS_APPROX_RCP` false
            // under `strict_ieee754`, so the strict build takes the exact `rcp()` above),
            // and a fixup for edges that rare is cycles the tiers came here to save.
            y = y * self.nmul_adde(y, Self::TWO);
        }

        y
    }

    #[inline(always)]
    fn approx_div<P: Policy>(self, rhs: Self) -> Self {
        // Same shape as `reciprocal`: a real divide when the estimate is either not
        // precise enough or, under `Preserve`, not permitted. Only `Worst` without
        // `Preserve` reaches the multiply, so the estimate is taken exactly where it was
        // asked for.
        //
        // Unlike `approx_reciprocal` the capability is not a free pass here: with no
        // hardware estimate `rhs.rcp()` is a division, so the fallthrough would be a
        // divide AND a multiply for a doubly-rounded answer. The divide below is cheaper
        // and better.
        //
        // `strict_ieee754` joins the escape outright rather than patching the estimate
        // the way `approx_reciprocal` does, because a select cannot fix this path. The
        // estimate treats a DENORMAL `rhs` as zero (regardless of MXCSR), so `0 / 2e-39`
        // manufactures `0 * inf = NaN` and `x / 2e-39` a wrong infinity, and the second
        // one's correct answer is a huge *finite* quotient only a real divide can
        // produce. Unlike `approx_reciprocal`, `HAS_APPROX_RCP = false` does NOT make
        // this moot under strict: even the exact `1/rhs` OVERFLOWS to infinity for
        // `rhs` below ~2^-126-ish, so `0 * inf = NaN` survives an exact reciprocal and
        // only `self / rhs` avoids it. Compiled out, not policy-gated, so a hand-built
        // flush policy cannot reintroduce it under the strict build.
        if const {
            P::POLICY.precision.gt(PrecisionPolicy::Worst)
                || matches!(P::POLICY.denormal_behavior, DenormalBehavior::Preserve)
                || cfg!(feature = "strict_ieee754")
        } {
            return self / rhs;
        }

        self * rhs.rcp()
    }

    /// `$a/\sqrt{b}$`, spelled `a.approx_div_sqrt(b)`, as one kernel rather than a
    /// divide bolted onto a square root.
    ///
    /// The shape is [`approx_div`](Self::approx_div)'s, one level up: multiply by the
    /// reciprocal square root where the estimate is both permitted and profitable, and
    /// take the exact route otherwise. Same two escapes, for the same reasons: `Best`
    /// wants full precision, and `Preserve` cannot use the estimate at all, because
    /// `rsqrtps` treats a denormal operand as zero in hardware regardless of MXCSR (see
    /// `generic::sqrt::inverse_sqrt_internal`).
    ///
    /// Like `approx_div`, the capability is not a free pass: with no hardware estimate
    /// `rsqrt()` *is* `ONE / sqrt()`, so multiplying by it would be a square root, a
    /// divide and a multiply for a doubly-rounded answer. The exact form below is both
    /// cheaper and better there.
    ///
    /// # Why there is no Newton step with the numerator in it
    ///
    /// The obvious wish is a refinement that corrects the whole quotient rather than just
    /// the root. There is not one, and both halves of that are worth recording because
    /// both look like they should work.
    ///
    /// **Refining an estimate cannot see the numerator.** Newton for
    /// `$r \approx 1/\sqrt{b}$` is `$r' = r(3 - br^2)/2$`, whose correction factor is built
    /// from `b` and `r` alone. Carrying `a` through it as `$y = ar$`,
    /// `$y' = y(3 - br^2)/2$` leaves the rounding of `$ar$` uncorrected, because the
    /// residual has no term that knows about `a`. An iteration that does see it means solving
    /// `$f(y) = b - a^2/y^2$`, whose step is `$y(3a^2 - by^2)/(2a^2)$`: it reintroduces a
    /// division by `$a^2$`, and squaring the numerator overflows on inputs where the
    /// function itself is perfectly finite.
    ///
    /// **A Karp-Markstein correction on the exact path buys nothing here.** The tempting
    /// form is `$r = 1/s$`, `$y = ar$`, `$e = a - sy$` (exact under a true FMA),
    /// `$y' = y + re$`, the standard trick for recovering a correctly rounded quotient. It
    /// measured **bit-identical to `self / s` on 4096 random inputs**, because a hardware
    /// divide *already* returns the correctly rounded `$a/s$`. The trick exists for
    /// machines that synthesize division from a reciprocal, and on every backend here
    /// `divps`/`fdiv` is already the thing it reconstructs. A divide, three operations and
    /// an overflow guard for zero gain.
    ///
    /// What neither form can reach is the half ulp already inside `s` itself, which
    /// `$a/s$` inherits. Removing *that* means compensating the root, which is
    /// double-double work and belongs to `Reference` rather than to this function.
    ///
    /// What knowing the numerator *does* buy is scheduling: it is folded into the
    /// estimate before the correction multiply (`$y = (ar)(3 - br^2)/2$` rather than
    /// `$a \cdot r(3 - br^2)/2$`), so `$ar$` issues in parallel with `$r^2$` and the
    /// post-`rsqrt` dependency chain is one multiply shorter, at the same instruction
    /// count and the same one interior rounding.
    #[inline(always)]
    fn approx_div_sqrt<P: Policy>(self, denom: Self) -> Self {
        if const {
            P::POLICY.precision.ge(PrecisionPolicy::Best)
                || matches!(P::POLICY.denormal_behavior, DenormalBehavior::Preserve)
                || !Self::HAS_APPROX_RSQRT
        } {
            return self / denom.sqrt();
        }

        // `rsqrt` and the Newton step are written out here rather than delegated to
        // `inverse_sqrt`, deliberately. That method carries its own `Best`/`Preserve`
        // escapes and its own capability gate, so composing the two would make one answer
        // depend on two independent policy ladders, and a tier could silently take an
        // exact route inside an approximate one. The gate above is the only gate.
        //
        // `HAS_APPROX_RSQRT` is already known true here: the exact route above claims
        // every backend without an estimate, so this really is the hardware instruction.
        //
        // The numerator is folded in BEFORE the correction multiply, not after: `ar` and
        // `y0.square()` depend only on `y0`, so they issue in parallel and the post-rsqrt
        // chain is square -> fma -> mul rather than square -> fma -> mul -> mul. Same
        // instruction count, one multiply shorter in latency. Accuracy is unchanged, since
        // both spellings carry one interior rounding (`round(a*r)*u` vs `round(r*u)*a`),
        // and for a power-of-two numerator (the hypot kernels) `a * y0` is exact.
        let y0 = denom.rsqrt();
        let ar = self * y0;
        let mut y = ar;

        if const { P::POLICY.precision.gt(PrecisionPolicy::Worst) } {
            // One iteration of Newton's method, the same step `inverse_sqrt_internal`
            // takes, scaled through by the numerator: y' = (a y)(3 - b y^2)/2.
            let nx2 = denom.scale(const { <Self::Element as FloatElement>::ConstRatio::<{ -1 }, { 2 }>::VALUE });
            let threehalfs = Self::splat(const { <Self::Element as FloatElement>::ConstRatio::<{ 3 }, { 2 }>::VALUE });

            y = ar * y0.square().mul_adde(nx2, threehalfs);

            if const { P::POLICY.check_overflow && cfg!(not(target_arch = "aarch64")) } {
                // The step is only valid where the estimate is finite and nonzero, which
                // is exactly the interior of the domain. At both ends it manufactures a
                // NaN out of a correct answer:
                //
                //   b = 0    -> y0 = +inf, and the step is inf * (inf * -0.0 + 1.5)
                //   b = inf  -> y0 = 0,    and the step is 0   * (0   * -inf + 1.5)
                //
                // Both are `inf * NaN`. The raw `a * y0` is already exactly right in both
                // cases (`a/sqrt(0)` is a signed infinity, `a/sqrt(inf)` is a signed
                // zero), so keep it rather than patching the result afterwards. The
                // condition is still built from `y0`: `ar` mixes in the numerator's own
                // zeros and infinities, which are interior points, not step failures.
                //
                // This is the same failure mode `inverse_sqrt_internal` documents for a
                // denormal operand, one step further out: there the estimate itself is
                // wrong, here the estimate is right and the refinement breaks it.
                y = y0.is_finite().bitandnot(y0.is_zero()).select(y, ar);
            }
        }

        #[cfg(target_arch = "aarch64")]
        if const { P::POLICY.check_overflow } {
            // NEON's register-level `rsqrt` is NaN at both ends itself (see
            // `inverse_sqrt_internal`), so the raw estimate cannot be kept and even the raw
            // tier needs the patch. Keyed on the denominator, not the estimate: `ar` mixes
            // in the numerator's own zeros and infinities, which are interior points.
            // `a * (+inf | 0)` gives `a/sqrt(0)` (signed infinity, NaN for a = 0) and
            // `a/sqrt(inf)` (signed zero, NaN for an infinite `a`) for free.
            let zero = denom.is_zero();
            let fixed = self * zero.select(Self::INFINITY, Self::ZERO);
            y = (zero | denom.cmp_eq(Self::INFINITY)).select(fixed, y);
        }

        y
    }

    /// `numer / sum(1/x_i)` the direct way, backing both
    /// [`harmonic_mean`](Self::harmonic_mean) and [`inv_sum_inv`](Self::inv_sum_inv), which
    /// differ only in whether the numerator is `N` or `1`.
    ///
    /// This is the form every type can run, and is the default precisely because it needs
    /// no ordering. Real f32/f64 vectors override it with a version that scales by the
    /// smallest element to keep the sum from overflowing. That rewrite is meaningless on
    /// `Complex`, whose `min` is *lexicographic by (re, im)* and can therefore return an
    /// element of large magnitude, giving no protection at all and possibly making matters
    /// worse. See `generic::inv_sum_inv_internal`.
    ///
    /// On a plain float vector both limits fall out here without a guard, which the scaled
    /// form cannot claim: a zero input sends its reciprocal to infinity so the answer is `0`,
    /// and an all-infinite input sums to `0` so the answer is infinite.
    ///
    /// That is a property of **IEEE division**, not of this function or of being real-valued,
    /// and neither composite inherits it. `Complex` division forms `1/(c^2 + d^2)` first, so
    /// a zero gives `0 * inf =` NaN with no infinity to sum toward. `Compensated` is real and
    /// still loses it, because double-double division forms `two_prod(q1, rhs)`, which is
    /// `inf * 0` at a zero divisor and poisons the error term. Both therefore return NaN at a
    /// zero input, in each case exactly what that type's own `1/x` returns, so the behavior
    /// is inherited rather than invented here. Their test suites pin it.
    #[inline(always)]
    fn inv_sum_inv_direct<P: Policy, const N: usize>(mut values: [Self; N], numer: Self) -> Self {
        let mut i = 0;
        while i < N {
            values[i] = Self::approx_reciprocal::<P>(values[i]);
            i += 1;
        }

        // Log-depth rather than a running accumulator: the adds are otherwise a serial
        // dependency chain N deep, and it is the same reduction `hypot_n` uses.
        crate::math::algorithms::reduce_in_place(&mut values, |a, b| a + b);

        Self::approx_div::<P>(numer, values[0])
    }

    #[inline(always)]
    fn harmonic_mean_n<P: Policy, const N: usize>(values: [Self; N]) -> Self {
        let n = Self::splat(Self::Element::from_int(N as crate::LargeInt));
        Self::inv_sum_inv_direct::<P, N>(values, n)
    }

    #[inline(always)]
    fn inv_sum_inv_n<P: Policy, const N: usize>(values: [Self; N]) -> Self {
        Self::inv_sum_inv_direct::<P, N>(values, Self::ONE)
    }

    /// `$1/\sum_i 1/x_i$` over a runtime-length slice.
    ///
    /// [`inv_sum_inv_direct`](Self::inv_sum_inv_direct) with the array pass rewritten as a
    /// slice pass: one reciprocal per element, one sum, one divide. Read that function for
    /// the zero-input behavior, which is inherited here unchanged.
    ///
    /// No composite overrides `inv_sum_inv_n`, so there is nothing for a runtime-length form
    /// to inherit by routing back through the const kernel, and routing through it would
    /// only add an inversion per batch. Real vectors, which *do* want something else, take
    /// `generic::inv_sum_inv_slice_internal` via the `ps`/`pd` override instead.
    ///
    /// The empty sum of reciprocals is `0`, so the answer is `numer / 0`, matching the const
    /// form at `N = 0` and the real-vector slice form.
    #[inline(always)]
    fn inv_sum_inv<P: Policy>(values: &[Self]) -> Self {
        let mut acc = Self::ZERO;
        for &v in values {
            acc += Self::approx_reciprocal::<P>(v);
        }

        Self::approx_div::<P>(Self::ONE, acc)
    }

    /// `$N/\sum_i 1/x_i$` over a runtime-length slice.
    ///
    /// `N` is the only thing the harmonic mean adds to
    /// [`inv_sum_inv`](Self::inv_sum_inv), and it cannot be folded in per chunk without
    /// counting the padding, so it is applied once at the end. The mean of no values is
    /// `$0 \cdot \infty =$` NaN, which is the empty-average convention rather than an
    /// invented value.
    #[inline(always)]
    fn harmonic_mean<P: Policy>(values: &[Self]) -> Self {
        let n = Self::splat(Self::Element::from_int(values.len() as crate::LargeInt));

        n * Self::inv_sum_inv::<P>(values)
    }

    #[inline(always)]
    fn reciprocal_adde<P: Policy>(self, a: Self) -> Self {
        // See `reciprocal`. Same two escapes, same reasons, same split over which
        // condition belongs to the policy and which to the capability.
        if const {
            P::POLICY.precision.ge(PrecisionPolicy::Best)
                || matches!(P::POLICY.denormal_behavior, DenormalBehavior::Preserve)
        } {
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
            x = Self::approx_reciprocal::<P>(x);

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

        x = e.is_negative().select(Self::approx_reciprocal::<P>(x), x);
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

    /// `x * ln_of_y`, with `x == 0` winning over an infinite log but a NaN `y` winning over
    /// both. Shared by [`xlogy`](Self::xlogy), [`xlog1py`](Self::xlog1py) and
    /// [`entr`](SpecializedRealMath::entr); the other two members of the family guard on the
    /// sign of both arguments instead and cannot use it.
    ///
    /// `y` is passed separately from its logarithm because the NaN test belongs to `y`: a
    /// negative `y` makes the log NaN without being NaN itself, and there the zero guard
    /// still applies.
    #[inline(always)]
    fn xlog_guarded(x: Self, y: Self, ln_y: Self) -> Self {
        (x.is_zero() & !y.is_nan()).select(Self::ZERO, x * ln_y)
    }

    #[inline(always)]
    fn xlogy<P: Policy>(self, y: Self) -> Self {
        Self::xlog_guarded(self, y, Self::ln::<P>(y))
    }

    #[inline(always)]
    fn xlog1py<P: Policy>(self, y: Self) -> Self {
        Self::xlog_guarded(self, y, Self::ln_1p::<P>(y))
    }

    fn sinhc<P: Policy>(self) -> Self;

    fn atanhc<P: Policy>(self) -> Self;

    fn sinh_cosh<P: Policy>(self) -> (Self, Self);

    #[inline(always)]
    fn sinh<P: Policy>(self) -> Self {
        Self::sinh_cosh::<P>(self).0
    }

    /// `cosh(x) - 1 = 2 sinh^2(x/2)`, an exact identity, so no type needs to override this:
    /// the composition inherits whatever accuracy that type's `sinh` has, and near zero
    /// `sinh(x/2)` is already `x/2` to full relative precision, giving `x^2/2` with no
    /// cancellation anywhere. Same treatment as `versin` above.
    #[inline(always)]
    fn cosh_m1<P: Policy>(self) -> Self {
        let s = Self::sinh::<P>(self * Self::HALF);
        let h = s * s;
        h + h
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
    fn sqrt1mexp<P: Policy>(self) -> Self {
        // sqrt(1 - e^-x) = sqrt(-expm1(-x)), since the direct form annihilates for small x.
        // No singularity to patch: x = 0 gives 0, x = +inf gives 1, and x < 0 is out of
        // domain and correctly yields NaN from the square root of a negative.
        Self::exp_m1::<P>(-self).neg().sqrt()
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
        if const { P::POLICY.precision.le(PrecisionPolicy::Average) || !matches!(Self::HAS_NATIVE_FMA, tribool::True) }
        {
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

        // The correction is only meaningful (and only safe) where both the residual and the
        // result are finite. `e = inf` is a legitimate overflow, where `p_lo * inf + inf` is
        // NaN for a negative residual; `p_lo` itself is NaN at the domain edge x = -1, where
        // `p = n * -inf` and the residual is `inf - inf`. In both cases the uncorrected value
        // is already the exact limit, so the answer is to skip the correction, not to patch
        // the result afterwards.
        (p_lo.is_finite() & e.is_finite()).select(p_lo.mul_adde(e, e), e)
    }

    #[inline(always)]
    fn powf_m1<P: Policy>(self, e: Self) -> Self {
        // x^e - 1 = expm1(e * ln(x)); avoids the outer cancellation of pow(x, e) - 1.
        //
        // At `Worst`, take the log as `ln_1p(x - 1)` instead. It is the same value, but
        // `ln_1p` has a small-argument shortcut this tier badly needs and `ln` does not:
        // the `Worst` log is a linear function of the bit pattern with ~0.04 ABSOLUTE
        // error, and near x = 1, which is where this whole function lives, that leaves
        // roughly a constant 0.0397 instead of the true small value. `ln(1.000189)` at
        // `UltraPerformance` gives **3.986e-02** against a true 1.889e-04, which made
        // `powf_m1(1.000189, 4.857)` return 0.214 where 9.18e-04 was wanted.
        //
        // One subtract, at one tier, and the hot `ln` path is untouched, since `ln` is
        // called from far more places than this and the low tiers cannot afford extra
        // cycles. `(x - 1) + 1` recovers `x` to one rounding, nothing against this tier's
        // own 0.04.
        //
        // The tier bump is not new policy, as `powf` already does exactly this ("the
        // 'Worst' log2 precision is _terrible_, so just use medium to give anything
        // reasonable back"), and it pays the same price for the same reason: any ABSOLUTE
        // error `d` in the log becomes a RELATIVE error `e*d` in the result, so the
        // `Worst` log's 0.04 leaves ~2 bits for an exponent of 5. Before the bump it was
        // 1.6 bits at its best magnitude, against a 5-bit floor.
        let l = if const { P::POLICY.precision.eq(PrecisionPolicy::Worst) } {
            Self::ln_1p::<MediumPrecision<P>>(self - Self::ONE)
        } else {
            Self::ln::<P>(self)
        };

        let p = e * l;

        // As in `compound`, the residual needs a real FMA. Non-FMA backends keep the
        // uncorrected form rather than pay emulation.
        if const { P::POLICY.precision.le(PrecisionPolicy::Average) || !matches!(Self::HAS_NATIVE_FMA, tribool::True) }
        {
            return Self::exp_m1::<P>(p);
        }

        // Same Dekker product-residual as `compound` (see there): expm1(p + lo) =
        // expm1(p) + e^p * lo = r + (r + 1) * lo to first order.
        let p_lo = e.mul_sube(l, p);
        let r = Self::exp_m1::<P>(p);

        // As in `compound`: no correction unless the residual and the result are both
        // finite. `p_lo` is NaN at x = 0, where `p = e * -inf` makes the residual
        // `inf - inf`, and `r = -1` there is already exact.
        (p_lo.is_finite() & r.is_finite()).select(p_lo.mul_adde(r + Self::ONE, r), r)
    }

    #[inline(always)]
    fn compound_m1<P: Policy>(self, n: Self) -> Self {
        // (1 + x)^n - 1 = expm1(n * ln(1 + x)). `ln_1p` keeps x's low bits, which
        // `powf_m1(1 + x, n)` would round away, and `expm1` keeps the outer subtraction
        // from cancelling, which `compound(x, n) - 1` would not.
        let l = Self::ln_1p::<P>(self);
        let p = n * l;

        // As in `compound`, the Dekker residual needs a real FMA; non-FMA backends keep the
        // uncorrected form rather than pay emulation.
        if const { P::POLICY.precision.le(PrecisionPolicy::Average) || !matches!(Self::HAS_NATIVE_FMA, tribool::True) }
        {
            return Self::exp_m1::<P>(p);
        }

        let p_lo = n.mul_sube(l, p);
        let r = Self::exp_m1::<P>(p);

        // Same guard as `powf_m1`: at the domain edge x = -1 the residual is `inf - inf`,
        // and the uncorrected -1 (or +inf for n < 0) is already the limit.
        (p_lo.is_finite() & r.is_finite()).select(p_lo.mul_adde(r + Self::ONE, r), r)
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
    fn versinc<P: Policy>(self) -> Self {
        // (1 - cos x)/x^2 = 2 sin^2(x/2) / x^2 = (1/2) * (sin(x/2)/(x/2))^2, an exact
        // identity rather than an approximation, so there is no series and no cutoff.
        // Every singularity is `sinc`'s: it already returns 1 at the origin (giving 1/2
        // here, the true limit) and 0 at infinity (giving 0, likewise correct).
        let s = Self::sinc::<P>(self * Self::HALF);
        (s * s) * Self::HALF
    }

    #[inline(always)]
    fn cos_m1<P: Policy>(self) -> Self {
        // cos(x) - 1 = -(1 - cos(x))
        -Self::versin::<P>(self)
    }

    #[inline(always)]
    fn nth_root_n<P: Policy, const N: usize>(self) -> Self {
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

    /// The runtime twin of [`nth_root_n`](Self::nth_root_n): the same arithmetic with the degree
    /// as a value, so the two agree to the bit at every `n`. The special cases are one uniform
    /// branch on `n` rather than a compile-time fold.
    #[inline(always)]
    fn nth_root<P: Policy>(self, n: u32) -> Self {
        let mut x = self;

        match n {
            0 => Self::NAN, // undefined
            1 => x,
            2 => x.sqrt(),
            3 => x.cbrt_p::<P>(),
            4 if const { P::POLICY.precision.le(PrecisionPolicy::Average) } => x.sqrt().sqrt(),

            _ => {
                let odd = n & 1 == 1;
                let mut is_neg = GenericMask::FALSY;

                // for odd powers, work with absolute value and restore sign later
                if odd {
                    is_neg = x.is_negative();
                    x = x.abs();
                }

                let y = x.powf_p::<LessPrecision<P>>(Self::splat(E::from_ratio(1, n as crate::LargeInt)));
                let y_n = y.powi_p::<P>(n as i32);

                let np1 = Self::splat(E::from_int((n + 1) as crate::LargeInt));
                let nm1 = Self::splat(E::from_int((n - 1) as crate::LargeInt));

                // The dimensionless Halley step. See the const form for why.
                let q = y_n / x;
                let t = (Self::ONE - q) / q.mul_adde(np1, nm1);
                let mut y2 = t.mul_adde(y + y, y);

                if const { P::POLICY.check_overflow } {
                    y2 = (x.cmp_eq(Self::ZERO) | x.is_infinite()).select(y, y2);
                }

                if odd {
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
    // cancellation-safe ln_1p keeps the near-zero accuracy, and scaling by a constant
    // preserves the relative error.
    #[inline(always)]
    fn log2_p1<P: Policy>(self) -> Self {
        Self::ln_1p::<P>(self) * Self::LOG2_E
    }

    #[inline(always)]
    fn log10_p1<P: Policy>(self) -> Self {
        Self::ln_1p::<P>(self) * Self::LOG10_E
    }

    /// The direct form, which cancels near zero (see the trait method's docs). Real f32/f64
    /// vectors override this with `generic::log1pmx_internal`. The default exists so that
    /// ordered-comparison-free types (`Complex` above all, which cannot select a window at
    /// all) still get a correct answer rather than blocking the whole method.
    #[inline(always)]
    fn log1pmx<P: Policy>(self) -> Self {
        Self::ln_1p::<P>(self) - self
    }

    fn log_n_n<P: Policy, const N: usize>(self) -> Self;

    /// The runtime twin of [`log_n_n`](Self::log_n_n). This default goes through
    /// [`log`](Self::log). The real f32/f64 vectors override it with the same table lookup the
    /// const form uses, so the two agree to the bit there.
    #[inline(always)]
    fn log_n<P: Policy>(self, n: u32) -> Self {
        match n {
            0 => Self::ZERO,
            1 => <Self as FloatVector>::INFINITY,
            _ => Self::log::<P>(self, Self::splat(E::from_int(n as crate::LargeInt))),
        }
    }

    #[inline(always)]
    fn log<P: Policy>(self, base: Self) -> Self {
        Self::log2::<P>(self) / Self::log2::<P>(base)
    }

    /// ln(1 - e^(-x))
    #[inline(always)]
    fn ln1m_expnx<P: Policy>(self) -> Self {
        Self::ln::<P>(Self::ONE - Self::exp::<P>(-self))
    }

    fn ln1m_expnx_ext<P: Policy>(self, lnx: Self) -> Self;
}

/// Reduce a slice with `f`, seeded from its first element. The slice must be non-empty.
///
/// A serial fold rather than the log-depth `algorithms::reduce_*`: those take an array, and
/// the whole point of the slice forms is that they do not have one. Dependency depth is the
/// price of a runtime length, and these are the convenient spelling rather than the fast
/// path. A caller who wants the tree writes `*_n`.
#[inline(always)]
fn fold_slice<V: Copy>(values: &[V], f: impl Fn(V, V) -> V) -> V {
    let (&first, rest) = values.split_first().expect("fold_slice on an empty slice");

    let mut acc = first;
    for &v in rest {
        acc = f(acc, v);
    }

    acc
}

pub trait SpecializedSpatialMath<E>: SpecializedCoreMath<E> {
    // type Scalar: SpecializedRealMath<E>;

    #[inline(always)]
    fn hypot_n<P: Policy, const N: usize>(values: [Self; N]) -> Self {
        generic::hypot_n_recip_scaled::<Self, E, P, N, false>(values)
    }

    /// The two-argument spelling, and nothing more than a spelling.
    ///
    /// Both lowerings of `hypot_n` write `N = 2` out as their own arm (the real-vector
    /// `generic::hypot_n_pow2_scaled` and the composite `generic::hypot_n_recip_scaled`),
    /// so this delegates rather than carrying a third kernel that would have to be kept in
    /// step with them. Overriding it in a backend is therefore almost always the wrong
    /// move: override `hypot_n`'s `N = 2` arm instead, where the N-ary callers benefit too.
    #[inline(always)]
    fn hypot<P: Policy>(self, y: Self) -> Self {
        Self::hypot_n::<P, 2>([self, y])
    }

    #[inline(always)]
    fn inv_hypot_n<P: Policy, const N: usize>(values: [Self; N]) -> Self {
        generic::hypot_n_recip_scaled::<Self, E, P, N, true>(values)
    }

    #[inline(always)]
    fn hypot_s<P: Policy>(values: &[Self]) -> Self {
        generic::hypot_slice_recip_scaled::<Self, E, P, false>(values)
    }

    #[inline(always)]
    fn inv_hypot<P: Policy>(values: &[Self]) -> Self {
        generic::hypot_slice_recip_scaled::<Self, E, P, true>(values)
    }

    fn l1_norm<P: Policy>(self) -> Self;

    #[inline(always)]
    fn l2_norm<P: Policy>(self) -> Self {
        Self::l2_norm_squared::<P>(self).sqrt()
    }

    fn l2_norm_squared<P: Policy>(self) -> Self;
}

/// Backing trait of [`PrimalMathWithPolicy`](crate::math::PrimalMathWithPolicy):
/// a marker for _single-value_ real numbers (plain float vectors, `Compensated`),
/// never derivative- or component-carrying composites like `Dual` or `Complex`.
///
/// Implementing this is also what _provides_ the [`PrimalProjection`] fixpoint:
/// the blanket impl in [`crate::math`] gives every `SpecializedPrimalMath` type
/// `Primal = Self` with identity conversions.
pub trait SpecializedPrimalMath<E>:
    SpecializedRealMath<E> + SpecializedCoreMath<E> + PrimalProjection<Primal = Self>
{
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

    /// `-x ln x`, `0` at zero, `-inf` below it. The zero case rides the shared guard, since
    /// `ln 0` is the same `0 * -inf` that `xlogy` exists to absorb. Only the negative branch
    /// is specific to this one.
    #[inline(always)]
    fn entr<P: Policy>(self) -> Self {
        let v = Self::xlog_guarded(self, self, Self::ln::<P>(self));
        self.cmp_lt(Self::ZERO).select(Self::NEG_INFINITY, -v)
    }

    /// `ln(x/y)` for positive `x` and `y`, accurate near `x = y` where the plain ratio is
    /// not. The shared core of [`rel_entr`](Self::rel_entr) and [`kl_div`](Self::kl_div).
    ///
    /// `x/y` rounds to a relative `eps`, so `ln(x/y)` carries an *absolute* error of `eps`
    /// while the answer itself is `O((x-y)/y)`, a relative error of `eps*y/(x-y)`, which is
    /// unbounded as the arguments approach each other. That is the regime a converging
    /// optimizer lives in, so it is the regime that matters. Near the diagonal this instead
    /// uses `ln1p((x-y)/y)`, where `x - y` is exact by Sterbenz and no cancellation occurs.
    ///
    /// Neither form is good everywhere, which is why this is a select and not a rewrite.
    /// `ln1p` degrades as `x/y -> 0`, its argument approaching `-1`, while the ratio degrades
    /// on the diagonal. Measured against a 50-digit oracle over 211 points, the split at
    /// `|x - y| < y/2` is never worse than the plain ratio and is up to ten orders better:
    /// worst case `2.0e-16` against `3.7e-6`.
    ///
    /// A third form, `ln x - ln y`, is the only one that survives a ratio past the exponent
    /// range (`x = 1, y = 1e-320` overflows both of the others). It costs a second `log`
    /// everywhere and cancels on the diagonal exactly as the ratio does, so it is not used
    /// here. SciPy has the same overflow behavior.
    #[inline(always)]
    fn ln_ratio<P: Policy>(x: Self, y: Self) -> Self {
        let d = x - y;
        let near = d.abs().cmp_lt(y * Self::HALF);

        if const { !P::POLICY.avoid_branching } {
            if near.all() {
                return Self::ln_1p::<P>(d / y);
            }
            if near.none() {
                return Self::ln::<P>(x / y);
            }
        }

        near.select(Self::ln_1p::<P>(d / y), Self::ln::<P>(x / y))
    }

    /// `x ln(x/y)`, the Kullback-Leibler summand, extended by `0` at `x = 0, y >= 0` and
    /// `+inf` everywhere else in the plane.
    #[inline(always)]
    fn rel_entr<P: Policy>(self, y: Self) -> Self {
        // Explicit comparisons rather than the sign-bit predicates: -0.0 has to count as
        // zero here, and `is_negative` would call it negative.
        let inside = self.cmp_gt(Self::ZERO) & y.cmp_gt(Self::ZERO);
        let at_zero = self.cmp_eq(Self::ZERO) & y.cmp_ge(Self::ZERO);

        let v = self * Self::ln_ratio::<P>(self, y);

        at_zero.select(Self::ZERO, inside.select(v, Self::INFINITY))
    }

    /// [`rel_entr`](Self::rel_entr) plus the Bregman tail `-x + y`, which is what makes this
    /// non-negative for unnormalized arguments. Note the `x = 0` case is `y`, not `0`: the
    /// tail survives when the log term vanishes.
    ///
    /// This one has a cancellation of its own, worse than `rel_entr`'s and in a different
    /// place. With `y = x(1 + u)`, the log term is `-xu + xu^2/2` and the tail is `+xu`, so
    /// two first-order quantities cancel to a **second**-order answer: near the diagonal
    /// the textbook spelling is not merely imprecise, it is 100% wrong (measured, against a
    /// 50-digit oracle). Written instead as
    ///
    /// ```text
    /// kl_div(x, y) = -x * log1pmx((y - x)/x)
    /// ```
    ///
    /// the cancellation moves inside [`log1pmx`](SpecializedTranscendentalMath::log1pmx),
    /// which exists to absorb exactly it. That is an identity, not an approximation, and the
    /// same one the Poisson deviance uses: `bd0(k, lambda)` in `thermite-special` is this
    /// function under another name.
    ///
    /// The parametrization degrades as `u -> -1`, i.e. `y << x`, so past `|u| >= 1/2` the
    /// direct form runs instead. There the two terms no longer cancel, the answer being
    /// `O(x)` rather than `O(x u^2)`. Worst case over 160 sampled points: `1.8e-15` against
    /// the direct form's `1.0e+00`.
    #[inline(always)]
    fn kl_div<P: Policy>(self, y: Self) -> Self {
        let inside = self.cmp_gt(Self::ZERO) & y.cmp_gt(Self::ZERO);
        let at_zero = self.cmp_eq(Self::ZERO) & y.cmp_ge(Self::ZERO);

        let u = (y - self) / self;
        let near = u.abs().cmp_lt(Self::HALF);

        let v = if const { !P::POLICY.avoid_branching } && near.all() {
            -(self * Self::log1pmx::<P>(u))
        } else if const { !P::POLICY.avoid_branching } && near.none() {
            self.mul_adde(Self::ln_ratio::<P>(self, y), y - self)
        } else {
            near.select(
                -(self * Self::log1pmx::<P>(u)),
                self.mul_adde(Self::ln_ratio::<P>(self, y), y - self),
            )
        };

        at_zero.select(y, inside.select(v, Self::INFINITY))
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
            // a == b == +-inf makes a - b NaN, but the answer is that infinity (= m).
            r = d.is_nan().select(m, r);
        }

        r
    }

    #[inline(always)]
    fn logmean<P: Policy>(self, other: Self) -> Self {
        // (x - y)/(ln x - ln y) = (x - y)/(2 atanh(f)) with f = (x - y)/(x + y), and since
        // 2 atanh(f) = 2 f atanhc(f) while (x - y)/(2 f) is exactly the arithmetic mean, that
        // whole expression collapses to
        //
        //     logmean(x, y) = (x + y) / (2 atanhc(f))
        //
        // Both halves of the defining form cancel as x approaches y. This one does not: for
        // nearby arguments the subtraction is exact by Sterbenz's lemma, and atanhc is at its
        // most accurate near zero, which is exactly where the ratio lands.
        //
        // Going through `atanhc` rather than `atanh` also moves the x == y limit into a
        // function that already fills it in (`atanhc(0) = 1`, giving the arithmetic mean,
        // which is the correct limit), so the guard below is only for exact equality, where
        // f is 0/0 rather than 0.
        let d = self - other;
        let s = self + other;
        let f = Self::approx_div::<P>(d, s);
        let mut r = Self::approx_div::<P>(s, Self::atanhc::<P>(f) * Self::TWO);

        if const { P::POLICY.check_overflow } {
            // x == y makes f itself 0/0, so the limit has to be supplied here.
            r = d.is_zero().select(self, r);
        }

        r
    }

    #[inline(always)]
    fn logsumexp<P: Policy>(values: &[Self]) -> Self {
        // [`logsumexp_n`](Self::logsumexp_n) with the array passes rewritten as slice
        // passes. Every step below is that function's (the shared max, the `ln_1p` split
        // that drops exactly one dominant term, the `used` tie-breaker, the non-finite max
        // select) and is documented there rather than repeated here. The tree
        // reductions become serial folds, which is the whole cost of a runtime length.
        let Some((&first, rest)) = values.split_first() else {
            // The empty sum is 0, and ln(0) = -inf.
            return Self::NEG_INFINITY;
        };

        if rest.is_empty() {
            return first;
        }

        if rest.len() == 1 {
            return Self::logaddexp::<P>(first, rest[0]);
        }

        let m = fold_slice(values, |a, b| a.max(b));

        let mut acc = Self::ZERO;
        let mut r;

        if const { P::POLICY.precision.le(PrecisionPolicy::Worst) } {
            for &v in values {
                acc += Self::exp::<P>(v - m);
            }

            r = m + Self::ln::<P>(acc);
        } else {
            let mut used = <Self::Mask as GenericMask>::FALSY;

            for &v in values {
                let d = v - m;
                let dominant = d.cmp_eq(Self::ZERO).bitandnot(used);

                used |= dominant;
                acc += Self::exp::<P>(d).nz(dominant);
            }

            r = m + Self::ln_1p::<P>(acc);
        }

        if const { P::POLICY.check_overflow } {
            r = m.is_finite().select(r, m);
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
                (ba.approx_reciprocal_p::<P>(), xa / ba)
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

        #[cfg(not(target_arch = "spirv"))]
        if let Some(new_y) = FlushDenormals::<P>::flush_denormals([y]) {
            y = new_y[0];
        }

        //                             // Initial guess: y - 2y * (1 - y) * (y - 0.5)
        // While we have a good initial guess for the inverse, S-curves are most stable at the
        // midpoint, so start there. Converges much faster this way.
        //
        // The search always runs in unit t-space and rescales once at the end: mapping the
        // bracket through an approximate reciprocal of (b - a) put the endpoints a rounding
        // error inside [0, 1], so S(t_max) < 1 and y = 1 had no bracket.
        let x0 = Self::HALF; //(y + y).nmul_adde((Self::ONE - y) * (y - Self::HALF), y);

        if let Some((a, b)) = edges {
            ba = b - a;

            match N {
                0 => return y.step_p::<P>(Self::HALF).mul_adde(ba, a),
                1 => return y.mul_adde(ba, a),
                _ => {}
            }
        }

        // The bracket needs f(0) = -y strictly negative and f(1) = 1 - y not, so the solve
        // runs on y clamped into (0, 1]; lanes at or past either end get the exact endpoint
        // afterwards (S(0) = 0, S(1) = 1), which also covers out-of-range y.
        let ys = y.clamp(Self::MIN_POSITIVE, Self::ONE);

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

        let bounds = Some((Self::ZERO, Self::ONE));

        #[rustfmt::skip]
        let (v, _converged) = algorithms::newtons_method::<Self, P, _>(x0, Self::tolerance::<P>(), GenericMask::TRUTHY, bounds, #[inline(always)] move |t: Self| {
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

            // NOTE: derivative does not need to be clamped here, since Newton is bracketed.
            (t.mul_sube(xn1 * fx, ys), fpx * xn1)
        });

        let t = y.cmp_le(Self::ZERO).select(Self::ZERO, v);
        let t = y.cmp_ge(Self::ONE).select(Self::ONE, t);

        match edges {
            Some((a, _)) => t.mul_adde(ba, a),
            None => t,
        }
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
        let mut res = d.approx_reciprocal_p::<ExtraPrecision<P>>();

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
        let l = k * (y.approx_reciprocal_p::<P>() - Self::ONE).ln_p::<P>();

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
        // `N as i32 - 1` (not `(N - 1) as i32`) so the N=0 case, an empty coeff
        // array whose loop never runs and leaves `n` unused, doesn't underflow `usize`
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
