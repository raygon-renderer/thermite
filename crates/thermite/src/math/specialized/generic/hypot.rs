//! Norms: every `hypot` lowering, at a const count and at a runtime one.
//!
//! # There are two hypot implementation pairs, and both have to exist
//!
//! This file carries `hypot_n_pow2_scaled`/`hypot_slice_pow2_scaled` AND
//! `hypot_n_recip_scaled`/`hypot_slice_recip_scaled`, which look like duplicates and are
//! not. The split is a *capability* one, named for how each rescales and visible in the
//! bounds:
//!
//! | | bound | how it rescales |
//! |---|---|---|
//! | `*_pow2_scaled` | `FloatVectorWithBits` | exponent-field surgery: `s = 2^-k`, **exact** |
//! | `*_recip_scaled` | `SpecializedSpatialMath` only | `max_abs.approx_reciprocal_p()`, a rounded divide |
//!
//! A norm is a range problem, so the whole kernel is scaling. Real f32/f64 vectors can
//! take the exponent apart and scale by an exact power of two, which is both faster and
//! free of rounding. `Complex`, `Dual`, `Interval` and `Compensated` have no exponent
//! field to take apart (there is no `FloatVectorWithBits` for them), so they scale by a
//! reciprocal and eat the extra rounding. Neither body can serve the other's callers.
//!
//! `ps.rs`/`pd.rs` override the trait methods to reach the `*_pow2_scaled` pair.
//! Everything else falls through to the trait defaults in `specialized/mod.rs`, which
//! delegate to the `*_recip_scaled` pair. (That composite fallback is the one exception
//! to this directory's "real-vector lowerings only" rule. It lives here so every `hypot`
//! body is in one file, not because real vectors reach it.)
//!
//! All four use `approx_div_sqrt` on the inverse path. For the `*_recip_scaled` pair the
//! reason is plain, since its scale factor is a rounded reciprocal and folding the
//! multiply into the divide removes a rounding outright. For `*_pow2_scaled` it looks
//! unnecessary, because an exact power-of-two scale makes `s * (1/sqrt(acc))` and
//! `s / sqrt(acc)` bit-identical across the entire normal range (measured: 0 differences
//! in 11940 cases, and none at any approximate tier). It is not unnecessary, because the
//! inverse path's whole purpose is to stay representable where the norm overflows, and
//! those answers are **subnormal**, at which point the multiply rounds as well and the
//! separate form rounds twice.

use crate::math::specialized::*;

/// The scale pair every pow2 kernel shares: clamp the max magnitude `m` so both scales
/// stay normal, then take its exponent field apart into `s = 2^-k` and `is = 2^k`, both
/// exact. See [`hypot_n_pow2_scaled`]'s doc for why the clamp is in the float domain and
/// how it makes the edge cases disappear.
#[inline(always)]
fn pow2_scales<V, E>(m: V) -> (V, V)
where
    E: FloatElementWithBits,
    V: FloatVectorWithBits<Element = E>,
{
    // `EXP_MASK` is the FULL field. Getting it wrong is nearly silent, because dropping
    // the low exponent bit still yields a valid, merely coarser, power-of-two scale. It
    // only shows up on subnormals, and only with the float-domain clamp.
    //
    // Every constant here comes out of existing `FloatElementWithBits` members with only
    // `!`, `&`, `+` and `-`. No shifts, which the element's `Bits` type does not offer
    // generically, and no new trait members.
    //
    //   exp_mask  the full biased-exponent field   (f32 0x7F80_0000)
    //   top       2 * bias, pre-shifted, so `top - e` is the field of `2^-k`  (f32 254 << 23)
    //   lo        2^(1 - bias), the smallest normal
    //   hi        2^(bias - 1), the largest normal whose reciprocal is also normal
    //
    // `top` is the exponent field of the largest finite value, which is `2 * bias` by
    // construction: `MAX_FINITE_PATTERN` has biased exponent `2*bias`, one below the
    // all-ones field reserved for inf/NaN.
    let exp_mask = !E::SIGN_MANTISSA_MASK;
    let top = E::MAX_FINITE_PATTERN & exp_mask;
    let lo_bits = E::MAX_SUBNORMAL + <E::Bits as crate::element::Element>::ONE; // one past the largest subnormal
    let hi_bits = top - lo_bits;

    // Clamped in the FLOAT domain: the integer form needs a 64-bit integer min/max that
    // AVX2 does not have.
    let m = m
        .max(V::from_bits(V::Bits::splat(lo_bits)))
        .min(V::from_bits(V::Bits::splat(hi_bits)));

    let e = V::Bits::from_bits(m) & V::Bits::splat(exp_mask);

    let s = V::from_bits(V::Bits::splat(top) - e); // 2^-k, exact
    let is = V::from_bits(e); //                      2^k,  exact

    (s, is)
}

/// Runtime-length form of [`hypot_n_pow2_scaled`], behind the slice-taking
/// [`hypot_s`](SpecializedSpatialMath::hypot_s)/[`inv_hypot`](SpecializedSpatialMath::inv_hypot).
///
/// Identical scaling, identical constants, identical guards. Read that function for why
/// any of it is the way it is. The differences are only what a runtime length forces: the
/// `N = 0`/`N = 1` shortcuts become ordinary branches, and neither pass unrolls.
#[inline(always)]
pub fn hypot_slice_pow2_scaled<V, E, P, const INV: bool>(values: &[V]) -> V
where
    E: FloatElementWithBits,
    V: FloatVectorWithBits<Element = E> + SpecializedSpatialMath<E>,
    P: Policy,
{
    let Some((&first, rest)) = values.split_first() else {
        return if INV { V::INFINITY } else { V::ZERO };
    };

    if rest.is_empty() {
        // sqrt(x^2) == |x|, with none of the scaling needed.
        let res = first.abs();
        return if INV { res.approx_reciprocal_p::<P>() } else { res };
    }

    let mut m = first.abs();
    for &v in rest {
        m = m.max(v.abs());
    }

    let (s, is) = pow2_scales::<V, E>(m);

    let mut acc = V::ZERO;
    for &v in values {
        let u = v * s;
        acc = u.mul_adde(u, acc);
    }

    let mut res = if INV {
        // Fused rather than `s * acc.inverse_sqrt_p::<P>()`, for the reason recorded on
        // the same line in `hypot_n_pow2_scaled`.
        s.approx_div_sqrt_p::<P>(acc)
    } else {
        acc.sqrt() * is
    };

    if const { P::POLICY.check_overflow } {
        let mut any_inf = first.is_infinite();
        for &v in rest {
            any_inf |= v.is_infinite();
        }

        res = any_inf.select(if INV { V::ZERO } else { V::INFINITY }, res);
    }

    res
}

/// Power-of-two-rescaled `sqrt(sum x_i^2)`, the real-vector override behind
/// [`SpecializedSpatialMath::hypot_n`] and [`SpecializedSpatialMath::inv_hypot_n`].
///
/// **This kernel is policy-invariant on purpose.** `hypot` exists to answer "the naive
/// form overflows", so a tier that skips the range handling is not a faster `hypot`, it
/// is a `sqrt` of a sum of squares wearing `hypot`'s name. On the `precision <= Worst`
/// fast path this replaces, float32 `hypot(3e38, 4e37)`, `hypot(1e30, 1e30)` and
/// `hypot(9.5e-23, 3.2e20)` all returned `inf`, and `hypot(1e-40, 1e-40)` returned
/// **0.0**. The underflow half is the one that hands a renderer a zero-length direction
/// vector.
///
/// # Why a power of two rather than dividing by the largest element
///
/// The classical form is `m * sqrt(1 + (n/m)^2)` for `m = max`, `n = min`, which costs a
/// divide, and the N-ary generalization costs a reciprocal, approximate below `Best`, so
/// its scaling is not even exact. Scaling by `2^-k` taken straight out of the
/// exponent field is **exact** (the significand is untouched, so no rounding happens at
/// all), needs no division, and is branchless.
///
/// It measured *faster than the divide form it replaces* on znver3, AVX2 256-bit, via
/// llvm-mca against the shipped kernels:
///
/// | | uOps | RThroughput | latency |
/// |---|---|---|---|
/// | f32 old `min/max` | 17 | 8.0 | 49 |
/// | f32 this | 18 | **5.0** | 45 |
/// | f64 old `min/max` | 17 | 14.0 | 57 |
/// | f64 this | 18 | **9.0** | 51 |
///
/// The divide and the square root are both FP1 on Zen 3, so the old form pinned FP1 at
/// exactly 14.00, which *was* its throughput. Dropping the divide leaves the sqrt alone
/// at 9.00, and the extra instructions are free because they land on FP0/FP2/FP3, which
/// the old form left idle. It also ties the *naive* path on throughput, paying only
/// latency (33 -> 45 cycles at f32).
///
/// # How the clamp makes the edge cases disappear
///
/// The scale comes from `max |x_i|` clamped into `[2^(1-bias), 2^(bias-1)]` **in the
/// float domain**, which clamps its exponent field into the range where both `2^k` and
/// `2^-k` are normal. Clamping the integer field instead is equivalent but needs
/// `vpmaxq`/`vpminq`, which **AVX2 does not have**. LLVM then emits a
/// `vpcmpgtq` + `vpblendvb` pair per bound and float64 costs 21 uOps and 59 cycles
/// instead of 18 and 51.
///
/// Three edge cases then need no code at all:
///
/// - **Zero.** Everything clamps to the low bound, `0 * 2^(bias-1)` is `0`, and
///   `sqrt(0) * 2^(1-bias)` is `0`. The old form needed a `cmp_eq`/`select` to stop
///   `min/max` becoming `0/0`.
/// - **Subnormals.** The low clamp lifts them into the normal range before squaring.
/// - **NaN.** The sum below uses the ORIGINAL values, never `min`/`max`, so a NaN cannot
///   be voted away. The old form lost it: `maxps`/`minps` both return their *second*
///   operand when either is NaN, so `hypot(NaN, 1.0)` returned **1.4142135** and
///   `hypot(NaN, 0.0)` returned **0.0**, while `hypot(1.0, NaN)` was correctly NaN.
///
/// Whatever the clamp decides for a NaN or infinite input is irrelevant for the same
/// reason: the scale is always a finite power of two, and `inf * 2^-k` is still `inf`.
///
/// # What still needs a guard
///
/// `hypot(+-inf, y) == +inf` for **any** `y`, NaN included (C99 F.10.4.3, and IEEE 754-2019
/// clause 9.2 for the same function). That does not fall out of the arithmetic, because
/// `inf` and `NaN` in the same sum give `NaN`. It is gated on `check_overflow` like every
/// other edge patch-up in this tree, so the tiers that opt out get `NaN` there.
#[inline(always)]
pub fn hypot_n_pow2_scaled<V, E, P, const N: usize, const INV: bool>(values: [V; N]) -> V
where
    E: FloatElementWithBits,
    V: FloatVectorWithBits<Element = E> + SpecializedSpatialMath<E>,
    P: Policy,
{
    if const { N == 0 } {
        return if INV { V::INFINITY } else { V::ZERO };
    }

    if const { N == 1 } {
        // sqrt(x^2) == |x|, with none of the scaling needed.
        let res = values[0].abs();
        return if INV { res.approx_reciprocal_p::<P>() } else { res };
    }

    // The general body below is already correct at this length (one max and two FMAs), so
    // this arm only saves the max's second `abs`, since the scale only ever needed the
    // larger magnitude, and the two-operand infinity test, which needs no accumulator.
    //
    // Everything else is the general path verbatim, including the clamp. Do not "simplify"
    // it to `sqrt(x*x + y*y)`, for the range reason in this function's doc.
    if const { N == 2 } {
        let x = values[0].abs();
        let y = values[1].abs();

        let (s, is) = pow2_scales::<V, E>(x.max(y));

        let u = x * s;
        let v = y * s;
        let acc = u.mul_adde(u, v * v);

        let mut res = if INV {
            // Fused for the subnormal tail, as in the general path below.
            s.approx_div_sqrt_p::<P>(acc)
        } else {
            acc.sqrt() * is
        };

        if const { P::POLICY.check_overflow } {
            // Both operands, not `m`: `max` returns its second operand when either is
            // NaN, so `m` is not reliably the infinity and `hypot(inf, NaN)` would miss.
            let any_inf = x.is_infinite() | y.is_infinite();
            res = any_inf.select(if INV { V::ZERO } else { V::INFINITY }, res);
        }

        return res;
    }

    // Exponent field of the largest magnitude, clamped so both scales stay normal. The
    // constant surgery is `pow2_scales`, shared with the `N == 2` arm and the slice form.
    let mut m = values[0].abs();
    let mut i = 1;
    while i < N {
        m = m.max(values[i].abs());
        i += 1;
    }

    let (s, is) = pow2_scales::<V, E>(m);

    // Sum of squares of the SCALED values. The largest is in [1, 2), so its square is in
    // [1, 4) and the sum cannot overflow for any N this is stamped at. A term small enough
    // to flush to zero is one whose contribution is below the result's last bit anyway.
    let mut acc = V::ZERO;
    let mut i = 0;
    while i < N {
        let u = values[i] * s;
        acc = u.mul_adde(u, acc);
        i += 1;
    }

    let mut res = if INV {
        // `s * acc.inverse_sqrt_p::<P>()` is bit-identical to this across the whole normal
        // range, since `s` is an exact power of two and scaling by it adds no rounding. It
        // is NOT equivalent where the result underflows, which is the regime this path
        // exists for: `inv_hypot` stays representable at magnitudes where the norm itself
        // overflows, and that answer is subnormal. There the multiply rounds too, so the
        // separate form rounds twice, measured worse on 48 of the 58 subnormal cases where
        // they differ and better on 10. One divide instead of a divide and a multiply, as
        // well.
        s.approx_div_sqrt_p::<P>(acc)
    } else {
        acc.sqrt() * is
    };

    if const { P::POLICY.check_overflow } {
        let mut any_inf = values[0].is_infinite();
        let mut i = 1;
        while i < N {
            any_inf |= values[i].is_infinite();
            i += 1;
        }

        // `hypot(+-inf, y) = +inf` for any y, NaN included, so the inverse is +0.
        res = any_inf.select(if INV { V::ZERO } else { V::INFINITY }, res);
    }

    res
}

/// Runtime-length form of [`hypot_n_recip_scaled`], the default behind the slice-taking
/// `hypot_s`/`inv_hypot`.
///
/// The same function as [`hypot_n_recip_scaled`]'s N-ary path with the array passes
/// rewritten as slice passes: identical scaling, identical zero and infinity guards,
/// identical fusing of `INV` into `inverse_sqrt_p` so the norm is never formed in the
/// inverse direction. Read that function for why any of it is the way it is. The
/// differences are only what a runtime length forces. The `N = 0` and `N = 1` shortcuts
/// become ordinary branches, the log-depth `reduce_*` become serial folds, and nothing can
/// be scaled in place because the slice is shared, so the scale and the square happen
/// inside the accumulation loop.
///
/// This mirrors [`hypot_slice_pow2_scaled`], which is the same rewrite of the
/// real-vector kernel. Real vectors never arrive here: `ps`/`pd` override `hypot_s` and
/// `inv_hypot` with that one, so this is the composite path exclusively.
///
/// # What a composite override owes
///
/// `Complex` needs none: the body opens with `abs()`, which over C is the modulus, so
/// everything after it is real and this computes the norm rather than the analytic
/// continuation, the same mechanism its `hypot_n` override uses. `Dual` and `Interval` do
/// override, for the same reasons their `hypot_n` overrides exist (a `0/0` gradient at the
/// origin, and a scaling built on certainly-compares that goes to `[0, inf]`), and both
/// express it with a fixed number of accumulators rather than a buffer over the slice.
#[inline(always)]
pub fn hypot_slice_recip_scaled<V, E, P, const INV: bool>(values: &[V]) -> V
where
    E: FloatElement,
    V: SpecializedSpatialMath<E>,
    P: Policy,
{
    // sqrt of the empty sum is 0, and 1/0 is infinity. Matches `hypot_n_recip_scaled` at `N = 0`.
    let Some((&first, rest)) = values.split_first() else {
        return if INV { V::INFINITY } else { V::ZERO };
    };

    #[inline(always)]
    fn prep<V: FloatVector, P: Policy>(v: V) -> V {
        let v = v.abs();

        #[cfg(not(target_arch = "spirv"))]
        if let Some([flushed]) = FlushDenormals::<P>::flush_denormals([v]) {
            return flushed;
        }

        v
    }

    let first = prep::<V, P>(first);

    if rest.is_empty() {
        // sqrt(x^2) == |x|, with none of the scaling needed. Matches `N = 1`.
        return if INV { first.approx_reciprocal_p::<P>() } else { first };
    }

    // Computed from the operands rather than from the running max: `max` returns its second
    // operand when either is NaN, so the max is not reliably the infinity and `hypot(inf,
    // NaN)` would be missed.
    let mut any_inf = V::Mask::FALSY;
    if const { P::POLICY.check_overflow } {
        any_inf = first.is_infinite();
        for &v in rest {
            any_inf |= v.is_infinite();
        }
    }

    let mut max_abs = first;
    for &v in rest {
        max_abs = max_abs.max(prep::<V, P>(v));
    }

    // A zero max would make every term 0/0. Scaling by 1 instead gives a zero sum, hence a
    // zero norm and an infinite inverse, the limit and what `hypot_n_recip_scaled` gives.
    let is_zero = max_abs.cmp_eq(V::ZERO);
    let scale = is_zero.select(V::ONE, max_abs.approx_reciprocal_p::<P>());

    let mut acc = V::ZERO;
    for &v in values {
        let u = prep::<V, P>(v) * scale;
        acc = u.mul_adde(u, acc);
    }

    let mut res = if INV {
        // Scaled, so the norm itself is never formed: this stays representable at
        // magnitudes where `max_abs * sqrt(acc)` would overflow.
        scale * acc.inverse_sqrt_p::<P>()
    } else {
        max_abs * acc.sqrt()
    };

    if const { P::POLICY.check_overflow } {
        // `hypot(+-inf, y) = +inf` for ANY y, NaN included (C99 F.10.4.3).
        res = any_inf.select(if INV { V::ZERO } else { V::INFINITY }, res);
    }

    res
}

/// Divide-rescaled `sqrt(sum x_i^2)`, the composite fallback behind the
/// [`SpecializedSpatialMath::hypot_n`]/[`inv_hypot_n`](SpecializedSpatialMath::inv_hypot_n)
/// trait defaults. Real f32/f64 vectors never reach it, since `ps`/`pd` override with
/// [`hypot_n_pow2_scaled`], so this serves `Complex`, `Compensated`, and anything else
/// without an exponent field to take apart. See the module doc for the split.
#[inline(always)]
pub fn hypot_n_recip_scaled<V, E, P, const N: usize, const INV: bool>(mut values: [V; N]) -> V
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
            res = res.approx_reciprocal_p::<P>();
        }

        return res;
    }

    // special case N=2 which saves a couple instructions
    if const { N == 2 } {
        let x = values[0];
        let y = values[1];

        // NO precision gate. `hypot` solves a RANGE problem, so a tier that skips the
        // scaling is not a cheaper `hypot`, it is `sqrt(x*x + y*y)` under another name.
        // On the float32 fast path this replaces, `hypot(3e38, 4e37)` and
        // `hypot(9.5e-23, 3.2e20)` both returned `inf`, and `hypot(1e-40, 1e-40)`
        // returned 0.0. Real float vectors take the power-of-two form in
        // `hypot_n_pow2_scaled`, which is both faster and range-safe. This is the
        // fallback for composites with no exponent field.
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
        } else {
            res = max * s.sqrt();
        }

        if const { P::POLICY.check_overflow } {
            // `hypot(+-inf, y) = +inf` for ANY y, NaN included (C99 F.10.4.3). Testing
            // BOTH operands rather than `max`: `max`/`min` return their second operand
            // when either is NaN, so `max` is not reliably the infinity.
            let any_inf = x.is_infinite() | y.is_infinite();
            res = any_inf.select(if INV { V::ZERO } else { V::INFINITY }, res);
        }

        return res;
    }

    // No precision gate here either. See the `N == 2` note above.

    // take absolute value of each element in place, zero dependencies,
    // since we're squaring anyway this doesn't lose any information
    for x in &mut values {
        *x = x.abs();
    }

    // Computed before `values` is consumed below, and from the operands rather than from
    // `max_abs`: `max` returns its second operand when either is NaN, so `max_abs` is not
    // reliably the infinity, and `hypot(inf, NaN)` would be missed.
    let mut any_inf = V::Mask::FALSY;
    if const { P::POLICY.check_overflow } {
        for x in &values {
            any_inf |= x.is_infinite();
        }
    }

    let max_abs = crate::math::algorithms::reduce_array(values, |a, b| a.max(b));
    let is_zero = max_abs.cmp_eq(V::ZERO);

    let scale = is_zero.select(V::ONE, max_abs.approx_reciprocal_p::<P>());

    for x in &mut values {
        *x *= scale; // scale to prevent overflow
        *x = x.square(); // square in place
    }

    // sum squares in place
    crate::math::algorithms::reduce_in_place(&mut values, |a, b| a + b);

    let mut res;

    if INV {
        // `scale` is a rounded reciprocal here, not the exact power of two the real-vector
        // kernel uses, so `scale * (1/sqrt(acc))` would round twice. One kernel rounds once.
        res = scale.approx_div_sqrt_p::<P>(values[0]);
    } else {
        res = max_abs * values[0].sqrt();
    }

    if const { P::POLICY.check_overflow } {
        // `hypot(+-inf, y) = +inf` for ANY y, NaN included (C99 F.10.4.3).
        res = any_inf.select(if INV { V::ZERO } else { V::INFINITY }, res);
    }

    res
}
