use super::*;

use crate::element::FloatElementWithBits;
use generic_array::typenum::Unsigned;

#[inline(always)]
#[allow(unused)]
pub fn fix_min<R: FloatRegister>(a: Storage<R>, b: Storage<R>, mut min: Storage<R>) -> Storage<R> {
    #[cfg(not(feature = "strict_ieee754"))]
    return min;

    let is_nan = R::is_nan(b);

    // This will copy the negative sign if min(-0.0, +0.0),
    // since if they are non-zero but the equal, the sign is already identical.
    let same = R::eq(a, b);
    min = R::blendv(same, min, R::bitor(a, b));

    R::blendv(is_nan, min, a)
}

#[inline(always)]
#[allow(unused)]
pub fn fix_max<R: FloatRegister>(a: Storage<R>, b: Storage<R>, mut max: Storage<R>) -> Storage<R> {
    #[cfg(not(feature = "strict_ieee754"))]
    return max;

    let is_nan = R::is_nan(b);

    // This will remove the negative sign if max(+0.0, -0.0),
    // since if they are non-zero but the equal, the sign is already identical.
    let same = R::eq(a, b);
    max = R::blendv(same, max, R::bitand(a, b));

    R::blendv(is_nan, max, a)
}

// ---------------------------------------------------------------------------
// Correctly-rounded FMA emulation via rounding to odd
//
// Boldo & Melquiond, "Emulation of FMA and correctly-rounded sums: proved
// algorithms using rounding to odd", IEEE Trans. Comput. 57(4), 2008
// (doi:10.1109/TC.2007.70819, Coq-proved, Pff library). Their Theorem 4:
// with `(u_h, u_l) = ExactMult(a, b)`, `(t_h, t_l) = ExactAdd(c, u_h)`,
// `v = RO(t_l + u_l)` and `z = RN(t_h + v)`, then `z = RN(a*b + c)` exactly
// (bit-identical to a hardware FMA), provided only that `u_l` (the product
// error term) is representable and p >= 5. There is NO hypothesis on `c`
// and none about subnormal intermediates.
//
// Assumes round-to-nearest-even, the Rust/Thermite baseline rounding mode.
//
// Every 2Sum below is `R::two_sum`: Knuth's unconditional form for `FAST = false` (the
// form Theorem 4 is proved with), Dekker's for `FAST = true`.
//
// Going through the register method matters: spelled out of `R::add`/`R::sub` it is
// reassociable on the scalar backend under `algebraic-scalar`, and LLVM folds the error
// term to zero (measured, `tests/eft.rs`). Every guarantee in this file depends on that
// error term being real.
// ---------------------------------------------------------------------------

/// Magnitude-sorted Fast2Sum: same `(s, err)` as plain 2Sum (`s = RN(x + y)`
/// and `s + err == x + y` exactly), via Dekker's 3-op form, whose
/// `|big| >= |small|` precondition is established by an explicit sort.
///
/// The payoff is the dependency chain, not the op count: the sum is computed
/// from the original operands (same value in either order), so it issues
/// immediately while the compare and blends run in parallel beside it. The
/// error term then needs only two serial subtracts, versus 2Sum's four-deep
/// tail. Graillat-Muller suggest exactly this trade, which attacks the
/// measured dependency-bound bottleneck of the emulated FMA.
///
/// Equal magnitudes may pick either operand as `big` (both orderings satisfy
/// Dekker's condition), zeros and subnormals are exact as in 2Sum, and
/// non-finite inputs produce the same NaN/inf garbage classes 2Sum does, and
/// callers' guards and post-checks are indifferent to which.
/// The sort stays local - it is a per-call-site trade, not a primitive - but the three
/// arithmetic operations after it go through [`FloatRegister::two_sum`] with `FAST`, for
/// the same strictness reason given in the section note above.
///
/// One change from the hand-rolled version: the sum now comes from the sorted pair, so it
/// waits on the blends instead of issuing beside them. Same value, one blend deeper. The
/// -7..-9% latency figure above was measured on the older form.
#[inline(always)]
fn two_sum_sorted<R: FloatRegister>(x: Storage<R>, y: Storage<R>) -> (Storage<R>, Storage<R>) {
    let x_larger = R::ge(R::abs(x), R::abs(y));
    let big = R::blendv(x_larger, y, x);
    let small = R::blendv(x_larger, x, y);

    R::two_sum::<true>(big, small)
}

/// Round-to-odd addition: `RO(x + y)`, branch-free.
///
/// 2Sum recovers the rounding error of `RN(x + y)`. When it is nonzero and the
/// rounded sum's last significand bit is even, the raw bits are stepped one ulp
/// toward the error's sign (BM 2008 Listing 1, CRlibm's `CorrectRoundedSum3`
/// macro). The bit step is `nextafter`'s mechanism: exact everywhere, including
/// across the subnormal boundary and exponent changes. A sum that lands
/// subnormal was exact (`err == 0`), so no adjustment fires there.
///
/// The `is_finite` guard keeps an infinite `s` untouched: 2Sum around infinity
/// produces a NaN error term, and stepping infinity's bit pattern would turn it
/// into a NaN.
#[inline(always)]
pub fn odd_round_add<R: FloatRegister>(x: Storage<R>, y: Storage<R>) -> Storage<R> {
    odd_round_add_full::<R, false>(x, y).0
}

/// As [`odd_round_add`], but also returns the exact pre-adjustment error term
/// and the all-ones "adjustment fired" lane mask (in the bits domain). The
/// rescue path reconstructs the exact sign of `(x + y) - RO(x + y)` from
/// these: it is `sign(err)` when no adjustment fired (the odd value is the
/// nearest sum) and `-sign(err)` when one did (the odd step overshoots the
/// true value by construction: |err| < |step|), and is zero iff `err` is.
#[inline(always)]
fn odd_round_add_full<R: FloatRegister, const SORTED: bool>(
    x: Storage<R>,
    y: Storage<R>,
) -> (Storage<R>, Storage<R>, Storage<R::Bits>) {
    // SORTED trades ports for a shorter chain. Measured worthwhile for the
    // f64 path (this EFT sits on `fmadd_ro`'s critical path: latency -7..-9%
    // for +2..+8% rthr, llvm-mca znver3) but not for the f32 widen path,
    // whose doubled ArrayRegister halves pay the sort twice for the same
    // chain win (+21% rthr), so it passes `false`.
    let (s, err) = if SORTED {
        two_sum_sorted::<R>(x, y)
    } else {
        R::two_sum::<false>(x, y)
    };

    let s_bits = <R::Bits as BitCastRegister<R>>::from_bits(s);
    let err_bits = <R::Bits as BitCastRegister<R>>::from_bits(err);

    // NOTE: `finite` also guards a real failure in `fmadd_ro`, beyond keeping
    // specials sane. A split carry on a near-MAX operand (e.g. `fma(MAX, 1.0,
    // c)`) produces an infinite `u_l` while `u_h`/`t_h` stay finite. Without
    // this guard the adjustment can decrement infinity's bit pattern into MAX
    // and return a finite but wrong `z` that the `is_finite(z)` post-check
    // cannot see.
    let inexact = R::ne(err, R::ZERO); // float compare: -0.0 counts as exact
    let finite = R::is_finite(s);

    let inexact = <<R::Bits as CoreRegister>::Mask as CastMaskRegister<R::Mask>>::mask_from(inexact);
    let finite = <<R::Bits as CoreRegister>::Mask as CastMaskRegister<R::Mask>>::mask_from(finite);

    // All-ones where the last significand bit is even: `(bits & 1) - 1` is 0
    // for odd, wraps to all-ones for even. Pure integer ops, cheaper than an
    // `eq`-vs-zero, which is a multi-op polyfill for 64-bit lanes below SSE4.1.
    let even = R::Bits::sub(R::Bits::bitand(s_bits, R::Bits::ONE), R::Bits::ONE);

    let adjust = R::Bits::bitand(
        <R::Bits as CoreRegister>::from_mask(<<R::Bits as CoreRegister>::Mask as BitwiseRegister>::bitand(
            inexact, finite,
        )),
        even,
    );

    // Direction: increment the raw bits (magnitude up) when `s` and `err` share a
    // sign, decrement when they differ, equivalent to Listing 1's
    // `(err > 0) ^ (s < 0)` given `err != 0`, and `s != 0` whenever `err != 0`.
    let sign_shift = (size_of::<<R::Element as FloatElementWithBits>::Bits>() * 8 - 1) as u32;
    let differ = R::Bits::shr(R::Bits::bitxor(s_bits, err_bits), sign_shift); // 0 or 1
    let step = R::Bits::sub(R::Bits::ONE, R::Bits::shl(differ, 1)); // +1, or -1 by wraparound

    // Masked add without a blend: `step & adjust` is the step where the
    // adjustment fires and an additive zero elsewhere.
    let odd = <R as BitCastRegister<R::Bits>>::from_bits(R::Bits::add(s_bits, R::Bits::bitand(step, adjust)));
    (odd, err, adjust)
}

/// Correctly-rounded fused multiply-add at `R`'s own precision: `RN(a*b + c)`
/// with a single rounding, bit-identical to a hardware FMA for every input.
///
/// The intended `mul_add` lowering for f64 registers on backends without
/// hardware FMA. (f32 registers should use [`fmadd_widen_ro`] instead, where
/// one widened multiply and an odd-rounded add replace the whole Dekker product.)
///
/// The packed path is BM 2008 Algorithm 1 with an integer-add Veltkamp split.
/// Theorem 4's one numeric hypothesis (that the product error term `u_l` is
/// representable) is guaranteed by a pre-gate on the sum of the biased
/// exponents of `a` and `b`: their product's error is a multiple of
/// `2^(ea + eb - 2(p-1))`, which must not fall below the smallest subnormal,
/// or in biased terms `Ea + Eb >= BIAS + p`, i.e. 1076 for f64. Lanes with a
/// zero `a` or `b` pass regardless: a zero product is exact, so the packed
/// path returns `RN(c)` for them. `c` itself needs no gate at all.
///
/// There is no overflow pre-gate. Every overflow surfaces as a non-finite `z`,
/// whether genuine, spurious (a finite true result whose intermediate `t_h`
/// overflows), a split carry into the exponent top, or an inf/NaN input,
/// because the algorithm cannot create an inexact finite result (BM 2008
/// Sec. VI). Both that post-check and the pre-gate route the whole packet
/// through `fmadd_ro_rescue`, a fully vectorized (no scalar loop, no libm)
/// all-range path, so the emulation is total and every lane of every input
/// is correctly rounded.
#[inline(always)]
pub fn fmadd_ro<R: FloatRegister>(a: Storage<R>, b: Storage<R>, c: Storage<R>) -> Storage<R> {
    // Under `ignore_denormals` the entire underflow side is compiled out: the
    // gate exists only to keep subnormal-scale product error terms exact, which
    // that feature declares out of scope (and hardware running FTZ/DAZ, the
    // usual reason to enable it, flushes those terms regardless of what we
    // do). Sub-gate products then round faithfully instead of exactly, and
    // subnormal-scale results lose the exactness guarantee. Every normal-range
    // result from normal-range products stays bit-identical to hardware FMA
    // except a rounding tie whose breaker sits below 2^-(BIAS - p). The split
    // and EFTs are scale-free (zeros and subnormals split exactly), so nothing
    // below the removed gate misbehaves structurally, and the `is_finite`
    // post-check (overflow/specials, unrelated to denormals) still stands.
    if const { !crate::features::IGNORE_DENORMALS } {
        let mantissa_bits = <R::Element as FloatElementWithBits>::MANTISSA_BITS;
        let exp_bits = <R::Element as FloatElementWithBits>::EXP_BITS;
        let p = mantissa_bits + 1;

        let a_bits = <R::Bits as BitCastRegister<R>>::from_bits(a);
        let b_bits = <R::Bits as BitCastRegister<R>>::from_bits(b);

        // Biased exponent fields, sign discarded by the left shift.
        let ea = R::Bits::shr(R::Bits::shl(a_bits, 1), mantissa_bits + 1);
        let eb = R::Bits::shr(R::Bits::shl(b_bits, 1), mantissa_bits + 1);
        let esum = R::Bits::add(ea, eb);

        // threshold = BIAS + p, with BIAS = ((1 << EXP_BITS) - 1) >> 1. Built from
        // ONE so it folds to a constant. A subnormal operand reads field 0, which
        // undershoots its true scale, so it conservatively fails the gate.
        let exp_field_max = R::Bits::sub(R::Bits::shl(R::Bits::ONE, exp_bits), R::Bits::ONE);
        let thresh = R::Bits::add(
            R::Bits::shri::<1>(exp_field_max),
            R::Bits::splat(Element::from_u16(p as u16)),
        );

        let ul_ok = R::Bits::ge(esum, thresh);

        // Zero rescue: a zero `a` or `b` makes every product term an exact zero, so
        // the packed path degenerates to `RN(c)`, which is correct. Zeroed lanes are
        // common (masked-off data), and without this they would scalarize.
        let zero_ab = R::Mask::bitor(R::eq(a, R::ZERO), R::eq(b, R::ZERO));
        let zero_ab = <<R::Bits as CoreRegister>::Mask as CastMaskRegister<R::Mask>>::mask_from(zero_ab);

        let safe = <<R::Bits as CoreRegister>::Mask as BitwiseRegister>::bitor(ul_ok, zero_ab);
        if !<<R::Bits as CoreRegister>::Mask as MaskRegister>::all(safe) {
            core::hint::cold_path();
            return fmadd_ro_rescue::<R>(a, b, c);
        }
    }

    let (t_h, t_l, u_l) = bm_core::<R>(a, b, c);
    let v = odd_round_add_full::<R, true>(t_l, u_l).0;
    let z = R::add(t_h, v);

    // Exact-zero sign: `v == 0` means `t_l + u_l == 0` exactly (a faithful
    // rounding is zero only for an exact zero), so the true result is `t_h`,
    // but `RN(t_h + 0.0)` turns a `-0.0` `t_h` into `+0.0` (e.g. a = -0.0,
    // c = -0.0 must produce -0.0, matching hardware FMA). Return `t_h` itself
    // on those lanes. When `v != 0` cancels `t_h` exactly instead, `+0.0` is
    // the correct IEEE sign for the freshly created zero, and `z` already has it.
    let z = R::blendv(R::eq(v, R::ZERO), z, t_h);

    if !<R::Mask as MaskRegister>::all(R::is_finite(z)) {
        core::hint::cold_path();
        return fmadd_ro_rescue::<R>(a, b, c);
    }

    z
}

/// The shared BM 2008 middle: Dekker product (integer-add Veltkamp split) and
/// ExactAdd with `c`. Returns `(t_h, t_l, u_l)` with
/// `t_h + t_l + u_l == a*b + c` exactly, under Theorem 4's hypothesis that
/// `u_l` (the product error term) is representable.
///
/// Split note: the integer-add split rounds the low `ceil(p/2)` significand
/// bits into the kept ones (add `2^(s-1)` to the sign-magnitude encoding,
/// which rounds the magnitude identically for either sign), then clears them.
/// The high part has at most `p - s` significant bits, the low part at most
/// `s`, and `hi + lo == x` exactly, so all four partial products are exact.
/// A carry running into the exponent top (near-MAX operands) yields an
/// infinite high part and poisons the result to non-finite, so callers must
/// route non-finite outcomes to the rescue/specials handling.
#[inline(always)]
fn bm_core<R: FloatRegister>(a: Storage<R>, b: Storage<R>, c: Storage<R>) -> (Storage<R>, Storage<R>, Storage<R>) {
    let mantissa_bits = <R::Element as FloatElementWithBits>::MANTISSA_BITS;

    let a_bits = <R::Bits as BitCastRegister<R>>::from_bits(a);
    let b_bits = <R::Bits as BitCastRegister<R>>::from_bits(b);

    let split = (mantissa_bits + 2) / 2;
    let round_bit = R::Bits::shl(R::Bits::ONE, split - 1);
    let keep_mask = R::Bits::not(R::Bits::sub(R::Bits::shl(R::Bits::ONE, split), R::Bits::ONE));

    let a_hi = <R as BitCastRegister<R::Bits>>::from_bits(R::Bits::bitand(R::Bits::add(a_bits, round_bit), keep_mask));
    let a_lo = R::sub(a, a_hi);
    let b_hi = <R as BitCastRegister<R::Bits>>::from_bits(R::Bits::bitand(R::Bits::add(b_bits, round_bit), keep_mask));
    let b_lo = R::sub(b, b_hi);

    // Dekker product: u_h + u_l == a * b exactly.
    let u_h = R::mul(a, b);
    let e1 = R::sub(R::mul(a_hi, b_hi), u_h);
    let e2 = R::add(e1, R::mul(a_hi, b_lo));
    let e3 = R::add(e2, R::mul(a_lo, b_hi));
    let u_l = R::add(e3, R::mul(a_lo, b_lo));

    // ExactAdd(c, u_h).
    let (t_h, t_l) = R::two_sum::<false>(u_h, c);

    (t_h, t_l, u_l)
}

/// Correctly-rounded fused multiply-add for a register whose
/// [`ExtendedPrecision`](FloatRegister::ExtendedPrecision) type is genuinely
/// wider: `RN(a*b + c)` bit-identical to a hardware FMA, branch-free, no gate.
///
/// The widened product is exact (2p <= P), so `RO_wide(a*b + c)` followed by
/// the narrowing round-to-nearest is correctly rounded by BM 2008 Theorem 3
/// (requires P >= p + 2 and the wide format's exponent range to extend at
/// least 2 below, and f32 in f64 has 29 extra digits and hundreds of exponent
/// headroom, while the theorem's Coq proof covers subnormal results). Infinities and
/// NaNs pass through: nothing overflows f64 from f32 operands, and
/// [`odd_round_add`]'s finite guard leaves special values untouched.
///
/// Caller contract: `R::ExtendedPrecision` must be a wider format, not the
/// `Self` fallback. At `Self` this computes a double rounding, not an FMA.
#[inline(always)]
pub fn fmadd_widen_ro<R: FloatRegister>(a: Storage<R>, b: Storage<R>, c: Storage<R>) -> Storage<R> {
    let wa = <R::ExtendedPrecision as CastRegister<R>>::cast_from(a);
    let wb = <R::ExtendedPrecision as CastRegister<R>>::cast_from(b);
    let wc = <R::ExtendedPrecision as CastRegister<R>>::cast_from(c);

    let prod = R::ExtendedPrecision::mul(wa, wb); // exact: 2p <= P
    let v = odd_round_add::<R::ExtendedPrecision>(prod, wc);

    <R as CastRegister<R::ExtendedPrecision>>::cast_from(v)
}

/// Whole-packet vectorized rescue for [`fmadd_ro`]: lanes outside the
/// exponent-safe range (subnormal-scale products) or a non-finite packed
/// result (overflow, split carry, inf/NaN inputs). Still correctly rounded
/// for every lane, still branch-free inside, and entirely register-resident:
/// no scalar loop, no libm.
///
/// Strategy: normalize `a` and `b` to ~2^0 with exact power-of-two multiplies
/// built from their own exponent fields (this also normalizes subnormal
/// inputs), run the same BM core at centered exponents (where the product
/// error term is always representable, the split cannot carry off the top,
/// and `t_h` cannot overflow), then place the result back at scale
/// `K = Ea + Eb - 2*BIAS` with a single rounding:
///
///   - normal destination: exact power-of-two multiplies of `RN(t_h + v)`,
///     chained with clamped factors so every intermediate stays normal. The
///     overflow boundary maps exactly under scaling, so a destination that
///     rounds to infinity does so consistently with a direct RN.
///   - subnormal destination: explicit round-to-nearest-even of the odd sum
///     `RO(t_h + v)` via a variable shift with guard/sticky bits. The odd
///     last bit summarizes all lower-order inexactness (BM Theorem 3), and
///     doing the last rounding in integer arithmetic sidesteps the
///     double-rounding-at-the-subnormal-boundary trap entirely (the musl
///     fma bug shape).
///
/// `c` interacts through three regimes, selected per lane by the biased
/// exponent gap `g = Ec - BIAS - K` (field arithmetic, and subnormal fields
/// misread their true exponent by at most p - 1, which the margins absorb):
///
///   - |g| <= T (T = 4p + 8): `c * 2^-K` is scaled exactly into the centered
///     frame (clamped factor chain, and the exponent stays within ~T + p of
///     zero, so every intermediate is normal).
///   - g > T ("c dominates"): |a*b| < ulp(c)/4, and `c` is never a midpoint,
///     so RN(c + ab) = c. Answer is literally `c`.
///   - g < -T ("product dominates"): |c * 2^-K| < 2^(-T + p + 2), far below
///     the distance from the exact 2p-bit product-sum to the nearest rounding
///     boundary (>= 2^(-2p - 1) in the centered frame), so `c` only matters
///     as a same-signed sticky: replace it with sign(c) * 2^-(4p + 3).
///     A true zero `c` stays zero (a sticky would break RN ties wrongly).
///
/// Special inputs bypass all of it: non-finite `a`/`b` (and zero `a`/`b`,
/// whose lanes carry garbage `K`) get the IEEE-identical `RN(a*b) + c`, and
/// non-finite `c` with finite `a`, `b` gets `c` (an overflowing product
/// cannot beat an infinity, and NaN passes through).
///
/// Dispatched (registers are `HasIsa`), which is what makes outlining this
/// body sound: the per-backend trampoline carries the `#[target_feature]`
/// set, so the rescue stays out of the caller's hot block without degrading
/// its intrinsics to out-of-line calls (the rule-zero trap that previously
/// forced it `#[inline(always)]`). The callers in [`fmadd_ro`] keep their
/// `cold_path()` branch hints.
#[thermite_macros::dispatch(R, thermite = "crate")]
fn fmadd_ro_rescue<R: FloatRegister>(a: Storage<R>, b: Storage<R>, c: Storage<R>) -> Storage<R> {
    type SB<R> = <R as FloatRegister>::SignedBits;
    type B<R> = <R as FloatRegister>::Bits;

    let m = <R::Element as FloatElementWithBits>::MANTISSA_BITS;
    let exp_bits = <R::Element as FloatElementWithBits>::EXP_BITS;
    let p = m + 1;

    // Small signed-integer constants, built from ONE so they fold.
    let one = SB::<R>::ONE;
    let bias = SB::<R>::sub(SB::<R>::shl(one, exp_bits - 1), one);
    let t_gap = SB::<R>::splat(Element::from_u16((4 * p + 8) as u16));

    // Biased exponent fields as signed integers (sign bit discarded by shl).
    let field = |x: Storage<R>| -> Storage<SB<R>> {
        let bits = <B<R> as BitCastRegister<R>>::from_bits(x);
        <SB<R> as BitCastRegister<B<R>>>::from_bits(B::<R>::shr(B::<R>::shl(bits, 1), m + 1))
    };

    let ea = field(a);
    let eb = field(b);
    let ec = field(c);

    // 2^n as a float, for n + BIAS in [1, 2*BIAS] (callers clamp to +-(BIAS-1)).
    let pow2 = |n: Storage<SB<R>>| -> Storage<R> {
        <R as BitCastRegister<SB<R>>>::from_bits(SB::<R>::shl(SB::<R>::add(n, bias), m))
    };

    // x * 2^n via three clamped exact factors (|n| <= 3*(BIAS-1) covers every
    // K and destination this path sees). Exactness of the intermediates is a
    // per-call-site argument (see the block comments below).
    let clamp_hi = SB::<R>::sub(bias, one);
    let mul_pow2 = |x: Storage<R>, n: Storage<SB<R>>| -> Storage<R> {
        let k1 = SB::<R>::max(SB::<R>::min(n, clamp_hi), SB::<R>::neg(clamp_hi));
        let r = SB::<R>::sub(n, k1);
        let k2 = SB::<R>::max(SB::<R>::min(r, clamp_hi), SB::<R>::neg(clamp_hi));
        let k3 = SB::<R>::sub(r, k2);
        R::mul(R::mul(R::mul(x, pow2(k1)), pow2(k2)), pow2(k3))
    };

    // Normalize: a' = a * 2^(BIAS - Ea) lands every finite nonzero a near
    // [2^-(p-1), 2), subnormals included (a power-of-two multiply of a
    // subnormal that lands normal is exact). The scale exponent is clamped at
    // -(BIAS - 1): 2^-BIAS is not a normal float (pow2 would build a zero),
    // and a top-exponent operand then normalizes to ~[2, 4) instead, which the
    // centered-frame bounds all carry one binade of slack for. K is
    // computed from the applied scales so the bookkeeping stays exact.
    let n_a = SB::<R>::max(SB::<R>::sub(bias, ea), SB::<R>::neg(clamp_hi));
    let n_b = SB::<R>::max(SB::<R>::sub(bias, eb), SB::<R>::neg(clamp_hi));
    let a1 = R::mul(a, pow2(n_a));
    let b1 = R::mul(b, pow2(n_b));
    let k_scale = SB::<R>::neg(SB::<R>::add(n_a, n_b));

    let gap = SB::<R>::sub(SB::<R>::sub(ec, bias), k_scale);
    let case_c = SB::<R>::gt(gap, t_gap);
    let case_p = SB::<R>::lt(gap, SB::<R>::neg(t_gap));

    // S-regime scaled c (exact: target exponent within ~T + p of zero and the
    // clamped chain keeps intermediates normal, since |Ec - BIAS| <= BIAS and
    // each factor moves at most BIAS - 1 toward a near-zero target).
    let c_scaled = mul_pow2(c, SB::<R>::neg(k_scale));

    // P-regime sticky: sign(c) * 2^-(4p + 3), only for nonzero c.
    let sticky_mag = pow2(SB::<R>::neg(SB::<R>::splat(Element::from_u16((4 * p + 3) as u16))));
    let sticky = R::bitor(R::signed_zero(c), sticky_mag);
    let use_sticky = <R::Mask as CastMaskRegister<<SB<R> as CoreRegister>::Mask>>::mask_from(case_p);
    let use_sticky = R::Mask::bitandnot(
        use_sticky,
        <R::Mask as CastMaskRegister<R::Mask>>::mask_from(R::eq(c, R::ZERO)),
    );
    let c_in = R::blendv(use_sticky, c_scaled, sticky);

    // The BM core at centered exponents: no gate needed (u_l' always
    // representable), no split carry, no t_h overflow (|c_in| <= ~2^(T+1),
    // |u'| < 8).
    let (t_h, t_l, u_l) = bm_core::<R>(a1, b1, c_in);

    // Theorem 4's own final step gives the correctly rounded 53-bit result:
    // s_rn = RN(t_h + RO(t_l + u_l)). The subnormal-destination rounder below
    // additionally needs the exact sign of `total - s_rn` (an odd summary
    // alone is one spare bit short when the destination sits one bit below
    // normal, a real 1-ulp failure caught by the random sweep). It is
    // reconstructed exactly from the two error terms the construction already
    // yields:  total - s_rn = e_rn + dv,  where dv = (t_l + u_l) - v.
    // Case split (proof: notes FMA_PROOFS.md, L7): if err_v == 0 then dv == 0
    // and e_rn alone is the residual, which covers every massive-cancellation
    // shape, where Sterbenz makes t_l == 0 and v = u_l exact. If err_v != 0,
    // the inexact u_h + c forces |t_h| >= max(|u_h|, |c|)/2, hence
    // e(v) <= e(t_h) - (m-1), so a nonzero e_rn is a multiple of ulp(v)
    // while |dv| < ulp(v) strictly, and sign(e_rn) dominates. Either way:
    // sign(e_rn) decides unless e_rn == 0, then dv decides. dv's sign is
    // sign(err_v), flipped when the odd adjustment fired (the step
    // overshoots: |step| > |err_v| even across a downward binade crossing),
    // and dv == 0 iff err_v == 0.
    let (v, err_v, fired) = odd_round_add_full::<R, true>(t_l, u_l);
    let (s_rn, e_rn) = R::two_sum::<false>(t_h, v);

    // Exact-zero sign, as in the main path: v == 0 means the exact result is
    // t_h, whose zero keeps its sign.
    let v_zero = R::eq(v, R::ZERO);
    let s_rn = R::blendv(v_zero, s_rn, t_h);

    // Normal/overflow destinations: exact clamped factor chain. When the
    // destination is normal every factor step stays normal (dest >= 1 - BIAS
    // and e_res >= -(2T + p) bound the path). When it overflows, the final
    // rounding-to-infinity happens at a boundary that maps exactly under
    // power-of-two scaling, so it agrees with a direct RN.
    let z_norm = mul_pow2(s_rn, k_scale);

    // Under `ignore_denormals` the rescue is only reachable through the
    // overflow post-check and specials (the underflow gate in `fmadd_ro` is
    // compiled out), so every non-blended destination is normal or infinite:
    // the whole subnormal-destination rounder below is dead and folds away.
    // Subnormal destinations that arrive anyway (none today) would flush
    // through the multiply chain's final rounding instead of being exact.
    let z = if const { crate::features::IGNORE_DENORMALS } {
        z_norm
    } else {
        // Residual sign bit and zero-ness, in the bits domain.
        let sign_shift = (size_of::<<R::Element as FloatElementWithBits>::Bits>() * 8 - 1) as u32;
        let sign_mask = B::<R>::shl(B::<R>::ONE, sign_shift);
        let dv_sign = B::<R>::bitxor(
            B::<R>::bitand(<B<R> as BitCastRegister<R>>::from_bits(err_v), sign_mask),
            B::<R>::bitand(fired, sign_mask),
        );
        let ern_zero = R::eq(e_rn, R::ZERO);
        let res_sign = R::blendv(
            ern_zero,
            R::signed_zero(e_rn),
            <R as BitCastRegister<B<R>>>::from_bits(dv_sign),
        );
        let res_zero = R::Mask::bitand(ern_zero, R::eq(err_v, R::ZERO));

        // Destination exponent. `s_rn == 0` (exact zero result) rides the normal-
        // destination multiply chain, which preserves the sign of zero exactly.
        let e_res = SB::<R>::sub(field(s_rn), bias);
        let e_dest = SB::<R>::add(e_res, k_scale);

        // Subnormal destinations: explicit round-to-nearest-even by a variable
        // shift of s_rn's significand, with the exact residual deciding ties and
        // near-tie direction. shift in [1, p] does real rounding, and the clamp at 63
        // turns "far below the smallest subnormal" into a correct flush to zero
        // through the same code. |residual| < half of s_rn's lsb, so it can only
        // matter when rem sits exactly on a boundary (rem == half, or rem == 0
        // where it cannot change the rounding), never push rem across one.
        let min_norm_e = SB::<R>::sub(one, bias);
        let dest_sub = SB::<R>::lt(e_dest, min_norm_e);
        let s_zero = R::eq(s_rn, R::ZERO);

        let mant_mask = B::<R>::sub(B::<R>::shl(B::<R>::ONE, m), B::<R>::ONE);
        let implicit = B::<R>::shl(B::<R>::ONE, m);
        let rn_bits = <B<R> as BitCastRegister<R>>::from_bits(s_rn);
        let mant = B::<R>::bitor(B::<R>::bitand(rn_bits, mant_mask), implicit);

        let shift = SB::<R>::sub(min_norm_e, e_dest);
        let shift = SB::<R>::min(shift, SB::<R>::splat(Element::from_u16(63)));
        let shift = SB::<R>::max(shift, one); // garbage lanes clamp instead of UB-adjacent counts
        let shift = <B<R> as BitCastRegister<SB<R>>>::from_bits(shift);

        let kept = B::<R>::shrv(mant, shift);
        let rem = B::<R>::sub(mant, B::<R>::shlv(kept, shift));
        let half = B::<R>::shlv(B::<R>::ONE, B::<R>::sub(shift, B::<R>::ONE));

        // Tie handling in magnitude space: the residual moves the true value away
        // from zero when its sign matches s_rn's. all-ones masks, integer built.
        let res_bits = <B<R> as BitCastRegister<R>>::from_bits(res_sign);
        let toward = B::<R>::shr(B::<R>::bitxor(res_bits, rn_bits), sign_shift); // 1 = residual opposes s_rn
        let away = B::<R>::sub(toward, B::<R>::ONE); // all-ones when residual and s_rn agree
        let res_nonzero = <B<R> as CoreRegister>::from_mask(<<B<R> as CoreRegister>::Mask as CastMaskRegister<
            R::Mask,
        >>::mask_from(R::Mask::not(res_zero)));
        let kept_odd = B::<R>::sub(B::<R>::ZERO, B::<R>::bitand(kept, B::<R>::ONE)); // all-ones when odd

        // Round up on rem > half. On the exact boundary rem == half, round up if
        // the true value lies above it (nonzero residual pointing away from
        // zero), or on a true tie when the kept part is odd (ties-to-even).
        let tie_up = B::<R>::bitor(
            B::<R>::bitand(res_nonzero, away),
            B::<R>::bitandnot(kept_odd, res_nonzero), // odd & !nonzero
        );
        let round_up = B::<R>::bitor(
            B::<R>::bitand(B::<R>::from_mask(B::<R>::gt(rem, half)), B::<R>::ONE),
            B::<R>::bitand(
                B::<R>::from_mask(B::<R>::eq(rem, half)),
                B::<R>::bitand(tie_up, B::<R>::ONE),
            ),
        );

        let sign_bit = B::<R>::shl(B::<R>::ONE, m + exp_bits);
        let z_sub_bits = B::<R>::bitor(B::<R>::add(kept, round_up), B::<R>::bitand(rn_bits, sign_bit));
        let z_sub = <R as BitCastRegister<B<R>>>::from_bits(z_sub_bits);

        let dest_sub = <R::Mask as CastMaskRegister<<SB<R> as CoreRegister>::Mask>>::mask_from(dest_sub);
        let dest_sub = R::Mask::bitandnot(dest_sub, <R::Mask as CastMaskRegister<R::Mask>>::mask_from(s_zero));
        R::blendv(dest_sub, z_norm, z_sub)
    };

    // c-dominates lanes and non-finite c (with finite a, b) answer `c`.
    // A zero `c` is excluded from c-dominates: RN(-0.0 + tiny_positive) is
    // +0.0, not -0.0, so zero `c` lanes must ride the compute path (their
    // scaled c is an exact signed zero, and the subnormal-destination rounder
    // carries the product's sign).
    let a_fin = R::is_finite(a);
    let b_fin = R::is_finite(b);
    let ab_fin = R::Mask::bitand(a_fin, b_fin);
    let case_c = <R::Mask as CastMaskRegister<<SB<R> as CoreRegister>::Mask>>::mask_from(case_c);
    let case_c = R::Mask::bitandnot(case_c, R::eq(c, R::ZERO));
    let c_nonfin = R::Mask::bitandnot(ab_fin, R::is_finite(c));
    let z = R::blendv(R::Mask::bitor(case_c, c_nonfin), z, c);

    // Non-finite or zero a/b: IEEE-identical naive form (their `K` is garbage).
    let zero_ab = R::Mask::bitor(R::eq(a, R::ZERO), R::eq(b, R::ZERO));
    let naive_mask = R::Mask::bitor(R::Mask::not(ab_fin), zero_ab);
    R::blendv(naive_mask, z, R::add(R::mul(a, b), c))
}
