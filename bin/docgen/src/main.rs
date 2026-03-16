/// NOTE: Much of this file is AI-generated. I did't feel like writing out a bunch of SVG
/// generation code by hand, and the analysis logic is very repetitive and well-suited to generation as well.
use thermite::{
    backend::x86_v3::prelude::*,
    math::{
        TranscendentalMath,
        policy::{DefaultPolicy, DenormalBehavior, PolicyParameters, PrecisionPolicy},
    },
};

use thermite_special::SpecialMathWithPolicy;

use std::{collections::HashMap, fmt::Write as FmtWrite, fs::File, io::Write as _};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

trait AnalysisKernel<V: FloatVector> {
    fn actual(&self, x: V) -> V;
    fn expected(&self, x: V) -> V;

    #[inline(always)]
    fn both(&self, x: V) -> (V, V) {
        (self.actual(x), self.expected(x))
    }
}

/// Per-domain statistics for negative, positive denormal, and positive normal inputs.
#[derive(Debug, Clone, Copy)]
pub struct DomainStats {
    pub count: u64,
    pub abs_ulp_sum: u64,
    pub signed_ulp_sum: i128,
    pub sum_sq_ulp: u128,
    pub max_ulp: u64,
    pub worst_x: f32,
    pub worst_actual: f32,
    pub worst_expected: f32,
}

impl Default for DomainStats {
    fn default() -> Self {
        Self {
            count: 0,
            abs_ulp_sum: 0,
            signed_ulp_sum: 0,
            sum_sq_ulp: 0,
            max_ulp: 0,
            worst_x: 0.0,
            worst_actual: 0.0,
            worst_expected: 0.0,
        }
    }
}

impl DomainStats {
    pub fn mean_abs_ulp(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.abs_ulp_sum as f64 / self.count as f64
    }

    pub fn mean_signed_ulp(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.signed_ulp_sum as f64 / self.count as f64
    }

    pub fn rms_ulp(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        (self.sum_sq_ulp as f64 / self.count as f64).sqrt()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Domain {
    Negative = 0,
    PositiveDenormal = 1,
    PositiveNormal = 2,
}

impl Domain {
    #[inline(always)]
    pub fn classify(x_bits: u32) -> Self {
        if x_bits & 0x8000_0000 != 0 {
            Domain::Negative
        } else if x_bits & 0x7F80_0000 == 0 {
            Domain::PositiveDenormal
        } else {
            Domain::PositiveNormal
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Domain::Negative => "Negative",
            Domain::PositiveDenormal => "Positive Denormal",
            Domain::PositiveNormal => "Positive Normal",
        }
    }
}

// ── Important Range Stats ──

pub const NUM_IMPORTANT_BINS: usize = 512;

#[derive(Clone, Copy)]
pub struct ImportantRangeStats {
    pub lo: f32,
    pub hi: f32,
    /// Per-bin: (sum of abs ULP diffs, count of valid inputs)
    pub bins: [(u64, u64); NUM_IMPORTANT_BINS],
    /// Aggregate stats across the entire important range
    pub count: u64,
    pub ulp_sum: u64,
    pub max_ulp: u64,
    pub worst_x: f32,
    pub worst_actual: f32,
    pub worst_expected: f32,
    pub ulp_distribution: [u64; 16],
    pub ulp_greater_than_15: u64,
}

impl ImportantRangeStats {
    pub fn new(lo: f32, hi: f32) -> Self {
        Self {
            lo,
            hi,
            bins: [(0u64, 0u64); NUM_IMPORTANT_BINS],
            count: 0,
            ulp_sum: 0,
            max_ulp: 0,
            worst_x: 0.0,
            worst_actual: 0.0,
            worst_expected: 0.0,
            ulp_distribution: [0; 16],
            ulp_greater_than_15: 0,
        }
    }

    pub fn has_range(&self) -> bool {
        self.lo.is_finite() && self.hi.is_finite()
    }

    pub fn mean_abs_ulp(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.ulp_sum as f64 / self.count as f64
        }
    }

    pub fn bin_width(&self) -> f32 {
        (self.hi - self.lo) / NUM_IMPORTANT_BINS as f32
    }

    pub fn bin_mean_ulp(&self, bin: usize) -> f64 {
        let (sum, count) = self.bins[bin];
        if count == 0 { 0.0 } else { sum as f64 / count as f64 }
    }
}

// ── FnRecording ──

#[derive(Clone, Copy)]
pub struct FnRecording {
    /// Absolute ULP difference summed across all valid inputs.
    pub abs_ulp_diff: u64,

    /// Signed ULP difference summed (expected - actual). Positive = actual underestimates.
    pub signed_ulp_diff: i128,

    /// Sum of squared ULP differences for RMS calculation.
    pub sum_sq_ulp: u128,

    /// Number of valid, finite inputs evaluated (excluding NaN/Inf in/out).
    pub count: u64,

    /// Number of NaN outputs from the actual function.
    pub actual_nans: u64,
    /// Number of NaN outputs from the expected function.
    pub expected_nans: u64,

    /// Whether the actual function preserves monotonicity relative to expected.
    pub monotonic: bool,
    /// Whether the actual function is exactly correct at key points (0, 1, -1).
    pub exact_at_key_points: bool,

    /// Number of times expected output was NaN but actual was not, or vice versa.
    pub nan_mismatches: u64,
    /// Number of times expected output was Inf but actual was finite, or vice versa.
    pub inf_mismatches: u64,
    /// Number of times expected output was zero but actual was non-zero, or vice versa.
    pub zero_mismatches: u64,

    // ULP Distribution Buckets
    pub ulp_distribution: [u64; 16],
    pub ulp_greater_than_15: u64,

    /// The absolute worst finite ULP difference recorded.
    pub max_ulp_diff: u64,
    /// The specific data related to the max_ulp_diff.
    pub worst_x: f32,
    pub worst_actual: f32,
    pub worst_expected: f32,

    /// Histogram of the MAXIMUM ULP difference observed for each f32 exponent (0-255).
    pub exp_histogram: [u64; 256],
    /// The specific f32 input that caused the maximum ULP difference in each exponent bucket.
    pub worst_input_for_exp: [f32; 256],

    /// Sum of absolute ULP differences per f32 exponent bucket.
    /// Combined with the known population per bucket (2 * 2^23), yields mean ULP per exponent.
    pub exp_ulp_sum: [u64; 256],

    /// Per-domain breakdown: [Negative, PositiveDenormal, PositiveNormal]
    pub domain_stats: [DomainStats; 3],

    /// Detailed stats within a user-specified "important" input range.
    pub important_range: ImportantRangeStats,

    /// Detailed stats within the "reasonable" range [-1e7, 1e7].
    pub reasonable_range: ImportantRangeStats,
}

impl FnRecording {
    pub fn mean_abs_ulp(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.abs_ulp_diff as f64 / self.count as f64
    }

    pub fn mean_signed_ulp(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.signed_ulp_diff as f64 / self.count as f64
    }

    pub fn rms_ulp(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        (self.sum_sq_ulp as f64 / self.count as f64).sqrt()
    }
}

// Helper to check exactness for specific key points
#[inline(always)]
fn is_key_point<S: NativeSimdVectorsWithRegisters>(
    x: <S as NativeSimdVectors>::f32xN,
) -> <<S as NativeSimdVectors>::f32xN as GenericVector>::Mask {
    let zero = <S as NativeSimdVectors>::f32xN::splat(0.0);
    let one = <S as NativeSimdVectors>::f32xN::splat(1.0);
    let neg_one = <S as NativeSimdVectors>::f32xN::splat(-1.0);

    x.cmp_eq(zero) | x.cmp_eq(one) | x.cmp_eq(neg_one)
}

#[thermite::dispatch(S)]
fn analyze_f32<S: NativeSimdVectorsWithRegisters, K>(
    kernel: &K,
    important_lo: f32,
    important_hi: f32,
    pb: Option<&ProgressBar>,
) -> FnRecording
where
    K: AnalysisKernel<<S as NativeSimdVectors>::f32xN>,
{
    let mut recording = FnRecording {
        abs_ulp_diff: 0,
        signed_ulp_diff: 0,
        sum_sq_ulp: 0,
        count: 0,
        actual_nans: 0,
        expected_nans: 0,
        monotonic: true,
        exact_at_key_points: true,
        nan_mismatches: 0,
        inf_mismatches: 0,
        zero_mismatches: 0,
        ulp_distribution: [0; 16],
        ulp_greater_than_15: 0,
        max_ulp_diff: 0,
        worst_x: 0.0,
        worst_actual: 0.0,
        worst_expected: 0.0,
        exp_histogram: [0; 256],
        worst_input_for_exp: [0.0; 256],
        exp_ulp_sum: [0; 256],
        domain_stats: [DomainStats::default(); 3],
        important_range: ImportantRangeStats::new(important_lo, important_hi),
        reasonable_range: ImportantRangeStats::new(-1e7, 1e7),
    };

    // Iterate over the full u32 space to hit every possible f32 bit pattern.
    let mut start_bits: u64 = 0;
    let end_bits: u64 = u32::MAX as u64;

    let lanes = <S as NativeSimdVectors>::i32xN::LANES as u64;
    let mut bits = <S as NativeSimdVectors>::i32xN::indexed();

    // Monotonicity: track the last lane's linear-order values (scalar) for correct adjacency.
    let first_x = <S as NativeSimdVectors>::f32xN::from_bits(bits);
    let (first_actual, first_expected) = kernel.both(first_x);
    let last_lane = lanes as usize - 1;
    let mut prev_actual_linear_last: i32 = first_actual.linear_order().extractv(last_lane);
    let mut prev_expected_linear_last: i32 = first_expected.linear_order().extractv(last_lane);
    let mut prev_last_was_finite: bool = {
        let x_val = first_x.extractv(last_lane);
        let a_val = first_actual.extractv(last_lane);
        let e_val = first_expected.extractv(last_lane);
        x_val.is_finite() && a_val.is_finite() && e_val.is_finite()
    };

    // Important range: splat bounds for vectorized comparison.
    // When bounds are NaN, cmp_ge/cmp_le produce false for all lanes → zero overhead.
    let important_lo_v = <S as NativeSimdVectors>::f32xN::splat(important_lo);
    let important_hi_v = <S as NativeSimdVectors>::f32xN::splat(important_hi);
    let inv_bin_width = if important_lo.is_finite() && important_hi.is_finite() {
        NUM_IMPORTANT_BINS as f32 / (important_hi - important_lo)
    } else {
        0.0
    };

    // Reasonable range: fixed [-1e7, 1e7]
    let reasonable_lo: f32 = -1e7;
    let reasonable_hi: f32 = 1e7;
    let reasonable_lo_v = <S as NativeSimdVectors>::f32xN::splat(reasonable_lo);
    let reasonable_hi_v = <S as NativeSimdVectors>::f32xN::splat(reasonable_hi);
    let reasonable_inv_bin_width = NUM_IMPORTANT_BINS as f32 / (reasonable_hi - reasonable_lo);

    while start_bits <= end_bits {
        let x = <S as NativeSimdVectors>::f32xN::from_bits(bits);

        let (actual, expected) = kernel.both(x);

        let u32_sign_mask = <S as NativeSimdVectors>::u32xN::splat(0x7FFF_FFFF);
        let u32_exp_mask = <S as NativeSimdVectors>::u32xN::splat(0x7F80_0000);

        let x_bits_u32 = <S as NativeSimdVectors>::u32xN::from_bits(x);
        let x_abs = x_bits_u32 & u32_sign_mask;
        let actual_abs = <S as NativeSimdVectors>::u32xN::from_bits(actual) & u32_sign_mask;
        let expected_abs = <S as NativeSimdVectors>::u32xN::from_bits(expected) & u32_sign_mask;

        let x_nan = x_abs.cmp_gt(u32_exp_mask);
        let x_inf = x_abs.cmp_eq(u32_exp_mask);
        let actual_nan = actual_abs.cmp_gt(u32_exp_mask);
        let actual_inf = actual_abs.cmp_eq(u32_exp_mask);
        let expected_nan = expected_abs.cmp_gt(u32_exp_mask);
        let expected_inf = expected_abs.cmp_eq(u32_exp_mask);

        // ── 1. NaN counting & mismatches ──
        recording.actual_nans += actual_nan.native_bitmask().unwrap().count_ones() as u64;
        recording.expected_nans += expected_nan.native_bitmask().unwrap().count_ones() as u64;

        let nan_mismatch = expected_nan ^ actual_nan;
        recording.nan_mismatches += nan_mismatch.native_bitmask().unwrap().count_ones() as u64;

        // Inf and zero mismatch counting
        let inf_mismatch = expected_inf ^ actual_inf;
        recording.inf_mismatches += inf_mismatch.native_bitmask().unwrap().count_ones() as u64;

        let expected_zero = expected.cmp_eq(<S as NativeSimdVectors>::f32xN::splat(0.0));
        let actual_zero = actual.cmp_eq(<S as NativeSimdVectors>::f32xN::splat(0.0));
        recording.zero_mismatches += (expected_zero ^ actual_zero).native_bitmask().unwrap().count_ones() as u64;

        // ── 2. Exactness at key points ──
        let key_points_mask = is_key_point::<S>(x);
        if key_points_mask.any() {
            let exact_match = actual.cmp_eq(expected);
            let valid = !key_points_mask | exact_match;
            if !valid.all() {
                recording.exact_at_key_points = false;
            }
        }

        // ── 3. ULP difference calculation ──
        let expected_linear = expected.linear_order();
        let actual_linear = actual.linear_order();

        let ulp_diff = expected_linear - actual_linear; // signed: positive = actual underestimates
        let zero_int = <S as NativeSimdVectors>::i32xN::splat(0);
        let mut abs_ulp_diff = ulp_diff.abs();

        // FIX: Exclude lanes where both are NaN (different NaN payloads are "equal").
        let both_nan = expected_nan & actual_nan;
        let both_nan_i32 = both_nan.cast::<<<S as NativeSimdVectors>::i32xN as GenericVector>::Mask>();
        abs_ulp_diff = both_nan_i32.select(zero_int, abs_ulp_diff);

        // FIX: Exclude both-NaN agreement lanes from valid count, not just mismatches.
        let is_valid_mask = !(nan_mismatch | inf_mismatch | x_nan | x_inf | both_nan)
            .cast::<<<S as NativeSimdVectors>::i32xN as GenericVector>::Mask>();

        let valid_abs_ulp_diff = is_valid_mask.select(abs_ulp_diff, zero_int);
        let valid_signed_ulp_diff = is_valid_mask.select(ulp_diff, zero_int);

        // Widen to u64 per-lane before summing to avoid i32 overflow with pathological errors.
        let mut abs_sum_u64: u64 = 0;
        let mut signed_sum_i64: i64 = 0;
        for i in 0..lanes as usize {
            abs_sum_u64 += valid_abs_ulp_diff.extractv(i) as u64;
            signed_sum_i64 += valid_signed_ulp_diff.extractv(i) as i64;
        }
        recording.abs_ulp_diff += abs_sum_u64;
        recording.signed_ulp_diff += signed_sum_i64 as i128;

        let valid_count = is_valid_mask.native_bitmask().unwrap().count_ones() as u64;
        recording.count += valid_count;

        // ── 3b. Domain counting (vectorized via classify + popcount) ──
        let is_valid_u32 = !(nan_mismatch | inf_mismatch | x_nan | x_inf | both_nan);
        let sign_set = x_bits_u32.cmp_gt(<S as NativeSimdVectors>::u32xN::splat(0x7FFF_FFFF));
        let exp_zero = (x_bits_u32 & <S as NativeSimdVectors>::u32xN::splat(0x7F80_0000))
            .cmp_eq(<S as NativeSimdVectors>::u32xN::splat(0));

        let neg_mask = is_valid_u32 & sign_set;
        let pos_denorm_mask = is_valid_u32 & !sign_set & exp_zero;
        let pos_normal_mask = is_valid_u32 & !sign_set & !exp_zero;

        recording.domain_stats[Domain::Negative as usize].count +=
            neg_mask.native_bitmask().unwrap().count_ones() as u64;
        recording.domain_stats[Domain::PositiveDenormal as usize].count +=
            pos_denorm_mask.native_bitmask().unwrap().count_ones() as u64;
        recording.domain_stats[Domain::PositiveNormal as usize].count +=
            pos_normal_mask.native_bitmask().unwrap().count_ones() as u64;

        // Optimistically add all valid inputs to the 0 ULP bucket.
        recording.ulp_distribution[0] += valid_count;

        // ── 4. Scalar extraction for histogram, distribution, domain error stats, RMS ──
        if valid_abs_ulp_diff.cmp_gt(zero_int).any() {
            let exps = (x_bits_u32 >> 23) & <S as NativeSimdVectors>::u32xN::splat(0xFF);

            for i in 0..lanes as usize {
                let current_ulp = valid_abs_ulp_diff.extractv(i) as u64;
                let current_signed = valid_signed_ulp_diff.extractv(i) as i64;

                if current_ulp > 0 {
                    let x_val = x.extractv(i);
                    let x_bits_scalar = x_val.to_bits();
                    let domain = Domain::classify(x_bits_scalar);
                    let ds = &mut recording.domain_stats[domain as usize];

                    ds.abs_ulp_sum += current_ulp;
                    ds.signed_ulp_sum += current_signed as i128;
                    ds.sum_sq_ulp += (current_ulp as u128) * (current_ulp as u128);

                    if current_ulp > ds.max_ulp {
                        ds.max_ulp = current_ulp;
                        ds.worst_x = x_val;
                        ds.worst_actual = actual.extractv(i);
                        ds.worst_expected = expected.extractv(i);
                    }

                    // Global sum-of-squares
                    recording.sum_sq_ulp += (current_ulp as u128) * (current_ulp as u128);

                    // Correct the optimistic 0 ULP bucket
                    recording.ulp_distribution[0] -= 1;

                    if current_ulp < 16 {
                        recording.ulp_distribution[current_ulp as usize] += 1;
                    } else {
                        recording.ulp_greater_than_15 += 1;
                    }
                }

                let exp_idx = exps.extractv(i) as usize;
                recording.exp_ulp_sum[exp_idx] += current_ulp;
                if current_ulp > recording.exp_histogram[exp_idx] {
                    recording.exp_histogram[exp_idx] = current_ulp;
                    recording.worst_input_for_exp[exp_idx] = x.extractv(i);
                }

                // Global worst ULP tracker
                if current_ulp > recording.max_ulp_diff {
                    recording.max_ulp_diff = current_ulp;
                    recording.worst_x = x.extractv(i);
                    recording.worst_actual = actual.extractv(i);
                    recording.worst_expected = expected.extractv(i);
                }
            }
        }

        // ── 4b. Important range binned histogram ──
        // Vectorized range check: cmp against NaN is always false, so this
        // is a no-op when no important range is specified.
        let in_range = x.cmp_ge(important_lo_v) & x.cmp_le(important_hi_v);
        let in_range_u32 = in_range.cast::<<<S as NativeSimdVectors>::u32xN as GenericVector>::Mask>();
        let in_range_valid = is_valid_u32 & in_range_u32;

        if in_range_valid.any() {
            let in_range_i32 = in_range_valid.cast::<<<S as NativeSimdVectors>::i32xN as GenericVector>::Mask>();
            let ir_abs_ulp = in_range_i32.select(abs_ulp_diff, zero_int);
            let bitmask = in_range_valid.native_bitmask().unwrap() as u64;

            for i in 0..lanes as usize {
                if bitmask & (1u64 << i) == 0 {
                    continue;
                }

                let x_val = x.extractv(i);
                let current_ulp = ir_abs_ulp.extractv(i) as u64;
                let ir = &mut recording.important_range;

                let bin = ((x_val - important_lo) * inv_bin_width) as usize;
                let bin = bin.min(NUM_IMPORTANT_BINS - 1);

                ir.bins[bin].0 += current_ulp;
                ir.bins[bin].1 += 1;
                ir.count += 1;
                ir.ulp_sum += current_ulp;

                if current_ulp < 16 {
                    ir.ulp_distribution[current_ulp as usize] += 1;
                } else {
                    ir.ulp_greater_than_15 += 1;
                }

                if current_ulp > ir.max_ulp {
                    ir.max_ulp = current_ulp;
                    ir.worst_x = x_val;
                    ir.worst_actual = actual.extractv(i);
                    ir.worst_expected = expected.extractv(i);
                }
            }
        }

        // ── 4c. Reasonable range [-1e7, 1e7] binned histogram ──
        let in_reasonable = x.cmp_ge(reasonable_lo_v) & x.cmp_le(reasonable_hi_v);
        let in_reasonable_u32 = in_reasonable.cast::<<<S as NativeSimdVectors>::u32xN as GenericVector>::Mask>();
        let in_reasonable_valid = is_valid_u32 & in_reasonable_u32;

        if in_reasonable_valid.any() {
            let in_reasonable_i32 =
                in_reasonable_valid.cast::<<<S as NativeSimdVectors>::i32xN as GenericVector>::Mask>();
            let rr_abs_ulp = in_reasonable_i32.select(abs_ulp_diff, zero_int);
            let bitmask = in_reasonable_valid.native_bitmask().unwrap() as u64;

            for i in 0..lanes as usize {
                if bitmask & (1u64 << i) == 0 {
                    continue;
                }

                let x_val = x.extractv(i);
                let current_ulp = rr_abs_ulp.extractv(i) as u64;
                let rr = &mut recording.reasonable_range;

                let bin = ((x_val - reasonable_lo) * reasonable_inv_bin_width) as usize;
                let bin = bin.min(NUM_IMPORTANT_BINS - 1);

                rr.bins[bin].0 += current_ulp;
                rr.bins[bin].1 += 1;
                rr.count += 1;
                rr.ulp_sum += current_ulp;

                if current_ulp < 16 {
                    rr.ulp_distribution[current_ulp as usize] += 1;
                } else {
                    rr.ulp_greater_than_15 += 1;
                }

                if current_ulp > rr.max_ulp {
                    rr.max_ulp = current_ulp;
                    rr.worst_x = x_val;
                    rr.worst_actual = actual.extractv(i);
                    rr.worst_expected = expected.extractv(i);
                }
            }
        }

        // ── 5. Monotonicity check (correct adjacent-lane comparison) ──
        if recording.monotonic {
            for i in 0..lanes as usize {
                let cur_a = actual_linear.extractv(i);
                let cur_e = expected_linear.extractv(i);

                let x_val = x.extractv(i);
                let a_val = actual.extractv(i);
                let e_val = expected.extractv(i);
                let cur_finite = x_val.is_finite() && a_val.is_finite() && e_val.is_finite();

                let (prev_a, prev_e, prev_finite) = if i == 0 {
                    (prev_actual_linear_last, prev_expected_linear_last, prev_last_was_finite)
                } else {
                    let prev_x = x.extractv(i - 1);
                    let prev_av = actual.extractv(i - 1);
                    let prev_ev = expected.extractv(i - 1);
                    (
                        actual_linear.extractv(i - 1),
                        expected_linear.extractv(i - 1),
                        prev_x.is_finite() && prev_av.is_finite() && prev_ev.is_finite(),
                    )
                };

                if cur_finite && prev_finite {
                    let a_dir = (cur_a - prev_a).signum();
                    let e_dir = (cur_e - prev_e).signum();
                    if a_dir * e_dir < 0 {
                        recording.monotonic = false;
                        break;
                    }
                }
            }

            prev_actual_linear_last = actual_linear.extractv(last_lane);
            prev_expected_linear_last = expected_linear.extractv(last_lane);
            let lx = x.extractv(last_lane);
            let la = actual.extractv(last_lane);
            let le = expected.extractv(last_lane);
            prev_last_was_finite = lx.is_finite() && la.is_finite() && le.is_finite();
        }

        // ── 6. Progress ──
        if let Some(pb) = pb {
            // Flush every ~16M inputs to avoid per-iteration overhead on the draw lock.
            if start_bits & 0x00FF_FFFF < lanes {
                pb.set_position(start_bits);
            }
        }

        bits += <S as NativeSimdVectors>::i32xN::splat(lanes as i32);
        start_bits += lanes;
    }

    if let Some(pb) = pb {
        pb.finish();
    }
    recording
}

// ── SVG Builder ──

/// Population of valid inputs per exponent bucket in an exhaustive f32 sweep.
/// Each biased exponent 0..=254 has 2^23 positive + 2^23 negative mantissa patterns.
/// Exponent 255 is all NaN/Inf, excluded.
const INPUTS_PER_EXP: u64 = 2 * (1 << 23);

/// Builder for an individual SVG `<text>` element, with optional attributes via chaining.
struct TextEl {
    x: f64,
    y: f64,
    content: String,
    class: Option<&'static str>,
    anchor: Option<&'static str>,
    font_size: Option<f64>,
    fill: Option<&'static str>,
    font_family: Option<&'static str>,
    font_weight: Option<&'static str>,
    rotation: Option<(f64, f64, f64)>,
}

impl TextEl {
    fn new(x: f64, y: f64, content: impl Into<String>) -> Self {
        Self {
            x,
            y,
            content: content.into(),
            class: None,
            anchor: None,
            font_size: None,
            fill: None,
            font_family: None,
            font_weight: None,
            rotation: None,
        }
    }

    fn class(mut self, c: &'static str) -> Self {
        self.class = Some(c);
        self
    }
    fn anchor(mut self, a: &'static str) -> Self {
        self.anchor = Some(a);
        self
    }
    fn font_size(mut self, s: f64) -> Self {
        self.font_size = Some(s);
        self
    }
    fn fill(mut self, f: &'static str) -> Self {
        self.fill = Some(f);
        self
    }
    fn font_family(mut self, f: &'static str) -> Self {
        self.font_family = Some(f);
        self
    }
    fn font_weight(mut self, w: &'static str) -> Self {
        self.font_weight = Some(w);
        self
    }
    fn rotate(mut self, deg: f64, cx: f64, cy: f64) -> Self {
        self.rotation = Some((deg, cx, cy));
        self
    }

    fn render(&self, out: &mut String) {
        write!(out, "<text x=\"{:.1}\" y=\"{:.1}\"", self.x, self.y).unwrap();
        if let Some(c) = self.class {
            write!(out, " class=\"{}\"", c).unwrap();
        }
        if let Some(a) = self.anchor {
            write!(out, " text-anchor=\"{}\"", a).unwrap();
        }
        if let Some(s) = self.font_size {
            write!(out, " font-size=\"{}\"", s).unwrap();
        }
        if let Some(f) = self.fill {
            write!(out, " fill=\"{}\"", f).unwrap();
        }
        if let Some(f) = self.font_family {
            write!(out, " font-family=\"{}\"", f).unwrap();
        }
        if let Some(w) = self.font_weight {
            write!(out, " font-weight=\"{}\"", w).unwrap();
        }
        if let Some((deg, cx, cy)) = self.rotation {
            write!(out, " transform=\"rotate({},{:.1},{:.1})\"", deg, cx, cy).unwrap();
        }
        write!(out, ">{}</text>", self.content).unwrap();
    }
}

/// Accumulates SVG elements and renders the final `<svg>` document.
struct SvgDocument {
    width: f64,
    height: f64,
    clip_defs: String,
    style: String,
    body: String,
}

#[rustfmt::skip]
impl SvgDocument {
    fn new(width: f64, height: f64) -> Self {
        Self {
            width,
            height,
            clip_defs: String::new(),
            style: String::new(),
            body: String::with_capacity(8192),
        }
    }

    fn set_style(&mut self, css: &str) {
        self.style = css.to_string();
    }

    fn add_clip_rect(&mut self, id: &str, x: f64, y: f64, w: f64, h: f64) {
        write!(self.clip_defs, "<clipPath id=\"{id}\"><rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{w:.1}\" height=\"{h:.1}\"/></clipPath>").unwrap();
    }

    fn line(&mut self, class: &str, x1: f64, y1: f64, x2: f64, y2: f64) {
        write!(self.body, "<line class=\"{class}\" x1=\"{x1:.1}\" y1=\"{y1:.1}\" x2=\"{x2:.1}\" y2=\"{y2:.1}\"/>").unwrap();
    }

    fn text(&mut self, el: TextEl) {
        el.render(&mut self.body);
    }

    fn rect(&mut self, x: f64, y: f64, w: f64, h: f64, fill: &str, stroke: &str, stroke_width: f64) {
        write!(self.body, "<rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{w:.1}\" height=\"{h:.1}\" fill=\"{fill}\" stroke=\"{stroke}\" stroke-width=\"{stroke_width}\"/>").unwrap();
    }

    fn circle(&mut self, cx: f64, cy: f64, r: f64, fill: &str) {
        write!(self.body, "<circle cx=\"{cx:.1}\" cy=\"{cy:.1}\" r=\"{r}\" fill=\"{fill}\"/>").unwrap();
    }

    fn polyline_clipped(&mut self, points: &str, stroke: &str, stroke_width: f64, clip_id: &str) {
        write!(self.body, "<polyline points=\"{points}\" fill=\"none\" stroke=\"{stroke}\" stroke-width=\"{stroke_width}\" stroke-linejoin=\"round\" clip-path=\"url(#{clip_id})\"/>").unwrap();
    }

    fn polygon_clipped(&mut self, points: &str, fill: &str, fill_opacity: f64, clip_id: &str) {
        write!(
            self.body,
            "<polygon points=\"{points}\" fill=\"{fill}\" fill-opacity=\"{fill_opacity}\" clip-path=\"url(#{clip_id})\"/>",
        )
        .unwrap();
    }

    fn path_clipped(
        &mut self,
        d: &str,
        fill: &str,
        fill_opacity: f64,
        stroke: &str,
        stroke_width: f64,
        linejoin: &str,
        clip_id: &str,
    ) {
        write!(self.body, "<path d=\"{d}\" fill=\"{fill}\" fill-opacity=\"{fill_opacity}\" stroke=\"{stroke}\" stroke-width=\"{stroke_width}\" stroke-linejoin=\"{linejoin}\" clip-path=\"url(#{clip_id})\"/>").unwrap();
    }

    /// Wrap previously-pushed content in a clipped group.
    /// `content_fn` pushes elements into `body`; they'll be wrapped in `<g clip-path="...">`.
    fn clipped_group(&mut self, clip_id: &str, content_fn: impl FnOnce(&mut Self)) {
        let start = self.body.len();
        content_fn(self);
        let inner = self.body[start..].to_string();
        self.body.truncate(start);
        write!(self.body, "<g clip-path=\"url(#{clip_id})\">{inner}</g>").unwrap();
    }

    fn build(self) -> String {
        let mut out = String::with_capacity(self.body.len() + 512);
        write!(
            out,
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {} {}\" width=\"{}\" height=\"{}\">",
            self.width, self.height, self.width, self.height,
        ).unwrap();

        if !self.clip_defs.is_empty() {
            write!(out, "<defs>{}</defs>", self.clip_defs).unwrap();
        }

        if !self.style.is_empty() {
            write!(out, "<style>{}</style>", self.style).unwrap();
        }

        out.push_str(&self.body);
        out.push_str("</svg>");
        out
    }
}

/// Chart coordinate system and layout constants.
struct SvgChart {
    width: f64,
    height: f64,
    margin_left: f64,
    margin_right: f64,
    margin_top: f64,
    margin_bottom: f64,
}

const CHART_STYLE: &str = "svg{background:#fafafa}\
    .gl{stroke:#e5e7eb;stroke-width:.5;fill:none}\
    .yl{font-size:11px;fill:#6b7280;font-family:monospace;text-anchor:end}\
    .xt{stroke:#9ca3af;stroke-width:1;fill:none}";

impl SvgChart {
    fn plot_w(&self) -> f64 {
        self.width - self.margin_left - self.margin_right
    }
    fn plot_h(&self) -> f64 {
        self.height - self.margin_top - self.margin_bottom
    }

    fn x_of(&self, frac: f64) -> f64 {
        self.margin_left + frac * self.plot_w()
    }
    fn y_of(&self, log_val: f64, max_log: f64) -> f64 {
        self.margin_top + self.plot_h() - (log_val / max_log) * self.plot_h()
    }
    fn y_bottom(&self) -> f64 {
        self.margin_top + self.plot_h()
    }
    fn center_x(&self) -> f64 {
        self.margin_left + self.plot_w() / 2.0
    }

    /// Initialize an `SvgDocument` with this chart's dimensions, clip path, style, and plot background.
    fn new_document(&self, extra_style: &str) -> SvgDocument {
        let mut doc = SvgDocument::new(self.width, self.height);
        doc.add_clip_rect("pc", self.margin_left, self.margin_top, self.plot_w(), self.plot_h());
        doc.set_style(&format!("{}{}", CHART_STYLE, extra_style));
        doc.rect(
            self.margin_left,
            self.margin_top,
            self.plot_w(),
            self.plot_h(),
            "#fff",
            "#d1d5db",
            0.5,
        );
        doc
    }

    /// Emit a horizontal grid line at the given Y coordinate, plus its label.
    fn y_grid(&self, doc: &mut SvgDocument, y: f64, label: &str) {
        doc.line("gl", self.margin_left, y, self.margin_left + self.plot_w(), y);
        doc.text(TextEl::new(self.margin_left - 8.0, y + 4.0, label).class("yl"));
    }

    /// Emit a vertical tick mark at `x` below the plot area, with a label.
    fn x_tick(&self, doc: &mut SvgDocument, x: f64, label: &str, label_class: &'static str) {
        doc.line("xt", x, self.y_bottom(), x, self.y_bottom() + 5.0);
        doc.text(TextEl::new(x, self.y_bottom() + 18.0, label).class(label_class));
    }

    /// Emit a rotated X-axis tick (for the exponent chart where labels overlap).
    fn x_tick_rotated(&self, doc: &mut SvgDocument, x: f64, label: &str, label_class: &'static str) {
        doc.line("xt", x, self.y_bottom(), x, self.y_bottom() + 5.0);
        let ty = self.y_bottom() + 14.0;
        doc.text(TextEl::new(x, ty, label).class(label_class).rotate(-45.0, x, ty));
    }

    fn title(&self, doc: &mut SvgDocument, content: &str) {
        doc.text(
            TextEl::new(self.width / 2.0, 22.0, content)
                .anchor("middle")
                .font_size(14.0)
                .font_weight("600")
                .fill("#111827")
                .font_family("sans-serif"),
        );
    }

    fn x_axis_label(&self, doc: &mut SvgDocument, label: &str) {
        doc.text(
            TextEl::new(self.center_x(), self.height - 6.0, label)
                .anchor("middle")
                .font_size(12.0)
                .fill("#374151")
                .font_family("sans-serif"),
        );
    }

    fn y_axis_label(&self, doc: &mut SvgDocument, label: &str) {
        let y_mid = self.margin_top + self.plot_h() / 2.0;
        doc.text(
            TextEl::new(16.0, y_mid, label)
                .anchor("middle")
                .font_size(12.0)
                .fill("#374151")
                .font_family("sans-serif")
                .rotate(-90.0, 16.0, y_mid),
        );
    }
}

/// Generate an SVG chart of mean absolute ULP error vs input exponent (log-scale).
///
/// X axis: biased f32 exponent 0..254 (log2 of input magnitude)
/// Y axis: log10(mean_abs_ulp + 1)
pub fn generate_ulp_svg(name: &str, exp_ulp_sum: &[u64; 256]) -> String {
    let chart = SvgChart {
        width: 1000.0,
        height: 420.0,
        margin_left: 72.0,
        margin_right: 24.0,
        margin_top: 36.0,
        margin_bottom: 64.0,
    };

    let avg: Vec<f64> = (0..255)
        .map(|e| exp_ulp_sum[e] as f64 / INPUTS_PER_EXP as f64)
        .collect();

    let max_log = avg.iter().map(|a| (a + 1.0).log10()).fold(0.0_f64, f64::max).max(0.1);

    // Build polyline and filled-area point strings.
    let mut points = String::with_capacity(255 * 16);
    let mut area_points = String::with_capacity(255 * 16 + 64);
    write!(area_points, "{:.1},{:.1}", chart.x_of(0.0), chart.y_of(0.0, max_log)).unwrap();
    for (e, &a) in avg.iter().enumerate() {
        let x = chart.x_of(e as f64 / 254.0);
        let y = chart.y_of((a + 1.0).log10(), max_log);
        if e > 0 {
            points.push(' ');
        }
        write!(points, "{:.1},{:.1}", x, y).unwrap();
        write!(area_points, " {:.1},{:.1}", x, y).unwrap();
    }
    write!(area_points, " {:.1},{:.1}", chart.x_of(1.0), chart.y_of(0.0, max_log)).unwrap();

    let mut doc = chart.new_document(".xl{font-size:10px;fill:#6b7280;font-family:monospace;text-anchor:end}");
    chart.title(
        &mut doc,
        &format!("{} - Mean Absolute ULP Error by Input Exponent", name),
    );

    // Y-axis grid
    let max_pow = max_log.ceil() as i32;
    let mut drew_nonzero = false;
    for p in 0..=max_pow {
        let log_val = p as f64;
        if log_val > max_log {
            break;
        }
        if p > 0 {
            drew_nonzero = true;
        }
        let label = if p == 0 { "0".to_string() } else { format!("1e{}", p) };
        doc.clipped_group("pc", |d| {
            d.line(
                "gl",
                chart.margin_left,
                chart.y_of(log_val, max_log),
                chart.margin_left + chart.plot_w(),
                chart.y_of(log_val, max_log),
            );
        });
        doc.text(TextEl::new(chart.margin_left - 8.0, chart.y_of(log_val, max_log) + 4.0, &label).class("yl"));
    }
    if !drew_nonzero {
        let top_ulp = 10.0_f64.powf(max_log) - 1.0;
        let y = chart.y_of(max_log, max_log);
        doc.clipped_group("pc", |d| {
            d.line("gl", chart.margin_left, y, chart.margin_left + chart.plot_w(), y);
        });
        doc.text(TextEl::new(chart.margin_left - 8.0, y + 4.0, format!("{:.1}", top_ulp)).class("yl"));
    }

    // Data series
    doc.polygon_clipped(&area_points, "#3b82f6", 0.12, "pc");
    doc.polyline_clipped(&points, "#2563eb", 1.5, "pc");

    // Peak annotation
    let (peak_exp, peak_avg) = avg
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(e, &a)| (e, a))
        .unwrap_or((0, 0.0));

    if peak_avg > 1.0 {
        let px = chart.x_of(peak_exp as f64 / 254.0);
        let py = chart.y_of((peak_avg + 1.0).log10(), max_log);
        doc.circle(px, py, 3.0, "#dc2626");
        doc.text(
            TextEl::new(
                px,
                py + 18.0,
                format!("peak: {:.1} ULP avg (exp {})", peak_avg, peak_exp),
            )
            .anchor("middle")
            .font_size(10.0)
            .fill("#dc2626")
            .font_family("sans-serif"),
        );
    }

    // X-axis ticks at notable exponents
    for &(exp, label) in &[
        (0, "denorm"),
        (63, "1e-19"),
        (96, "1e-9"),
        (117, "1e-3"),
        (127, "1.0"),
        (137, "1e+3"),
        (150, "8e+6"),
        (170, "3e+12"),
        (190, "6e+18"),
        (210, "5e+24"),
        (230, "4e+30"),
        (254, "1e+38"),
    ] {
        chart.x_tick_rotated(&mut doc, chart.x_of(exp as f64 / 254.0), label, "xl");
    }

    chart.x_axis_label(&mut doc, "Input Magnitude (f32 biased exponent)");
    chart.y_axis_label(&mut doc, "Mean Abs ULP Error (log\u{2081}\u{2080} scale)");

    doc.build()
}

/// Generate an SVG bar chart of mean absolute ULP error across the important
/// input range, with a linear X axis subdivided into bins.
pub fn generate_important_range_svg(name: &str, ir: &ImportantRangeStats) -> Option<String> {
    if !ir.has_range() || ir.count == 0 {
        return None;
    }

    let chart = SvgChart {
        width: 1000.0,
        height: 400.0,
        margin_left: 72.0,
        margin_right: 24.0,
        margin_top: 36.0,
        margin_bottom: 52.0,
    };

    let bin_avgs: Vec<f64> = (0..NUM_IMPORTANT_BINS).map(|b| ir.bin_mean_ulp(b)).collect();

    // loglog Y transform: log10(log10(avg + 1) + 1)
    // At avg=0 → 0 (baseline). Compresses large spikes that would dominate a single-log axis.
    let lly = |avg: f64| -> f64 { ((avg + 1.0).log10() + 1.0).log10() };

    let max_log = bin_avgs.iter().map(|&a| lly(a)).fold(0.0_f64, f64::max).max(lly(1.0));

    let baseline_y = chart.y_of(0.0, max_log);
    let lo = ir.lo as f64;
    let hi = ir.hi as f64;

    // Symlog transform: sign(x) * log10(1 + |x|).
    let symlog = |x: f64| -> f64 { x.signum() * (1.0 + x.abs()).log10() };
    let sl_lo = symlog(lo);
    let sl_hi = symlog(hi);
    let sl_range = sl_hi - sl_lo;
    let x_of_val = |v: f64| -> f64 { chart.margin_left + ((symlog(v) - sl_lo) / sl_range) * chart.plot_w() };
    let bin_val = |b: usize, side: f64| lo + (b as f64 + side) / NUM_IMPORTANT_BINS as f64 * (hi - lo);

    // Stepped contour path with non-uniform bin widths in symlog space.
    let mut step_path = String::with_capacity(NUM_IMPORTANT_BINS * 24 + 64);
    let first_y = chart.y_of(lly(bin_avgs[0]), max_log);
    write!(
        step_path,
        "M {:.1},{:.1} V {:.1}",
        x_of_val(bin_val(0, 0.0)),
        baseline_y,
        first_y
    )
    .unwrap();
    let mut cur_y = first_y;
    for b in 0..NUM_IMPORTANT_BINS {
        let next_y = if b + 1 < NUM_IMPORTANT_BINS {
            chart.y_of(lly(bin_avgs[b + 1]), max_log)
        } else {
            baseline_y
        };
        if (next_y - cur_y).abs() > 0.05 || b == NUM_IMPORTANT_BINS - 1 {
            write!(step_path, " H {:.1} V {:.1}", x_of_val(bin_val(b, 1.0)), next_y).unwrap();
            cur_y = next_y;
        }
    }
    step_path.push('Z');

    let mut doc = chart.new_document(".xlm{font-size:10px;fill:#6b7280;font-family:monospace;text-anchor:middle}");

    chart.title(&mut doc, &format!("{} - ULP Error in [{}, {}]", name, ir.lo, ir.hi));

    // Summary annotation
    doc.text(
        TextEl::new(
            chart.margin_left + chart.plot_w(),
            chart.margin_top - 6.0,
            format!(
                "mean: {:.3} ULP | max: {} ULP | n={}",
                ir.mean_abs_ulp(),
                ir.max_ulp,
                ir.count
            ),
        )
        .anchor("end")
        .font_size(10.0)
        .fill("#374151")
        .font_family("monospace"),
    );

    // Y-axis grid at ULP values 0, 1, 10, 100, ...
    let max_raw_ulp = bin_avgs.iter().cloned().fold(0.0_f64, f64::max);
    let mut ulp_ticks: Vec<f64> = vec![0.0];
    let mut k = 0i32;
    loop {
        let v = 10f64.powi(k);
        ulp_ticks.push(v);
        if v > max_raw_ulp {
            break;
        }
        k += 1;
    }
    for &ulp_val in &ulp_ticks {
        let y_log = lly(ulp_val);
        if y_log > max_log {
            continue;
        }
        let y = chart.y_of(y_log, max_log);
        let label = if ulp_val == 0.0 {
            "0".to_string()
        } else {
            format!("1e{}", ulp_val.log10().round() as i32)
        };
        chart.y_grid(&mut doc, y, &label);
    }

    // Data path (clipped)
    doc.path_clipped(&step_path, "#2563eb", 0.15, "#2563eb", 1.5, "miter", "pc");

    // X-axis ticks at decade boundaries and zero within [lo, hi], via symlog.
    let mut tick_vals: Vec<f64> = vec![0.0];
    let max_mag = lo.abs().max(hi.abs());
    let mut k = 0i32;
    loop {
        let v = 10f64.powi(k);
        if v > max_mag * 1.01 {
            break;
        }
        if v >= lo && v <= hi {
            tick_vals.push(v);
        }
        if -v >= lo && -v <= hi {
            tick_vals.push(-v);
        }
        k += 1;
    }
    tick_vals.retain(|&v| v >= lo && v <= hi);
    tick_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    tick_vals.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

    for val in &tick_vals {
        let label = if val.abs() < 1e-9 {
            "0".to_string()
        } else if val.abs() < 0.01 {
            format!("{:.2e}", val)
        } else if val.abs() >= 1000.0 {
            format!("{:.0e}", val)
        } else if val.fract().abs() < 1e-6 {
            format!("{:.0}", val)
        } else {
            format!("{:.2}", val)
        };
        chart.x_tick(&mut doc, x_of_val(*val), &label, "xlm");
    }

    chart.x_axis_label(&mut doc, "Input Value (symlog scale)");
    chart.y_axis_label(&mut doc, "Mean Abs ULP (log log scale)");

    Some(doc.build())
}

// ── Doc markdown generation ──

fn percent_encode_svg(svg: &str) -> String {
    let mut out = String::with_capacity(svg.len() + svg.len() / 8);
    for b in svg.bytes() {
        match b {
            b'%' => out.push_str("%25"),
            b'#' => out.push_str("%23"),
            b'"' => out.push_str("%22"),
            b'\n' | b'\r' => {} // strip
            0x80.. => write!(out, "%{:02X}", b).unwrap(),
            _ => out.push(b as char),
        }
    }
    out
}

/// Generate a markdown file suitable for `#[doc = include_str!(...)]` embedding.
/// Contains the SVG charts as percent-encoded `<img>` tags inside a `<details>` block.
pub fn generate_doc_markdown(
    name: &str,
    recording: &FnRecording,
    full_svg: &str,
    range_svg: Option<&str>,
    reasonable_svg: Option<&str>,
) -> String {
    let mut md = String::with_capacity(8192);

    writeln!(
        md,
        "<details><summary>Accuracy Analysis ({} valid inputs)</summary>\n",
        recording.count
    )
    .unwrap();

    // Summary table
    writeln!(md, "| Metric | Value |").unwrap();
    writeln!(md, "|:---|---:|").unwrap();
    writeln!(md, "| Mean Abs ULP | {:.4} |", recording.mean_abs_ulp()).unwrap();
    writeln!(md, "| Mean Signed ULP | {:+.4} |", recording.mean_signed_ulp()).unwrap();
    writeln!(md, "| RMS ULP | {:.4} |", recording.rms_ulp()).unwrap();
    writeln!(md, "| Max ULP | {} |", recording.max_ulp_diff).unwrap();
    writeln!(md, "| Monotonic | {} |", recording.monotonic).unwrap();
    writeln!(md, "| Exact at 0, ±1 | {} |\n", recording.exact_at_key_points).unwrap();

    // Important range section
    let ir = &recording.important_range;
    if ir.has_range() && ir.count > 0 {
        writeln!(md, "**Within [{}, {}]** ({} inputs):\n", ir.lo, ir.hi, ir.count).unwrap();
        writeln!(md, "| Metric | Value |").unwrap();
        writeln!(md, "|:---|---:|").unwrap();
        writeln!(md, "| Mean Abs ULP | {:.4} |", ir.mean_abs_ulp()).unwrap();
        writeln!(md, "| Max ULP | {} |", ir.max_ulp).unwrap();

        let total = ir.count as f64;
        writeln!(
            md,
            "| 0 ULP (exact) | {:.2}% |",
            ir.ulp_distribution[0] as f64 / total * 100.0
        )
        .unwrap();
        writeln!(
            md,
            "| ≤1 ULP | {:.2}% |",
            (ir.ulp_distribution[0] + ir.ulp_distribution[1]) as f64 / total * 100.0
        )
        .unwrap();
        writeln!(md).unwrap();

        if let Some(range_svg_str) = range_svg {
            writeln!(
                md,
                r#"<img src="data:image/svg+xml,{}" alt="{} ULP error in important range" />"#,
                percent_encode_svg(range_svg_str),
                name,
            )
            .unwrap();
            writeln!(md).unwrap();
        }
    }

    // Reasonable range section
    let rr = &recording.reasonable_range;
    if rr.has_range() && rr.count > 0 {
        writeln!(
            md,
            "**Within [{}, {}]** (reasonable range, {} inputs):\n",
            rr.lo, rr.hi, rr.count
        )
        .unwrap();
        writeln!(md, "| Metric | Value |").unwrap();
        writeln!(md, "|:---|---:|").unwrap();
        writeln!(md, "| Mean Abs ULP | {:.4} |", rr.mean_abs_ulp()).unwrap();
        writeln!(md, "| Max ULP | {} |", rr.max_ulp).unwrap();

        let rr_total = rr.count as f64;
        writeln!(
            md,
            "| 0 ULP (exact) | {:.2}% |",
            rr.ulp_distribution[0] as f64 / rr_total * 100.0
        )
        .unwrap();
        writeln!(
            md,
            "| ≤1 ULP | {:.2}% |",
            (rr.ulp_distribution[0] + rr.ulp_distribution[1]) as f64 / rr_total * 100.0
        )
        .unwrap();
        writeln!(md).unwrap();

        if let Some(reasonable_svg_str) = reasonable_svg {
            writeln!(
                md,
                r#"<img src="data:image/svg+xml,{}" alt="{} ULP error in reasonable range" />"#,
                percent_encode_svg(reasonable_svg_str),
                name,
            )
            .unwrap();
            writeln!(md).unwrap();
        }
    }

    // Full-range chart
    writeln!(md, "**Full f32 range:**\n").unwrap();
    writeln!(
        md,
        r#"<img src="data:image/svg+xml,{}" alt="{} ULP error chart (full range)" />"#,
        percent_encode_svg(&full_svg),
        name,
    )
    .unwrap();

    writeln!(md, "\n</details>").unwrap();
    md
}

// ── Main ──

fn main() {
    struct BenchmarkPolicy;

    impl Policy for BenchmarkPolicy {
        const POLICY: PolicyParameters = PolicyParameters {
            check_overflow: true,
            unroll_loops: true,
            precision: PrecisionPolicy::Best,
            avoid_branching: false,
            max_iterations: 10000,
            use_compensation: true,
            denormal_behavior: DenormalBehavior::Preserve,
        };
    }

    let pool = rayon::ThreadPoolBuilder::new().num_threads(22).build().unwrap();

    let mp = MultiProgress::new();
    let pb_style = ProgressStyle::with_template(
        "{prefix:>8} [{bar:30.cyan/dim}] {percent:>3}% | {human_pos}/{human_len} | ETA {eta}",
    )
    .unwrap()
    .progress_chars("━╸─");

    let total: u64 = u32::MAX as u64 + 1; // full f32 bit-space

    let recordings = pool.scope(|s| {
        let (tx, rx) = std::sync::mpsc::channel();

        macro_rules! spawn_analysis {
            // With important range
            ($func:ident vs $libm:ident, $name:expr, $lo:expr, $hi:expr) => {
                paste::paste! {
                    let tx = tx.clone();
                    let pb = mp.add(ProgressBar::new(total));
                    pb.set_style(pb_style.clone());
                    pb.set_prefix($name);
                    s.spawn(move |_| {
                        struct [<$func:camel Kernel>];

                        impl<V: TranscendentalMath<Element = f32> + SpecialMathWithPolicy<Element = f32>> AnalysisKernel<V>
                            for [<$func:camel Kernel>]
                        {
                            #[inline(always)]
                            fn actual(&self, x: V) -> V {
                                x.[<$func _p>]::<BenchmarkPolicy>()
                            }

                            #[inline(always)]
                            fn expected(&self, x: V) -> V {
                                // NOTE: libm f64→f32 is a good practical reference but not perfectly rounded.
                                // For 0-vs-1 ULP discrimination, consider using MPFR (e.g. the `rug` crate).
                                x.map(|x| libm::$libm(x as f64) as f32)
                            }
                        }

                        let recording = analyze_f32::<X86V3, _>(&[<$func:camel Kernel>], $lo, $hi, Some(&pb));
                        tx.send(($name, recording)).expect("Failed to send recording");
                    });
                }
            };
            // Without important range (NaN bounds → range check always false, zero overhead)
            ($func:ident vs $libm:ident, $name:expr) => {
                spawn_analysis!($func vs $libm, $name, f32::NAN, f32::NAN);
            };
        }

        spawn_analysis!(exp2 vs exp2, "exp2", -100f32, 100f32);
        spawn_analysis!(exp vs exp, "exp", -100f32, 100f32);
        spawn_analysis!(log2 vs log2, "log2", 0f32, 1e10f32);
        spawn_analysis!(log10 vs log10, "log10", 0f32, 1e10f32);
        spawn_analysis!(ln vs log, "ln", 0f32, 1e10f32);
        spawn_analysis!(sin vs sin, "sin", -100.0_f32, 100.0_f32);
        spawn_analysis!(cos vs cos, "cos", -100.0_f32, 100.0_f32);
        spawn_analysis!(asin vs asin, "asin", -1.0_f32, 1.0_f32);
        spawn_analysis!(acos vs acos, "acos", -1.0_f32, 1.0_f32);
        spawn_analysis!(atan vs atan, "atan", -1e6_f32, 1e6_f32);
        spawn_analysis!(sinh vs sinh, "sinh", -10f32, 10f32);
        spawn_analysis!(cosh vs cosh, "cosh", -10f32, 10f32);
        spawn_analysis!(tanh vs tanh, "tanh", -500f32, 500f32);
        spawn_analysis!(asinh vs asinh, "asinh", 500f32, 500f32);
        spawn_analysis!(acosh vs acosh, "acosh", 0.45f32, 100f32);
        spawn_analysis!(atanh vs atanh, "atanh", -2f32, 2f32);
        spawn_analysis!(cbrt vs cbrt, "cbrt", -100f32, 100f32);
        spawn_analysis!(tgamma vs tgamma, "tgamma", -20f32, 20f32);
        spawn_analysis!(lgamma vs lgamma, "lgamma", -100f32, 100f32);
        spawn_analysis!(erf vs erf, "erf", -100f32, 100f32);

        drop(tx);

        let mut recordings: HashMap<&'static str, FnRecording> = HashMap::new();

        while let Ok((name, recording)) = rx.recv() {
            recordings.insert(name, recording);
        }

        recordings
    });

    // ── Markdown report ──

    let mut file = File::create("analysis_results.md").expect("Failed to create markdown file");
    writeln!(file, "# Analysis Results\n").unwrap();

    for (recording_name, recording) in &recordings {
        let c = recording.count as f64;

        writeln!(file, "## Results for: `{}`\n", recording_name).unwrap();

        // ── Global Error Summary ──
        writeln!(file, "### Global Error Summary ({} valid inputs)\n", recording.count).unwrap();
        writeln!(file, "| Metric | Value |").unwrap();
        writeln!(file, "|---|---|").unwrap();
        writeln!(file, "| Mean Abs ULP | {:.6} |", recording.mean_abs_ulp()).unwrap();
        writeln!(file, "| Mean Signed | {:+.6} |", recording.mean_signed_ulp()).unwrap();
        writeln!(file, "| RMS ULP | {:.6} |\n", recording.rms_ulp()).unwrap();

        // ── ULP Distribution ──
        writeln!(file, "### ULP Distribution\n").unwrap();
        writeln!(file, "| ULP | Count | Percentage |").unwrap();
        writeln!(file, "|---:|---:|---:|").unwrap();
        for i in 0..16 {
            let bucket_count = recording.ulp_distribution[i];
            writeln!(
                file,
                "| {} | {} | {:.2}% |",
                i,
                bucket_count,
                (bucket_count as f64 / c) * 100.0
            )
            .unwrap();
        }
        writeln!(
            file,
            "| >15 | {} | {:.2}% |\n",
            recording.ulp_greater_than_15,
            (recording.ulp_greater_than_15 as f64 / c) * 100.0
        )
        .unwrap();

        // ── Important Range Stats ──
        let ir = &recording.important_range;
        if ir.has_range() && ir.count > 0 {
            writeln!(
                file,
                "### Important Range [{}, {}] ({} inputs)\n",
                ir.lo, ir.hi, ir.count
            )
            .unwrap();
            writeln!(file, "| Metric | Value |").unwrap();
            writeln!(file, "|---|---|").unwrap();
            writeln!(file, "| Mean Abs ULP | {:.6} |", ir.mean_abs_ulp()).unwrap();
            writeln!(file, "| Max ULP | {} |", ir.max_ulp).unwrap();
            let total = ir.count as f64;
            writeln!(
                file,
                "| 0 ULP (exact) | {:.2}% |",
                ir.ulp_distribution[0] as f64 / total * 100.0
            )
            .unwrap();
            writeln!(
                file,
                "| ≤1 ULP | {:.2}% |\n",
                (ir.ulp_distribution[0] + ir.ulp_distribution[1]) as f64 / total * 100.0
            )
            .unwrap();

            writeln!(file, "| ULP | Count | Percentage |").unwrap();
            writeln!(file, "|---:|---:|---:|").unwrap();
            for i in 0..16 {
                let bucket_count = ir.ulp_distribution[i];
                writeln!(
                    file,
                    "| {} | {} | {:.2}% |",
                    i,
                    bucket_count,
                    (bucket_count as f64 / total) * 100.0
                )
                .unwrap();
            }
            writeln!(
                file,
                "| >15 | {} | {:.2}% |\n",
                ir.ulp_greater_than_15,
                (ir.ulp_greater_than_15 as f64 / total) * 100.0
            )
            .unwrap();
        }

        // ── Reasonable Range Stats ──
        let rr = &recording.reasonable_range;
        if rr.has_range() && rr.count > 0 {
            writeln!(
                file,
                "### Reasonable Range [{}, {}] ({} inputs)\n",
                rr.lo, rr.hi, rr.count
            )
            .unwrap();
            writeln!(file, "| Metric | Value |").unwrap();
            writeln!(file, "|---|---|").unwrap();
            writeln!(file, "| Mean Abs ULP | {:.6} |", rr.mean_abs_ulp()).unwrap();
            writeln!(file, "| Max ULP | {} |", rr.max_ulp).unwrap();
            let rr_total = rr.count as f64;
            writeln!(
                file,
                "| 0 ULP (exact) | {:.2}% |",
                rr.ulp_distribution[0] as f64 / rr_total * 100.0
            )
            .unwrap();
            writeln!(
                file,
                "| ≤1 ULP | {:.2}% |\n",
                (rr.ulp_distribution[0] + rr.ulp_distribution[1]) as f64 / rr_total * 100.0
            )
            .unwrap();

            writeln!(file, "| ULP | Count | Percentage |").unwrap();
            writeln!(file, "|---:|---:|---:|").unwrap();
            for i in 0..16 {
                let bucket_count = rr.ulp_distribution[i];
                writeln!(
                    file,
                    "| {} | {} | {:.2}% |",
                    i,
                    bucket_count,
                    (bucket_count as f64 / rr_total) * 100.0
                )
                .unwrap();
            }
            writeln!(
                file,
                "| >15 | {} | {:.2}% |\n",
                rr.ulp_greater_than_15,
                (rr.ulp_greater_than_15 as f64 / rr_total) * 100.0
            )
            .unwrap();
        }

        // ── Mismatches ──
        writeln!(file, "### Mismatches\n").unwrap();
        writeln!(file, "| Type | Mismatch Count | Actual Total | Expected Total |").unwrap();
        writeln!(file, "|---|---|---|---|").unwrap();
        writeln!(
            file,
            "| NaN | {} | {} | {} |",
            recording.nan_mismatches, recording.actual_nans, recording.expected_nans
        )
        .unwrap();
        writeln!(file, "| Inf | {} | - | - |", recording.inf_mismatches).unwrap();
        writeln!(file, "| Zero | {} | - | - |\n", recording.zero_mismatches).unwrap();

        // ── Properties ──
        writeln!(file, "### Properties\n").unwrap();
        writeln!(file, "| Property | Status |").unwrap();
        writeln!(file, "|---|---|").unwrap();
        writeln!(file, "| Monotonic | `{}` |", recording.monotonic).unwrap();
        writeln!(file, "| Exact at key points | `{}` |\n", recording.exact_at_key_points).unwrap();

        // ── Worst Finite Offender ──
        writeln!(file, "### Worst Finite Offender\n").unwrap();
        writeln!(file, "**Max ULP Diff:** {}\n", recording.max_ulp_diff).unwrap();
        writeln!(file, "| Value | Float | Hex |").unwrap();
        writeln!(file, "|---|---|---|").unwrap();
        writeln!(
            file,
            "| Input (x) | `{:e}` | `0x{:08X}` |",
            recording.worst_x,
            recording.worst_x.to_bits()
        )
        .unwrap();
        writeln!(
            file,
            "| Actual | `{:e}` | `0x{:08X}` |",
            recording.worst_actual,
            recording.worst_actual.to_bits()
        )
        .unwrap();
        writeln!(
            file,
            "| Expected | `{:e}` | `0x{:08X}` |\n",
            recording.worst_expected,
            recording.worst_expected.to_bits()
        )
        .unwrap();

        // ── Per-Domain Breakdown ──
        writeln!(file, "### Per-Domain Breakdown\n").unwrap();
        writeln!(file, "| Domain | Count | Mean Abs ULP | Mean Signed | RMS ULP | Max ULP | Worst Input | Worst Actual | Worst Expected |").unwrap();
        writeln!(file, "|---|---|---|---|---|---|---|---|---|").unwrap();
        for domain in [Domain::Negative, Domain::PositiveDenormal, Domain::PositiveNormal] {
            let ds = &recording.domain_stats[domain as usize];
            if ds.count == 0 {
                writeln!(file, "| {} | 0 | - | - | - | - | - | - | - |", domain.name()).unwrap();
                continue;
            }
            if ds.max_ulp > 0 {
                writeln!(
                    file,
                    "| {} | {} | {:.6} | {:+.6} | {:.6} | {} | `{:e}`<br>`0x{:08X}` | `{:e}`<br>`0x{:08X}` | `{:e}`<br>`0x{:08X}` |",
                    domain.name(),
                    ds.count,
                    ds.mean_abs_ulp(),
                    ds.mean_signed_ulp(),
                    ds.rms_ulp(),
                    ds.max_ulp,
                    ds.worst_x, ds.worst_x.to_bits(),
                    ds.worst_actual, ds.worst_actual.to_bits(),
                    ds.worst_expected, ds.worst_expected.to_bits()
                ).unwrap();
            } else {
                writeln!(
                    file,
                    "| {} | {} | {:.6} | {:+.6} | {:.6} | {} | - | - | - |",
                    domain.name(),
                    ds.count,
                    ds.mean_abs_ulp(),
                    ds.mean_signed_ulp(),
                    ds.rms_ulp(),
                    ds.max_ulp
                )
                .unwrap();
            }
        }
        writeln!(file, "\n---").unwrap();
    }

    // ── SVG + doc markdown generation ──

    std::fs::create_dir_all("docs").unwrap();

    for (recording_name, recording) in &recordings {
        // Full-range exponent chart
        let full_svg = generate_ulp_svg(recording_name, &recording.exp_ulp_sum);
        let svg_path = format!("docs/{}_ulp_chart.svg", recording_name);
        File::create(&svg_path).unwrap().write_all(full_svg.as_bytes()).unwrap();
        println!("Wrote {}", svg_path);

        // Important-range chart
        let range_svg = generate_important_range_svg(recording_name, &recording.important_range);
        if let Some(ref svg) = range_svg {
            let path = format!("docs/{}_ulp_important_range.svg", recording_name);
            File::create(&path).unwrap().write_all(svg.as_bytes()).unwrap();
            println!("Wrote {}", path);
        }

        // Reasonable-range chart [-1e7, 1e7]
        let reasonable_svg = generate_important_range_svg(recording_name, &recording.reasonable_range);
        if let Some(ref svg) = reasonable_svg {
            let path = format!("docs/{}_ulp_reasonable_range.svg", recording_name);
            File::create(&path).unwrap().write_all(svg.as_bytes()).unwrap();
            println!("Wrote {}", path);
        }

        // Doc markdown with embedded SVGs for rustdoc
        let doc_md = generate_doc_markdown(
            recording_name,
            recording,
            &full_svg,
            range_svg.as_deref(),
            reasonable_svg.as_deref(),
        );
        let md_path = format!("docs/{}_accuracy.md", recording_name);
        File::create(&md_path).unwrap().write_all(doc_md.as_bytes()).unwrap();
        println!("Wrote {}", md_path);
    }

    println!("Analysis complete. Results written to 'analysis_results.md' and docs/");
}
