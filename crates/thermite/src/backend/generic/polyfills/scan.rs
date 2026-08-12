//! Inclusive prefix scans over the lanes of a single register.
//!
//! A *forward* scan replaces lane `i` with `op(v[0], v[1], .., v[i])`; a *reverse*
//! (suffix) scan replaces it with `op(v[i], .., v[LANES-1])`. Both are computed with
//! the classic Hillis-Steele doubling ladder - `ceil(log2(LANES))` stages, each one a
//! cross-register [`align`](Register::align) plus one vector op - rather than the
//! `LANES-1` sequential steps a scalar loop needs.
//!
//! These are the primitive behind SAH bin offsets, stream-compaction write offsets,
//! and running-extent queries, where the alternative is spilling the register to
//! memory and walking it.
//!
//! # Why the ladder is gated
//!
//! Every stage is an `align`. Where a register has no native cross-register align
//! ([`HAS_NATIVE_ALIGN`](Register::HAS_NATIVE_ALIGN) is false - the scalar backend,
//! SPIR-V, and the odd-lane-count [`ReducedRegister`](crate::register::reduced::ReducedRegister)),
//! each stage expands to the generic `swizzle_const` default instead, and a ladder of
//! those loses to just walking the lanes. So the choice is a compile-time `if const`:
//! vector ladder when the align is real, scalar loop when it is not.
//!
//! # Fill values, and why `min`/`max` do not use `MIN`/`MAX`
//!
//! Each stage shifts the vector by `s` lanes and combines; the `s` lanes shifted in
//! need a value that leaves the already-final lanes untouched.
//!
//! For `sum` that is [`ZERO`](NumericRegister::ZERO), exact and free.
//!
//! For `min`/`max` the obvious choice is [`MAX`](NumericRegister::MAX) /
//! [`MIN`](NumericRegister::MIN), but those are `f32::MAX`/`f32::MIN` on float
//! registers, **not** `+/-inf`. A lane holding `+inf` would come back as `f32::MAX`
//! (`min(inf, f32::MAX) == f32::MAX`). Instead the fill is a broadcast of the *edge
//! lane* - `v[0]` forward, `v[LANES-1]` reverse - which is exact for every input,
//! infinities included: forward, the invariant is `prefix[i] = min(v[0..=i]) <= v[0]`,
//! so re-applying `min(prefix[i], v[0]) == prefix[i]` is harmless (symmetrically for
//! `max`, and for the reverse direction against the last lane). The broadcast is
//! loop-invariant, one hoisted instruction.
//!
//! # NaN
//!
//! `min`/`max` scans inherit the backend's `min`/`max` NaN behaviour, and the ladder
//! combines lanes in tree order while the scalar fallback walks them in sequence, so on
//! NaN input the two need not agree on *which* operand wins. Same caveat as
//! [`min`](NumericRegister::min) itself, which the differential harness tests at
//! `Tol::ExactOrNan`. Results are exact and backend-identical for NaN-free input.
//!
//! # Masked scans
//!
//! There is no `_c`/`_m`/`_z` variant: the generated form would be "scan everything,
//! then blend", which is not what a masked scan means. To scan only selected lanes,
//! neutralise the others first - `v.zz(mask).prefix_sum()` for a sum, or
//! `mask.select(v, Self::splat(inf)).prefix_min()` for a min.

use generic_array::typenum::Unsigned;

use crate::register::{CoreRegister, NumericRegister, Register, Storage};

/// Lane count as a `usize`, for the `if const` / `match const` gates below.
macro_rules! lanes {
    ($r:ty) => {
        <<$r as CoreRegister>::Lanes as Unsigned>::USIZE
    };
}

/// True when the vector ladder is the right lowering for `R`: the align has to be
/// real, and the width has to be one the forward offset table covers.
macro_rules! use_ladder {
    ($r:ty) => {
        const { <$r as Register>::HAS_NATIVE_ALIGN && lanes!($r).is_power_of_two() && lanes!($r) <= 64 }
    };
}

/// One forward (toward higher lanes) doubling ladder.
///
/// Stage `s` wants `shifted[i] = v[i - s]`, with the low `s` lanes taking `fill`.
/// `align::<OFFSET>(a, b)[i] == concat(a, b)[OFFSET + i]`, so with `a = fill` and
/// `b = v` that is `OFFSET == LANES - s`: for `i >= s` the index lands in `v` at
/// `i - s`, and for `i < s` it lands back in `fill`.
///
/// `OFFSET` is a const-generic argument and so must be a literal - `LANES - s` is not
/// expressible on stable. Hence the match on the (compile-time) lane count with a
/// precomputed offset list per width; exactly one arm survives monomorphization, and
/// `use_ladder!` has already excluded every width without an arm.
macro_rules! forward_ladder {
    ($r:ty, $v:expr, $fill:expr, $op:path) => {{
        let mut v = $v;
        let f = $fill;
        #[rustfmt::skip]
        let () = match const { lanes!($r) } {
            0 | 1 => {}
            2  => { v = $op(v, <$r as Register>::align::<1>(f, v)); }
            4  => { v = $op(v, <$r as Register>::align::<3>(f, v));
                    v = $op(v, <$r as Register>::align::<2>(f, v)); }
            8  => { v = $op(v, <$r as Register>::align::<7>(f, v));
                    v = $op(v, <$r as Register>::align::<6>(f, v));
                    v = $op(v, <$r as Register>::align::<4>(f, v)); }
            16 => { v = $op(v, <$r as Register>::align::<15>(f, v));
                    v = $op(v, <$r as Register>::align::<14>(f, v));
                    v = $op(v, <$r as Register>::align::<12>(f, v));
                    v = $op(v, <$r as Register>::align::<8>(f, v)); }
            32 => { v = $op(v, <$r as Register>::align::<31>(f, v));
                    v = $op(v, <$r as Register>::align::<30>(f, v));
                    v = $op(v, <$r as Register>::align::<28>(f, v));
                    v = $op(v, <$r as Register>::align::<24>(f, v));
                    v = $op(v, <$r as Register>::align::<16>(f, v)); }
            64 => { v = $op(v, <$r as Register>::align::<63>(f, v));
                    v = $op(v, <$r as Register>::align::<62>(f, v));
                    v = $op(v, <$r as Register>::align::<60>(f, v));
                    v = $op(v, <$r as Register>::align::<56>(f, v));
                    v = $op(v, <$r as Register>::align::<48>(f, v));
                    v = $op(v, <$r as Register>::align::<32>(f, v)); }
            // unreachable: `use_ladder!` gates on a power-of-two width <= 64. Panicking
            // is the right failure mode if a new width ever slips past that guard.
            _ => unreachable!(),
            };
        v
    }};
}

/// One reverse (toward lower lanes) doubling ladder.
///
/// Stage `s` wants `shifted[i] = v[i + s]`, with the high `s` lanes taking `fill`; that
/// is `align::<s>(v, fill)`. Unlike the forward direction the offset *is* the shift, so
/// it is already a literal and needs no per-width table - the `if const` chain just
/// stops once the shift covers the register.
macro_rules! reverse_ladder {
    ($r:ty, $v:expr, $fill:expr, $op:path) => {{
        let mut v = $v;
        let f = $fill;
        #[rustfmt::skip]
        let () = {
            if const { lanes!($r) >  1 } { v = $op(v, <$r as Register>::align::<1>(v, f)); }
            if const { lanes!($r) >  2 } { v = $op(v, <$r as Register>::align::<2>(v, f)); }
            if const { lanes!($r) >  4 } { v = $op(v, <$r as Register>::align::<4>(v, f)); }
            if const { lanes!($r) >  8 } { v = $op(v, <$r as Register>::align::<8>(v, f)); }
            if const { lanes!($r) > 16 } { v = $op(v, <$r as Register>::align::<16>(v, f)); }
            if const { lanes!($r) > 32 } { v = $op(v, <$r as Register>::align::<32>(v, f)); }
                            };
        v
    }};
}

/// Sequential scans through the lanes: the fallback for registers with no native
/// align, and the oracle the ladder is checked against in the differential suite.
///
/// The element-level comparisons mirror the operand order the ladder uses -
/// `min(current, accumulated)` - so a backend's `minps`-style "return the second
/// operand on NaN" semantics line up with the fallback as closely as a
/// tree-vs-sequential reassociation allows.
macro_rules! scalar_scan {
    ($name:ident, forward, $combine:expr) => {
        #[inline(always)]
        pub fn $name<R: NumericRegister>(mut value: Storage<R>) -> Storage<R> {
            let s = R::as_mut_slice(&mut value);
            let mut i = 1;
            while i < s.len() {
                let acc = s[i - 1];
                s[i] = $combine(s[i], acc);
                i += 1;
            }
            value
        }
    };
    ($name:ident, reverse, $combine:expr) => {
        #[inline(always)]
        pub fn $name<R: NumericRegister>(mut value: Storage<R>) -> Storage<R> {
            let s = R::as_mut_slice(&mut value);
            let mut i = s.len().saturating_sub(1);
            while i > 0 {
                i -= 1;
                let acc = s[i + 1];
                s[i] = $combine(s[i], acc);
            }
            value
        }
    };
}

scalar_scan!(scalar_prefix_sum, forward, |cur, acc| cur + acc);
scalar_scan!(scalar_prefix_min, forward, |cur, acc| if cur < acc { cur } else { acc });
scalar_scan!(scalar_prefix_max, forward, |cur, acc| if cur > acc { cur } else { acc });
scalar_scan!(scalar_reverse_prefix_sum, reverse, |cur, acc| cur + acc);
scalar_scan!(scalar_reverse_prefix_min, reverse, |cur, acc| if cur < acc {
    cur
} else {
    acc
});
scalar_scan!(scalar_reverse_prefix_max, reverse, |cur, acc| if cur > acc {
    cur
} else {
    acc
});

/// Broadcast of lane 0 - the `min`/`max` forward fill (see the module docs).
#[inline(always)]
fn first_lane<R: Register>(value: Storage<R>) -> Storage<R> {
    R::broadcast::<0>(value)
}

/// Broadcast of the last lane - the `min`/`max` reverse fill. `LANES - 1` is not a
/// literal, so this reverses first and broadcasts lane 0; both are single ops and the
/// pair is loop-invariant.
#[inline(always)]
fn last_lane<R: Register>(value: Storage<R>) -> Storage<R> {
    R::broadcast::<0>(R::reverse(value))
}

/// Inclusive forward prefix sum: `out[i] = v[0] + .. + v[i]`.
#[inline(always)]
pub fn prefix_sum<R: NumericRegister>(value: Storage<R>) -> Storage<R> {
    if const { !use_ladder!(R) } {
        return scalar_prefix_sum::<R>(value);
    }
    forward_ladder!(R, value, R::ZERO, R::add)
}

/// Inclusive forward prefix minimum: `out[i] = min(v[0], .., v[i])`.
#[inline(always)]
pub fn prefix_min<R: NumericRegister>(value: Storage<R>) -> Storage<R> {
    if const { !use_ladder!(R) } {
        return scalar_prefix_min::<R>(value);
    }
    forward_ladder!(R, value, first_lane::<R>(value), R::min)
}

/// Inclusive forward prefix maximum: `out[i] = max(v[0], .., v[i])`.
#[inline(always)]
pub fn prefix_max<R: NumericRegister>(value: Storage<R>) -> Storage<R> {
    if const { !use_ladder!(R) } {
        return scalar_prefix_max::<R>(value);
    }
    forward_ladder!(R, value, first_lane::<R>(value), R::max)
}

/// Inclusive reverse (suffix) sum: `out[i] = v[i] + .. + v[LANES-1]`.
#[inline(always)]
pub fn reverse_prefix_sum<R: NumericRegister>(value: Storage<R>) -> Storage<R> {
    if const { !use_ladder!(R) } {
        return scalar_reverse_prefix_sum::<R>(value);
    }
    reverse_ladder!(R, value, R::ZERO, R::add)
}

/// Inclusive reverse (suffix) minimum: `out[i] = min(v[i], .., v[LANES-1])`.
#[inline(always)]
pub fn reverse_prefix_min<R: NumericRegister>(value: Storage<R>) -> Storage<R> {
    if const { !use_ladder!(R) } {
        return scalar_reverse_prefix_min::<R>(value);
    }
    reverse_ladder!(R, value, last_lane::<R>(value), R::min)
}

/// Inclusive reverse (suffix) maximum: `out[i] = max(v[i], .., v[LANES-1])`.
#[inline(always)]
pub fn reverse_prefix_max<R: NumericRegister>(value: Storage<R>) -> Storage<R> {
    if const { !use_ladder!(R) } {
        return scalar_reverse_prefix_max::<R>(value);
    }
    reverse_ladder!(R, value, last_lane::<R>(value), R::max)
}
