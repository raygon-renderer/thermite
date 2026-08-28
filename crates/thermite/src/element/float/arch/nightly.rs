//! Nightly rung: the generic LLVM float intrinsics straight from `core`.
//!
//! These lower identically to the standard library's methods, and, unlike every
//! `cfg!`-driven rung below, LLVM selects them using the **enclosing function's**
//! target features. Inside a `#[thermite::dispatch]` trampoline they become a single
//! instruction even when the crate baseline has nothing, which is precisely what the
//! `cfg!` ladder cannot do. This rung therefore supersedes `std` entirely and is
//! checked first.
//!
//! `core::intrinsics` is an internal, unstable surface that does churn (there is no
//! `rintf64`, for one), which is why it sits behind the `nightly` feature with the
//! stable rungs left intact underneath.

macro_rules! impl_nightly {
    ($t:ty, $sqrt:ident, $floor:ident, $ceil:ident, $trunc:ident, $round:ident, $fma:ident,
     [$i_sqrt:ident, $i_floor:ident, $i_ceil:ident, $i_trunc:ident, $i_round:ident, $i_fma:ident]) => {
        #[inline(always)]
        pub fn $sqrt(x: $t) -> $t {
            core::intrinsics::$i_sqrt(x)
        }

        #[inline(always)]
        pub fn $floor(x: $t) -> $t {
            core::intrinsics::$i_floor(x)
        }

        #[inline(always)]
        pub fn $ceil(x: $t) -> $t {
            core::intrinsics::$i_ceil(x)
        }

        #[inline(always)]
        pub fn $trunc(x: $t) -> $t {
            core::intrinsics::$i_trunc(x)
        }

        // Ties to even, matching the vector backends.
        #[inline(always)]
        pub fn $round(x: $t) -> $t {
            core::intrinsics::$i_round(x)
        }

        #[inline(always)]
        pub fn $fma(x: $t, y: $t, z: $t) -> $t {
            core::intrinsics::$i_fma(x, y, z)
        }
    };
}

impl_nightly!(
    f64,
    sqrt,
    floor,
    ceil,
    trunc,
    round,
    fma,
    [sqrtf64, floorf64, ceilf64, truncf64, round_ties_even_f64, fmaf64]
);

impl_nightly!(
    f32,
    sqrtf,
    floorf,
    ceilf,
    truncf,
    roundf,
    fmaf,
    [sqrtf32, floorf32, ceilf32, truncf32, round_ties_even_f32, fmaf32]
);
