/// Reduces a 4-lane `f32` SIMD vector using two operations simultaneously, returning both results.
///
/// A single shuffle feeds both `$op1` and `$op2`, then a shared `movehdup` feeds both finals.
#[rustfmt::skip]
macro_rules! _mm_reduce2_ps_v2 {
    ($value:expr; $op1:ident $last1:ident, $op2:ident $last2:ident) => {#[allow(unused_unsafe)] unsafe {
        let xmm0 = $value;
        let xmm1 = arch::_mm_shuffle_ps(xmm0, xmm0, 0b11_10_11_10);
        let xmm_a = arch::$op1(xmm0, xmm1);
        let xmm_b = arch::$op2(xmm0, xmm1);
        let xmm1 = arch::_mm_movehdup_ps(xmm_a);
        let xmm2 = arch::_mm_movehdup_ps(xmm_b);
        (
            arch::_mm_cvtss_f32(arch::$last1(xmm_a, xmm1)),
            arch::_mm_cvtss_f32(arch::$last2(xmm_b, xmm2)),
        )
    }};
}

/// Reduces an 8-lane 16-bit integer SIMD vector to a single lane value using the specified
/// lane-wise op (e.g. `_mm_min_epi16`, `_mm_add_epi16`). Log-tree fold: 8 -> 4 -> 2 -> 1 via
/// successive byte-shift-down + op, then extract lane 0. Returns an `i16` (cast at the call
/// site for unsigned). Only valid for commutative+associative ops; `add`/`mullo` wrap.
#[rustfmt::skip]
macro_rules! _mm_reduce_epi16_v2 {
    ($value:expr; $op:ident) => {{#[allow(unused_unsafe)] unsafe {
        let x = $value;
        let x = arch::$op(x, arch::_mm_bsrli_si128(x, 8)); // fold lanes 4..8 into 0..4
        let x = arch::$op(x, arch::_mm_bsrli_si128(x, 4)); // fold lanes 2..4 into 0..2
        let x = arch::$op(x, arch::_mm_bsrli_si128(x, 2)); // fold lane 1 into lane 0
        arch::_mm_extract_epi16::<0>(x) as i16
    }}};
}

/// Reduces a 16-lane 8-bit integer SIMD vector to a single lane value using the specified
/// lane-wise op (e.g. `_mm_min_epi8`, `_mm_add_epi8`). Log-tree fold: 16 -> 8 -> 4 -> 2 -> 1
/// via successive byte-shift-down + op, then extract lane 0. Returns an `i8` (cast at the call
/// site for unsigned). Only valid for commutative+associative ops; `add` wraps.
#[rustfmt::skip]
macro_rules! _mm_reduce_epi8_v2 {
    ($value:expr; $op:ident) => {{#[allow(unused_unsafe)] unsafe {
        let x = $value;
        let x = arch::$op(x, arch::_mm_bsrli_si128(x, 8)); // fold lanes 8..16 into 0..8
        let x = arch::$op(x, arch::_mm_bsrli_si128(x, 4)); // fold lanes 4..8 into 0..4
        let x = arch::$op(x, arch::_mm_bsrli_si128(x, 2)); // fold lanes 2..4 into 0..2
        let x = arch::$op(x, arch::_mm_bsrli_si128(x, 1)); // fold lane 1 into lane 0
        arch::_mm_extract_epi8::<0>(x) as i8
    }}};
}

/// Reduces a 4-lane `f32` SIMD vector to a single `f32` value using the specified operation.
#[rustfmt::skip]
macro_rules! _mm_reduce_ps_v2 {
    ($value:expr; $op:ident $last:ident) => {#[allow(unused_unsafe)] unsafe {
        let xmm0 = $value;
        // duplicate higher half to lower half
        let xmm1 = arch::_mm_shuffle_ps(xmm0, xmm0, 0b11_10_11_10);
        // first reduce
        let xmm0 = arch::$op(xmm0, xmm1);
        // Duplicate odd-indexed elements (1, 1, 3, 3)
        let xmm1 = arch::_mm_movehdup_ps(xmm0);
        // final reduce and extract
        arch::_mm_cvtss_f32(arch::$last(xmm0, xmm1))
    }};
}
