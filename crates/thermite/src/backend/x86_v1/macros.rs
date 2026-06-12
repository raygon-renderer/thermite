/// Reduces a 2-lane `f64` SIMD vector using two operations simultaneously, returning both results.
#[rustfmt::skip]
macro_rules! _mm_reduce2_pd_v1 {
    ($value:expr; $op1:ident $last1:ident, $op2:ident $last2:ident) => {#[allow(unused_unsafe)] unsafe {
        let xmm0 = $value;
        let xmm1 = arch::_mm_unpackhi_pd(xmm0, xmm0);
        (
            arch::_mm_cvtsd_f64(arch::$last1(xmm0, xmm1)),
            arch::_mm_cvtsd_f64(arch::$last2(xmm0, xmm1)),
        )
    }};
}

/// Reduces a 2-lane `f64` SIMD vector to a single `f64` value using the specified operation.
#[rustfmt::skip]
macro_rules! _mm_reduce_pd_v1 {
    ($value:expr; $op:ident $last:ident) => {#[allow(unused_unsafe)] unsafe {
        let xmm0 = $value;
        // Duplicate odd-indexed elements (1, 1)
        let xmm1 = arch::_mm_unpackhi_pd(xmm0, xmm0);
        // final reduce and extract
        arch::_mm_cvtsd_f64(arch::$last(xmm0, xmm1))
    }};
}

/// Reduces 4-lane `f32` SIMD vector to a single `f32` value using the specified operation,
/// ignoring the last lane entirely.
#[rustfmt::skip]
macro_rules! _mm_reduce_ps3_v1 {
    ($value:expr; $last:ident) => {#[allow(unused_unsafe)] unsafe {
        let xmm0 = $value;
        let xmm1 = arch::_mm_shuffle_ps(xmm0, xmm0, 0b00_00_00_01);
        let xmm2 = arch::_mm_shuffle_ps(xmm0, xmm0, 0b00_00_00_10);
        let xmm0 = arch::$last(xmm0, xmm1);
        arch::_mm_cvtss_f32(arch::$last(xmm0, xmm2))
    }};
}

/// Reduces a 4-lane `i32` SIMD vector to a single `i32` value using the specified operation.
#[rustfmt::skip]
macro_rules! _mm_reduce_epi32_v1 {
    ($value:expr; $op:ident $last:ident) => {{#[allow(unused_unsafe)] unsafe {
        let xmm0 = $value;
        // duplicate higher half to lower half
        let xmm1 = arch::_mm_shuffle_epi32(xmm0, 0b11_10_11_10);
        // first reduce
        let xmm0 = arch::$op(xmm0, xmm1);
        // Duplicate odd-indexed elements (1, 1, 3, 3).
        let xmm1 = arch::_mm_shuffle_epi32(xmm0, 0b11_11_01_01);
        // final reduce and extract
        arch::_mm_cvtsi128_si32(arch::$last(xmm0, xmm1))
    }}};
}

/// `[a0,a1,a2,a3]` hadd `[b0,b1,b2,b3]` -> `[a0+a1, a2+a3, b0+b1, b2+b3]`
#[rustfmt::skip]
macro_rules! _mm_pairwise_sum_ps_v1 {
    ($lhs:expr, $rhs:expr) => {#[allow(unused_unsafe)] unsafe {
        let evens = arch::_mm_shuffle_ps($lhs, $rhs, 0b10_00_10_00); // [a0,a2,b0,b2]
        let odds  = arch::_mm_shuffle_ps($lhs, $rhs, 0b11_01_11_01); // [a1,a3,b1,b3]
        arch::_mm_add_ps(evens, odds)
    }};
}

/// `[a0,a1]` hadd `[b0,b1]` -> `[a0+a1, b0+b1]`
#[rustfmt::skip]
macro_rules! _mm_pairwise_sum_pd_v1 {
    ($lhs:expr, $rhs:expr) => {#[allow(unused_unsafe)] unsafe {
        let lo = arch::_mm_shuffle_pd($lhs, $rhs, 0b00); // [a0,b0]
        let hi = arch::_mm_shuffle_pd($lhs, $rhs, 0b11); // [a1,b1]
        arch::_mm_add_pd(lo, hi)
    }};
}

/// `[a0,a1,a2,a3]` hadd `[b0,b1,b2,b3]` -> `[a0+a1, a2+a3, b0+b1, b2+b3]`
/// Uses float shuffle to mix two sources (no 2-source `_mm_shuffle_epi32`).
#[rustfmt::skip]
macro_rules! _mm_pairwise_sum_epi32_v1 {
    ($lhs:expr, $rhs:expr) => {{#[allow(unused_unsafe)] unsafe {
        let lhs_ps = arch::_mm_castsi128_ps($lhs);
        let rhs_ps = arch::_mm_castsi128_ps($rhs);
        let lo = arch::_mm_shuffle_ps(lhs_ps, rhs_ps, 0b10_00_10_00); // [a0,a2,b0,b2]
        let hi = arch::_mm_shuffle_ps(lhs_ps, rhs_ps, 0b11_01_11_01); // [a1,a3,b1,b3]
        arch::_mm_add_epi32(arch::_mm_castps_si128(lo), arch::_mm_castps_si128(hi))
    }}};
}

/// `[a0,a1]` hadd `[b0,b1]` -> `[a0+a1, b0+b1]`
/// Uses double shuffle to mix two sources (no 2-source `_mm_shuffle_epi32` for 64-bit).
#[rustfmt::skip]
macro_rules! _mm_pairwise_sum_epi64_v1 {
    ($lhs:expr, $rhs:expr) => {{#[allow(unused_unsafe)] unsafe {
        let lhs_pd = arch::_mm_castsi128_pd($lhs);
        let rhs_pd = arch::_mm_castsi128_pd($rhs);
        let lo = arch::_mm_shuffle_pd(lhs_pd, rhs_pd, 0b00); // [a0,b0]
        let hi = arch::_mm_shuffle_pd(lhs_pd, rhs_pd, 0b11); // [a1,b1]
        arch::_mm_add_epi64(arch::_mm_castpd_si128(lo), arch::_mm_castpd_si128(hi))
    }}};
}

/// Reduces a 2-lane `i64` SIMD vector to a single `i64` value using the specified operation.
#[rustfmt::skip]
macro_rules! _mm_reduce_epi64_v1 {
    ($value:expr; $last:ident) => {{#[allow(unused_unsafe)] unsafe {
        let xmm0 = $value;
        // copy high 64-bits to low 64-bits
        let xmm1 = arch::_mm_shuffle_epi32(xmm0, 0b11_10_11_10);
        // final reduce and extract
        arch::_mm_cvtsi128_si64(arch::$last(xmm0, xmm1))
    }}};
}

/// Reduces a 4-lane `f32` SIMD vector to a single `f32` value using the specified operation.
/// SSE2 version of `_mm_reduce_ps_v2!` (`movehdup` is SSE3, so a shuffle is used instead).
#[rustfmt::skip]
macro_rules! _mm_reduce_ps_v1 {
    ($value:expr; $op:ident $last:ident) => {#[allow(unused_unsafe)] unsafe {
        let xmm0 = $value;
        // duplicate higher half to lower half
        let xmm1 = arch::_mm_shuffle_ps(xmm0, xmm0, 0b11_10_11_10);
        // first reduce
        let xmm0 = arch::$op(xmm0, xmm1);
        // Duplicate odd-indexed elements (1, 1, 3, 3)
        let xmm1 = arch::_mm_shuffle_ps(xmm0, xmm0, 0b11_11_01_01);
        // final reduce and extract
        arch::_mm_cvtss_f32(arch::$last(xmm0, xmm1))
    }};
}

/// Reduces a 4-lane `f32` SIMD vector using two operations simultaneously, returning both results.
/// SSE2 version of `_mm_reduce2_ps_v2!`.
#[rustfmt::skip]
macro_rules! _mm_reduce2_ps_v1 {
    ($value:expr; $op1:ident $last1:ident, $op2:ident $last2:ident) => {#[allow(unused_unsafe)] unsafe {
        let xmm0 = $value;
        let xmm1 = arch::_mm_shuffle_ps(xmm0, xmm0, 0b11_10_11_10);
        let xmm_a = arch::$op1(xmm0, xmm1);
        let xmm_b = arch::$op2(xmm0, xmm1);
        let xmm1 = arch::_mm_shuffle_ps(xmm_a, xmm_a, 0b11_11_01_01);
        let xmm2 = arch::_mm_shuffle_ps(xmm_b, xmm_b, 0b11_11_01_01);
        (
            arch::_mm_cvtss_f32(arch::$last1(xmm_a, xmm1)),
            arch::_mm_cvtss_f32(arch::$last2(xmm_b, xmm2)),
        )
    }};
}
