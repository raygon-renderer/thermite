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
