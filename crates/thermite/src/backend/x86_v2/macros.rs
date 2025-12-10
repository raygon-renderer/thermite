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
