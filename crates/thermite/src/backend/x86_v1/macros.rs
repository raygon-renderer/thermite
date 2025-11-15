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
        // Duplicate odd-indexed elements (1, 1, 3, 3)
        let xmm1 = arch::_mm_shuffle_epi32(xmm0, 0b00_00_11_11);
        // final reduce and extract
        arch::_mm_cvtsi128_si32(arch::$last(xmm0, xmm1))
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
