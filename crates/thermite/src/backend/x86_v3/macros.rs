/// Reduces an 8-lane `f32` SIMD vector to a single `f32` value using the specified operation.
#[rustfmt::skip]
macro_rules! _mm256_reduce_ps_v3 {
    ($value:expr; $op:ident $last:ident) => {#[allow(unused_unsafe)] unsafe {
        let ymm0 = $value;
        let xmm0 = arch::_mm256_castps256_ps128(ymm0);
        let xmm1 = arch::_mm256_extractf128_ps(ymm0, 1);

        let xmm0 = arch::$op(xmm0, xmm1);

        _mm_reduce_ps_v2!(xmm0; $op $last)
    }};
}

/// Reduces a 4-lane `f64` SIMD vector to a single `f64` value using the specified operation.
#[rustfmt::skip]
macro_rules! _mm256_reduce_pd_v3 {
    ($value:expr; $op:ident $last:ident) => {#[allow(unused_unsafe)] unsafe {
        let ymm0 = $value;
        let xmm0 = arch::_mm256_castpd256_pd128(ymm0);
        let xmm1 = arch::_mm256_extractf128_pd(ymm0, 1);

        let xmm0 = arch::$op(xmm0, xmm1);

        _mm_reduce_pd_v1!(xmm0; $op $last)
    }};
}

/// Reduces a 4-lane `f64` SIMD vector to a single `f64` value using the specified operation,
/// ignoring the last lane entirely.
#[rustfmt::skip]
macro_rules! _mm256_reduce_pd3_v3 {
    ($value:expr; $last:ident) => {#[allow(unused_unsafe)] unsafe {
        let ymm0 = $value;

        // [0, 1]
        let xmm0 = arch::_mm256_castpd256_pd128(ymm0);
        // [1, 0]
        let xmm1 = arch::_mm256_castpd256_pd128(arch::_mm256_permute4x64_pd(ymm0, 0b00_00_00_01));
        let xmm1 = arch::$last(xmm0, xmm1); // [0 + 1, 1]

        // [2, 0]
        let xmm2 = arch::_mm256_castpd256_pd128(arch::_mm256_permute4x64_pd(ymm0, 0b00_00_00_10));

        // [0 + 1 + 2, 1]
        arch::_mm_cvtsd_f64(arch::$last(xmm1, xmm2))

    }};
}

/// Reduces an 8-lane `i32` SIMD vector to a single `i32` value using the specified operation.
#[rustfmt::skip]
macro_rules! _mm256_reduce_epi32_v3 {
    ($value:expr; $op:ident $last:ident) => {{#[allow(unused_unsafe)] unsafe {
        let ymm0 = $value;
        let xmm0 = arch::_mm256_castsi256_si128(ymm0);
        let xmm1 = arch::_mm256_extractf128_si256(ymm0, 1);

        let xmm0 = arch::$op(xmm0, xmm1);

        _mm_reduce_epi32_v1!(xmm0; $op $last)
    }}};
}

macro_rules! _mm256_reduce_epi64_v3 {
    ($value:expr; $op:ident $last:ident) => {{#[allow(unused_unsafe)] unsafe {
        let ymm0 = $value;
        let xmm0 = arch::_mm256_castsi256_si128(ymm0);
        let xmm1 = arch::_mm256_extractf128_si256(ymm0, 1);

        let xmm0 = arch::$op(xmm0, xmm1);

        _mm_reduce_epi64_v1!(xmm0; $last)
    }}};
}
