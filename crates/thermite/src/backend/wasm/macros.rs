/// Reduces a 4-lane `f32` SIMD vector to a single `f32` value using the specified operation.
#[rustfmt::skip]
macro_rules! reduce_32x4 {
    ($prefix:ident $value:expr; $op:ident $last:ident) => {{paste::paste! {
        let v0 = $value;

        // 1. Duplicate high half to lower half.
        // Intel: _mm_shuffle_ps(xmm0, xmm0, 0b11_10_11_10) -> Lanes [2, 3, 2, 3]
        let v1 = arch::u8x16_swizzle(v0, const { arch::x4indices(2, 3, 2, 3) });

        // First reduce (e.g., f32x4_add)
        let v0 = arch::$op(v0, v1);

        // 2. Duplicate odd-indexed elements.
        // Intel: _mm_movehdup_ps(xmm0) -> Lanes [1, 1, 3, 3]
        let v1 = arch::u8x16_swizzle(v0, const { arch::x4indices(1, 1, 3, 3) });

        // Final reduce and extract
        arch::[<$prefix 32x4_extract_lane>]::<0>(arch::$last(v0, v1))
    }}};
}

/// Reduces a 2-lane `f64` SIMD vector to a single `f64` value using the specified operation.
#[rustfmt::skip]
macro_rules! reduce_64x2 {
    ($prefix:ident $value:expr; $op:ident $last:ident) => {{paste::paste! {
        let v0 = $value;

        // Duplicate the high lane (Lane 1) into both slots.
        // Intel: _mm_movedup_pd(xmm0) -> Lanes [1, 1]
        // We use x2indices to select index 1 for both 64-bit lanes.
        let v1 = arch::u8x16_swizzle(v0, const { arch::x2indices(1, 1) });

        // Final reduce and extract.
        // v0: [A, B], v1: [B, B]
        // Result: [A op B, B op B] -> Extract Lane 0
        arch::[<$prefix 64x2_extract_lane>]::<0>(arch::$last(v0, v1))
    }}};
}
