/// Reduces a 4-lane `f32` SIMD vector to a single `f32` value using the specified operation.
#[rustfmt::skip]
macro_rules! _mm_reduce_ps {
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

/// Reduces 4-lane `f32` SIMD vector to a single `f32` value using the specified operation,
/// ignoring the last lane entirely.
#[rustfmt::skip]
macro_rules! _mm_reduce_ps3 {
    ($value:expr; $last:ident) => {#[allow(unused_unsafe)] unsafe {
        let xmm0 = $value;
        let xmm1 = arch::_mm_shuffle_ps(xmm0, xmm0, 0b00_00_00_01);
        let xmm2 = arch::_mm_shuffle_ps(xmm0, xmm0, 0b00_00_00_10);
        let xmm0 = arch::$last(xmm0, xmm1);
        arch::_mm_cvtss_f32(arch::$last(xmm0, xmm2))
    }};
}

/// Reduces a 2-lane `f64` SIMD vector to a single `f64` value using the specified operation.
#[rustfmt::skip]
macro_rules! _mm_reduce_pd {
    ($value:expr; $op:ident $last:ident) => {#[allow(unused_unsafe)] unsafe {
        let xmm0 = $value;
        // Duplicate odd-indexed elements (1, 1)
        let xmm1 = arch::_mm_movedup_pd(xmm0);
        // final reduce and extract
        arch::_mm_cvtsd_f64(arch::$last(xmm0, xmm1))
    }};
}

/// Reduces a 4-lane `i32` SIMD vector to a single `i32` value using the specified operation.
#[rustfmt::skip]
macro_rules! _mm_reduce_epi32 {
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
macro_rules! _mm_reduce_epi64 {
    ($value:expr; $last:ident) => {{#[allow(unused_unsafe)] unsafe {
        let xmm0 = $value;
        // copy high 64-bits to low 64-bits
        let xmm1 = arch::_mm_shuffle_epi32(xmm0, 0b11_10_11_10);
        // final reduce and extract
        arch::_mm_cvtsi128_si64(arch::$last(xmm0, xmm1))
    }}};
}

/// Reduces an 8-lane `f32` SIMD vector to a single `f32` value using the specified operation.
#[rustfmt::skip]
macro_rules! _mm256_reduce_ps {
    ($value:expr; $op:ident $last:ident) => {#[allow(unused_unsafe)] unsafe {
        let ymm0 = $value;
        let xmm0 = arch::_mm256_castps256_ps128(ymm0);
        let xmm1 = arch::_mm256_extractf128_ps(ymm0, 1);

        let xmm0 = arch::$op(xmm0, xmm1);

        _mm_reduce_ps!(xmm0; $op $last)
    }};
}

/// Reduces a 4-lane `f64` SIMD vector to a single `f64` value using the specified operation.
#[rustfmt::skip]
macro_rules! _mm256_reduce_pd {
    ($value:expr; $op:ident $last:ident) => {#[allow(unused_unsafe)] unsafe {
        let ymm0 = $value;
        let xmm0 = arch::_mm256_castpd256_pd128(ymm0);
        let xmm1 = arch::_mm256_extractf128_pd(ymm0, 1);

        let xmm0 = arch::$op(xmm0, xmm1);

        _mm_reduce_pd!(xmm0; $op $last)
    }};
}

/// Reduces a 4-lane `f64` SIMD vector to a single `f64` value using the specified operation,
/// ignoring the last lane entirely.
#[rustfmt::skip]
macro_rules! _mm256_reduce_pd3 {
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
macro_rules! _mm256_reduce_epi32 {
    ($value:expr; $op:ident $last:ident) => {{#[allow(unused_unsafe)] unsafe {
        let ymm0 = $value;
        let xmm0 = arch::_mm256_castsi256_si128(ymm0);
        let xmm1 = arch::_mm256_extractf128_si256(ymm0, 1);

        let xmm0 = arch::$op(xmm0, xmm1);

        _mm_reduce_epi32!(xmm0; $op $last)
    }}};
}

macro_rules! _mm256_reduce_epi64 {
    ($value:expr; $op:ident $last:ident) => {{#[allow(unused_unsafe)] unsafe {
        let ymm0 = $value;
        let xmm0 = arch::_mm256_castsi256_si128(ymm0);
        let xmm1 = arch::_mm256_extractf128_si256(ymm0, 1);

        let xmm0 = arch::$op(xmm0, xmm1);

        _mm_reduce_epi64!(xmm0; $last)
    }}};
}

macro_rules! impl_bit_casts {
    ($($from:ty as $to:ty => $conv:ident),* $(,)?) => {
        const _: () = {$(
            impl $crate::register::BitsRegister<$from> for $to {
                #[inline(always)]
                fn from_bits(value: <$from as $crate::register::Register>::Storage) -> Self::Storage {
                    unsafe { arch::$conv(value) }
                }
            }
        )*};
    };
}

macro_rules! impl_type_casts {
    ($($from:ty as $to:ty => $conv:ident $(| $fast_conv:ident)?),* $(,)?) => {
        const _: () = {$(
            impl $crate::register::CastRegister<$from> for $to {
                #[inline(always)]
                fn cast_from(value: <$from as $crate::register::Register>::Storage) -> Self::Storage {
                    unsafe { arch::$conv(value) }
                }

                $(
                    #[inline(always)]
                    fn fast_cast_from(value: <$from as $crate::register::Register>::Storage) -> Self::Storage {
                        unsafe { arch::$fast_conv(value) }
                    }
                )?
            }
        )*};
    };
}

macro_rules! impl_mask_casts {
    ($($from:ty as $to:ty => $conv:ident),* $(,)?) => {
        const _: () = {$(
            impl $crate::register::CastMaskRegister<$from> for $to {
                #[inline(always)]
                fn mask_from(value: <$from as $crate::register::Register>::Storage) -> Self::Storage {
                    unsafe { arch::$conv(value) }
                }
            }
        )*};
    };
}
