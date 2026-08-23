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

/// Native two-register element align (`Register::align`) for any full-width
/// (128-bit) wasm register, via `i8x16.shuffle`.
///
/// wasm SIMD128 has no whole-register byte shift (no `pslldq` analogue), but
/// `i8x16_shuffle` is a two-source shuffle with compile-time byte indices where
/// index `>= 16` selects from the second operand - which is exactly an align:
/// the 16-byte window starting at byte `ob` of `concat(a, b)` is the index run
/// `ob..ob+16`. One instruction.
///
/// The match is keyed on the byte offset `ob = OFFSET * size_of::<Element>()`, so
/// this serves every wasm register width, not just the byte ones; `ob` const-folds
/// to exactly one arm. `ob > 16` means `OFFSET > LANES` (out of range) and falls
/// back to the generic `swizzle_const` default.
///
/// Without this, wasm had no native `align` on any register and fell through to
/// `swizzle_const` - two `i8x16_swizzle`s plus a bitselect.
#[rustfmt::skip]
macro_rules! impl_wasm_align_shuffle {
    () => {
        const HAS_NATIVE_ALIGN: bool = true;

        #[inline(always)]
        fn align<const OFFSET: usize>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
            match const { OFFSET * core::mem::size_of::<Self::Element>() } {
                0 => a,
                1  => arch::i8x16_shuffle::<1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16>(a, b),
                2  => arch::i8x16_shuffle::<2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17>(a, b),
                3  => arch::i8x16_shuffle::<3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18>(a, b),
                4  => arch::i8x16_shuffle::<4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19>(a, b),
                5  => arch::i8x16_shuffle::<5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20>(a, b),
                6  => arch::i8x16_shuffle::<6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21>(a, b),
                7  => arch::i8x16_shuffle::<7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22>(a, b),
                8  => arch::i8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23>(a, b),
                9  => arch::i8x16_shuffle::<9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24>(a, b),
                10 => arch::i8x16_shuffle::<10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25>(a, b),
                11 => arch::i8x16_shuffle::<11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26>(a, b),
                12 => arch::i8x16_shuffle::<12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27>(a, b),
                13 => arch::i8x16_shuffle::<13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28>(a, b),
                14 => arch::i8x16_shuffle::<14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29>(a, b),
                15 => arch::i8x16_shuffle::<15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30>(a, b),
                16 => b,
                _  => Self::swizzle_const::<$crate::swizzle::AlignIndices<OFFSET, Self::Lanes>>(a, b),
            }
        }
    };
}

/// Emit the wasm overrides of
/// [`Register::widen_index_bytes`](crate::register::Register::widen_index_bytes)
/// and [`Register::permutev_row`](crate::register::Register::permutev_row):
/// widen the register's `LANES`-byte index array into the unsigned index
/// register with the SIMD128 extend ladder (`u8 -> u16 -> u32 -> u64`), stopping
/// at the register's own lane width, since wasm has no single-step `pmovzxbd`
/// equivalent. The ladder already produces a `v128`, so nothing is stored back
/// out.
///
/// Shape tag is the lane count. Invoke inside the register's own `impl Register`
/// block, where `arch` is in scope. Each `@body` load is exactly `LANES` bytes
/// wide, since the argument is only that wide, while `@row` keeps the 8-byte
/// table row it is handed.
#[rustfmt::skip]
macro_rules! impl_widen_index_bytes_wasm {
    ($shape:ident) => {
        #[inline(always)]
        fn widen_index_bytes(
            bytes: &generic_array::GenericArray<u8, <Self as $crate::register::CoreRegister>::Lanes>,
        ) -> $crate::register::Storage<<Self as $crate::register::Register>::Unsigned> {
            unsafe { impl_widen_index_bytes_wasm!(@body bytes, $shape) }
        }

        // Straight from the byte row: no widen, no narrow, no clamp.
        #[inline(always)]
        fn permutev_row(
            value: $crate::register::Storage<Self>,
            row: &generic_array::GenericArray<u8, generic_array::typenum::U8>,
        ) -> $crate::register::Storage<Self> {
            arch::u8x16_relaxed_swizzle(value, unsafe { impl_widen_index_bytes_wasm!(@row row, $shape) })
        }
    };

    (@row $row:ident, x2) => { arch::wasm_lane_table_row::<2>($row.as_ptr()) };
    (@row $row:ident, x4) => { arch::wasm_lane_table_row::<4>($row.as_ptr()) };
    (@row $row:ident, x8) => { arch::wasm_lane_table_row::<8>($row.as_ptr()) };

    // 8 lanes of u16: one extend off the 8-byte index array.
    (@body $idxs:ident, x8) => {{
        arch::u16x8_extend_low_u8x16(arch::v128_load64_zero($idxs.as_ptr() as *const u64))
    }};

    // 4 lanes of u32: two rungs, off a 4-byte load.
    (@body $idxs:ident, x4) => {{
        let w16 = arch::u16x8_extend_low_u8x16(arch::v128_load32_zero($idxs.as_ptr() as *const u32));
        arch::u32x4_extend_low_u16x8(w16)
    }};

    // 2 lanes of u64: three rungs. Only two bytes are readable, so they ride in
    // on a `u32` splat, and the upper halves are dropped by the first extend.
    (@body $idxs:ident, x2) => {{
        let lo = arch::u32x4_splat(core::ptr::read_unaligned($idxs.as_ptr() as *const u16) as u32);
        let w16 = arch::u16x8_extend_low_u8x16(lo);
        arch::u64x2_extend_low_u32x4(arch::u32x4_extend_low_u16x8(w16))
    }};
}
