/// Attach the generic-default `PackedFloatRegister` impls for the two fp8 formats
/// ([`Fp8E4M3`](crate::element::float::spec::Fp8E4M3) /
/// [`Fp8E5M2`](crate::element::float::spec::Fp8E5M2)) on one or more `(u8 register, f32 register)`
/// pairs. No hardware transcodes fp8, so every backend just takes the branchless defaults; this
/// macro is the per-backend wiring (invoked from each `registers/mod.rs`).
macro_rules! impl_packed_fp8 {
    ($($u8:ty => $f32:ty),* $(,)?) => {$(
        impl $crate::register::PackedFloatRegister<$crate::element::float::spec::Fp8E4M3, $f32> for $u8 {}
        impl $crate::register::PackedFloatRegister<$crate::element::float::spec::Fp8E5M2, $f32> for $u8 {}
    )*};
}

/// Native two-register byte align (`IntegerRegister::align`) for a 128-bit byte
/// register, via `_mm_alignr_epi8` (SSSE3+). Drop into an `impl IntegerRegister`
/// block for an i8x16/u8x16-shaped register.
///
/// `align::<OFFSET>(a, b)` is the 16-byte window at byte `OFFSET` of the
/// concatenation `[a, b]` with `a` as the low half. `_mm_alignr_epi8::<n>(hi, lo)`
/// yields `concat(lo:hi)[n..]`, so we pass `(b, a)`. The immediate cannot be a
/// const expression of `OFFSET` on stable, hence the (verbose) match supplying
/// each literal; `OFFSET > 16` falls back to the generic `swizzle_const` default.
macro_rules! impl_byte_align_alignr {
    () => {
        fn align<const OFFSET: usize>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
            match OFFSET {
                0 => unsafe { arch::_mm_alignr_epi8::<0>(b, a) },
                1 => unsafe { arch::_mm_alignr_epi8::<1>(b, a) },
                2 => unsafe { arch::_mm_alignr_epi8::<2>(b, a) },
                3 => unsafe { arch::_mm_alignr_epi8::<3>(b, a) },
                4 => unsafe { arch::_mm_alignr_epi8::<4>(b, a) },
                5 => unsafe { arch::_mm_alignr_epi8::<5>(b, a) },
                6 => unsafe { arch::_mm_alignr_epi8::<6>(b, a) },
                7 => unsafe { arch::_mm_alignr_epi8::<7>(b, a) },
                8 => unsafe { arch::_mm_alignr_epi8::<8>(b, a) },
                9 => unsafe { arch::_mm_alignr_epi8::<9>(b, a) },
                10 => unsafe { arch::_mm_alignr_epi8::<10>(b, a) },
                11 => unsafe { arch::_mm_alignr_epi8::<11>(b, a) },
                12 => unsafe { arch::_mm_alignr_epi8::<12>(b, a) },
                13 => unsafe { arch::_mm_alignr_epi8::<13>(b, a) },
                14 => unsafe { arch::_mm_alignr_epi8::<14>(b, a) },
                15 => unsafe { arch::_mm_alignr_epi8::<15>(b, a) },
                16 => unsafe { arch::_mm_alignr_epi8::<16>(b, a) },
                _ => Self::swizzle_const::<$crate::swizzle::AlignIndices<OFFSET, Self::Lanes>>(a, b),
            }
        }
    };
}

/// Native two-register element align for a 256-bit register (AVX2). Drop into an
/// `impl IntegerRegister` block for an i8x32/i16x16/i32x8/i64x4-shaped register.
///
/// A full 256-bit align is two instructions: `mid = permute2x128(a, b, 0x21)`
/// (`[a.hi, b.lo]`), then a single `_mm256_alignr_epi8` (which aligns per
/// 128-bit lane). For byte offset `ob = OFFSET * size_of::<Element>()` in
/// `0..=16` the window is `alignr::<ob>(mid, a)`; in `16..=32` it is
/// `alignr::<ob-16>(b, mid)`. The match is keyed on `ob` so the immediate is a
/// literal (stable rejects a const expr of `OFFSET`); `ob` const-folds to one arm.
macro_rules! impl_byte_align_alignr256 {
    () => {
        fn align<const OFFSET: usize>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
            let mid = unsafe { arch::_mm256_permute2x128_si256::<0x21>(a, b) };
            match OFFSET * core::mem::size_of::<Self::Element>() {
                0 => unsafe { arch::_mm256_alignr_epi8::<0>(mid, a) },
                1 => unsafe { arch::_mm256_alignr_epi8::<1>(mid, a) },
                2 => unsafe { arch::_mm256_alignr_epi8::<2>(mid, a) },
                3 => unsafe { arch::_mm256_alignr_epi8::<3>(mid, a) },
                4 => unsafe { arch::_mm256_alignr_epi8::<4>(mid, a) },
                5 => unsafe { arch::_mm256_alignr_epi8::<5>(mid, a) },
                6 => unsafe { arch::_mm256_alignr_epi8::<6>(mid, a) },
                7 => unsafe { arch::_mm256_alignr_epi8::<7>(mid, a) },
                8 => unsafe { arch::_mm256_alignr_epi8::<8>(mid, a) },
                9 => unsafe { arch::_mm256_alignr_epi8::<9>(mid, a) },
                10 => unsafe { arch::_mm256_alignr_epi8::<10>(mid, a) },
                11 => unsafe { arch::_mm256_alignr_epi8::<11>(mid, a) },
                12 => unsafe { arch::_mm256_alignr_epi8::<12>(mid, a) },
                13 => unsafe { arch::_mm256_alignr_epi8::<13>(mid, a) },
                14 => unsafe { arch::_mm256_alignr_epi8::<14>(mid, a) },
                15 => unsafe { arch::_mm256_alignr_epi8::<15>(mid, a) },
                16 => unsafe { arch::_mm256_alignr_epi8::<16>(mid, a) },
                17 => unsafe { arch::_mm256_alignr_epi8::<1>(b, mid) },
                18 => unsafe { arch::_mm256_alignr_epi8::<2>(b, mid) },
                19 => unsafe { arch::_mm256_alignr_epi8::<3>(b, mid) },
                20 => unsafe { arch::_mm256_alignr_epi8::<4>(b, mid) },
                21 => unsafe { arch::_mm256_alignr_epi8::<5>(b, mid) },
                22 => unsafe { arch::_mm256_alignr_epi8::<6>(b, mid) },
                23 => unsafe { arch::_mm256_alignr_epi8::<7>(b, mid) },
                24 => unsafe { arch::_mm256_alignr_epi8::<8>(b, mid) },
                25 => unsafe { arch::_mm256_alignr_epi8::<9>(b, mid) },
                26 => unsafe { arch::_mm256_alignr_epi8::<10>(b, mid) },
                27 => unsafe { arch::_mm256_alignr_epi8::<11>(b, mid) },
                28 => unsafe { arch::_mm256_alignr_epi8::<12>(b, mid) },
                29 => unsafe { arch::_mm256_alignr_epi8::<13>(b, mid) },
                30 => unsafe { arch::_mm256_alignr_epi8::<14>(b, mid) },
                31 => unsafe { arch::_mm256_alignr_epi8::<15>(b, mid) },
                32 => unsafe { arch::_mm256_alignr_epi8::<16>(b, mid) },
                _ => Self::swizzle_const::<$crate::swizzle::AlignIndices<OFFSET, Self::Lanes>>(a, b),
            }
        }
    };
}

/// Whole-register byte-shift `align` override (`Register::align`) for a full,
/// unpadded 128-bit integer register with native full-width byte shifts
/// (`bshli`/`bshri` = `pslldq`/`psrldq`), e.g. SSE2 where there is no `palignr`.
/// Drop into an `impl Register` block.
///
/// `align::<OFFSET>(a, b)` is `(a >> ob) | (b << (16 - ob))` in bytes, where
/// `ob = OFFSET * size_of::<Element>()`. The match is keyed on `ob` so each
/// arm's shift counts are literals (stable rejects a const expr of `OFFSET` in
/// const-generic position); `ob` const-folds to one arm. `ob > 16` means
/// `OFFSET > LANES` (out of range) and falls back to the generic default.
///
/// NOTE: 128-bit only - AVX2 `_mm256_bslli/bsrli_epi128` shift per 128-bit lane.
macro_rules! impl_byteshift_align {
    () => {
        fn align<const OFFSET: usize>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
            match const { OFFSET * core::mem::size_of::<Self::Element>() } {
                0 => Self::bitor(Self::bshri::<0>(a), Self::bshli::<16>(b)),
                1 => Self::bitor(Self::bshri::<1>(a), Self::bshli::<15>(b)),
                2 => Self::bitor(Self::bshri::<2>(a), Self::bshli::<14>(b)),
                3 => Self::bitor(Self::bshri::<3>(a), Self::bshli::<13>(b)),
                4 => Self::bitor(Self::bshri::<4>(a), Self::bshli::<12>(b)),
                5 => Self::bitor(Self::bshri::<5>(a), Self::bshli::<11>(b)),
                6 => Self::bitor(Self::bshri::<6>(a), Self::bshli::<10>(b)),
                7 => Self::bitor(Self::bshri::<7>(a), Self::bshli::<9>(b)),
                8 => Self::bitor(Self::bshri::<8>(a), Self::bshli::<8>(b)),
                9 => Self::bitor(Self::bshri::<9>(a), Self::bshli::<7>(b)),
                10 => Self::bitor(Self::bshri::<10>(a), Self::bshli::<6>(b)),
                11 => Self::bitor(Self::bshri::<11>(a), Self::bshli::<5>(b)),
                12 => Self::bitor(Self::bshri::<12>(a), Self::bshli::<4>(b)),
                13 => Self::bitor(Self::bshri::<13>(a), Self::bshli::<3>(b)),
                14 => Self::bitor(Self::bshri::<14>(a), Self::bshli::<2>(b)),
                15 => Self::bitor(Self::bshri::<15>(a), Self::bshli::<1>(b)),
                16 => Self::bitor(Self::bshri::<16>(a), Self::bshli::<0>(b)),
                _ => Self::swizzle_const::<$crate::swizzle::AlignIndices<OFFSET, Self::Lanes>>(a, b),
            }
        }
    };
}

macro_rules! impl_bit_casts {
    ($($from:ty as $to:ty => $conv:ident),* $(,)?) => {
        const _: () = {$(
            #[thermite_macros::inline_always]
            impl $crate::register::BitCastRegister<$from> for $to {
                fn from_bits(value: Storage<$from>) -> Storage<Self> {
                    unsafe { arch::$conv(value) }
                }
            }
        )*};
    };
}

macro_rules! impl_type_casts {
    ($($from:ty as $to:ty => $conv:ident $(| $fast_conv:ident)?),* $(,)?) => {
        const _: () = {$(
            #[thermite_macros::inline_always]
            impl $crate::register::CastRegister<$from> for $to {
                fn cast_from(value: Storage<$from>) -> Storage<Self> {
                    unsafe { arch::$conv(value) }
                }

                $(
                    fn fast_cast_from(value: Storage<$from>) -> Storage<Self> {
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
            #[thermite_macros::inline_always]
            impl $crate::register::CastMaskRegister<$from> for $to {
                fn mask_from(value: Storage<$from>) -> Storage<Self> {
                    unsafe { arch::$conv(value) }
                }
            }
        )*};
    };
}

macro_rules! impl_concat_bool_register2 {
    ($e:ty, $r:ty) => {
        const _: () = {
            use $crate::element::MaskElement;

            #[thermite_macros::inline_always]
            impl $crate::register::ConcatRegister<bool> for $r {
                fn concat(lo: Storage<bool>, hi: Storage<bool>) -> Storage<Self> {
                    <Self as $crate::register::ConcatRegister<$e>>::concat(<$e>::from_bool(lo), <$e>::from_bool(hi))
                }

                fn split(value: Storage<Self>) -> (Storage<bool>, Storage<bool>) {
                    let (lo, hi) = <Self as $crate::register::ConcatRegister<$e>>::split(value);
                    (lo.to_bool(), hi.to_bool())
                }
            }

            #[thermite_macros::inline_always]
            impl $crate::register::ExtendRegister<bool> for $r {
                fn extend(value: Storage<bool>) -> Storage<Self> {
                    <Self as $crate::register::ExtendRegister<$e>>::extend(<$e>::from_bool(value))
                }

                fn narrow(value: Storage<Self>) -> Storage<bool> {
                    <Self as $crate::register::ExtendRegister<$e>>::narrow(value).to_bool()
                }
            }
        };
    };
}

macro_rules! impl_newregister {
    ($($r:ty),*) => {$(
        impl $crate::register::NewRegister<
            <$r as $crate::register::Register>::Element,
            <$r as $crate::register::CoreRegister>::Lanes,
            <$r as $crate::register::CoreRegister>::Storage
        > for $r {
            type New<T: $crate::vector::splat::NewConst<
                <$r as $crate::register::Register>::Element,
                <$r as $crate::register::CoreRegister>::Lanes>
            > = Self;
        }

        impl<T> $crate::vector::splat::VectorValue<T, Storage<Self>> for $r
        where
            T: $crate::vector::splat::NewConst<
                <$r as $crate::register::Register>::Element,
                <$r as $crate::register::CoreRegister>::Lanes
            >,
        {
            const VALUE: Storage<Self> = const { unsafe { $crate::generic_array::const_transmute(T::VALUES) } };
        }
    )*};
}

/// Add `Register::compress` + `compress_z` overrides to a register `impl` block
/// that delegate to the `<= 8`-lane table polyfill ([`compress_permute`]).
/// Invoke inside `impl Register for <Reg> { ... }` for any `Register` with at
/// most 8 lanes.
macro_rules! compress_via_table {
    () => {
        #[inline(always)]
        fn compress(
            value: $crate::register::Storage<Self>,
            mask: $crate::register::Storage<<Self as $crate::register::CoreRegister>::Mask>,
        ) -> $crate::register::Storage<Self> {
            $crate::backend::generic::polyfills::compress_permute::<Self>(value, mask)
        }

        #[inline(always)]
        fn compress_z(
            value: $crate::register::Storage<Self>,
            mask: $crate::register::Storage<<Self as $crate::register::CoreRegister>::Mask>,
        ) -> $crate::register::Storage<Self> {
            // Zero the unselected lanes, then compact: they carry into the tail.
            $crate::backend::generic::polyfills::compress_permute::<Self>(
                <Self as $crate::register::CoreRegister>::zz(mask, value),
                mask,
            )
        }
    };
}

/// Add `Register::compress` + `compress_z` overrides that delegate to the wide
/// polyfill ([`compress_permute_wide`]). Invoke inside `impl Register for <Reg>
/// { ... }` for any `Register` whose lane count is a multiple of 8 in `8..=64`
/// (the 16/32-lane byte and short vectors).
macro_rules! compress_via_wide {
    () => {
        #[inline(always)]
        fn compress(
            value: $crate::register::Storage<Self>,
            mask: $crate::register::Storage<<Self as $crate::register::CoreRegister>::Mask>,
        ) -> $crate::register::Storage<Self> {
            $crate::backend::generic::polyfills::compress_permute_wide::<Self>(value, mask)
        }

        #[inline(always)]
        fn compress_z(
            value: $crate::register::Storage<Self>,
            mask: $crate::register::Storage<<Self as $crate::register::CoreRegister>::Mask>,
        ) -> $crate::register::Storage<Self> {
            $crate::backend::generic::polyfills::compress_permute_wide::<Self>(
                <Self as $crate::register::CoreRegister>::zz(mask, value),
                mask,
            )
        }
    };
}
