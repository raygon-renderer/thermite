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

/// Stamp [`BitCastRegister`](crate::register::BitCastRegister) between two registers of
/// the same total width whose `Storage` types differ, via a by-value transmute. This is
/// the `ArrayRegister` counterpart to [`impl_bit_casts_identity`]: the scalar backend's
/// registers are `ArrayRegister<u8, 16>` / `ArrayRegister<u64, 2>` etc., which are
/// distinct types rather than one shared intrinsic, and the element-wise array bitcast
/// (`register/array.rs`) only relates arrays with the SAME lane count.
///
/// The transmute is by value, so the differing alignments of `[u8; 16]` and `[u64; 2]`
/// are irrelevant (alignment constrains references, not value copies). Byte order within
/// the wider lane is target-endian, which is immaterial to the SAD family: the SWAR
/// cascade sums all bytes of a lane regardless of their position within it.
macro_rules! impl_bit_casts_transmute {
    ($($from:ty as $to:ty),* $(,)?) => {
        const _: () = {$(
            #[thermite_macros::inline_always]
            impl $crate::register::BitCastRegister<$from> for $to {
                fn from_bits(value: $crate::register::Storage<$from>) -> $crate::register::Storage<Self> {
                    unsafe { $crate::generic_array::const_transmute(value) }
                }
            }
        )*};
    };
}

/// Attach the generic-default SAD impls ([`Sad16Register`](crate::register::Sad16Register)
/// / [`Sad32Register`](crate::register::Sad32Register) /
/// [`Sad64Register`](crate::register::Sad64Register)) on one or more
/// `u8 register => (u16, u32, u64 register)` groups. The three output registers must have
/// the same total width as the `u8` register - that shape is what the traits rely on
/// instead of a lane-count bound. Backends override individual methods where hardware
/// wins (x86 `psadbw`, NEON `vpaddl`, wasm `extadd_pairwise`); this macro is the wiring
/// that opts every register into the SWAR defaults.
macro_rules! impl_sad {
    // Single-grouping arms, reused by the other SAD macros. The SWAR cascade needs the
    // output register to be a same-width reinterpret of the byte register, which holds
    // whenever the group divides the lane count evenly.
    (@swar16 $u8:ty => $u16:ty) => {
        #[thermite_macros::inline_always]
        impl $crate::register::Sad16Register<$u16> for $u8 {
            fn sad16(
                a: $crate::register::Storage<Self>,
                b: $crate::register::Storage<Self>,
            ) -> $crate::register::Storage<$u16> {
                $crate::register::sad_cascade_u8_16::<Self, $u16>(
                    <Self as $crate::register::UnsignedIntegerRegister>::abs_diff(a, b),
                )
            }
        }
    };
    (@swar32 $u8:ty => $u32:ty) => {
        #[thermite_macros::inline_always]
        impl $crate::register::Sad32Register<$u32> for $u8 {
            fn sad32(
                a: $crate::register::Storage<Self>,
                b: $crate::register::Storage<Self>,
            ) -> $crate::register::Storage<$u32> {
                $crate::register::sad_cascade_u8_32::<Self, $u32>(
                    <Self as $crate::register::UnsignedIntegerRegister>::abs_diff(a, b),
                )
            }
        }
    };
    (@swar64 $u8:ty => $u64:ty) => {
        #[thermite_macros::inline_always]
        impl $crate::register::Sad64Register<$u64> for $u8 {
            fn sad64(
                a: $crate::register::Storage<Self>,
                b: $crate::register::Storage<Self>,
            ) -> $crate::register::Storage<$u64> {
                $crate::register::sad_cascade_u8_64::<Self, $u64>(
                    <Self as $crate::register::UnsignedIntegerRegister>::abs_diff(a, b),
                )
            }
        }
    };

    ($($u8:ty => ($u16:ty, $u32:ty, $u64:ty)),* $(,)?) => {$(
        impl_sad!(@swar16 $u8 => $u16);
        impl_sad!(@swar32 $u8 => $u32);
        impl_sad!(@swar64 $u8 => $u64);
    )*};
}

/// SAD on `u16` inputs: `sad32` sums adjacent lane pairs, `sad64` groups of four. Same
/// same-width-reinterpret shape as the `u8` family, one element size up. `$swar` selects
/// the SWAR cascade (full-width registers) or the lane-wise path (sub-native ones).
macro_rules! impl_sad_u16 {
    (@swar $($u16:ty => ($u32:ty, $u64:ty)),* $(,)?) => {$(
        impl_sad_u16!(@body $u16 => ($u32, $u64),
            |a, b| $crate::register::sad_cascade_u16_32::<Self, $u32>(
                <Self as $crate::register::UnsignedIntegerRegister>::abs_diff(a, b)),
            |a, b| $crate::register::sad_cascade_u16_64::<Self, $u64>(
                <Self as $crate::register::UnsignedIntegerRegister>::abs_diff(a, b)));
    )*};
    (@scalar $($u16:ty => ($u32:ty, $u64:ty)),* $(,)?) => {$(
        impl_sad_u16!(@body $u16 => ($u32, $u64),
            |a, b| $crate::register::sad_scalar_u16_32::<Self, $u32>(a, b),
            |a, b| $crate::register::sad_scalar_u16_64::<Self, $u64>(a, b));
    )*};
    (@body $u16:ty => ($u32:ty, $u64:ty), |$a32:ident, $b32:ident| $e32:expr, |$a64:ident, $b64:ident| $e64:expr) => {
        #[thermite_macros::inline_always]
        impl $crate::register::Sad32Register<$u32> for $u16 {
            fn sad32(
                $a32: $crate::register::Storage<Self>,
                $b32: $crate::register::Storage<Self>,
            ) -> $crate::register::Storage<$u32> {
                $e32
            }
        }

        #[thermite_macros::inline_always]
        impl $crate::register::Sad64Register<$u64> for $u16 {
            fn sad64(
                $a64: $crate::register::Storage<Self>,
                $b64: $crate::register::Storage<Self>,
            ) -> $crate::register::Storage<$u64> {
                $e64
            }
        }
    };
}

/// SAD on `u32` inputs: `sad64` sums adjacent lane pairs.
macro_rules! impl_sad_u32 {
    (@swar $($u32:ty => $u64:ty),* $(,)?) => {$(
        impl_sad_u32!(@body $u32 => $u64,
            |a, b| $crate::register::sad_cascade_u32_64::<Self, $u64>(
                <Self as $crate::register::UnsignedIntegerRegister>::abs_diff(a, b)));
    )*};
    (@scalar $($u32:ty => $u64:ty),* $(,)?) => {$(
        impl_sad_u32!(@body $u32 => $u64, |a, b| $crate::register::sad_scalar_u32_64::<Self, $u64>(a, b));
    )*};
    (@body $u32:ty => $u64:ty, |$a:ident, $b:ident| $e:expr) => {
        #[thermite_macros::inline_always]
        impl $crate::register::Sad64Register<$u64> for $u32 {
            fn sad64(
                $a: $crate::register::Storage<Self>,
                $b: $crate::register::Storage<Self>,
            ) -> $crate::register::Storage<$u64> {
                $e
            }
        }
    };
}

/// Lane-wise SAD for the sub-native (`< 128`-bit) `u8` ladder. Below a full register there
/// is no SIMD win, so `ReducedRegister`/`ArrayRegister` byte registers just sum their lanes
/// (see `sad_scalar_*` in `register/mod.rs`). Where the register holds fewer bytes than one
/// group, the single output lane sums the whole register.
macro_rules! impl_sad_scalar {
    ($($u8:ty => ($u16:ty, $u32:ty, $u64:ty)),* $(,)?) => {$(
        #[thermite_macros::inline_always]
        impl $crate::register::Sad16Register<$u16> for $u8 {
            fn sad16(
                a: $crate::register::Storage<Self>,
                b: $crate::register::Storage<Self>,
            ) -> $crate::register::Storage<$u16> {
                $crate::register::sad_scalar_u8_16::<Self, $u16>(a, b)
            }
        }

        #[thermite_macros::inline_always]
        impl $crate::register::Sad32Register<$u32> for $u8 {
            fn sad32(
                a: $crate::register::Storage<Self>,
                b: $crate::register::Storage<Self>,
            ) -> $crate::register::Storage<$u32> {
                $crate::register::sad_scalar_u8_32::<Self, $u32>(a, b)
            }
        }

        #[thermite_macros::inline_always]
        impl $crate::register::Sad64Register<$u64> for $u8 {
            fn sad64(
                a: $crate::register::Storage<Self>,
                b: $crate::register::Storage<Self>,
            ) -> $crate::register::Storage<$u64> {
                $crate::register::sad_scalar_u8_64::<Self, $u64>(a, b)
            }
        }
    )*};
}

/// As [`impl_sad`], but overriding the `u64` grouping with a native sum-of-absolute-
/// differences instruction (x86 `PSADBW` via `_mm_sad_epu8` / `_mm256_sad_epu8`), which
/// does the absolute difference AND the 8-byte horizontal sum in one op. Available on
/// every x86 tier - `psadbw` is SSE2 - so all three take this path. The 16/32 groupings
/// have no single-instruction x86 form and keep the SWAR defaults.
macro_rules! impl_sad_native_u64 {
    // SSSE3+ variant: the narrow groupings also get native forms. `pmaddubsw` against a
    // vector of ones sums adjacent byte pairs into `u16` (max 510, so the instruction's
    // saturation never triggers), and `pmaddwd` against ones sums adjacent `u16` pairs
    // into `u32` - the 2- and 4-byte groupings in one instruction each past `abs_diff`.
    (@ssse3 $($u8:ty => ($u16:ty, $u32:ty, $u64:ty) via $sad:ident),* $(,)?) => {$(
        #[thermite_macros::inline_always]
        impl $crate::register::Sad16Register<$u16> for $u8 {
            fn sad16(
                a: $crate::register::Storage<Self>,
                b: $crate::register::Storage<Self>,
            ) -> $crate::register::Storage<$u16> {
                unsafe {
                    arch::_mm_maddubs_epi16(
                        <Self as $crate::register::UnsignedIntegerRegister>::abs_diff(a, b),
                        arch::_mm_set1_epi8(1),
                    )
                }
            }
        }

        #[thermite_macros::inline_always]
        impl $crate::register::Sad32Register<$u32> for $u8 {
            fn sad32(
                a: $crate::register::Storage<Self>,
                b: $crate::register::Storage<Self>,
            ) -> $crate::register::Storage<$u32> {
                unsafe {
                    arch::_mm_madd_epi16(
                        <Self as $crate::register::Sad16Register<$u16>>::sad16(a, b),
                        arch::_mm_set1_epi16(1),
                    )
                }
            }
        }

        #[thermite_macros::inline_always]
        impl $crate::register::Sad64Register<$u64> for $u8 {
            fn sad64(
                a: $crate::register::Storage<Self>,
                b: $crate::register::Storage<Self>,
            ) -> $crate::register::Storage<$u64> {
                unsafe { arch::$sad(a, b) }
            }
        }
    )*};

    ($($u8:ty => ($u16:ty, $u32:ty, $u64:ty) via $sad:ident),* $(,)?) => {$(
        impl_sad!(@swar16 $u8 => $u16);
        impl_sad!(@swar32 $u8 => $u32);

        #[thermite_macros::inline_always]
        impl $crate::register::Sad64Register<$u64> for $u8 {
            fn sad64(
                a: $crate::register::Storage<Self>,
                b: $crate::register::Storage<Self>,
            ) -> $crate::register::Storage<$u64> {
                unsafe { arch::$sad(a, b) }
            }
        }
    )*};
}

/// Stamp `ExtendRegister<$elem>` (extend-from-scalar) on one or more concrete
/// native register types. Zero-extend places the scalar in lane 0 and zeros the
/// rest -- exactly `Register::single` -- and narrow reads lane 0 back with
/// `extract::<0>`. This is what satisfies the `Register: ExtendRegister<Self::Element>`
/// supertrait for every native (non-emulated) register width. Invoked from each
/// backend's `registers/mod.rs`, once per `(register, element)` pair.
///
/// Float masks are `Mask = Self`, so a float register's mask-extend obligation is
/// discharged by this same impl; integer/usize masks are handled where they differ.
macro_rules! impl_native_extend_from_scalar {
    ($($reg:ty => $elem:ty),* $(,)?) => {$(
        #[thermite_macros::inline_always]
        impl $crate::register::ExtendRegister<$elem> for $reg {
            fn extend(value: $crate::register::Storage<$elem>) -> $crate::register::Storage<Self> {
                <Self as $crate::register::Register>::single(value)
            }

            fn narrow(value: $crate::register::Storage<Self>) -> $crate::register::Storage<$elem> {
                <Self as $crate::register::Register>::extract::<0>(value)
            }
        }
    )*};
}

/// Emit an optimal `Register::extract<const I>` method body for an x86 register,
/// overriding the generic `as_slice[I]` default. Placed INSIDE the `impl Register`
/// block (like `impl_native_radix3!`). Intrinsic paths are absolute
/// (`core::arch::x86_64::*`) - a `:path` fragment can't take a `::<I>` turbofish,
/// and definition-site hygiene would break an unqualified `arch::`. The `as _`
/// casts adapt the intrinsic's `i32`/`i64` result to signed OR unsigned
/// `Self::Element`, so one tag serves both. Shapes are named by element/width:
/// `epi8x16`/`epi16x8`/`epi32x4`/`epi64x2` (128-bit int), their `x32`/`x16`/`x8`/`x4`
/// 256-bit counterparts, and `ps128`/`ps256`/`pd128`/`pd256` for floats.
///
/// SSE4.1 introduced most of these intrinsics; SSE2 (x86_v1) uses the `_v1`-suffixed
/// shapes below, which reach lane 0 with a `cvt` and any other lane with one shuffle.
macro_rules! impl_native_extract {
    // ===== 128-bit integer =====
    (@epi64x2) => {
        fn extract<const I: usize>(value: $crate::register::Storage<Self>) -> Self::Element {
            const {
                assert!(I < 2, "Index out of bounds for register lane extraction");
            }
            unsafe {
                match I {
                    0 => core::arch::x86_64::_mm_extract_epi64::<0>(value) as _,
                    _ => core::arch::x86_64::_mm_extract_epi64::<1>(value) as _,
                }
            }
        }
    };
    (@epi32x4) => {
        fn extract<const I: usize>(value: $crate::register::Storage<Self>) -> Self::Element {
            const {
                assert!(I < 4, "Index out of bounds for register lane extraction");
            }
            unsafe {
                match I {
                    0 => core::arch::x86_64::_mm_extract_epi32::<0>(value) as _,
                    1 => core::arch::x86_64::_mm_extract_epi32::<1>(value) as _,
                    2 => core::arch::x86_64::_mm_extract_epi32::<2>(value) as _,
                    _ => core::arch::x86_64::_mm_extract_epi32::<3>(value) as _,
                }
            }
        }
    };
    (@epi16x8) => {
        fn extract<const I: usize>(value: $crate::register::Storage<Self>) -> Self::Element {
            const {
                assert!(I < 8, "Index out of bounds for register lane extraction");
            }
            unsafe {
                match I {
                    0 => core::arch::x86_64::_mm_extract_epi16::<0>(value) as _,
                    1 => core::arch::x86_64::_mm_extract_epi16::<1>(value) as _,
                    2 => core::arch::x86_64::_mm_extract_epi16::<2>(value) as _,
                    3 => core::arch::x86_64::_mm_extract_epi16::<3>(value) as _,
                    4 => core::arch::x86_64::_mm_extract_epi16::<4>(value) as _,
                    5 => core::arch::x86_64::_mm_extract_epi16::<5>(value) as _,
                    6 => core::arch::x86_64::_mm_extract_epi16::<6>(value) as _,
                    _ => core::arch::x86_64::_mm_extract_epi16::<7>(value) as _,
                }
            }
        }
    };
    (@epi8x16) => {
        fn extract<const I: usize>(value: $crate::register::Storage<Self>) -> Self::Element {
            const {
                assert!(I < 16, "Index out of bounds for register lane extraction");
            }
            unsafe {
                match I {
                    0 => core::arch::x86_64::_mm_extract_epi8::<0>(value) as _,
                    1 => core::arch::x86_64::_mm_extract_epi8::<1>(value) as _,
                    2 => core::arch::x86_64::_mm_extract_epi8::<2>(value) as _,
                    3 => core::arch::x86_64::_mm_extract_epi8::<3>(value) as _,
                    4 => core::arch::x86_64::_mm_extract_epi8::<4>(value) as _,
                    5 => core::arch::x86_64::_mm_extract_epi8::<5>(value) as _,
                    6 => core::arch::x86_64::_mm_extract_epi8::<6>(value) as _,
                    7 => core::arch::x86_64::_mm_extract_epi8::<7>(value) as _,
                    8 => core::arch::x86_64::_mm_extract_epi8::<8>(value) as _,
                    9 => core::arch::x86_64::_mm_extract_epi8::<9>(value) as _,
                    10 => core::arch::x86_64::_mm_extract_epi8::<10>(value) as _,
                    11 => core::arch::x86_64::_mm_extract_epi8::<11>(value) as _,
                    12 => core::arch::x86_64::_mm_extract_epi8::<12>(value) as _,
                    13 => core::arch::x86_64::_mm_extract_epi8::<13>(value) as _,
                    14 => core::arch::x86_64::_mm_extract_epi8::<14>(value) as _,
                    _ => core::arch::x86_64::_mm_extract_epi8::<15>(value) as _,
                }
            }
        }
    };
    // ===== SSE2-only (x86_v1) counterparts =====
    // `_mm_extract_epi32`/`_mm_extract_epi64`/`_mm_extract_epi8`/`_mm_extract_ps` are all
    // SSE4.1, so v1 reaches lane 0 with a `cvt` (free -- the value is already in the low
    // element) and every other lane with one shuffle/unpack first. `@epi16x8` and `@pd128`
    // are already SSE2-legal, so v1 uses those arms directly rather than getting `_v1` twins.
    (@epi64x2_v1) => {
        fn extract<const I: usize>(value: $crate::register::Storage<Self>) -> Self::Element {
            const {
                assert!(I < 2, "Index out of bounds for register lane extraction");
            }
            unsafe {
                match I {
                    0 => core::arch::x86_64::_mm_cvtsi128_si64(value) as _,
                    _ => {
                        core::arch::x86_64::_mm_cvtsi128_si64(core::arch::x86_64::_mm_unpackhi_epi64(value, value)) as _
                    }
                }
            }
        }
    };
    (@epi32x4_v1) => {
        fn extract<const I: usize>(value: $crate::register::Storage<Self>) -> Self::Element {
            const {
                assert!(I < 4, "Index out of bounds for register lane extraction");
            }
            unsafe {
                match I {
                    0 => core::arch::x86_64::_mm_cvtsi128_si32(value) as _,
                    1 => core::arch::x86_64::_mm_cvtsi128_si32(core::arch::x86_64::_mm_shuffle_epi32::<0b01_01_01_01>(
                        value,
                    )) as _,
                    2 => {
                        core::arch::x86_64::_mm_cvtsi128_si32(core::arch::x86_64::_mm_unpackhi_epi64(value, value)) as _
                    }
                    _ => core::arch::x86_64::_mm_cvtsi128_si32(core::arch::x86_64::_mm_shuffle_epi32::<0b11_11_11_11>(
                        value,
                    )) as _,
                }
            }
        }
    };
    // Byte `I` is the low half of word `I / 2` when `I` is even, the high half when odd.
    // The `as u8` before `as _` makes this bit-preserving for a signed OR unsigned element.
    (@epi8x16_v1) => {
        fn extract<const I: usize>(value: $crate::register::Storage<Self>) -> Self::Element {
            const {
                assert!(I < 16, "Index out of bounds for register lane extraction");
            }
            unsafe {
                let word = match I / 2 {
                    0 => core::arch::x86_64::_mm_extract_epi16::<0>(value),
                    1 => core::arch::x86_64::_mm_extract_epi16::<1>(value),
                    2 => core::arch::x86_64::_mm_extract_epi16::<2>(value),
                    3 => core::arch::x86_64::_mm_extract_epi16::<3>(value),
                    4 => core::arch::x86_64::_mm_extract_epi16::<4>(value),
                    5 => core::arch::x86_64::_mm_extract_epi16::<5>(value),
                    6 => core::arch::x86_64::_mm_extract_epi16::<6>(value),
                    _ => core::arch::x86_64::_mm_extract_epi16::<7>(value),
                };
                ((word >> (8 * (I % 2))) as u8) as _
            }
        }
    };
    (@ps128_v1) => {
        fn extract<const I: usize>(value: $crate::register::Storage<Self>) -> Self::Element {
            const {
                assert!(I < 4, "Index out of bounds for register lane extraction");
            }
            unsafe {
                match I {
                    0 => core::arch::x86_64::_mm_cvtss_f32(value),
                    1 => core::arch::x86_64::_mm_cvtss_f32(core::arch::x86_64::_mm_shuffle_ps::<0b01_01_01_01>(
                        value, value,
                    )),
                    2 => core::arch::x86_64::_mm_cvtss_f32(core::arch::x86_64::_mm_unpackhi_ps(value, value)),
                    _ => core::arch::x86_64::_mm_cvtss_f32(core::arch::x86_64::_mm_shuffle_ps::<0b11_11_11_11>(
                        value, value,
                    )),
                }
            }
        }
    };
    // ===== 256-bit integer: extract the 128-bit lane, then the element =====
    (@epi64x4) => {
        fn extract<const I: usize>(value: $crate::register::Storage<Self>) -> Self::Element {
            const {
                assert!(I < 4, "Index out of bounds for register lane extraction");
            }
            unsafe {
                match I {
                    0 => core::arch::x86_64::_mm_extract_epi64::<0>(core::arch::x86_64::_mm256_extracti128_si256::<0>(
                        value,
                    )) as _,
                    1 => core::arch::x86_64::_mm_extract_epi64::<1>(core::arch::x86_64::_mm256_extracti128_si256::<0>(
                        value,
                    )) as _,
                    2 => core::arch::x86_64::_mm_extract_epi64::<0>(core::arch::x86_64::_mm256_extracti128_si256::<1>(
                        value,
                    )) as _,
                    _ => core::arch::x86_64::_mm_extract_epi64::<1>(core::arch::x86_64::_mm256_extracti128_si256::<1>(
                        value,
                    )) as _,
                }
            }
        }
    };
    (@epi32x8) => {
        fn extract<const I: usize>(value: $crate::register::Storage<Self>) -> Self::Element {
            const {
                assert!(I < 8, "Index out of bounds for register lane extraction");
            }
            unsafe {
                let (lo, hi) = (
                    core::arch::x86_64::_mm256_extracti128_si256::<0>(value),
                    core::arch::x86_64::_mm256_extracti128_si256::<1>(value),
                );
                match I {
                    0 => core::arch::x86_64::_mm_extract_epi32::<0>(lo) as _,
                    1 => core::arch::x86_64::_mm_extract_epi32::<1>(lo) as _,
                    2 => core::arch::x86_64::_mm_extract_epi32::<2>(lo) as _,
                    3 => core::arch::x86_64::_mm_extract_epi32::<3>(lo) as _,
                    4 => core::arch::x86_64::_mm_extract_epi32::<0>(hi) as _,
                    5 => core::arch::x86_64::_mm_extract_epi32::<1>(hi) as _,
                    6 => core::arch::x86_64::_mm_extract_epi32::<2>(hi) as _,
                    _ => core::arch::x86_64::_mm_extract_epi32::<3>(hi) as _,
                }
            }
        }
    };
    (@epi16x16) => {
        fn extract<const I: usize>(value: $crate::register::Storage<Self>) -> Self::Element {
            const {
                assert!(I < 16, "Index out of bounds for register lane extraction");
            }
            unsafe {
                let (lo, hi) = (
                    core::arch::x86_64::_mm256_extracti128_si256::<0>(value),
                    core::arch::x86_64::_mm256_extracti128_si256::<1>(value),
                );
                match I {
                    0 => core::arch::x86_64::_mm_extract_epi16::<0>(lo) as _,
                    1 => core::arch::x86_64::_mm_extract_epi16::<1>(lo) as _,
                    2 => core::arch::x86_64::_mm_extract_epi16::<2>(lo) as _,
                    3 => core::arch::x86_64::_mm_extract_epi16::<3>(lo) as _,
                    4 => core::arch::x86_64::_mm_extract_epi16::<4>(lo) as _,
                    5 => core::arch::x86_64::_mm_extract_epi16::<5>(lo) as _,
                    6 => core::arch::x86_64::_mm_extract_epi16::<6>(lo) as _,
                    7 => core::arch::x86_64::_mm_extract_epi16::<7>(lo) as _,
                    8 => core::arch::x86_64::_mm_extract_epi16::<0>(hi) as _,
                    9 => core::arch::x86_64::_mm_extract_epi16::<1>(hi) as _,
                    10 => core::arch::x86_64::_mm_extract_epi16::<2>(hi) as _,
                    11 => core::arch::x86_64::_mm_extract_epi16::<3>(hi) as _,
                    12 => core::arch::x86_64::_mm_extract_epi16::<4>(hi) as _,
                    13 => core::arch::x86_64::_mm_extract_epi16::<5>(hi) as _,
                    14 => core::arch::x86_64::_mm_extract_epi16::<6>(hi) as _,
                    _ => core::arch::x86_64::_mm_extract_epi16::<7>(hi) as _,
                }
            }
        }
    };
    (@epi8x32) => {
        fn extract<const I: usize>(value: $crate::register::Storage<Self>) -> Self::Element {
            const {
                assert!(I < 32, "Index out of bounds for register lane extraction");
            }
            unsafe {
                let (lo, hi) = (
                    core::arch::x86_64::_mm256_extracti128_si256::<0>(value),
                    core::arch::x86_64::_mm256_extracti128_si256::<1>(value),
                );
                match I {
                    0 => core::arch::x86_64::_mm_extract_epi8::<0>(lo) as _,
                    1 => core::arch::x86_64::_mm_extract_epi8::<1>(lo) as _,
                    2 => core::arch::x86_64::_mm_extract_epi8::<2>(lo) as _,
                    3 => core::arch::x86_64::_mm_extract_epi8::<3>(lo) as _,
                    4 => core::arch::x86_64::_mm_extract_epi8::<4>(lo) as _,
                    5 => core::arch::x86_64::_mm_extract_epi8::<5>(lo) as _,
                    6 => core::arch::x86_64::_mm_extract_epi8::<6>(lo) as _,
                    7 => core::arch::x86_64::_mm_extract_epi8::<7>(lo) as _,
                    8 => core::arch::x86_64::_mm_extract_epi8::<8>(lo) as _,
                    9 => core::arch::x86_64::_mm_extract_epi8::<9>(lo) as _,
                    10 => core::arch::x86_64::_mm_extract_epi8::<10>(lo) as _,
                    11 => core::arch::x86_64::_mm_extract_epi8::<11>(lo) as _,
                    12 => core::arch::x86_64::_mm_extract_epi8::<12>(lo) as _,
                    13 => core::arch::x86_64::_mm_extract_epi8::<13>(lo) as _,
                    14 => core::arch::x86_64::_mm_extract_epi8::<14>(lo) as _,
                    15 => core::arch::x86_64::_mm_extract_epi8::<15>(lo) as _,
                    16 => core::arch::x86_64::_mm_extract_epi8::<0>(hi) as _,
                    17 => core::arch::x86_64::_mm_extract_epi8::<1>(hi) as _,
                    18 => core::arch::x86_64::_mm_extract_epi8::<2>(hi) as _,
                    19 => core::arch::x86_64::_mm_extract_epi8::<3>(hi) as _,
                    20 => core::arch::x86_64::_mm_extract_epi8::<4>(hi) as _,
                    21 => core::arch::x86_64::_mm_extract_epi8::<5>(hi) as _,
                    22 => core::arch::x86_64::_mm_extract_epi8::<6>(hi) as _,
                    23 => core::arch::x86_64::_mm_extract_epi8::<7>(hi) as _,
                    24 => core::arch::x86_64::_mm_extract_epi8::<8>(hi) as _,
                    25 => core::arch::x86_64::_mm_extract_epi8::<9>(hi) as _,
                    26 => core::arch::x86_64::_mm_extract_epi8::<10>(hi) as _,
                    27 => core::arch::x86_64::_mm_extract_epi8::<11>(hi) as _,
                    28 => core::arch::x86_64::_mm_extract_epi8::<12>(hi) as _,
                    29 => core::arch::x86_64::_mm_extract_epi8::<13>(hi) as _,
                    30 => core::arch::x86_64::_mm_extract_epi8::<14>(hi) as _,
                    _ => core::arch::x86_64::_mm_extract_epi8::<15>(hi) as _,
                }
            }
        }
    };
    // ===== floats: lane 0 stays in xmm via cvt; other lanes via extract_ps/unpackhi =====
    (@ps128) => {
        fn extract<const I: usize>(value: $crate::register::Storage<Self>) -> Self::Element {
            const {
                assert!(I < 4, "Index out of bounds for register lane extraction");
            }
            unsafe {
                match I {
                    0 => core::arch::x86_64::_mm_cvtss_f32(value),
                    1 => f32::from_bits(core::arch::x86_64::_mm_extract_ps::<1>(value) as u32),
                    2 => f32::from_bits(core::arch::x86_64::_mm_extract_ps::<2>(value) as u32),
                    _ => f32::from_bits(core::arch::x86_64::_mm_extract_ps::<3>(value) as u32),
                }
            }
        }
    };
    (@ps256) => {
        fn extract<const I: usize>(value: $crate::register::Storage<Self>) -> Self::Element {
            const {
                assert!(I < 8, "Index out of bounds for register lane extraction");
            }
            unsafe {
                let (lo, hi) = (
                    core::arch::x86_64::_mm256_extractf128_ps::<0>(value),
                    core::arch::x86_64::_mm256_extractf128_ps::<1>(value),
                );
                match I {
                    0 => core::arch::x86_64::_mm_cvtss_f32(lo),
                    1 => f32::from_bits(core::arch::x86_64::_mm_extract_ps::<1>(lo) as u32),
                    2 => f32::from_bits(core::arch::x86_64::_mm_extract_ps::<2>(lo) as u32),
                    3 => f32::from_bits(core::arch::x86_64::_mm_extract_ps::<3>(lo) as u32),
                    4 => core::arch::x86_64::_mm_cvtss_f32(hi),
                    5 => f32::from_bits(core::arch::x86_64::_mm_extract_ps::<1>(hi) as u32),
                    6 => f32::from_bits(core::arch::x86_64::_mm_extract_ps::<2>(hi) as u32),
                    _ => f32::from_bits(core::arch::x86_64::_mm_extract_ps::<3>(hi) as u32),
                }
            }
        }
    };
    (@pd128) => {
        fn extract<const I: usize>(value: $crate::register::Storage<Self>) -> Self::Element {
            const {
                assert!(I < 2, "Index out of bounds for register lane extraction");
            }
            unsafe {
                match I {
                    0 => core::arch::x86_64::_mm_cvtsd_f64(value),
                    _ => core::arch::x86_64::_mm_cvtsd_f64(core::arch::x86_64::_mm_unpackhi_pd(value, value)),
                }
            }
        }
    };
    (@pd256) => {
        fn extract<const I: usize>(value: $crate::register::Storage<Self>) -> Self::Element {
            const {
                assert!(I < 4, "Index out of bounds for register lane extraction");
            }
            unsafe {
                let (lo, hi) = (
                    core::arch::x86_64::_mm256_extractf128_pd::<0>(value),
                    core::arch::x86_64::_mm256_extractf128_pd::<1>(value),
                );
                match I {
                    0 => core::arch::x86_64::_mm_cvtsd_f64(lo),
                    1 => core::arch::x86_64::_mm_cvtsd_f64(core::arch::x86_64::_mm_unpackhi_pd(lo, lo)),
                    2 => core::arch::x86_64::_mm_cvtsd_f64(hi),
                    _ => core::arch::x86_64::_mm_cvtsd_f64(core::arch::x86_64::_mm_unpackhi_pd(hi, hi)),
                }
            }
        }
    };
}

/// Native two-register element align (`Register::align`) for any 128-bit integer
/// register, via `_mm_alignr_epi8` (SSSE3+). Drop into an `impl Register` block.
///
/// `align::<OFFSET>(a, b)` is the `LANES`-lane window at lane `OFFSET` of the
/// concatenation `[a, b]` with `a` as the low half. `_mm_alignr_epi8::<n>(hi, lo)`
/// yields `concat(lo:hi)[n..]` in BYTES, so we pass `(b, a)` and key the match on
/// the byte offset `ob = OFFSET * size_of::<Element>()` - which makes this work
/// for i32x4/u64x2/... and not just the byte registers. The immediate cannot be a
/// const expression of `OFFSET` on stable, hence the (verbose) match supplying
/// each literal; `ob` const-folds to exactly one arm. `ob > 16` means
/// `OFFSET > LANES` (out of range) and falls back to the generic `swizzle_const`
/// default.
macro_rules! impl_byte_align_alignr {
    () => {
        const HAS_NATIVE_ALIGN: bool = true;

        fn align<const OFFSET: usize>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
            match const { OFFSET * core::mem::size_of::<Self::Element>() } {
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
        const HAS_NATIVE_ALIGN: bool = true;

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
        const HAS_NATIVE_ALIGN: bool = true;

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

/// Native two-register element align (`Register::align`) for a FLOAT register,
/// routed through its bitwise-identical unsigned integer register. Drop into an
/// `impl Register` block for a float register; `$bits` must be the same register
/// named by its [`FloatRegister::Bits`](crate::register::FloatRegister::Bits).
///
/// `align` is pure lane movement, so reinterpreting as the same-shape integer
/// register, aligning there, and reinterpreting back is exact for every bit
/// pattern (NaN payloads and signalling bits included). The reinterprets are free:
/// on x86 `_mm_castps_si128` and friends emit no instruction.
///
/// The integer registers carry native `align` overrides (`palignr`,
/// `pslldq`/`psrldq`, the AVX2 256-bit sequence); the float registers would
/// otherwise fall through to the generic `swizzle_const` default - two `permutev`s
/// plus a `blendv` at best, and on SSE2 (`HAS_PERMUTEV == false`) a scalar memory
/// round-trip.
macro_rules! impl_float_align_via_bits {
    ($bits:ty) => {
        // inherited, not asserted: this is only as native as the register it routes to
        const HAS_NATIVE_ALIGN: bool = <$bits as $crate::register::Register>::HAS_NATIVE_ALIGN;

        fn align<const OFFSET: usize>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
            <Self as $crate::register::BitCastRegister<$bits>>::from_bits(
                <$bits as $crate::register::Register>::align::<OFFSET>(
                    <$bits as $crate::register::BitCastRegister<Self>>::from_bits(a),
                    <$bits as $crate::register::BitCastRegister<Self>>::from_bits(b),
                ),
            )
        }
    };
}

/// Stamp [`WidenIndexRegister`](crate::register::WidenIndexRegister) for x86
/// registers: widen a `u8` compress/expand table row into the `u32` permute
/// control with the `pmovzxbd` family.
///
/// The shape tag is the register's **lane count**, not its width - the control
/// array follows `LANES` (`f64x4` and `f32x4` both want four `u32`s). `x8h`
/// ("halves") is the SSE4.1-only form for 8-lane registers on a tier without
/// `_mm256_cvtepu8_epi32`, which widens in two 128-bit steps.
///
/// Invoke once per backend in its `registers/mod.rs`, where `arch` is in scope.
macro_rules! impl_widen_indices_x86 {
    ($($reg:ty => $shape:ident),* $(,)?) => {
        $(
            #[thermite_macros::inline_always]
            impl $crate::register::WidenIndexRegister for $reg {
                fn widen_indices(
                    idxs: &generic_array::GenericArray<u8, generic_array::typenum::U8>,
                ) -> generic_array::GenericArray<u32, <Self as $crate::register::CoreRegister>::Lanes> {
                    unsafe { impl_widen_indices_x86!(@body idxs, $shape) }
                }
            }
        )*
    };

    // 2 lanes (8 bytes of control): widen four and keep the low half.
    (@body $idxs:ident, x2) => {{
        let q = arch::_mm_cvtepu8_epi32(arch::_mm_cvtsi32_si128(
            core::ptr::read_unaligned($idxs.as_ptr() as *const i32),
        ));
        core::mem::transmute_copy(&q)
    }};
    // 4 lanes (16 bytes): exactly one `pmovzxbd`.
    (@body $idxs:ident, x4) => {{
        let q = arch::_mm_cvtepu8_epi32(arch::_mm_cvtsi32_si128(
            core::ptr::read_unaligned($idxs.as_ptr() as *const i32),
        ));
        core::mem::transmute_copy(&q)
    }};
    // 8 lanes (32 bytes) with AVX2: one `vpmovzxbd` off a 64-bit load.
    (@body $idxs:ident, x8) => {{
        let o = arch::_mm256_cvtepu8_epi32(arch::_mm_loadl_epi64($idxs.as_ptr() as *const arch::__m128i));
        core::mem::transmute_copy(&o)
    }};
    // 8 lanes without AVX2: two 128-bit widenings.
    (@body $idxs:ident, x8h) => {{
        let p = $idxs.as_ptr();
        let lo = arch::_mm_cvtepu8_epi32(arch::_mm_cvtsi32_si128(core::ptr::read_unaligned(p as *const i32)));
        let hi = arch::_mm_cvtepu8_epi32(arch::_mm_cvtsi32_si128(core::ptr::read_unaligned(
            p.add(4) as *const i32,
        )));
        core::mem::transmute_copy(&[lo, hi])
    }};
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

/// Stamp [`BitCastRegister`](crate::register::BitCastRegister) for register pairs that
/// already share a `Storage` type, where the reinterpret is the identity function and no
/// `arch::` intrinsic exists (or is needed). On x86 every 128/256-bit integer register is
/// the same `__m128i`/`__m256i`, and on wasm everything is `v128`, so this covers the
/// whole integer matrix for those backends.
///
/// Complements [`impl_bit_casts`] (float <-> int, which needs a real cast intrinsic). The
/// pairs stamped here differ from the usual same-lane-count reinterprets: they relate
/// registers of the SAME TOTAL WIDTH but DIFFERENT lane counts (`u8x16` <-> `u64x2`),
/// which is what group-wise reductions like the SAD family need in order to view a byte
/// register as wider accumulator lanes.
macro_rules! impl_bit_casts_identity {
    ($($from:ty as $to:ty),* $(,)?) => {
        const _: () = {$(
            #[thermite_macros::inline_always]
            impl $crate::register::BitCastRegister<$from> for $to {
                fn from_bits(value: Storage<$from>) -> Storage<Self> {
                    value
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

/// Same-width float -> int register casts, the only pairs where the two
/// strengths are separate instructions.
///
/// `$conv` is the fast hardware/polyfill conversion, whose documented
/// precondition is in-range finite inputs (out-of-range/NaN lanes are
/// backend-defined). `$sat` is the `as`-exact counterpart: NaN -> 0,
/// out-of-range clamps. `$fast_conv`, where a backend has one, is narrower
/// still.
///
/// The `strict_ieee754` redirect is NOT here. It lives at the vector layer, in
/// the `Vector<R>` impl of `CastVector`, so it covers every pair rather than
/// only the ones this macro stamps.
macro_rules! impl_float_to_int_casts {
    ($($from:ty as $to:ty => $conv:ident sat $sat:ident $(| $fast_conv:ident)?),* $(,)?) => {
        const _: () = {$(
            #[thermite_macros::inline_always]
            impl $crate::register::CastRegister<$from> for $to {
                fn cast_from(value: Storage<$from>) -> Storage<Self> {
                    unsafe { arch::$conv(value) }
                }

                fn saturating_cast_from(value: Storage<$from>) -> Storage<Self> {
                    unsafe { arch::$sat(value) }
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

/// Cross-width casts composed from two existing legs: `$from -> $mid`, then
/// `$mid -> $to`. Both strengths compose independently, each through its own
/// pair of legs, which is what keeps `saturating_cast_from` bit-identical to a
/// direct `as`: saturation is idempotent across nested ranges, so clamping at
/// `$mid` and again at `$to` lands where clamping once at `$to` would.
///
/// `cast_from` composes the fast legs and inherits their contract, matching what
/// the hand-written float -> narrow-int impls used to spell out (a same-width
/// `cvtt` followed by an integer narrow).
macro_rules! impl_cast_via {
    ($($from:ty as $to:ty => via $mid:ty),* $(,)?) => {
        const _: () = {$(
            #[thermite_macros::inline_always]
            impl $crate::register::CastRegister<$from> for $to {
                fn cast_from(value: Storage<$from>) -> Storage<Self> {
                    <Self as $crate::register::CastRegister<$mid>>::cast_from(
                        <$mid as $crate::register::CastRegister<$from>>::cast_from(value),
                    )
                }

                fn saturating_cast_from(value: Storage<$from>) -> Storage<Self> {
                    <Self as $crate::register::CastRegister<$mid>>::saturating_cast_from(
                        <$mid as $crate::register::CastRegister<$from>>::saturating_cast_from(value),
                    )
                }
            }
        )*};
    };
}

/// f32 -> 64-bit int casts: widen exactly to f64 (`$mid`), then the same-width
/// f64 -> int cast. The widen is lossless, so it is a plain `cast_from` on both
/// paths; doing it first means the clamp happens at the DESTINATION's range,
/// matching a direct `as` rather than clamping at `i32` on the way through.
macro_rules! impl_cast_via_widen {
    ($($from:ty as $to:ty => via $mid:ty),* $(,)?) => {
        const _: () = {$(
            #[thermite_macros::inline_always]
            impl $crate::register::CastRegister<$from> for $to {
                fn cast_from(value: Storage<$from>) -> Storage<Self> {
                    <Self as $crate::register::CastRegister<$mid>>::cast_from(
                        <$mid as $crate::register::CastRegister<$from>>::cast_from(value),
                    )
                }

                fn saturating_cast_from(value: Storage<$from>) -> Storage<Self> {
                    <Self as $crate::register::CastRegister<$mid>>::saturating_cast_from(
                        <$mid as $crate::register::CastRegister<$from>>::cast_from(value),
                    )
                }
            }
        )*};
    };
}

/// Cross-width casts composed from two legs, defining **only** `cast_from` and
/// leaving `saturating_cast_from` to default to it. For pairs where the two
/// strengths are the same conversion, which is every pair this stamps.
///
/// Two families qualify, both matching the scalar oracle:
///
/// - **sign-changing integer**, composed as a same-signedness width change
///   (`$mid`, the source's signedness at the destination's width) then the free
///   same-width sign reinterpret. Scalar has no saturating lowering for these
///   either - `impl_nontrivial_casts!` stamps one `value as _` body and lets the
///   default do the rest - so `as` semantics are the whole contract.
/// - **int -> float**, which cannot go out of range, only lose precision, so
///   `cast_from` is already exact and a separate saturating body would be dead
///   weight.
///
/// Threading both strengths through the legs, as [`impl_cast_via`] does, would
/// pick up the clamp from a same-signedness narrow leg and disagree with the
/// oracle: `300u32 -> i8` would give `127` where scalar gives `44`.
macro_rules! impl_cast_from_via {
    ($($from:ty as $to:ty => via $mid:ty),* $(,)?) => {
        const _: () = {$(
            #[thermite_macros::inline_always]
            impl $crate::register::CastRegister<$from> for $to {
                fn cast_from(value: Storage<$from>) -> Storage<Self> {
                    <Self as $crate::register::CastRegister<$mid>>::cast_from(
                        <$mid as $crate::register::CastRegister<$from>>::cast_from(value),
                    )
                }
            }
        )*};
    };
}

/// The sign-changing pairs crossing between the 8-bit ladder and the 32/64-bit
/// one. Row layout: `[i32, u32, i64, u64, i8, u8]`.
///
/// Which crossings need stating is a property of the row's register shapes, not
/// of the element types, which is why these come in three separately-invocable
/// pieces. The array cast ladder in `register/array.rs` derives a pair only when
/// BOTH endpoints are `ArrayRegister`s; a native register on either end puts the
/// pair here. At the widest lane counts a backend's 8-bit slot is typically the
/// last native one left.
macro_rules! impl_sign_cast_matrix_8_to_32_64 {
    ($([$i32:ty, $u32:ty, $i64:ty, $u64:ty, $i8:ty, $u8:ty]),* $(,)?) => {$(
        impl_cast_from_via! {
            // widen: extend in the source's signedness, then reinterpret
            $i8 as $u32 => via $i32,
            $i8 as $u64 => via $i64,
            $u8 as $i32 => via $u32,
            $u8 as $i64 => via $u64,
            // narrow: truncate in the source's signedness, then reinterpret
            $i32 as $u8 => via $i8,
            $u32 as $i8 => via $u8,
            $i64 as $u8 => via $i8,
            $u64 as $i8 => via $u8,
        }
    )*};
}

/// The sign-changing pairs crossing between the 16-bit ladder and the 32/64-bit
/// one. Row layout: `[i32, u32, i64, u64, i16, u16]`. See
/// [`impl_sign_cast_matrix_8_to_32_64`] for why this is separable.
macro_rules! impl_sign_cast_matrix_16_to_32_64 {
    ($([$i32:ty, $u32:ty, $i64:ty, $u64:ty, $i16:ty, $u16:ty]),* $(,)?) => {$(
        impl_cast_from_via! {
            $i16 as $u32 => via $i32,
            $i16 as $u64 => via $i64,
            $u16 as $i32 => via $u32,
            $u16 as $i64 => via $u64,
            $i32 as $u16 => via $i16,
            $u32 as $i16 => via $u16,
            $i64 as $u16 => via $i16,
            $u64 as $i16 => via $u16,
        }
    )*};
}

/// Both of the 8/16-bit crossing halves. Row layout:
/// `[i32, u32, i64, u64, i16, u16, i8, u8]`.
macro_rules! impl_sign_cast_matrix_8_16_to_32_64 {
    ($([$i32:ty, $u32:ty, $i64:ty, $u64:ty, $i16:ty, $u16:ty, $i8:ty, $u8:ty]),* $(,)?) => {$(
        impl_sign_cast_matrix_8_to_32_64! {
            [$i32, $u32, $i64, $u64, $i8, $u8]
        }
        impl_sign_cast_matrix_16_to_32_64! {
            [$i32, $u32, $i64, $u64, $i16, $u16]
        }
    )*};
}

/// The four sign-changing 32 <-> 64 crossings, for one or more lane-count rows.
/// Row layout: `[i32, u32, i64, u64]`.
///
/// Kept apart from [`impl_sign_cast_matrix_8_16_to_32_64`] because a row whose
/// 32- and 64-bit slots are both `ArrayRegister`s a factor of two apart gets
/// these from the array cast ladder, and stamping them anyway is a coherence
/// conflict rather than a duplicate.
macro_rules! impl_sign_cast_matrix_32_64 {
    ($([$i32:ty, $u32:ty, $i64:ty, $u64:ty]),* $(,)?) => {$(
        impl_cast_from_via! {
            $i32 as $u64 => via $i64,
            $u32 as $i64 => via $u64,
            $i64 as $u32 => via $i32,
            $u64 as $i32 => via $u32,
        }
    )*};
}

/// Both halves of the sign-changing matrix for one or more lane-count rows, in
/// the same row layout as [`impl_float_cast_matrix`]:
/// `[f32, f64, i32, u32, i64, u64, i16, u16, i8, u8]` (the two float slots are
/// unused here, kept so a backend can pass one row shape to both macros).
///
/// For a row where the array ladder already derives the 32 <-> 64 crossings,
/// invoke [`impl_sign_cast_matrix_8_16_to_32_64`] alone instead.
macro_rules! impl_sign_cast_matrix {
    ($([$f32:ty, $f64:ty, $i32:ty, $u32:ty, $i64:ty, $u64:ty, $i16:ty, $u16:ty, $i8:ty, $u8:ty]),* $(,)?) => {$(
        impl_sign_cast_matrix_8_16_to_32_64! {
            [$i32, $u32, $i64, $u64, $i16, $u16, $i8, $u8]
        }
        impl_sign_cast_matrix_32_64! {
            [$i32, $u32, $i64, $u64]
        }
    )*};
}

/// The 8 <-> 16 sign-changing pairs, split out of [`impl_sign_cast_matrix`]
/// because the 2-lane rows get them from the scalar `ArrayRegister` impls and
/// would collide. Row layout is the four integer slots only:
/// `[i16, u16, i8, u8]`.
macro_rules! impl_sign_cast_matrix_8_16 {
    ($([$i16:ty, $u16:ty, $i8:ty, $u8:ty]),* $(,)?) => {$(
        impl_cast_from_via! {
            $i8 as $u16 => via $i16,
            $u8 as $i16 => via $u16,
            $i16 as $u8 => via $i8,
            $u16 as $i8 => via $u8,
        }
    )*};
}

/// Stamp the full cross-width float -> int cast matrix for one or more
/// lane-count rows of a backend's `Simd` slot types. The same-width float -> int
/// casts and the integer-narrowing matrix must already exist for the row;
/// everything here is composed from them (or from the exact f32 -> f64 widen for
/// the 64-bit destinations).
///
/// Row layout: `[f32, f64, i32, u32, i64, u64, i16, u16, i8, u8]` - the
/// backend's concrete types for one lane count.
macro_rules! impl_float_cast_matrix {
    ($([$f32:ty, $f64:ty, $i32:ty, $u32:ty, $i64:ty, $u64:ty, $i16:ty, $u16:ty, $i8:ty, $u8:ty]),* $(,)?) => {$(
        impl_cast_via! {
            $f32 as $i16 => via $i32,
            $f32 as $i8 => via $i32,
            $f32 as $u16 => via $u32,
            $f32 as $u8 => via $u32,
            $f64 as $i32 => via $i64,
            $f64 as $i16 => via $i64,
            $f64 as $i8 => via $i64,
            $f64 as $u32 => via $u64,
            $f64 as $u16 => via $u64,
            $f64 as $u8 => via $u64,
        }
        impl_cast_via_widen! {
            $f32 as $i64 => via $f64,
            $f32 as $u64 => via $f64,
        }
    )*};
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

/// Add `Register::compress` + `compress_z` + `expand` + `expand_z` overrides to
/// a register `impl` block that delegate to the `<= 8`-lane table polyfills
/// ([`compress_permute`] / [`expand_permute`]). Invoke inside
/// `impl Register for <Reg> { ... }` for any `Register` with at most 8 lanes.
///
/// One macro emits both directions so a register cannot opt into a fast compress
/// while silently leaving expand on the scalar default. The `_m` merge forms need
/// no entry here: their trait defaults blend over these overridden bodies and
/// inherit the fast path.
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

        #[inline(always)]
        fn expand(
            value: $crate::register::Storage<Self>,
            mask: $crate::register::Storage<<Self as $crate::register::CoreRegister>::Mask>,
        ) -> $crate::register::Storage<Self> {
            $crate::backend::generic::polyfills::expand_permute::<Self>(value, mask)
        }

        #[inline(always)]
        fn expand_z(
            value: $crate::register::Storage<Self>,
            mask: $crate::register::Storage<<Self as $crate::register::CoreRegister>::Mask>,
        ) -> $crate::register::Storage<Self> {
            // Expand, then zero the unselected lanes: the zz composes AFTER here
            // (the packed front must be routed before masking), the mirror of
            // compress_z's zz-before.
            <Self as $crate::register::CoreRegister>::zz(
                mask,
                $crate::backend::generic::polyfills::expand_permute::<Self>(value, mask),
            )
        }
    };
}

/// Add `Register::compress` + `compress_z` + `expand` + `expand_z` overrides
/// that delegate to the wide polyfills ([`compress_permute_wide`] /
/// [`expand_permute_wide`]). Invoke inside `impl Register for <Reg> { ... }`
/// for any `Register` whose lane count is a multiple of 8 in `8..=64` (the
/// 16/32-lane byte and short vectors). Emits both directions, same as
/// [`compress_via_table!`].
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

        #[inline(always)]
        fn expand(
            value: $crate::register::Storage<Self>,
            mask: $crate::register::Storage<<Self as $crate::register::CoreRegister>::Mask>,
        ) -> $crate::register::Storage<Self> {
            $crate::backend::generic::polyfills::expand_permute_wide::<Self>(value, mask)
        }

        #[inline(always)]
        fn expand_z(
            value: $crate::register::Storage<Self>,
            mask: $crate::register::Storage<<Self as $crate::register::CoreRegister>::Mask>,
        ) -> $crate::register::Storage<Self> {
            <Self as $crate::register::CoreRegister>::zz(
                mask,
                $crate::backend::generic::polyfills::expand_permute_wide::<Self>(value, mask),
            )
        }
    };
}

/// Native radix-3 `interleave_radix`/`deinterleave_radix` overrides, stamped from
/// the backend's `(de)interleave3` register intrinsics. Drop into an
/// `impl Register for <Reg> { ... }` block.
///
/// Only the native radix-3 sequence is provided here; radix 2 and every other
/// radix fall back to the shared defaults
/// ([`interleave_radix_default`](crate::backend::generic::polyfills::interleave_radix_default)
/// / [`deinterleave_radix_default`](crate::backend::generic::polyfills::deinterleave_radix_default)),
/// so the `N == 2` forward and the permute+blend gather live in one place. `N` is
/// a compile-time constant at every call site, so the `if N == 3` folds and the
/// array plumbing evaporates (verified equal to the old tuple `interleave3` asm).
///
/// `$ilv3`/`$dilv3` are `unsafe fn(a, b, c) -> (x, y, z)` register intrinsics,
/// e.g. `arch::_mm256_interleave3_ps`.
macro_rules! impl_native_radix3 {
    ($ilv3:path, $dilv3:path) => {
        #[inline(always)]
        fn interleave_radix<const N: usize>(
            inputs: [$crate::register::Storage<Self>; N],
        ) -> [$crate::register::Storage<Self>; N] {
            if const { N == 3 } {
                // SAFETY: `N == 3` on this arm, so lanes 0/1/2 are in bounds.
                let (x, y, z) = unsafe {
                    (
                        *inputs.get_unchecked(0),
                        *inputs.get_unchecked(1),
                        *inputs.get_unchecked(2),
                    )
                };
                let (a, b, c) = unsafe { $ilv3(x, y, z) };
                let mut out = [<Self as $crate::register::CoreRegister>::EMPTY; N];
                // SAFETY: as above.
                unsafe {
                    *out.get_unchecked_mut(0) = a;
                    *out.get_unchecked_mut(1) = b;
                    *out.get_unchecked_mut(2) = c;
                }
                out
            } else {
                $crate::backend::generic::polyfills::interleave_radix_default::<Self, N>(inputs)
            }
        }

        #[inline(always)]
        fn deinterleave_radix<const N: usize>(
            inputs: [$crate::register::Storage<Self>; N],
        ) -> [$crate::register::Storage<Self>; N] {
            if const { N == 3 } {
                // SAFETY: `N == 3` on this arm.
                let (a, b, c) = unsafe {
                    (
                        *inputs.get_unchecked(0),
                        *inputs.get_unchecked(1),
                        *inputs.get_unchecked(2),
                    )
                };
                let (x, y, z) = unsafe { $dilv3(a, b, c) };
                let mut out = [<Self as $crate::register::CoreRegister>::EMPTY; N];
                // SAFETY: as above.
                unsafe {
                    *out.get_unchecked_mut(0) = x;
                    *out.get_unchecked_mut(1) = y;
                    *out.get_unchecked_mut(2) = z;
                }
                out
            } else {
                $crate::backend::generic::polyfills::deinterleave_radix_default::<Self, N>(inputs)
            }
        }
    };
}

/// Stamp `NumericRegister::sort` + `bitonic_clean` onto a register from the
/// sorting-network polyfills. Invoke inside `impl NumericRegister for <Reg>`,
/// passing the register's LANE COUNT: `sort_via_network!(8);`
///
/// Only 2 and 4 have dedicated arms. Everything else falls through to the trait
/// defaults, which are a real `sort_lanes` network up to 16 lanes (scalar walk
/// only past that). See the paragraph below and the note in the body for why
/// the old `(8)` arm was removed.
///
/// Both methods are emitted together on purpose. A cross-register merge needs
/// `bitonic_clean` to be the cheap `log2(LANES)`-layer form; if a register had a
/// fast `sort` and a defaulted `bitonic_clean`, the merge would silently pay a
/// full scalar sort per chunk with every test still green. Same no-drift rule as
/// `compress_via_table!`.
/// Lane counts with no arm here (8, 16, 32, ...) keep the trait defaults, so a
/// macro-stamped backend can pass its lane count straight through without the
/// caller filtering widths. Since the defaults became a real network
/// (`sort_lanes`) rather than a scalar walk, falling through is no longer a
/// silent de-optimization at any width up to 16 - and at 8 lanes it is a 21%
/// latency *win*, which is why that arm was removed. See the note in the body.
macro_rules! sort_via_network {
    (2) => { sort_via_network!(@emit sort_2, bitonic_clean_2); };
    (4) => { sort_via_network!(@emit sort_4, bitonic_clean_4); };

    // NOTE there is deliberately no `(8)` arm. It used to select `sort_8`, and
    // the trait default beat it: measured identical (30 insns, 31 uOps, 9.0
    // RThroughput) and **34 cycles latency against 43**, which is the metric a
    // lane sort pays. Both are depth-6 bitonic; the default fuses the "reverse
    // the second half" shuffle into a comparator's partner permutation instead
    // of issuing it standalone, so its critical path is 6 serial shuffles rather
    // than 7. `bitonic_clean` at 8 lanes is unaffected - the default's halving
    // strides generate `bitonic_clean_8`'s exact indices and keep-masks.
    //
    // Widths with no arm fall through to the trait defaults, which are a real
    // network up to 16 lanes and the scalar walk past that.
    ($other:tt) => {};

    (@emit $sort:ident, $clean:ident) => {
        #[inline(always)]
        fn sort_by<O: $crate::sort::SortOrder>(
            value: $crate::register::Storage<Self>,
        ) -> $crate::register::Storage<Self> {
            $crate::backend::generic::polyfills::sort::$sort::<Self, O>(value)
        }

        #[inline(always)]
        fn bitonic_clean_by<O: $crate::sort::SortOrder>(
            value: $crate::register::Storage<Self>,
        ) -> $crate::register::Storage<Self> {
            $crate::backend::generic::polyfills::sort::$clean::<Self, O>(value)
        }
    };
}
