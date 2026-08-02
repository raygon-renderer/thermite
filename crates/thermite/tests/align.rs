//! `IntegerVector::align` (two-register `palignr`-style element align).
//!
//! `a.align::<OFFSET>(b)` is the window of `LANES` lanes starting at lane
//! `OFFSET` of the concatenation `[a, b]`. With `a[i] = i + 1` and
//! `b[i] = N + i + 1` the concatenation is `concat[k] = k + 1`, so the result
//! must be `got[i] = i + OFFSET + 1` - a clean oracle covering `OFFSET == 0`
//! (returns `a`), `OFFSET == LANES` (returns `b`), and every spill in between.
//! Exercised across the scalar backend and, on x86, v1/v2/v3.

use thermite::backend::scalar::Scalar;
use thermite::prelude::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use thermite::backend::{x86_v1::X86V1, x86_v2::X86V2, x86_v3::X86V3};

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use thermite::backend::wasm::Wasm;

#[cfg(target_arch = "aarch64")]
use thermite::backend::neon::Neon;

macro_rules! check {
    ($name:ident, $vty:ty, $elem:ty, $n:literal, [$($off:literal),*]) => {
        #[test]
        fn $name() {
            type V = $vty;
            const N: usize = $n;

            // Distinct nonzero values so a misplaced/dropped lane is visible;
            // concat[k] == k + 1.
            let mut da = [0 as $elem; N];
            let mut db = [0 as $elem; N];
            for i in 0..N {
                da[i] = (i + 1) as $elem;
                db[i] = (N + i + 1) as $elem;
            }
            let a = V::new(da);
            let b = V::new(db);

            $({
                const OFF: usize = $off;
                let got = a.align::<OFF>(b);
                for i in 0..N {
                    assert_eq!(
                        got.as_slice()[i],
                        (i + OFF + 1) as $elem,
                        "align::<{}> width={} lane={}",
                        OFF, N, i
                    );
                }
            })*
        }
    };
}

macro_rules! suite {
    ($modname:ident, $backend:ty) => {
        mod $modname {
            use super::*;
            check!(
                u8x16,
                thermite::simd::u8x16<$backend>,
                u8,
                16,
                [0, 1, 2, 7, 8, 15, 16]
            );
            check!(
                i8x16,
                thermite::simd::i8x16<$backend>,
                i8,
                16,
                [0, 1, 2, 7, 8, 15, 16]
            );
            check!(u16x8, thermite::simd::u16x8<$backend>, u16, 8, [0, 1, 4, 7, 8]);
            check!(
                u16x16,
                thermite::simd::u16x16<$backend>,
                u16,
                16,
                [0, 1, 8, 15, 16]
            );
            check!(u32x4, thermite::simd::u32x4<$backend>, u32, 4, [0, 1, 2, 3, 4]);
            check!(u32x8, thermite::simd::u32x8<$backend>, u32, 8, [0, 1, 5, 8]);
            check!(i16x8, thermite::simd::i16x8<$backend>, i16, 8, [0, 1, 4, 8]);
            check!(i32x4, thermite::simd::i32x4<$backend>, i32, 4, [0, 2, 4]);
            check!(u64x2, thermite::simd::u64x2<$backend>, u64, 2, [0, 1, 2]);
            // 256-bit on v3 (native), ArrayRegister on v1/v2, 1-lane-chunk array on scalar.
            check!(
                i16x16,
                thermite::simd::i16x16<$backend>,
                i16,
                16,
                [0, 1, 8, 15, 16]
            );
            check!(i32x8, thermite::simd::i32x8<$backend>, i32, 8, [0, 1, 4, 7, 8]);
            check!(i64x4, thermite::simd::i64x4<$backend>, i64, 4, [0, 1, 2, 3, 4]);
            check!(u64x4, thermite::simd::u64x4<$backend>, u64, 4, [0, 1, 2, 4]);
            // Float widths: align is on GenericVector now, so it works for any
            // element type. f32x8/f64x4 are native 256 on v3, ArrayRegister on
            // v1/v2, 1-lane-chunk array on scalar. Values are exact integers, so
            // the lane-movement result compares exactly.
            check!(f32x8, thermite::simd::f32x8<$backend>, f32, 8, [0, 1, 5, 8]);
            check!(f64x4, thermite::simd::f64x4<$backend>, f64, 4, [0, 1, 2, 4]);
            // 128-bit floats: these route through the same-shape unsigned integer
            // register (`impl_float_align_via_bits!`), so they exercise the bitcast
            // round-trip on top of the integer register's native align.
            check!(f32x4, thermite::simd::f32x4<$backend>, f32, 4, [0, 1, 2, 3, 4]);
            check!(f64x2, thermite::simd::f64x2<$backend>, f64, 2, [0, 1, 2]);
        }
    };
}

suite!(scalar, Scalar);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
suite!(v1, X86V1);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
suite!(v2, X86V2);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
suite!(v3, X86V3);

// The native backends for the other architectures. Without these the suite only
// ever ran `Scalar` off-x86, leaving the wasm `i8x16.shuffle` align and the NEON
// `vext` align compiled but unexercised.
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
suite!(wasm, Wasm);
#[cfg(target_arch = "aarch64")]
suite!(neon, Neon);

// Native 256-bit byte align on v3 (AVX2): `i8xN`/`u8xN` are 32 lanes here, so
// `ob == OFFSET` and this exercises every arm of the 33-way match - including the
// odd byte offsets that the wider element types (EB >= 2) never reach.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod v3_native_bytes {
    use super::*;
    check!(
        i8xN,
        thermite::simd::i8xN<X86V3>,
        i8,
        32,
        [0, 1, 2, 3, 15, 16, 17, 18, 31, 32]
    );
    check!(
        u8xN,
        thermite::simd::u8xN<X86V3>,
        u8,
        32,
        [0, 1, 2, 3, 15, 16, 17, 18, 31, 32]
    );
}

/// Every full-width register on a real backend must advertise a native `align`.
///
/// A performance invariant, not a correctness one, hence the explicit pin:
/// `HAS_NATIVE_ALIGN` selects between two paths that agree on results, so a register
/// dropping to `false` changes nothing a functional test can observe - it just routes
/// the prefix-scan family (`NumericVector::prefix_sum` and friends) onto the
/// sequential fallback.
///
/// It has already happened: `u8x16`/`u8x32`/`u16x16`/`u32x8`/`u64x4` write their
/// `align` body by hand rather than through an `impl_*_align*!` macro, so they never
/// picked up the flag the macros emit, and every float register inheriting through
/// `impl_float_align_via_bits!` inherited the `false`. `f32x8` on AVX2 was scanning
/// one lane at a time with every test green.
mod native_align_flag {
    use thermite::register::Register;

    macro_rules! assert_native {
        ($($r:ty),* $(,)?) => {$(
            assert!(
                <$r as Register>::HAS_NATIVE_ALIGN,
                "{} lost its native align; the prefix-scan family silently \
                 degrades to the sequential fallback on this register",
                stringify!($r),
            );
        )*};
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn x86() {
        use thermite::backend::{
            x86_v1::registers as v1, x86_v2::registers as v2, x86_v3::registers as v3,
        };
        assert_native!(
            v1::F32x4V1, v1::F64x2V1, v1::I8x16V1, v1::I16x8V1, v1::I32x4V1, v1::I64x2V1,
            v1::U8x16V1, v1::U16x8V1, v1::U32x4V1, v1::U64x2V1,
        );
        assert_native!(
            v2::F32x4V2, v2::F64x2V2, v2::I8x16V2, v2::I16x8V2, v2::I32x4V2, v2::I64x2V2,
            v2::U8x16V2, v2::U16x8V2, v2::U32x4V2, v2::U64x2V2,
        );
        assert_native!(
            v3::F32x4V3, v3::F32x8V3, v3::F64x2V3, v3::F64x4V3,
            v3::I8x16V3, v3::I8x32V3, v3::I16x8V3, v3::I16x16V3,
            v3::I32x4V3, v3::I32x8V3, v3::I64x2V3, v3::I64x4V3,
            v3::U8x16V3, v3::U8x32V3, v3::U16x8V3, v3::U16x16V3,
            v3::U32x4V3, v3::U32x8V3, v3::U64x2V3, v3::U64x4V3,
        );
    }

    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    #[test]
    fn wasm() {
        use thermite::backend::wasm::registers as w;
        assert_native!(
            w::F32x4Wasm, w::F64x2Wasm, w::I8x16Wasm, w::I16x8Wasm, w::I32x4Wasm,
            w::I64x2Wasm, w::U8x16Wasm, w::U16x8Wasm, w::U32x4Wasm, w::U64x2Wasm,
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon() {
        use thermite::backend::neon::registers as n;
        assert_native!(
            n::F32x4Neon, n::F64x2Neon, n::I8x16Neon, n::I16x8Neon, n::I32x4Neon,
            n::I64x2Neon, n::U8x16Neon, n::U16x8Neon, n::U32x4Neon, n::U64x2Neon,
        );
    }

    /// `GenericVector::HAS_NATIVE_ALIGN` has to report what the register underneath
    /// actually does. A wrong answer here is invisible to every functional test -
    /// both align paths agree on results - and only shows up as a composite scan
    /// picking the wrong lowering, so it is asserted directly.
    #[test]
    fn vector_layer_forwards_the_register() {
        use thermite::prelude::*;

        macro_rules! assert_forwards {
            ($($v:ty => $r:ty),* $(,)?) => {$(
                assert_eq!(
                    <$v as GenericVector>::HAS_NATIVE_ALIGN,
                    <$r as Register>::HAS_NATIVE_ALIGN,
                    "{} does not forward {}'s HAS_NATIVE_ALIGN",
                    stringify!($v),
                    stringify!($r),
                );
            )*};
        }

        // The scalar backend is the interesting direction: it has no native align,
        // so this pins that `false` propagates as readily as `true`.
        assert_forwards!(Vector<f32> => f32, Vector<f64> => f64);

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            use thermite::backend::x86_v3::{self, registers as v3};
            assert_forwards!(
                x86_v3::f32x4 => v3::F32x4V3,
                x86_v3::f32x8 => v3::F32x8V3,
                x86_v3::f64x4 => v3::F64x4V3,
                x86_v3::i32x8 => v3::I32x8V3,
            );
        }
    }
}
