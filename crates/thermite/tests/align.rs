//! `IntegerVector::align` (two-register `palignr`-style element align).
//!
//! `a.align::<OFFSET>(b)` is the window of `LANES` lanes starting at lane
//! `OFFSET` of the concatenation `[a, b]`. With `a[i] = i + 1` and
//! `b[i] = N + i + 1` the concatenation is `concat[k] = k + 1`, so the result
//! must be `got[i] = i + OFFSET + 1` - a clean oracle covering `OFFSET == 0`
//! (returns `a`), `OFFSET == LANES` (returns `b`), and every spill in between.
//! Exercised on every backend.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use thermite::prelude::*;

macro_rules! check {
    ($vty:ty, $elem:ty, $n:literal, [$($off:literal),*]) => {{
        type V = $vty;
        const N: usize = $n;

        // Distinct nonzero values so a misplaced/dropped lane is visible:
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
    }};
}

/// The native-width byte registers have a per-backend lane count (16/32/64), so
/// the input is built from `LANES` and the offsets stay <= 16.
macro_rules! check_native {
    ($vty:ty, $elem:ty, [$($off:literal),*]) => {{
        type V = $vty;
        const N: usize = <V as GenericVector>::LANES;

        let mut da = [0 as $elem; N];
        let mut db = [0 as $elem; N];
        for i in 0..N {
            da[i] = (i + 1) as $elem;
            db[i] = (N + i + 1) as $elem;
        }
        let a = V::new(da.into());
        let b = V::new(db.into());

        $({
            const OFF: usize = $off;
            // offsets past the lane count are meaningless (Scalar's i8xN is 1 lane)
            if OFF <= N {
                let got = a.align::<OFF>(b);
                for i in 0..N {
                    assert_eq!(
                        got.as_slice()[i],
                        (i + OFF + 1) as $elem,
                        "align::<{}> width={} lane={}",
                        OFF, N, i
                    );
                }
            }
        })*
    }};
}

for_each_backend_concrete! {
    fn u8x16() { check!(thermite::simd::u8x16<S>, u8, 16, [0, 1, 2, 7, 8, 15, 16]) }
    fn i8x16() { check!(thermite::simd::i8x16<S>, i8, 16, [0, 1, 2, 7, 8, 15, 16]) }
    fn u16x8() { check!(thermite::simd::u16x8<S>, u16, 8, [0, 1, 4, 7, 8]) }
    fn u16x16() { check!(thermite::simd::u16x16<S>, u16, 16, [0, 1, 8, 15, 16]) }
    fn u32x4() { check!(thermite::simd::u32x4<S>, u32, 4, [0, 1, 2, 3, 4]) }
    fn u32x8() { check!(thermite::simd::u32x8<S>, u32, 8, [0, 1, 5, 8]) }
    fn u32x16() { check!(thermite::simd::u32x16<S>, u32, 16, [0, 1, 7, 8, 9, 15, 16]) }
    fn i16x8() { check!(thermite::simd::i16x8<S>, i16, 8, [0, 1, 4, 8]) }
    fn i32x4() { check!(thermite::simd::i32x4<S>, i32, 4, [0, 2, 4]) }
    fn u64x2() { check!(thermite::simd::u64x2<S>, u64, 2, [0, 1, 2]) }
    fn i16x16() { check!(thermite::simd::i16x16<S>, i16, 16, [0, 1, 8, 15, 16]) }
    fn i32x8() { check!(thermite::simd::i32x8<S>, i32, 8, [0, 1, 4, 7, 8]) }
    fn i64x4() { check!(thermite::simd::i64x4<S>, i64, 4, [0, 1, 2, 3, 4]) }
    fn u64x4() { check!(thermite::simd::u64x4<S>, u64, 4, [0, 1, 2, 4]) }
    fn u64x8() { check!(thermite::simd::u64x8<S>, u64, 8, [0, 1, 3, 4, 5, 8]) }
    fn f32x8() { check!(thermite::simd::f32x8<S>, f32, 8, [0, 1, 5, 8]) }
    fn f32x16() { check!(thermite::simd::f32x16<S>, f32, 16, [0, 1, 7, 8, 9, 16]) }
    fn f64x4() { check!(thermite::simd::f64x4<S>, f64, 4, [0, 1, 2, 4]) }
    fn f64x8() { check!(thermite::simd::f64x8<S>, f64, 8, [0, 1, 4, 8]) }
    fn f32x4() { check!(thermite::simd::f32x4<S>, f32, 4, [0, 1, 2, 3, 4]) }
    fn f64x2() { check!(thermite::simd::f64x2<S>, f64, 2, [0, 1, 2]) }

    // Native byte width: 16 on SSE/WASM/NEON, 32 on AVX2, 64 on AVX-512.
    fn native_bytes() {
        check_native!(thermite::simd::i8xN<S>, i8, [0, 1, 2, 3, 7, 8, 15, 16]);
        check_native!(thermite::simd::u8xN<S>, u8, [0, 1, 2, 3, 7, 8, 15, 16]);
    }
}

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
        use thermite::backend::{x86_v1::registers as v1, x86_v2::registers as v2, x86_v3::registers as v3};
        assert_native!(
            v1::F32x4V1,
            v1::F64x2V1,
            v1::I8x16V1,
            v1::I16x8V1,
            v1::I32x4V1,
            v1::I64x2V1,
            v1::U8x16V1,
            v1::U16x8V1,
            v1::U32x4V1,
            v1::U64x2V1,
        );
        assert_native!(
            v2::F32x4V2,
            v2::F64x2V2,
            v2::I8x16V2,
            v2::I16x8V2,
            v2::I32x4V2,
            v2::I64x2V2,
            v2::U8x16V2,
            v2::U16x8V2,
            v2::U32x4V2,
            v2::U64x2V2,
        );
        assert_native!(
            v3::F32x4V3,
            v3::F32x8V3,
            v3::F64x2V3,
            v3::F64x4V3,
            v3::I8x16V3,
            v3::I8x32V3,
            v3::I16x8V3,
            v3::I16x16V3,
            v3::I32x4V3,
            v3::I32x8V3,
            v3::I64x2V3,
            v3::I64x4V3,
            v3::U8x16V3,
            v3::U8x32V3,
            v3::U16x8V3,
            v3::U16x16V3,
            v3::U32x4V3,
            v3::U32x8V3,
            v3::U64x2V3,
            v3::U64x4V3,
        );
    }

    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    #[test]
    fn wasm() {
        use thermite::backend::wasm::registers as w;
        assert_native!(
            w::F32x4Wasm,
            w::F64x2Wasm,
            w::I8x16Wasm,
            w::I16x8Wasm,
            w::I32x4Wasm,
            w::I64x2Wasm,
            w::U8x16Wasm,
            w::U16x8Wasm,
            w::U32x4Wasm,
            w::U64x2Wasm,
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon() {
        use thermite::backend::neon::registers as n;
        assert_native!(
            n::F32x4Neon,
            n::F64x2Neon,
            n::I8x16Neon,
            n::I16x8Neon,
            n::I32x4Neon,
            n::I64x2Neon,
            n::U8x16Neon,
            n::U16x8Neon,
            n::U32x4Neon,
            n::U64x2Neon,
        );
    }

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
