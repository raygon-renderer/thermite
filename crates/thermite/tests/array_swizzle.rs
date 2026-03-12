use thermite::{
    backend::x86_v3::prelude::*,
    divider::{BranchfreeDivider, Divider},
};

use thermite::register::array::ArrayRegister;
use thermite::register::{CoreRegister, NumericRegister, Register, Storage, SwizzleIndices, SwizzleRegister};

use generic_array::{
    GenericArray,
    typenum::{self, Unsigned},
};

use rand::{Rng, RngExt, rngs::SmallRng};

/// Helper to execute and verify a `permutev` operation against its scalar fallback.
fn verify_permutev<R: SwizzleRegister>(input: Storage<R>, idxs: GenericArray<u32, R::Lanes>)
where
    R::Element: PartialEq + core::fmt::Debug,
{
    // The scalar_permutev method acts as our perfect ground truth
    let expected = R::scalar_permutev(input, idxs.clone());
    let actual = R::permutev(input, idxs.clone());

    let expected_arr = R::as_array(&expected);
    let actual_arr = R::as_array(&actual);

    assert_eq!(
        expected_arr,
        actual_arr,
        "permutev mismatch!\nIndices: {:?}\ninput: {:?}",
        idxs,
        R::as_array(&input)
    );
}

/// Helper to execute and verify a `swizzle` operation against its scalar fallback.
fn verify_swizzle<R: SwizzleRegister>(a: Storage<R>, b: Storage<R>, idxs: GenericArray<u32, R::Lanes>)
where
    R::Element: PartialEq + core::fmt::Debug,
{
    // The scalar_swizzle method acts as our perfect ground truth
    let expected = R::scalar_swizzle(a, b, idxs.clone());
    let actual = R::swizzle(a, b, idxs.clone());

    let expected_arr = R::as_array(&expected);
    let actual_arr = R::as_array(&actual);

    assert_eq!(
        expected_arr,
        actual_arr,
        "swizzle mismatch!\nIndices: {:?}\na: {:?}\nb: {:?}",
        idxs,
        R::as_array(&a),
        R::as_array(&b)
    );
}

/// Executes a rigorous sequence of routing tests against a concrete SwizzleRegister.
pub fn run_swizzle_tests<R: SwizzleRegister>(a: Storage<R>, b: Storage<R>)
where
    R::Element: PartialEq + core::fmt::Debug,
{
    let lanes = <R::Lanes as Unsigned>::USIZE;
    let mut idxs = GenericArray::<u32, R::Lanes>::default();

    // ----------------------------------------------------------------------
    // 1. Common Structured Patterns
    // ----------------------------------------------------------------------

    // Identity
    for i in 0..lanes {
        idxs[i] = i as u32;
    }
    verify_permutev::<R>(a, idxs.clone());
    verify_swizzle::<R>(a, b, idxs.clone());

    // Reverse
    for i in 0..lanes {
        idxs[i] = (lanes - 1 - i) as u32;
    }
    verify_permutev::<R>(a, idxs.clone());
    verify_swizzle::<R>(a, b, idxs.clone());

    // Broadcasts (Every target index broadcasted to all lanes)
    for target in 0..lanes {
        for i in 0..lanes {
            idxs[i] = target as u32;
        }
        verify_permutev::<R>(a, idxs.clone());
    }

    // Double-wide Broadcasts (Swizzle)
    for target in 0..(2 * lanes) {
        for i in 0..lanes {
            idxs[i] = target as u32;
        }
        verify_swizzle::<R>(a, b, idxs.clone());
    }

    // ----------------------------------------------------------------------
    // 2. Exhaustive Single-Lane Routing (O(N^2))
    // Ensures no single input-to-output chunk mapping is broken or off-by-one.
    // ----------------------------------------------------------------------
    for out_lane in 0..lanes {
        // permutev exhaustive
        for in_lane in 0..lanes {
            // zero out indices, set exactly one route
            for i in 0..lanes {
                idxs[i] = 0;
            }
            idxs[out_lane] = in_lane as u32;
            verify_permutev::<R>(a, idxs.clone());
        }

        // swizzle exhaustive
        for in_lane in 0..(2 * lanes) {
            for i in 0..lanes {
                idxs[i] = 0;
            }
            idxs[out_lane] = in_lane as u32;
            verify_swizzle::<R>(a, b, idxs.clone());
        }
    }

    // ----------------------------------------------------------------------
    // 3. Dense Accumulation Fuzzing
    // Stresses the `blendv` accumulation by heavily mixing chunk targets.
    // ----------------------------------------------------------------------
    let mut prng: SmallRng = rand::make_rng();

    // permutev fuzzing
    for _ in 0..5000 {
        for i in 0..lanes {
            idxs[i] = prng.random_range(0..lanes as u32);
        }
        verify_permutev::<R>(a, idxs.clone());
    }

    // swizzle fuzzing
    for _ in 0..5000 {
        for i in 0..lanes {
            idxs[i] = prng.random_range(0..(2 * lanes) as u32);
        }
        verify_swizzle::<R>(a, b, idxs.clone());
    }
}

#[test]
fn test_arrayregister_swizzle() {
    type RA = ArrayRegister<<X86V3 as Simd>::f32x4, 4>;

    let a = RA::indexed();
    let b = RA::map(RA::indexed(), |v| v + 100.0);

    run_swizzle_tests::<RA>(a, b);
}

#[test]
fn test_regular_swizzle() {
    type R = <X86V3 as Simd>::f32x16;

    let a = R::indexed();
    let b = R::map(a, |v| v + 100f32);

    run_swizzle_tests::<R>(a, b);
}

/// A macro to generate a test case for `precomputed_permutev`.
/// It creates an anonymous struct, implements the trait, and compares
/// the hardware intrinsic result against the known-good scalar result.
macro_rules! assert_permute_eq {
    ($reg_ty:ty, $chunks_ty:ty, $val:expr, [$($idx:expr),* $(,)?]) => {{
        let expected = <$reg_ty>::scalar_permutev($val, generic_array::arr![$($idx),*]);
        let val = Vector::<$reg_ty>($val);
        let actual = thermite::swizzle!(val, [$($idx),*]).0;

        assert_eq!(
            <$reg_ty>::as_array(&expected),
            <$reg_ty>::as_array(&actual),
            "Permute failed for indices: {:?}", generic_array::arr![$($idx),*]
        );
    }};
}

/// A macro to generate a test case for `precomputed_swizzle`.
macro_rules! assert_swizzle_eq {
    ($reg_ty:ty, $chunks_ty:ty, $a:expr, $b:expr, [$($idx:expr),* $(,)?]) => {{
        let expected = <$reg_ty>::scalar_swizzle($a, $b, generic_array::arr![$($idx),*]);
        let val_a = Vector::<$reg_ty>($a);
        let val_b = Vector::<$reg_ty>($b);
        let actual = thermite::swizzle!(val_a, val_b, [$($idx),*]).0;

        assert_eq!(
            <$reg_ty>::as_array(&expected),
            <$reg_ty>::as_array(&actual),
            "Swizzle failed for indices: {:?}", generic_array::arr![$($idx),*]
        );
    }};
}

// ----------------------------------------------------------------------------
// Test Suites
// ----------------------------------------------------------------------------

/// Runs tests assuming an ArrayRegister of 2 chunks, each with 4 lanes (8 lanes total).
/// Replace `R4` with your actual 4-lane register type (e.g., `f32x4` or `i32x4`).
pub fn test_array_register_2x4<R: SwizzleRegister<Lanes = typenum::U8>>(val_a: Storage<R>, val_b: Storage<R>)
where
    R::Element: PartialEq + core::fmt::Debug,
{
    // --- Permute Tests (Single Register, Indices 0..7) ---

    // 1. Identity
    assert_permute_eq!(R, Chunks, val_a, [0, 1, 2, 3, 4, 5, 6, 7]);

    // 2. Reversal (Stresses cross-chunk routing entirely)
    assert_permute_eq!(R, Chunks, val_a, [7, 6, 5, 4, 3, 2, 1, 0]);

    // 3. Chunk Swap (Moves hi chunk to lo chunk and vice versa)
    assert_permute_eq!(R, Chunks, val_a, [4, 5, 6, 7, 0, 1, 2, 3]);

    // 4. Single Element Broadcasts
    assert_permute_eq!(R, Chunks, val_a, [0, 0, 0, 0, 0, 0, 0, 0]);
    assert_permute_eq!(R, Chunks, val_a, [4, 4, 4, 4, 4, 4, 4, 4]);
    assert_permute_eq!(R, Chunks, val_a, [7, 7, 7, 7, 7, 7, 7, 7]);

    // 5. Interleaved (Stresses `blendv` mask merging)
    assert_permute_eq!(R, Chunks, val_a, [0, 4, 1, 5, 2, 6, 3, 7]);

    // --- Swizzle Tests (Double Register, Indices 0..15) ---

    // 1. Identity (Extract exactly `a` then exactly `b`)
    assert_swizzle_eq!(R, Chunks, val_a, val_b, [0, 1, 2, 3, 4, 5, 6, 7]);

    // 2. Full Reversal (b reversed, then a reversed)
    assert_swizzle_eq!(R, Chunks, val_a, val_b, [15, 14, 13, 12, 11, 10, 9, 8]);

    // 3. Cross-Register Broadcasts
    assert_swizzle_eq!(R, Chunks, val_a, val_b, [0, 0, 0, 0, 0, 0, 0, 0]); // Only touches a[0]
    assert_swizzle_eq!(R, Chunks, val_a, val_b, [15, 15, 15, 15, 15, 15, 15, 15]); // Only touches b[7]

    // 4. Interleaved A and B
    assert_swizzle_eq!(R, Chunks, val_a, val_b, [0, 8, 1, 9, 2, 10, 3, 11]);

    // 5. Out of bounds verification (Tests your `max_idx` wrapping/clamping)
    assert_swizzle_eq!(R, Chunks, val_a, val_b, [16, 17, 31, 32, 100, 255, 0, 1]);
}

#[test]
fn test_constant_swizzle() {
    type R = ArrayRegister<<X86V3 as Simd>::i64x2, 4>;

    let a = R::indexed();
    let b = R::map(a, |v| v + 100);

    test_array_register_2x4::<R>(a, b);
}
