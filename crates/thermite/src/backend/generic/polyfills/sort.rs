use generic_array::{GenericArray, typenum};

use super::*;

/// Helper: Permute, Compare, and Blend.
///
/// 1. Permutes `v` using `PERM_MASK`.
/// 2. Compares original `v` vs permuted version.
/// 3. Blends Min/Max back together: `0` in `BLEND_MASK` selects Min, `1` selects Max.
#[inline(always)]
fn cmp_merge<R, const PERM_MASK: i32, const BLEND_MASK: i32>(v: Storage<R>) -> Storage<R>
where
    R: NumericRegister + BlendRegister + PermuteRegister,
{
    let v_shuf = R::permute::<PERM_MASK>(v);
    let v_min = R::min(v, v_shuf);
    let v_max = R::max(v, v_shuf);
    R::blend::<BLEND_MASK>(v_min, v_max)
}

/// Sorts a 2-element vector (e.g. f64x2, u64x2).
/// Network: Single Comparator (Depth: 1, Size: 1)
#[inline(always)]
pub fn sort_2<R>(v: Storage<R>) -> Storage<R>
where
    R: NumericRegister<Lanes = typenum::U2> + BlendRegister + PermuteRegister,
{
    // 1. Swap elements: [1, 0]
    // Mask 0x01 (binary 01) swaps index 0 and 1.
    let v_shuf = R::permute::<0x01>(v);

    let v_min = R::min(v, v_shuf);
    let v_max = R::max(v, v_shuf);

    // 2. Blend Min and Max
    // We want Index 0 to have Min (Mask bit 0 = 0)
    // We want Index 1 to have Max (Mask bit 1 = 1)
    // Mask: 0b10 -> 0x02
    R::blend::<0x02>(v_min, v_max)
}

/// Sorts a 4-lane register (e.g., f64x4 or u64x4).
/// Uses an Optimal Sorting Network (Depth 3, Size 5).
#[inline(always)]
pub fn sort_4<R>(v: Storage<R>) -> Storage<R>
where
    R: NumericRegister<Lanes = typenum::U4> + BlendRegister + PermuteRegister,
{
    // Layer 1: Swap Neighbors (0,1) and (2,3)
    // Permute: [1, 0, 3, 2] -> _MM_SHUFFLE(2, 3, 0, 1) -> 0xB1
    // Blend: Min at 0, 2; Max at 1, 3 -> 0b1010 -> 0xA
    let v = cmp_merge::<R, 0xB1, 0xA>(v);

    // Layer 2: Swap Pairs (0,2) and (1,3)
    // Permute: [2, 3, 0, 1] -> _MM_SHUFFLE(1, 0, 3, 2) -> 0x4E
    // Blend: Min at 0, 1; Max at 2, 3 -> 0b1100 -> 0xC
    let v = cmp_merge::<R, 0x4E, 0xC>(v);

    // Layer 3: Swap Inner (1,2)
    // Permute: [0, 2, 1, 3] -> _MM_SHUFFLE(3, 1, 2, 0) -> 0xD8
    // Blend: Min at 1; Max at 2 -> 0b0100 -> 0x4
    cmp_merge::<R, 0xD8, 0x4>(v)
}

/// Sorts an 8-lane register (e.g., f32x8 or u32x8).
/// Strategy: ILP Parallel Sort -> Cross-Lane Merge.
#[inline(always)]
pub fn sort_8<R>(v: Storage<R>) -> Storage<R>
where
    R: NumericRegister<Lanes = typenum::U8> + BlendRegister + PermuteRegister,
{
    // --- PHASE 1: Independent Local Sorts (ILP) ---
    // We sort the lower 4 lanes and upper 4 lanes independently.
    // We assume `R::permute` acts like `_mm256_shuffle_ps`, applying the
    // permutation to both 128-bit halves identically.

    // 1. Swap Neighbors: (0,1), (2,3), (4,5), (6,7)
    // Permute: [1, 0, 3, 2] repeating -> 0xB1
    // Blend: Max at 1, 3, 5, 7 -> 0xAA
    let v = cmp_merge::<R, 0xB1, 0xAA>(v);

    // 2. Swap Stride 2: (0,2), (1,3), (4,6), (5,7)
    // Permute: [2, 3, 0, 1] repeating -> 0x4E
    // Blend: Max at 2, 3, 6, 7 -> 0xCC
    let v = cmp_merge::<R, 0x4E, 0xCC>(v);

    // 3. Swap Inner: (1,2), (5,6)
    // Permute: [0, 2, 1, 3] repeating -> 0xD8
    // Blend: Max at 2 and 6 -> 0x44
    let v = cmp_merge::<R, 0xD8, 0x44>(v);

    // --- PHASE 2: Cross-Lane Merge ---
    // Current state: Sorted [A,B,C,D] and Sorted [E,F,G,H].
    // To perform a Bitonic Merge, we effectively need one sequence reversed.

    // 4. Reverse the Upper Lane: [A, B, C, D | H, G, F, E]
    // We use permutev because a simple immediate shuffle usually cannot
    // reorder specific indices purely in the high lane without AVX512.
    // Indices: 0, 1, 2, 3 (Keep Low), 7, 6, 5, 4 (Reverse High)
    let v = R::permutev(v, GenericArray::from_array([0, 1, 2, 3, 7, 6, 5, 4]));

    // 5. Cross-Lane Swap (Stride 4)
    // Compare Low Half vs High Half: (0,7), (1,6), (2,5), (3,4)
    // We need to bring the high elements to the low positions and vice versa.
    // Indices: 4, 5, 6, 7, 0, 1, 2, 3
    let v_shuf = R::permutev(v, GenericArray::from_array([4, 5, 6, 7, 0, 1, 2, 3]));

    let v_min = R::min(v, v_shuf);
    let v_max = R::max(v, v_shuf);

    // Blend: Keep Min in Lower Half (0-3), Max in Upper Half (4-7)
    // Mask: 0b11110000 -> 0xF0
    let v = R::blend::<0xF0>(v_min, v_max);

    // --- PHASE 3: Final Cleanup ---

    // 6. Stride 2 Cleanup
    // Permute: [2, 3, 0, 1] repeating -> 0x4E
    // Blend: Max at 2, 3, 6, 7 -> 0xCC
    let v = cmp_merge::<R, 0x4E, 0xCC>(v);

    // 7. Stride 1 Cleanup
    // Permute: [1, 0, 3, 2] repeating -> 0xB1
    // Blend: Max at 1, 3, 5, 7 -> 0xAA
    cmp_merge::<R, 0xB1, 0xAA>(v)
}

#[inline(always)]
pub fn sort_any<R: NumericRegister>(mut value: Storage<R>) -> Storage<R> {
    let s = R::as_array_mut(&mut value);

    /// Compare-and-Swap: The atomic primitive of sorting networks.
    /// LLVM optimizes this to `cmp` + `cmov` (Conditional Move), which is branchless.
    #[inline(always)]
    fn cas<T: PartialOrd>(s: &mut [T], i: usize, j: usize) {
        // Note: slice indexing checks bounds.
        // For maximal performance, you could use `get_unchecked` if unsafe is permitted,
        // but the optimizer often elides checks in fixed-size networks anyway.
        if s[i] > s[j] {
            s.swap(i, j);
        }
    }

    #[rustfmt::skip]
    let () = match s.len() {
        2 => cas(s, 0, 1),
        4 => {
            cas(s, 0, 1); cas(s, 2, 3); // Layer 1
            cas(s, 0, 2); cas(s, 1, 3); // Layer 2
            cas(s, 1, 2);               // Layer 3
        },
        // For N=8 or others, Insertion Sort is compact and very fast for N < 20
        _ => {
            for i in 1..s.len() {
                let mut j = i;
                // The compiler unrolls this loop well for small fixed bounds
                while j > 0 && s[j - 1] > s[j] {
                    s.swap(j - 1, j);
                    j -= 1;
                }
            }
        }
    };

    value
}
