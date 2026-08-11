# thermite-sort

Vectorized quicksort over slices, plus the pieces it is built from: a branchless
bidirectional partition, a sorting-network base case that finishes short ranges
entirely in registers, run detection at entry, and `ancestor_pivot` handling for
duplicate runs. Unstable and allocation-free, except where noted.

```rust
use thermite::backend::x86_v3::i32x8;
use thermite_sort::{sort, sort_by, Descending};

let mut keys = [5, 2, 9, 2, 7, 1, 8, 3];
sort::<i32x8>(&mut keys);                  // ascending
sort_by::<i32x8, Descending>(&mut keys);   // same cost - see "order is free"
```

## 1. Slice sort

```rust
sort::<V>(&mut [V::Element])               // ascending
sort_by::<V, O: SortOrder>(&mut [V::Element])
```

**Choosing `V`: the widest NATIVE vector for the ISA** - `i32x8` on AVX2,
`i32x4` on SSE4.2/NEON/wasm. Do **not** reach for a wider `ArrayRegister`
composite (`i32x8` on a 128-bit tier): the hot ops are `compress` and the merge
swizzles, neither of which scales across sub-registers the way `min`/`max`
does, so a two-chunk composite measured ~0.71x of the native width it is built
from. There is no auto-dispatching entry point - the caller names `V`, usually
under a `dispatch_dyn!`.

ISAs where the vector path cannot pay for itself route to `core`'s
`sort_unstable` at compile time (`Scalar`, `Unknown`, and x86-v1/SSE2, whose
`compress` polyfill has no byte shuffle to build on).

**NaN**: swept to the **back** of the slice for both orders, bit patterns
preserved, ordered prefix sorted in front. Always a permutation of the input.
Integer sorts compile none of the sweep (gated on `Element::HAS_UNORDERED`).

## 2. Key-value sort (two parallel slices)

```rust
use thermite::backend::x86_v3::{u32x4, u64x4};
use thermite_sort::{sort_kv, sort_kv_by};

sort_kv::<VK, PM>(&mut [VK::Element], &mut [PM::Element])          // ascending
sort_kv_by::<VK, PM, O>(&mut keys, &mut vals)
sort_kv_by::<u64x4, u32x4, Ascending>(&mut keys, &mut idx)         // "K64V32"
sort_kv_by::<u64x4, u64x4, Ascending>(&mut keys, &mut payload)     // "K64V64"
```

Keys decide the order; `vals[i]` stays glued to `keys[i]`. The two streams stay
**separate slices (SoA)** rather than interleaved records, which is what keeps
compares plain full-width ops on the key vector and makes the payload's element
width a free parameter: `PM` narrower than `VK` halves payload memory traffic
(in registers the payload rides at key width, converting only at load/store,
one instruction per vector).

- **Integer keys and payloads only**, as a bound. Float keys go through the
  `cached_key` transforms below, which also keeps NaN out of this path
  entirely. (The load/store conversion is a numeric cast: integers round-trip,
  a float conversion would rewrite the payload's bits.)
- Signed pairs with signed in the cast matrix: `i64x4 -> i32x4` exists,
  `i64x4 -> u32x4` does not.
- Unstable. For a stable key+index sort use `cached_key` (below).
- Panics if the slice lengths differ.

## 3. Cached-key object sort

Sort arbitrary `T` by a key computed **once per object**, `std` only (needs
scratch). Two tiers:

```rust
use thermite_sort::cached_key::{
    sort_by_cached_key, sort_by_cached_key_in, sort_by_cached_key_u64,
    ordered_key_i32, ordered_key_f32, ordered_key_f64, ordered_bits_f64,
};

// u32 key, STABLE, ~2x faster: packs (key << 32) | index into one u64 and
// sorts that. The unique index makes ties break by original position.
sort_by_cached_key::<u64x4, _, _>(&mut objs, |o| ordered_key_f64(o.cost));
sort_by_cached_key_in::<u64x4, _, _>(&mut objs, &mut scratch, key);  // no_std-friendly

// u64 key, UNSTABLE, exact: rides the key-value sort (keys + u32 index).
sort_by_cached_key_u64::<u64x4, u32x4, _, _>(&mut objs, |o| ordered_bits_f64(o.cost));
```

**Key contract: a `u32`/`u64` whose UNSIGNED order is the sort order.** The
transforms map other types onto that, as a radix sort would:

| fn | maps | notes |
|---|---|---|
| `ordered_key_i32` | `i32 -> u32` | one xor of the sign bit |
| `ordered_key_f32` | `f32 -> u32` | IEEE total order, exact |
| `ordered_bits_f64` | `f64 -> u64` | IEEE total order, exact |
| `ordered_key_f64` | `f64 -> u32` | truncated: sign + full exponent + top 20 mantissa bits (~1e-6 relative); collapsed keys keep original order |

These are the **scalar counterpart of core's vector
`FloatVectorWithBits::total_order()`**, which yields the same order as *signed*
bits: `ordered_key_f32(x) == (x.total_order() as u32) ^ 0x8000_0000` (pinned by
`tests/cached_key.rs::matches_core_total_order`). Reach for `total_order()` when
generating keys a vector at a time.

Descending is a key transform too: `!ordered_key_f32(x)` reverses the order and
**stays stable** (ties still break by ascending index), where running the packed
sort with `Descending` would reverse the tiebreak as well.

Costs: `n` words of scratch (so not allocation-free), plus a random-access
permutation-apply pass that grows with `size_of::<T>()`. Slices are limited to
`u32::MAX` elements (the index must fit the low word).

## 4. Building blocks (usable on their own)

```rust
partition::<V, O>(&mut keys, pivot) -> usize      // branchless bidirectional, in place
partition_equal::<V, O>(...)                      // equals go LEFT (duplicate runs)
partition_unordered_last::<V>(...)                // NaN sweep, same kernel
sort_columns_{2,4,8,16}[_by]::<V[, O]>([V; N])    // columnar network across vectors
merge::sort_block_{2,4,8,16}::<V, O>([V; N])      // column-sorted -> one sorted run
base::base_case::<V, O>(&mut keys)                // <= 16 * LANES, all in registers
runs::{is_ordered, is_reverse_ordered, has_unordered}::<V[, O]>(&keys)
```

`MergeVector` (= `NumericVector + SwizzleVector`) is the bound the block sort
needs; the merges cover `LANES <= 16`.

## 5. Core primitives (in `thermite::sort`, not this crate)

```rust
pub trait SortOrder { const IS_ASCENDING: bool; first/last; vector_first/vector_last; ... }
pub struct Ascending;  pub struct Descending;   // re-exported by thermite-sort

R::sort_by::<O>(v)  R::bitonic_clean_by::<O>(v)  // NumericRegister methods, backend-overridable
sort_lanes_by_key::<V, O, K>(v)                  // lane sort routed by a KEY comparison
bitonic_clean_lanes_by_key::<V, O, K>(v)         // ... for already-bitonic input
```

**Order is free.** Flipping every comparator turns an ascending network into a
descending one at identical instruction count - only which side of the blend
receives `min` vs `max` changes. Never sort ascending and `reverse()`.

`sort_lanes_by_key` + `SortKey` is how **composite** vectors sort: a
compare-exchange derives one routing mask from the key and `select`s every
component with it, so `Dual` sorts by its primal, `Compensated` and `Complex`
lexicographically. Two rules, both load-bearing: each compare-exchange needs
**two strict key tests** (one mask plus its negation duplicates a composite on
key ties), and the key must be a **static trait method** - an `F: Fn` bound left
the comparison as out-of-line base-ISA calls.

## 6. What to expect (znver3, AVX2, against `slice::sort_unstable`)

| input | speedup |
|---|---|
| i32 uniform | 2.8-2.9x |
| i32 few-unique | ~3x |
| u64 uniform | ~2.5x |
| sorted / reverse-sorted | 1.3-3.8x, and 10-20x this sort's own random-input rate |
| key-value (vs sorting zipped tuples) | ~1.5x |

Two shapes to know. **Sorted and reverse-sorted input finish in a single scan**
- the entry run detectors catch them before any partitioning, which is why
those rows are an order of magnitude above the random-input rate rather than a
few percent. **Duplicate-heavy input is also fast**, via `ancestor_pivot`
(equals sent left consume a whole run of duplicates per pass) plus a
uniform-remainder harvest in the equal-run partition, so a k-distinct input
costs about `O(n*k)` instead of degrading.

One shape that is *not* fast: almost-sorted input, where a single early
violation defeats both run detectors and the full recursion runs anyway. Expect
uniform-input numbers there, not sorted-input numbers.

## Gotchas

- **`V` must be a native register width**, not an `ArrayRegister` composite -
  see section 1. This is the most likely way to leave 30% on the floor.
- **Doctests that run a sort need `no_run`.** rustdoc compiles doctest bodies
  *unoptimized even under `--release`*, and at `opt-level = 0` the inlined
  kernel becomes one ~1 MiB stack frame, overflowing the Windows main thread's
  1 MiB reserve. `opt-level = 1` brings it under 32 KiB. Same caveat for any
  fully-unoptimized downstream build calling the sort from a small stack.
- **`sort` is unstable**; only `sort_by_cached_key` (the packed u32 tier) is
  stable, and only because the index tiebreak makes it so.
- **Everything is allocation-free except the `cached_key` entry points.** Use
  `sort_by_cached_key_in` to supply the scratch.
- **The `stats` feature** compiles structural counters (partitions, passes over
  `n`, equal runs, base cases) into the real recursion; `examples/counts` reads
  them. Wall-clock noise here is +-7%, so pass counts are the only trustworthy
  signal for pivot/partition changes - and the counters must live in the
  shipped code, never in a reimplementation.
