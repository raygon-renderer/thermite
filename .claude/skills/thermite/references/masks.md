# Masks

Comparisons return a `Mask`. `Mask<R>` wraps `Storage<R::Mask>`: full-width
all-ones/all-zeros lanes pre-AVX512, a k-register on AVX-512. Trait:
`GenericMask` (`crates/thermite/src/mask.rs:66`), prelude re-exported.

```rust
use thermite::prelude::*;

let m: V::Mask = a.cmp_lt(b);   // from PartialOrdVector: cmp_lt/le/gt/ge/eq/ne
```

## GenericMask API

```rust
m.all()  -> bool          // every lane true
m.any()  -> bool          // some lane true
m.none() -> bool          // no lane true  (== !any)

// scanning: turn a mask into find/count primitives (build on native_bitmask)
m.first_set() -> Option<usize>   // index of lowest true lane; with cmp_eq this is a SIMD memchr
m.last_set()  -> Option<usize>   // index of highest true lane
m.count_set() -> usize           // number of true lanes (popcount)

m.select(t, f) -> S       // per-lane: mask ? t : f   (one instruction; S: GenericSelectable)

// bitwise combine
m1 & m2     m1 | m2     m1 ^ m2     !m     m1.bitandnot(m2)   // m1 & !m2 (register layer is reversed)
Mask::ternlog::<IMM>(a, b, c)

// constants
Mask::TRUTHY   // all true       Mask::FALSY    // all false (also Default)

// bit representations
m.native_bitmask() -> Option<u64>   // packed bits if the backend supports it
m.bitmask() -> BitArray             // always works (software fallback; needs `bitvec` feature)

// reinterpret a mask to another vector's mask type (same lane count / compatible width)
let m2: OtherVec::Mask = m.cast::<_>();    // via CastMask
m.swap(&mut a, &mut b)                      // conditionally swap lanes of a and b
```

## Selecting and masking: the idioms

```rust
// Branchless choice between two computed vectors (preferred over data-dependent branches):
let r = a.cmp_lt(b).select(a, b);          // == a.min(b) but generalizes to any pair

// Zero-out helpers on the VECTOR (note: zero-if-false is `zz`, zero-if-true is `nz`):
v.zz(mask)    // keep where mask true, else 0
v.nz(mask)    // keep where mask false, else 0

// Masked operation variants put the result only where the mask is true:
a.add_c(mask, b)         // mask ? a+b : a
v.sqrt_z(mask)           // mask ? sqrt(v) : 0
```

See [vector-api.md](vector-api.md) for the full `_c`/`_m`/`_z` system.

## Branch only when it pays

SIMD lanes diverge, so a real branch helps only when *all* lanes agree, and the
math policy may forbid branching anyway (`P::POLICY.avoid_branching`). Gate an
expensive skip behind a runtime `.all()` / `.any()` check; otherwise prefer
`select`. Watch for `0 * NaN = NaN` poisoning when one branch is conditionally
invalid -- use `select` to keep the good lane instead of arithmetic that touches
the NaN. (See [performance.md](performance.md) section 9.)

## Casting masks between types

Masks for same-lane-count/width vectors (`f32x8` vs `i32x8`) convert free via
`m.cast()` (`CastMask`) -- e.g. float-compare mask applied to the integer view.
`GenericVector::Mask` is bounded `CastMask<Unsigned::Mask> + CastMask<Signed::Mask>`,
so these casts always exist within a vector's own family.
