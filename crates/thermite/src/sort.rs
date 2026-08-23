//! Sort order: which direction a comparator points.
//!
//! A sorting network is a fixed sequence of compare-exchanges, and a
//! compare-exchange is fully described by "which of the two values goes to the
//! lower index". [`SortOrder`] is that choice, as a compile-time parameter, so
//! one network body serves every direction.
//!
//! # The direction flip is free
//!
//! Flipping every comparator in an *ascending* network yields a *descending*
//! one - for any network, not just the bitonic ones here. If `f` is an
//! order-reversing bijection then `min(f a, f b) = f(max(a, b))`, so a flipped
//! network `N'` satisfies `N'(x) = f^-1(N(f(x)))`: the same permutation network
//! run on reversed values, which is `x` sorted descending.
//!
//! At the machine level the flip is not even an extra instruction. A network
//! layer is `permute`, `min`, `max`, `blend`; reversing it swaps which of the
//! `min`/`max` results feeds the blend's true arm. Measured on znver3 (llvm-mca,
//! `crates/sortasm` probe), `f32x8`:
//!
//! | | insns | `vminps` | `vmaxps` | shuffles | RThroughput | latency |
//! |---|---|---|---|---|---|---|
//! | [`Ascending`] | 29 | 6 | 6 | 8 | 9.0 | **43 cyc** |
//! | [`Descending`] | 29 | 6 | 6 | 8 | 9.0 | **43 cyc** |
//! | ascending then `reverse` | 31 | 6 | 6 | 11 | 8.5 | 50 cyc |
//!
//! So a descending sort costs exactly nothing, where post-processing an
//! ascending one with [`reverse`](crate::register::Register::reverse) costs 3
//! shuffles and 16% latency. The counts of `min` and `max` are unchanged in both
//! directions because each layer issues one of each regardless; only the blend
//! operands swap.
//!
//! # Generic methods, not a generic trait
//!
//! The obvious shape parameterizes the trait by the register
//! (`trait SortOrder<R: NumericRegister>`). It serves native registers fine and
//! then hard-blocks on the first *delegating* one: inside
//! `ArrayRegister::<R, N>::sort_by::<O>`, calling `R::sort_by::<O>` needs
//! `O: SortOrder<R>`, which `O: SortOrder<Self>` does not imply - and the bound
//! cannot be added, because the method signature is fixed by the trait
//! declaration, which has no `R` to name. Each nesting level would want another
//! bound.
//!
//! Generic *methods* keep the bound flat: `O: SortOrder` is the whole
//! requirement at every layer, whatever the register, however deeply nested.

use crate::register::{NumericRegister, Storage};
use crate::vector::NumericVector;

/// The direction a comparator points, as a compile-time parameter.
///
/// Implementors are zero-sized markers ([`Ascending`], [`Descending`]) selected
/// with turbofish - `R::sort_by::<Descending>(v)`. Both methods must be a
/// consistent pair: `first` and `last` are the two halves of one
/// compare-exchange, so for any `a`, `b` the multiset `{first(a,b), last(a,b)}`
/// must equal `{a, b}`, and `first(a,b)` must not come after `last(a,b)` under
/// the order. Violating that does not just misorder - it duplicates and drops
/// values, because a network never re-reads what it overwrote.
///
/// Deliberately only these two methods. They are the whole of what a *network*
/// needs; the padding sentinels (a value that sorts after everything, for
/// filling a partial register) and the members a partition-based sort would want
/// (`compare`, `prev_value`) get added when something calls them.
pub trait SortOrder {
    /// Whether this order is smallest-first.
    ///
    /// **Must agree with [`first`](Self::first) and [`last`](Self::last)** - it
    /// is the same fact stated a second way, and nothing checks that the two
    /// statements match.
    ///
    /// It exists because the scalar fallback
    /// ([`sort_any`](crate::backend::generic::polyfills::sort::sort_any), used
    /// by any register with no network for its lane count) sorts *elements*
    /// through `PartialOrd`, where the register-level `first`/`last` cannot
    /// reach. Rather than widen this trait with a scalar comparator pair used by
    /// one slow path, that path sorts ascending and reverses on this flag. The
    /// cost is one permute on a body that is already quadratic.
    const IS_ASCENDING: bool;

    /// The value that belongs at the **lower** index of a comparator pair.
    ///
    /// Named for the position rather than for `min`, because the two stop
    /// coinciding as soon as the order is anything but ascending.
    fn first<R: NumericRegister>(a: Storage<R>, b: Storage<R>) -> Storage<R>;

    /// The value that belongs at the **higher** index of a comparator pair.
    fn last<R: NumericRegister>(a: Storage<R>, b: Storage<R>) -> Storage<R>;

    /// [`first`](Self::first) at the vector layer.
    ///
    /// A second pair rather than one generic over both layers because
    /// [`NumericVector`] has no associated register type to route through (only
    /// its float/signed/unsigned sub-traits do), and `Storage<R>` is a
    /// projection, so no helper trait can cover both. Keeping the pairs on one
    /// trait is what stops a marker from meaning ascending at one layer and
    /// descending at the other.
    fn vector_first<V: NumericVector>(a: V, b: V) -> V;

    /// [`last`](Self::last) at the vector layer.
    fn vector_last<V: NumericVector>(a: V, b: V) -> V;

    /// A value that sorts **after** every real input under this order - the
    /// padding sentinel.
    ///
    /// Filling the unused lanes of a partial register with this lets a fixed
    /// network sort a run shorter than the register: the sentinels sort to the
    /// tail and are never stored back. Ascending wants the order maximum
    /// (`+inf` for floats, not `MAX`); descending wants the order minimum, which
    /// is why this lives on the order rather than on the register.
    ///
    /// **NaN is not covered.** No value sorts past NaN because NaN is unordered,
    /// so a float sort must remove NaN before the network runs - see
    /// `thermite_sort`. Padding with `+inf` alongside a NaN in the data gives a
    /// backend-dependent result, since `min`/`max` NaN semantics legitimately
    /// differ across ISAs (the differential suite carries a `Tol::ExactOrNan`
    /// for precisely this).
    #[inline(always)]
    fn last_value<V: NumericVector>() -> V {
        V::splat(if const { Self::IS_ASCENDING } {
            <V::Element as crate::element::Element>::ORDER_MAX
        } else {
            <V::Element as crate::element::Element>::ORDER_MIN
        })
    }

    /// A value that sorts **before** every real input under this order. The
    /// mirror of [`last_value`](Self::last_value); same NaN caveat.
    #[inline(always)]
    fn first_value<V: NumericVector>() -> V {
        V::splat(if const { Self::IS_ASCENDING } {
            <V::Element as crate::element::Element>::ORDER_MIN
        } else {
            <V::Element as crate::element::Element>::ORDER_MAX
        })
    }
}

/// Sort smallest-first. The default everywhere a direction is not named.
pub struct Ascending;

/// Sort largest-first.
///
/// Costs exactly the same as [`Ascending`] - see the module docs.
pub struct Descending;

impl SortOrder for Ascending {
    const IS_ASCENDING: bool = true;

    #[inline(always)]
    fn first<R: NumericRegister>(a: Storage<R>, b: Storage<R>) -> Storage<R> {
        R::min(a, b)
    }

    #[inline(always)]
    fn last<R: NumericRegister>(a: Storage<R>, b: Storage<R>) -> Storage<R> {
        R::max(a, b)
    }

    #[inline(always)]
    fn vector_first<V: NumericVector>(a: V, b: V) -> V {
        a.min(b)
    }

    #[inline(always)]
    fn vector_last<V: NumericVector>(a: V, b: V) -> V {
        a.max(b)
    }
}

impl SortOrder for Descending {
    const IS_ASCENDING: bool = false;

    #[inline(always)]
    fn first<R: NumericRegister>(a: Storage<R>, b: Storage<R>) -> Storage<R> {
        R::max(a, b)
    }

    #[inline(always)]
    fn last<R: NumericRegister>(a: Storage<R>, b: Storage<R>) -> Storage<R> {
        R::min(a, b)
    }

    #[inline(always)]
    fn vector_first<V: NumericVector>(a: V, b: V) -> V {
        a.max(b)
    }

    #[inline(always)]
    fn vector_last<V: NumericVector>(a: V, b: V) -> V {
        a.min(b)
    }
}

// ---------------------------------------------------------------------------
// Compare-exchange layer descriptions
//
// Pure index math over a lane count - no register, no vector, no layer. Both
// the register-layer networks (`backend::generic::polyfills::sort`) and the
// vector-layer block sort (`thermite_sort::merge`) build their layers from
// these, and there is no reason for two copies of the arithmetic.
// ---------------------------------------------------------------------------

use crate::register::SwizzleIndices;
use generic_array::{ArrayLength, GenericArray};

/// Lane `i` faces lane `i ^ K`: the partner set of a distance-`K`
/// compare-exchange, and the shuffle behind Highway's `SortPairsDistance{K}` /
/// `SwapAdjacentPairs` / `SwapAdjacentQuads`.
///
/// `K >= N` is clamped to the identity rather than left out of range. Such a
/// stage is dead code that a caller's `if const` ladder never reaches, but its
/// `INDICES` are still const-evaluated, and an out-of-range swizzle index is
/// undefined behavior by contract rather than merely unused.
pub struct XorIdx<const K: usize, N>(core::marker::PhantomData<N>);

impl<const K: usize, N: ArrayLength> SwizzleIndices<N> for XorIdx<K, N> {
    const INDICES: GenericArray<u32, N> = const {
        // `GenericArray<u32, N>` has no const literal constructor for a generic
        // `N`, so zero-initialize (all-zero `u32` is valid) and fill in place.
        let mut idxs: GenericArray<u32, N> = unsafe { core::mem::zeroed() };
        let ptr = &mut idxs as *mut GenericArray<u32, N> as *mut u32;

        let k = if K < N::USIZE { K } else { 0 };

        let mut i = 0;
        while i < N::USIZE {
            unsafe { *ptr.add(i) = (i ^ k) as u32 };
            i += 1;
        }
        idxs
    };
}

/// Lane `i` faces its mirror within its contiguous group of `K` lanes - the
/// shuffle behind Highway's `ReverseKeys{K}` and `SortPairsReverse{K}`.
///
/// `K > N` is clamped to a full reverse; see [`XorIdx`] for why that matters.
pub struct RevIdx<const K: usize, N>(core::marker::PhantomData<N>);

impl<const K: usize, N: ArrayLength> SwizzleIndices<N> for RevIdx<K, N> {
    const INDICES: GenericArray<u32, N> = const {
        let mut idxs: GenericArray<u32, N> = unsafe { core::mem::zeroed() };
        let ptr = &mut idxs as *mut GenericArray<u32, N> as *mut u32;

        let k = if K < N::USIZE { K } else { N::USIZE };

        let mut i = 0;
        while i < N::USIZE {
            unsafe { *ptr.add(i) = ((i & !(k - 1)) | (k - 1 - (i & (k - 1)))) as u32 };
            i += 1;
        }
        idxs
    };
}

/// Lanes whose index has `bit` set - the lanes that receive
/// [`SortOrder::last`] of their pair. `bit == 0` means no lane does.
///
/// Bits at or above the lane count are ignored by
/// [`from_native_bitmask`](crate::mask::GenericMask::from_native_bitmask), so
/// one 64-bit constant serves every width.
pub const fn keep_bits(bit: usize) -> u64 {
    let mut mask = 0u64;
    if bit == 0 {
        return mask;
    }
    let mut i = 0;
    while i < 64 {
        if i & bit != 0 {
            mask |= 1 << i;
        }
        i += 1;
    }
    mask
}

/// One in-register compare-exchange layer: a partner permutation, plus the lane
/// mask saying which side of each pair keeps the later element.
///
/// A network layer is `permute` + `first` + `last` + `blend`, and this is the
/// half of it that varies. Splitting it out as a type rather than two const
/// parameters is what lets [`RevPairs`] derive its mask from `K / 2` - a const
/// *expression*, which is stable, where a const-generic expression is not.
pub trait PairStage<N: ArrayLength> {
    /// Where each lane finds its partner.
    type Indices: SwizzleIndices<N>;

    /// Bit `i` set means lane `i` receives [`SortOrder::last`] of its pair.
    const KEEP_LAST: u64;
}

/// Compare-exchange lane `i` with lane `i ^ K` (Highway's
/// `SortPairsDistance{K}`). The high lane of each pair keeps the last.
///
/// The halving strides `LANES/2 .. 1` of this are exactly a bitonic cleanup.
pub struct Distance<const K: usize>;

impl<const K: usize, N: ArrayLength> PairStage<N> for Distance<K> {
    type Indices = XorIdx<K, N>;
    const KEEP_LAST: u64 = keep_bits(K);
}

/// Compare-exchange each lane with its mirror in its group of `K` (Highway's
/// `SortPairsReverse{K}`).
///
/// The upper *half* of each reversed group keeps the last, so the blend mask is
/// the distance-`K/2` one. That identity is what collapses the two families of
/// odd-even blend (`OddEvenKeys` / `OddEvenPairs` / `OddEvenQuads`) onto one
/// [`keep_bits`].
pub struct RevPairs<const K: usize>;

impl<const K: usize, N: ArrayLength> PairStage<N> for RevPairs<K> {
    type Indices = RevIdx<K, N>;
    const KEEP_LAST: u64 = keep_bits(K / 2);
}

// ---------------------------------------------------------------------------
// Key-driven lane sorts at the vector layer
//
// The register-layer networks move *elements* through `min`/`max`, which is
// only correct when a lane is one value. A composite vector (autodiff dual,
// compensated pair, complex) is several component vectors whose lanes move as
// a unit under a KEY comparison - the primal for `Dual`, the value for
// `Compensated`, lexicographic (re, im) for `Complex`. These run the exact
// same stage sequence (`RevPairs`/`Distance`, the depth-minimal construction
// behind `sort_lanes`), but each compare-exchange derives one *routing mask*
// from the key and applies it to every component with a single whole-vector
// `select` - which the composite's own `Mask::select` already does
// componentwise.
//
// The alternative - computing the sorting permutation of the key lanes and
// `permute`-ing every component once (an argsort) - costs fewer ops when the
// component count is large, since the network then only carries (key, index).
// At the component counts that exist today (2-5) the direct form is at parity
// or better and needs no index machinery; revisit if a many-component
// composite shows up hot.
// ---------------------------------------------------------------------------

use crate::mask::GenericMask;
use crate::vector::GenericVector;

/// The comparison a key-driven lane sort routes on: `key_lt(a, b)` masks the
/// lanes of `a` strictly before those of `b` under the type's sort key.
///
/// A static trait method rather than an `F: Fn` parameter ON PURPOSE. The
/// closure/fn-item form was measured leaving the comparison out of line - the
/// `Fn::call` shim through `&F` survived to codegen for the ternlog-carrying
/// lexicographic comparisons, one base-ISA `call` per network stage - and an
/// `#[inline(always)]` on a static method cannot be declined that way.
///
/// Requirements: a *strict* order test on the key alone (never the payload
/// components), consistent across lanes. Typically implemented by the
/// composite for itself (`impl SortKey<Self> for Self`) as its `cmp_lt` or a
/// component's `cmp_lt`.
pub trait SortKey<V: GenericVector> {
    /// Which lanes of `a` sort strictly before those of `b`, by the key.
    fn key_lt(a: V, b: V) -> V::Mask;
}

/// One key-driven compare-exchange layer: partner permutation, two strict key
/// comparisons, one routing select applied to the whole vector.
///
/// **Both sides of a pair need their own strict test.** With one mask `m =
/// lt(partner, self)` and the keep-last side taking `!m`, a key *tie* routes
/// the same composite to both lanes - one value duplicated, its partner
/// dropped, invisibly to any test whose payloads tie too. Two strict tests
/// (`lt(partner, self)` for keep-first lanes, `lt(self, partner)` for
/// keep-last) make every tied pair keep itself on both sides, so the layer is
/// a permutation for every input.
#[inline(always)]
fn key_stage<V, O, S, K>(v: V) -> V
where
    V: GenericVector + crate::swizzle::Swizzle<V::Lanes>,
    O: SortOrder,
    S: PairStage<V::Lanes>,
    K: SortKey<V>,
{
    let partner = v.permutev_const::<S::Indices>();

    let (m_first, m_last) = if const { O::IS_ASCENDING } {
        (K::key_lt(partner, v), K::key_lt(v, partner))
    } else {
        (K::key_lt(v, partner), K::key_lt(partner, v))
    };

    // A constant bitmask, so this folds to a materialized mask constant.
    let keep_last = V::Mask::from_native_bitmask(S::KEEP_LAST);
    let take_partner = (m_first & !keep_last) | (m_last & keep_last);
    take_partner.select(partner, v)
}

/// Sort the lanes of `v` in `O` order of `K`'s key. Whole lanes move
/// together: every component of a composite follows its key.
///
/// Same depth-minimal construction as the register-layer `sort_lanes` (the
/// widening tails at one chunk); covers power-of-two lane counts up to 16.
/// Ties by key keep the input's lane order within each compare-exchange (see
/// `key_stage`) but the sort as a whole is not stable.
#[inline(always)]
pub fn sort_lanes_by_key<V, O, K>(v: V) -> V
where
    V: GenericVector + crate::swizzle::Swizzle<V::Lanes>,
    O: SortOrder,
    K: SortKey<V>,
{
    // NOT a const assert: an `if const` gate in a caller does not stop this
    // from being monomorphized for wider vectors (guards do not prevent
    // monomorphization), so a hard compile-time check would break any caller
    // that gates and falls back. Callers must gate on `V::LANES <= 16`.
    debug_assert!(
        V::LANES <= 16 && V::LANES.is_power_of_two(),
        "sort_lanes_by_key covers power-of-two lane counts up to 16"
    );

    let v = if const { V::LANES >= 2 } {
        key_stage::<V, O, RevPairs<2>, K>(v)
    } else {
        v
    };
    let v = if const { V::LANES >= 4 } {
        let v = key_stage::<V, O, RevPairs<4>, K>(v);
        key_stage::<V, O, Distance<1>, K>(v)
    } else {
        v
    };
    let v = if const { V::LANES >= 8 } {
        let v = key_stage::<V, O, RevPairs<8>, K>(v);
        let v = key_stage::<V, O, Distance<2>, K>(v);
        key_stage::<V, O, Distance<1>, K>(v)
    } else {
        v
    };

    if const { V::LANES >= 16 } {
        let v = key_stage::<V, O, RevPairs<16>, K>(v);
        let v = key_stage::<V, O, Distance<4>, K>(v);
        let v = key_stage::<V, O, Distance<2>, K>(v);
        key_stage::<V, O, Distance<1>, K>(v)
    } else {
        v
    }
}

/// [`sort_lanes_by_key`] for an already-**bitonic** vector: the halving
/// compare-exchange strides `LANES/2 .. 1`. Garbage in, garbage out on
/// non-bitonic input, exactly like the register-layer `bitonic_clean_lanes`.
#[inline(always)]
pub fn bitonic_clean_lanes_by_key<V, O, K>(v: V) -> V
where
    V: GenericVector + crate::swizzle::Swizzle<V::Lanes>,
    O: SortOrder,
    K: SortKey<V>,
{
    // See `sort_lanes_by_key` for why this is not a const assert.
    debug_assert!(
        V::LANES <= 16 && V::LANES.is_power_of_two(),
        "bitonic_clean_lanes_by_key covers power-of-two lane counts up to 16"
    );

    let v = if const { V::LANES >= 16 } {
        key_stage::<V, O, Distance<8>, K>(v)
    } else {
        v
    };
    let v = if const { V::LANES >= 8 } {
        key_stage::<V, O, Distance<4>, K>(v)
    } else {
        v
    };
    let v = if const { V::LANES >= 4 } {
        key_stage::<V, O, Distance<2>, K>(v)
    } else {
        v
    };

    if const { V::LANES >= 2 } {
        key_stage::<V, O, Distance<1>, K>(v)
    } else {
        v
    }
}
