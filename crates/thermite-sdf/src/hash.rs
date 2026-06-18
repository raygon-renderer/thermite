//! Spatial hashing for lattice-based procedural fields (fBM noise, Voronoi).
//!
//! A lattice field assigns pseudo-random data to each integer cell of a grid;
//! that mapping is the *hash*. Two flavors are needed, sharing the same zero-cost
//! strategy-type (ZST) pattern as the [`SmoothKernel`](crate::ops::SmoothKernel):
//!
//! - [`LatticeHash`] - one scalar in `$[0,1)$` per cell (fBM value/size noise).
//! - [`VectorHash`] - an N-vector in `$[0,1)^N$` per cell (a Voronoi feature
//!   *position* inside the cell).
//!
//! The standard-library `Hash`/`Hasher` traits are a poor fit: they stream bytes
//! into a single `u64`, whereas an SDF needs a *branchless, per-lane* map from N
//! float coordinates, evaluated for every lane of a SIMD vector at once. So these
//! traits take the lattice coordinates as a [`Vector<V, N>`] and compute in
//! floating point.
//!
//! Two hashes are provided, each implementing both traits:
//! - [`SinHash`] - the classic `fract(sin(dot(i, k)) * beta)`; one transcendental
//!   call, good distribution.
//! - [`HoskinsHash`] - a trig-free nonlinear `fract`/multiply fold (Dave Hoskins'
//!   `hash11`), for backends where `sin` is expensive or absent.
//!
//! # Writing your own hash
//!
//! Both traits are static (`fn`, no `self`): the implementing type is a pure
//! marker, and the hash is a *deterministic function of the integer cell only*.
//! Whatever you write must satisfy these invariants, which the consuming fields
//! rely on:
//!
//! - **Deterministic & stateless.** `hash(c)` / `jitter(c)` must depend only on
//!   `c`, returning the same value every time and on every lane. The same cell is
//!   visited from many different sample points (it is a *neighbor* of all of
//!   them); if its feature moved between visits the field would tear. No global
//!   state, no `self`.
//! - **Branchless & per-lane (SIMD).** `c` is a SoA vector: every lane is a
//!   *different* cell, hashed simultaneously. Use only lane-wise arithmetic
//!   (`+ - * fract floor`, `mul_adde`, etc.); never a data-dependent `if` on lane
//!   values - branch on `const`/type parameters only.
//! - **Output range.** Return values in `$[0, 1)$` per component (typically via a
//!   final `fract`). Consumers assume this: fBM uses the scalar as a radius/value,
//!   Voronoi uses the vector as an in-cell position.
//! - **Precision policy `P`.** Thread `P` into any transcendental call
//!   (`x.sin_p::<P>()`); a trig-free hash simply ignores it.
//! - **Integer-valued input.** `c` holds exact integers (a `floor`ed cell index),
//!   possibly large far from the origin. Keep transcendental arguments in a
//!   precision-friendly range (very large `sin` arguments lose entropy in `f32`).
//!
//! ## `LatticeHash` specifics
//!
//! Provide one well-distributed scalar. Aim for low autocorrelation between
//! adjacent cells (neighbors should look unrelated) and make the axes
//! *distinguishable* - i.e. avoid an accidental symmetry where `c = (x, y)` and
//! `(y, x)` collide - unless that symmetry is intended. A single linear form
//! pushed through `fract(sin(.) * beta)`, or a nonlinear fold, both work.
//!
//! ## `VectorHash` specifics (the subtle one)
//!
//! The N components must be **mutually independent**: the joint output must fill
//! `$[0,1)^N$` *uniformly*, not lie on a lower-dimensional subset. The classic
//! mistake is to compute one scalar `s = dot(c, k)` and derive every component
//! from it (`fract(sin(s + phase_j) * beta)`): all components are then functions
//! of the *single* scalar `s`, so the feature is stuck on a 1-D curve inside the
//! cell. Features line up, the nearest-neighbor search mis-resolves, and the
//! lattice shows through as axis-aligned seams.
//!
//! The fix is that each component must depend on a *distinct* combination of all
//! input coordinates - e.g. an independent (non-parallel, well-conditioned) weight
//! vector per component, as in IQ's `hash2`/`hash3`. Beware "almost independent"
//! schemes too: weight vectors that are merely *near*-parallel (e.g. successive
//! terms of a geometric recurrence) are ill-conditioned and degrade the same way,
//! just more subtly.
//!
//! # Adapters
//!
//! [`Scalarize`] and [`Vectorize`] convert between the two traits. The directions
//! are not equally safe - see each type's docs - but in short: `VectorHash ->
//! LatticeHash` ([`Scalarize`]) is always correct; `LatticeHash -> VectorHash`
//! ([`Vectorize`]) is best-effort and inherits the wrapped hash's conditioning, so
//! a purpose-built `VectorHash` is preferred for production.

use thermite::math::RealMathWithPolicy;
use thermite::math::policy::Policy;

use thermite_geometry::prim::Vector;

use crate::consts::{frac, vint};
use crate::{SdfVector};

/// Strategy for hashing an integer lattice cell to a single per-lane
/// pseudo-random value in `$[0, 1)$`. Implemented as zero-sized strategy types so
/// fields like [`FbmDetail`](crate::fbm::FbmDetail) can be parameterized over it.
///
/// `P` is the precision policy for any transcendental ops the implementation uses
/// (ignored by trig-free hashes). For a per-axis vector offset (e.g. a Voronoi
/// feature position) use [`VectorHash`] instead.
pub trait LatticeHash: Copy {
    /// Pseudo-random value in `$[0, 1)$` per lane for integer lattice coordinates `i`.
    fn hash<V: SdfVector + RealMathWithPolicy, P: Policy, const N: usize>(i: Vector<V, N>) -> V;
}

/// Strategy for hashing an integer lattice cell to a per-cell feature offset in
/// `$[0,1)^N$` - one independent pseudo-random component per axis, so a jittered
/// feature fills the cell uniformly.
///
/// This is the vector counterpart of [`LatticeHash`]: a Voronoi feature needs an
/// N-vector *position*, and each component must be a genuinely independent
/// function of the cell. Deriving the components by phase-shifting one scalar
/// hash (`fract(sin(dot(i, k) + phase))`) collapses the offset onto a 1-D curve
/// inside the cell - features line up and the lattice shows through as grid
/// artifacts - so this generalizes IQ's `hash2`/`hash3` to N dimensions with an
/// independent weight vector per output axis.
pub trait VectorHash: Copy {
    /// Per-axis feature offset in `$[0,1)^N$` for the integer lattice cell `cell`.
    fn jitter<V: SdfVector + RealMathWithPolicy, P: Policy, const N: usize>(cell: Vector<V, N>) -> Vector<V, N>;
}

/// Trigonometric hash: `$\operatorname{fract}\bigl(\sin(i\cdot k)\,\beta\bigr)$`
/// with `$\beta = 43758.54$`, where the per-axis weights `$k_j$` come from a
/// multiplicative recurrence so the dot product extends to any `N`:
///
/// ```math
/// h(i) = \operatorname{fract}\!\left( \sin\!\Bigl( \sum_{j=0}^{N-1} i_j\, k_j \Bigr) \beta \right),
/// \qquad k_0 = 127.1, \quad k_{j+1} = 1.324\, k_j + 74.7.
/// ```
///
/// One transcendental call. The constants are arbitrary irrational-ish seeds.
#[derive(Debug, Clone, Copy, Default)]
pub struct SinHash;

impl LatticeHash for SinHash {
    #[inline(always)]
    fn hash<V: SdfVector + RealMathWithPolicy, P: Policy, const N: usize>(i: Vector<V, N>) -> V {
        // dot(i, k) with k_0 = 127.1, k_{j+1} = 1.324*k_j + 74.7
        let mut k = frac::<V, 1271, 10>();
        let mut h = V::ZERO;
        let mut j = 0;
        while j < N {
            h = i[j].mul_adde(k, h);
            k = k.mul_adde(frac::<V, 1324, 1000>(), frac::<V, 747, 10>());
            j += 1;
        }
        (h.sin_p::<P>() * frac::<V, 4375854, 100>()).fract() // *43758.54
    }
}

/// A distinct pseudo-random integer weight in `$[64, 320)$` for entry `idx` of
/// the [`SinHash`] weight matrix (SplitMix64 on the flattened `(c, j)` index).
///
/// The weights must be *random*, not a geometric recurrence: a recurrence
/// `$k_{j+1} \approx g\, k_j$` makes every component's weight vector approximately
/// `$\propto (1, g, g^2, \dots)$` - i.e. all rows near-parallel - which collapses
/// the offset onto a low-dimensional locus and produces lattice-aligned artifacts.
#[inline(always)]
fn sin_weight<V: SdfVector>(idx: usize) -> V {
    let mut s = (idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    s ^= s >> 30;
    s = s.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    s ^= s >> 27;
    s = s.wrapping_mul(0x94D0_49BB_1331_11EB);
    s ^= s >> 31;
    vint::<V>(64 + (s % 256) as thermite::LargeInt)
}

impl VectorHash for SinHash {
    /// Each output component `c` uses its own independent (non-parallel) weight
    /// vector `$k^{(c)}$` of [`sin_weight`]s, then
    ///
    /// ```math
    /// o_c = \operatorname{fract}\!\Bigl(\sin\bigl(\textstyle\sum_j i_j\, k^{(c)}_j\bigr)\,\beta\Bigr),
    /// \qquad \beta = 43758.54 .
    /// ```
    #[inline(always)]
    fn jitter<V: SdfVector + RealMathWithPolicy, P: Policy, const N: usize>(cell: Vector<V, N>) -> Vector<V, N> {
        let mut o = Vector::ZERO;
        let mut c = 0;
        while c < N {
            let mut h = V::ZERO;
            let mut j = 0;
            while j < N {
                h = cell[j].mul_adde(sin_weight::<V>(c * N + j), h);
                j += 1;
            }
            o[c] = (h.sin_p::<P>() * frac::<V, 4375854, 100>()).fract(); // *43758.54
            c += 1;
        }
        o
    }
}

/// Trig-free hash: a sequential, nonlinear fold of Dave Hoskins' `hash11`
/// (<https://www.shadertoy.com/view/4djSRW>) over the axes,
///
/// ```math
/// h_0 = \tfrac12, \qquad h_{j+1} = \operatorname{hash11}(h_j + i_j), \qquad h(i) = h_N.
/// ```
///
/// The fold is order-sensitive, so the axes stay distinguishable. Pure
/// multiply/`fract`, so it is cheaper than [`SinHash`] and needs no
/// transcendental unit - useful on backends where `sin` is expensive or absent.
#[derive(Debug, Clone, Copy, Default)]
pub struct HoskinsHash;

impl HoskinsHash {
    /// Hoskins `hash11`, scalar `$\to [0, 1)$`:
    ///
    /// ```math
    /// \operatorname{hash11}(p) = \operatorname{fract}\bigl(q\,(q + q)\bigr),
    /// \quad q = \operatorname{fract}(0.1031\,p)\,(\operatorname{fract}(0.1031\,p) + 33.33).
    /// ```
    #[inline(always)]
    pub(crate) fn hash11<V: SdfVector>(p: V) -> V {
        let q = (p * frac::<V, 1031, 10000>()).fract(); // p*0.1031
        let q = q * (q + frac::<V, 3333, 100>()); // q*(q+33.33)
        (q * (q + q)).fract() // fract(q*2q)
    }
}

impl LatticeHash for HoskinsHash {
    #[inline(always)]
    fn hash<V: SdfVector + RealMathWithPolicy, P: Policy, const N: usize>(i: Vector<V, N>) -> V {
        // Sequential, nonlinear fold so axis order is significant (no symmetry).
        let mut h = frac::<V, 1, 2>(); // seed 0.5
        let mut j = 0;
        while j < N {
            h = Self::hash11(h + i[j]);
            j += 1;
        }
        h
    }
}

impl VectorHash for HoskinsHash {
    /// Each output component is an independent Hoskins `hash11` fold over the axes,
    /// started from a distinct seed `$\tfrac12 + \tfrac{c}{3}$`, so the components
    /// decorrelate while each still depends on every coordinate.
    #[inline(always)]
    fn jitter<V: SdfVector + RealMathWithPolicy, P: Policy, const N: usize>(cell: Vector<V, N>) -> Vector<V, N> {
        let mut o = Vector::ZERO;
        let mut c = 0;
        while c < N {
            // seed 0.5 + c/3 distinguishes the per-axis folds
            let mut h = vint::<V>(c as thermite::LargeInt).mul_adde(frac::<V, 1, 3>(), frac::<V, 1, 2>());
            let mut j = 0;
            while j < N {
                h = HoskinsHash::hash11(h + cell[j]);
                j += 1;
            }
            o[c] = h;
            c += 1;
        }
        o
    }
}

// ---------------------------------------------------------------------------
// Adapters between the two traits
// ---------------------------------------------------------------------------

/// Wraps a [`VectorHash`] `H` to expose it as a [`LatticeHash`] by taking the
/// first component of the offset, `$h(c) = \operatorname{jitter}(c)_0$`.
///
/// **This direction is always safe.** A `VectorHash` already yields N independent,
/// well-distributed scalars; projecting onto one of them is a uniformly
/// distributed value in `$[0,1)$`, exactly what `LatticeHash` requires.
///
/// **Cost caveat.** `jitter` computes *all* N components and this throws N-1 away,
/// so `Scalarize<H>` is wasteful relative to a native `LatticeHash` (e.g. it does
/// N `sin`s where [`SinHash`]'s scalar form does one). Use it for convenience -
/// e.g. driving [`FbmDetail`](crate::fbm::FbmDetail) with a custom `VectorHash` -
/// not on a hot path where a purpose-built scalar hash would do.
///
/// ```
/// use thermite_sdf::hash::{Scalarize, SinHash};
/// // a LatticeHash derived from the vector hash, usable wherever fBM wants one:
/// type ScalarFromVec = Scalarize<SinHash>;
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Scalarize<H: VectorHash>(pub H);

impl<H: VectorHash> LatticeHash for Scalarize<H> {
    #[inline(always)]
    fn hash<V: SdfVector + RealMathWithPolicy, P: Policy, const N: usize>(i: Vector<V, N>) -> V {
        H::jitter::<V, P, N>(i)[0]
    }
}

/// Wraps a [`LatticeHash`] `H` to expose it as a [`VectorHash`], producing
/// component `c` by hashing the input with its coordinates **cyclically rotated**
/// by `c`: `$o_c = H(\operatorname{rot}_c(i))$`.
///
/// Rotation (rather than an additive offset) is what makes this work at all:
/// adding a per-component constant only shifts a hash that is internally `f(dot(i,
/// k))` by a *phase* of the same scalar, collapsing every component onto a 1-D
/// curve (see the [module docs](self) - this is the canonical Voronoi-seam bug).
/// Rotating the coordinates instead feeds the wrapped hash a genuinely different
/// argument per component.
///
/// # Limitations - prefer a native [`VectorHash`]
///
/// The output quality is **only as good as the wrapped hash is well-conditioned
/// under coordinate rotation**, which this wrapper cannot guarantee:
///
/// - If `H` is *symmetric* under coordinate permutation (e.g. it hashes
///   `$\sum_j i_j$`), all rotations give the **same** value - the components
///   become identical and the offset collapses to the cell diagonal. Such an `H`
///   is unusable here.
/// - If `H`'s effective weight vectors are merely *near*-parallel - notably
///   [`SinHash`], whose scalar form uses a **geometric** weight recurrence, so its
///   rotations are ill-conditioned - the distribution is poor (the same failure
///   mode, subtler). `Vectorize<SinHash>` is therefore noticeably worse than the
///   purpose-built `<SinHash as VectorHash>` impl (which uses *random*,
///   well-separated per-component weights). [`HoskinsHash`]'s order-sensitive fold
///   rotates cleanly and fares better.
///
/// In short: this is a convenience for reusing an existing scalar hash, not a
/// substitute for a real vector hash. For `N = 1` it is exactly `H` (the rotation
/// is the identity). When in doubt, write a `VectorHash` directly.
///
/// ```
/// use thermite_sdf::hash::{Vectorize, HoskinsHash};
/// // a (best-effort) vector hash derived from a scalar one:
/// type VecFromScalar = Vectorize<HoskinsHash>;
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct Vectorize<H: LatticeHash>(pub H);

impl<H: LatticeHash> VectorHash for Vectorize<H> {
    #[inline(always)]
    fn jitter<V: SdfVector + RealMathWithPolicy, P: Policy, const N: usize>(cell: Vector<V, N>) -> Vector<V, N> {
        let mut o = Vector::ZERO;
        let mut c = 0;
        while c < N {
            // hash the cell with coordinates cyclically rotated by c
            let mut rotated = Vector::ZERO;
            let mut j = 0;
            while j < N {
                rotated[j] = cell[(j + c) % N];
                j += 1;
            }
            o[c] = H::hash::<V, P, N>(rotated);
            c += 1;
        }
        o
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FbmDetail, SDF, Sphere3D, Voronoi};
    use thermite::math::policy::DefaultPolicy;
    use thermite::prelude::*;
    use thermite_geometry::prim::{Vector2, Vector3};

    type V = thermite::Vector<f32>;

    fn cell2(x: f32, y: f32) -> Vector2<V> {
        Vector2::new([V::splat(x), V::splat(y)])
    }

    // Scalarize<H>: a VectorHash projected to one scalar - in range, deterministic.
    #[test]
    fn scalarize_is_latticehash() {
        let c = cell2(3.0, -7.0);
        let h = <Scalarize<SinHash> as LatticeHash>::hash::<V, DefaultPolicy, 2>(c).extract::<0>();
        assert!((0.0..1.0).contains(&h), "out of range: {h}");
        let h2 = <Scalarize<SinHash> as LatticeHash>::hash::<V, DefaultPolicy, 2>(c).extract::<0>();
        assert_eq!(h, h2, "deterministic");
    }

    // Vectorize<H>: each component in range, deterministic.
    #[test]
    fn vectorize_is_vectorhash() {
        let c = cell2(3.0, -7.0);
        let o = <Vectorize<HoskinsHash> as VectorHash>::jitter::<V, DefaultPolicy, 2>(c);
        for k in 0..2 {
            let v = o[k].extract::<0>();
            assert!((0.0..1.0).contains(&v), "comp {k} out of range: {v}");
        }
        let o2 = <Vectorize<HoskinsHash> as VectorHash>::jitter::<V, DefaultPolicy, 2>(c);
        assert_eq!(o[0].extract::<0>(), o2[0].extract::<0>(), "deterministic");
    }

    // The adapters drop into the real fields: Vectorize -> Voronoi, Scalarize -> fBM.
    #[test]
    fn adapters_compose_into_fields() {
        let vor = Voronoi::<V, DefaultPolicy, Vectorize<HoskinsHash>>::new();
        let d = vor.eval(cell2(0.3, 0.6));
        assert!(d.extract::<0>().is_finite() && d.extract::<0>() >= 0.0);

        let fbm: FbmDetail<V, Sphere3D<V>, DefaultPolicy, Scalarize<SinHash>> =
            FbmDetail::new(Sphere3D { radius: V::splat(1.0) }).with_hash(Scalarize::<SinHash>::default());
        let d = fbm.eval(Vector3::new([V::splat(1.2), V::splat(0.0), V::splat(0.0)]));
        assert!(d.extract::<0>().is_finite());
    }
}
