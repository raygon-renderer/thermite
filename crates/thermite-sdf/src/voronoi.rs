//! Voronoi (cellular) distance fields, after Inigo Quilez's articles
//! (<https://iquilezles.org/articles/smoothvoronoi>,
//! <https://iquilezles.org/articles/voronoilines>), generalized to N dimensions.
//!
//! A Voronoi field partitions space into a unit integer lattice, jitters one
//! feature point into each cell (via a [`VectorHash`](crate::hash::VectorHash)),
//! and measures distance to those features. Three fields are provided, all
//! **unsigned** distance estimators (`>= 0`, zero on their respective zero-set)
//! and all valid (1-Lipschitz) SDFs suitable for raymarching and procedural
//! texturing:
//!
//! - [`Voronoi`] - F1, the distance to the nearest feature point ("cellular").
//! - [`SmoothVoronoi`] - the same, but the discontinuous `min` is replaced by a
//!   [`SmoothKernel`] smooth-min so the field is filterable / antialiasable.
//! - [`VoronoiEdges`] - distance to the cell-boundary network. Its `EXACT` const
//!   parameter selects the mathematically correct two-pass metric or the cheaper
//!   single-pass approximation.
//!
//! All three work in "cell space" like IQ's reference: the sample's integer part
//! is dropped before hashing, so the feature jitter is computed relative to the
//! cell rather than in world space (better precision far from the origin). The
//! lattice has unit spacing; scale the domain (multiply `p`, or wrap in
//! [`Scale`](crate::ops::Scale)) to change the cell size.
//!
//! # Neighborhood radius (`RINGS`)
//!
//! Each field scans a `$(2\,\text{RINGS}+1)^N$` block of cells around the sample;
//! `RINGS` is a const generic defaulting to `2` (a `$5^N$` scan):
//!
//! - `RINGS = 2` (default) always reaches the true nearest feature - a site 3
//!   cells out is `$\ge 2 > \sqrt N$` away, so it can never be nearest - making F1
//!   *exact* and the field genuinely 1-Lipschitz.
//! - `RINGS = 1` (a `$3^N$` scan, IQ's textbook size) is ~2.8x cheaper in 2D but
//!   only approximate: the nearest site can sit two cells away, which shows up as
//!   discontinuities at cell boundaries (and, for the exact edges, can make pass 1
//!   pick the wrong cell). Use it when speed matters more than a clean metric.
//!
//! Cost per eval is `$(2\,\text{RINGS}+1)^N$` cells (the exact edge field pays for
//! two such passes). In practice pick the radius with a type alias, e.g.
//! `type FastVoronoi<V> = Voronoi<V, DefaultPolicy, SinHash, 1>;`. Practical to
//! roughly `N = 4`.
//!
//! These are infinite fields with no bounding box, so no [`BoundedSdf`]; the
//! gradient is numeric (wrap in [`FiniteDiff`](crate::ops::FiniteDiff)).
//!
//! # Examples
//!
//! In practice, pin down a configuration with a type alias and use that:
//!
//! ```
//! use thermite::prelude::*;
//! use thermite::math::policy::DefaultPolicy;
//! use thermite_geometry::soa::prim::Vector2;
//! use thermite_sdf::{SDF, Voronoi, VoronoiEdges, SinHash, HoskinsHash};
//!
//! // Cheap 1-ring cellular noise: approximate, but ~2.8x fewer cells in 2D.
//! type FastVoronoi<V> = Voronoi<V, DefaultPolicy, SinHash, 1>;
//!
//! // Trig-free cellular noise (no `sin`) - handy on WASM / sin-light backends.
//! type TrigFreeVoronoi<V> = Voronoi<V, DefaultPolicy, HoskinsHash>;
//!
//! // The cheap single-pass edge field (one scan; uneven line widths).
//! type CheapEdges<V> = VoronoiEdges<V, DefaultPolicy, false>;
//!
//! type F = Vector<f32>;
//! let noise = FastVoronoi::<F>::new();
//! let d = noise.eval(Vector2::new([F::splat(0.3), F::splat(0.7)]));
//! assert!(d.extract::<0>() >= 0.0);
//! # let _ = (TrigFreeVoronoi::<F>::new(), CheapEdges::<F>::new());
//! ```

use core::marker::PhantomData;

use thermite::mask::{GenericMask as _, GenericSelectable as _};
use thermite::math::RealMathWithPolicy;
use thermite::math::policy::{DefaultPolicy, Policy};

use thermite_geometry::soa::prim::{Vector, vector::VectorOps as _};

use crate::consts::{frac, vint};
use crate::hash::{SinHash, VectorHash};
use crate::ops::{Quadratic, SmoothKernel, smin_k};
use crate::{SDF, SdfVector};

/// Squared bail-out used to discard the originating cell in the exact edge pass:
/// a neighbor whose offset from the nearest feature is below this is the feature
/// itself (its bisector is degenerate / at infinity).
#[inline(always)]
fn edge_eps<V: SdfVector>() -> V {
    frac::<V, 1, 100000>() // 1e-5, matching IQ's reference threshold
}

/// Side length of a `RINGS`-radius neighborhood: `$2\,\text{RINGS}+1$`.
#[inline(always)]
const fn side(rings: usize) -> usize {
    2 * rings + 1
}

/// Decodes the `m`-th offset of a `base^N` neighborhood scan into `$\{lo, \dots,
/// lo+base-1\}^N$`, as a float vector. `m` ranges over `0..base^N`.
#[inline(always)]
fn neighbor_offset<V: SdfVector, const N: usize>(mut m: usize, base: usize, lo: thermite::LargeInt) -> Vector<V, N> {
    let mut g = Vector::ZERO;
    let mut k = 0;
    while k < N {
        g[k] = vint::<V>((m % base) as thermite::LargeInt + lo);
        m /= base;
        k += 1;
    }
    g
}

/// `base^N` as a loop bound (the size of an `N`-D neighborhood of side `base`).
#[inline(always)]
const fn pow_n(base: usize, n: usize) -> usize {
    let mut acc = 1;
    let mut i = 0;
    while i < n {
        acc *= base;
        i += 1;
    }
    acc
}

/// Splits `p` into its integer cell `i = floor(p)` and the in-cell fraction
/// `f = p - i`.
#[inline(always)]
fn cell_split<V: SdfVector, const N: usize>(p: Vector<V, N>) -> (Vector<V, N>, Vector<V, N>) {
    let mut i = Vector::ZERO;
    let mut k = 0;
    while k < N {
        i[k] = p[k].floor();
        k += 1;
    }
    (i, p - i)
}

// ---------------------------------------------------------------------------
// F1 - distance to the nearest feature point
// ---------------------------------------------------------------------------

/// Classic Voronoi / cellular field: the Euclidean distance to the nearest
/// jittered feature point, scanning the `$(2\,\text{RINGS}+1)^N$` cells around the
/// sample.
///
/// With `$i = \lfloor p \rfloor$`, `$f = p - i$`, a per-cell feature offset
/// `$o(c) \in [0,1)^N$` and `$R = \text{RINGS}$`,
///
/// ```math
/// \operatorname{F_1}(p) = \min_{g \in \{-R, \dots, R\}^N}
///   \bigl\lVert\, g + o(i+g) - f \,\bigr\rVert_2 .
/// ```
///
/// The min is taken on squared distances (one final `sqrt`). At the default
/// `RINGS = 2` the scan always reaches the true nearest site, so this is the
/// *exact* F1 distance - a valid unsigned SDF, exactly 1-Lipschitz, whose
/// zero-set is the feature points. See the [module docs](self) for the `RINGS`
/// trade-off.
#[derive(Debug, Clone, Copy)]
pub struct Voronoi<V: SdfVector, P: Policy = DefaultPolicy, H: VectorHash = SinHash, const RINGS: usize = 2> {
    pub hash: H,
    _policy: PhantomData<(V, P)>,
}

impl<V: SdfVector, P: Policy, H: VectorHash + Default, const RINGS: usize> Voronoi<V, P, H, RINGS> {
    /// Builds with the hash `H` (defaults to [`SinHash`] when unspecified).
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            hash: H::default(),
            _policy: PhantomData,
        }
    }
}

impl<V: SdfVector, P: Policy, H: VectorHash + Default, const RINGS: usize> Default for Voronoi<V, P, H, RINGS> {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl<V: SdfVector, P: Policy, H: VectorHash, const RINGS: usize> Voronoi<V, P, H, RINGS> {
    /// Swaps in a different [`VectorHash`], keeping the ring radius.
    #[inline(always)]
    pub fn with_hash<H2: VectorHash>(self, hash: H2) -> Voronoi<V, P, H2, RINGS> {
        Voronoi {
            hash,
            _policy: PhantomData,
        }
    }
}

impl<V: SdfVector + RealMathWithPolicy, P: Policy, H: VectorHash, const RINGS: usize, const N: usize> SDF<V, N>
    for Voronoi<V, P, H, RINGS>
{
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        let (i, f) = cell_split(p);
        let (base, lo) = (side(RINGS), -(RINGS as thermite::LargeInt));
        let mut md = V::INFINITY; // min squared distance
        let count = pow_n(base, N);
        let mut m = 0;
        while m < count {
            let g = neighbor_offset::<V, N>(m, base, lo);
            let o = H::jitter::<V, P, N>(i + g);
            let r = (g + o) - f;
            md = md.min(r.dot(&r));
            m += 1;
        }
        md.sqrt()
    }
}

// ---------------------------------------------------------------------------
// Smooth Voronoi - smooth-min of feature distances
// ---------------------------------------------------------------------------

/// Smooth Voronoi: the F1 field with the discontinuous `min` replaced by a
/// [`SmoothKernel`] smooth-min of blend radius `smooth`, so the field (and its
/// gradient) is continuous and filterable (<https://iquilezles.org/articles/smoothvoronoi>).
///
/// Each of the `$(2\,\text{RINGS}+1)^N$` neighbor distances is folded in with
/// `$\operatorname{smin}_K$`; as `smooth -> 0` it converges to [`Voronoi`]. The
/// smooth-min is `$\le \min$`, so the field stays a (slightly contracted) valid
/// distance estimator.
#[derive(Debug, Clone, Copy)]
pub struct SmoothVoronoi<
    V: SdfVector,
    P: Policy = DefaultPolicy,
    K: SmoothKernel = Quadratic,
    H: VectorHash = SinHash,
    const RINGS: usize = 2,
> {
    /// Smooth-min blend radius (in distance units). Must be `> 0`.
    pub smooth: V,
    pub kernel: K,
    pub hash: H,
    _policy: PhantomData<(V, P)>,
}

impl<V: SdfVector, P: Policy, K: SmoothKernel + Default, H: VectorHash + Default, const RINGS: usize>
    SmoothVoronoi<V, P, K, H, RINGS>
{
    /// Builds with the given blend radius; the kernel `K` and hash `H` default to
    /// [`Quadratic`] and [`SinHash`] when unspecified.
    #[inline(always)]
    pub fn new(smooth: V) -> Self {
        Self {
            smooth,
            kernel: K::default(),
            hash: H::default(),
            _policy: PhantomData,
        }
    }
}

impl<V: SdfVector, P: Policy, K: SmoothKernel, H: VectorHash, const RINGS: usize> SmoothVoronoi<V, P, K, H, RINGS> {
    /// Swaps in a different smooth-min [`SmoothKernel`], keeping all else.
    #[inline(always)]
    pub fn with_kernel<K2: SmoothKernel>(self, kernel: K2) -> SmoothVoronoi<V, P, K2, H, RINGS> {
        SmoothVoronoi {
            smooth: self.smooth,
            kernel,
            hash: self.hash,
            _policy: PhantomData,
        }
    }

    /// Swaps in a different [`VectorHash`], keeping all else.
    #[inline(always)]
    pub fn with_hash<H2: VectorHash>(self, hash: H2) -> SmoothVoronoi<V, P, K, H2, RINGS> {
        SmoothVoronoi {
            smooth: self.smooth,
            kernel: self.kernel,
            hash,
            _policy: PhantomData,
        }
    }
}

impl<V: SdfVector + RealMathWithPolicy, P: Policy, K: SmoothKernel, H: VectorHash, const RINGS: usize, const N: usize>
    SDF<V, N> for SmoothVoronoi<V, P, K, H, RINGS>
{
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        let (i, f) = cell_split(p);
        let (base, lo) = (side(RINGS), -(RINGS as thermite::LargeInt));
        let mut md = V::INFINITY;
        let count = pow_n(base, N);
        let mut m = 0;
        while m < count {
            let g = neighbor_offset::<V, N>(m, base, lo);
            let o = H::jitter::<V, P, N>(i + g);
            let r = (g + o) - f;
            let d = r.l2_norm(); // actual distance (smin works in distance units)
            md = smin_k::<K, V>(md, d, self.smooth);
            m += 1;
        }
        md
    }
}

// ---------------------------------------------------------------------------
// Voronoi edges - distance to the cell-boundary network
// ---------------------------------------------------------------------------

/// Distance to the Voronoi cell borders (<https://iquilezles.org/articles/voronoilines>).
///
/// `F2 - F1` is *not* a real distance (border width pulses with the local cell
/// spacing). The fix measures distance to the bisecting hyperplane between cells,
/// in the cell-relative frame centered at the sample. The `EXACT` const selects
/// the algorithm (orthogonal to the [`RINGS`](self) scan radius):
///
/// - `EXACT = false` (cheap): one pass finds the two nearest features `a, b`; the
///   border distance is the projection onto the `a-b` bisector,
///   `$\langle \tfrac12(a+b),\ \widehat{b-a} \rangle$`. Fast, but wrong near the
///   corners where three cells meet (the second-nearest changes discontinuously).
///
/// - `EXACT = true` (two-pass): pass 1 finds the nearest cell; pass 2 re-centers
///   on it and takes the min distance to each neighbor's bisector, `$\min_r
///   \langle \tfrac12(m_r + r),\ \widehat{r - m_r} \rangle$`. This is the
///   mathematically correct cell-edge metric (uniform line widths, equidistant
///   isolines), at the cost of a second scan. At the default `RINGS = 2`, pass 1
///   never misses the nearest site (IQ's reference uses a 1-ring pass 1, which
///   can, yielding a slightly negative distance); so the result stays `>= 0`.
///
/// The result is an unsigned, ~1-Lipschitz SDF whose zero-set is the cell-border
/// network.
#[derive(Debug, Clone, Copy)]
pub struct VoronoiEdges<
    V: SdfVector,
    P: Policy = DefaultPolicy,
    const EXACT: bool = true,
    H: VectorHash = SinHash,
    const RINGS: usize = 2,
> {
    pub hash: H,
    _policy: PhantomData<(V, P)>,
}

impl<V: SdfVector, P: Policy, const EXACT: bool, H: VectorHash + Default, const RINGS: usize>
    VoronoiEdges<V, P, EXACT, H, RINGS>
{
    /// Builds with the hash `H` (defaults to [`SinHash`] when unspecified).
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            hash: H::default(),
            _policy: PhantomData,
        }
    }
}

impl<V: SdfVector, P: Policy, const EXACT: bool, H: VectorHash + Default, const RINGS: usize> Default
    for VoronoiEdges<V, P, EXACT, H, RINGS>
{
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl<V: SdfVector, P: Policy, const EXACT: bool, H: VectorHash, const RINGS: usize> VoronoiEdges<V, P, EXACT, H, RINGS> {
    /// Swaps in a different [`VectorHash`], keeping all else.
    #[inline(always)]
    pub fn with_hash<H2: VectorHash>(self, hash: H2) -> VoronoiEdges<V, P, EXACT, H2, RINGS> {
        VoronoiEdges {
            hash,
            _policy: PhantomData,
        }
    }
}

impl<V: SdfVector + RealMathWithPolicy, P: Policy, const EXACT: bool, H: VectorHash, const RINGS: usize, const N: usize>
    SDF<V, N> for VoronoiEdges<V, P, EXACT, H, RINGS>
{
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        let (i, f) = cell_split(p);
        let (base, lo) = (side(RINGS), -(RINGS as thermite::LargeInt));
        let count = pow_n(base, N);

        if const { EXACT } {
            // pass 1: nearest feature (squared dist md, relative position mr, cell offset mg)
            let mut md = V::INFINITY;
            let mut mr = Vector::ZERO;
            let mut mg = Vector::ZERO;
            let mut m = 0;
            while m < count {
                let g = neighbor_offset::<V, N>(m, base, lo);
                let o = H::jitter::<V, P, N>(i + g);
                let r = (g + o) - f;
                let d = r.dot(&r);
                let lt = d.cmp_lt(md);
                md = lt.select(d, md);
                mr = Vector::select(lt, r, mr);
                mg = Vector::select(lt, g, mg);
                m += 1;
            }

            // pass 2: min distance to each neighbor's bisector, re-centered on mg
            let eps = edge_eps::<V>();
            let mut res = V::INFINITY;
            let mut m = 0;
            while m < count {
                let g = mg + neighbor_offset::<V, N>(m, base, lo);
                let o = H::jitter::<V, P, N>(i + g);
                let r = (g + o) - f;
                let diff = r - mr; // r - mr
                let len2 = diff.dot(&diff);
                // skip the originating cell (diff ~ 0); its bisector is degenerate
                let skip = len2.cmp_le(eps);
                let mid = (mr + r) * V::HALF;
                let bd = mid.dot(&diff.normalize());
                res = res.min(skip.select(V::INFINITY, bd));
                m += 1;
            }
            res
        } else {
            // single pass: track nearest (md1, mra) and second-nearest (md2, mrb)
            let mut md1 = V::INFINITY;
            let mut md2 = V::INFINITY;
            let mut mra = Vector::ZERO;
            let mut mrb = Vector::ZERO;
            let mut m = 0;
            while m < count {
                let g = neighbor_offset::<V, N>(m, base, lo);
                let o = H::jitter::<V, P, N>(i + g);
                let r = (g + o) - f;
                let d = r.dot(&r);
                // compare against the OLD running minima, then shift them down
                let lt1 = d.cmp_lt(md1);
                let lt2 = d.cmp_lt(md2);
                md2 = lt1.select(md1, lt2.select(d, md2));
                mrb = Vector::select(lt1, mra, Vector::select(lt2, r, mrb));
                md1 = lt1.select(d, md1);
                mra = Vector::select(lt1, r, mra);
                m += 1;
            }
            // distance to the bisector of the two nearest features
            let ab = mrb - mra;
            let mid = (mra + mrb) * V::HALF;
            mid.dot(&ab.normalize())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::HoskinsHash;
    use thermite::prelude::*;
    use thermite_geometry::soa::prim::{Vector2, Vector3};

    type V = thermite::Vector<f32>;

    #[inline]
    fn v(x: f32) -> V {
        V::splat(x)
    }
    #[inline]
    fn p(x: f32, y: f32) -> Vector2<V> {
        Vector2::new([v(x), v(y)])
    }
    #[inline]
    fn p3(x: f32, y: f32, z: f32) -> Vector3<V> {
        Vector3::new([v(x), v(y), v(z)])
    }
    #[inline]
    fn s(x: V) -> f32 {
        x.extract::<0>()
    }

    // A spread of off-lattice 2D sample points (avoid exact integer/half coords).
    const PTS2: [(f32, f32); 6] = [
        (0.137, 0.642),
        (1.733, -0.512),
        (-2.21, 3.07),
        (4.51, 2.09),
        (-1.27, -3.88),
        (7.03, -5.41),
    ];
    const PTS3: [(f32, f32, f32); 4] = [
        (0.137, 0.642, -0.31),
        (1.733, -0.512, 2.04),
        (-2.21, 3.07, 1.11),
        (4.51, 2.09, -3.6),
    ];

    #[test]
    fn f1_nonnegative_finite_deterministic() {
        let f = Voronoi::<V>::new();
        for &(x, y) in &PTS2 {
            let d = s(f.eval(p(x, y)));
            assert!(d.is_finite() && d >= 0.0, "2D F1 {d} at ({x},{y})");
            assert_eq!(d, s(f.eval(p(x, y))), "deterministic");
            // a unit lattice always has a feature within ~sqrt(2) of any point
            assert!(d <= 1.5, "2D F1 {d} unexpectedly large");
        }
        for &(x, y, z) in &PTS3 {
            let d = s(f.eval(p3(x, y, z)));
            assert!(d.is_finite() && d >= 0.0, "3D F1 {d}");
            assert!(d <= 1.8, "3D F1 {d} unexpectedly large");
        }
    }

    // Distance to a point set is exactly 1-Lipschitz: |f(p) - f(p+dp)| <= |dp|.
    #[test]
    fn f1_is_one_lipschitz() {
        let f = Voronoi::<V>::new();
        let eps = 0.01f32;
        for &(x, y) in &PTS2 {
            let d0 = s(f.eval(p(x, y)));
            for (dx, dy) in [(eps, 0.0), (0.0, eps), (eps, eps)] {
                let d1 = s(f.eval(p(x + dx, y + dy)));
                let step = (dx * dx + dy * dy).sqrt();
                assert!((d1 - d0).abs() <= step + 1e-4, "Lipschitz: |{d1}-{d0}| > {step}");
            }
        }
    }

    // Smooth-min is <= min, so SmoothVoronoi <= F1; and as smooth -> 0 it converges.
    #[test]
    fn smooth_voronoi_bounds_and_limit() {
        let f1 = Voronoi::<V>::new();
        let sv = SmoothVoronoi::<V>::new(v(0.05));
        let sv_tiny = SmoothVoronoi::<V>::new(v(1e-3));
        for &(x, y) in &PTS2 {
            let d_f1 = s(f1.eval(p(x, y)));
            let d_sv = s(sv.eval(p(x, y)));
            assert!(d_sv.is_finite() && d_sv >= 0.0, "smooth {d_sv}");
            assert!(d_sv <= d_f1 + 1e-4, "smooth {d_sv} should be <= F1 {d_f1}");
            // a tiny blend radius reproduces F1 closely
            let d_tiny = s(sv_tiny.eval(p(x, y)));
            assert!((d_tiny - d_f1).abs() < 5e-3, "smooth->F1: {d_tiny} vs {d_f1}");
        }
    }

    #[test]
    fn edges_exact_nonnegative_and_lipschitz() {
        let e = VoronoiEdges::<V, DefaultPolicy, true>::new();
        let eps = 0.01f32;
        for &(x, y) in &PTS2 {
            let d0 = s(e.eval(p(x, y)));
            assert!(d0.is_finite() && d0 >= -1e-4, "exact edge {d0} at ({x},{y})");
            assert_eq!(d0, s(e.eval(p(x, y))), "deterministic");
            for (dx, dy) in [(eps, 0.0), (0.0, eps)] {
                let d1 = s(e.eval(p(x + dx, y + dy)));
                let step = (dx * dx + dy * dy).sqrt();
                // exact edge distance is a min of bisector half-space distances,
                // hence ~1-Lipschitz (small slack for the discrete cell selection)
                assert!((d1 - d0).abs() <= step + 5e-3, "edge Lipschitz: |{d1}-{d0}| > {step}");
            }
        }
        // 3D exact path runs and stays finite/non-negative
        for &(x, y, z) in &PTS3 {
            let d = s(e.eval(p3(x, y, z)));
            assert!(d.is_finite() && d >= -1e-4, "3D exact edge {d}");
        }
    }

    #[test]
    fn edges_cheap_runs() {
        let e = VoronoiEdges::<V, DefaultPolicy, false>::new();
        for &(x, y) in &PTS2 {
            let d = s(e.eval(p(x, y)));
            assert!(d.is_finite() && d >= -1e-4, "cheap edge {d} at ({x},{y})");
        }
    }

    // The trig-free HoskinsHash works as a drop-in for every field.
    #[test]
    fn hoskins_hash_drop_in() {
        let f = Voronoi::<V>::new().with_hash(HoskinsHash);
        let e = VoronoiEdges::<V, DefaultPolicy, true>::new().with_hash(HoskinsHash);
        for &(x, y) in &PTS2 {
            assert!(s(f.eval(p(x, y))).is_finite());
            assert!(s(e.eval(p(x, y))).is_finite());
        }
    }

    // A 1-ring (RINGS = 1) field is the cheap approximation; it must still be a
    // finite, non-negative, deterministic distance. (It is *not* exactly
    // 1-Lipschitz - that is the documented trade-off - so we don't assert it.)
    #[test]
    fn rings_const_generic() {
        let f1_fast = Voronoi::<V, DefaultPolicy, SinHash, 1>::new();
        let edges_fast = VoronoiEdges::<V, DefaultPolicy, true, SinHash, 1>::new();
        for &(x, y) in &PTS2 {
            let d = s(f1_fast.eval(p(x, y)));
            assert!(d.is_finite() && d >= 0.0, "1-ring F1 {d}");
            assert_eq!(d, s(f1_fast.eval(p(x, y))), "deterministic");
            assert!(s(edges_fast.eval(p(x, y))).is_finite());
        }
        // a wider scan never reports a *larger* nearest distance than a narrower one
        let f1_wide = Voronoi::<V, DefaultPolicy, SinHash, 3>::new();
        for &(x, y) in &PTS2 {
            assert!(s(f1_wide.eval(p(x, y))) <= s(f1_fast.eval(p(x, y))) + 1e-4);
        }
    }
}
