//! fBM (fractal-noise) detail for SDFs, after
//! <https://iquilezles.org/articles/fbmsdf>, generalized to N dimensions.
//!
//! Adding a noise field to an SDF breaks the metric (the sum of two SDFs is not
//! an SDF). Instead this "adds" detail the SDF way: each octave is an SDF of a
//! random primitive lattice that is *smooth-clipped* to a slightly inflated copy
//! of the host surface (`smax`) and then *smooth-unioned* onto it (`smin`). The
//! result stays a valid (approximate) SDF suitable for raymarching, distance
//! lighting and collision - unlike naive displacement.
//!
//! Everything here is dimension-generic and composed from zero-cost strategy
//! types (hash, octave transform, lattice primitive, smooth kernel). The base
//! layer mins over the `$2^N$` cells around the sample, so cost per octave grows
//! as `$2^N$` (practical to roughly `N = 6`). The inter-octave decorrelation
//! rotation uses the bespoke rational matrix from IQ for `N = 3` and a generic
//! scaled axis-permutation rotation otherwise.
//!
//! # Examples
//!
//! [`FbmDetail`] carries four strategy type parameters ([`LatticeHash`],
//! [`OctaveTransform`], [`LatticePrimitive`], [`SmoothKernel`]); a type alias over
//! the host shape `S` names a reusable configuration, which the `with_*` builders
//! then produce:
//!
//! ```
//! use thermite::prelude::*;
//! use thermite::math::policy::DefaultPolicy;
//! use thermite_geometry::prim::Vector3;
//! use thermite_sdf::{FbmDetail, SDF, Sphere3D};
//! use thermite_sdf::{SinHash, HoskinsHash, IqRotation, GivensRotation, BoxCell};
//!
//! type F = Vector<f32>;
//! type Ball = Sphere3D<F>;
//!
//! // Trig-free fBM (no `sin` in the hash) - for WASM / sin-light backends.
//! type TrigFreeFbm<S> = FbmDetail<F, S, DefaultPolicy, HoskinsHash>;
//!
//! // Crystalline fBM: box (L-inf) cells give a blockier, faceted surface.
//! type CrystalFbm<S> = FbmDetail<F, S, DefaultPolicy, HoskinsHash, IqRotation, BoxCell>;
//!
//! // For N != 3, GivensRotation decorrelates octaves better than IqRotation.
//! type NdFbm<S> = FbmDetail<F, S, DefaultPolicy, SinHash, GivensRotation>;
//!
//! // Build via the ergonomic setters; each result has exactly the aliased type.
//! let _trig: TrigFreeFbm<Ball> = FbmDetail::new(Sphere3D { radius: F::splat(1.0) }).with_hash(HoskinsHash);
//! let _nd: NdFbm<Ball> = FbmDetail::new(Sphere3D { radius: F::splat(1.0) }).with_transform(GivensRotation);
//! let detail: CrystalFbm<Ball> =
//!     FbmDetail::new(Sphere3D { radius: F::splat(1.0) }).with_hash(HoskinsHash).with_primitive(BoxCell);
//!
//! let d = detail.eval(Vector3::new([F::splat(1.5), F::splat(0.0), F::splat(0.0)]));
//! assert!(d.extract::<0>().is_finite());
//! ```

use core::marker::PhantomData;

use thermite::math::RealMathWithPolicy;
use thermite::math::policy::{DefaultPolicy, Policy};

use thermite_geometry::prim::{Vector, vector::VectorOps as _};

use crate::consts::frac;
use crate::hash::{LatticeHash, SinHash};
use crate::ops::{Quadratic, SmoothKernel, smax_k, smin_k};
use crate::{SDF, SdfVector};

/// Strategy for the primitive placed at each lattice vertex. Given the offset
/// `d` from the corner to the sample point and a per-lane random `h` in
/// `[0, 1)`, it returns that primitive's signed distance. Pluggable like
/// [`LatticeHash`]; all built-ins are 1-Lipschitz so the lattice min stays a
/// valid (conservative) field.
///
/// The field's *look* is dominated by this choice, and so is part of its cost -
/// e.g. [`SphereCell`] needs a `sqrt` per corner while [`BoxCell`] does not.
pub trait LatticePrimitive: Copy {
    /// Signed distance from local offset `d` (= point - corner) to the primitive
    /// seeded by `h`.
    fn distance<V: SdfVector, const N: usize>(d: Vector<V, N>, h: V) -> V;
}

/// Sphere of random radius `$r = \tfrac12 h \in [0, 0.5)$` - the reference
/// primitive (isotropic). Euclidean (`$L^2$`), so it costs one `sqrt` per corner:
///
/// ```math
/// \operatorname{dist}(d, h) = \lVert d \rVert_2 - \tfrac12 h.
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct SphereCell;

impl LatticePrimitive for SphereCell {
    #[inline(always)]
    fn distance<V: SdfVector, const N: usize>(d: Vector<V, N>, h: V) -> V {
        d.l2_norm() - V::HALF * h
    }
}

/// Axis-aligned cube of random half-extent `$r = \tfrac12 h \in [0, 0.5)$`,
/// measured in the `$L^\infty$` (Chebyshev) norm. Sqrt-free, so it is the fastest
/// built-in primitive; gives a blockier, more crystalline noise:
///
/// ```math
/// \operatorname{dist}(d, h) = \max_i |d_i| - \tfrac12 h.
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct BoxCell;

impl LatticePrimitive for BoxCell {
    #[inline(always)]
    fn distance<V: SdfVector, const N: usize>(d: Vector<V, N>, h: V) -> V {
        let mut m = V::ZERO;
        for k in 0..N {
            m = m.max(d[k].abs());
        }
        m - V::HALF * h
    }
}

/// `sdBase`: an infinite cubic lattice (unit spacing) of [`LatticePrimitive`]s
/// `C` with random size, evaluated as the min over the `$2^N$` cell corners
/// surrounding `p`. With `$i = \lfloor p \rfloor$` and `$f = p - i$`,
///
/// ```math
/// \operatorname{sdBase}(p) = \min_{c \in \{0,1\}^N}
///   \operatorname{dist}_C\!\bigl(f - c,\ \operatorname{hash}(i + c)\bigr).
/// ```
///
/// The corner reduction uses two interleaved `min` accumulators to halve the
/// dependency chain (the per-corner hash/distance work then overlaps across the
/// unrolled `$2^N$` iterations).
#[inline(always)]
fn sd_base<V: SdfVector + RealMathWithPolicy, P: Policy, H: LatticeHash, C: LatticePrimitive, const N: usize>(
    p: Vector<V, N>,
) -> V {
    let mut i = Vector::ZERO;
    for k in 0..N {
        i[k] = p[k].floor();
    }
    let f = p - i;
    let (mut d0, mut d1) = (V::INFINITY, V::INFINITY);
    let corners = 1usize << N;
    let mut m = 0usize;
    while m < corners {
        let mut c = Vector::ZERO;
        for k in 0..N {
            c[k] = if (m >> k) & 1 == 1 { V::ONE } else { V::ZERO };
        }
        let h = H::hash::<V, P, N>(i + c);
        let dist = C::distance(f - c, h);
        // alternate accumulators (m is const per unrolled iteration)
        if m & 1 == 0 {
            d0 = d0.min(dist);
        } else {
            d1 = d1.min(dist);
        }
        m += 1;
    }
    d0.min(d1)
}

/// Strategy for the domain transform applied between fBM octaves: a rotation
/// scaled ~2x that doubles the frequency while breaking axis alignment so the
/// octaves do not stack coherently. Pluggable like [`LatticeHash`].
///
/// It only needs to be (approximately) `2 *` an orthonormal map; the per-vertex
/// [`LatticeHash`] supplies the rest of the decorrelation. All built-in members
/// are trig-free.
pub trait OctaveTransform: Copy {
    /// Maps the domain for the next octave.
    fn apply<V: SdfVector, const N: usize>(p: Vector<V, N>) -> Vector<V, N>;
}

/// Default inter-octave transform: a frequency-doubling rotation, specialized to
/// IQ's hand-tuned rational matrix at `N = 3` (so the 3D field reproduces the
/// reference) and a generic scaled cyclic permutation otherwise.
///
/// # `N = 3`
///
/// Applies the fixed matrix
///
/// ```math
/// M = \begin{pmatrix}
///   0   & -1.6  & -1.2 \\
///   1.6 &  0.72 & -0.96 \\
///   1.2 & -0.96 &  1.28
/// \end{pmatrix}.
/// ```
///
/// Every row and column has norm exactly `$2$`, so `$M = 2R$` with `$R$` an
/// orthonormal rotation and `$\det M = 2^3 = 8$`. Hence
/// `$\lVert M p \rVert = 2\lVert p \rVert$`: each octave doubles the frequency,
/// while `$R$` tilts the lattice off-axis so successive octaves decorrelate. The
/// odd-looking rational entries are IQ's chosen seed for that tilt.
///
/// # General `N`
///
/// Uses `$M = 2P$`, where `$P$` is the cyclic-shift permutation that reads each
/// component from the next axis,
///
/// ```math
/// (M p)_k = 2\, p_{(k+1) \bmod N} \quad (k \ge 1),
/// \qquad (M p)_0 = \pm\, 2\, p_1,
/// ```
///
/// with the `$-$` sign taken when `$N$` is even so that `$\det P = +1$` (a proper
/// rotation) and `$\det M = 2^N$`. Because `$P^N = I$`, the orientation returns to
/// itself every `$N$` octaves - only the `$2^N$` scaling persists - so octaves
/// re-align periodically. For `$N \ne 3$` prefer [`GivensRotation`], whose angle is
/// incommensurate with any such period.
#[derive(Debug, Clone, Copy, Default)]
pub struct IqRotation;

impl OctaveTransform for IqRotation {
    #[inline(always)]
    fn apply<V: SdfVector, const N: usize>(p: Vector<V, N>) -> Vector<V, N> {
        let mut q = Vector::ZERO;
        if const { N == 3 } {
            // columns (0,1.6,1.2), (-1.6,0.72,-0.96), (-1.2,-0.96,1.28); rows have
            // norm 2, i.e. exactly 2 * a rotation.
            let (x, y, z) = (p[0], p[1], p[2]);
            let c16 = frac::<V, 16, 10>();
            let c12 = frac::<V, 12, 10>();
            let c072 = frac::<V, 72, 100>();
            let c096 = frac::<V, 96, 100>();
            let c128 = frac::<V, 128, 100>();
            q[0] = (-c16).mul_adde(y, -c12 * z);
            q[1] = c16.mul_adde(x, c072.mul_adde(y, -c096 * z));
            q[2] = c12.mul_adde(x, (-c096).mul_adde(y, c128 * z));
        } else {
            // 2 * (cyclic shift e_k -> e_{k+1}). A cyclic shift has det (-1)^(N-1);
            // flip one component for N even so it stays a proper rotation.
            for k in 0..N {
                q[k] = p[(k + 1) % N] * V::TWO;
            }
            if const { N.is_multiple_of(2) } {
                q[0] = -q[0];
            }
        }
        q
    }
}

/// Inter-octave transform built from a chain of planar (Givens) rotations, each
/// by the fixed 3-4-5 angle `$\theta = \arccos\frac{4}{5} \approx 36.87°$`
/// (`$\cos\theta = 0.8,\ \sin\theta = 0.6$`), then scaled by `$2$`.
///
/// For each adjacent coordinate pair `$(k, k{+}1)$` it applies the Givens
/// rotation `$G_k$`, which is the identity off that plane and within it acts as
///
/// ```math
/// \begin{pmatrix} q_k \\ q_{k+1} \end{pmatrix} \mapsto
/// \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}
/// \begin{pmatrix} q_k \\ q_{k+1} \end{pmatrix}.
/// ```
///
/// Sweeping `$k = 0, \dots, N{-}2$` and scaling composes them into
///
/// ```math
/// M = 2\, G_{N-2} \cdots G_1 G_0,
/// ```
///
/// a proper rotation with `$\det M = 2^N$` that doubles the frequency like
/// [`IqRotation`] but, crucially, never re-aligns. By Niven's theorem a rational
/// `$\cos\theta$` with `$\theta/\pi$` rational forces
/// `$\cos\theta \in \{0, \pm\tfrac12, \pm1\}$`; since `$\cos\theta = \tfrac45$` is
/// none of these, `$\theta/\pi$` is irrational and `$\theta$` is incommensurate
/// with `$2\pi$`, so no finite number of octaves returns the lattice to a
/// previous orientation. That gives better decorrelation than [`IqRotation`]'s
/// order-`$N$` permutation for general `$N$`, while staying rational and trig-free.
///
/// At `$N = 3$` it does **not** reproduce the IQ reference; use [`IqRotation`] for
/// that. For `$N \le 1$` there is no rotation plane, so it reduces to the bare
/// `$2\times$` scaling.
#[derive(Debug, Clone, Copy, Default)]
pub struct GivensRotation;

impl OctaveTransform for GivensRotation {
    #[inline(always)]
    fn apply<V: SdfVector, const N: usize>(p: Vector<V, N>) -> Vector<V, N> {
        let c = frac::<V, 8, 10>(); // cos = 0.8
        let s = frac::<V, 6, 10>(); // sin = 0.6
        let mut q = p;
        for k in 0..N.saturating_sub(1) {
            let (a, b) = (q[k], q[k + 1]);
            q[k] = c.mul_adde(a, -(s * b)); // c*a - s*b
            q[k + 1] = s.mul_adde(a, c * b); // s*a + c*b
        }
        for k in 0..N {
            let qk = q[k];
            q[k] += qk; // * 2
        }
        q
    }
}

/// Grows fBM detail on top of a host N-D SDF (`<https://iquilezles.org/articles/fbmsdf>`).
///
/// Starting from the host field `$d_0 = \text{shape}(p)$` and `$q_0 = p$`, each
/// octave `$o = 0, 1, \dots$` evaluates a `sd_base` layer at amplitude
/// `$s_o = 2^{-o}$`, **smooth-clips** it to a slightly inflated copy of the
/// running surface, **smooth-combines** it in, then rotates/doubles the domain
/// for the next octave (with `$\iota = \text{inflate}$`, `$\kappa = \text{smooth}$`):
///
/// ```math
/// \begin{aligned}
/// n_o &= s_o\, \operatorname{sdBase}(q_o), \\
/// d_{o+1} &= \begin{cases}
///   \operatorname{smin}\!\bigl(\operatorname{smax}(n_o,\ d_o - \iota s_o,\ \kappa s_o),\ d_o,\ \kappa s_o\bigr)
///     & \text{(additive)} \\[4pt]
///   \operatorname{smax}\!\bigl(d_o,\ -n_o,\ \kappa s_o\bigr)
///     & \text{(subtractive)}
/// \end{cases} \\
/// q_{o+1} &= M\, q_o, \qquad s_{o+1} = \tfrac12 s_o.
/// \end{aligned}
/// ```
///
/// The clip keeps each octave near the host surface, and because the blend bands
/// `$\iota s_o, \kappa s_o$` scale with the amplitude the detail is self-similar
/// at every scale. `smin`/`smax` are the chosen [`SmoothKernel`]; `$M$` is the
/// [`OctaveTransform`] (a `$2\times$` rotation, so frequency doubles each octave).
/// Additive detail (`subtract = false`) builds a terrain-like crust; subtractive
/// (`subtract = true`) carves eroded solids.
///
/// The behavior is composed from four zero-cost (ZST, monomorphized) strategies,
/// each with an ergonomic builder:
/// - hash `H` (default [`SinHash`]) - [`with_hash`](Self::with_hash);
/// - inter-octave domain rotation `T` (default [`IqRotation`]) -
///   [`with_transform`](Self::with_transform);
/// - lattice primitive `C` (default [`SphereCell`]) -
///   [`with_primitive`](Self::with_primitive);
/// - smooth-combine kernel `K` (default [`Quadratic`]) -
///   [`with_kernel`](Self::with_kernel).
///
/// Works in any dimension via the [`SDF<V, N>`] impl (cost per octave is `$2^N$`).
#[derive(Debug, Clone, Copy)]
pub struct FbmDetail<
    V: SdfVector,
    S,
    P: Policy = DefaultPolicy,
    H: LatticeHash = SinHash,
    T: OctaveTransform = IqRotation,
    C: LatticePrimitive = SphereCell,
    K: SmoothKernel = Quadratic,
> {
    pub shape: S,
    /// Number of octaves (frequency-doubling layers). IQ uses 11.
    pub octaves: u32,
    /// Surface-tracking inflation per octave (`0.1` in the reference).
    pub inflate: V,
    /// Smooth-union/clip blend per octave (`0.3` in the reference).
    pub smooth: V,
    /// Carve detail out (`true`) instead of adding it (`false`).
    pub subtract: bool,
    /// Per-lattice-vertex hash strategy.
    pub hash: H,
    /// Inter-octave domain transform strategy.
    pub transform: T,
    /// Lattice primitive strategy.
    pub primitive: C,
    /// Smooth-combine kernel strategy.
    pub kernel: K,
    _policy: PhantomData<P>,
}

impl<V: SdfVector, S, P: Policy> FbmDetail<V, S, P, SinHash, IqRotation, SphereCell, Quadratic> {
    /// Additive detail with the reference defaults (11 octaves, inflate 0.1,
    /// smooth 0.3) and the [`SinHash`].
    #[inline(always)]
    pub fn new(shape: S) -> Self {
        Self {
            shape,
            octaves: 11,
            inflate: frac::<V, 1, 10>(),
            smooth: frac::<V, 3, 10>(),
            subtract: false,
            hash: SinHash,
            transform: IqRotation,
            primitive: SphereCell,
            kernel: Quadratic,
            _policy: PhantomData,
        }
    }

    /// Subtractive (erosion) detail with the reference defaults.
    #[inline(always)]
    pub fn carved(shape: S) -> Self {
        Self {
            subtract: true,
            ..Self::new(shape)
        }
    }
}

impl<V: SdfVector, S, P: Policy, H: LatticeHash, T: OctaveTransform, C: LatticePrimitive, K: SmoothKernel>
    FbmDetail<V, S, P, H, T, C, K>
{
    /// Swaps in a different [`LatticeHash`], keeping all other settings.
    #[inline(always)]
    pub fn with_hash<H2: LatticeHash>(self, hash: H2) -> FbmDetail<V, S, P, H2, T, C, K> {
        FbmDetail {
            shape: self.shape,
            octaves: self.octaves,
            inflate: self.inflate,
            smooth: self.smooth,
            subtract: self.subtract,
            hash,
            transform: self.transform,
            primitive: self.primitive,
            kernel: self.kernel,
            _policy: PhantomData,
        }
    }

    /// Swaps in a different [`OctaveTransform`], keeping all other settings.
    #[inline(always)]
    pub fn with_transform<T2: OctaveTransform>(self, transform: T2) -> FbmDetail<V, S, P, H, T2, C, K> {
        FbmDetail {
            shape: self.shape,
            octaves: self.octaves,
            inflate: self.inflate,
            smooth: self.smooth,
            subtract: self.subtract,
            hash: self.hash,
            transform,
            primitive: self.primitive,
            kernel: self.kernel,
            _policy: PhantomData,
        }
    }

    /// Swaps in a different [`LatticePrimitive`], keeping all other settings.
    #[inline(always)]
    pub fn with_primitive<C2: LatticePrimitive>(self, primitive: C2) -> FbmDetail<V, S, P, H, T, C2, K> {
        FbmDetail {
            shape: self.shape,
            octaves: self.octaves,
            inflate: self.inflate,
            smooth: self.smooth,
            subtract: self.subtract,
            hash: self.hash,
            transform: self.transform,
            primitive,
            kernel: self.kernel,
            _policy: PhantomData,
        }
    }

    /// Swaps in a different smooth-combine [`SmoothKernel`], keeping all other
    /// settings.
    #[inline(always)]
    pub fn with_kernel<K2: SmoothKernel>(self, kernel: K2) -> FbmDetail<V, S, P, H, T, C, K2> {
        FbmDetail {
            shape: self.shape,
            octaves: self.octaves,
            inflate: self.inflate,
            smooth: self.smooth,
            subtract: self.subtract,
            hash: self.hash,
            transform: self.transform,
            primitive: self.primitive,
            kernel,
            _policy: PhantomData,
        }
    }
}

impl<
    V: SdfVector + RealMathWithPolicy,
    const N: usize,
    S: SDF<V, N>,
    P: Policy,
    H: LatticeHash,
    T: OctaveTransform,
    C: LatticePrimitive,
    K: SmoothKernel,
> SDF<V, N> for FbmDetail<V, S, P, H, T, C, K>
{
    #[inline(always)]
    fn eval(&self, p: Vector<V, N>) -> V {
        let mut d = self.shape.eval(p);
        let mut q = p;
        let mut s = V::ONE;
        let mut octave = 0;
        while octave < self.octaves {
            let n = s * sd_base::<V, P, H, C, N>(q);
            let sm = self.smooth * s;
            if self.subtract {
                // carve: smax(d, -n, smooth*s)
                d = smax_k::<K, V>(d, -n, sm);
            } else {
                // add: clip the octave to the inflated host, then smooth-union
                let clipped = smax_k::<K, V>(n, d - self.inflate * s, sm);
                d = smin_k::<K, V>(clipped, d, sm);
            }
            q = T::apply(q);
            s *= V::HALF;
            octave += 1;
        }
        d
    }
}
