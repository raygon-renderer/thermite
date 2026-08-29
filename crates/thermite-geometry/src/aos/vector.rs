//! Vector and point operations on a bare 3-lane register.
//!
//! These are extension traits with blanket impls, not newtypes: the register
//! already has the arithmetic operators, and `Vector3Ext` only adds the
//! geometry (norms, normalization, reflection) on top. Import it and every
//! 3-lane float register in scope gains the methods.

use thermite::{
    element::FloatElement,
    math::{ScalarMath, SpatialMath},
    prelude::*,
};

use super::Vec3;

/// Geometric operations on a 3-lane register interpreted as a direction.
///
/// Everything that reduces across the components (`dot3`, the norms) returns a
/// **scalar**, not a vector, since one object per register means one answer.
pub trait Vector3Ext: Vec3 {
    /// `$\|v\|^2 = v \cdot v$`. No square root, so prefer it whenever you only
    /// compare lengths.
    #[inline(always)]
    fn norm_sqr(self) -> Self::Element {
        self.dot3(self)
    }

    /// `$\|v\| = \sqrt{v \cdot v}$`.
    #[inline(always)]
    fn norm(self) -> Self::Element {
        FloatElement::sqrt(self.norm_sqr())
    }

    /// The sum of the absolute components, `$\sum_i |v_i|$`.
    ///
    /// The `3` suffix follows the `dot3`/`sum_elements3` convention for a
    /// **horizontal** reduction over the first three lanes, and distinguishes it
    /// from [`SpatialMath::l1_norm`](thermite::math::SpatialMath::l1_norm),
    /// which is per-lane and returns a vector.
    #[inline(always)]
    fn l1_norm3(self) -> Self::Element {
        self.abs().sum_elements3()
    }

    /// The largest absolute component, `$\max_i |v_i|$`, the Chebyshev norm.
    ///
    /// Horizontal, like [`l1_norm3`](Self::l1_norm3). See there on the naming.
    #[inline(always)]
    fn linf_norm3(self) -> Self::Element {
        self.abs().max_element3()
    }

    /// `$\hat{v} = v / \|v\|$`. A zero-length input gives `NaN`; see
    /// [`try_normalize`](Self::try_normalize).
    #[inline(always)]
    fn normalize(self) -> Self {
        // One reciprocal shared by all three components, rather than three divides.
        self * Self::splat(Self::Element::ONE / self.norm())
    }

    /// The normalized vector *and* the original norm, sharing one reciprocal.
    #[inline(always)]
    fn normalize_norm(self) -> (Self, Self::Element) {
        let norm = self.norm();

        (self * Self::splat(Self::Element::ONE / norm), norm)
    }

    /// Like [`normalize`](Self::normalize), but a zero-length or non-finite
    /// vector comes back as zero instead of `NaN`.
    ///
    /// Unlike the SoA counterpart this is a real branch: there is one vector
    /// here, so there is one answer, and a predictable branch beats computing
    /// both sides.
    #[inline(always)]
    fn try_normalize(self) -> Self {
        let norm_sqr = self.norm_sqr();

        // Finiteness is tested on the vector rather than the scalar norm:
        // `FloatElement` has no `is_finite`, and a non-finite component would
        // reach `norm_sqr` as an infinity or NaN anyway.
        if norm_sqr > Self::Element::ZERO && self.is_all_finite() {
            self * Self::splat(Self::Element::ONE / FloatElement::sqrt(norm_sqr))
        } else {
            Self::ZERO
        }
    }

    /// The component-wise reciprocal by **exact IEEE division**, so `1/0` is a
    /// signed infinity rather than `NaN`.
    ///
    /// Distinct from [`CoreMath::reciprocal`](thermite::math::CoreMath::reciprocal),
    /// which is policy-based: on any backend with `HAS_APPROX_RCP` (every x86
    /// one) it computes `rcp` plus a Newton refinement below `Best` precision,
    /// and that step, `$y(2 - dy)$`, evaluates to `inf * (2 - 0 * inf)` =
    /// `NaN` at `$d = 0$`. It is also merely approximate elsewhere (`1/2`
    /// measured as `0.49999997`).
    ///
    /// Use this wherever a division by zero must yield an infinity that later
    /// min/max operations act on, above all
    /// [`Bounds3::intersect_ray`](super::Bounds3::intersect_ray), whose
    /// empty-or-full interval per axis depends on exactly that behaviour.
    #[inline(always)]
    fn reciprocal_exact(self) -> Self {
        Self::ONE / self
    }

    /// Linear interpolation from `self` to `other` by `t`.
    #[inline(always)]
    fn lerp(self, other: Self, t: Self::Element) -> Self {
        Self::splat(t).mix(self, other)
    }

    /// Reflects `self` about the unit normal `n`:
    /// `$\mathbf{i} - 2(\mathbf{n}\cdot\mathbf{i})\mathbf{n}$`.
    ///
    /// GLSL convention: the incident vector points *into* the surface, so the
    /// result points away from it. The AoS mirror of
    /// [`soa`'s `reflect`](crate::soa::prim::vector::VectorOps::reflect).
    #[inline(always)]
    fn reflect(self, n: Self) -> Self {
        // 2 (n.i) built at the vector level: `Self::TWO` is a const-folded splat,
        // where the scalar `2` would need a splat of its own.
        let coeff = self.dot3(n);

        n.nmul_adde(Self::splat(coeff + coeff), self)
    }

    // No `refract` here: [`LinAlg3Vector::refract`] already is the
    // implementation, a backend primitive that computes the replicated dot
    // product with the cross-product rotations rather than an extract-and-splat.
    // Only the checked form below adds anything.

    /// [`LinAlg3Vector::refract`], plus whether it actually refracted (`false`
    /// means total internal reflection, and the result is the zero vector).
    #[inline(always)]
    fn refract_checked(self, n: Self, eta: Self::Element) -> (Self, bool) {
        let d = self.dot3(n);
        let k = Self::Element::ONE - eta * eta * (Self::Element::ONE - d * d);

        (LinAlg3Vector::refract(self, n, eta), k >= Self::Element::ZERO)
    }

    /// Returns `self` flipped, if needed, to lie in the hemisphere opposing
    /// `incident`.
    ///
    /// The test is against `reference` (usually the geometric normal), which is
    /// what lets a geometric normal decide the orientation of a shading normal.
    #[inline(always)]
    fn faceforward(self, incident: Self, reference: Self) -> Self {
        if incident.dot3(reference) < Self::Element::ZERO {
            self
        } else {
            -self
        }
    }

    /// The angle between two vectors in radians, by the numerically stable
    /// `atan2` form rather than `acos(dot)`.
    ///
    /// `acos` loses all precision for nearly-parallel inputs (its derivative is
    /// unbounded at `+/-1`, and `dot` saturates there), while the cross/dot form stays
    /// accurate across the whole range.
    ///
    /// `atan2` is the one operation here that `FloatElement` does not provide,
    /// so the extra bound sits on this method rather than on the whole trait.
    #[inline(always)]
    fn angle_between(self, other: Self) -> Self::Element
    where
        Self: SpatialMath,
        Self::Element: ScalarMath,
    {
        self.cross3::<false>(other).norm().scalar_atan2(self.dot3(other))
    }

    /// The component of `self` parallel to `onto` (which need not be unit).
    #[inline(always)]
    fn project_onto(self, onto: Self) -> Self {
        onto * Self::splat(self.dot3(onto) / onto.norm_sqr())
    }

    /// The component of `self` perpendicular to `from`.
    #[inline(always)]
    fn reject_from(self, from: Self) -> Self {
        self - self.project_onto(from)
    }

    /// True when every component is finite.
    #[inline(always)]
    fn is_all_finite(self) -> bool {
        FloatVector::is_finite(self).all()
    }
}

impl<V: Vec3> Vector3Ext for V {}

/// Operations on a 3-lane register interpreted as a position.
///
/// A point and a vector share a representation here, so this trait exists for
/// the operations whose *meaning* is affine: distances between positions rather
/// than lengths of displacements.
pub trait Point3Ext: Vec3 {
    /// The Euclidean distance between two points.
    #[inline(always)]
    fn distance(self, other: Self) -> Self::Element {
        (self - other).norm()
    }

    /// The squared Euclidean distance. No square root, enough for comparisons.
    #[inline(always)]
    fn distance_sqr(self, other: Self) -> Self::Element {
        (self - other).norm_sqr()
    }

    /// The point halfway between `self` and `other`.
    #[inline(always)]
    fn midpoint(self, other: Self) -> Self {
        Self::HALF.mix(self, other)
    }
}

impl<V: Vec3> Point3Ext for V {}
