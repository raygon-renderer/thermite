//! Unit quaternion, one per 4-lane register.
//!
//! The rotation operations here are thin wrappers over `LinAlg4Vector`
//! primitives, [`quat4_product`](thermite::vector::LinAlg4Vector::quat4_product)
//! for composition and
//! [`quat4_vec3_product`](thermite::vector::LinAlg4Vector::quat4_vec3_product)
//! for rotating a vector, because those are what the backends specialize. The
//! vector rotation in particular picks its algorithm per ISA: the Double-Cross
//! (Giesen) method where shuffles are cheap, two dot products and a cross where
//! they are not.

use core::ops::{Add, Mul, Neg, Sub};

use thermite::{math::ScalarMath, prelude::*, simd::Simd3Vectors};

use super::{AosFloat, Matrix4, V3, V4, vector::Vector3Ext as _};

/// A quaternion `$[x, y, z, w]$`, `w` scalar, stored in one 4-lane register.
///
/// The rotation methods assume a **unit** quaternion. Use
/// [`normalize`](Self::normalize) or [`try_normalize`](Self::try_normalize) when
/// that is not guaranteed.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct Quaternion<S: Simd3Vectors, E: AosFloat<S>>(pub V4<S, E>);

impl<S: Simd3Vectors, E: AosFloat<S>> Default for Quaternion<S, E> {
    #[inline(always)]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl<S: Simd3Vectors, E: AosFloat<S>> Quaternion<S, E> {
    /// The rotation that does nothing, `$[0, 0, 0, 1]$`.
    ///
    /// The `w = 1` basis vector, so this is the same rodata-backed constant as
    /// the identity matrix's last column.
    pub const IDENTITY: Self = Self(Matrix4::<S, E>::IDENTITY.0[3]);

    #[inline(always)]
    pub const fn from_register(q: V4<S, E>) -> Self {
        Self(q)
    }

    #[inline(always)]
    pub fn new(x: E, y: E, z: E, w: E) -> Self {
        Self(V4::<S, E>::from_slice(&[x, y, z, w]))
    }

    /// From the vector (imaginary) part and the scalar part.
    #[inline(always)]
    pub fn from_parts(xyz: V3<S, E>, w: E) -> Self {
        Self(xyz.extend::<V4<S, E>>().insert::<3>(w))
    }

    #[inline(always)]
    pub fn x(self) -> E {
        self.0.extract::<0>()
    }

    #[inline(always)]
    pub fn y(self) -> E {
        self.0.extract::<1>()
    }

    #[inline(always)]
    pub fn z(self) -> E {
        self.0.extract::<2>()
    }

    #[inline(always)]
    pub fn w(self) -> E {
        self.0.extract::<3>()
    }

    /// The vector (imaginary) part, `$[x, y, z]$`.
    #[inline(always)]
    pub fn xyz(self) -> V3<S, E> {
        GenericVector::narrow(self.0)
    }

    #[inline(always)]
    pub fn dot(self, other: Self) -> E {
        self.0.dot4(other.0)
    }

    #[inline(always)]
    pub fn norm_sqr(self) -> E {
        self.dot(self)
    }

    #[inline(always)]
    pub fn norm(self) -> E {
        thermite::element::FloatElement::sqrt(self.norm_sqr())
    }

    /// Scales to unit length. A zero quaternion gives `NaN`; see
    /// [`try_normalize`](Self::try_normalize).
    #[inline(always)]
    pub fn normalize(self) -> Self {
        Self(self.0 * V4::<S, E>::splat(E::ONE / self.norm()))
    }

    /// Like [`normalize`](Self::normalize), but a zero or non-finite quaternion
    /// comes back as the identity rotation.
    #[inline(always)]
    pub fn try_normalize(self) -> Self {
        let norm_sqr = self.norm_sqr();

        if norm_sqr > E::ZERO && FloatVector::is_finite(self.0).all() {
            Self(self.0 * V4::<S, E>::splat(E::ONE / thermite::element::FloatElement::sqrt(norm_sqr)))
        } else {
            Self::IDENTITY
        }
    }

    /// The conjugate `$[-x, -y, -z, w]$`, which for a unit quaternion is the
    /// inverse rotation.
    #[inline(always)]
    pub fn conjugate(self) -> Self {
        // Negate the vector part by flipping the sign bits of the first three
        // lanes. The scalar lane is left alone.
        Self(-self.0).with_w(self.w())
    }

    /// The multiplicative inverse `$\bar{q}/\|q\|^2$`.
    ///
    /// For a unit quaternion this is just the [`conjugate`](Self::conjugate),
    /// which is cheaper, so prefer it when the quaternion is known normalized.
    #[inline(always)]
    pub fn inverse(self) -> Self {
        let s = V4::<S, E>::splat(E::ONE / self.norm_sqr());

        Self(self.conjugate().0 * s)
    }

    /// Replaces the scalar part.
    #[inline(always)]
    fn with_w(self, w: E) -> Self {
        Self(self.0.insert::<3>(w))
    }

    /// True when every component is finite.
    #[inline(always)]
    pub fn is_finite(self) -> bool {
        FloatVector::is_finite(self.0).all()
    }

    /// Rotates a vector by this **unit** quaternion.
    ///
    /// In lieu of a full policy system, `FAST` is used to pick the internal
    /// behavior. It is forwarded to
    /// [`quat4_vec3_product`](thermite::vector::LinAlg4Vector::quat4_vec3_product),
    /// see there for what it does.
    #[inline(always)]
    pub fn rotate_vector<const FAST: bool>(self, v: V3<S, E>) -> V3<S, E> {
        GenericVector::narrow(self.0.quat4_vec3_product::<FAST>(v.extend()))
    }

    /// Rotates a point by this quaternion.
    ///
    /// Identical to [`rotate_vector`](Self::rotate_vector), since a rotation has no
    /// translation and a point and a direction transform the same way, but
    /// named separately so the call site says which one it means.
    #[inline(always)]
    pub fn transform_point<const FAST: bool>(self, p: V3<S, E>) -> V3<S, E> {
        self.rotate_vector::<FAST>(p)
    }

    /// The rotation matrix for this **unit** quaternion.
    ///
    /// Trig-free: the entries are pairwise products of `$x, y, z, w$`.
    #[inline(always)]
    pub fn to_matrix(self) -> Matrix4<S, E> {
        Matrix4(self.0.quat_to_mat4::<true>())
    }

    /// The three rotated basis vectors (the columns of the rotation matrix).
    #[inline(always)]
    pub fn to_basis(self) -> [V3<S, E>; 3] {
        self.0.quat_to_mat3::<true>().map(GenericVector::narrow)
    }
}

impl<S: Simd3Vectors, E: AosFloat<S>> Quaternion<S, E> {
    /// Extracts a unit quaternion from the upper-left 3x3 rotation block of `m`,
    /// by Shepperd's method.
    ///
    /// Shepperd's method exists to avoid dividing by a near-zero component: it
    /// solves for whichever of `w, x, y, z` is largest and derives the other
    /// three from it. With one matrix per call that choice is a real four-way
    /// branch, where the SoA counterpart must compute all four candidates and
    /// blend. That is the payoff of the AoS layout on a branchy algorithm.
    #[inline(always)]
    pub fn from_matrix(m: &Matrix4<S, E>) -> Self {
        let (m00, m11, m22) = (m.get(0, 0), m.get(1, 1), m.get(2, 2));

        let trace = m00 + m11 + m22;

        let half = E::from_ratio(1, 2);

        if trace > E::ZERO {
            let s = thermite::element::FloatElement::sqrt(E::ONE + trace);
            let r = half / s;

            Self::new(
                (m.get(2, 1) - m.get(1, 2)) * r,
                (m.get(0, 2) - m.get(2, 0)) * r,
                (m.get(1, 0) - m.get(0, 1)) * r,
                s * half,
            )
        } else if m00 > m11 && m00 > m22 {
            let s = thermite::element::FloatElement::sqrt(E::ONE + m00 - (m11 + m22));
            let r = half / s;

            Self::new(
                s * half,
                (m.get(1, 0) + m.get(0, 1)) * r,
                (m.get(2, 0) + m.get(0, 2)) * r,
                (m.get(2, 1) - m.get(1, 2)) * r,
            )
        } else if m11 > m22 {
            let s = thermite::element::FloatElement::sqrt(E::ONE + m11 - (m22 + m00));
            let r = half / s;

            Self::new(
                (m.get(0, 1) + m.get(1, 0)) * r,
                s * half,
                (m.get(2, 1) + m.get(1, 2)) * r,
                (m.get(0, 2) - m.get(2, 0)) * r,
            )
        } else {
            let s = thermite::element::FloatElement::sqrt(E::ONE + m22 - (m00 + m11));
            let r = half / s;

            Self::new(
                (m.get(0, 2) + m.get(2, 0)) * r,
                (m.get(1, 2) + m.get(2, 1)) * r,
                s * half,
                (m.get(1, 0) - m.get(0, 1)) * r,
            )
        }
    }
}

impl<S: Simd3Vectors, E: AosFloat<S> + ScalarMath> Quaternion<S, E> {
    /// Rotation of `angle` radians about a **pre-normalized** axis:
    /// `$q = [\hat{a}\sin(\theta/2),\, \cos(\theta/2)]$`.
    #[inline(always)]
    pub fn from_axis_angle_raw(axis: V3<S, E>, angle: E) -> Self {
        let (sin_half, cos_half) = (angle * E::from_ratio(1, 2)).scalar_sin_cos();

        Self::from_parts(axis * V3::<S, E>::splat(sin_half), cos_half)
    }

    /// Rotation from an axis-angle vector: the direction is the axis, the
    /// magnitude is the angle in radians. A zero vector gives the identity.
    #[inline(always)]
    pub fn from_axis_angle(axis_angle: V3<S, E>) -> Self {
        let (axis, angle) = axis_angle.normalize_norm();

        if angle > E::ZERO && axis.is_all_finite() {
            Self::from_axis_angle_raw(axis, angle)
        } else {
            Self::IDENTITY
        }
    }

    /// Look-at rotation: the rotation carrying `+Z` onto the direction from
    /// `origin` to `target`.
    ///
    /// Matches [`Transform::look_at`](super::Transform::look_at) and the rest of
    /// the crate, where the camera looks down its own `+Z`.
    ///
    /// Two degeneracies, both real branches here where the SoA twin must blend:
    /// a target coincident with the origin (no direction to look along, so the
    /// identity), and a forward direction nearly parallel to `+/-Z`, where the
    /// axis `$\hat{z} \times \mathbf{f}$` vanishes. Looking along `+Z` is
    /// already the identity, and looking along `-Z` is a half turn, taken about
    /// `+Y` since any perpendicular axis will do.
    #[inline(always)]
    pub fn look_at(origin: V3<S, E>, target: V3<S, E>) -> Self {
        let (forward, len) = (target - origin).normalize_norm();

        // The tiny cutoff keeps the normalization from amplifying noise into a
        // direction, matching the SoA twin's 1e-12 on the squared length.
        let tiny = E::from_ratio(1, 1_000_000);

        // The finiteness test comes first: a NaN input normalizes to a NaN
        // direction, so this catches it before the ordered comparison below,
        // which a NaN would silently pass.
        if !forward.is_all_finite() || len <= tiny {
            return Self::IDENTITY;
        }

        let fz = forward.extract::<2>();

        let near_pole = E::from_ratio(999, 1000);

        let up = V3::<S, E>::from_slice(&[E::ZERO, E::ONE, E::ZERO]);

        if fz.abs() > near_pole {
            // At the poles the axis is zero. Looking along -Z is a half turn
            // about any perpendicular, and +Y is as good as any.
            let angle = if fz < E::ZERO { E::PI } else { E::ZERO };

            return Self::from_axis_angle_raw(up, angle);
        }

        // axis = z_hat x forward = (-fy, fx, 0), and the angle between +Z and
        // forward has cosine fz.
        let axis = V3::<S, E>::from_slice(&[-forward.extract::<1>(), forward.extract::<0>(), E::ZERO]);

        Self::from_axis_angle_raw(axis.normalize(), fz.scalar_acos())
    }

    /// From Euler angles in radians as `$(yaw, pitch, roll) = (R_Y, R_X, R_Z)$`,
    /// applied in that order.
    #[inline(always)]
    pub fn from_euler(yaw: E, pitch: E, roll: E) -> Self {
        let half = E::from_ratio(1, 2);

        let (sy, cy) = (yaw * half).scalar_sin_cos();
        let (sp, cp) = (pitch * half).scalar_sin_cos();
        let (sr, cr) = (roll * half).scalar_sin_cos();

        Self::new(
            (cy * cp * sr) - (sy * sp * cr),
            (sy * cp * sr) + (cy * sp * cr),
            (sy * cp * cr) - (cy * sp * sr),
            (cy * cp * cr) + (sy * sp * sr),
        )
        .normalize()
    }

    /// The axis-angle form of this rotation: the unit axis and the angle in
    /// radians, the inverse of
    /// [`from_axis_angle_raw`](Self::from_axis_angle_raw).
    ///
    /// Uses `$\theta = 2\,\mathrm{atan2}(\|xyz\|, w)$` rather than
    /// `$2\arccos w$`: `acos` loses precision near the identity, exactly where a
    /// small rotation needs to be read back accurately. A zero rotation has no
    /// axis, so it reports `+X` with a zero angle.
    #[inline(always)]
    pub fn to_axis_angle(self) -> (V3<S, E>, E) {
        let (axis, sin_half) = self.xyz().normalize_norm();

        if sin_half > E::ZERO {
            (axis, (E::ONE + E::ONE) * sin_half.scalar_atan2(self.w()))
        } else {
            (V3::<S, E>::from_slice(&[E::ONE, E::ZERO, E::ZERO]), E::ZERO)
        }
    }

    /// Spherical linear interpolation along the **shortest** arc.
    ///
    /// Falls back to a normalized lerp once the quaternions are within
    /// `$\cos\Omega > 0.9995$` of each other, where `$\sin\Omega$` cancels
    /// catastrophically and the two paths agree to within float precision.
    #[inline(always)]
    pub fn slerp(self, other: Self, t: E) -> Self {
        let d = self.dot(other);

        // q and -q are the same rotation, so flip the far one to take the short way.
        let (other, d) = if d < E::ZERO { (-other, -d) } else { (other, d) };

        let near = E::from_ratio(1999, 2000);

        if d > near {
            // Nearly coincident: a plain lerp, renormalized.
            return Self(self.0 + (other.0 - self.0) * V4::<S, E>::splat(t)).normalize();
        }

        let theta = d.scalar_acos();
        let (sin, cos) = (theta * t).scalar_sin_cos();

        // The component of `other` orthogonal to `self`, which with `self` spans
        // the arc between them.
        let perp = Self(other.0 - self.0 * V4::<S, E>::splat(d)).normalize();

        Self(self.0 * V4::<S, E>::splat(cos) + perp.0 * V4::<S, E>::splat(sin))
    }
}

impl<S: Simd3Vectors, E: AosFloat<S>> Quaternion<S, E> {
    /// The shortest-arc rotation taking `start` onto `end`.
    ///
    /// Half-angle construction
    /// `$q = [s \times e,\; \|s\|\|e\| + s \cdot e]$`, left unnormalized until
    /// the end. Antiparallel inputs collapse it to zero, since the rotation is a
    /// half turn about an arbitrary perpendicular axis that the formula cannot
    /// name, so those fall back to a turn about any perpendicular.
    #[inline(always)]
    pub fn from_rotation_arc(start: V3<S, E>, end: V3<S, E>) -> Self {
        let cross = start.cross3::<false>(end);

        let w = thermite::element::FloatElement::sqrt(start.norm_sqr() * end.norm_sqr()) + start.dot3(end);

        let q = Self::from_parts(cross, w);

        if q.norm_sqr() > E::ZERO {
            q.normalize()
        } else {
            // Antiparallel: any perpendicular axis is a valid half turn. Take the
            // cross with whichever cardinal axis is least aligned with `start`.
            let axis = start.cross3::<false>(least_aligned_axis(start));

            Self::from_parts(axis.try_normalize(), E::ZERO)
        }
    }
}

/// The cardinal axis least aligned with `v`, the standard trick for picking a
/// perpendicular that will not degenerate.
///
/// Generic over the vector type rather than over `(S, E)`, so the type falls out
/// of the argument, since inference cannot see through the `V3<S, E>` projection.
#[inline(always)]
fn least_aligned_axis<V: LinAlg3Vector>(v: V) -> V {
    let a = v.abs();
    let (x, y, z) = (a.extract::<0>(), a.extract::<1>(), a.extract::<2>());

    let zero = <V::Element as thermite::element::Element>::ZERO;
    let one = <V::Element as thermite::element::Element>::ONE;

    if x <= y && x <= z {
        V::from_slice(&[one, zero, zero])
    } else if y <= z {
        V::from_slice(&[zero, one, zero])
    } else {
        V::from_slice(&[zero, zero, one])
    }
}

/// Quaternion composition: `self * rhs` rotates by `rhs` first, then by `self`.
impl<S: Simd3Vectors, E: AosFloat<S>> Mul for Quaternion<S, E> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // TODO: `Mul` cannot carry a const param, so this pins the historical
        // (fast, one-sided-fused) arm. Deciding whether the operator should
        // instead be the exact-cancelling `false` arm, with speed opt-in via an
        // explicit `product::<FAST>` method, is still open.
        Self(self.0.quat4_product::<true>(rhs.0))
    }
}

impl<S: Simd3Vectors, E: AosFloat<S>> Mul<E> for Quaternion<S, E> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: E) -> Self {
        Self(self.0 * V4::<S, E>::splat(rhs))
    }
}

impl<S: Simd3Vectors, E: AosFloat<S>> Add for Quaternion<S, E> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl<S: Simd3Vectors, E: AosFloat<S>> Sub for Quaternion<S, E> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl<S: Simd3Vectors, E: AosFloat<S>> Neg for Quaternion<S, E> {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl<S: Simd3Vectors, E: AosFloat<S> + ScalarMath> Quaternion<S, E> {
    /// The exponential map: turns a **rotation vector** into the rotation it
    /// describes.
    ///
    /// The input is an axis-angle vector living in the quaternion's tangent
    /// space (direction is the axis, magnitude is the angle in radians), and
    /// the result is `$[\hat{v}\sin(\|v\|/2),\, \cos(\|v\|/2)]$`.
    ///
    /// Same construction as [`from_axis_angle`](Self::from_axis_angle). `exp` is
    /// what the literature calls it, and it is the half of the `exp`/`log` pair
    /// that [`log`](Self::log) inverts. Picture rotations as living on a curved
    /// space (the unit 4-sphere) and rotation vectors on the flat tangent plane
    /// at the identity: `exp` maps flat to curved, `log` maps back. Averaging,
    /// differencing and integrating rotations only make sense on the flat side.
    ///
    /// A zero vector maps to the identity rather than dividing by zero.
    #[inline(always)]
    pub fn exp(rotation_vector: V3<S, E>) -> Self {
        Self::from_axis_angle(rotation_vector)
    }

    /// The logarithmic map: recovers the rotation vector from a **unit**
    /// quaternion, the inverse of [`exp`](Self::exp).
    ///
    /// Returns the axis scaled by the angle in radians, so its magnitude is the
    /// amount of rotation. The identity maps to the zero vector.
    ///
    /// # What it is for
    ///
    /// `log` is what makes rotations subtractable. The "difference" between two
    /// orientations is `$\log(q_1 q_0^{-1})$`, a plain vector giving the axis
    /// and amount that carries one onto the other, and once rotations are
    /// vectors you can average them, filter them, hand them to a solver, or
    /// measure convergence with them. `exp` turns the answer back into a
    /// rotation. Slerp is exactly `$q_0 \exp(t \log(q_1 q_0^{-1}))$`, so
    /// [`slerp`](Self::slerp) and this pair are two views of one operation.
    #[inline(always)]
    pub fn log(self) -> V3<S, E> {
        let (axis, angle) = self.to_axis_angle();

        axis * V3::<S, E>::splat(angle)
    }

    /// Advances an orientation by an angular velocity over a timestep.
    ///
    /// `angular_velocity` is in radians per second, expressed as a vector whose
    /// direction is the instantaneous axis of spin and whose magnitude is the
    /// rate. `dt` is the timestep.
    ///
    /// # What it is for
    ///
    /// This is the inner loop of every rigid-body integrator. A body's
    /// orientation obeys `$\dot{q} = \tfrac{1}{2}\,\omega\,q$` (with `$\omega$`
    /// the angular velocity as a pure quaternion), and integrating that over one
    /// step is what advances a spinning object from one frame to the next.
    ///
    /// Uses the **exact** exponential update `$q' = \exp(\omega\,dt/2)\,q$`
    /// rather than the usual first-order `$q + \tfrac{1}{2}\,dt\,\omega q$`. The
    /// cheap form leaves the quaternion off the unit sphere and needs
    /// renormalizing every step, and it systematically under-rotates, so a body
    /// spinning fast enough visibly lags. The exponential form is a true
    /// rotation by construction, so it stays normalized and is accurate at any
    /// spin rate, at the cost of one `sin_cos` the cheap form skips.
    ///
    /// The result is renormalized anyway, to stop round-off accumulating over
    /// thousands of steps.
    #[inline(always)]
    pub fn integrate(self, angular_velocity: V3<S, E>, dt: E) -> Self {
        // The rotation vector covered during this step is omega * dt, and `exp`
        // halves it internally, which is the 1/2 in the kinematic equation.
        let delta = Self::exp(angular_velocity * V3::<S, E>::splat(dt));

        (delta * self).try_normalize()
    }
}
