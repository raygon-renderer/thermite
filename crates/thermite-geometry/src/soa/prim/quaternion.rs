use core::ops::{Add, Div, Mul, Neg, Sub};

use thermite::{
    mask::{GenericMask, GenericSelectable},
    math::{RealMath, SpatialMath, TranscendentalMath},
    vector::FloatVector,
};

use super::{Matrix, Point, Vector, vector::VectorOps as _};

/// Unit quaternion `[x, y, z, w]`, `w` scalar, one quaternion per lane.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct Quaternion<V: FloatVector>(pub [V; 4]);

impl<V: FloatVector> Default for Quaternion<V> {
    #[inline(always)]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl<V: FloatVector> GenericSelectable for Quaternion<V> {
    type SelectableMask = V::Mask;

    #[inline(always)]
    fn select<M>(mask: M, t: Self, f: Self) -> Self
    where
        Self::SelectableMask: thermite::mask::CastMask<M>,
    {
        let mask = <V::Mask as thermite::mask::CastMask<M>>::mask_from(mask);

        let mut result = t;

        for i in 0..4 {
            result.0[i] = mask.select(t.0[i], f.0[i]);

            unsafe { result.0[i].block_autovectorization() };
        }

        result
    }
}

impl<V: FloatVector> Quaternion<V> {
    /// The rotation that does nothing: `[0, 0, 0, 1]`.
    pub const IDENTITY: Self = Self([V::ZERO, V::ZERO, V::ZERO, V::ONE]);

    #[inline(always)]
    pub const fn new(x: V, y: V, z: V, w: V) -> Self {
        Self([x, y, z, w])
    }

    #[inline(always)]
    pub const fn from_array(components: [V; 4]) -> Self {
        Self(components)
    }

    /// From the vector (imaginary) part and the scalar part.
    #[inline(always)]
    pub const fn from_parts(xyz: Vector<V, 3>, w: V) -> Self {
        Self([xyz.0[0], xyz.0[1], xyz.0[2], w])
    }

    #[inline(always)]
    pub const fn x(self) -> V {
        self.0[0]
    }

    #[inline(always)]
    pub const fn y(self) -> V {
        self.0[1]
    }

    #[inline(always)]
    pub const fn z(self) -> V {
        self.0[2]
    }

    #[inline(always)]
    pub const fn w(self) -> V {
        self.0[3]
    }

    /// The vector (imaginary) part, `[x, y, z]`.
    #[inline(always)]
    pub const fn xyz(self) -> Vector<V, 3> {
        Vector([self.0[0], self.0[1], self.0[2]])
    }

    /// A mask that is set for the lanes whose every component is finite.
    #[inline(always)]
    pub fn is_finite(self) -> V::Mask {
        let mut mask = self.0[0].is_finite();

        for i in 1..4 {
            mask &= self.0[i].is_finite();
        }

        mask
    }

    #[inline(always)]
    pub fn dot(self, other: Self) -> V {
        let mut result = self.0[0] * other.0[0];

        for i in 1..4 {
            result = self.0[i].mul_adde(other.0[i], result);
        }

        result
    }

    #[inline(always)]
    pub fn norm_sqr(self) -> V {
        self.dot(self)
    }

    #[inline(always)]
    pub fn norm(self) -> V {
        self.norm_sqr().sqrt()
    }

    /// Scales to unit length. Zero-length lanes come back as `NaN`; use
    /// [`try_normalize`](Self::try_normalize) if that is possible.
    #[inline(always)]
    pub fn normalize(self) -> Self {
        self * (V::ONE / self.norm())
    }

    /// Like [`normalize`](Self::normalize), but lanes whose norm is zero or
    /// non-finite come back as the identity rotation.
    ///
    /// The guard is a select rather than a branch: the reciprocal is computed for
    /// every lane (with the norm forced to one where it would be degenerate, so no
    /// lane ever divides by zero and no `NaN` is manufactured), then the degenerate
    /// lanes are overwritten with the identity.
    #[inline(always)]
    pub fn try_normalize(self) -> Self {
        let norm_sqr = self.norm_sqr();

        let ok = norm_sqr.cmp_gt(V::ZERO) & norm_sqr.is_finite();

        let inv_norm = V::ONE / ok.select(norm_sqr, V::ONE).sqrt();

        ok.select(self * inv_norm, Self::IDENTITY)
    }

    /// The conjugate `[-x, -y, -z, w]`, which for a unit quaternion is the
    /// inverse rotation.
    #[inline(always)]
    pub fn conjugate(self) -> Self {
        Self([-self.0[0], -self.0[1], -self.0[2], self.0[3]])
    }

    /// The multiplicative inverse, `$\bar{\mathbf{q}} / \|\mathbf{q}\|^2$`.
    ///
    /// For a unit quaternion this is just the [`conjugate`](Self::conjugate), which
    /// is cheaper, so prefer that when the quaternion is known to be normalized.
    #[inline(always)]
    pub fn inverse(self) -> Self {
        self.conjugate() * (V::ONE / self.norm_sqr())
    }

    /// Rotate a free vector by this (unit) quaternion.
    ///
    /// There is no hardware quaternion-vector primitive in the SoA layout, so this
    /// is the two-cross-product identity
    /// `$\mathbf{v}' = \mathbf{v} + 2\,\mathbf{u} \times (\mathbf{u} \times \mathbf{v} + w\,\mathbf{v})$`
    /// with `$\mathbf{u}$` the vector part, which costs far less than expanding the
    /// full `$\mathbf{q}\mathbf{v}\bar{\mathbf{q}}$` sandwich.
    #[inline(always)]
    pub fn rotate_vector(self, v: Vector<V, 3>) -> Vector<V, 3> {
        let u = self.xyz();
        let w = self.0[3];

        let t = u.cross(v) * V::TWO;

        // t * w + v, then + u x t: the FMA folds the scale of t into the add.
        t.mul_adde(w, v) + u.cross(t)
    }

    /// Rotate a point by this quaternion. Identical to
    /// [`rotate_vector`](Self::rotate_vector) (a rotation has no translation), but
    /// named separately for clarity at the call site.
    #[inline(always)]
    pub fn transform_point(self, p: Point<V, 3>) -> Point<V, 3> {
        Point::from(self.rotate_vector(Vector::from(p)))
    }

    /// The rotation matrix for this quaternion.
    ///
    /// Uses the standard expansion (see the struct docs), which requires a unit
    /// quaternion, so the input is normalized first. Lanes whose norm is zero or
    /// non-finite are replaced by the identity quaternion, the same select as
    /// [`try_normalize`](Self::try_normalize), which expands to the identity
    /// matrix, so no branch is needed on the matrix itself.
    #[rustfmt::skip]
    #[inline(always)]
    pub fn to_matrix(self) -> Matrix<V, 4, 4> {
        let q = self.try_normalize();

        let (x, y, z, w) = (q.0[0], q.0[1], q.0[2], q.0[3]);

        let (xx, yy, zz) = (x * x, y * y, z * z);
        let (wx, wy, wz) = (w * x, w * y, w * z);
        let (xy, xz, yz) = (x * y, x * z, y * z);

        let m00 = V::TWO.nmul_adde(yy + zz, V::ONE);
        let m01 = V::TWO * (xy - wz);
        let m02 = V::TWO * (xz + wy);
        let m10 = V::TWO * (xy + wz);
        let m11 = V::TWO.nmul_adde(xx + zz, V::ONE);
        let m12 = V::TWO * (yz - wx);
        let m20 = V::TWO * (xz - wy);
        let m21 = V::TWO * (yz + wx);
        let m22 = V::TWO.nmul_adde(xx + yy, V::ONE);

        // Column-major: each inner array is one column.
        Matrix([
            [m00, m10, m20, V::ZERO],
            [m01, m11, m21, V::ZERO],
            [m02, m12, m22, V::ZERO],
            [V::ZERO, V::ZERO, V::ZERO, V::ONE],
        ])
    }

    /// The three rotated basis vectors, the columns of
    /// [`to_matrix`](Self::to_matrix)'s linear part.
    ///
    /// Cheaper than building the matrix when all you want is where the axes
    /// went, and it is what a [`TangentFrame`](super::TangentFrame) or a
    /// camera basis is assembled from.
    #[inline(always)]
    pub fn to_basis(self) -> [Vector<V, 3>; 3] {
        let m = self.to_matrix();

        [
            Vector([m.0[0][0], m.0[0][1], m.0[0][2]]),
            Vector([m.0[1][0], m.0[1][1], m.0[1][2]]),
            Vector([m.0[2][0], m.0[2][1], m.0[2][2]]),
        ]
    }

    /// Extracts a unit quaternion from the upper-left `$3\times3$` rotation
    /// submatrix, by Shepperd's method.
    ///
    /// Shepperd's method exists to avoid dividing by a near-zero component: it
    /// solves for whichever of `w, x, y, z` is largest (equivalently, whichever of
    /// the trace and the three diagonal entries dominates) and derives the other
    /// three from it. That choice is a four-way branch in scalar code, but here all
    /// four candidate quaternions are computed and a chain of selects picks the
    /// right one per lane. The three unselected candidates may hold `NaN` (their
    /// radicand goes negative) or infinities, which is harmless because a select never
    /// propagates the value of the lane it discards.
    #[inline(always)]
    pub fn from_matrix(m: &Matrix<V, 4, 4>) -> Self {
        // m_ij = row i, column j. The storage is column-major, hence [j][i].
        let m00 = m.0[0][0];
        let m01 = m.0[1][0];
        let m02 = m.0[2][0];
        let m10 = m.0[0][1];
        let m11 = m.0[1][1];
        let m12 = m.0[2][1];
        let m20 = m.0[0][2];
        let m21 = m.0[1][2];
        let m22 = m.0[2][2];

        let trace = m00 + m11 + m22;

        // Each candidate divides by twice its own (largest) component, and the select
        // below guarantees the divisor is the one that is far from zero.
        let sw = (V::ONE + trace).sqrt();
        let sx = ((m00 - (m11 + m22)) + V::ONE).sqrt();
        let sy = ((m11 - (m22 + m00)) + V::ONE).sqrt();
        let sz = ((m22 - (m00 + m11)) + V::ONE).sqrt();

        // A zero s would give an infinite r, so force those lanes to zero, which
        // yields the zero quaternion and lets try_normalize fall back to identity.
        let rw = sw.cmp_ne(V::ZERO).select(V::HALF / sw, V::ZERO);
        let rx = sx.cmp_ne(V::ZERO).select(V::HALF / sx, V::ZERO);
        let ry = sy.cmp_ne(V::ZERO).select(V::HALF / sy, V::ZERO);
        let rz = sz.cmp_ne(V::ZERO).select(V::HALF / sz, V::ZERO);

        let qw = Self([(m21 - m12) * rw, (m02 - m20) * rw, (m10 - m01) * rw, sw * V::HALF]);
        let qx = Self([sx * V::HALF, (m10 + m01) * rx, (m20 + m02) * rx, (m21 - m12) * rx]);
        let qy = Self([(m01 + m10) * ry, sy * V::HALF, (m21 + m12) * ry, (m02 - m20) * ry]);
        let qz = Self([(m02 + m20) * rz, (m12 + m21) * rz, sz * V::HALF, (m10 - m01) * rz]);

        let use_w = trace.cmp_gt(V::ZERO);
        // The nesting order encodes the original if/else-if priority, so these two
        // masks do not need to exclude the cases tested above them.
        let use_y = m11.cmp_gt(m00) & m11.cmp_gt(m22);
        let use_z = m22.cmp_gt(m00);

        let q = use_w.select(qw, use_y.select(qy, use_z.select(qz, qx)));

        q.try_normalize()
    }
}

impl<V: FloatVector + TranscendentalMath> Quaternion<V> {
    /// Spherical linear interpolation between two unit quaternions.
    ///
    /// `$\mathrm{slerp}(\mathbf{q}_0, \mathbf{q}_1, t) = \mathbf{q}_0 \cos(t\Omega) + \mathbf{q}_\perp \sin(t\Omega)$`
    /// where `$\cos\Omega = \mathbf{q}_0 \cdot \mathbf{q}_1$` and `$\mathbf{q}_\perp$`
    /// is the component of `$\mathbf{q}_1$` orthogonal to `$\mathbf{q}_0$`.
    ///
    /// Both paths (the spherical one and the normalized-lerp fallback used when
    /// `$\Omega \approx 0$`, where `$\sin\Omega$` cancels catastrophically) are
    /// evaluated for every lane and blended, since different lanes may need
    /// different paths. The fallback lanes' spherical result is `NaN` (the
    /// orthogonal component is zero and cannot be normalized), which the select
    /// discards.
    #[inline(always)]
    pub fn slerp(self, other: Self, t: V) -> Self {
        let d = self.dot(other);

        // Shortest arc: q and -q are the same rotation, so flip the far one. A
        // masked negate does this in one op per component, where negating the whole
        // quaternion and blending it back would cost a negate and a select each.
        let flip = d.cmp_lt(V::ZERO);

        let mut other = other;

        for i in 0..4 {
            other.0[i] = other.0[i].neg_c(flip);
        }

        let cos_theta = d.abs();

        let nlerp = (self * (V::ONE - t) + other * t).normalize();

        let theta = cos_theta.clamp(V::NEG_ONE, V::ONE).acos();
        let qperp = (other - self * cos_theta).normalize();
        let (sin, cos) = (theta * t).sin_cos();
        let spherical = self * cos + qperp * sin;

        // 0.9995 is the usual cutoff: beyond it the two quaternions are close
        // enough that a normalized lerp is within float precision of the arc.
        let near: V = thermite::const_splat!(ratio <V::Element>: 1999, 2000);

        cos_theta.cmp_gt(near).select(nlerp, spherical)
    }

    /// Build from Euler angles in radians as `$(yaw, pitch, roll) = (R_Y, R_X, R_Z)$`
    /// applied in that order.
    #[inline(always)]
    pub fn from_euler(yaw: V, pitch: V, roll: V) -> Self {
        let (sy, cy) = (yaw * V::HALF).sin_cos();
        let (sp, cp) = (pitch * V::HALF).sin_cos();
        let (sr, cr) = (roll * V::HALF).sin_cos();

        Self([
            (cy * cp * sr) - (sy * sp * cr),
            (sy * cp * sr) + (cy * sp * cr),
            (sy * cp * cr) - (cy * sp * sr),
            (cy * cp * cr) + (sy * sp * sr),
        ])
        .normalize()
    }

    /// Rotation of `angle` radians around a **pre-normalized** axis.
    ///
    /// `$\mathbf{q} = [\hat{\mathbf{a}}\sin(\theta/2),\; \cos(\theta/2)]$`
    #[inline(always)]
    pub fn from_axis_angle_raw(axis: Vector<V, 3>, angle: V) -> Self {
        let (sin_half, cos_half) = (angle * V::HALF).sin_cos();

        let xyz = axis * sin_half;

        Self([xyz[0], xyz[1], xyz[2], cos_half])
    }
}

impl<V: FloatVector + SpatialMath> Quaternion<V> {
    /// Shortest-arc rotation taking `start` onto `end`.
    ///
    /// Half-angle construction:
    /// `$\mathbf{q} = [\mathbf{s} \times \mathbf{e},\; \|\mathbf{s}\|\|\mathbf{e}\| + \mathbf{s}\cdot\mathbf{e}]$`,
    /// unnormalized. Antiparallel inputs collapse this to the zero quaternion (the
    /// rotation is a half turn about an arbitrary perpendicular axis, which the
    /// formula cannot name), and [`try_normalize`](Self::try_normalize) turns those
    /// lanes into the identity.
    #[inline(always)]
    pub fn from_rotation_arc(start: Vector<V, 3>, end: Vector<V, 3>) -> Self {
        let cross = start.cross(end);

        let w = (start.norm_sqr() * end.norm_sqr()).sqrt() + start.dot(&end);

        Self([cross[0], cross[1], cross[2], w]).try_normalize()
    }
}

impl<V: FloatVector + RealMath> Quaternion<V> {
    /// The axis-angle form of this rotation: the unit axis and the angle in
    /// radians, the inverse of
    /// [`from_axis_angle_raw`](Self::from_axis_angle_raw).
    ///
    /// Uses `$\theta = 2\,\mathrm{atan2}(\|xyz\|, w)$` rather than
    /// `$2\arccos w$`. `acos` loses precision exactly where it matters most -
    /// near the identity, where `w` approaches 1 and its derivative blows up -
    /// which is where a small rotation needs to be read back accurately. The
    /// `atan2` form stays accurate across the whole range and needs no clamping.
    ///
    /// A zero rotation names no axis, so those lanes report `+X` with a zero
    /// angle rather than `NaN`.
    #[inline(always)]
    pub fn to_axis_angle(self) -> (Vector<V, 3>, V) {
        // The vector part has length sin(theta/2), and normalizing it hands back both
        // the axis and that sine in one pass.
        let (axis, sin_half) = self.xyz().normalize_norm();

        let ok = sin_half.cmp_gt(V::ZERO) & sin_half.is_finite();

        (
            Vector::select(ok, axis, Vector::<V, 3>::X),
            ok.select(V::TWO * sin_half.atan2(self.0[3]), V::ZERO),
        )
    }

    /// Rotation from an axis-angle vector: direction is the axis, magnitude is the
    /// angle in radians.
    ///
    /// A zero-length input names no axis, so those lanes select the identity rather
    /// than dividing by zero. The division itself is made safe first (the length is
    /// forced to one where it is zero), so the degenerate lanes never produce a
    /// `NaN` that a later `sin_cos` would have to chew through.
    #[inline(always)]
    pub fn from_axis_angle(axis_angle: Vector<V, 3>) -> Self {
        let len_sqr = axis_angle.norm_sqr();

        let ok = len_sqr.cmp_gt(V::ZERO) & len_sqr.is_finite();

        let angle = ok.select(len_sqr, V::ONE).sqrt();
        let axis = axis_angle * (V::ONE / angle);

        ok.select(Self::from_axis_angle_raw(axis, angle), Self::IDENTITY)
    }

    /// Look-at rotation: the rotation carrying `+Z` onto the direction from
    /// `origin` to `target`.
    ///
    /// Matches [`Transform::look_at`](super::Transform::look_at) and the rest of
    /// the crate, where the camera looks down its own `+Z`.
    ///
    /// Two degeneracies, both handled by selects instead of early returns: a
    /// target coincident with the origin (no direction to look along, so the
    /// identity), and a forward direction nearly parallel to `+/-Z`, where the
    /// axis `$\hat{z} \times \mathbf{f}$` vanishes. Looking along `+Z` is
    /// already the identity, and looking along `-Z` is a half turn, taken about
    /// `+Y` since any perpendicular axis will do.
    #[inline(always)]
    pub fn look_at(origin: Point<V, 3>, target: Point<V, 3>) -> Self {
        let diff = target - origin;

        let len_sqr = diff.norm_sqr();

        // 1e-12 keeps the reciprocal below from amplifying noise into a direction.
        let tiny: V = thermite::const_splat!(ratio <V::Element>: 1, 1_000_000_000_000);

        let ok = len_sqr.cmp_gt(tiny) & len_sqr.is_finite();

        let forward = diff * (V::ONE / ok.select(len_sqr, V::ONE).sqrt());

        let (fx, fy, fz) = (forward[0], forward[1], forward[2]);

        // The rotation axis is z_hat x forward = (-fy, fx, 0), and the angle is
        // the one between +Z and forward, so cos(angle) is just fz.
        let axis = Vector([-fy, fx, V::ZERO]);
        let general = Self::from_axis_angle_raw(axis.normalize(), fz.acos());

        // At the poles that axis is zero. Looking along -Z is a half turn about
        // any perpendicular, and +Y is as good as any.
        let pole_angle = fz.cmp_lt(V::ZERO).select(V::PI, V::ZERO);
        let pole = Self::from_axis_angle_raw(Vector::<V, 3>::Y, pole_angle);

        let near_pole: V = thermite::const_splat!(ratio <V::Element>: 999, 1000);

        let q = fz.abs().cmp_gt(near_pole).select(pole, general);

        ok.select(q, Self::IDENTITY)
    }
}

impl<V: FloatVector> Add for Quaternion<V> {
    type Output = Self;

    #[inline(always)]
    fn add(mut self, rhs: Self) -> Self {
        for i in 0..4 {
            self.0[i] += rhs.0[i];

            unsafe { self.0[i].block_autovectorization() };
        }

        self
    }
}

impl<V: FloatVector> Sub for Quaternion<V> {
    type Output = Self;

    #[inline(always)]
    fn sub(mut self, rhs: Self) -> Self {
        for i in 0..4 {
            self.0[i] -= rhs.0[i];

            unsafe { self.0[i].block_autovectorization() };
        }

        self
    }
}

impl<V: FloatVector> Neg for Quaternion<V> {
    type Output = Self;

    #[inline(always)]
    fn neg(mut self) -> Self {
        for i in 0..4 {
            self.0[i] = -self.0[i];

            unsafe { self.0[i].block_autovectorization() };
        }

        self
    }
}

impl<V: FloatVector> Mul<V> for Quaternion<V> {
    type Output = Self;

    #[inline(always)]
    fn mul(mut self, rhs: V) -> Self {
        for i in 0..4 {
            self.0[i] *= rhs;

            unsafe { self.0[i].block_autovectorization() };
        }

        self
    }
}

impl<V: FloatVector> Div<V> for Quaternion<V> {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: V) -> Self {
        // One reciprocal shared by all four components, as elsewhere in the crate.
        self * (V::ONE / rhs)
    }
}

/// Quaternion composition: the Hamilton product `self * rhs` rotates by `rhs`
/// first, then by `self`.
impl<V: FloatVector> Mul for Quaternion<V> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let (ax, ay, az, aw) = (self.0[0], self.0[1], self.0[2], self.0[3]);
        let (bx, by, bz, bw) = (rhs.0[0], rhs.0[1], rhs.0[2], rhs.0[3]);

        Self([
            aw.mul_adde(bx, ax.mul_adde(bw, ay.mul_adde(bz, -(az * by)))),
            aw.mul_adde(by, ay.mul_adde(bw, az.mul_adde(bx, -(ax * bz)))),
            aw.mul_adde(bz, az.mul_adde(bw, ax.mul_adde(by, -(ay * bx)))),
            aw.mul_adde(bw, -ax.mul_adde(bx, ay.mul_adde(by, az * bz))),
        ])
    }
}

impl<V: FloatVector + RealMath> Quaternion<V> {
    /// The exponential map: turns a **rotation vector** into the rotation it
    /// describes.
    ///
    /// The input is an axis-angle vector living in the quaternion's tangent
    /// space (direction is the axis, magnitude is the angle in radians), and the
    /// result is `$[\hat{v}\sin(\|v\|/2),\, \cos(\|v\|/2)]$`.
    ///
    /// This is the same construction as
    /// [`from_axis_angle`](Self::from_axis_angle). The name `exp` is what the
    /// literature calls it, and it is the half of the `exp`/`log` pair that
    /// [`log`](Self::log) inverts. Think of rotations as living on a curved
    /// space (the unit 4-sphere) and rotation vectors as living on the flat
    /// tangent plane at the identity: `exp` maps flat to curved, `log` maps back.
    /// Averaging, differencing and integrating rotations are all operations that
    /// only make sense on the flat side.
    ///
    /// A zero vector maps to the identity rather than dividing by zero.
    #[inline(always)]
    pub fn exp(rotation_vector: Vector<V, 3>) -> Self {
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
    /// vectors you can average them, filter them, feed them to a solver, or
    /// measure convergence with them. Going the other way, `exp` turns the
    /// answer back into a rotation. Slerp is exactly
    /// `$q_0 \exp(t \log(q_1 q_0^{-1}))$`, which is why
    /// [`slerp`](Self::slerp) and this pair are two views of one operation.
    #[inline(always)]
    pub fn log(self) -> Vector<V, 3> {
        let (axis, angle) = self.to_axis_angle();

        axis * angle
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
    /// orientation obeys
    /// `$\dot{q} = \tfrac{1}{2}\,\omega\,q$` (with `$\omega$` the angular
    /// velocity as a pure quaternion), and integrating that over one step is
    /// exactly what advances a spinning object from one frame to the next.
    ///
    /// Uses the **exact** exponential update `$q' = \exp(\omega\,dt/2)\,q$`
    /// rather than the usual first-order `$q + \tfrac{1}{2}\,dt\,\omega q$`.
    /// The cheap form leaves the quaternion off the unit sphere and needs
    /// renormalizing every step, and it systematically under-rotates, so a body
    /// spinning fast enough visibly lags. The exponential form is a true
    /// rotation by construction, so it stays normalized and is accurate at any
    /// spin rate, at the cost of one `sin_cos` that the cheap form skips.
    ///
    /// The result is renormalized anyway, to stop round-off accumulating over
    /// thousands of steps.
    #[inline(always)]
    pub fn integrate(self, angular_velocity: Vector<V, 3>, dt: V) -> Self {
        // The rotation vector covered during this step is omega * dt, and `exp`
        // halves it internally, which is the 1/2 in the kinematic equation.
        let delta = Self::exp(angular_velocity * dt);

        (delta * self).try_normalize()
    }
}
