use thermite::{mask::GenericMask, math::SpatialMath, vector::FloatVector};

use super::{Matrix, Transform, Vector, vector::VectorOps as _};

/// Orthonormal shading frame (tangent, bitangent, normal).
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct TangentFrame<V: FloatVector> {
    pub normal: Vector<V, 3>,
    pub tangent: Vector<V, 3>,
    pub bitangent: Vector<V, 3>,
}

impl<V: FloatVector> TangentFrame<V> {
    /// An all-zero frame. Degenerate, and only useful as a placeholder.
    pub const ZERO: Self = Self {
        normal: Vector::ZERO,
        tangent: Vector::ZERO,
        bitangent: Vector::ZERO,
    };

    /// Build an orthonormal frame around a **unit** normal, picking an arbitrary
    /// tangent.
    ///
    /// The Frisvad/Duff et al. construction (JCGT 2017,
    /// <https://jcgt.org/published/0006/01/01/paper.pdf>): branchless and stable
    /// for every normal, including `$n_z$` near `$-1$`, where the naive Frisvad
    /// formula loses all precision. The sign trick is what buys that: reflecting
    /// the frame with `$s = \mathrm{copysign}(1, n_z)$` folds the southern
    /// hemisphere onto the northern one, so the `$1/(s + n_z)$` denominator is
    /// never near zero.
    #[inline(always)]
    pub fn new(normal: Vector<V, 3>) -> Self {
        let (nx, ny, nz) = (normal[0], normal[1], normal[2]);

        let sign = V::ONE.copysign(nz);

        let a = V::NEG_ONE / (sign + nz);
        let b = nx * ny * a;

        let tangent = Vector([a.mul_adde(sign * (nx * nx), V::ONE), sign * b, -(sign * nx)]);
        let bitangent = Vector([b, a.mul_adde(ny * ny, sign), -ny]);

        Self {
            normal,
            tangent,
            bitangent,
        }
    }

    /// Rotate a shading-space vector into world space:
    /// `$\mathbf{v}_w = x\,\hat{\mathbf{t}} + y\,\hat{\mathbf{b}} + z\,\hat{\mathbf{n}}$`.
    #[inline(always)]
    pub fn to_world(&self, local: Vector<V, 3>) -> Vector<V, 3> {
        self.tangent
            .mul_adde(local[0], self.bitangent.mul_adde(local[1], self.normal * local[2]))
    }

    /// The frame as a [`Transform`]: `forward` maps tangent space to world space.
    ///
    /// The basis is orthonormal and unshifted, so the inverse is exactly the
    /// transpose and no inversion is performed.
    #[inline(always)]
    pub fn to_transform(&self) -> Transform<V> {
        let forward = Matrix::from_basis(self.tangent, self.bitangent, self.normal, Vector::ZERO);

        Transform::from_raw(forward, forward.transpose())
    }
}

impl<V: SpatialMath> TangentFrame<V> {
    /// Build a frame around a normal using a supplied (mesh) tangent, falling
    /// back to [`new`](Self::new) wherever that tangent is unusable.
    ///
    /// A tangent is unusable when it is zero, non-finite, or parallel to the
    /// normal. In each case the cross product that defines the bitangent has no
    /// direction, which the length test below catches in one shot. Both frames
    /// are computed and blended, because with one frame per lane a branch could
    /// not serve a batch where only some tangents are degenerate.
    ///
    /// The supplied tangent is used as-is (only normalized), not re-orthogonalized
    /// against the normal.
    #[inline(always)]
    pub fn partial(normal: Vector<V, 3>, tangent: Vector<V, 3>) -> Self {
        let (bitangent, bitangent_norm) = normal.cross(tangent).normalize_norm();

        let ok = bitangent_norm.cmp_gt(V::ZERO) & bitangent_norm.is_finite();

        let fallback = Self::new(normal);

        Self {
            normal,
            tangent: ok.select(tangent.normalize(), fallback.tangent),
            bitangent: ok.select(bitangent, fallback.bitangent),
        }
    }

    /// Project a world-space vector into shading space, where `$+Z$` is the normal:
    /// `$\mathbf{v}_l = [\mathbf{v}\cdot\hat{\mathbf{t}},\; \mathbf{v}\cdot\hat{\mathbf{b}},\; \mathbf{v}\cdot\hat{\mathbf{n}}]$`.
    #[inline(always)]
    pub fn to_local(&self, world: Vector<V, 3>) -> Vector<V, 3> {
        Vector([
            world.dot(&self.tangent),
            world.dot(&self.bitangent),
            world.dot(&self.normal),
        ])
    }
}
