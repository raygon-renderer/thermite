//! Orthonormal shading frame.

use thermite::{prelude::*, simd::Simd3Vectors};

use super::{AosFloat, Matrix4, Transform, V3, vector::Vector3Ext as _};

/// An orthonormal shading frame (tangent, bitangent, normal).
///
/// Shading space has `+Z` along the normal, so [`to_local`](Self::to_local)
/// makes `$\cos\theta$` just the `z` component, the reason every BSDF wants to
/// work in this frame.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct TangentFrame<S: Simd3Vectors, E: AosFloat<S>> {
    pub normal: V3<S, E>,
    pub tangent: V3<S, E>,
    pub bitangent: V3<S, E>,
}

impl<S: Simd3Vectors, E: AosFloat<S>> TangentFrame<S, E> {
    /// Builds an orthonormal frame around a **unit** normal, picking an
    /// arbitrary tangent.
    ///
    /// The Frisvad/Duff et al. construction (JCGT 2017,
    /// <https://jcgt.org/published/0006/01/01/paper.pdf>): branchless and stable
    /// for every normal, including `$n_z$` near `$-1$` where the naive Frisvad
    /// formula loses all precision. The sign trick buys that: reflecting the
    /// frame with `$s = \mathrm{copysign}(1, n_z)$` folds the southern
    /// hemisphere onto the northern one, so the `$1/(s + n_z)$` denominator is
    /// never near zero.
    #[inline(always)]
    pub fn new(normal: V3<S, E>) -> Self {
        let nz = normal.extract::<2>();

        let sign = if nz < E::ZERO { -E::ONE } else { E::ONE };

        let nx = normal.extract::<0>();
        let ny = normal.extract::<1>();

        let a = -E::ONE / (sign + nz);
        let b = nx * ny * a;

        Self {
            normal,
            tangent: V3::<S, E>::from_slice(&[E::ONE + sign * nx * nx * a, sign * b, -(sign * nx)]),
            bitangent: V3::<S, E>::from_slice(&[b, sign + ny * ny * a, -ny]),
        }
    }

    /// Builds a frame around a normal using a supplied (mesh) tangent, falling
    /// back to [`new`](Self::new) wherever that tangent is unusable.
    ///
    /// A tangent is unusable when it is zero, non-finite, or parallel to the
    /// normal. In each case the cross product defining the bitangent has no
    /// direction, which the length test catches in one shot.
    ///
    /// The supplied tangent is used as-is (only normalized), not
    /// re-orthogonalized against the normal.
    #[inline(always)]
    pub fn partial(normal: V3<S, E>, tangent: V3<S, E>) -> Self {
        let (bitangent, len) = normal.cross3::<false>(tangent).normalize_norm();

        if len > E::ZERO && bitangent.is_all_finite() {
            Self {
                normal,
                tangent: tangent.normalize(),
                bitangent,
            }
        } else {
            Self::new(normal)
        }
    }

    /// Rotates a shading-space vector into world space:
    /// `$\mathbf{v}_w = x\hat{t} + y\hat{b} + z\hat{n}$`.
    #[inline(always)]
    pub fn to_world(&self, local: V3<S, E>) -> V3<S, E> {
        self.tangent.mul_adde(
            local.broadcast::<0>(),
            self.bitangent
                .mul_adde(local.broadcast::<1>(), self.normal * local.broadcast::<2>()),
        )
    }

    /// Projects a world-space vector into shading space, where `+Z` is the
    /// normal:
    /// `$\mathbf{v}_l = [\mathbf{v}\cdot\hat{t},\, \mathbf{v}\cdot\hat{b},\, \mathbf{v}\cdot\hat{n}]$`.
    #[inline(always)]
    pub fn to_local(&self, world: V3<S, E>) -> V3<S, E> {
        V3::<S, E>::from_slice(&[
            world.dot3(self.tangent),
            world.dot3(self.bitangent),
            world.dot3(self.normal),
        ])
    }

    /// The frame as a [`Transform`]: `forward` maps tangent space to world
    /// space.
    ///
    /// The basis is orthonormal and unshifted, so the inverse is exactly the
    /// transpose and nothing is inverted.
    #[inline(always)]
    pub fn to_transform(&self) -> Transform<S, E> {
        let forward = Matrix4::from_basis(
            self.tangent,
            self.bitangent,
            self.normal,
            <V3<S, E> as NumericVector>::ZERO,
        );

        Transform::from_raw(forward, forward.transpose())
    }
}
