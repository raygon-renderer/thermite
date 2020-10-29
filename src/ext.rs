use crate::*;

pub trait SimdFloatVectorExt<S: Simd>: SimdFloatVector<S> {
    #[inline]
    fn approx_eq(self, other: Self, tolerance: Self) -> Mask<S, Self> {
        (self - other).abs().lt(tolerance)
    }

    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        self.min(max).max(min)
    }

    /// Clamps self to between 0 and 1
    #[inline]
    fn saturate(self) -> Self {
        self.clamp(Self::zero(), Self::one())
    }

    /// Scales values between `in_min` and `in_max`, to between `out_min` and `out_max`
    #[inline]
    fn scale(self, in_min: Self, in_max: Self, out_min: Self, out_max: Self) -> Self {
        ((self - in_min) / (in_max - in_min)).mul_add(out_max - out_min, out_min)
    }

    /// Linearly interpolates between `a` and `b` using `self`
    ///
    /// Equivalent to `(1 - t) * a + t * b`, but uses fused multiply-add operations
    /// to improve performance while maintaining precision
    #[inline]
    fn lerp(self, a: Self, b: Self) -> Self {
        self.mul_add(b - a, a)
    }

    /// Clamps input to positive numbers before calling `sqrt`
    #[inline]
    fn safe_sqrt(self) -> Self {
        self.max(Self::zero()).sqrt()
    }
}

impl<S: Simd, T> SimdFloatVectorExt<S> for T where T: SimdFloatVector<S> {}
