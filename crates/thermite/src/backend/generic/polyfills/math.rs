use super::*;

#[inline(always)]
pub fn fix_min<R: FloatRegister>(a: Storage<R>, b: Storage<R>, mut min: Storage<R>) -> Storage<R> {
    #[cfg(not(feature = "strict_ieee754"))]
    return min;

    let is_nan = R::is_nan(b);

    // This will copy the negative sign if min(-0.0, +0.0),
    // since if they are non-zero but the equal, the sign is already identical.
    let same = R::eq(a, b);
    min = R::blendv(same, min, R::bitor(a, b));

    R::blendv(is_nan, min, a)
}

#[inline(always)]
pub fn fix_max<R: FloatRegister>(a: Storage<R>, b: Storage<R>, mut max: Storage<R>) -> Storage<R> {
    #[cfg(not(feature = "strict_ieee754"))]
    return max;

    let is_nan = R::is_nan(b);

    // This will remove the negative sign if max(+0.0, -0.0),
    // since if they are non-zero but the equal, the sign is already identical.
    let same = R::eq(a, b);
    max = R::blendv(same, max, R::bitand(a, b));

    R::blendv(is_nan, max, a)
}
