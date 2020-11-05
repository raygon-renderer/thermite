use crate::*;

#[inline(always)]
pub fn poly_2<S: Simd, V: SimdFloatVector<S>>(x: V, x2: V, c0: V, c1: V, c2: V) -> V {
    x2.mul_add(c2, x.mul_add(c1, c0))
}

#[inline(always)]
pub fn poly_3<S: Simd, V: SimdFloatVector<S>>(x: V, x2: V, c0: V, c1: V, c2: V, c3: V) -> V {
    // x^2 * (x * c3 + c2) + (x*c1 + c0)
    x2.mul_add(x.mul_add(c3, c2), x.mul_add(c1, c0))
}

#[inline(always)]
pub fn poly_4<S: Simd, V: SimdFloatVector<S>>(x: V, x2: V, x4: V, c0: V, c1: V, c2: V, c3: V, c4: V) -> V {
    // x^4 * c4 + (x^2 * (x * c3 + c2) + (x*c1 + c0))
    x4.mul_add(c4, x2.mul_add(x.mul_add(c3, c2), x.mul_add(c1, c0)))
}

#[inline(always)]
pub fn poly_5<S: Simd, V: SimdFloatVector<S>>(x: V, x2: V, x4: V, c0: V, c1: V, c2: V, c3: V, c4: V, c5: V) -> V {
    // x^4 * (x * c5 + c4) + (x^2 * (x * c3 + c2) + (x*c1 + c0))
    x4.mul_add(x.mul_add(c4, c4), x2.mul_add(x.mul_add(c3, c2), x.mul_add(c1, c0)))
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_6<S: Simd, V: SimdFloatVector<S>>(x: V, x2: V, x4: V, c0: V, c1: V, c2: V, c3: V, c4: V, c5: V, c6: V) -> V {
    // x^4 * (x^2 * c6 + (x * c5 + c4)) + (x^2 * (x * c3 + c2) + (x * c1 + c0))
    x4.mul_add(
        x2.mul_add(c6, x.mul_add(c5, c4)),
        x2.mul_add(x.mul_add(c3, c2), x.mul_add(c1, c0)),
    )
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_7<S: Simd, V: SimdFloatVector<S>>(x: V, x2: V, x4: V, c0: V, c1: V, c2: V, c3: V, c4: V, c5: V, c6: V, c7: V) -> V {
    x4.mul_add(
        x2.mul_add(x.mul_add(c7, c6), x.mul_add(c5, c4)),
        x2.mul_add(x.mul_add(c3, c2), x.mul_add(c1, c0)),
    )
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_8<S: Simd, V: SimdFloatVector<S>>(
    x: V, x2: V, x4: V, x8: V,
    c0: V, c1: V, c2: V, c3: V, c4: V, c5: V, c6: V, c7: V, c8: V
) -> V {
    x8.mul_add(c8, x4.mul_add(
        x2.mul_add(x.mul_add(c7, c6), x.mul_add(c5, c4)),
        x2.mul_add(x.mul_add(c3, c2), x.mul_add(c1, c0)),
    ))
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_9<S: Simd, V: SimdFloatVector<S>>(
    x: V, x2: V, x4: V, x8: V,
    c0: V, c1: V, c2: V, c3: V, c4: V, c5: V, c6: V, c7: V, c8: V, c9: V
) -> V {
    x8.mul_add(x.mul_add(c9, c8), x4.mul_add(
        x2.mul_add(x.mul_add(c7, c6), x.mul_add(c5, c4)),
        x2.mul_add(x.mul_add(c3, c2), x.mul_add(c1, c0)),
    ))
}
