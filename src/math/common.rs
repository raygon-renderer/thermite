use crate::*;

// All of these polynomials use Estrin's scheme to reduce the
// dependency chain length and encourage instruction-level parallelism, which has
// the potential to improve performance despite the powers of X being required upfront

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
    x4.mul_add(x.mul_add(c5, c4), x2.mul_add(x.mul_add(c3, c2), x.mul_add(c1, c0)))
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

#[rustfmt::skip]
#[inline(always)]
pub fn poly_12<S: Simd, V: SimdFloatVector<S>>(
    x: V, x2: V, x4: V, x8: V,
    c0: V, c1: V, c2: V, c3: V, c4: V, c5: V, c6: V, c7: V, c8: V, c9: V, c10: V, c11: V, c12: V,
) -> V {
    x8.mul_add(
        x4.mul_add(
            c12,
            x2.mul_add(x.mul_add(c11, c10), x.mul_add(c9, c8)),
        ),
        x4.mul_add(
            x2.mul_add(x.mul_add(c7, c6), x.mul_add(c5, c4)),
            x2.mul_add(x.mul_add(c3, c2), x.mul_add(c1, c0)),
        ),
    )
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_13<S: Simd, V: SimdFloatVector<S>>(
    x: V, x2: V, x4: V, x8: V,
    c0: V, c1: V, c2: V, c3: V, c4: V, c5: V, c6: V, c7: V, c8: V, c9: V, c10: V, c11: V, c12: V, c13: V,
) -> V {
    x8.mul_add(
        x4.mul_add(
            x.mul_add(c13, c12),
            x2.mul_add(x.mul_add(c11, c10), x.mul_add(c9, c8)),
        ),
        x4.mul_add(
            x2.mul_add(x.mul_add(c7, c6), x.mul_add(c5, c4)),
            x2.mul_add(x.mul_add(c3, c2), x.mul_add(c1, c0)),
        ),
    )
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_15<S: Simd, V: SimdFloatVector<S>>(
    x: V, x2: V, x4: V, x8: V,
    c0: V, c1: V, c2: V, c3: V, c4: V, c5: V, c6: V, c7: V, c8: V, c9: V, c10: V, c11: V, c12: V, c13: V, c14: V, c15: V
) -> V {
    // (((C0+C1x) + (C2+C3x)x2) + ((C4+C5x) + (C6+C7x)x2)x4) + (((C8+C9x) + (C10+C11x)x2) + ((C12+C13x) + (C14+C15x)x2)x4)x8
    x8.mul_add(
        x4.mul_add(
            x2.mul_add(x.mul_add(c15, c14), x.mul_add(c13, c12)),
            x2.mul_add(x.mul_add(c11, c10), x.mul_add(c9, c8)),
        ),
        x4.mul_add(
            x2.mul_add(x.mul_add(c7, c6), x.mul_add(c5, c4)),
            x2.mul_add(x.mul_add(c3, c2), x.mul_add(c1, c0)),
        ),
    )
}

#[rustfmt::skip]
#[inline(always)]
pub fn poly_30<S: Simd, V: SimdFloatVector<S>>(
    x: V, x2: V, x4: V, x8: V, x16: V,
    c00: V, c01: V, c02: V, c03: V, c04: V, c05: V, c06: V, c07: V, c08: V, c09: V, c10: V, c11: V, c12: V, c13: V, c14: V, c15: V,
    c16: V, c17: V, c18: V, c19: V, c20: V, c21: V, c22: V, c23: V, c24: V, c25: V, c26: V, c27: V, c28: V, c29: V, c30: V, c31: V
) -> V {
    x16.mul_add(
        x8.mul_add(
            x4.mul_add(
                x2.mul_add(x.mul_add(c31, c30), x.mul_add(c29, c28)),
                x2.mul_add(x.mul_add(c27, c26), x.mul_add(c25, c24)),
            ),
            x4.mul_add(
                x2.mul_add(x.mul_add(c23, c22), x.mul_add(c21, c20)),
                x2.mul_add(x.mul_add(c19, c18), x.mul_add(c17, c16)),
            ),
        ),
        x8.mul_add(
            x4.mul_add(
                x2.mul_add(x.mul_add(c15, c14), x.mul_add(c13, c12)),
                x2.mul_add(x.mul_add(c11, c10), x.mul_add(c09, c08)),
            ),
            x4.mul_add(
                x2.mul_add(x.mul_add(c07, c06), x.mul_add(c05, c04)),
                x2.mul_add(x.mul_add(c03, c02), x.mul_add(c01, c00)),
            ),
        )
    )
}
