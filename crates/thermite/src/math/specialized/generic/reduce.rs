//! Reciprocal-sum reductions, at a const count and at a runtime one.

use super::super::*;

/// `numer / sum(1/x_i)` with every reciprocal scaled by the smallest input, the real-vector
/// override behind [`SpecializedCoreMath::harmonic_mean`] and
/// [`SpecializedCoreMath::inv_sum_inv`].
///
/// Written directly, `sum(1/x_i)` overflows the moment any input is denormal: the reciprocal
/// saturates to infinity, the sum with it, and the answer collapses to zero when the true
/// value is merely small. Scaling by `m = min(x_i)` makes every term `m/x_i <= 1` by
/// construction, so the sum lands in `[1, N]` and cannot overflow whatever the spread of the
/// inputs. Recovering the answer is exact, since `sum(1/x_i) = s/m`.
///
/// It has to be the *smallest* element. The largest reciprocal is the one that overflows, so
/// it is the one that must normalize to 1; scaling by the largest input, which is what
/// `hypot_n` does for the opposite reason, would leave the failure exactly where it was.
/// Measured against a 60-digit oracle, this is exact across the full representable spread
/// (`5e-324` against `1e300`) where the direct form returns zero.
///
/// The scaling is what costs the two guards the direct form does not need: at `m = 0` every
/// term is `0/x_i` and the sum is `0`, giving `0/0` where the limit is `0`, and at an
/// all-infinite input every term is `inf/inf = NaN` where the limit is infinite.
///
/// This is confined to real vectors on purpose. `Complex::min` is lexicographic by
/// `(re, im)`, so it can return a large-magnitude element and the scaling would protect
/// nothing, so composites take the direct form instead.
#[inline(always)]
pub fn inv_sum_inv_internal<V, E: FloatElement, P, const N: usize>(mut values: [V; N], numer: V) -> V
where
    V: FloatVectorWithBits<Element = E> + SpecializedCoreMath<E>,
    P: Policy,
{
    if const { P::POLICY.precision.lt(PrecisionPolicy::Average) } {
        return V::inv_sum_inv_direct::<P, N>(values, numer);
    }

    // `reduce_array` copies, so `values` is still intact after this.
    let m = crate::math::algorithms::reduce_array(values, |a, b| a.min(b));

    let mut i = 0;
    while i < N {
        values[i] = m / values[i];
        i += 1;
    }

    crate::math::algorithms::reduce_in_place(&mut values, |a, b| a + b);

    let r = (numer * m) / values[0];

    if const { !P::POLICY.check_overflow } {
        return r;
    }

    m.cmp_eq(V::INFINITY)
        .select(V::INFINITY, m.cmp_eq(V::ZERO).select(V::ZERO, r))
}

/// Runtime-length form of [`inv_sum_inv_internal`], behind the slice-taking
/// `harmonic_mean`/`inv_sum_inv`.
///
/// Same scaling, same guards, same reasoning. See that function. The only difference is
/// mechanical: a shared slice cannot be scaled in place, so the min and the sum are serial
/// folds over `values` instead of a copy plus two log-depth reductions.
#[inline(always)]
pub fn inv_sum_inv_slice_internal<V, E: FloatElement, P>(values: &[V], numer: V) -> V
where
    V: FloatVectorWithBits<Element = E> + SpecializedCoreMath<E>,
    P: Policy,
{
    if const { P::POLICY.precision.lt(PrecisionPolicy::Average) } {
        // The direct form, unscaled: below Average the const kernel skips the scaling
        // too, and with it the denormal protection. Same trade, same tier.
        let mut acc = V::ZERO;
        for &v in values {
            acc += V::approx_reciprocal::<P>(v);
        }

        return V::approx_div::<P>(numer, acc);
    }

    let Some((&first, rest)) = values.split_first() else {
        // The empty product of reciprocals is 0, so `numer / 0` is the answer the const
        // form gives at N = 0 as well.
        return V::approx_div::<P>(numer, V::ZERO);
    };

    let mut m = first;
    for &v in rest {
        m = m.min(v);
    }

    let mut acc = V::ZERO;
    for &v in values {
        acc += m / v;
    }

    let r = (numer * m) / acc;

    if const { !P::POLICY.check_overflow } {
        return r;
    }

    m.cmp_eq(V::INFINITY)
        .select(V::INFINITY, m.cmp_eq(V::ZERO).select(V::ZERO, r))
}
