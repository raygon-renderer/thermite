//! Polynomial evaluation shared by the f32 and f64 backends.

use super::super::*;

use crate::vector::NumVector;

/// Shared body of the real-vector [`SpecializedCoreMath::poly_n_primal`] override.
///
/// A real float vector is its own primal, so the coefficients arrive already splatted:
/// this is [`poly`](SpecializedCoreMath::poly) minus the per-term `splat`, ILP lowering
/// and all. Identical for f32 and f64, so both backends delegate here.
#[inline(always)]
pub fn poly_n_primal_internal<V, P, N>(x: V, coeffs: &GenericArray<V, N>) -> V
where
    V: FloatVector,
    P: Policy,
    N: ArrayLength,
{
    let n = const { N::USIZE };

    if const {
        !P::POLICY.unroll_loops
            || P::POLICY.precision.ge(PrecisionPolicy::Best)
            || !V::ISA.has_instruction_level_parallelism()
    } {
        let mut res = coeffs[n - 1];
        for &c in coeffs.iter().rev().skip(1) {
            res = res.mul_adde(x, c);
        }
        return res;
    }

    // `poly_f` rather than `poly_f_n`: a typenum length cannot be passed as a const
    // generic argument on stable. `poly_f` pins fast_polynomial's own `LENGTH` to 0, so
    // its internal `assert_unchecked(n == LENGTH)` hint does NOT fire, and the 16-arm
    // length match folds only via caller inlining plus constant propagation of
    // `N::USIZE`. It does: `bin/poly_n_primal_probe` emits byte-identical asm for this and
    // the const-N `poly_f_n` lowering at N=13 on AVX2 (Estrin, no call, no jump table).
    // Re-run that probe if this is ever restructured.
    //
    // NumVector provides the num_traits::MulAdd implementation fast_polynomial needs.
    let res = fast_polynomial::poly_f(NumVector(x), n, |i| unsafe { NumVector(*coeffs.get_unchecked(i)) });

    res.0
}

/// Runtime-length form of [`poly_n_primal_internal`], behind the real-vector
/// `poly_primal` override.
///
/// Same body with `N::USIZE` replaced by `coeffs.len()`, which is what
/// [`SpecializedCoreMath::poly`] already does for element coefficients. The ILP lowering
/// does not need the length to be constant, only the unrolling does.
///
/// `REV` walks the coefficients descending, so the two spellings share one body rather
/// than duplicating the `poly_f` call twice over.
#[inline(always)]
pub fn poly_primal_slice_internal<V, P, const REV: bool>(x: V, coeffs: &[V]) -> V
where
    V: FloatVector,
    P: Policy,
{
    let n = coeffs.len();

    if const {
        !P::POLICY.unroll_loops
            || P::POLICY.precision.ge(PrecisionPolicy::Best)
            || !V::ISA.has_instruction_level_parallelism()
    } {
        if crate::unlikely(n == 0) {
            return V::ZERO;
        }

        if REV {
            let mut res = coeffs[0];
            for &c in coeffs.iter().skip(1) {
                res = res.mul_adde(x, c);
            }
            return res;
        }

        let mut res = coeffs[n - 1];
        for &c in coeffs.iter().rev().skip(1) {
            res = res.mul_adde(x, c);
        }
        return res;
    }

    // See `poly_n_primal_internal` for why this is `poly_f` and not `poly_f_n`.
    let res = fast_polynomial::poly_f(NumVector(x), n, |i| unsafe {
        NumVector(*coeffs.get_unchecked(if REV { n - 1 - i } else { i }))
    });

    res.0
}

/// [`poly_n_primal_internal`] with the coefficients in reverse (descending) order,
/// backing the real-vector [`SpecializedCoreMath::poly_rev_n_primal`] override.
#[inline(always)]
pub fn poly_rev_n_primal_internal<V, P, N>(x: V, coeffs: &GenericArray<V, N>) -> V
where
    V: FloatVector,
    P: Policy,
    N: ArrayLength,
{
    let n = const { N::USIZE };

    if const {
        !P::POLICY.unroll_loops
            || P::POLICY.precision.ge(PrecisionPolicy::Best)
            || !V::ISA.has_instruction_level_parallelism()
    } {
        let mut res = coeffs[0];
        for &c in coeffs.iter().skip(1) {
            res = res.mul_adde(x, c);
        }
        return res;
    }

    // See `poly_n_primal_internal` for why this is `poly_f` and not `poly_f_n`.
    let res = fast_polynomial::poly_f(NumVector(x), n, |i| unsafe {
        NumVector(*coeffs.get_unchecked(n - 1 - i))
    });

    res.0
}

