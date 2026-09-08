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

// --- const-length polynomial evaluation, real vectors only ---------------------------
//
// Everything below is bounded on `FloatVectorWithBits` on purpose. Estrin, the unrolling
// ladder and the compensated arm trade extra operations for ILP or exact error terms,
// which only pays on real lanes. A composite (`Complex`, `Dual`, `Compensated`) already
// saturates the ILP, and an error-free transformation is not well posed on a type whose
// addition is more than one rounding. Those take the plain Horner fallback.

/// Shared body of the real-vector [`SpecializedCoreMath::poly`] and
/// [`SpecializedCoreMath::poly_rev`] overrides.
///
/// The runtime-length twin of [`poly_n_internal`]: same ILP lowering, no unrolling ladder
/// or compensated arm. `REV` walks the coefficients descending.
///
/// An empty slice is `0`; only the Horner arm checks, since `fast_polynomial` handles a
/// zero length itself.
#[inline(always)]
pub fn poly_slice_internal<V, E, P, const REV: bool>(x: V, coeffs: &[E]) -> V
where
    E: FloatElementWithBits,
    V: FloatVectorWithBits<Element = E>,
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
            let mut res = V::splat(coeffs[0]);
            for &c in coeffs.iter().skip(1) {
                res = res.mul_adde(x, V::splat(c));
            }
            return res;
        }

        let mut res = V::splat(coeffs[n - 1]);
        for &c in coeffs.iter().rev().skip(1) {
            res = res.mul_adde(x, V::splat(c));
        }
        return res;
    }

    // NumVector provides the num_traits::MulAdd implementation fast_polynomial needs.
    let res = fast_polynomial::poly_f(NumVector(x), n, |i| unsafe {
        NumVector(V::splat(*coeffs.get_unchecked(if REV { n - 1 - i } else { i })))
    });

    res.0
}

/// Compensated Horner (Graillat-Langlois-Louvet): the value plus a running error term.
///
/// `REV` picks the coefficient order: `false` is constant-term-first (`poly_n`), `true` is
/// leading-term-first (`poly_rev_n`).
///
/// Opt-in only, about 10 operations per term against 1 FMA. A call site asks for it with
/// `UseCompensation<P, true>`, which is how the Bessel rationals reach it.
///
/// With a native FMA this is the full scheme: both error sources captured, condition
/// number entering squared. Without one (including wasm's `Indeterminate`) the product
/// error is dropped and only the sums are compensated, improving the bound from
/// `gamma_{2n}` to `gamma_n`. The emulated correctly-rounded FMA would keep the full
/// scheme but measured 27x on the 1-lane f64 seed and 4x on f64x4 for roughly 2x of
/// accuracy, which is also why this does not call [`FloatVectorWithBits::two_prod`].
#[inline(always)]
pub fn compensated_horner<V, E, const N: usize, const REV: bool>(x: V, coeffs: &[E; N]) -> V
where
    E: FloatElementWithBits,
    V: FloatVectorWithBits<Element = E>,
{
    // Leading coefficient: last slot when constant-first, first slot when leading-first.
    let mut s = V::splat(coeffs[if REV { 0 } else { N - 1 }]);
    let mut e = V::ZERO;

    let mut i = 1usize;
    while i < N {
        let c = V::splat(coeffs[if REV { i } else { N - 1 - i }]);

        let p = s * x;

        // Knuth's 2Sum, since the operands are not magnitude-ordered. Through the register
        // layer so the scalar backend keeps it strict under `algebraic-scalar`.
        let (t, sigma) = p.two_sum(c);

        let inc = if const { matches!(V::HAS_NATIVE_FMA, tribool::True) } {
            s.mul_add(x, -p) + sigma
        } else {
            sigma
        };

        // The error terms ride the same recurrence. `mul_adde`, not `mul_add`: `e` is
        // already ~eps, so its own rounding is second order, and a correctly-rounded FMA
        // here would drag the emulated path back in.
        e = e.mul_adde(x, inc);
        s = t;

        i += 1;
    }

    s + e
}

/// Shared body of the real-vector [`SpecializedCoreMath::poly_n`] override.
///
/// Constant-term-first. Compensated Horner when opted in, plain Horner when unrolling is
/// off or the policy is `Best`+ or the ISA has no ILP, otherwise Estrin.
#[inline(always)]
pub fn poly_n_internal<V, E, P, const N: usize>(x: V, coeffs: &[E; N]) -> V
where
    E: FloatElementWithBits,
    V: FloatVectorWithBits<Element = E>,
    P: Policy,
{
    // Opt-in only: no standard policy sets `use_compensation`.
    if const { P::POLICY.use_compensation } {
        return compensated_horner::<V, E, N, false>(x, coeffs);
    }

    if const {
        !P::POLICY.unroll_loops
            || P::POLICY.precision.ge(PrecisionPolicy::Best)
            || !V::ISA.has_instruction_level_parallelism()
    } {
        // SPIR-V is terrible at unrolling loops like this, so we do it ourselves.
        #[cfg(all(feature = "spirv", target_arch = "spirv"))]
        {
            use crunchy::unroll;

            let mut res = V::splat(coeffs[N - 1]);

            macro_rules! unroll_poly {
                ($($len:tt),*) => {
                    $(if const { N == $len } {
                        unroll! {
                            for i in 1..$len {
                                res = res.mul_adde(x, V::splat(coeffs[
                                    const { if $len > i + 1 { $len - 1 - i } else { 0 } }
                                ]));
                            }
                        }
                    } else )* {
                        let mut i = const { N - 1 };
                        while i > 0 {
                            i -= 1;
                            unsafe { core::hint::assert_unchecked(i < N) };
                            res = res.mul_adde(x, V::splat(coeffs[i]));
                        }
                    }
                };
            }

            unroll_poly!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

            return res;
        }

        // Basic Horner: compact and accurate, even without FMA.
        #[allow(unreachable_code)]
        {
            let mut res = V::splat(coeffs[N - 1]);
            for &c in coeffs.iter().rev().skip(1) {
                res = res.mul_adde(x, V::splat(c));
            }
            return res;
        }
    }

    // NumVector provides the num_traits::MulAdd implementation fast_polynomial needs.
    let res = fast_polynomial::poly_f_n::<_, _, N>(NumVector(x), |i| unsafe {
        NumVector(V::splat(*coeffs.get_unchecked(i)))
    });

    res.0
}

/// The leading-term-first twin of [`poly_n_internal`].
///
/// `poly_rational_n`'s reciprocal branch goes through here, so it needs the compensated
/// arm too.
#[inline(always)]
pub fn poly_rev_n_internal<V, E, P, const N: usize>(x: V, coeffs: &[E; N]) -> V
where
    E: FloatElementWithBits,
    V: FloatVectorWithBits<Element = E>,
    P: Policy,
{
    if const { P::POLICY.use_compensation } {
        return compensated_horner::<V, E, N, true>(x, coeffs);
    }

    if const {
        !P::POLICY.unroll_loops
            || P::POLICY.precision.ge(PrecisionPolicy::Best)
            || !V::ISA.has_instruction_level_parallelism()
    } {
        #[cfg(all(feature = "spirv", target_arch = "spirv"))]
        {
            use crunchy::unroll;

            let mut res = V::splat(coeffs[0]);

            macro_rules! unroll_poly {
                ($($len:tt),*) => {
                    $(if const { N == $len } {
                        unroll! {
                            for i in 1..$len {
                                res = res.mul_adde(x, V::splat(coeffs[i]));
                            }
                        }
                    } else )* {
                        let mut i = 1usize;
                        while i < N {
                            unsafe { core::hint::assert_unchecked(i < N) };
                            res = res.mul_adde(x, V::splat(coeffs[i]));
                            i += 1;
                        }
                    }
                };
            }

            unroll_poly!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

            return res;
        }

        #[allow(unreachable_code)]
        {
            let mut res = V::splat(coeffs[0]);
            for &c in coeffs.iter().skip(1) {
                res = res.mul_adde(x, V::splat(c));
            }
            return res;
        }
    }

    let res = fast_polynomial::poly_f_n::<_, _, N>(NumVector(x), |i| unsafe {
        NumVector(V::splat(*coeffs.get_unchecked(N - 1 - i)))
    });

    res.0
}