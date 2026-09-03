//! The Bessel family's glue: what sits between the kernels in
//! [`generic::bessel`](super::generic::bessel) and the per-element `SpecialMath` impls.
//!
//! Three things live here. The **reflection helpers** for negative whole orders, which
//! every entry point applies after evaluating at `|N|`. The **entry-point stamping
//! macros** for the spherical and Airy families, whose bodies differ per element only in
//! the tables they name. And [`BesselDetails`], the per-arithmetic decisions of the
//! real-order `I`/`K` kernel, which is what lets `thermite-complex` run that kernel in
//! complex arithmetic (together with the doc-hidden [`kernels`] re-export it reaches the
//! bodies through).

use thermite::{
    math::{PrimalProjection, specialized::SpecializedPrimalMath},
    register::FloatElement,
};

/// Whether a negative Bessel order owes its result a sign flip.
///
/// The four families reflect differently at negative **integer** order:
/// `$J_{-n} = (-1)^n J_n$` and `$Y_{-n} = (-1)^n Y_n$`, while `$I_{-n} = I_n$` and
/// `$K_{-n} = K_n$` outright. Every kernel evaluates at `|N|` (the absolute value cannot live
/// in a const-generic argument without `generic_const_exprs`), so this is the sign the entry
/// point still owes, and is `true` only for `J`/`Y` at odd negative orders.
///
/// A flip is exact, so it costs no accuracy and the `Reference` tier stays bit-identical.
#[inline(always)]
pub(crate) const fn bessel_reflect_negates(n: i32) -> bool {
    n < 0 && n % 2 != 0
}

/// The per-lane twin of [`bessel_reflect_negates`], for the runtime-order `J`/`Y` entries.
///
/// Returns `(|nu|, flip)`: the magnitude to evaluate at, and the lanes owed a negation. The
/// recurrences only walk upward from order 0, so a negative order has to arrive as its
/// magnitude and be reflected afterwards.
///
/// The parity test stays in the float domain deliberately. `nu` is an exact integer here, so
/// `nu/2` is either integral or exactly half-integral, and comparing against its own floor
/// answers "odd?" without casting an integer mask across to the float mask type.
#[inline(always)]
pub(crate) fn bessel_reflect_v<V: thermite::vector::FloatVector>(nu: V) -> (V, V::Mask) {
    let na = nu.abs();
    let half = na * V::HALF;

    (na, nu.cmp_lt(V::ZERO) & half.cmp_ne(half.floor()))
}

/// Stamps the spherical Bessel entry points for one element type.
///
/// The four families share two kernels and one derivative formula. What differs per entry is
/// which slot of the returned pair is the value, whether the neighbour enters negated, and
/// whether the exponential is factored out.
macro_rules! impl_sph_bessel_entries {
    ($e:ty, $bi0:expr) => {
        #[inline(always)]
        fn sph_bessel_j_n<P: Policy, const N: usize>(self) -> Self {
            generic::bessel::spherical::sph_jy_impl_n::<P, $e, _, N>(self).1
        }

        #[inline(always)]
        fn sph_bessel_y_n<P: Policy, const N: usize>(self) -> Self {
            generic::bessel::spherical::sph_jy_impl_n::<P, $e, _, N>(self).3
        }

        #[inline(always)]
        fn sph_bessel_i_n<P: Policy, const N: usize>(self) -> Self {
            generic::bessel::spherical::sph_ik_impl_n::<P, $e, _, N,false>(self, $bi0.far_threshold).1
        }

        #[inline(always)]
        fn sph_bessel_i_scaled_n<P: Policy, const N: usize>(self) -> Self {
            generic::bessel::spherical::sph_ik_impl_n::<P, $e, _, N,true>(self, $bi0.far_threshold).1
        }

        #[inline(always)]
        fn sph_bessel_k_n<P: Policy, const N: usize>(self) -> Self {
            generic::bessel::spherical::sph_ik_impl_n::<P, $e, _, N,false>(self, $bi0.far_threshold).3
        }

        #[inline(always)]
        fn sph_bessel_k_scaled_n<P: Policy, const N: usize>(self) -> Self {
            generic::bessel::spherical::sph_ik_impl_n::<P, $e, _, N,true>(self, $bi0.far_threshold).3
        }

        #[inline(always)]
        fn sph_bessel_j_with_deriv_n<P: Policy, const N: usize>(self) -> (Self, Self) {
            let (prev, v, _, _) = generic::bessel::spherical::sph_jy_impl_n::<P, $e, _, N>(self);
            (v, generic::bessel::spherical::sph_deriv_n::<$e, _, N,false>(self, prev, v))
        }

        #[inline(always)]
        fn sph_bessel_y_with_deriv_n<P: Policy, const N: usize>(self) -> (Self, Self) {
            let (_, _, prev, v) = generic::bessel::spherical::sph_jy_impl_n::<P, $e, _, N>(self);
            (v, generic::bessel::spherical::sph_deriv_n::<$e, _, N,false>(self, prev, v))
        }

        #[inline(always)]
        fn sph_bessel_i_with_deriv_n<P: Policy, const N: usize, const SCALED: bool>(self) -> (Self, Self) {
            let (prev, v, _, _) = generic::bessel::spherical::sph_ik_impl_n::<P, $e, _, N,SCALED>(self, $bi0.far_threshold);
            let d = generic::bessel::spherical::sph_deriv_n::<$e, _, N,false>(self, prev, v);
            // `d/dx (e^{-x} i_n) = e^{-x}(i_n' - i_n)`: the scaling's own derivative, which
            // does not cancel.
            (v, if const { SCALED } { d - v } else { d })
        }

        #[inline(always)]
        fn sph_bessel_k_with_deriv_n<P: Policy, const N: usize, const SCALED: bool>(self) -> (Self, Self) {
            let (_, _, prev, v) = generic::bessel::spherical::sph_ik_impl_n::<P, $e, _, N,SCALED>(self, $bi0.far_threshold);
            // `k`'s neighbour enters negated: it is the decaying solution.
            let d = generic::bessel::spherical::sph_deriv_n::<$e, _, N,true>(self, prev, v);
            // At the origin `d + v` is `-inf + inf`, but the derivative of `e^{x} k_n` is still
            // `-inf` there, which `d` already holds.
            (v, if const { SCALED } { self.is_zero().select(d, d + v) } else { d })
        }

        // ---- runtime-order twins, one line each -------------------------------------------

        #[inline(always)]
        fn sph_bessel_j<P: Policy>(self, n: u32) -> Self {
            generic::bessel::spherical::sph_jy_impl::<P, $e, _>(self, n).1
        }

        #[inline(always)]
        fn sph_bessel_y<P: Policy>(self, n: u32) -> Self {
            generic::bessel::spherical::sph_jy_impl::<P, $e, _>(self, n).3
        }

        #[inline(always)]
        fn sph_bessel_i<P: Policy>(self, n: u32) -> Self {
            generic::bessel::spherical::sph_ik_impl::<P, $e, _, false>(self, n, $bi0.far_threshold).1
        }

        #[inline(always)]
        fn sph_bessel_i_scaled<P: Policy>(self, n: u32) -> Self {
            generic::bessel::spherical::sph_ik_impl::<P, $e, _, true>(self, n, $bi0.far_threshold).1
        }

        #[inline(always)]
        fn sph_bessel_k<P: Policy>(self, n: u32) -> Self {
            generic::bessel::spherical::sph_ik_impl::<P, $e, _, false>(self, n, $bi0.far_threshold).3
        }

        #[inline(always)]
        fn sph_bessel_k_scaled<P: Policy>(self, n: u32) -> Self {
            generic::bessel::spherical::sph_ik_impl::<P, $e, _, true>(self, n, $bi0.far_threshold).3
        }

        #[inline(always)]
        fn sph_bessel_j_with_deriv<P: Policy>(self, n: u32) -> (Self, Self) {
            let (prev, v, _, _) = generic::bessel::spherical::sph_jy_impl::<P, $e, _>(self, n);
            (v, generic::bessel::spherical::sph_deriv::<$e, _, false>(self, n, prev, v))
        }

        #[inline(always)]
        fn sph_bessel_y_with_deriv<P: Policy>(self, n: u32) -> (Self, Self) {
            let (_, _, prev, v) = generic::bessel::spherical::sph_jy_impl::<P, $e, _>(self, n);
            (v, generic::bessel::spherical::sph_deriv::<$e, _, false>(self, n, prev, v))
        }

        #[inline(always)]
        fn sph_bessel_i_with_deriv<P: Policy, const SCALED: bool>(self, n: u32) -> (Self, Self) {
            let (prev, v, _, _) = generic::bessel::spherical::sph_ik_impl::<P, $e, _, SCALED>(self, n, $bi0.far_threshold);
            let d = generic::bessel::spherical::sph_deriv::<$e, _, false>(self, n, prev, v);
            (v, if const { SCALED } { d - v } else { d })
        }

        #[inline(always)]
        fn sph_bessel_k_with_deriv<P: Policy, const SCALED: bool>(self, n: u32) -> (Self, Self) {
            let (_, _, prev, v) = generic::bessel::spherical::sph_ik_impl::<P, $e, _, SCALED>(self, n, $bi0.far_threshold);
            let d = generic::bessel::spherical::sph_deriv::<$e, _, true>(self, n, prev, v);
            (v, if const { SCALED } { self.is_zero().select(d, d + v) } else { d })
        }
    };
}

/// Stamps the ten Airy entry points for one element type.
///
/// They differ only in which of the four outputs each asks the kernel for, and whether the
/// exponential is factored out: four `const bool`s and one more. Writing them out twice, once
/// per element, would be eighty lines of near-identical text in which a single transposed flag
/// would be invisible. Here the flags line up in a column.
///
/// The flag order is `(Ai, Ai', Bi, Bi')`, matching the return tuple and SciPy's `airy`.
macro_rules! impl_airy_entries {
    ($e:ty, $nh:literal, $ne:literal, $no:literal, $fnum:literal, $fden:literal, $lgamma:expr, $zero:expr, $bi0:expr) => {
        /// `(Ai, Ai', Bi, Bi')`, all four.
        #[inline(always)]
        fn airy_tuple<P: Policy>(self) -> (Self, Self, Self, Self) {
            generic::bessel::airy::airy_impl::<P, $e, _, $nh, $ne, $no, $fnum, $fden, false, true, true, true, true>(
                self, $lgamma, $zero, $bi0.far_threshold,
            )
        }

        /// `(Ai, Ai', Bi, Bi')`, all four, with the exponential factored out on the positive axis.
        #[inline(always)]
        fn airy_tuple_scaled<P: Policy>(self) -> (Self, Self, Self, Self) {
            generic::bessel::airy::airy_impl::<P, $e, _, $nh, $ne, $no, $fnum, $fden, true, true, true, true, true>(
                self, $lgamma, $zero, $bi0.far_threshold,
            )
        }

        /// `Ai` alone: one Bessel pass, and `K` only within it.
        #[inline(always)]
        fn airy_ai<P: Policy>(self) -> Self {
            generic::bessel::airy::airy_impl::<P, $e, _, $nh, $ne, $no, $fnum, $fden, false, true, false, false, false>(
                self, $lgamma, $zero, $bi0.far_threshold,
            )
            .0
        }

        /// `e^zeta Ai` on the positive axis.
        #[inline(always)]
        fn airy_ai_scaled<P: Policy>(self) -> Self {
            generic::bessel::airy::airy_impl::<P, $e, _, $nh, $ne, $no, $fnum, $fden, true, true, false, false, false>(
                self, $lgamma, $zero, $bi0.far_threshold,
            )
            .0
        }

        /// `Bi` alone: one Bessel pass, `I` and `K` within it.
        #[inline(always)]
        fn airy_bi<P: Policy>(self) -> Self {
            generic::bessel::airy::airy_impl::<P, $e, _, $nh, $ne, $no, $fnum, $fden, false, false, false, true, false>(
                self, $lgamma, $zero, $bi0.far_threshold,
            )
            .2
        }

        /// `e^-zeta Bi` on the positive axis.
        #[inline(always)]
        fn airy_bi_scaled<P: Policy>(self) -> Self {
            generic::bessel::airy::airy_impl::<P, $e, _, $nh, $ne, $no, $fnum, $fden, true, false, false, true, false>(
                self, $lgamma, $zero, $bi0.far_threshold,
            )
            .2
        }

        /// `Ai'` alone: the order-2/3 pass, `K` only.
        #[inline(always)]
        fn airy_ai_prime<P: Policy>(self) -> Self {
            generic::bessel::airy::airy_impl::<P, $e, _, $nh, $ne, $no, $fnum, $fden, false, false, true, false, false>(
                self, $lgamma, $zero, $bi0.far_threshold,
            )
            .1
        }

        /// `e^zeta Ai'` on the positive axis.
        #[inline(always)]
        fn airy_ai_prime_scaled<P: Policy>(self) -> Self {
            generic::bessel::airy::airy_impl::<P, $e, _, $nh, $ne, $no, $fnum, $fden, true, false, true, false, false>(
                self, $lgamma, $zero, $bi0.far_threshold,
            )
            .1
        }

        /// `Bi'` alone: the order-2/3 pass, `I` and `K`.
        #[inline(always)]
        fn airy_bi_prime<P: Policy>(self) -> Self {
            generic::bessel::airy::airy_impl::<P, $e, _, $nh, $ne, $no, $fnum, $fden, false, false, false, false, true>(
                self, $lgamma, $zero, $bi0.far_threshold,
            )
            .3
        }

        /// `e^-zeta Bi'` on the positive axis.
        #[inline(always)]
        fn airy_bi_prime_scaled<P: Policy>(self) -> Self {
            generic::bessel::airy::airy_impl::<P, $e, _, $nh, $ne, $no, $fnum, $fden, true, false, false, false, true>(
                self, $lgamma, $zero, $bi0.far_threshold,
            )
            .3
        }
    };
}

/// Per-arithmetic decisions of the real-order modified Bessel kernel,
/// [`bessel_ik_real`](kernels::bessel_ik_real).
///
/// The kernel's arithmetic (Temme's series, two continued fractions, the Wronskian, the
/// asymptotic series) is the same over R and over C. What changes is every place it
/// _compares_ the argument: region selects by magnitude, the domain test, the overflow
/// corner of the exponential. On a real vector those are plain comparisons. On `Complex`
/// `cmp_lt` is a lexicographic sort order, not a modulus, and would route far-off-axis
/// points into the wrong arm. Same split as [`ExpIntDetails`](super::ExpIntDetails).
///
/// The **order** is always real (`V::Primal`), which is why every threshold here is a
/// primal and why the kernel takes the order as a separate real vector.
///
/// The blanket impl below covers every primal type (real `f32`/`f64` vectors,
/// `Compensated`) with the real-line defaults. `Complex` overrides all of it.
pub trait BesselDetails<V: thermite::vector::FloatVector + PrimalProjection> {
    /// Lanes in the small-argument region, `|z| <= 2`, where `K` comes from Temme's series.
    #[inline(always)]
    fn near(z: V) -> V::Mask {
        z.cmp_le(V::TWO)
    }

    /// Lanes with `|z| >= threshold`, per lane: where `I` takes the asymptotic series.
    #[inline(always)]
    fn beyond(z: V, threshold: V::Primal) -> V::Mask {
        z.cmp_ge(V::from_primal(threshold))
    }

    /// Lanes inside the kernel's domain: the open positive axis, or the closed right
    /// half-plane less the origin. The origin itself is selected to its limits by the
    /// kernel. Everything else outside this mask is NaN.
    #[inline(always)]
    fn valid(z: V) -> V::Mask {
        z.cmp_gt(V::ZERO)
    }

    /// Lanes where a single `e^z` overflows before `e^z * a` does, so the exponential is
    /// halved and applied twice: `Re z >= threshold`.
    #[inline(always)]
    fn exp_far(z: V, threshold: V::Primal) -> V::Mask {
        z.cmp_ge(V::from_primal(threshold))
    }

    /// Whether the large-argument expansion of `I` carries its second exponential.
    ///
    /// `$I_\nu(z) \sim \frac{e^{z}}{\sqrt{2\pi z}}\sum(-1)^k a_k z^{-k} + \frac{e^{-z \pm
    /// (\nu+1/2)\pi i}}{\sqrt{2\pi z}}\sum a_k z^{-k}$` (DLMF 10.40.5). On the real line the
    /// second term is `$e^{-2x}$` relative and below epsilon wherever the arm runs, so the
    /// default drops it. Off the axis its modulus is `$e^{-2\,\mathrm{Re}\,z}$`, which on
    /// the imaginary axis is **one**, so `Complex` keeps it.
    const ASYM_TWO_TERMS: bool = false;

    /// The exponent of that second term in the _scaled_ domain, `$-2z \pm (\nu+1/2)\pi i$`
    /// with the sign of `Im z`. Only read when [`ASYM_TWO_TERMS`](Self::ASYM_TWO_TERMS).
    #[inline(always)]
    fn asym_second_exponent(z: V, _nu: V::Primal) -> V {
        z
    }
}

impl<E: FloatElement, V: thermite::vector::FloatVector<Element = E> + SpecializedPrimalMath<E>> BesselDetails<V> for V {}

/// The shared Bessel kernels a downstream arithmetic instantiates for itself.
///
/// `thermite-complex` runs `bessel_ik_real` in complex arithmetic with its own
/// [`BesselDetails`], and nothing here is API for anyone else.
#[doc(hidden)]
pub mod kernels {
    pub use crate::specialized::generic::bessel::ik::{asymptotic_series_g, unscale_i_pair_masked};
    pub use crate::specialized::generic::bessel::ik_real::bessel_ik_real;
}
