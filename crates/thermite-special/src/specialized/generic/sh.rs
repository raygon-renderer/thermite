//! Real spherical harmonics, evaluated directly from Cartesian components.
//!
//! # Conventions
//!
//! Orthonormal real spherical harmonics. The `CS` const parameter selects the phase
//! convention, either [`NO_PHASE`] (the standard real-SH tables, sphericart, most math
//! references) or [`CONDON_SHORTLEY`]. Everything below describes [`NO_PHASE`]. See
//! [the phase section](#the-condon-shortley-phase) for what the other one changes.
//!
//! ```math
//! \int_{S^2} Y_{\ell m}^2 \, d\Omega = 1,
//! \qquad
//! Y_{\ell m} =
//! \begin{cases}
//! \sqrt{2}\, K_\ell^m P_\ell^m(\cos\theta)\cos(m\varphi) & m > 0 \\
//! K_\ell^0 P_\ell(\cos\theta) & m = 0 \\
//! \sqrt{2}\, K_\ell^{|m|} P_\ell^{|m|}(\cos\theta)\sin(|m|\varphi) & m < 0
//! \end{cases}
//! ```
//!
//! with `$K_\ell^m = \sqrt{\tfrac{2\ell+1}{4\pi}\tfrac{(\ell-m)!}{(\ell+m)!}}$` and
//! `$P_\ell^m$` the associated Legendre functions _without_ `$(-1)^m$`. So
//! `$Y_{00} = \sqrt{1/4\pi}$`, `$Y_{1,-1} = \sqrt{3/4\pi}\,y$`, `$Y_{10} = \sqrt{3/4\pi}\,z$`,
//! `$Y_{11} = \sqrt{3/4\pi}\,x$`.
//!
//! Outputs are written in the flat `l * (l + 1) + m` order (`m` from `-l` to `l`),
//! the layout every SH-lighting pipeline uses.
//!
//! ## The Condon-Shortley phase
//!
//! The two conventions differ by `$(-1)^{|m|}$`: odd `|m|` is negated, even `|m|`
//! agrees exactly. Which one a body of data was projected against is _not_ recoverable
//! from the data (the difference is invisible in any rotationally-averaged or
//! squared quantity), so mixing them is a silent, plausible-looking wrong answer.
//! Hence the explicit parameter rather than a fixed choice.
//!
//! Sloan's widely-copied `SHEval` generated code (_Efficient Spherical Harmonic
//! Evaluation_, JCGT 2(2), 2013) **does** carry the phase. Its diagonal recurrence
//! is `P_m^m = (1 - 2m) P_{m-1}^{m-1}`, negative for every `m >= 1`. Its order-3
//! listing emits `pSH[3] = -0.48860251 * x`, matching [`CONDON_SHORTLEY`] here, while
//! [`NO_PHASE`] gives `+0.48860251 * x`.
//!
//! `CS` is baked into the constant table, so neither choice costs an instruction.
//! Internally it is applied in two places, the second easy to overlook: the diagonal
//! seeds for odd `m` (which propagates to a whole column, and to both the `+m` and
//! `-m` slots that share it), and _every_ `z`-derivative ratio in `f`, because that
//! ratio crosses between adjacent columns whose signs always disagree.
//!
//! # Algorithm
//!
//! No trigonometry and no division anywhere. The evaluation factors each harmonic as
//! `$Y_{\ell,\pm m} = q_\ell^m(z) \cdot \{c_m, s_m\}$` where
//!
//! * `$c_m + i s_m = (x + iy)^m$`, accumulated by the complex-multiplication pair
//!   recurrence. Since `$x + iy = \sin\theta\, e^{i\varphi}$` on the unit sphere, this
//!   _is_ `$\sin^m\theta \{\cos, \sin\}(m\varphi)$`, i.e. the `$\sin^m\theta$` factor of
//!   `$P_\ell^m$` moved into the azimuthal term, which removes the `$1/\sin\theta$`
//!   pole from every recurrence (the factoring used by sphericart, Bigi et al.,
//!   J. Chem. Phys. 159, 064802, 2023).
//! * `$q_\ell^m(z)$` is the fully-normalized sin-factored associated Legendre part,
//!   via the standard normalized three-term recurrences (Holmes & Featherstone 2002,
//!   J. Geodesy 76): a constant diagonal, one `$\sqrt{2m+3}\, z$` step, then
//!   `$q_\ell^m = a_\ell^m z\, q_{\ell-1}^m - b_\ell^m q_{\ell-2}^m$`. All coefficients
//!   are precomputed at compile time ([`ShConsts`]). Intermediate values stay `O(1)`,
//!   so there is no overflow at any order either format can index.
//!
//! Cost is `O(L^2)` FMAs per call (two per harmonic past the seeds) with zero
//! transcendentals.
//!
//! # Domain and gradient semantics
//!
//! `(x, y, z)` is assumed to be a **unit vector**. Nothing renormalizes. Off the unit
//! sphere the recurrences still evaluate a perfectly good polynomial in `(x, y, z)`
//! (the one that agrees with `$Y_{\ell m}$` on the sphere), which is exactly what
//! [`sh_d_impl`]'s derivatives differentiate: the **ambient Cartesian gradient of that
//! polynomial form**, evaluated at the given point. This is the convention machine
//! learning interatomic potentials and finite-difference checks want. A caller who
//! needs the _tangential_ (spherical) gradient projects out the radial component:
//! `g_tan = g - (g . n) n`.
//!
//! The derivative combinations are exact identities on the recurrence outputs:
//! `$\partial_x c_m = m c_{m-1}$`, `$\partial_y c_m = -m s_{m-1}$` (and the mirrored
//! pair for `$s_m$`), and `$\partial_z q_\ell^m = f_\ell^m q_\ell^{m+1}$` where
//! `$f_\ell^m$` is a tabulated norm ratio. So the gradient pass reuses every value
//! the value pass produced and adds no new recurrences.

use core::f64::consts::PI;

use thermite::{
    math::{PrimalProjection, policy::Policy, scalar::Unwrap},
    prelude::*,
    register::FloatElement,
};

/// Triangular index base: `q_l^m` lives at `tri(l) + m`.
///
/// `tri(L) + L = L(L+3)/2 < (L+1)^2` for every `L`, so the triangular tables always
/// fit in the same `N = (L+1)^2` allocation the flat output uses.
#[inline(always)]
pub const fn tri(l: usize) -> usize {
    l * (l + 1) / 2
}

/// `sqrt` for positive finite values in const context.
///
/// Bit-shift seed plus Newton iterations. Converges to within 1 ulp long before the
/// iteration cap for any normal positive input. Not guaranteed correctly rounded
/// (irrelevant at 1 ulp for approximation coefficients), but fully deterministic,
/// which is what matters for reproducible tables.
const fn csqrt(x: f64) -> f64 {
    assert!(x > 0.0 && x < f64::INFINITY);

    // Halving the exponent bits lands within ~2x of sqrt(x), and each Newton step then
    // squares the relative accuracy, so 6 steps are already past f64 precision.
    let mut y = f64::from_bits((x.to_bits() >> 1) + 0x1FF8_0000_0000_0000);

    let mut i = 0;
    while i < 8 {
        y = 0.5 * (y + x / y);
        i += 1;
    }

    y
}

/// Precomputed recurrence coefficients for all `(l, m)` with `l <= L`.
///
/// Every array is sized `N = (L+1)^2` (the flat output size) rather than its exact
/// need, because the exact sizes (`L+1`, triangular) are generic const expressions
/// that stable Rust cannot spell in a type. The waste is compile-time data only.
///
/// Indexing: `qmm` and `em` by `m`, and `a`, `nb`, `f` by `tri(l) + m`.
///
/// Instantiated two ways. `ShTable<E, N>` over a scalar element is the compile-time
/// form behind [`ShConsts`], read by the unrolled kernels. `ShTable<V, N>` over a
/// _vector_ is the runtime form produced by [`sh_table_impl`] and consumed by
/// [`sh_eval_impl`], which is what lifts the degree cap and makes the kernels work on
/// element types that have no const table.
///
/// `#[repr(C)]` so that the two are layout-compatible when the element and vector
/// types are (the scalar-math layer reinterprets `&mut ShTable<f32, N>` as
/// `&mut ShTable<Vector<f32>, N>`). `repr(Rust)` gives no such guarantee across
/// distinct type arguments.
///
/// `Clone` but deliberately not `Copy`: a table is `6 * N` elements (about 4.8 KB at
/// `L = 4` on `f32x8`, 54 KB at `L = 16`), and implicit copies of that are not
/// something to make easy. Pass it by reference. It is read-only after filling.
#[repr(C)]
#[derive(Clone, Debug)]
pub struct ShTable<E, const N: usize> {
    /// Diagonal values `q_m^m`, pure constants, since the `$\sin^m\theta$` that made
    /// the diagonal `z`-dependent lives in the azimuthal recurrence instead.
    pub qmm: [E; N],
    /// First off-diagonal step: `q_{m+1}^m = em[m] * z * q_m^m`, `em[m] = sqrt(2m+3)`.
    pub em: [E; N],
    /// Three-term recurrence: `q_l^m = a * z * q_{l-1}^m + nb * q_{l-2}^m`.
    pub a: [E; N],
    /// The `b` coefficient, stored negated so the recurrence is a single `mul_adde`.
    pub nb: [E; N],
    /// `z`-derivative norm ratio: `d(q_l^m)/dz = f[tri(l)+m] * q_l^{m+1}` (zero at `m = l`).
    pub f: [E; N],
    /// `m` as a float, for the azimuthal derivative factor (`d c_m = m c_{m-1}` etc.).
    pub mf: [E; N],
}

const fn build_f64<const L: usize, const N: usize, const CS: bool>() -> ShTable<f64, N> {
    assert!(N == (L + 1) * (L + 1));

    let mut t = ShTable {
        qmm: [0.0; N],
        em: [0.0; N],
        a: [0.0; N],
        nb: [0.0; N],
        f: [0.0; N],
        mf: [0.0; N],
    };

    let mut m = 0;
    while m <= L {
        t.mf[m] = m as f64;
        m += 1;
    }

    // Diagonal: q_0^0 = Y_00 = sqrt(1/4pi), and each step multiplies by
    // sqrt((2m+1)/(2m)), with one extra sqrt(2) at m = 1. That is the sqrt(2 - delta_{m0})
    // of the real-harmonic normalization entering the recurrence exactly once.
    t.qmm[0] = csqrt(1.0 / (4.0 * PI));

    let mut m = 1;
    while m <= L {
        let mut d = csqrt((2 * m + 1) as f64 / (2 * m) as f64);
        if m == 1 {
            d *= csqrt(2.0);
        }
        t.qmm[m] = t.qmm[m - 1] * d;
        m += 1;
    }

    let mut m = 0;
    while m < L {
        t.em[m] = csqrt((2 * m + 3) as f64);
        m += 1;
    }

    let mut l = 1;
    while l <= L {
        let lf = l as f64;

        let mut mm = 0;
        while mm <= l {
            let k = tri(l) + mm;
            let mf = mm as f64;

            if l >= mm + 2 {
                // Holmes & Featherstone fully-normalized coefficients. The common
                // sqrt(2 - delta) / 4pi prefactors cancel in the ratios, so these are
                // identical for the m = 0 and m > 0 columns.
                t.a[k] = csqrt(((2.0 * lf + 1.0) * (2.0 * lf - 1.0)) / ((lf - mf) * (lf + mf)));
                t.nb[k] = -csqrt(
                    ((2.0 * lf + 1.0) * (lf - mf - 1.0) * (lf + mf - 1.0)) / ((lf - mf) * (lf + mf) * (2.0 * lf - 3.0)),
                );
            }

            // d(q_l^m)/dz = f * q_l^{m+1}: the norm ratio n_{l,m}/n_{l,m+1} applied to
            // the classical dQ_l^m/dz = Q_l^{m+1}. The m = 0 column picks up a 1/sqrt(2)
            // from sqrt(2 - delta_{m0}) changing between the columns.
            t.f[k] = if mm == l {
                0.0 // q_l^{l+1} = 0
            } else if mm == 0 {
                csqrt(lf * (lf + 1.0) / 2.0)
            } else {
                csqrt((lf - mf) * (lf + mf + 1.0))
            };

            mm += 1;
        }

        l += 1;
    }

    // --- Condon-Shortley phase, if requested: scale column m by (-1)^m ---
    //
    // Two touch-ups suffice, and the second is the one that is easy to miss.
    //
    // Every q_l^m in a column is generated from that column's diagonal seed by
    // recurrences that are linear and homogeneous in it, so negating q_m^m negates
    // the whole column. Since the +m and -m outputs share one q, both slots
    // flip together, which is the (-1)^|m| the convention asks for.
    //
    // But `f` relates ADJACENT columns (d(q_l^m)/dz = f * q_l^{m+1}), whose signs
    // now always disagree: the ratio (-1)^m / (-1)^{m+1} is -1 for every m. So the
    // derivative ratios flip globally, independent of m's parity. Miss this and the
    // values are right while every z-gradient carries the wrong sign.
    if CS {
        let mut m = 1;
        while m <= L {
            if m % 2 == 1 {
                t.qmm[m] = -t.qmm[m];
            }
            m += 1;
        }

        let mut l = 0;
        while l <= L {
            let mut mm = 0;
            // Strictly below the diagonal: f is an exact zero at mm == l, and
            // negating that would only manufacture a -0.0.
            while mm < l {
                t.f[tri(l) + mm] = -t.f[tri(l) + mm];
                mm += 1;
            }
            l += 1;
        }
    }

    t
}

const fn arr_to_f32<const N: usize>(a: &[f64; N]) -> [f32; N] {
    let mut o = [0.0f32; N];
    let mut i = 0;
    while i < N {
        o[i] = a[i] as f32;
        i += 1;
    }
    o
}

const fn build_f32<const L: usize, const N: usize, const CS: bool>() -> ShTable<f32, N> {
    let t = build_f64::<L, N, CS>();

    ShTable {
        qmm: arr_to_f32(&t.qmm),
        em: arr_to_f32(&t.em),
        a: arr_to_f32(&t.a),
        nb: arr_to_f32(&t.nb),
        f: arr_to_f32(&t.f),
        mf: arr_to_f32(&t.mf),
    }
}

/// `CS` argument selecting **no** Condon-Shortley phase (the real-SH tables,
/// sphericart, and most math references). This is the conventional default here.
pub const NO_PHASE: bool = false;

/// `CS` argument selecting the Condon-Shortley `$(-1)^{|m|}$` phase (Sloan's
/// `SHEval`, and the physics convention).
pub const CONDON_SHORTLEY: bool = true;

/// Compile-time spherical-harmonic coefficient tables for one element type, at one
/// degree and one phase convention.
///
/// Follows the per-element const-table pattern of `thermite-complex`'s `Weideman`
/// trait: the `f64` table is computed once in const eval and narrowed per element, so
/// the kernels read plain constants and pay no runtime conversion. `CS` is baked into
/// the table rather than applied at runtime, so the phase costs literally nothing.
/// The two conventions differ only in which constants get emitted.
pub trait ShConsts<const L: usize, const N: usize, const CS: bool>: FloatElement {
    const TABLE: ShTable<Self, N>;
}

impl<const L: usize, const N: usize, const CS: bool> ShConsts<L, N, CS> for f64 {
    const TABLE: ShTable<f64, N> = build_f64::<L, N, CS>();
}

impl<const L: usize, const N: usize, const CS: bool> ShConsts<L, N, CS> for f32 {
    const TABLE: ShTable<f32, N> = build_f32::<L, N, CS>();
}

impl<V: FloatVector, const N: usize> ShTable<V, N> {
    /// An all-zero table, to be filled by [`sh_table_impl`].
    #[inline(always)]
    pub fn zeroed() -> Self {
        Self {
            qmm: [V::ZERO; N],
            em: [V::ZERO; N],
            a: [V::ZERO; N],
            nb: [V::ZERO; N],
            f: [V::ZERO; N],
            mf: [V::ZERO; N],
        }
    }

    /// Lifts every entry into a composite `W` whose primal is `V`, via
    /// [`from_primal`](PrimalProjection::from_primal), so constants with zeroed
    /// augmentation. The identity copy when `W` is its own primal.
    ///
    /// Hand-rolled loops rather than `array::map`, which fails to inline in
    /// `target_feature` code.
    #[inline(always)]
    pub fn lift<W>(&self) -> ShTable<W, N>
    where
        W: FloatVector + PrimalProjection<Primal = V>,
    {
        let mut out = ShTable::<W, N>::zeroed();
        let mut i = 0;
        while i < N {
            out.qmm[i] = W::from_primal(self.qmm[i]);
            out.em[i] = W::from_primal(self.em[i]);
            out.a[i] = W::from_primal(self.a[i]);
            out.nb[i] = W::from_primal(self.nb[i]);
            out.f[i] = W::from_primal(self.f[i]);
            out.mf[i] = W::from_primal(self.mf[i]);
            i += 1;
        }
        out
    }
}

/// Scalar-layer bridge for the table arguments, mirroring `thermite`'s `Unwrap for
/// &mut [Vector<R>; N]`.
///
/// The `ScalarSpecialMath` aggregate runs the vector kernels at width 1, wrapping each
/// argument on the way in. A table is a by-reference parameter, so it is reinterpreted
/// in place rather than copied. This is sound because `Vector<R>` is
/// `#[repr(transparent)]` over `Storage<R>`, `R: Register<Storage = R>` pins that to
/// `R`, and [`ShTable`] is `#[repr(C)]` so the two instantiations agree on layout.
impl<'a, R, const N: usize> Unwrap for &'a ShTable<Vector<R>, N>
where
    R: thermite::register::Register<Storage = R>,
{
    type Unwrapped = &'a ShTable<R, N>;

    #[inline(always)]
    fn wrap(value: Self::Unwrapped) -> Self {
        // SAFETY: see the doc comment above (repr(transparent) + repr(C)).
        unsafe { &*(value as *const ShTable<R, N> as *const ShTable<Vector<R>, N>) }
    }

    #[inline(always)]
    fn unwrap(self) -> Self::Unwrapped {
        // SAFETY: as in `wrap`.
        unsafe { &*(self as *const ShTable<Vector<R>, N> as *const ShTable<R, N>) }
    }
}

impl<'a, R, const N: usize> Unwrap for &'a mut ShTable<Vector<R>, N>
where
    R: thermite::register::Register<Storage = R>,
{
    type Unwrapped = &'a mut ShTable<R, N>;

    #[inline(always)]
    fn wrap(value: Self::Unwrapped) -> Self {
        // SAFETY: see the doc comment above (repr(transparent) + repr(C)).
        unsafe { &mut *(value as *mut ShTable<R, N> as *mut ShTable<Vector<R>, N>) }
    }

    #[inline(always)]
    fn unwrap(self) -> Self::Unwrapped {
        // SAFETY: as in `wrap`.
        unsafe { &mut *(self as *mut ShTable<Vector<R>, N> as *mut ShTable<R, N>) }
    }
}

// Index-safety invariant shared by everything below: `N == (L + 1)^2` is
// const-asserted in both kernels, and for `m <= l <= L`
//   flat:       l(l+1) + m  <=  L(L+1) + L  =  N - 1
//   triangular: tri(l) + m  <=  L(L+3)/2    <   N
//   column:     m <= L < N
// so every access is in bounds by construction. The checked-indexing forms are not
// used because their bounds checks defeat LLVM's unroller and scheduler (measured:
// the whole kernel stayed rolled with panic paths at every store).

/// Unchecked fixed-array read under the module's index invariant.
#[inline(always)]
fn at<T: Copy, const N: usize>(a: &[T; N], i: usize) -> T {
    debug_assert!(i < N);
    // SAFETY: see the index invariant above.
    unsafe { *a.get_unchecked(i) }
}

/// Unchecked fixed-array write under the module's index invariant.
#[inline(always)]
fn put<T, const N: usize>(a: &mut [T; N], i: usize, v: T) {
    debug_assert!(i < N);
    // SAFETY: see the index invariant above.
    unsafe {
        *a.get_unchecked_mut(i) = v;
    }
}

/// Writes `q * {c, s}` into the two `(l, +-m)` output slots.
///
/// `m = 0` stores `q` directly: `c_0 = 1` and the `-m` slot is the same slot.
#[inline(always)]
fn emit<V: FloatVector, const N: usize>(out: &mut [V; N], l: usize, m: usize, q: V, c: V, s: V) {
    let base = l * (l + 1);

    if m == 0 {
        put(out, base, q);
    } else {
        put(out, base + m, q * c);
        put(out, base - m, q * s);
    }
}

/// Highest degree the stamped ladders below cover. Beyond it the kernels fall back to
/// the rolled, runtime-coefficient path ([`sh_table_impl`] + [`sh_eval_impl`]), which
/// is correct at any degree but roughly an order of magnitude slower. Extending the
/// ladder is mechanical: append literals to every `0 1 2 ... 16` list.
pub const MAX_DEGREE: usize = 16;

// --- The literal ladders ---
//
// LLVM's unroller declines these triangular nests outright: measured on the loop
// form, L = 4 and L = 8 produced near-identical fully-rolled code with runtime
// l*(l+1) index arithmetic (`imul`/`shl`) and table loads through a register index,
// even with all bounds checks elided. So the unrolling is done in the source (the
// same guard-ladder fix the interleave engine uses): every (m, l) pair through
// MAX_DEGREE is stamped with LITERAL indices behind `if <lit> <= L` guards. `L` is
// a monomorphized constant, so dead pairs fold away and live indices become
// compile-time constants: coefficient loads fold to `vbroadcast` from `.rodata`,
// stores get fixed offsets, and no integer arithmetic survives to runtime.
//
// Hygiene note: the recurrence state (`q_prev`/`q_cur`, `c`/`s`) is threaded between
// rules as `ident` arguments. Locals introduced in one expansion are invisible to
// tokens written in another rule, but a captured ident keeps its context.

/// Value-kernel ladder: z-recurrence columns fused with the `emit` sink.
macro_rules! sh_value_columns {
    ($L:ident, $V:ident, $t:ident, $x:ident, $y:ident, $z:ident, $out:ident) => {
        let mut c = $V::ONE;
        let mut s = $V::ZERO;
        sh_value_columns!(@m $L, $V, $t, $x, $y, $z, c, s, $out;
            0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16);
    };
    (@m $L:ident, $V:ident, $t:ident, $x:ident, $y:ident, $z:ident, $c:ident, $s:ident, $out:ident; $($mv:literal)*) => { $(
        if $mv <= $L {
            let q_diag = $V::splat(at(&$t.qmm, $mv));
            emit($out, $mv, $mv, q_diag, $c, $s);

            if $mv < $L {
                let mut q_prev = q_diag;
                let mut q_cur = ($z * q_prev) * $V::splat(at(&$t.em, $mv));
                emit($out, $mv + 1, $mv, q_cur, $c, $s);

                sh_value_columns!(@l $L, $V, $t, $z, $c, $s, $out, $mv, q_prev, q_cur;
                    0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16);

                // (c, s) *= (x + iy)
                let c_next = $y.nmul_adde($s, $x * $c);
                let s_next = $y.mul_adde($c, $x * $s);
                $c = c_next;
                $s = s_next;
            }
        }
    )* };
    (@l $L:ident, $V:ident, $t:ident, $z:ident, $c:ident, $s:ident, $out:ident, $mv:literal, $qp:ident, $qc:ident; $($lv:literal)*) => { $(
        if $lv >= $mv + 2 && $lv <= $L {
            let q_next = ($z * $qc)
                .mul_adde($V::splat(at(&$t.a, tri($lv) + $mv)), $qp * $V::splat(at(&$t.nb, tri($lv) + $mv)));
            emit($out, $lv, $mv, q_next, $c, $s);
            $qp = $qc;
            $qc = q_next;
        }
    )* };
}

/// Scratch-fill ladder: the same z-recurrence with the triangular `q` array as the
/// sink (pass 1 of [`sh_d_impl`]). Kept separate from [`sh_value_columns`] rather
/// than parameterized by a sink callback, since the duplication is ~20 lines and the
/// parameterized form costs far more in readability.
macro_rules! sh_q_columns {
    ($L:ident, $V:ident, $t:ident, $z:ident, $q:ident) => {
        sh_q_columns!(@m $L, $V, $t, $z, $q; 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16);
    };
    (@m $L:ident, $V:ident, $t:ident, $z:ident, $q:ident; $($mv:literal)*) => { $(
        if $mv <= $L {
            let q_diag = $V::splat(at(&$t.qmm, $mv));
            put(&mut $q, tri($mv) + $mv, q_diag);

            if $mv < $L {
                let mut q_prev = q_diag;
                let mut q_cur = ($z * q_prev) * $V::splat(at(&$t.em, $mv));
                put(&mut $q, tri($mv + 1) + $mv, q_cur);

                sh_q_columns!(@l $L, $V, $t, $z, $q, $mv, q_prev, q_cur;
                    0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16);
            }
        }
    )* };
    (@l $L:ident, $V:ident, $t:ident, $z:ident, $q:ident, $mv:literal, $qp:ident, $qc:ident; $($lv:literal)*) => { $(
        if $lv >= $mv + 2 && $lv <= $L {
            let q_next = ($z * $qc)
                .mul_adde($V::splat(at(&$t.a, tri($lv) + $mv)), $qp * $V::splat(at(&$t.nb, tri($lv) + $mv)));
            put(&mut $q, tri($lv) + $mv, q_next);
            $qp = $qc;
            $qc = q_next;
        }
    )* };
}

/// Gradient-emission ladder (pass 2 of [`sh_d_impl`]): reads the `q` scratch, no
/// recurrence state beyond the rolling `(c, s)` / `(cp, sp)` azimuthal window, so the
/// inner rule is stateless.
macro_rules! sh_grad_columns {
    ($L:ident, $V:ident, $t:ident, $x:ident, $y:ident, $z:ident, $q:ident, $out:ident, $ddx:ident, $ddy:ident, $ddz:ident) => {
        let mut c = $V::ONE;
        let mut s = $V::ZERO;
        let mut cp = $V::ZERO; // unused at m = 0 (the m factor is zero there)
        let mut sp = $V::ZERO;
        sh_grad_columns!(@m $L, $V, $t, $x, $y, $z, c, s, cp, sp, $q, $out, $ddx, $ddy, $ddz;
            0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16);
    };
    (@m $L:ident, $V:ident, $t:ident, $x:ident, $y:ident, $z:ident, $c:ident, $s:ident, $cp:ident, $sp:ident, $q:ident, $out:ident, $ddx:ident, $ddy:ident, $ddz:ident; $($mv:literal)*) => { $(
        if $mv <= $L {
            let mv = $V::splat(at(&$t.mf, $mv));

            sh_grad_columns!(@l $L, $V, $t, $c, $s, $cp, $sp, $q, $out, $ddx, $ddy, $ddz, $mv, mv;
                0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16);

            if $mv < $L {
                let c_next = $y.nmul_adde($s, $x * $c);
                let s_next = $y.mul_adde($c, $x * $s);
                $cp = $c;
                $sp = $s;
                $c = c_next;
                $s = s_next;
            }
        }
    )* };
    (@l $L:ident, $V:ident, $t:ident, $c:ident, $s:ident, $cp:ident, $sp:ident, $q:ident, $out:ident, $ddx:ident, $ddy:ident, $ddz:ident, $mv:literal, $mfv:ident; $($lv:literal)*) => { $(
        if $lv >= $mv && $lv <= $L {
            let k = tri($lv) + $mv;
            let base = $lv * ($lv + 1);

            let qv = at(&$q, k);

            // q_l^{m+1}, the z-derivative partner, zero on the diagonal.
            let qn = if $mv == $lv { $V::ZERO } else { at(&$q, k + 1) };
            let dq = $V::splat(at(&$t.f, k)) * qn;

            if $mv == 0 {
                // Y_{l,0} = q_l^0(z): no x/y dependence in the polynomial form.
                put($out, base, qv);
                put($ddx, base, $V::ZERO);
                put($ddy, base, $V::ZERO);
                put($ddz, base, dq);
            } else {
                put($out, base + $mv, qv * $c);
                put($out, base - $mv, qv * $s);

                // d{c,s}_m = m * {c,s}_{m-1} rotated: dc/dx = m c', dc/dy = -m s',
                // ds/dx = m s', ds/dy = m c'.
                let mq = $mfv * qv;
                let mq_cp = mq * $cp;
                let mq_sp = mq * $sp;

                put($ddx, base + $mv, mq_cp);
                put($ddy, base + $mv, -mq_sp);
                put($ddx, base - $mv, mq_sp);
                put($ddy, base - $mv, mq_cp);

                put($ddz, base + $mv, dq * $c);
                put($ddz, base - $mv, dq * $s);
            }
        }
    )* };
}

/// All real spherical harmonics through degree `L` at the unit direction `(x, y, z)`.
///
/// `out[l * (l + 1) + m]` receives `$Y_{\ell m}$` for `m` in `-l..=l`, and `N` must
/// equal `(L + 1)^2` (compile-time checked). See the [module docs](self) for
/// conventions, the algorithm, and the unit-vector requirement.
///
/// The policy parameter is currently unused (the evaluation is pure polynomial
/// arithmetic with one fixed, FMA-preferring lowering). It is accepted so the
/// signature matches every sibling kernel and leaves room for policy-driven variants.
// The `unused_comparisons`/`unused_assignments` allows are ladder artifacts: the
// `0 <= L` guard of the first stamped column and the dead state hand-off of the last
// stamped row are structurally unavoidable in machine-stamped straight-line code.
#[allow(
    clippy::extra_unused_type_parameters,
    clippy::int_plus_one,
    unused_comparisons,
    unused_assignments
)]
#[inline(always)]
pub fn sh_impl<P, E, V, const L: usize, const N: usize, const CS: bool>(x: V, y: V, z: V, out: &mut [V; N])
where
    P: Policy,
    E: FloatElement + ShConsts<L, N, CS>,
    V: FloatVector<Element = E>,
{
    const {
        assert!(N == (L + 1) * (L + 1));
    }

    // Above the stamped ladder there is no unrolled code to run, so this delegates to
    // the general path rather than silently leaving the high bands unwritten. Note the
    // guard cannot live in the caller: rustc monomorphizes both arms of an `if const`
    // whose condition involves a generic const parameter, so a cap assert here would
    // fire from a statically-dead call site.
    if const { L > MAX_DEGREE } {
        let mut table = ShTable::<V, N>::zeroed();
        sh_table_impl::<V, L, N, CS>(&mut table);
        sh_eval_impl::<V, L, N>(&table, x, y, z, out);
        return;
    }

    let t = &<E as ShConsts<L, N, CS>>::TABLE;

    sh_value_columns!(L, V, t, x, y, z, out);
}

/// [`sh_impl`] plus the ambient Cartesian gradient of every harmonic.
///
/// `out` receives the values exactly as [`sh_impl`] produces them, and `ddx`/`ddy`/`ddz`
/// receive `$\partial Y_{\ell m}/\partial\{x,y,z\}$` of the polynomial form at the
/// given (unit) input. See the module docs for what that means off the sphere and
/// how to project to the tangential gradient.
///
/// Two passes over an internal `q` scratch: the pure `z`-recurrence first, then one
/// combining sweep that emits values and all three derivatives from tabulated ratios,
/// with no recurrences beyond those [`sh_impl`] already runs.
// PERF: the [V; N] scratch is zero-initialized (O(N) stores) and lives on the stack,
// ~4 KB at L = 10 / f32x8. Fine for a leaf. Revisit (MaybeUninit or caller scratch)
// if profiles ever notice.
#[allow(
    clippy::extra_unused_type_parameters,
    clippy::int_plus_one,
    unused_comparisons,
    unused_assignments
)]
#[inline(always)]
pub fn sh_d_impl<P, E, V, const L: usize, const N: usize, const CS: bool>(
    x: V,
    y: V,
    z: V,
    out: &mut [V; N],
    ddx: &mut [V; N],
    ddy: &mut [V; N],
    ddz: &mut [V; N],
) where
    P: Policy,
    E: FloatElement + ShConsts<L, N, CS>,
    V: FloatVector<Element = E>,
{
    const {
        assert!(N == (L + 1) * (L + 1));
    }

    // See `sh_impl`: above the ladder, delegate to the general path.
    if const { L > MAX_DEGREE } {
        let mut table = ShTable::<V, N>::zeroed();
        sh_table_impl::<V, L, N, CS>(&mut table);
        sh_eval_d_impl::<V, L, N>(&table, x, y, z, out, ddx, ddy, ddz);
        return;
    }

    let t = &<E as ShConsts<L, N, CS>>::TABLE;

    // Pass 1: every q_l^m, by column, into triangular scratch.
    let mut q = [V::ZERO; N];
    sh_q_columns!(L, V, t, z, q);

    // Pass 2: combine scratch, tabulated ratios, and the rolling azimuthal window
    // ((c, s) at column m, (cp, sp) at column m - 1) into values and gradients.
    sh_grad_columns!(L, V, t, x, y, z, q, out, ddx, ddy, ddz);
}

// --- The general path: runtime coefficients, rolled loops, any degree ---
//
// The same normalized recurrence as the unrolled kernels above, with the constants
// computed rather than tabulated. That buys two things the const-table path cannot
// offer: degrees beyond `MAX_DEGREE` (whose table would be `6 * (L+1)^2` entries of
// rodata, roughly half a megabyte at `L = 100`), and element types that have no
// `ShConsts` impl at all, which is every composite.
//
// Numerically this IS the fast path, not an approximation of it: identical
// recurrence, identical `O(1)` intermediates, no overflow at any degree. Only the
// provenance of the coefficients differs, so the two can be diffed directly.
//
// All three are `#[inline(always)]`, not `#[inline]`. That is rule zero, not a
// preference: target features propagate into a callee only when it is inlined, so a
// merely-`#[inline]` kernel that rustc declines to inline compiles at the base ISA. It
// was measured doing exactly that: 1851 instructions of SSE2 with no FMA and not one
// `ymm` register, called from an AVX2 caller.
//
// Everything here needs nothing beyond `FloatVector` (add, mul, div, sqrt). Notably
// `l` and `m` are carried as running `V` values incremented by `V::ONE` rather than
// converted from integers, which keeps even the element-conversion traits out of the
// bounds. Integer values this small are exact in any float format.

/// Computes the recurrence coefficients for degree `L` into a runtime table.
///
/// The expensive half of the general path (two `sqrt` and two divisions per `(l, m)`),
/// and the reason it is a separate entry point: it depends only on `L` and `CS`, never
/// on the direction, so a caller evaluating many directions computes it once.
///
/// `CS` is baked in here, which is why [`sh_eval_impl`] does not take it. A filled
/// table already knows its phase convention.
#[allow(clippy::extra_unused_type_parameters)]
#[inline(always)]
pub fn sh_table_impl<V, const L: usize, const N: usize, const CS: bool>(t: &mut ShTable<V, N>)
where
    V: FloatVector,
{
    const {
        assert!(N == (L + 1) * (L + 1));
    }

    let two = V::ONE + V::ONE;

    // q_0^0 = sqrt(1/4pi) = (1/sqrt(pi)) / 2, exactly what the fast table seeds with.
    let mut mag = V::FRAC_1_SQRT_PI / two;
    put(&mut t.qmm, 0, mag);

    // Diagonal: multiply by sqrt((2m+1)/(2m)) per step, with one extra sqrt(2) at
    // m = 1 (the sqrt(2 - delta_{m0}) of the real normalization, entering once).
    //
    // The Condon-Shortley sign is applied to each entry as it is stored, and the
    // recurrence is carried in `mag`, which stays unsigned. Feeding a SIGNED entry
    // back into the next step instead would compound the phases: column m would come
    // out with (-1)^(number of odd columns below it) rather than (-1)^m, which is
    // right for odd m and wrong for even m.
    let mut mv = V::ZERO;
    let mut m = 1;
    while m <= L {
        mv += V::ONE;
        let two_m = mv * two;
        let mut d = ((two_m + V::ONE) / two_m).sqrt();
        if m == 1 {
            d *= two.sqrt();
        }
        mag *= d;
        put(&mut t.qmm, m, if CS && m % 2 == 1 { -mag } else { mag });
        m += 1;
    }

    let mut mv = V::ZERO;
    let mut m = 0;
    while m <= L {
        put(&mut t.mf, m, mv);

        if m < L {
            put(&mut t.em, m, (mv * two + V::ONE + two).sqrt());
        }

        let mut lv = mv;
        let mut l = m;
        while l <= L {
            let k = tri(l) + m;

            let lm_lo = lv - mv; // l - m
            let lm_hi = lv + mv; // l + m
            let denom = lm_lo * lm_hi;
            let two_l = lv * two;

            if l >= m + 2 {
                put(&mut t.a, k, ((two_l + V::ONE) * (two_l - V::ONE) / denom).sqrt());
                put(
                    &mut t.nb,
                    k,
                    -(((two_l + V::ONE) * (lm_lo - V::ONE) * (lm_hi - V::ONE)) / (denom * (two_l - two - V::ONE)))
                        .sqrt(),
                );
            }

            // d(q_l^m)/dz = f * q_l^{m+1}. Zero on the diagonal, and the m = 0 column
            // picks up a 1/sqrt(2) because sqrt(2 - delta_{m0}) differs between the
            // two columns the ratio spans.
            if l > m {
                let fv = if m == 0 {
                    (lv * (lv + V::ONE) / two).sqrt()
                } else {
                    (lm_lo * (lm_hi + V::ONE)).sqrt()
                };

                // The ratio crosses columns m and m + 1, whose Condon-Shortley signs
                // always disagree, so under CS every f flips regardless of parity.
                put(&mut t.f, k, if CS { -fv } else { fv });
            }

            lv += V::ONE;
            l += 1;
        }

        mv += V::ONE;
        m += 1;
    }
}

/// Evaluates all harmonics through degree `L` from a table filled by [`sh_table_impl`].
///
/// The rolled counterpart of [`sh_impl`], for any `L` and any `V`. The phase convention
/// comes from the table, so there is no `CS` parameter here.
#[allow(clippy::extra_unused_type_parameters)]
#[inline(always)]
pub fn sh_eval_impl<V, const L: usize, const N: usize>(t: &ShTable<V, N>, x: V, y: V, z: V, out: &mut [V; N])
where
    V: FloatVector,
{
    const {
        assert!(N == (L + 1) * (L + 1));
    }

    let mut c = V::ONE;
    let mut s = V::ZERO;

    let mut m = 0;
    while m <= L {
        let q_diag = at(&t.qmm, m);
        emit(out, m, m, q_diag, c, s);

        if m < L {
            let mut q_prev = q_diag;
            let mut q_cur = (z * q_prev) * at(&t.em, m);
            emit(out, m + 1, m, q_cur, c, s);

            let mut l = m + 2;
            while l <= L {
                let k = tri(l) + m;
                let q_next = (z * q_cur).mul_adde(at(&t.a, k), q_prev * at(&t.nb, k));
                emit(out, l, m, q_next, c, s);

                q_prev = q_cur;
                q_cur = q_next;
                l += 1;
            }

            let c_next = y.nmul_adde(s, x * c);
            let s_next = y.mul_adde(c, x * s);
            c = c_next;
            s = s_next;
        }

        m += 1;
    }
}

/// [`sh_eval_impl`] over a table stored in `W`'s _primal_ type, each coefficient
/// lifted through [`from_primal`](PrimalProjection::from_primal) as it is read.
///
/// The generic fallback behind `spherical_harmonics_with` now that tables are
/// `Self::Primal`-typed. For a type that is its own primal (`Vector`, `Compensated`)
/// the lift is the identity and this folds to exactly [`sh_eval_impl`], FMAs
/// included. A composite gets correct-but-unspecialized code (its constants carry
/// zeroed augmentation through full composite multiplies), which is why `Dual`
/// overrides the method with [`sh_eval_mixed_impl`] instead.
#[allow(clippy::extra_unused_type_parameters)]
#[inline(always)]
pub fn sh_eval_lifted_impl<W, const L: usize, const N: usize>(
    t: &ShTable<W::Primal, N>,
    x: W,
    y: W,
    z: W,
    out: &mut [W; N],
) where
    W: FloatVector + PrimalProjection,
{
    const {
        assert!(N == (L + 1) * (L + 1));
    }

    let mut c = W::ONE;
    let mut s = W::ZERO;

    let mut m = 0;
    while m <= L {
        let q_diag = W::from_primal(at(&t.qmm, m));
        emit(out, m, m, q_diag, c, s);

        if m < L {
            let mut q_prev = q_diag;
            let mut q_cur = (z * q_prev) * W::from_primal(at(&t.em, m));
            emit(out, m + 1, m, q_cur, c, s);

            let mut l = m + 2;
            while l <= L {
                let k = tri(l) + m;
                let q_next = (z * q_cur).mul_adde(W::from_primal(at(&t.a, k)), q_prev * W::from_primal(at(&t.nb, k)));
                emit(out, l, m, q_next, c, s);

                q_prev = q_cur;
                q_cur = q_next;
                l += 1;
            }

            let c_next = y.nmul_adde(s, x * c);
            let s_next = y.mul_adde(c, x * s);
            c = c_next;
            s = s_next;
        }

        m += 1;
    }
}

/// [`sh_eval_impl`] with the coefficients kept in a _different_, simpler type than the
/// values.
///
/// The case this exists for is a composite `W` (a `Dual`, say) evaluated against a
/// table of plain real coefficients. Every recurrence constant has a zero derivative,
/// so carrying it as a `Dual` means computing `a.re * 0.0` cross terms for each one,
/// which LLVM cannot fold away under strict IEEE (`a.re` could be an infinity or a
/// NaN). Typing the table by `R`'s primal instead turns each of those into
/// `Dual * real`, which `thermite-dual` implements as `1 + N` multiplies rather than
/// `1 + 2N`.
///
/// It also shrinks the table itself, which is the larger saving in practice: a
/// `ShTable<Dual<V, 3>, 25>` is 600 vector stores to fill, against 150 for
/// `ShTable<V, 25>`, and the general path fills one per call.
///
/// Deliberately _not_ a generalization of [`sh_eval_impl`]. The single-type version
/// folds its recurrence into `mul_adde`, and no fused multiply-add spans two operand
/// types, so merging them would cost the real path its FMAs to benefit the composite
/// one. The duplicated body is about twenty lines and neither copy has to compromise.
#[allow(clippy::extra_unused_type_parameters)]
#[inline(always)]
pub fn sh_eval_mixed_impl<W, R, const L: usize, const N: usize>(
    t: &ShTable<R::Primal, N>,
    x: W,
    y: W,
    z: W,
    out: &mut [W; N],
) where
    W: FloatVector + core::ops::Mul<R, Output = W>,
    R: PrimalProjection,
{
    const {
        assert!(N == (L + 1) * (L + 1));
    }

    let mut c = W::ONE;
    let mut s = W::ZERO;

    let mut m = 0;
    while m <= L {
        // The table is `R::Primal`-typed. `R::from_primal` lifts an entry to `R` (the
        // identity for a plain real `R`), and `W::ONE * r` lifts that into `W` without
        // needing a conversion trait: for a real `W` it is the identity LLVM folds
        // away, and for a `Dual` it produces the constant with a zero derivative
        // directly.
        let q_diag = W::ONE * R::from_primal(at(&t.qmm, m));
        emit(out, m, m, q_diag, c, s);

        if m < L {
            let mut q_prev = q_diag;
            let mut q_cur = (z * q_prev) * R::from_primal(at(&t.em, m));
            emit(out, m + 1, m, q_cur, c, s);

            let mut l = m + 2;
            while l <= L {
                let k = tri(l) + m;
                let q_next = (z * q_cur) * R::from_primal(at(&t.a, k)) + q_prev * R::from_primal(at(&t.nb, k));
                emit(out, l, m, q_next, c, s);

                q_prev = q_cur;
                q_cur = q_next;
                l += 1;
            }

            let c_next = y.nmul_adde(s, x * c);
            let s_next = y.mul_adde(c, x * s);
            c = c_next;
            s = s_next;
        }

        m += 1;
    }
}

/// [`sh_eval_impl`] plus the ambient Cartesian gradients. The rolled counterpart of
/// [`sh_d_impl`], with the same two-pass structure and gradient semantics.
#[allow(clippy::extra_unused_type_parameters, clippy::too_many_arguments)]
#[inline(always)]
pub fn sh_eval_d_impl<V, const L: usize, const N: usize>(
    t: &ShTable<V, N>,
    x: V,
    y: V,
    z: V,
    out: &mut [V; N],
    ddx: &mut [V; N],
    ddy: &mut [V; N],
    ddz: &mut [V; N],
) where
    V: FloatVector,
{
    const {
        assert!(N == (L + 1) * (L + 1));
    }

    // Pass 1: the z-recurrence into triangular scratch.
    let mut q = [V::ZERO; N];

    let mut m = 0;
    while m <= L {
        let q_diag = at(&t.qmm, m);
        put(&mut q, tri(m) + m, q_diag);

        if m < L {
            let mut q_prev = q_diag;
            let mut q_cur = (z * q_prev) * at(&t.em, m);
            put(&mut q, tri(m + 1) + m, q_cur);

            let mut l = m + 2;
            while l <= L {
                let k = tri(l) + m;
                let q_next = (z * q_cur).mul_adde(at(&t.a, k), q_prev * at(&t.nb, k));
                put(&mut q, k, q_next);

                q_prev = q_cur;
                q_cur = q_next;
                l += 1;
            }
        }

        m += 1;
    }

    // Pass 2: values and gradients from the scratch and the tabulated ratios.
    let mut c = V::ONE;
    let mut s = V::ZERO;
    let mut cp = V::ZERO;
    let mut sp = V::ZERO;

    let mut m = 0;
    while m <= L {
        let mv = at(&t.mf, m);

        let mut l = m;
        while l <= L {
            let k = tri(l) + m;
            let base = l * (l + 1);

            let qv = at(&q, k);
            let qn = if m == l { V::ZERO } else { at(&q, k + 1) };
            let dq = at(&t.f, k) * qn;

            if m == 0 {
                put(out, base, qv);
                put(ddx, base, V::ZERO);
                put(ddy, base, V::ZERO);
                put(ddz, base, dq);
            } else {
                put(out, base + m, qv * c);
                put(out, base - m, qv * s);

                let mq = mv * qv;
                let mq_cp = mq * cp;
                let mq_sp = mq * sp;

                put(ddx, base + m, mq_cp);
                put(ddy, base + m, -mq_sp);
                put(ddx, base - m, mq_sp);
                put(ddy, base - m, mq_cp);

                put(ddz, base + m, dq * c);
                put(ddz, base - m, dq * s);
            }

            l += 1;
        }

        if m < L {
            let c_next = y.nmul_adde(s, x * c);
            let s_next = y.mul_adde(c, x * s);
            cp = c;
            sp = s;
            c = c_next;
            s = s_next;
        }

        m += 1;
    }
}
