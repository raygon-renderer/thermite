//! The batch Zernike kernel: every mode through degree `L` at one point, in Cartesian
//! coordinates.
//!
//! # The Cartesian substitution
//!
//! The polar definition `$Z_n^m = R_n^{|m|}(\rho)\cos(m\theta)$` suggests a `sin_cos` per
//! mode and a `powi` per mode. Both disappear under one substitution.
//!
//! Write `$s = x^2 + y^2 = \rho^2$`. The radial polynomial factors as
//!
//! ```math
//! R_n^{|m|}(\rho) = \rho^{|m|}\, Q_{k,|m|}(s), \qquad
//! Q_{k,m}(s) = P_k^{(0,m)}(2s - 1), \qquad k = \tfrac{n - |m|}{2}
//! ```
//!
//! so the `$\rho^{|m|}$` is the *only* place an odd power of `$\rho$` appears, and
//! `$Q$` is an honest polynomial in `s`. Meanwhile
//!
//! ```math
//! (x + iy)^m = \rho^m\left(\cos m\theta + i \sin m\theta\right)
//! ```
//!
//! so `$\rho^{|m|}\cos(m\theta)$` and `$\rho^{|m|}\sin(m\theta)$` are exactly the real and
//! imaginary parts of `$(x+iy)^{|m|}$`, which come off a two-line complex ladder. The
//! `$\rho^{|m|}$` the radial part needed and the `$\rho^{|m|}$` the angular part produced
//! are the same factor, so they never have to be formed separately:
//!
//! ```math
//! Z_n^m = Q_{k,|m|}(s) \times \begin{cases}\operatorname{Re}(x+iy)^{m} & m \ge 0\\
//!                                          \operatorname{Im}(x+iy)^{|m|} & m < 0\end{cases}
//! ```
//!
//! The whole basis is therefore pure polynomial arithmetic in `(x, y)`: no `atan2`, no
//! `sqrt`, no trigonometry, no division, `$O(L^2)$` FMAs total, and no singularity at the
//! pupil centre (which the polar form has, in `$\partial_\theta Z / \rho$`).
//!
//! Taking `(x, y)` rather than `$(\rho, \theta)$` is thus not a convenience: a polar entry
//! point would make the caller pay an `atan2` per sample to build an angle this kernel
//! immediately destroys. Pupil samples arrive as Cartesian coordinates anyway.
//!
//! # The recurrence
//!
//! `$Q_{k,m}$` is the Jacobi three-term recurrence rewritten in `s` rather than
//! `$t = 2s-1$`, which folds the change of variable into the coefficients instead of
//! spending an operation on it per mode:
//!
//! ```math
//! Q_{0,m} = 1,\qquad Q_{1,m}(s) = (m+2)s - (m+1)
//! ```
//! ```math
//! Q_{k,m} = (A_{k,m}\,s + B_{k,m})\,Q_{k-1,m} - C_{k,m}\,Q_{k-2,m}
//! ```
//!
//! with, writing `$c = 2k(k+m)(2k+m-2)$`,
//!
//! ```math
//! A = \frac{2(2k+m-1)(2k+m)(2k+m-2)}{c},\quad
//! B = \frac{-(2k+m-1)(m^2 + (2k+m)(2k+m-2))}{c},\quad
//! C = \frac{2(k-1)(k+m-1)(2k+m)}{c}
//! ```
//!
//! Every coefficient is a ratio of small integers - the largest through `L = 16` is 3360,
//! comfortably exact in f32 - and at stamped literal `(k, m)` they fold to `.rodata`
//! constants. Three operations per mode: one FMA for `As + B`, one multiply, one FMA.
//!
//! Running the recurrence in `k` at fixed `m` is what makes this `$O(L^2)$` rather than
//! the `$O(L^3)$` of calling the single-mode entry point per mode, which restarts the
//! recurrence from `k = 0` every time.
//!
//! # Layout
//!
//! `out[j]` for the ANSI Z80.28 / OSA index `j = (n(n+2) + m)/2`, so `N` must be
//! `(L+1)(L+2)/2`. ANSI is the layout rather than Noll or Fringe because it is the
//! scheme whose index is a closed form *and* whose degree truncation is contiguous;
//! [`noll_to_ansi`](crate::zernike::noll_to_ansi) and
//! [`fringe_to_ansi`](crate::zernike::fringe_to_ansi) gather from it.
//!
//! Nothing normalizes `(x, y)` onto the unit disc, exactly as the spherical-harmonic
//! kernels do not renormalize their direction. Outside it the polynomials are still
//! evaluated correctly and simply are not orthogonal.

use thermite::{
    math::{CoreMath, policy::Policy},
    prelude::*,
    register::FloatElement,
};

use crate::zernike::{ZERNIKE_ORTHONORMAL, ZERNIKE_UNIT_PEAK};

/// Highest degree the stamped ladder below covers. Beyond it the kernel takes the
/// rolled path, which is correct at any degree but computes its coefficients at runtime
/// and does not unroll. Extending the ladder is mechanical: append literals to the `m`
/// list and, every two degrees, to the `k` list.
pub const MAX_DEGREE: usize = 16;

// Index invariant, so the kernel can use unchecked accesses. `N == (L+1)(L+2)/2` is
// const-asserted, and every emitted mode satisfies |m| <= n <= L, whose ANSI index
//   (n(n+2) + m)/2  <=  (L(L+2) + L)/2  =  L(L+3)/2  =  N - 1
// so every store is in bounds by construction. The checked forms are not used because
// their panic paths defeat the unroller, the same finding the SH kernel records.

/// The `(A, B, C)` coefficients of the `Q` recurrence at step `k >= 2`, order `m`.
///
/// ```text
/// Q_k = (A s + B) Q_{k-1} - C Q_{k-2}
/// ```
///
/// Written once and shared by all three consumers (the stamped ladder folds this to
/// literals, while the rolled path and the single-mode radial call it with runtime
/// `(k, m)`), so the recurrence exists in exactly one place. The divisions are of small integers in
/// the element type, never of the vector, so they stay off the recurrence's critical
/// path even where they are not folded away.
#[inline(always)]
fn q_coeffs<E: FloatElement>(k: thermite::LargeInt, m: thermite::LargeInt) -> (E, E, E) {
    let c = 2 * k * (k + m) * (2 * k + m - 2);

    let an = (2 * k + m - 1) * (2 * k + m) * (2 * k + m - 2);
    let bn = (2 * k + m - 1) * m * m;
    let cn = 2 * (k - 1) * (k + m - 1) * (2 * k + m);

    (
        E::from_ratio(2 * an, c),
        E::from_ratio(-(bn + an), c),
        E::from_ratio(cn, c),
    )
}

/// The reduced radial polynomial `$Q_{k,m}(s) = P_k^{(0,m)}(2s - 1)$` at runtime `(k, m)`.
///
/// The single-mode counterpart of one column of the batch ladder. Used by
/// [`zernike_r`](crate::SpecialMath::zernike_r) in place of a general `jacobi` call: the
/// general form carries runtime `alpha`/`beta` and divides *the vector* once per step,
/// putting a full divide latency in the dependency chain, where this divides small
/// integers in the element type instead.
#[inline(always)]
pub fn reduced_radial_impl<E, V>(s: V, k: u32, m: u32) -> V
where
    E: FloatElement,
    V: FloatVector<Element = E>,
{
    if k == 0 {
        return V::ONE;
    }

    let mf = m as thermite::LargeInt;

    // Q_1(s) = (m+2)s - (m+1)
    let mut q_prev = V::ONE;
    let mut q_cur = s.mul_sube(V::splat(E::from_int(mf + 2)), V::splat(E::from_int(mf + 1)));

    let mut kk = 2;
    while kk <= k {
        let (a, b, c) = q_coeffs::<E>(kk as thermite::LargeInt, mf);

        let q_next = s
            .mul_adde(V::splat(a), V::splat(b))
            .mul_sube(q_cur, V::splat(c) * q_prev);

        q_prev = q_cur;
        q_cur = q_next;

        kk += 1;
    }

    q_cur
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

/// The ANSI Z80.28 / OSA slot for `(n, m)`, as a `usize` for indexing.
#[inline(always)]
const fn slot(n: usize, m: i32) -> usize {
    ((n * (n + 2)) as i32 + m) as usize / 2
}

/// `$N_n^m$` for the requested normalization, as an element constant.
///
/// Folds to a literal at stamped `(n, m)`: `NORM` is a monomorphized constant and the
/// `sqrt` is of an exactly-representable integer.
#[inline(always)]
fn norm<E: FloatElement, const NORM: u8>(n: usize, m: usize) -> E {
    if const { NORM == ZERNIKE_UNIT_PEAK } {
        return <E as thermite::register::Element>::ONE;
    }

    // sqrt(2(n+1) / (1 + delta_{m,0}))
    let radicand = if m == 0 { n + 1 } else { 2 * (n + 1) };

    FloatElement::sqrt(E::from_int(radicand as thermite::LargeInt))
}

/// Writes the one or two modes of degree `n` and azimuthal order `+-m`.
///
/// `q` is the reduced radial polynomial `$Q_{k,m}(s)$`; `u` and `v` are the real and
/// imaginary parts of `$(x+iy)^m$`, which already carry the `$\rho^m$` the radial part
/// omitted. `m = 0` has a single mode, and `v` is zero there anyway.
#[inline(always)]
fn emit<E, V, const N: usize, const NORM: u8>(out: &mut [V; N], n: usize, m: usize, q: V, u: V, v: V)
where
    E: FloatElement,
    V: FloatVector<Element = E>,
{
    let qn = q * V::splat(norm::<E, NORM>(n, m));

    if m == 0 {
        put(out, slot(n, 0), qn);
    } else {
        put(out, slot(n, m as i32), qn * u);
        put(out, slot(n, -(m as i32)), qn * v);
    }
}

/// [`emit`] plus the Cartesian gradient of the same one or two modes.
///
/// `dq` is `$\partial Q_{k,m}/\partial s$`, and `up`/`vp` are the real and imaginary parts
/// of `$(x+iy)^{m-1}$` - the previous rung of the same ladder `u`/`v` came from.
///
/// Both factors of the mode depend on the point, so both differentiate. The radial half
/// goes through `s`, giving `$\partial s/\partial x = 2x$`; the azimuthal half is a
/// complex power, so `$\partial_x (x+iy)^m = m(x+iy)^{m-1}$` and
/// `$\partial_y (x+iy)^m = im(x+iy)^{m-1}$`, which is why the `y` derivative crosses the
/// real and imaginary parts over and flips one sign.
///
/// `tx`/`ty` are `2x` and `2y`, hoisted by the caller since every mode uses them.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn emit_d<E, V, const N: usize, const NORM: u8>(
    out: &mut [V; N],
    ddx: &mut [V; N],
    ddy: &mut [V; N],
    n: usize,
    m: usize,
    (tx, ty): (V, V),
    (q, dq): (V, V),
    (u, v): (V, V),
    (up, vp): (V, V),
) where
    E: FloatElement,
    V: FloatVector<Element = E>,
{
    let scale = V::splat(norm::<E, NORM>(n, m));

    let qn = q * scale;
    let dqn = dq * scale;

    if m == 0 {
        // u = 1 and du/dx = 0 * u_{-1} = 0, so only the radial half survives.
        put(out, slot(n, 0), qn);
        put(ddx, slot(n, 0), tx * dqn);
        put(ddy, slot(n, 0), ty * dqn);

        return;
    }

    let mq = qn * V::splat(E::from_int(m as thermite::LargeInt));

    let (rx, ry) = (tx * dqn, ty * dqn);

    let jp = slot(n, m as i32);
    let jm = slot(n, -(m as i32));

    put(out, jp, qn * u);
    put(ddx, jp, rx.mul_adde(u, mq * up));
    put(ddy, jp, ry.mul_sube(u, mq * vp));

    put(out, jm, qn * v);
    put(ddx, jm, rx.mul_adde(v, mq * vp));
    put(ddy, jm, ry.mul_adde(v, mq * up));
}

// --- The literal ladder ---
//
// Stamped rather than looped, for the reason the SH kernel documents at length: LLVM
// declines to unroll a triangular nest over a const-generic bound, leaving runtime index
// arithmetic and register-indexed coefficient loads. Behind `if <lit> <= L` guards with
// `L` a monomorphized constant, dead modes fold away and live ones become fixed offsets
// and `.rodata` broadcasts.
//
// Hygiene note, as in `sh`: recurrence state (`q_prev`/`q_cur`, `u`/`v`) is threaded
// between rules as `ident` arguments so it keeps its definition context.

macro_rules! zernike_columns {
    ($L:ident, $E:ident, $V:ident, $NORM:ident, $s:ident, $x:ident, $y:ident, $out:ident) => {
        // (u, v) = Re/Im of (x + iy)^m, starting at m = 0.
        let mut u = $V::ONE;
        let mut v = $V::ZERO;

        zernike_columns!(@m $L, $E, $V, $NORM, $s, $x, $y, u, v, $out;
            0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16);
    };

    (@m $L:ident, $E:ident, $V:ident, $NORM:ident, $s:ident, $x:ident, $y:ident,
     $u:ident, $v:ident, $out:ident; $($mv:literal)*) => { $(
        if $mv <= $L {
            // k = 0: Q = 1, so the mode is the azimuthal factor alone. This is the
            // n = |m| diagonal, R_m^m = rho^m.
            emit::<$E, $V, _, $NORM>($out, $mv, $mv, $V::ONE, $u, $v);

            if $mv + 2 <= $L {
                // k = 1: Q_{1,m}(s) = (m+2)s - (m+1)
                let mut q_prev = $V::ONE;
                let mut q_cur = $s.mul_sube(
                    $V::splat($E::from_int($mv + 2)),
                    $V::splat($E::from_int($mv + 1)),
                );

                emit::<$E, $V, _, $NORM>($out, $mv + 2, $mv, q_cur, $u, $v);

                zernike_columns!(@k $L, $E, $V, $NORM, $s, $u, $v, $out, $mv, q_prev, q_cur;
                    2 3 4 5 6 7 8);
            }

            if $mv < $L {
                // (u, v) *= (x + iy)
                let u_next = $x.difference_of_products($u, $y, $v);
                let v_next = $x.sum_of_products($v, $y, $u);
                $u = u_next;
                $v = v_next;
            }
        }
    )* };

    (@k $L:ident, $E:ident, $V:ident, $NORM:ident, $s:ident, $u:ident, $v:ident, $out:ident,
     $mv:literal, $qp:ident, $qc:ident; $($kv:literal)*) => { $(
        if $mv + 2 * $kv <= $L {
            // Literal (k, m), so `q_coeffs` folds to three `.rodata` constants.
            let (ae, be, ce) = q_coeffs::<$E>($kv, $mv);

            let a = $V::splat(ae);
            let b = $V::splat(be);
            let c = $V::splat(ce);

            let q_next = $s.mul_adde(a, b).mul_sube($qc, c * $qp);


            emit::<$E, $V, _, $NORM>($out, $mv + 2 * $kv, $mv, q_next, $u, $v);

            $qp = $qc;
            $qc = q_next;
        }
    )* };
}

/// The gradient ladder: the same columns carrying `(Q, dQ/ds)` and a one-rung window on
/// the complex power ladder.
///
/// Kept separate from [`zernike_columns`] rather than parameterized by a sink, on the
/// same judgement the SH kernel records: the duplication is short and the parameterized
/// form costs far more in readability. Unlike `sh_d_impl` this needs no scratch array and
/// no second pass: differentiating the `Q` recurrence gives another recurrence of the
/// same shape, so value and slope advance together in one sweep.
macro_rules! zernike_grad_columns {
    ($L:ident, $E:ident, $V:ident, $NORM:ident, $s:ident, $t:ident, $x:ident, $y:ident,
     $out:ident, $ddx:ident, $ddy:ident) => {
        // (u, v) at column m, (up, vp) at column m - 1. The m = 0 column never reads the
        // window, since its azimuthal factor is the constant 1.
        let mut u = $V::ONE;
        let mut v = $V::ZERO;
        let mut up = $V::ZERO;
        let mut vp = $V::ZERO;

        zernike_grad_columns!(@m $L, $E, $V, $NORM, $s, $t, $x, $y, u, v, up, vp, $out, $ddx, $ddy;
            0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16);
    };

    (@m $L:ident, $E:ident, $V:ident, $NORM:ident, $s:ident, $t:ident, $x:ident, $y:ident,
     $u:ident, $v:ident, $up:ident, $vp:ident, $out:ident, $ddx:ident, $ddy:ident;
     $($mv:literal)*) => { $(
        if $mv <= $L {
            // k = 0: Q = 1, dQ/ds = 0.
            emit_d::<$E, $V, _, $NORM>(
                $out, $ddx, $ddy, $mv, $mv, $t, ($V::ONE, $V::ZERO), ($u, $v), ($up, $vp),
            );

            if $mv + 2 <= $L {
                // k = 1: Q = (m+2)s - (m+1), dQ/ds = m+2.
                let mut q_prev = $V::ONE;
                let mut d_prev = $V::ZERO;

                let slope = $V::splat($E::from_int($mv + 2));

                let mut q_cur = $s.mul_sube(slope, $V::splat($E::from_int($mv + 1)));
                let mut d_cur = slope;

                emit_d::<$E, $V, _, $NORM>(
                    $out, $ddx, $ddy, $mv + 2, $mv, $t, (q_cur, d_cur), ($u, $v), ($up, $vp),
                );

                zernike_grad_columns!(@k $L, $E, $V, $NORM, $s, $t, $u, $v, $up, $vp,
                    $out, $ddx, $ddy, $mv, q_prev, q_cur, d_prev, d_cur; 2 3 4 5 6 7 8);
            }

            if $mv < $L {
                // (up, vp) = (u, v); (u, v) *= (x + iy)
                let u_next = $x.difference_of_products($u, $y, $v);
                let v_next = $x.sum_of_products($v, $y, $u);

                $up = $u;
                $vp = $v;
                $u = u_next;
                $v = v_next;
            }
        }
    )* };

    (@k $L:ident, $E:ident, $V:ident, $NORM:ident, $s:ident, $t:ident,
     $u:ident, $v:ident, $up:ident, $vp:ident, $out:ident, $ddx:ident, $ddy:ident,
     $mv:literal, $qp:ident, $qc:ident, $dp:ident, $dc:ident; $($kv:literal)*) => { $(
        if $mv + 2 * $kv <= $L {
            let (ae, be, ce) = q_coeffs::<$E>($kv, $mv);

            let a = $V::splat(ae);
            let b = $V::splat(be);
            let c = $V::splat(ce);

            let lin = $s.mul_adde(a, b);

            // Q_k   = (As + B) Q_{k-1} - C Q_{k-2}
            // Q'_k  = A Q_{k-1} + (As + B) Q'_{k-1} - C Q'_{k-2}
            let q_next = lin.mul_sube($qc, c * $qp);
            let d_next = a.mul_adde($qc, lin.mul_sube($dc, c * $dp));

            emit_d::<$E, $V, _, $NORM>(
                $out, $ddx, $ddy, $mv + 2 * $kv, $mv, $t, (q_next, d_next), ($u, $v), ($up, $vp),
            );

            $qp = $qc;
            $qc = q_next;
            $dp = $dc;
            $dc = d_next;
        }
    )* };
}

/// Every Zernike mode through degree `L` at the Cartesian point `(x, y)`.
///
/// `out[(n(n+2) + m)/2]` receives `$Z_n^m$` in the normalization named by `NORM`, and `N`
/// must equal `(L+1)(L+2)/2` (compile-time checked). See the module docs for the
/// algorithm and the domain note.
///
/// The policy parameter is unused: evaluation is pure polynomial arithmetic with one
/// fixed FMA-preferring lowering. It is accepted so the signature matches its siblings.
// The `unused_comparisons`/`unused_assignments` allows are ladder artifacts, exactly as
// in `sh`: the `0 <= L` guard of the first stamped column, and the dead state hand-off
// of the last stamped row.
#[allow(
    clippy::extra_unused_type_parameters,
    clippy::int_plus_one,
    unused_comparisons,
    unused_assignments
)]
#[inline(always)]
pub fn zernike_basis_impl<P, E, V, const L: usize, const NORM: u8, const N: usize>(x: V, y: V, out: &mut [V; N])
where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E> + CoreMath,
{
    const {
        assert!(N == (L + 1) * (L + 2) / 2, "zernike_basis: N must equal (L+1)(L+2)/2");
        assert!(
            NORM == ZERNIKE_UNIT_PEAK || NORM == ZERNIKE_ORTHONORMAL,
            "zernike_basis: NORM must be ZERNIKE_UNIT_PEAK or ZERNIKE_ORTHONORMAL"
        );
    }

    let s = x.mul_adde(x, y * y);

    // Above the stamped ladder there is no unrolled code to run, so this delegates to
    // the rolled path rather than silently leaving the high degrees unwritten. As in
    // `sh_impl`, the guard cannot live in the caller: rustc monomorphizes both arms of
    // an `if const` over a generic const parameter.
    if const { L > MAX_DEGREE } {
        rolled::<E, V, L, NORM, N>(s, x, y, out);
        return;
    }

    zernike_columns!(L, E, V, NORM, s, x, y, out);
}

/// The same recurrence with runtime coefficients and rolled loops, for `L > MAX_DEGREE`.
///
/// Correct at any degree and considerably slower: the coefficients are divisions rather
/// than folded constants, and the index arithmetic survives to runtime.
#[inline(always)]
fn rolled<E, V, const L: usize, const NORM: u8, const N: usize>(s: V, x: V, y: V, out: &mut [V; N])
where
    E: FloatElement,
    V: FloatVector<Element = E> + CoreMath,
{
    let mut u = V::ONE;
    let mut v = V::ZERO;

    let mut m = 0;
    while m <= L {
        emit::<E, V, N, NORM>(out, m, m, V::ONE, u, v);

        if m + 2 <= L {
            let mf = m as thermite::LargeInt;

            let mut q_prev = V::ONE;
            let mut q_cur = s.mul_sube(V::splat(E::from_int(mf + 2)), V::splat(E::from_int(mf + 1)));

            emit::<E, V, N, NORM>(out, m + 2, m, q_cur, u, v);

            let mut k = 2;
            while m + 2 * k <= L {
                let (a, b, c) = q_coeffs::<E>(k as thermite::LargeInt, mf);

                let q_next = s
                    .mul_adde(V::splat(a), V::splat(b))
                    .mul_sube(q_cur, V::splat(c) * q_prev);

                emit::<E, V, N, NORM>(out, m + 2 * k, m, q_next, u, v);

                q_prev = q_cur;
                q_cur = q_next;

                k += 1;
            }
        }

        if m < L {
            let u_next = x.difference_of_products(u, y, v);
            let v_next = x.sum_of_products(v, y, u);
            u = u_next;
            v = v_next;
        }

        m += 1;
    }
}

/// [`zernike_basis_impl`] plus the Cartesian gradient of every mode.
///
/// `out` receives the values exactly as [`zernike_basis_impl`] produces them, and
/// `ddx`/`ddy` receive `$\partial Z_n^m/\partial\{x,y\}$` at the same point.
///
/// This is what a Shack-Hartmann reconstruction integrates against: the sensor measures
/// wavefront *slopes*, so the fit matrix is built from the gradient basis rather than the
/// value basis. Prefer it over seeding a `Dual<V, 2>` and calling the value form, which
/// carries two derivative components through every operation of the whole ladder, where
/// this shares the `Q` recurrence between the value and both gradients and differentiates
/// only the two factors that actually depend on the point.
///
/// The gradient is finite everywhere including the pupil centre, which is the practical
/// payoff of the Cartesian formulation: the polar `$\partial_\theta Z/\rho$` is singular
/// there.
#[allow(
    clippy::extra_unused_type_parameters,
    clippy::int_plus_one,
    unused_comparisons,
    unused_assignments
)]
#[inline(always)]
pub fn zernike_basis_d_impl<P, E, V, const L: usize, const NORM: u8, const N: usize>(
    x: V,
    y: V,
    out: &mut [V; N],
    ddx: &mut [V; N],
    ddy: &mut [V; N],
) where
    P: Policy,
    E: FloatElement,
    V: FloatVector<Element = E> + CoreMath,
{
    const {
        assert!(N == (L + 1) * (L + 2) / 2, "zernike_basis_d: N must equal (L+1)(L+2)/2");
        assert!(
            NORM == ZERNIKE_UNIT_PEAK || NORM == ZERNIKE_ORTHONORMAL,
            "zernike_basis_d: NORM must be ZERNIKE_UNIT_PEAK or ZERNIKE_ORTHONORMAL"
        );
    }

    let s = x.mul_adde(x, y * y);

    // ds/dx and ds/dy, hoisted: every mode's radial half is scaled by these.
    let t = (x + x, y + y);

    if const { L > MAX_DEGREE } {
        rolled_d::<E, V, L, NORM, N>(s, t, x, y, out, ddx, ddy);
        return;
    }

    zernike_grad_columns!(L, E, V, NORM, s, t, x, y, out, ddx, ddy);
}

/// The gradient kernel with runtime coefficients and rolled loops, for `L > MAX_DEGREE`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn rolled_d<E, V, const L: usize, const NORM: u8, const N: usize>(
    s: V,
    t: (V, V),
    x: V,
    y: V,
    out: &mut [V; N],
    ddx: &mut [V; N],
    ddy: &mut [V; N],
) where
    E: FloatElement,
    V: FloatVector<Element = E> + CoreMath,
{
    let mut u = V::ONE;
    let mut v = V::ZERO;
    let mut up = V::ZERO;
    let mut vp = V::ZERO;

    let mut m = 0;
    while m <= L {
        emit_d::<E, V, N, NORM>(out, ddx, ddy, m, m, t, (V::ONE, V::ZERO), (u, v), (up, vp));

        if m + 2 <= L {
            let mf = m as thermite::LargeInt;

            let mut q_prev = V::ONE;
            let mut d_prev = V::ZERO;

            let slope = V::splat(E::from_int(mf + 2));

            let mut q_cur = s.mul_sube(slope, V::splat(E::from_int(mf + 1)));
            let mut d_cur = slope;

            emit_d::<E, V, N, NORM>(out, ddx, ddy, m + 2, m, t, (q_cur, d_cur), (u, v), (up, vp));

            let mut k = 2;
            while m + 2 * k <= L {
                let (ae, be, ce) = q_coeffs::<E>(k as thermite::LargeInt, mf);

                let a = V::splat(ae);
                let c = V::splat(ce);

                let lin = s.mul_adde(a, V::splat(be));

                let q_next = lin.mul_sube(q_cur, c * q_prev);
                let d_next = a.mul_adde(q_cur, lin.mul_sube(d_cur, c * d_prev));

                emit_d::<E, V, N, NORM>(out, ddx, ddy, m + 2 * k, m, t, (q_next, d_next), (u, v), (up, vp));

                q_prev = q_cur;
                q_cur = q_next;
                d_prev = d_cur;
                d_cur = d_next;

                k += 1;
            }
        }

        if m < L {
            let u_next = x.difference_of_products(u, y, v);
            let v_next = x.sum_of_products(v, y, u);

            up = u;
            vp = v;
            u = u_next;
            v = v_next;
        }

        m += 1;
    }
}
