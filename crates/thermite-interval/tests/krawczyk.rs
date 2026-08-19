//! Krawczyk root certification (`dual` feature): the verdicts are proofs, so
//! the tests check soundness first (never certify what is not there) and
//! completeness second (find and prove every root that is).

#![cfg(feature = "dual")]

use thermite::math::TranscendentalMath;
use thermite::prelude::*;
use thermite_interval::verify::{DualInterval, certify_roots, krawczyk_step};
use thermite_interval::{Balanced, Interval, Tightest, WideningPolicy};

type V1 = Vector<f64>;
type I<W> = Interval<V1, W>;
type D<W> = DualInterval<V1, W>;

fn c<W: WideningPolicy>(v: f64) -> D<W> {
    thermite_dual::Dual::constant(Interval::degenerate(V1::splat(v)))
}

fn f_sin<W: WideningPolicy>(x: D<W>) -> D<W> {
    x.sin() - x * c(1.0 / 3.0)
}

fn g_exp<W: WideningPolicy>(x: D<W>) -> D<W> {
    x.exp() - x * c(3.0)
}

/// (x - 1)(x - 2)(x - 3): three exact integer roots.
fn cubic<W: WideningPolicy>(x: D<W>) -> D<W> {
    (x - c(1.0)) * (x - c(2.0)) * (x - c(3.0))
}

/// (x - 1)^2: a double root, so Krawczyk must NOT certify uniqueness.
fn double_root<W: WideningPolicy>(x: D<W>) -> D<W> {
    let d = x - c(1.0);
    d * d
}

/// x^2 + 1: no real roots.
fn rootless<W: WideningPolicy>(x: D<W>) -> D<W> {
    x * x + c(1.0)
}

fn contains(lo: f64, hi: f64, v: f64) -> bool {
    lo <= v && v <= hi
}

#[test]
fn one_step_verdicts() {
    let x: I<Tightest> = Interval::bounds(V1::splat(2.0), V1::splat(2.5));
    let (k, v) = krawczyk_step(f_sin::<Tightest>, x);
    assert!(v.unique.all(), "[2, 2.5] holds exactly one root of sin(x) - x/3");
    assert!(!v.excluded.any());
    // K(X) is strictly inside X.
    assert!(k.lo().extract::<0>() > 2.0 && k.hi().extract::<0>() < 2.5);
    assert!(contains(
        k.lo().extract::<0>(),
        k.hi().extract::<0>(),
        2.278862660075816
    ));

    let x: I<Tightest> = Interval::bounds(V1::splat(0.5), V1::splat(1.5));
    let (_, v) = krawczyk_step(f_sin::<Tightest>, x);
    assert!(v.excluded.all(), "[0.5, 1.5] is root-free");
    assert!(!v.unique.any());

    // Empty in, empty out (and reported excluded), no NaN leakage.
    let e: I<Tightest> = Interval::empty();
    let (k, v) = krawczyk_step(f_sin::<Tightest>, e);
    assert!(k.is_empty().all() && v.excluded.all() && !v.unique.any());
}

fn check_sin<W: WideningPolicy>() {
    let cert = certify_roots::<V1, W, _>(f_sin::<W>, -4.0, 4.0, 1e-9, 10_000);
    assert_eq!(
        cert.roots.len(),
        3,
        "sin(x) - x/3 has 3 roots on [-4, 4]: {:?}",
        cert.roots
    );
    assert!(cert.unresolved.is_empty(), "nothing unresolved: {:?}", cert.unresolved);

    let mut roots = cert.roots.clone();
    roots.sort_by(|p, q| p.lo.partial_cmp(&q.lo).unwrap());
    let truth = [-2.2788626600758283, 0.0, 2.2788626600758283];
    for (r, t) in roots.iter().zip(truth) {
        assert!(contains(r.lo, r.hi, t), "[{}, {}] must contain {t}", r.lo, r.hi);
        assert!(r.hi - r.lo < 1e-9, "refined below tol: {}", r.hi - r.lo);
    }
}

#[test]
fn certifies_all_roots_tightest() {
    check_sin::<Tightest>();
}

#[test]
fn certifies_all_roots_balanced() {
    check_sin::<Balanced>();
}

#[test]
fn certifies_exp_roots() {
    let cert = certify_roots::<V1, Tightest, _>(g_exp::<Tightest>, -2.0, 4.0, 1e-9, 10_000);
    assert_eq!(cert.roots.len(), 2, "{:?}", cert.roots);
    assert!(cert.unresolved.is_empty());
    let mut roots = cert.roots.clone();
    roots.sort_by(|p, q| p.lo.partial_cmp(&q.lo).unwrap());
    assert!(contains(roots[0].lo, roots[0].hi, 0.6190612867359450));
    assert!(contains(roots[1].lo, roots[1].hi, 1.5121345516578424));
}

#[test]
fn certifies_integer_roots_of_cubic() {
    let cert = certify_roots::<V1, Tightest, _>(cubic::<Tightest>, 0.0, 4.0, 1e-9, 10_000);
    assert_eq!(cert.roots.len(), 3, "{:?}", cert.roots);
    assert!(cert.unresolved.is_empty(), "{:?}", cert.unresolved);
    let mut roots = cert.roots.clone();
    roots.sort_by(|p, q| p.lo.partial_cmp(&q.lo).unwrap());
    for (r, t) in roots.iter().zip([1.0, 2.0, 3.0]) {
        assert!(contains(r.lo, r.hi, t), "[{}, {}] must contain {t}", r.lo, r.hi);
    }
}

/// SOUNDNESS: a double root is never certified as unique. It must surface as
/// unresolved (the honest answer), and every unresolved box must contain it.
#[test]
fn double_root_is_never_certified() {
    let cert = certify_roots::<V1, Tightest, _>(double_root::<Tightest>, 0.0, 2.0, 1e-6, 10_000);
    assert!(
        cert.roots.is_empty(),
        "a double root cannot be certified unique: {:?}",
        cert.roots
    );
    assert!(!cert.unresolved.is_empty(), "the tangency must be reported");
    // Exclusion is a proof of absence, so the root can never be lost: some
    // unresolved box contains it, and all of them cluster at the tangency
    // (near a double root f is tiny, so nearby boxes cannot be excluded).
    assert!(
        cert.unresolved.iter().any(|u| contains(u.lo, u.hi, 1.0)),
        "the root box must survive: {:?}",
        cert.unresolved
    );
    for u in &cert.unresolved {
        assert!(
            (u.lo - 1.0).abs() < 1e-2 && (u.hi - 1.0).abs() < 1e-2,
            "unresolved box [{}, {}] far from the root",
            u.lo,
            u.hi
        );
    }
}

/// SOUNDNESS: a root-free function is entirely excluded, nothing certified
/// and nothing unresolved.
#[test]
fn rootless_is_fully_excluded() {
    let cert = certify_roots::<V1, Tightest, _>(rootless::<Tightest>, -5.0, 5.0, 1e-9, 10_000);
    assert!(cert.roots.is_empty(), "{:?}", cert.roots);
    assert!(cert.unresolved.is_empty(), "{:?}", cert.unresolved);
    assert!(cert.boxes < 64, "exclusion should be fast: {} boxes", cert.boxes);
}

/// Refinement after a certificate: iterating Krawczyk on a certified box
/// contracts it toward the root without ever excluding it, and the root stays
/// inside every iterate. (Strict interiority, the certificate itself, need
/// not hold once the box reaches the enclosure floor, where K(X) ~ X. The
/// proof was earned on the wide box and containment carries it down.)
#[test]
fn refinement_preserves_containment() {
    let mut x: I<Tightest> = Interval::bounds(V1::splat(2.0), V1::splat(2.5));
    let (_, v) = krawczyk_step(f_sin::<Tightest>, x);
    assert!(v.unique.all(), "start from a certified box");

    let mut last_w = f64::INFINITY;
    for _ in 0..12 {
        let (k, v) = krawczyk_step(f_sin::<Tightest>, x);
        assert!(!v.excluded.any(), "a certified box can never be excluded on refinement");
        let w = k.width().extract::<0>();
        assert!(w <= last_w, "width must not grow: {w} > {last_w}");
        assert!(contains(
            k.lo().extract::<0>(),
            k.hi().extract::<0>(),
            2.2788626600758283
        ));
        last_w = w;
        x = k;
    }
    assert!(last_w < 1e-12, "converged: width {last_w}");
}
