//! `difference_of_products` / `sum_of_products`: the exactness property, the
//! compensation, and the tier boundary.
//!
//! The property being pinned is not "accurate". It is **exact when the two products
//! are equal**: `a*b - b*a` must be zero, not small. Cross products, perp-dots, 2x2
//! determinants and every sign test built on one of them depend on that, and a test
//! that samples random inputs and checks a relative error will not notice when it
//! breaks. Hence the explicit `f(v, v) == 0` cases below.
//!
//! It breaks for exactly one of the three lowerings. Naive `a*b - c*d` has the
//! property (both products round identically, then cancel). The compensated form has
//! it (the recovered residual cancels). The _one-sided_ fused form
//! `a.mul_sube(b, c * d)` does not: it leaves `a*b` exact and rounds `c*d`, so the
//! difference is the discarded rounding rather than zero. It does that only where
//! hardware FMA exists, which is why the same source gave a zero self-cross-product
//! on baseline x86 and a denormal on AArch64. The one-sided form is confined to
//! `Medium` and below. `Average`, the default tier, and up compensate.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]

use thermite::math::policy::policies::{HighPerformance, Performance, Precision};
use thermite::math::{CoreMath, CoreMathWithPolicy};
use thermite::prelude::*;
use thermite::vector::ops::MulAddExt;

type D = Vector<f64>;
type F = Vector<f32>;

/// Whether the scalar seed fuses. False on baseline x86, true on AArch64 and under
/// `RUSTFLAGS="-C target-feature=+fma"`. Both arms are tested wherever the property
/// holds unconditionally. This only gates the claims that differ by lowering.
const SEED_FUSES: bool = <D as MulAddExt>::HAS_TRUE_FMA;

/// Operand pairs whose product is not exactly representable, so a lowering that
/// rounds one side and not the other cannot accidentally pass. The last pair sits
/// near the top of the f64 range, where a compensated form that scales rather than
/// rebalances would overflow.
const PAIRS: &[(f64, f64)] = &[
    (0.1, 0.3),
    (1.0 / 3.0, 7.0 / 11.0),
    (1e-8, 3.7e12),
    (9.007199254740993e15, 1.0000000000000002),
    (-2.718281828459045, 3.141592653589793),
    (1.7976931348623157e150, 1.1102230246251565e-16),
];

/// The f32 twin. Kept separate rather than cast down from [`PAIRS`]: the wide f64
/// entries saturate to infinity in f32, and `inf - inf` is a correct NaN that has
/// nothing to say about the lowering under test.
const PAIRS32: &[(f32, f32)] = &[
    (0.1, 0.3),
    (1.0 / 3.0, 7.0 / 11.0),
    (1e-8, 3.7e12),
    (16777217.0, 1.0000001),
    (-2.7182817, 3.1415927),
    (1.9721523e30, 5.9604645e-8),
];

fn ulp(x: f64) -> f64 {
    let a = x.abs();
    if a == 0.0 { f64::MIN_POSITIVE } else { a.next_up() - a }
}

/// `a*b - b*a` is zero, at every tier that compensates.
///
/// Both surviving arms give exact zero here (naive because the two products round
/// identically, compensated because the residual it adds back is the one it just
/// subtracted), so this holds with or without hardware FMA and needs no gate.
#[test]
fn equal_products_cancel_exactly() {
    for &(a, b) in PAIRS {
        let got = D::splat(a)
            .difference_of_products(D::splat(b), D::splat(b), D::splat(a))
            .extract::<0>();
        assert_eq!(got, 0.0, "f64 default policy: {a:e} * {b:e} - {b:e} * {a:e}");

        let got = D::splat(a)
            .difference_of_products_p::<Precision>(D::splat(b), D::splat(b), D::splat(a))
            .extract::<0>();
        assert_eq!(got, 0.0, "f64 Precision: {a:e} * {b:e} - {b:e} * {a:e}");
    }

    for &(a, b) in PAIRS32 {
        let got = F::splat(a)
            .difference_of_products(F::splat(b), F::splat(b), F::splat(a))
            .extract::<0>();
        assert_eq!(got, 0.0, "f32 default policy: {a:e} * {b:e} - {b:e} * {a:e}");
    }
}

/// The `sum_of_products` twin: `a*b + (-b)*a` is zero. The compensation enters with
/// the opposite sign there, and getting that backwards is the easy mistake.
#[test]
fn sum_of_products_cancels_exactly() {
    for &(a, b) in PAIRS {
        let got = D::splat(a)
            .sum_of_products(D::splat(b), D::splat(-b), D::splat(a))
            .extract::<0>();
        assert_eq!(got, 0.0, "f64 default policy: {a:e} * {b:e} + -{b:e} * {a:e}");

        let got = D::splat(a)
            .sum_of_products_p::<Precision>(D::splat(b), D::splat(-b), D::splat(a))
            .extract::<0>();
        assert_eq!(got, 0.0, "f64 Precision: {a:e} * {b:e} + -{b:e} * {a:e}");
    }
}

/// The same property at the native register width.
///
/// The scalar seed and the SIMD backends reach the FMA gate by different routes, and
/// on this workspace's usual x86 hosts they disagree about it: the seed reads
/// baseline target features (no FMA) while the AVX2 backend has it. So this is not a
/// duplicate of the test above. It usually exercises the _other_ arm.
#[test]
fn equal_products_cancel_exactly_at_native_width() {
    for &(a, b) in PAIRS {
        let got = thermite::dispatch_dyn!(for<S> |a: f64, b: f64| -> f64 {
            f64xN::splat(a)
                .difference_of_products(f64xN::splat(b), f64xN::splat(b), f64xN::splat(a))
                .extract::<0>()
        });
        assert_eq!(got, 0.0, "f64xN: {a:e} * {b:e} - {b:e} * {a:e}");
    }

    for &(a, b) in PAIRS32 {
        let got = thermite::dispatch_dyn!(for<S> |a: f32, b: f32| -> f32 {
            f32xN::splat(a)
                .difference_of_products(f32xN::splat(b), f32xN::splat(b), f32xN::splat(a))
                .extract::<0>()
        });
        assert_eq!(got, 0.0, "f32xN: {a:e} * {b:e} - {b:e} * {a:e}");
    }
}

/// Kahan's cancelling case: the compensated form lands within an ulp of the exact
/// value where the uncompensated ones are off by ~1e7 ulp.
///
/// `a*b` and `c*d` agree to 9 significant figures, so all but the last few bits of
/// each product cancel and whatever error the products carry is all that survives.
/// Exact value computed with `fractions.Fraction`. The compensated form's measured
/// relative error is `1.6519e-16`, or 0.74 ulp.
#[test]
fn compensation_recovers_cancellation() {
    const A: f64 = 33962.035;
    const B: f64 = -30438.8;
    const C: f64 = 41563.4;
    const DD: f64 = -24871.969;
    const EXACT: f64 = 5.376599994516417;

    let dop = |x: D| x.extract::<0>();

    let compensated = dop(D::splat(A).difference_of_products_p::<Performance>(D::splat(B), D::splat(C), D::splat(DD)));
    let cheap = dop(D::splat(A).difference_of_products_p::<HighPerformance>(D::splat(B), D::splat(C), D::splat(DD)));

    if SEED_FUSES {
        assert!(
            (compensated - EXACT).abs() <= 2.0 * ulp(EXACT),
            "compensated: got {compensated:?}, want {EXACT:?} ({} ulp)",
            (compensated - EXACT).abs() / ulp(EXACT)
        );
        // And the cheap tier really is the cheap tier. If this stops being much
        // worse, the tier boundary has silently moved.
        assert!(
            (cheap - EXACT).abs() > 1000.0 * ulp(EXACT),
            "cheap tier unexpectedly accurate: got {cheap:?}, want {EXACT:?}"
        );
    } else {
        // No hardware FMA: every tier is the naive form, which has the exactness
        // property but not the accuracy. Pinned so the arm is not silently swapped
        // for an emulated FMA, which is against house rules and would show up here.
        assert_eq!(compensated, cheap, "no-FMA host: tiers must agree");
        assert!(
            (compensated - EXACT).abs() > 1000.0 * ulp(EXACT),
            "no-FMA host produced a compensated result: got {compensated:?}"
        );
    }
}

/// Below `Average` the one-sided form is deliberate, and on a fusing host it does
/// _not_ give exact zero. Pinned as a positive statement so the tier boundary is
/// visible in the test suite rather than only in the docs.
#[test]
fn low_tier_takes_the_one_sided_form() {
    let (a, b) = (0.1f64, 0.3f64);

    let got = D::splat(a)
        .difference_of_products_p::<HighPerformance>(D::splat(b), D::splat(b), D::splat(a))
        .extract::<0>();

    if SEED_FUSES {
        assert_ne!(got, 0.0, "one-sided form should leave the discarded rounding behind");
        assert!(
            got.abs() <= ulp(a * b),
            "one-sided residual should be within an ulp of the product, got {got:e}"
        );
    } else {
        assert_eq!(got, 0.0, "no-FMA host: the naive form is exact here");
    }
}

/// The motivating consumer: a cross product of a vector with itself is the zero
/// vector, and a 2x2 determinant of a matrix with two equal rows is zero.
///
/// Spelled out longhand rather than through `thermite-geometry` so this test does not
/// depend on that crate having been migrated yet.
#[test]
fn self_cross_product_is_zero() {
    let v = [0.1f64, 0.3, -7.0 / 11.0];

    for (i, j, k) in [(0usize, 1usize, 2usize), (1, 2, 0), (2, 0, 1)] {
        let component = D::splat(v[j])
            .difference_of_products(D::splat(v[k]), D::splat(v[k]), D::splat(v[j]))
            .extract::<0>();
        assert_eq!(component, 0.0, "cross(v, v) component {i} was {component:e}");
    }

    let (a, b) = (0.1f64, 0.3f64);
    let det = D::splat(a)
        .difference_of_products(D::splat(b), D::splat(a), D::splat(b))
        .extract::<0>();
    assert_eq!(det, 0.0, "det([[a, b], [a, b]]) was {det:e}");
}
