//! The slice-taking reductions against their const-length originals.
//!
//! `hypot_s`, `inv_hypot`, `harmonic_mean`, `inv_sum_inv` and `logsumexp` are the
//! runtime-length forms of `*_n`. The const form is the tested one, so the useful
//! assertion is agreement between the two rather than a fresh reference table: any
//! disagreement is a defect in the loop rewrite, which is the only new machinery here.
//!
//! These types take the `specialized::ps`/`pd` overrides, which route to the
//! `generic::*_slice_internal` bodies. The COMPOSITE defaults are a separate
//! implementation and are covered in `thermite-compensated/tests/slice_reductions.rs`.

use thermite::prelude::*;

type D = Vector<f64>;
type F = Vector<f32>;

fn v(x: f64) -> D {
    D::splat(x)
}

/// Agreement is exact for lengths at or below one chunk: the fold runs the const kernel
/// once, on the same values, with the tail padded by the identity.
#[test]
fn single_chunk_is_exact() {
    let arr = [v(1.0), v(2.0), v(4.0), v(8.0), v(16.0)];

    assert_eq!(
        D::hypot_s(&arr).extract::<0>(),
        D::hypot_n(arr).extract::<0>(),
        "hypot_s disagrees with hypot_n within one chunk"
    );

    assert_eq!(
        D::inv_hypot(&arr).extract::<0>(),
        D::inv_hypot_n(arr).extract::<0>(),
        "inv_hypot disagrees with inv_hypot_n within one chunk"
    );

    assert_eq!(
        D::logsumexp(&arr).extract::<0>(),
        D::logsumexp_n(arr).extract::<0>(),
        "logsumexp disagrees with logsumexp_n within one chunk"
    );
}

/// Past one chunk the fold is a genuine partition, so agreement is to rounding rather
/// than exact.
#[test]
fn multiple_chunks_agree() {
    let xs: [f64; 21] = core::array::from_fn(|i| 0.5 + (i as f64) * 1.375);
    let arr: [D; 21] = xs.map(v);

    let close = |a: f64, b: f64, tol: f64, what: &str| {
        assert!(
            (a - b).abs() <= tol * b.abs().max(1.0),
            "{what}: slice {a} vs const {b}"
        );
    };

    close(
        D::hypot_s(&arr).extract::<0>(),
        D::hypot_n(arr).extract::<0>(),
        1e-15,
        "hypot_s",
    );

    close(
        D::inv_hypot(&arr).extract::<0>(),
        D::inv_hypot_n(arr).extract::<0>(),
        1e-15,
        "inv_hypot",
    );

    close(
        D::logsumexp(&arr).extract::<0>(),
        D::logsumexp_n(arr).extract::<0>(),
        1e-15,
        "logsumexp",
    );

    close(
        D::harmonic_mean(&arr).extract::<0>(),
        D::harmonic_mean_n(arr).extract::<0>(),
        1e-15,
        "harmonic_mean",
    );

    close(
        D::inv_sum_inv(&arr).extract::<0>(),
        D::inv_sum_inv_n(arr).extract::<0>(),
        1e-15,
        "inv_sum_inv",
    );
}

/// The identity used as tail padding has to be a true identity, or every length that is
/// not a multiple of the chunk is wrong. This is the assertion that would have caught it.
#[test]
fn tail_padding_does_not_contribute() {
    for n in 1..=17usize {
        let arr: Vec<D> = (0..n).map(|i| v(1.0 + i as f64)).collect();

        let want_norm: f64 = (1..=n).map(|i| (i as f64) * (i as f64)).sum::<f64>().sqrt();
        let got_norm = D::hypot_s(&arr).extract::<0>();
        assert!(
            (got_norm - want_norm).abs() <= 1e-14 * want_norm,
            "hypot_s over {n} values: {got_norm} want {want_norm}"
        );

        let want_isi = 1.0 / (1..=n).map(|i| 1.0 / i as f64).sum::<f64>();
        let got_isi = D::inv_sum_inv(&arr).extract::<0>();
        assert!(
            (got_isi - want_isi).abs() <= 1e-14 * want_isi,
            "inv_sum_inv over {n} values: {got_isi} want {want_isi}"
        );

        let want_hm = want_isi * n as f64;
        let got_hm = D::harmonic_mean(&arr).extract::<0>();
        assert!(
            (got_hm - want_hm).abs() <= 1e-14 * want_hm,
            "harmonic_mean over {n} values: {got_hm} want {want_hm}"
        );
    }
}

/// The empty slice is the identity of each reduction, which is what makes folding over a
/// partition well defined in the first place.
#[test]
fn empty_slice_is_the_identity() {
    let none: [D; 0] = [];

    assert_eq!(D::hypot_s(&none).extract::<0>(), 0.0);
    assert_eq!(D::inv_hypot(&none).extract::<0>(), f64::INFINITY);
    assert_eq!(D::logsumexp(&none).extract::<0>(), f64::NEG_INFINITY);
    assert_eq!(D::inv_sum_inv(&none).extract::<0>(), f64::INFINITY);
    assert!(D::harmonic_mean(&none).extract::<0>().is_nan());
}

/// The range safety is the whole reason `hypot` is not `sqrt(sum of squares)`, and the
/// slice form has to keep it. Both of these overflow or underflow if evaluated directly.
#[test]
fn slice_hypot_keeps_its_range() {
    let big = [v(1e300), v(2e300), v(3e300)];
    let got = D::hypot_s(&big).extract::<0>();
    assert!(got.is_finite(), "hypot_s overflowed: {got}");
    assert!((got - 3.7416573867739413e300).abs() <= 1e-14 * got);

    let small = [v(1e-300), v(2e-300), v(3e-300)];
    let got = D::hypot_s(&small).extract::<0>();
    assert!(got > 0.0, "hypot_s underflowed: {got}");
    assert!((got - 3.7416573867739413e-300).abs() <= 1e-14 * got);

    // Straddling a chunk boundary, so the partial norms themselves span the range.
    let wide: [D; 12] = core::array::from_fn(|i| if i < 6 { v(1e-290) } else { v(1e290) });
    let got = D::hypot_s(&wide).extract::<0>();
    assert!(got.is_finite() && got > 0.0, "hypot_s over a wide spread: {got}");
    assert!((got - 2.449489742783178e290).abs() <= 1e-14 * got);
}

/// `logsumexp` over a slice must not overflow either, for the same reason.
#[test]
fn slice_logsumexp_keeps_its_range() {
    let arr: [D; 11] = core::array::from_fn(|i| v(1000.0 + i as f64));
    let got = D::logsumexp(&arr).extract::<0>();
    assert!(got.is_finite(), "logsumexp overflowed: {got}");

    // ln(sum e^(1000+i)) = 1010 + ln(sum e^(i-10))
    let want = 1010.0 + (0..11).map(|i| ((i as f64) - 10.0).exp()).sum::<f64>().ln();
    assert!((got - want).abs() <= 1e-12, "logsumexp: {got} want {want}");
}

/// float32 as well, since the two element types take different specialized bodies.
#[test]
fn float32_agrees() {
    let arr: [F; 13] = core::array::from_fn(|i| F::splat(0.25 + i as f32));

    let a = F::hypot_s(&arr).extract::<0>();
    let b = F::hypot_n(arr).extract::<0>();
    assert!((a - b).abs() <= 1e-6 * b, "f32 hypot_s: {a} vs {b}");

    let a = F::logsumexp(&arr).extract::<0>();
    let b = F::logsumexp_n(arr).extract::<0>();
    assert!((a - b).abs() <= 1e-5 * b.abs(), "f32 logsumexp: {a} vs {b}");
}

/// `poly_primal` and `poly_rev_primal`: the slice twins of the `GenericArray` forms.
///
/// A real vector is its own primal, so the coefficients arrive pre-splatted and the check
/// is against `poly`/`poly_rev` over the same values as elements.
#[test]
fn primal_poly_slices_agree() {
    let coeffs = [1.5, -0.75, 0.25, 2.0, -1.25, 0.5, 0.125];
    let splatted: [D; 7] = coeffs.map(v);

    for &x in &[-2.0, -0.5, 0.0, 0.75, 3.0] {
        let a = D::splat(x);

        assert_eq!(
            a.poly_primal(&splatted).extract::<0>(),
            a.poly(&coeffs).extract::<0>(),
            "poly_primal at x={x}"
        );

        assert_eq!(
            a.poly_rev_primal(&splatted).extract::<0>(),
            a.poly_rev(&coeffs).extract::<0>(),
            "poly_rev_primal at x={x}"
        );
    }

    let none: [D; 0] = [];
    assert_eq!(D::splat(2.0).poly_primal(&none).extract::<0>(), 0.0);
    assert_eq!(D::splat(2.0).poly_rev_primal(&none).extract::<0>(), 0.0);
}
