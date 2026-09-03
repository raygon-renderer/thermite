//! What the marker layer promises on `Complex` beyond what `bessel.rs` grades: `Scaled<J>`
//! is the complex `jve`, not the real-line forward, and the Hankel markers select the slots
//! of the four-output kernel they name.

#![cfg(feature = "special")]

use thermite::math::policy::policies::Precision;
use thermite::prelude::*;
use thermite_complex::Complex;
use thermite_complex::math::special::{ComplexSpecialMathWithPolicy, H1, H2};
use thermite_special::bessel::{J, Scaled, Y};
use thermite_special::{BesselOrder, SpecialMathWithPolicy};

type V = Vector<f64>;
type C = Complex<V>;

fn c(re: f64, im: f64) -> C {
    Complex::new(V::splat(re), V::splat(im))
}

fn bits(z: C) -> (u64, u64) {
    (z.re.extract::<0>().to_bits(), z.im.extract::<0>().to_bits())
}

#[test]
fn complex_scaled_j_is_not_the_unscaled_value() {
    // Off the axis the scale factor is e^{-|Im z|}, so the two spellings must differ. The
    // scaled one must be what the Hankel combination gives: 2 jve = e^{-|Im z|} (H1 + H2).
    let z = c(1.0, 3.0);
    let order = BesselOrder::Real(c(0.5, 0.0));
    let orderv = BesselOrder::Real(V::splat(0.5));
    let plain = z.bessel_p::<Precision, J>(order);
    let scaled = z.bessel_p::<Precision, Scaled<J>>(order);
    assert_ne!(bits(plain), bits(scaled));

    let h1 = z.hankel_p::<Precision, H1>(orderv);
    let h2 = z.hankel_p::<Precision, H2>(orderv);
    let two_j = h1 + h2;
    let rel = ((two_j - plain - plain).norm_sqr() / (plain + plain).norm_sqr())
        .extract::<0>()
        .sqrt();
    assert!(rel < 1e-12, "H1 + H2 vs 2J: {rel:e}");
}

#[test]
fn scaled_hankel_markers_carry_the_scaling() {
    let z = c(2.0, 1.0);
    let orderv = BesselOrder::Real(V::splat(1.0 / 3.0));
    for (plain, scaled) in [
        (
            z.hankel_p::<Precision, H1>(orderv),
            z.hankel_p::<Precision, Scaled<H1>>(orderv),
        ),
        (
            z.hankel_p::<Precision, H2>(orderv),
            z.hankel_p::<Precision, Scaled<H2>>(orderv),
        ),
    ] {
        assert_ne!(bits(plain), bits(scaled));
    }
    // Y through the same kernel, for symmetry with J above.
    let order = BesselOrder::Real(c(1.0 / 3.0, 0.0));
    assert_ne!(
        bits(z.bessel_p::<Precision, Y>(order)),
        bits(z.bessel_p::<Precision, Scaled<Y>>(order))
    );
}
