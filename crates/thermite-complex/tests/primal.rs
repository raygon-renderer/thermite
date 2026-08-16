//! Compile-time checks of the `Primal` associated type projections.
//!
//! `Primal` (on `PrimalProjection`) is the unaugmented value type: `Self` for
//! plain vectors, the inner value vector's primal (recursively) for composites.
//! `Compensated` is deliberately its own primal (the error half of a
//! double-double constant is information, not augmentation), so it acts as a
//! fixpoint of the recursion.
//!
//! Everything here is checked by the compiler. The test body only has to build.

use thermite::math::PrimalProjection;
use thermite::prelude::*;
use thermite_complex::Complex;

fn assert_primal<V: PrimalProjection<Primal = P>, P>() {}

/// Compiles only because `PrimalMath` carries `PrimalProjection<Primal = Self>`:
/// every primal type is its own primal, as a law rather than a convention.
fn assert_own_primal<P: thermite::math::PrimalMath>() {
    assert_primal::<P, P>();
}

#[test]
fn primal_projections() {
    // Plain vectors are their own primal.
    assert_primal::<Vector<f32>, Vector<f32>>();
    assert_primal::<Vector<f64>, Vector<f64>>();
    assert_own_primal::<Vector<f64>>();
    #[cfg(feature = "compensated")]
    assert_own_primal::<thermite_compensated::Compensated<Vector<f64>>>();

    // A real coefficient's imaginary part is identically zero: Complex strips
    // down to the real vector.
    assert_primal::<Complex<Vector<f64>>, Vector<f64>>();

    #[cfg(feature = "dual")]
    {
        use thermite_dual::Dual;

        // A constant's derivative parts are identically zero: Dual strips down
        // to the value vector.
        assert_primal::<Dual<Vector<f64>, 2>, Vector<f64>>();

        // The recursion composes: both augmentations strip in one projection.
        assert_primal::<Complex<Dual<Vector<f64>, 1>>, Vector<f64>>();
    }

    #[cfg(feature = "compensated")]
    {
        use thermite_compensated::Compensated;

        // Compensated is the fixpoint: double-double tables keep their error half.
        assert_primal::<Compensated<Vector<f64>>, Compensated<Vector<f64>>>();

        // Composites over Compensated stop stripping at the double-double.
        assert_primal::<Complex<Compensated<Vector<f64>>>, Compensated<Vector<f64>>>();

        #[cfg(feature = "dual")]
        assert_primal::<thermite_dual::Dual<Compensated<Vector<f64>>, 3>, Compensated<Vector<f64>>>();
    }
}

/// `from_primal` zeroes the non-primal fields, and `to_primal` discards them.
#[test]
fn primal_round_trips() {
    let p = Vector::<f64>::splat(3.5);

    let z = Complex::<Vector<f64>>::from_primal(p);
    assert_eq!(z.re.extract::<0>(), 3.5);
    assert_eq!(z.im.extract::<0>(), 0.0);
    assert_eq!(z.to_primal().extract::<0>(), 3.5);

    #[cfg(feature = "dual")]
    {
        use thermite_dual::Dual;

        let d = Dual::<Vector<f64>, 2>::from_primal(p);
        assert_eq!(d.re.extract::<0>(), 3.5);
        assert_eq!(d.dual[0].extract::<0>(), 0.0);
        assert_eq!(d.dual[1].extract::<0>(), 0.0);
        assert_eq!(d.to_primal().extract::<0>(), 3.5);

        // Both augmentations strip through one projection.
        let zd = Complex::<Dual<Vector<f64>, 1>>::from_primal(p);
        assert_eq!(zd.re.re.extract::<0>(), 3.5);
        assert_eq!(zd.re.dual[0].extract::<0>(), 0.0);
        assert_eq!(zd.im.re.extract::<0>(), 0.0);
        assert_eq!(zd.to_primal().extract::<0>(), 3.5);
    }

    #[cfg(feature = "compensated")]
    {
        use thermite_compensated::Compensated;

        // The fixpoint round-trips the whole double-double, error half included.
        let tau = Compensated::<Vector<f64>>::TAU;
        let c = Compensated::<Vector<f64>>::from_primal(tau);
        assert_eq!(c.value().extract::<0>(), tau.value().extract::<0>());
        assert_eq!(c.error().extract::<0>(), tau.error().extract::<0>());
        assert_ne!(c.error().extract::<0>(), 0.0);

        // A composite over Compensated embeds it losslessly too.
        let zc = Complex::<Compensated<Vector<f64>>>::from_primal(tau);
        assert_eq!(zc.re.error().extract::<0>(), tau.error().extract::<0>());
        assert_eq!(zc.to_primal().error().extract::<0>(), tau.error().extract::<0>());
    }
}
