//! `poly_primal`/`poly_rev_primal` against `poly`/`poly_rev` across the composite types.
//!
//! Each pair evaluates the same polynomial and differs only in where the coefficients
//! live. `poly` carries them in `Self`, storing and then adding the augmented fields
//! of a constant (which are zero). `poly_primal` carries them in `Self::Primal` and
//! adds only to the value part. Any disagreement means a composite's
//! `mul_add_primal` override is wrong.

use thermite::generic_array::GenericArray;
use thermite::generic_array::sequence::GenericSequence;
use thermite::generic_array::typenum::U6;
use thermite::math::{CoreMath, CoreMathWithPolicy};
use thermite::prelude::*;
use thermite_compensated::Compensated;
use thermite_complex::Complex;
use thermite_dual::Dual;

type V = Vector<f64>;

const COEFFS: [f64; 6] = [1.5, -0.25, 3.0, 0.75, -1.125, 2.0];

fn splat_primal<T: CoreMathWithPolicy>(c: [f64; 6]) -> GenericArray<T::Primal, U6>
where
    T::Primal: GenericVector<Element = f64>,
{
    GenericArray::generate(|i| T::Primal::splat(c[i]))
}

#[test]
fn poly_primal_matches_poly_on_real_vectors() {
    for x in [-2.0f64, -0.5, 0.0, 0.25, 1.0, 3.5] {
        let a = V::splat(x).poly(&COEFFS).extract::<0>();
        let b = V::splat(x).poly_primal(&splat_primal::<V>(COEFFS)).extract::<0>();
        assert_eq!(a, b, "real poly at {x}: poly {a}, poly_primal {b}");

        let a = V::splat(x).poly_rev(&COEFFS).extract::<0>();
        let b = V::splat(x).poly_rev_primal(&splat_primal::<V>(COEFFS)).extract::<0>();
        assert_eq!(a, b, "real poly_rev at {x}: poly_rev {a}, poly_rev_primal {b}");
    }
}

#[test]
fn poly_primal_matches_poly_on_dual() {
    type D = Dual<V, 2>;

    for x in [-2.0f64, -0.5, 0.25, 1.0, 3.5] {
        // Seeded as a variable so the derivative is exercised, not just the value.
        let d = D::variable(V::splat(x), 0);

        let lifted: [Dual<f64, 2>; 6] = core::array::from_fn(|i| Dual::constant(COEFFS[i]));
        let a = d.poly(&lifted);
        let b = d.poly_primal(&splat_primal::<D>(COEFFS));

        assert_eq!(a.value().extract::<0>(), b.value().extract::<0>(), "dual value at {x}");
        for j in 0..2 {
            assert_eq!(
                a.gradient()[j].extract::<0>(),
                b.gradient()[j].extract::<0>(),
                "dual d/d{j} at {x}"
            );
        }

        let a = d.poly_rev(&lifted);
        let b = d.poly_rev_primal(&splat_primal::<D>(COEFFS));

        assert_eq!(
            a.value().extract::<0>(),
            b.value().extract::<0>(),
            "dual rev value at {x}"
        );
        for j in 0..2 {
            assert_eq!(
                a.gradient()[j].extract::<0>(),
                b.gradient()[j].extract::<0>(),
                "dual rev d/d{j} at {x}"
            );
        }
    }
}

#[test]
fn poly_primal_matches_poly_on_complex() {
    type C = Complex<V>;

    for (re, im) in [(0.5f64, 0.25f64), (-1.5, 2.0), (0.0, -0.75), (3.0, 0.0)] {
        let z = C::new(V::splat(re), V::splat(im));

        let lifted: [Complex<f64>; 6] = core::array::from_fn(|i| Complex::new(COEFFS[i], 0.0));
        let a = z.poly(&lifted);
        let b = z.poly_primal(&splat_primal::<C>(COEFFS));

        assert_eq!(a.re.extract::<0>(), b.re.extract::<0>(), "complex re at ({re},{im})");
        assert_eq!(a.im.extract::<0>(), b.im.extract::<0>(), "complex im at ({re},{im})");

        let a = z.poly_rev(&lifted);
        let b = z.poly_rev_primal(&splat_primal::<C>(COEFFS));

        assert_eq!(
            a.re.extract::<0>(),
            b.re.extract::<0>(),
            "complex rev re at ({re},{im})"
        );
        assert_eq!(
            a.im.extract::<0>(),
            b.im.extract::<0>(),
            "complex rev im at ({re},{im})"
        );
    }
}

#[test]
fn poly_primal_matches_poly_on_compensated() {
    // `Compensated` is its own primal (the low word of a double-double constant is
    // precision, not augmentation), so its primal coefficients are themselves
    // `Compensated`, and `poly_primal` must reduce to `poly` exactly.
    type C = Compensated<V>;

    for x in [-2.0f64, 0.25, 1.0, 3.5] {
        let c = C::new(V::splat(x));

        let lifted: [Compensated<f64>; 6] = core::array::from_fn(|i| Compensated::new(COEFFS[i]));
        let splatted: GenericArray<C, U6> = GenericArray::generate(|i| C::new(V::splat(COEFFS[i])));

        let a = c.poly(&lifted).value().extract::<0>();
        let b = c.poly_primal(&splatted).value().extract::<0>();

        assert_eq!(a, b, "compensated poly at {x}: poly {a}, poly_primal {b}");

        let a = c.poly_rev(&lifted).value().extract::<0>();
        let b = c.poly_rev_primal(&splatted).value().extract::<0>();

        assert_eq!(a, b, "compensated poly_rev at {x}: poly_rev {a}, poly_rev_primal {b}");
    }
}

/// Real vectors must agree with `poly` under _every_ policy, not just the default.
///
/// The two lowerings diverge on policy: `unroll_loops` with less-than-`Best` precision
/// takes the Estrin/ILP path, anything else takes Horner. `poly_primal` overrides the
/// real-vector case specifically to keep that split rather than inherit the
/// primal-Horner default, so both arms need checking.
#[test]
fn poly_primal_matches_poly_under_each_policy() {
    use thermite::math::policy::policies::{Performance, Precision, Reference, UltraPerformance};

    macro_rules! check {
        ($($policy:ty),*) => {$(
            for x in [-2.0f64, -0.5, 0.0, 0.25, 1.0, 3.5] {
                let v = V::splat(x);
                let a = v.poly_p::<$policy, 6>(&COEFFS).extract::<0>();
                let b = v.poly_primal_p::<$policy, U6>(&splat_primal::<V>(COEFFS)).extract::<0>();
                assert!(
                    (a - b).abs() <= 1e-14 * (1.0 + a.abs()),
                    "{} at {x}: poly {a}, poly_primal {b}",
                    core::any::type_name::<$policy>()
                );

                let a = v.poly_rev_p::<$policy, 6>(&COEFFS).extract::<0>();
                let b = v.poly_rev_primal_p::<$policy, U6>(&splat_primal::<V>(COEFFS)).extract::<0>();
                assert!(
                    (a - b).abs() <= 1e-14 * (1.0 + a.abs()),
                    "{} rev at {x}: poly_rev {a}, poly_rev_primal {b}",
                    core::any::type_name::<$policy>()
                );
            }
        )*};
    }

    check!(UltraPerformance, Performance, Precision, Reference);
}
