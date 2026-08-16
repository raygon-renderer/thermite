//! The full composite tower: `Complex<Dual<Compensated<V>, 1>>`.
//!
//! Each layer contributes one thing, and none of them know about the others:
//!
//! - `Compensated<V>` carries every value as a double-double (~32 significant
//!   digits),
//! - `Dual<_, 1>` carries one derivative direction through the chain rule,
//! - `Complex<_>` makes the whole thing a point in C.
//!
//! There is no `Dual<Complex<..>>` and none is needed: the tensor product
//! commutes (`C (x) R[eps]` = `R[eps] (x) C`), so `Complex<Dual<..>>` already
//! _is_ the algebra of "dual complex numbers". Structurally the complex layer
//! must be outermost anyway, since `Dual`'s chain rule runs on real-valued math
//! (`RealMathWithPolicy`), which `Complex` deliberately does not implement.
//!
//! The demo differentiates a holomorphic function. Seed `d/dx` on the real
//! component only. For holomorphic `f`, the Cauchy-Riemann equations give the
//! complex derivative from that single real sweep:
//!
//! ```text
//! f'(z) = du/dx + i dv/dx
//! ```
//!
//! We evaluate `f(z) = z * e^z` and compare the AD derivative against the
//! analytic `f'(z) = (1 + z) e^z` computed on a plain `Complex<Compensated<V>>`
//! (no dual anywhere, so the two paths share no derivative code). The residual
//! lands around `1e-31`, about fifteen digits _below_ what `f64` could even
//! represent, which is the compensated layer earning its keep.
//!
//! Run with:
//!
//! ```text
//! cargo run --release -p thermite-complex --example tower --features dual,compensated
//! ```

use thermite::math::PrimalProjection;
use thermite::prelude::*;
use thermite_compensated::Compensated;
use thermite_complex::Complex;
use thermite_dual::Dual;

type V = Vector<f64>;
/// Double-double: ~106 bits of significand.
type DD = Compensated<V>;
/// One derivative direction over that.
type D = Dual<DD, 1>;
/// A complex number whose components are derivative-carrying double-doubles.
type Z = Complex<D>;

/// Both halves of a double-double lane 0, for printing.
fn dd(x: DD) -> (f64, f64) {
    (x.value().extract::<0>(), x.error().extract::<0>())
}

/// `|x|` of lane 0 as a plain f64, error half folded in.
fn dd_abs(x: DD) -> f64 {
    let (v, e) = dd(x);
    (v + e).abs()
}

fn main() {
    // z = 0.7 + 1.3i, in double-double.
    let x = DD::new(V::splat(0.7));
    let y = DD::new(V::splat(1.3));

    // Seed d/dx on the real component. The imaginary component is a constant.
    let z = Z::new(Dual::variable(x, 0), Dual::constant(y));

    // f(z) = z * e^z, evaluated through all three layers at once.
    let f = z * z.exp();

    // Cauchy-Riemann: f'(z) = du/dx + i dv/dx.
    let f_ad = Complex::<DD>::new(f.re.dual[0], f.im.dual[0]);

    // Independent reference: f'(z) = (1 + z) e^z on Complex<DD>, no dual layer.
    let zc = Complex::<DD>::new(x, y);
    let one = Complex::<DD>::new(DD::new(V::ONE), DD::ZERO);
    let f_ref = (one + zc) * zc.exp();

    let diff = f_ad - f_ref;
    let residual = dd_abs(diff.re).max(dd_abs(diff.im));

    let (vre, ere) = dd(f_ad.re);
    let (vim, eim) = dd(f_ad.im);
    println!("f'(z) via AD + Cauchy-Riemann:");
    println!("  re = {vre:+.17e} + {ere:+.17e}");
    println!("  im = {vim:+.17e} + {eim:+.17e}");
    let (vre, ere) = dd(f_ref.re);
    let (vim, eim) = dd(f_ref.im);
    println!("f'(z) analytic, (1 + z) e^z:");
    println!("  re = {vre:+.17e} + {ere:+.17e}");
    println!("  im = {vim:+.17e} + {eim:+.17e}");
    println!("max |component residual| = {residual:.3e}");

    // Well below f64 epsilon: only possible because every intermediate was a
    // double-double. The two paths agree in the ERROR halves, not just the
    // representable f64 part.
    assert!(residual < 1e-28, "tower lost precision: {residual:.3e}");

    // The Primal projection collapses the whole tower in one step:
    // Complex -> (strip imaginary) Dual -> (strip derivatives) Compensated,
    // and Compensated is the fixpoint (its error half is precision, not
    // augmentation), so it survives. Constants enter the tower the same way,
    // full double-double resolution included, without hand-zeroing the
    // augmented fields.
    let pi: Z = Z::from_primal(DD::PI);
    let (pv, pe) = dd(pi.re.to_primal());
    println!("pi lifted through the tower and back: {pv:+.17e} + {pe:+.17e}");
    assert_ne!(pe, 0.0, "the double-double error half must survive the round trip");
    assert_eq!(dd(pi.re.dual[0]).0, 0.0);
    assert_eq!(dd(pi.im.to_primal()).0, 0.0);
}
