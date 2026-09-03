//! Are the scipy disagreements a policy-tier trade or a real accuracy gap?
use thermite::math::policy::policies::{Performance, Precision, Reference};
use thermite::prelude::*;
use thermite_complex::Complex;
use thermite_complex::prelude::ComplexSpecialMathWithPolicy;
use thermite_special::{RealSpecialMathWithPolicy, SpecialMathWithPolicy};

type D = Vector<f64>;
fn s(x: f64) -> D {
    D::splat(x)
}

fn main() {
    let x = 5.2834604670233245e-08;
    println!("erfcx({x})");
    println!("  perf  {:.17e}", s(x).erfcx_p::<Performance>().extract::<0>());
    println!("  prec  {:.17e}", s(x).erfcx_p::<Precision>().extract::<0>());
    println!("  ref   {:.17e}", s(x).erfcx_p::<Reference>().extract::<0>());

    println!("erfinv(-0.99)");
    println!("  perf  {:.17e}", s(-0.99).erfinv_p::<Performance>().extract::<0>());
    println!("  prec  {:.17e}", s(-0.99).erfinv_p::<Precision>().extract::<0>());
    println!("  ref   {:.17e}", s(-0.99).erfinv_p::<Reference>().extract::<0>());

    println!("probit(0.21)");
    println!("  perf  {:.17e}", s(0.21).probit_p::<Performance>().extract::<0>());
    println!("  prec  {:.17e}", s(0.21).probit_p::<Precision>().extract::<0>());
    println!("  ref   {:.17e}", s(0.21).probit_p::<Reference>().extract::<0>());

    let e = 0.8717676911661476;
    println!("expint1({e})");
    println!("  perf  {:.17e}", s(e).expint_n_p::<Performance, 1>().extract::<0>());
    println!("  prec  {:.17e}", s(e).expint_n_p::<Precision, 1>().extract::<0>());
    println!("  ref   {:.17e}", s(e).expint_n_p::<Reference, 1>().extract::<0>());

    let z = Complex::new(s(-8.4), s(0.0));
    println!("Re w(-8.4 + 0i)   [true = e^-70.56 = 2.2708e-31]");
    println!("  perf  {:.6e}", z.faddeeva_w_p::<Performance>().re.extract::<0>());
    println!("  prec  {:.6e}", z.faddeeva_w_p::<Precision>().re.extract::<0>());
    let z2 = Complex::new(s(-8.4), s(1e-6));
    println!("Re w(-8.4 + 1e-6 i)");
    println!("  perf  {:.6e}", z2.faddeeva_w_p::<Performance>().re.extract::<0>());
    println!("  prec  {:.6e}", z2.faddeeva_w_p::<Precision>().re.extract::<0>());
}
