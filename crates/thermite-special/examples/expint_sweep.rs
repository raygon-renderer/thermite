//! `expint` accuracy across order and argument, at Reference tier.
use thermite::math::policy::policies::Reference;
use thermite::prelude::*;
use thermite_special::SpecialMathWithPolicy;
type D = Vector<f64>;
fn main() {
    println!("n,x,v");
    let xs: Vec<f64> = (0..=60).map(|i| 0.5 + 1.5 * i as f64).collect();
    for &x in &xs {
        let v = D::splat(x);
        println!("1,{x:.17e},{:.17e}", v.expint_p::<Reference, 1>().extract::<0>());
        println!("3,{x:.17e},{:.17e}", v.expint_p::<Reference, 3>().extract::<0>());
        println!("5,{x:.17e},{:.17e}", v.expint_p::<Reference, 5>().extract::<0>());
        println!("8,{x:.17e},{:.17e}", v.expint_p::<Reference, 8>().extract::<0>());
        println!("12,{x:.17e},{:.17e}", v.expint_p::<Reference, 12>().extract::<0>());
        println!("20,{x:.17e},{:.17e}", v.expint_p::<Reference, 20>().extract::<0>());
    }
}
