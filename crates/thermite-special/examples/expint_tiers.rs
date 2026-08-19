//! `expint` accuracy per precision tier, in the fraction's expensive regime, both formats.
use thermite::math::policy::policies::{HighPerformance, Performance, Precision, Reference, UltraPerformance};
use thermite::prelude::*;
use thermite_special::SpecialMathWithPolicy;
type D = Vector<f64>;
type F = Vector<f32>;
fn main() {
    println!("fmt,tier,n,x,v");
    macro_rules! row {
        ($p:ty, $name:literal) => {
            for (n, xs) in [
                (3usize, [11.5f64, 20.0, 60.0]),
                (8, [6.5, 20.0, 60.0]),
                (20, [10.0, 20.0, 60.0]),
            ] {
                for x in xs {
                    let v = match n {
                        3 => D::splat(x).expint_p::<$p, 3>(),
                        8 => D::splat(x).expint_p::<$p, 8>(),
                        _ => D::splat(x).expint_p::<$p, 20>(),
                    };
                    println!("f64,{},{n},{x:.17e},{:.17e}", $name, v.extract::<0>());
                    let w = match n {
                        3 => F::splat(x as f32).expint_p::<$p, 3>(),
                        8 => F::splat(x as f32).expint_p::<$p, 8>(),
                        _ => F::splat(x as f32).expint_p::<$p, 20>(),
                    };
                    println!("f32,{},{n},{x:.17e},{:.17e}", $name, w.extract::<0>());
                }
            }
        };
    }
    row!(Reference, "Reference");
    row!(Precision, "Precision");
    row!(Performance, "Performance");
    row!(HighPerformance, "HighPerformance");
    row!(UltraPerformance, "UltraPerformance");
}
