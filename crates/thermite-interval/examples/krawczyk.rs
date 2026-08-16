//! Krawczyk root certification, one candidate box per SIMD lane.
//!
//! Every root of a transcendental function on an interval is found AND
//! proven. Each reported box provably contains exactly one root, and every
//! discarded box provably contains none. The derivative enclosures the
//! operator needs come from `Dual<Interval<V>, 1>`: automatic differentiation
//! whose chain rule runs over interval arithmetic, so `f'` is itself
//! rigorously enclosed. Nothing here is a heuristic.
//!
//! Run with:
//!
//! ```text
//! cargo run --release -p thermite-interval --example krawczyk --features dual
//! ```

use thermite::math::TranscendentalMath;
use thermite::prelude::*;
use thermite_interval::verify::{DualInterval, certify_roots, krawczyk_step};
use thermite_interval::{Interval, Tightest};

/// f(x) = sin(x) - x/3: three real roots (0 and +-2.2788...), and the trig
/// enclosure has to do real work (quadrant analysis) over the search boxes.
fn f_sin<V: FloatVector + TranscendentalMath>(x: V) -> V {
    x.sin() - x.scale(<V::Element as thermite::element::FloatElement>::from_ratio(1, 3))
}

/// g(x) = e^x - 3x: two real roots (0.6190... and 1.5121...).
fn g_exp<V: FloatVector + TranscendentalMath>(x: V) -> V {
    x.exp() - x.scale(<V::Element as thermite::element::FloatElement>::from_int(3))
}

#[thermite::dispatch(S)]
fn run<S: Simd>() {
    type V<S> = Vector<<S as thermite::simd::NativeSimd>::f64xN>;
    type I<S> = Interval<V<S>, Tightest>;

    println!("lanes = {} (each Krawczyk step certifies {} boxes at once)\n", V::<S>::LANES, V::<S>::LANES);

    // --- one step, by hand -------------------------------------------------
    // A box known to contain a root: K(X) lands strictly inside X -> proof.
    let x = I::<S>::bounds(V::<S>::splat(2.0), V::<S>::splat(2.5));
    let (k, v) = krawczyk_step(|d: DualInterval<V<S>, Tightest>| f_sin(d), x);
    println!("sin(x) - x/3 on [2.0, 2.5]:");
    println!("  K(X) = [{:.6}, {:.6}]  unique = {}  excluded = {}", k.lo().extract::<0>(), k.hi().extract::<0>(),
             v.unique.all(), v.excluded.all());

    // A box with no root: K(X) misses X entirely -> proof of absence.
    let x = I::<S>::bounds(V::<S>::splat(0.5), V::<S>::splat(1.5));
    let (_, v) = krawczyk_step(|d: DualInterval<V<S>, Tightest>| f_sin(d), x);
    println!("sin(x) - x/3 on [0.5, 1.5]:  unique = {}  excluded = {}\n", v.unique.all(), v.excluded.all());

    // --- full certification by subdivision --------------------------------
    for (name, a, b, kind) in [("sin(x) - x/3", -4.0, 4.0, 0), ("e^x - 3x", -2.0, 4.0, 1)] {
        let cert = if kind == 0 {
            certify_roots::<V<S>, Tightest, _>(|d| f_sin(d), a, b, 1e-12, 100_000)
        } else {
            certify_roots::<V<S>, Tightest, _>(|d| g_exp(d), a, b, 1e-12, 100_000)
        };

        println!("{name} on [{a}, {b}]: {} steps over {} boxes", cert.steps, cert.boxes);
        let mut roots = cert.roots.clone();
        roots.sort_by(|p, q| p.lo.partial_cmp(&q.lo).unwrap());
        for r in &roots {
            println!("  PROVEN unique root in [{:.15}, {:.15}]  (width {:.2e})", r.lo, r.hi, r.hi - r.lo);
        }
        for u in &cert.unresolved {
            println!("  unresolved: [{:.15}, {:.15}]", u.lo, u.hi);
        }
        println!();
    }
}

fn main() {
    thermite::dispatch_dyn!(run());
}
