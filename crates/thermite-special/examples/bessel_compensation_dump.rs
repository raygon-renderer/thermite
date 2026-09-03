//! Dump `J_1`/`Y_1` from all three compensation arms so they can be graded against mpmath.
//!
//! The question this answers: **does compensating only the sums retain any accuracy?**
//! `compensated_horner` captures both the product and the sum error where the hardware fuses,
//! and only the sum error where it does not. The product half needs a correctly rounded FMA
//! that costs more, off-FMA, than the term it is compensating.
//!
//! Theory says sums-only is a constant factor (`gamma_{2n} -> gamma_n`, about 2x) rather than
//! the precision doubling full compensation gives, because the surviving product errors are
//! still amplified by the full condition number. This measures whether that holds here.
//!
//! Three columns, all from one process so the inputs are bit-identical:
//!
//! | column | backend | tier | what runs |
//! |---|---|---|---|
//! | `s_none` | scalar 1-lane | `Performance` | no compensation at all |
//! | `s_sums` | scalar 1-lane | `Precision` | sums only (no hardware FMA) |
//! | `v_full` | x86-v3 f64x4 | `Precision` | full, both error terms |
//!
//! `s_none` is the baseline the other two have to beat. Values are printed as raw bit patterns
//! so nothing is lost to decimal formatting.
//!
//! Run: `cargo run -p thermite-special --release --example bessel_compensation_dump`

use thermite::backend::x86_v3::X86V3;
use thermite::math::policy::policies::{Performance, Precision};
use thermite::prelude::*;
use thermite_special::SpecialMathWithPolicy;
use thermite_special::bessel::{J, Y};

type V1 = Vector<f64>;
type V4 = Vector<<X86V3 as Simd>::f64x4>;

/// Dispatched so the vector row gets real AVX2 rather than the SSE2 baseline. The scalar rows
/// go through the same wrapper for symmetry. They have no FMA either way, which is the point.
#[thermite::dispatch(V)]
fn eval<V, P, const ORDER: i32, const IS_Y: bool>(x: V) -> V
where
    V: FloatVector<Element = f64> + thermite::math::TranscendentalMath + SpecialMathWithPolicy,
    P: thermite::math::policy::Policy,
{
    if IS_Y {
        x.bessel_n_p::<P, Y, ORDER>()
    } else {
        x.bessel_n_p::<P, J, ORDER>()
    }
}

fn main() {
    // Region 2 of `J_1`/`Y_1`, the band the compensation was aimed at, plus a little of the
    // Hankel arm so the reading is not purely one code path.
    const N: usize = 4000;
    let lo = 4.0f64;
    let hi = 12.0f64;

    println!("x s_none_j1 s_sums_j1 v_full_j1 s_none_y1 s_sums_y1 v_full_y1");

    for i in 0..N {
        let x = lo + (hi - lo) * (i as f64) / (N as f64);

        let s = V1::splat(x);
        let v = V4::splat(x);

        let nj = eval::<_, Performance, 1, false>(s).extract::<0>();
        let sj = eval::<_, Precision, 1, false>(s).extract::<0>();
        let fj = eval::<_, Precision, 1, false>(v).extract::<0>();

        let ny = eval::<_, Performance, 1, true>(s).extract::<0>();
        let sy = eval::<_, Precision, 1, true>(s).extract::<0>();
        let fy = eval::<_, Precision, 1, true>(v).extract::<0>();

        println!(
            "{:016x} {:016x} {:016x} {:016x} {:016x} {:016x} {:016x}",
            x.to_bits(),
            nj.to_bits(),
            sj.to_bits(),
            fj.to_bits(),
            ny.to_bits(),
            sy.to_bits(),
            fy.to_bits()
        );
    }
}
