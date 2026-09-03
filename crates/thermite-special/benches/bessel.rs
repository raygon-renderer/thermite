//! The Bessel family against libm, at two widths.
//!
//! `J`/`Y` are the only Bessel functions with a scalar baseline anywhere in reach: libm ships
//! `j0`/`j1`/`y0`/`y1` (the C/POSIX XSI set) and has **no modified Bessel at all**, so `I`/`K`
//! are benchmarked against themselves across regimes instead. Nothing else in the ecosystem
//! has them to compare to. No CPU SIMD library ships any Bessel function.
//!
//! Three things are separated deliberately:
//!
//! - **scalar**: `Vector<f64>`, the 1-lane seed. Against `libm` this is an algorithm-vs-
//!   algorithm comparison with the vector width taken out: Boost's 3 regions and one Hankel
//!   arm against fdlibm's 4-way split envelope. If this is not competitive, the SIMD number
//!   is measuring the wrong thing.
//! - **x86v3**: `f64x4`/`f32x8`. The ratio against the scalar row is the width win. The ratio
//!   against libm is what a caller actually gets.
//! - **regime**: for `I`, below the 7.75 series/asymptotic seam, straddling it, and out past
//!   the order-2 recurrence crossover at 40 where the asymptotic arm takes over. The
//!   straddling row is the honest worst case, since a packet pays for every arm its lanes want.
//!
//! Same inputs, one process, so the ratios hold even where the absolute numbers drift.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use thermite::backend::x86_v3::X86V3;
use thermite::prelude::*;
use thermite_special::bessel::{I, J, K, Scaled, Y};
use thermite_special::{SpecialMath, SpecialMathWithPolicy};

type V1 = Vector<f64>;
type S1 = Vector<f32>;
type V4 = Vector<<X86V3 as Simd>::f64x4>;
type F8 = Vector<<X86V3 as Simd>::f32x8>;

/// One dispatched kernel per measured form.
///
/// **Without this the vector rows are meaningless.** A generic-over-`V` body with no
/// `#[thermite::dispatch]` ancestor compiles at the base ISA (SSE2 here, since nothing sets
/// `target-cpu`), so an AVX2 vector type lowers to out-of-line calls, one per intrinsic. The
/// first version of this file had no dispatch and reported f64x4 at only 1.9x the 1-lane row.
/// `benches/langevin.rs` carries the same warning for the same reason.
macro_rules! kernel {
    ($name:ident, |$x:ident: $V:ident| $body:expr) => {
        #[thermite::dispatch($V)]
        fn $name<$V>(xs: &[$V]) -> $V
        where
            $V: FloatVector + thermite::math::TranscendentalMath + SpecialMathWithPolicy,
        {
            let mut acc = $V::ZERO;
            for &$x in xs {
                acc = acc + $body;
            }
            acc
        }
    };
}

/// Same shape, but the precision tier is a parameter.
///
/// This exists to price the accuracy work in `J`/`Y`: the opt-in compensated Horner on eleven
/// `poly_rational` calls (about 10 ops per term against 1 FMA), the `cos(2x)` cancellation
/// repair, and the bounded-variable substitution. The repair is gated at `Average` and above,
/// so `Worst` is the only row that skips it. The compensation is NOT tiered and every row
/// pays it.
macro_rules! kernel_p {
    ($name:ident, |$x:ident: $V:ident, $P:ident| $body:expr) => {
        #[thermite::dispatch($V)]
        fn $name<$V, $P>(xs: &[$V]) -> $V
        where
            $V: FloatVector + thermite::math::TranscendentalMath + SpecialMathWithPolicy,
            $P: thermite::math::policy::Policy,
        {
            let mut acc = $V::ZERO;
            for &$x in xs {
                acc = acc + $body;
            }
            acc
        }
    };
}

kernel_p!(kp_j0, |x: V, P| x.bessel_n_p::<P, J, 0>());
kernel_p!(kp_j1, |x: V, P| x.bessel_n_p::<P, J, 1>());
kernel_p!(kp_y0, |x: V, P| x.bessel_n_p::<P, Y, 0>());
kernel_p!(kp_y1, |x: V, P| x.bessel_n_p::<P, Y, 1>());

kernel!(k_j0, |x: V| x.bessel_n::<J, 0>());
kernel!(k_j1, |x: V| x.bessel_n::<J, 1>());
kernel!(k_y0, |x: V| x.bessel_n::<Y, 0>());
kernel!(k_i0, |x: V| x.bessel_n::<I, 0>());
kernel!(k_i0s, |x: V| x.bessel_n::<Scaled<I>, 0>());
kernel!(k_i2, |x: V| x.bessel_n::<I, 2>());
kernel!(k_k0, |x: V| x.bessel_n::<K, 0>());
kernel!(k_j2, |x: V| x.bessel_n::<J, 2>());
kernel!(k_j8, |x: V| x.bessel_n::<J, 8>());
kernel!(k_j20, |x: V| x.bessel_n::<J, 20>());
kernel!(k_y2, |x: V| x.bessel_n::<Y, 2>());
kernel!(k_y8, |x: V| x.bessel_n::<Y, 8>());
kernel!(k_y20, |x: V| x.bessel_n::<Y, 20>());
kernel!(k_i8, |x: V| x.bessel_n::<I, 8>());
kernel!(k_k8, |x: V| x.bessel_n::<K, 8>());

const COUNT: usize = 256;

/// Low-discrepancy lanes over `[lo, hi)`, so a straddling range really does land on both
/// sides of a seam rather than clustering on one.
fn scalars(count: usize, lo: f64, hi: f64) -> Vec<f64> {
    (0..count * 4)
        .map(|i| lo + (hi - lo) * ((i as f64) * 0.6180339887498949).fract())
        .collect()
}

fn packed4(src: &[f64]) -> Vec<V4> {
    src.chunks_exact(4).map(|c| V4::new([c[0], c[1], c[2], c[3]])).collect()
}

/// 1-lane vectors, so the scalar row can go through the SAME dispatched kernel as the wide
/// ones. Without this the scalar row compiles at the SSE2 baseline while the vector rows get
/// AVX2, and the width ratio is measuring the dispatch boundary as much as the width.
fn packed1(src: &[f64]) -> Vec<V1> {
    src.iter().map(|&x| V1::splat(x)).collect()
}

fn packed1f(src: &[f32]) -> Vec<S1> {
    src.iter().map(|&x| S1::splat(x)).collect()
}

fn as_f32(src: &[f64]) -> Vec<f32> {
    src.iter().map(|&x| x as f32).collect()
}

fn packed8f(src: &[f64]) -> Vec<F8> {
    src.chunks_exact(8)
        .map(|c| {
            let mut lanes = [0.0f32; 8];
            for (k, v) in lanes.iter_mut().enumerate() {
                *v = c[k] as f32;
            }
            F8::new(lanes)
        })
        .collect()
}

/// `J_0` and `Y_0` against libm, at both widths.
fn jy_vs_libm(c: &mut Criterion) {
    let src = scalars(COUNT, 0.1, 60.0);
    let v4 = packed4(&src);
    let f8 = packed8f(&src);

    let v1 = packed1(&src);
    let mut g = c.benchmark_group("bessel_j0");
    g.bench_function("libm scalar", |b| {
        b.iter(|| src.iter().map(|&x| libm::j0(black_box(x))).sum::<f64>())
    });
    g.bench_function("thermite scalar", |b| b.iter(|| k_j0(black_box(&v1))));
    g.bench_function("thermite x86v3 f64x4", |b| b.iter(|| k_j0(black_box(&v4))));
    g.bench_function("thermite x86v3 f32x8", |b| b.iter(|| k_j0(black_box(&f8))));
    g.finish();

    // The f32 ladder, which is the fair comparison for the f32x8 row above: libm's own f32
    // entry point, thermite at one f32 lane, thermite at eight. Comparing f32x8 against f64
    // libm answers "what if f32 is enough". This answers "at the same precision, how much of
    // the win is the algorithm and how much is the width".
    let src32 = as_f32(&src);
    let s1 = packed1f(&src32);
    let mut g = c.benchmark_group("bessel_j0_f32");
    g.bench_function("libm scalar", |b| {
        b.iter(|| src32.iter().map(|&x| libm::j0f(black_box(x))).sum::<f32>())
    });
    g.bench_function("thermite scalar", |b| b.iter(|| k_j0(black_box(&s1))));
    g.bench_function("thermite x86v3 f32x8", |b| b.iter(|| k_j0(black_box(&f8))));
    g.finish();

    let mut g = c.benchmark_group("bessel_y0_f32");
    g.bench_function("libm scalar", |b| {
        b.iter(|| src32.iter().map(|&x| libm::y0f(black_box(x))).sum::<f32>())
    });
    g.bench_function("thermite scalar", |b| {
        b.iter(|| {
            src32
                .iter()
                .map(|&x| S1::splat(black_box(x)).bessel_n::<Y, 0>().extract::<0>())
                .sum::<f32>()
        })
    });
    g.finish();

    let mut g = c.benchmark_group("bessel_y0");
    g.bench_function("libm scalar", |b| {
        b.iter(|| src.iter().map(|&x| libm::y0(black_box(x))).sum::<f64>())
    });
    g.bench_function("thermite scalar", |b| {
        b.iter(|| {
            src.iter()
                .map(|&x| V1::splat(black_box(x)).bessel_n::<Y, 0>().extract::<0>())
                .sum::<f64>()
        })
    });
    g.bench_function("thermite x86v3 f64x4", |b| b.iter(|| k_y0(black_box(&v4))));
    g.finish();

    // Does the packet pay for regions its lanes do not want? `J` has three, and the wide
    // [0.1, 60) range above guarantees that essentially every 4-lane packet straddles at
    // least one boundary, so the vector rows evaluate arms they then discard, while the
    // 1-lane rows genuinely branch. A narrow band entirely inside the Hankel region isolates
    // that: if width scaling is much better here, the wide-range sub-linearity is region
    // straddling and not a codegen problem.
    let narrow = scalars(COUNT, 20.0, 30.0);
    let n4 = packed4(&narrow);
    let n8 = packed8f(&narrow);
    let mut g = c.benchmark_group("bessel_j0_narrow");
    g.bench_function("thermite scalar", |b| {
        b.iter(|| {
            narrow
                .iter()
                .map(|&x| V1::splat(black_box(x)).bessel_n::<J, 0>().extract::<0>())
                .sum::<f64>()
        })
    });
    g.bench_function("thermite x86v3 f64x4", |b| b.iter(|| k_j0(black_box(&n4))));
    g.bench_function("thermite x86v3 f32x8", |b| b.iter(|| k_j0(black_box(&n8))));
    g.finish();

    // `J_1` separately: its Hankel arm carries one more term than `J_0`, and libm's `j1`
    // splits its envelope the same four ways `j0` does, so the shape of the gap should match.
    let mut g = c.benchmark_group("bessel_j1");
    g.bench_function("libm scalar", |b| {
        b.iter(|| src.iter().map(|&x| libm::j1(black_box(x))).sum::<f64>())
    });
    g.bench_function("thermite x86v3 f64x4", |b| b.iter(|| k_j1(black_box(&v4))));
    g.finish();
}

/// What the accuracy work costs: every `J`/`Y` order-0/1 form at `Average` and `Best`, both
/// widths, against libm.
///
/// `Average` IS the default policy (`Performance`), so the `Average` rows are what a plain
/// `bessel_j::<0>()` call gets. `Worst` is included on one row only, as the floor: it is the
/// single tier that skips the `cos(2x)` repair, and still pays the compensated Horner,
/// which no policy turns off.
fn policy_cost(c: &mut Criterion) {
    use thermite::math::policy::policies::{Performance as Average, Precision as Best, UltraPerformance as Worst};

    let src = scalars(COUNT, 0.1, 60.0);
    let v4 = packed4(&src);
    let f8 = packed8f(&src);
    let src32 = as_f32(&src);

    macro_rules! row {
        ($g:expr, $k:ident, $v4:expr, $f8:expr) => {
            $g.bench_function("f64x4 average", |b| b.iter(|| $k::<_, Average>(black_box($v4))));
            $g.bench_function("f64x4 best", |b| b.iter(|| $k::<_, Best>(black_box($v4))));
            $g.bench_function("f32x8 average", |b| b.iter(|| $k::<_, Average>(black_box($f8))));
            $g.bench_function("f32x8 best", |b| b.iter(|| $k::<_, Best>(black_box($f8))));
        };
    }

    let mut g = c.benchmark_group("policy/j0");
    g.bench_function("libm f64", |b| {
        b.iter(|| src.iter().map(|&x| libm::j0(black_box(x))).sum::<f64>())
    });
    g.bench_function("libm f32", |b| {
        b.iter(|| src32.iter().map(|&x| libm::j0f(black_box(x))).sum::<f32>())
    });
    row!(g, kp_j0, &v4, &f8);
    g.bench_function("f64x4 worst", |b| b.iter(|| kp_j0::<_, Worst>(black_box(&v4))));
    g.finish();

    let mut g = c.benchmark_group("policy/j1");
    g.bench_function("libm f64", |b| {
        b.iter(|| src.iter().map(|&x| libm::j1(black_box(x))).sum::<f64>())
    });
    row!(g, kp_j1, &v4, &f8);
    g.finish();

    let mut g = c.benchmark_group("policy/y0");
    g.bench_function("libm f64", |b| {
        b.iter(|| src.iter().map(|&x| libm::y0(black_box(x))).sum::<f64>())
    });
    row!(g, kp_y0, &v4, &f8);
    g.finish();

    let mut g = c.benchmark_group("policy/y1");
    g.bench_function("libm f64", |b| {
        b.iter(|| src.iter().map(|&x| libm::y1(black_box(x))).sum::<f64>())
    });
    row!(g, kp_y1, &v4, &f8);
    g.finish();

    // Region 2 alone (4, 8]: the band the substitution and the compensation were aimed at, with
    // no Hankel packet in the mix to dilute the reading.
    let mid = scalars(COUNT, 4.0, 8.0);
    let m4 = packed4(&mid);
    let m8 = packed8f(&mid);
    let mut g = c.benchmark_group("policy/y1_region2");
    g.bench_function("libm f64", |b| {
        b.iter(|| mid.iter().map(|&x| libm::y1(black_box(x))).sum::<f64>())
    });
    row!(g, kp_y1, &m4, &m8);
    g.finish();
}

/// `I` and `K` by regime. No libm row. It has neither.
fn ik_regimes(c: &mut Criterion) {
    for (name, lo, hi) in [
        ("small", 0.1, 7.0),         // ascending series only
        ("straddle", 6.0, 10.0),     // both arms live in one packet: the honest worst case
        ("large", 10.0, 39.0),       // asymptotic envelope, still under the order-2 crossover
        ("asymptotic", 50.0, 400.0), // past 40, where the series arm takes over from recurrence
    ] {
        let src = scalars(COUNT, lo, hi);
        let v4 = packed4(&src);
        let f8 = packed8f(&src);

        let mut g = c.benchmark_group(format!("bessel_i0/{name}"));
        g.bench_function("scalar", |b| {
            b.iter(|| {
                src.iter()
                    .map(|&x| V1::splat(black_box(x)).bessel_n::<I, 0>().extract::<0>())
                    .sum::<f64>()
            })
        });
        g.bench_function("x86v3 f64x4", |b| b.iter(|| k_i0(black_box(&v4))));
        g.bench_function("x86v3 f32x8", |b| b.iter(|| k_i0(black_box(&f8))));
        g.bench_function("x86v3 f64x4 scaled", |b| b.iter(|| k_i0s(black_box(&v4))));
        g.finish();

        // Order 2 is the first that reaches the recurrence, so this is where the trip count
        // shows up, and where the asymptotic arm should flatten it out again.
        let mut g = c.benchmark_group(format!("bessel_i2/{name}"));
        g.bench_function("x86v3 f64x4", |b| b.iter(|| k_i2(black_box(&v4))));
        g.finish();

        let mut g = c.benchmark_group(format!("bessel_k0/{name}"));
        g.bench_function("x86v3 f64x4", |b| b.iter(|| k_k0(black_box(&v4))));
        g.finish();
    }
}

/// Higher orders, split by which arm the order/argument pair selects.
///
/// `J_n` is the interesting one: forward when `N < x` (N-1 FMAs) and a downward continued
/// fraction otherwise, so the same order costs very different amounts either side of `N = x`.
/// `Y_n` has one arm at every order, `K_n` likewise, and `I_n` has the reverse split.
fn higher_orders(c: &mut Criterion) {
    // `above` puts every lane past the order (forward arm for J). `below` puts every lane
    // under it (downward arm). Order 8 is the pivot.
    for (band, lo, hi) in [("N_below_x", 30.0, 60.0), ("N_above_x", 0.5, 4.0)] {
        let src = scalars(COUNT, lo, hi);
        let v4 = packed4(&src);

        let mut g = c.benchmark_group(format!("bessel_order8/{band}"));
        g.bench_function("J_8 f64x4", |b| b.iter(|| k_j8(black_box(&v4))));
        g.bench_function("J_8 libm scalar", |b| {
            b.iter(|| src.iter().map(|&x| libm::jn(8, black_box(x))).sum::<f64>())
        });
        g.bench_function("Y_8 f64x4", |b| b.iter(|| k_y8(black_box(&v4))));
        g.bench_function("Y_8 libm scalar", |b| {
            b.iter(|| src.iter().map(|&x| libm::yn(8, black_box(x))).sum::<f64>())
        });
        g.bench_function("I_8 f64x4", |b| b.iter(|| k_i8(black_box(&v4))));
        g.bench_function("K_8 f64x4", |b| b.iter(|| k_k8(black_box(&v4))));
        g.finish();
    }

    // Cost against order, on one fixed band, to see what each driver charges per order.
    let src = scalars(COUNT, 1.0, 12.0);
    let v4 = packed4(&src);
    let mut g = c.benchmark_group("bessel_by_order");
    g.bench_function("J_2", |b| b.iter(|| k_j2(black_box(&v4))));
    g.bench_function("J_8", |b| b.iter(|| k_j8(black_box(&v4))));
    g.bench_function("J_20", |b| b.iter(|| k_j20(black_box(&v4))));
    g.bench_function("Y_2", |b| b.iter(|| k_y2(black_box(&v4))));
    g.bench_function("Y_8", |b| b.iter(|| k_y8(black_box(&v4))));
    g.bench_function("Y_20", |b| b.iter(|| k_y20(black_box(&v4))));
    g.bench_function("I_2", |b| b.iter(|| k_i2(black_box(&v4))));
    g.bench_function("I_8", |b| b.iter(|| k_i8(black_box(&v4))));
    g.bench_function("K_8", |b| b.iter(|| k_k8(black_box(&v4))));
    g.finish();
}

criterion_group!(benches, policy_cost, jy_vs_libm, ik_regimes, higher_orders);
criterion_main!(benches);
