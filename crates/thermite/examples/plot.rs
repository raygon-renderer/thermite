#![allow(unused)]

use thermite::*;

pub mod geo;

use thermite::backends::avx2::AVX2;

type Vf32 = <AVX2 as Simd>::Vf32;
type Vf64 = <AVX2 as Simd>::Vf64;
type Vi32 = <AVX2 as Simd>::Vi32;
type Vu64 = <AVX2 as Simd>::Vu64;
type Vu32 = <AVX2 as Simd>::Vu32;
type Vi64 = <AVX2 as Simd>::Vi64;

use plotly::common::{ColorScale, ColorScalePalette, DashType, Fill, Font, Line, LineShape, Marker, Mode, Title};
use plotly::layout::{Axis, BarMode, Layout, Legend, TicksDirection};
use plotly::{Bar, NamedColor, Plot, Rgb, Rgba, Scatter};

fn plot_function<F>(name: &str, x_axis: &Vec<f32>, plot: &mut Plot, mut f: F)
where
    F: FnMut(Vf32) -> Vf32,
{
    let mut y_axis = vec![0.0; x_axis.len()];

    for (src, dst) in x_axis
        .chunks(Vf32::NUM_ELEMENTS)
        .zip(y_axis.chunks_mut(Vf32::NUM_ELEMENTS))
    {
        f(Vf32::load_unaligned(src))
            //.clamp(Vf32::splat(-400.0), Vf32::splat(400.0))
            .store_unaligned(dst);
    }

    plot.add_trace(Scatter::new(x_axis.clone(), y_axis).mode(Mode::Lines).name(name));
}

fn main() {
    let num_points = Vf32::NUM_ELEMENTS * 1000;

    let x_axis: Vec<f32> = (0..num_points)
        .into_iter()
        .map(|x| (x as f32 / num_points as f32) * 30.0 - 15.0)
        .collect();

    let layout = Layout::new().title(Title::new("Gamma function"));
    let mut plot = Plot::new();

    //for i in 0..5 {
    //    plot_function(&format!("Y{}", i), &x_axis, &mut plot, |x| {
    //        x.bessel_y_p::<policies::Precision>(i)
    //    });
    //}

    //plot_function("cos(x) [Precision]", &x_axis, &mut plot, |x| {
    //    x.cos_p::<policies::Precision>()
    //});
    //plot_function("cos(x) [Reference]", &x_axis, &mut plot, |x| {
    //    x.cos_p::<policies::Reference>()
    //});
    //
    //plot_function("sin(x) [Precision]", &x_axis, &mut plot, |x| {
    //    x.sin_p::<policies::Precision>()
    //});
    //plot_function("sin(x) [Reference]", &x_axis, &mut plot, |x| {
    //    x.sin_p::<policies::Reference>()
    //});

    //plot_function("tgamma(x)", &x_axis, &mut plot, |x| x.tgamma());
    //plot_function("lgamma(x)", &x_axis, &mut plot, |x| x.lgamma());
    //plot_function("ln(tgamma(x))", &x_axis, &mut plot, |x| x.tgamma().ln());
    //plot_function("diff*1000", &x_axis, &mut plot, |x| {
    //    (x.tgamma().ln() - x.lgamma()) * Vf32::splat(1000.0)
    //});

    //plot_function("digamma(x)", &x_axis, &mut plot, |x| x.digamma());

    /*
    plot_function("Gamma Avg", &x_axis, &mut plot, |x| x.tgamma());
    plot_function("Gamma Worst", &x_axis, &mut plot, |x| {
        x.tgamma_p::<policies::UltraPerformance>()
    });

    plot_function("Diffx100", &x_axis, &mut plot, |x| {
        (x.tgamma() - x.tgamma_p::<policies::UltraPerformance>()) * Vf32::splat(100.0)
    });
     */

    plot_function("Ln Avg", &x_axis, &mut plot, |x| x.ln());
    plot_function("Ln Worst", &x_axis, &mut plot, |x| {
        x.ln_p::<policies::UltraPerformance>()
    });

    plot_function("Diffx100", &x_axis, &mut plot, |x| {
        (x.ln() - x.ln_p::<policies::UltraPerformance>()) * Vf32::splat(100.0)
    });

    /*
    for i in 0..5 {
        plot_function(&format!("beta(x, {}) [UP]", i), &x_axis, &mut plot, |x| {
            x.beta_p::<policies::UltraPerformance>(Vf32::splat_as(i + 1))
        });
    }

    for i in 0..5 {
        plot_function(&format!("beta(x, {}) [Precision]", i), &x_axis, &mut plot, |x| {
            x.beta_p::<policies::Precision>(Vf32::splat_as(i + 1))
        });
    }
     */

    plot.show();
}
