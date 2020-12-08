#![allow(unused)]

use thermite::*;

pub mod geo;

use thermite::backends::AVX2;

type Vf32 = <AVX2 as Simd>::Vf32;
type Vf64 = <AVX2 as Simd>::Vf64;
type Vi32 = <AVX2 as Simd>::Vi32;
type Vu64 = <AVX2 as Simd>::Vu64;
type Vu32 = <AVX2 as Simd>::Vu32;
type Vi64 = <AVX2 as Simd>::Vi64;

use plotly::common::{ColorScale, ColorScalePalette, DashType, Fill, Font, Line, LineShape, Marker, Mode, Title};
use plotly::layout::{Axis, BarMode, Layout, Legend, TicksDirection};
use plotly::{Bar, NamedColor, Plot, Rgb, Rgba, Scatter};

fn plot_function<F>(name: &str, x_axis: &Vec<f64>, plot: &mut Plot, mut f: F)
where
    F: FnMut(Vf64) -> Vf64,
{
    let mut y_axis = vec![0.0; x_axis.len()];

    for (src, dst) in x_axis
        .chunks(Vf64::NUM_ELEMENTS)
        .zip(y_axis.chunks_mut(Vf64::NUM_ELEMENTS))
    {
        f(Vf64::load_unaligned(src))
            .clamp(Vf64::splat(-40.0), Vf64::splat(40.0))
            .store_unaligned(dst);
    }

    plot.add_trace(Scatter::new(x_axis.clone(), y_axis).mode(Mode::Lines).name(name));
}

fn main() {
    let num_points = Vf64::NUM_ELEMENTS * 1000 * 30;

    let x_axis: Vec<f64> = (0..num_points)
        .into_iter()
        .map(|x| (x as f64 / num_points as f64) * 30.0 - 15.0)
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

    plot_function("tgamma(x)", &x_axis, &mut plot, |x| x.tgamma());
    plot_function("lgamma(x)", &x_axis, &mut plot, |x| x.lgamma());
    //plot_function("ln(tgamma(x))", &x_axis, &mut plot, |x| x.tgamma().ln());
    //plot_function("diff*1000", &x_axis, &mut plot, |x| {
    //    (x.tgamma().ln() - x.lgamma()) * Vf64::splat(1000.0)
    //});

    plot_function("digamma(x)", &x_axis, &mut plot, |x| x.digamma());

    plot.show();
}
