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

fn main() {
    let num_points = Vf32::NUM_ELEMENTS * 1000;

    let x_axis: Vec<f32> = (0..num_points)
        .into_iter()
        .map(|x| (x as f32 / num_points as f32) * 30.0 - 35.0)
        .collect();

    let mut y_axis = vec![0.0; x_axis.len()];

    for (src, dst) in x_axis
        .chunks(Vf32::NUM_ELEMENTS)
        .zip(y_axis.chunks_mut(Vf32::NUM_ELEMENTS))
    {
        Vf32::load_unaligned(src)
            .tgamma()
            .clamp(Vf32::splat(-100.0), Vf32::splat(1000.0))
            .store_unaligned(dst);
    }

    let trace = Scatter::new(x_axis, y_axis).mode(Mode::Lines);

    let layout = Layout::new().title(Title::new("Gamma function"));
    let mut plot = Plot::new();
    plot.add_trace(trace);
    plot.show();
}
