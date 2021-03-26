#![allow(unused)]

use thermite::*;

type Vi32 = <backends::avx2::AVX2 as Simd>::Vi32;
type Vu32 = <backends::avx2::AVX2 as Simd>::Vu32;
type Vf32 = <backends::avx2::AVX2 as Simd>::Vf32;
type Vf64 = <backends::avx2::AVX2 as Simd>::Vf64;

#[test]
fn test_powi() {
    let x = Vf32::splat(5.5);

    let y0 = x.reciprocal_p::<policies::UltraPerformance>();
    let y1 = x.reciprocal_p::<policies::Performance>();
    let y2 = x.reciprocal_p::<policies::Precision>();
    let y3 = x.reciprocal_p::<policies::Reference>();

    println!(
        "{} == {} == {} == {} == {}",
        1.0 / x.extract(0),
        y0.extract(0),
        y1.extract(0),
        y2.extract(0),
        y3.extract(0),
    );
}
