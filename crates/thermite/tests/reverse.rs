#![allow(unused)]

use thermite::*;

type Vi32 = <backends::avx2::AVX2 as Simd>::Vi32;
type Vu32 = <backends::avx2::AVX2 as Simd>::Vu32;
type Vu64 = <backends::avx2::AVX2 as Simd>::Vu64;
type Vf64 = <backends::avx2::AVX2 as Simd>::Vf64;
type Vf32 = <backends::avx2::AVX2 as Simd>::Vf32;
type Vi64 = <backends::avx2::AVX2 as Simd>::Vi64;

#[test]
fn test_bitreversal_32bit() {
    for i in -1000..1000 {
        let x = Vi32::splat(i) * (Vi32::indexed() + Vi32::one());

        let y = x.reverse_bits();

        for j in 0..Vi32::NUM_ELEMENTS {
            let x = x.extract(j).reverse_bits();
            let y = y.extract(j);

            assert_eq!(x, y);
        }
    }
}

#[test]
fn test_bitreversal_64bit() {
    for i in -1000..1000 {
        let x = Vi64::splat(i) * (Vi64::indexed() + Vi64::one());

        let y = x.reverse_bits();

        for j in 0..Vi64::NUM_ELEMENTS {
            let x = x.extract(j).reverse_bits();
            let y = y.extract(j);

            assert_eq!(x, y);
        }
    }
}
