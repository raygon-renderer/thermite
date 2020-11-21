#![allow(unused)]

use thermite::*;

type Vi32 = <backends::AVX2 as Simd>::Vi32;
type Vu32 = <backends::AVX2 as Simd>::Vu32;
type Vu64 = <backends::AVX2 as Simd>::Vu64;
type Vf64 = <backends::AVX2 as Simd>::Vf64;
type Vf32 = <backends::AVX2 as Simd>::Vf32;
type Vi64 = <backends::AVX2 as Simd>::Vi64;

#[test]
fn test_popcnt_32bit() {
    for i in -1000..1000 {
        let x = Vi32::splat(i) * (Vi32::indexed() + Vi32::one());

        let bits = x.count_ones();

        for j in 0..Vi32::NUM_ELEMENTS {
            let x = x.extract(j);
            let b = bits.extract(j) as u32;

            assert_eq!(x.count_ones(), b, "0b{:b} {} == {}", x, x.count_ones(), b);
        }
    }
}

#[test]
fn test_popcnt_64bit() {
    for i in -1000..1000 {
        let x = Vi64::splat(i) * (Vi64::indexed() + Vi64::one());

        let bits = x.count_ones();

        for j in 0..Vi64::NUM_ELEMENTS {
            let x = x.extract(j);
            let b = bits.extract(j) as u32;

            assert_eq!(x.count_ones(), b, "0b{:b} {} == {}", x, x.count_ones(), b);
        }
    }
}

#[test]
fn test_tzc_64bit() {
    for i in -1000..1000 {
        let x = Vi64::splat(i) * (Vi64::indexed() + Vi64::one());

        let bits = x.trailing_zeros();

        for j in 0..Vi64::NUM_ELEMENTS {
            let x = x.extract(j);
            let b = bits.extract(j) as u32;

            assert_eq!(x.trailing_zeros(), b, "0b{:b} {} == {}", x, x.trailing_zeros(), b);
        }
    }
}
