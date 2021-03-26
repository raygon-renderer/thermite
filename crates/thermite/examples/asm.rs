#![allow(unused)]

// NOTE: This example only exists to be compiled and inspected as assembly via the command:
// `cargo rustc --example asm --release -- -C target-feature=+sse2 --emit asm`
// It's easier to access the example output in the `target/release/examples` directory

use no_panic::no_panic;

use thermite::*;

pub mod geo;

use thermite::backends::avx2::AVX2;
use thermite::rng::SimdRng;

type Vf32 = <AVX2 as Simd>::Vf32;
type Vf64 = <AVX2 as Simd>::Vf64;
type Vi32 = <AVX2 as Simd>::Vi32;
type Vu64 = <AVX2 as Simd>::Vu64;
type Vu32 = <AVX2 as Simd>::Vu32;
type Vi64 = <AVX2 as Simd>::Vi64;

type Vector3xN = geo::Vector3xN<AVX2>;

type Xoshiro128Plus = thermite::rng::xoshiro::Xoshiro128Plus<AVX2>;

#[no_mangle]
#[inline(never)]
pub fn test_dynamic_dispatch(value: &mut [f32]) {
    assert_eq!(value.len(), 8);

    #[dispatch]
    fn test<S: Simd>(value: &mut [f32]) {
        thermite::Vf32::<S>::load_unaligned(value).exp2().store_unaligned(value);
    }

    dispatch_dyn!({ test::<S>(value) })
}

#[no_mangle]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn test_simdrng(rng: &mut Xoshiro128Plus) -> Vf64 {
    rng.next_f64()
}

#[no_mangle]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn test_revbits(x: Vi32) -> Vi32 {
    x.reverse_bits()
}

#[no_mangle]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn test_normalize(v: &mut Vector3xN) {
    *v = v.normalize()
}

#[no_mangle]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn test_u64div(a: Vu64, b: Vu64) -> Vu64 {
    a / b
}

#[no_mangle]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn test_bitmask(b: u16) -> Vu64 {
    Mask::from_bitmask(b).value()
}

#[no_mangle]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn test_cross(a: Vector3xN, b: Vector3xN) -> Vector3xN {
    a.cross(&b)
}

#[no_mangle]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn do_alloc(count: usize) -> VectorBuffer<AVX2, Vf32> {
    Vf32::alloc(count)
}

#[no_mangle]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn test_powf_ps(y: Vf32, x: Vf32) -> Vf32 {
    y.powf(x)
}

#[no_mangle]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn test_powf_pd(y: Vf64, x: Vf64) -> Vf64 {
    y.powf(x)
}

#[no_mangle]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn test_smootheststep(x: Vf32) -> Vf32 {
    x.smootheststep()
}

#[no_mangle]
#[inline(never)]
//#[target_feature(enable = "avx2,fma")]
pub unsafe fn test_pdsin(x: Vf64) -> Vf64 {
    x.sin()
}

#[no_mangle]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn test_pssin_cos(x: Vf32) -> (Vf32, Vf32) {
    x.sin_cos_p::<policies::UltraPerformance>()
}

#[no_mangle]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn test_select_neg_ps(x: Vf32, a: Vf32, b: Vf32) -> Vf32 {
    x.is_negative().select(a, b)
}

#[no_mangle]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn test_select_neg_epi32(x: Vi32, a: Vi32, b: Vi32) -> Vi32 {
    x.is_negative().select(a, b)
}

#[no_mangle]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
#[no_panic]
pub unsafe fn test_shuffle(x: Vf64, y: Vf64) -> Vf64 {
    match Vf64::NUM_ELEMENTS {
        4 => shuffle!(x, y, [6, 2, 1, 7]),
        8 => shuffle!(x, y, [5, 6, 10, 9, 2, 8, 6, 4]),
        _ => unimplemented!(),
    }
}

#[no_mangle]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn test_shuffle_dyn_unchecked(a: Vf32, b: Vf32, indices: &[usize]) -> Vf32 {
    a.shuffle_dyn_unchecked(b, indices)
}

//#[no_mangle]
//#[inline(never)]
//#[target_feature(enable = "avx2,fma")]
//pub unsafe fn test_shuffle_dyn(x: Vf32, y: Vf32, indices: &[usize; 8]) -> Vf32 {
//    x.shuffle(y, &indices[..])
//}

#[no_mangle]
#[inline(never)]
//#[target_feature(enable = "avx2,fma")]
pub unsafe fn test_pstgamma(x: Vf32) -> Vf32 {
    x.tgamma_p::<policies::UltraPerformance>()
}

#[no_mangle]
#[inline(never)]
//#[target_feature(enable = "avx2,fma")]
pub unsafe fn test_pdtgamma(x: Vf64) -> Vf64 {
    x.tgamma()
}

#[no_mangle]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn test_pserf(x: Vf32) -> Vf32 {
    x.erf()
}

#[no_mangle]
#[inline(never)]
pub unsafe fn test_psexp(x: Vf32) -> Vf32 {
    x.exp()
}

#[no_mangle]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn test_pderfinv(x: Vf64) -> Vf64 {
    x.erfinv()
}

#[no_mangle]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn test_pscbrt(x: Vf32) -> Vf32 {
    x.cbrt()
}

#[no_mangle]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn test_ps_bessel_y4(x: Vf32) -> Vf32 {
    x.bessel_y_p::<policies::Precision>(4)
}

#[no_mangle]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn test_poly(x: Vf32, e: &[f32]) -> Vf32 {
    x.poly_f(128, |i| Vf32::splat(*e.get_unchecked(i)))
}

#[no_mangle]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn test_rational_poly(x: Vf32, e: &[f32], d: &[f32]) -> Vf32 {
    let n0 = x.poly_f(19, |i| Vf32::splat(*e.get_unchecked(i)));
    let n1 = x.poly_f(19, |i| Vf32::splat(*d.get_unchecked(i)));

    n0 / n1
}

#[no_mangle]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn test_rational_poly2(x: Vf32, e: &[f32], d: &[f32]) -> Vf32 {
    assert!(e.len() == 19 && e.len() == d.len());

    x.poly_rational_p::<policies::Size>(e, d)
}

#[no_mangle]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn test_poly2(x: Vf32) -> Vf32 {
    x.poly_f(128, |i| {
        Vf32::splat((-1.0f32).powi(i as i32) * (2f32.powi(i as i32) - i as f32))
    })
}

#[no_mangle]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn test_pdcbrt(x: Vf64) -> Vf64 {
    x.cbrt()
}

#[no_mangle]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn test_pdsinh(x: Vf64) -> Vf64 {
    x.sinh_p::<policies::Precision>()
}

#[no_mangle]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn test_pssinh(x: Vf32) -> Vf32 {
    x.sinh_p::<policies::Precision>()
}

#[no_mangle]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn test_jacobi(x: Vf32, alpha: Vf32, beta: Vf32, n: u32, m: u32) -> Vf32 {
    x.legendre(50, 0)
}

#[no_mangle]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn test_cast2(x: Vf64) -> Vi64 {
    x.cast()
}

fn main() {}
