use thermite::*;

pub mod geo;

use thermite::backends::AVX2;

type Vf32 = <AVX2 as Simd>::Vf32;

type Vector3xN = geo::Vector3xN<AVX2>;

#[no_mangle]
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
pub unsafe extern "C" fn test_normalize(v: &mut Vector3xN) {
    *v = v.normalize()
}

fn main() {}
