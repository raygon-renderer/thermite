//! Both lowerings of the 16-bit per-lane variable shift at v3 must agree.
//!
//! Only one of each pair is reachable from a register impl: the widen form
//! wins everywhere except 256-bit `sllv`, where LLVM rewrites `vpackusdw`
//! into a five-op shuffle chain and the decomposition comes out ahead
//! (llvm-mca, znver3: 4.5 vs 5.5 cycles/iter). Testing the unreachable arm
//! anyway keeps it from rotting: if the pick is ever revisited on another
//! microarchitecture, the alternative is known-good rather than assumed-good.
//!
//! `packus` saturating instead of truncating is exactly the defect
//! fearless_simd shipped (their #287/#289), and the widen form is where that
//! hazard actually lives in Thermite, so it is the arm most worth pinning.
#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use rand::{RngExt as _, SeedableRng};
use rand::rngs::SmallRng;

use thermite::backend::x86_v3::arch;

#[target_feature(enable = "avx2,fma,popcnt")]
unsafe fn run() { unsafe {
    use core::arch::x86_64::*;

    let mut rng = SmallRng::seed_from_u64(0x5EED_1616);

    // Edge values plus randoms. Shift counts stay in `0..16`: past the element
    // width the result is unspecified by contract, so the two lowerings are
    // free to differ and agreement would not be a meaningful assertion.
    let edges: [u16; 8] = [0, 1, 3, 0x7FFF, 0x8000, 0xFFFF, 0xAAAA, 0x0100];

    for trial in 0..4096 {
        let mut v = [0u16; 16];
        let mut s = [0u16; 16];
        for i in 0..16 {
            v[i] = if trial % 3 == 0 { edges[i % edges.len()] } else { rng.random() };
            // In-range only: past 16 the two lowerings are both "unspecified"
            // and are not required to agree (the decomposition ignores the high
            // count bits, the widen form masks them off).
            s[i] = if trial % 2 == 0 { (i as u16) % 16 } else { rng.random::<u16>() % 16 };
        }

        let a = _mm256_loadu_si256(v.as_ptr() as *const __m256i);
        let b = _mm256_loadu_si256(s.as_ptr() as *const __m256i);

        let pairs: [(__m256i, __m256i, &str); 3] = [
            (arch::_mm256_sllv_epi16x_v3(a, b), arch::_mm256_sllv_epi16_widex_v3(a, b), "sllv"),
            (arch::_mm256_srlv_epi16x_v3(a, b), arch::_mm256_srlv_epi16_widex_v3(a, b), "srlv"),
            (arch::_mm256_srav_epi16x_v3(a, b), arch::_mm256_srav_epi16_widex_v3(a, b), "srav"),
        ];

        for (dec, wide, name) in pairs {
            let mut d = [0u16; 16];
            let mut w = [0u16; 16];
            _mm256_storeu_si256(d.as_mut_ptr() as *mut __m256i, dec);
            _mm256_storeu_si256(w.as_mut_ptr() as *mut __m256i, wide);
            assert_eq!(
                d, w,
                "256-bit {name} lowerings disagree\n  values = {v:?}\n  shifts = {s:?}"
            );
        }

        // 128-bit halves, same contract.
        let a4 = _mm_loadu_si128(v.as_ptr() as *const __m128i);
        let b4 = _mm_loadu_si128(s.as_ptr() as *const __m128i);
        let pairs128: [(__m128i, __m128i, &str); 3] = [
            (arch::_mm_sllv_epi16x_v1(a4, b4), arch::_mm_sllv_epi16_widex_v3(a4, b4), "sllv"),
            (arch::_mm_srlv_epi16x_v1(a4, b4), arch::_mm_srlv_epi16_widex_v3(a4, b4), "srlv"),
            (arch::_mm_srav_epi16x_v1(a4, b4), arch::_mm_srav_epi16_widex_v3(a4, b4), "srav"),
        ];
        for (dec, wide, name) in pairs128 {
            let mut d = [0u16; 8];
            let mut w = [0u16; 8];
            _mm_storeu_si128(d.as_mut_ptr() as *mut __m128i, dec);
            _mm_storeu_si128(w.as_mut_ptr() as *mut __m128i, wide);
            assert_eq!(d, w, "128-bit {name} lowerings disagree");
        }
    }
}}

#[test]
fn lowerings_agree() {
    if !std::arch::is_x86_feature_detected!("avx2") {
        eprintln!("skipping: no AVX2");
        return;
    }
    unsafe { run() }
}
