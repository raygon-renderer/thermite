//! Coverage for the `ScalarMath` / `ScalarMathWithPolicy` shortcut traits (the
//! `scalar_*` methods on bare `f32`/`f64`) and the `Unwrap` plumbing they use
//! (`math/scalar.rs`). The tuple-returning ones (`scalar_sin_cos`, `scalar_frexp`,
//! ...) exercise the tuple `Unwrap` impls.
//!
//! Spot-checked against `libm` (loose `Performance`-policy tolerance); the rest
//! just need to execute. `ScalarMath` is implemented directly on the scalar types.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

use thermite::math::policy::policies::Performance;
use thermite::math::{ScalarMath, ScalarMathWithPolicy};

macro_rules! suite {
    ($name:ident, $e:ty, $sin:path, $cos:path, $exp:path, $ln:path, $atan2:path, $hypot:path, $tol:expr) => {
        #[test]
        fn $name() {
            let x: $e = 1.3;
            let close = |g: $e, w: $e| {
                assert!(
                    (g as f64 - w as f64).abs() <= $tol * (w as f64).abs().max(1.0),
                    "got {g} want {w}"
                )
            };

            // unary, spot-checked vs libm
            close(x.scalar_sin(), $sin(x));
            close(x.scalar_cos(), $cos(x));
            close(x.scalar_exp(), $exp(x));
            close(x.scalar_ln(), $ln(x));

            // tuple-returning -> exercises the tuple `Unwrap` impls
            let (s, c) = x.scalar_sin_cos();
            close(s, $sin(x));
            close(c, $cos(x));
            let (sh, ch) = x.scalar_sinh_cosh();
            assert!((sh as f64).is_finite() && (ch as f64) >= 1.0);
            // frexp: x == mantissa * 2^exp, mantissa in [0.5, 1)
            let (m, e) = x.scalar_frexp();
            let recon = (m as f64) * 2f64.powi(e as i32);
            assert!(
                (recon - x as f64).abs() <= 1e-6 && (m as f64).abs() >= 0.5 && (m as f64).abs() < 1.0,
                "frexp {m} {e}"
            );
            // ldexp inverts frexp's exponent
            close(m.scalar_ldexp(e), x);

            // binary
            close(x.scalar_atan2(0.5 as $e), $atan2(x, 0.5 as $e));
            close(<$e>::scalar_hypot_n::<2>([x, 0.5 as $e]), $hypot(x, 0.5 as $e));

            // a broad sweep that only needs to execute (correctness covered by diff_math)
            let _ = x.scalar_tan();
            let _ = x.scalar_exp2();
            let _ = x.scalar_log2();
            let _ = x.scalar_log10();
            let _ = x.scalar_cbrt();
            let _ = x.scalar_inverse_sqrt();
            let _ = (0.5 as $e).scalar_asin();
            let _ = (0.5 as $e).scalar_acos();
            let _ = x.scalar_atan();
            let _ = x.scalar_sinh();
            let _ = x.scalar_cosh();
            let _ = x.scalar_tanh();
            let _ = x.scalar_asinh();
            let _ = x.scalar_acosh();
            let _ = (0.5 as $e).scalar_atanh();
            let _ = x.scalar_exp_m1();
            let _ = (0.5 as $e).scalar_ln_1p();
            let _ = x.scalar_to_degrees();
            let _ = x.scalar_to_radians();
            let _ = x.scalar_approx_reciprocal();
            let _ = x.scalar_powf(2.0 as $e);
            let _ = x.scalar_lerp(0.0 as $e, 10.0 as $e);

            // policy-aware variants (ScalarMathWithPolicy)
            close(x.scalar_sin_p::<Performance>(), $sin(x));
            let _ = x.scalar_exp_p::<Performance>();
            let (sp, cp) = x.scalar_sin_cos_p::<Performance>();
            close(sp, $sin(x));
            close(cp, $cos(x));
        }
    };
}

suite!(
    f32_scalar_math,
    f32,
    libm::sinf,
    libm::cosf,
    libm::expf,
    libm::logf,
    libm::atan2f,
    libm::hypotf,
    2.0e-3
);
suite!(
    f64_scalar_math,
    f64,
    libm::sin,
    libm::cos,
    libm::exp,
    libm::log,
    libm::atan2,
    libm::hypot,
    1.0e-6
);

// Diagnostic: does ldexp/frexp round-trip at the *vector* (FloatVectorWithBits) layer?
// If this passes but `scalar_ldexp` returns 0, the bug is in the scalar `Unwrap` of the
// SignedBits exponent argument, not in ldexp itself.
#[test]
fn vector_ldexp_roundtrip() {
    use thermite::Vector;
    use thermite::prelude::*;
    use thermite::simd::Simd;
    type V = Vector<<thermite::backend::scalar::Scalar as Simd>::f32x4>;
    type U = <V as GenericVector>::Unsigned; // scalar u32x4 (ArrayRegister<u32,4>)

    // Primitive diagnostics on the scalar bits register (used by the bit-manip ldexp):
    let u = U::new([0x3F26_6666, 8, 0xFF00_00FF, 1]);
    assert_eq!(
        (u >> 23u32).into_array().as_slice(),
        &[0x7E, 0, 0x1FE, 0],
        "scalar u32 shr"
    );
    assert_eq!(
        (U::new([1, 2, 3, 0x7F]) << 23u32).into_array().as_slice(),
        &[1 << 23, 2 << 23, 3 << 23, 0x7F << 23],
        "scalar u32 shl"
    );
    let f = V::new([1.3, -2.5, 0.1, 7.0]);
    assert_eq!(
        V::from_bits(f.into_bits::<U>()).into_array().as_slice(),
        f.into_array().as_slice(),
        "scalar bitcast roundtrip"
    );

    let v = V::new([1.3, 2.5, 0.1, 7.0]);
    let (m, e) = v.frexp();
    let back = m.ldexp(e);
    assert_eq!(
        back.into_array().as_slice(),
        v.into_array().as_slice(),
        "vector ldexp(frexp(x)) round-trip"
    );
}
