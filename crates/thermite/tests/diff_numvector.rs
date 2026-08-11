//! Coverage for `NumVector<V>` (`vector/num.rs`), the `num_traits` compatibility
//! wrapper (was ~2%). The interesting logic is the all/any reduction semantics:
//! `is_zero`/`is_one`/`eq` are **all-lane**, `ne`/`is_nan`/`is_sign_negative` are
//! **any-lane**, plus `partial_cmp` and `classify`. The forwarded arithmetic /
//! math ops just need to execute (correctness is covered elsewhere).
// `num_traits::Float for NumVector<V>` is `#[cfg(feature = "std")]`-gated in
// `vector/num.rs`, but the dev-dependency on num-traits enables *its* `std`, so the
// trait exists here regardless. Without this gate the file fails to compile under a
// plain `cargo check --all-targets` (196 x E0277) while passing under `--features std`.
#![cfg(feature = "std")]
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

use core::cmp::Ordering;
use core::num::FpCategory;

use num_traits::float::{Float, FloatCore};
use num_traits::{Bounded, ConstOne, ConstZero, MulAdd, Num, NumCast, One, ToPrimitive, Zero};

use thermite::Vector;
use thermite::mask::GenericSelectable;
use thermite::prelude::*;
use thermite::simd::Simd;
use thermite::vector::NumVector;

use thermite::backend::scalar::Scalar;

macro_rules! numvector_suite {
    ($mod:ident, $backend:ty) => {
        mod $mod {
            use super::*;
            type VF = Vector<<$backend as Simd>::f32x4>;
            type VI = Vector<<$backend as Simd>::i32x4>;
            type NF = NumVector<VF>;
            type NI = NumVector<VI>;

            fn nf(a: [f32; 4]) -> NF {
                NumVector(VF::new(a))
            }
            fn ni(a: [i32; 4]) -> NI {
                NumVector(VI::new(a))
            }
            fn rdf(v: NF) -> [f32; 4] {
                let g = v.0.into_array();
                [g[0], g[1], g[2], g[3]]
            }
            fn rdi(v: NI) -> [i32; 4] {
                let g = v.0.into_array();
                [g[0], g[1], g[2], g[3]]
            }

            #[test]
            fn operators() {
                let a = ni([10, 20, 30, 40]);
                let b = ni([1, 2, 3, 4]);
                assert_eq!(rdi(a + b), [11, 22, 33, 44]);
                assert_eq!(rdi(a - b), [9, 18, 27, 36]);
                assert_eq!(rdi(a * b), [10, 40, 90, 160]);
                assert_eq!(rdi(a / b), [10, 10, 10, 10]);
                assert_eq!(rdi(a % b), [0, 0, 0, 0]);
                assert_eq!(rdi(-b), [-1, -2, -3, -4]);
                // bitwise
                let c = ni([0b1100, 0b1010, 0xFF, 0x0F]);
                let d = ni([0b1010, 0b0110, 0x0F, 0xFF]);
                assert_eq!(rdi(c & d), [0b1000, 0b0010, 0x0F, 0x0F]);
                assert_eq!(rdi(c | d), [0b1110, 0b1110, 0xFF, 0xFF]);
                assert_eq!(rdi(c ^ d), [0b0110, 0b1100, 0xF0, 0xF0]);
                // shifts (u32 and NumVector<Unsigned>)
                assert_eq!(rdi(b << 2u32), [4, 8, 12, 16]);
                assert_eq!(rdi(ni([16, 32, 48, 64]) >> 2u32), [4, 8, 12, 16]);
                let sh = NumVector(<VI as GenericVector>::Unsigned::splat(1u32));
                assert_eq!(rdi(b << sh), [2, 4, 6, 8]);
                // assign forms
                let mut t = a;
                t += b;
                assert_eq!(rdi(t), [11, 22, 33, 44]);
                let mut t = c;
                t &= d;
                assert_eq!(rdi(t), [0b1000, 0b0010, 0x0F, 0x0F]);
                let mut t = b;
                t <<= 1u32;
                assert_eq!(rdi(t), [2, 4, 6, 8]);
            }

            #[test]
            fn identities_and_bounds() {
                assert!(<NI as Zero>::zero().is_zero());
                assert!(!ni([0, 1, 0, 0]).is_zero()); // not ALL zero
                assert!(<NI as One>::one().is_one());
                assert!(!ni([1, 1, 2, 1]).is_one()); // not ALL one
                assert_eq!(rdi(<NI as ConstZero>::ZERO), [0; 4]);
                assert_eq!(rdi(<NI as ConstOne>::ONE), [1; 4]);
                assert_eq!(rdi(<NI as Bounded>::min_value()), [i32::MIN; 4]);
                assert_eq!(rdi(<NI as Bounded>::max_value()), [i32::MAX; 4]);
                // Num::from_str_radix splats a parsed scalar
                let parsed = <NI as Num>::from_str_radix("7", 10).unwrap();
                assert_eq!(rdi(parsed), [7; 4]);
                // NumCast splats
                let cast: NI = NumCast::from(5i32).unwrap();
                assert_eq!(rdi(cast), [5; 4]);
            }

            #[test]
            fn equality_and_ordering() {
                let a = ni([1, 2, 3, 4]);
                // eq: all lanes equal; ne: any lane differs
                assert!(a == ni([1, 2, 3, 4]));
                assert!(a != ni([1, 2, 3, 5]));
                assert!(!(a == ni([1, 2, 3, 5]))); // one lane differs -> not all-equal
                // partial_cmp: Some only when ALL lanes share the relation
                assert_eq!(a.partial_cmp(&ni([5, 6, 7, 8])), Some(Ordering::Less));
                assert_eq!(a.partial_cmp(&ni([0, 1, 2, 3])), Some(Ordering::Greater));
                assert_eq!(a.partial_cmp(&ni([1, 2, 3, 4])), Some(Ordering::Equal));
                assert_eq!(a.partial_cmp(&ni([5, 1, 7, 8])), None); // mixed
            }

            #[test]
            fn to_primitive() {
                // ToPrimitive extracts lane 0
                let a = ni([42, 1, 2, 3]);
                assert_eq!(a.to_i32(), Some(42));
                assert_eq!(a.to_i64(), Some(42));
                assert_eq!(a.to_f64(), Some(42.0));
                assert_eq!(a.to_u32(), Some(42));
                let f = nf([2.5, 0.0, 0.0, 0.0]);
                assert_eq!(f.to_f32(), Some(2.5));
                assert_eq!(f.to_i32(), Some(2));
            }

            #[test]
            fn floatcore_classify_and_predicates() {
                // classify priority: NaN > Inf > Zero(all) > Subnormal > Normal
                assert_eq!(
                    FloatCore::classify(nf([f32::NAN, 1.0, 2.0, 3.0])),
                    FpCategory::Nan
                );
                assert_eq!(
                    FloatCore::classify(nf([f32::INFINITY, 1.0, 2.0, 3.0])),
                    FpCategory::Infinite
                );
                assert_eq!(FloatCore::classify(nf([0.0, 0.0, 0.0, 0.0])), FpCategory::Zero);
                assert_eq!(
                    FloatCore::classify(nf([f32::MIN_POSITIVE / 2.0, 0.0, 0.0, 0.0])),
                    FpCategory::Subnormal
                );
                assert_eq!(FloatCore::classify(nf([1.0, 2.0, 3.0, 4.0])), FpCategory::Normal);

                // any/all predicates
                assert!(FloatCore::is_nan(nf([f32::NAN, 1.0, 1.0, 1.0])));
                assert!(!FloatCore::is_finite(nf([f32::INFINITY, 1.0, 1.0, 1.0])));
                assert!(FloatCore::is_finite(nf([1.0, 2.0, 3.0, 4.0])));
                assert!(FloatCore::is_infinite(nf([f32::INFINITY, 1.0, 1.0, 1.0])));
                assert!(FloatCore::is_normal(nf([1.0, 2.0, 3.0, 4.0])));
                assert!(Float::is_sign_negative(nf([-1.0, 2.0, 3.0, 4.0]))); // any negative
                assert!(Float::is_sign_positive(nf([1.0, 2.0, 3.0, 4.0]))); // all positive

                // FloatCore rounding/sign + to_degrees/recip
                assert_eq!(
                    rdf(FloatCore::floor(nf([1.7, 2.2, -1.3, 3.9]))),
                    [1.0, 2.0, -2.0, 3.0]
                );
                assert_eq!(
                    rdf(FloatCore::abs(nf([-1.0, 2.0, -3.0, 4.0]))),
                    [1.0, 2.0, 3.0, 4.0]
                );
                let r = rdf(FloatCore::recip(nf([2.0, 4.0, 5.0, 8.0])));
                assert!((r[0] - 0.5).abs() < 1e-4 && (r[1] - 0.25).abs() < 1e-4);

                // remaining FloatCore rounding/sign + constants (execute-only / spot checks)
                let m = nf([1.7, 2.2, -1.3, 3.9]);
                assert_eq!(rdf(FloatCore::ceil(m)), [2.0, 3.0, -1.0, 4.0]);
                assert_eq!(rdf(FloatCore::trunc(m)), [1.0, 2.0, -1.0, 3.0]);
                let _ = FloatCore::round(m);
                let _ = FloatCore::fract(m);
                assert_eq!(
                    rdf(FloatCore::signum(nf([-2.0, 3.0, -4.0, 5.0]))),
                    [-1.0, 1.0, -1.0, 1.0]
                );
                assert_eq!(
                    rdf(FloatCore::min(nf([1.0, 5.0, 2.0, 8.0]), nf([3.0, 2.0, 9.0, 1.0]))),
                    [1.0, 2.0, 2.0, 1.0]
                );
                assert_eq!(
                    rdf(FloatCore::max(nf([1.0, 5.0, 2.0, 8.0]), nf([3.0, 2.0, 9.0, 1.0]))),
                    [3.0, 5.0, 9.0, 8.0]
                );
                let _ = FloatCore::to_degrees(nf([1.0; 4]));
                let _ = FloatCore::to_radians(nf([1.0; 4]));
                let _ = FloatCore::integer_decode(nf([1.0; 4]));
                let _ = <NF as FloatCore>::infinity();
                let _ = <NF as FloatCore>::neg_infinity();
                let _ = <NF as FloatCore>::nan();
                let _ = <NF as FloatCore>::neg_zero();
                let _ = <NF as FloatCore>::min_value();
                let _ = <NF as FloatCore>::max_value();
                let _ = <NF as FloatCore>::min_positive_value();
                let _ = <NF as FloatCore>::epsilon();
                // Float-trait constant constructors too
                let _ = <NF as Float>::infinity();
                let _ = <NF as Float>::neg_infinity();
                let _ = <NF as Float>::nan();
                let _ = <NF as Float>::neg_zero();
                let _ = <NF as Float>::min_value();
                let _ = <NF as Float>::max_value();
                let _ = <NF as Float>::min_positive_value();
                let _ = <NF as Float>::epsilon();
                assert!(matches!(Float::classify(nf([0.0; 4])), FpCategory::Zero));
            }

            #[test]
            fn float_math_forwards() {
                // These just need to execute; correctness is covered by diff_math.
                let a = nf([1.0, 2.0, 3.0, 4.0]);
                let b = nf([0.5, 0.5, 0.5, 0.5]);
                let _ = Float::sqrt(a);
                let _ = Float::exp(b);
                let _ = Float::ln(a);
                let _ = Float::powf(a, b);
                let _ = Float::powi(a, 3);
                let _ = Float::hypot(a, b);
                let _ = Float::atan2(a, b);
                let (_s, _c) = Float::sin_cos(a);
                let _ = Float::cbrt(a);
                let _ = Float::recip(a);
                let _ = Float::abs_sub(a, b);
                let _ = Float::clamp(a, nf([0.0; 4]), nf([2.0; 4]));
                let _ = Float::copysign(a, nf([-1.0; 4]));
                // remaining transcendental forwards (execute-only)
                let _ = Float::exp2(b);
                let _ = Float::log(a, nf([2.0; 4]));
                let _ = Float::log2(a);
                let _ = Float::log10(a);
                let _ = Float::exp_m1(b);
                let _ = Float::ln_1p(b);
                let _ = Float::sin(a);
                let _ = Float::cos(a);
                let _ = Float::tan(b);
                let _ = Float::asin(b); // |b| <= 1
                let _ = Float::acos(b);
                let _ = Float::atan(a);
                let _ = Float::sinh(b);
                let _ = Float::cosh(b);
                let _ = Float::tanh(b);
                let _ = Float::asinh(b);
                let _ = Float::acosh(a); // a >= 1
                let _ = Float::atanh(b); // |b| < 1
                let _ = Float::to_degrees(a);
                let _ = Float::to_radians(a);
                let _ = Float::min(a, b);
                let _ = Float::max(a, b);
                let _ = Float::integer_decode(a);
                let _ = Float::is_subnormal(a);
                // mul_add via both the num_traits::MulAdd and Float::mul_add paths
                let m1 = rdf(MulAdd::mul_add(a, b, nf([1.0; 4])));
                let m2 = rdf(Float::mul_add(a, b, nf([1.0; 4])));
                assert_eq!(m1, m2);
                assert_eq!(m1, [1.5, 2.0, 2.5, 3.0]); // 1*0.5+1, 2*0.5+1, ...
                let mut t = a;
                num_traits::MulAddAssign::mul_add_assign(&mut t, b, nf([1.0; 4]));
                assert_eq!(rdf(t), [1.5, 2.0, 2.5, 3.0]);
            }

            #[test]
            fn deref_and_select() {
                // Deref exposes the underlying vector's inherent methods.
                let a = nf([3.0, 1.0, 4.0, 2.0]);
                assert_eq!(a.sum_elements(), 10.0);
                // GenericSelectable on the wrapper
                let mask = a.0.cmp_lt(VF::splat(3.0)); // lanes 1 (1.0) and 3 (2.0) true
                let sel = <NF as GenericSelectable>::select(mask, nf([1.0; 4]), nf([0.0; 4]));
                assert_eq!(rdf(sel), [0.0, 1.0, 0.0, 1.0]);
            }
        }
    };
}

numvector_suite!(scalar, Scalar);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    use thermite::backend::x86_v1::X86V1;
    use thermite::backend::x86_v2::X86V2;
    use thermite::backend::x86_v3::X86V3;
    numvector_suite!(v3, X86V3);
    numvector_suite!(v2, X86V2);
    numvector_suite!(v1, X86V1);
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use thermite::backend::wasm::Wasm;
    numvector_suite!(wasm, Wasm);
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use thermite::backend::neon::Neon;
    numvector_suite!(neon, Neon);
}
