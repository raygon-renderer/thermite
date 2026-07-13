//! Coverage for the `Vector<R>` wrapper surface in `vector/vector.rs` that the
//! other suites don't reach: utility/accessor methods, the `square_*` and
//! `div_*` **masked** variants, the `num_traits` impls implemented directly on
//! `Vector<R>` (Zero/One/Saturating/Wrapping/Bounded/Sum/Product/Index/PartialEq),
//! and `total_order`/`linear_order`.
//!
//! These impls are generic over `R`, so a single concrete instantiation per impl
//! covers its source lines; we use `X86V3` (+ `Scalar` for the masked ops, whose
//! `_c`/`_m`/`_z` lowering differs from the x86 blend path).
#![cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "wasm32", all(feature = "neon", target_arch = "aarch64")))]

use num_traits::{Bounded, One, Saturating, SaturatingAdd, SaturatingSub, WrappingAdd, WrappingMul, WrappingSub, Zero};

use thermite::Vector;
use thermite::mask::Mask;
use thermite::prelude::*;
use thermite::register::Storage;
use thermite::simd::Simd;
use thermite::vector::{Concat, Extend, VectorWithRegister};

use thermite::backend::scalar::Scalar;

// helper: the element value `3`, generically (for the mask threshold in square_masked)
fn num_traits_three<R: thermite::register::NumericRegister>() -> R::Element
where
    Vector<R>: NumericVector<Element = R::Element>,
{
    // 1 + 1 + 1
    let three = Vector::<R>::ONE + Vector::<R>::ONE + Vector::<R>::ONE;
    three.extract::<0>()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    use thermite::backend::x86_v3::X86V3;

    type VI = Vector<<X86V3 as Simd>::i32x4>;
    type VF = Vector<<X86V3 as Simd>::f32x4>;

    #[test]
    fn utility_and_accessors() {
        let v = VI::new([3, 1, 4, 1]);

        // accessors
        assert_eq!(v.as_slice(), &[3, 1, 4, 1]);
        assert_eq!(v.as_slice(), &[3, 1, 4, 1]);
        assert_eq!(v.into_array().as_slice(), &[3, 1, 4, 1]);
        let mut m = v;
        m.as_mut_slice()[0] = 9;
        m.as_mut_slice()[1] = 8;
        assert_eq!(m.as_slice(), &[9, 8, 4, 1]);

        // splat_const (const fn — exercise at runtime), default, clone, Debug
        const C: VI = VI::splat_const(7);
        assert_eq!(C.as_slice(), &[7; 4]);
        assert_eq!(VI::splat_const(5).as_slice(), &[5; 4]);
        assert!(VI::default().as_slice().iter().all(|&x| x == 0));
        #[allow(clippy::clone_on_copy)]
        let c = v.clone();
        assert_eq!(c.as_slice(), v.as_slice());
        assert!(format!("{v:?}").contains('3'));

        // offset / indexed
        let _ = VI::offset();
        assert_eq!(VI::indexed().as_slice(), &[0, 1, 2, 3]);

        // into_register / from_register round-trip
        let reg: Storage<<X86V3 as Simd>::i32x4> = v.into_register();
        let back = VI::from_register(reg);
        assert_eq!(back.as_slice(), v.as_slice());
    }

    #[test]
    fn unsigned_predicates_dividers_fastcast() {
        type VU = Vector<<X86V3 as Simd>::u32x4>;

        // is_power_of_two (UnsignedIntegerVector)
        let u = VU::new([1, 3, 8, 0]);
        let pot = u.is_power_of_two().select(VU::ONE, VU::ZERO).into_array();
        // 1 and 8 are powers of two; 3 is not. (0 is reported as a power of two — a known quirk.)
        assert_eq!(pot.as_slice(), &[1, 0, 1, 1]);

        // create_divider / create_branchfree_divider (NumericVector helpers)
        let d = VI::create_divider(7);
        let bf = VI::create_branchfree_divider(7);
        let v = VI::new([14, 21, 30, 49]);
        assert_eq!((v / d).into_array().as_slice(), &[2, 3, 4, 7]);
        assert_eq!((v / bf).into_array().as_slice(), &[2, 3, 4, 7]);

        // fast_cast (relaxed numeric cast) i32 -> f32
        let f: VF = VI::new([1, 2, 3, 4]).fast_cast_into();
        assert_eq!(f.into_array().as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn float_bit_orderings() {
        // total_order / linear_order map floats to a sign-magnitude-monotone integer
        // key: a < b (as floats) => key(a) < key(b). Check the ordering is preserved.
        let xs = VF::new([-2.0, -0.0, 1.5, 3.0]);
        let to = xs.total_order().into_array();
        let lo = xs.linear_order().into_array();
        for ord in [to, lo] {
            assert!(
                ord[0] < ord[1] && ord[1] <= ord[2] && ord[2] < ord[3],
                "ordering not monotone: {ord:?}"
            );
        }
    }

    #[test]
    fn square_masked() {
        fn check<R>()
        where
            R: thermite::register::NumericRegister,
            Vector<R>: NumericVector<Element = R::Element>,
            R::Element: PartialEq + core::fmt::Debug,
        {
            let v = Vector::<R>::indexed() + Vector::<R>::ONE; // [1, 2, 3, ...]
            let sq = v.square();
            let src = Vector::<R>::TWO; // explicit non-zero merge source
            let mask = v.cmp_lt(Vector::<R>::splat(num_traits_three::<R>())); // first two lanes true
            // _c: mask ? v² : v ; _m: mask ? v² : src ; _z: mask ? v² : 0
            let zero = Vector::<R>::ZERO;
            let c = v.square_c(mask);
            let m = v.square_m(src, mask);
            let z = v.square_z(mask);
            let exp_c = mask.select(sq, v).into_array();
            let exp_m = mask.select(sq, src).into_array();
            let exp_z = mask.select(sq, zero).into_array();
            assert_eq!(c.into_array().as_slice(), exp_c.as_slice());
            assert_eq!(m.into_array().as_slice(), exp_m.as_slice());
            assert_eq!(z.into_array().as_slice(), exp_z.as_slice());
        }
        check::<<X86V3 as Simd>::i32x4>();
        check::<<Scalar as Simd>::i32x4>();
        check::<<X86V3 as Simd>::f32x4>();
    }

    #[test]
    fn div_masked_constant_dividers() {
        use thermite::{BranchfreeDivider, Divider};
        let v = VI::new([10, 20, 30, 40]);
        let mask = v.cmp_gt(VI::splat(15)); // [F, T, T, T]
        let d = Divider::i32(7);
        let bf = BranchfreeDivider::i32(7);
        let q = VI::new([10 / 7, 20 / 7, 30 / 7, 40 / 7]);

        // Divider masked variants
        assert_eq!(
            v.div_c(mask, d).into_array().as_slice(),
            mask.select(q, v).into_array().as_slice()
        );
        assert_eq!(
            v.div_m(VI::splat(-1), mask, d).into_array().as_slice(),
            mask.select(q, VI::splat(-1)).into_array().as_slice()
        );
        assert_eq!(
            v.div_z(mask, d).into_array().as_slice(),
            mask.select(q, VI::ZERO).into_array().as_slice()
        );

        // BranchfreeDivider masked variants
        assert_eq!(
            v.div_c(mask, bf).into_array().as_slice(),
            mask.select(q, v).into_array().as_slice()
        );
        assert_eq!(
            v.div_m(VI::splat(-1), mask, bf).into_array().as_slice(),
            mask.select(q, VI::splat(-1)).into_array().as_slice()
        );
        assert_eq!(
            v.div_z(mask, bf).into_array().as_slice(),
            mask.select(q, VI::ZERO).into_array().as_slice()
        );
    }

    #[test]
    fn mask_reshape_and_store_masked() {
        type VI8 = Vector<<X86V3 as Simd>::i32x8>;
        type MI8 = Mask<<X86V3 as Simd>::i32x8>;

        let lo = VI::new([1, 0, 1, 0]).cmp_ne(VI::ZERO); // [T,F,T,F]
        let hi = VI::new([0, 0, 1, 1]).cmp_ne(VI::ZERO); // [F,F,T,T]
        let rb4 = |m: Mask<<X86V3 as Simd>::i32x4>| m.select(VI::ONE, VI::ZERO).into_array().as_slice().to_vec();
        let rb8 = |m: MI8| m.select(VI8::ONE, VI8::ZERO).into_array().as_slice().to_vec();

        // Concat: low half = lo, high half = hi
        let wide: MI8 = Concat::concat(lo, hi);
        assert_eq!(rb8(wide), vec![1, 0, 1, 0, 0, 0, 1, 1]);
        // split is the inverse
        let (slo, shi) = Concat::split(wide);
        assert_eq!(rb4(slo), vec![1, 0, 1, 0]);
        assert_eq!(rb4(shi), vec![0, 0, 1, 1]);
        // extend: low half = lo, high half = false; narrow keeps the low half
        let ext: MI8 = Extend::extend(lo);
        assert_eq!(rb8(ext), vec![1, 0, 1, 0, 0, 0, 0, 0]);
        assert_eq!(rb4(ext.narrow()), vec![1, 0, 1, 0]);

        // store_masked: only lanes where the mask is true are written
        let v = VI::new([11, 22, 33, 44]);
        let mut buf = [0i32; 4];
        unsafe { v.store_masked(lo, buf.as_mut_ptr()) };
        assert_eq!(buf, [11, 0, 33, 0]);
    }

    #[test]
    fn vector_num_traits() {
        let a = VI::new([5, -3, 8, 2]);
        let b = VI::new([1, 4, 8, 2]);

        // Zero
        assert!(<VI as Zero>::zero().as_slice().iter().all(|&x| x == 0));
        assert!(Zero::is_zero(&VI::ZERO));
        assert!(!Zero::is_zero(&a));
        let mut z = a;
        z.set_zero();
        assert!(Zero::is_zero(&z));

        // One
        assert!(<VI as One>::one().as_slice().iter().all(|&x| x == 1));
        assert!(One::is_one(&VI::ONE));
        assert!(!One::is_one(&a));
        let mut o = a;
        o.set_one();
        assert!(One::is_one(&o));

        // Bounded
        assert_eq!(<VI as Bounded>::max_value().as_slice(), &[i32::MAX; 4]);
        assert_eq!(<VI as Bounded>::min_value().as_slice(), &[i32::MIN; 4]);

        // Saturating (by value) + SaturatingAdd/Sub (by ref)
        let big = VI::splat(i32::MAX);
        assert_eq!(Saturating::saturating_add(big, VI::ONE).as_slice(), &[i32::MAX; 4]);
        assert_eq!(
            Saturating::saturating_sub(VI::splat(i32::MIN), VI::ONE).as_slice(),
            &[i32::MIN; 4]
        );
        assert_eq!(SaturatingAdd::saturating_add(&big, &VI::ONE).as_slice(), &[i32::MAX; 4]);
        assert_eq!(
            SaturatingSub::saturating_sub(&VI::splat(i32::MIN), &VI::ONE).as_slice(),
            &[i32::MIN; 4]
        );

        // Wrapping add/sub/mul
        assert_eq!(WrappingAdd::wrapping_add(&a, &b).as_slice(), &[6, 1, 16, 4]);
        assert_eq!(WrappingSub::wrapping_sub(&a, &b).as_slice(), &[4, -7, 0, 0]);
        assert_eq!(WrappingMul::wrapping_mul(&a, &b).as_slice(), &[5, -12, 64, 4]);

        // PartialEq (Vector == Vector): all lanes equal
        assert!(VI::splat(2) == VI::splat(2));
        assert!(VI::splat(2) != a);

        // Index / IndexMut
        assert_eq!(a[2], 8);
        let mut im = a;
        im[0] = 99;
        assert_eq!(im[0], 99);

        // Sum / Product over an iterator of vectors (lane-wise)
        let vs = [VI::new([1, 2, 3, 4]), VI::new([10, 20, 30, 40]), VI::splat(100)];
        let s: VI = vs.into_iter().sum();
        assert_eq!(s.as_slice(), &[111, 122, 133, 144]);
        let p: VI = [VI::splat(2), VI::splat(3), VI::new([1, 2, 3, 4])]
            .into_iter()
            .product();
        assert_eq!(p.as_slice(), &[6, 12, 18, 24]);
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use thermite::backend::wasm::Wasm;

    type VI = Vector<<Wasm as Simd>::i32x4>;
    type VF = Vector<<Wasm as Simd>::f32x4>;

    #[test]
    fn utility_and_accessors() {
        let v = VI::new([3, 1, 4, 1]);

        // accessors
        assert_eq!(v.as_slice(), &[3, 1, 4, 1]);
        assert_eq!(v.as_slice(), &[3, 1, 4, 1]);
        assert_eq!(v.into_array().as_slice(), &[3, 1, 4, 1]);
        let mut m = v;
        m.as_mut_slice()[0] = 9;
        m.as_mut_slice()[1] = 8;
        assert_eq!(m.as_slice(), &[9, 8, 4, 1]);

        // splat_const (const fn — exercise at runtime), default, clone, Debug
        const C: VI = VI::splat_const(7);
        assert_eq!(C.as_slice(), &[7; 4]);
        assert_eq!(VI::splat_const(5).as_slice(), &[5; 4]);
        assert!(VI::default().as_slice().iter().all(|&x| x == 0));
        #[allow(clippy::clone_on_copy)]
        let c = v.clone();
        assert_eq!(c.as_slice(), v.as_slice());
        assert!(format!("{v:?}").contains('3'));

        // offset / indexed
        let _ = VI::offset();
        assert_eq!(VI::indexed().as_slice(), &[0, 1, 2, 3]);

        // into_register / from_register round-trip
        let reg: Storage<<Wasm as Simd>::i32x4> = v.into_register();
        let back = VI::from_register(reg);
        assert_eq!(back.as_slice(), v.as_slice());
    }

    #[test]
    fn unsigned_predicates_dividers_fastcast() {
        type VU = Vector<<Wasm as Simd>::u32x4>;

        // is_power_of_two (UnsignedIntegerVector)
        let u = VU::new([1, 3, 8, 0]);
        let pot = u.is_power_of_two().select(VU::ONE, VU::ZERO).into_array();
        // 1 and 8 are powers of two; 3 is not. (0 is reported as a power of two — a known quirk.)
        assert_eq!(pot.as_slice(), &[1, 0, 1, 1]);

        // create_divider / create_branchfree_divider (NumericVector helpers)
        let d = VI::create_divider(7);
        let bf = VI::create_branchfree_divider(7);
        let v = VI::new([14, 21, 30, 49]);
        assert_eq!((v / d).into_array().as_slice(), &[2, 3, 4, 7]);
        assert_eq!((v / bf).into_array().as_slice(), &[2, 3, 4, 7]);

        // fast_cast (relaxed numeric cast) i32 -> f32
        let f: VF = VI::new([1, 2, 3, 4]).fast_cast_into();
        assert_eq!(f.into_array().as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn float_bit_orderings() {
        // total_order / linear_order map floats to a sign-magnitude-monotone integer
        // key: a < b (as floats) => key(a) < key(b). Check the ordering is preserved.
        let xs = VF::new([-2.0, -0.0, 1.5, 3.0]);
        let to = xs.total_order().into_array();
        let lo = xs.linear_order().into_array();
        for ord in [to, lo] {
            assert!(
                ord[0] < ord[1] && ord[1] <= ord[2] && ord[2] < ord[3],
                "ordering not monotone: {ord:?}"
            );
        }
    }

    #[test]
    fn square_masked() {
        fn check<R>()
        where
            R: thermite::register::NumericRegister,
            Vector<R>: NumericVector<Element = R::Element>,
            R::Element: PartialEq + core::fmt::Debug,
        {
            let v = Vector::<R>::indexed() + Vector::<R>::ONE; // [1, 2, 3, ...]
            let sq = v.square();
            let src = Vector::<R>::TWO; // explicit non-zero merge source
            let mask = v.cmp_lt(Vector::<R>::splat(num_traits_three::<R>())); // first two lanes true
            // _c: mask ? v² : v ; _m: mask ? v² : src ; _z: mask ? v² : 0
            let zero = Vector::<R>::ZERO;
            let c = v.square_c(mask);
            let m = v.square_m(src, mask);
            let z = v.square_z(mask);
            let exp_c = mask.select(sq, v).into_array();
            let exp_m = mask.select(sq, src).into_array();
            let exp_z = mask.select(sq, zero).into_array();
            assert_eq!(c.into_array().as_slice(), exp_c.as_slice());
            assert_eq!(m.into_array().as_slice(), exp_m.as_slice());
            assert_eq!(z.into_array().as_slice(), exp_z.as_slice());
        }
        check::<<Wasm as Simd>::i32x4>();
        check::<<Scalar as Simd>::i32x4>();
        check::<<Wasm as Simd>::f32x4>();
    }

    #[test]
    fn div_masked_constant_dividers() {
        use thermite::{BranchfreeDivider, Divider};
        let v = VI::new([10, 20, 30, 40]);
        let mask = v.cmp_gt(VI::splat(15)); // [F, T, T, T]
        let d = Divider::i32(7);
        let bf = BranchfreeDivider::i32(7);
        let q = VI::new([10 / 7, 20 / 7, 30 / 7, 40 / 7]);

        // Divider masked variants
        assert_eq!(
            v.div_c(mask, d).into_array().as_slice(),
            mask.select(q, v).into_array().as_slice()
        );
        assert_eq!(
            v.div_m(VI::splat(-1), mask, d).into_array().as_slice(),
            mask.select(q, VI::splat(-1)).into_array().as_slice()
        );
        assert_eq!(
            v.div_z(mask, d).into_array().as_slice(),
            mask.select(q, VI::ZERO).into_array().as_slice()
        );

        // BranchfreeDivider masked variants
        assert_eq!(
            v.div_c(mask, bf).into_array().as_slice(),
            mask.select(q, v).into_array().as_slice()
        );
        assert_eq!(
            v.div_m(VI::splat(-1), mask, bf).into_array().as_slice(),
            mask.select(q, VI::splat(-1)).into_array().as_slice()
        );
        assert_eq!(
            v.div_z(mask, bf).into_array().as_slice(),
            mask.select(q, VI::ZERO).into_array().as_slice()
        );
    }

    #[test]
    fn mask_reshape_and_store_masked() {
        type VI8 = Vector<<Wasm as Simd>::i32x8>;
        type MI8 = Mask<<Wasm as Simd>::i32x8>;

        let lo = VI::new([1, 0, 1, 0]).cmp_ne(VI::ZERO); // [T,F,T,F]
        let hi = VI::new([0, 0, 1, 1]).cmp_ne(VI::ZERO); // [F,F,T,T]
        let rb4 = |m: Mask<<Wasm as Simd>::i32x4>| m.select(VI::ONE, VI::ZERO).into_array().as_slice().to_vec();
        let rb8 = |m: MI8| m.select(VI8::ONE, VI8::ZERO).into_array().as_slice().to_vec();

        // Concat: low half = lo, high half = hi
        let wide: MI8 = Concat::concat(lo, hi);
        assert_eq!(rb8(wide), vec![1, 0, 1, 0, 0, 0, 1, 1]);
        // split is the inverse
        let (slo, shi) = Concat::split(wide);
        assert_eq!(rb4(slo), vec![1, 0, 1, 0]);
        assert_eq!(rb4(shi), vec![0, 0, 1, 1]);
        // extend: low half = lo, high half = false; narrow keeps the low half
        let ext: MI8 = Extend::extend(lo);
        assert_eq!(rb8(ext), vec![1, 0, 1, 0, 0, 0, 0, 0]);
        assert_eq!(rb4(ext.narrow()), vec![1, 0, 1, 0]);

        // store_masked: only lanes where the mask is true are written
        let v = VI::new([11, 22, 33, 44]);
        let mut buf = [0i32; 4];
        unsafe { v.store_masked(lo, buf.as_mut_ptr()) };
        assert_eq!(buf, [11, 0, 33, 0]);
    }

    #[test]
    fn vector_num_traits() {
        let a = VI::new([5, -3, 8, 2]);
        let b = VI::new([1, 4, 8, 2]);

        // Zero
        assert!(<VI as Zero>::zero().as_slice().iter().all(|&x| x == 0));
        assert!(Zero::is_zero(&VI::ZERO));
        assert!(!Zero::is_zero(&a));
        let mut z = a;
        z.set_zero();
        assert!(Zero::is_zero(&z));

        // One
        assert!(<VI as One>::one().as_slice().iter().all(|&x| x == 1));
        assert!(One::is_one(&VI::ONE));
        assert!(!One::is_one(&a));
        let mut o = a;
        o.set_one();
        assert!(One::is_one(&o));

        // Bounded
        assert_eq!(<VI as Bounded>::max_value().as_slice(), &[i32::MAX; 4]);
        assert_eq!(<VI as Bounded>::min_value().as_slice(), &[i32::MIN; 4]);

        // Saturating (by value) + SaturatingAdd/Sub (by ref)
        let big = VI::splat(i32::MAX);
        assert_eq!(Saturating::saturating_add(big, VI::ONE).as_slice(), &[i32::MAX; 4]);
        assert_eq!(
            Saturating::saturating_sub(VI::splat(i32::MIN), VI::ONE).as_slice(),
            &[i32::MIN; 4]
        );
        assert_eq!(SaturatingAdd::saturating_add(&big, &VI::ONE).as_slice(), &[i32::MAX; 4]);
        assert_eq!(
            SaturatingSub::saturating_sub(&VI::splat(i32::MIN), &VI::ONE).as_slice(),
            &[i32::MIN; 4]
        );

        // Wrapping add/sub/mul
        assert_eq!(WrappingAdd::wrapping_add(&a, &b).as_slice(), &[6, 1, 16, 4]);
        assert_eq!(WrappingSub::wrapping_sub(&a, &b).as_slice(), &[4, -7, 0, 0]);
        assert_eq!(WrappingMul::wrapping_mul(&a, &b).as_slice(), &[5, -12, 64, 4]);

        // PartialEq (Vector == Vector): all lanes equal
        assert!(VI::splat(2) == VI::splat(2));
        assert!(VI::splat(2) != a);

        // Index / IndexMut
        assert_eq!(a[2], 8);
        let mut im = a;
        im[0] = 99;
        assert_eq!(im[0], 99);

        // Sum / Product over an iterator of vectors (lane-wise)
        let vs = [VI::new([1, 2, 3, 4]), VI::new([10, 20, 30, 40]), VI::splat(100)];
        let s: VI = vs.into_iter().sum();
        assert_eq!(s.as_slice(), &[111, 122, 133, 144]);
        let p: VI = [VI::splat(2), VI::splat(3), VI::new([1, 2, 3, 4])]
            .into_iter()
            .product();
        assert_eq!(p.as_slice(), &[6, 12, 18, 24]);
    }
}

#[cfg(all(feature = "neon", target_arch = "aarch64"))]
mod neon {
    use super::*;
    use thermite::backend::neon::Neon;

    type VI = Vector<<Neon as Simd>::i32x4>;
    type VF = Vector<<Neon as Simd>::f32x4>;

    #[test]
    fn utility_and_accessors() {
        let v = VI::new([3, 1, 4, 1]);

        // accessors
        assert_eq!(v.as_slice(), &[3, 1, 4, 1]);
        assert_eq!(v.as_slice(), &[3, 1, 4, 1]);
        assert_eq!(v.into_array().as_slice(), &[3, 1, 4, 1]);
        let mut m = v;
        m.as_mut_slice()[0] = 9;
        m.as_mut_slice()[1] = 8;
        assert_eq!(m.as_slice(), &[9, 8, 4, 1]);

        // splat_const (const fn — exercise at runtime), default, clone, Debug
        const C: VI = VI::splat_const(7);
        assert_eq!(C.as_slice(), &[7; 4]);
        assert_eq!(VI::splat_const(5).as_slice(), &[5; 4]);
        assert!(VI::default().as_slice().iter().all(|&x| x == 0));
        #[allow(clippy::clone_on_copy)]
        let c = v.clone();
        assert_eq!(c.as_slice(), v.as_slice());
        assert!(format!("{v:?}").contains('3'));

        // offset / indexed
        let _ = VI::offset();
        assert_eq!(VI::indexed().as_slice(), &[0, 1, 2, 3]);

        // into_register / from_register round-trip
        let reg: Storage<<Neon as Simd>::i32x4> = v.into_register();
        let back = VI::from_register(reg);
        assert_eq!(back.as_slice(), v.as_slice());
    }

    #[test]
    fn unsigned_predicates_dividers_fastcast() {
        type VU = Vector<<Neon as Simd>::u32x4>;

        // is_power_of_two (UnsignedIntegerVector)
        let u = VU::new([1, 3, 8, 0]);
        let pot = u.is_power_of_two().select(VU::ONE, VU::ZERO).into_array();
        // 1 and 8 are powers of two; 3 is not. (0 is reported as a power of two — a known quirk.)
        assert_eq!(pot.as_slice(), &[1, 0, 1, 1]);

        // create_divider / create_branchfree_divider (NumericVector helpers)
        let d = VI::create_divider(7);
        let bf = VI::create_branchfree_divider(7);
        let v = VI::new([14, 21, 30, 49]);
        assert_eq!((v / d).into_array().as_slice(), &[2, 3, 4, 7]);
        assert_eq!((v / bf).into_array().as_slice(), &[2, 3, 4, 7]);

        // fast_cast (relaxed numeric cast) i32 -> f32
        let f: VF = VI::new([1, 2, 3, 4]).fast_cast_into();
        assert_eq!(f.into_array().as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn float_bit_orderings() {
        // total_order / linear_order map floats to a sign-magnitude-monotone integer
        // key: a < b (as floats) => key(a) < key(b). Check the ordering is preserved.
        let xs = VF::new([-2.0, -0.0, 1.5, 3.0]);
        let to = xs.total_order().into_array();
        let lo = xs.linear_order().into_array();
        for ord in [to, lo] {
            assert!(
                ord[0] < ord[1] && ord[1] <= ord[2] && ord[2] < ord[3],
                "ordering not monotone: {ord:?}"
            );
        }
    }

    #[test]
    fn square_masked() {
        fn check<R>()
        where
            R: thermite::register::NumericRegister,
            Vector<R>: NumericVector<Element = R::Element>,
            R::Element: PartialEq + core::fmt::Debug,
        {
            let v = Vector::<R>::indexed() + Vector::<R>::ONE; // [1, 2, 3, ...]
            let sq = v.square();
            let src = Vector::<R>::TWO; // explicit non-zero merge source
            let mask = v.cmp_lt(Vector::<R>::splat(num_traits_three::<R>())); // first two lanes true
            // _c: mask ? v² : v ; _m: mask ? v² : src ; _z: mask ? v² : 0
            let zero = Vector::<R>::ZERO;
            let c = v.square_c(mask);
            let m = v.square_m(src, mask);
            let z = v.square_z(mask);
            let exp_c = mask.select(sq, v).into_array();
            let exp_m = mask.select(sq, src).into_array();
            let exp_z = mask.select(sq, zero).into_array();
            assert_eq!(c.into_array().as_slice(), exp_c.as_slice());
            assert_eq!(m.into_array().as_slice(), exp_m.as_slice());
            assert_eq!(z.into_array().as_slice(), exp_z.as_slice());
        }
        check::<<Neon as Simd>::i32x4>();
        check::<<Scalar as Simd>::i32x4>();
        check::<<Neon as Simd>::f32x4>();
    }

    #[test]
    fn div_masked_constant_dividers() {
        use thermite::{BranchfreeDivider, Divider};
        let v = VI::new([10, 20, 30, 40]);
        let mask = v.cmp_gt(VI::splat(15)); // [F, T, T, T]
        let d = Divider::i32(7);
        let bf = BranchfreeDivider::i32(7);
        let q = VI::new([10 / 7, 20 / 7, 30 / 7, 40 / 7]);

        // Divider masked variants
        assert_eq!(
            v.div_c(mask, d).into_array().as_slice(),
            mask.select(q, v).into_array().as_slice()
        );
        assert_eq!(
            v.div_m(VI::splat(-1), mask, d).into_array().as_slice(),
            mask.select(q, VI::splat(-1)).into_array().as_slice()
        );
        assert_eq!(
            v.div_z(mask, d).into_array().as_slice(),
            mask.select(q, VI::ZERO).into_array().as_slice()
        );

        // BranchfreeDivider masked variants
        assert_eq!(
            v.div_c(mask, bf).into_array().as_slice(),
            mask.select(q, v).into_array().as_slice()
        );
        assert_eq!(
            v.div_m(VI::splat(-1), mask, bf).into_array().as_slice(),
            mask.select(q, VI::splat(-1)).into_array().as_slice()
        );
        assert_eq!(
            v.div_z(mask, bf).into_array().as_slice(),
            mask.select(q, VI::ZERO).into_array().as_slice()
        );
    }

    #[test]
    fn mask_reshape_and_store_masked() {
        type VI8 = Vector<<Neon as Simd>::i32x8>;
        type MI8 = Mask<<Neon as Simd>::i32x8>;

        let lo = VI::new([1, 0, 1, 0]).cmp_ne(VI::ZERO); // [T,F,T,F]
        let hi = VI::new([0, 0, 1, 1]).cmp_ne(VI::ZERO); // [F,F,T,T]
        let rb4 = |m: Mask<<Neon as Simd>::i32x4>| m.select(VI::ONE, VI::ZERO).into_array().as_slice().to_vec();
        let rb8 = |m: MI8| m.select(VI8::ONE, VI8::ZERO).into_array().as_slice().to_vec();

        // Concat: low half = lo, high half = hi
        let wide: MI8 = Concat::concat(lo, hi);
        assert_eq!(rb8(wide), vec![1, 0, 1, 0, 0, 0, 1, 1]);
        // split is the inverse
        let (slo, shi) = Concat::split(wide);
        assert_eq!(rb4(slo), vec![1, 0, 1, 0]);
        assert_eq!(rb4(shi), vec![0, 0, 1, 1]);
        // extend: low half = lo, high half = false; narrow keeps the low half
        let ext: MI8 = Extend::extend(lo);
        assert_eq!(rb8(ext), vec![1, 0, 1, 0, 0, 0, 0, 0]);
        assert_eq!(rb4(ext.narrow()), vec![1, 0, 1, 0]);

        // store_masked: only lanes where the mask is true are written
        let v = VI::new([11, 22, 33, 44]);
        let mut buf = [0i32; 4];
        unsafe { v.store_masked(lo, buf.as_mut_ptr()) };
        assert_eq!(buf, [11, 0, 33, 0]);
    }

    #[test]
    fn vector_num_traits() {
        let a = VI::new([5, -3, 8, 2]);
        let b = VI::new([1, 4, 8, 2]);

        // Zero
        assert!(<VI as Zero>::zero().as_slice().iter().all(|&x| x == 0));
        assert!(Zero::is_zero(&VI::ZERO));
        assert!(!Zero::is_zero(&a));
        let mut z = a;
        z.set_zero();
        assert!(Zero::is_zero(&z));

        // One
        assert!(<VI as One>::one().as_slice().iter().all(|&x| x == 1));
        assert!(One::is_one(&VI::ONE));
        assert!(!One::is_one(&a));
        let mut o = a;
        o.set_one();
        assert!(One::is_one(&o));

        // Bounded
        assert_eq!(<VI as Bounded>::max_value().as_slice(), &[i32::MAX; 4]);
        assert_eq!(<VI as Bounded>::min_value().as_slice(), &[i32::MIN; 4]);

        // Saturating (by value) + SaturatingAdd/Sub (by ref)
        let big = VI::splat(i32::MAX);
        assert_eq!(Saturating::saturating_add(big, VI::ONE).as_slice(), &[i32::MAX; 4]);
        assert_eq!(
            Saturating::saturating_sub(VI::splat(i32::MIN), VI::ONE).as_slice(),
            &[i32::MIN; 4]
        );
        assert_eq!(SaturatingAdd::saturating_add(&big, &VI::ONE).as_slice(), &[i32::MAX; 4]);
        assert_eq!(
            SaturatingSub::saturating_sub(&VI::splat(i32::MIN), &VI::ONE).as_slice(),
            &[i32::MIN; 4]
        );

        // Wrapping add/sub/mul
        assert_eq!(WrappingAdd::wrapping_add(&a, &b).as_slice(), &[6, 1, 16, 4]);
        assert_eq!(WrappingSub::wrapping_sub(&a, &b).as_slice(), &[4, -7, 0, 0]);
        assert_eq!(WrappingMul::wrapping_mul(&a, &b).as_slice(), &[5, -12, 64, 4]);

        // PartialEq (Vector == Vector): all lanes equal
        assert!(VI::splat(2) == VI::splat(2));
        assert!(VI::splat(2) != a);

        // Index / IndexMut
        assert_eq!(a[2], 8);
        let mut im = a;
        im[0] = 99;
        assert_eq!(im[0], 99);

        // Sum / Product over an iterator of vectors (lane-wise)
        let vs = [VI::new([1, 2, 3, 4]), VI::new([10, 20, 30, 40]), VI::splat(100)];
        let s: VI = vs.into_iter().sum();
        assert_eq!(s.as_slice(), &[111, 122, 133, 144]);
        let p: VI = [VI::splat(2), VI::splat(3), VI::new([1, 2, 3, 4])]
            .into_iter()
            .product();
        assert_eq!(p.as_slice(), &[6, 12, 18, 24]);
    }
}
