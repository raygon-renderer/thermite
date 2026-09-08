//! The error-free transformations on `FloatRegister` stay exact.
//!
//! The failure mode is a silently zero error term beside a plausible value, so every case
//! asserts the residual is the specific nonzero number it must be.
//!
//! Run under `--features algebraic-scalar` as well; without it the strict overrides on
//! the scalar backend are not exercised.

use thermite::register::FloatRegister;

/// `black_box` so nothing is constant-folded at compile time, where reassociation
/// cannot be observed.
fn bb<T>(v: T) -> T {
    core::hint::black_box(v)
}

/// Spells the default body's arithmetic (`alg_add`/`alg_sub` on this backend) beside the
/// strict override and prints whether they agree. Asserts only on the strict path, since
/// whether the optimizer takes the reassociation it is allowed is not stable.
#[test]
fn strict_override_versus_the_algebraic_default_body() {
    use thermite::register::NumericRegister;

    let a = bb(1.0f64);
    let b = bb(f64::from_bits(0x3C30_0000_0000_0000)); // 2^-60

    // The default body, verbatim.
    let s = <f64 as NumericRegister>::add(a, b);
    let v = <f64 as NumericRegister>::sub(s, a);
    let naive = <f64 as NumericRegister>::add(
        <f64 as NumericRegister>::sub(a, <f64 as NumericRegister>::sub(s, v)),
        <f64 as NumericRegister>::sub(b, v),
    );

    let (_, strict) = <f64 as FloatRegister>::two_sum::<false>(a, b);

    println!(
        "algebraic-scalar = {}: default-body error = {naive:e}, strict override = {strict:e}",
        cfg!(feature = "algebraic-scalar"),
    );

    assert_eq!(strict, b, "the strict override must keep the whole residual");
}

#[test]
fn two_sum_f64_keeps_the_residual() {
    // s rounds to exactly 1.0 and the whole of b survives in the error term.
    let a = bb(1.0f64);
    let b = bb(f64::from_bits(0x3C30_0000_0000_0000)); // 2^-60

    let (s, e) = <f64 as FloatRegister>::two_sum::<false>(a, b);

    assert_eq!(s, 1.0, "sum should round to 1.0");
    assert_eq!(e, b, "the error term is the whole of b, not zero");
}

#[test]
fn two_sum_f32_keeps_the_residual() {
    let a = bb(1.0f32);
    let b = bb(f32::from_bits(0x3380_0000)); // 2^-24

    let (s, e) = <f32 as FloatRegister>::two_sum::<false>(a, b);

    assert_eq!(s, 1.0);
    assert_eq!(e, b);
}

#[test]
fn two_diff_keeps_the_residual() {
    let a = bb(1.0f64);
    let b = bb(f64::from_bits(0x3C30_0000_0000_0000)); // 2^-60

    let (s, e) = <f64 as FloatRegister>::two_diff::<false>(a, b);

    assert_eq!(s, 1.0);
    assert_eq!(e, -b);
}

#[test]
fn two_prod_is_exact() {
    // (1 + 2^-52)^2 = 1 + 2^-51 + 2^-104. The first two bits fit a double; the last does
    // not, so it must come back in the error term.
    let a = bb(1.0f64 + f64::from_bits(0x3CB0_0000_0000_0000)); // 1 + 2^-52
    let (p, e) = <f64 as FloatRegister>::two_prod::<false>(a, a);

    assert_eq!(p, 1.0 + f64::from_bits(0x3CC0_0000_0000_0000), "1 + 2^-51");
    assert_eq!(e, f64::from_bits(0x3970_0000_0000_0000), "2^-104");
}

/// `SQUARE = true` agrees with the general path bit for bit and ignores `b`. The second
/// operand is deliberately not `a`, so a wrong wiring would show.
#[test]
fn square_arm_agrees_with_the_general_one_and_ignores_b() {
    for bits in [0x3FF0_0000_0000_0001u64, 0x4009_21FB_5444_2D18, 0x0010_0000_0000_0000] {
        let a = bb(f64::from_bits(bits));

        let (p, e) = <f64 as FloatRegister>::two_prod::<false>(a, a);
        let (sp, se) = <f64 as FloatRegister>::two_prod::<true>(a, bb(3.0f64));

        assert_eq!(sp.to_bits(), p.to_bits(), "value mismatch at {bits:#x}");
        assert_eq!(se.to_bits(), e.to_bits(), "error mismatch at {bits:#x}");
    }
}

/// The Dekker path, called directly since `two_prod` takes the FMA arm on this machine.
#[test]
fn veltkamp_split_reconstructs_and_survives_large_operands() {
    for bits in [0x3FF0_0000_0000_0001u64, 0x7FE0_0000_0000_0000, 0x0010_0000_0000_0000] {
        let a = bb(f64::from_bits(bits));

        let (sa, sb) = <f64 as FloatRegister>::rebalance_for_split(a, bb(1.0f64));
        let (hi, lo) = <f64 as FloatRegister>::veltkamp_split(sa);

        assert!(hi.is_finite(), "split overflowed at {bits:#x}");
        assert_eq!(hi + lo, sa, "split does not reconstruct at {bits:#x}");
        assert_eq!(sa * sb, a, "rebalance changed the product at {bits:#x}");
    }
}

/// `MAX * 0.5` overflows an unguarded split to NaN. `two_prod(MAX, 1.0)` is not here; it
/// is the documented residual limit of Dekker's method.
#[test]
fn two_prod_survives_the_overflow_threshold() {
    let a = bb(f64::MAX);
    let b = bb(0.5f64);

    let (p, e) = <f64 as FloatRegister>::two_prod::<false>(a, b);

    assert_eq!(p, f64::MAX * 0.5);
    assert!(e.is_finite(), "error term overflowed: {e}");
    assert_eq!(e, 0.0, "MAX * 0.5 is exact, so the residual is zero");
}

/// `ArrayRegister<f64, N>` must forward the EFTs per lane. Its `add`/`sub` are `alg_*`
/// under `algebraic-scalar`, so the `FloatRegister` default would fold.
#[test]
fn emulated_widths_keep_their_residual() {
    use thermite::simd::Simd;
    use thermite::vector::FloatVectorWithBits;

    use thermite::prelude::*;
    type W = thermite::vector::Vector<<thermite::backend::scalar::Scalar as Simd>::f64x4>;

    let a = bb(W::splat(1.0));
    let b = bb(W::splat(f64::from_bits(0x3C30_0000_0000_0000))); // 2^-60

    let (s, e) = a.two_sum(b);

    println!("ArrayRegister two_sum: s={} e={:e}", s.extract::<0>(), e.extract::<0>());
    assert_eq!(s.extract::<0>(), 1.0);
    assert_eq!(
        e.extract::<0>(),
        2f64.powi(-60),
        "emulated width folded its residual - ArrayRegister needs to delegate the EFTs \
         per lane instead of taking the FloatRegister default"
    );
}

/// 2Quotient's remainder is exact and its high word is a correctly-rounded division.
/// `49` is the divisor that measured worst under `algebraic-scalar` with a bare `/`.
#[test]
fn two_quot_is_exact_and_correctly_rounded() {
    use thermite::prelude::*;
    use thermite::vector::FloatVectorWithBits;

    type V = thermite::vector::Vector<f64>;

    for &(a, b) in &[(1.0f64, 49.0f64), (1.0, 3.0), (-7.5, 0.1), (2.0, 2.0), (1e300, 7.0)] {
        let (q, r) = V::splat(bb(a)).two_quot(V::splat(bb(b)));
        let (q, r) = (q.extract::<0>(), r.extract::<0>());

        // `q` is the correctly-rounded quotient: the same value strict IEEE division gives.
        assert_eq!(q, bb(a) / bb(b), "two_quot({a}, {b}) high word is not RN(a / b)");

        // `a == q*b + r` exactly, checked in double-double so the check itself is exact.
        let (p, e) = V::splat(q).two_prod(V::splat(b));
        let (p, e) = (p.extract::<0>(), e.extract::<0>());
        assert_eq!((a - p) - e, r, "two_quot({a}, {b}) remainder is not exact");
    }
}

/// Same as `emulated_widths_keep_their_residual`, for `two_quot`.
#[test]
fn emulated_widths_keep_their_quotient_strict() {
    use thermite::simd::Simd;
    use thermite::vector::FloatVectorWithBits;

    use thermite::prelude::*;
    type W = thermite::vector::Vector<<thermite::backend::scalar::Scalar as Simd>::f64x4>;

    let a = bb(W::splat(1.0));
    let b = bb(W::splat(49.0));

    let (q, r) = a.two_quot(b);

    assert_eq!(q.extract::<0>(), 1.0f64 / 49.0);
    assert_ne!(r.extract::<0>(), 0.0, "emulated width folded its division remainder");

    // Per lane, not lane 0 broadcast.
    assert_eq!(q.extract::<3>(), 1.0f64 / 49.0);
    assert_eq!(r.extract::<3>(), r.extract::<0>());
}
