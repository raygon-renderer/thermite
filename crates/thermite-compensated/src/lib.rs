// #![no_std]
#![allow(unused_braces)]

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};

use num_traits::{NumAssignOps, NumOps};
use thermite::element::SignedElement;
use thermite::vector::{NewConst, NewVector, SplatVector, VectorValue};
use thermite::{LargeInt, mask::GenericSelectable, prelude::*};

use thermite::vector::ops::{MulAddAssignExt, MulAddExt, Square, SquareMasked};

pub mod consts;
pub mod math;

#[cfg(feature = "special")]
pub mod special;

/// Scalar values that can be used in compensated arithmetic.
///
/// This also applies to [`Vector`]s whose elements implement this trait.
///
/// This can be implemented for anything so long as a suitable Veltkamp's
/// splitting constant can be provided. It just doesn't make much sense
/// on anything but scalar-like floating point types.
///
/// However, it's worth noting that if the type has true FMA support,
/// as indicated by `MulAddExt::HAS_TRUE_FMA`, then
/// the splitting constant is never used, so it can be a dummy value in that case.
pub trait ScalarValue:
    Copy + NumOps + NumAssignOps + MulAddExt<Output = Self> + Neg<Output = Self> + consts::SplitFloatConsts<Self>
{
    /// for Veltkamp's splitting
    const SPLITTER: Self;

    /// The value zero. Named this way to avoid conflicts.
    const SCALAR_ZERO: Self;

    /// The value one. Named this way to avoid conflicts.
    const SCALAR_ONE: Self;

    /// Empirical maximum |x| for which the erf_inv Maclaurin series converges
    /// within 64 terms to full precision. This is only used when the `special`
    /// crate feature is enabled, for the `erf_inv` function.
    const MAX_ERFINV_SERIES: Self;

    /// Returns the value truncated to its integer component.
    ///
    /// Named this way to avoid conflicts. Required for the `Rem` implementation.
    fn scalar_trunc(self) -> Self;

    /// Marker type for splatting a compile-time integer constant as `Compensated<Self>`.
    ///
    /// Each concrete impl can choose the precision strategy: `f32` uses a `f64` intermediate
    /// to capture the rounding error in the error term; `f64` stores zero error (would need
    /// `f128` for better); `Vector<R>` delegates to the inner element and splats via
    /// `VectorValue`.
    type CompensatedConstInt<const N: LargeInt>: SplatConst<Compensated<Self>>;

    /// Marker type for splatting a compile-time rational constant `N/D` as `Compensated<Self>`.
    ///
    /// Same precision strategy as `CompensatedConstInt`.
    type CompensatedConstRatio<const N: LargeInt, const D: LargeInt>: SplatConst<Compensated<Self>>;

    #[inline(always)]
    fn two_sum(a: Self, b: Self) -> (Self, Self) {
        let s = a + b;
        let v = s - a;
        let e = (a - (s - v)) + (b - v);
        (s, e)
    }

    #[inline(always)]
    fn two_diff(a: Self, b: Self) -> (Self, Self) {
        let s = a - b;
        let v = s - a;
        let e = (a - (s - v)) - (b + v);
        (s, e)
    }

    #[inline(always)]
    fn two_prod(a: Self, b: Self) -> (Self, Self) {
        // fast path if we have FMA available
        if Self::HAS_TRUE_FMA {
            let p = a * b;
            let e = a.mul_sub(b, p);

            return (p, e);
        }

        let splitter = Self::SPLITTER;

        // Split a
        let c_a = a * splitter;
        let a_hi = c_a - (c_a - a);
        let a_lo = a - a_hi;

        // Split b
        let c_b = b * splitter;
        let b_hi = c_b - (c_b - b);
        let b_lo = b - b_hi;

        // exact product
        let p = a * b;

        let err = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;

        (p, err)
    }

    #[inline(always)]
    fn square(a: Self) -> (Self, Self) {
        // fast path if we have FMA available
        if Self::HAS_TRUE_FMA {
            let p = a * a;
            let e = a.mul_sub(a, p);

            return (p, e);
        }

        let splitter = Self::SPLITTER;

        // Split a
        let c_a = a * splitter;
        let a_hi = c_a - (c_a - a);
        let a_lo = a - a_hi;

        // exact product
        let p = a * a;

        let d = a_hi * a_lo;
        let err = ((a_hi * a_hi - p) + d + d) + a_lo * a_lo;

        (p, err)
    }
}

impl ScalarValue for f32 {
    const SPLITTER: Self = ((1u64 << 12) + 1) as f32; // 2^12 + 1
    const SCALAR_ZERO: Self = 0.0;
    const SCALAR_ONE: Self = 1.0;
    const MAX_ERFINV_SERIES: Self = 0.75;

    #[inline(always)]
    fn scalar_trunc(self) -> Self {
        FloatElement::trunc(self)
    }

    type CompensatedConstInt<const N: LargeInt> = F32CompensatedIntConst<N>;
    type CompensatedConstRatio<const N: LargeInt, const D: LargeInt> = F32CompensatedRatioConst<N, D>;
}

impl ScalarValue for f64 {
    const SPLITTER: Self = ((1u64 << 27) + 1) as f64; // 2^27 + 1
    const SCALAR_ZERO: Self = 0.0;
    const SCALAR_ONE: Self = 1.0;
    const MAX_ERFINV_SERIES: Self = 0.545;

    #[inline(always)]
    fn scalar_trunc(self) -> Self {
        FloatElement::trunc(self)
    }

    type CompensatedConstInt<const N: LargeInt> = F64CompensatedIntConst<N>;
    type CompensatedConstRatio<const N: LargeInt, const D: LargeInt> = F64CompensatedRatioConst<N, D>;
}

impl<R: thermite::register::FloatRegister> ScalarValue for Vector<R>
where
    R::Element: ScalarValue,
{
    const SPLITTER: Self = Self::splat_const(<R::Element as ScalarValue>::SPLITTER);
    const SCALAR_ZERO: Self = Self::ZERO;
    const SCALAR_ONE: Self = Self::ONE;
    const MAX_ERFINV_SERIES: Self = Self::splat_const(<R::Element as ScalarValue>::MAX_ERFINV_SERIES);

    #[inline(always)]
    fn scalar_trunc(self) -> Self {
        self.trunc()
    }

    type CompensatedConstInt<const N: LargeInt> =
        CompensatedVectorConst<<R::Element as ScalarValue>::CompensatedConstInt<N>>;

    type CompensatedConstRatio<const N: LargeInt, const D: LargeInt> =
        CompensatedVectorConst<<R::Element as ScalarValue>::CompensatedConstRatio<N, D>>;
}

// /// NOTE: Nesting Compensated is not recommended. This is only implemented
// /// for completeness. If you need higher precision, consider using a wider
// /// base type instead, potentially a `BigFloat` from `thermite-bignum` instead of
// /// `Compensated` values altogether.
// impl<V: ScalarValue> ScalarValue for Compensated<V> {
//     const SPLITTER: Self = const {
//         assert!(
//             V::HAS_TRUE_FMA,
//             "Compensated<S> requires true FMA support to implement ScalarValue"
//         );

//         Self {
//             value: V::SPLITTER,
//             error: V::SCALAR_ZERO,
//         }
//     };

//     const SCALAR_ZERO: Self = Self {
//         value: V::SCALAR_ZERO,
//         error: V::SCALAR_ZERO,
//     };

//     const SCALAR_ONE: Self = Self {
//         value: V::SCALAR_ONE,
//         error: V::SCALAR_ZERO,
//     };

//     fn scalar_trunc(self) -> Self {
//         Self {
//             value: self.value.scalar_trunc(),
//             error: V::SCALAR_ZERO,
//         }
//     }
// }

/// Trait for float vector types that can be used in compensated arithmetic.
pub trait CompensatedFloatVector: ScalarValue + FloatVector<Element: ScalarValue> + CastVector<Self> {}
impl<V> CompensatedFloatVector for V where V: ScalarValue + FloatVector<Element: ScalarValue> + CastVector<V> {}

#[rustfmt::skip]
impl<E: ScalarValue + Element> Element for Compensated<E> {
    type Signed = <E as Element>::Signed;
    type Unsigned = <E as Element>::Unsigned;

    const ONE: Self = Self { value: E::ONE, error: E::ZERO };
    const ZERO: Self = Self { value: E::ZERO, error: E::ZERO };

    fn from_i8(value: i8) -> Self { Self { value: E::from_i8(value), error: E::ZERO } }
    fn from_u8(value: u8) -> Self { Self { value: E::from_u8(value), error: E::ZERO } }
    fn from_u16(value: u16) -> Self { Self { value: E::from_u16(value), error: E::ZERO } }
}

#[rustfmt::skip]
impl<E: ScalarValue + SignedElement> SignedElement for Compensated<E> {
    #[inline(always)]
    fn abs(self) -> Self {
        if self.value() < E::ZERO {
            -self
        } else {
            self
        }
    }

    #[inline(always)]
    fn signum(self) -> Self {
        Self::new(self.value().signum())
    }
}

use core::marker::PhantomData;

// EFT two_sum: returns (s, e) such that s + e = a + b exactly, s = fl(a + b).
const fn two_sum_f64(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    (s, e)
}

// EFT two_product via Dekker splitting: returns (p, e) such that p + e = a * b exactly,
// p = fl(a * b). Requires no FMA; accurate when |a|, |b| < 2^996 (no overflow in split).
const fn two_product_f64(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let c = f64::SPLITTER * a;
    let a_hi = c - (c - a);
    let a_lo = a - a_hi;
    let c = f64::SPLITTER * b;
    let b_hi = c - (c - b);
    let b_lo = b - b_hi;
    let e = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo;
    (p, e)
}

// --- f32: uses f64 intermediate to capture rounding error in the error term ---

pub struct F32CompensatedIntConst<const N: LargeInt>;
pub struct F32CompensatedRatioConst<const N: LargeInt, const D: LargeInt>;

impl<const N: LargeInt> SplatConst<Compensated<f32>> for F32CompensatedIntConst<N> {
    const VALUE: Compensated<f32> = {
        let value = N as f32;
        let error = (N as f64 - value as f64) as f32;
        Compensated { value, error }
    };
}

impl<const N: LargeInt, const D: LargeInt> SplatConst<Compensated<f32>> for F32CompensatedRatioConst<N, D> {
    const VALUE: Compensated<f32> = {
        assert!(D != 0, "CompensatedRatioConst: denominator must not be zero");
        let (q, r) = (N / D, N % D);
        let hi64 = (q as f64) + (r as f64) / (D as f64);
        let value = hi64 as f32;
        let error = (hi64 - value as f64) as f32;
        Compensated { value, error }
    };
}

// --- f64: double-double EFT to capture the rounding error without needing f128 ---

pub struct F64CompensatedIntConst<const N: LargeInt>;
pub struct F64CompensatedRatioConst<const N: LargeInt, const D: LargeInt>;

impl<const N: LargeInt> SplatConst<Compensated<f64>> for F64CompensatedIntConst<N> {
    const VALUE: Compensated<f64> = {
        // If |N| ≤ 2^53 the cast is exact, so error = 0. Otherwise the rounding error
        // is an integer ≤ ulp(value)/2, which is always exactly representable in f64.
        let value = N as f64;
        let error = (N - value as LargeInt) as f64;
        Compensated { value, error }
    };
}

impl<const N: LargeInt, const D: LargeInt> SplatConst<Compensated<f64>> for F64CompensatedRatioConst<N, D> {
    const VALUE: Compensated<f64> = {
        assert!(D != 0, "CompensatedRatioConst: denominator must not be zero");
        let (q, r) = (N / D, N % D);
        let q_f64 = q as f64;
        let r_f64 = r as f64;
        let d_f64 = D as f64;

        // frac = fl(r / D), with rounding error frac_err = r/D - frac.
        let frac = r_f64 / d_f64;

        // value = fl(q + frac); two_sum gives us the exact rounding error e_add.
        // q_f64 + frac = value + e_add (exactly).
        let (value, e_add) = two_sum_f64(q_f64, frac);

        // Recover frac * D exactly via Dekker two_product, so we can compute
        // r - frac*D = (r/D - frac)*D, the numerator of the division error.
        let (prod, e_prod) = two_product_f64(frac, d_f64);

        // r - frac*D = r_f64 - prod - e_prod. Compute (r_f64 - prod) with two_sum
        // to avoid cancellation, then fold in e_prod.
        let (diff, e_diff) = two_sum_f64(r_f64, -prod);
        let frac_err = (diff + (e_diff - e_prod)) / d_f64;

        // Total: value + error = q + r/D = N/D (to full double-double precision,
        // exact when |q|, |r|, |D| each fit in 2^53).
        let error = frac_err + e_add;

        Compensated { value, error }
    };
}

// --- Vector: lifts a scalar SplatConst<Compensated<V::Element>> to SplatConst<Compensated<V>> ---
// Delegates to the existing VectorValue impl which splats value and error independently.

pub struct CompensatedVectorConst<Inner>(PhantomData<Inner>);

// --- New (per-lane values) support for Compensated<V> ---

pub struct CompensatedNewImpl;

struct CompensatedValueConst<C, V>(PhantomData<(C, V)>);
struct CompensatedErrorConst<C, V>(PhantomData<(C, V)>);

impl<C, V: CompensatedFloatVector> NewConst<V::Element, V::Lanes> for CompensatedValueConst<C, V>
where
    C: NewConst<Compensated<V::Element>, V::Lanes>,
{
    const VALUES: thermite::generic_array::GenericArray<V::Element, V::Lanes> = const {
        let c_vals = C::VALUES;
        let src = c_vals.as_slice();
        let mut out: thermite::generic_array::GenericArray<V::Element, V::Lanes> = unsafe { core::mem::zeroed() };
        let dst = out.as_mut_slice();
        let mut i = 0;
        while i < V::LANES {
            dst[i] = src[i].value;
            i += 1;
        }
        core::mem::forget(c_vals);
        out
    };
}

impl<C, V: CompensatedFloatVector> NewConst<V::Element, V::Lanes> for CompensatedErrorConst<C, V>
where
    C: NewConst<Compensated<V::Element>, V::Lanes>,
{
    const VALUES: thermite::generic_array::GenericArray<V::Element, V::Lanes> = const {
        let c_vals = C::VALUES;
        let src = c_vals.as_slice();
        let mut out: thermite::generic_array::GenericArray<V::Element, V::Lanes> = unsafe { core::mem::zeroed() };
        let dst = out.as_mut_slice();
        let mut i = 0;
        while i < V::LANES {
            dst[i] = src[i].error;
            i += 1;
        }
        core::mem::forget(c_vals);
        out
    };
}

impl<T, V: CompensatedFloatVector> VectorValue<T, Compensated<V>> for CompensatedNewImpl
where
    T: NewConst<Compensated<V::Element>, V::Lanes>,
{
    const VALUE: Compensated<V> = Compensated {
        value: <<V as NewVector<V::Element, V::Lanes>>::New<CompensatedValueConst<T, V>> as VectorValue<
            CompensatedValueConst<T, V>,
            V,
        >>::VALUE,
        error: <<V as NewVector<V::Element, V::Lanes>>::New<CompensatedErrorConst<T, V>> as VectorValue<
            CompensatedErrorConst<T, V>,
            V,
        >>::VALUE,
    };
}

impl<V: CompensatedFloatVector> NewVector<Compensated<V::Element>, V::Lanes> for Compensated<V> {
    type New<T: NewConst<Compensated<V::Element>, V::Lanes>> = CompensatedNewImpl;
}

impl<V, Inner> SplatConst<Compensated<V>> for CompensatedVectorConst<Inner>
where
    V: CompensatedFloatVector,
    Inner: SplatConst<Compensated<V::Element>>,
{
    const VALUE: Compensated<V> = <Compensated<V> as VectorValue<Inner, Compensated<V>>>::VALUE;
}

#[rustfmt::skip]
impl<E: ScalarValue + FloatElement> FloatElement for Compensated<E> {
    #[inline(always)]
    fn sqrt(this: Self) -> Self {
        let s = E::sqrt(this.value);

        let (p, e) = E::two_prod(s, s);

        // sum of differences
        let remainder = (this.value - p) + (this.error - e);

        // correction term
        let corr = remainder / (s + s);

        Self::renormalized(s, corr)
    }

    #[inline(always)] fn floor(this: Self) -> Self { Self::new(E::floor(this.value())) }
    #[inline(always)] fn ceil(this: Self) -> Self { Self::new(E::ceil(this.value())) }
    #[inline(always)] fn round(this: Self) -> Self { Self::new(E::round(this.value())) }
    #[inline(always)] fn trunc(this: Self) -> Self { Self::new(E::trunc(this.value())) }

    // for these two, we rely on `renormalized` to avoid infinite error values
    #[inline(always)] fn next_up(this: Self) -> Self { Self::renormalized(this.value, E::next_up(this.error)) }
    #[inline(always)] fn next_down(this: Self) -> Self { Self::renormalized(this.value, E::next_down(this.error)) }

    // TODO: Represent these more accurately
    #[inline(always)]
    fn try_from_int(value: LargeInt) -> Option<Self> {
        E::try_from_int(value).map(|v| Self::new(v))
    }

    #[inline(always)]
    fn try_from_ratio(n: LargeInt, d: LargeInt) -> Option<Self> {
        if d == 0 {
            return None;
        }

        let df= <E as FloatElement>::try_from_int(d)?;

        // fast path for values that both fit in the float exactly
        if let Some(n) = <E as FloatElement>::try_from_int(n) {
            return Some(Self::from_fraction(n, df));
        }

        let (q, r) = (n / d, n % d);

        let mut result = Self::try_from_int(q)?;

        if r != 0 {
            let rf = <E as FloatElement>::try_from_int(r)?;

            result += Self::from_fraction(rf, df);
        }

        Some(result)
    }

    const HAS_INFINITY: bool = E::HAS_INFINITY;
    const HAS_SIGNED_ZERO: bool = E::HAS_SIGNED_ZERO;
    const HAS_SUBNORMALS: bool = E::HAS_SUBNORMALS;

    type ConstInt<const N: thermite::LargeInt> = E::CompensatedConstInt<N>;

    type ConstRatio<const N: thermite::LargeInt, const D: thermite::LargeInt> = E::CompensatedConstRatio<N, D>;
}

/// Compensated arithmetic number type.
///
/// This type represents a number as the sum of two components: a high-order value and a low-order error term.
/// Using these, it can effectively double the mantissa precision of standard floating-point types,
/// providing significantly improved accuracy for a wide range of numerical computations.
///
/// `Compensated<f32 | f64>` have some functionality required for use as an `Element`
/// in vectorized types, but cannot use the math library. Use `Vector<f32>` or `Vector<f64>`
/// as the inner type for full functionality.
#[derive(Debug, Default, Clone, Copy, PartialEq, PartialOrd)]
#[repr(C)]
pub struct Compensated<V> {
    pub value: V,
    pub error: V,
}

impl<V: ScalarValue> thermite::const_default::ConstDefault for Compensated<V> {
    const DEFAULT: Self = Compensated {
        value: V::SCALAR_ZERO,
        error: V::SCALAR_ZERO,
    };
}

impl<V: ScalarValue> Compensated<V> {
    /// Creates a new compensated number with zero error term.
    #[inline(always)]
    pub const fn new(value: V) -> Self {
        Self {
            value,
            error: V::SCALAR_ZERO,
        }
    }

    /// Returns the normalized value `(value + error)`.
    #[inline(always)]
    pub fn value(self) -> V {
        self.value + self.error
    }

    /// Returns the uncompensated value, with no error term applied.
    #[inline(always)]
    pub const fn uncompensated(self) -> V {
        self.value
    }

    /// Returns the error term.
    #[inline(always)]
    pub const fn error(self) -> V {
        self.error
    }

    /// Renormalizes a compensated number from a value and error term.
    #[inline(always)]
    pub(crate) fn renormalized(value: V, error: V) -> Self {
        let sum = value + error;
        let err = (value - sum) + error;
        Self { value: sum, error: err }
    }

    #[inline(always)]
    pub fn normalize(self) -> Self {
        Self::renormalized(self.value, self.error)
    }
}

impl<V: CompensatedFloatVector> Compensated<V> {
    pub fn splat_value(value: V::Element) -> Self {
        Self {
            value: V::splat(value),
            error: V::ZERO,
        }
    }
}

// for testing
const ALLOW_UNNORMALIZED: bool = true;

impl<V: ScalarValue> Compensated<V> {
    /// Accumulate rhs into self without renormalization.
    ///
    /// This should only be used in specific scenarios where renormalization is not desired,
    /// such as within iterative series expansions.
    #[inline(always)]
    pub fn accumulate_unnormalized(&mut self, rhs: Self) {
        if ALLOW_UNNORMALIZED {
            let (s, e) = V::two_sum(self.value, rhs.value);
            self.value = s;
            self.error += e + rhs.error;
        } else {
            *self += rhs;
        }
    }

    /// Reduce rhs from self without renormalization.
    ///
    /// This should only be used in specific scenarios where renormalization is not desired,
    /// such as within iterative series expansions.
    #[inline(always)]
    pub fn reduce_unnormalized(&mut self, rhs: Self) {
        if ALLOW_UNNORMALIZED {
            let (s, e) = V::two_diff(self.value, rhs.value);
            self.value = s;
            self.error = e + (self.error - rhs.error);
        } else {
            *self -= rhs;
        }
    }
}

#[rustfmt::skip]
impl<V: ScalarValue> Neg for Compensated<V> {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        Self { value: -self.value, error: -self.error }
    }
}

impl<V: ScalarValue> Add<Self> for Compensated<V> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let (s, e) = V::two_sum(self.value, rhs.value);
        Self::renormalized(s, e + self.error + rhs.error)
    }
}

impl<V: ScalarValue> Add<V> for Compensated<V> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: V) -> Self::Output {
        let (s, e) = V::two_sum(self.value, rhs);
        Self::renormalized(s, e + self.error)
    }
}

impl<V: ScalarValue> Sub<Self> for Compensated<V> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        let (s, e) = V::two_diff(self.value, rhs.value);
        Self::renormalized(s, e + (self.error - rhs.error))
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl<V: ScalarValue> Sub<V> for Compensated<V> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: V) -> Self::Output {
        let (s, e) = V::two_diff(self.value, rhs);
        Self::renormalized(s, e + self.error)
    }
}

impl<V: ScalarValue> Square for Compensated<V> {
    type Output = Self;

    #[inline(always)]
    fn square(self) -> Self {
        let (p, e) = V::square(self.value);

        let d = self.error * self.value;

        Self::renormalized(p, d + d + e)
    }
}

impl<V: CompensatedFloatVector> SquareMasked<V::Mask> for Compensated<V> {
    #[inline(always)]
    fn square_c(self, mask: V::Mask) -> Self::Output {
        mask.select(self.square(), self)
    }

    #[inline(always)]
    fn square_m(self, src: Self, mask: V::Mask) -> Self::Output {
        mask.select(self.square(), src)
    }

    #[inline(always)]
    fn square_z(self, mask: V::Mask) -> Self::Output {
        mask.select(self.square(), Self::ZERO)
    }
}

impl<V: ScalarValue> Mul<Self> for Compensated<V> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        let (p, e) = V::two_prod(self.value, rhs.value);

        let e = self.error.mul_adde(rhs.value, self.value.mul_adde(rhs.error, e));

        Self::renormalized(p, e)
    }
}

impl<V: ScalarValue> Mul<V> for Compensated<V> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: V) -> Self {
        // (a0 + a1) * b = a0*b + a1*b
        let (p, e1) = V::two_prod(self.value, rhs);
        // We just add a1*b to the error term
        Self::renormalized(p, self.error.mul_adde(rhs, e1))
    }
}

impl<V: ScalarValue> Div<Self> for Compensated<V> {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        let q1 = self.value / rhs.value;

        let (p_hi, p_lo) = V::two_prod(q1, rhs.value);

        // calculate the remainder r
        // let r = (self.value - p_hi) - p_lo + self.error - (q1 * rhs.error);
        let r = (self.value - p_hi) - p_lo + q1.nmul_adde(rhs.error, self.error);

        Self::renormalized(q1, r / rhs.value)
    }
}

impl<V: ScalarValue> Compensated<V> {
    pub fn div_scalar(num: V, denom: Self) -> Self {
        let q1 = num / denom.value;

        let (p_hi, p_lo) = V::two_prod(q1, denom.value);

        // calculate the remainder r
        // let r = (self.value - p_hi) - p_lo + self.error - (q1 * rhs.error);
        let r = (num - p_hi) - p_lo - (q1 * denom.error);

        Compensated::renormalized(q1, r / denom.value)
    }
}

impl<V: ScalarValue> Compensated<V> {
    /// Creates a compensated number from a fraction `numerator / denominator`,
    /// dividing with compensation.
    #[inline(always)]
    pub fn from_fraction(numerator: V, denominator: V) -> Self {
        let q1 = numerator / denominator;

        let (p_hi, p_lo) = V::two_prod(q1, denominator);

        // calculate the remainder r
        let r = (numerator - p_hi) - p_lo;

        Self::renormalized(q1, r / denominator)
    }
}

impl Compensated<f32> {
    /// Create a compensated f32 value from an f64 value,
    /// preserving as much precision as possible.
    #[inline(always)]
    pub const fn from_f64(v: f64) -> Self {
        let v_f32 = v as f32;
        let err = v - (v_f32 as f64);
        Self {
            value: v_f32,
            error: err as f32,
        }
    }
}

impl<V: ScalarValue> Div<V> for Compensated<V> {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: V) -> Self {
        // same as regular division, but rhs has no error term
        let q1 = self.value / rhs;

        let (p_hi, p_lo) = V::two_prod(q1, rhs);

        // calculate the remainder r
        let r = (self.value - p_hi) - p_lo + self.error;

        Self::renormalized(q1, r / rhs)
    }
}

impl<V: ScalarValue> Rem<Self> for Compensated<V> {
    type Output = Self;

    #[inline(always)]
    fn rem(self, rhs: Self) -> Self {
        let q = self / rhs;
        let n = Compensated::new(-q.value.scalar_trunc());
        rhs.mul_add(n, self)
    }
}

impl<V: ScalarValue> Rem<V> for Compensated<V> {
    type Output = Self;

    #[inline(always)]
    fn rem(self, rhs: V) -> Self {
        let q = self / rhs;
        let n = Compensated::new(-q.value.scalar_trunc());
        MulAddExt::mul_add(n, rhs, self)
    }
}

#[rustfmt::skip]
impl<V: ScalarValue> MulAddExt<Self, Self> for Compensated<V> {
    type Output = Self;

    // Compensated mul-add is always accurate, and have the same code paths,
    // so we can just set this to true.
    const HAS_TRUE_FMA: bool = true;

    #[inline(always)]
    fn mul_add(self, b: Self, c: Self) -> Self {
        let (p, e_prod_base) = V::two_prod(self.value, b.value);
        let (s, e_sum) = V::two_sum(p, c.value);

        let e_prod = self.error.mul_adde(b.value, self.value.mul_adde(b.error, e_prod_base + e_sum));

        Self::renormalized(s, e_prod + c.error)
    }

    #[inline(always)]
    fn mul_sub(self, b: Self, c: Self) -> Self::Output {
        let (p, e_prod_base) = V::two_prod(self.value, b.value);
        let (s, e_diff) = V::two_diff(p, c.value);

        let e_prod = self.error.mul_adde(b.value, self.value.mul_adde(b.error, e_prod_base + e_diff));

        // Subtract c.error because the operation is (a*b) - c
        // The total error is the product error + subtraction error - c's error component
        Self::renormalized(s, e_prod - c.error)
    }

    #[inline(always)] fn nmul_add(self, a: Self, b: Self) -> Self::Output { self.mul_add(-a, b) }
    #[inline(always)] fn nmul_sub(self, a: Self, b: Self) -> Self::Output { self.mul_sub(-a, b) }
    #[inline(always)] fn mul_adde(self, a: Self, b: Self) -> Self::Output { self.mul_add(a, b) }
    #[inline(always)] fn mul_sube(self, a: Self, b: Self) -> Self::Output { self.mul_sub(a, b) }
    #[inline(always)] fn nmul_adde(self, a: Self, b: Self) -> Self::Output { self.nmul_add(a, b) }
    #[inline(always)] fn nmul_sube(self, a: Self, b: Self) -> Self::Output { self.nmul_sub(a, b) }
}

#[rustfmt::skip]
impl<V: ScalarValue> MulAddExt<V, Self> for Compensated<V> {
    type Output = Self;

    const HAS_TRUE_FMA: bool = true;

    #[inline(always)]
    fn mul_add(self, b: V, c: Self) -> Self::Output {
        let (p, e_prod_base) = V::two_prod(self.value, b);
        let (s, e_sum) = V::two_sum(p, c.value);

        let e_prod = self.error.mul_adde(b, e_prod_base + e_sum);

        Self::renormalized(s, e_prod + c.error)
    }

    #[inline(always)]
    fn mul_sub(self, b: V, c: Self) -> Self::Output {
        let (p, e_prod_base) = V::two_prod(self.value, b);
        let (s, e_diff) = V::two_diff(p, c.value);

        let e_prod = self.error.mul_adde(b, e_prod_base + e_diff);

        Self::renormalized(s, e_prod - c.error)
    }

    #[inline(always)] fn nmul_add(self, a: V, b: Self) -> Self::Output { self.mul_add(-a, b) }
    #[inline(always)] fn nmul_sub(self, a: V, b: Self) -> Self::Output { self.mul_sub(-a, b) }
    #[inline(always)] fn mul_adde(self, a: V, b: Self) -> Self::Output { self.mul_add(a, b) }
    #[inline(always)] fn mul_sube(self, a: V, b: Self) -> Self::Output { self.mul_sub(a, b) }
    #[inline(always)] fn nmul_adde(self, a: V, b: Self) -> Self::Output { self.nmul_add(a, b) }
    #[inline(always)] fn nmul_sube(self, a: V, b: Self) -> Self::Output { self.nmul_sub(a, b) }
}

#[rustfmt::skip]
impl<V: ScalarValue> MulAddExt<Self, V> for Compensated<V> {
    type Output = Self;

    const HAS_TRUE_FMA: bool = true;

    #[inline(always)]
    fn mul_add(self, a: Self, b: V) -> Self::Output {
        let (p, e_prod_base) = V::two_prod(self.value, a.value);
        let (s, e_sum) = V::two_sum(p, b);

        let e_prod = self.error.mul_add(a.value, self.value.mul_add(a.error, e_prod_base + e_sum));

        Self::renormalized(s, e_prod)
    }

    #[inline(always)]
    fn mul_sub(self, b: Self, c: V) -> Self::Output {
        let (p, e_prod_base) = V::two_prod(self.value, b.value);
        let (s, e_diff) = V::two_diff(p, c);

        let e_prod = self.error.mul_adde(b.value, self.value.mul_adde(b.error, e_prod_base + e_diff));

        Self::renormalized(s, e_prod)
    }

    #[inline(always)] fn nmul_add(self, a: Self, b: V) -> Self::Output { self.mul_add(-a, b) }
    #[inline(always)] fn nmul_sub(self, a: Self, b: V) -> Self::Output { self.mul_sub(-a, b) }
    #[inline(always)] fn mul_adde(self, a: Self, b: V) -> Self::Output { self.mul_add(a, b) }
    #[inline(always)] fn mul_sube(self, a: Self, b: V) -> Self::Output { self.mul_sub(a, b) }
    #[inline(always)] fn nmul_adde(self, a: Self, b: V) -> Self::Output { self.nmul_add(a, b) }
    #[inline(always)] fn nmul_sube(self, a: Self, b: V) -> Self::Output { self.nmul_sub(a, b) }
}

impl<V: Copy, T> AddAssign<T> for Compensated<V>
where
    Self: Add<T, Output = Self>,
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: T) {
        *self = *self + rhs;
    }
}

impl<V: Copy, T> SubAssign<T> for Compensated<V>
where
    Self: Sub<T, Output = Self>,
{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: T) {
        *self = *self - rhs;
    }
}

impl<V: Copy, T> MulAssign<T> for Compensated<V>
where
    Self: Mul<T, Output = Self>,
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: T) {
        *self = *self * rhs;
    }
}

impl<V: Copy, T> DivAssign<T> for Compensated<V>
where
    Self: Div<T, Output = Self>,
{
    #[inline(always)]
    fn div_assign(&mut self, rhs: T) {
        *self = *self / rhs;
    }
}

impl<V: Copy, T> RemAssign<T> for Compensated<V>
where
    Self: Rem<T, Output = Self>,
{
    #[inline(always)]
    fn rem_assign(&mut self, rhs: T) {
        *self = *self % rhs;
    }
}

#[rustfmt::skip]
impl<V: Copy, A, B> MulAddAssignExt<A, B> for Compensated<V>
where
    Self: MulAddExt<A, B, Output = Self>,
{
    #[inline(always)] fn mul_add_assign(&mut self, a: A, b: B) { *self = self.mul_add(a, b); }
    #[inline(always)] fn mul_sub_assign(&mut self, a: A, b: B) { *self = self.mul_sub(a, b); }
    #[inline(always)] fn nmul_add_assign(&mut self, a: A, b: B) { *self = self.nmul_add(a, b); }
    #[inline(always)] fn nmul_sub_assign(&mut self, a: A, b: B) { *self = self.nmul_sub(a, b); }
    #[inline(always)] fn mul_adde_assign(&mut self, a: A, b: B) { *self = self.mul_adde(a, b); }
    #[inline(always)] fn mul_sube_assign(&mut self, a: A, b: B) { *self = self.mul_sube(a, b); }
    #[inline(always)] fn nmul_adde_assign(&mut self, a: A, b: B) { *self = self.nmul_adde(a, b); }
    #[inline(always)] fn nmul_sube_assign(&mut self, a: A, b: B) { *self = self.nmul_sube(a, b); }
}

macro_rules! impl_masked {
    (MUL_ADD: $($method:ident),*) => {paste::paste! {
        impl<V: CompensatedFloatVector, A, B> thermite::vector::ops::MulAddExtMasked<V::Mask, A, B> for Compensated<V>
        where
            Compensated<V>: MulAddExt<A, B, Output = Self>,
        {
            $(
                #[inline(always)]
                fn [<$method _c>](self, mask: V::Mask, a: A, b: B) -> Self {
                    mask.select(self.[<$method>](a, b), self)
                }

                #[inline(always)]
                fn [<$method _m>](self, src: Self, mask: V::Mask, a: A, b: B) -> Self {
                    mask.select(self.[<$method>](a, b), src)
                }

                #[inline(always)]
                fn [<$method _z>](self, mask: V::Mask, a: A, b: B) -> Self {
                    mask.select(self.[<$method>](a, b), Self::EMPTY)
                }
            )*
        }

        impl<V: CompensatedFloatVector, A, B> thermite::vector::ops::MulAddAssignExtMasked<V::Mask, A, B> for Compensated<V>
        where
            Compensated<V>: MulAddExt<A, B, Output = Self>,
        {
            $(
                #[inline(always)]
                fn [<$method _assign_c>](&mut self, mask: V::Mask, a: A, b: B) {
                    *self = mask.select(self.[<$method>](a, b), *self);
                }

                #[inline(always)]
                fn [<$method _assign_m>](&mut self, src: Self, mask: V::Mask, a: A, b: B) {
                    *self = mask.select(self.[<$method>](a, b), src);
                }

                #[inline(always)]
                fn [<$method _assign_z>](&mut self, mask: V::Mask, a: A, b: B) {
                    *self = mask.select(self.[<$method>](a, b), Self::EMPTY);
                }
            )*
        }
    }};

    ($trait:ident::$method:ident) => {paste::paste! {
        impl<V: CompensatedFloatVector, Rhs> thermite::vector::ops::[<$trait Masked>]<V::Mask, Rhs> for Compensated<V>
        where
            Compensated<V>: $trait<Rhs, Output = Self>,
        {
            #[inline(always)]
            fn [<$method _c>](self, mask: V::Mask, rhs: Rhs) -> Self {
                mask.select(self.$method(rhs), self)
            }

            #[inline(always)]
            fn [<$method _m>](self, src: Self, mask: V::Mask, rhs: Rhs) -> Self {
                mask.select(self.$method(rhs), src)
            }

            #[inline(always)]
            fn [<$method _z>](self, mask: V::Mask, rhs: Rhs) -> Self {
                mask.select(self.$method(rhs), Self::EMPTY)
            }
        }

        impl<V: CompensatedFloatVector, Rhs> thermite::vector::ops::[<$trait AssignMasked>]<V::Mask, Rhs> for Compensated<V>
        where
            Compensated<V>: $trait<Rhs, Output = Self>,
        {
            #[inline(always)]
            fn [<$method _assign_c>](&mut self, mask: V::Mask, rhs: Rhs) {
                *self = mask.select(self.$method(rhs), *self);
            }

            #[inline(always)]
            fn [<$method _assign_m>](&mut self, src: Self, mask: V::Mask, rhs: Rhs) {
                *self = mask.select(self.$method(rhs), src);
            }

            #[inline(always)]
            fn [<$method _assign_z>](&mut self, mask: V::Mask, rhs: Rhs) {
                *self = mask.select(self.$method(rhs), Self::EMPTY);
            }
        }
    }};
}

impl_masked!(MUL_ADD: mul_add, mul_sub, nmul_add, nmul_sub, mul_adde, mul_sube, nmul_adde, nmul_sube);
impl_masked!(Add::add);
impl_masked!(Sub::sub);
impl_masked!(Mul::mul);
impl_masked!(Div::div);
impl_masked!(Rem::rem);

impl<V: CompensatedFloatVector> GenericSelectable for Compensated<V> {
    type SelectableMask = <V as GenericSelectable>::SelectableMask;

    #[inline(always)]
    fn select<M>(mask: M, t: Self, f: Self) -> Self
    where
        Self::SelectableMask: CastMask<M>,
    {
        let mask = <Self::SelectableMask as CastMask<M>>::mask_from(mask);

        Self {
            value: mask.select(t.value, f.value),
            error: mask.select(t.error, f.error),
        }
    }
}

impl<V: thermite::simd::HasIsa> thermite::simd::HasIsa for Compensated<V> {
    const ISA: thermite::isa::InstructionSet = V::ISA;
}

impl<V: CompensatedFloatVector> SplatVector<Compensated<V::Element>> for Compensated<V> {
    type Splat<T: SplatConst<Compensated<V::Element>>> = Self;
}

#[rustfmt::skip]
impl<V: CompensatedFloatVector, E: SplatConst<Compensated<V::Element>>> VectorValue<E, Compensated<V>> for Compensated<V> {
    const VALUE: Compensated<V> = const {
        struct Value<V: CompensatedFloatVector, E: SplatConst<Compensated<V::Element>>>(core::marker::PhantomData<(V, E)>);
        struct Error<V: CompensatedFloatVector, E: SplatConst<Compensated<V::Element>>>(core::marker::PhantomData<(V, E)>);

        impl<V: CompensatedFloatVector, E: SplatConst<Compensated<V::Element>>> SplatConst<V::Element> for Value<V, E> {
            const VALUE: V::Element = <E as SplatConst<Compensated<V::Element>>>::VALUE.value;
        }

        impl<V: CompensatedFloatVector, E: SplatConst<Compensated<V::Element>>> SplatConst<V::Element> for Error<V, E> {
            const VALUE: V::Element = <E as SplatConst<Compensated<V::Element>>>::VALUE.error;
        }

        Compensated {
            value: thermite::vector::const_splat::<V, Value<V, E>>(),
            error: thermite::vector::const_splat::<V, Error<V, E>>(),
        }
    };
}

#[rustfmt::skip]
impl<V: CompensatedFloatVector> Interleave for Compensated<V> {
    #[inline(always)]
    fn interleave(self, other: Self) -> (Self, Self) {
        let (value_lo, value_hi) = self.value.interleave(other.value);
        let (error_lo, error_hi) = self.error.interleave(other.error);

        (
            Self { value: value_lo, error: error_lo },
            Self { value: value_hi, error: error_hi },
        )
    }

    #[inline(always)]
    fn deinterleave(self, other: Self) -> (Self, Self) {
        let (value_lo, value_hi) = self.value.deinterleave(other.value);
        let (error_lo, error_hi) = self.error.deinterleave(other.error);

        (
            Self { value: value_lo, error: error_lo },
            Self { value: value_hi, error: error_hi },
        )
    }
}

#[rustfmt::skip]
impl<V: CompensatedFloatVector> GenericVector for Compensated<V> {
    type Element = Compensated<V::Element>;

    const EMPTY: Self = Self::new(V::ZERO);
    const LANES: usize = V::LANES;

    type Lanes = V::Lanes;

    type Unsigned = V::Unsigned;
    type Signed = V::Signed;

    type Mask = V::Mask;

    #[inline(always)]
    fn new<const N: usize>(value: [Self::Element; N]) -> Self
    where
        thermite::generic_array::typenum::Const<N>: thermite::generic_array::IntoArrayLength<ArrayLength = Self::Lanes>
    {
        Compensated {
            value: V::new(value.map(|c| c.value)),
            error: V::new(value.map(|c| c.error)),
        }
    }

    #[inline(always)]
    fn splat(value: Self::Element) -> Self {
        Self {
            value: V::splat(value.value),
            error: V::splat(value.error),
        }
    }

    #[inline(always)]
    fn single(value: Self::Element) -> Self {
        Self::new(V::single(value.value))
    }

    #[inline(always)]
    unsafe fn load(ptr: *const Self::Element) -> Self {
        let ptr = ptr as *const V::Element;
        let a = unsafe { V::load(ptr) };
        let b = unsafe { V::load(ptr.add(V::LANES)) };
        let (value, error) = a.deinterleave(b);
        Self { value, error }
    }

    #[inline(always)]
    unsafe fn load_m(src: Self, mask: Self::Mask, ptr: *const Self::Element) -> Self {
        let ptr = ptr as *const V::Element;
        // Expand mask to cover the interleaved (value, error) pairs in memory:
        // lane i of mask -> positions 2i and 2i+1 in the interleaved layout.
        let (a_mask, b_mask) = mask.interleave(mask);
        let (src_a, src_b) = src.value.interleave(src.error);
        let a = unsafe { V::load_m(src_a, a_mask, ptr) };
        let b = unsafe { V::load_m(src_b, b_mask, ptr.add(V::LANES)) };
        let (value, error) = a.deinterleave(b);
        Self { value, error }
    }

    #[inline(always)]
    unsafe fn load_z(mask: Self::Mask, ptr: *const Self::Element) -> Self {
        let ptr = ptr as *const V::Element;
        // Expand mask to cover the interleaved (value, error) pairs in memory.
        let (a_mask, b_mask) = mask.interleave(mask);
        let a = unsafe { V::load_z(a_mask, ptr) };
        let b = unsafe { V::load_z(b_mask, ptr.add(V::LANES)) };
        let (value, error) = a.deinterleave(b);
        Self { value, error }
    }

    #[inline(always)]
    unsafe fn load_unaligned(ptr: *const Self::Element) -> Self {
        let ptr = ptr as *const V::Element;
        let a = unsafe { V::load_unaligned(ptr) };
        let b = unsafe { V::load_unaligned(ptr.add(V::LANES)) };
        let (value, error) = a.deinterleave(b);
        Self { value, error }
    }

    #[inline(always)]
    unsafe fn load_streaming(ptr: *const Self::Element) -> Self {
        let ptr = ptr as *const V::Element;
        let a = unsafe { V::load_streaming(ptr) };
        let b = unsafe { V::load_streaming(ptr.add(V::LANES)) };
        let (value, error) = a.deinterleave(b);
        Self { value, error }
    }

    #[inline(always)]
    unsafe fn store(self, ptr: *mut Self::Element) {
        let ptr = ptr as *mut V::Element;
        let (a, b) = self.value.interleave(self.error);
        unsafe {
            a.store(ptr);
            b.store(ptr.add(V::LANES));
        }
    }

    #[inline(always)]
    unsafe fn store_masked(self, mask: Self::Mask, ptr: *mut Self::Element) {
        let ptr = ptr as *mut V::Element;
        // Expand mask to cover the interleaved (value, error) pairs in memory.
        let (a_mask, b_mask) = mask.interleave(mask);
        let (a, b) = self.value.interleave(self.error);
        unsafe {
            a.store_masked(a_mask, ptr);
            b.store_masked(b_mask, ptr.add(V::LANES));
        }
    }

    #[inline(always)]
    unsafe fn store_unaligned(self, ptr: *mut Self::Element) {
        let ptr = ptr as *mut V::Element;
        let (a, b) = self.value.interleave(self.error);
        unsafe {
            a.store_unaligned(ptr);
            b.store_unaligned(ptr.add(V::LANES));
        }
    }

    #[inline(always)]
    unsafe fn store_streaming(self, ptr: *mut Self::Element) {
        let ptr = ptr as *mut V::Element;
        let (a, b) = self.value.interleave(self.error);
        unsafe {
            a.store_streaming(ptr);
            b.store_streaming(ptr.add(V::LANES));
        }
    }

    #[inline(always)]
    unsafe fn lookup_unchecked(values: &[Self::Element], indices: Self::Unsigned) -> Self {
        if values.len() > Self::LANES * 2 {
            // for large lookup tables just fallback to scalar
            let mut res = Self::EMPTY;

            for i in 0..Self::LANES {
                let Ok(idx) = indices.extractv(i).try_into() else {
                    panic!("Index out of bounds for usize");
                };

                res = res.insertv(i, values[idx]);
            }

            return res;
        }

        let values = unsafe  {
            core::slice::from_raw_parts(values.as_ptr() as *const V::Element, values.len() * 2)
        };

        let value_idx = indices << 1;
        let error_idx = value_idx + Self::Unsigned::ONE;

        let value = unsafe { V::lookup_unchecked(values, value_idx) };
        let error = unsafe { V::lookup_unchecked(values, error_idx) };

        Self { value, error }
    }

    #[inline(always)]
    fn broadcast<const I: usize>(self) -> Self {
        Self {
            value: V::broadcast::<I>(self.value),
            error: V::broadcast::<I>(self.error),
        }
    }

    #[inline(always)]
    fn broadcastv(self, idx: usize) -> Self {
        Self {
            value: V::broadcastv(self.value, idx),
            error: V::broadcastv(self.error, idx),
        }
    }

    #[inline(always)]
    fn extract<const I: usize>(self) -> Self::Element {
        let value = V::extract::<I>(self.value);
        let error = V::extract::<I>(self.error);

        Compensated { value, error }
    }

    #[inline(always)]
    fn extractv(self, idx: usize) -> Self::Element {
        let value = V::extractv(self.value, idx);
        let error = V::extractv(self.error, idx);

        Compensated { value, error }
    }

    #[inline(always)]
    fn insert<const I: usize>(self, value: Self::Element) -> Self {
        let Compensated { value, error } = value;

        Self {
            value: V::insert::<I>(self.value, value),
            error: V::insert::<I>(self.error, error),
        }
    }

    #[inline(always)]
    fn insertv(self, idx: usize, value: Self::Element) -> Self {
        let Compensated { value, error } = value;

        Self {
            value: V::insertv(self.value, idx, value),
            error: V::insertv(self.error, idx, error),
        }
    }

    #[inline(always)]
    fn reverse(self) -> Self {
        Self {
            value: self.value.reverse(),
            error: self.error.reverse(),
        }
    }

    #[inline(always)]
    fn swap_bytes(self) -> Self {
        Self {
            value: self.value.swap_bytes(),
            error: self.error.swap_bytes(),
        }
    }

    #[inline(always)]
    fn zz(self, mask: Self::Mask) -> Self {
        Self {
            value: self.value.zz(mask),
            error: self.error.zz(mask),
        }
    }

    #[inline(always)]
    fn nz(self, mask: Self::Mask) -> Self {
        Self {
            value: self.value.nz(mask),
            error: self.error.nz(mask),
        }
    }


    fn map<F>(mut self, f: F) -> Self
    where
        F: Fn(Self::Element) -> Self::Element,
    {
        for i in 0..Self::LANES {
            self = self.insertv(i, f(self.extractv(i)));
        }

        self
    }

    fn fold<F>(self, mut init: Self::Element, f: F) -> Self::Element
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element,
    {
        for i in 0..Self::LANES {
            init = f(init, self.extractv(i));
        }

        init
    }

    fn reduce<F>(self, f: F) -> Self::Element
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element,
    {
        let mut result = self.extractv(0);

        for i in 1..Self::LANES {
            result = f(result, self.extractv(i));
        }

        result
    }

    #[inline(always)] fn splat_m(src: Self, mask: Self::Mask, value: Self::Element) -> Self { mask.select(Self::splat(value), src) }
    #[inline(always)] fn splat_z(mask: Self::Mask, value: Self::Element) -> Self { mask.select(Self::splat(value), Self::EMPTY) }
    #[inline(always)] fn broadcast_c<const I: usize>(self, mask: Self::Mask) -> Self { mask.select(self.broadcast::<I>(), self) }
    #[inline(always)] fn broadcast_m<const I: usize>(self, src: Self, mask: Self::Mask) -> Self { mask.select(self.broadcast::<I>(), src) }
    #[inline(always)] fn broadcast_z<const I: usize>(self, mask: Self::Mask) -> Self { mask.select(self.broadcast::<I>(), Self::EMPTY) }
    #[inline(always)] fn broadcastv_c(self, mask: Self::Mask, idx: usize) -> Self { mask.select(self.broadcastv(idx), self) }
    #[inline(always)] fn broadcastv_m(self, src: Self, mask: Self::Mask, idx: usize) -> Self { mask.select(self.broadcastv(idx), src) }
    #[inline(always)] fn broadcastv_z(self, mask: Self::Mask, idx: usize) -> Self { mask.select(self.broadcastv(idx), Self::EMPTY) }
    #[inline(always)] fn reverse_c(self, mask: Self::Mask) -> Self { mask.select(self.reverse(), self) }
    #[inline(always)] fn reverse_m(self, src: Self, mask: Self::Mask) -> Self { mask.select(self.reverse(), src) }
    #[inline(always)] fn reverse_z(self, mask: Self::Mask) -> Self { mask.select(self.reverse(), Self::EMPTY) }
    #[inline(always)] fn swap_bytes_c(self, mask: Self::Mask) -> Self { mask.select(self.swap_bytes(), self) }
    #[inline(always)] fn swap_bytes_m(self, src: Self, mask: Self::Mask) -> Self { mask.select(self.swap_bytes(), src) }
    #[inline(always)] fn swap_bytes_z(self, mask: Self::Mask) -> Self { mask.select(self.swap_bytes(), Self::EMPTY) }
}

#[rustfmt::skip]
impl<V: CompensatedFloatVector> PartialOrdVector for Compensated<V> {
    #[inline(always)]
    fn cmp_eq(self, other: Self) -> Self::Mask {
        // Strictly equal if both components match
        self.value.cmp_eq(other.value) & self.error.cmp_eq(other.error)
    }

    #[inline(always)]
    fn cmp_ne(self, other: Self) -> Self::Mask {
        // Not equal if either component differs
        self.value.cmp_ne(other.value) | self.error.cmp_ne(other.error)
    }

    #[inline(always)]
    fn cmp_lt(self, other: Self) -> Self::Mask {
        let val_lt = self.value.cmp_lt(other.value);
        let val_eq = self.value.cmp_eq(other.value);
        let err_lt = self.error.cmp_lt(other.error);

        // (value < other.value) OR (value == other.value AND error < other.error)
        GenericMask::ternlog::<{ thermite::ternlog_imm!(A | (B & C)) }>(val_lt, val_eq, err_lt)
    }

    #[inline(always)]
    fn cmp_gt(self, other: Self) -> Self::Mask {
        let val_gt = self.value.cmp_gt(other.value);
        let val_eq = self.value.cmp_eq(other.value);
        let err_gt = self.error.cmp_gt(other.error);

        // (value > other.value) OR (value == other.value AND error > other.error)
        GenericMask::ternlog::<{ thermite::ternlog_imm!(A | (B & C)) }>(val_gt, val_eq, err_gt)
    }

    #[inline(always)]
    fn cmp_le(self, other: Self) -> Self::Mask {
        let val_lt = self.value.cmp_lt(other.value);
        let val_eq = self.value.cmp_eq(other.value);
        let err_le = self.error.cmp_le(other.error);

        // (value < other.value) OR (value == other.value AND error <= other.error)
        GenericMask::ternlog::<{ thermite::ternlog_imm!(A | (B & C)) }>(val_lt, val_eq, err_le)
    }

    #[inline(always)]
    fn cmp_ge(self, other: Self) -> Self::Mask {
        let val_gt = self.value.cmp_gt(other.value);
        let val_eq = self.value.cmp_eq(other.value);
        let err_ge = self.error.cmp_ge(other.error);

        // (value > other.value) OR (value == other.value AND error >= other.error)
        GenericMask::ternlog::<{ thermite::ternlog_imm!(A | (B & C)) }>(val_gt, val_eq, err_ge)
    }
}

impl<V: ScalarValue> core::iter::Sum for Compensated<V> {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut iter = iter.into_iter();

        let Some(mut total) = iter.next() else {
            return Compensated::new(V::SCALAR_ZERO);
        };

        for v in iter {
            total += v;
        }

        total
    }
}

impl<V: ScalarValue> core::iter::Product for Compensated<V> {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        let mut iter = iter.into_iter();

        let Some(mut total) = iter.next() else {
            return Compensated::new(V::SCALAR_ONE); // multiplicative identity
        };

        for v in iter {
            total *= v;
        }

        total
    }
}

// These are odd in that (value + error) can exceed the bounds of V,
// but this is the most sensible implementation.
#[rustfmt::skip]
impl<V: CompensatedFloatVector> num_traits::Bounded for Compensated<V> {
    #[inline(always)] fn min_value() -> Self { Self { value: V::MIN, error: V::MIN } }
    #[inline(always)] fn max_value() -> Self { Self { value: V::MAX, error: V::MAX } }
}

impl<V: CompensatedFloatVector> NumericVector for Compensated<V> {
    const ZERO: Self = Self::new(V::ZERO);
    const ONE: Self = Self::new(V::ONE);
    const TWO: Self = Self::new(V::TWO);

    const MIN: Self = Self {
        value: V::MIN,
        error: V::MIN,
    };

    const MAX: Self = Self {
        value: V::MAX,
        error: V::MAX,
    };

    #[inline(always)]
    fn is_zero(self) -> Self::Mask {
        self.value().is_zero()
    }

    #[inline(always)]
    fn is_all_zero(self) -> bool {
        // if value+error is nonzero, then one of the components must be nonzero,
        // and this allows for faster vertical reduction versus two calls to is_all_zero()
        self.value().is_all_zero()
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        self.cmp_lt(other).select(self, other)
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        self.cmp_gt(other).select(self, other)
    }

    #[inline(always)]
    fn clamp(self, min: Self, max: Self) -> Self {
        let x = self.value();
        let min_value = min.value();
        let max_value = max.value();

        let is_lt = x.cmp_lt(min_value);
        let is_gt = x.cmp_gt(max_value);

        let value = is_lt.select(min.value, is_gt.select(max.value, self.value));
        let error = is_lt.select(min.error, is_gt.select(max.error, self.error));

        Self { value, error }
    }

    #[inline(always)]
    fn min_element(self) -> Self::Element {
        let mut min_elem = self.extractv(0);
        let mut min_value = min_elem.value();

        for i in 1..Self::LANES {
            let elem = self.extractv(i);
            let value = elem.value();

            if value < min_value {
                min_elem = elem;
                min_value = value;
            }
        }

        min_elem
    }

    #[inline(always)]
    fn max_element(self) -> Self::Element {
        let mut max_elem = self.extractv(0);
        let mut max_value = max_elem.value();

        for i in 1..Self::LANES {
            let elem = self.extractv(i);
            let value = elem.value();

            if value > max_value {
                max_elem = elem;
                max_value = value;
            }
        }

        max_elem
    }

    fn sum_elements(self) -> Self::Element {
        self.reduce(|a, b| a + b)
    }

    fn prod_elements(self) -> Self::Element {
        self.reduce(|a, b| a * b)
    }

    #[inline(always)]
    fn offset() -> Self {
        Self::new(V::offset())
    }

    #[inline(always)]
    fn indexed() -> Self {
        Self::new(V::indexed())
    }

    fn min_c(self, _mask: Self::Mask, _other: Self) -> Self {
        todo!()
    }

    fn min_m(self, _src: Self, _mask: Self::Mask, _other: Self) -> Self {
        todo!()
    }

    fn min_z(self, _mask: Self::Mask, _other: Self) -> Self {
        todo!()
    }

    fn max_c(self, _mask: Self::Mask, _other: Self) -> Self {
        todo!()
    }

    fn max_m(self, _src: Self, _mask: Self::Mask, _other: Self) -> Self {
        todo!()
    }

    fn max_z(self, _mask: Self::Mask, _other: Self) -> Self {
        todo!()
    }

    fn scale(self, _factor: Self::Element) -> Self {
        todo!()
    }

    fn scale_c(self, _mask: Self::Mask, _factor: Self::Element) -> Self {
        todo!()
    }

    fn scale_m(self, _src: Self, _mask: Self::Mask, _factor: Self::Element) -> Self {
        todo!()
    }

    fn scale_z(self, _mask: Self::Mask, _factor: Self::Element) -> Self {
        todo!()
    }

    fn pairwise_sum(_lo: Self, _hi: Self) -> Self {
        todo!()
    }

    fn relaxed_pairwise_sum(_lo: Self, _hi: Self) -> Self {
        todo!()
    }

    fn min_max_element(self) -> (Self::Element, Self::Element) {
        (self.min_element(), self.max_element())
    }

    fn arg_minmax(self) -> (usize, usize) {
        todo!()
    }
}

impl<V: CompensatedFloatVector> thermite::vector::ops::NegMasked<V::Mask> for Compensated<V> {
    #[inline(always)]
    fn neg_c(mut self, mask: V::Mask) -> Self {
        self.value = self.value.neg_c(mask);
        self.error = self.error.neg_c(mask);

        self
    }

    #[inline(always)]
    fn neg_m(mut self, src: Self, mask: V::Mask) -> Self {
        self.value = self.value.neg_m(src.value, mask);
        self.error = self.error.neg_m(src.error, mask);

        self
    }

    #[inline(always)]
    fn neg_z(mut self, mask: V::Mask) -> Self {
        self.value = self.value.neg_z(mask);
        self.error = self.error.neg_z(mask);

        self
    }
}

impl<V: CompensatedFloatVector> SignedVector for Compensated<V> {
    const NEG_ONE: Self = Self::new(V::NEG_ONE);
    const MIN_POSITIVE: Self = Self::new(V::MIN_POSITIVE);

    #[inline(always)]
    fn abs(self) -> Self {
        self.neg_c(self.value().cmp_lt(V::ZERO))
    }

    #[inline(always)]
    fn signum(self) -> Self {
        Self::new(self.value().signum())
    }

    #[inline(always)]
    fn is_positive(self) -> Self::Mask {
        self.value().is_positive()
    }

    #[inline(always)]
    fn is_negative(self) -> Self::Mask {
        self.value().is_negative()
    }

    #[inline(always)]
    fn select_negative(self, if_neg: Self, if_pos: Self) -> Self {
        self.is_negative().select(if_neg, if_pos)
    }

    #[inline(always)]
    fn copysign(self, sign: Self) -> Self {
        let self_is_neg = self.is_negative();
        let sign_is_neg = sign.is_negative();

        self.neg_c(self_is_neg ^ sign_is_neg)
    }

    fn abs_c(self, _mask: Self::Mask) -> Self {
        todo!()
    }

    fn abs_m(self, _src: Self, _mask: Self::Mask) -> Self {
        todo!()
    }

    fn abs_z(self, _mask: Self::Mask) -> Self {
        todo!()
    }
    fn copysign_c(self, _mask: Self::Mask, _sign: Self) -> Self {
        todo!()
    }
    fn copysign_m(self, _src: Self, _mask: Self::Mask, _sign: Self) -> Self {
        todo!()
    }
    fn copysign_z(self, _mask: Self::Mask, _sign: Self) -> Self {
        todo!()
    }
}

// We need a single generic form of this since Rust only allows one
// implementation of this given the nested generic parameters.
impl<FROM, TO> CastVector<Compensated<FROM>> for Compensated<TO>
where
    FROM: CompensatedFloatVector + CastVector<TO>,
    TO: CompensatedFloatVector + CastVector<FROM>,
{
    fn cast_into(self) -> Compensated<FROM> {
        Compensated::<FROM>::cast_from(self)
    }

    fn cast_from(from: Compensated<FROM>) -> Self {
        let from_size = size_of::<FROM::Element>();
        let to_size = size_of::<TO::Element>();

        // TODO: Maybe match on cmp ordering?
        // Ord::cmp(&size_of::<FROM::Element>(), &size_of::<TO::Element>());
        if from_size > to_size {
            // --- Downsampling (f64-like -> f32-like) ---
            // We lose precision, so we must capture the lost bits in the new error term.

            // Project High -> Low
            let value = TO::cast_from(from.value);

            // Project Low -> High (check our work)
            // Calculate residual (bits lost in cast) in High Precision
            let delta = from.value - FROM::cast_from(value);

            Self {
                value,
                // Accumulate total error (new lost bits + old error)
                error: TO::cast_from(delta + from.error),
            }
        } else if from_size < to_size {
            // --- Upsampling (f32-like -> f64-like) ---
            // The larger type can hold the entire double-double sum losslessly.
            // We collapse the pair into the single 'value' field to normalize it.

            let v_hi = TO::cast_from(from.value);
            let e_hi = TO::cast_from(from.error);

            // Since f32+f32 (48 bits effective) fits in f64 (53 bits),
            // this sum is exact.
            Self {
                value: v_hi + e_hi,
                error: TO::ZERO,
            }
        } else {
            // --- Same Precision (f64 -> f64 or f32 -> f32) ---
            // Just a type conversion (or SIMD layout change) without precision change.
            // Preserve the structure exactly.
            Self {
                value: TO::cast_from(from.value),
                error: TO::cast_from(from.error),
            }
        }
    }
}

#[rustfmt::skip]
impl<V: CompensatedFloatVector> FloatVector for Compensated<V> {
    const HALF: Self = Self::new(<V as FloatVector>::HALF);
    const NEG_ZERO: Self = Self::new(<V as FloatVector>::NEG_ZERO);
    const INFINITY: Self = Self::new(<V as FloatVector>::INFINITY);
    const NEG_INFINITY: Self = Self::new(<V as FloatVector>::NEG_INFINITY);
    const NAN: Self = Self::new(<V as FloatVector>::NAN);

    const EPSILON: Self = Compensated {
        value: V::ZERO,
        error: <V as FloatVector>::EPSILON, // lower order bits get the epsilon
    };

    /// Don't use Compensated if you need to go higher precision than it provides.
    ///
    /// If you absolutely must, use [`CastVector`] to convert to a higher-precision type.
    type ExtendedPrecision = Self;

    // These are designed to provide reasonable results with reasonable performance.
    #[inline(always)] fn is_infinite(self) -> Self::Mask { self.value().is_infinite() }
    #[inline(always)] fn is_finite(self) -> Self::Mask { self.value().is_finite() }
    #[inline(always)] fn is_nan(self) -> Self::Mask { self.value.is_nan() | self.error.is_nan() }
    #[inline(always)] fn is_zero_or_subnormal(self) -> Self::Mask { self.value().is_zero_or_subnormal() }
    #[inline(always)] fn is_normal(self) -> Self::Mask { self.value().is_normal() }
    #[inline(always)] fn is_subnormal(self) -> Self::Mask { self.value.is_subnormal() | self.error.is_subnormal() }

    const HAS_APPROX_RCP: bool = false;
    const HAS_APPROX_RSQRT: bool = false;

    #[inline(always)]
    fn sqrt(self) -> Self {
        let s = V::sqrt(self.value);

        let (p, e) = ScalarValue::square(s);

        // sum of differences
        let remainder = (self.value - p) + (self.error - e);

        // correction term
        let corr = remainder / (s + s);

        Self::renormalized(s, corr)
    }

    #[inline(always)] fn rsqrt(self) -> Self { Self::div_scalar(V::ONE, self.sqrt()) }
    #[inline(always)] fn rcp(self) -> Self { Self::div_scalar(V::ONE, self) }

    #[inline(always)] fn floor(self) -> Self { Self::new(self.value().floor()) }
    #[inline(always)] fn ceil(self) -> Self { Self::new(self.value().ceil()) }
    #[inline(always)] fn round(self) -> Self { Self::new(self.value().round()) }
    #[inline(always)] fn trunc(self) -> Self { Self::new(self.value().trunc()) }
    #[inline(always)] fn fract(self) -> Self { self - self.trunc() }

    #[inline(always)]
    fn mul_sign(self, sign: Self) -> Self {
        let sign = sign.value();

        Self {
            value: self.value.mul_sign(sign),
            error: self.error.mul_sign(sign),
        }
    }

    #[inline(always)] fn signed_zero(self) -> Self { Self::new(self.value().signed_zero()) }

    #[inline(always)] fn next_up(self) -> Self { Self::renormalized(self.value, self.error.next_up()) }
    #[inline(always)] fn next_down(self) -> Self { Self::renormalized(self.value, self.error.next_down()) }

    unsafe fn block_autovectorization(&mut self) {
        unsafe {
            self.value.block_autovectorization();
            self.error.block_autovectorization();
        }
    }

    fn sqrt_c(self, _mask: Self::Mask) -> Self {
        todo!()
    }

    fn sqrt_m(self, _src: Self, _mask: Self::Mask) -> Self {
        todo!()
    }

    fn sqrt_z(self, _mask: Self::Mask) -> Self {
        todo!()
    }

    fn rsqrt_c(self, _mask: Self::Mask) -> Self {
        todo!()
    }

    fn rsqrt_m(self, _src: Self, _mask: Self::Mask) -> Self {
        todo!()
    }

    fn rsqrt_z(self, _mask: Self::Mask) -> Self {
        todo!()
    }

    fn rcp_c(self, _mask: Self::Mask) -> Self {
        todo!()
    }

    fn rcp_m(self, _src: Self, _mask: Self::Mask) -> Self {
        todo!()
    }

    fn rcp_z(self, _mask: Self::Mask) -> Self {
        todo!()
    }

    fn floor_c(self, _mask: Self::Mask) -> Self {
        todo!()
    }

    fn floor_m(self, _src: Self, _mask: Self::Mask) -> Self {
        todo!()
    }

    fn floor_z(self, _mask: Self::Mask) -> Self {
        todo!()
    }

    fn ceil_c(self, _mask: Self::Mask) -> Self {
        todo!()
    }

    fn ceil_m(self, _src: Self, _mask: Self::Mask) -> Self {
        todo!()
    }

    fn ceil_z(self, _mask: Self::Mask) -> Self {
        todo!()
    }

    fn round_c(self, _mask: Self::Mask) -> Self {
        todo!()
    }

    fn round_m(self, _src: Self, _mask: Self::Mask) -> Self {
        todo!()
    }

    fn round_z(self, _mask: Self::Mask) -> Self {
        todo!()
    }

    fn trunc_c(self, _mask: Self::Mask) -> Self {
        todo!()
    }

    fn trunc_m(self, _src: Self, _mask: Self::Mask) -> Self {
        todo!()
    }

    fn trunc_z(self, _mask: Self::Mask) -> Self {
        todo!()
    }

    fn fract_c(self, _mask: Self::Mask) -> Self {
        todo!()
    }

    fn fract_m(self, _src: Self, _mask: Self::Mask) -> Self {
        todo!()
    }

    fn fract_z(self, _mask: Self::Mask) -> Self {
        todo!()
    }

    fn mul_sign_c(self, _mask: Self::Mask, _sign: Self) -> Self {
        todo!()
    }

    fn mul_sign_m(self, _src: Self, _mask: Self::Mask, _sign: Self) -> Self {
        todo!()
    }

    fn mul_sign_z(self, _mask: Self::Mask, _sign: Self) -> Self {
        todo!()
    }

    fn signed_zero_c(self, _mask: Self::Mask) -> Self {
        todo!()
    }

    fn signed_zero_m(self, _src: Self, _mask: Self::Mask) -> Self {
        todo!()
    }

    fn signed_zero_z(self, _mask: Self::Mask) -> Self {
        todo!()
    }

    fn next_up_c(self, _mask: Self::Mask) -> Self {
        todo!()
    }

    fn next_up_m(self, _src: Self, _mask: Self::Mask) -> Self {
        todo!()
    }

    fn next_up_z(self, _mask: Self::Mask) -> Self {
        todo!()
    }

    fn next_down_c(self, _mask: Self::Mask) -> Self {
        todo!()
    }

    fn next_down_m(self, _src: Self, _mask: Self::Mask) -> Self {
        todo!()
    }

    fn next_down_z(self, _mask: Self::Mask) -> Self {
        todo!()
    }

    fn mix(self, a: Self, b: Self) -> Self {
        todo!()
    }
}

use core::fmt;

impl<V: PrettyPrintScalar> fmt::Display for Compensated<V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <V as PrettyPrintScalar>::fmt(self.value, self.error, f)
    }
}

trait PrettyPrintScalar: ScalarValue {
    fn fmt(value: Self, error: Self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

impl PrettyPrintScalar for f32 {
    fn fmt(value: Self, error: Self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if error == 0.0 {
            write!(f, "{value}")
        } else {
            write!(f, "{}", (value as f64) + (error as f64))
        }
    }
}

impl PrettyPrintScalar for f64 {
    fn fmt(mut value: Self, mut error: Self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if error == 0.0 {
            return write!(f, "{value}");
        };

        if value < 0.0 {
            write!(f, "-")?;

            value = -value;
            error = -error;
        }

        let c = Compensated { value, error };

        let int_part = Compensated::new(c.value().trunc());
        let mut frac_part = c - int_part;

        write!(f, "{}", int_part.value as u64)?;

        if frac_part.value() == 0.0 {
            return Ok(());
        }

        f.write_str(".")?;

        let p = f.precision().unwrap_or(17); // default to max precision for f64

        for _ in 0..p {
            frac_part *= 10.0;

            let digit = frac_part.value().trunc();

            write!(f, "{}", digit as u64)?;

            frac_part -= Compensated::new(digit);

            if frac_part.value == 0.0 && frac_part.error == 0.0 {
                break;
            }
        }

        Ok(())
    }
}
