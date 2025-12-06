// #![no_std]
#![allow(unsafe_op_in_unsafe_fn)]

use thermite::{
    Mask, Vector,
    mask::Selectable,
    math::FloatConsts,
    register::{CastMaskRegister, CastRegister, FloatElement, FloatRegister, MaskRegister},
};

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};

use num_traits::{
    ConstOne, ConstZero, MulAdd, MulAddAssign, Num, NumCast, One, Signed, ToPrimitive, Zero, float::FloatCore,
};

//pub mod polyfills;
//pub mod reg;
pub mod consts;
use consts::SplitFloatConsts;

mod math;

pub trait CompensatedElement: FloatElement + Signed + FloatConsts + SplitFloatConsts<Self> {
    /// for Veltkamp's splitting, defined as 2^(ceil(p/2)) + 1,
    /// where p is the number of bits in the significand.
    const SPLITTER: Self;

    /// Empirical maximum |x| for which the erf_inv Maclaurin series converges
    /// within 64 terms to full precision.
    const MAX_ERFINV_SERIES: Self;
}

impl CompensatedElement for f32 {
    const SPLITTER: Self = ((1u64 << 12) + 1) as f32; // 2^12 + 1

    const MAX_ERFINV_SERIES: Self = 0.75;
}

impl CompensatedElement for f64 {
    const SPLITTER: Self = ((1u64 << 27) + 1) as f64; // 2^27 + 1

    const MAX_ERFINV_SERIES: Self = 0.545;
}

pub trait CompensatedRegister: FloatRegister<Element: CompensatedElement> {}

impl<R: FloatRegister> CompensatedRegister for R where R::Element: CompensatedElement {}

#[repr(C)]
pub struct Compensated<R: CompensatedRegister> {
    value: Vector<R>,
    error: Vector<R>,
}

const _: () = {
    use core::fmt;

    impl<R: CompensatedRegister> fmt::Debug for Compensated<R> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("Compensated")
                .field("value", &self.value)
                .field("error", &self.error)
                .finish()
        }
    }
};

impl<R: CompensatedRegister> Selectable<R> for Compensated<R> {
    #[inline(always)]
    fn select<M: MaskRegister>(mask: Mask<M>, truthy: Self, falsy: Self) -> Self
    where
        R: CastMaskRegister<M, Lanes = M::Lanes>,
    {
        let mask: Mask<R> = mask.cast(); // do this upfront for both parts

        Compensated {
            value: mask.select(truthy.value, falsy.value),
            error: mask.select(truthy.error, falsy.error),
        }
    }
}

impl<R: CompensatedRegister> Clone for Compensated<R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: CompensatedRegister> Copy for Compensated<R> {}

impl<R: CompensatedRegister> Compensated<R> {
    pub const LANES: usize = <R::Lanes as thermite::generic_array::typenum::Unsigned>::USIZE;

    pub const EMPTY: Self = Self::new(Vector::EMPTY);
    pub const ZERO: Self = Self::new(Vector::ZERO);
    pub const ONE: Self = Self::new(Vector::ONE);
    pub const NEG_ONE: Self = Self::new(Vector::NEG_ONE);
    pub const MIN_POSITIVE: Self = Self::new(Vector::MIN_POSITIVE);
    pub const TWO: Self = Self::new(Vector::TWO);
    pub const HALF: Self = Self::new(Vector::HALF);

    pub const MIN: Self = Self::new(Vector::MIN);
    pub const MAX: Self = Self::new(Vector::MAX);

    pub const NAN: Self = Self::new(Vector::NAN);
    pub const INFINITY: Self = Self::new(Vector::INFINITY);
    pub const NEG_INFINITY: Self = Self::new(Vector::NEG_INFINITY);
    pub const NEG_ZERO: Self = Self::new(Vector::NEG_ZERO);
    pub const EPSILON: Self = FloatConsts::EPSILON; // use split epsilon

    #[inline(always)]
    pub const fn new(value: Vector<R>) -> Self {
        Self {
            value,
            error: Vector::ZERO,
        }
    }

    #[inline(always)]
    pub fn splat(value: R::Element) -> Self {
        Self::new(Vector::splat(value))
    }

    #[inline(always)]
    pub const fn splat_const(value: R::Element) -> Self {
        Self::new(Vector::splat_const(value))
    }

    #[inline(always)]
    pub const fn from_parts(value: Vector<R>, error: Vector<R>) -> Self {
        Self { value, error }
    }

    #[inline(always)]
    pub fn value(&self) -> Vector<R> {
        self.value + self.error
    }

    #[inline(always)]
    pub const fn raw_value(&self) -> Vector<R> {
        self.value
    }

    #[inline(always)]
    pub const fn error(&self) -> Vector<R> {
        self.error
    }

    #[inline(always)]
    pub fn cast<T>(self) -> Compensated<T>
    where
        T: CastRegister<R> + CompensatedRegister,
        R: CastRegister<T>,
    {
        let value = self.value.cast();

        let error = if const { size_of::<T::Element>() < size_of::<R::Element>() } {
            // accumulate downcasting error
            self.error + (self.value - value.cast())
        } else {
            self.error
        };

        Compensated {
            value,
            error: error.cast(),
        }
    }

    #[inline(always)]
    pub(crate) fn renormalized(value: Vector<R>, error: Vector<R>) -> Self {
        let sum = value + error;
        let err = error + (value - sum);
        Self { value: sum, error: err }
    }

    #[inline(always)]
    pub fn normalize(self) -> Self {
        Self::renormalized(self.value, self.error)
    }

    #[inline(always)]
    pub fn sqrt(self) -> Self {
        let s = self.value.sqrt();

        let (p, e) = two_prod(s, s);

        // sum of differences
        let remainder = (self.value - p) + (self.error - e);

        // correction term
        let corr = remainder / (s + s);

        Self::renormalized(s, corr)
    }

    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        let mask = self.value().cmp_lt(other.value());
        Compensated {
            value: mask.select(self.value, other.value),
            error: mask.select(self.error, other.error),
        }
    }

    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        let mask = self.value().cmp_gt(other.value());
        Compensated {
            value: mask.select(self.value, other.value),
            error: mask.select(self.error, other.error),
        }
    }

    #[inline(always)]
    pub fn min_max(self, other: Self) -> (Self, Self) {
        let mask = self.value().cmp_lt(other.value());

        let min = Compensated {
            value: mask.select(self.value, other.value),
            error: mask.select(self.error, other.error),
        };

        let max = Compensated {
            value: mask.select(other.value, self.value),
            error: mask.select(other.error, self.error),
        };

        (min, max)
    }

    #[inline(always)]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        let x = self.value();
        let min_value = min.value();
        let max_value = max.value();

        let is_lt = x.cmp_lt(min_value);
        let is_gt = x.cmp_gt(max_value);

        let value = is_lt.select(min.value, is_gt.select(max.value, self.value));
        let error = is_lt.select(min.error, is_gt.select(max.error, self.error));

        Compensated { value, error }
    }

    #[inline(always)]
    pub fn cmp_eq(&self, other: Self) -> Mask<R> {
        self.value.cmp_eq(other.value) & self.error.cmp_eq(other.error)
    }

    #[inline(always)]
    pub fn conditional_negate(self, mask: Mask<R>) -> Self {
        Compensated {
            value: self.value.conditional_negate(mask),
            error: self.error.conditional_negate(mask),
        }
    }
}

#[inline(always)]
fn two_sum<R: CompensatedRegister>(a: Vector<R>, b: Vector<R>) -> (Vector<R>, Vector<R>) {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    (s, e)
}

#[inline(always)]
fn two_diff<R: CompensatedRegister>(a: Vector<R>, b: Vector<R>) -> (Vector<R>, Vector<R>) {
    let s = a - b;
    let v = s - a;
    let e = (a - (s - v)) - (b + v);
    (s, e)
}

#[inline(always)]
fn two_prod<R: CompensatedRegister>(a: Vector<R>, b: Vector<R>) -> (Vector<R>, Vector<R>)
where
    R::Element: CompensatedElement,
{
    if R::HAS_TRUE_FMA {
        let p = a * b;
        let e = a.mul_sub(b, p);

        return (p, e);
    }

    let splitter = Vector::splat(R::Element::SPLITTER);

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

impl<R: CompensatedRegister> Add for Compensated<R> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        let (s, e) = two_sum(self.value, rhs.value);
        Self::renormalized(s, e + self.error + rhs.error)
    }
}

// for testing
const ALLOW_UNNORMALIZED: bool = true;

impl<R: CompensatedRegister> Compensated<R> {
    /// Accumulate rhs into self without renormalization.
    ///
    /// This should only be used in specific scenarios where renormalization is not desired,
    /// such as within iterative series expansions.
    #[inline(always)]
    pub fn accumulate_unnormalized(&mut self, rhs: Self) {
        if ALLOW_UNNORMALIZED {
            let (s, e) = two_sum(self.value, rhs.value);
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
            let (s, e) = two_diff(self.value, rhs.value);
            self.value = s;
            self.error = e + (self.error - rhs.error);
        } else {
            *self -= rhs;
        }
    }
}

impl<R: CompensatedRegister> Sub for Compensated<R> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        let (s, e) = two_diff(self.value, rhs.value);
        Self::renormalized(s, e + (self.error - rhs.error))
    }
}

impl<R: CompensatedRegister> Mul for Compensated<R> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let (p, e) = two_prod(self.value, rhs.value);

        let e = if R::HAS_TRUE_FMA {
            // 2 fmas
            self.error.mul_add(rhs.value, self.value.mul_add(rhs.error, e))
        } else {
            // 2 muls, 2 adds
            let cross1 = self.value * rhs.error;
            let cross2 = rhs.value * self.error;

            e + cross1 + cross2
        };

        Self::renormalized(p, e)
    }
}

impl<R: CompensatedRegister> Div for Compensated<R> {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        let q1 = self.value / rhs.value;

        let (p_hi, p_lo) = two_prod(q1, rhs.value);

        // calculate the remainder r
        let r = (self.value - p_hi) - p_lo + self.error - (q1 * rhs.error);

        Self::renormalized(q1, r / rhs.value)
    }
}

impl<R: CompensatedRegister> Rem for Compensated<R> {
    type Output = Self;

    #[inline(always)]
    fn rem(self, rhs: Self) -> Self {
        let q = self / rhs;
        let n = Compensated::new(-q.value.trunc());
        rhs.mul_add(n, self)
    }
}

impl<R: CompensatedRegister> Add<Vector<R>> for Compensated<R> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Vector<R>) -> Self {
        let (s, e) = two_sum(self.value, rhs);
        Self::renormalized(s, e + self.error)
    }
}

impl<R: CompensatedRegister> Sub<Vector<R>> for Compensated<R> {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline(always)]
    fn sub(self, rhs: Vector<R>) -> Self {
        let (s, e) = two_diff(self.value, rhs);
        Self::renormalized(s, e + self.error)
    }
}

impl<R: CompensatedRegister> Mul<Vector<R>> for Compensated<R> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Vector<R>) -> Self {
        // (a0 + a1) * b = a0*b + a1*b
        let (p, e1) = two_prod(self.value, rhs);
        // We just add a1*b to the error term
        Self::renormalized(p, self.error.mul_adde(rhs, e1))
    }
}

impl<R: CompensatedRegister> Div<Vector<R>> for Compensated<R> {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Vector<R>) -> Self {
        // same as regular division, but rhs has no error term
        let q1 = self.value / rhs;

        let (p_hi, p_lo) = two_prod(q1, rhs);

        // calculate the remainder r
        let r = (self.value - p_hi) - p_lo + self.error;

        Self::renormalized(q1, r / rhs)
    }
}

impl<R: CompensatedRegister> Rem<Vector<R>> for Compensated<R> {
    type Output = Self;

    #[inline(always)]
    fn rem(self, rhs: Vector<R>) -> Self {
        let q = self / rhs;
        let n = Compensated::new(-q.value.trunc());
        MulAdd::mul_add(n, rhs, self)
    }
}

impl<R: CompensatedRegister> Compensated<R> {
    #[inline(always)]
    pub fn mul_sub(self, b: Self, c: Self) -> Self {
        let (p, e_prod_base) = two_prod(self.value, b.value);
        let (s, e_diff) = two_diff(p, c.value);

        let e_prod = if R::HAS_TRUE_FMA {
            // 2 fmas
            self.error.mul_add(b.value, self.value.mul_add(b.error, e_prod_base))
        } else {
            // 2 muls, 2 adds
            let cross1 = self.value * b.error;
            let cross2 = b.value * self.error;

            e_prod_base + cross1 + cross2
        };

        // Subtract c.error because the operation is (a*b) - c
        // The total error is the product error + subtraction error - c's error component
        Self::renormalized(s, e_prod + e_diff - c.error)
    }
}

impl<R: CompensatedRegister> MulAdd for Compensated<R> {
    type Output = Self;

    #[inline(always)]
    fn mul_add(self, b: Self, c: Self) -> Self {
        let (p, e_prod_base) = two_prod(self.value, b.value);
        let (s, e_sum) = two_sum(p, c.value);

        let e_prod = if R::HAS_TRUE_FMA {
            // 2 fmas
            self.error.mul_add(b.value, self.value.mul_add(b.error, e_prod_base))
        } else {
            // 2 muls, 2 adds
            let cross1 = self.value * b.error;
            let cross2 = b.value * self.error;
            e_prod_base + cross1 + cross2
        };

        Self::renormalized(s, e_prod + e_sum + c.error)
    }
}

impl<R: CompensatedRegister> MulAdd<Vector<R>> for Compensated<R> {
    type Output = Self;

    #[inline(always)]
    fn mul_add(self, b: Vector<R>, c: Self) -> Self {
        let (p, e_prod_base) = two_prod(self.value, b);
        let (s, e_sum) = two_sum(p, c.value);

        let e_prod = if R::HAS_TRUE_FMA {
            // 2 fmas
            self.error.mul_add(b, e_prod_base)
        } else {
            // 1 mul, 1 add
            e_prod_base + (self.error * b)
        };

        Self::renormalized(s, e_prod + e_sum + c.error)
    }
}

impl<R: CompensatedRegister> MulAdd<Vector<R>, Vector<R>> for Compensated<R> {
    type Output = Self;

    #[inline(always)]
    fn mul_add(self, a: Vector<R>, b: Vector<R>) -> Self {
        let (p, e_prod_base) = two_prod(self.value, a);
        let (s, e_sum) = two_sum(p, b);

        let e_prod = if R::HAS_TRUE_FMA {
            // 1 fma
            self.error.mul_add(a, e_prod_base)
        } else {
            // 1 mul, 1 add
            e_prod_base + (self.error * a)
        };

        Self::renormalized(s, e_prod + e_sum)
    }
}

impl<R: CompensatedRegister> MulAdd<Self, Vector<R>> for Compensated<R> {
    type Output = Self;

    #[inline(always)]
    fn mul_add(self, a: Self, b: Vector<R>) -> Self::Output {
        let (p, e_prod_base) = two_prod(self.value, a.value);
        let (s, e_sum) = two_sum(p, b);

        let e_prod = if R::HAS_TRUE_FMA {
            // 2 fmas
            self.error.mul_add(a.value, self.value.mul_add(a.error, e_prod_base))
        } else {
            // 2 muls, 2 adds
            let cross1 = self.value * a.error;
            let cross2 = a.value * self.error;

            e_prod_base + cross1 + cross2
        };

        Self::renormalized(s, e_prod + e_sum)
    }
}

impl<R: CompensatedRegister> AddAssign for Compensated<R> {
    #[inline(always)]
    fn add_assign(&mut self, other: Self) {
        *self = self.add(other);
    }
}

impl<R: CompensatedRegister> SubAssign for Compensated<R> {
    #[inline(always)]
    fn sub_assign(&mut self, other: Self) {
        *self = self.sub(other);
    }
}

impl<R: CompensatedRegister> MulAssign for Compensated<R> {
    #[inline(always)]
    fn mul_assign(&mut self, other: Self) {
        *self = self.mul(other);
    }
}

impl<R: CompensatedRegister> DivAssign for Compensated<R> {
    #[inline(always)]
    fn div_assign(&mut self, other: Self) {
        *self = self.div(other);
    }
}

impl<R: CompensatedRegister> AddAssign<Vector<R>> for Compensated<R> {
    #[inline(always)]
    fn add_assign(&mut self, a: Vector<R>) {
        *self = self.add(Compensated::new(a));
    }
}

impl<R: CompensatedRegister> SubAssign<Vector<R>> for Compensated<R> {
    #[inline(always)]
    fn sub_assign(&mut self, a: Vector<R>) {
        *self = self.sub(Compensated::new(a));
    }
}

impl<R: CompensatedRegister> MulAssign<Vector<R>> for Compensated<R> {
    #[inline(always)]
    fn mul_assign(&mut self, a: Vector<R>) {
        *self = self.mul(Compensated::new(a));
    }
}

impl<R: CompensatedRegister> DivAssign<Vector<R>> for Compensated<R> {
    #[inline(always)]
    fn div_assign(&mut self, a: Vector<R>) {
        *self = self.div(Compensated::new(a));
    }
}

impl<R: CompensatedRegister> RemAssign<Vector<R>> for Compensated<R> {
    #[inline(always)]
    fn rem_assign(&mut self, a: Vector<R>) {
        *self = self.rem(Compensated::new(a));
    }
}

impl<R: CompensatedRegister, A, B> MulAddAssign<A, B> for Compensated<R>
where
    Self: MulAdd<A, B, Output = Self>,
{
    #[inline(always)]
    fn mul_add_assign(&mut self, a: A, b: B) {
        *self = self.mul_add(a, b);
    }
}

impl<R: CompensatedRegister> One for Compensated<R> {
    #[inline(always)]
    fn one() -> Self {
        Self::new(Vector::ONE)
    }
}

impl<R: CompensatedRegister> Zero for Compensated<R> {
    #[inline(always)]
    fn zero() -> Self {
        Self::new(Vector::ZERO)
    }

    /// Returns `true` if all elements are zero.
    #[inline(always)]
    fn is_zero(&self) -> bool {
        (self.value.is_zero() & self.error.is_zero()).all()
    }
}

impl<R: CompensatedRegister> ConstOne for Compensated<R> {
    const ONE: Self = Self::new(Vector::ONE);
}

impl<R: CompensatedRegister> ConstZero for Compensated<R> {
    const ZERO: Self = Self::new(Vector::ZERO);
}

impl<R: CompensatedRegister> PartialEq for Compensated<R> {
    /// Returns true if all elements are equal.
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        (self.value.cmp_eq(other.value) & self.error.cmp_eq(other.error)).all()
    }

    /// Returns true if any elements are not equal.
    #[allow(clippy::partialeq_ne_impl)]
    #[inline(always)]
    fn ne(&self, other: &Self) -> bool {
        (self.value.cmp_ne(other.value) | self.error.cmp_ne(other.error)).any()
    }
}

impl<R: CompensatedRegister> Num for Compensated<R> {
    type FromStrRadixErr = <R::Element as Num>::FromStrRadixErr;

    #[inline(always)]
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        Ok(Self::new(Vector::from_str_radix(str, radix)?))
    }
}

impl<R: CompensatedRegister> Neg for Compensated<R> {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            value: -self.value,
            error: -self.error,
        }
    }
}

impl<R: CompensatedRegister> Signed for Compensated<R> {
    /// Computes the absolute value of `self`, without losing precision.
    #[inline(always)]
    fn abs(&self) -> Self {
        let sign = self.value() & Vector::NEG_ZERO;

        // the final value may differ in sign from the components,
        // so negate both components based on the final value's sign
        Self {
            value: self.value ^ sign,
            error: self.error ^ sign,
        }
    }

    #[inline(always)]
    fn abs_sub(&self, other: &Self) -> Self {
        (*self - *other).max(Self::ZERO)
    }

    #[inline(always)]
    fn is_negative(&self) -> bool {
        Signed::is_negative(&self.value())
    }

    #[inline(always)]
    fn is_positive(&self) -> bool {
        Signed::is_positive(&self.value())
    }

    #[inline(always)]
    fn signum(&self) -> Self {
        Compensated::new(Signed::signum(&self.value()))
    }
}

impl<R: CompensatedRegister> PartialOrd for Compensated<R> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.value().partial_cmp(&other.value())
    }
}

#[rustfmt::skip]
impl<R: CompensatedRegister> ToPrimitive for Compensated<R>
where
    R::Element: num_traits::ToPrimitive,
{
    #[inline(always)] fn to_isize(&self) -> Option<isize> { self.value().extract::<0>().to_isize() }
    #[inline(always)] fn to_i8(&self) -> Option<i8> { self.value().extract::<0>().to_i8() }
    #[inline(always)] fn to_i16(&self) -> Option<i16> { self.value().extract::<0>().to_i16() }
    #[inline(always)] fn to_i32(&self) -> Option<i32> { self.value().extract::<0>().to_i32() }
    #[inline(always)] fn to_i128(&self) -> Option<i128> { self.value().extract::<0>().to_i128() }
    #[inline(always)] fn to_usize(&self) -> Option<usize> { self.value().extract::<0>().to_usize() }
    #[inline(always)] fn to_u8(&self) -> Option<u8> { self.value().extract::<0>().to_u8() }
    #[inline(always)] fn to_u16(&self) -> Option<u16> { self.value().extract::<0>().to_u16() }
    #[inline(always)] fn to_u32(&self) -> Option<u32> { self.value().extract::<0>().to_u32() }
    #[inline(always)] fn to_u128(&self) -> Option<u128> { self.value().extract::<0>().to_u128() }
    #[inline(always)] fn to_f32(&self) -> Option<f32> { self.value().extract::<0>().to_f32() }
    #[inline(always)] fn to_f64(&self) -> Option<f64> { self.value().extract::<0>().to_f64() }
    #[inline(always)] fn to_i64(&self) -> Option<i64> { self.value().extract::<0>().to_i64() }
    #[inline(always)] fn to_u64(&self) -> Option<u64> { self.value().extract::<0>().to_u64() }
}

impl<R: CompensatedRegister> NumCast for Compensated<R> {
    #[inline(always)]
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        <Vector<R> as NumCast>::from(n).map(Compensated::new)
    }
}

#[rustfmt::skip]
impl<R: CompensatedRegister> FloatCore for Compensated<R> {
    #[inline(always)] fn is_nan(self) -> bool { FloatCore::is_nan(self.value()) }
    #[inline(always)] fn is_infinite(self) -> bool { FloatCore::is_infinite(self.value()) }
    #[inline(always)] fn is_finite(self) -> bool { FloatCore::is_finite(self.value()) }
    #[inline(always)] fn is_normal(self) -> bool { FloatCore::is_normal(self.value()) }
    #[inline(always)] fn is_subnormal(self) -> bool { FloatCore::is_subnormal(self.value()) }
    #[inline(always)] fn floor(self) -> Self { Compensated::new(self.value().floor()) }
    #[inline(always)] fn ceil(self) -> Self { Compensated::new(self.value().ceil()) }
    #[inline(always)] fn round(self) -> Self { Compensated::new(self.value().round()) }
    #[inline(always)] fn trunc(self) -> Self { Compensated::new(self.value().trunc()) }
    #[inline(always)] fn fract(self) -> Self { Compensated::new(self.value().fract()) }
    #[inline(always)] fn abs(self) -> Self { Signed::abs(&self) }
    #[inline(always)] fn signum(self) -> Self { Signed::signum(&self) }
    #[inline(always)] fn is_sign_positive(self) -> bool { Signed::is_positive(&self) }
    #[inline(always)] fn is_sign_negative(self) -> bool { Signed::is_negative(&self) }

    #[inline(always)] fn min(self, other: Self) -> Self { self.min(other) }
    #[inline(always)] fn max(self, other: Self) -> Self { self.max(other) }
    #[inline(always)] fn clamp(self, min: Self, max: Self) -> Self { self.clamp(min, max) }

    #[inline(always)] fn recip(self) -> Self { Self::ONE / self }
    #[inline(always)] fn infinity() -> Self { Self::INFINITY }
    #[inline(always)] fn neg_infinity() -> Self { Self::NEG_INFINITY }
    #[inline(always)] fn nan() -> Self { Self::NAN }
    #[inline(always)] fn neg_zero() -> Self { Self::NEG_ZERO }
    #[inline(always)] fn min_value() -> Self { Self::MIN }
    #[inline(always)] fn min_positive_value() -> Self { Self::MIN_POSITIVE }
    #[inline(always)] fn epsilon() -> Self { Self::EPSILON }
    #[inline(always)] fn max_value() -> Self { Self::MAX }
    #[inline(always)] fn classify(self) -> core::num::FpCategory { FloatCore::classify(self.value()) }
    #[inline(always)] fn to_degrees(self) -> Self { self * Self::FRAC_180_PI }
    #[inline(always)] fn to_radians(self) -> Self { self * Self::FRAC_PI_180 }

    #[inline(always)]
    fn integer_decode(self) -> (u64, i16, i8) {
        FloatCore::integer_decode(self.value())
    }
}
