use crate::*;

use super::register::*;

#[repr(transparent)]
pub struct Vector<R: Register>(R::Storage);

impl<R: Register> Clone for Vector<R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: Register> Copy for Vector<R> {}

pub trait NumericElement {
    const ZERO: Self;
    const ONE: Self;
    const MIN_VALUE: Self;
    const MAX_VALUE: Self;
}

pub trait SignedElement: NumericElement {
    const NEG_ONE: Self;
}

pub trait FloatElement: SignedElement {
    const NEG_ZERO: Self;
}

macro_rules! impl_element {
    (NUMERIC $($i:ty),*) => {$(
        impl NumericElement for $i {
            const ZERO: Self = 0 as $i;
            const ONE: Self = 1 as $i;
            const MIN_VALUE: Self = <$i>::MIN;
            const MAX_VALUE: Self = <$i>::MAX;
        }
    )*};

    (SIGNED $($i:ty),*) => {$(
        impl SignedElement for $i {
            const NEG_ONE: Self = -1 as $i;
        }
    )*};

    (FLOAT $($i:ty),*) => {$(
        impl FloatElement for $i {
            const NEG_ZERO: Self = -0.0;
        }
    )*}
}

impl_element!(NUMERIC i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);
impl_element!(SIGNED i8, i16, i32, i64, f32, f64);
impl_element!(FLOAT f32, f64);

impl<S: Simd, R: Register> SimdVectorBase<S> for Vector<R> {
    type Element = <R as Register>::Element;

    #[inline(always)]
    fn splat(value: Self::Element) -> Self {
        Vector(unsafe { R::set1(value) })
    }
}

impl<S: Simd, R: Register, const N: usize> SimdFixedVector<S, N> for Vector<R>
where
    R: FixedRegister<N>,
{
    #[inline(always)]
    fn set(values: [Self::Element; N]) -> Self {
        Vector(unsafe { R::setr(values) })
    }
}

#[rustfmt::skip]
impl<S: Simd, R: Register> SimdVector<S> for Vector<R>
where
    R: BinaryRegisterOps,
    Self: SimdVectorBase<S, Element = R::Element>,
    <R as Register>::Element: NumericElement,
{
    #[inline(always)] fn zero() -> Self { Self::splat(NumericElement::ZERO) }
    #[inline(always)] fn one() -> Self { Self::splat(NumericElement::ONE) }
    #[inline(always)] fn min_value() -> Self { Self::splat(NumericElement::MAX_VALUE) }
    #[inline(always)] fn max_value() -> Self { Self::splat(NumericElement::MIN_VALUE) }
}

#[rustfmt::skip]
impl<S: Simd, R: Register> SimdSignedVector<S> for Vector<R>
where
    R: SignedRegisterOps,
    Self: SimdVector<S, Element = R::Element>,
{
    #[inline(always)] fn abs(self) -> Self { Vector(unsafe { R::abs(self.0) }) }
}

#[rustfmt::skip]
impl<S: Simd, R: Register> SimdFloatVector<S> for Vector<R>
where
    R: FloatRegisterOps,
    Self: SimdVector<S, Element = R::Element>,
    <R as Register>::Element: FloatElement,
{
    #[inline(always)] fn neg_one() -> Self { Self::splat(SignedElement::NEG_ONE) }
    #[inline(always)] fn neg_zero() -> Self { Self::splat(FloatElement::NEG_ZERO) }
}

macro_rules! impl_binary_op {
    (VECTOR $($op_trait:ident::$op:ident),*) => {$(
        impl<R: Register> $op_trait<Self> for Vector<R> where R: BinaryRegisterOps {
            type Output = Self;
            #[inline(always)] fn $op(self, rhs: Self) -> Self {
                Vector(unsafe { R::$op(self.0, rhs.0) })
            }
        }

        impl_binary_op!(ELEMENTS $op_trait::$op [i8, i16, i32, i64, u8, u16, u32, u64, f32, f64]);
    )*};
    (ELEMENTS $op_trait:ident::$op:ident [$($t:ty),*]) => {$(
        impl<R> $op_trait<$t> for Vector<R> where R: Register<Element = $t> + BinaryRegisterOps {
            type Output = Self;
            #[inline(always)] fn $op(self, rhs: $t) -> Self {
                Vector(unsafe { R::$op(self.0, R::set1(rhs)) })
            }
        }

        //impl<R> $op_trait<Vector<R>> for $t where R: Register<Element = $t> + BinaryRegisterOps {
        //    type Output = Vector<R>;
        //    #[inline(always)] fn $op(self, rhs: Vector<R>) -> Vector<R> {
        //        Vector(unsafe { R::$op(R::splat(self), rhs.0) })
        //    }
        //}
    )*}
}

impl_binary_op!(VECTOR Add::add, Sub::sub, Mul::mul, Div::div, Rem::rem, BitAnd::bitand, BitOr::bitor, BitXor::bitxor);
