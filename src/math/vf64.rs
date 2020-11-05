use super::{common::*, *};

impl<S: Simd> SimdVectorizedMathInternal<S> for f64
where
    <S as Simd>::Vf64: SimdFloatVector<S, Element = f64>,
{
    type Vf = <S as Simd>::Vf64;

    fn sin_cos(xx: Self::Vf) -> (Self::Vf, Self::Vf) {
        unimplemented!()
    }

    fn sinh(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }

    fn tanh(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn asin(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn acos(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn atan(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn atan2(y: Self::Vf, x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn asinh(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn acosh(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn atanh(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn exp(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn exph(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn exp2(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn exp10(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn powf(x: Self::Vf, e: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn ln(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn ln_1p(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn log2(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn log10(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn erf(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn ierf(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
}
