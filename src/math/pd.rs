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

    #[rustfmt::skip]
    #[inline(always)]
    fn erf(x: Self::Vf) -> Self::Vf {
        let a0 = Vf64::<S>::one();
        let a1 = Vf64::<S>::splat(0.141047395888);
        let a2 = Vf64::<S>::splat(0.0895246554342);
        let a3 = Vf64::<S>::splat(0.024538446357);
        let a4 = Vf64::<S>::splat(0.00339526031482);
        let a5 = Vf64::<S>::splat(0.00127101693092);
        let a6 = Vf64::<S>::splat(0.000343596421733);
        let a7 = Vf64::<S>::splat(-0.0000282694821623);
        let a8 = Vf64::<S>::splat(0.0000153312079619);
        let a9 = Vf64::<S>::splat(0.00000806034527525);
        let a10 = Vf64::<S>::splat(-0.00000491119825703);
        let a11 = Vf64::<S>::splat(0.00000190850200269);
        let a12 = Vf64::<S>::splat(-4.5433487004e-7);
        let a13 = Vf64::<S>::splat(7.5111413853e-8);
        let a14 = Vf64::<S>::splat(-7.4944859806e-9);
        let a15 = Vf64::<S>::splat(3.8381832932e-10);

        let b = a0 - (a0 - x.abs()); // crush denormals
        let b2 = b * b;
        let b4 = b2 * b2;

        let r = poly_15(
            b, b2, b4, b4 * b4,
            a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,
        );

        let r2 = r * r;
        let r4 = r2 * r2;
        let r8 = r4 * r4;

        (a0 - a0 / r8).copysign(x)
    }

    fn erfinv(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
}
