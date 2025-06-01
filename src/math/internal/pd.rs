use super::*;

impl<R> MathInternal<f64> for R
where
    R: FloatRegister<Element = f64>,
{
    fn sincos<P: Policy>(x: Vf<Self>) -> (Vf<Self>, Vf<Self>) {
        todo!()
    }

    fn sinh<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn cosh<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn tanh<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn asin<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn acos<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn atan<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn atan2<P: Policy>(y: Vf<Self>, x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn asinh<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn acosh<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn atanh<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn exp<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn exph<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn exp2<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn exp10<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn exp_m1<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn powf<P: Policy>(x: Vf<Self>, e: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn cbrt<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn ln<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn ln1p<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn log2<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn log10<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn erf<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn erfc<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn erfinv<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }
}
