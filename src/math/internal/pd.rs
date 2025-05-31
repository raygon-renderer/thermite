use super::*;

impl<R> MathInternal<f64> for R
where
    R: FloatRegister<Element = f64>,
{
    fn sincos<P: Policy>(x: Vf<Self>) -> (Vf<Self>, Vf<Self>) {
        todo!()
    }
}
