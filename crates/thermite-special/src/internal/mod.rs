#![allow(clippy::excessive_precision)]

use thermite::{
    math::{FloatConsts, policy::Policy},
    register::FloatRegister,
    vector::Vector,
};

use super::SpecialMathWithPolicy as _;

pub(crate) type Vf<R> = Vector<R>;
pub(crate) type Vu<R> = Vector<<R as FloatRegister>::Bits>;
pub(crate) type Vs<R> = Vector<<R as FloatRegister>::Signed>;

mod ps;

pub trait SpecialMathInternal<E: FloatConsts>: FloatRegister<Element = E> {
    fn tgamma<P: Policy>(x: Vf<Self>) -> Vf<Self>;
    fn lgamma<P: Policy>(x: Vf<Self>) -> (Vf<Self>, Vf<Self>);
}
