#![allow(clippy::excessive_precision)]

pub mod consts;
pub mod policy;

use crate::{
    Vector,
    register::{FloatRegister, Register},
};

pub mod internal;

use internal::MathInternal;
use policy::DefaultPolicy;

impl<E, R: MathInternal<E>> Vector<R>
where
    R: FloatRegister<Element = E>,
{
    #[inline(always)]
    pub fn sincos(self) -> (Self, Self) {
        R::sincos::<DefaultPolicy>(self)
    }

    #[inline(always)]
    pub fn powiv(self, e: Vector<R::Signed>) -> Self {
        R::powiv::<DefaultPolicy>(self, e)
    }
}
