use crate::element::FloatElementWithBits;

use super::*;

/// A FloatElement type that can itself be used as a FloatRegister,
/// with its associated Signed and Bits types also being fully formed.
pub trait WellFormedFloatElement:
    FloatElementWithBits<
        Signed: WellFormedSignedIntegerElement<
            Element = <Self as FloatElementWithBits>::Signed,
            Storage = <Self as FloatElementWithBits>::Signed,
        >,
        Bits: WellFormedUnsignedIntegerElement<
            Element = <Self as FloatElementWithBits>::Bits,
            Storage = <Self as FloatElementWithBits>::Bits,
        >,
        USize: WellFormedUnsignedIntegerElement<Element = <Self as Element>::USize, Storage = <Self as Element>::USize>,
        ISize: WellFormedSignedIntegerElement<Element = <Self as Element>::ISize, Storage = <Self as Element>::ISize>,
    > + FloatRegister<Element = Self, Storage = Self>
{
}

/// A Signed IntegerElement type that can itself be used as a SignedIntegerRegister.
pub trait WellFormedSignedIntegerElement:
    IntegerElement<USize: CoreRegister>
    + num_traits::Signed
    + SignedIntegerRegister<Element = Self, Storage = Self, ISize = Self, USize = <Self as Element>::USize>
{
}

/// An Unsigned IntegerElement type that can itself be used as an UnsignedIntegerRegister.
pub trait WellFormedUnsignedIntegerElement:
    IntegerElement<ISize: CoreRegister>
    + IntegerRegister<Element = Self, Storage = Self, USize = Self, ISize = <Self as Element>::ISize>
    + UnsignedIntegerRegister<Element = Self, Storage = Self>
{
}

impl<F> WellFormedFloatElement for F where
    F: FloatElementWithBits<
            Signed: WellFormedSignedIntegerElement<
                Element = <F as FloatElementWithBits>::Signed,
                Storage = <F as FloatElementWithBits>::Signed,
            >,
            Bits: WellFormedUnsignedIntegerElement<
                Element = <F as FloatElementWithBits>::Bits,
                Storage = <F as FloatElementWithBits>::Bits,
            >,
            USize: WellFormedUnsignedIntegerElement<Element = <F as Element>::USize, Storage = <F as Element>::USize>,
            ISize: WellFormedSignedIntegerElement<Element = <F as Element>::ISize, Storage = <F as Element>::ISize>,
        > + FloatRegister<Element = F, Storage = F>
{
}

impl<I> WellFormedSignedIntegerElement for I where
    I: IntegerElement<USize: CoreRegister>
        + num_traits::Signed
        + SignedIntegerRegister<Element = I, Storage = I, ISize = I, USize = <I as Element>::USize>
{
}

impl<U> WellFormedUnsignedIntegerElement for U where
    U: IntegerElement<ISize: CoreRegister>
        + IntegerRegister<Element = U, Storage = U, USize = U, ISize = <U as Element>::ISize>
        + UnsignedIntegerRegister<Element = U, Storage = U>
{
}

pub trait WellFormedRegister:
    Register<
        Element: Element<USize: WellFormedUnsignedIntegerElement, ISize: WellFormedSignedIntegerElement>,
        USize: WellFormedUnsignedIntegerRegister<Element = <Self::Element as Element>::USize>,
        ISize: WellFormedSignedIntegerRegister<Element = <Self::Element as Element>::ISize>,
    >
{
}

impl<R> WellFormedRegister for R where
    R: Register<
            Element: Element<USize: WellFormedUnsignedIntegerElement, ISize: WellFormedSignedIntegerElement>,
            USize: WellFormedUnsignedIntegerRegister<Element = <R::Element as Element>::USize>,
            ISize: WellFormedSignedIntegerRegister<Element = <R::Element as Element>::ISize>,
        >
{
}

pub trait WellFormedSignedIntegerRegister:
    SignedIntegerRegister<
        Element: WellFormedSignedIntegerElement,
        USize: Register<Element = <Self::Element as Element>::USize>,
        ISize = Self,
    >
{
}

pub trait WellFormedUnsignedIntegerRegister:
    UnsignedIntegerRegister<
        Element: WellFormedUnsignedIntegerElement,
        USize = Self,
        ISize: Register<Element = <Self::Element as Element>::ISize>,
    >
{
}

impl<R> WellFormedSignedIntegerRegister for R where
    R: SignedIntegerRegister<
            Element: WellFormedSignedIntegerElement,
            USize: Register<Element = <R::Element as Element>::USize>,
            ISize = R,
        >
{
}

impl<R> WellFormedUnsignedIntegerRegister for R where
    R: UnsignedIntegerRegister<
            Element: WellFormedUnsignedIntegerElement,
            USize = R,
            ISize: Register<Element = <R::Element as Element>::ISize>,
        >
{
}

pub trait WellFormedFloatRegister:
    FloatRegister<
        Element: WellFormedFloatElement,
        Signed: WellFormedSignedIntegerRegister<Element = <Self::Element as FloatElementWithBits>::Signed>,
        Bits: WellFormedUnsignedIntegerRegister<Element = <Self::Element as FloatElementWithBits>::Bits>,
    >
{
}

impl<F> WellFormedFloatRegister for F where
    F: FloatRegister<
            Element: WellFormedFloatElement,
            Signed: WellFormedSignedIntegerRegister<Element = <F::Element as FloatElementWithBits>::Signed>,
            Bits: WellFormedUnsignedIntegerRegister<Element = <F::Element as FloatElementWithBits>::Bits>,
        >
{
}
