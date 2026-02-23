use crate::element::FloatElementWithBits;

use super::*;

/// A FloatElement type that can itself be used as a FloatRegister,
/// with its associated SignedBits and Bits types also being fully formed.
pub trait WellFormedFloatElement:
    FloatElementWithBits<
        SignedBits: WellFormedSignedIntegerElement<
            Element = <Self as FloatElementWithBits>::SignedBits,
            Storage = <Self as FloatElementWithBits>::SignedBits,
        >,
        Bits: WellFormedUnsignedIntegerElement<
            Element = <Self as FloatElementWithBits>::Bits,
            Storage = <Self as FloatElementWithBits>::Bits,
        >,
        Unsigned: WellFormedUnsignedIntegerElement<
            Element = <Self as Element>::Unsigned,
            Storage = <Self as Element>::Unsigned,
        >,
        Signed: WellFormedSignedIntegerElement<
            Element = <Self as Element>::Signed,
            Storage = <Self as Element>::Signed,
        >,
    > + FloatRegister<Element = Self, Storage = Self>
{
}

/// A SignedBits IntegerElement type that can itself be used as a SignedIntegerRegister.
pub trait WellFormedSignedIntegerElement:
    IntegerElement<Unsigned: CoreRegister>
    + num_traits::Signed
    + SignedIntegerRegister<Element = Self, Storage = Self, Signed = Self, Unsigned = <Self as Element>::Unsigned>
{
}

/// An Unsigned IntegerElement type that can itself be used as an UnsignedIntegerRegister.
pub trait WellFormedUnsignedIntegerElement:
    IntegerElement<Signed: CoreRegister>
    + IntegerRegister<Element = Self, Storage = Self, Unsigned = Self, Signed = <Self as Element>::Signed>
    + UnsignedIntegerRegister<Element = Self, Storage = Self>
{
}

impl<F> WellFormedFloatElement for F where
    F: FloatElementWithBits<
            SignedBits: WellFormedSignedIntegerElement<
                Element = <F as FloatElementWithBits>::SignedBits,
                Storage = <F as FloatElementWithBits>::SignedBits,
            >,
            Bits: WellFormedUnsignedIntegerElement<
                Element = <F as FloatElementWithBits>::Bits,
                Storage = <F as FloatElementWithBits>::Bits,
            >,
            Unsigned: WellFormedUnsignedIntegerElement<
                Element = <F as Element>::Unsigned,
                Storage = <F as Element>::Unsigned,
            >,
            Signed: WellFormedSignedIntegerElement<Element = <F as Element>::Signed, Storage = <F as Element>::Signed>,
        > + FloatRegister<Element = F, Storage = F>
{
}

impl<I> WellFormedSignedIntegerElement for I where
    I: IntegerElement<Unsigned: CoreRegister>
        + num_traits::Signed
        + SignedIntegerRegister<Element = I, Storage = I, Signed = I, Unsigned = <I as Element>::Unsigned>
{
}

impl<U> WellFormedUnsignedIntegerElement for U where
    U: IntegerElement<Signed: CoreRegister>
        + IntegerRegister<Element = U, Storage = U, Unsigned = U, Signed = <U as Element>::Signed>
        + UnsignedIntegerRegister<Element = U, Storage = U>
{
}

pub trait WellFormedRegister:
    Register<
        Element: Element<Unsigned: WellFormedUnsignedIntegerElement, Signed: WellFormedSignedIntegerElement>,
        Unsigned: WellFormedUnsignedIntegerRegister<Element = <Self::Element as Element>::Unsigned>,
        Signed: WellFormedSignedIntegerRegister<Element = <Self::Element as Element>::Signed>,
    >
{
}

impl<R> WellFormedRegister for R where
    R: Register<
            Element: Element<Unsigned: WellFormedUnsignedIntegerElement, Signed: WellFormedSignedIntegerElement>,
            Unsigned: WellFormedUnsignedIntegerRegister<Element = <R::Element as Element>::Unsigned>,
            Signed: WellFormedSignedIntegerRegister<Element = <R::Element as Element>::Signed>,
        >
{
}

pub trait WellFormedSignedIntegerRegister:
    SignedIntegerRegister<
        Element: WellFormedSignedIntegerElement,
        Unsigned: Register<Element = <Self::Element as Element>::Unsigned>,
        Signed = Self,
    >
{
}

pub trait WellFormedUnsignedIntegerRegister:
    UnsignedIntegerRegister<
        Element: WellFormedUnsignedIntegerElement,
        Unsigned = Self,
        Signed: Register<Element = <Self::Element as Element>::Signed>,
    >
{
}

impl<R> WellFormedSignedIntegerRegister for R where
    R: SignedIntegerRegister<
            Element: WellFormedSignedIntegerElement,
            Unsigned: Register<Element = <R::Element as Element>::Unsigned>,
            Signed = R,
        >
{
}

impl<R> WellFormedUnsignedIntegerRegister for R where
    R: UnsignedIntegerRegister<
            Element: WellFormedUnsignedIntegerElement,
            Unsigned = R,
            Signed: Register<Element = <R::Element as Element>::Signed>,
        >
{
}

pub trait WellFormedFloatRegister:
    FloatRegister<
        Element: WellFormedFloatElement,
        SignedBits: WellFormedSignedIntegerRegister<Element = <Self::Element as FloatElementWithBits>::SignedBits>,
        Bits: WellFormedUnsignedIntegerRegister<Element = <Self::Element as FloatElementWithBits>::Bits>,
    >
{
}

impl<F> WellFormedFloatRegister for F where
    F: FloatRegister<
            Element: WellFormedFloatElement,
            SignedBits: WellFormedSignedIntegerRegister<Element = <F::Element as FloatElementWithBits>::SignedBits>,
            Bits: WellFormedUnsignedIntegerRegister<Element = <F::Element as FloatElementWithBits>::Bits>,
        >
{
}
