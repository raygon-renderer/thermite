use super::*;

pub trait SimdFloatVectorConsts<S: Simd>: SimdFloatVector<S> {
    /// Euler’s number (e)
    fn E() -> Self;

    /// 1/π
    fn FRAC_1_PI() -> Self;

    /// 1/sqrt(2)
    fn FRAC_1_SQRT_2() -> Self;

    /// 2/π
    fn FRAC_2_PI() -> Self;

    /// 2/sqrt(π)
    fn FRAC_2_SQRT_PI() -> Self;

    /// π/2
    fn FRAC_PI_2() -> Self;

    /// π/3
    fn FRAC_PI_3() -> Self;

    /// π/4
    fn FRAC_PI_4() -> Self;

    /// π/6
    fn FRAC_PI_6() -> Self;

    /// π/8
    fn FRAC_PI_8() -> Self;

    /// ln(2)
    fn LN_2() -> Self;

    /// ln(10)
    fn LN_10() -> Self;

    /// log2(10)
    fn LOG2_10() -> Self;

    /// log2(e)
    fn LOG2_E() -> Self;

    /// log10(2)
    fn LOG10_2() -> Self;

    /// log10(e)
    fn LOG10_E() -> Self;

    /// Archimedes’ constant (π)
    fn PI() -> Self;

    /// sqrt(2)
    fn SQRT_2() -> Self;

    /// The full circle constant (τ)
    fn TAU() -> Self;

    /// sqrt(pi/2)
    fn SQRT_FRAC_PI_2() -> Self;
}

#[doc(hidden)]
pub trait SimdFloatVectorConstsInternal<S: Simd>: SimdElement {
    type Vf: SimdFloatVector<S, Element = Self>;

    fn E() -> Self::Vf;
    fn FRAC_1_PI() -> Self::Vf;
    fn FRAC_1_SQRT_2() -> Self::Vf;
    fn FRAC_2_PI() -> Self::Vf;
    fn FRAC_2_SQRT_PI() -> Self::Vf;
    fn FRAC_PI_2() -> Self::Vf;
    fn FRAC_PI_3() -> Self::Vf;
    fn FRAC_PI_4() -> Self::Vf;
    fn FRAC_PI_6() -> Self::Vf;
    fn FRAC_PI_8() -> Self::Vf;
    fn LN_2() -> Self::Vf;
    fn LN_10() -> Self::Vf;
    fn LOG2_10() -> Self::Vf;
    fn LOG2_E() -> Self::Vf;
    fn LOG10_2() -> Self::Vf;
    fn LOG10_E() -> Self::Vf;
    fn PI() -> Self::Vf;
    fn SQRT_2() -> Self::Vf;
    fn TAU() -> Self::Vf;
    fn SQRT_FRAC_PI_2() -> Self::Vf;
}

macro_rules! impl_internal_consts {
    ($t:ident: $vf:ident => $($name:ident),*) => {
        $(
            #[inline(always)]
            fn $name() -> Self::Vf {
                Self::Vf::splat(core::$t::consts::$name)
            }
        )*
    }
}

impl<S: Simd> SimdFloatVectorConstsInternal<S> for f32 {
    type Vf = <S as Simd>::Vf32;

    fn SQRT_FRAC_PI_2() -> Self::Vf {
        Self::Vf::splat(1.2533141373155002512078826424055226265034933703049691583149617881)
    }

    impl_internal_consts!(f32: Vf32 => E, FRAC_1_PI, FRAC_1_SQRT_2, FRAC_2_PI, FRAC_2_SQRT_PI, FRAC_PI_2, FRAC_PI_3, FRAC_PI_4, FRAC_PI_6, FRAC_PI_8, LN_2, LN_10, LOG2_10, LOG2_E, LOG10_2, LOG10_E, PI, SQRT_2, TAU);
}

impl<S: Simd> SimdFloatVectorConstsInternal<S> for f64 {
    type Vf = <S as Simd>::Vf64;

    fn SQRT_FRAC_PI_2() -> Self::Vf {
        Self::Vf::splat(1.2533141373155002512078826424055226265034933703049691583149617881)
    }

    impl_internal_consts!(f64: Vf64 => E, FRAC_1_PI, FRAC_1_SQRT_2, FRAC_2_PI, FRAC_2_SQRT_PI, FRAC_PI_2, FRAC_PI_3, FRAC_PI_4, FRAC_PI_6, FRAC_PI_8, LN_2, LN_10, LOG2_10, LOG2_E, LOG10_2, LOG10_E, PI, SQRT_2, TAU);
}

macro_rules! impl_consts {
    ($($name:ident),*) => {
        $(
            #[inline(always)] fn $name() -> Self {
                <<Self as SimdVectorBase<S>>::Element as SimdFloatVectorConstsInternal<S>>::$name()
            }
        )*
    }
}

#[rustfmt::skip]
impl<T, S: Simd> SimdFloatVectorConsts<S> for T
where
    T: SimdFloatVector<S>,
    <T as SimdVectorBase<S>>::Element: SimdFloatVectorConstsInternal<S, Vf = T>,
{
    impl_consts!(E, FRAC_1_PI, FRAC_1_SQRT_2, FRAC_2_PI, FRAC_2_SQRT_PI, FRAC_PI_2, FRAC_PI_3, FRAC_PI_4, FRAC_PI_6, FRAC_PI_8, LN_2, LN_10, LOG2_10, LOG2_E, LOG10_2, LOG10_E, PI, SQRT_2, TAU, SQRT_FRAC_PI_2);
}
