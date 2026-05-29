#![allow(clippy::excessive_precision)]

/// Extensive set of constant special values used in float operations.
pub trait FloatConsts {
    /// Negative zero (-0) (only sign bit set)
    const NEG_ZERO: Self;

    /// Euler’s number (e)
    const E: Self;

    /// Euler-Mascheroni constant (γ)
    const EULER_GAMMA: Self;

    /// π^2
    const PI_SQUARED: Self;

    /// π^3
    const PI_CUBED: Self;

    /// π^4
    const PI_FOURTH: Self;

    /// 1/π
    const FRAC_1_PI: Self;

    /// 1/sqrt(2)
    const FRAC_1_SQRT_2: Self;

    /// 1/sqrt(3)
    const FRAC_1_SQRT_3: Self;

    /// 2/π
    const FRAC_2_PI: Self;

    /// 1/sqrt(π)
    const FRAC_1_SQRT_PI: Self;

    /// 2/sqrt(π)
    const FRAC_2_SQRT_PI: Self;

    /// sqrt(pi)/2
    const FRAC_SQRT_PI_2: Self;

    /// 1/sqrt(2π)
    const FRAC_1_SQRT_TAU: Self;

    /// π/2
    const FRAC_PI_2: Self;

    /// π/3
    const FRAC_PI_3: Self;

    /// π/4
    const FRAC_PI_4: Self;

    /// π/6
    const FRAC_PI_6: Self;

    /// π/8
    const FRAC_PI_8: Self;

    /// π/180
    const FRAC_PI_180: Self;

    /// 180/π
    const FRAC_180_PI: Self;

    /// ln(2)
    const LN_2: Self;

    /// ln(10)
    const LN_10: Self;

    /// ln(π)
    const LN_PI: Self;

    /// ln(pi)/2
    const FRAC_LN_PI_2: Self;

    /// log2(10)
    const LOG2_10: Self;

    /// log2(e)
    const LOG2_E: Self;

    /// log10(2)
    const LOG10_2: Self;

    /// log10(e)
    const LOG10_E: Self;

    /// Archimedes’ constant (π)
    const PI: Self;

    /// sqrt(2)
    const SQRT_2: Self;

    /// sqrt(3)
    const SQRT_3: Self;

    /// sqrt(e)
    const SQRT_E: Self;

    /// The machine epsilon
    const EPSILON: Self;

    /// The square root of the machine epsilon (sqrt(epsilon))
    const SQRT_EPSILON: Self;

    /// The fourth root of the machine epsilon (fourth_root(epsilon))
    const FOURTH_ROOT_EPSILON: Self;

    /// The full circle constant (τ)
    const TAU: Self;

    /// sqrt(π/2)
    const SQRT_FRAC_PI_2: Self;

    /// sqrt(2π)
    const SQRT_TAU: Self;

    /// The golden ratio (φ)
    const PHI: Self;

    /// 1/3
    const FRAC_1_3: Self;

    /// 2/3
    const FRAC_2_3: Self;

    /// 1/4
    const FRAC_1_4: Self;

    /// 1/6
    const FRAC_1_6: Self;

    /// -1/e
    const FRAC_NEG_1_E: Self;
}

use crate::vector::{SplatConst, SplatVector, VectorValue};

macro_rules! impl_consts {
    (@ $ty:ty { $($name:ident = $value:expr),* $(,)? }) => {
        impl FloatConsts for $ty {
            $(const $name: Self = $value;)*
        }
    };

    ($($name:ident),*) => {
        impl<R: FloatRegister<Element: FloatConsts>> FloatConsts for Vector<R> {
            $(const $name: Self = const {
                struct FC<R: FloatRegister>(core::marker::PhantomData<R>);
                impl<R: FloatRegister> SplatConst<R::Element> for FC<R> { const VALUE: R::Element = <R::Element as FloatConsts>::$name; }
                <<Vector<R> as SplatVector<R::Element>>::Splat<FC<R>> as VectorValue<FC<R>, Vector<R>>>::VALUE
            };)*
        }
    };
}

impl_consts!(
    NEG_ZERO,
    E,
    EULER_GAMMA,
    PI_SQUARED,
    PI_CUBED,
    PI_FOURTH,
    FRAC_1_PI,
    FRAC_1_SQRT_2,
    FRAC_1_SQRT_3,
    FRAC_2_PI,
    FRAC_1_SQRT_PI,
    FRAC_2_SQRT_PI,
    FRAC_SQRT_PI_2,
    FRAC_1_SQRT_TAU,
    FRAC_PI_2,
    FRAC_PI_3,
    FRAC_PI_4,
    FRAC_PI_6,
    FRAC_PI_8,
    FRAC_PI_180,
    FRAC_180_PI,
    LN_2,
    LN_10,
    LN_PI,
    FRAC_LN_PI_2,
    LOG2_10,
    LOG2_E,
    LOG10_2,
    LOG10_E,
    PI,
    SQRT_2,
    SQRT_3,
    SQRT_E,
    EPSILON,
    SQRT_EPSILON,
    FOURTH_ROOT_EPSILON,
    TAU,
    SQRT_FRAC_PI_2,
    SQRT_TAU,
    PHI,
    FRAC_1_3,
    FRAC_2_3,
    FRAC_1_4,
    FRAC_1_6,
    FRAC_NEG_1_E
);

use crate::{Vector, register::FloatRegister};
use core::f32::consts as f32c;
use core::f64::consts as f64c;

impl_consts! {@
    f32 {
        NEG_ZERO = -0.0f32,
        E = f32c::E,
        EULER_GAMMA = 5.772156649015328606065120900824024310e-01,
        PI_SQUARED = 9.8696044010893586188344909998761511353136994072408,
        PI_CUBED = 31.006276680299820175476315067101395202225288565885,
        PI_FOURTH = 97.409091034002437236440332688705111249727585672685,
        FRAC_1_PI = f32c::FRAC_1_PI,
        FRAC_1_SQRT_2 = f32c::FRAC_1_SQRT_2,
        FRAC_1_SQRT_3 = 0.577350269189625764509148780501957456,
        FRAC_2_PI = f32c::FRAC_2_PI,
        FRAC_1_SQRT_PI = 0.5641895835477562869480794515607725858440506293289988568440857217,
        FRAC_2_SQRT_PI = f32c::FRAC_2_SQRT_PI,
        FRAC_SQRT_PI_2 = 0.88622692545275801364908374167057259139877472806119,
        FRAC_1_SQRT_TAU = 0.398942280401432677939946059934381868,
        FRAC_PI_2 = f32c::FRAC_PI_2,
        FRAC_PI_3 = f32c::FRAC_PI_3,
        FRAC_PI_4 = f32c::FRAC_PI_4,
        FRAC_PI_6 = f32c::FRAC_PI_6,
        FRAC_PI_8 = f32c::FRAC_PI_8,
        FRAC_PI_180 = f32c::PI / 180.0,
        FRAC_180_PI = 180.0 / f32c::PI,
        LN_2 = f32c::LN_2,
        LN_10 = f32c::LN_10,
        LN_PI = 1.1447298858494001741434273513530587116472948129153115715136230714,
        FRAC_LN_PI_2 = 0.5723649429247000870717136756765293558236474064576557857568115357360688849424130,
        LOG2_10 = f32c::LOG2_10,
        LOG2_E = f32c::LOG2_E,
        LOG10_2 = f32c::LOG10_2,
        LOG10_E = f32c::LOG10_E,
        PI = f32c::PI,
        SQRT_2 = f32c::SQRT_2,
        SQRT_3 = 1.732050807568877293527446341505872367,
        SQRT_E = 1.6487212707001281468486507878141635716537761007101480115750793116,
        EPSILON = f32::EPSILON,
        SQRT_EPSILON = 0.0003452669836517821464776144458809047877858776827733458406716232,
        FOURTH_ROOT_EPSILON = 0.0185813611894226453755641436815143067322274318448624272659973571195104883,
        TAU = f32c::TAU,
        SQRT_FRAC_PI_2 = 1.2533141373155002512078826424055226265034933703049691583149617881,
        SQRT_TAU = 2.506628274631000502415765284811045253006986740609938316629923576342293654607842,
        PHI = 1.618033988749894848204586834365638118,
        FRAC_1_3 = 1.0 / 3.0,
        FRAC_2_3 = 2.0 / 3.0,
        FRAC_1_4 = 0.25,
        FRAC_1_6 = 1.0 / 6.0,
        FRAC_NEG_1_E = -0.367879441171442321595523770161460867445811131031767834507836801,
    }
}

impl_consts! {@
    f64 {
        NEG_ZERO = -0.0f64,
        E = f64c::E,
        EULER_GAMMA = 5.772156649015328606065120900824024310e-01,
        PI_SQUARED = 9.8696044010893586188344909998761511353136994072408,
        PI_CUBED = 31.006276680299820175476315067101395202225288565885,
        PI_FOURTH = 97.409091034002437236440332688705111249727585672685,
        FRAC_1_PI = f64c::FRAC_1_PI,
        FRAC_1_SQRT_2 = f64c::FRAC_1_SQRT_2,
        FRAC_1_SQRT_3 = 0.577350269189625764509148780501957456,
        FRAC_2_PI = f64c::FRAC_2_PI,
        FRAC_1_SQRT_PI = 0.5641895835477562869480794515607725858440506293289988568440857217,
        FRAC_2_SQRT_PI = f64c::FRAC_2_SQRT_PI,
        FRAC_SQRT_PI_2 = 0.88622692545275801364908374167057259139877472806119,
        FRAC_1_SQRT_TAU = 0.398942280401432677939946059934381868,
        FRAC_PI_2 = f64c::FRAC_PI_2,
        FRAC_PI_3 = f64c::FRAC_PI_3,
        FRAC_PI_4 = f64c::FRAC_PI_4,
        FRAC_PI_6 = f64c::FRAC_PI_6,
        FRAC_PI_8 = f64c::FRAC_PI_8,
        FRAC_PI_180 = f64c::PI / 180.0,
        FRAC_180_PI = 180.0 / f64c::PI,
        LN_2 = f64c::LN_2,
        LN_10 = f64c::LN_10,
        LN_PI = 1.1447298858494001741434273513530587116472948129153115715136230714,
        FRAC_LN_PI_2 = 0.5723649429247000870717136756765293558236474064576557857568115357360688849424130,
        LOG2_10 = f64c::LOG2_10,
        LOG2_E = f64c::LOG2_E,
        LOG10_2 = f64c::LOG10_2,
        LOG10_E = f64c::LOG10_E,
        PI = f64c::PI,
        SQRT_2 = f64c::SQRT_2,
        SQRT_3 = 1.732050807568877293527446341505872367,
        SQRT_E = 1.6487212707001281468486507878141635716537761007101480115750793116,
        EPSILON = f64::EPSILON,
        SQRT_EPSILON = 1.4901161193847656314265919999999999861416556075118966152884e-8,
        FOURTH_ROOT_EPSILON = 0.000122070312500000000263233208319999999148543320525530928655351028080390676,
        TAU = f64c::TAU,
        SQRT_FRAC_PI_2 = 1.2533141373155002512078826424055226265034933703049691583149617881,
        SQRT_TAU = 2.506628274631000502415765284811045253006986740609938316629923576342293654607842,
        PHI = 1.618033988749894848204586834365638118,
        FRAC_1_3 = 1.0 / 3.0,
        FRAC_2_3 = 2.0 / 3.0,
        FRAC_1_4 = 0.25,
        FRAC_1_6 = 1.0 / 6.0,
        FRAC_NEG_1_E = -0.367879441171442321595523770161460867445811131031767834507836801,
    }
}
