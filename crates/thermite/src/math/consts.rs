pub trait FloatConsts {
    /// Euler’s number (e)
    const E: Self;

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

    /// ln(2)
    const LN_2: Self;

    /// ln(10)
    const LN_10: Self;

    /// ln(π)
    const LN_PI: Self;

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

    /// The full circle constant (τ)
    const TAU: Self;

    /// sqrt(π/2)
    const SQRT_FRAC_PI_2: Self;

    /// The golden ratio (φ)
    const PHI: Self;
}

macro_rules! impl_consts {
    (@ $ty:ty { $($name:ident = $value:expr),* $(,)? }) => {
        impl FloatConsts for $ty {
            $(const $name: Self = $value;)*
        }
    };

    ($($name:ident),*) => {
        impl<R: FloatRegister<Element: FloatConsts>> FloatConsts for Vector<R> {
            $(const $name: Self = const { Self::splat_const(<R::Element as FloatConsts>::$name) };)*
        }
    };
}

impl_consts!(
    E,
    FRAC_1_PI,
    FRAC_1_SQRT_2,
    FRAC_1_SQRT_3,
    FRAC_2_PI,
    FRAC_1_SQRT_PI,
    FRAC_2_SQRT_PI,
    FRAC_1_SQRT_TAU,
    FRAC_PI_2,
    FRAC_PI_3,
    FRAC_PI_4,
    FRAC_PI_6,
    FRAC_PI_8,
    LN_2,
    LN_10,
    LN_PI,
    LOG2_10,
    LOG2_E,
    LOG10_2,
    LOG10_E,
    PI,
    SQRT_2,
    SQRT_3,
    SQRT_E,
    TAU,
    SQRT_FRAC_PI_2,
    PHI
);

use crate::{Vector, register::FloatRegister};
use core::f32::consts as f32c;
use core::f64::consts as f64c;

impl_consts! {@
    f32 {
        E = f32c::E,
        FRAC_1_PI = f32c::FRAC_1_PI,
        FRAC_1_SQRT_2 = f32c::FRAC_1_SQRT_2,
        FRAC_1_SQRT_3 = 0.577350269189625764509148780501957456,
        FRAC_2_PI = f32c::FRAC_2_PI,
        FRAC_1_SQRT_PI = 0.5641895835477562869480794515607725858440506293289988568440857217,
        FRAC_2_SQRT_PI = f32c::FRAC_2_SQRT_PI,
        FRAC_1_SQRT_TAU = 0.398942280401432677939946059934381868,
        FRAC_PI_2 = f32c::FRAC_PI_2,
        FRAC_PI_3 = f32c::FRAC_PI_3,
        FRAC_PI_4 = f32c::FRAC_PI_4,
        FRAC_PI_6 = f32c::FRAC_PI_6,
        FRAC_PI_8 = f32c::FRAC_PI_8,
        LN_2 = f32c::LN_2,
        LN_10 = f32c::LN_10,
        LN_PI = 1.1447298858494001741434273513530587116472948129153115715136230714,
        LOG2_10 = f32c::LOG2_10,
        LOG2_E = f32c::LOG2_E,
        LOG10_2 = f32c::LOG10_2,
        LOG10_E = f32c::LOG10_E,
        PI = f32c::PI,
        SQRT_2 = f32c::SQRT_2,
        SQRT_3 = 1.732050807568877293527446341505872367,
        SQRT_E = 1.6487212707001281468486507878141635716537761007101480115750793116,
        TAU = f32c::TAU,
        SQRT_FRAC_PI_2 = 1.2533141373155002512078826424055226265034933703049691583149617881,
        PHI = 1.618033988749894848204586834365638118,
    }
}

impl_consts! {@
    f64 {
        E = f64c::E,
        FRAC_1_PI = f64c::FRAC_1_PI,
        FRAC_1_SQRT_2 = f64c::FRAC_1_SQRT_2,
        FRAC_1_SQRT_3 = 0.577350269189625764509148780501957456,
        FRAC_2_PI = f64c::FRAC_2_PI,
        FRAC_1_SQRT_PI = 0.5641895835477562869480794515607725858440506293289988568440857217,
        FRAC_2_SQRT_PI = f64c::FRAC_2_SQRT_PI,
        FRAC_1_SQRT_TAU = 0.398942280401432677939946059934381868,
        FRAC_PI_2 = f64c::FRAC_PI_2,
        FRAC_PI_3 = f64c::FRAC_PI_3,
        FRAC_PI_4 = f64c::FRAC_PI_4,
        FRAC_PI_6 = f64c::FRAC_PI_6,
        FRAC_PI_8 = f64c::FRAC_PI_8,
        LN_2 = f64c::LN_2,
        LN_10 = f64c::LN_10,
        LN_PI = 1.1447298858494001741434273513530587116472948129153115715136230714,
        LOG2_10 = f64c::LOG2_10,
        LOG2_E = f64c::LOG2_E,
        LOG10_2 = f64c::LOG10_2,
        LOG10_E = f64c::LOG10_E,
        PI = f64c::PI,
        SQRT_2 = f64c::SQRT_2,
        SQRT_3 = 1.732050807568877293527446341505872367,
        SQRT_E = 1.6487212707001281468486507878141635716537761007101480115750793116,
        TAU = f64c::TAU,
        SQRT_FRAC_PI_2 = 1.2533141373155002512078826424055226265034933703049691583149617881,
        PHI = 1.618033988749894848204586834365638118,
    }
}
