#![allow(clippy::approx_constant)]

use thermite::{Vector, math::FloatConsts};

pub struct CompensatedConst<T>(pub T, pub T);

use super::{Compensated, CompensatedRegister};

macro_rules! impl_consts {
    ($($const:ident),*) => {
        pub trait SplitFloatConsts<T>: FloatConsts {
            $(const $const: CompensatedConst<T>; )*
        }

        impl<R: CompensatedRegister> FloatConsts for Compensated<R> {
            $(const $const: Self = {
                let CompensatedConst(hi, lo) = <R::Element as SplitFloatConsts<R::Element>>::$const;
                Compensated::from_parts(Vector::splat_const(hi), Vector::splat_const(lo))
            };)*
        }
    };

    ($t:ty { $($const:ident = ($high:literal, $low:literal)),* $(,)? }) => {paste::paste! {
        impl FloatConsts for CompensatedConst<$t> {
            $(const $const: Self = CompensatedConst(hexf::[<hex $t>]!($high), hexf::[<hex $t>]!($low));)*
        }

        impl SplitFloatConsts<$t> for $t {
            $(const $const: CompensatedConst<$t> = <CompensatedConst<$t> as FloatConsts>::$const;)*
        }
    }};
}

#[rustfmt::skip]
impl_consts!(
    ZERO,ONE,E,EGAMMA,FRAC_1_PI,FRAC_1_SQRT_2,FRAC_1_SQRT_3,FRAC_2_PI,FRAC_1_SQRT_PI,
    FRAC_2_SQRT_PI,FRAC_1_SQRT_TAU,FRAC_PI_2,FRAC_PI_3,FRAC_PI_4,FRAC_PI_6,FRAC_PI_8,
    FRAC_PI_180,FRAC_180_PI,LN_2,LN_10,LN_PI,FRAC_LN_PI_2,LOG2_10,LOG2_E,LOG10_2,LOG10_E,
    PI,SQRT_2,SQRT_3,SQRT_E,TAU,SQRT_FRAC_PI_2,SQRT_2_PI,PHI,EPSILON,SQRT_EPSILON,FOURTH_ROOT_EPSILON);

impl_consts!(f32 {
    ZERO = ("0x0.0p+0", "0x0.0p+0"),
    ONE = ("0x1.0000000000000p+0", "0x0.0p+0"),
    E = ("0x1.5bf0a80000000p+1", "0x1.628aee0000000p-24"),
    EGAMMA = ("0x1.2788d00000000p-1", "-0x1.c824f40000000p-28"),
    FRAC_1_PI = ("0x1.45f3060000000p-2", "0x1.b939100000000p-27"),
    FRAC_1_SQRT_2 = ("0x1.6a09e60000000p-1", "0x1.9fcef40000000p-27"),
    FRAC_1_SQRT_3 = ("0x1.279a740000000p-1", "0x1.640cc80000000p-27"),
    FRAC_2_PI = ("0x1.45f3060000000p-1", "0x1.b939100000000p-26"),
    FRAC_1_SQRT_PI = ("0x1.20dd760000000p-1", "-0x1.f7ac920000000p-26"),
    FRAC_2_SQRT_PI = ("0x1.20dd760000000p+0", "-0x1.f7ac920000000p-25"),
    FRAC_1_SQRT_TAU = ("0x1.9884540000000p-2", "-0x1.8579360000000p-27"),
    FRAC_PI_2 = ("0x1.921fb60000000p+0", "-0x1.777a5c0000000p-25"),
    FRAC_PI_3 = ("0x1.0c15240000000p+0", "-0x1.f4a3260000000p-26"),
    FRAC_PI_4 = ("0x1.921fb60000000p-1", "-0x1.777a5c0000000p-26"),
    FRAC_PI_6 = ("0x1.0c15240000000p-1", "-0x1.f4a3260000000p-27"),
    FRAC_PI_8 = ("0x1.921fb60000000p-2", "-0x1.777a5c0000000p-27"),
    FRAC_PI_180 = ("0x1.1df46a0000000p-6", "0x1.294e9c0000000p-33"),
    FRAC_180_PI = ("0x1.ca5dc20000000p+5", "-0x1.670f820000000p-21"),
    LN_2 = ("0x1.62e4300000000p-1", "-0x1.05c6100000000p-29"),
    LN_10 = ("0x1.26bb1c0000000p+1", "-0x1.12aaba0000000p-25"),
    LN_PI = ("0x1.250d040000000p+0", "0x1.1cf4380000000p-25"),
    FRAC_LN_PI_2 = ("0x1.250d040000000p-1", "0x1.1cf4380000000p-26"),
    LOG2_10 = ("0x1.a934f00000000p+1", "0x1.2f346e0000000p-24"),
    LOG2_E = ("0x1.7154760000000p+0", "0x1.4ae0c00000000p-26"),
    LOG10_2 = ("0x1.3441360000000p-2", "-0x1.ec10c00000000p-27"),
    LOG10_E = ("0x1.bcb7b20000000p-2", "-0x1.5b235e0000000p-27"),
    PI = ("0x1.921fb60000000p+1", "-0x1.777a5c0000000p-24"),
    SQRT_2 = ("0x1.6a09e60000000p+0", "0x1.9fcef40000000p-26"),
    SQRT_3 = ("0x1.bb67ae0000000p+0", "0x1.0b09960000000p-25"),
    SQRT_E = ("0x1.a612980000000p+0", "0x1.c3c0d40000000p-25"),
    TAU = ("0x1.921fb60000000p+2", "-0x1.777a5c0000000p-23"),
    SQRT_FRAC_PI_2 = ("0x1.9884540000000p-1", "-0x1.8579360000000p-26"),
    SQRT_2_PI = ("0x1.40d9320000000p+1", "-0x1.3b1f4e0000000p-32"),
    PHI = ("0x1.9e377a0000000p+0", "-0x1.1a02d60000000p-26"),
    EPSILON = ("0x1.0000000000000p-47", "0x1.6ece7c0000000p-103"),
    SQRT_EPSILON = ("0x1.6a09e60000000p-24", "0x1.9fcef40000000p-50"),
    FOURTH_ROOT_EPSILON = ("0x1.306fe00000000p-12", "0x1.4636e20000000p-37"),
});

impl_consts!(f64 {
    ZERO = ("0x0.0p+0", "0x0.0p+0"),
    ONE = ("0x1.0000000000000p+0", "0x0.0p+0"),
    E = ("0x1.5bf0a8b145769p+1", "0x1.4d57ee2b1013ap-53"),
    EGAMMA = ("0x1.2788cfc6fb619p-1", "-0x1.6cb90701fbfabp-58"),
    FRAC_1_PI = ("0x1.45f306dc9c883p-2", "-0x1.6b01ec5417056p-56"),
    FRAC_1_SQRT_2 = ("0x1.6a09e667f3bcdp-1", "-0x1.bdd3413b26456p-55"),
    FRAC_1_SQRT_3 = ("0x1.279a74590331cp-1", "0x1.34863e0792bedp-55"),
    FRAC_2_PI = ("0x1.45f306dc9c883p-1", "-0x1.6b01ec5417056p-55"),
    FRAC_1_SQRT_PI = ("0x1.20dd750429b6dp-1", "0x1.1ae3a914fed80p-57"),
    FRAC_2_SQRT_PI = ("0x1.20dd750429b6dp+0", "0x1.1ae3a914fed80p-56"),
    FRAC_1_SQRT_TAU = ("0x1.9884533d43651p-2", "-0x1.cbc0d30ebfd15p-56"),
    FRAC_PI_2 = ("0x1.921fb54442d18p+0", "0x1.1a62633145c07p-54"),
    FRAC_PI_3 = ("0x1.0c152382d7366p+0", "-0x1.ee6913347c2a6p-54"),
    FRAC_PI_4 = ("0x1.921fb54442d18p-1", "0x1.1a62633145c07p-55"),
    FRAC_PI_6 = ("0x1.0c152382d7366p-1", "-0x1.ee6913347c2a6p-55"),
    FRAC_PI_8 = ("0x1.921fb54442d18p-2", "0x1.1a62633145c07p-56"),
    FRAC_PI_180 = ("0x1.1df46a2529d39p-6", "0x1.5c1d8becdd291p-62"),
    FRAC_180_PI = ("0x1.ca5dc1a63c1f8p+5", "-0x1.1e7ab456405f9p-49"),
    LN_2 = ("0x1.62e42fefa39efp-1", "0x1.abc9e3b39803fp-56"),
    LN_10 = ("0x1.26bb1bbb55516p+1", "-0x1.f48ad494ea3e9p-53"),
    LN_PI = ("0x1.250d048e7a1bdp+0", "0x1.7abf2ad8d5088p-57"),
    FRAC_LN_PI_2 = ("0x1.250d048e7a1bdp-1", "0x1.7abf2ad8d5088p-58"),
    LOG2_10 = ("0x1.a934f0979a371p+1", "0x1.7f2495fb7fa6dp-53"),
    LOG2_E = ("0x1.71547652b82fep+0", "0x1.777d0ffda0d24p-56"),
    LOG10_2 = ("0x1.34413509f79ffp-2", "-0x1.9dc1da994fd21p-59"),
    LOG10_E = ("0x1.bcb7b1526e50ep-2", "0x1.95355baaafad3p-57"),
    PI = ("0x1.921fb54442d18p+1", "0x1.1a62633145c07p-53"),
    SQRT_2 = ("0x1.6a09e667f3bcdp+0", "-0x1.bdd3413b26456p-54"),
    SQRT_3 = ("0x1.bb67ae8584caap+0", "0x1.cec95d0b5c1e3p-54"),
    SQRT_E = ("0x1.a61298e1e069cp+0", "-0x1.b4690082a4906p-55"),
    TAU = ("0x1.921fb54442d18p+2", "0x1.1a62633145c07p-52"),
    SQRT_FRAC_PI_2 = ("0x1.9884533d43651p-1", "-0x1.cbc0d30ebfd15p-55"),
    SQRT_2_PI = ("0x1.40d931ff62706p+1", "-0x1.a6a0d6f814637p-53"),
    PHI = ("0x1.9e3779b97f4a8p+0", "-0x1.f506319fcfd19p-55"),
    EPSILON = ("0x1.0000000000000p-105", "0x1.2b837a498a57cp-228"),
    SQRT_EPSILON = ("0x1.0000000000000p-105", "0x1.2b837a498a57cp-228"),
    FOURTH_ROOT_EPSILON = ("0x1.0000000000000p-105", "0x1.2b837a498a57cp-228"),
});
