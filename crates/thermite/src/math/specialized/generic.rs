use super::*;

#[inline(always)]
pub fn inverse_sqrt_internal<V, E: FloatElement, P>(x: V) -> V
where
    V: FloatVectorWithBits<Element = E> + SpecializedTranscendentalMath<E>,
    P: Policy,
{
    let mut y = x.rsqrt();

    if const { V::HAS_APPROX_RSQRT && P::POLICY.precision.gt(PrecisionPolicy::Worst) } {
        let nx2 = x.scale(const { E::ConstRatio::<{-1}, {2}>::VALUE }); // -0.5*x
        let threehalfs = V::splat(const { E::ConstRatio::<{3}, {2}>::VALUE }); // 1.5

        // one iteration of Newton's method
        y = y * y.square().mul_adde(nx2, threehalfs);
    }

    y
}

#[inline(always)]
pub fn sinc_internal<V, E: FloatElement, P>(x: V) -> V
where
    V: FloatVectorWithBits<Element = E> + SpecializedTranscendentalMath<E>,
    P: Policy,
{
    if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
        let mut y = V::sin::<P>(x).approx_div_p::<P>(x);

        if const { P::POLICY.check_overflow } {
            y = x.is_zero().select(V::ONE, y);
            y = x.is_infinite().select(V::ZERO, y);
        }

        return y;
    }

    let is_tiny = x.abs().cmp_le(V::FOURTH_ROOT_EPSILON);

    let x2 = x.square();

    // if branching, use Taylor series for tiny x without calling sine.
    if const { !P::POLICY.avoid_branching } && crate::unlikely(is_tiny.all()) {
        if const { P::POLICY.precision.le(PrecisionPolicy::Average) } {
            // use fma instead of division then subtraction, for improved performance
            // at the cost of a tiny bit of precision with the 120 denominator
            return x2.mul_adde(
                x2.mul_sube(V::splat(FloatElement::from_ratio(1, 120)), V::FRAC_1_6),
                V::ONE,
            );
        }

        let res = x2 / V::splat(const { E::ConstInt::<{120}>::VALUE });
        return x2.mul_add(res - V::FRAC_1_6, V::ONE);
    }

    // For very small x, sinc(x) ~ 1 - x^2/6 + x^4/120
    let num = is_tiny.select(x2, V::sin::<P>(x));
    let den = is_tiny.select(V::splat(const { E::ConstInt::<{120}>::VALUE }), x);

    // combined division, since division is expensive
    let mut y = num.approx_div_p::<P>(den);

    y = is_tiny.select(x2.mul_adde(y - V::FRAC_1_6, V::ONE), y);

    if const { P::POLICY.check_overflow } {
        y = x.is_infinite().select(V::ZERO, y);
    }

    y
}

pub trait SincPiConsts: FloatConsts {
    const FRAC_PI_SQR_OVER_6: Self;
    const FRAC_120_OVER_PI_SQR: Self;
    const FRAC_PI_4_OVER_120: Self;
}

impl SincPiConsts for f32 {
    const FRAC_PI_SQR_OVER_6: Self = 1.6449340668482264364724151666460251892189499012068; // pi^2/6
    const FRAC_120_OVER_PI_SQR: Self = 1.2319178705621202226983339920542432970193362224366; // 120/pi^2
    const FRAC_PI_4_OVER_120: Self = 0.81174242528335364363700277240587592708106321393905; // pi^4/120
}

impl SincPiConsts for f64 {
    const FRAC_PI_SQR_OVER_6: Self = 1.6449340668482264364724151666460251892189499012068; // pi^2/6
    const FRAC_120_OVER_PI_SQR: Self = 1.2319178705621202226983339920542432970193362224366; // 120/pi^2
    const FRAC_PI_4_OVER_120: Self = 0.81174242528335364363700277240587592708106321393905; // pi^4/120
}

#[inline(always)]
pub fn sinc_pi_internal<V, E: FloatElement + SincPiConsts, P>(x: V) -> V
where
    V: FloatVectorWithBits<Element = E> + SpecializedTranscendentalMath<E>,
    P: Policy,
{
    if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
        // forwards to the above medium-precision sinc implementation,
        // which uses sin(x) * rcp(x)
        return V::sinc_p::<P>(x.scale(FloatConsts::PI));
    }

    let pi_2_frac_6: V = V::splat(E::FRAC_PI_SQR_OVER_6);
    let frac_120_pi_4: V = V::splat(E::FRAC_120_OVER_PI_SQR);

    let is_tiny = x.abs().cmp_le(V::FOURTH_ROOT_EPSILON);

    let x2 = x.square();

    // if branching, use Taylor series for tiny x without calling sine.
    if const { !P::POLICY.avoid_branching } && crate::unlikely(is_tiny.all()) {
        let pi_4_frac_120: V = V::splat(E::FRAC_PI_4_OVER_120);

        // unlike sinc, which has x^2/120 with 120 being an exact integer,
        // sinc_pi has pi^4/120, and since pi is irrational and imprecise anyway, we
        // can avoid the exact division by 120 in favor of multiplying by pi^4/120
        return x2.mul_add(x2.mul_sube(pi_4_frac_120, pi_2_frac_6), V::ONE);
    }

    // for very small x, sinc_pi(x) ~ 1 - (pi^2/6)*x^2 + (pi^4/120)*x^4
    let num = is_tiny.select(x2, V::sin_pi::<P>(x));
    let den = is_tiny.select(frac_120_pi_4, x.scale(FloatConsts::PI)); // NOTE: first term is flipped for division

    // combined division, since division is expensive
    let mut y = num / den;

    y = is_tiny.select(x2.mul_adde(y - pi_2_frac_6, V::ONE), y);

    if const { P::POLICY.check_overflow } {
        y = x.is_infinite().select(V::ZERO, y);
    }

    y
}

pub trait LogNHelper: FloatConsts + Sized {
    const LOG2_TABLE: [Self; 30];

    fn fallback<P: Policy, const N: usize>() -> Self;
}

macro_rules! impl_log2_table {
    ($($value:expr),* $(,)?) => {
        impl LogNHelper for f32 {
            const LOG2_TABLE: [Self; 30] = [$($value),*];

            #[inline(always)]
            fn fallback<P: Policy, const N: usize>() -> Self {
                cfg_if::cfg_if! {
                    if #[cfg(all(feature = "spirv", target_arch = "spirv"))] {
                        Vector::<f32>(N as f32).log2_p::<P>().0
                    } else {
                        libm::log2f(N as f32)
                    }
                }
            }
        }

        impl LogNHelper for f64 {
            const LOG2_TABLE: [Self; 30] = [$($value),*];

            #[inline(always)]
            fn fallback<P: Policy, const N: usize>() -> Self {
                cfg_if::cfg_if! {
                    if #[cfg(all(feature = "spirv", target_arch = "spirv"))] {
                        Vector::<f64>(N as f64).log2_p::<P>().0
                    } else {
                        libm::log2(N as f64)
                    }
                }
            }
        }
    };
}

impl_log2_table![
    // precomputed N[Table[1/log_2(x), {x, 3, 32}], 20]
    0.63092975357145743710,
    0.50000000000000000000,
    0.43067655807339305067,
    0.38685280723454158687,
    0.35620718710802217651,
    0.33333333333333333333,
    0.31546487678572871855,
    0.30102999566398119521,
    0.28906482631788785927,
    0.27894294565112984319,
    0.27023815442731974129,
    0.26264953503719354798,
    0.25595802480981548939,
    0.25000000000000000000,
    0.24465054211822603039,
    0.23981246656813144474,
    0.23540891336663823645,
    0.23137821315975917426,
    0.22767024869695299798,
    0.22424382421757543948,
    0.22106472945750374615,
    0.21810429198553155923,
    0.21533827903669652534,
    0.21274605355336315361,
    0.21030991785715247903,
    0.20801459767650945760,
    0.20584683246043445731,
    0.20379504709050619003,
    0.20184908658209985072,
    0.20000000000000000000,
];

#[inline(always)]
pub fn log_n_internal<V, E: FloatElement + LogNHelper, P, const N: usize>(x: V) -> V
where
    V: FloatVectorWithBits<Element = E> + SpecializedTranscendentalMath<E>,
    P: Policy,
{
    if const { N > 32 } {
        return V::log2::<P>(x).approx_div_p::<P>(V::splat(E::fallback::<P, N>()));
    }

    match N {
        // 0 and 1 are special cases, and these are what Wolfram Alpha returns
        0 => V::ZERO,               // log(x)/log(0) = log(x)/-infinity = 0
        1 => FloatVector::INFINITY, // log(x)/log(1) = log(x)/0 = complex infinity, only return real part
        2 => V::log2::<P>(x),
        10 => V::log10::<P>(x),
        _ => V::log2::<P>(x).scale(const { if N <= 32 { E::LOG2_TABLE[N - 3] } else { E::ZERO } }),
    }
}
