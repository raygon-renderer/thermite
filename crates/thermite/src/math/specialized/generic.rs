use super::*;

#[inline(always)]
pub fn inverse_sqrt_internal<V, E: FloatElement, P>(x: V) -> V
where
    V: FloatVectorWithBits<Element = E> + SpecializedTranscendentalMath<E>,
    P: Policy,
{
    let mut y = x.rsqrt();

    if const { V::HAS_APPROX_RSQRT && P::POLICY.precision.gt(PrecisionPolicy::Worst) } {
        let nx2 = x * V::splat(E::from_f64(-0.5));
        let threehalfs = V::splat(E::from_f64(1.5));

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
    if !P::POLICY.avoid_branching && crate::unlikely(is_tiny.all()) {
        if const { P::POLICY.precision.le(PrecisionPolicy::Average) } {
            // use fma instead of division then subtraction, for improved performance
            // at the cost of a tiny bit of precision with the 120 denominator
            return x2.mul_adde(
                x2.mul_sube(V::splat(FloatElement::from_ratio(1, 120)), V::FRAC_1_6),
                V::ONE,
            );
        }

        let res = x2 / V::splat(FloatElement::from_i64(120));
        return x2.mul_add(res - V::FRAC_1_6, V::ONE);
    }

    // For very small x, sinc(x) ~ 1 - x^2/6 + x^4/120
    let num = is_tiny.select(x2, V::sin::<P>(x));
    let den = is_tiny.select(V::splat(FloatElement::from_i64(120)), x);

    // combined division, since division is expensive
    let mut y = num.approx_div_p::<P>(den);

    y = is_tiny.select(x2.mul_adde(y - V::FRAC_1_6, V::ONE), y);

    if P::POLICY.check_overflow {
        y = x.is_infinite().select(V::ZERO, y);
    }

    y
}

#[inline(always)]
pub fn sinc_pi_internal<V, E: FloatElement, P>(x: V) -> V
where
    V: FloatVectorWithBits<Element = E> + SpecializedTranscendentalMath<E>,
    P: Policy,
{
    if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
        // forwards to the above medium-precision sinc implementation,
        // which uses sin(x) * rcp(x)
        return V::sinc_p::<P>(x * FloatConsts::PI);
    }

    let pi_2_frac_6: V = V::splat(FloatElementWithBits::from_f64(
        1.6449340668482264364724151666460251892189499012068, // pi^2/6
    ));

    let frac_120_pi_4: V = V::splat(FloatElementWithBits::from_f64(
        1.2319178705621202226983339920542432970193362224366, // 120/pi^4, flipped for division
    ));

    let is_tiny = x.abs().cmp_le(V::FOURTH_ROOT_EPSILON);

    let x2 = x.square();

    // if branching, use Taylor series for tiny x without calling sine.
    if !P::POLICY.avoid_branching && crate::unlikely(is_tiny.all()) {
        let pi_4_frac_120: V = V::splat(FloatElementWithBits::from_f64(
            0.81174242528335364363700277240587592708106321393905,
        ));

        // unlike sinc, which has x^2/120 with 120 being an exact integer,
        // sinc_pi has pi^4/120, and since pi is irrational and imprecise anyway, we
        // can avoid the exact division by 120 in favor of multiplying by pi^4/120
        return x2.mul_add(x2.mul_sube(pi_4_frac_120, pi_2_frac_6), V::ONE);
    }

    // for very small x, sinc_pi(x) ~ 1 - (pi^2/6)*x^2 + (pi^4/120)*x^4
    let num = is_tiny.select(x2, V::sin_pi::<P>(x));
    let den = is_tiny.select(frac_120_pi_4, x * V::PI); // NOTE: first term is flipped for division

    // combined division, since division is expensive
    let mut y = num / den;

    y = is_tiny.select(x2.mul_adde(y - pi_2_frac_6, V::ONE), y);

    if P::POLICY.check_overflow {
        y = x.is_infinite().select(V::ZERO, y);
    }

    y
}

#[inline(always)]
pub fn log_n_internal<V, E: FloatElement, P, const N: usize>(x: V) -> V
where
    V: FloatVectorWithBits<Element = E> + SpecializedTranscendentalMath<E>,
    P: Policy,
{
    match N {
        // 0 and 1 are special cases, and these are what Wolfram Alpha returns
        0 => V::ZERO,               // log(x)/log(0) = log(x)/-infinity = 0
        1 => FloatVector::INFINITY, // log(x)/log(1) = log(x)/0 = complex infinity, only return real part
        2 => V::log2::<P>(x),
        10 => V::log10::<P>(x),
        n if n <= 32 => {
            #[rustfmt::skip] #[allow(clippy::approx_constant)]
            const LOG_TABLE: [f64; 30] = [ // precomputed 1/Table[log(x), {x, 3, 32}]
                1.0 / 1.0986122886681096913952452369225257046474905578227, 1.0 / 1.3862943611198906188344642429163531361510002687205,
                1.0 / 1.6094379124341003746007593332261876395256013542685, 1.0 / 1.7917594692280550008124773583807022727229906921830,
                1.0 / 1.9459101490553133051053527434431797296370847295819, 1.0 / 2.0794415416798359282516963643745297042265004030808,
                1.0 / 2.1972245773362193827904904738450514092949811156455, 1.0 / 2.3025850929940456840179914546843642076011014886288,
                1.0 / 2.3978952727983705440619435779651292998217068539374, 1.0 / 2.4849066497880003102297094798388788407984908265433,
                1.0 / 2.5649493574615367360534874415653186048052679447602, 1.0 / 2.6390573296152586145225848649013562977125848639421,
                1.0 / 2.7080502011022100659960045701487133441730919120913, 1.0 / 2.7725887222397812376689284858327062723020005374410,
                1.0 / 2.8332133440562160802495346178731265355882030125857, 1.0 / 2.8903717578961646922077225953032279773704812500058,
                1.0 / 2.9444389791664404600090274318878535372373792612991, 1.0 / 2.9957322735539909934352235761425407756766016229890,
                1.0 / 3.0445224377234229965005979803657054342845752874046, 1.0 / 3.0910424533583158534791756994233058678972069882977,
                1.0 / 3.1354942159291496908067528318101961184423803148404, 1.0 / 3.1780538303479456196469416012970554088739909609035,
                1.0 / 3.2188758248682007492015186664523752790512027085370, 1.0 / 3.2580965380214820454707195630234951728807680791205,
                1.0 / 3.2958368660043290741857357107675771139424716734682, 1.0 / 3.3322045101752039239398169863595328657880849983024,
                1.0 / 3.3672958299864740271832720323619116054945129139227, 1.0 / 3.4011973816621553754132366916068899122485920464515,
                1.0 / 3.4339872044851462459291643245423572104499389304806, 1.0 / 3.4657359027997265470861606072908828403775006718013,
            ];

            V::ln::<P>(x) * V::splat(E::from_f64(LOG_TABLE[n - 3]))
        }
        _ => V::ln::<P>(x) / V::splat(E::from_f64(libm::log(N as f64))),
    }
}
