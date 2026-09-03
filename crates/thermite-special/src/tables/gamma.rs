//! The coefficient tables behind the Gamma family.
//!
//! Public so that other crates can build their own kernels on the same constants -
//! `thermite-complex` needs them for the complex Gamma family, since the tables are
//! element-specific but not real-specific. The shared real implementations that
//! consume them are re-exported at the bottom.
//!
//! The module is `#[doc(hidden)]`: sibling crates are the whole intended audience,
//! and none of this is stable surface. Coefficients, array orders and names follow
//! whatever the current approximation needs.
//!
//! A warning about reuse. The *tables* generalize; the *implementations* do not.
//! Every `*_impl` below branches on real orderings (`cmp_lt`, `floor`, `signum`) and
//! is bounded on `FloatVectorWithBits`, so none of them applies over C even where the
//! coefficients do. And within a single table, parts differ: the Lanczos sums and
//! `Digamma::p_large` are genuine analytic approximations valid off the real axis,
//! whereas `Digamma`'s `[1, 2]` rational and every `Trigamma` region are minimax
//! fits to intervals of the real line and mean nothing away from it.

/// The Lanczos approximation parameters for one element type.
///
/// All four coefficient arrays have the same length `N` (6 for f32 / `lanczos6m24`,
/// 13 for f64 / `lanczos13m53`), but they come in two *different orders* because the
/// two consumers want opposite conventions - mixing them up is silent and expensive:
///
/// * `p_rev` / `q_rev` are the unscaled Lanczos sum in **leading-term-first** order,
///   for `tgamma`, which evaluates them with `poly_rev_n_p` (that often optimizes better
///   than the rational form).
/// * `p_expg_scaled` / `q` are the `exp(g)`-scaled sum in **constant-term-first**
///   order, which is what `poly_rational_n_p` expects, for `lgamma_r` and `beta`.
///
/// The `q`/`q_rev` pair is the same polynomial `z(z+1)...(z+N-2)` written both ways.
/// Coefficients from Boost.Math's Lanczos approximations (BSL-1.0). Boost lists both `num` and
/// `denom` constant-term-first, so `p_rev`/`q_rev` are its arrays reversed.
pub struct Lanczos<E, const N: usize> {
    pub g: E,
    pub p_rev: [E; N],
    pub q_rev: [E; N],
    pub p_expg_scaled: [E; N],
    pub q: [E; N],
}

/// `ln(f32::MAX)`: the `tgamma` threshold past which the `pow` is split in two.
pub const LN_MAX_F32: f32 = 88.722839053130621324601674778549183073943430402325230485234240247;

/// `ln(f64::MAX)`: the `tgamma` threshold past which the `pow` is split in two.
pub const LN_MAX_F64: f64 = 709.782712893383973096206318586483;

/// Boost.Math `lanczos6m24` (BSL-1.0).
pub const LANCZOS_F32: Lanczos<f32, 6> = Lanczos {
    g: 1.428456135094165802001953125,

    // Boost's `num`, reversed into leading-term-first order for `poly_rev_n_p`.
    p_rev: [
        2.50662858515256974113978724717473206342,
        27.5192015197455403062503721613097825345,
        112.2526547883668146736465390902227161763,
        211.0971093028510041839168287718170827259,
        182.5248962595894264831189414768236280862,
        58.52061591769095910314047740215847630266,
    ],

    // z(z+1)(z+2)(z+3)(z+4), leading-term-first.
    q_rev: [1.0, 10.0, 35.0, 50.0, 24.0, 0.0],

    p_expg_scaled: [
        14.0261432874996476619570577285003839357,
        43.74732405540314316089531289293124360129,
        50.59547402616588964511581430025589038612,
        26.90456680562548195593733429204228910299,
        6.595765571169314946316366571954421695196,
        0.6007854010515290065101128585795542383721,
    ],

    // The same polynomial as `q_rev`, constant-term-first, for `poly_rational_n_p`.
    // NOT interchangeable with it: the two orderings agree only at z = 1, so swapping
    // them is invisible in a spot-check and wrong everywhere else.
    q: [0.0, 24.0, 50.0, 35.0, 10.0, 1.0],
};

/// Boost.Math `lanczos13m53` (BSL-1.0).
pub const LANCZOS_F64: Lanczos<f64, 13> = Lanczos {
    g: 6.024680040776729583740234375,

    p_rev: [
        2.506628274631000270164908177133837338626,
        210.8242777515793458725097339207133627117,
        8071.672002365816210638002902272250613822,
        186056.2653952234950402949897160456992822,
        2876370.628935372441225409051620849613599,
        31426415.58540019438061423162831820536287,
        248874557.8620541565114603864132294232163,
        1439720407.311721673663223072794912393972,
        6039542586.352028005064291644307297921070,
        17921034426.03720969991975575445893111267,
        35711959237.35566804944018545154716670596,
        42919803642.64909876895789904700198885093,
        23531376880.41075968857200767445163675473,
    ],

    // z(z+1)...(z+11), leading-term-first.
    q_rev: [
        1.0,
        66.0,
        1925.0,
        32670.0,
        357423.0,
        2637558.0,
        13339535.0,
        45995730.0,
        105258076.0,
        150917976.0,
        120543840.0,
        39916800.0,
        0.0,
    ],

    p_expg_scaled: [
        56906521.91347156388090791033559122686859,
        103794043.1163445451906271053616070238554,
        86363131.28813859145546927288977868422342,
        43338889.32467613834773723740590533316085,
        14605578.08768506808414169982791359218571,
        3481712.15498064590882071018964774556468,
        601859.6171681098786670226533699352302507,
        75999.29304014542649875303443598909137092,
        6955.999602515376140356310115515198987526,
        449.9445569063168119446858607650988409623,
        19.51992788247617482847860966235652136208,
        0.5098416655656676188125178644804694509993,
        0.006061842346248906525783753964555936883222,
    ],

    // As above: the same polynomial as `q_rev`, constant-term-first.
    q: [
        0.0,
        39916800.0,
        120543840.0,
        150917976.0,
        105258076.0,
        45995730.0,
        13339535.0,
        2637558.0,
        357423.0,
        32670.0,
        1925.0,
        66.0,
        1.0,
    ],
};

/// The digamma (`psi`) coefficients for one element type.
///
/// Two unrelated approximations, joined by a recurrence that moves an argument into
/// whichever one applies:
///
/// * `y` / `roots` / `p_12` / `q_12` are the `[1, 2]` rational
///   `psi(x) = (x - root)(Y + R(x - 1))`, where `root` is summed from `roots` by
///   staged subtraction to preserve bits. Being a minimax fit on a real interval, it
///   has no meaning off the real axis - a complex `digamma` can use `p_large` and
///   must not touch this part.
/// * `p_large` is the `x >= 10` asymptotic expansion in `1/(x-1)^2`, which is a
///   genuine asymptotic series and does carry over to C.
///
/// Coefficients from Boost.Math `digamma_imp_1_2` / `digamma_imp_large` (BSL-1.0).
pub struct Digamma<E, const NR: usize, const NL: usize, const NP: usize, const NQ: usize> {
    pub y: E,
    pub roots: [E; NR],
    pub p_large: [E; NL],
    pub p_12: [E; NP],
    pub q_12: [E; NQ],
}

/// 9-digit precision (24-bit mantissa).
pub const DIGAMMA_F32: Digamma<f32, 2, 3, 4, 4> = Digamma {
    y: 0.99558162689208984,

    // root = ROOTS[0] + ROOTS[1]
    roots: [
        1532632.0 / 1048576.0, // / 2^20
        0.3700660185912626595423257213284682051735604e-6,
    ],

    p_large: [
        0.083333333333333333333333333333333333333333333333333,
        -0.0083333333333333333333333333333333333333333333333333,
        0.003968253968253968253968253968253968253968253968254,
    ],

    p_12: [
        0.25479851023250261,
        -0.44981331915268368,
        -0.43916936919946835,
        -0.061041765350579073,
    ],

    q_12: [
        0.1e1,
        0.15890202430554952e1,
        0.65341249856146947,
        0.63851690523355715e-1,
    ],
};

/// 17/18-digit precision (53-bit mantissa).
pub const DIGAMMA_F64: Digamma<f64, 3, 8, 6, 7> = Digamma {
    y: 0.99558162689208984,

    // root = ROOTS[0] + ROOTS[1] + ROOTS[2]
    roots: [
        1569415565.0 / 1073741824.0,                 // / 2^30
        (381566830.0 / 1073741824.0) / 1073741824.0, // / 2^60
        0.9016312093258695918615325266959189453125e-19,
    ],

    p_large: [
        0.083333333333333333333333333333333333333333333333333,
        -0.0083333333333333333333333333333333333333333333333333,
        0.003968253968253968253968253968253968253968253968254,
        -0.0041666666666666666666666666666666666666666666666667,
        0.0075757575757575757575757575757575757575757575757576,
        -0.021092796092796092796092796092796092796092796092796,
        0.083333333333333333333333333333333333333333333333333,
        -0.44325980392156862745098039215686274509803921568627,
    ],

    p_12: [
        0.25479851061131551,
        -0.32555031186804491,
        -0.65031853770896507,
        -0.28919126444774784,
        -0.045251321448739056,
        -0.0020713321167745952,
    ],

    q_12: [
        1.0,
        2.0767117023730469,
        1.4606242909763515,
        0.43593529692665969,
        0.054151797245674225,
        0.0021284987017821144,
        -0.55789841321675513e-6,
    ],
};

/// Minimax rational coefficients for `trigamma`, in three regions of `x >= 1`.
///
/// Every array is constant-term-first, matching `poly_n_p`. Boost.Math fits these for
/// 53-bit precision and uses the *same* set for `float`, since its tag dispatch is
/// `precision <= 53 ? 53 : ...` and float is 24 bits - so unlike `digamma`, there is
/// no separate f32 table to port, and `TRIGAMMA_F32` below is this one rounded down.
pub struct Trigamma<E> {
    /// Additive offset for the `x <= 2` region, which absorbs its leading digits.
    pub offset: E,
    pub p_1_2: [E; 6],
    pub q_1_2: [E; 6],
    pub p_2_4: [E; 6],
    pub q_2_4: [E; 6],
    pub p_4_inf: [E; 7],
    pub q_4_inf: [E; 7],
}

/// Coefficients from Boost.Math `trigamma_prec` at the 53-bit tag (BSL-1.0).
/// Max error in the interpolated forms: 3.736e-17, 1.159e-17, 6.896e-18.
macro_rules! trigamma_coeffs {
    ($t:ty) => {
        Trigamma::<$t> {
            offset: 2.1093254089355469,
            p_1_2: [
                -1.1093280605946045,
                -3.8310674472619321,
                -3.3703848401898283,
                0.28080574467981213,
                1.6638069578676164,
                0.64468386819102836,
            ],
            q_1_2: [
                1.0,
                3.4535389668541151,
                4.5208926987851437,
                2.7012734178351534,
                0.64468798399785611,
                -0.20314516859987728e-6,
            ],
            p_2_4: [
                -0.13803835004508849e-7,
                0.50000049158540261,
                1.6077979838469348,
                2.5645435828098254,
                2.0534873203680393,
                0.74566981111565923,
            ],
            q_2_4: [
                1.0,
                2.8822787662376169,
                4.1681660554090917,
                2.7853527819234466,
                0.74967671848044792,
                -0.00057069112416246805,
            ],
            p_4_inf: [
                0.68947581948701249e-17,
                0.49999999999998975,
                1.0177274392923795,
                2.498208511343429,
                2.1921221359427595,
                1.5897035272532764,
                0.40154388356961734,
            ],
            q_4_inf: [
                1.0,
                1.7021215452463932,
                4.4290431747556469,
                2.9745631894384922,
                2.3013614809773616,
                0.28360399799075752,
                0.022892987908906897,
            ],
        }
    };
}

pub const TRIGAMMA_F32: Trigamma<f32> = trigamma_coeffs!(f32);
pub const TRIGAMMA_F64: Trigamma<f64> = trigamma_coeffs!(f64);

pub use crate::specialized::generic::digamma::digamma_impl;
pub use crate::specialized::generic::gamma::{beta_impl, lgamma_r_impl, tgamma_impl};
pub use crate::specialized::generic::trigamma::trigamma_impl;
