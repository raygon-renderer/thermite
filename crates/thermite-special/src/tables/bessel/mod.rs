//! Coefficient tables behind the modified Bessel functions `I_0` and `I_1`.
//!
//! Coefficients from Boost.Math's I0/I1 rational approximations (BSL-1.0), taking the
//! 24-digit sets for f32 and the 53-digit sets for f64. All arrays are
//! **constant-term-first**, matching `evaluate_polynomial` there and
//! [`poly_n`](thermite::math::CoreMath::poly_n) here.
//!
//! # Why three regions and not two
//!
//! The scalar sources split `x` at 7.75 and again near the top of the range. The second
//! split isn't about the polynomial. It's there so `exp(x)` does not overflow before
//! the `/sqrt(x)` brings the product back down. The kernel handles that case by halving
//! the exponent instead, so f64 never needs [`BesselI::far`] and only reads it on the
//! `unlikely` overflow path.
//!
//! f32 is the exception, a genuine accuracy region rather than an overflow one:
//! its `large` fit is minimax over `[7.75, 50]`, and its constant term is
//! `3.98942651588e-1` against a true limit of `3.98942280401e-1`. Extending it to infinity
//! is wrong by about 16 ulp, so the far fit earns its place.
//!
//! `far` is spelled for every table even where it is unreachable (f32 `I_1`, whose single
//! large fit is already the exp-split form and is valid to infinity) so that the kernel has
//! one uniform shape and no `const`-gated array lengths.

// Boost writes its f32 sets at f64 width. Kept verbatim so the tables diff cleanly against
// the source headers. The extra digits round away at the literal.
#![allow(clippy::excessive_precision)]

pub mod airy;
pub mod jy;

/// The three minimax fits behind one modified Bessel order, plus their breakpoints.
///
/// * `small` is in `a = x^2/4`. `I_0` reads it as `1 + a P(a)`, and `I_1` as
///   `(x/2)(1 + a(1/2 + a P(a)))`, which is Boost's nested `Q` written out.
/// * `large` and `far` are in `1/x`, and both give the **scaled** value
///   `e^{-x} I_n(x) = P(1/x)/sqrt(x)`. The unscaled form multiplies the exponential back
///   in at the call site, which is why the scaled entry points cost one transcendental
///   less rather than one more.
pub struct BesselI<E, const NS: usize, const NL: usize, const NF: usize> {
    /// Constant-term-first in `a = x^2/4`, for `x < small_threshold`.
    pub small: [E; NS],
    /// Constant-term-first in `1/x`, for `small_threshold <= x < far_threshold`.
    pub large: [E; NL],
    /// Constant-term-first in `1/x`, for `x >= far_threshold`.
    pub far: [E; NF],
    /// Where the ascending series hands over to the asymptotic form. 7.75 in every table.
    pub small_threshold: E,
    /// Where `large` hands over to `far`.
    pub far_threshold: E,
}

/// Boost.Math `bessel_i0_imp<float, 24>` (BSL-1.0).
pub const BESSEL_I0_F32: BesselI<f32, 9, 5, 3> = BesselI {
    small: [
        1.00000003928615375e+00,
        2.49999576572179639e-01,
        2.77785268558399407e-02,
        1.73560257755821695e-03,
        6.96166518788906424e-05,
        1.89645733877137904e-06,
        4.29455004657565361e-08,
        3.90565476357034480e-10,
        1.48095934745267240e-11,
    ],
    large: [
        3.98942651588301770e-01,
        4.98327234176892844e-02,
        2.91866904423115499e-02,
        1.35614940793742178e-02,
        1.31409251787866793e-01,
    ],
    far: [
        3.98942391532752700e-01,
        4.98455950638200020e-02,
        2.94835666900682535e-02,
    ],
    small_threshold: 7.75,
    far_threshold: 50.0,
};

/// Boost.Math `bessel_i0_imp<double, 53>` (BSL-1.0).
///
/// The `large` fit is degree 21 with coefficients running to `2.2e15` in alternating sign.
/// That is not a defect: at `1/x <= 0.129` the tail terms are still shrinking, and the
/// polynomial tends to `1/sqrt(2 pi)` as `1/x -> 0`, which is the correct asymptotic limit,
/// so extending it past its fitted `[7.75, 500]` costs nothing, and `far` is only ever read
/// where `exp(x)` itself would overflow.
pub const BESSEL_I0_F64: BesselI<f64, 15, 22, 5> = BesselI {
    small: [
        1.00000000000000000e+00,
        2.49999999999999909e-01,
        2.77777777777782257e-02,
        1.73611111111023792e-03,
        6.94444444453352521e-05,
        1.92901234513219920e-06,
        3.93675991102510739e-08,
        6.15118672704439289e-10,
        7.59407002058973446e-12,
        7.59389793369836367e-14,
        6.27767773636292611e-16,
        4.34709704153272287e-18,
        2.63417742690109154e-20,
        1.13943037744822825e-22,
        9.07926920085624812e-25,
    ],
    large: [
        3.98942280401425088e-01,
        4.98677850604961985e-02,
        2.80506233928312623e-02,
        2.92211225166047873e-02,
        4.44207299493659561e-02,
        1.30970574605856719e-01,
        -3.35052280231727022e+00,
        2.33025711583514727e+02,
        -1.13366350697172355e+04,
        4.24057674317867331e+05,
        -1.23157028595698731e+07,
        2.80231938155267516e+08,
        -5.01883999713777929e+09,
        7.08029243015109113e+10,
        -7.84261082124811106e+11,
        6.76825737854096565e+12,
        -4.49034849696138065e+13,
        2.24155239966958995e+14,
        -8.13426467865659318e+14,
        2.02391097391687777e+15,
        -3.08675715295370878e+15,
        2.17587543863819074e+15,
    ],
    far: [
        3.98942280401432905e-01,
        4.98677850491434560e-02,
        2.80506308916506102e-02,
        2.92179096853915176e-02,
        4.53371208762579442e-02,
    ],
    small_threshold: 7.75,
    far_threshold: 500.0,
};

/// Boost.Math `bessel_i1_imp<float, 24>` (BSL-1.0).
///
/// Boost gives this order a single large fit, already in exp-split form and valid to
/// infinity, so `far` repeats it and `far_threshold` is only a bound on where the kernel
/// halves the exponent.
pub const BESSEL_I1_F32: BesselI<f32, 8, 5, 5> = BesselI {
    small: [
        8.333333221e-02,
        6.944453712e-03,
        3.472097211e-04,
        1.158047174e-05,
        2.739745142e-07,
        5.135884609e-09,
        5.262251502e-11,
        1.331933703e-12,
    ],
    large: [
        3.98942115977513013e-01,
        -1.49581264836620262e-01,
        -4.76475741878486795e-02,
        -2.65157315524784407e-02,
        -1.47148600683672014e-01,
    ],
    far: [
        3.98942115977513013e-01,
        -1.49581264836620262e-01,
        -4.76475741878486795e-02,
        -2.65157315524784407e-02,
        -1.47148600683672014e-01,
    ],
    small_threshold: 7.75,
    far_threshold: 50.0,
};

/// Boost.Math `bessel_i1_imp<double, 53>` (BSL-1.0).
pub const BESSEL_I1_F64: BesselI<f64, 13, 22, 5> = BesselI {
    small: [
        8.333333333333333803e-02,
        6.944444444444341983e-03,
        3.472222222225921045e-04,
        1.157407407354987232e-05,
        2.755731926254790268e-07,
        4.920949692800671435e-09,
        6.834657311305621830e-11,
        7.593969849687574339e-13,
        6.904822652741917551e-15,
        5.220157095351373194e-17,
        3.410720494727771276e-19,
        1.625212890947171108e-21,
        1.332898928162290861e-23,
    ],
    large: [
        3.989422804014406054e-01,
        -1.496033551613111533e-01,
        -4.675104253598537322e-02,
        -4.090895951581637791e-02,
        -5.719036414430205390e-02,
        -1.528189554374492735e-01,
        3.458284470977172076e+00,
        -2.426181371595021021e+02,
        1.178785865993440669e+04,
        -4.404655582443487334e+05,
        1.277677779341446497e+07,
        -2.903390398236656519e+08,
        5.192386898222206474e+09,
        -7.313784438967834057e+10,
        8.087824484994859552e+11,
        -6.967602516005787001e+12,
        4.614040809616582764e+13,
        -2.298849639457172489e+14,
        8.325554073334618015e+14,
        -2.067285045778906105e+15,
        3.146401654361325073e+15,
        -2.213318202179221945e+15,
    ],
    far: [
        3.989422804014314820e-01,
        -1.496033551467584157e-01,
        -4.675105322571775911e-02,
        -4.090421597376992892e-02,
        -5.843630344778927582e-02,
    ],
    small_threshold: 7.75,
    far_threshold: 500.0,
};

/// The two rational fits behind one modified Bessel function of the **second** kind, plus the
/// constant split out of the large one.
///
/// The shape differs from [`BesselI`] in three ways, all forced by `$K_\nu$` having a
/// logarithmic singularity at the origin rather than a finite value:
///
/// * The small arm is not the whole answer. `$K_0 = P(x^2) - \ln(x) I_0(x)$` and
///   `$K_1 = R(x^2)x + 1/x + \ln(x) I_1(x)$`, so the kernels _call the `I` kernels_ rather
///   than carrying a second copy of those coefficients. Boost fits its own cut-down `I` on
///   `[0,1]` instead. Reusing the shipped one costs a longer Horner and removes a table.
/// * Both arms are rationals here, where `$I$` needed only polynomials. `small_den` is `[1]`
///   wherever the source uses a plain polynomial, which folds away.
/// * `large_offset` is Boost's `Y`: a constant deliberately split out of the large fit so the
///   rational itself stays small and contributes less relative error. It is exactly
///   representable and must not be folded into the coefficients.
///
/// There is no `far` region. The third arm in the scalar sources guards `$e^{-x}$` against
/// _underflow_ (`$K$` decays where `$I$` grows), and the kernel handles it the same way, by
/// halving the exponent under [`thermite::unlikely`].
pub struct BesselK<E, const NS: usize, const DS: usize, const NL: usize, const DL: usize> {
    /// Numerator, constant-term-first, in `x^2`, for `x <= small_threshold`.
    pub small_num: [E; NS],
    /// Denominator for the same, or `[1]` where the source fit is a plain polynomial.
    pub small_den: [E; DS],
    /// Numerator, constant-term-first, in `1/x`, for `x > small_threshold`.
    pub large_num: [E; NL],
    /// Denominator for the same.
    pub large_den: [E; DL],
    /// Boost's `Y`, added to the large rational before scaling.
    pub large_offset: E,
    /// Where the series hands over to the asymptotic form. 1 in every table.
    pub small_threshold: E,
    /// Past this, `exp(-x)` is halved and applied twice so it cannot reach zero before
    /// `1/sqrt(x)` scales the product back up. Set below each type's `exp` underflow point
    /// (about 745 for f64, 103 for f32).
    pub exp_split_threshold: E,
}

/// Boost.Math `bessel_k0_imp<float, 24>` (BSL-1.0).
pub const BESSEL_K0_F32: BesselK<f32, 5, 1, 4, 4> = BesselK {
    small_num: [
        1.159315158e-01,
        2.789828686e-01,
        2.524902861e-02,
        8.457241514e-04,
        1.530051997e-05,
    ],
    small_den: [1.0],
    large_num: [2.533141220e-01, 5.221502603e-01, 6.380180669e-02, -5.934976547e-02],
    large_den: [1.000000000e+00, 2.679722431e+00, 1.561635813e+00, 1.573660661e-01],
    large_offset: 1.0,
    small_threshold: 1.0,
    exp_split_threshold: 80.0,
};

/// Boost.Math `bessel_k0_imp<double, 53>` (BSL-1.0).
pub const BESSEL_K0_F64: BesselK<f64, 8, 1, 9, 9> = BesselK {
    small_num: [
        1.159315156584124484e-01,
        2.789828789146031732e-01,
        2.524892993216121934e-02,
        8.460350907213637784e-04,
        1.491471924309617534e-05,
        1.627106892422088488e-07,
        1.208266102392756055e-09,
        6.611686391749704310e-12,
    ],
    small_den: [1.0],
    large_num: [
        2.533141373155002416e-01,
        3.628342133984595192e+00,
        1.868441889406606057e+01,
        4.306243981063412784e+01,
        4.424116209627428189e+01,
        1.562095339356220468e+01,
        -1.810138978229410898e+00,
        -1.414237994269995877e+00,
        -9.369168119754924625e-02,
    ],
    large_den: [
        1.000000000000000000e+00,
        1.494194694879908328e+01,
        8.265296455388554217e+01,
        2.162779506621866970e+02,
        2.845145155184222157e+02,
        1.851714491916334995e+02,
        5.486540717439723515e+01,
        6.118075837628957015e+00,
        1.586261269326235053e-01,
    ],
    large_offset: 1.0,
    small_threshold: 1.0,
    exp_split_threshold: 700.0,
};

/// Boost.Math `bessel_k1_imp<float, 24>` (BSL-1.0).
pub const BESSEL_K1_F32: BesselK<f32, 4, 1, 4, 4> = BesselK {
    small_num: [-3.079657469e-01, -8.537108913e-02, -4.640275408e-03, -1.156442414e-04],
    small_den: [1.0],
    large_num: [-1.970280088e-01, 2.188747807e-02, 7.270394756e-01, 2.490678196e-01],
    large_den: [1.000000000e+00, 2.274292882e+00, 9.904984851e-01, 4.585534549e-02],
    large_offset: 1.450342178,
    small_threshold: 1.0,
    exp_split_threshold: 80.0,
};

/// Boost.Math `bessel_k1_imp<double, 53>` (BSL-1.0).
pub const BESSEL_K1_F64: BesselK<f64, 4, 4, 9, 9> = BesselK {
    small_num: [
        -3.07965757829206184e-01,
        -7.80929703673074907e-02,
        -2.70619343754051620e-03,
        -2.49549522229072008e-05,
    ],
    small_den: [
        1.00000000000000000e+00,
        -2.36316836412163098e-02,
        2.64524577525962719e-04,
        -1.49749618004162787e-06,
    ],
    large_num: [
        -1.97028041029226295e-01,
        -2.32408961548087617e+00,
        -7.98269784507699938e+00,
        -2.39968410774221632e+00,
        3.28314043780858713e+01,
        5.67713761158496058e+01,
        3.30907788466509823e+01,
        6.62582288933739787e+00,
        3.08851840645286691e-01,
    ],
    large_den: [
        1.00000000000000000e+00,
        1.41811409298826118e+01,
        7.35979466317556420e+01,
        1.77821793937080859e+02,
        2.11014501598705982e+02,
        1.19425262951064454e+02,
        2.88448064302447607e+01,
        2.27912927104139732e+00,
        2.50358186953478678e-02,
    ],
    large_offset: 1.45034217834472656,
    small_threshold: 1.0,
    exp_split_threshold: 700.0,
};

/// The fits behind one _oscillatory_ Bessel function of the first kind, `$J_0$` or `$J_1$`.
///
/// Three regions, and the two below 8 are **root-factored**: each carries a zero of the
/// function, split into `root_hi/256` and a small residual so that `(x - root_hi/256) - root_lo`
/// is computed exactly near it. Without that the rational alone would lose every significant
/// digit at the zero, because the value passes through 0 while the coefficients do not.
///
/// The `hi` region is the Hankel asymptotic: an amplitude pair `(pc/qc, ps/qs)` in `$(8/x)^2$`
/// against `sin x` and `cos x`. One region for all `x > 8`, where fdlibm splits the same
/// envelope four ways (see the kernel's module docs for why that matters here and not there).
pub struct BesselJ<E, const N1: usize, const N2: usize, const NH: usize> {
    /// `(0, 4]`: rational in `x^2`, times the factored root.
    pub p1: [E; N1],
    pub q1: [E; N1],
    /// `(4, 8]`: rational in `1 - x^2/64`, times the factored root.
    pub p2: [E; N2],
    pub q2: [E; N2],
    /// `(8, inf)`: the Hankel amplitude pair, both in `(8/x)^2`.
    pub pc: [E; NH],
    pub qc: [E; NH],
    pub ps: [E; NH],
    pub qs: [E; NH],
    /// First zero, and its exact split.
    pub root1: E,
    pub root1_hi: E,
    pub root1_lo: E,
    /// Second zero, and its exact split.
    pub root2: E,
    pub root2_hi: E,
    pub root2_lo: E,
}

/// The fits behind one oscillatory Bessel function of the second kind, `$Y_0$` or `$Y_1$`.
///
/// Same shape as [`BesselJ`] with one region more and one term more. `$Y_\nu$` is singular at
/// the origin, and the singularity is carried by a `$\frac{2}{\pi}\ln(x/x_k)J_\nu(x)$` term
/// rather than by the rational, so these kernels **call the `J` kernels**, exactly as `K`
/// calls `I`. Writing the log as `$\ln(x/x_k)$` about the region's own root, rather than
/// `$\ln x$`, is what keeps that term from swamping the rational near the zero.
///
/// `$Y_0$` genuinely has three regions below 8. `$Y_1$` has two and repeats the second, so the
/// kernel has one shape and the unused threshold is simply never reached.
pub struct BesselY<E, const N1: usize, const N2: usize, const N3: usize, const NH: usize> {
    pub p1: [E; N1],
    pub q1: [E; N1],
    pub p2: [E; N2],
    pub q2: [E; N2],
    pub p3: [E; N3],
    pub q3: [E; N3],
    pub pc: [E; NH],
    pub qc: [E; NH],
    pub ps: [E; NH],
    pub qs: [E; NH],
    pub root1: E,
    pub root1_hi: E,
    pub root1_lo: E,
    pub root2: E,
    pub root2_hi: E,
    pub root2_lo: E,
    pub root3: E,
    pub root3_hi: E,
    pub root3_lo: E,
    /// Upper edge of region 1, then of region 2. Region 3 runs to 8.
    pub threshold1: E,
    pub threshold2: E,
}
