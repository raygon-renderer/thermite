//! Coefficients for `$\ln\Gamma(1+v)$` on `$\lvert v\rvert \le 1/2$`.
//!
//! ```math
//! \ln\Gamma(1+v) = -\gamma v + \sum_{k\ge 2} \frac{(-1)^k \zeta(k)}{k} v^k
//! ```
//!
//! These are exact constants, **not** a fit. `$\zeta(k)/k$` is computed to 60 digits and
//! rounded once, so nothing here needed the parked minimax tooling, which is the whole reason
//! this route was taken over a rational approximation.
//!
//! # Split by parity, because both signs are always wanted
//!
//! The one consumer, Temme's series, needs `$\ln\Gamma(1+v)$` **and** `$\ln\Gamma(1-v)$`. A
//! power series gives both from a single evaluation in `$w = v^2$`:
//!
//! ```math
//! \ln\Gamma(1\pm v) = \underbrace{w\,P_{\text{even}}(w)}_{\text{even }k} \pm
//!                     \underbrace{v\,P_{\text{odd}}(w)}_{\text{odd }k}
//! ```
//!
//! so the two are one polynomial pair and a sign, rather than two full evaluations. It also
//! halves the degree each polynomial has to reach.
//!
//! # Term counts
//!
//! The largest term at `$\lvert v\rvert = 1/2$` is about `$2^{-k}/k$`, so the series has to run
//! until that falls under the format's epsilon: **k = 50** for binary64 (7.4e-17 at k = 48) and
//! **k = 22** for binary32. Split by parity that is 25 + 25 and 11 + 11.
//!
//! Long for a polynomial, but every coefficient is exact and the alternative was a fitted
//! rational. This only runs on the small-`x` arm.

#![allow(clippy::excessive_precision)]

/// The parity-split series for `$\ln\Gamma(1\pm v)$`. See the [module docs](self).
pub struct LogGamma1p<E, const NE: usize, const NO: usize> {
    /// `$\zeta(k)/k$` for even `k`, starting at `k = 2`. Multiplies `$w^{1..}$`.
    pub even: [E; NE],
    /// `$-\gamma$` then `$-\zeta(k)/k$` for odd `k`. Multiplies `$v\,w^{0..}$`.
    pub odd: [E; NO],
}

pub const LGAMMA1P_F64: LogGamma1p<f64, 25, 25> = LogGamma1p {
    even: [
        0.8224670334241132,
        0.27058080842778454,
        0.1695571769974082,
        0.12550966952474304,
        0.1000994575127818,
        0.083353840546109,
        0.07143294629536133,
        0.06250095514121304,
        0.055555767627403614,
        0.05000004769810169,
        0.04545455629320467,
        0.04166666915034121,
        0.03846153903467518,
        0.035714285847333355,
        0.03333333336437758,
        0.03125000000727597,
        0.029411764707594344,
        0.027777777778181998,
        0.02631578947377995,
        0.025000000000022737,
        0.023809523809529224,
        0.02272727272727402,
        0.021739130434782917,
        0.02083333333333341,
        0.020000000000000018,
    ],
    odd: [
        -0.5772156649015329,
        -0.40068563438653143,
        -0.20738555102867398,
        -0.1440498967688461,
        -0.11133426586956469,
        -0.09095401714582904,
        -0.0769325164113522,
        -0.06666870588242046,
        -0.058823978658684585,
        -0.05263167937961666,
        -0.047619070330142226,
        -0.04347826605304026,
        -0.04000000119214014,
        -0.037037037312989324,
        -0.034482758684919304,
        -0.03225806453115042,
        -0.030303030306558044,
        -0.02857142857226011,
        -0.027027027027223673,
        -0.025641025641072283,
        -0.024390243902450117,
        -0.023255813953491015,
        -0.022222222222222855,
        -0.021276595744681003,
        -0.02040816326530616,
    ],
};

pub const LGAMMA1P_F32: LogGamma1p<f32, 11, 11> = LogGamma1p {
    even: [
        0.8224670334241132,
        0.27058080842778454,
        0.1695571769974082,
        0.12550966952474304,
        0.1000994575127818,
        0.083353840546109,
        0.07143294629536133,
        0.06250095514121304,
        0.055555767627403614,
        0.05000004769810169,
        0.04545455629320467,
    ],
    odd: [
        -0.5772156649015329,
        -0.40068563438653143,
        -0.20738555102867398,
        -0.1440498967688461,
        -0.11133426586956469,
        -0.09095401714582904,
        -0.0769325164113522,
        -0.06666870588242046,
        -0.058823978658684585,
        -0.05263167937961666,
        -0.047619070330142226,
    ],
};
