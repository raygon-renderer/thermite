//! The four Airy values at the origin.
//!
//! These are the exact limits, not an approximation, rounded once from 40 digits:
//!
//! ```math
//! \mathrm{Ai}(0) = \frac{1}{3^{2/3}\Gamma(2/3)},\qquad
//! \mathrm{Ai}'(0) = -\frac{1}{3^{1/3}\Gamma(1/3)},\qquad
//! \mathrm{Bi}(0) = \frac{1}{3^{1/6}\Gamma(2/3)},\qquad
//! \mathrm{Bi}'(0) = \frac{3^{1/6}}{\Gamma(1/3)}
//! ```
//!
//! They exist because the Bessel route to Airy goes through
//! `$\zeta = \tfrac{2}{3}\lvert x\rvert^{3/2}$`, and at `$\zeta = 0$` the negative-order
//! Bessel functions are infinite while the `$\sqrt{\lvert x\rvert}$` in front is zero. That
//! `$0\cdot\infty$` is the only place the general route fails. It fails at `$x = 0$`,
//! which is not an exotic input.
//!
//! The guard is `$\zeta$` below the smallest normal number, not a neighbourhood of the origin.
//! `$\zeta$` is subnormal for `$3\times 10^{-216} < \lvert x\rvert < 8\times 10^{-206}$` and
//! zero below that, and across that whole span `$\mathrm{Ai}(x)$` rounds to `$\mathrm{Ai}(0)$`
//! anyway, so the substitution is exact rather than merely close. The subnormal span has to
//! be inside the guard: a subnormal `$\zeta$` carries only a few bits, and the Bessel route
//! returns them as a relative error of `$\zeta^{-1/3}$` (2.2e-2 at `$x = 10^{-215}$` when the
//! guard was `$\zeta = 0$`). Boost instead guards `$\lvert x^3\rvert/6 < \varepsilon$`, i.e.
//! every `$\lvert x\rvert < 1.1\times 10^{-5}$`. Measured, the general route is 0.7 to 4.2 eps
//! throughout that region, so the wider guard is not buying accuracy.
#![allow(clippy::excessive_precision)]

/// The four Airy values at the origin. See the [module documentation](self).
pub struct AiryZero<E> {
    /// `$\mathrm{Ai}(0)$`.
    pub ai: E,
    /// `$\mathrm{Ai}'(0)$`.
    pub aip: E,
    /// `$\mathrm{Bi}(0)$`.
    pub bi: E,
    /// `$\mathrm{Bi}'(0)$`.
    pub bip: E,
}

pub const AIRY_ZERO_F64: AiryZero<f64> = AiryZero {
    ai: 0.3550280538878172392600632,
    aip: -0.2588194037928067984051836,
    bi: 0.6149266274460007351509224,
    bip: 0.4482883573538263579148237,
};

pub const AIRY_ZERO_F32: AiryZero<f32> = AiryZero {
    ai: 0.3550280538878172392600632,
    aip: -0.2588194037928067984051836,
    bi: 0.6149266274460007351509224,
    bip: 0.4482883573538263579148237,
};
