# thermite-special

Special functions, generic over any float vector (composites too). Same trait
pattern as core math: `_p::<P>()` policy variant per method, default-policy
method without suffix, `scalar_`-prefixed surface on `f32`/`f64`.

```rust
use thermite::prelude::*;
use thermite_special::{SpecialMath, RealSpecialMath};

fn activation<V: RealSpecialMath>(x: V, beta: V) -> (V, V) { x.swish_d(beta) }
let e = Vector::<f64>::splat(1.0).erf();   // 0.842700792949715
```

Blanket impl:
`impl<E, V: FloatVector<Element=E>> SpecialMathWithPolicy for V where V: SpecializedSpecialMath<E>`
-- any float vector gets the whole surface.

## `SpecialMath` (real and complex vectors; requires `TranscendentalMath`)

```rust
v.erf()                     v.erfc()
v.logistic_sigmoid()        // 1 / (1 + e^-x)
v.softplus(k, rcp_k)        // (1/k) ln(1 + e^{kx})
v.tgamma()  v.lgamma()  v.digamma()   v.beta(y)
v.gaussian(a, c)            // a * exp(-0.5 (x/c)^2)
v.hermite::<N>()            v.hermitev(n_vec)            // physicists' Hermite H_n (raw; overflows ~deg 48 f32)
v.hermite_function::<N>()   v.hermite_function_series::<N>(&c) // orthonormal psi_n = H_n e^{-x^2/2}/sqrt(2^n n! sqrt pi); O(1) to deg ~1400
v.laguerre::<N>(alpha)      v.laguerrev(alpha, n_vec)    // generalized Laguerre L_n^a (raw)
v.laguerre_function::<N>(alpha) v.laguerre_function_series::<N>(alpha, &c) // orthonormal l_n^a; O(1), x to ~2800 f64
v.legendre(n, m)            v.jacobi(alpha, beta, n, m)  // single P_n^m (Condon-Shortley), P_n^{(a,b)} (m = derivative order)
v.legendre_series::<N>(&c)  // Clenshaw sum c_k P_k: phase functions by Legendre moment, multipoles
v.chebyshev::<K, N>(&coeffs)// Clenshaw eval of an N-term series; K = kind 1..=4 (T/U/V/W); Reinsch endpoint form under Best
v.lambert_w()  -> (V, V)    // (W_0(x), W_{-1}(x)), both branches at once
v.expint::<N>()            // exponential integral E_n(x)
k.poisson_pmf(lambda)  k.poisson_log_pmf(lambda)  // e^-l l^k/k! at REAL k >= 0; Loader saddle-point form (stirlerr+bd0), not exp(k ln l - l - lgamma)
```

Orthogonal-polynomial shapes, deliberately not uniform: `chebyshev` is series-only (no
single-`T_n` entry point - Chebyshev is an approximation basis, its quadrature nodes are
closed-form, and the one single-`T_n` use, filter response at `|x|>1`, wants
`cosh(n acosh x)` not the recurrence). `legendre`/`jacobi` are single-polynomial (Gauss
quadrature node-finding, Wigner-d, and the engine under SH/Zernike) with `legendre_series`
beside. `hermite`/`laguerre` raw forms are conventional but overflow early; the
`*_function` forms carry the weight and normalization *inside* the recurrence and are the
ones the documented applications (QHO eigenstates, beam modes, spectral methods) want.
The kernels are in `specialized/generic/{chebyshev,legendre,hermite,laguerre}.rs`; the
series/function tests in `tests/{legendre,hermite_function,laguerre_function}.rs` with
mpmath tables under `tests/*_ref/` from `scripts/orthonormal_ref.py`.

`laguerre_function` seed (`generic/laguerre.rs`, `l_0 = x^{a/2} e^{-x/2}/sqrt(Gamma(a+1))`)
is branched by weight: `a = 0` -> `g = f`; `_i` integer `a <= LAGUERRE_PRODUCT_SEED_CAP`
(170 f64 / 29 f32, a `SpecializedSpecialMath` const) -> exact factorial + `powi`, no
transcendentals; any other real `a` -> one shared path over `generic/poisson.rs::pmf_parts`
(Loader's saddle-point form: `stirlerr` series + `bd0` series; `a < 9` shifted up by an
integer with an exact product, so no `lgamma` anywhere; TwoSum on the `x/4` exponent) -
0-3 ulp at the peak where the old log form was 25-75. `poisson_pmf` is the same core. `bd0` could be spelled `-k * log1pmx((lambda-k)/k)` with
core's `log1pmx` (the DPQ/Welinder form); NOT done - the current `bd0` is the tested one.
`laguerre_function_i` is `#[skip_dispatch]` (inline into the caller) on purpose. Never
call `FloatElement::sqrt` on an `E` inside these kernels: under `no_std` it is libm's
legacy `sqrtsd` asm and cost 30x; root the splat instead.

There is no Bessel function. `bessel_j` was removed from `SpecialMath`: only f32
`J_0` was ever implemented, which left every order beyond it - and every
composite type built on it - with nothing to do but panic. It will come back as a
whole family or not at all.

## `RealSpecialMath` (real vectors only; uses ordering/sign/|x|)

```rust
v.erfinv()   v.probit()                 // inverse error fn / inverse normal CDF
v.boxcox(lambda)                        // (x^l - 1)/l, ln x at l=0; powf_m1-based, no series needed
v.boxcox_1p(lambda)                     // shifted: ((1+x)^l - 1)/l, ln1p(x) at l=0; core compound_m1
v.inv_boxcox(lambda)                    // (l y + 1)^(1/l), e^y at l=0; = exp(ln1p(l y)/l), never forms the base
v.inv_boxcox_1p(lambda)                 // shifted inverse; same exponent, expm1 outside
v.yeo_johnson(lambda)                   // Box-Cox extended to the whole real line
v.inv_yeo_johnson(lambda)
v.gelu(alpha)                           // GELU activation (alpha=1 standard)
v.swish(beta)                           // Swish/SiLU (beta=1 standard)
v.algebraic_sigmoid::<N>()              // x / (1 + |x|^N)^(1/N);  N=1 is softsign
v.algebraic_swish()                     // exp-free swish
v.lgamma_r()  -> (V, V)                 // (ln|Gamma(x)|, sign(Gamma(x)))
V::gaussian_integral(x0, x1, a, c)      // definite integral of a Gaussian
```

The power-transform family is five spellings of two functions. Yeo-Johnson is *not* a
separate algorithm: `psi(y, l) = +-boxcox_1p(|y|, l or 2 - l)`, sign folded first, which
collapses its two logarithmic special cases (`l = 0` above zero, `l = 2` below) onto the
single `l = 0` seam `boxcox_1p` already blends. The `_1p` forms are the ones Yeo-Johnson
needs and are not a convenience: `psi(y) ~ y` at the origin and the origin is where the data
is, so `boxcox(1 + y, l)` would round `y` away below `eps` and return a flat zero. Tests in
`tests/power_transform.rs`, reference generator `scripts/power_transform_ref.py` (which
writes out all four published YJ cases rather than the fold, on purpose). Autodiff through
the whole family works from the trait defaults - no `Dual` overrides.

## `RealPrimalMath` -- value + derivative pairs

For activation functions where you want the derivative alongside the value (handy
for backprop on plain real vectors, distinct from `Dual`):

```rust
let (y, dy) = x.softplus_d(k, rcp_k);
let (y, dy) = x.gelu_d(alpha);
let (y, dy) = x.swish_d(beta);
let (y, dy) = x.algebraic_sigmoid_d::<N>();
let (y, dy) = x.algebraic_swish_d();
```

## Elliptic integrals

Carlson symmetric forms and Legendre forms, dispatched via small request structs
(`thermite_special::elliptic`):

```rust
use thermite_special::elliptic::{CarlsonRf, EllintK, EllintF};

let rf = x.carlson(CarlsonRf { x, y, z });          // R_F(x,y,z)
let k  = Vector::<f64>::splat(0.5).ellint(EllintK { k });        // complete K(k)
let f  = phi.ellint(EllintF { phi, k });                         // incomplete F(phi,k)
```

Carlson kinds: `CarlsonRf`, `CarlsonRc`, `CarlsonRd`, `CarlsonRj`, `CarlsonRg`.
Legendre kinds: complete `EllintK`, `EllintE`, `EllintD`, `EllintPi`; incomplete
`EllintF`, `EllintEInc`, `EllintDInc`, `EllintPiInc`. All forms (including
phi-range reduction for the incomplete integrals and `R_J` with `p < 0`) are
implemented and tested against reference values in `tests/special_vs_libm.rs`;
remaining work is optimization-grade (f32 sweep validation, large-|phi|
accuracy), not correctness.

## Scalar surface

`ScalarSpecialMath` / `ScalarSpecialMathWithPolicy` on `f32`/`f64`:

```rust
use thermite_special::ScalarSpecialMath;
let e = 0.5_f32.scalar_erf();
let (y, _dy) = 1.0_f64.scalar_swish(1.0_f64);
```

## WIP flags

Nothing in `thermite-special` panics: every function on the public traits is
implemented for both f32 and f64, and the 209-test suite covers them against
libm. The remaining `todo!()`s in the family live *downstream*, in the composite
wrappers - `Dual::trigamma`, and six functions on `Complex<Compensated<..>>`.
See [composite-types.md](composite-types.md).
