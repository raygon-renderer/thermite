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
v.hermite_n::<N>()  v.hermite(n)  v.hermitev(n_vec)   // physicists' Hermite H_n (raw; overflows ~deg 48 f32)
v.hermite_function_n::<N>() v.hermite_function(n)  v.hermite_function_series_n::<N>(&c) // orthonormal psi_n = H_n e^{-x^2/2}/sqrt(2^n n! sqrt pi); O(1) to deg ~1400
v.laguerre_n::<N>(alpha)  v.laguerre(alpha, n)  v.laguerrev(alpha, n_vec)  // generalized Laguerre L_n^a (raw)
v.laguerre_function_n::<N>(alpha) v.laguerre_function(alpha, n) v.laguerre_function_series_n::<N>(alpha, &c) // orthonormal l_n^a; O(1), x to ~2800 f64
v.legendre(n, m)            v.jacobi(alpha, beta, n, m)  // single P_n^m (Condon-Shortley), P_n^{(a,b)} (m = derivative order)
v.legendre_series::<N>(&c)  // Clenshaw sum c_k P_k: phase functions by Legendre moment, multipoles
v.chebyshev::<K, N>(&coeffs)// Clenshaw eval of an N-term series; K = kind 1..=4 (T/U/V/W); Reinsch endpoint form under Best
v.lambert_w()  -> (V, V)    // (W_0(x), W_{-1}(x)), both branches at once
v.expint_n::<N>()  v.expint(n)   // exponential integral E_n(x); phi_n::<N>() / phi(n) likewise
v.bessel_n::<J, N>()  v.bessel::<Scaled<I>>(order)      // cylindrical J/Y/I/K by marker; const i32 order, or a per-lane BesselOrder
v.sph_bessel_n::<K, N>()  v.sph_bessel::<Scaled<K>>(n)   // spherical j/y/i/k, const usize order or runtime u32
v.airy::<Ai>()  v.airy::<Scaled<BiPrime>>()  v.airy_all::<SCALED>()  // one Airy value by marker, or all four
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

Bessel: all four families, spherical, Airy, and the scaled twins, for real and complex
vectors, behind **five marker-selected entry points** (`thermite_special::bessel::{J, Y, I,
K, Scaled, Ai, AiPrime, Bi, BiPrime}`):

```rust
use thermite_special::bessel::{J, I, K, Scaled, Ai, BiPrime};
x.bessel_n::<J, 2>()                              // J_2; N is i32, negative orders reflect (J_{-n} = (-1)^n J_n, I_{-n} = I_n)
x.bessel_n::<Scaled<K>, 0>()                      // e^x K_0: the natively scaled kernel, one transcendental cheaper, no underflow
x.bessel::<Scaled<I>>(BesselOrder::Real(nu))      // per-lane order of any class (Integer/HalfInteger/Thirds/Real), see bessel.rs
x.sph_bessel_n::<K, 3>()  x.sph_bessel::<J>(n)    // spherical: usize const order or runtime u32; j_0 is exactly sinc
x.airy::<Ai>()  x.airy::<Scaled<BiPrime>>()       // one Airy value, one Bessel pass (Ai skips the I half: ~1/4 of the tuple)
x.airy_all::<true>()                              // (Ai, Ai', Bi, Bi'), scaled on x > 0 (SciPy airye): 1-3 eps vs 684 unscaled at x = 100
x.bessel_n_p::<Precision, J, 2>()  2.5_f64.scalar_bessel_n::<J, 2>()   // policy and scalar forms as everywhere
z.hankel::<H1>(order)  z.hankel::<Scaled<H2>>(order)   // Complex only (ComplexSpecialMath; H1/H2 in thermite_complex::math::special): complex-valued even at real z
```

`Scaled<J>` / `Scaled<Y>` are SciPy's `jve`/`yve`, `e^{-|Im z|} J`: a unit factor on the real
axis (a real vector forwards to `J`), the bounded form on `Complex`. The family is a type
parameter because it never carries a value. Only the order does. Each marker's `BesselFamily` impl
is the dispatch into the per-family hooks on `SpecializedSpecialMath` (verbose, one per cell),
which is where `Dual` and `Complex` override, so a marker call on a composite reaches its
hand-written derivative or complex kernel with no marker-level code. Design record:
`notes/special/BESSEL_API_PLAN.md`.

```rust
v.polylog(PolylogOrder::Integer(2))   // Li_2(z), the dilogarithm; any integer order, both signs
v.polylog(PolylogOrder::Real(1.5))    // Li_s(z) at any real order (Fermi-Dirac: -Li_{j+1}(-e^x))
```

`polylog` (`SpecialMath`, so real AND complex vectors) takes a **scalar, packet-uniform**
order tagged by class: `PolylogOrder<E, S> { Integer(S), Real(E) }` with `E = Self::Element`
and `S` the signed lane element (`PolylogOrder<f64, i64>` on an `f64` vector, and on a complex
vector `Real` holds a `Complex<f64>` that must be real, on a dual vector a `Dual` element
with no derivative part), because integer order is table-driven (a few ulp) and real order
runs a live `zeta(s-k)` sweep per call (Roughan's 1e-12 class). The whole precompute runs
in the element type through the scalar math surface. Nothing converts through `f64`. Real vectors return the
**real part** of the principal value (on the cut `z > 1` the imaginary part is dropped,
and accuracy there is normwise). Complex vectors return the full value. On the cut it
follows the sign of `Im z`'s zero, C99 style (`-0` = the value mpmath gives a bare real).
Real(s) exactly whole snaps to Integer. Autodiff closes by `Li_s' = Li_{s-1}/z`. Kernel:
`specialized/generic/polylog.rs` (real, Goertzel real parts) and thermite-complex's
`math/special/polylog.rs`. Design and measurements are in `notes/special/`.

## `RealSpecialMath` (real vectors only; uses ordering/sign/|x|)

```rust
v.erfinv()   v.probit()                 // inverse error fn / inverse normal CDF
v.ndtr()                                // standard normal CDF Phi(x) = erfc(-x/sqrt2)/2, the forward of probit; underflows at x < -38.6 (f64)
v.log_ndtr()                            // ln Phi(x), finite for every finite x (log_ndtr(-100) = -5004.6); erfc in the moderate region
                                        // (bit-identical to ln(ndtr)), erfcx two tiers up in the tail, ln_1p of the complement on the right
v.logerfc()                             // ln erfc(x), finite where erfc has underflowed (x > 27); ln_1p(+-erf|x|) below 1/2, so below
                                        // Best it is only ABSOLUTELY accurate there (f64 erf(0) = 2.2e-16 at the default tier; Best+
                                        // takes fdlibm's small-x arm and is ~1 ulp relative everywhere)
v.inv_log_ndtr()                        // x with ln Phi(x) = y, y <= 0 (SciPy ndtri_exp); Newton on log_ndtr, finite where probit(e^y) isn't
v.inv_digamma()                         // x > 0 with digamma(x) = y (digammainv); Newton + Stirling fixed point above y = 3
v.wright_omega()                        // w with w + ln w = x, i.e. W_0(e^x) without forming e^x; series below x = -7
                                        // All three: newtons_method under MaxIterations<P, 8>, seeds at LessPrecision<P>, Dual by the
                                        // implicit function theorem (never through the loop). See generic/inverses.rs for the shape.
v.bessel_ratio::<I>(nu)                 // I_nu(x)/I_{nu-1}(x): vMF mean resultant length, p = 2 nu (nu = 3/2 is langevin); series/CF/quotient
v.inv_bessel_ratio::<I>(nu)              // kappa with that ratio = r: the vMF MLE concentration in any dimension; ill-conditioned as r -> 1
v.bessel_ratio_1m::<I>(nu)              // 1 - that ratio, full relative accuracy in the tail (x >= 8 nu evaluated directly)
v.inv_bessel_ratio_1m::<I>(nu)           // kappa from t = 1 - r: the well-conditioned form for nearly concentrated data (inv_langevin_1m move)
k.gauss_legendre(n) -> (x_k, w_k)       // k-th root of P_n and its weight, INDEX PER LANE (k as a float vector): a packet of 0..n IS the rule
k.gauss_hermite(n)                      // same shape, weight e^{-x^2}; unscaled weights; n <= ~170 f64 / ~40 f32
k.gauss_laguerre(alpha, n)              // same shape, weight x^alpha e^{-x}, alpha per lane; unscaled; n <= ~170 f64 / ~20 f32
                                        // Compensated panics on the Bessel-ratio ones (as for bessel_iv); Dual: closed-form A', constants for Gauss.
v.boxcox(lambda)                        // (x^l - 1)/l, ln x at l=0; powf_m1-based, no series needed
v.boxcox_1p(lambda)                     // shifted: ((1+x)^l - 1)/l, ln1p(x) at l=0; core compound_m1
v.inv_boxcox(lambda)                    // (l y + 1)^(1/l), e^y at l=0; = exp(ln1p(l y)/l), never forms the base
v.inv_boxcox_1p(lambda)                 // shifted inverse; same exponent, expm1 outside
v.yeo_johnson(lambda)                   // Box-Cox extended to the whole real line
v.inv_yeo_johnson(lambda)
v.gelu(alpha)                           // GELU activation (alpha=1 standard)
v.swish(beta)                           // Swish/SiLU (beta=1 standard)
v.algebraic_sigmoid_n::<N>()  v.algebraic_sigmoid(n)  // x / (1 + |x|^N)^(1/N);  N=1 is softsign
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
let (y, dy) = x.algebraic_sigmoid_d_n::<N>();   // or algebraic_sigmoid_d(n) at a runtime degree
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

Works on `Dual` (derivatives through the duplication/AGM iterations, checked against central
differences in every argument) and `Compensated` (full double-double, ~3e-32, and carries its own
`EllipticConsts` thresholds: the `R_C` series cutoff is per element type, an f64 cutoff capped
`R_J` at 1e-19). NOT on `Complex`: a compile error, because `Complex<E>` has no `EllipticConsts`
and the kernels' region tests are real-line comparisons. Adding a composite = one
`EllipticConsts` impl for its element.

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
