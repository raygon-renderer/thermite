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
v.hermite::<N>()            v.hermitev(n_vec)            // physicists' Hermite H_n
v.legendre(n, m)            v.jacobi(alpha, beta, n, m)
v.chebyshev::<K, N>(&coeffs)// Clenshaw eval of an N-term series; K = kind 1..=4 (T/U/V/W)
v.lambert_w()  -> (V, V)    // (W_0(x), W_{-1}(x)), both branches at once
v.expint::<N>()            // exponential integral E_n(x)
```

There is no Bessel function. `bessel_j` was removed from `SpecialMath`: only f32
`J_0` was ever implemented, which left every order beyond it - and every
composite type built on it - with nothing to do but panic. It will come back as a
whole family or not at all.

## `RealSpecialMath` (real vectors only; uses ordering/sign/|x|)

```rust
v.erfinv()   v.probit()                 // inverse error fn / inverse normal CDF
v.gelu(alpha)                           // GELU activation (alpha=1 standard)
v.swish(beta)                           // Swish/SiLU (beta=1 standard)
v.algebraic_sigmoid::<N>()              // x / (1 + |x|^N)^(1/N);  N=1 is softsign
v.algebraic_swish()                     // exp-free swish
v.lgamma_r()  -> (V, V)                 // (ln|Gamma(x)|, sign(Gamma(x)))
V::gaussian_integral(x0, x1, a, c)      // definite integral of a Gaussian
```

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
