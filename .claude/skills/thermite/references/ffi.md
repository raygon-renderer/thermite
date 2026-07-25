# thermite-ffi

C ABI over Thermite's math library: batch (array-at-a-time) `extern "C"`
functions, runtime-dispatched to the best backend. Crate type `cdylib` (name
`thermite`) -> `thermite.dll` / `libthermite.so`.

> **Rust code should ignore this crate** -- depend on `thermite` directly for
> inlining and type safety. This is for calling from C or another C-FFI language.

## Building the library

Requires **nightly** (the only part of Thermite that does). Header via
`cbindgen`; build with the `release-ffi` profile + `-nostdlib` (mirrors CI in
`.github/workflows/ffi_artifacts.yaml`):

```bash
cbindgen -q --config crates/thermite-ffi/cbindgen.toml --crate thermite-ffi --output thermite.h
RUSTFLAGS="-C link-arg=-nostdlib" \
  cargo +nightly build --profile release-ffi -p thermite-ffi --target x86_64-pc-windows-msvc
# (or --target x86_64-unknown-linux-gnu for libthermite.so)
```

Output in `target/<target>/release-ffi/`: `thermite.dll` (+ `.dll.lib` import
lib) or `libthermite.so`. `release-ffi` = opt-level=3, fat LTO, codegen-units=1,
strip, panic=abort. Windows CI compresses the dll with `mpress`. The header
comes straight from `cbindgen.toml` + the crate (no `cargo expand` step).

## C API shape

Global functions, or a caller-held vtable:

```c
#include "thermite.h"

void        thermite_init(void);                              // best backend, default policy
void        thermite_init_with_policy(ThermitePrecisionPolicy);
void        thermite_init_vtable(Thermite *vtable, ThermitePrecisionPolicy);
const char *thermite_backend_name(void);                      // e.g. "x86_v3"
ThermiteDenormalResult thermite_disable_denormals(void);
ThermiteDenormalResult thermite_enable_denormals(void);
// Policy enum: DefaultPolicy = 0, HighPerformance = -1, HighPrecision = 1
```

`Thermite` = vtable of function pointers + `name` and `alignment`.

**Naming scheme** (from the `decl_methods!` macro in `src/lib.rs`): every batch
op has an f32 and f64 form plus a mandatory **argument-shape suffix**:

- shape: `_v` (one vector in, one out), `_vv` (two vector inputs), `_vs`
  (vector + scalar), `_vvv`, etc.
- **f32**: element marker `f` right before the shape -> `<op>f_<shape>`
  (`addf_v`, `sinf_v`, `sin_cosf_vv`, `clampf_vs`).
- **f64**: no `f` -> `<op>_<shape>` (`add_v`, `sin_v`, `clamp_vs`).
- vtable fields use exactly those names; free functions prefix `thermite_`.

Signature: `void op(size_t len, const T *in..., T *out...[, T scalars])`.
In-place allowed (same pointer in/out); overlapping-but-offset aliasing is not.

Usage (from `crates/thermite-ffi/examples/bench.c`):

```c
Thermite *vtable = malloc(sizeof(Thermite));
thermite_init_vtable(vtable, HighPerformance);
printf("backend: %s\n", vtable->name);
vtable->sin_cosf_vv(len, in, sin_out, cos_out);  // one input, two outputs
vtable->sinf_v(len, data, data);                 // in place
vtable->enable_denormals();
free(vtable);
```

Or after `thermite_init()`, free functions: `thermite_sinf_v(len, in, out)`,
`thermite_addf_v(len, a, b, out)`, etc.

## What's exported

Roughly the whole real-math surface, each op in f32 + f64 forms. Base names:

- **Arithmetic / rounding**: add, sub, mul, div, rem, min, max, abs, signum,
  round, floor, ceil, trunc, fract, next_up, next_down,
  mul_add/mul_sub/nmul_add/nmul_sub.
- **Core/transcendental**: inverse_sqrt, reciprocal, sin, cos, tan, sin_cos
  (2 out), sin_cos_pi (2 out), sinh, cosh, tanh, sinh_cosh (2 out), asin, acos,
  atan, asinh, acosh, atanh, exp, exph, exp2, exp10, exp_m1, ln, ln_1p, log2,
  log10, log (base), cbrt, powf, atan2, hypot, sin_pi/cos_pi/tan_pi/sinc/sinc_pi.
- **Real math**: wrap_angle, angle_diff, to_degrees, to_radians, lerp,
  smoothstep, smootherstep, inverse_smoothstep, inverse_smootherstep,
  smooth_interpolator, smooth_interpolator_inverse, step, clamp, gaussian, powi.
- **Special**: erf, erfc, logistic_sigmoid, tgamma, lgamma, beta, erfinv.

Shape suffix = arity: `_v`, `_vv` (`atan2`, `hypot`, `beta`), `_vs`
(`clampf_vs(len, x, min, max, out)`), `_vvv`. Multi-output ops take two `*mut`
outputs.

## Crate features

`high_performance` / `high_precision` (default on) gate those policy paths;
`disable_dispatch` inlines everything (max speed, large binary);
`ignore_denormals` / `preserve_denormals` set denormal handling.
