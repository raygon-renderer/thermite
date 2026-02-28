#ifndef THERMITE_H
#define THERMITE_H

#include <stdint.h>
#ifdef _WIN32
    // If we are consuming the DLL, import the symbols
    #define THERMITE_API __declspec(dllimport)
#else
    // Fallback for Linux/macOS or static linking
    #define THERMITE_API
#endif


enum ThermitePrecisionPolicy
#ifdef __cplusplus
  : int32_t
#endif // __cplusplus
 {
  /**
   * Smart default that balances performance and accuracy, using faster approximations when they are sufficiently accurate.
   */
  DefaultPolicy = 0,
  /**
   * Prioritizes performance over accuracy, using the fastest available approximations.
   */
  HighPerformance = -1,
  /**
   * Prioritizes accuracy over performance, using the most precise approximations available, or covering
   * more edge cases at the cost of performance.
   */
  HighPrecision = 1,
};
#ifndef __cplusplus
typedef int32_t ThermitePrecisionPolicy;
#endif // __cplusplus

typedef struct Thermite {
  /**
   * `inverse_sqrt` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*inverse_sqrtf)(uintptr_t len, const float *x, float *out);
  /**
   * `inverse_sqrt` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*inverse_sqrt)(uintptr_t len, const double *x, double *out);
  /**
   * `reciprocal` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*reciprocalf)(uintptr_t len, const float *x, float *out);
  /**
   * `reciprocal` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*reciprocal)(uintptr_t len, const double *x, double *out);
  /**
   * `sin` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*sinf)(uintptr_t len, const float *x, float *y);
  /**
   * `sin` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*sin)(uintptr_t len, const double *x, double *y);
  /**
   * `cos` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*cosf)(uintptr_t len, const float *x, float *y);
  /**
   * `cos` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*cos)(uintptr_t len, const double *x, double *y);
  /**
   * `tan` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*tanf)(uintptr_t len, const float *x, float *y);
  /**
   * `tan` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*tan)(uintptr_t len, const double *x, double *y);
  /**
   * `sin_pi` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*sin_pif)(uintptr_t len, const float *x, float *y);
  /**
   * `sin_pi` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*sin_pi)(uintptr_t len, const double *x, double *y);
  /**
   * `cos_pi` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*cos_pif)(uintptr_t len, const float *x, float *y);
  /**
   * `cos_pi` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*cos_pi)(uintptr_t len, const double *x, double *y);
  /**
   * `tan_pi` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*tan_pif)(uintptr_t len, const float *x, float *y);
  /**
   * `tan_pi` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*tan_pi)(uintptr_t len, const double *x, double *y);
  /**
   * `sinc` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*sincf)(uintptr_t len, const float *x, float *y);
  /**
   * `sinc` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*sinc)(uintptr_t len, const double *x, double *y);
  /**
   * `sinc_pi` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*sinc_pif)(uintptr_t len, const float *x, float *y);
  /**
   * `sinc_pi` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*sinc_pi)(uintptr_t len, const double *x, double *y);
  /**
   * `sinh` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*sinhf)(uintptr_t len, const float *x, float *y);
  /**
   * `sinh` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*sinh)(uintptr_t len, const double *x, double *y);
  /**
   * `cosh` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*coshf)(uintptr_t len, const float *x, float *y);
  /**
   * `cosh` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*cosh)(uintptr_t len, const double *x, double *y);
  /**
   * `tanh` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*tanhf)(uintptr_t len, const float *x, float *y);
  /**
   * `tanh` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*tanh)(uintptr_t len, const double *x, double *y);
  /**
   * `asin` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*asinf)(uintptr_t len, const float *y, float *x);
  /**
   * `asin` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*asin)(uintptr_t len, const double *y, double *x);
  /**
   * `acos` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*acosf)(uintptr_t len, const float *y, float *x);
  /**
   * `acos` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*acos)(uintptr_t len, const double *y, double *x);
  /**
   * `atan` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*atanf)(uintptr_t len, const float *y, float *x);
  /**
   * `atan` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*atan)(uintptr_t len, const double *y, double *x);
  /**
   * `asinh` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*asinhf)(uintptr_t len, const float *y, float *x);
  /**
   * `asinh` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*asinh)(uintptr_t len, const double *y, double *x);
  /**
   * `acosh` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*acoshf)(uintptr_t len, const float *y, float *x);
  /**
   * `acosh` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*acosh)(uintptr_t len, const double *y, double *x);
  /**
   * `atanh` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*atanhf)(uintptr_t len, const float *y, float *x);
  /**
   * `atanh` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*atanh)(uintptr_t len, const double *y, double *x);
  /**
   * `exp` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*expf)(uintptr_t len, const float *x, float *y);
  /**
   * `exp` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*exp)(uintptr_t len, const double *x, double *y);
  /**
   * `exph` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*exphf)(uintptr_t len, const float *x, float *y);
  /**
   * `exph` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*exph)(uintptr_t len, const double *x, double *y);
  /**
   * `exp2` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*exp2f)(uintptr_t len, const float *x, float *y);
  /**
   * `exp2` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*exp2)(uintptr_t len, const double *x, double *y);
  /**
   * `exp10` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*exp10f)(uintptr_t len, const float *x, float *y);
  /**
   * `exp10` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*exp10)(uintptr_t len, const double *x, double *y);
  /**
   * `exp_m1` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*exp_m1f)(uintptr_t len, const float *x, float *y);
  /**
   * `exp_m1` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*exp_m1)(uintptr_t len, const double *x, double *y);
  /**
   * `ln` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*lnf)(uintptr_t len, const float *x, float *y);
  /**
   * `ln` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*ln)(uintptr_t len, const double *x, double *y);
  /**
   * `ln_1p` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*ln_1pf)(uintptr_t len, const float *x, float *y);
  /**
   * `ln_1p` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*ln_1p)(uintptr_t len, const double *x, double *y);
  /**
   * `log2` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*log2f)(uintptr_t len, const float *x, float *y);
  /**
   * `log2` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*log2)(uintptr_t len, const double *x, double *y);
  /**
   * `log10` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*log10f)(uintptr_t len, const float *x, float *y);
  /**
   * `log10` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*log10)(uintptr_t len, const double *x, double *y);
  /**
   * `cbrt` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*cbrtf)(uintptr_t len, const float *x, float *y);
  /**
   * `cbrt` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*cbrt)(uintptr_t len, const double *x, double *y);
  /**
   * `powf` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*powff)(uintptr_t len, const float *x, const float *e, float *y);
  /**
   * `powf` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*powf)(uintptr_t len, const double *x, const double *e, double *y);
  /**
   * `wrap_angle` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*wrap_anglef)(uintptr_t len, const float *x, float *y);
  /**
   * `wrap_angle` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*wrap_angle)(uintptr_t len, const double *x, double *y);
  /**
   * `to_degrees` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*to_degreesf)(uintptr_t len, const float *x, float *y);
  /**
   * `to_degrees` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*to_degrees)(uintptr_t len, const double *x, double *y);
  /**
   * `to_radians` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*to_radiansf)(uintptr_t len, const float *x, float *y);
  /**
   * `to_radians` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*to_radians)(uintptr_t len, const double *x, double *y);
  /**
   * `atan2` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*atan2f)(uintptr_t len, const float *y, const float *x, float *t);
  /**
   * `atan2` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*atan2)(uintptr_t len, const double *y, const double *x, double *t);
  /**
   * `erf` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*erff)(uintptr_t len, const float *x, float *y);
  /**
   * `erf` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*erf)(uintptr_t len, const double *x, double *y);
  /**
   * `erfc` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*erfcf)(uintptr_t len, const float *x, float *y);
  /**
   * `erfc` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*erfc)(uintptr_t len, const double *x, double *y);
  /**
   * `tgamma` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*tgammaf)(uintptr_t len, const float *x, float *y);
  /**
   * `tgamma` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*tgamma)(uintptr_t len, const double *x, double *y);
  /**
   * `lgamma` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*lgammaf)(uintptr_t len, const float *x, float *y);
  /**
   * `lgamma` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*lgamma)(uintptr_t len, const double *x, double *y);
  /**
   * `smoothstep` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*smoothstepf)(uintptr_t len, const float *x, float *y);
  /**
   * `smoothstep` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*smoothstep)(uintptr_t len, const double *x, double *y);
  /**
   * `inverse_smoothstep` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*inverse_smoothstepf)(uintptr_t len, const float *y, float *x);
  /**
   * `inverse_smoothstep` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*inverse_smoothstep)(uintptr_t len, const double *y, double *x);
  /**
   * `smootherstep` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*smootherstepf)(uintptr_t len, const float *x, float *y);
  /**
   * `smootherstep` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*smootherstep)(uintptr_t len, const double *x, double *y);
  /**
   * `inverse_smootherstep` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*inverse_smootherstepf)(uintptr_t len, const float *y, float *x);
  /**
   * `inverse_smootherstep` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*inverse_smootherstep)(uintptr_t len, const double *y, double *x);
  /**
   * `lerpv` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*lerpvf)(uintptr_t len, const float *t, const float *a, const float *b, float *y);
  /**
   * `lerpv` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*lerpv)(uintptr_t len, const double *t, const double *a, const double *b, double *y);
  /**
   * `sin_cos` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*sin_cosf)(uintptr_t len, const float *x, float *sin, float *cos);
  /**
   * `sin_cos` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*sin_cos)(uintptr_t len, const double *x, double *sin, double *cos);
  /**
   * `sin_cos_pi` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*sin_cos_pif)(uintptr_t len, const float *x, float *sin, float *cos);
  /**
   * `sin_cos_pi` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*sin_cos_pi)(uintptr_t len, const double *x, double *sin, double *cos);
  /**
   * `sinh_cosh` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*sinh_coshf)(uintptr_t len, const float *x, float *sinh, float *cosh);
  /**
   * `sinh_cosh` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*sinh_cosh)(uintptr_t len, const double *x, double *sinh, double *cosh);
  uintptr_t alignment;
  const char *name;
} Thermite;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * `inverse_sqrt` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_inverse_sqrtf(uintptr_t len,
                            const float *x,
                            float *out);

/**
 * `inverse_sqrt` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_inverse_sqrt(uintptr_t len,
                           const double *x,
                           double *out);

/**
 * `reciprocal` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_reciprocalf(uintptr_t len,
                          const float *x,
                          float *out);

/**
 * `reciprocal` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_reciprocal(uintptr_t len,
                         const double *x,
                         double *out);

/**
 * `sin` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_sinf(uintptr_t len,
                   const float *x,
                   float *y);

/**
 * `sin` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_sin(uintptr_t len,
                  const double *x,
                  double *y);

/**
 * `cos` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_cosf(uintptr_t len,
                   const float *x,
                   float *y);

/**
 * `cos` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_cos(uintptr_t len,
                  const double *x,
                  double *y);

/**
 * `tan` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_tanf(uintptr_t len,
                   const float *x,
                   float *y);

/**
 * `tan` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_tan(uintptr_t len,
                  const double *x,
                  double *y);

/**
 * `sin_pi` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_sin_pif(uintptr_t len,
                      const float *x,
                      float *y);

/**
 * `sin_pi` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_sin_pi(uintptr_t len,
                     const double *x,
                     double *y);

/**
 * `cos_pi` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_cos_pif(uintptr_t len,
                      const float *x,
                      float *y);

/**
 * `cos_pi` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_cos_pi(uintptr_t len,
                     const double *x,
                     double *y);

/**
 * `tan_pi` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_tan_pif(uintptr_t len,
                      const float *x,
                      float *y);

/**
 * `tan_pi` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_tan_pi(uintptr_t len,
                     const double *x,
                     double *y);

/**
 * `sinc` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_sincf(uintptr_t len,
                    const float *x,
                    float *y);

/**
 * `sinc` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_sinc(uintptr_t len,
                   const double *x,
                   double *y);

/**
 * `sinc_pi` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_sinc_pif(uintptr_t len,
                       const float *x,
                       float *y);

/**
 * `sinc_pi` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_sinc_pi(uintptr_t len,
                      const double *x,
                      double *y);

/**
 * `sinh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_sinhf(uintptr_t len,
                    const float *x,
                    float *y);

/**
 * `sinh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_sinh(uintptr_t len,
                   const double *x,
                   double *y);

/**
 * `cosh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_coshf(uintptr_t len,
                    const float *x,
                    float *y);

/**
 * `cosh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_cosh(uintptr_t len,
                   const double *x,
                   double *y);

/**
 * `tanh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_tanhf(uintptr_t len,
                    const float *x,
                    float *y);

/**
 * `tanh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_tanh(uintptr_t len,
                   const double *x,
                   double *y);

/**
 * `asin` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_asinf(uintptr_t len,
                    const float *y,
                    float *x);

/**
 * `asin` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_asin(uintptr_t len,
                   const double *y,
                   double *x);

/**
 * `acos` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_acosf(uintptr_t len,
                    const float *y,
                    float *x);

/**
 * `acos` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_acos(uintptr_t len,
                   const double *y,
                   double *x);

/**
 * `atan` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_atanf(uintptr_t len,
                    const float *y,
                    float *x);

/**
 * `atan` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_atan(uintptr_t len,
                   const double *y,
                   double *x);

/**
 * `asinh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_asinhf(uintptr_t len,
                     const float *y,
                     float *x);

/**
 * `asinh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_asinh(uintptr_t len,
                    const double *y,
                    double *x);

/**
 * `acosh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_acoshf(uintptr_t len,
                     const float *y,
                     float *x);

/**
 * `acosh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_acosh(uintptr_t len,
                    const double *y,
                    double *x);

/**
 * `atanh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_atanhf(uintptr_t len,
                     const float *y,
                     float *x);

/**
 * `atanh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_atanh(uintptr_t len,
                    const double *y,
                    double *x);

/**
 * `exp` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_expf(uintptr_t len,
                   const float *x,
                   float *y);

/**
 * `exp` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_exp(uintptr_t len,
                  const double *x,
                  double *y);

/**
 * `exph` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_exphf(uintptr_t len,
                    const float *x,
                    float *y);

/**
 * `exph` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_exph(uintptr_t len,
                   const double *x,
                   double *y);

/**
 * `exp2` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_exp2f(uintptr_t len,
                    const float *x,
                    float *y);

/**
 * `exp2` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_exp2(uintptr_t len,
                   const double *x,
                   double *y);

/**
 * `exp10` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_exp10f(uintptr_t len,
                     const float *x,
                     float *y);

/**
 * `exp10` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_exp10(uintptr_t len,
                    const double *x,
                    double *y);

/**
 * `exp_m1` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_exp_m1f(uintptr_t len,
                      const float *x,
                      float *y);

/**
 * `exp_m1` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_exp_m1(uintptr_t len,
                     const double *x,
                     double *y);

/**
 * `ln` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_lnf(uintptr_t len,
                  const float *x,
                  float *y);

/**
 * `ln` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_ln(uintptr_t len,
                 const double *x,
                 double *y);

/**
 * `ln_1p` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_ln_1pf(uintptr_t len,
                     const float *x,
                     float *y);

/**
 * `ln_1p` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_ln_1p(uintptr_t len,
                    const double *x,
                    double *y);

/**
 * `log2` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_log2f(uintptr_t len,
                    const float *x,
                    float *y);

/**
 * `log2` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_log2(uintptr_t len,
                   const double *x,
                   double *y);

/**
 * `log10` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_log10f(uintptr_t len,
                     const float *x,
                     float *y);

/**
 * `log10` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_log10(uintptr_t len,
                    const double *x,
                    double *y);

/**
 * `cbrt` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_cbrtf(uintptr_t len,
                    const float *x,
                    float *y);

/**
 * `cbrt` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_cbrt(uintptr_t len,
                   const double *x,
                   double *y);

/**
 * `powf` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_powff(uintptr_t len,
                    const float *x,
                    const float *e,
                    float *y);

/**
 * `powf` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_powf(uintptr_t len,
                   const double *x,
                   const double *e,
                   double *y);

/**
 * `wrap_angle` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_wrap_anglef(uintptr_t len,
                          const float *x,
                          float *y);

/**
 * `wrap_angle` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_wrap_angle(uintptr_t len,
                         const double *x,
                         double *y);

/**
 * `to_degrees` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_to_degreesf(uintptr_t len,
                          const float *x,
                          float *y);

/**
 * `to_degrees` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_to_degrees(uintptr_t len,
                         const double *x,
                         double *y);

/**
 * `to_radians` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_to_radiansf(uintptr_t len,
                          const float *x,
                          float *y);

/**
 * `to_radians` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_to_radians(uintptr_t len,
                         const double *x,
                         double *y);

/**
 * `atan2` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_atan2f(uintptr_t len,
                     const float *y,
                     const float *x,
                     float *t);

/**
 * `atan2` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_atan2(uintptr_t len,
                    const double *y,
                    const double *x,
                    double *t);

/**
 * `erf` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_erff(uintptr_t len,
                   const float *x,
                   float *y);

/**
 * `erf` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_erf(uintptr_t len,
                  const double *x,
                  double *y);

/**
 * `erfc` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_erfcf(uintptr_t len,
                    const float *x,
                    float *y);

/**
 * `erfc` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_erfc(uintptr_t len,
                   const double *x,
                   double *y);

/**
 * `tgamma` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_tgammaf(uintptr_t len,
                      const float *x,
                      float *y);

/**
 * `tgamma` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_tgamma(uintptr_t len,
                     const double *x,
                     double *y);

/**
 * `lgamma` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_lgammaf(uintptr_t len,
                      const float *x,
                      float *y);

/**
 * `lgamma` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_lgamma(uintptr_t len,
                     const double *x,
                     double *y);

/**
 * `smoothstep` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_smoothstepf(uintptr_t len,
                          const float *x,
                          float *y);

/**
 * `smoothstep` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_smoothstep(uintptr_t len,
                         const double *x,
                         double *y);

/**
 * `inverse_smoothstep` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_inverse_smoothstepf(uintptr_t len,
                                  const float *y,
                                  float *x);

/**
 * `inverse_smoothstep` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_inverse_smoothstep(uintptr_t len,
                                 const double *y,
                                 double *x);

/**
 * `smootherstep` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_smootherstepf(uintptr_t len,
                            const float *x,
                            float *y);

/**
 * `smootherstep` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_smootherstep(uintptr_t len,
                           const double *x,
                           double *y);

/**
 * `inverse_smootherstep` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_inverse_smootherstepf(uintptr_t len,
                                    const float *y,
                                    float *x);

/**
 * `inverse_smootherstep` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_inverse_smootherstep(uintptr_t len,
                                   const double *y,
                                   double *x);

/**
 * `lerpv` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_lerpvf(uintptr_t len,
                     const float *t,
                     const float *a,
                     const float *b,
                     float *y);

/**
 * `lerpv` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_lerpv(uintptr_t len,
                    const double *t,
                    const double *a,
                    const double *b,
                    double *y);

/**
 * Initializes a Thermite FFI VTable instance with the specified precision policy, allowing the caller to choose
 * between different performance and accuracy trade-offs. This function does not allocate, and simply fills in the
 * provided VTable struct with the appropriate function pointers based on the given precision policy and available instruction set.
 *
 * # Safety
 * The caller must ensure that `vtable` is a valid pointer to a `VTable` instance.
 */
THERMITE_API
void thermite_init_vtable(struct Thermite *vtable,
                          ThermitePrecisionPolicy policy);

/**
 * Initializes the Thermite FFI, setting up the function pointers based on the current precision policy and available instruction set.
 *
 * If not set, the default precision policy is `ThermitePrecisionPolicy::DefaultPolicy`, which provides a good balance of
 * performance and accuracy for most use cases. The caller can change the precision policy by calling
 * `thermite_init_with_policy` instead of this function.
 */
THERMITE_API
void thermite_init(void);

/**
 * Initializes the Thermite FFI with a specific precision policy, allowing the caller to choose between
 * different performance and accuracy trade-offs.
 */
THERMITE_API
void thermite_init_with_policy(ThermitePrecisionPolicy policy);

THERMITE_API const char *thermite_backend_name(void);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  /* THERMITE_H */
