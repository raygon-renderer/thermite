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
   * `add` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*addf_v)(uintptr_t len, const float *a, const float *b, float *y);
  /**
   * `add` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*add_v)(uintptr_t len, const double *a, const double *b, double *y);
  /**
   * `sub` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*subf_v)(uintptr_t len, const float *a, const float *b, float *y);
  /**
   * `sub` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*sub_v)(uintptr_t len, const double *a, const double *b, double *y);
  /**
   * `mul` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*mulf_v)(uintptr_t len, const float *a, const float *b, float *y);
  /**
   * `mul` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*mul_v)(uintptr_t len, const double *a, const double *b, double *y);
  /**
   * `div` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*divf_v)(uintptr_t len, const float *a, const float *b, float *y);
  /**
   * `div` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*div_v)(uintptr_t len, const double *a, const double *b, double *y);
  /**
   * `rem` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*remf_v)(uintptr_t len, const float *a, const float *b, float *y);
  /**
   * `rem` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*rem_v)(uintptr_t len, const double *a, const double *b, double *y);
  /**
   * `round` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*roundf_v)(uintptr_t len, const float *x, float *y);
  /**
   * `round` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*round_v)(uintptr_t len, const double *x, double *y);
  /**
   * `trunc` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*truncf_v)(uintptr_t len, const float *x, float *y);
  /**
   * `trunc` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*trunc_v)(uintptr_t len, const double *x, double *y);
  /**
   * `inverse_sqrt` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*inverse_sqrtf_v)(uintptr_t len, const float *x, float *out);
  /**
   * `inverse_sqrt` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*inverse_sqrt_v)(uintptr_t len, const double *x, double *out);
  /**
   * `reciprocal` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*reciprocalf_v)(uintptr_t len, const float *x, float *out);
  /**
   * `reciprocal` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*reciprocal_v)(uintptr_t len, const double *x, double *out);
  /**
   * `sin_cos` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*sin_cosf_vv)(uintptr_t len, const float *x, float *sin, float *cos);
  /**
   * `sin_cos` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*sin_cos_vv)(uintptr_t len, const double *x, double *sin, double *cos);
  /**
   * `sin_cos_pi` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*sin_cos_pif_vv)(uintptr_t len, const float *x, float *sin, float *cos);
  /**
   * `sin_cos_pi` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*sin_cos_pi_vv)(uintptr_t len, const double *x, double *sin, double *cos);
  /**
   * `sinh_cosh` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*sinh_coshf_vv)(uintptr_t len, const float *x, float *sinh, float *cosh);
  /**
   * `sinh_cosh` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*sinh_cosh_vv)(uintptr_t len, const double *x, double *sinh, double *cosh);
  /**
   * `sin` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*sinf_v)(uintptr_t len, const float *x, float *y);
  /**
   * `sin` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*sin_v)(uintptr_t len, const double *x, double *y);
  /**
   * `cos` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*cosf_v)(uintptr_t len, const float *x, float *y);
  /**
   * `cos` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*cos_v)(uintptr_t len, const double *x, double *y);
  /**
   * `tan` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*tanf_v)(uintptr_t len, const float *x, float *y);
  /**
   * `tan` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*tan_v)(uintptr_t len, const double *x, double *y);
  /**
   * `sin_pi` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*sin_pif_v)(uintptr_t len, const float *x, float *y);
  /**
   * `sin_pi` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*sin_pi_v)(uintptr_t len, const double *x, double *y);
  /**
   * `cos_pi` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*cos_pif_v)(uintptr_t len, const float *x, float *y);
  /**
   * `cos_pi` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*cos_pi_v)(uintptr_t len, const double *x, double *y);
  /**
   * `tan_pi` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*tan_pif_v)(uintptr_t len, const float *x, float *y);
  /**
   * `tan_pi` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*tan_pi_v)(uintptr_t len, const double *x, double *y);
  /**
   * `sinc` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*sincf_v)(uintptr_t len, const float *x, float *y);
  /**
   * `sinc` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*sinc_v)(uintptr_t len, const double *x, double *y);
  /**
   * `sinc_pi` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*sinc_pif_v)(uintptr_t len, const float *x, float *y);
  /**
   * `sinc_pi` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*sinc_pi_v)(uintptr_t len, const double *x, double *y);
  /**
   * `sinh` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*sinhf_v)(uintptr_t len, const float *x, float *y);
  /**
   * `sinh` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*sinh_v)(uintptr_t len, const double *x, double *y);
  /**
   * `cosh` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*coshf_v)(uintptr_t len, const float *x, float *y);
  /**
   * `cosh` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*cosh_v)(uintptr_t len, const double *x, double *y);
  /**
   * `tanh` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*tanhf_v)(uintptr_t len, const float *x, float *y);
  /**
   * `tanh` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*tanh_v)(uintptr_t len, const double *x, double *y);
  /**
   * `asin` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*asinf_v)(uintptr_t len, const float *y, float *x);
  /**
   * `asin` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*asin_v)(uintptr_t len, const double *y, double *x);
  /**
   * `acos` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*acosf_v)(uintptr_t len, const float *y, float *x);
  /**
   * `acos` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*acos_v)(uintptr_t len, const double *y, double *x);
  /**
   * `atan` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*atanf_v)(uintptr_t len, const float *y, float *x);
  /**
   * `atan` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*atan_v)(uintptr_t len, const double *y, double *x);
  /**
   * `asinh` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*asinhf_v)(uintptr_t len, const float *y, float *x);
  /**
   * `asinh` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*asinh_v)(uintptr_t len, const double *y, double *x);
  /**
   * `acosh` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*acoshf_v)(uintptr_t len, const float *y, float *x);
  /**
   * `acosh` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*acosh_v)(uintptr_t len, const double *y, double *x);
  /**
   * `atanh` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*atanhf_v)(uintptr_t len, const float *y, float *x);
  /**
   * `atanh` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*atanh_v)(uintptr_t len, const double *y, double *x);
  /**
   * `exp` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*expf_v)(uintptr_t len, const float *x, float *y);
  /**
   * `exp` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*exp_v)(uintptr_t len, const double *x, double *y);
  /**
   * `exph` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*exphf_v)(uintptr_t len, const float *x, float *y);
  /**
   * `exph` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*exph_v)(uintptr_t len, const double *x, double *y);
  /**
   * `exp2` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*exp2f_v)(uintptr_t len, const float *x, float *y);
  /**
   * `exp2` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*exp2_v)(uintptr_t len, const double *x, double *y);
  /**
   * `exp10` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*exp10f_v)(uintptr_t len, const float *x, float *y);
  /**
   * `exp10` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*exp10_v)(uintptr_t len, const double *x, double *y);
  /**
   * `exp_m1` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*exp_m1f_v)(uintptr_t len, const float *x, float *y);
  /**
   * `exp_m1` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*exp_m1_v)(uintptr_t len, const double *x, double *y);
  /**
   * `ln` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*lnf_v)(uintptr_t len, const float *x, float *y);
  /**
   * `ln` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*ln_v)(uintptr_t len, const double *x, double *y);
  /**
   * `ln_1p` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*ln_1pf_v)(uintptr_t len, const float *x, float *y);
  /**
   * `ln_1p` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*ln_1p_v)(uintptr_t len, const double *x, double *y);
  /**
   * `log2` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*log2f_v)(uintptr_t len, const float *x, float *y);
  /**
   * `log2` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*log2_v)(uintptr_t len, const double *x, double *y);
  /**
   * `log10` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*log10f_v)(uintptr_t len, const float *x, float *y);
  /**
   * `log10` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*log10_v)(uintptr_t len, const double *x, double *y);
  /**
   * `log` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*logf_v)(uintptr_t len, const float *x, const float *base, float *y);
  /**
   * `log` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*log_v)(uintptr_t len, const double *x, const double *base, double *y);
  /**
   * `cbrt` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*cbrtf_v)(uintptr_t len, const float *x, float *y);
  /**
   * `cbrt` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*cbrt_v)(uintptr_t len, const double *x, double *y);
  /**
   * `powf` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*powff_v)(uintptr_t len, const float *x, const float *e, float *y);
  /**
   * `powf` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*powf_v)(uintptr_t len, const double *x, const double *e, double *y);
  /**
   * `wrap_angle` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*wrap_anglef_v)(uintptr_t len, const float *x, float *y);
  /**
   * `wrap_angle` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*wrap_angle_v)(uintptr_t len, const double *x, double *y);
  /**
   * `angle_diff` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*angle_difff_v)(uintptr_t len, const float *a, const float *b, float *d);
  /**
   * `angle_diff` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*angle_diff_v)(uintptr_t len, const double *a, const double *b, double *d);
  /**
   * `to_degrees` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*to_degreesf_v)(uintptr_t len, const float *x, float *y);
  /**
   * `to_degrees` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*to_degrees_v)(uintptr_t len, const double *x, double *y);
  /**
   * `to_radians` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*to_radiansf_v)(uintptr_t len, const float *x, float *y);
  /**
   * `to_radians` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*to_radians_v)(uintptr_t len, const double *x, double *y);
  /**
   * `atan2` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*atan2f_v)(uintptr_t len, const float *y, const float *x, float *t);
  /**
   * `atan2` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*atan2_v)(uintptr_t len, const double *y, const double *x, double *t);
  /**
   * `lerp` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*lerpf_v)(uintptr_t len, const float *t, const float *a, const float *b, float *y);
  /**
   * `lerp` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*lerp_v)(uintptr_t len, const double *t, const double *a, const double *b, double *y);
  /**
   * `hypot` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*hypotf_v)(uintptr_t len, const float *x, const float *y, float *out);
  /**
   * `hypot` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*hypot_v)(uintptr_t len, const double *x, const double *y, double *out);
  /**
   * `erf` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*erff_v)(uintptr_t len, const float *x, float *y);
  /**
   * `erf` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*erf_v)(uintptr_t len, const double *x, double *y);
  /**
   * `erfc` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*erfcf_v)(uintptr_t len, const float *x, float *y);
  /**
   * `erfc` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*erfc_v)(uintptr_t len, const double *x, double *y);
  /**
   * `erfinv` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*erfinvf_v)(uintptr_t len, const float *y, float *x);
  /**
   * `erfinv` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*erfinv_v)(uintptr_t len, const double *y, double *x);
  /**
   * `tgamma` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*tgammaf_v)(uintptr_t len, const float *x, float *y);
  /**
   * `tgamma` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*tgamma_v)(uintptr_t len, const double *x, double *y);
  /**
   * `lgamma` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*lgammaf_v)(uintptr_t len, const float *x, float *y);
  /**
   * `lgamma` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*lgamma_v)(uintptr_t len, const double *x, double *y);
  /**
   * `beta` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*betaf_v)(uintptr_t len, const float *x, const float *y, float *z);
  /**
   * `beta` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*beta_v)(uintptr_t len, const double *x, const double *y, double *z);
  /**
   * `smoothstep` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*smoothstepf_v)(uintptr_t len, const float *x, float *y);
  /**
   * `smoothstep` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*smoothstep_v)(uintptr_t len, const double *x, double *y);
  /**
   * `inverse_smoothstep` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*inverse_smoothstepf_v)(uintptr_t len, const float *y, float *x);
  /**
   * `inverse_smoothstep` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*inverse_smoothstep_v)(uintptr_t len, const double *y, double *x);
  /**
   * `smootherstep` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*smootherstepf_v)(uintptr_t len, const float *x, float *y);
  /**
   * `smootherstep` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*smootherstep_v)(uintptr_t len, const double *x, double *y);
  /**
   * `inverse_smootherstep` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*inverse_smootherstepf_v)(uintptr_t len, const float *y, float *x);
  /**
   * `inverse_smootherstep` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*inverse_smootherstep_v)(uintptr_t len, const double *y, double *x);
  /**
   * `smooth_interpolator` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*smooth_interpolatorf_v)(uintptr_t len, const float *x, float *y, float k);
  /**
   * `smooth_interpolator` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*smooth_interpolator_v)(uintptr_t len, const double *x, double *y, double k);
  /**
   * `step` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*stepf_v)(uintptr_t len, const float *x, float *y, float edge);
  /**
   * `step` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*step_v)(uintptr_t len, const double *x, double *y, double edge);
  /**
   * `lerp` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*lerpf_vs)(uintptr_t len, const float *x, float *y, float a, float b);
  /**
   * `lerp` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*lerp_vs)(uintptr_t len, const double *x, double *y, double a, double b);
  /**
   * `powi` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*powif_vs)(uintptr_t len, const float *x, float *y, int32_t exp);
  /**
   * `powi` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*powi_vs)(uintptr_t len, const double *x, double *y, int32_t exp);
  /**
   * `gaussian` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*gaussianf_vs)(uintptr_t len, const float *x, float *y, float a, float c);
  /**
   * `gaussian` operation using the given Thermite backend.
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*gaussian_vs)(uintptr_t len, const double *x, double *y, double a, double c);
  uintptr_t alignment;
  const char *name;
} Thermite;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * `add` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_addf_v(uintptr_t len,
                     const float *a,
                     const float *b,
                     float *y);

/**
 * `add` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_add_v(uintptr_t len,
                    const double *a,
                    const double *b,
                    double *y);

/**
 * `sub` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_subf_v(uintptr_t len,
                     const float *a,
                     const float *b,
                     float *y);

/**
 * `sub` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_sub_v(uintptr_t len,
                    const double *a,
                    const double *b,
                    double *y);

/**
 * `mul` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_mulf_v(uintptr_t len,
                     const float *a,
                     const float *b,
                     float *y);

/**
 * `mul` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_mul_v(uintptr_t len,
                    const double *a,
                    const double *b,
                    double *y);

/**
 * `div` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_divf_v(uintptr_t len,
                     const float *a,
                     const float *b,
                     float *y);

/**
 * `div` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_div_v(uintptr_t len,
                    const double *a,
                    const double *b,
                    double *y);

/**
 * `rem` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_remf_v(uintptr_t len,
                     const float *a,
                     const float *b,
                     float *y);

/**
 * `rem` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_rem_v(uintptr_t len,
                    const double *a,
                    const double *b,
                    double *y);

/**
 * `round` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_roundf_v(uintptr_t len,
                       const float *x,
                       float *y);

/**
 * `round` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_round_v(uintptr_t len,
                      const double *x,
                      double *y);

/**
 * `trunc` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_truncf_v(uintptr_t len,
                       const float *x,
                       float *y);

/**
 * `trunc` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_trunc_v(uintptr_t len,
                      const double *x,
                      double *y);

/**
 * `inverse_sqrt` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_inverse_sqrtf_v(uintptr_t len,
                              const float *x,
                              float *out);

/**
 * `inverse_sqrt` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_inverse_sqrt_v(uintptr_t len,
                             const double *x,
                             double *out);

/**
 * `reciprocal` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_reciprocalf_v(uintptr_t len,
                            const float *x,
                            float *out);

/**
 * `reciprocal` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_reciprocal_v(uintptr_t len,
                           const double *x,
                           double *out);

/**
 * `sin_cos` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_sin_cosf_vv(uintptr_t len,
                          const float *x,
                          float *sin,
                          float *cos);

/**
 * `sin_cos` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_sin_cos_vv(uintptr_t len,
                         const double *x,
                         double *sin,
                         double *cos);

/**
 * `sin_cos_pi` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_sin_cos_pif_vv(uintptr_t len,
                             const float *x,
                             float *sin,
                             float *cos);

/**
 * `sin_cos_pi` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_sin_cos_pi_vv(uintptr_t len,
                            const double *x,
                            double *sin,
                            double *cos);

/**
 * `sinh_cosh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_sinh_coshf_vv(uintptr_t len,
                            const float *x,
                            float *sinh,
                            float *cosh);

/**
 * `sinh_cosh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_sinh_cosh_vv(uintptr_t len,
                           const double *x,
                           double *sinh,
                           double *cosh);

/**
 * `sin` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_sinf_v(uintptr_t len,
                     const float *x,
                     float *y);

/**
 * `sin` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_sin_v(uintptr_t len,
                    const double *x,
                    double *y);

/**
 * `cos` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_cosf_v(uintptr_t len,
                     const float *x,
                     float *y);

/**
 * `cos` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_cos_v(uintptr_t len,
                    const double *x,
                    double *y);

/**
 * `tan` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_tanf_v(uintptr_t len,
                     const float *x,
                     float *y);

/**
 * `tan` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_tan_v(uintptr_t len,
                    const double *x,
                    double *y);

/**
 * `sin_pi` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_sin_pif_v(uintptr_t len,
                        const float *x,
                        float *y);

/**
 * `sin_pi` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_sin_pi_v(uintptr_t len,
                       const double *x,
                       double *y);

/**
 * `cos_pi` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_cos_pif_v(uintptr_t len,
                        const float *x,
                        float *y);

/**
 * `cos_pi` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_cos_pi_v(uintptr_t len,
                       const double *x,
                       double *y);

/**
 * `tan_pi` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_tan_pif_v(uintptr_t len,
                        const float *x,
                        float *y);

/**
 * `tan_pi` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_tan_pi_v(uintptr_t len,
                       const double *x,
                       double *y);

/**
 * `sinc` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_sincf_v(uintptr_t len,
                      const float *x,
                      float *y);

/**
 * `sinc` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_sinc_v(uintptr_t len,
                     const double *x,
                     double *y);

/**
 * `sinc_pi` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_sinc_pif_v(uintptr_t len,
                         const float *x,
                         float *y);

/**
 * `sinc_pi` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_sinc_pi_v(uintptr_t len,
                        const double *x,
                        double *y);

/**
 * `sinh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_sinhf_v(uintptr_t len,
                      const float *x,
                      float *y);

/**
 * `sinh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_sinh_v(uintptr_t len,
                     const double *x,
                     double *y);

/**
 * `cosh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_coshf_v(uintptr_t len,
                      const float *x,
                      float *y);

/**
 * `cosh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_cosh_v(uintptr_t len,
                     const double *x,
                     double *y);

/**
 * `tanh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_tanhf_v(uintptr_t len,
                      const float *x,
                      float *y);

/**
 * `tanh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_tanh_v(uintptr_t len,
                     const double *x,
                     double *y);

/**
 * `asin` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_asinf_v(uintptr_t len,
                      const float *y,
                      float *x);

/**
 * `asin` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_asin_v(uintptr_t len,
                     const double *y,
                     double *x);

/**
 * `acos` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_acosf_v(uintptr_t len,
                      const float *y,
                      float *x);

/**
 * `acos` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_acos_v(uintptr_t len,
                     const double *y,
                     double *x);

/**
 * `atan` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_atanf_v(uintptr_t len,
                      const float *y,
                      float *x);

/**
 * `atan` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_atan_v(uintptr_t len,
                     const double *y,
                     double *x);

/**
 * `asinh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_asinhf_v(uintptr_t len,
                       const float *y,
                       float *x);

/**
 * `asinh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_asinh_v(uintptr_t len,
                      const double *y,
                      double *x);

/**
 * `acosh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_acoshf_v(uintptr_t len,
                       const float *y,
                       float *x);

/**
 * `acosh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_acosh_v(uintptr_t len,
                      const double *y,
                      double *x);

/**
 * `atanh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_atanhf_v(uintptr_t len,
                       const float *y,
                       float *x);

/**
 * `atanh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_atanh_v(uintptr_t len,
                      const double *y,
                      double *x);

/**
 * `exp` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_expf_v(uintptr_t len,
                     const float *x,
                     float *y);

/**
 * `exp` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_exp_v(uintptr_t len,
                    const double *x,
                    double *y);

/**
 * `exph` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_exphf_v(uintptr_t len,
                      const float *x,
                      float *y);

/**
 * `exph` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_exph_v(uintptr_t len,
                     const double *x,
                     double *y);

/**
 * `exp2` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_exp2f_v(uintptr_t len,
                      const float *x,
                      float *y);

/**
 * `exp2` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_exp2_v(uintptr_t len,
                     const double *x,
                     double *y);

/**
 * `exp10` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_exp10f_v(uintptr_t len,
                       const float *x,
                       float *y);

/**
 * `exp10` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_exp10_v(uintptr_t len,
                      const double *x,
                      double *y);

/**
 * `exp_m1` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_exp_m1f_v(uintptr_t len,
                        const float *x,
                        float *y);

/**
 * `exp_m1` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_exp_m1_v(uintptr_t len,
                       const double *x,
                       double *y);

/**
 * `ln` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_lnf_v(uintptr_t len,
                    const float *x,
                    float *y);

/**
 * `ln` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_ln_v(uintptr_t len,
                   const double *x,
                   double *y);

/**
 * `ln_1p` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_ln_1pf_v(uintptr_t len,
                       const float *x,
                       float *y);

/**
 * `ln_1p` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_ln_1p_v(uintptr_t len,
                      const double *x,
                      double *y);

/**
 * `log2` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_log2f_v(uintptr_t len,
                      const float *x,
                      float *y);

/**
 * `log2` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_log2_v(uintptr_t len,
                     const double *x,
                     double *y);

/**
 * `log10` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_log10f_v(uintptr_t len,
                       const float *x,
                       float *y);

/**
 * `log10` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_log10_v(uintptr_t len,
                      const double *x,
                      double *y);

/**
 * `log` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_logf_v(uintptr_t len,
                     const float *x,
                     const float *base,
                     float *y);

/**
 * `log` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_log_v(uintptr_t len,
                    const double *x,
                    const double *base,
                    double *y);

/**
 * `cbrt` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_cbrtf_v(uintptr_t len,
                      const float *x,
                      float *y);

/**
 * `cbrt` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_cbrt_v(uintptr_t len,
                     const double *x,
                     double *y);

/**
 * `powf` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_powff_v(uintptr_t len,
                      const float *x,
                      const float *e,
                      float *y);

/**
 * `powf` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_powf_v(uintptr_t len,
                     const double *x,
                     const double *e,
                     double *y);

/**
 * `wrap_angle` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_wrap_anglef_v(uintptr_t len,
                            const float *x,
                            float *y);

/**
 * `wrap_angle` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_wrap_angle_v(uintptr_t len,
                           const double *x,
                           double *y);

/**
 * `angle_diff` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_angle_difff_v(uintptr_t len,
                            const float *a,
                            const float *b,
                            float *d);

/**
 * `angle_diff` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_angle_diff_v(uintptr_t len,
                           const double *a,
                           const double *b,
                           double *d);

/**
 * `to_degrees` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_to_degreesf_v(uintptr_t len,
                            const float *x,
                            float *y);

/**
 * `to_degrees` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_to_degrees_v(uintptr_t len,
                           const double *x,
                           double *y);

/**
 * `to_radians` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_to_radiansf_v(uintptr_t len,
                            const float *x,
                            float *y);

/**
 * `to_radians` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_to_radians_v(uintptr_t len,
                           const double *x,
                           double *y);

/**
 * `atan2` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_atan2f_v(uintptr_t len,
                       const float *y,
                       const float *x,
                       float *t);

/**
 * `atan2` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_atan2_v(uintptr_t len,
                      const double *y,
                      const double *x,
                      double *t);

/**
 * `lerp` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_lerpf_v(uintptr_t len,
                      const float *t,
                      const float *a,
                      const float *b,
                      float *y);

/**
 * `lerp` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_lerp_v(uintptr_t len,
                     const double *t,
                     const double *a,
                     const double *b,
                     double *y);

/**
 * `hypot` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_hypotf_v(uintptr_t len,
                       const float *x,
                       const float *y,
                       float *out);

/**
 * `hypot` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_hypot_v(uintptr_t len,
                      const double *x,
                      const double *y,
                      double *out);

/**
 * `erf` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_erff_v(uintptr_t len,
                     const float *x,
                     float *y);

/**
 * `erf` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_erf_v(uintptr_t len,
                    const double *x,
                    double *y);

/**
 * `erfc` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_erfcf_v(uintptr_t len,
                      const float *x,
                      float *y);

/**
 * `erfc` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_erfc_v(uintptr_t len,
                     const double *x,
                     double *y);

/**
 * `erfinv` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_erfinvf_v(uintptr_t len,
                        const float *y,
                        float *x);

/**
 * `erfinv` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_erfinv_v(uintptr_t len,
                       const double *y,
                       double *x);

/**
 * `tgamma` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_tgammaf_v(uintptr_t len,
                        const float *x,
                        float *y);

/**
 * `tgamma` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_tgamma_v(uintptr_t len,
                       const double *x,
                       double *y);

/**
 * `lgamma` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_lgammaf_v(uintptr_t len,
                        const float *x,
                        float *y);

/**
 * `lgamma` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_lgamma_v(uintptr_t len,
                       const double *x,
                       double *y);

/**
 * `beta` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_betaf_v(uintptr_t len,
                      const float *x,
                      const float *y,
                      float *z);

/**
 * `beta` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_beta_v(uintptr_t len,
                     const double *x,
                     const double *y,
                     double *z);

/**
 * `smoothstep` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_smoothstepf_v(uintptr_t len,
                            const float *x,
                            float *y);

/**
 * `smoothstep` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_smoothstep_v(uintptr_t len,
                           const double *x,
                           double *y);

/**
 * `inverse_smoothstep` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_inverse_smoothstepf_v(uintptr_t len,
                                    const float *y,
                                    float *x);

/**
 * `inverse_smoothstep` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_inverse_smoothstep_v(uintptr_t len,
                                   const double *y,
                                   double *x);

/**
 * `smootherstep` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_smootherstepf_v(uintptr_t len,
                              const float *x,
                              float *y);

/**
 * `smootherstep` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_smootherstep_v(uintptr_t len,
                             const double *x,
                             double *y);

/**
 * `inverse_smootherstep` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_inverse_smootherstepf_v(uintptr_t len,
                                      const float *y,
                                      float *x);

/**
 * `inverse_smootherstep` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_inverse_smootherstep_v(uintptr_t len,
                                     const double *y,
                                     double *x);

/**
 * `smooth_interpolator` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_smooth_interpolatorf_v(uintptr_t len,
                                     const float *x,
                                     float *y,
                                     float k);

/**
 * `smooth_interpolator` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_smooth_interpolator_v(uintptr_t len,
                                    const double *x,
                                    double *y,
                                    double k);

/**
 * `step` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_stepf_v(uintptr_t len,
                      const float *x,
                      float *y,
                      float edge);

/**
 * `step` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_step_v(uintptr_t len,
                     const double *x,
                     double *y,
                     double edge);

/**
 * `lerp` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_lerpf_vs(uintptr_t len,
                       const float *x,
                       float *y,
                       float a,
                       float b);

/**
 * `lerp` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_lerp_vs(uintptr_t len,
                      const double *x,
                      double *y,
                      double a,
                      double b);

/**
 * `powi` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_powif_vs(uintptr_t len,
                       const float *x,
                       float *y,
                       int32_t exp);

/**
 * `powi` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_powi_vs(uintptr_t len,
                      const double *x,
                      double *y,
                      int32_t exp);

/**
 * `gaussian` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_gaussianf_vs(uintptr_t len,
                           const float *x,
                           float *y,
                           float a,
                           float c);

/**
 * `gaussian` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_gaussian_vs(uintptr_t len,
                          const double *x,
                          double *y,
                          double a,
                          double c);

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
