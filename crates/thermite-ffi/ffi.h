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

typedef void (*InplacePtr32)(float*, uintptr_t);

typedef void (*InplacePtr64)(double*, uintptr_t);

typedef struct Thermite {
  /**
   * In-place `inverse_sqrt` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 inverse_sqrtf_inplace;
  /**
   * In-place `inverse_sqrt` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 inverse_sqrt_inplace;
  /**
   * In-place `reciprocal` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 reciprocalf_inplace;
  /**
   * In-place `reciprocal` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 reciprocal_inplace;
  /**
   * In-place `sin` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 sinf_inplace;
  /**
   * In-place `sin` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 sin_inplace;
  /**
   * In-place `cos` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 cosf_inplace;
  /**
   * In-place `cos` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 cos_inplace;
  /**
   * In-place `tan` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 tanf_inplace;
  /**
   * In-place `tan` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 tan_inplace;
  /**
   * In-place `sin_pi` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 sin_pif_inplace;
  /**
   * In-place `sin_pi` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 sin_pi_inplace;
  /**
   * In-place `cos_pi` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 cos_pif_inplace;
  /**
   * In-place `cos_pi` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 cos_pi_inplace;
  /**
   * In-place `tan_pi` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 tan_pif_inplace;
  /**
   * In-place `tan_pi` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 tan_pi_inplace;
  /**
   * In-place `sinc` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 sincf_inplace;
  /**
   * In-place `sinc` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 sinc_inplace;
  /**
   * In-place `sinc_pi` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 sinc_pif_inplace;
  /**
   * In-place `sinc_pi` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 sinc_pi_inplace;
  /**
   * In-place `sinh` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 sinhf_inplace;
  /**
   * In-place `sinh` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 sinh_inplace;
  /**
   * In-place `cosh` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 coshf_inplace;
  /**
   * In-place `cosh` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 cosh_inplace;
  /**
   * In-place `asin` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 asinf_inplace;
  /**
   * In-place `asin` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 asin_inplace;
  /**
   * In-place `acos` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 acosf_inplace;
  /**
   * In-place `acos` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 acos_inplace;
  /**
   * In-place `atan` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 atanf_inplace;
  /**
   * In-place `atan` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 atan_inplace;
  /**
   * In-place `asinh` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 asinhf_inplace;
  /**
   * In-place `asinh` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 asinh_inplace;
  /**
   * In-place `acosh` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 acoshf_inplace;
  /**
   * In-place `acosh` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 acosh_inplace;
  /**
   * In-place `atanh` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 atanhf_inplace;
  /**
   * In-place `atanh` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 atanh_inplace;
  /**
   * In-place `exp` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 expf_inplace;
  /**
   * In-place `exp` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 exp_inplace;
  /**
   * In-place `exph` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 exphf_inplace;
  /**
   * In-place `exph` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 exph_inplace;
  /**
   * In-place `exp2` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 exp2f_inplace;
  /**
   * In-place `exp2` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 exp2_inplace;
  /**
   * In-place `exp10` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 exp10f_inplace;
  /**
   * In-place `exp10` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 exp10_inplace;
  /**
   * In-place `exp_m1` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 exp_m1f_inplace;
  /**
   * In-place `exp_m1` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 exp_m1_inplace;
  /**
   * In-place `ln` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 lnf_inplace;
  /**
   * In-place `ln` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 ln_inplace;
  /**
   * In-place `ln_1p` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 ln_1pf_inplace;
  /**
   * In-place `ln_1p` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 ln_1p_inplace;
  /**
   * In-place `log2` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 log2f_inplace;
  /**
   * In-place `log2` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 log2_inplace;
  /**
   * In-place `log10` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 log10f_inplace;
  /**
   * In-place `log10` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 log10_inplace;
  /**
   * In-place `cbrt` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 cbrtf_inplace;
  /**
   * In-place `cbrt` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 cbrt_inplace;
  /**
   * In-place `wrap_angle` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 wrap_anglef_inplace;
  /**
   * In-place `wrap_angle` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 wrap_angle_inplace;
  /**
   * In-place `to_degrees` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 to_degreesf_inplace;
  /**
   * In-place `to_degrees` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 to_degrees_inplace;
  /**
   * In-place `to_radians` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 to_radiansf_inplace;
  /**
   * In-place `to_radians` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 to_radians_inplace;
  /**
   * In-place `erf` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 erff_inplace;
  /**
   * In-place `erf` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 erf_inplace;
  /**
   * In-place `erfc` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 erfcf_inplace;
  /**
   * In-place `erfc` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 erfc_inplace;
  /**
   * In-place `smoothstep` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 smoothstepf_inplace;
  /**
   * In-place `smoothstep` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 smoothstep_inplace;
  /**
   * In-place `inverse_smoothstep` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 inverse_smoothstepf_inplace;
  /**
   * In-place `inverse_smoothstep` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 inverse_smoothstep_inplace;
  /**
   * In-place `smootherstep` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 smootherstepf_inplace;
  /**
   * In-place `smootherstep` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 smootherstep_inplace;
  /**
   * In-place `inverse_smootherstep` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr32 inverse_smootherstepf_inplace;
  /**
   * In-place `inverse_smootherstep` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
   */
  InplacePtr64 inverse_smootherstep_inplace;
  const char *name;
} Thermite;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * In-place `inverse_sqrt` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_inverse_sqrtf_inplace(float *ptr, uintptr_t len);

/**
 * In-place `inverse_sqrt` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_inverse_sqrt_inplace(double *ptr, uintptr_t len);

/**
 * In-place `reciprocal` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_reciprocalf_inplace(float *ptr, uintptr_t len);

/**
 * In-place `reciprocal` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_reciprocal_inplace(double *ptr, uintptr_t len);

/**
 * In-place `sin` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_sinf_inplace(float *ptr, uintptr_t len);

/**
 * In-place `sin` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_sin_inplace(double *ptr, uintptr_t len);

/**
 * In-place `cos` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_cosf_inplace(float *ptr, uintptr_t len);

/**
 * In-place `cos` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_cos_inplace(double *ptr, uintptr_t len);

/**
 * In-place `tan` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_tanf_inplace(float *ptr, uintptr_t len);

/**
 * In-place `tan` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_tan_inplace(double *ptr, uintptr_t len);

/**
 * In-place `sin_pi` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_sin_pif_inplace(float *ptr, uintptr_t len);

/**
 * In-place `sin_pi` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_sin_pi_inplace(double *ptr, uintptr_t len);

/**
 * In-place `cos_pi` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_cos_pif_inplace(float *ptr, uintptr_t len);

/**
 * In-place `cos_pi` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_cos_pi_inplace(double *ptr, uintptr_t len);

/**
 * In-place `tan_pi` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_tan_pif_inplace(float *ptr, uintptr_t len);

/**
 * In-place `tan_pi` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_tan_pi_inplace(double *ptr, uintptr_t len);

/**
 * In-place `sinc` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_sincf_inplace(float *ptr, uintptr_t len);

/**
 * In-place `sinc` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_sinc_inplace(double *ptr, uintptr_t len);

/**
 * In-place `sinc_pi` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_sinc_pif_inplace(float *ptr, uintptr_t len);

/**
 * In-place `sinc_pi` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_sinc_pi_inplace(double *ptr, uintptr_t len);

/**
 * In-place `sinh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_sinhf_inplace(float *ptr, uintptr_t len);

/**
 * In-place `sinh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_sinh_inplace(double *ptr, uintptr_t len);

/**
 * In-place `cosh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_coshf_inplace(float *ptr, uintptr_t len);

/**
 * In-place `cosh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_cosh_inplace(double *ptr, uintptr_t len);

/**
 * In-place `asin` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_asinf_inplace(float *ptr, uintptr_t len);

/**
 * In-place `asin` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_asin_inplace(double *ptr, uintptr_t len);

/**
 * In-place `acos` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_acosf_inplace(float *ptr, uintptr_t len);

/**
 * In-place `acos` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_acos_inplace(double *ptr, uintptr_t len);

/**
 * In-place `atan` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_atanf_inplace(float *ptr, uintptr_t len);

/**
 * In-place `atan` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_atan_inplace(double *ptr, uintptr_t len);

/**
 * In-place `asinh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_asinhf_inplace(float *ptr, uintptr_t len);

/**
 * In-place `asinh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_asinh_inplace(double *ptr, uintptr_t len);

/**
 * In-place `acosh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_acoshf_inplace(float *ptr, uintptr_t len);

/**
 * In-place `acosh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_acosh_inplace(double *ptr, uintptr_t len);

/**
 * In-place `atanh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_atanhf_inplace(float *ptr, uintptr_t len);

/**
 * In-place `atanh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_atanh_inplace(double *ptr, uintptr_t len);

/**
 * In-place `exp` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_expf_inplace(float *ptr, uintptr_t len);

/**
 * In-place `exp` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_exp_inplace(double *ptr, uintptr_t len);

/**
 * In-place `exph` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_exphf_inplace(float *ptr, uintptr_t len);

/**
 * In-place `exph` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_exph_inplace(double *ptr, uintptr_t len);

/**
 * In-place `exp2` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_exp2f_inplace(float *ptr, uintptr_t len);

/**
 * In-place `exp2` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_exp2_inplace(double *ptr, uintptr_t len);

/**
 * In-place `exp10` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_exp10f_inplace(float *ptr, uintptr_t len);

/**
 * In-place `exp10` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_exp10_inplace(double *ptr, uintptr_t len);

/**
 * In-place `exp_m1` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_exp_m1f_inplace(float *ptr, uintptr_t len);

/**
 * In-place `exp_m1` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_exp_m1_inplace(double *ptr, uintptr_t len);

/**
 * In-place `ln` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_lnf_inplace(float *ptr, uintptr_t len);

/**
 * In-place `ln` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_ln_inplace(double *ptr, uintptr_t len);

/**
 * In-place `ln_1p` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_ln_1pf_inplace(float *ptr, uintptr_t len);

/**
 * In-place `ln_1p` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_ln_1p_inplace(double *ptr, uintptr_t len);

/**
 * In-place `log2` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_log2f_inplace(float *ptr, uintptr_t len);

/**
 * In-place `log2` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_log2_inplace(double *ptr, uintptr_t len);

/**
 * In-place `log10` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_log10f_inplace(float *ptr, uintptr_t len);

/**
 * In-place `log10` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_log10_inplace(double *ptr, uintptr_t len);

/**
 * In-place `cbrt` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_cbrtf_inplace(float *ptr, uintptr_t len);

/**
 * In-place `cbrt` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_cbrt_inplace(double *ptr, uintptr_t len);

/**
 * In-place `wrap_angle` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_wrap_anglef_inplace(float *ptr, uintptr_t len);

/**
 * In-place `wrap_angle` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_wrap_angle_inplace(double *ptr, uintptr_t len);

/**
 * In-place `to_degrees` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_to_degreesf_inplace(float *ptr, uintptr_t len);

/**
 * In-place `to_degrees` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_to_degrees_inplace(double *ptr, uintptr_t len);

/**
 * In-place `to_radians` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_to_radiansf_inplace(float *ptr, uintptr_t len);

/**
 * In-place `to_radians` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_to_radians_inplace(double *ptr, uintptr_t len);

/**
 * In-place `erf` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_erff_inplace(float *ptr, uintptr_t len);

/**
 * In-place `erf` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_erf_inplace(double *ptr, uintptr_t len);

/**
 * In-place `erfc` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_erfcf_inplace(float *ptr, uintptr_t len);

/**
 * In-place `erfc` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_erfc_inplace(double *ptr, uintptr_t len);

/**
 * In-place `smoothstep` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_smoothstepf_inplace(float *ptr, uintptr_t len);

/**
 * In-place `smoothstep` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_smoothstep_inplace(double *ptr, uintptr_t len);

/**
 * In-place `inverse_smoothstep` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_inverse_smoothstepf_inplace(float *ptr, uintptr_t len);

/**
 * In-place `inverse_smoothstep` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_inverse_smoothstep_inplace(double *ptr, uintptr_t len);

/**
 * In-place `smootherstep` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_smootherstepf_inplace(float *ptr, uintptr_t len);

/**
 * In-place `smootherstep` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_smootherstep_inplace(double *ptr, uintptr_t len);

/**
 * In-place `inverse_smootherstep` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_inverse_smootherstepf_inplace(float *ptr, uintptr_t len);

/**
 * In-place `inverse_smootherstep` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
 */
THERMITE_API void thermite_inverse_smootherstep_inplace(double *ptr, uintptr_t len);

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
