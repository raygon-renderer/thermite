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

typedef void (*InplacePtr)(float*, uintptr_t);

typedef struct Thermite {
  /**
   * In-place `sin` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr sin_inplace;
  /**
   * In-place `cos` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr cos_inplace;
  /**
   * In-place `tan` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr tan_inplace;
  /**
   * In-place `sin_pi` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr sin_pi_inplace;
  /**
   * In-place `cos_pi` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr cos_pi_inplace;
  /**
   * In-place `tan_pi` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr tan_pi_inplace;
  /**
   * In-place `sinc` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr sinc_inplace;
  /**
   * In-place `sinc_pi` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr sinc_pi_inplace;
  /**
   * In-place `sinh` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr sinh_inplace;
  /**
   * In-place `cosh` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr cosh_inplace;
  /**
   * In-place `asin` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr asin_inplace;
  /**
   * In-place `acos` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr acos_inplace;
  /**
   * In-place `atan` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr atan_inplace;
  /**
   * In-place `asinh` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr asinh_inplace;
  /**
   * In-place `acosh` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr acosh_inplace;
  /**
   * In-place `atanh` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr atanh_inplace;
  /**
   * In-place `exp` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr exp_inplace;
  /**
   * In-place `exph` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr exph_inplace;
  /**
   * In-place `exp2` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr exp2_inplace;
  /**
   * In-place `exp10` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr exp10_inplace;
  /**
   * In-place `exp_m1` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr exp_m1_inplace;
  /**
   * In-place `ln` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr ln_inplace;
  /**
   * In-place `ln_1p` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr ln_1p_inplace;
  /**
   * In-place `log2` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr log2_inplace;
  /**
   * In-place `log10` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr log10_inplace;
  /**
   * In-place `cbrt` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr cbrt_inplace;
  /**
   * In-place `inverse_sqrt` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr inverse_sqrt_inplace;
  /**
   * In-place `reciprocal` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr reciprocal_inplace;
  /**
   * In-place `wrap_angle` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr wrap_angle_inplace;
  /**
   * In-place `to_degrees` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr to_degrees_inplace;
  /**
   * In-place `to_radians` operation using the current Thermite backend.
   * # Safety
   * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
   */
  InplacePtr to_radians_inplace;
  const char *name;
} Thermite;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * In-place `sin` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_sin_inplace(float *ptr, uintptr_t len);

/**
 * In-place `cos` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_cos_inplace(float *ptr, uintptr_t len);

/**
 * In-place `tan` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_tan_inplace(float *ptr, uintptr_t len);

/**
 * In-place `sin_pi` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_sin_pi_inplace(float *ptr, uintptr_t len);

/**
 * In-place `cos_pi` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_cos_pi_inplace(float *ptr, uintptr_t len);

/**
 * In-place `tan_pi` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_tan_pi_inplace(float *ptr, uintptr_t len);

/**
 * In-place `sinc` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_sinc_inplace(float *ptr, uintptr_t len);

/**
 * In-place `sinc_pi` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_sinc_pi_inplace(float *ptr, uintptr_t len);

/**
 * In-place `sinh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_sinh_inplace(float *ptr, uintptr_t len);

/**
 * In-place `cosh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_cosh_inplace(float *ptr, uintptr_t len);

/**
 * In-place `asin` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_asin_inplace(float *ptr, uintptr_t len);

/**
 * In-place `acos` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_acos_inplace(float *ptr, uintptr_t len);

/**
 * In-place `atan` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_atan_inplace(float *ptr, uintptr_t len);

/**
 * In-place `asinh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_asinh_inplace(float *ptr, uintptr_t len);

/**
 * In-place `acosh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_acosh_inplace(float *ptr, uintptr_t len);

/**
 * In-place `atanh` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_atanh_inplace(float *ptr, uintptr_t len);

/**
 * In-place `exp` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_exp_inplace(float *ptr, uintptr_t len);

/**
 * In-place `exph` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_exph_inplace(float *ptr, uintptr_t len);

/**
 * In-place `exp2` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_exp2_inplace(float *ptr, uintptr_t len);

/**
 * In-place `exp10` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_exp10_inplace(float *ptr, uintptr_t len);

/**
 * In-place `exp_m1` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_exp_m1_inplace(float *ptr, uintptr_t len);

/**
 * In-place `ln` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_ln_inplace(float *ptr, uintptr_t len);

/**
 * In-place `ln_1p` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_ln_1p_inplace(float *ptr, uintptr_t len);

/**
 * In-place `log2` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_log2_inplace(float *ptr, uintptr_t len);

/**
 * In-place `log10` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_log10_inplace(float *ptr, uintptr_t len);

/**
 * In-place `cbrt` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_cbrt_inplace(float *ptr, uintptr_t len);

/**
 * In-place `inverse_sqrt` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_inverse_sqrt_inplace(float *ptr, uintptr_t len);

/**
 * In-place `reciprocal` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_reciprocal_inplace(float *ptr, uintptr_t len);

/**
 * In-place `wrap_angle` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_wrap_angle_inplace(float *ptr, uintptr_t len);

/**
 * In-place `to_degrees` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_to_degrees_inplace(float *ptr, uintptr_t len);

/**
 * In-place `to_radians` operation using the current Thermite backend.
 * # Safety
 * The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
 */
THERMITE_API void thermite_to_radians_inplace(float *ptr, uintptr_t len);

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
