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
   * Floating-point addition
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*addf_v)(uintptr_t len, const float *a, const float *b, float *y);
  /**
   * Floating-point addition
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*add_v)(uintptr_t len, const double *a, const double *b, double *y);
  /**
   * Floating-point subtraction
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*subf_v)(uintptr_t len, const float *a, const float *b, float *y);
  /**
   * Floating-point subtraction
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*sub_v)(uintptr_t len, const double *a, const double *b, double *y);
  /**
   * Floating-point multiplication
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*mulf_v)(uintptr_t len, const float *a, const float *b, float *y);
  /**
   * Floating-point multiplication
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*mul_v)(uintptr_t len, const double *a, const double *b, double *y);
  /**
   * Floating-point division
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*divf_v)(uintptr_t len, const float *a, const float *b, float *y);
  /**
   * Floating-point division
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*div_v)(uintptr_t len, const double *a, const double *b, double *y);
  /**
   * Floating-point remainder (modulo/fmod)
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*remf_v)(uintptr_t len, const float *a, const float *b, float *y);
  /**
   * Floating-point remainder (modulo/fmod)
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*rem_v)(uintptr_t len, const double *a, const double *b, double *y);
  /**
   * Rounds a floating-point number to the nearest integer
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*roundf_v)(uintptr_t len, const float *x, float *y);
  /**
   * Rounds a floating-point number to the nearest integer
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*round_v)(uintptr_t len, const double *x, double *y);
  /**
   * Rounds a floating-point number down to the nearest integer
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*floorf_v)(uintptr_t len, const float *x, float *y);
  /**
   * Rounds a floating-point number down to the nearest integer
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*floor_v)(uintptr_t len, const double *x, double *y);
  /**
   * Rounds a floating-point number up to the nearest integer
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*ceilf_v)(uintptr_t len, const float *x, float *y);
  /**
   * Rounds a floating-point number up to the nearest integer
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*ceil_v)(uintptr_t len, const double *x, double *y);
  /**
   * Truncates a floating-point number, removing the fractional part
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*truncf_v)(uintptr_t len, const float *x, float *y);
  /**
   * Truncates a floating-point number, removing the fractional part
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*trunc_v)(uintptr_t len, const double *x, double *y);
  /**
   * Computes the fractional part of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*fractf_v)(uintptr_t len, const float *x, float *y);
  /**
   * Computes the fractional part of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*fract_v)(uintptr_t len, const double *x, double *y);
  /**
   * Computes the next representable floating-point value greater than the input
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*next_upf_v)(uintptr_t len, const float *x, float *y);
  /**
   * Computes the next representable floating-point value greater than the input
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*next_up_v)(uintptr_t len, const double *x, double *y);
  /**
   * Computes the next representable floating-point value less than the input
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*next_downf_v)(uintptr_t len, const float *x, float *y);
  /**
   * Computes the next representable floating-point value less than the input
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*next_down_v)(uintptr_t len, const double *x, double *y);
  /**
   * Computes the minimum of two floating-point numbers
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*minf_v)(uintptr_t len, const float *a, const float *b, float *y);
  /**
   * Computes the minimum of two floating-point numbers
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*min_v)(uintptr_t len, const double *a, const double *b, double *y);
  /**
   * Computes the maximum of two floating-point numbers
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*maxf_v)(uintptr_t len, const float *a, const float *b, float *y);
  /**
   * Computes the maximum of two floating-point numbers
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*max_v)(uintptr_t len, const double *a, const double *b, double *y);
  /**
   * Clamps a floating-point number between a minimum and maximum scalar value
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*clampf_vs)(uintptr_t len, const float *x, float *y, float min, float max);
  /**
   * Clamps a floating-point number between a minimum and maximum scalar value
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*clamp_vs)(uintptr_t len, const double *x, double *y, double min, double max);
  /**
   * Computes the absolute value of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*absf_v)(uintptr_t len, const float *x, float *y);
  /**
   * Computes the absolute value of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*abs_v)(uintptr_t len, const double *x, double *y);
  /**
   * Computes the sign of a floating-point number, returning -1.0 for negative values, 1.0 for positive values, and 0.0 for zero
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*signumf_v)(uintptr_t len, const float *x, float *y);
  /**
   * Computes the sign of a floating-point number, returning -1.0 for negative values, 1.0 for positive values, and 0.0 for zero
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*signum_v)(uintptr_t len, const double *x, double *y);
  /**
   * Computes (x * a) + b with only one rounding error, yielding a more accurate
   * result than a separate multiplication and addition
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*mul_addf_v)(uintptr_t len, const float *x, const float *a, const float *b, float *y);
  /**
   * Computes (x * a) + b with only one rounding error, yielding a more accurate
   * result than a separate multiplication and addition
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*mul_add_v)(uintptr_t len, const double *x, const double *a, const double *b, double *y);
  /**
   * Computes (x * a) - b with only one rounding error, yielding a more accurate
   * result than a separate multiplication and subtraction
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*mul_subf_v)(uintptr_t len, const float *x, const float *a, const float *b, float *y);
  /**
   * Computes (x * a) - b with only one rounding error, yielding a more accurate
   * result than a separate multiplication and subtraction
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*mul_sub_v)(uintptr_t len, const double *x, const double *a, const double *b, double *y);
  /**
   * Computes -(x * a) + b with only one rounding error, yielding a more accurate
   * result than a separate negated multiplication and addition
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*nmul_addf_v)(uintptr_t len, const float *x, const float *a, const float *b, float *y);
  /**
   * Computes -(x * a) + b with only one rounding error, yielding a more accurate
   * result than a separate negated multiplication and addition
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*nmul_add_v)(uintptr_t len, const double *x, const double *a, const double *b, double *y);
  /**
   * Computes -(x * a) - b with only one rounding error, yielding a more accurate
   * result than a separate negated multiplication and subtraction
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*nmul_subf_v)(uintptr_t len, const float *x, const float *a, const float *b, float *y);
  /**
   * Computes -(x * a) - b with only one rounding error, yielding a more accurate
   * result than a separate negated multiplication and subtraction
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*nmul_sub_v)(uintptr_t len, const double *x, const double *a, const double *b, double *y);
  /**
   * Computes the inverse square root, which may vary in accuracy and performance based on the chosen precision policy.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*inverse_sqrtf_v)(uintptr_t len, const float *x, float *out);
  /**
   * Computes the inverse square root, which may vary in accuracy and performance based on the chosen precision policy.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*inverse_sqrt_v)(uintptr_t len, const double *x, double *out);
  /**
   * Computes the reciprocal (1/x), which may vary in accuracy and performance based on the chosen precision policy.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*reciprocalf_v)(uintptr_t len, const float *x, float *out);
  /**
   * Computes the reciprocal (1/x), which may vary in accuracy and performance based on the chosen precision policy.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*reciprocal_v)(uintptr_t len, const double *x, double *out);
  /**
   * Compute both sine and cosine of the input simultaneously, which will be more efficient than computing them separately.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*sin_cosf_vv)(uintptr_t len, const float *x, float *sin, float *cos);
  /**
   * Compute both sine and cosine of the input simultaneously, which will be more efficient than computing them separately.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*sin_cos_vv)(uintptr_t len, const double *x, double *sin, double *cos);
  /**
   * Compute both sine and cosine of the input multiplied by π simultaneously, which will be more efficient than computing them separately.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*sin_cos_pif_vv)(uintptr_t len, const float *x, float *sin, float *cos);
  /**
   * Compute both sine and cosine of the input multiplied by π simultaneously, which will be more efficient than computing them separately.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*sin_cos_pi_vv)(uintptr_t len, const double *x, double *sin, double *cos);
  /**
   * Compute both hyperbolic sine and hyperbolic cosine of the input simultaneously, which will be more efficient than computing them separately.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*sinh_coshf_vv)(uintptr_t len, const float *x, float *sinh, float *cosh);
  /**
   * Compute both hyperbolic sine and hyperbolic cosine of the input simultaneously, which will be more efficient than computing them separately.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*sinh_cosh_vv)(uintptr_t len, const double *x, double *sinh, double *cosh);
  /**
   * Computes the sine of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*sinf_v)(uintptr_t len, const float *x, float *y);
  /**
   * Computes the sine of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*sin_v)(uintptr_t len, const double *x, double *y);
  /**
   * Computes the cosine of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*cosf_v)(uintptr_t len, const float *x, float *y);
  /**
   * Computes the cosine of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*cos_v)(uintptr_t len, const double *x, double *y);
  /**
   * Computes the tangent of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*tanf_v)(uintptr_t len, const float *x, float *y);
  /**
   * Computes the tangent of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*tan_v)(uintptr_t len, const double *x, double *y);
  /**
   * Computes the sine of the input multiplied by π, which may be more accurate for certain inputs than multiplying the input by π and then taking the sine.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*sin_pif_v)(uintptr_t len, const float *x, float *y);
  /**
   * Computes the sine of the input multiplied by π, which may be more accurate for certain inputs than multiplying the input by π and then taking the sine.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*sin_pi_v)(uintptr_t len, const double *x, double *y);
  /**
   * Computes the cosine of the input multiplied by π, which may be more accurate for certain inputs than multiplying the input by π and then taking the cosine.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*cos_pif_v)(uintptr_t len, const float *x, float *y);
  /**
   * Computes the cosine of the input multiplied by π, which may be more accurate for certain inputs than multiplying the input by π and then taking the cosine.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*cos_pi_v)(uintptr_t len, const double *x, double *y);
  /**
   * Computes the tangent of the input multiplied by π, which may be more accurate for certain inputs than multiplying the input by π and then taking the tangent.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*tan_pif_v)(uintptr_t len, const float *x, float *y);
  /**
   * Computes the tangent of the input multiplied by π, which may be more accurate for certain inputs than multiplying the input by π and then taking the tangent.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*tan_pi_v)(uintptr_t len, const double *x, double *y);
  /**
   * Computes the sinc function, defined as sin(πx)/(πx) for x != 0 and 1 for x = 0
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*sincf_v)(uintptr_t len, const float *x, float *y);
  /**
   * Computes the sinc function, defined as sin(πx)/(πx) for x != 0 and 1 for x = 0
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*sinc_v)(uintptr_t len, const double *x, double *y);
  /**
   * Computes the sinc function of the input multiplied by π, defined as sin(π^2 x)/(π^2 x) for x != 0 and 1 for x = 0,
   * which may be more accurate for certain inputs than multiplying the input by π and then taking the sinc.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*sinc_pif_v)(uintptr_t len, const float *x, float *y);
  /**
   * Computes the sinc function of the input multiplied by π, defined as sin(π^2 x)/(π^2 x) for x != 0 and 1 for x = 0,
   * which may be more accurate for certain inputs than multiplying the input by π and then taking the sinc.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*sinc_pi_v)(uintptr_t len, const double *x, double *y);
  /**
   * Computes the hyperbolic sine of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*sinhf_v)(uintptr_t len, const float *x, float *y);
  /**
   * Computes the hyperbolic sine of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*sinh_v)(uintptr_t len, const double *x, double *y);
  /**
   * Computes the hyperbolic cosine of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*coshf_v)(uintptr_t len, const float *x, float *y);
  /**
   * Computes the hyperbolic cosine of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*cosh_v)(uintptr_t len, const double *x, double *y);
  /**
   * Computes the hyperbolic tangent of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*tanhf_v)(uintptr_t len, const float *x, float *y);
  /**
   * Computes the hyperbolic tangent of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*tanh_v)(uintptr_t len, const double *x, double *y);
  /**
   * Computes the inverse sine (arcsine) of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*asinf_v)(uintptr_t len, const float *y, float *x);
  /**
   * Computes the inverse sine (arcsine) of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*asin_v)(uintptr_t len, const double *y, double *x);
  /**
   * Computes the inverse cosine (arccosine) of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*acosf_v)(uintptr_t len, const float *y, float *x);
  /**
   * Computes the inverse cosine (arccosine) of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*acos_v)(uintptr_t len, const double *y, double *x);
  /**
   * Computes the inverse tangent (arctangent) of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*atanf_v)(uintptr_t len, const float *y, float *x);
  /**
   * Computes the inverse tangent (arctangent) of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*atan_v)(uintptr_t len, const double *y, double *x);
  /**
   * Computes the inverse hyperbolic sine of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*asinhf_v)(uintptr_t len, const float *y, float *x);
  /**
   * Computes the inverse hyperbolic sine of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*asinh_v)(uintptr_t len, const double *y, double *x);
  /**
   * Computes the inverse hyperbolic cosine of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*acoshf_v)(uintptr_t len, const float *y, float *x);
  /**
   * Computes the inverse hyperbolic cosine of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*acosh_v)(uintptr_t len, const double *y, double *x);
  /**
   * Computes the inverse hyperbolic tangent of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*atanhf_v)(uintptr_t len, const float *y, float *x);
  /**
   * Computes the inverse hyperbolic tangent of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*atanh_v)(uintptr_t len, const double *y, double *x);
  /**
   * Computes the exponential of a floating-point number, which may vary in accuracy and performance based on the chosen precision policy.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*expf_v)(uintptr_t len, const float *x, float *y);
  /**
   * Computes the exponential of a floating-point number, which may vary in accuracy and performance based on the chosen precision policy.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*exp_v)(uintptr_t len, const double *x, double *y);
  /**
   * Computes the half-exponential of a floating-point number, defined as exp(x)/2
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*exphf_v)(uintptr_t len, const float *x, float *y);
  /**
   * Computes the half-exponential of a floating-point number, defined as exp(x)/2
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*exph_v)(uintptr_t len, const double *x, double *y);
  /**
   * Computes 2 raised to the power of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*exp2f_v)(uintptr_t len, const float *x, float *y);
  /**
   * Computes 2 raised to the power of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*exp2_v)(uintptr_t len, const double *x, double *y);
  /**
   * Computes 10 raised to the power of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*exp10f_v)(uintptr_t len, const float *x, float *y);
  /**
   * Computes 10 raised to the power of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*exp10_v)(uintptr_t len, const double *x, double *y);
  /**
   * Computes the exponential of a floating-point number minus one, which may be more accurate for small inputs than computing exp(x) - 1 directly.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*exp_m1f_v)(uintptr_t len, const float *x, float *y);
  /**
   * Computes the exponential of a floating-point number minus one, which may be more accurate for small inputs than computing exp(x) - 1 directly.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*exp_m1_v)(uintptr_t len, const double *x, double *y);
  /**
   * Computes the natural logarithm of a floating-point number, which may vary in accuracy and performance based on the chosen precision policy.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*lnf_v)(uintptr_t len, const float *x, float *y);
  /**
   * Computes the natural logarithm of a floating-point number, which may vary in accuracy and performance based on the chosen precision policy.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*ln_v)(uintptr_t len, const double *x, double *y);
  /**
   * Computes the natural logarithm of one plus a floating-point number, which may be more accurate for small inputs than computing ln(1 + x) directly.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*ln_1pf_v)(uintptr_t len, const float *x, float *y);
  /**
   * Computes the natural logarithm of one plus a floating-point number, which may be more accurate for small inputs than computing ln(1 + x) directly.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*ln_1p_v)(uintptr_t len, const double *x, double *y);
  /**
   * Computes the base-2 logarithm of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*log2f_v)(uintptr_t len, const float *x, float *y);
  /**
   * Computes the base-2 logarithm of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*log2_v)(uintptr_t len, const double *x, double *y);
  /**
   * Computes the base-10 logarithm of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*log10f_v)(uintptr_t len, const float *x, float *y);
  /**
   * Computes the base-10 logarithm of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*log10_v)(uintptr_t len, const double *x, double *y);
  /**
   * Computes the logarithm of a floating-point number with respect to an arbitrary base
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*logf_v)(uintptr_t len, const float *x, const float *base, float *y);
  /**
   * Computes the logarithm of a floating-point number with respect to an arbitrary base
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*log_v)(uintptr_t len, const double *x, const double *base, double *y);
  /**
   * Computes the cube root of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*cbrtf_v)(uintptr_t len, const float *x, float *y);
  /**
   * Computes the cube root of a floating-point number
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*cbrt_v)(uintptr_t len, const double *x, double *y);
  /**
   * Computes x raised to the power of y, which may vary in accuracy and performance based on the chosen precision policy.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*powff_v)(uintptr_t len, const float *x, const float *e, float *y);
  /**
   * Computes x raised to the power of y, which may vary in accuracy and performance based on the chosen precision policy.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*powf_v)(uintptr_t len, const double *x, const double *e, double *y);
  /**
   * Wraps an angle in radians to the range [-π, π)
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*wrap_anglef_v)(uintptr_t len, const float *x, float *y);
  /**
   * Wraps an angle in radians to the range [-π, π)
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*wrap_angle_v)(uintptr_t len, const double *x, double *y);
  /**
   * Computes the absolute difference between two angles
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*angle_difff_v)(uintptr_t len, const float *a, const float *b, float *d);
  /**
   * Computes the absolute difference between two angles
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*angle_diff_v)(uintptr_t len, const double *a, const double *b, double *d);
  /**
   * Converts an angle from radians to degrees
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*to_degreesf_v)(uintptr_t len, const float *x, float *y);
  /**
   * Converts an angle from radians to degrees
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*to_degrees_v)(uintptr_t len, const double *x, double *y);
  /**
   * Converts an angle from degrees to radians
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*to_radiansf_v)(uintptr_t len, const float *x, float *y);
  /**
   * Converts an angle from degrees to radians
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*to_radians_v)(uintptr_t len, const double *x, double *y);
  /**
   * Computes the angle (in radians) between the positive x-axis and the point (x, y), using the signs of both arguments to determine the correct quadrant of the result.
   *
   * This may vary in accuracy and performance based on the chosen precision policy.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*atan2f_v)(uintptr_t len, const float *y, const float *x, float *t);
  /**
   * Computes the angle (in radians) between the positive x-axis and the point (x, y), using the signs of both arguments to determine the correct quadrant of the result.
   *
   * This may vary in accuracy and performance based on the chosen precision policy.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*atan2_v)(uintptr_t len, const double *y, const double *x, double *t);
  /**
   * Performs linear interpolation between values a and b using t, where t is typically in the range [0, 1].
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*lerpf_v)(uintptr_t len, const float *t, const float *a, const float *b, float *y);
  /**
   * Performs linear interpolation between values a and b using t, where t is typically in the range [0, 1].
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*lerp_v)(uintptr_t len, const double *t, const double *a, const double *b, double *y);
  /**
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*hypotf_v)(uintptr_t len, const float *x, const float *y, float *out);
  /**
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*hypot_v)(uintptr_t len, const double *x, const double *y, double *out);
  /**
   * Computes the error function, which may vary in accuracy and performance based on the chosen precision policy.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*erff_v)(uintptr_t len, const float *x, float *y);
  /**
   * Computes the error function, which may vary in accuracy and performance based on the chosen precision policy.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*erf_v)(uintptr_t len, const double *x, double *y);
  /**
   * Computes the complementary error function, which may vary in accuracy and performance based on the chosen precision policy.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*erfcf_v)(uintptr_t len, const float *x, float *y);
  /**
   * Computes the complementary error function, which may vary in accuracy and performance based on the chosen precision policy.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*erfc_v)(uintptr_t len, const double *x, double *y);
  /**
   * Computes the inverse error function, which may vary in accuracy and performance based on the chosen precision policy.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*erfinvf_v)(uintptr_t len, const float *y, float *x);
  /**
   * Computes the inverse error function, which may vary in accuracy and performance based on the chosen precision policy.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*erfinv_v)(uintptr_t len, const double *y, double *x);
  /**
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*tgammaf_v)(uintptr_t len, const float *x, float *y);
  /**
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*tgamma_v)(uintptr_t len, const double *x, double *y);
  /**
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*lgammaf_v)(uintptr_t len, const float *x, float *y);
  /**
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*lgamma_v)(uintptr_t len, const double *x, double *y);
  /**
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*betaf_v)(uintptr_t len, const float *x, const float *y, float *z);
  /**
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*beta_v)(uintptr_t len, const double *x, const double *y, double *z);
  /**
   * 3rd-order smoothstep interpolation function
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*smoothstepf_v)(uintptr_t len, const float *x, float *y);
  /**
   * 3rd-order smoothstep interpolation function
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*smoothstep_v)(uintptr_t len, const double *x, double *y);
  /**
   * Inverse of the 3rd-order smoothstep function
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*inverse_smoothstepf_v)(uintptr_t len, const float *y, float *x);
  /**
   * Inverse of the 3rd-order smoothstep function
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*inverse_smoothstep_v)(uintptr_t len, const double *y, double *x);
  /**
   * 5th-order smoothstep interpolation function
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*smootherstepf_v)(uintptr_t len, const float *x, float *y);
  /**
   * 5th-order smoothstep interpolation function
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*smootherstep_v)(uintptr_t len, const double *x, double *y);
  /**
   * Inverse of the 5th-order smoothstep function
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*inverse_smootherstepf_v)(uintptr_t len, const float *y, float *x);
  /**
   * Inverse of the 5th-order smoothstep function
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*inverse_smootherstep_v)(uintptr_t len, const double *y, double *x);
  /**
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*smooth_interpolatorf_v)(uintptr_t len, const float *x, float *y, float k);
  /**
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*smooth_interpolator_v)(uintptr_t len, const double *x, double *y, double k);
  /**
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*smooth_interpolator_inversef_v)(uintptr_t len, const float *y, float *x, float k);
  /**
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*smooth_interpolator_inverse_v)(uintptr_t len, const double *y, double *x, double k);
  /**
   * Step function that returns 0.0 if x < edge and 1.0 if x >= edge
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*stepf_v)(uintptr_t len, const float *x, float *y, float edge);
  /**
   * Step function that returns 0.0 if x < edge and 1.0 if x >= edge
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*step_v)(uintptr_t len, const double *x, double *y, double edge);
  /**
   * Linear interpolation between scalars a and b by x, where x is in the range [0, 1]
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*lerpf_vs)(uintptr_t len, const float *x, float *y, float a, float b);
  /**
   * Linear interpolation between scalars a and b by x, where x is in the range [0, 1]
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*lerp_vs)(uintptr_t len, const double *x, double *y, double a, double b);
  /**
   * Raises x to the power of exp, where exp is an integer
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*powif_vs)(uintptr_t len, const float *x, float *y, int32_t exp);
  /**
   * Raises x to the power of exp, where exp is an integer
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
   */
  void (*powi_vs)(uintptr_t len, const double *x, double *y, int32_t exp);
  /**
   * Computes the Gaussian function with amplitude `a` and standard deviation `c`, defined as `a * exp(-0.5 * (self / c)^2)`.
   *
   * The position `b` is assumed to be zero. For a non-zero position, use `self - b` as the input.
   *
   * # Safety
   * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
   */
  void (*gaussianf_vs)(uintptr_t len, const float *x, float *y, float a, float c);
  /**
   * Computes the Gaussian function with amplitude `a` and standard deviation `c`, defined as `a * exp(-0.5 * (self / c)^2)`.
   *
   * The position `b` is assumed to be zero. For a non-zero position, use `self - b` as the input.
   *
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

/**
 * Floating-point addition
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_addf_v(uintptr_t len,
                     const float *a,
                     const float *b,
                     float *y);

/**
 * Floating-point addition
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_add_v(uintptr_t len,
                    const double *a,
                    const double *b,
                    double *y);

/**
 * Floating-point subtraction
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_subf_v(uintptr_t len,
                     const float *a,
                     const float *b,
                     float *y);

/**
 * Floating-point subtraction
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_sub_v(uintptr_t len,
                    const double *a,
                    const double *b,
                    double *y);

/**
 * Floating-point multiplication
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_mulf_v(uintptr_t len,
                     const float *a,
                     const float *b,
                     float *y);

/**
 * Floating-point multiplication
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_mul_v(uintptr_t len,
                    const double *a,
                    const double *b,
                    double *y);

/**
 * Floating-point division
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_divf_v(uintptr_t len,
                     const float *a,
                     const float *b,
                     float *y);

/**
 * Floating-point division
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_div_v(uintptr_t len,
                    const double *a,
                    const double *b,
                    double *y);

/**
 * Floating-point remainder (modulo/fmod)
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_remf_v(uintptr_t len,
                     const float *a,
                     const float *b,
                     float *y);

/**
 * Floating-point remainder (modulo/fmod)
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_rem_v(uintptr_t len,
                    const double *a,
                    const double *b,
                    double *y);

/**
 * Rounds a floating-point number to the nearest integer
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_roundf_v(uintptr_t len,
                       const float *x,
                       float *y);

/**
 * Rounds a floating-point number to the nearest integer
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_round_v(uintptr_t len,
                      const double *x,
                      double *y);

/**
 * Rounds a floating-point number down to the nearest integer
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_floorf_v(uintptr_t len,
                       const float *x,
                       float *y);

/**
 * Rounds a floating-point number down to the nearest integer
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_floor_v(uintptr_t len,
                      const double *x,
                      double *y);

/**
 * Rounds a floating-point number up to the nearest integer
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_ceilf_v(uintptr_t len,
                      const float *x,
                      float *y);

/**
 * Rounds a floating-point number up to the nearest integer
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_ceil_v(uintptr_t len,
                     const double *x,
                     double *y);

/**
 * Truncates a floating-point number, removing the fractional part
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_truncf_v(uintptr_t len,
                       const float *x,
                       float *y);

/**
 * Truncates a floating-point number, removing the fractional part
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_trunc_v(uintptr_t len,
                      const double *x,
                      double *y);

/**
 * Computes the fractional part of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_fractf_v(uintptr_t len,
                       const float *x,
                       float *y);

/**
 * Computes the fractional part of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_fract_v(uintptr_t len,
                      const double *x,
                      double *y);

/**
 * Computes the next representable floating-point value greater than the input
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_next_upf_v(uintptr_t len,
                         const float *x,
                         float *y);

/**
 * Computes the next representable floating-point value greater than the input
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_next_up_v(uintptr_t len,
                        const double *x,
                        double *y);

/**
 * Computes the next representable floating-point value less than the input
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_next_downf_v(uintptr_t len,
                           const float *x,
                           float *y);

/**
 * Computes the next representable floating-point value less than the input
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_next_down_v(uintptr_t len,
                          const double *x,
                          double *y);

/**
 * Computes the minimum of two floating-point numbers
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_minf_v(uintptr_t len,
                     const float *a,
                     const float *b,
                     float *y);

/**
 * Computes the minimum of two floating-point numbers
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_min_v(uintptr_t len,
                    const double *a,
                    const double *b,
                    double *y);

/**
 * Computes the maximum of two floating-point numbers
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_maxf_v(uintptr_t len,
                     const float *a,
                     const float *b,
                     float *y);

/**
 * Computes the maximum of two floating-point numbers
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_max_v(uintptr_t len,
                    const double *a,
                    const double *b,
                    double *y);

/**
 * Clamps a floating-point number between a minimum and maximum scalar value
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_clampf_vs(uintptr_t len,
                        const float *x,
                        float *y,
                        float min,
                        float max);

/**
 * Clamps a floating-point number between a minimum and maximum scalar value
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_clamp_vs(uintptr_t len,
                       const double *x,
                       double *y,
                       double min,
                       double max);

/**
 * Computes the absolute value of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_absf_v(uintptr_t len,
                     const float *x,
                     float *y);

/**
 * Computes the absolute value of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_abs_v(uintptr_t len,
                    const double *x,
                    double *y);

/**
 * Computes the sign of a floating-point number, returning -1.0 for negative values, 1.0 for positive values, and 0.0 for zero
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_signumf_v(uintptr_t len,
                        const float *x,
                        float *y);

/**
 * Computes the sign of a floating-point number, returning -1.0 for negative values, 1.0 for positive values, and 0.0 for zero
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_signum_v(uintptr_t len,
                       const double *x,
                       double *y);

/**
 * Computes (x * a) + b with only one rounding error, yielding a more accurate
 * result than a separate multiplication and addition
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_mul_addf_v(uintptr_t len,
                         const float *x,
                         const float *a,
                         const float *b,
                         float *y);

/**
 * Computes (x * a) + b with only one rounding error, yielding a more accurate
 * result than a separate multiplication and addition
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_mul_add_v(uintptr_t len,
                        const double *x,
                        const double *a,
                        const double *b,
                        double *y);

/**
 * Computes (x * a) - b with only one rounding error, yielding a more accurate
 * result than a separate multiplication and subtraction
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_mul_subf_v(uintptr_t len,
                         const float *x,
                         const float *a,
                         const float *b,
                         float *y);

/**
 * Computes (x * a) - b with only one rounding error, yielding a more accurate
 * result than a separate multiplication and subtraction
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_mul_sub_v(uintptr_t len,
                        const double *x,
                        const double *a,
                        const double *b,
                        double *y);

/**
 * Computes -(x * a) + b with only one rounding error, yielding a more accurate
 * result than a separate negated multiplication and addition
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_nmul_addf_v(uintptr_t len,
                          const float *x,
                          const float *a,
                          const float *b,
                          float *y);

/**
 * Computes -(x * a) + b with only one rounding error, yielding a more accurate
 * result than a separate negated multiplication and addition
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_nmul_add_v(uintptr_t len,
                         const double *x,
                         const double *a,
                         const double *b,
                         double *y);

/**
 * Computes -(x * a) - b with only one rounding error, yielding a more accurate
 * result than a separate negated multiplication and subtraction
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_nmul_subf_v(uintptr_t len,
                          const float *x,
                          const float *a,
                          const float *b,
                          float *y);

/**
 * Computes -(x * a) - b with only one rounding error, yielding a more accurate
 * result than a separate negated multiplication and subtraction
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_nmul_sub_v(uintptr_t len,
                         const double *x,
                         const double *a,
                         const double *b,
                         double *y);

/**
 * Computes the inverse square root, which may vary in accuracy and performance based on the chosen precision policy.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_inverse_sqrtf_v(uintptr_t len,
                              const float *x,
                              float *out);

/**
 * Computes the inverse square root, which may vary in accuracy and performance based on the chosen precision policy.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_inverse_sqrt_v(uintptr_t len,
                             const double *x,
                             double *out);

/**
 * Computes the reciprocal (1/x), which may vary in accuracy and performance based on the chosen precision policy.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_reciprocalf_v(uintptr_t len,
                            const float *x,
                            float *out);

/**
 * Computes the reciprocal (1/x), which may vary in accuracy and performance based on the chosen precision policy.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_reciprocal_v(uintptr_t len,
                           const double *x,
                           double *out);

/**
 * Compute both sine and cosine of the input simultaneously, which will be more efficient than computing them separately.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_sin_cosf_vv(uintptr_t len,
                          const float *x,
                          float *sin,
                          float *cos);

/**
 * Compute both sine and cosine of the input simultaneously, which will be more efficient than computing them separately.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_sin_cos_vv(uintptr_t len,
                         const double *x,
                         double *sin,
                         double *cos);

/**
 * Compute both sine and cosine of the input multiplied by π simultaneously, which will be more efficient than computing them separately.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_sin_cos_pif_vv(uintptr_t len,
                             const float *x,
                             float *sin,
                             float *cos);

/**
 * Compute both sine and cosine of the input multiplied by π simultaneously, which will be more efficient than computing them separately.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_sin_cos_pi_vv(uintptr_t len,
                            const double *x,
                            double *sin,
                            double *cos);

/**
 * Compute both hyperbolic sine and hyperbolic cosine of the input simultaneously, which will be more efficient than computing them separately.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_sinh_coshf_vv(uintptr_t len,
                            const float *x,
                            float *sinh,
                            float *cosh);

/**
 * Compute both hyperbolic sine and hyperbolic cosine of the input simultaneously, which will be more efficient than computing them separately.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_sinh_cosh_vv(uintptr_t len,
                           const double *x,
                           double *sinh,
                           double *cosh);

/**
 * Computes the sine of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_sinf_v(uintptr_t len,
                     const float *x,
                     float *y);

/**
 * Computes the sine of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_sin_v(uintptr_t len,
                    const double *x,
                    double *y);

/**
 * Computes the cosine of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_cosf_v(uintptr_t len,
                     const float *x,
                     float *y);

/**
 * Computes the cosine of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_cos_v(uintptr_t len,
                    const double *x,
                    double *y);

/**
 * Computes the tangent of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_tanf_v(uintptr_t len,
                     const float *x,
                     float *y);

/**
 * Computes the tangent of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_tan_v(uintptr_t len,
                    const double *x,
                    double *y);

/**
 * Computes the sine of the input multiplied by π, which may be more accurate for certain inputs than multiplying the input by π and then taking the sine.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_sin_pif_v(uintptr_t len,
                        const float *x,
                        float *y);

/**
 * Computes the sine of the input multiplied by π, which may be more accurate for certain inputs than multiplying the input by π and then taking the sine.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_sin_pi_v(uintptr_t len,
                       const double *x,
                       double *y);

/**
 * Computes the cosine of the input multiplied by π, which may be more accurate for certain inputs than multiplying the input by π and then taking the cosine.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_cos_pif_v(uintptr_t len,
                        const float *x,
                        float *y);

/**
 * Computes the cosine of the input multiplied by π, which may be more accurate for certain inputs than multiplying the input by π and then taking the cosine.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_cos_pi_v(uintptr_t len,
                       const double *x,
                       double *y);

/**
 * Computes the tangent of the input multiplied by π, which may be more accurate for certain inputs than multiplying the input by π and then taking the tangent.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_tan_pif_v(uintptr_t len,
                        const float *x,
                        float *y);

/**
 * Computes the tangent of the input multiplied by π, which may be more accurate for certain inputs than multiplying the input by π and then taking the tangent.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_tan_pi_v(uintptr_t len,
                       const double *x,
                       double *y);

/**
 * Computes the sinc function, defined as sin(πx)/(πx) for x != 0 and 1 for x = 0
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_sincf_v(uintptr_t len,
                      const float *x,
                      float *y);

/**
 * Computes the sinc function, defined as sin(πx)/(πx) for x != 0 and 1 for x = 0
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_sinc_v(uintptr_t len,
                     const double *x,
                     double *y);

/**
 * Computes the sinc function of the input multiplied by π, defined as sin(π^2 x)/(π^2 x) for x != 0 and 1 for x = 0,
 * which may be more accurate for certain inputs than multiplying the input by π and then taking the sinc.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_sinc_pif_v(uintptr_t len,
                         const float *x,
                         float *y);

/**
 * Computes the sinc function of the input multiplied by π, defined as sin(π^2 x)/(π^2 x) for x != 0 and 1 for x = 0,
 * which may be more accurate for certain inputs than multiplying the input by π and then taking the sinc.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_sinc_pi_v(uintptr_t len,
                        const double *x,
                        double *y);

/**
 * Computes the hyperbolic sine of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_sinhf_v(uintptr_t len,
                      const float *x,
                      float *y);

/**
 * Computes the hyperbolic sine of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_sinh_v(uintptr_t len,
                     const double *x,
                     double *y);

/**
 * Computes the hyperbolic cosine of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_coshf_v(uintptr_t len,
                      const float *x,
                      float *y);

/**
 * Computes the hyperbolic cosine of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_cosh_v(uintptr_t len,
                     const double *x,
                     double *y);

/**
 * Computes the hyperbolic tangent of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_tanhf_v(uintptr_t len,
                      const float *x,
                      float *y);

/**
 * Computes the hyperbolic tangent of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_tanh_v(uintptr_t len,
                     const double *x,
                     double *y);

/**
 * Computes the inverse sine (arcsine) of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_asinf_v(uintptr_t len,
                      const float *y,
                      float *x);

/**
 * Computes the inverse sine (arcsine) of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_asin_v(uintptr_t len,
                     const double *y,
                     double *x);

/**
 * Computes the inverse cosine (arccosine) of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_acosf_v(uintptr_t len,
                      const float *y,
                      float *x);

/**
 * Computes the inverse cosine (arccosine) of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_acos_v(uintptr_t len,
                     const double *y,
                     double *x);

/**
 * Computes the inverse tangent (arctangent) of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_atanf_v(uintptr_t len,
                      const float *y,
                      float *x);

/**
 * Computes the inverse tangent (arctangent) of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_atan_v(uintptr_t len,
                     const double *y,
                     double *x);

/**
 * Computes the inverse hyperbolic sine of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_asinhf_v(uintptr_t len,
                       const float *y,
                       float *x);

/**
 * Computes the inverse hyperbolic sine of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_asinh_v(uintptr_t len,
                      const double *y,
                      double *x);

/**
 * Computes the inverse hyperbolic cosine of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_acoshf_v(uintptr_t len,
                       const float *y,
                       float *x);

/**
 * Computes the inverse hyperbolic cosine of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_acosh_v(uintptr_t len,
                      const double *y,
                      double *x);

/**
 * Computes the inverse hyperbolic tangent of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_atanhf_v(uintptr_t len,
                       const float *y,
                       float *x);

/**
 * Computes the inverse hyperbolic tangent of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_atanh_v(uintptr_t len,
                      const double *y,
                      double *x);

/**
 * Computes the exponential of a floating-point number, which may vary in accuracy and performance based on the chosen precision policy.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_expf_v(uintptr_t len,
                     const float *x,
                     float *y);

/**
 * Computes the exponential of a floating-point number, which may vary in accuracy and performance based on the chosen precision policy.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_exp_v(uintptr_t len,
                    const double *x,
                    double *y);

/**
 * Computes the half-exponential of a floating-point number, defined as exp(x)/2
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_exphf_v(uintptr_t len,
                      const float *x,
                      float *y);

/**
 * Computes the half-exponential of a floating-point number, defined as exp(x)/2
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_exph_v(uintptr_t len,
                     const double *x,
                     double *y);

/**
 * Computes 2 raised to the power of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_exp2f_v(uintptr_t len,
                      const float *x,
                      float *y);

/**
 * Computes 2 raised to the power of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_exp2_v(uintptr_t len,
                     const double *x,
                     double *y);

/**
 * Computes 10 raised to the power of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_exp10f_v(uintptr_t len,
                       const float *x,
                       float *y);

/**
 * Computes 10 raised to the power of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_exp10_v(uintptr_t len,
                      const double *x,
                      double *y);

/**
 * Computes the exponential of a floating-point number minus one, which may be more accurate for small inputs than computing exp(x) - 1 directly.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_exp_m1f_v(uintptr_t len,
                        const float *x,
                        float *y);

/**
 * Computes the exponential of a floating-point number minus one, which may be more accurate for small inputs than computing exp(x) - 1 directly.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_exp_m1_v(uintptr_t len,
                       const double *x,
                       double *y);

/**
 * Computes the natural logarithm of a floating-point number, which may vary in accuracy and performance based on the chosen precision policy.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_lnf_v(uintptr_t len,
                    const float *x,
                    float *y);

/**
 * Computes the natural logarithm of a floating-point number, which may vary in accuracy and performance based on the chosen precision policy.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_ln_v(uintptr_t len,
                   const double *x,
                   double *y);

/**
 * Computes the natural logarithm of one plus a floating-point number, which may be more accurate for small inputs than computing ln(1 + x) directly.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_ln_1pf_v(uintptr_t len,
                       const float *x,
                       float *y);

/**
 * Computes the natural logarithm of one plus a floating-point number, which may be more accurate for small inputs than computing ln(1 + x) directly.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_ln_1p_v(uintptr_t len,
                      const double *x,
                      double *y);

/**
 * Computes the base-2 logarithm of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_log2f_v(uintptr_t len,
                      const float *x,
                      float *y);

/**
 * Computes the base-2 logarithm of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_log2_v(uintptr_t len,
                     const double *x,
                     double *y);

/**
 * Computes the base-10 logarithm of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_log10f_v(uintptr_t len,
                       const float *x,
                       float *y);

/**
 * Computes the base-10 logarithm of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_log10_v(uintptr_t len,
                      const double *x,
                      double *y);

/**
 * Computes the logarithm of a floating-point number with respect to an arbitrary base
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_logf_v(uintptr_t len,
                     const float *x,
                     const float *base,
                     float *y);

/**
 * Computes the logarithm of a floating-point number with respect to an arbitrary base
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_log_v(uintptr_t len,
                    const double *x,
                    const double *base,
                    double *y);

/**
 * Computes the cube root of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_cbrtf_v(uintptr_t len,
                      const float *x,
                      float *y);

/**
 * Computes the cube root of a floating-point number
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_cbrt_v(uintptr_t len,
                     const double *x,
                     double *y);

/**
 * Computes x raised to the power of y, which may vary in accuracy and performance based on the chosen precision policy.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_powff_v(uintptr_t len,
                      const float *x,
                      const float *e,
                      float *y);

/**
 * Computes x raised to the power of y, which may vary in accuracy and performance based on the chosen precision policy.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_powf_v(uintptr_t len,
                     const double *x,
                     const double *e,
                     double *y);

/**
 * Wraps an angle in radians to the range [-π, π)
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_wrap_anglef_v(uintptr_t len,
                            const float *x,
                            float *y);

/**
 * Wraps an angle in radians to the range [-π, π)
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_wrap_angle_v(uintptr_t len,
                           const double *x,
                           double *y);

/**
 * Computes the absolute difference between two angles
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_angle_difff_v(uintptr_t len,
                            const float *a,
                            const float *b,
                            float *d);

/**
 * Computes the absolute difference between two angles
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_angle_diff_v(uintptr_t len,
                           const double *a,
                           const double *b,
                           double *d);

/**
 * Converts an angle from radians to degrees
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_to_degreesf_v(uintptr_t len,
                            const float *x,
                            float *y);

/**
 * Converts an angle from radians to degrees
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_to_degrees_v(uintptr_t len,
                           const double *x,
                           double *y);

/**
 * Converts an angle from degrees to radians
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_to_radiansf_v(uintptr_t len,
                            const float *x,
                            float *y);

/**
 * Converts an angle from degrees to radians
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_to_radians_v(uintptr_t len,
                           const double *x,
                           double *y);

/**
 * Computes the angle (in radians) between the positive x-axis and the point (x, y), using the signs of both arguments to determine the correct quadrant of the result.
 *
 * This may vary in accuracy and performance based on the chosen precision policy.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_atan2f_v(uintptr_t len,
                       const float *y,
                       const float *x,
                       float *t);

/**
 * Computes the angle (in radians) between the positive x-axis and the point (x, y), using the signs of both arguments to determine the correct quadrant of the result.
 *
 * This may vary in accuracy and performance based on the chosen precision policy.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_atan2_v(uintptr_t len,
                      const double *y,
                      const double *x,
                      double *t);

/**
 * Performs linear interpolation between values a and b using t, where t is typically in the range [0, 1].
 *
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
 * Performs linear interpolation between values a and b using t, where t is typically in the range [0, 1].
 *
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
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_hypotf_v(uintptr_t len,
                       const float *x,
                       const float *y,
                       float *out);

/**
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_hypot_v(uintptr_t len,
                      const double *x,
                      const double *y,
                      double *out);

/**
 * Computes the error function, which may vary in accuracy and performance based on the chosen precision policy.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_erff_v(uintptr_t len,
                     const float *x,
                     float *y);

/**
 * Computes the error function, which may vary in accuracy and performance based on the chosen precision policy.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_erf_v(uintptr_t len,
                    const double *x,
                    double *y);

/**
 * Computes the complementary error function, which may vary in accuracy and performance based on the chosen precision policy.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_erfcf_v(uintptr_t len,
                      const float *x,
                      float *y);

/**
 * Computes the complementary error function, which may vary in accuracy and performance based on the chosen precision policy.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_erfc_v(uintptr_t len,
                     const double *x,
                     double *y);

/**
 * Computes the inverse error function, which may vary in accuracy and performance based on the chosen precision policy.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_erfinvf_v(uintptr_t len,
                        const float *y,
                        float *x);

/**
 * Computes the inverse error function, which may vary in accuracy and performance based on the chosen precision policy.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_erfinv_v(uintptr_t len,
                       const double *y,
                       double *x);

/**
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_tgammaf_v(uintptr_t len,
                        const float *x,
                        float *y);

/**
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_tgamma_v(uintptr_t len,
                       const double *x,
                       double *y);

/**
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_lgammaf_v(uintptr_t len,
                        const float *x,
                        float *y);

/**
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_lgamma_v(uintptr_t len,
                       const double *x,
                       double *y);

/**
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_betaf_v(uintptr_t len,
                      const float *x,
                      const float *y,
                      float *z);

/**
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_beta_v(uintptr_t len,
                     const double *x,
                     const double *y,
                     double *z);

/**
 * 3rd-order smoothstep interpolation function
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_smoothstepf_v(uintptr_t len,
                            const float *x,
                            float *y);

/**
 * 3rd-order smoothstep interpolation function
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_smoothstep_v(uintptr_t len,
                           const double *x,
                           double *y);

/**
 * Inverse of the 3rd-order smoothstep function
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_inverse_smoothstepf_v(uintptr_t len,
                                    const float *y,
                                    float *x);

/**
 * Inverse of the 3rd-order smoothstep function
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_inverse_smoothstep_v(uintptr_t len,
                                   const double *y,
                                   double *x);

/**
 * 5th-order smoothstep interpolation function
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_smootherstepf_v(uintptr_t len,
                              const float *x,
                              float *y);

/**
 * 5th-order smoothstep interpolation function
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_smootherstep_v(uintptr_t len,
                             const double *x,
                             double *y);

/**
 * Inverse of the 5th-order smoothstep function
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_inverse_smootherstepf_v(uintptr_t len,
                                      const float *y,
                                      float *x);

/**
 * Inverse of the 5th-order smoothstep function
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_inverse_smootherstep_v(uintptr_t len,
                                     const double *y,
                                     double *x);

/**
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_smooth_interpolatorf_v(uintptr_t len,
                                     const float *x,
                                     float *y,
                                     float k);

/**
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_smooth_interpolator_v(uintptr_t len,
                                    const double *x,
                                    double *y,
                                    double k);

/**
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_smooth_interpolator_inversef_v(uintptr_t len,
                                             const float *y,
                                             float *x,
                                             float k);

/**
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_smooth_interpolator_inverse_v(uintptr_t len,
                                            const double *y,
                                            double *x,
                                            double k);

/**
 * Step function that returns 0.0 if x < edge and 1.0 if x >= edge
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_stepf_v(uintptr_t len,
                      const float *x,
                      float *y,
                      float edge);

/**
 * Step function that returns 0.0 if x < edge and 1.0 if x >= edge
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_step_v(uintptr_t len,
                     const double *x,
                     double *y,
                     double edge);

/**
 * Linear interpolation between scalars a and b by x, where x is in the range [0, 1]
 *
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
 * Linear interpolation between scalars a and b by x, where x is in the range [0, 1]
 *
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
 * Raises x to the power of exp, where exp is an integer
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
 */
THERMITE_API
void thermite_powif_vs(uintptr_t len,
                       const float *x,
                       float *y,
                       int32_t exp);

/**
 * Raises x to the power of exp, where exp is an integer
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_powi_vs(uintptr_t len,
                      const double *x,
                      double *y,
                      int32_t exp);

/**
 * Computes the Gaussian function with amplitude `a` and standard deviation `c`, defined as `a * exp(-0.5 * (self / c)^2)`.
 *
 * The position `b` is assumed to be zero. For a non-zero position, use `self - b` as the input.
 *
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
 * Computes the Gaussian function with amplitude `a` and standard deviation `c`, defined as `a * exp(-0.5 * (self / c)^2)`.
 *
 * The position `b` is assumed to be zero. For a non-zero position, use `self - b` as the input.
 *
 * # Safety
 * The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
 */
THERMITE_API
void thermite_gaussian_vs(uintptr_t len,
                          const double *x,
                          double *y,
                          double a,
                          double c);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  /* THERMITE_H */
