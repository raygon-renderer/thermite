# Changelog

## 0.2.1 (2026-08-12)

### Fixed

- `cbrt` classified finite f64 above ~1.4e306 as non-finite (f32 bit pattern against the f64 high word), returned NaN for `f32::INFINITY`, and overflowed in the top binade. Denormals under `Preserve` now take the extended-precision path, which the fast form left with one significant bit.
- `ln_1p` on f32 split the exponent of `x` instead of `1 + x`. `ln`/`ln_1p` gave `-inf` for subnormals under `Preserve`.
- `ldexp(0.0, 300)` gave infinity, `ldexp(1.0, i32::MAX)` wrapped, and `Preserve` could not underflow `f32::MAX` to zero.
- `frexp` skipped renormalization off `Preserve`, so `frexp(1e-40f32)` broke the `0.5 <= |frac| < 1` postcondition. The fixup sits behind an `unlikely` branch on flushing policies (2.2 cyc/iter against 4.0 unconditional, crossover at ~1 subnormal element in 90); `AvoidBranching` forces it straight-line.
- Complex powers were NaN at the origin (`0 * inf` in `d * ln r`). `(0+0i)^2` and `0^0` are now 0 and 1.
- Avoided LLVM miscompilation of saturating downcasts on WASM.
- Improved CI pipeline to test x86, wasm, and aarch64.
- Improved elliptic integral accuracy and boundary handling.

### Added

- Payne-Hanek range reduction for f64 trig at `Best` and above. Without true FMA the Cody-Waite handoff moved from 1e13 to 1e7, where the 30-bit `dp1` stops being exact.
- `BitwiseRegister::HAS_NATIVE_TERNLOG`, forwarded to `BitwiseVector` and `GenericMask`. `ldexp` picks its lowering off it: 4 ternlogs on AVX-512, blends elsewhere (3.8 cyc/iter against 5.8, znver3).
- Saturating float-to-int casts, full cross-width matrix on every backend.
- Regression tests for all of the above: `cbrt_range`, `denormal_math`, `frexp_denormals`, `ldexp_ternlog`, `math_edge_cases`, `special_vs_libm`, `strict_cast`.

### Changed

- **BREAKING**: `SaturatingCastRegister`/`SaturatingCastVector` merged into `CastRegister`/`CastVector` as a `saturating_cast_from` method. It and `cast_from` default to each other, so a backend provides whichever ones it has a distinct lowering for, and `saturating_cast` resolves for every pair `cast` does (falling through where no saturating form exists, which for sign-changing integer casts means it wraps).
- `strict_ieee754` redirects float-source `cast` to the saturating lowering from one place, the `Vector<R>` impl of `CastVector`, so it now reaches every pair instead of only the same-width ones. Integer casts are untouched: `as` wraps for int-to-int, so redirecting them would clamp where the language wraps.
- AVX2 saturating narrows split to 128 bits before packing. LLVM canonicalizes the 256-bit `vpackssdw`/`vpackusdw` intrinsics into clamp plus truncate and never folds the clamps back out. `i32x8 -> i16x8` went from 9 instructions to 4, `f32x8 -> i16x8` from 16 to 12.
- `DefaultPolicy` is `Precision` under `strict_ieee754`.
- `ldexp` handles its full domain by default. A caller whose exponent is known in range (anything from `frexp`) can drop the checks with `ldexp_p::<CheckOverflow<P, false>>`.
