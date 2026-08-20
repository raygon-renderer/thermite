//! Architecture-specific scalar float operations.
//!
//! `libm`'s scalar routines are either opaque inline `asm!` blocks or out-of-line
//! soft-float calls. Neither constant-folds, and the `asm!` ones are pinned to the
//! legacy SSE encoding even when inlined into an AVX body (a measured ~30x hit from
//! the SSE/AVX transition). The rungs below lower to a single instruction that stays
//! transparent to LLVM wherever that can be established.
//!
//! Function names mirror `libm`'s (`sqrt`/`sqrtf`, ...) so `impl_float_element!` can
//! substitute one for the other.
//!
//! # The ladder
//!
//! | rung | when | dispatch-aware |
//! |---|---|---|
//! | [`nightly`] - `core::intrinsics` | `nightly` feature | **yes** |
//! | [`x86`] / [`neon`] / [`wasm`] - explicit intrinsics | baseline has the instruction | no |
//! | [`soft`] - std methods, else `libm` | everything else | only with `std` |
//!
//! The middle rungs read `cfg!(target_feature = ...)`, which sees the target spec
//! defaults plus `-C target-cpu` / `-C target-feature` but **not** a function-level
//! `#[target_feature]`. So a `#[thermite::dispatch]` trampoline does not unlock a
//! higher rung for scalar element ops inlined into its body, and on a stable `no_std`
//! build those ops stay at whatever the crate was compiled for. Prefer
//! `V::splat(x).sqrt()` over `FloatElement::sqrt(x)` inside a kernel regardless.
//!
//! The generic LLVM intrinsics behind the `nightly` and `std` rungs do not have that
//! limitation (LLVM selects them with the enclosing function's features), which is why
//! `nightly` sits above the explicit-intrinsic rungs and `std` backs them from below.
#![allow(dead_code, reason = "which rung is reachable depends on the target and features")]

use core::cfg_select;

mod soft;

cfg_select! {
    feature = "nightly" => {
        mod nightly;
        pub use nightly::*;
    }
    all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "sse2") => {
        mod x86;
        pub use x86::*;
    }
    all(target_arch = "aarch64", target_feature = "neon") => {
        mod neon;
        pub use neon::*;
    }
    all(feature = "wasm", any(target_arch = "wasm32", target_arch = "wasm64")) => {
        mod wasm;
        pub use wasm::*;
    }
    _ => {
        pub use soft::*;
    }
}

/// Whether scalar `mul_add` is a single hardware instruction.
///
/// Deliberately reads **baseline** target features only, on every rung. The `nightly`
/// and `std` rungs lower `mul_add` through `llvm.fma`, which becomes `vfmadd*` inside a
/// feature-enabled function but an out-of-line soft-float call outside one; a `const`
/// cannot tell which it will be. Thermite's kernels treat `HAS_TRUE_FMA == true` as a
/// promise that the FMA is *fast*, so the promise is only made where the baseline keeps
/// it unconditionally.
///
/// AArch64 is unconditionally true: AdvSIMD, and with it `fmadd`, is mandatory there.
pub const HAS_TRUE_FMA: bool = cfg!(any(
    all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "fma"),
    all(target_arch = "aarch64", target_feature = "neon"),
));
