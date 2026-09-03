// Shared by the packet-level tests: the widest register-width backend the target has, so they
// run on every ISA instead of only on x86. Every backend declares the full alias set
// (`f64x4`, `f32x8`, ...), composing sub-registers where the ISA is narrower, so the test
// bodies need no per-target lane counts.
//
// Pulled in with `include!("common/wide.rs")`. `Wide` is the `Simd` type; the glob brings in
// that backend's aliases plus the thermite prelude.

core::cfg_select! {
    any(target_arch = "x86", target_arch = "x86_64") => {
        #[allow(unused_imports)]
        use thermite::backend::x86_v3::{X86V3 as Wide, prelude::*};
    }
    target_arch = "aarch64" => {
        #[allow(unused_imports)]
        use thermite::backend::neon::{Neon as Wide, prelude::*};
    }
    all(feature = "wasm", any(target_arch = "wasm32", target_arch = "wasm64")) => {
        #[allow(unused_imports)]
        use thermite::backend::wasm::{Wasm as Wide, prelude::*};
    }
    _ => {
        #[allow(unused_imports)]
        use thermite::backend::scalar::{Scalar as Wide, prelude::*};
    }
}
