//! Rule A (out-of-line `core_arch` intrinsics), Rule B (featureless SIMD
//! work), the per-backend required-feature table, and the allowlist.

use std::cmp::Reverse;

use crate::ir::{FnRecord, Module, is_core_arch_path, is_detection_intrinsic, is_transfer_intrinsic, short_name};

/// A dispatch backend as it appears in demangled names.
///
/// Mirrors `BACKENDS` in `crates/thermite-macros/src/dispatch.rs` BY HAND.
/// Only the features every build of that backend enables are required. The
/// optional ones (`f16c`, `pclmulqdq`) are not, so an audit stays meaningful
/// under either default.
pub struct Backend {
    pub label: &'static str,
    pub markers: &'static [&'static str],
    pub required: &'static [&'static str],
    /// Hardware fused multiply-add, so reaching the software emulation on this
    /// backend is a gating bug. False for the two SSE rungs, and for wasm,
    /// where `HAS_NATIVE_FMA` is `Indeterminate` and the emulation is the
    /// documented fallback.
    pub native_fma: bool,
}

/// Highest tier first: a name mentioning several backends gets the strictest.
pub const BACKENDS: &[Backend] = &[
    Backend {
        label: "x86_v4",
        native_fma: true,
        markers: &["x86_v4::", "X86V4", "x86v4"],
        required: &["avx512f", "avx512vl", "avx512bw", "avx512dq"],
    },
    Backend {
        label: "x86_v3",
        native_fma: true,
        markers: &["x86_v3::", "X86V3", "x86v3"],
        required: &["avx2", "fma"],
    },
    Backend {
        label: "x86_v2",
        native_fma: false,
        markers: &["x86_v2::", "X86V2", "x86v2"],
        required: &["sse4.2"],
    },
    // sse2 is the x86_64 baseline: nothing to audit, so require nothing.
    Backend {
        label: "x86_v1",
        native_fma: false,
        markers: &["x86_v1::", "X86V1", "x86v1"],
        required: &[],
    },
    Backend {
        label: "neon",
        native_fma: true,
        markers: &["backend::neon::", "NEON", "__dispatch_neon"],
        required: &[],
    },
    Backend {
        label: "wasm",
        native_fma: false,
        markers: &["backend::wasm::", "WASM32", "__dispatch_wasm32"],
        required: &[],
    },
];

/// Functions that are featureless on purpose. Each entry cites why in
/// notes/thermite-audit/PLAN.md ("Allowlist"). Add nothing here without one.
pub const BUILTIN_ALLOW: &[&str] = &[
    // (_xgetbv / cpuid callers such as std_detect, chacha20, libm need no entry:
    // detection intrinsics are excluded by callee name in ir.rs.)
    "detect_once",                // one-time cpuid detection, runs before any ISA is known
    "isa::x86",                   // cpuid / xgetbv readers
    "thermite::cold",             // likely/unlikely hint, empty body
    "const_splat",                // const-eval entry point, a runtime survivor is a constant ret
    "splat_const",                // same
    "_fmadd_emulated",            // wasm: deliberately outlined, simd128 is baseline there
    "element::float::arch::soft", // outlined scalar math under outline_scalar_math
];

pub struct Finding<'a> {
    pub f: &'a FnRecord,
    pub backend: &'static str,
    pub missing: Vec<&'static str>,
    pub allowed: bool,
    /// Rule A only: every out-of-line intrinsic is data movement. A warning
    /// unless `--strict`.
    pub transfer_only: bool,
}

impl Finding<'_> {
    pub fn tag(&self) -> &'static str {
        if self.allowed {
            "[allowed]"
        } else if self.transfer_only {
            "[transfer-only]"
        } else {
            ""
        }
    }
}

pub struct Report<'a> {
    /// Rule A: functions with surviving `core_arch` calls.
    pub arch: Vec<Finding<'a>>,
    /// Rule B: featureless functions doing SIMD work.
    pub featureless: Vec<Finding<'a>>,
    /// Backends whose audit is vacuous because the baseline already has them.
    pub vacuous: Vec<&'static str>,
    /// Rule C: the software FMA reached from a backend that has the hardware.
    pub fma: Vec<FmaFinding<'a>>,
    /// True when no group anywhere enables AVX-512, so `X86V4` names are the
    /// v3 backend in disguise (thermite maps V4 hardware onto v3 then).
    pub v4_is_v3: bool,
}

/// One call into the round-to-odd FMA emulation from a backend that fuses in
/// hardware. `fmadd_ro` itself is `#[inline(always)]` and leaves no trace, but
/// its cold helper `fmadd_ro_rescue` is `#[dispatch]`-outlined on purpose, so
/// the call survives with the register type in its generic argument. That type
/// is what names the backend here, not the calling function.
pub struct FmaFinding<'a> {
    pub f: &'a FnRecord,
    pub backend: &'static str,
    pub callee: &'a str,
    pub count: u32,
}

/// Per-backend census, so an empty Rule B can be told apart from a parser that
/// saw nothing: functions naming the backend, how many carry its features,
/// how many do SIMD work.
pub struct Census {
    pub label: &'static str,
    pub named: u32,
    pub featured: u32,
    pub simd: u32,
}

impl Module {
    /// Per-backend tally over every function in the module.
    pub fn census(&self) -> Vec<Census> {
        let mut out: Vec<_> = BACKENDS
            .iter()
            .map(|b| Census {
                label: b.label,
                named: 0,
                featured: 0,
                simd: 0,
            })
            .collect();
        for f in &self.fns {
            let Some(b) = Backend::of(&f.name) else { continue };
            let c = out.iter_mut().find(|c| c.label == b.label).expect("backend in table");
            c.named += 1;
            if b.satisfied_by(self.features(f)) {
                c.featured += 1;
            }
            if f.does_simd_work() {
                c.simd += 1;
            }
        }
        out.retain(|c| c.named > 0);
        out
    }
}

impl Backend {
    /// The backend a demangled name belongs to, strictest tier first.
    pub fn of(name: &str) -> Option<&'static Backend> {
        BACKENDS.iter().find(|b| b.markers.iter().any(|m| name.contains(m)))
    }

    fn labelled(label: &str) -> &'static Backend {
        BACKENDS.iter().find(|b| b.label == label).expect("backend in table")
    }

    /// True when an attribute group's features cover everything this backend
    /// requires. Vacuously true for the backends that require nothing.
    pub fn satisfied_by(&self, feats: &[String]) -> bool {
        has_all(feats, self.required)
    }
}

/// True when `feats` (an attribute group's enabled features) contains every
/// feature in `required`.
fn has_all(feats: &[String], required: &[&str]) -> bool {
    required.iter().all(|r| feats.iter().any(|f| f == r))
}

impl<'a> Report<'a> {
    /// Run both rules over a parsed module.
    pub fn build(module: &'a Module, extra_allow: &[String]) -> Report<'a> {
        let allowed = |name: &str| {
            BUILTIN_ALLOW.iter().any(|a| name.contains(a)) || extra_allow.iter().any(|a| name.contains(a.as_str()))
        };

        // Baseline sanity: a backend whose features every group already has cannot
        // be audited (target-cpu=native). Without any AVX-512 group, `X86V4`
        // names are the v3 backend, which is where thermite maps V4 hardware.
        let all_groups_have =
            |required: &[&str]| !module.attrs.is_empty() && module.attrs.values().all(|feats| has_all(feats, required));
        let v4_is_v3 = !module.attrs.values().any(|feats| has_all(feats, &["avx512f"]));
        let vacuous = BACKENDS
            .iter()
            .filter(|b| !b.required.is_empty() && all_groups_have(b.required))
            .map(|b| b.label)
            .collect();

        let mut report = Report {
            arch: Vec::new(),
            featureless: Vec::new(),
            fma: Vec::new(),
            vacuous,
            v4_is_v3,
        };

        for f in &module.fns {
            let feats = module.features(f);
            let is_allowed = allowed(&f.name);

            // Rule C: the emulation is gated out at compile time wherever
            // `HAS_NATIVE_FMA` is `True`, so a surviving call from an FMA
            // backend means the gate was wrong.
            for (callee, count) in &f.calls {
                if !callee.contains("fmadd_ro") && !callee.contains("fmadd_widen_ro") {
                    continue;
                }
                if let Some(b) = Backend::of(callee)
                    && b.native_fma
                {
                    report.fma.push(FmaFinding {
                        f,
                        backend: b.label,
                        callee,
                        count: *count,
                    });
                }
            }

            // Rule A: a define OF a core_arch fn is an out-of-line copy someone forced.
            let is_copy = is_core_arch_path(&f.name);
            if is_copy && is_detection_intrinsic(&short_name(&f.name)) {
                continue;
            }
            if f.arch_calls > 0 || is_copy {
                let transfer_only = if is_copy {
                    is_transfer_intrinsic(&short_name(&f.name))
                } else {
                    f.compute_calls == 0
                };
                report.arch.push(Finding {
                    f,
                    backend: "",
                    missing: Vec::new(),
                    allowed: is_allowed,
                    transfer_only,
                });
            }

            // Rule B.
            let Some(mut backend) = Backend::of(&f.name) else {
                continue;
            };
            if backend.label == "x86_v4" && v4_is_v3 {
                backend = Backend::labelled("x86_v3");
            }
            let missing: Vec<_> = backend
                .required
                .iter()
                .copied()
                .filter(|r| !has_all(feats, &[r]))
                .collect();
            if !missing.is_empty() && f.does_simd_work() {
                report.featureless.push(Finding {
                    f,
                    backend: backend.label,
                    missing,
                    allowed: is_allowed,
                    transfer_only: false,
                });
            }
        }

        // Worst first: compute escapes before transfer-only, then by volume.
        report
            .arch
            .sort_by_key(|x| Reverse((x.f.compute_calls, x.f.arch_calls)));
        report.featureless.sort_by_key(|x| Reverse(x.f.simd_ops));
        report.fma.sort_by_key(|x| Reverse(x.count));
        report
    }
}
