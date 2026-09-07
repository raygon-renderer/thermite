//! Human-readable report for one audited module.

use std::borrow::Cow;
use std::fmt::Write as _;

use crate::AuditOpts;
use crate::ir::{Module, short_name};
use crate::rules::{Backend, Report};

/// Tallies for the verdict line.
#[derive(Default, Clone, Copy)]
pub struct Verdict {
    pub live: u32,
    pub warnings: u32,
    pub allowed: u32,
}

impl Verdict {
    pub fn failed(&self) -> bool {
        self.live > 0
    }

    /// One-line form for summaries.
    pub fn line(&self) -> String {
        let mut s = if self.live > 0 {
            format!("FAIL: {} finding(s)", self.live)
        } else {
            "OK".to_owned()
        };
        if self.warnings > 0 {
            s.push_str(&format!("  ({} transfer-only warning(s))", self.warnings));
        }
        s
    }
}

impl Report<'_> {
    /// Tally a report without printing anything.
    pub fn verdict(&self, opts: &AuditOpts) -> Verdict {
        let mut v = Verdict::default();
        for f in &self.arch {
            if f.allowed {
                v.allowed += 1;
            } else if f.transfer_only && !opts.strict {
                v.warnings += 1;
            } else {
                v.live += 1;
            }
        }
        for f in &self.featureless {
            if f.allowed { v.allowed += 1 } else { v.live += 1 }
        }
        // Rule C has no allowlist: the emulation is gated out at compile time,
        // so a surviving call is always a gating bug.
        v.live += self.fma.len() as u32;
        v
    }

    /// Print the report for one module under `title`. With `full` false, a clean
    /// module gets a single line (the driver's mode for targets that pass).
    pub fn print(&self, title: &str, module: &Module, opts: &AuditOpts, full: bool) -> Verdict {
        let v = self.verdict(opts);
        if !full && !v.failed() && !opts.verbose {
            println!("== {title}: {}", v.line());
            return v;
        }

        println!(
            "== {title} ({} functions, {} attribute groups)",
            module.fns.len(),
            module.attrs.len()
        );
        for b in &self.vacuous {
            println!(
                "WARNING: every attribute group already enables the {b} features (target-cpu=native?); the {b} audit is vacuous"
            );
        }
        if self.v4_is_v3 {
            println!("note: no AVX-512 in this build; X86V4 names are audited as x86_v3");
        }

        print_rule_a(module, self, opts);
        print_rule_b(module, self, opts);
        print_rule_c(self);
        print_census(module, opts);

        println!();
        if v.allowed > 0 {
            println!("allowlisted: {} (use --all to list)", v.allowed);
        }
        if v.warnings > 0 {
            println!("warnings: {} transfer-only (use --strict to fail on them)", v.warnings);
        }
        if v.live > 0 {
            println!("FAIL: {} finding(s)", v.live);
        } else {
            println!("OK: no findings");
        }
        println!();
        v
    }
}

fn print_rule_a(module: &Module, r: &Report, opts: &AuditOpts) {
    println!("\nRule A: out-of-line core_arch intrinsics");
    if r.arch.is_empty() {
        println!("  none");
        return;
    }

    // The surviving copies of the intrinsics themselves are a consequence of
    // the callers above them, so they fold into one line unless --all.
    let copies: Vec<_> = r.arch.iter().filter(|f| f.f.arch_calls == 0 && !f.allowed).collect();
    if !copies.is_empty() && !opts.show_all {
        let mut names: Vec<_> = copies.iter().map(|f| short_name(&f.f.name)).collect();
        names.sort_unstable();
        names.dedup();
        println!(
            "  {} out-of-line intrinsic copies ({} distinct): {}",
            copies.len(),
            names.len(),
            names.join(", ")
        );
    }

    let mut in_warnings = false;
    for f in &r.arch {
        if f.allowed && !opts.show_all {
            continue;
        }
        if f.f.arch_calls == 0 && !opts.show_all {
            continue;
        }
        let is_warning = f.transfer_only && !opts.strict;
        if is_warning && !in_warnings {
            in_warnings = true;
            println!("  -- transfer-only (load/store/extract out of line): warnings, fail with --strict --");
        }
        let feats = fmt_feats(module.features(f.f));
        if f.f.arch_calls == 0 {
            // A surviving copy of the intrinsic itself: it makes no core_arch
            // calls, so report who calls it instead.
            let short = short_name(&f.f.name);
            let callers: u32 = module.fns.iter().filter_map(|g| g.callees.get(&short)).sum();
            println!("  {}{} (out-of-line copy)", f.tag(), f.f.name);
            println!("      features: {feats}   called from {callers} site(s)");
            continue;
        }
        println!("  {}{}", f.tag(), f.f.name);
        println!("      features: {feats}   call sites: {}", f.f.arch_calls);

        let mut top: Vec<_> = f.f.callees.iter().collect();
        top.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        let mut shown = String::new();
        for (n, c) in top.iter().take(6) {
            if !shown.is_empty() {
                shown.push_str(", ");
            }
            let _ = write!(shown, "{n} x{c}");
        }
        println!("      top: {shown}");

        // Who calls this featureless function: the dispatch boundary it escaped.
        let mut callers: Vec<_> = module
            .fns
            .iter()
            .filter(|g| g.calls.contains_key(&f.f.name))
            .map(|g| &g.name)
            .collect();
        callers.sort_unstable();
        if !callers.is_empty() {
            let mut shown = String::new();
            for n in callers.iter().take(3) {
                if !shown.is_empty() {
                    shown.push_str(" | ");
                }
                shown.push_str(&truncate(n, 110));
            }
            let more = if callers.len() > 3 {
                format!(" (+{} more)", callers.len() - 3)
            } else {
                String::new()
            };
            println!("      called from: {shown}{more}");
        }
    }
}

fn print_rule_b(module: &Module, r: &Report, opts: &AuditOpts) {
    println!("\nRule B: featureless SIMD work");
    if r.featureless.is_empty() {
        println!("  none");
        return;
    }
    for f in &r.featureless {
        if f.allowed && !opts.show_all {
            continue;
        }
        println!("  {}{}", f.tag(), f.f.name);
        println!(
            "      backend: {}   missing: {}   simd ops: {}   inline asm: {}   widest: {} bits   features: {}",
            f.backend,
            f.missing.join(","),
            f.f.simd_ops,
            f.f.asm_calls,
            f.f.max_width,
            fmt_feats(module.features(f.f))
        );
    }
}

/// Rule C: software FMA reached from a backend that fuses in hardware.
fn print_rule_c(r: &Report) {
    println!(
        "
Rule C: emulated FMA on a hardware-FMA backend"
    );
    if r.fma.is_empty() {
        println!("  none");
        return;
    }
    for x in &r.fma {
        println!("  {}", x.f.name);
        println!(
            "      backend: {}   call sites: {}   emulation: {}",
            x.backend,
            x.count,
            short_name(x.callee)
        );
    }
}

fn print_census(module: &Module, opts: &AuditOpts) {
    println!("\nCensus (functions naming a backend / with its features / doing SIMD work):");
    for c in module.census() {
        println!(
            "  {:8} named: {:5}   featured: {:5}   simd: {:5}",
            c.label, c.named, c.featured, c.simd
        );
    }
    if !opts.verbose {
        return;
    }

    println!("\nBackend-named functions without their features (passed Rule B):");
    for f in &module.fns {
        let Some(b) = Backend::of(&f.name) else { continue };
        let feats = module.features(f);
        if !b.required.is_empty() && !b.satisfied_by(feats) && !f.does_simd_work() {
            println!("  {}   features: {}", f.name, fmt_feats(feats));
        }
    }

    let mut groups: Vec<_> = module.attrs.iter().collect();
    groups.sort_by_key(|(id, _)| **id);
    println!("\nAttribute groups:");
    for (id, feats) in groups {
        println!("  #{id}: {}", fmt_feats(feats));
    }
}

fn truncate(s: &str, n: usize) -> Cow<'_, str> {
    if s.chars().count() <= n {
        Cow::Borrowed(s)
    } else {
        let mut out: String = s.chars().take(n).collect();
        out.push_str("...");
        Cow::Owned(out)
    }
}

fn fmt_feats(feats: &[String]) -> Cow<'static, str> {
    if feats.is_empty() {
        return Cow::Borrowed("<none>");
    }
    // Baseline SSE noise is not informative. Show the interesting ones.
    let mut out = String::new();
    for f in feats.iter().map(String::as_str) {
        if matches!(f, "sse" | "sse2" | "fxsr" | "x87" | "cx8" | "cmov" | "mmx") {
            continue;
        }
        if !out.is_empty() {
            out.push(',');
        }
        out.push_str(f);
    }
    if out.is_empty() {
        Cow::Borrowed("baseline")
    } else {
        Cow::Owned(out)
    }
}
