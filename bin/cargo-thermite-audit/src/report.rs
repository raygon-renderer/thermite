//! Human-readable report for one audited module.

use crate::AuditOpts;
use crate::ir::{Module, short_name};
use crate::rules::{Finding, Report, backend_of, census, has_all};

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

/// Tally a report without printing anything.
pub fn verdict(r: &Report, opts: &AuditOpts) -> Verdict {
    let mut v = Verdict::default();
    for f in &r.arch {
        if f.allowed {
            v.allowed += 1;
        } else if f.transfer_only && !opts.strict {
            v.warnings += 1;
        } else {
            v.live += 1;
        }
    }
    for f in &r.featureless {
        if f.allowed { v.allowed += 1 } else { v.live += 1 }
    }
    v
}

/// Print the report for one module under `title`. With `full` false, a clean
/// module gets a single line (the driver's mode for targets that pass).
pub fn print(title: &str, module: &Module, r: &Report, opts: &AuditOpts, full: bool) -> Verdict {
    let v = verdict(r, opts);
    if !full && !v.failed() && !opts.verbose {
        println!("== {title}: {}", v.line());
        return v;
    }

    println!(
        "== {title} ({} functions, {} attribute groups)",
        module.fns.len(),
        module.attrs.len()
    );
    for b in &r.vacuous {
        println!(
            "WARNING: every attribute group already enables the {b} features (target-cpu=native?); the {b} audit is vacuous"
        );
    }
    if r.v4_is_v3 {
        println!("note: no AVX-512 in this build; X86V4 names are audited as x86_v3");
    }

    print_rule_a(module, r, opts);
    print_rule_b(module, r, opts);
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
            println!("  {}{} (out-of-line copy)", tag(f), f.f.name);
            println!("      features: {feats}   called from {callers} site(s)");
            continue;
        }
        println!("  {}{}", tag(f), f.f.name);
        println!("      features: {feats}   call sites: {}", f.f.arch_calls);

        let mut top: Vec<_> = f.f.callees.iter().collect();
        top.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        let shown: Vec<_> = top.iter().take(6).map(|(n, c)| format!("{n} x{c}")).collect();
        println!("      top: {}", shown.join(", "));

        // Who calls this featureless function: the dispatch boundary it escaped.
        let mut callers: Vec<_> = module
            .fns
            .iter()
            .filter(|g| g.calls.contains_key(&f.f.name))
            .map(|g| &g.name)
            .collect();
        callers.sort_unstable();
        if !callers.is_empty() {
            let shown: Vec<_> = callers.iter().take(3).map(|n| truncate(n, 110)).collect();
            let more = if callers.len() > 3 {
                format!(" (+{} more)", callers.len() - 3)
            } else {
                String::new()
            };
            println!("      called from: {}{more}", shown.join(" | "));
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
        println!("  {}{}", tag(f), f.f.name);
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

fn print_census(module: &Module, opts: &AuditOpts) {
    println!("\nCensus (functions naming a backend / with its features / doing SIMD work):");
    for c in census(module) {
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
        let Some(b) = backend_of(&f.name) else { continue };
        let feats = module.features(f);
        if !b.required.is_empty() && !has_all(feats, b.required) && !f.does_simd_work() {
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

fn tag(f: &Finding) -> &'static str {
    if f.allowed { "[allowed] " } else { "" }
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_owned()
    } else {
        format!("{}...", s.chars().take(n).collect::<String>())
    }
}

fn fmt_feats(feats: &[String]) -> String {
    if feats.is_empty() {
        return "<none>".into();
    }
    // Baseline SSE noise is not informative. Show the interesting ones.
    let interesting: Vec<_> = feats
        .iter()
        .filter(|f| !matches!(f.as_str(), "sse" | "sse2" | "fxsr" | "x87" | "cx8" | "cmov" | "mmx"))
        .cloned()
        .collect();
    if interesting.is_empty() {
        "baseline".into()
    } else {
        interesting.join(",")
    }
}
