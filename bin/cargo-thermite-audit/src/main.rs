//! cargo-thermite-audit: find Thermite kernels compiled without their
//! backend's target features (the missing `#[thermite::dispatch]` /
//! `#[inline(always)]` bug) by scanning LLVM IR. See notes/thermite-audit/PLAN.md.
//!
//! Two front doors:
//!
//! - `cargo thermite-audit [cargo selection] [audit flags]` builds each selected
//!   target with `--emit=llvm-ir` and audits the result (the `driver` module).
//! - `cargo thermite-audit --ll FILE.ll...` audits IR someone else emitted.

mod driver;
mod ir;
mod report;
mod rules;

use crate::rules::Report;
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "\
usage: cargo thermite-audit [SELECTION] [OPTIONS]
       cargo thermite-audit --ll FILE.ll... [OPTIONS]

Selection (cargo-style; default is every lib/bin/test/bench/example of the
selected packages, in the dev profile):
    -p, --package NAME     package to audit (repeatable; default: the current package)
    --workspace            every workspace member
    --lib / --bins / --tests / --benches / --examples / --all-targets
    --bin NAME / --test NAME / --bench NAME / --example NAME   (repeatable)
    --features LIST        (repeatable), --all-features, --no-default-features
    --release / --profile NAME
    --target TRIPLE        cross-compile (the IR is audited, nothing runs)

Options:
    --allow SUBSTR     treat functions whose demangled name contains SUBSTR as featureless on purpose
    --min-width BITS   vector width at which an instruction counts as SIMD work (default 256)
    --strict           out-of-line load/store/extract-only functions fail instead of warn
    --all              list allowlisted functions and every out-of-line intrinsic copy
    -v, --verbose      full report for clean targets too; list passed featureless fns and attribute groups
    --ll FILE...       audit these .ll files instead of building anything";

/// Flags that shape the audit itself, shared by both front doors.
pub struct AuditOpts {
    pub allow: Vec<String>,
    pub min_width: u32,
    pub show_all: bool,
    pub verbose: bool,
    /// Transfer-only Rule A findings (load/store/extract out of line) fail too.
    pub strict: bool,
}

enum Mode {
    Files(Vec<PathBuf>),
    Cargo(driver::Selection),
}

fn parse_args() -> Result<(Mode, AuditOpts), String> {
    let mut opts = AuditOpts {
        allow: Vec::new(),
        min_width: 256,
        show_all: false,
        verbose: false,
        strict: false,
    };
    let mut sel = driver::Selection::default();
    let mut files: Vec<PathBuf> = Vec::new();
    let mut ll_mode = false;

    let mut it = std::env::args().skip(1).peekable();
    // Invoked as `cargo thermite-audit`, cargo repeats the subcommand name.
    if it.peek().map(String::as_str) == Some("thermite-audit") {
        it.next();
    }

    let value = |it: &mut std::iter::Peekable<std::iter::Skip<std::env::Args>>, flag: &str| {
        it.next().ok_or_else(|| format!("{flag} needs a value"))
    };

    while let Some(a) = it.next() {
        match a.as_str() {
            "--allow" => opts.allow.push(value(&mut it, "--allow")?),
            "--min-width" => {
                let v = value(&mut it, "--min-width")?;
                opts.min_width = v.parse().map_err(|_| format!("bad --min-width {v}"))?;
            }
            "--all" => opts.show_all = true,
            "--strict" => opts.strict = true,
            "-v" | "--verbose" => opts.verbose = true,
            "-h" | "--help" => return Err(USAGE.to_owned()),
            "--ll" => ll_mode = true,

            "-p" | "--package" => sel.packages.push(value(&mut it, "-p")?),
            "--workspace" => sel.workspace = true,
            "--lib" => sel.kinds.lib = true,
            "--bins" => sel.kinds.bins = true,
            "--tests" => sel.kinds.tests = true,
            "--benches" => sel.kinds.benches = true,
            "--examples" => sel.kinds.examples = true,
            "--all-targets" => sel.kinds = driver::Kinds::ALL,
            "--bin" => sel.named.push((driver::Kind::Bin, value(&mut it, "--bin")?)),
            "--test" => sel.named.push((driver::Kind::Test, value(&mut it, "--test")?)),
            "--bench" => sel.named.push((driver::Kind::Bench, value(&mut it, "--bench")?)),
            "--example" => sel.named.push((driver::Kind::Example, value(&mut it, "--example")?)),
            "--features" | "-F" => sel.features.push(value(&mut it, "--features")?),
            "--all-features" => sel.all_features = true,
            "--no-default-features" => sel.no_default_features = true,
            "--release" => sel.profile = Some("release".to_owned()),
            "--profile" => sel.profile = Some(value(&mut it, "--profile")?),
            "--target" => sel.target = Some(value(&mut it, "--target")?),

            s if s.starts_with('-') => return Err(format!("unknown flag {s}\n{USAGE}")),
            _ => files.push(PathBuf::from(a)),
        }
    }

    if ll_mode || files.iter().any(|f| f.extension().is_some_and(|e| e == "ll")) {
        if files.is_empty() {
            return Err("--ll needs at least one .ll file".to_owned());
        }
        return Ok((Mode::Files(files), opts));
    }
    if !files.is_empty() {
        return Err(format!("unexpected argument {}\n{USAGE}", files[0].display()));
    }
    Ok((Mode::Cargo(sel), opts))
}

fn main() -> ExitCode {
    let (mode, opts) = match parse_args() {
        Ok(x) => x,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };

    let failed = match mode {
        Mode::Files(files) => {
            let mut failed = false;
            for path in &files {
                match ir::Module::parse(path, opts.min_width) {
                    Ok(module) => {
                        let r = Report::build(&module, &opts.allow);
                        failed |= r.print(&path.display().to_string(), &module, &opts, true).failed();
                    }
                    Err(e) => {
                        eprintln!("{}: {e}", path.display());
                        return ExitCode::from(2);
                    }
                }
            }
            failed
        }
        Mode::Cargo(sel) => match driver::run(&sel, &opts) {
            Ok(failed) => failed,
            Err(e) => {
                eprintln!("error: {e}");
                return ExitCode::from(2);
            }
        },
    };

    if failed { ExitCode::FAILURE } else { ExitCode::SUCCESS }
}
