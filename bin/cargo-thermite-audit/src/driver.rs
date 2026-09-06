//! The cargo front door: enumerate targets with `cargo metadata`, build each
//! one with `--emit=llvm-ir`, find the `.ll` cargo wrote, audit it.
//!
//! One `cargo rustc` per target: cargo only accepts one target per `rustc`
//! invocation, and the dependency graph is built once and cached, so each
//! extra target costs only its own leaf compile.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::SystemTime;

use serde_json::Value;

use crate::AuditOpts;
use crate::{ir, report, rules};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Lib,
    Bin,
    Test,
    Bench,
    Example,
}

impl Kind {
    fn flag(self) -> &'static str {
        match self {
            Kind::Lib => "--lib",
            Kind::Bin => "--bin",
            Kind::Test => "--test",
            Kind::Bench => "--bench",
            Kind::Example => "--example",
        }
    }

    /// Cargo's `target.kind` strings that mean this kind. Libraries report
    /// their crate types (`lib`, `rlib`, `cdylib`, ...). `proc-macro` and
    /// `custom-build` are deliberately not here.
    fn matches(self, kind: &str) -> bool {
        match self {
            Kind::Lib => matches!(kind, "lib" | "rlib" | "dylib" | "cdylib" | "staticlib"),
            Kind::Bin => kind == "bin",
            Kind::Test => kind == "test",
            Kind::Bench => kind == "bench",
            Kind::Example => kind == "example",
        }
    }
}

/// Which kinds of target to audit when none are named.
#[derive(Default, Clone, Copy)]
pub struct Kinds {
    pub lib: bool,
    pub bins: bool,
    pub tests: bool,
    pub benches: bool,
    pub examples: bool,
}

impl Kinds {
    pub const ALL: Kinds = Kinds {
        lib: true,
        bins: true,
        tests: true,
        benches: true,
        examples: true,
    };

    fn any(&self) -> bool {
        self.lib || self.bins || self.tests || self.benches || self.examples
    }

    fn wants(&self, kind: Kind) -> bool {
        match kind {
            Kind::Lib => self.lib,
            Kind::Bin => self.bins,
            Kind::Test => self.tests,
            Kind::Bench => self.benches,
            Kind::Example => self.examples,
        }
    }
}

/// Cargo-style target selection, as parsed from the command line.
#[derive(Default)]
pub struct Selection {
    pub packages: Vec<String>,
    pub workspace: bool,
    pub kinds: Kinds,
    pub named: Vec<(Kind, String)>,
    pub features: Vec<String>,
    pub all_features: bool,
    pub no_default_features: bool,
    /// `None` = dev, which is the right profile: `#[inline(always)]` is honored
    /// at every opt level and plain `#[inline]` is not at opt-level 1, so dev
    /// is the stricter audit. See notes/thermite-audit/LOG.md ("Dev profile").
    pub profile: Option<String>,
    /// `--target TRIPLE`, for cross-compiled audits (e.g. the aarch64 tests).
    pub target: Option<String>,
}

struct Target {
    package: String,
    kind: Kind,
    name: String,
}

fn cargo() -> Command {
    Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
}

/// Build and audit every selected target. Returns `Ok(true)` when any target
/// has live findings.
pub fn run(sel: &Selection, opts: &AuditOpts) -> Result<bool, String> {
    let targets = enumerate(sel)?;
    if targets.is_empty() {
        return Err("no targets selected".to_owned());
    }
    println!(
        "auditing {} target(s) in the {} profile",
        targets.len(),
        sel.profile.as_deref().unwrap_or("dev")
    );

    let mut rows: Vec<(String, String)> = Vec::new();
    let mut failed = false;
    for t in &targets {
        let title = format!("{} {} {}", t.package, t.kind.flag().trim_start_matches("--"), t.name);
        let line = match emit_ir(sel, t) {
            Err(e) => {
                failed = true;
                println!("== {title}: BUILD FAILED\n{e}");
                "BUILD FAILED".to_owned()
            }
            Ok(ll) => match ir::parse(&ll, opts.min_width) {
                Err(e) => {
                    failed = true;
                    println!("== {title}: cannot read {}: {e}", ll.display());
                    "UNREADABLE".to_owned()
                }
                Ok(module) => {
                    let r = rules::run(&module, &opts.allow);
                    let v = report::print(&title, &module, &r, opts, false);
                    failed |= v.failed();
                    v.line()
                }
            },
        };
        rows.push((title, line));
    }

    if rows.len() > 1 {
        println!("\nSummary:");
        let width = rows.iter().map(|(t, _)| t.len()).max().unwrap_or(0);
        for (title, line) in &rows {
            println!("  {title:width$}  {line}");
        }
    }
    Ok(failed)
}

/// `cargo metadata` -> the targets the selection asks for.
fn enumerate(sel: &Selection) -> Result<Vec<Target>, String> {
    let out = cargo()
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .stderr(Stdio::inherit())
        .output()
        .map_err(|e| format!("cannot run cargo metadata: {e}"))?;
    if !out.status.success() {
        return Err("cargo metadata failed".to_owned());
    }
    let meta: Value = serde_json::from_slice(&out.stdout).map_err(|e| format!("bad cargo metadata: {e}"))?;

    let packages = meta["packages"].as_array().ok_or("cargo metadata: no packages")?;
    let members: Vec<&str> = meta["workspace_members"]
        .as_array()
        .map_or(Vec::new(), |m| m.iter().filter_map(Value::as_str).collect());

    // Which packages: named ones, the whole workspace, or the current one.
    let wanted: Vec<&Value> = if !sel.packages.is_empty() {
        let mut found = Vec::new();
        for name in &sel.packages {
            let pkg = packages
                .iter()
                .find(|p| p["name"].as_str() == Some(name))
                .ok_or_else(|| format!("package `{name}` not found in the workspace"))?;
            found.push(pkg);
        }
        found
    } else if sel.workspace {
        packages
            .iter()
            .filter(|p| members.iter().any(|m| p["id"].as_str() == Some(m)))
            .collect()
    } else {
        let root = meta["resolve"]["root"].as_str();
        match root.and_then(|r| packages.iter().find(|p| p["id"].as_str() == Some(r))) {
            Some(p) => vec![p],
            None => return Err("not inside a package; pass -p NAME or --workspace".to_owned()),
        }
    };

    // Which kinds: explicit flags, else everything.
    let kinds = if sel.kinds.any() || !sel.named.is_empty() {
        sel.kinds
    } else {
        Kinds::ALL
    };

    let mut targets = Vec::new();
    for pkg in wanted {
        let package = pkg["name"].as_str().unwrap_or_default().to_owned();
        let Some(list) = pkg["targets"].as_array() else {
            continue;
        };
        for t in list {
            let name = t["name"].as_str().unwrap_or_default().to_owned();
            let cargo_kinds: Vec<&str> = t["kind"]
                .as_array()
                .map_or(Vec::new(), |k| k.iter().filter_map(Value::as_str).collect());
            for kind in [Kind::Lib, Kind::Bin, Kind::Test, Kind::Bench, Kind::Example] {
                if !cargo_kinds.iter().any(|k| kind.matches(k)) {
                    continue;
                }
                let named = sel.named.iter().any(|(k, n)| *k == kind && *n == name);
                if named || kinds.wants(kind) {
                    targets.push(Target {
                        package: package.clone(),
                        kind,
                        name: name.clone(),
                    });
                }
            }
        }
    }
    for (kind, name) in &sel.named {
        if !targets.iter().any(|t| t.kind == *kind && t.name == *name) {
            return Err(format!(
                "no {} target named `{name}`",
                kind.flag().trim_start_matches("--")
            ));
        }
    }
    Ok(targets)
}

/// `cargo rustc` one target with `--emit=llvm-ir` and return the `.ll` path.
fn emit_ir(sel: &Selection, t: &Target) -> Result<PathBuf, String> {
    let mut cmd = cargo();
    cmd.args(["rustc", "-p", &t.package, t.kind.flag()]);
    if t.kind != Kind::Lib {
        cmd.arg(&t.name);
    }
    for f in &sel.features {
        cmd.args(["--features", f]);
    }
    if sel.all_features {
        cmd.arg("--all-features");
    }
    if sel.no_default_features {
        cmd.arg("--no-default-features");
    }
    if let Some(p) = &sel.profile {
        cmd.args(["--profile", p]);
    }
    if let Some(t) = &sel.target {
        cmd.args(["--target", t]);
    }
    cmd.args(["--message-format=json-render-diagnostics", "--", "--emit=llvm-ir"]);
    cmd.stderr(Stdio::inherit());

    let out = cmd.output().map_err(|e| format!("cannot run cargo rustc: {e}"))?;
    let stdout = String::from_utf8_lossy(&out.stdout);

    // The artifact for OUR target: same package name, same target name, a kind
    // that matches. Cargo prints artifacts for every dependency too.
    let mut artifact: Option<PathBuf> = None;
    for line in stdout.lines() {
        let Ok(msg) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if msg["reason"].as_str() != Some("compiler-artifact") {
            continue;
        }
        let target = &msg["target"];
        if target["name"].as_str() != Some(t.name.as_str()) {
            continue;
        }
        let kinds: Vec<&str> = target["kind"]
            .as_array()
            .map_or(Vec::new(), |k| k.iter().filter_map(Value::as_str).collect());
        if !kinds.iter().any(|k| t.kind.matches(k)) {
            continue;
        }
        let path = msg["executable"].as_str().or_else(|| {
            msg["filenames"]
                .as_array()
                .and_then(|f| f.first())
                .and_then(Value::as_str)
        });
        if let Some(p) = path {
            artifact = Some(PathBuf::from(p));
        }
    }
    if !out.status.success() {
        return Err(format!("cargo rustc exited with {}", out.status));
    }
    let artifact = artifact.ok_or("cargo reported no artifact for this target")?;
    find_ll(&artifact).ok_or_else(|| format!("no .ll next to {}", artifact.display()))
}

/// The `.ll` rustc wrote for `artifact`. Same stem with the extension swapped
/// is the common case (`foo.exe` -> `foo.ll`, `libthermite-HASH.rlib` ->
/// `libthermite-HASH.ll`). Otherwise the newest `deps/<stem>-<hash>.ll`, with
/// or without a `lib` prefix.
fn find_ll(artifact: &Path) -> Option<PathBuf> {
    let direct = artifact.with_extension("ll");
    if direct.is_file() {
        return Some(direct);
    }

    let file = artifact.file_stem()?.to_str()?;
    let base = file.strip_prefix("lib").unwrap_or(file);
    let base = match base.rfind('-') {
        Some(i) if base[i + 1..].chars().all(|c| c.is_ascii_hexdigit()) => &base[..i],
        _ => base,
    };
    let dir = artifact.parent()?;
    let mut candidates: Vec<(SystemTime, PathBuf)> = Vec::new();
    for d in [
        dir.to_path_buf(),
        dir.join("deps"),
        dir.parent().map(|p| p.join("deps")).unwrap_or_default(),
    ] {
        let Ok(entries) = std::fs::read_dir(&d) else { continue };
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().is_none_or(|x| x != "ll") {
                continue;
            }
            let Some(stem) = p.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let stem = stem.strip_prefix("lib").unwrap_or(stem);
            let ok = stem == base
                || stem.strip_prefix(base).is_some_and(|rest| {
                    rest.strip_prefix('-')
                        .is_some_and(|h| !h.is_empty() && h.chars().all(|c| c.is_ascii_hexdigit()))
                });
            if ok && let Ok(m) = e.metadata().and_then(|m| m.modified()) {
                candidates.push((m, p));
            }
        }
    }
    candidates.sort();
    candidates.pop().map(|(_, p)| p)
}
