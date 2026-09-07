//! Streaming parser for the subset of LLVM textual IR the audit needs:
//! `define` headers, per-instruction vector widths, call targets, and the
//! `attributes #N` groups at the end of the module.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

/// One `define` in the module, with the counts the rules look at.
#[derive(Debug)]
pub struct FnRecord {
    /// Demangled symbol (or the raw symbol when it does not demangle).
    pub name: String,
    pub attr_id: Option<u32>,
    /// Instructions doing vector arithmetic at or above the width threshold.
    pub simd_ops: u32,
    /// Widest vector type (bits) seen in a counted instruction.
    pub max_width: u32,
    /// Calls whose callee lives under `core_arch::` (detection intrinsics excluded).
    pub arch_calls: u32,
    /// The subset of `arch_calls` that compute, as opposed to move data
    /// (load/store/set/extract/cast). A function with only transfer calls out
    /// of line pays one call per use. One with compute out of line is the bug.
    pub compute_calls: u32,
    /// Inline `asm` statements. Never subject to the target-feature inlining
    /// check, so AVX text in a featureless function assembles silently. Any
    /// asm in a backend-named featureless function counts as SIMD work.
    pub asm_calls: u32,
    /// Demangled `core_arch` callee (short name) -> call count.
    pub callees: HashMap<String, u32>,
    /// Every non-LLVM-intrinsic callee (full demangled name) -> call count,
    /// for reporting who calls a featureless function.
    pub calls: HashMap<String, u32>,
}

impl FnRecord {
    /// Vector arithmetic at or above the width threshold, or any inline asm.
    /// This is the test that lets the featureless outer half of a dispatched
    /// fn (one tail call, no arithmetic) pass Rule B.
    pub fn does_simd_work(&self) -> bool {
        self.simd_ops > 0 || self.asm_calls > 0
    }
}

/// The parsed module: every function plus the attribute groups.
#[derive(Debug, Default)]
pub struct Module {
    pub fns: Vec<FnRecord>,
    /// Attribute group id -> enabled target features (`+` entries, sign stripped).
    pub attrs: HashMap<u32, Vec<String>>,
}

impl Module {
    /// Features enabled on a function, empty when it has no group or no
    /// `"target-features"` entry.
    pub fn features(&self, f: &FnRecord) -> &[String] {
        f.attr_id
            .and_then(|id| self.attrs.get(&id))
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }
}

/// Opcodes that constitute "SIMD work" when they carry a wide vector type.
/// `ret`, `load`, `store`, `phi`, `alloca`, `bitcast` are deliberately absent:
/// a featureless function forwarding vector args does none of the work.
const SIMD_OPCODES: &[&str] = &[
    "fadd",
    "fsub",
    "fmul",
    "fdiv",
    "frem",
    "fneg",
    "fcmp",
    "add",
    "sub",
    "mul",
    "udiv",
    "sdiv",
    "and",
    "or",
    "xor",
    "shl",
    "lshr",
    "ashr",
    "icmp",
    "select",
    "shufflevector",
    "fptrunc",
    "fpext",
    "sitofp",
    "uitofp",
    "fptosi",
    "fptoui",
    "trunc",
    "zext",
    "sext",
    "extractelement",
    "insertelement",
];

impl Module {
    /// Parse one `.ll` file. `min_width` (bits) is the vector width at which an
    /// instruction counts as SIMD work.
    pub fn parse(path: &Path, min_width: u32) -> io::Result<Module> {
        let reader = BufReader::with_capacity(1 << 20, File::open(path)?);
        let mut module = Module::default();
        let mut current: Option<FnRecord> = None;

        for line in reader.split(b'\n') {
            let line = line?;
            let line = String::from_utf8_lossy(&line);
            let line = line.trim_end_matches('\r');

            if let Some(f) = current.as_mut() {
                if line == "}" {
                    module.fns.push(current.take().expect("open function"));
                    continue;
                }
                f.classify(line.trim_start(), min_width);
                continue;
            }

            if line.starts_with("define ") {
                current = Some(FnRecord::from_define(line));
            } else if line.starts_with("attributes #")
                && let Some((id, feats)) = parse_attr_group(line)
            {
                module.attrs.insert(id, feats);
            }
        }

        Ok(module)
    }
}

impl FnRecord {
    /// A fresh record from a `define` header: symbol, demangled name, and the
    /// trailing `#N` attribute group.
    fn from_define(line: &str) -> Self {
        let symbol = symbol_after_at(line).unwrap_or_default();
        let name = demangle(&symbol);

        // Attribute group: the last `#N` token before the opening brace.
        let attr_id = line
            .rsplit(' ')
            .filter_map(|tok| tok.strip_prefix('#'))
            .find_map(|digits| digits.parse::<u32>().ok());

        FnRecord {
            name,
            attr_id,
            simd_ops: 0,
            max_width: 0,
            arch_calls: 0,
            compute_calls: 0,
            asm_calls: 0,
            callees: HashMap::new(),
            calls: HashMap::new(),
        }
    }

    /// Fold one instruction line into this record's counts.
    fn classify(&mut self, line: &str, min_width: u32) {
        if line.is_empty() || line.starts_with(';') || line.ends_with(':') {
            return;
        }

        // `%x = [tail] opcode ...` or `opcode ...`
        let rhs = match line.find(" = ") {
            Some(i) if line.starts_with('%') => &line[i + 3..],
            _ => line,
        };
        let rhs = rhs
            .strip_prefix("tail ")
            .or_else(|| rhs.strip_prefix("musttail "))
            .or_else(|| rhs.strip_prefix("notail "))
            .unwrap_or(rhs);
        let opcode = rhs.split(' ').next().unwrap_or("");

        match opcode {
            "call" | "invoke" => {
                // `call <ty> asm [sideeffect] "text", "constraints"(args)`: the
                // `asm` keyword sits before the first quote.
                let head = &rhs[..rhs.find('"').unwrap_or(rhs.len())];
                if head.split(' ').any(|t| t == "asm") {
                    // `core::hint::black_box` is an EMPTY asm template, not work.
                    if rhs[head.len()..].starts_with("\"\"") {
                        return;
                    }
                    self.asm_calls += 1;
                    if let Some(w) = widest_vector(rhs) {
                        self.max_width = self.max_width.max(w);
                    }
                    return;
                }
                let Some(callee) = symbol_after_at(rhs) else { return };
                if callee.starts_with("llvm.") {
                    // LLVM intrinsics (fma, sqrt, minnum, x86.*) with vector operands are work.
                    if let Some(w) = widest_vector(rhs)
                        && w >= min_width
                    {
                        self.simd_ops += 1;
                        self.max_width = self.max_width.max(w);
                    }
                    return;
                }
                let name = demangle(&callee);
                *self.calls.entry(name.clone()).or_insert(0) += 1;
                // The function's own path must be under core_arch. A generic
                // argument naming `__m256` (`<[__m256; 2]>::try_map`) is not one.
                if is_core_arch_path(&name) {
                    let short = short_name(&name);
                    if is_detection_intrinsic(&short) {
                        return;
                    }
                    self.arch_calls += 1;
                    if !is_transfer_intrinsic(&short) {
                        self.compute_calls += 1;
                    }
                    *self.callees.entry(short).or_insert(0) += 1;
                }
            }
            op if SIMD_OPCODES.contains(&op) => {
                if let Some(w) = widest_vector(rhs)
                    && w >= min_width
                {
                    self.simd_ops += 1;
                    self.max_width = self.max_width.max(w);
                }
            }
            _ => {}
        }
    }
}

fn parse_attr_group(line: &str) -> Option<(u32, Vec<String>)> {
    let rest = line.strip_prefix("attributes #")?;
    let end = rest.find(' ')?;
    let id: u32 = rest[..end].parse().ok()?;

    let feats = match rest.find("\"target-features\"=\"") {
        Some(i) => {
            let s = &rest[i + "\"target-features\"=\"".len()..];
            let s = &s[..s.find('"').unwrap_or(s.len())];
            // LLVM repeats features it was told twice, so dedupe for display.
            let mut v: Vec<String> = s
                .split(',')
                .filter_map(|f| f.strip_prefix('+'))
                .map(str::to_owned)
                .collect();
            v.sort_unstable();
            v.dedup();
            v
        }
        None => Vec::new(),
    };
    Some((id, feats))
}

/// The symbol following the first `@` in `s`, quoted or bare.
fn symbol_after_at(s: &str) -> Option<String> {
    let i = s.find('@')?;
    let rest = &s[i + 1..];
    if let Some(q) = rest.strip_prefix('"') {
        let end = q.find('"')?;
        Some(q[..end].to_owned())
    } else {
        let end = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '_' | '$' | '.')))
            .unwrap_or(rest.len());
        Some(rest[..end].to_owned())
    }
}

/// Widest `<N x T>` vector type on the line, in bits. `ptr` vectors are ignored.
fn widest_vector(s: &str) -> Option<u32> {
    let mut best = None;
    let mut rest = s;
    while let Some(i) = rest.find('<') {
        rest = &rest[i + 1..];
        let Some(x) = rest.find(" x ") else { break };
        let Ok(n) = rest[..x].trim().parse::<u32>() else {
            continue;
        };
        let ty = &rest[x + 3..];
        let end = ty.find('>').unwrap_or(ty.len());
        let bits = match ty[..end].trim() {
            "half" | "bfloat" | "i16" => 16,
            "float" | "i32" => 32,
            "double" | "i64" => 64,
            "i8" => 8,
            "i1" => 1,
            _ => continue,
        };
        let w = n * bits;
        if best.is_none_or(|b| w > b) {
            best = Some(w);
        }
    }
    best
}

fn demangle(symbol: &str) -> String {
    // Strip the hash suffix so two instantiations with the same path merge.
    format!("{:#}", rustc_demangle::demangle(symbol))
}

/// `core::core_arch::x86::avx::_mm256_cmp_ps::<17>` -> `_mm256_cmp_ps`, and
/// LTO's duplicate-rename suffix (`_xgetbv.1593`) is dropped too.
pub fn short_name(name: &str) -> String {
    let base = name.split("::<").next().unwrap_or(name);
    let last = base.rsplit("::").next().unwrap_or(base);
    match last.rfind('.') {
        Some(i) if last[i + 1..].chars().all(|c| c.is_ascii_digit()) && i > 0 => last[..i].to_owned(),
        _ => last.to_owned(),
    }
}

/// True when the function itself lives under `core::core_arch::` (an
/// intrinsic), as opposed to merely mentioning a `core_arch` type in its
/// generic arguments.
pub fn is_core_arch_path(name: &str) -> bool {
    let path = name.split("::<").next().unwrap_or(name);
    path.starts_with("core::core_arch::") || path.starts_with("<core::core_arch::")
}

/// CPU feature detection: inherently featureless wherever it is called
/// (`_xgetbv` is `#[target_feature(enable = "xsave")]`, so it always stays out
/// of line), and never SIMD work.
pub fn is_detection_intrinsic(short: &str) -> bool {
    matches!(
        short,
        "_xgetbv" | "__cpuid" | "__cpuid_count" | "__get_cpuid_max" | "_rdtsc" | "__rdtscp"
    )
}

/// Data movement rather than arithmetic. Out of line these cost one call per
/// use and compute nothing wrong, so they are reported as warnings, not failures,
/// unless `--strict`.
pub fn is_transfer_intrinsic(short: &str) -> bool {
    const TRANSFER: &[&str] = &[
        // NEON: loads/stores, lane moves, duplicates, reinterprets, halves.
        "vld",
        "vst",
        "vdup",
        "vmov",
        "vget",
        "vset",
        "vreinterpret",
        "vcombine",
        "vcreate",
        // x86.
        "_load",
        "_store",
        "_set1_",
        "_set_",
        "_setr_",
        "_setzero",
        "_extract",
        "_insert",
        "_cast",
        "_cvtss_f32",
        "_cvtsd_f64",
        "_cvtsi",
        "_movemask",
        "_broadcast",
        "_maskload",
        "_maskstore",
        "_stream",
        "_lddqu",
        "_undefined",
        "_zeroupper",
        "_prefetch",
    ];
    TRANSFER.iter().any(|t| short.contains(t))
}
