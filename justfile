# Thermite task runner. Run `just` (or `just --list`) to see recipes.
#
# Scope: these recipes target the `thermite` crate only. The rest of the
# workspace contains backends and tools that don't build on an x86_64 host
# (spirv/neon/wasm), or are mid-rewrite, so a `--workspace` run would fail.

# Native PowerShell so the JSON parsing recipes work without `jq`/`sh`.
set windows-shell := ["powershell.exe", "-NoProfile", "-Command"]

# --- Configuration -----------------------------------------------------------

# Coverage/test toolchain. Deliberately decoupled from rust-toolchain.toml,
# which is pinned to a rust-gpu nightly for the (currently inactive) SPIR-V
# backend. Flip to a nightly only for `cov-branch` / `miri`.
toolchain := "stable"

# Crate under measurement.
pkg := "thermite"

# Host-safe feature set. NOT `--all-features`: thermite's features select
# mutually exclusive backends (spirv, neon, wasm, avx512-tier*, std_simd) that
# cannot co-compile on an x86_64 host. `std` is additive and lights up the
# fmt-gated panic paths, so it raises the honest line count.
features := "std"

_cargo := "cargo +" + toolchain
# Build/collect recipes need the feature flags; the `report` subcommand reuses
# stored profdata and rejects `--features`, so it takes the package alone.
_args := "-p " + pkg + " --features " + features
_report_args := "-p " + pkg

# Default: show the recipe list.
default:
    @just --list

# --- Tests -------------------------------------------------------------------

# Run the thermite test suite (release; the differential suites are slow in debug).
#
# `nextest` runs every test in its own process and schedules them across all
# cores, rather than running each test binary to completion in turn - worth ~25%
# here. It deliberately does not support doctests, so those run separately;
# `test` is the real gate and runs both. Pass a filter, e.g.
# `just test -E 'test(interleave)'` or `just test --test diff_ops`.
test *args:
    {{ _cargo }} nextest run {{ _args }} --release {{ args }}
    {{ _cargo }} test {{ _args }} --release --doc

# Tests only, no doctests - the fast inner-loop gate.
test-fast *args:
    {{ _cargo }} nextest run {{ _args }} --release {{ args }}

# --- WASM test suite ---------------------------------------------------------
#
# Runs the (wasm-applicable) thermite tests on `wasm32-wasip1` under wasmtime. The
# `tests/wasm-runner` crate is a thin wasmtime+WASI host that executes each compiled libtest
# binary; it's wired in via the CARGO_TARGET_WASM32_WASIP1_RUNNER env var (built fresh, so no
# machine-specific path is committed). `+simd128,+relaxed-simd` come from `.cargo/config.toml`.
#
# Prereqs (one-time): `rustup target add wasm32-wasip1`.
# `--tests` excludes the criterion benches (native-only). Pass extra cargo args, e.g.
# `just wasm-test --test diff_u8`.

# Build the wasmtime-based runner that executes wasm libtest binaries.
_wasm-runner:
    cargo +nightly build -p wasm-runner --release

# Run thermite's wasm-capable tests on wasm32-wasip1 via wasmtime. Defaults to all test targets
# (`--tests`); pass a filter to scope it, e.g. `just wasm-test "--test diff_u8"`.
# LTO is force-disabled: fat LTO (`profile.release.lto = true`) is a known LLVM weak spot on
# 32-bit targets and gives no benefit for tests. (Note: not what caused the diff_swizzle `-O3`
# "Cannot select" ICE - that was an integer compare-reduction in the test, fixed in-source.)
wasm-test *args="--tests": _wasm-runner
    $env:CARGO_TARGET_WASM32_WASIP1_RUNNER = (Resolve-Path "target/release/wasm-runner.exe").Path; $env:CARGO_PROFILE_RELEASE_LTO = "false"; $env:CARGO_PROFILE_BENCH_LTO = "false"; cargo +nightly test -p {{ pkg }} --no-default-features --features "wasm,std" --target wasm32-wasip1 {{ args }} -- --test-threads=1

# --- Code coverage (cargo-llvm-cov) ------------------------------------------
#
# Two-phase workflow: `cov-collect` runs the instrumented tests once and stores
# profdata under target/llvm-cov-target; the `cov-missing` / `cov-summary` /
# `cov-percent` report recipes then reuse it and are fast (no rebuild/re-run).
# `cov` is the one-shot convenience entry point.

# `--ignore-run-fail`: still collect coverage when a test *assertion* fails (a
# compile error aborts as normal). thermite has a known, accepted failure
# (frldexp denormal-flush, gated by the off-by-default `preserve_denormals`);
# we want its coverage regardless. Use `just test` for a strict pass/fail gate.

# One-shot: collect coverage and open the HTML report in a browser.
cov:
    {{ _cargo }} llvm-cov {{ _args }} --ignore-run-fail --html --open

# Collect coverage by running the tests; emit no report yet (feeds the report-only recipes below).
cov-collect:
    {{ _cargo }} llvm-cov --no-report {{ _args }} --ignore-run-fail

# Trust this over the summary's "Missed Lines" column (it counts per-instantiation
# segments and over-reports). Report-only: run `cov-collect` (or `cov`) first.
#
# Authoritative list of uncovered source lines.
cov-missing:
    {{ _cargo }} llvm-cov report {{ _report_args }} --show-missing-lines

# Per-file / total coverage table. Report-only.
cov-summary:
    {{ _cargo }} llvm-cov report {{ _report_args }} --summary-only

# Just the total line-coverage percentage (for CI / a badge). Report-only.
cov-percent:
    ({{ _cargo }} llvm-cov report {{ _report_args }} --json --summary-only | ConvertFrom-Json).data[0].totals.lines.percent

# Reads much lower than line coverage (LLVM doesn't credit diverging arms);
# informational only, don't badge it.
#
# Branch coverage (needs nightly).
cov-branch:
    cargo +nightly llvm-cov --branch {{ _args }} --summary-only

# Remove all coverage artifacts and stored profdata.
cov-clean:
    {{ _cargo }} llvm-cov clean --workspace

# --- Miri --------------------------------------------------------------------
#
# Best-effort: Miri does NOT implement x86 SIMD intrinsics, so the V2/V3 backend
# tests will error under it. Useful for scalar-backend / generic `unsafe` paths
# (slice iterators, gather/scatter bounds logic). Pass a filter, e.g.
# `just miri frldexp`, to stay on Miri-compatible tests.
miri filter="":
    cargo +nightly miri test {{ _args }} {{ filter }}

# --- Docs

doc:
    $env:RUSTDOCFLAGS="--html-in-header {{justfile_directory()}}/katex-header.html"; cargo doc --no-deps

doc-open:
    $env:RUSTDOCFLAGS="--html-in-header {{justfile_directory()}}/katex-header.html"; cargo doc --no-deps --open

# --- Maintenance -------------------------------------------------------------

# Crates whose rustdoc renders KaTeX math and therefore need a crate-local copy
# of katex-header.html (referenced by their docs.rs `--html-in-header` arg, a
# crate-relative path). Space-separated. Add a crate here when it starts using
# `math` doc blocks. The LICENSE files, by contrast, go to *every* crate.
katex_crates := "thermite thermite-special thermite-compensated thermite-blas thermite-geometry thermite-complex"

# Re-propagate the shared root assets into the member crates, force-overwriting
# each crate's copy. Run after editing any root original (LICENSE-MIT,
# LICENSE-APACHE, katex-header.html) so the published crates stay in sync.
#
# LICENSE-* -> every crate under crates/;  katex-header.html -> {{katex_crates}}.
sync-assets:
    Get-ChildItem crates -Directory | ForEach-Object { Copy-Item LICENSE-MIT, LICENSE-APACHE $_.FullName -Force; Write-Host "license -> $($_.Name)" }
    '{{katex_crates}}'.Split(' ') | ForEach-Object { Copy-Item katex-header.html "crates/$_" -Force; Write-Host "katex   -> $_" }

# --- Skill bundle ------------------------------------------------------------

# The Claude Code usage skill that ships to downstream Thermite users.
skill_dir := ".claude/skills/thermite"
skill_zip := ".claude/skills/thermite.zip"

# Bundle the `thermite` skill into a single zip for distribution: attach it to a
# GitHub release, and users unzip it into their own `.claude/skills/`. The archive
# roots at `thermite/` (SKILL.md, references/, and the plugin manifest), so it drops
# straight into place. Output: {{skill_zip}}.
bundle-skill:
    if (-not (Test-Path '{{skill_dir}}/SKILL.md')) { throw 'skill not found at {{skill_dir}}' }
    New-Item -ItemType Directory -Force -Path (Split-Path '{{skill_zip}}') | Out-Null
    if (Test-Path '{{skill_zip}}') { Remove-Item '{{skill_zip}}' -Force }
    Compress-Archive -Path '{{skill_dir}}' -DestinationPath '{{skill_zip}}'
    Write-Host "bundled {{skill_dir}} -> {{skill_zip}} ($([math]::Round((Get-Item '{{skill_zip}}').Length / 1KB, 1)) KB)"

# Copy the local thermite skill into the global ~/.claude/skills/ directory,
# overwriting the installed copy in place. Run after editing skill files to
# make changes available to the current Claude Code session immediately.
sync-skill:
    if (-not (Test-Path '{{skill_dir}}/SKILL.md')) { throw 'skill not found at {{skill_dir}}' }
    New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.claude\skills\thermite" | Out-Null
    Copy-Item -Path '{{skill_dir}}/*' -Destination "$env:USERPROFILE\.claude\skills\thermite" -Recurse -Force
    Write-Host "synced {{skill_dir}} -> $env:USERPROFILE\.claude\skills\thermite"
