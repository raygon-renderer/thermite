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

# Run the thermite test suite on the `dev` profile (`opt-level = 1`).
#
# These recipes used to pass `--release`, on the belief that the differential
# suites were too slow in debug. Measured 2026-08-23, that is no longer true -
# they are assert-and-oracle bound, not kernel-throughput bound, so the test
# crate's opt-level barely moves them: diff_math 613ms dev vs 576ms at O2,
# diff_polyfill 444 vs 507, diff_mask 1369 vs 1538, diff_real_math 60 vs 76.
# Release cost 413s of fat LTO and serial LLVM per cold build for that (the
# worst single unit, diff_mask, was 378s of it); `dev` builds the same 84
# binaries in 292s. `dev` also keeps incremental compilation, which
# `release`-derived profiles cannot.
#
# The bigger point is that `--release` was silently giving up DEBUG ASSERTIONS
# and OVERFLOW CHECKS, which `dev` has on - so the suite is now both cheaper to
# build and checking more than it did. Those checks are most of what `dev`
# spends: with them off, the worst unit drops 236s -> 103s. They stay on
# regardless (owner, 2026-08-23) - do not "optimize" the profile by disabling
# them. Raising opt-level does NOT help here (262s at O2 vs 236s at O1); the
# cost is checked-arithmetic IR volume, not optimization.
#
# `nextest` runs every test in its own process and schedules them across all
# cores, rather than running each test binary to completion in turn - worth ~25%
# here. It deliberately does not support doctests, so those run separately;
# `test` is the real gate and runs both. Pass a filter, e.g.
# `just test -E 'test(interleave)'` or `just test --test diff_ops`.
test *args:
    {{ _cargo }} nextest run {{ _args }} {{ args }}
    {{ _cargo }} test {{ _args }} --doc

# Tests only, no doctests - the fast inner-loop gate.
test-fast *args:
    {{ _cargo }} nextest run {{ _args }} {{ args }}

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
    $env:CARGO_TARGET_WASM32_WASIP1_RUNNER = (Resolve-Path "target/release/wasm-runner.exe").Path; $env:CARGO_PROFILE_RELEASE_LTO = "false"; $env:CARGO_PROFILE_BENCH_LTO = "false"; cargo test -p {{ pkg }} --no-default-features --features "wasm,std" --target wasm32-wasip1 {{ args }} -- --test-threads=1

# --- ARM NEON test suite (Raspberry Pi) ---------------------------------------
#
# Cross-builds the test binaries for 64-bit ARM Linux and stages them under
# `target/pi-stage/` with stable names plus a `run-all.sh`. Push them to the
# device and run there:
#
#     scp target/pi-stage/* pi@<host>:~/thermite-tests/
#     ssh pi@<host> 'cd ~/thermite-tests && chmod +x run-all.sh && ./run-all.sh'
#
# Target is aarch64-unknown-linux-musl + rust-lld (see .cargo/config.toml): no
# external cross-toolchain needed, and the static binaries run on any 64-bit
# OS image. The Pi must run a 64-bit OS (`uname -m` = aarch64). LTO is
# disabled for iteration speed, like the wasm recipe.
#
# Prereqs (one-time): `rustup target add aarch64-unknown-linux-musl`.
# Scope with e.g. `just pi-build "--test diff_ops"`.
pi-build *args="--tests":
    $env:CARGO_PROFILE_RELEASE_LTO = "false"; \
    $json = cargo +stable test -p {{ pkg }} --release --no-run --target aarch64-unknown-linux-musl --no-default-features --features "std" {{ args }} --message-format=json; \
    if ($LASTEXITCODE -ne 0) { $json | Where-Object { $_ -match '"level":"error"' } | ForEach-Object { ($_ | ConvertFrom-Json).message.rendered } | Write-Host; exit 1 }; \
    $exes = $json | ForEach-Object { $_ | ConvertFrom-Json } | Where-Object { $_.reason -eq 'compiler-artifact' -and $_.executable } | ForEach-Object { $_.executable }; \
    New-Item -ItemType Directory -Force target/pi-stage | Out-Null; \
    if ('{{ args }}' -eq '--tests') { Get-ChildItem target/pi-stage -File | Remove-Item -Force -Confirm:$false }; \
    $names = foreach ($e in $exes) { $n = [IO.Path]::GetFileName($e) -replace '-[0-9a-f]+$', ''; Copy-Item $e ("target/pi-stage/" + $n); $n }; \
    if ('{{ args }}' -eq '--tests') { $script = @('#!/bin/sh', 'set -u', 'fail=0') + ($names | ForEach-Object { "echo `"=== $_ ===`"; ./$_ --test-threads=2 || fail=1" }) + @('exit $fail'); [IO.File]::WriteAllText((Join-Path (Get-Location) 'target/pi-stage/run-all.sh'), (($script -join "`n") + "`n")) }; \
    Write-Host ("Staged " + @($names).Count + " test binaries in target/pi-stage/"); \
    Write-Host "Push:  scp target/pi-stage/* pi@<host>:~/thermite-tests/"

# Convenience push over scp (recursive: includes bench/ if staged); runs stay
# manual on the device.
# Usage: just pi-push pi@raspberrypi.local
pi-push host:
    scp -r target/pi-stage/* "{{ host }}:~/thermite-tests/"

# Run the staged aarch64 test binaries under qemu user emulation via a Podman
# arm64 container - a fast pre-Pi correctness gate on the dev box (emulation
# validates instruction semantics but not timing/hardware quirks; the Pi run
# is still the final word). Prereqs (once per podman-machine boot):
#   podman machine start
#   podman run --privileged --rm docker.io/tonistiigi/binfmt --install arm64
qemu-test:
    podman run --rm --platform linux/arm64 -v "$(Resolve-Path target/pi-stage):/tests:ro" docker.io/library/alpine:latest sh -c 'cd /tests && sh run-all.sh'

# --- AVX-512 test suite (Intel SDE emulation) ---------------------------------
#
# No AVX-512 hardware here, and QEMU's TCG has never implemented AVX-512, so the
# emulator is Intel SDE (a Pin-based DBT). It emulates CPUID *and* XCR0, so the
# runtime dispatcher selects the AVX-512 backend on its own - the test binaries
# are built normally (baseline ISA + `#[target_feature]` trampolines), exactly as
# they ship. See AVX512_BUILD_AND_EMULATION.md.
#
# Point `THERMITE_SDE` at sde.exe if it lives elsewhere. Emulation is 10-100x
# slower and its timings are meaningless: scope runs, and never bench under it.
sde := env("THERMITE_SDE", "F:/bin/sde/sde.exe")

# tier -> SDE chip flag. Each flag also turns on SDE's chip-check, which faults
# on any instruction outside that chip's ISA - i.e. it catches a tier-2 encoding
# leaking into a tier-1 build. `-cpx` (Cooper Lake) is deliberately not tier 3:
# it has BF16 but not the tier-2 set, so `avx512_tier()` calls it Tier1.
# Knights Landing (F+CD only) is below the ladder's floor and reports no tier
# at all, but current SDE builds have dropped `-knl`, so that arm of
# `avx512_tier()` is pinned by the synthetic-features test instead.
#
# `10.1`/`10.2` are AVX10 rather than rungs of the tier ladder: Granite Rapids
# is the first part to enumerate leaf 0x24 (version 1) and Diamond Rapids
# reports version 2. Both still report Tier3, since AVX10.1 is defined as the
# whole AVX-512 feature set - they exercise `Features::avx10`, not a new tier.
_sde-chip tier:
    @$c = @{ "1" = "skx"; "2" = "icx"; "3" = "spr"; "10.1" = "gnr"; "10.2" = "dmr" }["{{ tier }}"]; \
    if (-not $c) { Write-Error "tier must be 1..3, 10.1 or 10.2"; exit 1 }; $c

# Sanity gate: run the CPU-detection and ISA tests under an emulated AVX-512 CPU
# and confirm the ladder reports the expected rung. Run this BEFORE writing any
# backend code - if detection reads false, everything else silently tests x86-v3.
#

# Confirm AVX-512 detection works under an emulated CPU (tier 1..3, 10.1, 10.2).
sde-check tier="1":
    just _sde-chip {{ tier }} | Out-Null; \
    $chip = (just _sde-chip {{ tier }}); \
    $json = cargo test -p {{ pkg }} --features "{{ features }}" --lib --no-run --message-format=json; \
    $exe = $json | ForEach-Object { $_ | ConvertFrom-Json } | Where-Object { $_.reason -eq 'compiler-artifact' -and $_.executable } | ForEach-Object { $_.executable }; \
    Write-Host "=== SDE -$chip ==="; \
    & "{{ sde }}" "-$chip" -- $exe isa:: --test-threads=1

# Scope with cargo args, e.g. `just sde-test 3 "--test diff_ops"`.

# Run thermite's tests under an emulated AVX-512 CPU (tier 1..3, 10.1, 10.2).
sde-test tier="1" *args="--lib":
    $chip = (just _sde-chip {{ tier }}); \
    $json = cargo test -p {{ pkg }} --features "{{ features }}" {{ args }} --no-run --message-format=json; \
    if ($LASTEXITCODE -ne 0) { $json | Where-Object { $_ -match '"level":"error"' } | ForEach-Object { ($_ | ConvertFrom-Json).message.rendered } | Write-Host; exit 1 }; \
    $exes = $json | ForEach-Object { $_ | ConvertFrom-Json } | Where-Object { $_.reason -eq 'compiler-artifact' -and $_.executable } | ForEach-Object { $_.executable }; \
    $fail = 0; \
    foreach ($e in $exes) { Write-Host ("=== SDE -" + $chip + ": " + [IO.Path]::GetFileName($e) + " ==="); \
        & "{{ sde }}" "-$chip" -- $e --test-threads=1 --skip x86_fills_in_the_basics; if ($LASTEXITCODE -ne 0) { $fail = 1 } }; \
    exit $fail

# Dynamic instruction histogram for one already-built binary under emulation -
# the proof that EVEX encodings actually executed rather than the dispatcher
# quietly falling back to x86-v3. Look for the `*isa-set-AVX512*` rows.
# Usage: just sde-mix 2 target/release/deps/foo.exe

# Instruction-mix histogram for one binary under emulation (proves EVEX ran).
sde-mix tier exe:
    $chip = (just _sde-chip {{ tier }}); \
    $target = (Resolve-Path "{{ exe }}").Path; \
    & "{{ sde }}" "-$chip" -mix -omix target/sde-mix.out -- $target; \
    Select-String -Path target/sde-mix.out -Pattern '^\*isa-(set|ext)-' | ForEach-Object { $_.Line }

# Build thermite with a whole-tier baseline (NOT how the backend ships - this is
# for asm probes and for seeing what LLVM does when AVX-512 is unconditional).
# The explicit --target is required: bare RUSTFLAGS would also apply to build
# scripts and proc macros, which then execute here and die with 0xc000001d.

# cargo check thermite with a whole AVX-512 tier in the baseline (asm probes).
avx512-check tier="1" *args="":
    $f = @{ \
      "1" = "+avx512f,+avx512cd,+avx512bw,+avx512dq,+avx512vl"; \
      "2" = "+avx512f,+avx512cd,+avx512bw,+avx512dq,+avx512vl,+avx512vbmi,+avx512vbmi2,+avx512vnni,+avx512bitalg,+avx512vpopcntdq,+avx512ifma,+gfni,+vaes,+vpclmulqdq"; \
      "3" = "+avx512f,+avx512cd,+avx512bw,+avx512dq,+avx512vl,+avx512vbmi,+avx512vbmi2,+avx512vnni,+avx512bitalg,+avx512vpopcntdq,+avx512ifma,+gfni,+vaes,+vpclmulqdq,+avx512bf16" \
    }["{{ tier }}"]; \
    if (-not $f) { Write-Error "tier must be 1..3"; exit 1 }; \
    $env:RUSTFLAGS = "-C target-feature=$f"; \
    cargo check -p {{ pkg }} --features "{{ features }}" --target x86_64-pc-windows-msvc {{ args }}

# Cross-build the criterion bench binaries for the Pi and stage them under
# `target/pi-stage/bench/`. Uses `cross` (container build via Podman): criterion's
# `alloca` dep compiles C, so this needs the container's aarch64 C toolchain -
# unlike the pure-Rust test suite, which `pi-build` links with musl + rust-lld.
# The gnu-target binaries are dynamic; any 64-bit Raspberry Pi OS has the glibc
# they need. Prereqs: `cargo install cross`, podman machine running.
# On the Pi: `./math --noplot` etc.; `--test` runs one quick sanity iteration.
#
# Builds into `target/cross/` (its own CARGO_TARGET_DIR): the container's Linux
# x86_64 build-script artifacts would otherwise collide with the Windows ones in
# `target/`, and a stale one from a different image fails to exec. The glibc the
# binaries link against is pinned by the image in `Cross.toml` - see that file.
pi-bench-build:
    $env:CROSS_CONTAINER_ENGINE = "podman"; \
    $env:CARGO_TARGET_DIR = "target/cross"; \
    $json = cross bench -p {{ pkg }} --no-run --target aarch64-unknown-linux-gnu --no-default-features --features "std" --message-format=json; \
    if ($LASTEXITCODE -ne 0) { $json | Where-Object { $_ -match '"level":"error"' } | ForEach-Object { ($_ | ConvertFrom-Json).message.rendered } | Write-Host; exit 1 }; \
    $exes = $json | ForEach-Object { try { $_ | ConvertFrom-Json } catch { $null } } | Where-Object { $_ -and $_.reason -eq 'compiler-artifact' -and $_.executable -and $_.target.kind -contains 'bench' } | ForEach-Object { $_.executable }; \
    New-Item -ItemType Directory -Force target/pi-stage/bench | Out-Null; \
    $names = foreach ($e in $exes) { $base = [IO.Path]::GetFileName($e); $n = $base -replace '-[0-9a-f]+$', ''; Copy-Item -ErrorAction Stop ("target/cross/aarch64-unknown-linux-gnu/release/deps/" + $base) ("target/pi-stage/bench/" + $n); $n }; \
    $script = @('#!/bin/sh', '# NOTE: criterion needs --bench, else it runs a single "test mode" iteration', '# (prints Success, measures nothing). Extra args are forwarded, e.g. a filter.', 'set -u', 'fail=0') + ($names | ForEach-Object { "echo `"=== $_ ===`"; ./$_ --bench --noplot `"`$@`" || fail=1" }) + @('exit $fail'); \
    [IO.File]::WriteAllText((Join-Path (Get-Location) 'target/pi-stage/bench/run-bench.sh'), (($script -join "`n") + "`n")); \
    Write-Host ("Staged " + @($names).Count + " bench binaries + run-bench.sh in target/pi-stage/bench/: " + ($names -join ', ')); \
    Write-Host "Push: just pi-push pi@<host>  ->  on the Pi: cd ~/thermite-tests/bench && sh run-bench.sh"

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
# --- Dispatch audit ----------------------------------------------------------

# Scan compiled targets for kernels built without their backend's target
# features - the silent missing-`#[thermite::dispatch]` / `#[inline(always)]`
# bug. `cargo-thermite-audit` is a cargo subcommand (bin/cargo-thermite-audit):
# it builds each selected target with `--emit=llvm-ir` and audits the IR. Dev
# profile by default, which is the stricter audit (notes/thermite-audit).
# Usage: just audit -p thermite --features std --tests
#        just audit -p thermite-bvh --example pathtrace --features omm --release
#        just audit --ll target/release/examples/pathtrace.ll
audit *args:
    cargo run -p cargo-thermite-audit --release -- {{ args }}

# Install it on PATH so `cargo thermite-audit ...` works anywhere.
audit-install:
    cargo install --path bin/cargo-thermite-audit
