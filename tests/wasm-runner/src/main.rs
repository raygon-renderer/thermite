//! Cargo test runner for `wasm32-wasip1` thermite test binaries.
//!
//! Cargo invokes this as `wasm-runner <test.wasm> [libtest args...]` (wired up via the
//! `CARGO_TARGET_WASM32_WASIP1_RUNNER` env var in the `just wasm-test` recipe). It runs the
//! module's `_start` under a WASI preview1 context with SIMD + relaxed-SIMD enabled, forwards
//! stdio and the libtest arguments, and propagates the guest exit code so cargo sees real
//! pass/fail results.

use anyhow::{Result, anyhow};
use wasmtime::{Config, Engine, Linker, Module, Store};
use wasmtime_wasi::WasiCtxBuilder;
use wasmtime_wasi::p1::{self, WasiP1Ctx};

fn main() -> Result<()> {
    let mut args = std::env::args();
    let _runner = args.next();
    let module_path = args
        .next()
        .ok_or_else(|| anyhow!("usage: wasm-runner <module.wasm> [args...]"))?;
    let guest_args: Vec<String> = args.collect();

    // SIMD is enabled by default; relaxed-SIMD must be opted in (thermite's wasm registers use
    // `*_relaxed_swizzle` / `*_relaxed_laneselect`).
    let mut config = Config::new();
    config.wasm_relaxed_simd(true);
    let engine = Engine::new(&config)?;

    let module = Module::from_file(&engine, &module_path)?;

    let mut linker: Linker<WasiP1Ctx> = Linker::new(&engine);
    p1::add_to_linker_sync(&mut linker, |t| t)?;

    // Guest argv: argv[0] = program name (the module path, conventionally), then the libtest args.
    let mut builder = WasiCtxBuilder::new();
    builder.inherit_stdio();
    builder.arg(&module_path);
    for a in &guest_args {
        builder.arg(a);
    }
    let wasi = builder.build_p1();

    let mut store = Store::new(&engine, wasi);
    let instance = linker.instantiate(&mut store, &module)?;
    let start = instance.get_typed_func::<(), ()>(&mut store, "_start")?;

    match start.call(&mut store, ()) {
        Ok(()) => Ok(()),
        Err(err) => {
            // WASI `proc_exit` surfaces as an `I32Exit` trap carrying the libtest exit code.
            if let Some(exit) = err.downcast_ref::<wasmtime_wasi::I32Exit>() {
                std::process::exit(exit.0);
            }
            Err(err.into())
        }
    }
}
