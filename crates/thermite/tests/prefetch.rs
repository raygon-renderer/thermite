//! [`NativeIsa::prefetch`] across every backend the host can run.
//!
//! A prefetch has no architecturally visible effect, so there is nothing to
//! diff against the scalar oracle; what *is* worth pinning is the contract:
//! every `LOCALITY`/`WRITE` combination must be instantiable on every backend,
//! and the address is only ever a hint -- null, dangling, unaligned and far
//! out-of-bounds pointers must all be accepted without faulting. That last
//! property is what lets hot loops feed it `base.wrapping_add(i)` with no
//! bounds check, so it deserves to be executed rather than assumed.

use thermite::simd::NativeIsa;

/// Issue all eight `<LOCALITY, WRITE>` combinations at `ptr`.
fn all_hints<S: NativeIsa>(ptr: *const u8) {
    S::prefetch::<0, false>(ptr);
    S::prefetch::<1, false>(ptr);
    S::prefetch::<2, false>(ptr);
    S::prefetch::<3, false>(ptr);
    S::prefetch::<0, true>(ptr);
    S::prefetch::<1, true>(ptr);
    S::prefetch::<2, true>(ptr);
    S::prefetch::<3, true>(ptr);
}

/// Every pointer shape a caller might hand us, valid or not.
fn every_pointer<S: NativeIsa>() {
    let data = [0u32; 64];
    let base = data.as_ptr().cast::<u8>();

    all_hints::<S>(base);
    all_hints::<S>(base.wrapping_add(3)); // unaligned
    all_hints::<S>(base.wrapping_add(4096)); // far past the end
    all_hints::<S>(base.wrapping_sub(4096)); // before the start
    all_hints::<S>(core::ptr::null());
    all_hints::<S>(core::ptr::dangling());
    all_hints::<S>(usize::MAX as *const u8); // unmapped, non-canonical on x86-64

    let boxed = alloc::boxed::Box::new(0u64);
    let freed = (&*boxed as *const u64).cast::<u8>();
    drop(boxed);
    all_hints::<S>(freed); // freed: still just a number to a prefetch
}

extern crate alloc;

#[test]
fn scalar() {
    every_pointer::<thermite::backend::scalar::Scalar>();
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn x86() {
    use thermite::backend::{x86_v1::X86V1, x86_v2::X86V2, x86_v3::X86V3};

    every_pointer::<X86V1>();
    every_pointer::<X86V2>();
    every_pointer::<X86V3>();

    // x86 always has the SSE prefetch family, and the scalar backend rides the
    // same host implementation rather than degrading to a no-op.
    assert!(X86V1::HAS_PREFETCH && X86V2::HAS_PREFETCH && X86V3::HAS_PREFETCH);
    assert_eq!(
        <thermite::backend::scalar::Scalar as NativeIsa>::HAS_PREFETCH,
        thermite::backend::prefetch::HAS_PREFETCH
    );
}

#[cfg(all(feature = "neon", target_arch = "aarch64"))]
#[test]
fn neon() {
    every_pointer::<thermite::backend::neon::Neon>();

    // `prfm` is baseline aarch64, so both aarch64 backends issue a real hint.
    assert!(<thermite::backend::neon::Neon as NativeIsa>::HAS_PREFETCH);
    assert!(<thermite::backend::scalar::Scalar as NativeIsa>::HAS_PREFETCH);
}

#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
#[test]
fn wasm() {
    every_pointer::<thermite::backend::wasm::Wasm>();

    // WebAssembly has no prefetch hint: the calls compile away entirely.
    assert!(!<thermite::backend::wasm::Wasm as NativeIsa>::HAS_PREFETCH);
}
