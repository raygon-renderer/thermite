//! Integration tests for `thermite::dispatch_dyn!`.
//!
//! Lives in its own crate because the macro pulls in `thermite` (which itself depends
//! on `thermite-dispatch`), and proc-macro crates can't depend on the crates they're
//! tested against.
