use thermite_dispatch::dispatch;

// Stub since we can't import `thermite` here.
pub mod thermite {
    pub enum InstructionSet {
        Scalar,
        X86V1,
        X86V2,
        X86V3,
        Neon,
        WasmSimd128,
    }

    pub trait HasIsa {
        const ISA: InstructionSet;
    }
}

// -- Free functions -------------------------------------------------------------

// Explicit dispatch ident, return value.
#[dispatch(V, thermite = "thermite")]
fn free_basic<V: thermite::HasIsa>(x: V) -> V {
    x
}

// Default `S` dispatch ident.
#[dispatch(thermite = "thermite")]
fn free_default_ident<S: thermite::HasIsa>(x: S) -> S {
    x
}

// Multiple arguments and a where clause.
#[dispatch(V, thermite = "thermite")]
fn free_multi_arg<V>(x: V, y: V, n: usize) -> V
where
    V: thermite::HasIsa,
{
    let _ = (y, n);
    x
}

// Additional non-dispatch generic threaded alongside the dispatch ident.
#[dispatch(V, thermite = "thermite")]
fn free_extra_generic<V: thermite::HasIsa, T: Copy>(_extra: T, x: V) -> V {
    x
}

// Lifetime parameter (early-bound — survives the trampoline boundary).
#[dispatch(V, thermite = "thermite")]
fn free_with_lifetime<'a, V: thermite::HasIsa>(x: &'a V) -> &'a V {
    x
}

// `unsafe` free function.
#[dispatch(V, thermite = "thermite")]
unsafe fn free_unsafe<V: thermite::HasIsa>(x: V) -> V {
    x
}

// -- Types ---------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
struct SimdA(f32);

impl thermite::HasIsa for SimdA {
    const ISA: thermite::InstructionSet = thermite::InstructionSet::X86V3;
}

#[derive(Clone, Copy, Default)]
struct SimdB(f64);

impl thermite::HasIsa for SimdB {
    const ISA: thermite::InstructionSet = thermite::InstructionSet::Scalar;
}

// -- #[dispatch] on an impl block — all receiver shapes ------------------------
//
// The `Self` ident is valid here because the macro has the full impl context.

#[dispatch(Self, thermite = "thermite")]
impl SimdA {
    // Value receiver.
    fn consume(self) -> Self {
        self
    }

    // Shared reference.
    fn inspect(&self) -> f32 {
        self.0
    }

    // Mutable reference.
    fn scale(&mut self, factor: f32) {
        self.0 *= factor;
    }

    // No receiver (associated function).
    fn create(val: f32) -> Self {
        Self(val)
    }

    // Extra generic alongside dispatch.
    fn transform<T: Into<f32>>(&self, val: T) -> f32 {
        self.0 + val.into()
    }

    // Opted out — runs without any dispatch wrapping.
    #[skip_dispatch]
    fn helper(&self) -> f32 {
        self.0 * 2.0
    }
}

// -- Method-level #[dispatch] — concrete type must be supplied -----------------
//
// The macro only sees the method, not the surrounding impl block, so `Self` is
// unavailable as the impl target. The concrete type name is supplied explicitly.

impl SimdB {
    // Value receiver.
    #[dispatch(SimdB, thermite = "thermite")]
    fn method_value(self) -> Self {
        self
    }

    // Shared reference.
    #[dispatch(SimdB, thermite = "thermite")]
    fn method_ref(&self) -> f64 {
        self.0
    }

    // Mutable reference.
    #[dispatch(SimdB, thermite = "thermite")]
    fn method_mut(&mut self) {
        self.0 += 1.0;
    }

    // Multiple plain arguments.
    #[dispatch(SimdB, thermite = "thermite")]
    fn method_args(&self, x: f64, y: f64) -> f64 {
        self.0 + x + y
    }

    // Generic argument on the method itself.
    #[dispatch(SimdB, thermite = "thermite")]
    fn method_generic<T: Into<f64>>(&self, val: T) -> f64 {
        self.0 + val.into()
    }

    // Non-dispatched method in the same impl block — no attribute, no changes.
    fn plain(&self) -> f64 {
        self.0
    }
}

fn main() {}
