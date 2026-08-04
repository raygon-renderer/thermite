//! `HasIsa::Native` - the backend type reachable from any vector.
//!
//! Two properties are load-bearing and neither is visible to a functional
//! test, so they are asserted directly here:
//!
//! 1. **`Native` names the backend that actually executes the vector.** A
//!    full-width slot reports its own backend; an emulated `ArrayRegister`
//!    slot reports whatever its *lanes* run on, which is the owning backend
//!    for `ArrayRegister<F32x4V1, 2>` but `Scalar` for the sub-native
//!    `ArrayRegister<i16, 2>` shared by every backend. A slot that silently
//!    forwarded the wrong one would hand callers a plausible-but-wrong
//!    register budget and alignment.
//! 2. **`ISA` and `Native::ISA` agree.** `CoreRegister::ISA` defaults to
//!    `<Self::NativeIsa as HasIsa>::ISA`, so a register that overrides one but
//!    not the other would drift without any test noticing.

use thermite::backend::scalar::Scalar;
use thermite::prelude::*;
// The `f32x4<S>` / `f32xN<S>` aliases and `HasIsa` are not in the prelude.
use thermite::simd::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use thermite::backend::{x86_v1::X86V1, x86_v2::X86V2, x86_v3::X86V3};

/// Compile-time type equality: instantiating this is the assertion.
fn assert_native<V, S>()
where
    V: HasIsa<Native = S>,
    S: NativeIsa,
{
}

/// `V::ISA` and `V::Native::ISA` must be the same value.
fn assert_isa_agrees<V: HasIsa>() {
    assert_eq!(
        V::ISA,
        <V::Native as HasIsa>::ISA,
        "{:?} disagrees with its Native's ISA {:?}",
        V::ISA,
        <V::Native as HasIsa>::ISA
    );
}

#[test]
fn scalar_backend_names_itself() {
    assert_native::<Scalar, Scalar>();
    assert_native::<f32xN<Scalar>, Scalar>();
    assert_native::<f64x4<Scalar>, Scalar>();
    assert_native::<i16x2<Scalar>, Scalar>();

    assert_isa_agrees::<Scalar>();
    assert_isa_agrees::<f32xN<Scalar>>();
    assert_isa_agrees::<f64x4<Scalar>>();
}

/// Native-width and fixed-width slots carry their own backend, including the
/// `ArrayRegister`-emulated wide slots (`f32x16<X86V1>` is four `F32x4V1`s, so
/// it is still v1 code) and the `ReducedRegister`-backed narrow ones.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn x86_slots_name_their_own_backend() {
    assert_native::<X86V1, X86V1>();
    assert_native::<X86V2, X86V2>();
    assert_native::<X86V3, X86V3>();

    // Native width.
    assert_native::<f32xN<X86V1>, X86V1>();
    assert_native::<f32xN<X86V2>, X86V2>();
    assert_native::<f32xN<X86V3>, X86V3>();

    // Fixed width at, below, and above the native width.
    assert_native::<f32x4<X86V1>, X86V1>();
    assert_native::<f32x2<X86V1>, X86V1>();
    assert_native::<f32x16<X86V1>, X86V1>(); // ArrayRegister<F32x4V1, 4>
    assert_native::<f64x8<X86V2>, X86V2>();
    assert_native::<u8x16<X86V3>, X86V3>();
    assert_native::<f32x4<X86V3>, X86V3>(); // half-width on a 256-bit backend

    // Masks and the integer families ride along.
    assert_native::<u32x8<X86V3>, X86V3>();
    assert_native::<i64x2<X86V2>, X86V2>();
}

/// The documented exception: sub-native slots that are `ArrayRegister`s of
/// *scalar* lanes on every backend. `Scalar` is the correct answer for what
/// executes them - and the reason `Native` must not be read as "what machine
/// am I on".
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn shared_scalar_slots_report_scalar() {
    assert_native::<i16x2<X86V3>, Scalar>();
    assert_native::<u16x2<X86V3>, Scalar>();
    assert_native::<i8x2<X86V3>, Scalar>();
    assert_native::<u8x2<X86V3>, Scalar>();

    // Same type on every tier, so the answer cannot be tier-specific.
    assert_native::<i16x2<X86V1>, Scalar>();
    assert_native::<i16x2<Scalar>, Scalar>();

    assert_eq!(<i16x2<X86V3> as HasIsa>::ISA, thermite::isa::InstructionSet::Scalar);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn isa_agrees_with_native_isa() {
    assert_isa_agrees::<X86V1>();
    assert_isa_agrees::<X86V2>();
    assert_isa_agrees::<X86V3>();

    assert_isa_agrees::<f32xN<X86V1>>();
    assert_isa_agrees::<f32xN<X86V2>>();
    assert_isa_agrees::<f32xN<X86V3>>();
    assert_isa_agrees::<f32x16<X86V1>>();
    assert_isa_agrees::<f64x2<X86V2>>();
    assert_isa_agrees::<u8x16<X86V3>>();
    assert_isa_agrees::<i16x2<X86V3>>();
}

/// The point of the whole thing: per-ISA properties reachable from a bound
/// that never mentions the backend.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn properties_reachable_from_a_vector_only_bound() {
    use generic_array::typenum::Unsigned;

    fn native_lanes<V: FloatVector>() -> usize {
        <<V::Native as NativeIsa>::Native32Width as Unsigned>::USIZE
    }

    fn registers<V: FloatVector>() -> usize {
        <<V::Native as NativeIsa>::Registers as Unsigned>::USIZE
    }

    assert_eq!(native_lanes::<f32xN<X86V3>>(), 8);
    assert_eq!(native_lanes::<f32xN<X86V1>>(), 4);
    assert_eq!(native_lanes::<f32xN<Scalar>>(), 1);

    assert_eq!(registers::<f32xN<X86V1>>(), 8);
    assert_eq!(registers::<f32xN<X86V3>>(), 16);
}
