//! ISA capability traits and the SIMD type-name lattice.
//!
//! This module defines the trait stack that names every concrete register
//! and vector type a backend provides, plus the `Vector<...>` type aliases
//! that map "I want an `f32x8` on whatever backend `S` is" to the right
//! concrete type. Most users only ever interact with the type aliases; the
//! traits are mainly building blocks for backends and for generic library
//! code that needs to spell out the relationships between register widths
//! and element types.
//!
//! # The ISA stack
//!
//! ```text
//! HasIsa            -- announces which `InstructionSet` a backend targets, and
//!   |                  names the backend type itself as `Native`
//!   |
//! NativeIsa         -- adds register count, native 32/64-bit widths, alignment,
//!   |                  and the `enable_denormals` / `zeroupper` knobs
//! NativeSimd        -- names the "native-width" register types for each
//!   |                  supported element family (currently the
//!   |                  `f32xN` / `i32xN` / `u32xN` group and the
//!   |                  `f64xN` / `i64xN` / `u64xN` group)
//! Simd              -- names every fixed-width register type from x2 to x16,
//!   |                  for f32/i32/u32/f64/i64/u64 and usize
//! SizedSimd<F,I,U>  -- one element family at a time (32-bit OR 64-bit), with
//!   |                  short names: `fxN`, `ixN`, `uxN`, `fx2`, `fx4`, ...
//! FloatSimd<F>      -- like SizedSimd, but inferred from the float type alone
//! ```
//!
//! [`Simd3A`] and [`Simd3`] are extensions on top of [`Simd`] that add 3-lane
//! register types (`f32x3A` for "alpha-padded" 3-in-4 storage, `f32x3` for
//! true 3-lane representations on backends that have them, such as GPUs).
//!
//! # The Vector mirror
//!
//! Each `*Simd` trait above has a `*Vectors` companion that exposes the same
//! types as `Vector<...>` rather than raw registers. The `*WithRegisters`
//! marker companions on top of that bridge the two layers, so generic code
//! can move between [`Vector`] and its underlying register without losing
//! associated-type information:
//!
//! ```text
//! NativeSimdVectors -> NativeSimdVectorsWithRegisters
//! SimdVectors       -> SimdVectorsWithRegisters
//! Simd3AVectors     -> Simd3AVectorsWithRegisters
//! Simd3Vectors      -> Simd3VectorsWithRegisters
//! ```
//!
//! # Type aliases
//!
//! The free `pub type f32x4<S> = Vector<<S as Simd>::f32x4>;` aliases and
//! their friends are the recommended entry point. Pick a backend `S` (for
//! example `X86V3`, `Scalar`, or a generic `S: Simd` parameter), then write
//! `f32x4<S>` and you have a fully-typed vector.

#![allow(non_camel_case_types)]

use core::{hash::Hash, marker::PhantomData};

use generic_array::{
    ArrayLength,
    typenum::{U1, U2, U3, U4, U8, U16},
};

use crate::{
    Vector,
    element::{FloatElementWithBits, USize},
    isa::InstructionSet,
    register::{
        BitCastRegister, CastRegister, ConcatRegister, ExtendRegister, FloatRegister, FullyInteroperable,
        IndexableRegister, Lanes, LinAlg3Register, LinAlg4Register, PackedFloatRegister, Register, Sad16Register,
        Sad32Register, Sad64Register, SignedIntegerRegister, UnsignedIntegerRegister,
        reduced::ReducedRegister,
        well_formed::{
            WellFormedFloatElement, WellFormedFloatRegister, WellFormedSignedIntegerElement,
            WellFormedSignedIntegerRegister, WellFormedUnsignedIntegerElement, WellFormedUnsignedIntegerRegister,
        },
    },
};

/// Zero-sized type with a `repr(align(16))`. Used as a field marker to force
/// a containing struct up to 16-byte alignment without occupying any bytes
/// of its own. Picked by backends whose native vectors live in 128-bit
/// registers (SSE, NEON).
#[doc(hidden)]
#[repr(align(16))]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Align16;

/// Zero-sized type with a `repr(align(32))`. Used as a field marker to force
/// 32-byte alignment for containers of 256-bit registers (AVX/AVX2).
#[doc(hidden)]
#[repr(align(32))]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Align32;

/// Zero-sized type with a `repr(align(64))`. Used as a field marker to force
/// 64-byte alignment for containers of 512-bit registers (AVX-512).
#[doc(hidden)]
#[repr(align(64))]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Align64;

/// Root of the ISA trait stack: every type that can be traced back to a
/// backend advertises which [`InstructionSet`] it targets via this trait.
///
/// Implementors include the per-tier x86 types (`X86V1`/`X86V2`/`X86V3`),
/// the `Scalar` fallback backend, `SPIRV`, and the WASM backend, but also
/// every [`Vector`] and composite vector type ([`GenericVector`] requires
/// `HasIsa`). The `dispatch!` macro reads [`ISA`](Self::ISA) to pick the
/// right specialization at runtime.
///
/// [`Native`](Self::Native) additionally names the backend type itself, so
/// generic code holding only a vector can reach the per-ISA properties on
/// [`NativeIsa`] -- register count, native lane widths, the
/// [`NativeAlignment`](NativeIsa::NativeAlignment) marker, `prefetch`, and
/// the denormal toggles -- without threading a separate `S: Simd` parameter.
///
/// # `Native` describes the value, not the host
///
/// `Native` answers "what executes *this* vector", which is not always the
/// machine you are running on. A sub-native slot such as `i16x2<S>` is an
/// [`ArrayRegister`](crate::register::array::ArrayRegister) of scalar lanes on every
/// backend, so its `Native` is `Scalar` even on an AVX2 host -- correct for
/// that vector, but wrong if you wanted the host's register budget. Tuning
/// decisions that are about the *machine* (unroll factors, register
/// pressure) should still read the dispatched `S`, not `V::Native`.
pub trait HasIsa {
    /// The backend this type executes on.
    ///
    /// Backend types name themselves (`type Native = Self`); registers name
    /// the backend that owns them; [`Vector`] and the composite vector types
    /// forward their inner register's. Emulated registers forward the
    /// register they are built from, so an `ArrayRegister<F32x4V1, 2>` still
    /// reports `X86V1` while an `ArrayRegister<i16, 2>` reports `Scalar`.
    type Native: NativeIsa;

    /// The instruction set this backend implements.
    ///
    /// Defaults to the ISA of [`Native`](Self::Native), which is the right
    /// answer for everything except the backend types themselves -- their
    /// `Native` is `Self`, so they must state it explicitly or the default
    /// would recurse.
    const ISA: InstructionSet = <Self::Native as HasIsa>::ISA;
}

/// Properties of a runnable native ISA: register count, native widths,
/// alignment marker, and the CPU-state toggles ([`disable_denormals`](Self::disable_denormals),
/// [`enable_denormals`](Self::enable_denormals), [`zeroupper`](Self::zeroupper)).
///
/// Every concrete backend type that can actually execute SIMD code
/// implements this. The `Scalar` backend implements it with `U1` widths so
/// generic code can be written uniformly.
#[rustfmt::skip]
pub trait NativeIsa: HasIsa + core::fmt::Debug + Clone + Copy + PartialEq + Eq + Hash {
    /// Number of architectural SIMD registers the ISA exposes. For example,
    /// `U8` for legacy SSE (xmm0-xmm7), `U16` for SSE4.2/AVX2, `U32` for
    /// AVX-512. Used by some inlining heuristics in `transform::map_inplace`.
    type Registers: ArrayLength;

    /// Lane count of the widest natively-supported 32-bit-element register.
    /// `U4` on SSE (128 bits), `U8` on AVX2 (256 bits), `U16` on AVX-512.
    type Native32Width: Lanes;

    /// Lane count of the widest natively-supported 64-bit-element register.
    /// `U2` on SSE, `U4` on AVX2, `U8` on AVX-512.
    type Native64Width: Lanes;

    /// Lane count of the widest natively-supported 16-bit-element register: `U8` on
    /// SSE-class backends (128-bit) and WASM, `U16` on AVX2 (256-bit), `U32` on a future
    /// AVX-512 backend (512-bit).
    type Native16Width: Lanes;

    /// Lane count of the widest natively-supported 8-bit-element register: `U16` on
    /// SSE-class backends (128-bit) and WASM, `U32` on AVX2 (256-bit), `U64` on a future
    /// AVX-512 backend (512-bit).
    type Native8Width: Lanes;

    /// Opaque type with the minimum required alignment for native SIMD types.
    ///
    /// Include this as a field in structs that contain native SIMD types to ensure
    /// proper alignment of the containing struct.
    ///
    /// Because this cannot be instantiated easier (despite the `Default` impl),
    /// I'd recommend using a zero-length array of this type, like `[<S as NativeSimd>::NativeAlignment; 0]`,
    /// then it's easy to construction with `[]` and it doesn't take up any real space in the struct.
    ///
    /// You may also want `#[repr(C)]` to ensure the order of the fields is preserved. Note that only the first field
    /// is guaranteed to have the correct alignment, so the SIMD fields should be placed first in the struct, before any other fields.
    type NativeAlignment: Sized + Default + Copy + Ord + Hash + Send + Sync + Unpin + core::panic::UnwindSafe + core::panic::RefUnwindSafe + core::fmt::Debug + 'static;

    /// Attempt to disable slow subnormal/denormal fallback handling on the current ISA.
    /// This will try to set the behavior such that denormal values are flushed to zero,
    /// both as input and as output.
    ///
    /// This will return `Err` if disabling denormals isn't supported in software, and
    /// upon success will return `Ok(was_enabled)`, indicating if denormals
    /// were previously enabled.
    ///
    /// NOTE: x86 has two flags for this, but this method will treat them as one.
    ///
    /// # Safety
    ///
    /// This may modify low-level CPU registers that control the behavior of the entire
    /// thread. Use with caution.
    unsafe fn disable_denormals() -> Result<bool, UnsupportedError> { Err(UnsupportedError) }

    /// Attempt to enable subnormal/denormal handling on the current ISA. This will
    /// try to set the behavior such that denormal values are correctly handled,
    /// even at massive performance costs.
    ///
    /// This will return `Err` if enabling denormals isn't supported in software.
    ///
    /// NOTE: x86 has two flags for this, but this method will treat them as one.
    ///
    /// # Safety
    ///
    /// This may modify low-level CPU registers that control the behavior of the
    /// entire thread. Use with caution.
    unsafe fn enable_denormals() -> Result<(), UnsupportedError> { Err(UnsupportedError) }

    /// Whether [`prefetch`](Self::prefetch) issues a real instruction on this backend.
    /// `false` means it compiles away to nothing (wasm, SPIR-V, or any host without a
    /// software-prefetch hint), so callers can `if const { S::HAS_PREFETCH }` around
    /// address arithmetic that only exists to feed it.
    const HAS_PREFETCH: bool = false;

    /// Hint that the cache line containing `ptr` should be fetched into cache now,
    /// ahead of the access that will actually need it -- the classic way to overlap a
    /// DRAM miss with useful work (walk a pointer chase, prefetch the *next* node
    /// while decoding the current one).
    ///
    /// `LOCALITY` says how long the line is worth keeping, following the LLVM /
    /// `__builtin_prefetch` convention: `0` = none (streaming, evict ASAP), `1` = low,
    /// `2` = moderate, `3` = high (keep it in every cache level). Anything outside
    /// `0..=3` is a compile error. `WRITE` is `true` when the line is about to be
    /// written, so the hardware can fetch it in an exclusive state and skip the later
    /// read-for-ownership; backends with no write form use the read form instead.
    ///
    /// | `LOCALITY` | x86 (read / write) | aarch64 (read / write) |
    /// |---|---|---|
    /// | 3 | `prefetcht0` / `prefetchw` | `prfm pldl1keep` / `prfm pstl1keep` |
    /// | 2 | `prefetcht1` / `prefetchw`(t1) | `prfm pldl2keep` / `prfm pstl2keep` |
    /// | 1 | `prefetcht2` / `prefetchw`(t1) | `prfm pldl3keep` / `prfm pstl3keep` |
    /// | 0 | `prefetchnta` | `prfm pldl1strm` / `prfm pstl1strm` |
    ///
    /// (The x86 write forms need `prfchw`/`prefetchwt1` enabled; without them LLVM
    /// lowers them back to the read form, which is fine for a hint.)
    ///
    /// This is safe for **any** pointer -- dangling, null, unaligned or far out of
    /// bounds -- because a prefetch never dereferences the address and cannot fault.
    /// Branchless address arithmetic is therefore fine and preferred:
    ///
    /// ```
    /// use thermite::backend::scalar::Scalar;
    /// use thermite::simd::NativeIsa;
    ///
    /// let data = [0u32; 64];
    /// let next = data.as_ptr().wrapping_add(16); // no bounds check needed
    /// Scalar::prefetch::<3, false>(next.cast());
    /// ```
    ///
    /// The default is a no-op, like every other capability on this trait: a backend
    /// opts in by routing to `arch::prefetch` (see [`crate::backend::prefetch`] for
    /// the per-target lowering).
    #[inline(always)]
    fn prefetch<const LOCALITY: u8, const WRITE: bool>(ptr: *const u8) {
        const { assert!(LOCALITY <= 3, "prefetch LOCALITY must be 0 (non-temporal) ..= 3 (keep in every cache level)") };
        let _ = ptr;
    }

    /// If supported, zero out the upper parts of all SIMD registers. Not all architectures
    /// support this, and will return `false` if not, otherwise `true` if it succeeded.
    ///
    /// # Safety
    ///
    /// This is a hardware intrinsic that will intentionally destroy parts of all SIMD
    /// registers. Generally don't use this unless you know exactly what you're doing.
    unsafe fn zeroupper() -> bool { false }
}

/// Returned by [`NativeIsa::disable_denormals`] / [`NativeIsa::enable_denormals`]
/// when the backend has no way to influence the denormal flag (for example,
/// on the `Scalar` backend or under WASM).
pub struct UnsupportedError;

/// RAII guard for temporarily disabling denormals at the CPU level.
///
/// Construct one at the start of a region that is performance-sensitive to
/// subnormal inputs, perform the work, and let it drop at the end of the
/// scope -- on `Drop` the previous flag state is restored. If the backend
/// did not support disabling denormals in the first place, the guard
/// becomes a no-op (you can check with [`is_disabled`](Self::is_disabled)).
///
/// ```ignore
/// // Safety: we are not relying on subnormal-aware float behavior here.
/// let _guard = unsafe { DisableDenormals::<S>::disable_denormals() };
/// hot_loop(data);
/// // guard drops here, restoring the previous state
/// ```
pub struct DisableDenormals<S: NativeIsa>(Result<bool, UnsupportedError>, PhantomData<S>);

impl<S: NativeIsa> DisableDenormals<S> {
    /// # Safety
    ///
    /// See [`NativeIsa::disable_denormals`]
    #[inline(always)]
    #[allow(clippy::self_named_constructors)]
    pub unsafe fn disable_denormals() -> Self {
        unsafe { Self(S::disable_denormals(), PhantomData) }
    }

    /// Returns true if denormals were previously enabled before disabling them.
    #[inline(always)]
    pub fn was_enabled(&self) -> bool {
        matches!(self.0, Ok(true))
    }

    /// Returns true if denormals are currently disabled. (i.e. if the disabling was successful)
    #[inline(always)]
    pub fn is_disabled(&self) -> bool {
        self.0.is_ok()
    }
}

impl<S: NativeIsa> Drop for DisableDenormals<S> {
    #[inline(always)]
    fn drop(&mut self) {
        if let Ok(true) = self.0 {
            unsafe { _ = S::enable_denormals() };
        }
    }
}

/// Names the "native-width" register types of an ISA.
///
/// The `f32xN`/`i32xN`/`u32xN` group is sized to
/// [`NativeIsa::Native32Width`]; the `f64xN`/`i64xN`/`u64xN` group to
/// [`NativeIsa::Native64Width`]. These are the register types where the
/// backend's hardware is most directly expressed (256-bit on AVX2, 128-bit
/// on SSE, 1-lane on Scalar/SPIR-V), so generic code that wants to follow
/// the host's natural width should bind to these. Additional native
/// element families may be added in the future without breaking existing
/// implementors.
///
/// Use [`Simd`] when you need a specific width by name instead.
#[rustfmt::skip]
pub trait NativeSimd: NativeIsa {
    /// Native-width `f32` register. Lanes = [`Native32Width`](NativeIsa::Native32Width).
    type f32xN: FullyInteroperable<Self::i32xN, Self::u32xN, Lanes = Self::Native32Width, Element = f32, Unsigned = Self::u32xN, Signed = Self::i32xN>
        + WellFormedFloatRegister<Bits = Self::u32xN, SignedBits = Self::i32xN> + IndexableRegister<Self::u32xN>;
    /// Native-width signed 32-bit integer register. Same lane count as `f32xN`.
    type i32xN: FullyInteroperable<Self::f32xN, Self::u32xN, Lanes = Self::Native32Width, Element = i32, Unsigned = Self::u32xN, Signed = Self::i32xN>
        + WellFormedSignedIntegerRegister + IndexableRegister<Self::u32xN>;
    /// Native-width unsigned 32-bit integer register. Same lane count as `f32xN`.
    type u32xN: FullyInteroperable<Self::f32xN, Self::i32xN, Lanes = Self::Native32Width, Element = u32, Unsigned = Self::u32xN, Signed = Self::i32xN>
        + Sad64Register<Self::u64xN>
        + WellFormedUnsignedIntegerRegister + IndexableRegister<Self::u32xN>;

    /// Native-width `f64` register. Lanes = [`Native64Width`](NativeIsa::Native64Width),
    /// typically half of [`Native32Width`](NativeIsa::Native32Width).
    type f64xN: FullyInteroperable<Self::i64xN, Self::u64xN, Lanes = Self::Native64Width, Element = f64, Unsigned = Self::u64xN, Signed = Self::i64xN>
        + WellFormedFloatRegister<Bits = Self::u64xN, SignedBits = Self::i64xN> + IndexableRegister<Self::u64xN>;
    /// Native-width signed 64-bit integer register. Same lane count as `f64xN`.
    type i64xN: FullyInteroperable<Self::f64xN, Self::u64xN, Lanes = Self::Native64Width, Element = i64, Unsigned = Self::u64xN, Signed = Self::i64xN>
        + WellFormedSignedIntegerRegister + IndexableRegister<Self::u64xN>;
    /// Native-width unsigned 64-bit integer register. Same lane count as `f64xN`.
    type u64xN: FullyInteroperable<Self::f64xN, Self::i64xN, Lanes = Self::Native64Width, Element = u64, Unsigned = Self::u64xN, Signed = Self::i64xN>
        + WellFormedUnsignedIntegerRegister + IndexableRegister<Self::u64xN>;

    /// Native-width signed 16-bit register. Lanes = [`Native16Width`](NativeIsa::Native16Width).
    /// The 16-bit families have no floating-point partner (there is no `f16`), so these are
    /// modeled on the float-free `usize`/integer slots rather than the float-coupled ones.
    type i16xN: WellFormedSignedIntegerRegister<Element = i16, Lanes = Self::Native16Width>
        + IndexableRegister<Self::u16xN>;
    /// Native-width unsigned 16-bit register. Same lane count as `i16xN`.
    type u16xN: WellFormedUnsignedIntegerRegister<Element = u16, Lanes = Self::Native16Width>
        + IndexableRegister<Self::u16xN>
        // SAD lines up with the other native widths: pairs -> u32xN, quads -> u64xN.
        + Sad32Register<Self::u32xN> + Sad64Register<Self::u64xN>;

    /// Native-width signed 8-bit register. Lanes = [`Native8Width`](NativeIsa::Native8Width).
    type i8xN: WellFormedSignedIntegerRegister<Element = i8, Lanes = Self::Native8Width>
        + IndexableRegister<Self::u8xN>;
    /// Native-width unsigned 8-bit register. Same lane count as `i8xN`.
    type u8xN: WellFormedUnsignedIntegerRegister<Element = u8, Lanes = Self::Native8Width>
        + IndexableRegister<Self::u8xN>
        // SAD at native width: half/quarter/eighth the lanes, one step down the xN ladder
        // each time. On AVX2 this is the 256-bit `u8x32`, so `_mm256_sad_epu8` is reachable
        // even though there is no fixed-width `u8x32` slot.
        + Sad16Register<Self::u16xN> + Sad32Register<Self::u32xN> + Sad64Register<Self::u64xN>;
}

/// Helper traits bundling several register / vector relationships into one
/// bound, so trait-heavy signatures elsewhere stay readable.
///
/// These are auto-implemented for any type that satisfies their component
/// bounds; you never implement them by hand, you just write them in `where`
/// clauses.
pub mod helpers {
    use super::*;

    /// Bundle: a register that accepts `usize`-, `u32`- and `u64`-typed index
    /// registers of the same lane count for gather/scatter.
    ///
    /// Used as a single `IndexedBy<S::usizex4, S::u32x4, S::u64x4>` bound on
    /// e.g. an `f32x4` register, rather than three separate
    /// `IndexableRegister<...>` bounds.
    pub trait IndexedBy<
        USIZE: UnsignedIntegerRegister<Lanes = Self::Lanes>,
        U32: UnsignedIntegerRegister<Lanes = Self::Lanes>,
        U64: UnsignedIntegerRegister<Lanes = Self::Lanes>,
    >: Register + IndexableRegister<USIZE> + IndexableRegister<U32> + IndexableRegister<U64>
    {
    }

    impl<
        USIZE: UnsignedIntegerRegister<Lanes = Self::Lanes>,
        U32: UnsignedIntegerRegister<Lanes = Self::Lanes>,
        U64: UnsignedIntegerRegister<Lanes = Self::Lanes>,
        R,
    > IndexedBy<USIZE, U32, U64> for R
    where
        R: Register + IndexableRegister<USIZE> + IndexableRegister<U32> + IndexableRegister<U64>,
    {
    }

    /// Bundle: a register that can be zero-extended from `FROM`, including
    /// its mask type. Used to chain register-width promotions through
    /// generic code without re-stating the mask bound at every level.
    pub trait FullExtendRegister<FROM: Register>:
        Register<Mask: ExtendRegister<FROM::Mask>> + ExtendRegister<FROM>
    {
    }

    /// Bundle: a register that is the concatenation of two `HALF` registers,
    /// including its mask type. Implies [`FullExtendRegister`] and
    /// [`ConcatRegister`], so a single bound covers both halves-into-whole
    /// and whole-into-halves moves.
    pub trait FullConcatRegister<HALF: Register>:
        Register<Mask: ConcatRegister<HALF::Mask>> + FullExtendRegister<HALF> + ConcatRegister<HALF>
    {
    }

    impl<R: Register, F: Register> FullExtendRegister<F> for R where
        R: Register<Mask: ExtendRegister<F::Mask>> + ExtendRegister<F>
    {
    }
    impl<R: Register, H: Register> FullConcatRegister<H> for R where
        R: Register<Mask: ConcatRegister<H::Mask>> + FullExtendRegister<H> + ConcatRegister<H>
    {
    }

    use crate::element::float::spec::{Bf16, Fp8E4M3, Fp8E5M2, Fp16, Fp16Fast};

    /// Bundle: a `u16` register that transcodes all three 16-bit float formats - [`Fp16`],
    /// [`Fp16Fast`], and [`Bf16`] - to and from the `f32` register `F` (same lane count). A single
    /// `PackedF16Register<S::f32xK>` bound on a `u16xK` slot replaces three separate
    /// `PackedFloatRegister<{Fp16,Fp16Fast,Bf16}, ...>` bounds. Blanket-implemented, so any
    /// register carrying all three (the generic defaults, or F16C hardware overrides) satisfies it
    /// automatically.
    pub trait PackedF16Register<F>:
        PackedFloatRegister<Fp16, F> + PackedFloatRegister<Fp16Fast, F> + PackedFloatRegister<Bf16, F>
    where
        F: FloatRegister<Element = f32, Lanes = Self::Lanes, Bits: CastRegister<Self>>,
    {
    }

    impl<R, F> PackedF16Register<F> for R
    where
        F: FloatRegister<Element = f32, Lanes = R::Lanes, Bits: CastRegister<R>>,
        R: PackedFloatRegister<Fp16, F> + PackedFloatRegister<Fp16Fast, F> + PackedFloatRegister<Bf16, F>,
    {
    }

    /// Vector-level mirror of [`PackedF16Register`]: a `u16` vector that transcodes all three
    /// 16-bit float formats - [`Fp16`], [`Fp16Fast`], and [`Bf16`] - to and from the `f32` vector
    /// `F`. A single `PackedF16Vector<S::f32xK>` bound on a `u16xK` vector slot replaces three
    /// separate `PackedFloatVector<{Fp16,Fp16Fast,Bf16}, ...>` bounds. Blanket-implemented.
    pub trait PackedF16Vector<F>:
        PackedFloatVector<Fp16, F> + PackedFloatVector<Fp16Fast, F> + PackedFloatVector<Bf16, F>
    {
    }

    impl<V, F> PackedF16Vector<F> for V where
        V: PackedFloatVector<Fp16, F> + PackedFloatVector<Fp16Fast, F> + PackedFloatVector<Bf16, F>
    {
    }

    /// Bundle: a `u8` register that transcodes both fp8 formats - OCP [`Fp8E4M3`] and [`Fp8E5M2`] -
    /// to and from the `f32` register `F` (same lane count). A single `PackedF8Register<S::f32xK>`
    /// bound on a `u8xK` slot replaces two `PackedFloatRegister<{Fp8E4M3,Fp8E5M2}, ...>` bounds.
    /// Blanket-implemented; fp8 has no hardware transcoder, so every backend's `u8` registers
    /// satisfy it via the generic branchless defaults.
    pub trait PackedF8Register<F>: PackedFloatRegister<Fp8E4M3, F> + PackedFloatRegister<Fp8E5M2, F>
    where
        F: FloatRegister<Element = f32, Lanes = Self::Lanes, Bits: CastRegister<Self>>,
    {
    }

    impl<R, F> PackedF8Register<F> for R
    where
        F: FloatRegister<Element = f32, Lanes = R::Lanes, Bits: CastRegister<R>>,
        R: PackedFloatRegister<Fp8E4M3, F> + PackedFloatRegister<Fp8E5M2, F>,
    {
    }

    /// Vector-level mirror of [`PackedF8Register`]: a `u8` vector that transcodes both fp8 formats
    /// to and from the `f32` vector `F`. Blanket-implemented.
    pub trait PackedF8Vector<F>: PackedFloatVector<Fp8E4M3, F> + PackedFloatVector<Fp8E5M2, F> {}

    impl<V, F> PackedF8Vector<F> for V where V: PackedFloatVector<Fp8E4M3, F> + PackedFloatVector<Fp8E5M2, F> {}

    /// Vector-level mirror of [`IndexedBy`]: a vector that accepts `usize`-,
    /// `u32`- and `u64`-typed index vectors of the same lane count for
    /// gather/scatter.
    pub trait VectorIndexedBy<
        USIZE: UnsignedIntegerVector<Lanes = Self::Lanes>,
        U32: UnsignedIntegerVector<Lanes = Self::Lanes>,
        U64: UnsignedIntegerVector<Lanes = Self::Lanes>,
    >: GenericVector + IndexableVector<USIZE> + IndexableVector<U32> + IndexableVector<U64>
    {
    }

    impl<
        USIZE: UnsignedIntegerVector<Lanes = Self::Lanes>,
        U32: UnsignedIntegerVector<Lanes = Self::Lanes>,
        U64: UnsignedIntegerVector<Lanes = Self::Lanes>,
        R,
    > VectorIndexedBy<USIZE, U32, U64> for R
    where
        R: GenericVector + IndexableVector<USIZE> + IndexableVector<U32> + IndexableVector<U64>,
    {
    }
}

use self::helpers::*;

/// Names every fixed-width SIMD register a backend offers.
///
/// Where [`NativeSimd`] gives you only the host's "natural" register width,
/// `Simd` enumerates the full grid: lane counts of 2, 4, 8, and 16 for both
/// 32-bit and 64-bit element families, plus the platform-pointer-sized
/// `usizexN` group. Register widths that exceed the host's native width are
/// emulated -- a `Simd::f64x16` on SSE2 is an `ArrayRegister` of four
/// 2-lane `f64x2` registers -- but they expose the same interface, so
/// generic code never has to special-case the "register too wide" path.
///
/// `Simd` is fine to bind directly when you want a specific width by name;
/// [`NativeSimd`], [`SizedSimd`], [`FloatSimd`], and the free type aliases
/// (`f32x4<S>`, etc.) at the module level are alternative entry points that
/// can be more convenient in different contexts.
#[rustfmt::skip]
pub trait Simd: NativeSimd {type usizex2: WellFormedUnsignedIntegerRegister<Element = crate::element::USize, Lanes = U2>
        + IndexedBy<Self::usizex2, Self::u32x2, Self::u64x2> + FullConcatRegister<USize>;
    type usizex4: WellFormedUnsignedIntegerRegister<Element = crate::element::USize, Lanes = U4>
        + IndexedBy<Self::usizex4, Self::u32x4, Self::u64x4> + FullConcatRegister<Self::usizex2>;
    type usizex8: WellFormedUnsignedIntegerRegister<Element = crate::element::USize, Lanes = U8>
        + IndexedBy<Self::usizex8, Self::u32x8, Self::u64x8> + FullConcatRegister<Self::usizex4>;
    type usizex16: WellFormedUnsignedIntegerRegister<Element = crate::element::USize, Lanes = U16>
        + IndexedBy<Self::usizex16, Self::u32x16, Self::u64x16> + FullConcatRegister<Self::usizex8>;

    // 64/32-bit SIMD types, almost always composite of scalar types
    type f32x2: WellFormedFloatRegister<Bits = Self::u32x2, SignedBits = Self::i32x2>
        + FullyInteroperable<Self::i32x2, Self::u32x2, Lanes = U2, Element = f32, Unsigned = Self::u32x2, Signed = Self::i32x2>
        + CastRegister<Self::f64x2> + CastRegister<Self::i16x2> + CastRegister<Self::u16x2> + CastRegister<Self::i8x2> + CastRegister<Self::u8x2>
        + FullConcatRegister<f32> + IndexedBy<Self::usizex2, Self::u32x2, Self::u64x2>
        + CastRegister<Self::i64x2> + CastRegister<Self::u64x2>;
    type i32x2: WellFormedSignedIntegerRegister
        + FullyInteroperable<Self::f32x2, Self::u32x2, Lanes = U2, Element = i32, Unsigned = Self::u32x2, Signed = Self::i32x2>
        + CastRegister<Self::i64x2> + CastRegister<Self::i16x2> + CastRegister<Self::i8x2> + CastRegister<Self::f32x2> + CastRegister<Self::f64x2> + FullConcatRegister<i32> + IndexedBy<Self::usizex2, Self::u32x2, Self::u64x2>
        + CastRegister<Self::u8x2> + CastRegister<Self::u16x2> + CastRegister<Self::u64x2>;
    // SAD: two u32 lanes = exactly one pair.
    type u32x2: WellFormedUnsignedIntegerRegister + Sad64Register<u64>
        + FullyInteroperable<Self::f32x2, Self::i32x2, Lanes = U2, Element = u32, Unsigned = Self::u32x2, Signed = Self::i32x2>
        + CastRegister<Self::u64x2> + CastRegister<Self::u16x2> + CastRegister<Self::u8x2> + CastRegister<Self::f32x2> + CastRegister<Self::f64x2> + FullConcatRegister<u32> + IndexedBy<Self::usizex2, Self::u32x2, Self::u64x2>
        + CastRegister<Self::i8x2> + CastRegister<Self::i16x2> + CastRegister<Self::i64x2>;

    // 128/32-bit SIMD types
    type f32x4: WellFormedFloatRegister<Bits = Self::u32x4, SignedBits = Self::i32x4> + LinAlg4Register
        + FullyInteroperable<Self::i32x4, Self::u32x4, Lanes = U4, Element = f32, Unsigned = Self::u32x4, Signed = Self::i32x4>
        + CastRegister<Self::f64x4> + CastRegister<Self::i16x4> + CastRegister<Self::u16x4> + CastRegister<Self::i8x4> + CastRegister<Self::u8x4>
        + FullConcatRegister<Self::f32x2> + IndexedBy<Self::usizex4, Self::u32x4, Self::u64x4> + CastRegister<Self::u64x4>
        + CastRegister<Self::i64x4>;
    type i32x4: WellFormedSignedIntegerRegister
        + FullyInteroperable<Self::f32x4, Self::u32x4, Lanes = U4, Element = i32, Unsigned = Self::u32x4, Signed = Self::i32x4>
        + CastRegister<Self::i64x4> + CastRegister<Self::i16x4> + CastRegister<Self::i8x4> + CastRegister<Self::f32x4> + CastRegister<Self::f64x4> + FullConcatRegister<Self::i32x2> + IndexedBy<Self::usizex4, Self::u32x4, Self::u64x4>
        + CastRegister<Self::u8x4> + CastRegister<Self::u16x4> + CastRegister<Self::u64x4>;
    type u32x4: WellFormedUnsignedIntegerRegister
        + FullyInteroperable<Self::f32x4, Self::i32x4, Lanes = U4, Element = u32, Unsigned = Self::u32x4, Signed = Self::i32x4>
        + CastRegister<Self::u64x4> + CastRegister<Self::u16x4> + CastRegister<Self::u8x4> + CastRegister<Self::f32x4> + CastRegister<Self::f64x4> + FullConcatRegister<Self::u32x2> + IndexedBy<Self::usizex4, Self::u32x4, Self::u64x4>
        // Same-width reinterprets (see `u8x16`'s SAD bounds).
        + BitCastRegister<Self::u8x16> + BitCastRegister<Self::u16x8>
        + Sad64Register<Self::u64x2>
        + CastRegister<Self::i8x4> + CastRegister<Self::i16x4> + CastRegister<Self::i64x4>;

    // 256/32-bit SIMD types
    type f32x8: WellFormedFloatRegister<Bits = Self::u32x8, SignedBits = Self::i32x8>
        + FullyInteroperable<Self::i32x8, Self::u32x8, Lanes = U8, Element = f32, Unsigned = Self::u32x8, Signed = Self::i32x8>
        + CastRegister<Self::f64x8> + CastRegister<Self::i16x8> + CastRegister<Self::u16x8> + CastRegister<Self::i8x8> + CastRegister<Self::u8x8>
        + FullConcatRegister<Self::f32x4> + IndexedBy<Self::usizex8, Self::u32x8, Self::u64x8>
        + CastRegister<Self::i64x8> + CastRegister<Self::u64x8>;
    type i32x8: WellFormedSignedIntegerRegister
        + FullyInteroperable<Self::f32x8, Self::u32x8, Lanes = U8, Element = i32, Unsigned = Self::u32x8, Signed = Self::i32x8>
        + CastRegister<Self::i64x8> + CastRegister<Self::i16x8> + CastRegister<Self::i8x8> + CastRegister<Self::f32x8> + CastRegister<Self::f64x8> + FullConcatRegister<Self::i32x4> + IndexedBy<Self::usizex8, Self::u32x8, Self::u64x8>
        + CastRegister<Self::u8x8> + CastRegister<Self::u16x8> + CastRegister<Self::u64x8>;
    // SAD: pairs -> u64x4.
    type u32x8: WellFormedUnsignedIntegerRegister + Sad64Register<Self::u64x4>
        + FullyInteroperable<Self::f32x8, Self::i32x8, Lanes = U8, Element = u32, Unsigned = Self::u32x8, Signed = Self::i32x8>
        + CastRegister<Self::u64x8> + CastRegister<Self::u16x8> + CastRegister<Self::u8x8> + CastRegister<Self::f32x8> + CastRegister<Self::f64x8> + FullConcatRegister<Self::u32x4> + IndexedBy<Self::usizex8, Self::u32x8, Self::u64x8>
        + CastRegister<Self::i8x8> + CastRegister<Self::i16x8> + CastRegister<Self::i64x8>;

    // 128/64-bit SIMD types
    type f64x2: WellFormedFloatRegister<Bits = Self::u64x2, SignedBits = Self::i64x2>
        + FullyInteroperable<Self::i64x2, Self::u64x2, Lanes = U2, Element = f64, Unsigned = Self::u64x2, Signed = Self::i64x2>
        + CastRegister<Self::f32x2> + CastRegister<Self::i16x2> + CastRegister<Self::u16x2> + CastRegister<Self::i8x2> + CastRegister<Self::u8x2>
        + FullConcatRegister<f64> + IndexedBy<Self::usizex2, Self::u32x2, Self::u64x2>
        + CastRegister<Self::i32x2> + CastRegister<Self::u32x2>;
    type i64x2: WellFormedSignedIntegerRegister
        + FullyInteroperable<Self::f64x2, Self::u64x2, Lanes = U2, Element = i64, Unsigned = Self::u64x2, Signed = Self::i64x2>
        + CastRegister<Self::i32x2> + CastRegister<Self::i16x2> + CastRegister<Self::i8x2> + CastRegister<Self::f64x2> + CastRegister<Self::f32x2> + FullConcatRegister<i64> + IndexedBy<Self::usizex2, Self::u32x2, Self::u64x2>
        + CastRegister<Self::u8x2> + CastRegister<Self::u16x2> + CastRegister<Self::u32x2>;
    type u64x2: WellFormedUnsignedIntegerRegister
        + FullyInteroperable<Self::f64x2, Self::i64x2, Lanes = U2, Element = u64, Unsigned = Self::u64x2, Signed = Self::i64x2>
        + CastRegister<Self::u32x2> + CastRegister<Self::u16x2> + CastRegister<Self::u8x2> + CastRegister<Self::f64x2> + CastRegister<Self::f32x2> + FullConcatRegister<u64> + IndexedBy<Self::usizex2, Self::u32x2, Self::u64x2>
        // Same-width reinterprets (see `u8x16`'s SAD bounds).
        + BitCastRegister<Self::u8x16> + BitCastRegister<Self::u16x8> + BitCastRegister<Self::u32x4>
        + CastRegister<Self::i8x2> + CastRegister<Self::i16x2> + CastRegister<Self::i32x2>;

    // 256/64-bit SIMD types
    type f64x4: WellFormedFloatRegister<Bits = Self::u64x4, SignedBits = Self::i64x4> + LinAlg4Register
        + FullyInteroperable<Self::i64x4, Self::u64x4, Lanes = U4, Element = f64, Unsigned = Self::u64x4, Signed = Self::i64x4>
        + CastRegister<Self::f32x4> + CastRegister<Self::i16x4> + CastRegister<Self::u16x4> + CastRegister<Self::i8x4> + CastRegister<Self::u8x4>
        + FullConcatRegister<Self::f64x2> + IndexedBy<Self::usizex4, Self::u32x4, Self::u64x4>
        + CastRegister<Self::i32x4> + CastRegister<Self::u32x4>;
    type i64x4: WellFormedSignedIntegerRegister
        + FullyInteroperable<Self::f64x4, Self::u64x4, Lanes = U4, Element = i64, Unsigned = Self::u64x4, Signed = Self::i64x4>
        + CastRegister<Self::i32x4> + CastRegister<Self::i16x4> + CastRegister<Self::i8x4> + CastRegister<Self::f64x4> + CastRegister<Self::f32x4> + FullConcatRegister<Self::i64x2> + IndexedBy<Self::usizex4, Self::u32x4, Self::u64x4>
        + CastRegister<Self::u8x4> + CastRegister<Self::u16x4> + CastRegister<Self::u32x4>;
    type u64x4: WellFormedUnsignedIntegerRegister
        + FullyInteroperable<Self::f64x4, Self::i64x4, Lanes = U4, Element = u64, Unsigned = Self::u64x4, Signed = Self::i64x4>
        + CastRegister<Self::u32x4> + CastRegister<Self::u16x4> + CastRegister<Self::u8x4> + CastRegister<Self::f64x4> + CastRegister<Self::f32x4> + FullConcatRegister<Self::u64x2> + IndexedBy<Self::usizex4, Self::u32x4, Self::u64x4>
        + CastRegister<Self::i8x4> + CastRegister<Self::i16x4> + CastRegister<Self::i32x4>;

    // 512/64-bit SIMD types
    type f64x8: WellFormedFloatRegister<Bits = Self::u64x8, SignedBits = Self::i64x8>
        + FullyInteroperable<Self::i64x8, Self::u64x8, Lanes = U8, Element = f64, Unsigned = Self::u64x8, Signed = Self::i64x8>
        + CastRegister<Self::f32x8> + CastRegister<Self::i16x8> + CastRegister<Self::u16x8> + CastRegister<Self::i8x8> + CastRegister<Self::u8x8>
        + FullConcatRegister<Self::f64x4> + IndexedBy<Self::usizex8, Self::u32x8, Self::u64x8>
        + CastRegister<Self::i32x8> + CastRegister<Self::u32x8>;
    type i64x8: WellFormedSignedIntegerRegister
        + FullyInteroperable<Self::f64x8, Self::u64x8, Lanes = U8, Element = i64, Unsigned = Self::u64x8, Signed = Self::i64x8>
        + CastRegister<Self::i32x8> + CastRegister<Self::i16x8> + CastRegister<Self::i8x8> + CastRegister<Self::f64x8> + CastRegister<Self::f32x8> + FullConcatRegister<Self::i64x4> + IndexedBy<Self::usizex8, Self::u32x8, Self::u64x8>
        + CastRegister<Self::u8x8> + CastRegister<Self::u16x8> + CastRegister<Self::u32x8>;
    type u64x8: WellFormedUnsignedIntegerRegister
        + FullyInteroperable<Self::f64x8, Self::i64x8, Lanes = U8, Element = u64, Unsigned = Self::u64x8, Signed = Self::i64x8>
        + CastRegister<Self::u32x8> + CastRegister<Self::u16x8> + CastRegister<Self::u8x8> + CastRegister<Self::f64x8> + CastRegister<Self::f32x8> + FullConcatRegister<Self::u64x4> + IndexedBy<Self::usizex8, Self::u32x8, Self::u64x8>
        + CastRegister<Self::i8x8> + CastRegister<Self::i16x8> + CastRegister<Self::i32x8>;

    // 512/32-bit SIMD types
    type f32x16: WellFormedFloatRegister<Bits = Self::u32x16, SignedBits = Self::i32x16>
        + FullyInteroperable<Self::i32x16, Self::u32x16, Lanes = U16, Element = f32, Unsigned = Self::u32x16, Signed = Self::i32x16>
        + CastRegister<Self::f64x16> + CastRegister<Self::i16x16> + CastRegister<Self::u16x16> + CastRegister<Self::i8x16> + CastRegister<Self::u8x16>
        + FullConcatRegister<Self::f32x8> + IndexedBy<Self::usizex16, Self::u32x16, Self::u64x16>
        + CastRegister<Self::i64x16> + CastRegister<Self::u64x16>;
    type i32x16: WellFormedSignedIntegerRegister
        + FullyInteroperable<Self::f32x16, Self::u32x16, Lanes = U16, Element = i32, Unsigned = Self::u32x16, Signed = Self::i32x16>
        + CastRegister<Self::i64x16> + CastRegister<Self::i16x16> + CastRegister<Self::i8x16> + CastRegister<Self::f32x16> + CastRegister<Self::f64x16> + FullConcatRegister<Self::i32x8> + IndexedBy<Self::usizex16, Self::u32x16, Self::u64x16>
        + CastRegister<Self::u8x16> + CastRegister<Self::u16x16> + CastRegister<Self::u64x16>;
    // SAD: pairs -> u64x8.
    type u32x16: WellFormedUnsignedIntegerRegister + Sad64Register<Self::u64x8>
        + FullyInteroperable<Self::f32x16, Self::i32x16, Lanes = U16, Element = u32, Unsigned = Self::u32x16, Signed = Self::i32x16>
        + CastRegister<Self::u64x16> + CastRegister<Self::u16x16> + CastRegister<Self::u8x16> + CastRegister<Self::f32x16> + CastRegister<Self::f64x16> + FullConcatRegister<Self::u32x8> + IndexedBy<Self::usizex16, Self::u32x16, Self::u64x16>
        + CastRegister<Self::i8x16> + CastRegister<Self::i16x16> + CastRegister<Self::i64x16>;

    // 1024/64-bit SIMD types
    type f64x16: WellFormedFloatRegister<Bits = Self::u64x16, SignedBits = Self::i64x16>
        + FullyInteroperable<Self::i64x16, Self::u64x16, Lanes = U16, Element = f64, Unsigned = Self::u64x16, Signed = Self::i64x16>
        + CastRegister<Self::f32x16> + CastRegister<Self::i16x16> + CastRegister<Self::u16x16> + CastRegister<Self::i8x16> + CastRegister<Self::u8x16>
        + FullConcatRegister<Self::f64x8> + IndexedBy<Self::usizex16, Self::u32x16, Self::u64x16>
        + CastRegister<Self::i32x16> + CastRegister<Self::u32x16>;
    type i64x16: WellFormedSignedIntegerRegister
        + FullyInteroperable<Self::f64x16, Self::u64x16, Lanes = U16, Element = i64, Unsigned = Self::u64x16, Signed = Self::i64x16>
        + CastRegister<Self::i32x16> + CastRegister<Self::i16x16> + CastRegister<Self::i8x16> + CastRegister<Self::f64x16> + CastRegister<Self::f32x16> + FullConcatRegister<Self::i64x8> + IndexedBy<Self::usizex16, Self::u32x16, Self::u64x16>
        + CastRegister<Self::u8x16> + CastRegister<Self::u16x16> + CastRegister<Self::u32x16>;
    type u64x16: WellFormedUnsignedIntegerRegister
        + FullyInteroperable<Self::f64x16, Self::i64x16, Lanes = U16, Element = u64, Unsigned = Self::u64x16, Signed = Self::i64x16>
        + CastRegister<Self::u32x16> + CastRegister<Self::u16x16> + CastRegister<Self::u8x16> + CastRegister<Self::f64x16> + CastRegister<Self::f32x16> + FullConcatRegister<Self::u64x8> + IndexedBy<Self::usizex16, Self::u32x16, Self::u64x16>
        + CastRegister<Self::i8x16> + CastRegister<Self::i16x16> + CastRegister<Self::i32x16>;

    // ===== 8-bit and 16-bit integer families =====
    // No floating-point partner (there is no `f16`), so these are modeled on the float-free
    // `usize` slots rather than the float-coupled 32/64-bit integer slots. They carry the
    // lane-preserving 8 <-> 16 <-> 32 widen/narrow casts (the 8/16 -> 32 widen lives in the
    // `where` clause on the trait header above; the narrows are stated on the narrower slot here).

    /// Fixed 16-lane (128-bit) signed 8-bit register, available with the same lane count on
    /// every backend (native `__m128i`/`v128` everywhere; `ArrayRegister` on the scalar
    /// backend). The natural width for byte/text work. Carries the concat-to-half ladder and the
    /// `usize`/`u32`/`u64` gather index trio like the rest of the 8-bit ladder; the fp8 transcode
    /// lives on the unsigned form (`u8x16` carries [`PackedF8Register`] / the `u8 <-> u32` widen).
    type i8x16: WellFormedSignedIntegerRegister<Element = i8, Lanes = U16, Unsigned = Self::u8x16, Signed = Self::i8x16>
        + IndexableRegister<Self::u8x16> + CastRegister<Self::i32x16> + CastRegister<Self::i16x16> + CastRegister<Self::i64x16>
           + CastRegister<Self::f32x16> + CastRegister<Self::f64x16>
          + BitCastRegister<Self::u8x16>
        + FullConcatRegister<Self::i8x8> + IndexedBy<Self::usizex16, Self::u32x16, Self::u64x16>
        + CastRegister<Self::u16x16> + CastRegister<Self::u32x16> + CastRegister<Self::u64x16>;
    /// Fixed 16-lane (128-bit) unsigned 8-bit register. See [`i8x16`](Self::i8x16).
    type u8x16: WellFormedUnsignedIntegerRegister<Element = u8, Lanes = U16, Unsigned = Self::u8x16, Signed = Self::i8x16>
        + IndexableRegister<Self::u8x16> + CastRegister<Self::u32x16> + CastRegister<Self::u16x16> + CastRegister<Self::u64x16>
           + CastRegister<Self::f32x16> + CastRegister<Self::f64x16>
          + BitCastRegister<Self::i8x16>
        + FullConcatRegister<Self::u8x8> + IndexedBy<Self::usizex16, Self::u32x16, Self::u64x16> + PackedF8Register<Self::f32x16>
        // Sum of absolute differences, summed in groups of 2/4/8 byte lanes into the
        // same-width `u16`/`u32`/`u64` register. x86 does the `u64` grouping in one
        // `psadbw`; NEON/wasm do the narrower ones in one `vpaddl`/`extadd_pairwise`.
        + Sad16Register<Self::u16x8> + Sad32Register<Self::u32x4> + Sad64Register<Self::u64x2>
        + CastRegister<Self::i16x16> + CastRegister<Self::i32x16> + CastRegister<Self::i64x16>;

    // Sub-native 8-bit ladder (x2 = ArrayRegister, x4/x8 = ReducedRegister over the native 128-bit
    // register). `CastRegister<Self::i16xK>` is the 16 -> 8 narrow; `CastRegister<Self::i32xK>` the
    // 32 -> 8 narrow; the 8 -> 16 / 8 -> 32 widens live on the wider slots.
    type i8x2: WellFormedSignedIntegerRegister<Element = i8, Lanes = U2, Unsigned = Self::u8x2, Signed = Self::i8x2>
        + CastRegister<Self::i32x2> + CastRegister<Self::i16x2> + CastRegister<Self::i64x2>
           + CastRegister<Self::f32x2> + CastRegister<Self::f64x2>
          + BitCastRegister<Self::u8x2> + FullConcatRegister<i8> + IndexedBy<Self::usizex2, Self::u32x2, Self::u64x2>
        + CastRegister<Self::u16x2> + CastRegister<Self::u32x2> + CastRegister<Self::u64x2>;
    type u8x2: WellFormedUnsignedIntegerRegister<Element = u8, Lanes = U2, Unsigned = Self::u8x2, Signed = Self::i8x2>
        + CastRegister<Self::u32x2> + CastRegister<Self::u16x2> + CastRegister<Self::u64x2>
           + CastRegister<Self::f32x2> + CastRegister<Self::f64x2>
          + BitCastRegister<Self::i8x2> + FullConcatRegister<u8> + IndexedBy<Self::usizex2, Self::u32x2, Self::u64x2>
        + PackedF8Register<Self::f32x2>
        // SAD: two bytes is below every grouping, so all three sum the whole
        // register into a single lane (the 1-lane scalar registers).
        + Sad16Register<u16> + Sad32Register<u32> + Sad64Register<u64>
        + CastRegister<Self::i16x2> + CastRegister<Self::i32x2> + CastRegister<Self::i64x2>;

    type i8x4: WellFormedSignedIntegerRegister<Element = i8, Lanes = U4, Unsigned = Self::u8x4, Signed = Self::i8x4>
        + CastRegister<Self::i32x4> + CastRegister<Self::i16x4> + CastRegister<Self::i64x4>
           + CastRegister<Self::f32x4> + CastRegister<Self::f64x4>
          + BitCastRegister<Self::u8x4> + FullConcatRegister<Self::i8x2> + IndexedBy<Self::usizex4, Self::u32x4, Self::u64x4>
        + CastRegister<Self::u16x4> + CastRegister<Self::u32x4> + CastRegister<Self::u64x4>;
    type u8x4: WellFormedUnsignedIntegerRegister<Element = u8, Lanes = U4, Unsigned = Self::u8x4, Signed = Self::i8x4>
        + CastRegister<Self::u32x4> + CastRegister<Self::u16x4> + CastRegister<Self::u64x4>
           + CastRegister<Self::f32x4> + CastRegister<Self::f64x4>
          + BitCastRegister<Self::i8x4> + FullConcatRegister<Self::u8x2> + IndexedBy<Self::usizex4, Self::u32x4, Self::u64x4>
        + PackedF8Register<Self::f32x4>
        // SAD: 4 bytes = two 2-byte groups, one 4-byte group, and a partial 8-byte group.
        + Sad16Register<Self::u16x2> + Sad32Register<u32> + Sad64Register<u64>
        + CastRegister<Self::i16x4> + CastRegister<Self::i32x4> + CastRegister<Self::i64x4>;

    type i8x8: WellFormedSignedIntegerRegister<Element = i8, Lanes = U8, Unsigned = Self::u8x8, Signed = Self::i8x8>
        + CastRegister<Self::i32x8> + CastRegister<Self::i16x8> + CastRegister<Self::i64x8>
           + CastRegister<Self::f32x8> + CastRegister<Self::f64x8>
          + BitCastRegister<Self::u8x8> + FullConcatRegister<Self::i8x4> + IndexedBy<Self::usizex8, Self::u32x8, Self::u64x8>
        + CastRegister<Self::u16x8> + CastRegister<Self::u32x8> + CastRegister<Self::u64x8>;
    type u8x8: WellFormedUnsignedIntegerRegister<Element = u8, Lanes = U8, Unsigned = Self::u8x8, Signed = Self::i8x8>
        + CastRegister<Self::u32x8> + CastRegister<Self::u16x8> + CastRegister<Self::u64x8>
           + CastRegister<Self::f32x8> + CastRegister<Self::f64x8>
          + BitCastRegister<Self::i8x8> + FullConcatRegister<Self::u8x4> + IndexedBy<Self::usizex8, Self::u32x8, Self::u64x8>
        + PackedF8Register<Self::f32x8>
        // SAD: 8 bytes = four/two groups, and exactly one full 8-byte group.
        + Sad16Register<Self::u16x4> + Sad32Register<Self::u32x2> + Sad64Register<u64>
        + CastRegister<Self::i16x8> + CastRegister<Self::i32x8> + CastRegister<Self::i64x8>;

    // 16-bit ladder. `CastRegister<Self::i8xK>` is the 8 -> 16 widen; `CastRegister<Self::i32xK>`
    // the 16 -> 32 narrow.
    type i16x2: WellFormedSignedIntegerRegister<Element = i16, Lanes = U2, Unsigned = Self::u16x2, Signed = Self::i16x2>
        + CastRegister<Self::i32x2> + CastRegister<Self::i8x2> + CastRegister<Self::i64x2>
          + CastRegister<Self::f32x2> + CastRegister<Self::f64x2>
          + BitCastRegister<Self::u16x2> + FullConcatRegister<i16> + IndexedBy<Self::usizex2, Self::u32x2, Self::u64x2>
        + CastRegister<Self::u8x2> + CastRegister<Self::u32x2> + CastRegister<Self::u64x2>;
    // SAD: one full pair; the group of four is partial, so it sums both lanes.
    type u16x2: WellFormedUnsignedIntegerRegister<Element = u16, Lanes = U2, Unsigned = Self::u16x2, Signed = Self::i16x2>
        + Sad32Register<u32> + Sad64Register<u64>
        + CastRegister<Self::u32x2> + CastRegister<Self::u8x2> + CastRegister<Self::u64x2>
          + CastRegister<Self::f32x2> + CastRegister<Self::f64x2>
          + BitCastRegister<Self::i16x2> + FullConcatRegister<u16> + IndexedBy<Self::usizex2, Self::u32x2, Self::u64x2>
        + CastRegister<Self::i8x2> + CastRegister<Self::i32x2> + CastRegister<Self::i64x2>;

    type i16x4: WellFormedSignedIntegerRegister<Element = i16, Lanes = U4, Unsigned = Self::u16x4, Signed = Self::i16x4>
        + CastRegister<Self::i32x4> + CastRegister<Self::i8x4> + CastRegister<Self::i64x4>
          + CastRegister<Self::f32x4> + CastRegister<Self::f64x4>
          + BitCastRegister<Self::u16x4> + FullConcatRegister<Self::i16x2> + IndexedBy<Self::usizex4, Self::u32x4, Self::u64x4>
        + CastRegister<Self::u8x4> + CastRegister<Self::u32x4> + CastRegister<Self::u64x4>;
    // SAD: 4 u16 lanes = two pairs, and exactly one group of four.
    type u16x4: WellFormedUnsignedIntegerRegister<Element = u16, Lanes = U4, Unsigned = Self::u16x4, Signed = Self::i16x4>
        + Sad32Register<Self::u32x2> + Sad64Register<u64>
        + CastRegister<Self::u32x4> + CastRegister<Self::u8x4> + CastRegister<Self::u64x4>
          + CastRegister<Self::f32x4> + CastRegister<Self::f64x4>
          + BitCastRegister<Self::i16x4> + FullConcatRegister<Self::u16x2> + IndexedBy<Self::usizex4, Self::u32x4, Self::u64x4>
        + PackedF16Register<Self::f32x4>
        + CastRegister<Self::i8x4> + CastRegister<Self::i32x4> + CastRegister<Self::i64x4>;

    type i16x8: WellFormedSignedIntegerRegister<Element = i16, Lanes = U8, Unsigned = Self::u16x8, Signed = Self::i16x8>
        + CastRegister<Self::i32x8> + CastRegister<Self::i8x8> + CastRegister<Self::i64x8>
          + CastRegister<Self::f32x8> + CastRegister<Self::f64x8>
          + BitCastRegister<Self::u16x8> + FullConcatRegister<Self::i16x4> + IndexedBy<Self::usizex8, Self::u32x8, Self::u64x8>
        + CastRegister<Self::u8x8> + CastRegister<Self::u32x8> + CastRegister<Self::u64x8>;
    type u16x8: WellFormedUnsignedIntegerRegister<Element = u16, Lanes = U8, Unsigned = Self::u16x8, Signed = Self::i16x8>
        + CastRegister<Self::u32x8> + CastRegister<Self::u8x8> + CastRegister<Self::u64x8>
          + CastRegister<Self::f32x8> + CastRegister<Self::f64x8>
          + BitCastRegister<Self::i16x8> + FullConcatRegister<Self::u16x4> + IndexedBy<Self::usizex8, Self::u32x8, Self::u64x8>
        + PackedF16Register<Self::f32x8>
        // Same-width reinterpret of the byte register (see `u8x16`'s SAD bounds).
        + BitCastRegister<Self::u8x16>
        // SAD one element size up: pairs -> u32, quads -> u64.
        + Sad32Register<Self::u32x4> + Sad64Register<Self::u64x2>
        + CastRegister<Self::i8x8> + CastRegister<Self::i32x8> + CastRegister<Self::i64x8>;

    type i16x16: WellFormedSignedIntegerRegister<Element = i16, Lanes = U16, Unsigned = Self::u16x16, Signed = Self::i16x16>
        + CastRegister<Self::i32x16> + CastRegister<Self::i8x16> + CastRegister<Self::i64x16>
          + CastRegister<Self::f32x16> + CastRegister<Self::f64x16>
          + BitCastRegister<Self::u16x16> + FullConcatRegister<Self::i16x8> + IndexedBy<Self::usizex16, Self::u32x16, Self::u64x16>
        + CastRegister<Self::u8x16> + CastRegister<Self::u32x16> + CastRegister<Self::u64x16>;
    // SAD: pairs -> u32x8, groups of four -> u64x4.
    type u16x16: WellFormedUnsignedIntegerRegister<Element = u16, Lanes = U16, Unsigned = Self::u16x16, Signed = Self::i16x16>
        + Sad32Register<Self::u32x8> + Sad64Register<Self::u64x4>
        + CastRegister<Self::u32x16> + CastRegister<Self::u8x16> + CastRegister<Self::u64x16>
          + CastRegister<Self::f32x16> + CastRegister<Self::f64x16>
          + BitCastRegister<Self::i16x16> + FullConcatRegister<Self::u16x8> + IndexedBy<Self::usizex16, Self::u32x16, Self::u64x16>
        + PackedF16Register<Self::f32x16>
        + CastRegister<Self::i8x16> + CastRegister<Self::i32x16> + CastRegister<Self::i64x16>;}

/// Reduced registers to 3 lanes, for 3D math operations. They are backed by 4-lane registers internally
/// using [`ReducedRegister`], so they have the same alignment and nearly identical performance
/// characteristics as 4-lane registers, but only use 3 lanes for data.
#[rustfmt::skip]
pub trait Simd3A: Simd<
    usizex4: FullExtendRegister<Self::usizex3A>,
    f32x4: FullExtendRegister<Self::f32x3A>,
    i32x4: FullExtendRegister<Self::i32x3A>,
    u32x4: FullExtendRegister<Self::u32x3A>,
    f64x4: FullExtendRegister<Self::f64x3A>,
    i64x4: FullExtendRegister<Self::i64x3A>,
    u64x4: FullExtendRegister<Self::u64x3A>,
> {type usizex3A: WellFormedUnsignedIntegerRegister<Element = crate::element::USize, Lanes = U3>
        + IndexedBy<Self::usizex3A, Self::u32x3A, Self::u64x3A>;

    type f32x3A: WellFormedFloatRegister<Bits = Self::u32x3A, SignedBits = Self::i32x3A> + LinAlg3Register
        + FullyInteroperable<Self::i32x3A, Self::u32x3A, Lanes = U3, Element = f32, Unsigned = Self::u32x3A, Signed = Self::i32x3A>
        + CastRegister<Self::f64x3A> + IndexedBy<Self::usizex3A, Self::u32x3A, Self::u64x3A>
        + CastRegister<Self::i64x3A> + CastRegister<Self::u64x3A>;
    type i32x3A: WellFormedSignedIntegerRegister
        + FullyInteroperable<Self::f32x3A, Self::u32x3A, Lanes = U3, Element = i32, Unsigned = Self::u32x3A, Signed = Self::i32x3A>
        + CastRegister<Self::i64x3A> + CastRegister<Self::f32x3A> + CastRegister<Self::f64x3A> + IndexedBy<Self::usizex3A, Self::u32x3A, Self::u64x3A>
        + CastRegister<Self::u64x3A>;
    type u32x3A: WellFormedUnsignedIntegerRegister
        + FullyInteroperable<Self::f32x3A, Self::i32x3A, Lanes = U3, Element = u32, Unsigned = Self::u32x3A, Signed = Self::i32x3A>
        + CastRegister<Self::u64x3A> + CastRegister<Self::f32x3A> + CastRegister<Self::f64x3A> + IndexedBy<Self::usizex3A, Self::u32x3A, Self::u64x3A>
        + CastRegister<Self::i64x3A>;

    type f64x3A: WellFormedFloatRegister<Bits = Self::u64x3A, SignedBits = Self::i64x3A> + LinAlg3Register
        + FullyInteroperable<Self::i64x3A, Self::u64x3A, Lanes = U3, Element = f64, Unsigned = Self::u64x3A, Signed = Self::i64x3A>
        + CastRegister<Self::f32x3A> + IndexedBy<Self::usizex3A, Self::u32x3A, Self::u64x3A>
        + CastRegister<Self::i32x3A> + CastRegister<Self::u32x3A>;
    type i64x3A: WellFormedSignedIntegerRegister
        + FullyInteroperable<Self::f64x3A, Self::u64x3A, Lanes = U3, Element = i64, Unsigned = Self::u64x3A, Signed = Self::i64x3A>
        + CastRegister<Self::i32x3A> + CastRegister<Self::f64x3A> + CastRegister<Self::f32x3A> + IndexedBy<Self::usizex3A, Self::u32x3A, Self::u64x3A>
        + CastRegister<Self::u32x3A>;
    type u64x3A: WellFormedUnsignedIntegerRegister
        + FullyInteroperable<Self::f64x3A, Self::i64x3A, Lanes = U3, Element = u64, Unsigned = Self::u64x3A, Signed = Self::i64x3A>
        + CastRegister<Self::u32x3A> + CastRegister<Self::f64x3A> + CastRegister<Self::f32x3A> + IndexedBy<Self::usizex3A, Self::u32x3A, Self::u64x3A>
        + CastRegister<Self::i32x3A>;}

impl<S: Simd> Simd3A for S {
    type usizex3A = ReducedRegister<S::usizex4, U1>;

    type f32x3A = ReducedRegister<S::f32x4, U1>;
    type i32x3A = ReducedRegister<S::i32x4, U1>;
    type u32x3A = ReducedRegister<S::u32x4, U1>;

    type f64x3A = ReducedRegister<S::f64x4, U1>;
    type i64x3A = ReducedRegister<S::i64x4, U1>;
    type u64x3A = ReducedRegister<S::u64x4, U1>;
}

/// True 3-lane SIMD registers, for 3D math operations. Unlike [`Simd3A`], these registers
/// are **not** guaranteed to be four-element aligned - the underlying storage may be a true
/// 3-lane register on backends that support it natively (such as SPIR-V with `vec3`), or
/// any other 3-lane representation the backend chooses.
///
/// On most CPU backends, this will typically be implemented identically to [`Simd3A`]
/// (using a [`ReducedRegister`] backed by the 4-lane register), since hardware SIMD has
/// no true 3-lane representation. On GPU/shader backends, this can be implemented using
/// native 3-component vector types.
///
/// This trait is **not** automatically implemented - each backend must opt in and define
/// its own 3-lane register types.
#[rustfmt::skip]
pub trait Simd3: Simd3A<
    usizex4: FullExtendRegister<Self::usizex3>,
    f32x4: FullExtendRegister<Self::f32x3>,
    i32x4: FullExtendRegister<Self::i32x3>,
    u32x4: FullExtendRegister<Self::u32x3>,
    f64x4: FullExtendRegister<Self::f64x3>,
    i64x4: FullExtendRegister<Self::i64x3>,
    u64x4: FullExtendRegister<Self::u64x3>,
> {type usizex3: WellFormedUnsignedIntegerRegister<Element = crate::element::USize, Lanes = U3>
        + IndexedBy<Self::usizex3, Self::u32x3, Self::u64x3>;

    type f32x3: WellFormedFloatRegister<Bits = Self::u32x3, SignedBits = Self::i32x3> + LinAlg3Register
        + FullyInteroperable<Self::i32x3, Self::u32x3, Lanes = U3, Element = f32, Unsigned = Self::u32x3, Signed = Self::i32x3>
        + CastRegister<Self::f64x3> + IndexedBy<Self::usizex3, Self::u32x3, Self::u64x3>
        + CastRegister<Self::i64x3> + CastRegister<Self::u64x3>;
    type i32x3: WellFormedSignedIntegerRegister
        + FullyInteroperable<Self::f32x3, Self::u32x3, Lanes = U3, Element = i32, Unsigned = Self::u32x3, Signed = Self::i32x3>
        + CastRegister<Self::i64x3> + CastRegister<Self::f32x3> + CastRegister<Self::f64x3> + IndexedBy<Self::usizex3, Self::u32x3, Self::u64x3>
        + CastRegister<Self::u64x3>;
    type u32x3: WellFormedUnsignedIntegerRegister
        + FullyInteroperable<Self::f32x3, Self::i32x3, Lanes = U3, Element = u32, Unsigned = Self::u32x3, Signed = Self::i32x3>
        + CastRegister<Self::u64x3> + CastRegister<Self::f32x3> + CastRegister<Self::f64x3> + IndexedBy<Self::usizex3, Self::u32x3, Self::u64x3>
        + CastRegister<Self::i64x3>;

    type f64x3: WellFormedFloatRegister<Bits = Self::u64x3, SignedBits = Self::i64x3> + LinAlg3Register
        + FullyInteroperable<Self::i64x3, Self::u64x3, Lanes = U3, Element = f64, Unsigned = Self::u64x3, Signed = Self::i64x3>
        + CastRegister<Self::f32x3> + IndexedBy<Self::usizex3, Self::u32x3, Self::u64x3>
        + CastRegister<Self::i32x3> + CastRegister<Self::u32x3>;
    type i64x3: WellFormedSignedIntegerRegister
        + FullyInteroperable<Self::f64x3, Self::u64x3, Lanes = U3, Element = i64, Unsigned = Self::u64x3, Signed = Self::i64x3>
        + CastRegister<Self::i32x3> + CastRegister<Self::f64x3> + CastRegister<Self::f32x3> + IndexedBy<Self::usizex3, Self::u32x3, Self::u64x3>
        + CastRegister<Self::u32x3>;
    type u64x3: WellFormedUnsignedIntegerRegister
        + FullyInteroperable<Self::f64x3, Self::i64x3, Lanes = U3, Element = u64, Unsigned = Self::u64x3, Signed = Self::i64x3>
        + CastRegister<Self::u32x3> + CastRegister<Self::f64x3> + CastRegister<Self::f32x3> + IndexedBy<Self::usizex3, Self::u32x3, Self::u64x3>
        + CastRegister<Self::i32x3>;}

/// Names every SIMD register a backend offers, parameterized by a single
/// type-level lane count `Width`.
///
/// Where [`Simd`] gives you `f32x2`, `f32x4`, `f32x8`, `f32x16` as separate
/// associated types, `FixedWidthSimd<U4>` collapses them into a single
/// `f32xN` associated type, and likewise for the other element types. This
/// is what lets a function be written once and instantiated at the chosen
/// width by callers.
///
/// Implemented automatically for every backend at widths `U1`, `U2`, `U3`
/// (via [`Simd3`]), `U4`, `U8`, `U16`.
///
/// # Example
///
/// ```ignore
/// fn process<S, W>(data: &mut [f32])
/// where
///     S: FixedWidthSimd<W>,
///     W: Lanes,
/// {
///     // `S::f32xN` is the f32 vector with `W` lanes on backend `S`.
///     let (_, chunks, _) = Vector::<S::f32xN>::align_slice_mut(data);
///     for c in chunks { *c = c.sqrt(); }
/// }
/// ```
pub trait FixedWidthSimd<Width: Lanes>: Simd {
    type usizexN: WellFormedUnsignedIntegerRegister<Element = crate::element::USize, Lanes = Width>;

    type f32xN: WellFormedFloatRegister<
            Bits = <Self as FixedWidthSimd<Width>>::u32xN,
            SignedBits = <Self as FixedWidthSimd<Width>>::i32xN,
        > + FullyInteroperable<
            <Self as FixedWidthSimd<Width>>::i32xN,
            <Self as FixedWidthSimd<Width>>::u32xN,
            Lanes = Width,
            Element = f32,
            Unsigned = <Self as FixedWidthSimd<Width>>::u32xN,
            Signed = <Self as FixedWidthSimd<Width>>::i32xN,
        > + IndexedBy<
            <Self as FixedWidthSimd<Width>>::usizexN,
            <Self as FixedWidthSimd<Width>>::u32xN,
            <Self as FixedWidthSimd<Width>>::u64xN,
        > + CastRegister<<Self as FixedWidthSimd<Width>>::f64xN>;
    type i32xN: WellFormedSignedIntegerRegister
        + FullyInteroperable<
            <Self as FixedWidthSimd<Width>>::f32xN,
            <Self as FixedWidthSimd<Width>>::u32xN,
            Lanes = Width,
            Element = i32,
            Unsigned = <Self as FixedWidthSimd<Width>>::u32xN,
            Signed = <Self as FixedWidthSimd<Width>>::i32xN,
        > + IndexedBy<
            <Self as FixedWidthSimd<Width>>::usizexN,
            <Self as FixedWidthSimd<Width>>::u32xN,
            <Self as FixedWidthSimd<Width>>::u64xN,
        > + CastRegister<<Self as FixedWidthSimd<Width>>::i64xN>;
    type u32xN: WellFormedUnsignedIntegerRegister
        + FullyInteroperable<
            <Self as FixedWidthSimd<Width>>::f32xN,
            <Self as FixedWidthSimd<Width>>::i32xN,
            Lanes = Width,
            Element = u32,
            Unsigned = <Self as FixedWidthSimd<Width>>::u32xN,
            Signed = <Self as FixedWidthSimd<Width>>::i32xN,
        > + IndexedBy<
            <Self as FixedWidthSimd<Width>>::usizexN,
            <Self as FixedWidthSimd<Width>>::u32xN,
            <Self as FixedWidthSimd<Width>>::u64xN,
        > + CastRegister<<Self as FixedWidthSimd<Width>>::u64xN>;

    type f64xN: WellFormedFloatRegister<
            Bits = <Self as FixedWidthSimd<Width>>::u64xN,
            SignedBits = <Self as FixedWidthSimd<Width>>::i64xN,
        > + FullyInteroperable<
            <Self as FixedWidthSimd<Width>>::i64xN,
            <Self as FixedWidthSimd<Width>>::u64xN,
            Lanes = Width,
            Element = f64,
            Unsigned = <Self as FixedWidthSimd<Width>>::u64xN,
            Signed = <Self as FixedWidthSimd<Width>>::i64xN,
        > + IndexedBy<
            <Self as FixedWidthSimd<Width>>::usizexN,
            <Self as FixedWidthSimd<Width>>::u32xN,
            <Self as FixedWidthSimd<Width>>::u64xN,
        > + CastRegister<<Self as FixedWidthSimd<Width>>::f32xN>;
    type i64xN: WellFormedSignedIntegerRegister
        + FullyInteroperable<
            <Self as FixedWidthSimd<Width>>::f64xN,
            <Self as FixedWidthSimd<Width>>::u64xN,
            Lanes = Width,
            Element = i64,
            Unsigned = <Self as FixedWidthSimd<Width>>::u64xN,
            Signed = <Self as FixedWidthSimd<Width>>::i64xN,
        > + IndexedBy<
            <Self as FixedWidthSimd<Width>>::usizexN,
            <Self as FixedWidthSimd<Width>>::u32xN,
            <Self as FixedWidthSimd<Width>>::u64xN,
        > + CastRegister<<Self as FixedWidthSimd<Width>>::i32xN>;
    type u64xN: WellFormedUnsignedIntegerRegister
        + FullyInteroperable<
            <Self as FixedWidthSimd<Width>>::f64xN,
            <Self as FixedWidthSimd<Width>>::i64xN,
            Lanes = Width,
            Element = u64,
            Unsigned = <Self as FixedWidthSimd<Width>>::u64xN,
            Signed = <Self as FixedWidthSimd<Width>>::i64xN,
        > + IndexedBy<
            <Self as FixedWidthSimd<Width>>::usizexN,
            <Self as FixedWidthSimd<Width>>::u32xN,
            <Self as FixedWidthSimd<Width>>::u64xN,
        > + CastRegister<<Self as FixedWidthSimd<Width>>::u32xN>;
}

macro_rules! impl_wide_simd {
    ($($width:literal),*) => {paste::paste! { $(
        impl<S: Simd> FixedWidthSimd<[<U $width>]> for S {
            type usizexN = S::[<usizex $width>];

            type f32xN = S::[<f32x $width>];
            type i32xN = S::[<i32x $width>];
            type u32xN = S::[<u32x $width>];

            type f64xN = S::[<f64x $width>];
            type i64xN = S::[<i64x $width>];
            type u64xN = S::[<u64x $width>];
        }
    )* }}
}

impl<S: Simd> FixedWidthSimd<U1> for S {
    type usizexN = crate::element::USize;

    type f32xN = f32;
    type i32xN = i32;
    type u32xN = u32;

    type f64xN = f64;
    type i64xN = i64;
    type u64xN = u64;
}

impl_wide_simd!(2, 4, 8, 16);

impl<S: Simd3> FixedWidthSimd<U3> for S {
    type usizexN = S::usizex3;

    type f32xN = S::f32x3;
    type i32xN = S::i32x3;
    type u32xN = S::u32x3;

    type f64xN = S::f64x3;
    type i64xN = S::i64x3;
    type u64xN = S::u64x3;
}

/// SIMD types of the same element size but different lane counts, all based
/// on the given fully formed element types.
///
/// For example, `SizedSimd<f32, i32, u32>` provides all the fixed-size SIMD types
/// with 32-bit elements, and `SizedSimd<f64, i64, u64>` provides all the fixed-size SIMD types
/// with 64-bit elements.
///
/// You can use this generically to write code that works with either 32-bit or 64-bit
/// floating-point SIMD types and their associated integer types, like so:
/// ```ignore
/// fn my_func<S, T>(values: &[T]) -> T
/// where
///     T: WellFormedFloatElement,
///     S: SizedSimd<T, <T as FloatElementWithBits>::SignedBits, <T as FloatElementWithBits>::Bits>,
/// {
///     // do whatever you need with S::fxN, S::ixN, S::uxN, etc.
///     let (scalar_prefix, vectors, scalar_suffix) = Vector::<S::fxN>::from_slice(values);
/// }
/// ```
#[rustfmt::skip]
pub trait SizedSimd<
    F: WellFormedFloatElement + FloatElementWithBits<SignedBits = I, Bits = U>,
    I: WellFormedSignedIntegerElement,
    U: WellFormedUnsignedIntegerElement,
>: Simd {
    type NativeWidth: Lanes;

    // TODO: Figure out how to make these all WellFormed registers. There is currently some kind of mismatch
    // between the expected inner Element type and what is provided via the generic parameters. Specifying them
    // is weird/difficult.

    type fxN: FullyInteroperable<Self::ixN, Self::uxN, Lanes = Self::NativeWidth, Element = F, Unsigned = Self::uxN, Signed = Self::ixN>
        + FloatRegister<Bits = Self::uxN, SignedBits = Self::ixN> + IndexableRegister<Self::uxN>;
    type ixN: FullyInteroperable<Self::fxN, Self::uxN, Lanes = Self::NativeWidth, Element = I, Unsigned = Self::uxN, Signed = Self::ixN>
        + SignedIntegerRegister<Element = <F as FloatElementWithBits>::SignedBits> + IndexableRegister<Self::uxN>;
    type uxN: FullyInteroperable<Self::fxN, Self::ixN, Lanes = Self::NativeWidth, Element = U, Unsigned = Self::uxN, Signed = Self::ixN>
        + UnsignedIntegerRegister<Element = <F as FloatElementWithBits>::Bits> + IndexableRegister<Self::uxN>;

    type fx2: FullyInteroperable<Self::ix2, Self::ux2, Lanes = U2, Element = F, Unsigned = Self::ux2, Signed = Self::ix2>
        + FloatRegister<Bits = Self::ux2, SignedBits = Self::ix2> + FullConcatRegister<F> + IndexedBy<Self::usizex2, Self::u32x2, Self::u64x2>;
    type ix2: FullyInteroperable<Self::fx2, Self::ux2, Lanes = U2, Element = I, Unsigned = Self::ux2, Signed = Self::ix2>
        + SignedIntegerRegister<Element = <F as FloatElementWithBits>::SignedBits> + FullConcatRegister<I> + IndexedBy<Self::usizex2, Self::u32x2, Self::u64x2>;
    type ux2: FullyInteroperable<Self::fx2, Self::ix2, Lanes = U2, Element = U, Unsigned = Self::ux2, Signed = Self::ix2>
        + UnsignedIntegerRegister<Element = <F as FloatElementWithBits>::Bits> + FullConcatRegister<U> + IndexedBy<Self::usizex2, Self::u32x2, Self::u64x2>;

    type fx4: FullyInteroperable<Self::ix4, Self::ux4, Lanes = U4, Element = F, Unsigned = Self::ux4, Signed = Self::ix4>
        + FloatRegister<Bits = Self::ux4, SignedBits = Self::ix4> + LinAlg4Register
        + FullConcatRegister<Self::fx2> + IndexedBy<Self::usizex4, Self::u32x4, Self::u64x4>;
    type ix4: FullyInteroperable<Self::fx4, Self::ux4, Lanes = U4, Element = I, Unsigned = Self::ux4, Signed = Self::ix4>
        + SignedIntegerRegister<Element = <F as FloatElementWithBits>::SignedBits> + FullConcatRegister<Self::ix2> + IndexedBy<Self::usizex4, Self::u32x4, Self::u64x4>;
    type ux4: FullyInteroperable<Self::fx4, Self::ix4, Lanes = U4, Element = U, Unsigned = Self::ux4, Signed = Self::ix4>
        + UnsignedIntegerRegister<Element = <F as FloatElementWithBits>::Bits> + FullConcatRegister<Self::ux2> + IndexedBy<Self::usizex4, Self::u32x4, Self::u64x4>;

    type fx8: FullyInteroperable<Self::ix8, Self::ux8, Lanes = U8, Element = F, Unsigned = Self::ux8, Signed = Self::ix8>
        + FloatRegister<Bits = Self::ux8, SignedBits = Self::ix8> + FullConcatRegister<Self::fx4> + IndexedBy<Self::usizex8, Self::u32x8, Self::u64x8>;
    type ix8: FullyInteroperable<Self::fx8, Self::ux8, Lanes = U8, Element = I, Unsigned = Self::ux8, Signed = Self::ix8>
        + SignedIntegerRegister<Element = <F as FloatElementWithBits>::SignedBits> + FullConcatRegister<Self::ix4> + IndexedBy<Self::usizex8, Self::u32x8, Self::u64x8>;
    type ux8: FullyInteroperable<Self::fx8, Self::ix8, Lanes = U8, Element = U, Unsigned = Self::ux8, Signed = Self::ix8>
        + UnsignedIntegerRegister<Element = <F as FloatElementWithBits>::Bits> + FullConcatRegister<Self::ux4> + IndexedBy<Self::usizex8, Self::u32x8, Self::u64x8>;

    type fx16: FullyInteroperable<Self::ix16, Self::ux16, Lanes = U16, Element = F, Unsigned = Self::ux16, Signed = Self::ix16>
        + FloatRegister<Bits = Self::ux16, SignedBits = Self::ix16> + FullConcatRegister<Self::fx8> + IndexedBy<Self::usizex16, Self::u32x16, Self::u64x16>;
    type ix16: FullyInteroperable<Self::fx16, Self::ux16, Lanes = U16, Element = I, Unsigned = Self::ux16, Signed = Self::ix16>
        + SignedIntegerRegister<Element = <F as FloatElementWithBits>::SignedBits> + FullConcatRegister<Self::ix8> + IndexedBy<Self::usizex16, Self::u32x16, Self::u64x16>;
    type ux16: FullyInteroperable<Self::fx16, Self::ix16, Lanes = U16, Element = U, Unsigned = Self::ux16, Signed = Self::ix16>
        + UnsignedIntegerRegister<Element = <F as FloatElementWithBits>::Bits> + FullConcatRegister<Self::ux8> + IndexedBy<Self::usizex16, Self::u32x16, Self::u64x16>;
}

/// Convenience over [`SizedSimd`]: pick a float element type, and the
/// matching signed/unsigned integer "bits" types are inferred automatically.
///
/// For `F = f32` this is exactly `SizedSimd<f32, i32, u32>`; for `F = f64`,
/// `SizedSimd<f64, i64, u64>`. Prefer this when the float type is the only
/// parameter your function cares about and you do not want to spell out
/// `<F as FloatElementWithBits>::SignedBits` etc.
///
/// # Example
///
/// ```ignore
/// fn sin_inplace<S, F>(data: &mut [F])
/// where
///     F: WellFormedFloatElement,
///     S: FloatSimd<F>,
///     Vector<S::fxN>: TranscendentalMath,
/// {
///     let (_, chunks, _) = Vector::<S::fxN>::align_slice_mut(data);
///     for c in chunks { *c = c.sin(); }
/// }
/// ```
pub trait FloatSimd<F: WellFormedFloatElement + FloatElementWithBits>:
    SizedSimd<F, <F as FloatElementWithBits>::SignedBits, <F as FloatElementWithBits>::Bits>
{
}

impl<S, F> FloatSimd<F> for S
where
    F: WellFormedFloatElement + FloatElementWithBits,
    S: SizedSimd<F, <F as FloatElementWithBits>::SignedBits, <F as FloatElementWithBits>::Bits>,
{
}

impl<S: Simd> SizedSimd<f32, i32, u32> for S {
    type NativeWidth = S::Native32Width;

    type fxN = S::f32xN;
    type ixN = S::i32xN;
    type uxN = S::u32xN;

    type fx2 = S::f32x2;
    type ix2 = S::i32x2;
    type ux2 = S::u32x2;

    type fx4 = S::f32x4;
    type ix4 = S::i32x4;
    type ux4 = S::u32x4;

    type fx8 = S::f32x8;
    type ix8 = S::i32x8;
    type ux8 = S::u32x8;

    type fx16 = S::f32x16;
    type ix16 = S::i32x16;
    type ux16 = S::u32x16;
}

impl<S: Simd> SizedSimd<f64, i64, u64> for S {
    type NativeWidth = S::Native64Width;

    type fxN = S::f64xN;
    type ixN = S::i64xN;
    type uxN = S::u64xN;

    type fx2 = S::f64x2;
    type ix2 = S::i64x2;
    type ux2 = S::u64x2;

    type fx4 = S::f64x4;
    type ix4 = S::i64x4;
    type ux4 = S::u64x4;

    type fx8 = S::f64x8;
    type ix8 = S::i64x8;
    type ux8 = S::u64x8;

    type fx16 = S::f64x16;
    type ix16 = S::i64x16;
    type ux16 = S::u64x16;
}

// =============================================================================
// Vector type aliases, parameterized by a backend `S`.
//
// These are the recommended way to spell a SIMD vector type in user code:
// pick a backend (`X86V3`, `Scalar`, or a generic `S: Simd`), and write
// `f32x4<S>` rather than `Vector<<S as Simd>::f32x4>`.
//
// The `xN` suffix follows the host's native width via `NativeSimd`; the
// numeric suffixes (`x2`/`x4`/`x8`/`x16`) come from `Simd`. The `x3A` and
// `x3` variants come from `Simd3A`/`Simd3` (alpha-padded vs. true 3-lane).
// =============================================================================

/// `f32` vector at the backend's native 32-bit width.
pub type f32xN<S> = Vector<<S as NativeSimd>::f32xN>;
/// `i32` vector at the backend's native 32-bit width.
pub type i32xN<S> = Vector<<S as NativeSimd>::i32xN>;
/// `u32` vector at the backend's native 32-bit width.
pub type u32xN<S> = Vector<<S as NativeSimd>::u32xN>;

/// `f64` vector at the backend's native 64-bit width.
pub type f64xN<S> = Vector<<S as NativeSimd>::f64xN>;
/// `i64` vector at the backend's native 64-bit width.
pub type i64xN<S> = Vector<<S as NativeSimd>::i64xN>;
/// `u64` vector at the backend's native 64-bit width.
pub type u64xN<S> = Vector<<S as NativeSimd>::u64xN>;

/// 2-lane `f32` vector (64 bits on 32-bit-float backends).
pub type f32x2<S> = Vector<<S as Simd>::f32x2>;
/// 2-lane `i32` vector.
pub type i32x2<S> = Vector<<S as Simd>::i32x2>;
/// 2-lane `u32` vector.
pub type u32x2<S> = Vector<<S as Simd>::u32x2>;

/// 3-lane `f32` vector stored in a 4-lane register (alpha-padded). The
/// 4th lane carries no semantic value but keeps register alignment.
pub type f32x3A<S> = Vector<<S as Simd3A>::f32x3A>;
/// 3-lane `i32` vector in alpha-padded 4-lane storage. See [`f32x3A`].
pub type i32x3A<S> = Vector<<S as Simd3A>::i32x3A>;
/// 3-lane `u32` vector in alpha-padded 4-lane storage. See [`f32x3A`].
pub type u32x3A<S> = Vector<<S as Simd3A>::u32x3A>;

/// 3-lane `usize` vector in a true 3-lane representation (e.g. SPIR-V `vec3`).
pub type usizex3<S> = Vector<<S as Simd3>::usizex3>;
/// 3-lane `f32` vector in a true 3-lane representation. See [`usizex3`].
pub type f32x3<S> = Vector<<S as Simd3>::f32x3>;
/// 3-lane `i32` vector in a true 3-lane representation. See [`usizex3`].
pub type i32x3<S> = Vector<<S as Simd3>::i32x3>;
/// 3-lane `u32` vector in a true 3-lane representation. See [`usizex3`].
pub type u32x3<S> = Vector<<S as Simd3>::u32x3>;

/// 4-lane `f32` vector (128 bits). Supports 4D linear-algebra helpers.
pub type f32x4<S> = Vector<<S as Simd>::f32x4>;
/// 4-lane `i32` vector (128 bits).
pub type i32x4<S> = Vector<<S as Simd>::i32x4>;
/// 4-lane `u32` vector (128 bits).
pub type u32x4<S> = Vector<<S as Simd>::u32x4>;

/// 8-lane `f32` vector (256 bits). Native on AVX2+ backends, emulated below.
pub type f32x8<S> = Vector<<S as Simd>::f32x8>;
/// 8-lane `i32` vector (256 bits).
pub type i32x8<S> = Vector<<S as Simd>::i32x8>;
/// 8-lane `u32` vector (256 bits).
pub type u32x8<S> = Vector<<S as Simd>::u32x8>;

/// 2-lane `f64` vector (128 bits).
pub type f64x2<S> = Vector<<S as Simd>::f64x2>;
/// 2-lane `i64` vector (128 bits).
pub type i64x2<S> = Vector<<S as Simd>::i64x2>;
/// 2-lane `u64` vector (128 bits).
pub type u64x2<S> = Vector<<S as Simd>::u64x2>;

/// 3-lane `f64` vector in alpha-padded 4-lane storage. See [`f32x3A`].
pub type f64x3A<S> = Vector<<S as Simd3A>::f64x3A>;
/// 3-lane `i64` vector in alpha-padded 4-lane storage. See [`f32x3A`].
pub type i64x3A<S> = Vector<<S as Simd3A>::i64x3A>;
/// 3-lane `u64` vector in alpha-padded 4-lane storage. See [`f32x3A`].
pub type u64x3A<S> = Vector<<S as Simd3A>::u64x3A>;

/// 3-lane `f64` vector in a true 3-lane representation. See [`usizex3`].
pub type f64x3<S> = Vector<<S as Simd3>::f64x3>;
/// 3-lane `i64` vector in a true 3-lane representation. See [`usizex3`].
pub type i64x3<S> = Vector<<S as Simd3>::i64x3>;
/// 3-lane `u64` vector in a true 3-lane representation. See [`usizex3`].
pub type u64x3<S> = Vector<<S as Simd3>::u64x3>;

/// 4-lane `f64` vector (256 bits). Supports 4D linear-algebra helpers.
pub type f64x4<S> = Vector<<S as Simd>::f64x4>;
/// 4-lane `i64` vector (256 bits).
pub type i64x4<S> = Vector<<S as Simd>::i64x4>;
/// 4-lane `u64` vector (256 bits).
pub type u64x4<S> = Vector<<S as Simd>::u64x4>;

/// 8-lane `f64` vector (512 bits). Native on AVX-512, emulated below.
pub type f64x8<S> = Vector<<S as Simd>::f64x8>;
/// 8-lane `i64` vector (512 bits).
pub type i64x8<S> = Vector<<S as Simd>::i64x8>;
/// 8-lane `u64` vector (512 bits).
pub type u64x8<S> = Vector<<S as Simd>::u64x8>;

/// 16-lane `f32` vector (512 bits). Native on AVX-512, emulated below.
pub type f32x16<S> = Vector<<S as Simd>::f32x16>;
/// 16-lane `i32` vector (512 bits).
pub type i32x16<S> = Vector<<S as Simd>::i32x16>;
/// 16-lane `u32` vector (512 bits).
pub type u32x16<S> = Vector<<S as Simd>::u32x16>;

/// 16-lane `f64` vector (1024 bits, always emulated).
pub type f64x16<S> = Vector<<S as Simd>::f64x16>;
/// 16-lane `i64` vector (1024 bits, always emulated).
pub type i64x16<S> = Vector<<S as Simd>::i64x16>;
/// 16-lane `u64` vector (1024 bits, always emulated).
pub type u64x16<S> = Vector<<S as Simd>::u64x16>;

// 16-bit integer families (see `Simd`).

/// `i16` vector at the backend's widest native 16-bit width.
pub type i16xN<S> = Vector<<S as NativeSimd>::i16xN>;
/// `u16` vector at the backend's widest native 16-bit width.
pub type u16xN<S> = Vector<<S as NativeSimd>::u16xN>;

/// 2-lane `i16` vector (32 bits).
pub type i16x2<S> = Vector<<S as Simd>::i16x2>;
/// 2-lane `u16` vector (32 bits).
pub type u16x2<S> = Vector<<S as Simd>::u16x2>;
/// 4-lane `i16` vector (64 bits).
pub type i16x4<S> = Vector<<S as Simd>::i16x4>;
/// 4-lane `u16` vector (64 bits).
pub type u16x4<S> = Vector<<S as Simd>::u16x4>;
/// 8-lane `i16` vector (128 bits).
pub type i16x8<S> = Vector<<S as Simd>::i16x8>;
/// 8-lane `u16` vector (128 bits).
pub type u16x8<S> = Vector<<S as Simd>::u16x8>;
/// 16-lane `i16` vector (256 bits). Native on AVX2+, emulated below.
pub type i16x16<S> = Vector<<S as Simd>::i16x16>;
/// 16-lane `u16` vector (256 bits). Native on AVX2+, emulated below.
pub type u16x16<S> = Vector<<S as Simd>::u16x16>;

// 8-bit integer families (native width only, see `NativeSimd`).

/// `i8` vector at the backend's widest native 8-bit width (16 lanes on SSE, 32 on AVX2).
pub type i8xN<S> = Vector<<S as NativeSimd>::i8xN>;
/// `u8` vector at the backend's widest native 8-bit width (16 lanes on SSE, 32 on AVX2).
pub type u8xN<S> = Vector<<S as NativeSimd>::u8xN>;

/// 16-lane `i8` vector (128 bits), the same width on every backend.
pub type i8x16<S> = Vector<<S as Simd>::i8x16>;
/// 16-lane `u8` vector (128 bits), the same width on every backend.
pub type u8x16<S> = Vector<<S as Simd>::u8x16>;

/// 2-lane `i8` vector. See [`Simd::i8x2`].
pub type i8x2<S> = Vector<<S as Simd>::i8x2>;
/// 2-lane `u8` vector.
pub type u8x2<S> = Vector<<S as Simd>::u8x2>;
/// 4-lane `i8` vector. See [`Simd::i8x4`].
pub type i8x4<S> = Vector<<S as Simd>::i8x4>;
/// 4-lane `u8` vector.
pub type u8x4<S> = Vector<<S as Simd>::u8x4>;
/// 8-lane `i8` vector. See [`Simd::i8x8`].
pub type i8x8<S> = Vector<<S as Simd>::i8x8>;
/// 8-lane `u8` vector.
pub type u8x8<S> = Vector<<S as Simd>::u8x8>;

/// Generates a backend-local `aliases` submodule with every vector type
/// alias from [`crate::simd`] pre-bound to a specific backend.
///
/// Each backend invokes this once at its module root with its own
/// `<Backend>` type, producing `aliases::f32x4`, `aliases::f32xN`, etc.
/// without the `<S>` parameter -- so user code that imports
/// `backend::x86_v3::prelude::*` (which re-exports `aliases::*`) gets the
/// short names by default.
macro_rules! decl_aliases {
    ($simd:ty) => {
        #[allow(non_camel_case_types)]
        pub mod aliases {
            use super::*;

            pub type f32xN = crate::simd::f32xN<$simd>;
            pub type i32xN = crate::simd::i32xN<$simd>;
            pub type u32xN = crate::simd::u32xN<$simd>;

            pub type f64xN = crate::simd::f64xN<$simd>;
            pub type i64xN = crate::simd::i64xN<$simd>;
            pub type u64xN = crate::simd::u64xN<$simd>;

            pub type f32x2 = crate::simd::f32x2<$simd>;
            pub type i32x2 = crate::simd::i32x2<$simd>;
            pub type u32x2 = crate::simd::u32x2<$simd>;

            pub type f32x3A = crate::simd::f32x3A<$simd>;
            pub type i32x3A = crate::simd::i32x3A<$simd>;
            pub type u32x3A = crate::simd::u32x3A<$simd>;

            pub type usizex3 = crate::simd::usizex3<$simd>;
            pub type f32x3 = crate::simd::f32x3<$simd>;
            pub type i32x3 = crate::simd::i32x3<$simd>;
            pub type u32x3 = crate::simd::u32x3<$simd>;

            pub type f32x4 = crate::simd::f32x4<$simd>;
            pub type i32x4 = crate::simd::i32x4<$simd>;
            pub type u32x4 = crate::simd::u32x4<$simd>;

            pub type f32x8 = crate::simd::f32x8<$simd>;
            pub type i32x8 = crate::simd::i32x8<$simd>;
            pub type u32x8 = crate::simd::u32x8<$simd>;

            pub type f64x2 = crate::simd::f64x2<$simd>;
            pub type i64x2 = crate::simd::i64x2<$simd>;
            pub type u64x2 = crate::simd::u64x2<$simd>;

            pub type f64x3A = crate::simd::f64x3A<$simd>;
            pub type i64x3A = crate::simd::i64x3A<$simd>;
            pub type u64x3A = crate::simd::u64x3A<$simd>;

            pub type f64x3 = crate::simd::f64x3<$simd>;
            pub type i64x3 = crate::simd::i64x3<$simd>;
            pub type u64x3 = crate::simd::u64x3<$simd>;

            pub type f64x4 = crate::simd::f64x4<$simd>;
            pub type i64x4 = crate::simd::i64x4<$simd>;
            pub type u64x4 = crate::simd::u64x4<$simd>;

            pub type f64x8 = crate::simd::f64x8<$simd>;
            pub type i64x8 = crate::simd::i64x8<$simd>;
            pub type u64x8 = crate::simd::u64x8<$simd>;

            pub type f32x16 = crate::simd::f32x16<$simd>;
            pub type i32x16 = crate::simd::i32x16<$simd>;
            pub type u32x16 = crate::simd::u32x16<$simd>;

            pub type f64x16 = crate::simd::f64x16<$simd>;
            pub type i64x16 = crate::simd::i64x16<$simd>;
            pub type u64x16 = crate::simd::u64x16<$simd>;
        }
    };
}

use crate::vector::{
    CastVector, ConcatVector, ExtendVector, FloatVector, FloatVectorWithRegister, FullyInteroperable as FIV,
    GenericVector, IndexableVector, LinAlg3Vector, LinAlg4Vector, PackedFloatVector, Sad16Vector, Sad32Vector,
    Sad64Vector, SignedIntegerVector, SignedIntegerVectorWithRegister, SwizzleVector, UnsignedIntegerVector,
    UnsignedIntegerVectorWithRegister,
};

/// Vector-level mirror of [`NativeSimd`]: names the native-width register
/// types as [`Vector`] wrappers instead of raw registers.
///
/// Auto-implemented for every [`NativeSimd`] backend, so user code can use
/// `S::f32xN`, `S::i64xN`, etc. as fully-typed vector aliases. When you
/// also need the underlying [`crate::register::Register`] type
/// to be reachable, bound on [`NativeSimdVectorsWithRegisters`] instead.
pub trait NativeSimdVectors: NativeIsa {
    /// Native-width `f32` vector. Lane count matches [`NativeIsa::Native32Width`].
    type f32xN: FloatVector<Lanes = <Self as NativeIsa>::Native32Width, Element = f32>
        + IndexableVector<<Self as NativeSimdVectors>::u32xN>;
    /// Native-width signed 32-bit integer vector.
    type i32xN: SignedIntegerVector<Lanes = <Self as NativeIsa>::Native32Width, Element = i32>
        + IndexableVector<<Self as NativeSimdVectors>::u32xN>;
    /// Native-width unsigned 32-bit integer vector.
    type u32xN: UnsignedIntegerVector<Lanes = <Self as NativeIsa>::Native32Width, Element = u32>
        + IndexableVector<<Self as NativeSimdVectors>::u32xN>
        + Sad64Vector<<Self as NativeSimdVectors>::u64xN>;

    /// Native-width `f64` vector. Lane count matches [`NativeIsa::Native64Width`].
    type f64xN: FloatVector<Lanes = <Self as NativeIsa>::Native64Width, Element = f64>
        + IndexableVector<<Self as NativeSimdVectors>::u64xN>;
    /// Native-width signed 64-bit integer vector.
    type i64xN: SignedIntegerVector<Lanes = <Self as NativeIsa>::Native64Width, Element = i64>
        + IndexableVector<<Self as NativeSimdVectors>::u64xN>;
    /// Native-width unsigned 64-bit integer vector.
    type u64xN: UnsignedIntegerVector<Lanes = <Self as NativeIsa>::Native64Width, Element = u64>
        + IndexableVector<<Self as NativeSimdVectors>::u64xN>;

    /// Native-width signed 16-bit integer vector. Lane count matches [`NativeIsa::Native16Width`].
    type i16xN: SignedIntegerVector<Lanes = <Self as NativeIsa>::Native16Width, Element = i16> + SwizzleVector;
    /// Native-width unsigned 16-bit integer vector.
    type u16xN: UnsignedIntegerVector<Lanes = <Self as NativeIsa>::Native16Width, Element = u16>
        + SwizzleVector
        + Sad32Vector<Self::u32xN>
        + Sad64Vector<Self::u64xN>;

    /// Native-width signed 8-bit integer vector. Lane count matches [`NativeIsa::Native8Width`].
    type i8xN: SignedIntegerVector<Lanes = <Self as NativeIsa>::Native8Width, Element = i8> + SwizzleVector;
    /// Native-width unsigned 8-bit integer vector.
    type u8xN: UnsignedIntegerVector<Lanes = <Self as NativeIsa>::Native8Width, Element = u8>
        + SwizzleVector
        + Sad16Vector<Self::u16xN>
        + Sad32Vector<Self::u32xN>
        + Sad64Vector<Self::u64xN>;
}

/// Marker companion to [`NativeSimdVectors`] that also exposes each vector's
/// underlying [`crate::register::Register`] type via
/// [`VectorWithRegister`](crate::vector::VectorWithRegister).
///
/// Bind on this when generic code needs to round-trip a vector to its raw
/// register storage (rare; the usual reason is FFI or hand-tuned codegen).
pub trait NativeSimdVectorsWithRegisters: NativeSimd + NativeSimdVectors<
    // 32xN
    f32xN: FloatVectorWithRegister<
        Register = <Self as NativeSimd>::f32xN,
        SignedBits = <Self as NativeSimdVectors>::i32xN,
        Bits = <Self as NativeSimdVectors>::u32xN,
    > + FIV<<Self as NativeSimdVectors>::i32xN, <Self as NativeSimdVectors>::u32xN>,
    i32xN: SignedIntegerVectorWithRegister<Register = <Self as NativeSimd>::i32xN>
        + FIV<<Self as NativeSimdVectors>::f32xN, <Self as NativeSimdVectors>::u32xN>,
    u32xN: UnsignedIntegerVectorWithRegister<Register = <Self as NativeSimd>::u32xN>
        + FIV<<Self as NativeSimdVectors>::f32xN, <Self as NativeSimdVectors>::i32xN>,

    // 64xN
    f64xN: FloatVectorWithRegister<
        Register = <Self as NativeSimd>::f64xN,
        SignedBits = <Self as NativeSimdVectors>::i64xN,
        Bits = <Self as NativeSimdVectors>::u64xN,
    > + FIV<<Self as NativeSimdVectors>::i64xN, <Self as NativeSimdVectors>::u64xN>,
    i64xN: SignedIntegerVectorWithRegister<Register = <Self as NativeSimd>::i64xN>
        + FIV<<Self as NativeSimdVectors>::f64xN, <Self as NativeSimdVectors>::u64xN>,
    u64xN: UnsignedIntegerVectorWithRegister<Register = <Self as NativeSimd>::u64xN>
        + FIV<<Self as NativeSimdVectors>::f64xN, <Self as NativeSimdVectors>::i64xN>,

    // native 16-bit / 8-bit (no float partner, so no FIV)
    i16xN: SignedIntegerVectorWithRegister<Register = <Self as NativeSimd>::i16xN>,
    u16xN: UnsignedIntegerVectorWithRegister<Register = <Self as NativeSimd>::u16xN>,
    i8xN:  SignedIntegerVectorWithRegister<Register = <Self as NativeSimd>::i8xN>,
    u8xN:  UnsignedIntegerVectorWithRegister<Register = <Self as NativeSimd>::u8xN>,
>
{}

impl<S: NativeSimd> NativeSimdVectors for S {
    type f32xN = Vector<<Self as NativeSimd>::f32xN>;
    type i32xN = Vector<<Self as NativeSimd>::i32xN>;
    type u32xN = Vector<<Self as NativeSimd>::u32xN>;

    type f64xN = Vector<<Self as NativeSimd>::f64xN>;
    type i64xN = Vector<<Self as NativeSimd>::i64xN>;
    type u64xN = Vector<<Self as NativeSimd>::u64xN>;

    type i16xN = Vector<<Self as NativeSimd>::i16xN>;
    type u16xN = Vector<<Self as NativeSimd>::u16xN>;
    type i8xN = Vector<<Self as NativeSimd>::i8xN>;
    type u8xN = Vector<<Self as NativeSimd>::u8xN>;
}

impl<S: NativeSimd> NativeSimdVectorsWithRegisters for S {}

/// Vector-level mirror of [`Simd`]: names every fixed-width register type
/// the backend offers, as [`Vector`] wrappers.
///
/// Auto-implemented for every [`Simd`] backend. Provides the `usizexN`,
/// `f32xN`/`i32xN`/`u32xN`, and `f64xN`/`i64xN`/`u64xN` groups at lane
/// counts 2, 4, 8, and 16 -- the vector equivalents of the same names on
/// [`Simd`]. When you also need each vector's underlying
/// [`crate::register::Register`], bound on
/// [`SimdVectorsWithRegisters`] instead.
#[rustfmt::skip]
pub trait SimdVectors: NativeSimdVectors {type usizex2: UnsignedIntegerVector<Lanes = U2, Element = crate::element::USize>
        + ConcatVector<Vector<crate::element::USize>> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex2, <Self as SimdVectors>::u32x2, <Self as SimdVectors>::u64x2>;
    type usizex4: UnsignedIntegerVector<Lanes = U4, Element = crate::element::USize>
        + ConcatVector<<Self as SimdVectors>::usizex2> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex4, <Self as SimdVectors>::u32x4, <Self as SimdVectors>::u64x4>;
    type usizex8: UnsignedIntegerVector<Lanes = U8, Element = crate::element::USize>
        + ConcatVector<<Self as SimdVectors>::usizex4> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex8, <Self as SimdVectors>::u32x8, <Self as SimdVectors>::u64x8>;
    type usizex16: UnsignedIntegerVector<Lanes = U16, Element = crate::element::USize>
        + ConcatVector<<Self as SimdVectors>::usizex8> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex16, <Self as SimdVectors>::u32x16, <Self as SimdVectors>::u64x16>;

    type f32x2: FloatVector<Lanes = U2, Element = f32>
        + CastVector<<Self as SimdVectors>::f64x2>
        + ConcatVector<Vector<f32>> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex2, <Self as SimdVectors>::u32x2, <Self as SimdVectors>::u64x2>
        + CastVector<<Self as SimdVectors>::i16x2> + CastVector<<Self as SimdVectors>::i32x2>
        + CastVector<<Self as SimdVectors>::i64x2> + CastVector<<Self as SimdVectors>::i8x2>
        + CastVector<<Self as SimdVectors>::u16x2> + CastVector<<Self as SimdVectors>::u32x2>
        + CastVector<<Self as SimdVectors>::u64x2> + CastVector<<Self as SimdVectors>::u8x2>;
    type i32x2: SignedIntegerVector<Lanes = U2, Element = i32>
        + CastVector<<Self as SimdVectors>::i64x2> + CastVector<<Self as SimdVectors>::i64x2>
        + ConcatVector<Vector<i32>> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex2, <Self as SimdVectors>::u32x2, <Self as SimdVectors>::u64x2>
        + CastVector<<Self as SimdVectors>::f32x2> + CastVector<<Self as SimdVectors>::f64x2>
        + CastVector<<Self as SimdVectors>::i16x2> + CastVector<<Self as SimdVectors>::i8x2>
        + CastVector<<Self as SimdVectors>::u16x2> + CastVector<<Self as SimdVectors>::u32x2>
        + CastVector<<Self as SimdVectors>::u64x2> + CastVector<<Self as SimdVectors>::u8x2>;
    type u32x2: UnsignedIntegerVector<Lanes = U2, Element = u32> + Sad64Vector<Vector<u64>>
        + CastVector<<Self as SimdVectors>::u64x2> + CastVector<<Self as SimdVectors>::u64x2>
        + ConcatVector<Vector<u32>> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex2, <Self as SimdVectors>::u32x2, <Self as SimdVectors>::u64x2>
        + CastVector<<Self as SimdVectors>::f32x2> + CastVector<<Self as SimdVectors>::f64x2>
        + CastVector<<Self as SimdVectors>::i16x2> + CastVector<<Self as SimdVectors>::i32x2>
        + CastVector<<Self as SimdVectors>::i64x2> + CastVector<<Self as SimdVectors>::i8x2>
        + CastVector<<Self as SimdVectors>::u16x2> + CastVector<<Self as SimdVectors>::u8x2>;

    type f32x4: FloatVector<Lanes = U4, Element = f32>
        + CastVector<<Self as SimdVectors>::f64x4>
        + ConcatVector<<Self as SimdVectors>::f32x2> + SwizzleVector
        + LinAlg4Vector
        + VectorIndexedBy<<Self as SimdVectors>::usizex4, <Self as SimdVectors>::u32x4, <Self as SimdVectors>::u64x4>
        + CastVector<<Self as SimdVectors>::i16x4> + CastVector<<Self as SimdVectors>::i32x4>
        + CastVector<<Self as SimdVectors>::i64x4> + CastVector<<Self as SimdVectors>::i8x4>
        + CastVector<<Self as SimdVectors>::u16x4> + CastVector<<Self as SimdVectors>::u32x4>
        + CastVector<<Self as SimdVectors>::u64x4> + CastVector<<Self as SimdVectors>::u8x4>;
    type i32x4: SignedIntegerVector<Lanes = U4, Element = i32>
        + CastVector<<Self as SimdVectors>::i64x4> + CastVector<<Self as SimdVectors>::i64x4>
        + ConcatVector<<Self as SimdVectors>::i32x2> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex4, <Self as SimdVectors>::u32x4, <Self as SimdVectors>::u64x4>
        + CastVector<<Self as SimdVectors>::f32x4> + CastVector<<Self as SimdVectors>::f64x4>
        + CastVector<<Self as SimdVectors>::i16x4> + CastVector<<Self as SimdVectors>::i8x4>
        + CastVector<<Self as SimdVectors>::u16x4> + CastVector<<Self as SimdVectors>::u32x4>
        + CastVector<<Self as SimdVectors>::u64x4> + CastVector<<Self as SimdVectors>::u8x4>;
    type u32x4: UnsignedIntegerVector<Lanes = U4, Element = u32> + Sad64Vector<<Self as SimdVectors>::u64x2>
        + CastVector<<Self as SimdVectors>::u64x4> + CastVector<<Self as SimdVectors>::u64x4>
        + ConcatVector<<Self as SimdVectors>::u32x2> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex4, <Self as SimdVectors>::u32x4, <Self as SimdVectors>::u64x4>
        + CastVector<<Self as SimdVectors>::f32x4> + CastVector<<Self as SimdVectors>::f64x4>
        + CastVector<<Self as SimdVectors>::i16x4> + CastVector<<Self as SimdVectors>::i32x4>
        + CastVector<<Self as SimdVectors>::i64x4> + CastVector<<Self as SimdVectors>::i8x4>
        + CastVector<<Self as SimdVectors>::u16x4> + CastVector<<Self as SimdVectors>::u8x4>;

    type f32x8: FloatVector<Lanes = U8, Element = f32>
        + CastVector<<Self as SimdVectors>::f64x8>
        + ConcatVector<<Self as SimdVectors>::f32x4> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex8, <Self as SimdVectors>::u32x8, <Self as SimdVectors>::u64x8>
        + CastVector<<Self as SimdVectors>::i16x8> + CastVector<<Self as SimdVectors>::i32x8>
        + CastVector<<Self as SimdVectors>::i64x8> + CastVector<<Self as SimdVectors>::i8x8>
        + CastVector<<Self as SimdVectors>::u16x8> + CastVector<<Self as SimdVectors>::u32x8>
        + CastVector<<Self as SimdVectors>::u64x8> + CastVector<<Self as SimdVectors>::u8x8>;
    type i32x8: SignedIntegerVector<Lanes = U8, Element = i32>
        + CastVector<<Self as SimdVectors>::i64x8> + CastVector<<Self as SimdVectors>::i64x8>
        + ConcatVector<<Self as SimdVectors>::i32x4> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex8, <Self as SimdVectors>::u32x8, <Self as SimdVectors>::u64x8>
        + CastVector<<Self as SimdVectors>::f32x8> + CastVector<<Self as SimdVectors>::f64x8>
        + CastVector<<Self as SimdVectors>::i16x8> + CastVector<<Self as SimdVectors>::i8x8>
        + CastVector<<Self as SimdVectors>::u16x8> + CastVector<<Self as SimdVectors>::u32x8>
        + CastVector<<Self as SimdVectors>::u64x8> + CastVector<<Self as SimdVectors>::u8x8>;
    type u32x8: UnsignedIntegerVector<Lanes = U8, Element = u32> + Sad64Vector<<Self as SimdVectors>::u64x4>
        + CastVector<<Self as SimdVectors>::u64x8> + CastVector<<Self as SimdVectors>::u64x8>
        + ConcatVector<<Self as SimdVectors>::u32x4> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex8, <Self as SimdVectors>::u32x8, <Self as SimdVectors>::u64x8>
        + CastVector<<Self as SimdVectors>::f32x8> + CastVector<<Self as SimdVectors>::f64x8>
        + CastVector<<Self as SimdVectors>::i16x8> + CastVector<<Self as SimdVectors>::i32x8>
        + CastVector<<Self as SimdVectors>::i64x8> + CastVector<<Self as SimdVectors>::i8x8>
        + CastVector<<Self as SimdVectors>::u16x8> + CastVector<<Self as SimdVectors>::u8x8>;

    type f64x2: FloatVector<Lanes = U2, Element = f64>
        + CastVector<<Self as SimdVectors>::f32x2>
        + ConcatVector<Vector<f64>> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex2, <Self as SimdVectors>::u32x2, <Self as SimdVectors>::u64x2>
        + CastVector<<Self as SimdVectors>::i16x2> + CastVector<<Self as SimdVectors>::i32x2>
        + CastVector<<Self as SimdVectors>::i64x2> + CastVector<<Self as SimdVectors>::i8x2>
        + CastVector<<Self as SimdVectors>::u16x2> + CastVector<<Self as SimdVectors>::u32x2>
        + CastVector<<Self as SimdVectors>::u64x2> + CastVector<<Self as SimdVectors>::u8x2>;
    type i64x2: SignedIntegerVector<Lanes = U2, Element = i64>
        + CastVector<<Self as SimdVectors>::i32x2>
        + ConcatVector<Vector<i64>> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex2, <Self as SimdVectors>::u32x2, <Self as SimdVectors>::u64x2>
        + CastVector<<Self as SimdVectors>::f32x2> + CastVector<<Self as SimdVectors>::f64x2>
        + CastVector<<Self as SimdVectors>::i16x2> + CastVector<<Self as SimdVectors>::i8x2>
        + CastVector<<Self as SimdVectors>::u16x2> + CastVector<<Self as SimdVectors>::u32x2>
        + CastVector<<Self as SimdVectors>::u64x2> + CastVector<<Self as SimdVectors>::u8x2>;
    type u64x2: UnsignedIntegerVector<Lanes = U2, Element = u64>
        + CastVector<<Self as SimdVectors>::u32x2>
        + ConcatVector<Vector<u64>> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex2, <Self as SimdVectors>::u32x2, <Self as SimdVectors>::u64x2>
        + CastVector<<Self as SimdVectors>::f32x2> + CastVector<<Self as SimdVectors>::f64x2>
        + CastVector<<Self as SimdVectors>::i16x2> + CastVector<<Self as SimdVectors>::i32x2>
        + CastVector<<Self as SimdVectors>::i64x2> + CastVector<<Self as SimdVectors>::i8x2>
        + CastVector<<Self as SimdVectors>::u16x2> + CastVector<<Self as SimdVectors>::u8x2>;

    type f64x4: FloatVector<Lanes = U4, Element = f64>
        + CastVector<<Self as SimdVectors>::f32x4>
        + ConcatVector<<Self as SimdVectors>::f64x2> + SwizzleVector
        + LinAlg4Vector
        + VectorIndexedBy<<Self as SimdVectors>::usizex4, <Self as SimdVectors>::u32x4, <Self as SimdVectors>::u64x4>
        + CastVector<<Self as SimdVectors>::i16x4> + CastVector<<Self as SimdVectors>::i32x4>
        + CastVector<<Self as SimdVectors>::i64x4> + CastVector<<Self as SimdVectors>::i8x4>
        + CastVector<<Self as SimdVectors>::u16x4> + CastVector<<Self as SimdVectors>::u32x4>
        + CastVector<<Self as SimdVectors>::u64x4> + CastVector<<Self as SimdVectors>::u8x4>;
    type i64x4: SignedIntegerVector<Lanes = U4, Element = i64>
        + CastVector<<Self as SimdVectors>::i32x4>
        + ConcatVector<<Self as SimdVectors>::i64x2> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex4, <Self as SimdVectors>::u32x4, <Self as SimdVectors>::u64x4>
        + CastVector<<Self as SimdVectors>::f32x4> + CastVector<<Self as SimdVectors>::f64x4>
        + CastVector<<Self as SimdVectors>::i16x4> + CastVector<<Self as SimdVectors>::i8x4>
        + CastVector<<Self as SimdVectors>::u16x4> + CastVector<<Self as SimdVectors>::u32x4>
        + CastVector<<Self as SimdVectors>::u64x4> + CastVector<<Self as SimdVectors>::u8x4>;
    type u64x4: UnsignedIntegerVector<Lanes = U4, Element = u64>
        + CastVector<<Self as SimdVectors>::u32x4>
        + ConcatVector<<Self as SimdVectors>::u64x2> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex4, <Self as SimdVectors>::u32x4, <Self as SimdVectors>::u64x4>
        + CastVector<<Self as SimdVectors>::f32x4> + CastVector<<Self as SimdVectors>::f64x4>
        + CastVector<<Self as SimdVectors>::i16x4> + CastVector<<Self as SimdVectors>::i32x4>
        + CastVector<<Self as SimdVectors>::i64x4> + CastVector<<Self as SimdVectors>::i8x4>
        + CastVector<<Self as SimdVectors>::u16x4> + CastVector<<Self as SimdVectors>::u8x4>;

    type f64x8: FloatVector<Lanes = U8, Element = f64>
        + CastVector<<Self as SimdVectors>::f32x8>
        + ConcatVector<<Self as SimdVectors>::f64x4> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex8, <Self as SimdVectors>::u32x8, <Self as SimdVectors>::u64x8>
        + CastVector<<Self as SimdVectors>::i16x8> + CastVector<<Self as SimdVectors>::i32x8>
        + CastVector<<Self as SimdVectors>::i64x8> + CastVector<<Self as SimdVectors>::i8x8>
        + CastVector<<Self as SimdVectors>::u16x8> + CastVector<<Self as SimdVectors>::u32x8>
        + CastVector<<Self as SimdVectors>::u64x8> + CastVector<<Self as SimdVectors>::u8x8>;
    type i64x8: SignedIntegerVector<Lanes = U8, Element = i64>
        + CastVector<<Self as SimdVectors>::i32x8>
        + ConcatVector<<Self as SimdVectors>::i64x4> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex8, <Self as SimdVectors>::u32x8, <Self as SimdVectors>::u64x8>
        + CastVector<<Self as SimdVectors>::f32x8> + CastVector<<Self as SimdVectors>::f64x8>
        + CastVector<<Self as SimdVectors>::i16x8> + CastVector<<Self as SimdVectors>::i8x8>
        + CastVector<<Self as SimdVectors>::u16x8> + CastVector<<Self as SimdVectors>::u32x8>
        + CastVector<<Self as SimdVectors>::u64x8> + CastVector<<Self as SimdVectors>::u8x8>;
    type u64x8: UnsignedIntegerVector<Lanes = U8, Element = u64>
        + CastVector<<Self as SimdVectors>::u32x8>
        + ConcatVector<<Self as SimdVectors>::u64x4> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex8, <Self as SimdVectors>::u32x8, <Self as SimdVectors>::u64x8>
        + CastVector<<Self as SimdVectors>::f32x8> + CastVector<<Self as SimdVectors>::f64x8>
        + CastVector<<Self as SimdVectors>::i16x8> + CastVector<<Self as SimdVectors>::i32x8>
        + CastVector<<Self as SimdVectors>::i64x8> + CastVector<<Self as SimdVectors>::i8x8>
        + CastVector<<Self as SimdVectors>::u16x8> + CastVector<<Self as SimdVectors>::u8x8>;

    type f32x16: FloatVector<Lanes = U16, Element = f32>
        + CastVector<<Self as SimdVectors>::f64x16>
        + ConcatVector<<Self as SimdVectors>::f32x8> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex16, <Self as SimdVectors>::u32x16, <Self as SimdVectors>::u64x16>
        + CastVector<<Self as SimdVectors>::i16x16> + CastVector<<Self as SimdVectors>::i32x16>
        + CastVector<<Self as SimdVectors>::i64x16> + CastVector<<Self as SimdVectors>::i8x16>
        + CastVector<<Self as SimdVectors>::u16x16> + CastVector<<Self as SimdVectors>::u32x16>
        + CastVector<<Self as SimdVectors>::u64x16> + CastVector<<Self as SimdVectors>::u8x16>;
    type i32x16: SignedIntegerVector<Lanes = U16, Element = i32>
        + CastVector<<Self as SimdVectors>::i64x16> + CastVector<<Self as SimdVectors>::i64x16>
        + ConcatVector<<Self as SimdVectors>::i32x8> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex16, <Self as SimdVectors>::u32x16, <Self as SimdVectors>::u64x16>
        + CastVector<<Self as SimdVectors>::f32x16> + CastVector<<Self as SimdVectors>::f64x16>
        + CastVector<<Self as SimdVectors>::i16x16> + CastVector<<Self as SimdVectors>::i8x16>
        + CastVector<<Self as SimdVectors>::u16x16> + CastVector<<Self as SimdVectors>::u32x16>
        + CastVector<<Self as SimdVectors>::u64x16> + CastVector<<Self as SimdVectors>::u8x16>;
    type u32x16: UnsignedIntegerVector<Lanes = U16, Element = u32> + Sad64Vector<<Self as SimdVectors>::u64x8>
        + CastVector<<Self as SimdVectors>::u64x16> + CastVector<<Self as SimdVectors>::u64x16>
        + ConcatVector<<Self as SimdVectors>::u32x8> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex16, <Self as SimdVectors>::u32x16, <Self as SimdVectors>::u64x16>
        + CastVector<<Self as SimdVectors>::f32x16> + CastVector<<Self as SimdVectors>::f64x16>
        + CastVector<<Self as SimdVectors>::i16x16> + CastVector<<Self as SimdVectors>::i32x16>
        + CastVector<<Self as SimdVectors>::i64x16> + CastVector<<Self as SimdVectors>::i8x16>
        + CastVector<<Self as SimdVectors>::u16x16> + CastVector<<Self as SimdVectors>::u8x16>;

    type f64x16: FloatVector<Lanes = U16, Element = f64>
        + CastVector<<Self as SimdVectors>::f32x16>
        + ConcatVector<<Self as SimdVectors>::f64x8> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex16, <Self as SimdVectors>::u32x16, <Self as SimdVectors>::u64x16>
        + CastVector<<Self as SimdVectors>::i16x16> + CastVector<<Self as SimdVectors>::i32x16>
        + CastVector<<Self as SimdVectors>::i64x16> + CastVector<<Self as SimdVectors>::i8x16>
        + CastVector<<Self as SimdVectors>::u16x16> + CastVector<<Self as SimdVectors>::u32x16>
        + CastVector<<Self as SimdVectors>::u64x16> + CastVector<<Self as SimdVectors>::u8x16>;
    type i64x16: SignedIntegerVector<Lanes = U16, Element = i64>
        + CastVector<<Self as SimdVectors>::i32x16>
        + ConcatVector<<Self as SimdVectors>::i64x8> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex16, <Self as SimdVectors>::u32x16, <Self as SimdVectors>::u64x16>
        + CastVector<<Self as SimdVectors>::f32x16> + CastVector<<Self as SimdVectors>::f64x16>
        + CastVector<<Self as SimdVectors>::i16x16> + CastVector<<Self as SimdVectors>::i8x16>
        + CastVector<<Self as SimdVectors>::u16x16> + CastVector<<Self as SimdVectors>::u32x16>
        + CastVector<<Self as SimdVectors>::u64x16> + CastVector<<Self as SimdVectors>::u8x16>;
    type u64x16: UnsignedIntegerVector<Lanes = U16, Element = u64>
        + CastVector<<Self as SimdVectors>::u32x16>
        + ConcatVector<<Self as SimdVectors>::u64x8> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex16, <Self as SimdVectors>::u32x16, <Self as SimdVectors>::u64x16>
        + CastVector<<Self as SimdVectors>::f32x16> + CastVector<<Self as SimdVectors>::f64x16>
        + CastVector<<Self as SimdVectors>::i16x16> + CastVector<<Self as SimdVectors>::i32x16>
        + CastVector<<Self as SimdVectors>::i64x16> + CastVector<<Self as SimdVectors>::i8x16>
        + CastVector<<Self as SimdVectors>::u16x16> + CastVector<<Self as SimdVectors>::u8x16>;

    // ===== 8-bit and 16-bit integer vector families (mirror of the same on `Simd`) =====
    type i8x16: SignedIntegerVector<Lanes = U16, Element = i8, Unsigned = <Self as SimdVectors>::u8x16, Signed = <Self as SimdVectors>::i8x16>
        + CastVector<<Self as SimdVectors>::i32x16> + CastVector<<Self as SimdVectors>::i32x16> + ConcatVector<<Self as SimdVectors>::i8x8> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex16, <Self as SimdVectors>::u32x16, <Self as SimdVectors>::u64x16>
        + CastVector<<Self as SimdVectors>::f32x16> + CastVector<<Self as SimdVectors>::f64x16>
        + CastVector<<Self as SimdVectors>::i16x16> + CastVector<<Self as SimdVectors>::i64x16>
        + CastVector<<Self as SimdVectors>::u16x16> + CastVector<<Self as SimdVectors>::u32x16>
        + CastVector<<Self as SimdVectors>::u64x16> + CastVector<<Self as SimdVectors>::u8x16>;
    type u8x16: UnsignedIntegerVector<Lanes = U16, Element = u8, Unsigned = <Self as SimdVectors>::u8x16, Signed = <Self as SimdVectors>::i8x16>
        + CastVector<<Self as SimdVectors>::u32x16> + CastVector<<Self as SimdVectors>::u32x16> + ConcatVector<<Self as SimdVectors>::u8x8> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex16, <Self as SimdVectors>::u32x16, <Self as SimdVectors>::u64x16>
        + PackedF8Vector<<Self as SimdVectors>::f32x16>
        // Vector-level mirror of the `Simd::u8x16` SAD bounds.
        + Sad16Vector<<Self as SimdVectors>::u16x8>
        + Sad32Vector<<Self as SimdVectors>::u32x4>
        + Sad64Vector<<Self as SimdVectors>::u64x2>
        + CastVector<<Self as SimdVectors>::f32x16> + CastVector<<Self as SimdVectors>::f64x16>
        + CastVector<<Self as SimdVectors>::i16x16> + CastVector<<Self as SimdVectors>::i32x16>
        + CastVector<<Self as SimdVectors>::i64x16> + CastVector<<Self as SimdVectors>::i8x16>
        + CastVector<<Self as SimdVectors>::u16x16> + CastVector<<Self as SimdVectors>::u64x16>;

    type i8x2: SignedIntegerVector<Lanes = U2, Element = i8, Unsigned = <Self as SimdVectors>::u8x2, Signed = <Self as SimdVectors>::i8x2>
        + CastVector<<Self as SimdVectors>::i32x2> + CastVector<<Self as SimdVectors>::i32x2> + ConcatVector<Vector<i8>> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex2, <Self as SimdVectors>::u32x2, <Self as SimdVectors>::u64x2>
        + CastVector<<Self as SimdVectors>::f32x2> + CastVector<<Self as SimdVectors>::f64x2>
        + CastVector<<Self as SimdVectors>::i16x2> + CastVector<<Self as SimdVectors>::i64x2>
        + CastVector<<Self as SimdVectors>::u16x2> + CastVector<<Self as SimdVectors>::u32x2>
        + CastVector<<Self as SimdVectors>::u64x2> + CastVector<<Self as SimdVectors>::u8x2>;
    type u8x2: UnsignedIntegerVector<Lanes = U2, Element = u8, Unsigned = <Self as SimdVectors>::u8x2, Signed = <Self as SimdVectors>::i8x2>
        + CastVector<<Self as SimdVectors>::u32x2> + CastVector<<Self as SimdVectors>::u32x2> + ConcatVector<Vector<u8>> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex2, <Self as SimdVectors>::u32x2, <Self as SimdVectors>::u64x2>
        + PackedF8Vector<<Self as SimdVectors>::f32x2>
        + Sad16Vector<Vector<u16>> + Sad32Vector<Vector<u32>> + Sad64Vector<Vector<u64>>
        + CastVector<<Self as SimdVectors>::f32x2> + CastVector<<Self as SimdVectors>::f64x2>
        + CastVector<<Self as SimdVectors>::i16x2> + CastVector<<Self as SimdVectors>::i32x2>
        + CastVector<<Self as SimdVectors>::i64x2> + CastVector<<Self as SimdVectors>::i8x2>
        + CastVector<<Self as SimdVectors>::u16x2> + CastVector<<Self as SimdVectors>::u64x2>;

    type i8x4: SignedIntegerVector<Lanes = U4, Element = i8, Unsigned = <Self as SimdVectors>::u8x4, Signed = <Self as SimdVectors>::i8x4>
        + CastVector<<Self as SimdVectors>::i32x4> + CastVector<<Self as SimdVectors>::i32x4> + ConcatVector<<Self as SimdVectors>::i8x2> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex4, <Self as SimdVectors>::u32x4, <Self as SimdVectors>::u64x4>
        + CastVector<<Self as SimdVectors>::f32x4> + CastVector<<Self as SimdVectors>::f64x4>
        + CastVector<<Self as SimdVectors>::i16x4> + CastVector<<Self as SimdVectors>::i64x4>
        + CastVector<<Self as SimdVectors>::u16x4> + CastVector<<Self as SimdVectors>::u32x4>
        + CastVector<<Self as SimdVectors>::u64x4> + CastVector<<Self as SimdVectors>::u8x4>;
    type u8x4: UnsignedIntegerVector<Lanes = U4, Element = u8, Unsigned = <Self as SimdVectors>::u8x4, Signed = <Self as SimdVectors>::i8x4>
        + CastVector<<Self as SimdVectors>::u32x4> + CastVector<<Self as SimdVectors>::u32x4> + ConcatVector<<Self as SimdVectors>::u8x2> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex4, <Self as SimdVectors>::u32x4, <Self as SimdVectors>::u64x4>
        + PackedF8Vector<<Self as SimdVectors>::f32x4>
        + Sad16Vector<<Self as SimdVectors>::u16x2> + Sad32Vector<Vector<u32>> + Sad64Vector<Vector<u64>>
        + CastVector<<Self as SimdVectors>::f32x4> + CastVector<<Self as SimdVectors>::f64x4>
        + CastVector<<Self as SimdVectors>::i16x4> + CastVector<<Self as SimdVectors>::i32x4>
        + CastVector<<Self as SimdVectors>::i64x4> + CastVector<<Self as SimdVectors>::i8x4>
        + CastVector<<Self as SimdVectors>::u16x4> + CastVector<<Self as SimdVectors>::u64x4>;

    type i8x8: SignedIntegerVector<Lanes = U8, Element = i8, Unsigned = <Self as SimdVectors>::u8x8, Signed = <Self as SimdVectors>::i8x8>
        + CastVector<<Self as SimdVectors>::i32x8> + CastVector<<Self as SimdVectors>::i32x8> + ConcatVector<<Self as SimdVectors>::i8x4> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex8, <Self as SimdVectors>::u32x8, <Self as SimdVectors>::u64x8>
        + CastVector<<Self as SimdVectors>::f32x8> + CastVector<<Self as SimdVectors>::f64x8>
        + CastVector<<Self as SimdVectors>::i16x8> + CastVector<<Self as SimdVectors>::i64x8>
        + CastVector<<Self as SimdVectors>::u16x8> + CastVector<<Self as SimdVectors>::u32x8>
        + CastVector<<Self as SimdVectors>::u64x8> + CastVector<<Self as SimdVectors>::u8x8>;
    type u8x8: UnsignedIntegerVector<Lanes = U8, Element = u8, Unsigned = <Self as SimdVectors>::u8x8, Signed = <Self as SimdVectors>::i8x8>
        + CastVector<<Self as SimdVectors>::u32x8> + CastVector<<Self as SimdVectors>::u32x8> + ConcatVector<<Self as SimdVectors>::u8x4> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex8, <Self as SimdVectors>::u32x8, <Self as SimdVectors>::u64x8>
        + PackedF8Vector<<Self as SimdVectors>::f32x8>
        + Sad16Vector<<Self as SimdVectors>::u16x4> + Sad32Vector<<Self as SimdVectors>::u32x2> + Sad64Vector<Vector<u64>>
        + CastVector<<Self as SimdVectors>::f32x8> + CastVector<<Self as SimdVectors>::f64x8>
        + CastVector<<Self as SimdVectors>::i16x8> + CastVector<<Self as SimdVectors>::i32x8>
        + CastVector<<Self as SimdVectors>::i64x8> + CastVector<<Self as SimdVectors>::i8x8>
        + CastVector<<Self as SimdVectors>::u16x8> + CastVector<<Self as SimdVectors>::u64x8>;

    type i16x2: SignedIntegerVector<Lanes = U2, Element = i16, Unsigned = <Self as SimdVectors>::u16x2, Signed = <Self as SimdVectors>::i16x2>
        + CastVector<<Self as SimdVectors>::i32x2> + CastVector<<Self as SimdVectors>::i32x2> + ConcatVector<Vector<i16>> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex2, <Self as SimdVectors>::u32x2, <Self as SimdVectors>::u64x2>
        + CastVector<<Self as SimdVectors>::f32x2> + CastVector<<Self as SimdVectors>::f64x2>
        + CastVector<<Self as SimdVectors>::i64x2> + CastVector<<Self as SimdVectors>::i8x2>
        + CastVector<<Self as SimdVectors>::u16x2> + CastVector<<Self as SimdVectors>::u32x2>
        + CastVector<<Self as SimdVectors>::u64x2> + CastVector<<Self as SimdVectors>::u8x2>;
    type u16x2: UnsignedIntegerVector<Lanes = U2, Element = u16, Unsigned = <Self as SimdVectors>::u16x2, Signed = <Self as SimdVectors>::i16x2>
        + Sad32Vector<Vector<u32>> + Sad64Vector<Vector<u64>>
        + CastVector<<Self as SimdVectors>::u32x2> + CastVector<<Self as SimdVectors>::u32x2> + ConcatVector<Vector<u16>> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex2, <Self as SimdVectors>::u32x2, <Self as SimdVectors>::u64x2>
        + CastVector<<Self as SimdVectors>::f32x2> + CastVector<<Self as SimdVectors>::f64x2>
        + CastVector<<Self as SimdVectors>::i16x2> + CastVector<<Self as SimdVectors>::i32x2>
        + CastVector<<Self as SimdVectors>::i64x2> + CastVector<<Self as SimdVectors>::i8x2>
        + CastVector<<Self as SimdVectors>::u64x2> + CastVector<<Self as SimdVectors>::u8x2>;

    type i16x4: SignedIntegerVector<Lanes = U4, Element = i16, Unsigned = <Self as SimdVectors>::u16x4, Signed = <Self as SimdVectors>::i16x4>
        + CastVector<<Self as SimdVectors>::i32x4> + CastVector<<Self as SimdVectors>::i32x4> + ConcatVector<<Self as SimdVectors>::i16x2> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex4, <Self as SimdVectors>::u32x4, <Self as SimdVectors>::u64x4>
        + CastVector<<Self as SimdVectors>::f32x4> + CastVector<<Self as SimdVectors>::f64x4>
        + CastVector<<Self as SimdVectors>::i64x4> + CastVector<<Self as SimdVectors>::i8x4>
        + CastVector<<Self as SimdVectors>::u16x4> + CastVector<<Self as SimdVectors>::u32x4>
        + CastVector<<Self as SimdVectors>::u64x4> + CastVector<<Self as SimdVectors>::u8x4>;
    type u16x4: UnsignedIntegerVector<Lanes = U4, Element = u16, Unsigned = <Self as SimdVectors>::u16x4, Signed = <Self as SimdVectors>::i16x4>
        + Sad32Vector<<Self as SimdVectors>::u32x2> + Sad64Vector<Vector<u64>>
        + CastVector<<Self as SimdVectors>::u32x4> + CastVector<<Self as SimdVectors>::u32x4> + ConcatVector<<Self as SimdVectors>::u16x2> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex4, <Self as SimdVectors>::u32x4, <Self as SimdVectors>::u64x4>
        + PackedF16Vector<<Self as SimdVectors>::f32x4>
        + CastVector<<Self as SimdVectors>::f32x4> + CastVector<<Self as SimdVectors>::f64x4>
        + CastVector<<Self as SimdVectors>::i16x4> + CastVector<<Self as SimdVectors>::i32x4>
        + CastVector<<Self as SimdVectors>::i64x4> + CastVector<<Self as SimdVectors>::i8x4>
        + CastVector<<Self as SimdVectors>::u64x4> + CastVector<<Self as SimdVectors>::u8x4>;

    type i16x8: SignedIntegerVector<Lanes = U8, Element = i16, Unsigned = <Self as SimdVectors>::u16x8, Signed = <Self as SimdVectors>::i16x8>
        + CastVector<<Self as SimdVectors>::i32x8> + CastVector<<Self as SimdVectors>::i32x8> + ConcatVector<<Self as SimdVectors>::i16x4> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex8, <Self as SimdVectors>::u32x8, <Self as SimdVectors>::u64x8>
        + CastVector<<Self as SimdVectors>::f32x8> + CastVector<<Self as SimdVectors>::f64x8>
        + CastVector<<Self as SimdVectors>::i64x8> + CastVector<<Self as SimdVectors>::i8x8>
        + CastVector<<Self as SimdVectors>::u16x8> + CastVector<<Self as SimdVectors>::u32x8>
        + CastVector<<Self as SimdVectors>::u64x8> + CastVector<<Self as SimdVectors>::u8x8>;
    type u16x8: UnsignedIntegerVector<Lanes = U8, Element = u16, Unsigned = <Self as SimdVectors>::u16x8, Signed = <Self as SimdVectors>::i16x8>
        + CastVector<<Self as SimdVectors>::u32x8> + CastVector<<Self as SimdVectors>::u32x8> + ConcatVector<<Self as SimdVectors>::u16x4> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex8, <Self as SimdVectors>::u32x8, <Self as SimdVectors>::u64x8>
        + PackedF16Vector<<Self as SimdVectors>::f32x8>
        + Sad32Vector<<Self as SimdVectors>::u32x4> + Sad64Vector<<Self as SimdVectors>::u64x2>
        + CastVector<<Self as SimdVectors>::f32x8> + CastVector<<Self as SimdVectors>::f64x8>
        + CastVector<<Self as SimdVectors>::i16x8> + CastVector<<Self as SimdVectors>::i32x8>
        + CastVector<<Self as SimdVectors>::i64x8> + CastVector<<Self as SimdVectors>::i8x8>
        + CastVector<<Self as SimdVectors>::u64x8> + CastVector<<Self as SimdVectors>::u8x8>;

    type i16x16: SignedIntegerVector<Lanes = U16, Element = i16, Unsigned = <Self as SimdVectors>::u16x16, Signed = <Self as SimdVectors>::i16x16>
        + CastVector<<Self as SimdVectors>::i32x16> + CastVector<<Self as SimdVectors>::i32x16> + ConcatVector<<Self as SimdVectors>::i16x8> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex16, <Self as SimdVectors>::u32x16, <Self as SimdVectors>::u64x16>
        + CastVector<<Self as SimdVectors>::f32x16> + CastVector<<Self as SimdVectors>::f64x16>
        + CastVector<<Self as SimdVectors>::i64x16> + CastVector<<Self as SimdVectors>::i8x16>
        + CastVector<<Self as SimdVectors>::u16x16> + CastVector<<Self as SimdVectors>::u32x16>
        + CastVector<<Self as SimdVectors>::u64x16> + CastVector<<Self as SimdVectors>::u8x16>;
    type u16x16: UnsignedIntegerVector<Lanes = U16, Element = u16, Unsigned = <Self as SimdVectors>::u16x16, Signed = <Self as SimdVectors>::i16x16>
        + Sad32Vector<<Self as SimdVectors>::u32x8> + Sad64Vector<<Self as SimdVectors>::u64x4>
        + CastVector<<Self as SimdVectors>::u32x16> + CastVector<<Self as SimdVectors>::u32x16> + ConcatVector<<Self as SimdVectors>::u16x8> + SwizzleVector
        + VectorIndexedBy<<Self as SimdVectors>::usizex16, <Self as SimdVectors>::u32x16, <Self as SimdVectors>::u64x16>
        + PackedF16Vector<<Self as SimdVectors>::f32x16>
        + CastVector<<Self as SimdVectors>::f32x16> + CastVector<<Self as SimdVectors>::f64x16>
        + CastVector<<Self as SimdVectors>::i16x16> + CastVector<<Self as SimdVectors>::i32x16>
        + CastVector<<Self as SimdVectors>::i64x16> + CastVector<<Self as SimdVectors>::i8x16>
        + CastVector<<Self as SimdVectors>::u64x16> + CastVector<<Self as SimdVectors>::u8x16>;}

/// Marker companion to [`SimdVectors`] that also exposes each vector's
/// underlying [`crate::register::Register`] type.
///
/// Bind on this when generic code needs to move a fixed-width vector to or
/// from its raw register storage.
pub trait SimdVectorsWithRegisters: Simd + NativeSimdVectorsWithRegisters + SimdVectors<
    // usizes
    usizex2: UnsignedIntegerVectorWithRegister<Register = <Self as Simd>::usizex2>,
    usizex4: UnsignedIntegerVectorWithRegister<Register = <Self as Simd>::usizex4>,
    usizex8: UnsignedIntegerVectorWithRegister<Register = <Self as Simd>::usizex8>,
    usizex16: UnsignedIntegerVectorWithRegister<Register = <Self as Simd>::usizex16>,

    // 32x2
    f32x2: FloatVectorWithRegister<Register = <Self as Simd>::f32x2, SignedBits = <Self as SimdVectors>::i32x2, Bits = <Self as SimdVectors>::u32x2>
        + FIV<<Self as SimdVectors>::i32x2, <Self as SimdVectors>::u32x2>,
    i32x2: SignedIntegerVectorWithRegister<Register = <Self as Simd>::i32x2>
        + FIV<<Self as SimdVectors>::f32x2, <Self as SimdVectors>::u32x2>,
    u32x2: UnsignedIntegerVectorWithRegister<Register = <Self as Simd>::u32x2>
        + FIV<<Self as SimdVectors>::f32x2, <Self as SimdVectors>::i32x2>,

    // 32x4
    f32x4: FloatVectorWithRegister<Register = <Self as Simd>::f32x4, SignedBits = <Self as SimdVectors>::i32x4, Bits = <Self as SimdVectors>::u32x4>
        + FIV<<Self as SimdVectors>::i32x4, <Self as SimdVectors>::u32x4>,
    i32x4: SignedIntegerVectorWithRegister<Register = <Self as Simd>::i32x4>
        + FIV<<Self as SimdVectors>::f32x4, <Self as SimdVectors>::u32x4>,
    u32x4: UnsignedIntegerVectorWithRegister<Register = <Self as Simd>::u32x4>
        + FIV<<Self as SimdVectors>::f32x4, <Self as SimdVectors>::i32x4>,

    // 32x8
    f32x8: FloatVectorWithRegister<Register = <Self as Simd>::f32x8, SignedBits = <Self as SimdVectors>::i32x8, Bits = <Self as SimdVectors>::u32x8>
        + FIV<<Self as SimdVectors>::i32x8, <Self as SimdVectors>::u32x8>,
    i32x8: SignedIntegerVectorWithRegister<Register = <Self as Simd>::i32x8>
        + FIV<<Self as SimdVectors>::f32x8, <Self as SimdVectors>::u32x8>,
    u32x8: UnsignedIntegerVectorWithRegister<Register = <Self as Simd>::u32x8>
        + FIV<<Self as SimdVectors>::f32x8, <Self as SimdVectors>::i32x8>,

    // 32x16
    f32x16: FloatVectorWithRegister<Register = <Self as Simd>::f32x16, SignedBits = <Self as SimdVectors>::i32x16, Bits = <Self as SimdVectors>::u32x16>
        + FIV<<Self as SimdVectors>::i32x16, <Self as SimdVectors>::u32x16>,
    i32x16: SignedIntegerVectorWithRegister<Register = <Self as Simd>::i32x16>
        + FIV<<Self as SimdVectors>::f32x16, <Self as SimdVectors>::u32x16>,
    u32x16: UnsignedIntegerVectorWithRegister<Register = <Self as Simd>::u32x16>
        + FIV<<Self as SimdVectors>::f32x16, <Self as SimdVectors>::i32x16>,

    // 64x2
    f64x2: FloatVectorWithRegister<Register = <Self as Simd>::f64x2, SignedBits = <Self as SimdVectors>::i64x2, Bits = <Self as SimdVectors>::u64x2>,
    i64x2: SignedIntegerVectorWithRegister<Register = <Self as Simd>::i64x2>
        + FIV<<Self as SimdVectors>::f64x2, <Self as SimdVectors>::u64x2>,
    u64x2: UnsignedIntegerVectorWithRegister<Register = <Self as Simd>::u64x2>
        + FIV<<Self as SimdVectors>::f64x2, <Self as SimdVectors>::i64x2>,

    // 64x4
    f64x4: FloatVectorWithRegister<Register = <Self as Simd>::f64x4, SignedBits = <Self as SimdVectors>::i64x4, Bits = <Self as SimdVectors>::u64x4>
        + FIV<<Self as SimdVectors>::i64x4, <Self as SimdVectors>::u64x4>,
    i64x4: SignedIntegerVectorWithRegister<Register = <Self as Simd>::i64x4>
        + FIV<<Self as SimdVectors>::f64x4, <Self as SimdVectors>::u64x4>,
    u64x4: UnsignedIntegerVectorWithRegister<Register = <Self as Simd>::u64x4>
        + FIV<<Self as SimdVectors>::f64x4, <Self as SimdVectors>::i64x4>,

    // 64x8
    f64x8: FloatVectorWithRegister<Register = <Self as Simd>::f64x8, SignedBits = <Self as SimdVectors>::i64x8, Bits = <Self as SimdVectors>::u64x8>
        + FIV<<Self as SimdVectors>::i64x8, <Self as SimdVectors>::u64x8>,
    i64x8: SignedIntegerVectorWithRegister<Register = <Self as Simd>::i64x8>
        + FIV<<Self as SimdVectors>::f64x8, <Self as SimdVectors>::u64x8>,
    u64x8: UnsignedIntegerVectorWithRegister<Register = <Self as Simd>::u64x8>
        + FIV<<Self as SimdVectors>::f64x8, <Self as SimdVectors>::i64x8>,

    // 64x16
    f64x16: FloatVectorWithRegister<Register = <Self as Simd>::f64x16, SignedBits = <Self as SimdVectors>::i64x16, Bits = <Self as SimdVectors>::u64x16>
        + FIV<<Self as SimdVectors>::i64x16, <Self as SimdVectors>::u64x16>,
    i64x16: SignedIntegerVectorWithRegister<Register = <Self as Simd>::i64x16>
        + FIV<<Self as SimdVectors>::f64x16, <Self as SimdVectors>::u64x16>,
    u64x16: UnsignedIntegerVectorWithRegister<Register = <Self as Simd>::u64x16>
        + FIV<<Self as SimdVectors>::f64x16, <Self as SimdVectors>::i64x16>,

    // 8-bit / 16-bit fixed widths (no float partner, so no FIV)
    i8x16:  SignedIntegerVectorWithRegister<Register = <Self as Simd>::i8x16>,
    u8x16:  UnsignedIntegerVectorWithRegister<Register = <Self as Simd>::u8x16>,
    i8x2:   SignedIntegerVectorWithRegister<Register = <Self as Simd>::i8x2>,
    u8x2:   UnsignedIntegerVectorWithRegister<Register = <Self as Simd>::u8x2>,
    i8x4:   SignedIntegerVectorWithRegister<Register = <Self as Simd>::i8x4>,
    u8x4:   UnsignedIntegerVectorWithRegister<Register = <Self as Simd>::u8x4>,
    i8x8:   SignedIntegerVectorWithRegister<Register = <Self as Simd>::i8x8>,
    u8x8:   UnsignedIntegerVectorWithRegister<Register = <Self as Simd>::u8x8>,
    i16x2:  SignedIntegerVectorWithRegister<Register = <Self as Simd>::i16x2>,
    u16x2:  UnsignedIntegerVectorWithRegister<Register = <Self as Simd>::u16x2>,
    i16x4:  SignedIntegerVectorWithRegister<Register = <Self as Simd>::i16x4>,
    u16x4:  UnsignedIntegerVectorWithRegister<Register = <Self as Simd>::u16x4>,
    i16x8:  SignedIntegerVectorWithRegister<Register = <Self as Simd>::i16x8>,
    u16x8:  UnsignedIntegerVectorWithRegister<Register = <Self as Simd>::u16x8>,
    i16x16: SignedIntegerVectorWithRegister<Register = <Self as Simd>::i16x16>,
    u16x16: UnsignedIntegerVectorWithRegister<Register = <Self as Simd>::u16x16>,
>{}

/// Vector-level mirror of [`Simd3A`]: names the alpha-padded 3-lane vector
/// types (`f32x3A`, `i32x3A`, etc.) as [`Vector`] wrappers.
///
/// Auto-implemented for every [`Simd3A`] backend. The 3-lane vectors share
/// storage with the corresponding 4-lane registers; the 4th lane carries
/// no semantic value but keeps alignment intact, which makes shuffles and
/// other lane-aware ops cheap. For the truly-3-lane variant (e.g. SPIR-V
/// `vec3`), use [`Simd3Vectors`].
pub trait Simd3AVectors:
    SimdVectors<
        usizex4: ExtendVector<Self::usizex3A>,
        f32x4: ExtendVector<Self::f32x3A>,
        i32x4: ExtendVector<Self::i32x3A>,
        u32x4: ExtendVector<Self::u32x3A>,
        f64x4: ExtendVector<Self::f64x3A>,
        i64x4: ExtendVector<Self::i64x3A>,
        u64x4: ExtendVector<Self::u64x3A>,
    >
{
    type usizex3A: UnsignedIntegerVector<Lanes = U3, Element = crate::element::USize>
        + SwizzleVector
        + VectorIndexedBy<
            <Self as Simd3AVectors>::usizex3A,
            <Self as Simd3AVectors>::u32x3A,
            <Self as Simd3AVectors>::u64x3A,
        >;

    type f32x3A: FloatVector<Lanes = U3, Element = f32>
        + LinAlg3Vector
        + CastVector<<Self as Simd3AVectors>::f64x3A>
        + SwizzleVector
        + VectorIndexedBy<
            <Self as Simd3AVectors>::usizex3A,
            <Self as Simd3AVectors>::u32x3A,
            <Self as Simd3AVectors>::u64x3A,
        > + CastVector<<Self as Simd3AVectors>::i32x3A>
        + CastVector<<Self as Simd3AVectors>::i64x3A>
        + CastVector<<Self as Simd3AVectors>::u32x3A>
        + CastVector<<Self as Simd3AVectors>::u64x3A>;
    type i32x3A: SignedIntegerVector<Lanes = U3, Element = i32>
        + CastVector<<Self as Simd3AVectors>::i64x3A>
        + SwizzleVector
        + VectorIndexedBy<
            <Self as Simd3AVectors>::usizex3A,
            <Self as Simd3AVectors>::u32x3A,
            <Self as Simd3AVectors>::u64x3A,
        > + CastVector<<Self as Simd3AVectors>::f32x3A>
        + CastVector<<Self as Simd3AVectors>::f64x3A>
        + CastVector<<Self as Simd3AVectors>::u32x3A>
        + CastVector<<Self as Simd3AVectors>::u64x3A>;
    type u32x3A: UnsignedIntegerVector<Lanes = U3, Element = u32>
        + CastVector<<Self as Simd3AVectors>::u64x3A>
        + SwizzleVector
        + VectorIndexedBy<
            <Self as Simd3AVectors>::usizex3A,
            <Self as Simd3AVectors>::u32x3A,
            <Self as Simd3AVectors>::u64x3A,
        > + CastVector<<Self as Simd3AVectors>::f32x3A>
        + CastVector<<Self as Simd3AVectors>::f64x3A>
        + CastVector<<Self as Simd3AVectors>::i32x3A>
        + CastVector<<Self as Simd3AVectors>::i64x3A>;

    type f64x3A: FloatVector<Lanes = U3, Element = f64>
        + LinAlg3Vector
        + CastVector<<Self as Simd3AVectors>::f32x3A>
        + SwizzleVector
        + VectorIndexedBy<
            <Self as Simd3AVectors>::usizex3A,
            <Self as Simd3AVectors>::u32x3A,
            <Self as Simd3AVectors>::u64x3A,
        > + CastVector<<Self as Simd3AVectors>::i32x3A>
        + CastVector<<Self as Simd3AVectors>::i64x3A>
        + CastVector<<Self as Simd3AVectors>::u32x3A>
        + CastVector<<Self as Simd3AVectors>::u64x3A>;
    type i64x3A: SignedIntegerVector<Lanes = U3, Element = i64>
        + CastVector<<Self as Simd3AVectors>::i32x3A>
        + SwizzleVector
        + VectorIndexedBy<
            <Self as Simd3AVectors>::usizex3A,
            <Self as Simd3AVectors>::u32x3A,
            <Self as Simd3AVectors>::u64x3A,
        > + CastVector<<Self as Simd3AVectors>::f32x3A>
        + CastVector<<Self as Simd3AVectors>::f64x3A>
        + CastVector<<Self as Simd3AVectors>::u32x3A>
        + CastVector<<Self as Simd3AVectors>::u64x3A>;
    type u64x3A: UnsignedIntegerVector<Lanes = U3, Element = u64>
        + CastVector<<Self as Simd3AVectors>::u32x3A>
        + SwizzleVector
        + VectorIndexedBy<
            <Self as Simd3AVectors>::usizex3A,
            <Self as Simd3AVectors>::u32x3A,
            <Self as Simd3AVectors>::u64x3A,
        > + CastVector<<Self as Simd3AVectors>::f32x3A>
        + CastVector<<Self as Simd3AVectors>::f64x3A>
        + CastVector<<Self as Simd3AVectors>::i32x3A>
        + CastVector<<Self as Simd3AVectors>::i64x3A>;
}

/// Marker companion to [`Simd3AVectors`] that also exposes each vector's
/// underlying [`crate::register::Register`] type.
pub trait Simd3AVectorsWithRegisters: SimdVectorsWithRegisters + Simd3A + Simd3AVectors<
    // usizex3A
    usizex3A: UnsignedIntegerVectorWithRegister<Register = <Self as Simd3A>::usizex3A>,

    // 32x3A
    f32x3A: FloatVectorWithRegister<Register = <Self as Simd3A>::f32x3A, SignedBits = <Self as Simd3AVectors>::i32x3A, Bits = <Self as Simd3AVectors>::u32x3A>
        + FIV<<Self as Simd3AVectors>::i32x3A, <Self as Simd3AVectors>::u32x3A>,
    i32x3A: SignedIntegerVectorWithRegister<Register = <Self as Simd3A>::i32x3A>
        + FIV<<Self as Simd3AVectors>::f32x3A, <Self as Simd3AVectors>::u32x3A>,
    u32x3A: UnsignedIntegerVectorWithRegister<Register = <Self as Simd3A>::u32x3A>
        + FIV<<Self as Simd3AVectors>::f32x3A, <Self as Simd3AVectors>::i32x3A>,

    // 64x3A
    f64x3A: FloatVectorWithRegister<Register = <Self as Simd3A>::f64x3A, SignedBits = <Self as Simd3AVectors>::i64x3A, Bits = <Self as Simd3AVectors>::u64x3A>
        + FIV<<Self as Simd3AVectors>::i64x3A, <Self as Simd3AVectors>::u64x3A>,
    i64x3A: SignedIntegerVectorWithRegister<Register = <Self as Simd3A>::i64x3A>
        + FIV<<Self as Simd3AVectors>::f64x3A, <Self as Simd3AVectors>::u64x3A>,
    u64x3A: UnsignedIntegerVectorWithRegister<Register = <Self as Simd3A>::u64x3A>
        + FIV<<Self as Simd3AVectors>::f64x3A, <Self as Simd3AVectors>::i64x3A>,
>{}

/// Vector-level mirror of [`Simd3`]: names the truly-3-lane vector types
/// (`f32x3`, `i32x3`, etc.) as [`Vector`] wrappers.
///
/// Only implemented on backends that have a real 3-lane representation
/// (GPU/shader targets such as SPIR-V with `vec3`). On CPU backends, this
/// is normally identical to [`Simd3AVectors`] -- the implementation just
/// re-uses the alpha-padded form. Use it when you care about the
/// "no padding lane" semantics, e.g. when emitting GPU shader code.
pub trait Simd3Vectors:
    Simd3AVectors<
        usizex4: ExtendVector<Self::usizex3>,
        f32x4: ExtendVector<Self::f32x3>,
        i32x4: ExtendVector<Self::i32x3>,
        u32x4: ExtendVector<Self::u32x3>,
        f64x4: ExtendVector<Self::f64x3>,
        i64x4: ExtendVector<Self::i64x3>,
        u64x4: ExtendVector<Self::u64x3>,
    >
{
    type usizex3: UnsignedIntegerVector<Lanes = U3, Element = crate::element::USize>
        + SwizzleVector
        + VectorIndexedBy<<Self as Simd3Vectors>::usizex3, <Self as Simd3Vectors>::u32x3, <Self as Simd3Vectors>::u64x3>;

    type f32x3: FloatVector<Lanes = U3, Element = f32>
        + LinAlg3Vector
        + CastVector<<Self as Simd3Vectors>::f64x3>
        + SwizzleVector
        + VectorIndexedBy<<Self as Simd3Vectors>::usizex3, <Self as Simd3Vectors>::u32x3, <Self as Simd3Vectors>::u64x3>
        + CastVector<<Self as Simd3Vectors>::i32x3>
        + CastVector<<Self as Simd3Vectors>::i64x3>
        + CastVector<<Self as Simd3Vectors>::u32x3>
        + CastVector<<Self as Simd3Vectors>::u64x3>;
    type i32x3: SignedIntegerVector<Lanes = U3, Element = i32>
        + CastVector<<Self as Simd3Vectors>::i64x3>
        + SwizzleVector
        + VectorIndexedBy<<Self as Simd3Vectors>::usizex3, <Self as Simd3Vectors>::u32x3, <Self as Simd3Vectors>::u64x3>
        + CastVector<<Self as Simd3Vectors>::f32x3>
        + CastVector<<Self as Simd3Vectors>::f64x3>
        + CastVector<<Self as Simd3Vectors>::u32x3>
        + CastVector<<Self as Simd3Vectors>::u64x3>;
    type u32x3: UnsignedIntegerVector<Lanes = U3, Element = u32>
        + CastVector<<Self as Simd3Vectors>::u64x3>
        + SwizzleVector
        + VectorIndexedBy<<Self as Simd3Vectors>::usizex3, <Self as Simd3Vectors>::u32x3, <Self as Simd3Vectors>::u64x3>
        + CastVector<<Self as Simd3Vectors>::f32x3>
        + CastVector<<Self as Simd3Vectors>::f64x3>
        + CastVector<<Self as Simd3Vectors>::i32x3>
        + CastVector<<Self as Simd3Vectors>::i64x3>;

    type f64x3: FloatVector<Lanes = U3, Element = f64>
        + LinAlg3Vector
        + CastVector<<Self as Simd3Vectors>::f32x3>
        + SwizzleVector
        + VectorIndexedBy<<Self as Simd3Vectors>::usizex3, <Self as Simd3Vectors>::u32x3, <Self as Simd3Vectors>::u64x3>
        + CastVector<<Self as Simd3Vectors>::i32x3>
        + CastVector<<Self as Simd3Vectors>::i64x3>
        + CastVector<<Self as Simd3Vectors>::u32x3>
        + CastVector<<Self as Simd3Vectors>::u64x3>;
    type i64x3: SignedIntegerVector<Lanes = U3, Element = i64>
        + CastVector<<Self as Simd3Vectors>::i32x3>
        + SwizzleVector
        + VectorIndexedBy<<Self as Simd3Vectors>::usizex3, <Self as Simd3Vectors>::u32x3, <Self as Simd3Vectors>::u64x3>
        + CastVector<<Self as Simd3Vectors>::f32x3>
        + CastVector<<Self as Simd3Vectors>::f64x3>
        + CastVector<<Self as Simd3Vectors>::u32x3>
        + CastVector<<Self as Simd3Vectors>::u64x3>;
    type u64x3: UnsignedIntegerVector<Lanes = U3, Element = u64>
        + CastVector<<Self as Simd3Vectors>::u32x3>
        + SwizzleVector
        + VectorIndexedBy<<Self as Simd3Vectors>::usizex3, <Self as Simd3Vectors>::u32x3, <Self as Simd3Vectors>::u64x3>
        + CastVector<<Self as Simd3Vectors>::f32x3>
        + CastVector<<Self as Simd3Vectors>::f64x3>
        + CastVector<<Self as Simd3Vectors>::i32x3>
        + CastVector<<Self as Simd3Vectors>::i64x3>;
}

/// Marker companion to [`Simd3Vectors`] that also exposes each vector's
/// underlying [`crate::register::Register`] type.
pub trait Simd3VectorsWithRegisters: SimdVectorsWithRegisters + Simd3 + Simd3Vectors<
    // usizex3
    usizex3: UnsignedIntegerVectorWithRegister<Register = <Self as Simd3>::usizex3>,

    // 32x3
    f32x3: FloatVectorWithRegister<Register = <Self as Simd3>::f32x3, SignedBits = <Self as Simd3Vectors>::i32x3, Bits = <Self as Simd3Vectors>::u32x3>
        + FIV<<Self as Simd3Vectors>::i32x3, <Self as Simd3Vectors>::u32x3>,
    i32x3: SignedIntegerVectorWithRegister<Register = <Self as Simd3>::i32x3>
        + FIV<<Self as Simd3Vectors>::f32x3, <Self as Simd3Vectors>::u32x3>,
    u32x3: UnsignedIntegerVectorWithRegister<Register = <Self as Simd3>::u32x3>
        + FIV<<Self as Simd3Vectors>::f32x3, <Self as Simd3Vectors>::i32x3>,

    // 64x3
    f64x3: FloatVectorWithRegister<Register = <Self as Simd3>::f64x3, SignedBits = <Self as Simd3Vectors>::i64x3, Bits = <Self as Simd3Vectors>::u64x3>
        + FIV<<Self as Simd3Vectors>::i64x3, <Self as Simd3Vectors>::u64x3>,
    i64x3: SignedIntegerVectorWithRegister<Register = <Self as Simd3>::i64x3>
        + FIV<<Self as Simd3Vectors>::f64x3, <Self as Simd3Vectors>::u64x3>,
    u64x3: UnsignedIntegerVectorWithRegister<Register = <Self as Simd3>::u64x3>
        + FIV<<Self as Simd3Vectors>::f64x3, <Self as Simd3Vectors>::i64x3>,
>{}

impl<S: Simd> SimdVectors for S {
    type usizex2 = Vector<<Self as Simd>::usizex2>;
    type usizex4 = Vector<<Self as Simd>::usizex4>;
    type usizex8 = Vector<<Self as Simd>::usizex8>;
    type usizex16 = Vector<<Self as Simd>::usizex16>;

    type f32x2 = Vector<<Self as Simd>::f32x2>;
    type i32x2 = Vector<<Self as Simd>::i32x2>;
    type u32x2 = Vector<<Self as Simd>::u32x2>;

    type f32x4 = Vector<<Self as Simd>::f32x4>;
    type i32x4 = Vector<<Self as Simd>::i32x4>;
    type u32x4 = Vector<<Self as Simd>::u32x4>;

    type f32x8 = Vector<<Self as Simd>::f32x8>;
    type i32x8 = Vector<<Self as Simd>::i32x8>;
    type u32x8 = Vector<<Self as Simd>::u32x8>;

    type f64x2 = Vector<<Self as Simd>::f64x2>;
    type i64x2 = Vector<<Self as Simd>::i64x2>;
    type u64x2 = Vector<<Self as Simd>::u64x2>;

    type f64x4 = Vector<<Self as Simd>::f64x4>;
    type i64x4 = Vector<<Self as Simd>::i64x4>;
    type u64x4 = Vector<<Self as Simd>::u64x4>;

    type f64x8 = Vector<<Self as Simd>::f64x8>;
    type i64x8 = Vector<<Self as Simd>::i64x8>;
    type u64x8 = Vector<<Self as Simd>::u64x8>;

    type f32x16 = Vector<<Self as Simd>::f32x16>;
    type i32x16 = Vector<<Self as Simd>::i32x16>;
    type u32x16 = Vector<<Self as Simd>::u32x16>;

    type f64x16 = Vector<<Self as Simd>::f64x16>;
    type i64x16 = Vector<<Self as Simd>::i64x16>;
    type u64x16 = Vector<<Self as Simd>::u64x16>;

    type i8x16 = Vector<<Self as Simd>::i8x16>;
    type u8x16 = Vector<<Self as Simd>::u8x16>;
    type i8x2 = Vector<<Self as Simd>::i8x2>;
    type u8x2 = Vector<<Self as Simd>::u8x2>;
    type i8x4 = Vector<<Self as Simd>::i8x4>;
    type u8x4 = Vector<<Self as Simd>::u8x4>;
    type i8x8 = Vector<<Self as Simd>::i8x8>;
    type u8x8 = Vector<<Self as Simd>::u8x8>;

    type i16x2 = Vector<<Self as Simd>::i16x2>;
    type u16x2 = Vector<<Self as Simd>::u16x2>;
    type i16x4 = Vector<<Self as Simd>::i16x4>;
    type u16x4 = Vector<<Self as Simd>::u16x4>;
    type i16x8 = Vector<<Self as Simd>::i16x8>;
    type u16x8 = Vector<<Self as Simd>::u16x8>;
    type i16x16 = Vector<<Self as Simd>::i16x16>;
    type u16x16 = Vector<<Self as Simd>::u16x16>;
}

impl<S: Simd> SimdVectorsWithRegisters for S {}

impl<S: Simd3A> Simd3AVectors for S {
    type usizex3A = Vector<<Self as Simd3A>::usizex3A>;

    type f32x3A = Vector<<Self as Simd3A>::f32x3A>;
    type i32x3A = Vector<<Self as Simd3A>::i32x3A>;
    type u32x3A = Vector<<Self as Simd3A>::u32x3A>;

    type f64x3A = Vector<<Self as Simd3A>::f64x3A>;
    type i64x3A = Vector<<Self as Simd3A>::i64x3A>;
    type u64x3A = Vector<<Self as Simd3A>::u64x3A>;
}

impl<S: Simd3A> Simd3AVectorsWithRegisters for S {}

impl<S: Simd3> Simd3Vectors for S {
    type usizex3 = Vector<<Self as Simd3>::usizex3>;

    type f32x3 = Vector<<Self as Simd3>::f32x3>;
    type i32x3 = Vector<<Self as Simd3>::i32x3>;
    type u32x3 = Vector<<Self as Simd3>::u32x3>;

    type f64x3 = Vector<<Self as Simd3>::f64x3>;
    type i64x3 = Vector<<Self as Simd3>::i64x3>;
    type u64x3 = Vector<<Self as Simd3>::u64x3>;
}

impl<S: Simd3> Simd3VectorsWithRegisters for S {}
