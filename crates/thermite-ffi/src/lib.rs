#![cfg_attr(docsrs, feature(doc_cfg))]

//! Thermite SIMD-Accelerated batch operations for C-ABI applications.
//!
//! This crate provides a set of functions that take arbitrary length arrays and performs
//! operations on them, often using SIMD-accelerated algorithms. The algorithms are designed
//! to support the same pointer for inputs and outputs, so in-place mapping is implicit,
//! but it does NOT support writing to different portions of the same array
//! (i.e., giving an offset pointer to the output).
//!
//! To get the most performance, the library must be initialized with [`thermite_init`] or
//! functions must be accessed via a custom-allocated [`VTable`]
//! (exported to the C header as `Thermite`). Upon initialization, the vtable is populated
//! with methods from the appropriate backend with the desired precision policy.
//!
//! Either the global or individual vtables can be initialized with a given precision policy,
//! which affects the speed and accuracy of the results. See [`ThermitePrecisionPolicy`]
//! for more on that.
//!
//! This library is primarily intended to be dynamically linked, so code size matters and
//! not all functions are inlined nor all loops unrolled.
//!
//! The `disable_dispatch` feature is nonetheless **on by default here**, which is the
//! opposite of the advice for a normal Thermite consumer. Dispatch exists to choose an ISA
//! at the point of the call, and this crate has already chosen one: [`thermite_init`]
//! populates the vtable from a single backend, so every function in it is monomorphized for
//! exactly one ISA before anything calls it. The dispatch inside was re-deciding that, and
//! charging a function call plus a stack round-trip of the vectors to do it.
//!
//! Measured on a 5950X, `sin_cosf_vv` over 32K f32 (x86-v3, HighPerformance), turning it on
//! is **39% faster and slightly smaller** on both x86 and AArch64:
//!
//! ```text
//!                      ns/elem      .dll (x86)     .so (aarch64)
//!   dispatch            0.659        2,419,200        646,848
//!   disable_dispatch    0.398        2,404,352        643,120
//! ```
//!
//! The usual warning that inlining the math "bloats the binary considerably" is true when a
//! caller dispatches over several ISAs, and false here: there is only ever one ISA per
//! vtable, so there is nothing for the inlined math to be duplicated across. `llvm-mca` puts
//! the removed overhead at roughly 3.8 of the ~23 cycles per 8-lane block, before counting
//! the call itself.
//!
//! Build with `--no-default-features --features high_performance,high_precision` to get the
//! dispatched form back.
//!
//! Furthermore, using a tool like `mpress` to compress the binary may be desired, but that's
//! more of a personal preference in the end.

// cargo expand -p thermite-ffi --all-features > ffi.rs && cbindgen -q -l c --crate thermite-ffi ffi.rs > ffi.h && echo "Done"
// cargo build --profile release-ffi -p thermite-ffi && Copy-Item ../../target/release-ffi/thermite_ffi.dll && mpress -b -s thermite_ffi.dll && echo "Done"
// cl.exe test.c /O2 /GL /link "../../target/release-ffi/thermite_ffi.dll.lib" ntdll.lib /LTCG /OPT:REF /OPT:ICF

#![no_std]

mod map;

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

/// Apple's linker refuses to produce a dylib that does not link libSystem:
///
/// ```text
/// ld: dynamic executables or dylibs must link with libSystem.dylib
/// ```
///
/// Normally the platform default supplies it, but this crate graph is `no_std`,
/// so rustc passes `-nodefaultlibs` and nothing does. Asking for it here rather
/// than through `RUSTFLAGS` keeps a plain `cargo build` working on macOS.
///
/// Unconditional across Apple targets, not just aarch64: the requirement is the
/// linker's, and applies to an Intel Mac build just as much.
#[cfg(target_vendor = "apple")]
#[link(name = "System")]
unsafe extern "C" {}

// `panic = "abort"` still leaves a reference to the unwinding personality routine
// on ELF targets, and this library links against nothing (no DT_NEEDED entries at
// all), so there is no libc to resolve it from. Defining it keeps the shared
// object fully self-contained. It is never called: nothing here unwinds.
#[cfg(not(target_env = "msvc"))]
#[unsafe(no_mangle)]
extern "C" fn rust_eh_personality() {}

use core::ffi::c_char;

use thermite::{
    element::FloatElement,
    math::{CoreMathWithPolicy, RealMathWithPolicy, SpatialMathWithPolicy, TranscendentalMathWithPolicy},
    prelude::{FloatVector, Policy},
    simd::NativeIsa,
};
use thermite_special::{
    RealPrimalMathWithPolicy, RealSpecialMathWithPolicy, SpecialMathWithPolicy,
    specialized::SpecializedSpecialMath,
    elliptic::{
        EllipticConsts,
        CarlsonRc, CarlsonRd, CarlsonRf, CarlsonRg, CarlsonRj, EllintD, EllintDInc, EllintE, EllintEInc, EllintF,
        EllintK, EllintPi, EllintPiInc,
    },
};

// The method deliberately isn't named `into_array`: on a generic `V` receiver,
// method resolution prefers where-clause candidates, so a name shared with
// `GenericVector::into_array` resolves to that trait method instead of this shim.
trait IntoOutputs<T, const N: usize> {
    fn into_outputs(self) -> [T; N];
}

#[rustfmt::skip]
const _: () = {
    impl<T> IntoOutputs<T, 1> for T { #[inline(always)] fn into_outputs(self) -> [T; 1] { [self] } }
    impl<T> IntoOutputs<T, 2> for (T, T) { #[inline(always)] fn into_outputs(self) -> [T; 2] { [self.0, self.1] } }
};

/// Forms of RealMath methods with explicit generic parameters,
/// such as order, dimensions, edges, etc.
#[rustfmt::skip]
pub trait RealMathWithPolicyFfi: RealMathWithPolicy + RealPrimalMathWithPolicy {
    #[inline(always)] fn add_v_p<P: Policy>(self, other: Self) -> Self { self + other }
    #[inline(always)] fn sub_v_p<P: Policy>(self, other: Self) -> Self { self - other }
    #[inline(always)] fn mul_v_p<P: Policy>(self, other: Self) -> Self { self * other }
    #[inline(always)] fn div_v_p<P: Policy>(self, other: Self) -> Self { self / other }
    #[inline(always)] fn rem_v_p<P: Policy>(self, other: Self) -> Self { self % other }
    #[inline(always)] fn round_v_p<P: Policy>(self) -> Self { self.round() }
    #[inline(always)] fn floor_v_p<P: Policy>(self) -> Self { self.floor() }
    #[inline(always)] fn ceil_v_p<P: Policy>(self) -> Self { self.ceil() }
    #[inline(always)] fn trunc_v_p<P: Policy>(self) -> Self { self.trunc() }
    #[inline(always)] fn fract_v_p<P: Policy>(self) -> Self { self.fract() }
    #[inline(always)] fn next_up_v_p<P: Policy>(self) -> Self { self.next_up() }
    #[inline(always)] fn next_down_v_p<P: Policy>(self) -> Self { self.next_down() }
    #[inline(always)] fn min_v_p<P: Policy>(self, other: Self) -> Self { self.min(other) }
    #[inline(always)] fn max_v_p<P: Policy>(self, other: Self) -> Self { self.max(other) }
    #[inline(always)] fn clamp_vs_p<P: Policy>(self, min: Self::Element, max: Self::Element) -> Self {
        self.clamp(Self::splat(min), Self::splat(max))
    }
    #[inline(always)] fn abs_v_p<P: Policy>(self) -> Self { self.abs() }
    #[inline(always)] fn signum_v_p<P: Policy>(self) -> Self { self.signum() }

    #[inline(always)] fn mul_add_v_p<P: Policy>(self, a: Self, b: Self) -> Self { self.mul_add(a, b) }
    #[inline(always)] fn mul_sub_v_p<P: Policy>(self, a: Self, b: Self) -> Self { self.mul_sub(a, b) }
    #[inline(always)] fn nmul_add_v_p<P: Policy>(self, a: Self, b: Self) -> Self { self.nmul_add(a, b) }
    #[inline(always)] fn nmul_sub_v_p<P: Policy>(self, a: Self, b: Self) -> Self { self.nmul_sub(a, b) }

    /// 3rd-order smoothstep
    #[inline(always)]
    fn smoothstep_p<P: Policy>(self) -> Self {
        RealMathWithPolicy::smoothstep_p::<P, 2>(self, None)
    }

    #[inline(always)]
    fn inverse_smoothstep_p<P: Policy>(self) -> Self {
        RealMathWithPolicy::inverse_smoothstep_p::<P, 2>(self, None)
    }

    /// 5th-order smoothstep
    #[inline(always)]
    fn smootherstep_p<P: Policy>(self) -> Self {
        RealMathWithPolicy::smoothstep_p::<P, 3>(self, None)
    }

    #[inline(always)]
    fn inverse_smootherstep_p<P: Policy>(self) -> Self {
        RealMathWithPolicy::inverse_smoothstep_p::<P, 3>(self, None)
    }

    #[inline(always)]
    fn lerp_vs_p<P: Policy>(self, a: Self::Element, b: Self::Element) -> Self {
        RealMathWithPolicy::lerp_p::<P>(self, Self::splat(a), Self::splat(b))
    }

    #[inline(always)]
    fn powi_vs_p<P: Policy>(self, exp: i32) -> Self {
        CoreMathWithPolicy::powi_p::<P>(self, exp)
    }

    #[inline(always)]
    fn step_vs_p<P: Policy>(self, edge: Self::Element) -> Self {
        RealMathWithPolicy::step_p::<P>(self, Self::splat(edge))
    }

    #[inline(always)]
    fn smooth_interpolator_vs_p<P: Policy>(self, k: Self::Element) -> Self {
        RealMathWithPolicy::smooth_interpolator_p::<P>(self, None, Self::splat(k))
    }

    #[inline(always)]
    fn smooth_interpolator_inverse_vs_p<P: Policy>(self, k: Self::Element) -> Self {
        RealMathWithPolicy::smooth_interpolator_inverse_p::<P>(self, None, Self::splat(k))
    }

    #[inline(always)]
    fn gaussian_vs_p<P: Policy>(self, a: Self::Element, c: Self::Element) -> Self {
        SpecialMathWithPolicy::gaussian_p::<P>(self, Self::splat(a), Self::splat(c))
    }

    // --- Activations -------------------------------------------------------
    //
    // Each takes its shape parameter as a full vector in Rust. A C caller almost
    // always wants one parameter for the whole array, so these splat it, the same
    // way `gaussian_vs_p` above does.

    #[inline(always)]
    fn gelu_vs_p<P: Policy>(self, alpha: Self::Element) -> Self {
        RealSpecialMathWithPolicy::gelu_p::<P>(self, Self::splat(alpha))
    }

    #[inline(always)]
    fn swish_vs_p<P: Policy>(self, beta: Self::Element) -> Self {
        RealSpecialMathWithPolicy::swish_p::<P>(self, Self::splat(beta))
    }

    /// `softplus` wants both `k` and its reciprocal, since it would otherwise
    /// recompute the division per call. The C surface takes only `k` and derives
    /// the reciprocal once, which is what the extra argument was avoiding anyway.
    #[inline(always)]
    fn softplus_vs_p<P: Policy>(self, k: Self::Element) -> Self {
        let k = Self::splat(k);
        SpecialMathWithPolicy::softplus_p::<P>(self, k, CoreMathWithPolicy::reciprocal_p::<P>(k))
    }

    /// `x / (1 + |x|)`, the softsign function.
    #[inline(always)]
    fn algebraic_sigmoid_1_p<P: Policy>(self) -> Self {
        RealSpecialMathWithPolicy::algebraic_sigmoid_p::<P, 1>(self)
    }

    /// `x / sqrt(1 + x^2)`.
    #[inline(always)]
    fn algebraic_sigmoid_2_p<P: Policy>(self) -> Self {
        RealSpecialMathWithPolicy::algebraic_sigmoid_p::<P, 2>(self)
    }

    // --- Activations, value and derivative together ------------------------
    //
    // One pass returns both, which is what a training loop wants.

    #[inline(always)]
    fn gelu_d_vs_p<P: Policy>(self, alpha: Self::Element) -> (Self, Self) {
        RealPrimalMathWithPolicy::gelu_d_p::<P>(self, Self::splat(alpha))
    }

    #[inline(always)]
    fn swish_d_vs_p<P: Policy>(self, beta: Self::Element) -> (Self, Self) {
        RealPrimalMathWithPolicy::swish_d_p::<P>(self, Self::splat(beta))
    }

    #[inline(always)]
    fn softplus_d_vs_p<P: Policy>(self, k: Self::Element) -> (Self, Self) {
        let k = Self::splat(k);
        RealPrimalMathWithPolicy::softplus_d_p::<P>(self, k, CoreMathWithPolicy::reciprocal_p::<P>(k))
    }

    #[inline(always)]
    fn algebraic_sigmoid_d_1_p<P: Policy>(self) -> (Self, Self) {
        RealPrimalMathWithPolicy::algebraic_sigmoid_d_p::<P, 1>(self)
    }

    #[inline(always)]
    fn algebraic_sigmoid_d_2_p<P: Policy>(self) -> (Self, Self) {
        RealPrimalMathWithPolicy::algebraic_sigmoid_d_p::<P, 2>(self)
    }

    // --- Fixed instantiations of const-generic orders ----------------------

    /// The exponential integral `E_1(x)`, the order that actually gets called.
    #[inline(always)]
    fn expint_1_p<P: Policy>(self) -> Self {
        SpecialMathWithPolicy::expint_p::<P, 1>(self)
    }

    /// Euclidean length of a 3-vector, without intermediate overflow.
    #[inline(always)]
    fn hypot_3_p<P: Policy>(self, y: Self, z: Self) -> Self {
        SpatialMathWithPolicy::hypot_n_p::<P, 3>([self, y, z])
    }

    /// Euclidean length of a 4-vector, without intermediate overflow.
    #[inline(always)]
    fn hypot_4_p<P: Policy>(self, y: Self, z: Self, w: Self) -> Self {
        SpatialMathWithPolicy::hypot_n_p::<P, 4>([self, y, z, w])
    }

    /// Remap from `[in_min, in_max]` onto `[out_min, out_max]`.
    #[inline(always)]
    fn rescale_vs_p<P: Policy>(
        self,
        in_min: Self::Element,
        in_max: Self::Element,
        out_min: Self::Element,
        out_max: Self::Element,
    ) -> Self {
        RealMathWithPolicy::rescale_p::<P>(
            self,
            Self::splat(in_min),
            Self::splat(in_max),
            Self::splat(out_min),
            Self::splat(out_max),
        )
    }

}

impl<T> RealMathWithPolicyFfi for T where T: RealMathWithPolicy + RealPrimalMathWithPolicy {}

/// The elliptic integrals, one entry point per form.
///
/// The Rust API selects the form with a request struct, so the wrong argument
/// shape is a compile error. C has no such mechanism, so each form is spelled out
/// with exactly its own arguments.
///
/// The bounds live on the blanket impl below rather than in this trait's
/// supertrait list. `EllipticKind` needs `Self: SpecializedSpecialMath<E>` with
/// `E: EllipticConsts`, and naming `<Self as GenericVector>::Element` in a
/// supertrait position sends the resolver into a cycle. Declaring the methods
/// here and satisfying them in a bounded impl keeps `EllipticMathFfi<Element = _>`
/// nameable, which is what the vtable macro writes.
pub trait EllipticMathFfi: SpecialMathWithPolicy {
    /// Complete elliptic integral of the first kind, `K(k)`.
    fn ellint_k_p<P: Policy>(self) -> Self;
    /// Complete elliptic integral of the second kind, `E(k)`.
    fn ellint_e_p<P: Policy>(self) -> Self;
    /// Complete `D(k) = (K(k) - E(k)) / k^2`.
    fn ellint_d_p<P: Policy>(self) -> Self;
    /// Complete elliptic integral of the third kind, `Pi(n, k)`.
    fn ellint_pi_p<P: Policy>(self, k: Self) -> Self;

    // The incomplete forms again, with the parameters that describe the geometry
    // taken as scalars. Sweeping `phi` at a fixed modulus is the usual shape of
    // the problem, and the all-vector forms above would make the caller allocate
    // and stream an array holding one repeated constant.

    /// `F(phi, k)` over a range of `phi` at one fixed modulus.
    fn ellint_f_vs_p<P: Policy>(self, k: Self::Element) -> Self;
    /// `E(phi, k)` over a range of `phi` at one fixed modulus.
    fn ellint_e_inc_vs_p<P: Policy>(self, k: Self::Element) -> Self;
    /// `D(phi, k)` over a range of `phi` at one fixed modulus.
    fn ellint_d_inc_vs_p<P: Policy>(self, k: Self::Element) -> Self;
    /// `Pi(n, phi, k)` over a range of `phi` at one fixed characteristic and modulus.
    fn ellint_pi_inc_vs_p<P: Policy>(self, n: Self::Element, k: Self::Element) -> Self;

    /// Carlson `R_F(x, y, z)`, the symmetric integral of the first kind.
    fn carlson_rf_p<P: Policy>(self, y: Self, z: Self) -> Self;
    /// Carlson `R_C(x, y)`, the degenerate form.
    fn carlson_rc_p<P: Policy>(self, y: Self) -> Self;
    /// Carlson `R_D(x, y, z)`, the symmetric integral of the second kind.
    fn carlson_rd_p<P: Policy>(self, y: Self, z: Self) -> Self;
    /// Carlson `R_J(x, y, z, p)`, the symmetric integral of the third kind.
    fn carlson_rj_p<P: Policy>(self, y: Self, z: Self, p: Self) -> Self;
    /// Carlson `R_G(x, y, z)`, the completely symmetric integral.
    fn carlson_rg_p<P: Policy>(self, y: Self, z: Self) -> Self;
}

#[rustfmt::skip]
impl<E, V> EllipticMathFfi for V
where
    E: FloatElement + EllipticConsts,
    V: FloatVector<Element = E> + SpecializedSpecialMath<E> + SpecialMathWithPolicy,
{
    #[inline(always)] fn ellint_k_p<P: Policy>(self) -> Self { SpecialMathWithPolicy::ellint_p::<P, _>(EllintK { k: self }) }
    #[inline(always)] fn ellint_e_p<P: Policy>(self) -> Self { SpecialMathWithPolicy::ellint_p::<P, _>(EllintE { k: self }) }
    #[inline(always)] fn ellint_d_p<P: Policy>(self) -> Self { SpecialMathWithPolicy::ellint_p::<P, _>(EllintD { k: self }) }
    #[inline(always)] fn ellint_pi_p<P: Policy>(self, k: Self) -> Self { SpecialMathWithPolicy::ellint_p::<P, _>(EllintPi { n: self, k }) }

    #[inline(always)] fn ellint_f_vs_p<P: Policy>(self, k: E) -> Self { SpecialMathWithPolicy::ellint_p::<P, _>(EllintF { phi: self, k: Self::splat(k) }) }
    #[inline(always)] fn ellint_e_inc_vs_p<P: Policy>(self, k: E) -> Self { SpecialMathWithPolicy::ellint_p::<P, _>(EllintEInc { phi: self, k: Self::splat(k) }) }
    #[inline(always)] fn ellint_d_inc_vs_p<P: Policy>(self, k: E) -> Self { SpecialMathWithPolicy::ellint_p::<P, _>(EllintDInc { phi: self, k: Self::splat(k) }) }
    #[inline(always)] fn ellint_pi_inc_vs_p<P: Policy>(self, n: E, k: E) -> Self { SpecialMathWithPolicy::ellint_p::<P, _>(EllintPiInc { n: Self::splat(n), phi: self, k: Self::splat(k) }) }

    #[inline(always)] fn carlson_rf_p<P: Policy>(self, y: Self, z: Self) -> Self { SpecialMathWithPolicy::carlson_p::<P, _>(CarlsonRf { x: self, y, z }) }
    #[inline(always)] fn carlson_rc_p<P: Policy>(self, y: Self) -> Self { SpecialMathWithPolicy::carlson_p::<P, _>(CarlsonRc { x: self, y }) }
    #[inline(always)] fn carlson_rd_p<P: Policy>(self, y: Self, z: Self) -> Self { SpecialMathWithPolicy::carlson_p::<P, _>(CarlsonRd { x: self, y, z }) }
    #[inline(always)] fn carlson_rj_p<P: Policy>(self, y: Self, z: Self, p: Self) -> Self { SpecialMathWithPolicy::carlson_p::<P, _>(CarlsonRj { x: self, y, z, p }) }
    #[inline(always)] fn carlson_rg_p<P: Policy>(self, y: Self, z: Self) -> Self { SpecialMathWithPolicy::carlson_p::<P, _>(CarlsonRg { x: self, y, z }) }
}

use thermite::math::policy::{
    DefaultPolicy,
    policies::{HighPerformance, Precision as HighPrecision},
};

#[derive(Default, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ThermitePrecisionPolicy {
    /// Smart default that balances performance and accuracy, using faster approximations when they are sufficiently accurate.
    #[default]
    DefaultPolicy = 0,
    /// Prioritizes performance over accuracy, using the fastest available approximations.
    HighPerformance = -1,
    /// Prioritizes accuracy over performance, using the most precise approximations available, or covering
    /// more edge cases at the cost of performance.
    HighPrecision = 1,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ThermiteDenormalResult {
    /// Setting the denormal behavior is not supported.
    NotSupported = 0,
    /// Setting the denormal behavior was a success
    Success = 1,
    /// Setting the denormal behavior was a success,
    /// and denormals were previously enabled.
    SuccessWasEnabled = 2,
}

macro_rules! c_str {
    ($($s:expr),*) => { concat!($($s),*, "\0").as_ptr() as *const c_char };
}

// these are removed from the public API, but are useful in the macros for
// generating the VTable methods, and cbindgen will ignore them.

/// cbindgen:ignore
type Int32 = i32;
/// cbindgen:ignore
type Int32f = i32;
/// cbindgen:ignore
type Intf = i32;
/// cbindgen:ignore
type Int = i64;
/// cbindgen:ignore
type Floatf = f32;
/// cbindgen:ignore
type Float = f64;

#[inline(never)]
unsafe extern "C" fn disable_denormals_template<S: NativeIsa>() -> ThermiteDenormalResult {
    match unsafe { S::disable_denormals() } {
        Ok(true) => ThermiteDenormalResult::SuccessWasEnabled,
        Ok(false) => ThermiteDenormalResult::Success,
        Err(_) => ThermiteDenormalResult::NotSupported,
    }
}

#[inline(never)]
unsafe extern "C" fn enable_denormals_template<S: NativeIsa>() -> ThermiteDenormalResult {
    match unsafe { S::enable_denormals() } {
        Ok(_) => ThermiteDenormalResult::Success,
        Err(_) => ThermiteDenormalResult::NotSupported,
    }
}

macro_rules! decl_methods {
    (ISA $policy:ty => $path:ident::$isa:ident [$feature:literal] [$arch:meta]
        $( MAPPING [
            $(    ($($input:ident),+) $([ $($scalar:ident: $ty:ty),+ ])? $mapping:ident $suffix:ident ($($output:ident),+)    ),* $(,)?
        ] ),+
    ) => {paste::paste! {
        #[cfg($arch)]
        const fn [<$isa:lower _ $policy:snake>]() -> Self {$($(
            #[inline(never)] #[target_feature(enable = $feature)]
            unsafe extern "C" fn [<$mapping f_ $suffix>](len: usize, $($input: *const f32,)+ $($output: *mut f32,)+ $( $($scalar: [<$ty f>],)+ )?) {
                unsafe { map::map_overlapping::<thermite::backend::$path::$isa, _, _, _, _>(
                    len, [$($input,)+], [$($output,)+], &[<$policy $mapping:camel $suffix:upper KernelF>] { $( $($scalar: $scalar as _,)+ )? }
                ) };
            }

            #[inline(never)] #[target_feature(enable = $feature)]
            unsafe extern "C" fn [<$mapping _ $suffix>](len: usize, $($input: *const f64,)+ $($output: *mut f64,)+ $( $($scalar: $ty,)+ )?) {
                unsafe { map::map_overlapping::<thermite::backend::$path::$isa, _, _, _, _>(
                    len, [$($input,)+], [$($output,)+], &[<$policy $mapping:camel $suffix:upper Kernel>] { $( $($scalar,)+ )? }
                ) };
            })*)+

            Self {
                disable_denormals: disable_denormals_template::<thermite::backend::$path::$isa>,
                enable_denormals: enable_denormals_template::<thermite::backend::$path::$isa>,
                name: c_str!(stringify!($isa), "/", stringify!($policy)),
                alignment: align_of::<<thermite::backend::$path::$isa as thermite::simd::NativeIsa>::NativeAlignment>(),
                $($([<$mapping f_ $suffix>], [<$mapping _ $suffix>],)*)+
            }
        }
    }};

    (SCALAR $policy:ty => [$arch:meta]
        $( MAPPING [
            $(    ($($input:ident),+) $([ $($scalar:ident: $ty:ty),+ ])? $mapping:ident $suffix:ident ($($output:ident),+)    ),* $(,)?
        ] ),+
    ) => {paste::paste! {
        #[cfg($arch)]
        const fn [<scalar_ $policy:snake>]() -> Self {$($(
            #[inline(never)]
            unsafe extern "C" fn [<$mapping f_ $suffix>](len: usize, $($input: *const f32,)+ $($output: *mut f32,)+ $( $($scalar: [<$ty f>],)+ )?) {
                unsafe { map::map_overlapping::<thermite::backend::scalar::Scalar, _, _, _, _>(
                    len, [$($input,)+], [$($output,)+], &[<$policy $mapping:camel $suffix:upper KernelF>] { $( $($scalar: $scalar as _,)+ )? }
                ) };
            }

            #[inline(never)]
            unsafe extern "C" fn [<$mapping _ $suffix>](len: usize, $($input: *const f64,)+ $($output: *mut f64,)+ $( $($scalar: $ty,)+ )?) {
                unsafe { map::map_overlapping::<thermite::backend::scalar::Scalar, _, _, _, _>(
                    len, [$($input,)+], [$($output,)+], &[<$policy $mapping:camel $suffix:upper Kernel>] { $( $($scalar,)+ )? }
                ) };
            })*)+

            Self {
                disable_denormals: disable_denormals_template::<thermite::backend::scalar::Scalar>,
                enable_denormals: enable_denormals_template::<thermite::backend::scalar::Scalar>,
                name: c_str!("Scalar/", stringify!($policy)),
                alignment: align_of::<f32>(),
                $($([<$mapping f_ $suffix>], [<$mapping _ $suffix>],)*)+
            }
        }
    }};

    (@COUNT $($val:ident),*) => { <[&'static str]>::len(&[$(stringify!($val)),*]) };

    (POLICY $policy:ty =>
        // automatically derived mapping kernels
        $( MAPPING: $trait:ident [$(
            $(#[$meta:meta])*
            ($($input:ident),+) $([ $($scalar:ident: $ty:ty),+ ])? $mapping:ident $suffix:ident $method:ident ($($output:ident),+)
        ),* $(,)?] ),+
    ) => {paste::paste! {
        $($(
            struct [<$policy $mapping:camel $suffix:upper Kernel>] {
                $($($scalar: $ty,)+)?
            }

            struct [<$policy $mapping:camel $suffix:upper KernelF>] {
                $($($scalar: [<$ty f>],)+)?
            }

            const _: () = {
                const I: usize = decl_methods!(@COUNT $($input),*);
                const O: usize = decl_methods!(@COUNT $($output),*);

                impl<V: $trait<Element = f64>> map::MapKernel2<V, I, O> for [<$policy $mapping:camel $suffix:upper Kernel>] {
                    #[inline(always)] fn map(&self, [$($input),+]: [V; I]) -> [V; O] {
                        <V as $trait>::[<$method _p>]::<$policy>($($input,)+ $( $(self.$scalar),+ )? ).into_outputs()
                    }
                }

                impl<V: $trait<Element = f32>> map::MapKernel2<V, I, O> for [<$policy $mapping:camel $suffix:upper KernelF>] {
                    #[inline(always)] fn map(&self, [$($input),+]: [V; I]) -> [V; O] {
                        <V as $trait>::[<$method _p>]::<$policy>($($input,)+ $( $(self.$scalar),+ )? ).into_outputs()
                    }
                }
            };
        )*)+

        impl VTable {
            // Not on AArch64: AdvSIMD is mandatory there, so the scalar table is
            // unreachable and would only add ~250 dead functions to the binary.
            decl_methods!(SCALAR $policy => [not(target_arch = "aarch64")]
                MAPPING [$( $(($($input),+) $([ $($scalar: $ty),+ ])? $mapping $suffix ($($output),+) ),* ),* ]);
            decl_methods!(ISA $policy => x86_v2::X86V2 ["sse4.2"] [any(target_arch = "x86", target_arch = "x86_64")]
                MAPPING [$( $(($($input),+) $([ $($scalar: $ty),+ ])? $mapping $suffix ($($output),+) ),* ),* ]);
            decl_methods!(ISA $policy => x86_v3::X86V3 ["avx,avx2,fma"] [any(target_arch = "x86", target_arch = "x86_64")]
                MAPPING [$( $(($($input),+) $([ $($scalar: $ty),+ ])? $mapping $suffix ($($output),+) ),* ),* ]);
            // AdvSIMD is mandatory in AArch64, so this is the only backend there
            // and there is nothing to detect at runtime.
            decl_methods!(ISA $policy => neon::Neon ["neon"] [target_arch = "aarch64"]
                MAPPING [$( $(($($input),+) $([ $($scalar: $ty),+ ])? $mapping $suffix ($($output),+) ),* ),* ]);
        }
    }};

    (
        $( MAPPING: $trait:ident [$(
            $(#[$meta:meta])*
            ($($input:ident),+) $([ $($scalar:ident: $ty:ty),+ ])? $mapping:ident $suffix:ident $method:ident ($($output:ident),+)
        ),* $(,)?]),+
    ) => {paste::paste! {
        #[repr(C)]
        pub struct VTable {
            pub disable_denormals: unsafe extern "C" fn() -> ThermiteDenormalResult,
            pub enable_denormals: unsafe extern "C" fn() -> ThermiteDenormalResult,

            $($(
                $(#[$meta])*
                ///
                /// # Safety
                /// The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
                pub [<$mapping f_ $suffix>]: unsafe extern "C" fn(len: usize, $( $input: *const f32, )+ $( $output: *mut f32, )+ $( $($scalar: [<$ty f>],)+ )?),
                $(#[$meta])*
                ///
                /// # Safety
                /// The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
                pub [<$mapping _ $suffix>]: unsafe extern "C" fn(len: usize, $( $input: *const f64, )+ $( $output: *mut f64, )+ $( $($scalar: $ty,)+ )?),
            )*)+

            pub alignment: usize,
            pub name: *const c_char,
        }

        decl_methods!(POLICY DefaultPolicy =>
            $( MAPPING: $trait [$( $(#[$meta])* ($($input),+) $([ $($scalar: $ty),+ ])? $mapping $suffix $method ($($output),+) ),*] ),+ );

        #[cfg(feature = "high_performance")]
        decl_methods!(POLICY HighPerformance =>
            $( MAPPING: $trait [$( $(#[$meta])* ($($input),+) $([ $($scalar: $ty),+ ])? $mapping $suffix $method ($($output),+) ),*] ),+ );

        #[cfg(feature = "high_precision")]
        decl_methods!(POLICY HighPrecision =>
            $( MAPPING: $trait [$( $(#[$meta])* ($($input),+) $([ $($scalar: $ty),+ ])? $mapping $suffix $method ($($output),+) ),*] ),+ );

        $($(
            $(#[$meta])*
            ///
            /// # Safety
            /// The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
            #[inline(never)] #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<thermite_ $mapping f_ $suffix>](len: usize, $( $input: *const f32, )+ $( $output: *mut f32, )+ $( $($scalar: [<$ty f>],)+ )?)
            { unsafe { (THERMITE_VTABLE.[<$mapping f_ $suffix>])(len, $( $input, )+ $( $output, )+ $( $($scalar,)+ )? ) }; }

            $(#[$meta])*
            ///
            /// # Safety
            /// The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
            #[inline(never)] #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<thermite_ $mapping _ $suffix>](len: usize, $( $input: *const f64, )+ $( $output: *mut f64, )+ $( $($scalar: $ty,)+ )?)
            { unsafe { (THERMITE_VTABLE.[<$mapping _ $suffix>])(len, $( $input, )+ $( $output, )+ $( $($scalar,)+ )? ) }; }
        )*)+
    }};
}

static mut THERMITE_POLICY: ThermitePrecisionPolicy = ThermitePrecisionPolicy::DefaultPolicy;
// AArch64 always has AdvSIMD, so the NEON table is correct before `thermite_init`
// is ever called and there is no scalar stage to fall back through. Elsewhere the
// scalar table is the safe pre-init default until dispatch runs.
#[cfg(target_arch = "aarch64")]
static mut THERMITE_VTABLE: VTable = VTable::neon_default_policy();
#[cfg(not(target_arch = "aarch64"))]
static mut THERMITE_VTABLE: VTable = VTable::scalar_default_policy();

impl VTable {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn get(policy: ThermitePrecisionPolicy) -> Self {
        use thermite::isa::InstructionSet;

        match (policy, InstructionSet::get()) {
            #[cfg(feature = "high_performance")]
            (ThermitePrecisionPolicy::HighPerformance, InstructionSet::X86V2) => Self::x86v2_high_performance(),
            #[cfg(feature = "high_performance")]
            (ThermitePrecisionPolicy::HighPerformance, InstructionSet::X86V3) => Self::x86v3_high_performance(),

            #[cfg(feature = "high_precision")]
            (ThermitePrecisionPolicy::HighPrecision, InstructionSet::X86V2) => Self::x86v2_high_precision(),
            #[cfg(feature = "high_precision")]
            (ThermitePrecisionPolicy::HighPrecision, InstructionSet::X86V3) => Self::x86v3_high_precision(),

            (_, InstructionSet::X86V2) => Self::x86v2_default_policy(),
            (_, InstructionSet::X86V3) => Self::x86v3_default_policy(),

            #[cfg(feature = "high_performance")]
            (ThermitePrecisionPolicy::HighPerformance, _) => Self::scalar_high_performance(),
            #[cfg(feature = "high_precision")]
            (ThermitePrecisionPolicy::HighPrecision, _) => Self::scalar_high_precision(),

            (_, _) => Self::scalar_default_policy(),
        }
    }

    /// AdvSIMD is mandatory in AArch64, so there is nothing to detect: the policy
    /// is the only choice, and every rung below NEON is unreachable.
    #[cfg(target_arch = "aarch64")]
    pub fn get(policy: ThermitePrecisionPolicy) -> Self {
        match policy {
            #[cfg(feature = "high_performance")]
            ThermitePrecisionPolicy::HighPerformance => Self::neon_high_performance(),
            #[cfg(feature = "high_precision")]
            ThermitePrecisionPolicy::HighPrecision => Self::neon_high_precision(),
            _ => Self::neon_default_policy(),
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
    pub fn init() {
        unsafe { THERMITE_VTABLE = Self::get(THERMITE_POLICY) };
    }
}

/// Initializes a Thermite FFI VTable instance with the specified precision policy, allowing the caller to choose
/// between different performance and accuracy trade-offs. This function does not allocate, and simply fills in the
/// provided VTable struct with the appropriate function pointers based on the given precision policy and available instruction set.
///
/// # Safety
/// The caller must ensure that `vtable` is a valid pointer to a `VTable` instance.
#[inline(never)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn thermite_init_vtable(vtable: *mut VTable, policy: ThermitePrecisionPolicy) {
    unsafe { *vtable = VTable::get(policy) };
}

/// Initializes the Thermite FFI, setting up the function pointers based on the current precision policy and available instruction set.
///
/// If not set, the default precision policy is `ThermitePrecisionPolicy::DefaultPolicy`, which provides a good balance of
/// performance and accuracy for most use cases. The caller can change the precision policy by calling
/// `thermite_init_with_policy` instead of this function.
#[inline(never)]
#[unsafe(no_mangle)]
pub extern "C" fn thermite_init() {
    VTable::init();
}

/// Initializes the Thermite FFI with a specific precision policy, allowing the caller to choose between
/// different performance and accuracy trade-offs.
#[inline(never)]
#[unsafe(no_mangle)]
pub extern "C" fn thermite_init_with_policy(policy: ThermitePrecisionPolicy) {
    unsafe { THERMITE_POLICY = policy };
    thermite_init();
}

#[inline(never)]
#[unsafe(no_mangle)]
pub extern "C" fn thermite_backend_name() -> *const c_char {
    // SAFETY: The returned pointer is valid as long as the program is running,
    // and points to a null-terminated string. Even if VTABLE is changed,
    // the string it points to will still be valid.
    // #[allow(static_mut_refs)]
    unsafe { THERMITE_VTABLE.name }
}

/// Attempt to disable denormal handling on the current thread.
#[inline(never)]
#[unsafe(no_mangle)]
pub extern "C" fn thermite_disable_denormals() -> ThermiteDenormalResult {
    unsafe { (THERMITE_VTABLE.disable_denormals)() }
}

/// Attempt to enable denormal handling on the current thread.
#[inline(never)]
#[unsafe(no_mangle)]
pub extern "C" fn thermite_enable_denormals() -> ThermiteDenormalResult {
    unsafe { (THERMITE_VTABLE.enable_denormals)() }
}

decl_methods! {
    MAPPING: RealMathWithPolicyFfi [
        /// Floating-point addition
        (a, b)add v add_v(y),
        /// Floating-point subtraction
        (a, b)sub v sub_v(y),
        /// Floating-point multiplication
        (a, b)mul v mul_v(y),
        /// Floating-point division
        (a, b)div v div_v(y),
        /// Floating-point remainder (modulo/fmod)
        (a, b)rem v rem_v(y),
        /// Rounds a floating-point number to the nearest integer
        (x)round v round_v(y),
        /// Rounds a floating-point number down to the nearest integer
        (x)floor v floor_v(y),
        /// Rounds a floating-point number up to the nearest integer
        (x)ceil v ceil_v(y),
        /// Truncates a floating-point number, removing the fractional part
        (x)trunc v trunc_v(y),
        /// Computes the fractional part of a floating-point number
        (x)fract v fract_v(y),
        /// Computes the next representable floating-point value greater than the input
        (x)next_up v next_up_v(y),
        /// Computes the next representable floating-point value less than the input
        (x)next_down v next_down_v(y),
        /// Computes the minimum of two floating-point numbers
        (a, b)min v min_v(y),
        /// Computes the maximum of two floating-point numbers
        (a, b)max v max_v(y),
        /// Clamps a floating-point number between a minimum and maximum scalar value
        (x)[min: Float, max: Float] clamp vs clamp_vs(y),
        /// Computes the absolute value of a floating-point number
        (x)abs v abs_v(y),
        /// Computes the sign of a floating-point number, returning -1.0 for negative values, 1.0 for positive values, and 0.0 for zero
        (x)signum v signum_v(y),
        /// Computes (x * a) + b with only one rounding error, yielding a more accurate
        /// result than a separate multiplication and addition
        (x, a, b)mul_add v mul_add_v(y),
        /// Computes (x * a) - b with only one rounding error, yielding a more accurate
        /// result than a separate multiplication and subtraction
        (x, a, b)mul_sub v mul_sub_v(y),
        /// Computes -(x * a) + b with only one rounding error, yielding a more accurate
        /// result than a separate negated multiplication and addition
        (x, a, b)nmul_add v nmul_add_v(y),
        /// Computes -(x * a) - b with only one rounding error, yielding a more accurate
        /// result than a separate negated multiplication and subtraction
        (x, a, b)nmul_sub v nmul_sub_v(y)
    ],
    MAPPING: CoreMathWithPolicy [
        /// Computes the inverse square root, which may vary in accuracy and performance based on the chosen precision policy.
        (x)inverse_sqrt v inverse_sqrt(out),
        /// Computes the reciprocal (1/x), which may vary in accuracy and performance based on the chosen precision policy.
        (x)reciprocal v reciprocal(out)
    ],
    MAPPING: TranscendentalMathWithPolicy [
        /// Compute both sine and cosine of the input simultaneously, which will be more efficient than computing them separately.
        (x)sin_cos vv sin_cos(sin, cos),
        /// Compute both sine and cosine of the input multiplied by π simultaneously, which will be more efficient than computing them separately.
        (x)sin_cos_pi vv sincos_pi(sin, cos),
        /// Compute both hyperbolic sine and hyperbolic cosine of the input simultaneously, which will be more efficient than computing them separately.
        (x)sinh_cosh vv sinh_cosh(sinh, cosh),
        /// Computes the sine of a floating-point number
        (x)sin v sin(y),
        /// Computes the cosine of a floating-point number
        (x)cos v cos(y),
        /// Computes the tangent of a floating-point number
        (x)tan v tan(y),
        /// Computes the sine of the input multiplied by π, which may be more accurate for certain inputs than multiplying the input by π and then taking the sine.
        (x)sin_pi v sin_pi(y),
        /// Computes the cosine of the input multiplied by π, which may be more accurate for certain inputs than multiplying the input by π and then taking the cosine.
        (x)cos_pi v cos_pi(y),
        /// Computes the tangent of the input multiplied by π, which may be more accurate for certain inputs than multiplying the input by π and then taking the tangent.
        (x)tan_pi v tan_pi(y),
        /// Computes the sinc function, defined as sin(πx)/(πx) for x != 0 and 1 for x = 0
        (x)sinc v sinc(y),
        /// Computes the sinc function of the input multiplied by π, defined as sin(π^2 x)/(π^2 x) for x != 0 and 1 for x = 0,
        /// which may be more accurate for certain inputs than multiplying the input by π and then taking the sinc.
        (x)sinc_pi v sinc_pi(y),
        /// Computes the hyperbolic sine of a floating-point number
        (x)sinh v sinh(y),
        /// Computes the hyperbolic cosine of a floating-point number
        (x)cosh v cosh(y),
        /// Computes the hyperbolic tangent of a floating-point number
        (x)tanh v tanh(y),
        /// Computes the inverse sine (arcsine) of a floating-point number
        (y)asin v asin(x),
        /// Computes the inverse cosine (arccosine) of a floating-point number
        (y)acos v acos(x),
        /// Computes the inverse tangent (arctangent) of a floating-point number
        (y)atan v atan(x),
        /// Computes the inverse hyperbolic sine of a floating-point number
        (y)asinh v asinh(x),
        /// Computes the inverse hyperbolic cosine of a floating-point number
        (y)acosh v acosh(x),
        /// Computes the inverse hyperbolic tangent of a floating-point number
        (y)atanh v atanh(x),
        /// Computes the exponential of a floating-point number, which may vary in accuracy and performance based on the chosen precision policy.
        (x)exp v exp(y),
        /// Computes the half-exponential of a floating-point number, defined as exp(x)/2
        (x)exph v exph(y),
        /// Computes 2 raised to the power of a floating-point number
        (x)exp2 v exp2(y),
        /// Computes 10 raised to the power of a floating-point number
        (x)exp10 v exp10(y),
        /// Computes the exponential of a floating-point number minus one, which may be more accurate for small inputs than computing exp(x) - 1 directly.
        (x)exp_m1 v exp_m1(y),
        /// Computes the natural logarithm of a floating-point number, which may vary in accuracy and performance based on the chosen precision policy.
        (x)ln v ln(y),
        /// Computes the natural logarithm of one plus a floating-point number, which may be more accurate for small inputs than computing ln(1 + x) directly.
        (x)ln_1p v ln_1p(y),
        /// Computes the base-2 logarithm of a floating-point number
        (x)log2 v log2(y),
        /// Computes the base-10 logarithm of a floating-point number
        (x)log10 v log10(y),
        /// Computes the logarithm of a floating-point number with respect to an arbitrary base
        (x, base)log v log(y),
        /// Computes the cube root of a floating-point number
        (x)cbrt v cbrt(y),
        /// Computes x raised to the power of y, which may vary in accuracy and performance based on the chosen precision policy.
        (x, e)powf v powf(y),
        /// Computes cos(x) - 1, accurately for small inputs where computing the
        /// cosine and subtracting one loses every significant digit to cancellation.
        (x)cos_m1 v cos_m1(y),
        /// Computes the versed sine, 1 - cos(x).
        (x)versin v versin(y),
        /// Computes the haversine, (1 - cos(x)) / 2, the kernel of the great-circle
        /// distance formula.
        (x)haversin v haversin(y),
        /// Computes 2^x - 1, accurately for small inputs.
        (x)exp2_m1 v exp2_m1(y),
        /// Computes 10^x - 1, accurately for small inputs.
        (x)exp10_m1 v exp10_m1(y),
        /// Computes sqrt(1 + x) - 1, accurately for small inputs.
        (x)sqrt1pm1 v sqrt1pm1(y),
        /// Computes x^e - 1, accurately for results near zero.
        (x, e)powf_m1 v powf_m1(y),
        /// Computes (1 + x)^n, the compound-interest form, accurately for small x.
        (x, n)compound v compound(y),
        /// Computes log2(1 + x), accurately for small inputs.
        (x)log2_p1 v log2_p1(y),
        /// Computes log10(1 + x), accurately for small inputs.
        (x)log10_p1 v log10_p1(y),
        /// Computes ln(1 - exp(-x)), which is otherwise catastrophically inaccurate
        /// for small x and overflows for large x.
        (x)ln1m_expnx v ln1m_expnx(y),
        /// Computes ln(1 - exp(-x)) given a precomputed ln(x), for callers that
        /// already have it.
        (x, lnx)ln1m_expnx_ext v ln1m_expnx_ext(y)
    ],
    MAPPING: RealMathWithPolicy [
        /// Wraps an angle in radians to the range [-π, π)
        (x)wrap_angle v wrap_angle(y),
        /// Computes the absolute difference between two angles
        (a, b)angle_diff v angle_diff(d),
        /// Converts an angle from radians to degrees
        (x)to_degrees v to_degrees(y),
        /// Converts an angle from degrees to radians
        (x)to_radians v to_radians(y),
        /// Computes the angle (in radians) between the positive x-axis and the point (x, y), using the signs of both arguments to determine the correct quadrant of the result.
        ///
        /// This may vary in accuracy and performance based on the chosen precision policy.
        (y, x)atan2 v atan2(t),
        /// Performs linear interpolation between values a and b using t, where t is typically in the range [0, 1].
        (t, a, b)lerp v lerp(y),
        /// Computes ln(exp(a) + exp(b)) without overflowing on large inputs or
        /// underflowing on small ones. The log-sum-exp primitive behind softmax
        /// and most probability code that works in log space.
        (a, b)logaddexp v logaddexp(y)
    ],
    MAPPING: SpatialMathWithPolicy [
        (x, y)hypot v hypot(out)
    ],
    MAPPING: SpecialMathWithPolicy [
        /// Computes the error function, which may vary in accuracy and performance based on the chosen precision policy.
        (x)erf v erf(y),
        /// Computes the complementary error function, which may vary in accuracy and performance based on the chosen precision policy.
        (x)erfc v erfc(y),

        (x)tgamma v tgamma(y),
        (x)lgamma v lgamma(y),
        (x, y)beta v beta(z),

        /// Computes the Logistic sigmoid function, defined as 1 / (1 + exp(-x)), which maps any real-valued number into the range (0, 1).
        (x)logistic_sigmoid v logistic_sigmoid(y),

        /// Computes the digamma function, the logarithmic derivative of the gamma
        /// function.
        (x)digamma v digamma(y),
        /// Computes both real branches of the Lambert W function at once,
        /// W_0(x) and W_-1(x), where each satisfies w * exp(w) = x.
        ///
        /// W_0 is valid for x >= -1/e and W_-1 for -1/e <= x < 0. Outside those
        /// domains the respective output is NaN.
        (x)lambert_w vv lambert_w(w0, wm1)
    ],
    MAPPING: RealSpecialMathWithPolicy [
        /// Computes the inverse error function, which may vary in accuracy and performance based on the chosen precision policy.
        (y)erfinv v erfinv(x),
        /// Computes the probit function, the inverse of the standard normal
        /// cumulative distribution function.
        (p)probit v probit(x),
        /// Computes the natural logarithm of the absolute value of the gamma
        /// function, together with the sign of the gamma function itself.
        (x)lgamma_r vv lgamma_r(y, sign),
        /// Computes an exponential-free approximation of the swish activation.
        (x)algebraic_swish v algebraic_swish(y)
    ],
    MAPPING: RealPrimalMathWithPolicy [
        /// Computes the exponential-free swish activation and its derivative together.
        (x)algebraic_swish_d vv algebraic_swish_d(y, dy)
    ],
    MAPPING: RealMathWithPolicyFfi [
        /// 3rd-order smoothstep interpolation function
        (x)smoothstep v smoothstep(y),
        /// Inverse of the 3rd-order smoothstep function
        (y)inverse_smoothstep v inverse_smoothstep(x),
        /// 5th-order smoothstep interpolation function
        (x)smootherstep v smootherstep(y),
        /// Inverse of the 5th-order smoothstep function
        (y)inverse_smootherstep v inverse_smootherstep(x),
        (x) [k: Float] smooth_interpolator v smooth_interpolator_vs(y),
        (y) [k: Float] smooth_interpolator_inverse v smooth_interpolator_inverse_vs(x),
        /// Step function that returns 0.0 if x < edge and 1.0 if x >= edge
        (x) [edge: Float] step v step_vs(y),
        /// Linear interpolation between scalars a and b by x, where x is in the range [0, 1]
        (x) [a: Float, b: Float] lerp vs lerp_vs(y),
        /// Raises x to the power of exp, where exp is an integer
        (x) [exp: Int32] powi vs powi_vs(y),
        /// Computes the Gaussian function with amplitude `a` and standard deviation `c`, defined as `a * exp(-0.5 * (self / c)^2)`.
        ///
        /// The position `b` is assumed to be zero. For a non-zero position, use `self - b` as the input.
        (x) [a: Float, c: Float] gaussian vs gaussian_vs(y),

        /// Computes the GELU activation with shape parameter `alpha`.
        (x) [alpha: Float] gelu vs gelu_vs(y),
        /// Computes the swish (SiLU) activation, x * sigmoid(beta * x).
        (x) [beta: Float] swish vs swish_vs(y),
        /// Computes the softplus activation, a smooth approximation of ReLU, with
        /// sharpness `k`.
        (x) [k: Float] softplus vs softplus_vs(y),

        /// Computes the GELU activation and its derivative together, in one pass.
        (x) [alpha: Float] gelu_d vvs gelu_d_vs(y, dy),
        /// Computes the swish activation and its derivative together, in one pass.
        (x) [beta: Float] swish_d vvs swish_d_vs(y, dy),
        /// Computes the softplus activation and its derivative together, in one pass.
        (x) [k: Float] softplus_d vvs softplus_d_vs(y, dy),
        /// Computes x / (1 + |x|), the softsign function.
        (x)algebraic_sigmoid_1 v algebraic_sigmoid_1(y),
        /// Computes x / sqrt(1 + x^2).
        (x)algebraic_sigmoid_2 v algebraic_sigmoid_2(y),
        /// Computes the softsign function and its derivative together, in one pass.
        (x)algebraic_sigmoid_d_1 vv algebraic_sigmoid_d_1(y, dy),
        /// Computes x / sqrt(1 + x^2) and its derivative together, in one pass.
        (x)algebraic_sigmoid_d_2 vv algebraic_sigmoid_d_2(y, dy),

        /// Computes the exponential integral E_1(x).
        (x)expint_1 v expint_1(y),

        /// Computes the Euclidean length of a 3-vector without intermediate
        /// overflow or underflow.
        (x, y, z)hypot_3 v hypot_3(out),
        /// Computes the Euclidean length of a 4-vector without intermediate
        /// overflow or underflow.
        (x, y, z, w)hypot_4 v hypot_4(out),

        /// Remaps a value from the range [in_min, in_max] onto [out_min, out_max].
        (x) [in_min: Float, in_max: Float, out_min: Float, out_max: Float] rescale vs rescale_vs(y)
    ],
    MAPPING: EllipticMathFfi [
        /// Computes the complete elliptic integral of the first kind, K(k).
        (k)ellint_k v ellint_k(y),
        /// Computes the complete elliptic integral of the second kind, E(k).
        (k)ellint_e v ellint_e(y),
        /// Computes the complete elliptic integral D(k) = (K(k) - E(k)) / k^2.
        (k)ellint_d v ellint_d(y),
        /// Computes the complete elliptic integral of the third kind, Pi(n, k).
        (n, k)ellint_pi v ellint_pi(y),

        /// Computes F(phi, k) over an array of phi at a single fixed modulus k.
        (phi) [k: Float] ellint_f vs ellint_f_vs(y),
        /// Computes E(phi, k) over an array of phi at a single fixed modulus k.
        (phi) [k: Float] ellint_e_inc vs ellint_e_inc_vs(y),
        /// Computes D(phi, k) over an array of phi at a single fixed modulus k.
        (phi) [k: Float] ellint_d_inc vs ellint_d_inc_vs(y),
        /// Computes Pi(n, phi, k) over an array of phi at a single fixed
        /// characteristic n and modulus k.
        (phi) [n: Float, k: Float] ellint_pi_inc vs ellint_pi_inc_vs(y),

        /// Computes the Carlson symmetric elliptic integral of the first kind, R_F(x, y, z).
        (x, y, z)carlson_rf v carlson_rf(out),
        /// Computes the degenerate Carlson symmetric elliptic integral R_C(x, y).
        (x, y)carlson_rc v carlson_rc(out),
        /// Computes the Carlson symmetric elliptic integral of the second kind, R_D(x, y, z).
        (x, y, z)carlson_rd v carlson_rd(out),
        /// Computes the Carlson symmetric elliptic integral of the third kind, R_J(x, y, z, p).
        (x, y, z, p)carlson_rj v carlson_rj(out),
        /// Computes the completely symmetric Carlson elliptic integral R_G(x, y, z).
        (x, y, z)carlson_rg v carlson_rg(out)
    ]
}
