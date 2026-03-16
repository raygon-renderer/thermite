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
//! This library is primarily intended to be dynamically linked. To that end, code size has
//! been reduced as much as reasonably possible while retaining performance, but as a result
//! not all functions are inlined nor all loops unrolled. If a larger library binary is
//! acceptable, the `disable_dispatch` crate feature will disable dispatch indirection and
//! force all algorithms to be inlined. An example of this is how many functions here rely on
//! the `exp` function internally. Enabling `disable_dispatch` will force the compiler to
//! copy the entire `exp` implementation into each and every function that uses it,
//! potentially improving performance by removing a function call and allowing LLVM
//! to interweave the `exp` computation better, at the cost of bumping the binary size
//! considerably.
//!
//! Furthermore, using a tool like `mpress` to compress the binary may be desired, but that's
//! more of a personal preference in the end.

// cargo expand -p thermite-ffi --all-features > ffi.rs && cbindgen -q -l c --crate thermite-ffi ffi.rs > ffi.h && echo "Done"
// cargo build --profile release-ffi -p thermite-ffi && Copy-Item ../../target/release-ffi/thermite_ffi.dll && mpress -b -s thermite_ffi.dll && echo "Done"
// cl.exe test.c /O2 /GL /link "../../target/release-ffi/thermite_ffi.dll.lib" ntdll.lib /LTCG /OPT:REF /OPT:ICF

#![no_std]

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

use core::ffi::c_char;

use thermite::{
    math::{CoreMathWithPolicy, RealMathWithPolicy, SpatialMathWithPolicy, TranscendentalMathWithPolicy},
    prelude::Policy,
    simd::NativeIsa,
};
use thermite_special::SpecialMathWithPolicy;

trait IntoArray<T, const N: usize> {
    fn into_array(self) -> [T; N];
}

#[rustfmt::skip]
const _: () = {
    impl<T> IntoArray<T, 1> for T { #[inline(always)] fn into_array(self) -> [T; 1] { [self] } }
    impl<T> IntoArray<T, 2> for (T, T) { #[inline(always)] fn into_array(self) -> [T; 2] { [self.0, self.1] } }
};

/// Forms of RealMath methods with explicit generic parameters,
/// such as order, dimensions, edges, etc.
#[rustfmt::skip]
pub trait RealMathWithPolicyFfi: RealMathWithPolicy + SpecialMathWithPolicy {
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
}

impl<T> RealMathWithPolicyFfi for T where T: RealMathWithPolicy + SpecialMathWithPolicy {}

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
    (ISA $policy:ty => $path:ident::$isa:ident [$feature:literal]
        $( MAPPING [
            $(    ($($input:ident),+) $([ $($scalar:ident: $ty:ty),+ ])? $mapping:ident $suffix:ident ($($output:ident),+)    ),* $(,)?
        ] ),+
    ) => {paste::paste! {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        const fn [<$isa:lower _ $policy:snake>]() -> Self {$($(
            #[inline(never)] #[target_feature(enable = $feature)]
            unsafe extern "C" fn [<$mapping f_ $suffix>](len: usize, $($input: *const f32,)+ $($output: *mut f32,)+ $( $($scalar: [<$ty f>],)+ )?) {
                unsafe { thermite::transform::map_overlapping::<thermite::backend::$path::$isa, _, _, _, _>(
                    len, [$($input,)+], [$($output,)+], &[<$policy $mapping:camel $suffix:upper KernelF>] { $( $($scalar: $scalar as _,)+ )? }
                ) };
            }

            #[inline(never)] #[target_feature(enable = $feature)]
            unsafe extern "C" fn [<$mapping _ $suffix>](len: usize, $($input: *const f64,)+ $($output: *mut f64,)+ $( $($scalar: $ty,)+ )?) {
                unsafe { thermite::transform::map_overlapping::<thermite::backend::$path::$isa, _, _, _, _>(
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

    (SCALAR $policy:ty =>
        $( MAPPING [
            $(    ($($input:ident),+) $([ $($scalar:ident: $ty:ty),+ ])? $mapping:ident $suffix:ident ($($output:ident),+)    ),* $(,)?
        ] ),+
    ) => {paste::paste! {
        const fn [<scalar_ $policy:snake>]() -> Self {$($(
            #[inline(never)]
            unsafe extern "C" fn [<$mapping f_ $suffix>](len: usize, $($input: *const f32,)+ $($output: *mut f32,)+ $( $($scalar: [<$ty f>],)+ )?) {
                unsafe { thermite::transform::map_overlapping::<thermite::backend::scalar::Scalar, _, _, _, _>(
                    len, [$($input,)+], [$($output,)+], &[<$policy $mapping:camel $suffix:upper KernelF>] { $( $($scalar: $scalar as _,)+ )? }
                ) };
            }

            #[inline(never)]
            unsafe extern "C" fn [<$mapping _ $suffix>](len: usize, $($input: *const f64,)+ $($output: *mut f64,)+ $( $($scalar: $ty,)+ )?) {
                unsafe { thermite::transform::map_overlapping::<thermite::backend::scalar::Scalar, _, _, _, _>(
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

                impl<V: $trait<Element = f64>> thermite::transform::MapKernel2<V, I, O> for [<$policy $mapping:camel $suffix:upper Kernel>] {
                    #[inline(always)] fn map(&self, [$($input),+]: [V; I]) -> [V; O] {
                        <V as $trait>::[<$method _p>]::<$policy>($($input,)+ $( $(self.$scalar),+ )? ).into_array()
                    }
                }

                impl<V: $trait<Element = f32>> thermite::transform::MapKernel2<V, I, O> for [<$policy $mapping:camel $suffix:upper KernelF>] {
                    #[inline(always)] fn map(&self, [$($input),+]: [V; I]) -> [V; O] {
                        <V as $trait>::[<$method _p>]::<$policy>($($input,)+ $( $(self.$scalar),+ )? ).into_array()
                    }
                }
            };
        )*)+

        impl VTable {
            decl_methods!(SCALAR $policy =>
                MAPPING [$( $(($($input),+) $([ $($scalar: $ty),+ ])? $mapping $suffix ($($output),+) ),* ),* ]);
            decl_methods!(ISA $policy => x86_v2::X86V2 ["sse4.2"]
                MAPPING [$( $(($($input),+) $([ $($scalar: $ty),+ ])? $mapping $suffix ($($output),+) ),* ),* ]);
            decl_methods!(ISA $policy => x86_v3::X86V3 ["avx,avx2,fma"]
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

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
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
        (x, e)powf v powf(y)
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
        (t, a, b)lerp v lerp(y)
    ],
    MAPPING: SpatialMathWithPolicy [
        (x, y)hypot v hypot(out)
    ],
    MAPPING: SpecialMathWithPolicy [
        /// Computes the error function, which may vary in accuracy and performance based on the chosen precision policy.
        (x)erf v erf(y),
        /// Computes the complementary error function, which may vary in accuracy and performance based on the chosen precision policy.
        (x)erfc v erfc(y),
        /// Computes the inverse error function, which may vary in accuracy and performance based on the chosen precision policy.
        (y)erfinv v erfinv(x),
        (x)tgamma v tgamma(y),
        (x)lgamma v lgamma(y),
        (x, y)beta v beta(z),
        /// Computes the sigmoid function, defined as 1 / (1 + exp(-x)), which maps any real-valued number into the range (0, 1).
        (x)sigmoid v sigmoid(y)
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
        (x) [a: Float, c: Float] gaussian vs gaussian_vs(y)
    ]
}
