// cl.exe test.c /O2 /GL /link "../../target/release/thermite_ffi.dll.lib" ntdll.lib /LTCG /OPT:REF /OPT:ICF

#![no_std]

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

use core::ffi::c_char;

use thermite::{
    math::{CoreMathWithPolicy, RealMathWithPolicy, TranscendentalMathWithPolicy},
    prelude::Policy,
};
use thermite_special::SpecialMathWithPolicy;

/// Forms of RealMath methods with explicit generic parameters,
/// such as order, dimensions, edges, etc.
pub trait RealMathWithPolicyFfi: RealMathWithPolicy {
    #[inline(always)]
    fn smoothstep_p<P: Policy>(self) -> Self {
        RealMathWithPolicy::smoothstep_p::<P, 2>(self, None)
    }

    #[inline(always)]
    fn inverse_smoothstep_p<P: Policy>(self) -> Self {
        RealMathWithPolicy::inverse_smoothstep_p::<P, 2>(self, None)
    }

    #[inline(always)]
    fn smootherstep_p<P: Policy>(self) -> Self {
        RealMathWithPolicy::smoothstep_p::<P, 3>(self, None)
    }

    #[inline(always)]
    fn inverse_smootherstep_p<P: Policy>(self) -> Self {
        RealMathWithPolicy::inverse_smoothstep_p::<P, 3>(self, None)
    }
}

impl<T> RealMathWithPolicyFfi for T where T: RealMathWithPolicy {}

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

macro_rules! c_str {
    ($($s:expr),*) => {
        concat!($($s),*, "\0").as_ptr() as *const c_char
    };
}

pub type InplacePtr32 = unsafe extern "C" fn(*mut f32, usize);
pub type InplacePtr64 = unsafe extern "C" fn(*mut f64, usize);

macro_rules! decl_methods {
    (ISA $policy:ty => $path:ident::$isa:ident [$feature:literal]
        $( INPLACE: $trait:ident [$($inplace:ident[$unroll:literal]),*] ),+
    ) => {paste::paste! {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        pub const fn [<$isa:lower _ $policy:snake>]() -> Self {$($(
            #[inline(never)] #[target_feature(enable = $feature)]
            unsafe extern "C" fn [<$inplace f_inplace>](ptr: *mut f32, len: usize) {
                struct [<$isa $policy $inplace:camel Kernel>];

                impl<V: $trait> thermite::transform::MapKernel<V> for [<$isa $policy $inplace:camel Kernel>] {
                    #[inline(always)] fn map(&self, input: V) -> V { <V as $trait>::[<$inplace _p>]::<$policy>(input) }
                }

                unsafe { thermite::transform::map_inplace::<thermite::backend::$path::$isa, _, _, $unroll>(
                    core::slice::from_raw_parts_mut(ptr, len), &[<$isa $policy $inplace:camel Kernel>]
                ) };
            }

            #[inline(never)] #[target_feature(enable = $feature)]
            unsafe extern "C" fn [<$inplace _inplace>](ptr: *mut f64, len: usize) {
                struct [<$isa $policy $inplace:camel Kernel>];

                impl<V: $trait> thermite::transform::MapKernel<V> for [<$isa $policy $inplace:camel Kernel>] {
                    #[inline(always)] fn map(&self, input: V) -> V { <V as $trait>::[<$inplace _p>]::<$policy>(input) }
                }

                unsafe { thermite::transform::map_inplace::<thermite::backend::$path::$isa, _, _, $unroll>(
                    core::slice::from_raw_parts_mut(ptr, len), &[<$isa $policy $inplace:camel Kernel>]
                ) };
            })*)+

            Self {
                name: c_str!(stringify!($isa), "/", stringify!($policy)),
                $($([<$inplace f_inplace>], [<$inplace _inplace>],)*)+
            }
        }
    }};

    (SCALAR $policy:ty =>
        $( INPLACE: $trait:ident [$($inplace:ident),*] ),+
    ) => {paste::paste! {
        pub const fn [<scalar_ $policy:snake>]() -> Self {$($(
            #[inline(never)]
            unsafe extern "C" fn [<$inplace f_inplace>](ptr: *mut f32, len: usize) {
                struct [<Scalar $policy $inplace:camel Kernel>];

                impl<V: $trait> thermite::transform::MapKernel<V> for [<Scalar $policy $inplace:camel Kernel>] {
                    #[inline(always)] fn map(&self, input: V) -> V { <V as $trait>::[<$inplace _p>]::<$policy>(input) }
                }

                unsafe { thermite::transform::map_inplace::<thermite::backend::scalar::Scalar, _, _, 1>(
                    core::slice::from_raw_parts_mut(ptr, len), &[<Scalar $policy $inplace:camel Kernel>]
                ) };
            }

            #[inline(never)]
            unsafe extern "C" fn [<$inplace _inplace>](ptr: *mut f64, len: usize) {
                struct [<Scalar $policy $inplace:camel Kernel>];

                impl<V: $trait> thermite::transform::MapKernel<V> for [<Scalar $policy $inplace:camel Kernel>] {
                    #[inline(always)] fn map(&self, input: V) -> V { <V as $trait>::[<$inplace _p>]::<$policy>(input) }
                }

                unsafe { thermite::transform::map_inplace::<thermite::backend::scalar::Scalar, _, _, 1>(
                    core::slice::from_raw_parts_mut(ptr, len), &[<Scalar $policy $inplace:camel Kernel>]
                ) };
            })*)+

            Self {
                name: c_str!("Scalar/", stringify!($policy)),
                $($([<$inplace f_inplace>], [<$inplace _inplace>],)*)+
            }
        }
    }};

    ( $( INPLACE: $trait:ident [$($inplace:ident[$unroll:literal]),*] ),+) => {paste::paste! {
        #[repr(C)]
        pub struct VTable {
            $($(
                #[doc = " In-place `" $inplace "` operation using the current Thermite backend.\n"]
                /// # Safety
                /// The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
                pub [<$inplace f_inplace>]: InplacePtr32,

                #[doc = " In-place `" $inplace "` operation using the current Thermite backend.\n"]
                /// # Safety
                /// The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
                pub [<$inplace _inplace>]: InplacePtr64,
            )*)+

            pub name: *const c_char,
        }

        impl VTable {
            decl_methods!(SCALAR DefaultPolicy => $( INPLACE: $trait [$($inplace),*] ),+);
            decl_methods!(SCALAR HighPerformance => $( INPLACE: $trait [$($inplace),*] ),+);
            decl_methods!(SCALAR HighPrecision => $( INPLACE: $trait [$($inplace),*] ),+);

            decl_methods!(ISA DefaultPolicy => x86_v2::X86V2 ["sse4.2"] $( INPLACE: $trait [$($inplace[$unroll]),*] ),+);
            decl_methods!(ISA HighPerformance => x86_v2::X86V2 ["sse4.2"] $( INPLACE: $trait [$($inplace[$unroll]),*] ),+);
            decl_methods!(ISA HighPrecision => x86_v2::X86V2 ["sse4.2"] $( INPLACE: $trait [$($inplace[$unroll]),*] ),+);

            decl_methods!(ISA DefaultPolicy => x86_v3::X86V3 ["avx,avx2,fma"] $( INPLACE: $trait [$($inplace[$unroll]),*] ),+);
            decl_methods!(ISA HighPerformance => x86_v3::X86V3 ["avx,avx2,fma"] $( INPLACE: $trait [$($inplace[$unroll]),*] ),+);
            decl_methods!(ISA HighPrecision => x86_v3::X86V3 ["avx,avx2,fma"] $( INPLACE: $trait [$($inplace[$unroll]),*] ),+);
        }

        $($(
            #[doc = " In-place `" $inplace "` operation using the current Thermite backend.\n"]
            /// # Safety
            /// The caller must ensure that `ptr` is valid for reads and writes of `len` `f32` elements.
            #[inline(never)] #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<thermite_ $inplace f_inplace>](ptr: *mut f32, len: usize) { unsafe { (THERMITE_VTABLE.[<$inplace f_inplace>])(ptr, len) }; }

            #[doc = " In-place `" $inplace "` operation using the current Thermite backend.\n"]
            /// # Safety
            /// The caller must ensure that `ptr` is valid for reads and writes of `len` `f64` elements.
            #[inline(never)] #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<thermite_ $inplace _inplace>](ptr: *mut f64, len: usize) { unsafe { (THERMITE_VTABLE.[<$inplace _inplace>])(ptr, len) }; }
        )*)+
    }};
}

decl_methods! {
    INPLACE: CoreMathWithPolicy [
        inverse_sqrt[4], reciprocal[4]
    ],
    INPLACE: TranscendentalMathWithPolicy [
        sin[2], cos[2], tan[1], sin_pi[1], cos_pi[1], tan_pi[1], sinc[1], sinc_pi[1],
        sinh[2], cosh[2], asin[2], acos[2], atan[2], asinh[2], acosh[2], atanh[1],
        exp[1], exph[1], exp2[1], exp10[1], exp_m1[1],
        ln[2], ln_1p[2], log2[2], log10[2], cbrt[2]
    ],
    INPLACE: RealMathWithPolicy [
        wrap_angle[2], to_degrees[4], to_radians[4]
    ],
    INPLACE: SpecialMathWithPolicy [
        erf[2], erfc[2]
    ],
    INPLACE: RealMathWithPolicyFfi [
        smoothstep[2], inverse_smoothstep[2], smootherstep[2], inverse_smootherstep[2]
    ]
}

static mut THERMITE_POLICY: ThermitePrecisionPolicy = ThermitePrecisionPolicy::DefaultPolicy;
static mut THERMITE_VTABLE: VTable = VTable::scalar_default_policy();

impl VTable {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn get(policy: ThermitePrecisionPolicy) -> Self {
        use thermite::isa::InstructionSet;

        match (policy, InstructionSet::get()) {
            (ThermitePrecisionPolicy::DefaultPolicy, InstructionSet::X86V2) => Self::x86v2_default_policy(),
            (ThermitePrecisionPolicy::DefaultPolicy, InstructionSet::X86V3) => Self::x86v3_default_policy(),
            (ThermitePrecisionPolicy::HighPerformance, InstructionSet::X86V2) => Self::x86v2_high_performance(),
            (ThermitePrecisionPolicy::HighPerformance, InstructionSet::X86V3) => Self::x86v3_high_performance(),
            (ThermitePrecisionPolicy::HighPrecision, InstructionSet::X86V2) => Self::x86v2_high_precision(),
            (ThermitePrecisionPolicy::HighPrecision, InstructionSet::X86V3) => Self::x86v3_high_precision(),
            (ThermitePrecisionPolicy::DefaultPolicy, _) => Self::scalar_default_policy(),
            (ThermitePrecisionPolicy::HighPerformance, _) => Self::scalar_high_performance(),
            (ThermitePrecisionPolicy::HighPrecision, _) => Self::scalar_high_precision(),
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
    unsafe {
        *vtable = VTable::get(policy);
    }
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
