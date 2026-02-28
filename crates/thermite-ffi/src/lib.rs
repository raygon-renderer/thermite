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

    #[inline(always)]
    fn lerpv_p<P: Policy>(self, a: Self, b: Self) -> Self {
        RealMathWithPolicy::lerp_p::<P>(self, a, b)
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
    ($($s:expr),*) => { concat!($($s),*, "\0").as_ptr() as *const c_char };
}

macro_rules! decl_methods {
    (ISA $policy:ty => $path:ident::$isa:ident [$feature:literal]
        $( MAPPING [
            $(    ($($input:ident),+) $mapping:ident ($($output:ident),+)    ),*
        ] ),+
    ) => {paste::paste! {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        pub const fn [<$isa:lower _ $policy:snake>]() -> Self {$($(
            #[inline(never)] #[target_feature(enable = $feature)]
            unsafe extern "C" fn [<$mapping f>](len: usize, $($input: *const f32,)+ $($output: *mut f32),+) {
                unsafe { thermite::transform::map_overlapping::<thermite::backend::$path::$isa, _, _, _, _>(
                    len, [$($input,)+], [$($output,)+], &[<$policy $mapping:camel Kernel>]
                ) };
            }

            #[inline(never)] #[target_feature(enable = $feature)]
            unsafe extern "C" fn $mapping(len: usize, $($input: *const f64,)+ $($output: *mut f64),+) {
                unsafe { thermite::transform::map_overlapping::<thermite::backend::$path::$isa, _, _, _, _>(
                    len, [$($input,)+], [$($output,)+], &[<$policy $mapping:camel Kernel>]
                ) };
            })*)+

            Self {
                name: c_str!(stringify!($isa), "/", stringify!($policy)),
                alignment: align_of::<<thermite::backend::$path::$isa as thermite::simd::NativeIsa>::NativeAlignment>(),
                $($([<$mapping f>], $mapping,)*)+
            }
        }
    }};

    (SCALAR $policy:ty =>
        $( MAPPING [
            $(    ($($input:ident),+) $mapping:ident ($($output:ident),+)    ),*
        ] ),+
    ) => {paste::paste! {
        pub const fn [<scalar_ $policy:snake>]() -> Self {$($(
            #[inline(never)]
            unsafe extern "C" fn [<$mapping f>](len: usize, $($input: *const f32,)+ $($output: *mut f32),+) {
                unsafe { thermite::transform::map_overlapping::<thermite::backend::scalar::Scalar, _, _, _, _>(
                    len, [$($input,)+], [$($output,)+], &[<$policy $mapping:camel Kernel>]
                ) };
            }

            #[inline(never)]
            unsafe extern "C" fn $mapping(len: usize, $($input: *const f64,)+ $($output: *mut f64),+) {
                unsafe { thermite::transform::map_overlapping::<thermite::backend::scalar::Scalar, _, _, _, _>(
                    len, [$($input,)+], [$($output,)+], &[<$policy $mapping:camel Kernel>]
                ) };
            })*)+

            Self {
                name: c_str!("Scalar/", stringify!($policy)),
                alignment: align_of::<f32>(),
                $($([<$mapping f>], $mapping,)*)+
            }
        }
    }};

    (@COUNT $($val:ident),*) => { <[&'static str]>::len(&[$(stringify!($val)),*]) };

    (POLICY $policy:ty =>
        // automatically derived mapping kernels
        $( MAPPING: $trait:ident [$(
            ($($input:ident),+) $mapping:ident ($($output:ident),+)
        ),*], )+

        // implicitly defined mapping kernels defined elsewhere
        IMPLICIT: [ $(
            ($($input_i:ident),+) $implicit:ident ($($output_i:ident),+)
        ),*]
    ) => {paste::paste! {
        $($(
            /// cbindgen:ignore
            struct [<$policy $mapping:camel Kernel>];

            const _: () = {
                const I: usize = decl_methods!(@COUNT $($input),*);
                const O: usize = decl_methods!(@COUNT $($output),*);

                impl<V: $trait> thermite::transform::MapKernel2<V, I, O> for [<$policy $mapping:camel Kernel>] {
                    #[inline(always)] fn map(&self, [$($input),+]: [V; I]) -> [V; O] {
                        let res = <V as $trait>::[<$mapping _p>]::<$policy>($($input),+);

                        [res; 1]
                    }
                }
            };
        )*)+

        struct [<$policy SinCosKernel>];
        struct [<$policy SinCosPiKernel>];
        struct [<$policy SinhCoshKernel>];

        const _: () = {
            impl<V: TranscendentalMathWithPolicy> thermite::transform::MapKernel2<V, 1, 2> for [<$policy SinCosKernel>] {
                #[inline(always)] fn map(&self, [x]: [V; 1]) -> [V; 2] {
                    let (s, c) = <V as TranscendentalMathWithPolicy>::sin_cos_p::<$policy>(x); [s, c]
                }
            }

            impl<V: TranscendentalMathWithPolicy> thermite::transform::MapKernel2<V, 1, 2> for [<$policy SinCosPiKernel>] {
                #[inline(always)] fn map(&self, [x]: [V; 1]) -> [V; 2] {
                    let (s, c) = <V as TranscendentalMathWithPolicy>::sincos_pi_p::<$policy>(x); [s, c]
                }
            }

            impl<V: TranscendentalMathWithPolicy> thermite::transform::MapKernel2<V, 1, 2> for [<$policy SinhCoshKernel>] {
                #[inline(always)] fn map(&self, [x]: [V; 1]) -> [V; 2] {
                    let (sh, ch) = <V as TranscendentalMathWithPolicy>::sinh_cosh_p::<$policy>(x); [sh, ch]
                }
            }
        };

        impl VTable {
            decl_methods!(SCALAR $policy => MAPPING [
                $( $(($($input),+)  $mapping ($($output),+) ),* ,)*
                $( ($($input_i),+) $implicit ($($output_i),+) ),*
            ]);

            decl_methods!(ISA $policy => x86_v2::X86V2 ["sse4.2"] MAPPING [
                $( $(($($input),+)  $mapping ($($output),+) ),* ,)*
                $( ($($input_i),+) $implicit ($($output_i),+) ),*
            ]);
            decl_methods!(ISA $policy => x86_v3::X86V3 ["avx,avx2,fma"] MAPPING [
                $( $(($($input),+)  $mapping ($($output),+) ),* ,)*
                $( ($($input_i),+) $implicit ($($output_i),+) ),*
            ]);
        }
    }};

    (
        // automatically derived mapping kernels
        $( MAPPING: $trait:ident [$(
            ($($input:ident),+) $mapping:ident ($($output:ident),+)
        ),*],)+

        // implicitly defined mapping kernels defined elsewhere
        IMPLICIT: [ $(
            ($($input_i:ident),+) $implicit:ident ($($output_i:ident),+)
        ),*]
    ) => {paste::paste! {
        #[repr(C)]
        pub struct VTable {
            $($(
                #[doc = " `" $mapping "` operation using the given Thermite backend.\n"]
                /// # Safety
                /// The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
                pub [<$mapping f>]: unsafe extern "C" fn(len: usize, $( $input: *const f32, )+ $( $output: *mut f32 ),+),

                #[doc = " `" $mapping "` operation using the given Thermite backend.\n"]
                /// # Safety
                /// The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
                pub $mapping: unsafe extern "C" fn(len: usize, $( $input: *const f64, )+ $( $output: *mut f64 ),+),
            )*)+

            $(
                #[doc = " `" $implicit "` operation using the given Thermite backend.\n"]
                /// # Safety
                /// The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
                pub [<$implicit f>]: unsafe extern "C" fn(len: usize, $( $input_i: *const f32, )+ $( $output_i: *mut f32 ),+),

                #[doc = " `" $implicit "` operation using the given Thermite backend.\n"]
                /// # Safety
                /// The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
                pub $implicit: unsafe extern "C" fn(len: usize, $( $input_i: *const f64, )+ $( $output_i: *mut f64 ),+),
            )*

            pub alignment: usize,
            pub name: *const c_char,
        }

        decl_methods!(POLICY DefaultPolicy =>
            $( MAPPING: $trait [$( ($($input),+) $mapping ($($output),+) ),*], )+
            IMPLICIT: [$( ($($input_i),+) $implicit ($($output_i),+) ),*]
        );
        decl_methods!(POLICY HighPerformance =>
            $( MAPPING: $trait [$( ($($input),+) $mapping ($($output),+) ),*], )+
            IMPLICIT: [$( ($($input_i),+) $implicit ($($output_i),+) ),*]
        );
        decl_methods!(POLICY HighPrecision =>
            $( MAPPING: $trait [$( ($($input),+) $mapping ($($output),+) ),*], )+
            IMPLICIT: [$( ($($input_i),+) $implicit ($($output_i),+) ),*]
        );

        $($(
            #[doc = " `" $mapping "` operation using the current Thermite backend.\n"]
            /// # Safety
            /// The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
            #[inline(never)] #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<thermite_ $mapping f>](len: usize, $( $input: *const f32, )+ $( $output: *mut f32 ),+ )
            { unsafe { (THERMITE_VTABLE.[<$mapping f>])(len, $( $input, )+ $( $output ),+ ) }; }

            #[doc = " `" $mapping "` operation using the current Thermite backend.\n"]
            /// # Safety
            /// The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
            #[inline(never)] #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<thermite_ $mapping>](len: usize, $( $input: *const f64, )+ $( $output: *mut f64 ),+ )
            { unsafe { (THERMITE_VTABLE.$mapping)(len, $( $input, )+ $( $output ),+ ) }; }
        )*)+
    }};
}

decl_methods! {
    MAPPING: CoreMathWithPolicy [
        (x)inverse_sqrt(out), (x)reciprocal(out)
    ],
    MAPPING: TranscendentalMathWithPolicy [
        (x)sin(y), (x)cos(y), (x)tan(y), (x)sin_pi(y), (x)cos_pi(y), (x)tan_pi(y), (x)sinc(y), (x)sinc_pi(y),
        (x)sinh(y), (x)cosh(y), (x)tanh(y),
        (y)asin(x), (y)acos(x), (y)atan(x), (y)asinh(x), (y)acosh(x), (y)atanh(x),
        (x)exp(y), (x)exph(y), (x)exp2(y), (x)exp10(y), (x)exp_m1(y),
        (x)ln(y), (x)ln_1p(y), (x)log2(y), (x)log10(y),
        (x)cbrt(y), (x, e)powf(y)
    ],
    MAPPING: RealMathWithPolicy [
        (x)wrap_angle(y), (x)to_degrees(y), (x)to_radians(y), (y, x)atan2(t)
    ],
    MAPPING: SpecialMathWithPolicy [
        (x)erf(y), (x)erfc(y), (x)tgamma(y), (x)lgamma(y)
    ],
    MAPPING: RealMathWithPolicyFfi [
        (x)smoothstep(y), (y)inverse_smoothstep(x),
        (x)smootherstep(y), (y)inverse_smootherstep(x),
        (t, a, b)lerpv(y)
    ],
    IMPLICIT: [
        (x)sin_cos(sin, cos), (x)sin_cos_pi(sin, cos), (x)sinh_cosh(sinh, cosh)
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
