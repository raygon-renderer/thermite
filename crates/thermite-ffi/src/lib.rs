// cargo expand -p thermite-ffi > ffi.rs && cbindgen -q -l c --crate thermite-ffi ffi.rs > ffi.h && echo "Done"
// cargo build --profile release-ffi -p thermite-ffi && Copy-Item ../../target/release-ffi/thermite_ffi.dll && upx -9 --ultra-brute thermite_ffi.dll && echo "Done"
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
pub trait RealMathWithPolicyFfi: RealMathWithPolicy + SpecialMathWithPolicy {
    #[inline(always)]
    fn add_v_p<P: Policy>(self, other: Self) -> Self {
        self + other
    }

    #[inline(always)]
    fn sub_v_p<P: Policy>(self, other: Self) -> Self {
        self - other
    }

    #[inline(always)]
    fn mul_v_p<P: Policy>(self, other: Self) -> Self {
        self * other
    }

    #[inline(always)]
    fn div_v_p<P: Policy>(self, other: Self) -> Self {
        self / other
    }

    #[inline(always)]
    fn rem_v_p<P: Policy>(self, other: Self) -> Self {
        self % other
    }

    #[inline(always)]
    fn round_v_p<P: Policy>(self) -> Self {
        self.round()
    }

    #[inline(always)]
    fn trunc_v_p<P: Policy>(self) -> Self {
        self.trunc()
    }

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

macro_rules! c_str {
    ($($s:expr),*) => { concat!($($s),*, "\0").as_ptr() as *const c_char };
}

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

macro_rules! decl_methods {
    (ISA $policy:ty => $path:ident::$isa:ident [$feature:literal]
        $( MAPPING [
            $(    ($($input:ident),+) $([ $($scalar:ident: $ty:ty),+ ])? $mapping:ident $suffix:ident ($($output:ident),+)    ),* $(,)?
        ] ),+
    ) => {paste::paste! {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        pub const fn [<$isa:lower _ $policy:snake>]() -> Self {$($(
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
        pub const fn [<scalar_ $policy:snake>]() -> Self {$($(
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
            ($($input:ident),+) $([ $($scalar:ident: $ty:ty),+ ])? $mapping:ident $suffix:ident $method:ident ($($output:ident),+)
        ),* $(,)?]),+
    ) => {paste::paste! {
        #[repr(C)]
        pub struct VTable {
            $($(
                #[doc = " `" $mapping "` operation using the given Thermite backend.\n"]
                /// # Safety
                /// The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
                pub [<$mapping f_ $suffix>]: unsafe extern "C" fn(len: usize, $( $input: *const f32, )+ $( $output: *mut f32, )+ $( $($scalar: [<$ty f>],)+ )?),
                #[doc = " `" $mapping "` operation using the given Thermite backend.\n"]
                /// # Safety
                /// The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
                pub [<$mapping _ $suffix>]: unsafe extern "C" fn(len: usize, $( $input: *const f64, )+ $( $output: *mut f64, )+ $( $($scalar: $ty,)+ )?),
            )*)+

            pub alignment: usize,
            pub name: *const c_char,
        }

        decl_methods!(POLICY DefaultPolicy =>
            $( MAPPING: $trait [$( ($($input),+) $([ $($scalar: $ty),+ ])? $mapping $suffix $method ($($output),+) ),*] ),+ );
        decl_methods!(POLICY HighPerformance =>
            $( MAPPING: $trait [$( ($($input),+) $([ $($scalar: $ty),+ ])? $mapping $suffix $method ($($output),+) ),*] ),+ );
        decl_methods!(POLICY HighPrecision =>
            $( MAPPING: $trait [$( ($($input),+) $([ $($scalar: $ty),+ ])? $mapping $suffix $method ($($output),+) ),*] ),+ );

        $($(
            #[doc = " `" $mapping "` operation using the current Thermite backend.\n"]
            /// # Safety
            /// The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f32` elements.
            #[inline(never)] #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<thermite_ $mapping f_ $suffix>](len: usize, $( $input: *const f32, )+ $( $output: *mut f32, )+ $( $($scalar: [<$ty f>],)+ )?)
            { unsafe { (THERMITE_VTABLE.[<$mapping f_ $suffix>])(len, $( $input, )+ $( $output, )+ $( $($scalar,)+ )? ) }; }

            #[doc = " `" $mapping "` operation using the current Thermite backend.\n"]
            /// # Safety
            /// The caller must ensure that pointers are valid for reads (const ptrs) or writes of `len` `f64` elements.
            #[inline(never)] #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<thermite_ $mapping _ $suffix>](len: usize, $( $input: *const f64, )+ $( $output: *mut f64, )+ $( $($scalar: $ty,)+ )?)
            { unsafe { (THERMITE_VTABLE.[<$mapping _ $suffix>])(len, $( $input, )+ $( $output, )+ $( $($scalar,)+ )? ) }; }
        )*)+
    }};
}

decl_methods! {
    MAPPING: RealMathWithPolicyFfi [
        (a, b)add v add_v(y),
        (a, b)sub v sub_v(y),
        (a, b)mul v mul_v(y),
        (a, b)div v div_v(y),
        (a, b)rem v rem_v(y),
        (x)round v round_v(y),
        (x)trunc v trunc_v(y)
    ],
    MAPPING: CoreMathWithPolicy [
        (x)inverse_sqrt v inverse_sqrt(out),
        (x)reciprocal v reciprocal(out)
    ],
    MAPPING: TranscendentalMathWithPolicy [
        (x)sin_cos vv sin_cos(sin, cos),
        (x)sin_cos_pi vv sincos_pi(sin, cos),
        (x)sinh_cosh vv sinh_cosh(sinh, cosh),
        (x)sin v sin(y),
        (x)cos v cos(y),
        (x)tan v tan(y),
        (x)sin_pi v sin_pi(y),
        (x)cos_pi v cos_pi(y),
        (x)tan_pi v tan_pi(y),
        (x)sinc v sinc(y),
        (x)sinc_pi v sinc_pi(y),
        (x)sinh v sinh(y),
        (x)cosh v cosh(y),
        (x)tanh v tanh(y),
        (y)asin v asin(x),
        (y)acos v acos(x),
        (y)atan v atan(x),
        (y)asinh v asinh(x),
        (y)acosh v acosh(x),
        (y)atanh v atanh(x),
        (x)exp v exp(y),
        (x)exph v exph(y),
        (x)exp2 v exp2(y),
        (x)exp10 v exp10(y),
        (x)exp_m1 v exp_m1(y),
        (x)ln v ln(y),
        (x)ln_1p v ln_1p(y),
        (x)log2 v log2(y),
        (x)log10 v log10(y),
        (x, base)log v log(y),
        (x)cbrt v cbrt(y),
        (x, e)powf v powf(y)
    ],
    MAPPING: RealMathWithPolicy [
        (x)wrap_angle v wrap_angle(y),
        (a, b)angle_diff v angle_diff(d),
        (x)to_degrees v to_degrees(y),
        (x)to_radians v to_radians(y),
        (y, x)atan2 v atan2(t),
        (t, a, b)lerp v lerp(y)
    ],
    MAPPING: SpatialMathWithPolicy [
        (x, y)hypot v hypot(out)
    ],
    MAPPING: SpecialMathWithPolicy [
        (x)erf v erf(y),
        (x)erfc v erfc(y),
        (y)erfinv v erfinv(x),
        (x)tgamma v tgamma(y),
        (x)lgamma v lgamma(y),
        (x, y)beta v beta(z),
    ],
    MAPPING: RealMathWithPolicyFfi [
        (x)smoothstep v smoothstep(y),
        (y)inverse_smoothstep v inverse_smoothstep(x),
        (x)smootherstep v smootherstep(y),
        (y)inverse_smootherstep v inverse_smootherstep(x),
        (x) [k: Float] smooth_interpolator v smooth_interpolator_vs(y),
        (x) [edge: Float] step v step_vs(y),
        (x) [a: Float, b: Float] lerp vs lerp_vs(y),
        (x) [exp: Int32] powi vs powi_vs(y),
        (x) [a: Float, c: Float] gaussian vs gaussian_vs(y)
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
