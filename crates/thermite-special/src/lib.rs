#![doc = include_str!("../README.md")]
#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(clippy::needless_arbitrary_self_type, clippy::needless_range_loop)]

use thermite::{
    element::{Element, ElementExt, FloatElementWithBits},
    math::{
        PrimalMathWithPolicy, PrimalProjection, TranscendentalMathWithPolicy,
        policy::{DefaultPolicy, Policy},
        scalar::Unwrap,
    },
    vector::{FloatVector, FloatVectorWithBits},
};

pub mod specialized;

// Raw approximation coefficients, shared with the sibling crates. Documented on the
// module itself rather than here: an outer doc at this declaration site is merged with
// the module's own and then resolved in THIS scope, which breaks every link it makes
// to its own submodules.
#[doc(hidden)]
pub mod tables;

pub use tables::bernoulli::BernoulliNumbers;

pub mod bernoulli;
pub mod zernike;

// The two normalization flags appear in the `zernike` signature below as a const
// generic, so a caller has to be able to name them without reaching into the module.
pub use crate::zernike::{ZERNIKE_ORTHONORMAL, ZERNIKE_UNIT_PEAK};

use crate::specialized::{CarlsonKind, EllipticKind, WrapTo};

// Spherical-harmonic support: `ShTable` appears in the public signatures below, and
// `MAX_SH_DEGREE` is the documented degree at which they leave the unrolled path.
pub use crate::specialized::{MAX_SH_DEGREE, ShTable};

/// Elliptic integral request structs and the traits they implement:
///
/// - Carlson symmetric integrals (for [`SpecialMath::carlson`]): [`CarlsonRf`](elliptic::CarlsonRf),
///   [`CarlsonRc`](elliptic::CarlsonRc), [`CarlsonRd`](elliptic::CarlsonRd), [`CarlsonRj`](elliptic::CarlsonRj),
///   [`CarlsonRg`](elliptic::CarlsonRg), implementing [`CarlsonKind`].
/// - Legendre integrals (for [`SpecialMath::ellint`]): [`EllintK`](elliptic::EllintK)/[`EllintF`](elliptic::EllintF),
///   [`EllintE`](elliptic::EllintE)/[`EllintEInc`](elliptic::EllintEInc),
///   [`EllintD`](elliptic::EllintD)/[`EllintDInc`](elliptic::EllintDInc),
///   [`EllintPi`](elliptic::EllintPi)/[`EllintPiInc`](elliptic::EllintPiInc), implementing
///   [`EllipticKind`]. Completeness is encoded by the struct: a complete integral has no `phi` field.
pub mod elliptic {
    pub use crate::specialized::EllipticConsts;

    pub use crate::specialized::{CarlsonKind, CarlsonRc, CarlsonRd, CarlsonRf, CarlsonRg, CarlsonRj};

    pub use crate::specialized::{
        EllintD, EllintDInc, EllintE, EllintEInc, EllintF, EllintK, EllintPi, EllintPiInc, EllipticKind,
    };
}

macro_rules! decl_math {
    ($(
        $(#[$trait_meta:meta])*
        trait $trait:ident $(: $($bound:ident)&+)? { $(
            $(#[$meta:meta])*
            fn $name:ident [ $($generics:tt)* ][$($generic_names:ident),*]( $($arg_name:ident :$arg_ty:ty),* $(,)?) -> $ret:ty
                $(where [ $($where_clause:tt)* ])?;
        )*
        // Optional block of "kind-dispatched" methods: a single request-struct argument carrying
        // the operation's data (e.g. `CarlsonRf { x, y, z }`). The struct's `eval` (a CarlsonKind /
        // EllipticKind impl) does the work. This generates the full trait family (policy + default +
        // dispatched vector impl + scalar) around it. Because the struct's element backend
        // (EllipticEval) covers both `Vector<R>` and scalar floats, the same bound works at the
        // scalar layer, no Unwrap wrapping needed here.
        $(@kinds {$(
            $(#[$kmeta:meta])*
            fn $kname:ident : $ktrait:path;
        )*})?
        // Optional block of methods whose scalar-layer signature differs from the
        // vector one. A `Self::Primal`-typed table parameter has no spelling on a bare
        // scalar (`f32` implements no vector trait), but the scalar IS its own primal,
        // so the scalar form takes plain `Self` and the impl Unwrap-wraps it into the
        // width-1 vector table as usual. Each fn declares that form after `= scalar`.
        $(@scalar_sig {$(
            $(#[$vmeta:meta])*
            fn $vname:ident [ $($vgenerics:tt)* ][$($vgeneric_names:ident),*]( $($varg_name:ident :$varg_ty:ty),* $(,)?) -> $vret:ty
                = scalar( $($vsarg_name:ident : $vsarg_ty:ty),* $(,)?) -> $vsret:ty;
        )*})?
        }
    )*) => {paste::paste! {$(
        #[doc = "" $trait " Math functions for floating-point vectors with customizable policies.\n\n"]
        #[doc = "Each method has a `_p`-suffixed variant in this trait that accepts a leading `P: Policy` generic.\n\n"]
        #[doc = "All floating-point vector types that implement [`Specialized" $trait "Math`](specialized::SpecializedSpecialMath) will\n"]
        #[doc = "automatically implement this trait, and [`" $trait "Math`] as well."]
        $(#[$trait_meta])*
        #[thermite::dispatch(Self)]
        pub trait [<$trait MathWithPolicy>]: $($($bound +)+)? {$(
            $(#[$meta])* fn [<$name _p>]<P: Policy, $($generics)*>($($arg_name: $arg_ty),*) -> $ret
                $(where $($where_clause)*)?;
        )*
        $($(
            $(#[$kmeta])* fn [<$kname _p>]<P: Policy, K: $ktrait<Output = Self>>(kind: K) -> Self;
        )*)?
        $($(
            $(#[$vmeta])* fn [<$vname _p>]<P: Policy, $($vgenerics)*>($($varg_name: $varg_ty),*) -> $vret;
        )*)?
        }

        #[doc = "" $trait " Math functions for floating-point vectors using the default policy.\n\n"]
        #[doc = "Implementors of [`" $trait "MathWithPolicy`] automatically implement this trait.\n\n"]
        #[doc = "Each method here has a `_p`-suffixed counterpart in [`" $trait "MathWithPolicy`] that\n"]
        #[doc = "accepts a leading `P: Policy` generic for fine-grained precision/performance control."]
        $(#[$trait_meta])*
        #[thermite::dispatch(Self)]
        pub trait [<$trait Math>]: [<$trait MathWithPolicy>] {$(
            $(#[$meta])* #[inline(always)] fn $name<$($generics)*>($($arg_name: $arg_ty),*) -> $ret
                $(where $($where_clause)*)?
            { [<$trait MathWithPolicy>]::[<$name _p>]::<DefaultPolicy, $($generic_names),*>($($arg_name),*) }
        )*
        $($(
            $(#[$kmeta])* #[inline(always)] fn $kname<K: $ktrait<Output = Self>>(kind: K) -> Self
            { [<$trait MathWithPolicy>]::[<$kname _p>]::<DefaultPolicy, K>(kind) }
        )*)?
        $($(
            // `<Self as ...>` explicitly: a `Self::Primal`-typed argument cannot drive
            // `Self` inference (`Primal` is not injective).
            $(#[$vmeta])* #[inline(always)] fn $vname<$($vgenerics)*>($($varg_name: $varg_ty),*) -> $vret
            { <Self as [<$trait MathWithPolicy>]>::[<$vname _p>]::<DefaultPolicy, $($vgeneric_names),*>($($varg_name),*) }
        )*)?
        }

        impl<M> [<$trait Math>] for M where M: [<$trait MathWithPolicy>] {}

        #[thermite::dispatch(Self)]
        impl<E, V: FloatVector<Element = E> $(+ $($bound +)+)?> [<$trait MathWithPolicy>] for V
        where
            V: specialized::[<Specialized $trait Math>]<E>,
        {$(
            #[cfg(not(feature = "disable_dispatch"))]
            $(#[$meta])* #[inline(always)] fn [<$name _p>]<P: Policy, $($generics)*>($($arg_name: $arg_ty),*) -> $ret
                $(where $($where_clause)*)?
            { V::$name::<P, $($generic_names),*>($($arg_name),*) }

            #[cfg(feature = "disable_dispatch")]
            $(#[$meta])* #[skip_dispatch] #[inline(always)] fn [<$name _p>]<P: Policy, $($generics)*>($($arg_name: $arg_ty),*) -> $ret
                $(where $($where_clause)*)?
            { V::$name::<P, $($generic_names),*>($($arg_name),*) }
        )*
        $($(
            // Kind methods delegate to the request struct's own `eval`; `#[dispatch]` wraps this in
            // the per-ISA trampolines, so `eval`'s inner Carlson/AGM work runs under target_feature.
            #[cfg(not(feature = "disable_dispatch"))]
            $(#[$kmeta])* #[inline(always)] fn [<$kname _p>]<P: Policy, K: $ktrait<Output = Self>>(kind: K) -> Self
            { kind.eval::<P>() }

            #[cfg(feature = "disable_dispatch")]
            $(#[$kmeta])* #[skip_dispatch] #[inline(always)] fn [<$kname _p>]<P: Policy, K: $ktrait<Output = Self>>(kind: K) -> Self
            { kind.eval::<P>() }
        )*)?
        $($(
            #[cfg(not(feature = "disable_dispatch"))]
            $(#[$vmeta])* #[inline(always)] fn [<$vname _p>]<P: Policy, $($vgenerics)*>($($varg_name: $varg_ty),*) -> $vret
            { <V as specialized::[<Specialized $trait Math>]<E>>::$vname::<P, $($vgeneric_names),*>($($varg_name),*) }

            #[cfg(feature = "disable_dispatch")]
            $(#[$vmeta])* #[skip_dispatch] #[inline(always)] fn [<$vname _p>]<P: Policy, $($vgenerics)*>($($varg_name: $varg_ty),*) -> $vret
            { <V as specialized::[<Specialized $trait Math>]<E>>::$vname::<P, $($vgeneric_names),*>($($varg_name),*) }
        )*)?
        })*

        #[doc = "Aggregate of all scalar special-math traits with customizable policies."]
        #[doc = ""]
        #[doc = "This trait collects every method from the following trait families into a single"]
        #[doc = "trait implemented directly on `f32` and `f64`:"]
        #[doc = ""]
        $(#[doc = "- [`" [<$trait MathWithPolicy>] "`]"])*
        #[doc = ""]
        #[doc = "All methods are prefixed with `scalar_` to avoid conflicts with inherent methods"]
        #[doc = "on `f32`/`f64`. The policy-aware versions additionally carry a `_p` suffix."]
        #[doc = ""]
        #[doc = "# Limitations"]
        #[doc = ""]
        #[doc = "This trait is **only** implemented for bare scalar types. Code that is generic over"]
        #[doc = "a `FloatVector` bound will not accept a bare `f32` or `f64` - the scalar must be"]
        #[doc = "wrapped in [`Vector`](thermite::Vector) first (e.g., `Vector::<f32>(x)`) to satisfy"]
        #[doc = "that bound. `ScalarSpecialMath` exists purely as a convenience for call-sites that"]
        #[doc = "already hold a concrete scalar and do not need to be generic."]
        #[doc = ""]
        #[doc = "For convenience, a default-policy version is provided by [`ScalarSpecialMath`], which"]
        #[doc = "drops the `_p` suffix and uses [`DefaultPolicy`](thermite::math::policy::DefaultPolicy) for all operations."]
        #[thermite::dispatch(Self)]
        #[diagnostic::on_unimplemented(
            message = "`{Self}` is not a bare floating-point scalar",
            note = "`ScalarSpecialMathWithPolicy` is implemented only for the bare scalar types `f32` and `f64`. For SIMD vectors, bound on `FloatVector` plus the special-math traits (`SpecialMath`, `RealSpecialMath`, ...) instead."
        )]
        pub trait ScalarSpecialMathWithPolicy: ElementExt<Element = Self> + FloatElementWithBits {$($(
             $(#[$meta])* fn [<scalar_ $name _p>]<P: Policy, $($generics)*>($($arg_name: $arg_ty),*) -> $ret
                $(where $($where_clause)*)?;
        )*
        $($(
            $(#[$kmeta])* fn [<scalar_ $kname _p>]<P: Policy, K: WrapTo>(kind: K) -> Self
            where K::Wrapped: $ktrait, <K::Wrapped as $ktrait>::Output: Unwrap<Unwrapped = Self>;
        )*)?
        $($(
            $(#[$vmeta])* fn [<scalar_ $vname _p>]<P: Policy, $($vgenerics)*>($($vsarg_name: $vsarg_ty),*) -> $vsret;
        )*)?
        )*}

        #[doc = "Aggregate of all scalar special-math traits using the default policy."]
        #[doc = ""]
        #[doc = "This trait collects every method from the following trait families into a single"]
        #[doc = "trait implemented directly on `f32` and `f64`, using the default policy for all operations:"]
        #[doc = ""]
        $(#[doc = "- [`" [<$trait Math>] "`]"])*
        #[doc = ""]
        #[doc = "All methods are prefixed with `scalar_` to avoid conflicts with inherent methods"]
        #[doc = "on `f32`/`f64`. See [`ScalarSpecialMathWithPolicy`] for the policy-aware variant,"]
        #[doc = "which additionally carries a `_p` suffix on each method."]
        #[doc = ""]
        #[doc = "# Limitations"]
        #[doc = ""]
        #[doc = "This trait is **only** implemented for bare scalar types. Code that is generic over"]
        #[doc = "a `FloatVector` bound will not accept a bare `f32` or `f64` - the scalar must be"]
        #[doc = "wrapped in [`Vector`](thermite::Vector) first (e.g., `Vector::<f32>(x)`) to satisfy"]
        #[doc = "that bound. `ScalarSpecialMath` exists purely as a convenience for call-sites that"]
        #[doc = "already hold a concrete scalar and do not need to be generic."]
        #[doc = ""]
        #[doc = "All types that implement [`ScalarSpecialMathWithPolicy`] automatically implement this trait."]
        #[thermite::dispatch(Self)]
        #[diagnostic::on_unimplemented(
            message = "`{Self}` is not a bare floating-point scalar",
            note = "`ScalarSpecialMath` is implemented only for the bare scalar types `f32` and `f64`. For SIMD vectors, bound on `FloatVector` plus the special-math traits (`SpecialMath`, `RealSpecialMath`, ...) instead."
        )]
        pub trait ScalarSpecialMath: ScalarSpecialMathWithPolicy {$($(
            $(#[$meta])* #[inline(always)] fn [<scalar_ $name>]<$($generics)*>($($arg_name: $arg_ty),*) -> $ret
                $(where $($where_clause)*)?
            { ScalarSpecialMathWithPolicy::[<scalar_ $name _p>]::<DefaultPolicy, $($generic_names),*>($($arg_name),*) }
        )*
        $($(
            $(#[$kmeta])* #[inline(always)] fn [<scalar_ $kname>]<K: WrapTo>(kind: K) -> Self
            where K::Wrapped: $ktrait, <K::Wrapped as $ktrait>::Output: Unwrap<Unwrapped = Self>
            { ScalarSpecialMathWithPolicy::[<scalar_ $kname _p>]::<DefaultPolicy, K>(kind) }
        )*)?
        $($(
            // `<Self as ...>` explicitly, as in the vector layer: a table argument
            // cannot drive `Self` inference.
            $(#[$vmeta])* #[inline(always)] fn [<scalar_ $vname>]<$($vgenerics)*>($($vsarg_name: $vsarg_ty),*) -> $vsret
            { <Self as ScalarSpecialMathWithPolicy>::[<scalar_ $vname _p>]::<DefaultPolicy, $($vgeneric_names),*>($($vsarg_name),*) }
        )*)?
        )*}

        impl<M> ScalarSpecialMath for M where M: ScalarSpecialMathWithPolicy {}

        #[thermite::dispatch(Self)]
        impl<E: ElementExt<Element = Self> + FloatElementWithBits> ScalarSpecialMathWithPolicy for E
        where
            thermite::Vector<E>: Unwrap<Unwrapped = E> +
                FloatVectorWithBits<Element = E,
                    Signed: Unwrap<Unwrapped = <E as Element>::Signed>,
                    Unsigned: Unwrap<Unwrapped = <E as Element>::Unsigned>,
                    SignedBits: Unwrap<Unwrapped = <E as FloatElementWithBits>::SignedBits>,
                    Bits: Unwrap<Unwrapped = <E as FloatElementWithBits>::Bits>
                >
                // Pins `Primal = Self` on the width-1 vector so the primal-typed table
                // parameters normalize to what `Unwrap` produces.
                + PrimalProjection<Primal = thermite::Vector<E>>
                $(+ specialized::[<Specialized $trait Math>]<E>)*,
            E: thermite::register::FloatRegister<Storage = E>,
        {$($(
            $(#[$meta])* #[skip_dispatch] #[inline(always)] fn [<scalar_ $name _p>]<P: Policy, $($generics)*>($($arg_name: $arg_ty),*) -> $ret
                $(where $($where_clause)*)?
            {
                let ($(decl_math!(@SELF $arg_name this),)*) = Unwrap::wrap(($($arg_name,)*));

                let res = <thermite::Vector<E> as specialized::[<Specialized $trait Math>]<E>>::$name::<P, $($generic_names),*>($(decl_math!(@SELF $arg_name this)),*);

                Unwrap::unwrap(res)
            }
        )*
        $($(
            // Kind methods: wrap the scalar request into its width-1 vector form (WrapTo), run the
            // vector-only `eval`, then unwrap the scalar result. The backend stays vector-only.
            $(#[$kmeta])* #[skip_dispatch] #[inline(always)] fn [<scalar_ $kname _p>]<P: Policy, K: WrapTo>(kind: K) -> Self
            where K::Wrapped: $ktrait, <K::Wrapped as $ktrait>::Output: Unwrap<Unwrapped = Self>
            { Unwrap::unwrap(<K::Wrapped as Unwrap>::wrap(kind).eval::<P>()) }
        )*)?
        $($(
            // Scalar-signature methods: same wrap/call/unwrap as the plain fns, with the
            // scalar spelling of the arguments (a scalar is its own primal, so the table
            // wraps into the width-1 vector's primal table directly).
            $(#[$vmeta])* #[skip_dispatch] #[inline(always)] fn [<scalar_ $vname _p>]<P: Policy, $($vgenerics)*>($($vsarg_name: $vsarg_ty),*) -> $vsret
            {
                let ($(decl_math!(@SELF $vsarg_name this),)*) = Unwrap::wrap(($($vsarg_name,)*));

                let res = <thermite::Vector<E> as specialized::[<Specialized $trait Math>]<E>>::$vname::<P, $($vgeneric_names),*>($(decl_math!(@SELF $vsarg_name this)),*);

                Unwrap::unwrap(res)
            }
        )*)?
        )*}
    }};

    // rename `self` to `this`. Requires an existing ident to bind to.
    (@SELF self $rename:ident) => { $rename };
    (@SELF $other:ident $rename:ident) => { $other };
}

decl_math! {
    /// Special math functions that are valid for both real and complex floating-point vectors.
    #[diagnostic::on_unimplemented(
        message = "`{Self}` does not provide special math (`erf`, `gamma`, activations, ...)",
        note = "The special-math traits are auto-implemented for every float vector (any `FloatVector` whose element is `f32`/`f64`) and for composite float types. A bare `f32`/`f64` does not qualify. Wrap it in `Vector::<f32>::splat(x)`, or use `ScalarSpecialMath`'s `scalar_`-prefixed methods.",
        note = "If `{Self}` already is a `FloatVector` and only the method call fails to resolve, bring the trait into scope: `use thermite_special::SpecialMath;` (or the relevant `RealSpecialMath` / `RealPrimalMath`)."
    )]
    trait Special: TranscendentalMathWithPolicy {
        /// Computes the error function.
        ///
        /// For f32 vectors, this is still decently accurate even with the `Medium` and `Worst` precision policies,
        /// thanks to good approximations that don't rely on the precision of `exp`. Subsequently, performance
        /// of the lower precision policies is excellent. Furthermore, if using on a GPU with native `exp` support,
        /// all precision policies will have good performance and accuracy.
        fn erf[][](self: Self) -> Self;

        /// Computes the complementary error function.
        fn erfc[][](self: Self) -> Self;

        /// Computes the scaled complementary error function,
        /// `$\operatorname{erfcx}(x) = e^{x^2}\operatorname{erfc}(x)$`.
        ///
        /// `erfc` underflows to zero at `x ~ 27` in `f64` and `x ~ 9` in `f32`,
        /// where the true value is `$e^{-x^2}/(x\sqrt{\pi})$`, nonzero and merely too small to
        /// represent. Anything reading a Gaussian tail past that point silently gets zero:
        /// importance weights, log-likelihoods, censored-data models, the Voigt profile.
        /// `erfcx` removes the exponential and decays only as `$1/(x\sqrt{\pi})$`, so it is
        /// representable for every finite argument and keeps full relative accuracy.
        ///
        /// Computed on the real backends as the Faddeeva function restricted to the imaginary
        /// axis, `$w(ix) = \operatorname{erfcx}(x)$`, where Weideman's rational approximation
        /// degenerates to real arithmetic: one reciprocal and one Horner, no transcendental at
        /// all for `x >= 0`. That makes it cheaper than the `erfc` it complements, and
        /// measures 1.22 ulp worst over `$x \in [0, 10^{15}]$` at the `Best` tier and above.
        ///
        /// Negative arguments use `$\operatorname{erfcx}(-x) = 2e^{x^2} - \operatorname{erfcx}(x)$`
        /// and legitimately overflow below about `-26.6` (`f64`), the function itself growing
        /// like `$e^{x^2}$` in that direction.
        ///
        /// The two are related by `$\operatorname{erfc}(x) = e^{-x^2}\operatorname{erfcx}(x)$`,
        /// which is the numerically sound way to recover a tail value that `erfc` alone cannot
        /// hold. Keep the `$-x^2$` in the log domain rather than exponentiating it.
        fn erfcx[][](self: Self) -> Self;

        /// Computes the Logistic sigmoid function, defined as `$\sigma(x) = \frac{1}{1 + e^{-x}}$`.
        ///
        /// It's worth mentioning that the derivative of the logistic sigmoid can be computed very cheaply
        /// from the output of the logistic sigmoid itself, in the form of:
        ///
        /// ```rust,ignore
        /// let s = x.logistic_sigmoid();
        /// let derivative = s * (1.0 - s); // or s.nmul_adde(s, s), which may be slightly faster
        /// ```
        ///
        /// Notably, for `f32` and `f64` this implementation still has good precision for the `Worst`
        /// precision policy, and for the `Best` precision policies handles very large positive and negative
        /// inputs without overflow or underflow issues.
        fn logistic_sigmoid[][](self: Self) -> Self;

        /// Computes the logit `$\ln\!\frac{p}{1-p}$`, the inverse of
        /// [`logistic_sigmoid`](SpecialMath::logistic_sigmoid).
        ///
        /// Evaluated as `$\ln(p) - \ln_{1p}(-p)$`, which is accurate for small `p` where the direct
        /// quotient is not. For `p` approaching 1 no evaluation order helps. `$1 - p$` has already
        /// lost its low digits inside the input itself, and the information is not recoverable from
        /// `p`. A caller who knows `$q = 1 - p$` should pass it to
        /// [`logit_1m`](SpecialMath::logit_1m) instead, which is exact at the far end of the range.
        ///
        /// `p = 0` gives `-∞`, `p = 1` gives `+∞`, and `p` outside `[0, 1]` is out of domain.
        fn logit[][](self: Self) -> Self;

        /// Computes `$\mathrm{logit}(1 - q) = \ln\!\frac{1-q}{q}$` from the complement `q` directly.
        ///
        /// The companion entry point to [`logit`](SpecialMath::logit), in the same relationship as
        /// [`langevin_1m`](RealSpecialMath::langevin_1m) has to
        /// [`langevin`](RealSpecialMath::langevin). The logit diverges as its argument approaches 1,
        /// and near that end `$1 - p$` cannot be formed from `p` without losing every digit that
        /// matters. Working in `q` throughout sidesteps that: evaluated as
        /// `$\ln_{1p}(-q) - \ln(q)$`, accurate to a few ulp however small `q` is.
        ///
        /// Note the sign convention follows the substitution, so `logit_1m(q) == -logit(q)` as
        /// functions of the same number. The two differ in _which_ probability the argument names.
        fn logit_1m[][](self: Self) -> Self;

        /// Computes the softplus function, defined as `$\frac{1}{k}\ln(1 + e^{kx})$`.
        ///
        /// This is a smooth approximation to the ReLU function
        /// that is more numerically stable for large inputs.
        ///
        /// The parameter `k` controls the steepness of the curve, with larger values approaching ReLU more closely.
        /// Pass `k = 1` and `rcp_k = 1` for the standard softplus with no steepness scaling.
        ///
        /// `rcp_k` must equal `1/k`. It is passed explicitly so callers that invoke softplus repeatedly
        /// with the same `k` can pre-compute the reciprocal once rather than recomputing it per call.
        ///
        /// To also obtain the derivative with respect to `x`, use
        /// [`softplus_d`](crate::RealPrimalMath::softplus_d).
        fn softplus[][](self: Self, k: Self, rcp_k: Self) -> Self;

        /// Computes the Gamma function (`$\Gamma(z)$`) for any real input, for each value in a vector.
        ///
        /// This implementation uses a few different behaviors to ensure the greatest precision where possible.
        ///
        /// * For non-integer positive inputs, it uses the Lanczos approximation.
        /// * For small non-integer negative inputs, it uses the recursive identity `$\Gamma(z) = \Gamma(z+1)/z$` until `z` is positive.
        /// * For large non-integer negative inputs, it uses the reflection formula `$-\pi / (\Gamma(z)\sin(\pi z)\,z)$`.
        /// * For positive integers, it simply computes the factorial in a tight loop to ensure precision. Lookup tables could not be used with SIMD.
        /// * At zero, the result will be positive or negative infinity based on the input sign (signed zero is a thing).
        ///
        /// **NOTE**: The Gamma function is not defined for negative integers.
        fn tgamma[][](self: Self) -> Self;

        /// Computes the natural log of the Gamma function (`$\ln|\Gamma(x)|$`) for any real input, for each value in a vector.
        fn lgamma[][](self: Self) -> Self;

        /// The Poisson probability mass `$P(k; \lambda) = e^{-\lambda}\lambda^k / k!$` at `k = self`,
        /// for real `$k \ge 0$` and mean `$\lambda \ge 0$`.
        ///
        /// Not `exp(k ln lambda - lambda - lgamma(k+1))`: that forms an `$O(1)$` answer as the
        /// exponential of a difference of large numbers, and half an ulp of
        /// `$\ln\Gamma(k+1) = O(k \ln k)$` becomes that many ulp of the mass. For `$k \ge 9$` this
        /// uses Loader's saddle-point form (the one R's `dpois` uses),
        ///
        /// ```math
        /// P(k; \lambda) = \frac{e^{-\mathrm{stirlerr}(k) - \mathrm{bd0}(k, \lambda)}}{\sqrt{2\pi k}}
        /// ```
        ///
        /// with `stirlerr` the Stirling remainder (a short `$1/k^2$` series) and `bd0` the
        /// deviance `$k \ln(k/\lambda) + \lambda - k$` (a series in `$(k-\lambda)/(k+\lambda)$` near
        /// the peak, where the direct form cancels): both are small where the mass is not
        /// negligible, so the exponential amplifies nothing, and there is no `lgamma` and no
        /// `ln` at all near the peak. Below `$k = 9$` the same machinery is used after shifting
        /// `k` up by an integer, with the exact product `$(k+1)\cdots(k+m)$` taken back out, so
        /// there is no `lgamma` anywhere, and mixed vectors share one `ln`, one `stirlerr` and
        /// one `exp`. Real `k` is allowed because
        /// the Gamma density is the same function: `$f(x; a) = P(a-1; x)$` for shape `$a \ge 1$`
        /// (unit scale).
        ///
        /// Edges: `$\lambda = 0$` gives `1` at `$k = 0$` and `0` above; `$k = 0$` is `$e^{-\lambda}$`.
        fn poisson_pmf[][](self: Self, lambda: Self) -> Self;

        /// `$\ln P(k; \lambda)$`, the log of [`poisson_pmf`](SpecialMath::poisson_pmf), formed
        /// directly (no `exp` then `ln`) so it stays finite far in the tails where the mass
        /// itself underflows.
        fn poisson_log_pmf[][](self: Self, lambda: Self) -> Self;

        /// Computes the digamma function `$\psi(x) = \frac{\mathrm{d}}{\mathrm{d}x}\ln\Gamma(x) = \frac{\Gamma'(x)}{\Gamma(x)}$`
        /// for any real input, for each value in a vector.
        ///
        /// The argument is handled in three regimes:
        ///
        /// * For `x >= 10`, an asymptotic expansion in `$1/x^2$` is used.
        /// * For smaller `x`, the recurrence `$\psi(x) = \psi(x+1) - 1/x$` shifts the argument into
        ///   `[1, 2]`, where a rational minimax approximation `$\psi(x) = (x - x_0)(Y + R(x-1))$` is used
        ///   (`$x_0$` is the positive root of `$\psi$`).
        /// * For `x <= -1`, the reflection formula `$\psi(1-x) = \psi(x) + \pi\cot(\pi x)$` is applied.
        ///
        /// **NOTE**: The digamma function is not defined at zero or the negative integers. Those inputs
        /// yield NaN when overflow checking is enabled.
        fn digamma[][](self: Self) -> Self;

        /// Computes the Beta function `$\mathrm{B}(x, y)$`
        fn beta[][](self: Self, y: Self) -> Self;

        /// Computes `$\ln\left|\mathrm{B}(x, y)\right|$`, the log of the absolute Beta function.
        ///
        /// [`beta`](SpecialMath::beta) itself underflows to zero for quite ordinary arguments
        /// (`$\mathrm{B}(200, 200)$` is about `1e-121`, already gone in f32) and overflows for
        /// arguments straddling the poles. The log form has range to spare in both directions and is
        /// what the surrounding computation usually wants anyway, since Beta almost always appears
        /// inside a product of Gammas that is about to be logged.
        ///
        /// Evaluated as `$\ln\Gamma(x) + \ln\Gamma(y) - \ln\Gamma(x+y)$`. The absolute value follows
        /// [`lgamma`](SpecialMath::lgamma), so recover the sign from
        /// [`lgamma_r`](RealSpecialMath::lgamma_r) if the arguments can be negative.
        ///
        /// This buys range at some cost in relative accuracy. The three `lgamma` terms cancel
        /// against each other, shedding roughly `$\log_{10}\frac{\ln\Gamma(x+y)}{|\ln \mathrm{B}|}$`
        /// digits. That is under one digit at `$x = y = 200$`, and a little over two at
        /// `$x = 200,\ y = 1$` where the terms are near 860 and the answer is near -5.3. It remains
        /// far better conditioned than [`beta`](SpecialMath::beta), which simply has no value to
        /// return across most of that domain.
        fn lbeta[][](self: Self, y: Self) -> Self;

        /// Computes the m-th derivative of the n-th degree Jacobi polynomial
        ///
        /// A the special case where α and β are both zero, the Jacobi polynomial reduces to a
        /// Legendre polynomial.
        ///
        /// **NOTE**: Given constant α, β or `n`, LLVM will happily optimize those away and unroll loops.
        fn jacobi[][](self: Self, alpha: Self, beta: Self, n: u32, m: u32) -> Self;

        /// Computes the N-th degree physicists' [Hermite polynomial](https://en.wikipedia.org/wiki/Hermite_polynomials)
        /// `$H_N(x)$` where `x` is `self` and `N` is the polynomial degree.
        ///
        /// Evaluated by the three-term recurrence
        ///
        /// ```math
        /// H_{n+1}(x) = 2x\,H_n(x) - 2n\,H_{n-1}(x)
        /// ```
        ///
        /// seeded with `$H_0 = 1$` and `$H_1(x) = 2x$`. The trip count is `N`, with no data
        /// dependence, so LLVM unrolls the whole thing into straight-line FMA.
        ///
        /// The derivative is another member of the same family, `$H_n'(x) = 2n\,H_{n-1}(x)$`, so a
        /// value-and-slope pair costs one extra call rather than a separate kernel. The
        /// probabilists' polynomials are a rescaling, `$He_n(x) = 2^{-n/2} H_n(x/\sqrt{2})$`.
        ///
        /// **NOTE**: this is the raw polynomial, which grows fast: `$H_n(0) = (-2)^{n/2} (n-1)!!$` for
        /// even `n`, and `$H_n(x) \sim (2x)^n$` in the tails. It leaves binary32 range at the origin
        /// around degree 48 and binary64 around 300, and much earlier for `|x|` of a few units. If
        /// what you actually want is the *normalized* Hermite function (the quantum harmonic
        /// oscillator eigenstate, a Hermite-Gauss beam mode, or the basis of a Hermite spectral
        /// method), use [`hermite_function`](SpecialMath::hermite_function), which folds the
        /// Gaussian weight and the normalization into the recurrence and stays `$O(1)$` at every
        /// degree. The raw polynomial is the right primitive for Gauss-Hermite quadrature
        /// node-finding at modest `n` and for anything that genuinely wants `$H_n$` itself.
        fn hermite[const N: usize][N](self: Self) -> Self;

        /// Computes the n-th degree physicists' [Hermite polynomial](https://en.wikipedia.org/wiki/Hermite_polynomials)
        /// `H_n(x)` where `x` is `self` and `n` is a vector of unsigned integers representing the polynomial degree.
        ///
        /// The polynomial is calculated independently per-lane with the given degree in `n`.
        ///
        /// This uses the recurrence relation to compute the polynomial iteratively.
        fn hermitev[][](self: Self, n: Self::Unsigned) -> Self;

        /// Computes the orthonormal [Hermite function](https://en.wikipedia.org/wiki/Hermite_polynomials#Hermite_functions)
        ///
        /// ```math
        /// \psi_N(x) = \frac{1}{\sqrt{2^N N! \sqrt{\pi}}}\, e^{-x^2/2}\, H_N(x)
        /// ```
        ///
        /// where `x` is `self`. These are the eigenfunctions of the quantum harmonic oscillator
        /// and of the Fourier transform, the Hermite-Gauss modes of a paraxial beam, and the
        /// basis of Hermite spectral methods. They are orthonormal on the whole line,
        /// `$\int \psi_m \psi_n\, dx = \delta_{mn}$`.
        ///
        /// Evaluated by the recurrence on the functions themselves,
        ///
        /// ```math
        /// \psi_{n+1}(x) = \sqrt{\tfrac{2}{n+1}}\, x\, \psi_n(x) - \sqrt{\tfrac{n}{n+1}}\, \psi_{n-1}(x)
        /// ```
        ///
        /// which keeps every intermediate `$O(1)$` (the polynomial's growth and the Gaussian's
        /// decay cancel inside each step), so unlike [`hermite`](SpecialMath::hermite) it does not
        /// overflow at high degree. Both square roots are literals under the unrolled loop. The
        /// per-step cost is one FMA on the critical path.
        ///
        /// # Range
        ///
        /// The only quantity that can leave the exponent range is the Gaussian seed, which is
        /// carried as `$e^{-x^2/4}$` in two halves to double the reach. Full accuracy at every
        /// degree holds for `$|x|$` under about 18.7 (binary32) or 53 (binary64), which covers
        /// every degree up to about 175 / 1400 everywhere on the line, since past the turning
        /// point `$\sqrt{2n+1}$` the true value decays faster than the seed. Beyond that the result
        /// is still correct wherever `$e^{-x^2/4}$` is representable, and zero past it.
        ///
        /// Under a `Best`-or-better precision policy on true-FMA hardware, the rounding of `$x^2$`
        /// (which is the entire error budget of a Gaussian at large `x`) is recovered exactly and
        /// corrected to first order.
        fn hermite_function[const N: usize][N](self: Self) -> Self;

        /// Evaluates a finite series of Hermite functions at `x = self`:
        ///
        /// ```math
        /// \sum_{k=0}^{N-1} \mathrm{coeffs}[k] \cdot \psi_k(x)
        /// ```
        ///
        /// with `$\psi_k$` as in [`hermite_function`](SpecialMath::hermite_function). Evaluated by
        /// Clenshaw's backward recurrence, which is more stable than summing the functions one at
        /// a time and never forms them individually. `N` is the *length* of the coefficient array,
        /// so the highest function is `$\psi_{N-1}$`; `N = 0` is rejected.
        ///
        /// Same range as [`hermite_function`](SpecialMath::hermite_function): the coefficients are
        /// pre-scaled by half of the Gaussian and the outer factor carries the other half, so the
        /// running Clenshaw values grow no faster than `$e^{x^2/4}$`.
        #[skip_dispatch] fn hermite_function_series_n[const N: usize][N](self: Self, coeffs: &[Self::Element; N]) -> Self;

        /// [`hermite_function_series_n`](SpecialMath::hermite_function_series_n) over a
        /// runtime-length coefficient slice.
        ///
        /// Same recurrence, same pre-scaling, same range. The length is the only difference,
        /// and it costs real work rather than only unrolling: the recurrence coefficients
        /// `$\sqrt{2/(k+1)}$` and `$\sqrt{k/(k+1)}$` fold to literals when `N` is a constant
        /// and become per-step square roots when it is not. Prefer the const form when the
        /// degree is known.
        ///
        /// An empty coefficient slice is `0`, where the const form rejects `N = 0` at compile
        /// time.
        #[skip_dispatch] fn hermite_function_series[][](self: Self, coeffs: &[Self::Element]) -> Self;

        /// Computes the generalized (associated) [Laguerre polynomial](https://en.wikipedia.org/wiki/Laguerre_polynomials)
        /// `$L_N^{(\alpha)}(x)$`, where `x` is `self` and `N` is the polynomial degree.
        ///
        /// Passing `alpha = Self::ZERO` gives the ordinary Laguerre polynomial `$L_N(x)$`; because
        /// `alpha` is an ordinary argument rather than a const generic, that case folds away
        /// completely when the zero is visible at the call site.
        ///
        /// Evaluated by the three-term recurrence
        ///
        /// ```math
        /// (n+1)\,L_{n+1}^{(\alpha)}(x) = (2n + \alpha + 1 - x)\,L_n^{(\alpha)}(x) - (n + \alpha)\,L_{n-1}^{(\alpha)}(x)
        /// ```
        ///
        /// seeded with `$L_0^{(\alpha)} = 1$` and `$L_1^{(\alpha)}(x) = 1 + \alpha - x$`. The trip count
        /// is `N`, with no data dependence, so LLVM unrolls the whole thing into straight-line FMA.
        ///
        /// The derivative is another member of the same family,
        /// `$\frac{\mathrm{d}}{\mathrm{d}x} L_n^{(\alpha)}(x) = -L_{n-1}^{(\alpha+1)}(x)$`, so a
        /// value-and-slope pair costs one extra call rather than a separate kernel.
        ///
        /// **NOTE**: the forward recurrence is the standard evaluation route (Boost and GSL both use
        /// it) and is well behaved across the oscillatory region `$0 \le x \lesssim 4n$`. Past that
        /// `$L_n^{(\alpha)}$` itself grows like `$(-x)^n/n!$` and will overflow for large `N` and `x`
        /// on its own account.
        ///
        /// Laguerre-Gaussian beam modes, the radial part of the hydrogen wavefunction, the quantum
        /// harmonic oscillator and coherent-state expansions, and Gauss-Laguerre quadrature.
        fn laguerre[const N: usize][N](self: Self, alpha: Self) -> Self;

        /// Computes the generalized (associated) [Laguerre polynomial](https://en.wikipedia.org/wiki/Laguerre_polynomials)
        /// `$L_n^{(\alpha)}(x)$` where `n` is a vector of unsigned integers giving the degree per lane.
        ///
        /// The per-lane counterpart of [`laguerre`](SpecialMath::laguerre), in the same relation to it
        /// as [`hermitev`](SpecialMath::hermitev) is to [`hermite`](SpecialMath::hermite). The
        /// recurrence runs to the largest `n` in the vector and lanes freeze at their own degree, so
        /// the cost is set by `max(n)` rather than by any one lane.
        fn laguerrev[][](self: Self, alpha: Self, n: Self::Unsigned) -> Self;

        /// Computes the orthonormal generalized [Laguerre function](https://en.wikipedia.org/wiki/Laguerre_polynomials#Generalized_Laguerre_polynomials)
        ///
        /// ```math
        /// l_N^{(\alpha)}(x) = \sqrt{\frac{N!}{\Gamma(N+\alpha+1)}}\; x^{\alpha/2} e^{-x/2}\, L_N^{(\alpha)}(x)
        /// ```
        ///
        /// where `x` is `self`. Orthonormal on the half-line, `$\int_0^\infty l_m l_n\, dx = \delta_{mn}$`.
        /// This is the radial factor of Laguerre-Gauss beam modes and (up to a power of `x` from the
        /// spherical measure) of the hydrogen wavefunctions. Defined for `$x \ge 0$` and
        /// `$\alpha > -1$`, and nothing is checked outside that.
        ///
        /// Evaluated by the recurrence on the functions themselves, with
        /// `$s_k = \sqrt{(k+1)(k+\alpha+1)}$`:
        ///
        /// ```math
        /// l_{k+1} = \frac{(2k + \alpha + 1 - x)\, l_k - s_{k-1}\, l_{k-1}}{s_k}
        /// ```
        ///
        /// which keeps every intermediate `$O(1)$`, so unlike [`laguerre`](SpecialMath::laguerre)
        /// it does not overflow at high degree or large `x`. `alpha` is a runtime vector, so each
        /// step also carries a `sqrt` and a reciprocal, beside the recurrence rather than on its
        /// critical path, and folded to literals when `alpha` is a visible constant. The seed
        /// is skipped outright by a uniform branch when every lane has `alpha = 0`, which is the
        /// ordinary Laguerre function and by far the common case.
        ///
        /// # Range
        ///
        /// The Gaussian-like seed `$x^{\alpha/2} e^{-x/2}$` is carried as `$e^{-x/4}$` in two
        /// halves, as in [`hermite_function`](SpecialMath::hermite_function). Full accuracy at
        /// every degree for `x` under about 350 (binary32) or 2800 (binary64), covering every
        /// degree up to roughly 87 / 700 everywhere on the half-line (the turning point of
        /// `$l_n^{(\alpha)}$` is near `4n`).
        ///
        /// `alpha` is unrestricted over the same `x` range. The seed's whole parameter
        /// dependence, `$x^{\alpha/2}/\sqrt{\Gamma(\alpha+1)}$`, is the square root of the Poisson
        /// mass `$P(\alpha; x)$` and is evaluated as [`poisson_pmf`](SpecialMath::poisson_pmf)
        /// is (Loader's saddle-point form, one exponential of a small exponent), so neither
        /// factor materializes (separately `$x^{\alpha/2}$` overflows binary64 near
        /// `$\alpha = 250$` and `$1/\sqrt{\Gamma(\alpha+1)}$` underflows near `$\alpha = 320$`,
        /// and their overlap would be `inf * 0`) and nothing large is exponentiated: 0-3 ulp
        /// at the peak `x ~ alpha` out to `$\alpha = 1400$`, against a 50-digit oracle.
        fn laguerre_function[const N: usize][N](self: Self, alpha: Self) -> Self;

        /// [`laguerre_function`](SpecialMath::laguerre_function) at an integer weight, taken as a
        /// **scalar** `i32` rather than a vector.
        ///
        /// Same function and same range. What changes is what the compiler can see. Every
        /// quantity the recurrence derives from the weight (the `$s_k = \sqrt{(k+1)(k+\alpha+1)}$`
        /// and their reciprocals, and the `$2k+\alpha+1$` offsets) becomes a scalar constant
        /// instead of a vector `sqrt` and reciprocal per step, and folds to a literal outright
        /// when `alpha` is compile-time known.
        ///
        /// The seed changes too. Up to `$\alpha = 170$` (binary64) / `29` (binary32) the
        /// normalization `$x^{\alpha/2}/\sqrt{\alpha!}$` is a scalar factorial, a `powi` and at
        /// most one `sqrt`, with no `ln`, `lgamma` or second `exp` at all, and a few ulp *more*
        /// accurate than the log form, whose `lgamma` error is amplified by the exponential.
        /// `$\alpha = 0$` is a scalar test that skips even that. Beyond the cap it takes
        /// the vector form's saddle-point seed. Measured on AVX2 f64x4 at degree 4:
        /// about 5x faster than the vector form at a literal small weight, 2x at a runtime one.
        ///
        /// Prefer this whenever the weight is a non-negative integer, which every classical
        /// application has: the hydrogen radial functions use `$\alpha = 2\ell+1$` and the
        /// Laguerre-Gauss beam modes use `$\alpha = |\ell|$`. Negative values are out of domain,
        /// as `$\alpha \le -1$` is for the general form.
        ///
        /// Like the series forms this is inlined into the caller rather than given its own
        /// dispatch trampoline: the weight is a plain `i32` argument, and a shared
        /// out-of-line copy would take it at runtime, which both defeats the folding above
        /// and (measured) stops LLVM overlapping consecutive evaluations, at 7x the cost.
        /// Call it from inside a `#[thermite::dispatch]` body.
        #[skip_dispatch] fn laguerre_function_i[const N: usize][N](self: Self, alpha: i32) -> Self;

        /// Evaluates a finite series of generalized Laguerre functions at `x = self`:
        ///
        /// ```math
        /// \sum_{k=0}^{N-1} \mathrm{coeffs}[k] \cdot l_k^{(\alpha)}(x)
        /// ```
        ///
        /// with `$l_k^{(\alpha)}$` as in [`laguerre_function`](SpecialMath::laguerre_function).
        /// Clenshaw's backward recurrence, same range as the single function; `N` is the
        /// coefficient count and `N = 0` is rejected.
        #[skip_dispatch] fn laguerre_function_series_n[const N: usize][N](self: Self, alpha: Self, coeffs: &[Self::Element; N]) -> Self;

        /// [`laguerre_function_series_n`](SpecialMath::laguerre_function_series_n) over a
        /// runtime-length coefficient slice.
        ///
        /// Same recurrence, same pre-scaling, same range. The per-step weights are computed
        /// rather than folded, as in
        /// [`hermite_function_series`](SpecialMath::hermite_function_series). An empty
        /// coefficient slice is `0`.
        #[skip_dispatch] fn laguerre_function_series[][](self: Self, alpha: Self, coeffs: &[Self::Element]) -> Self;

        /// [`laguerre_function_series`](SpecialMath::laguerre_function_series) at a scalar integer
        /// weight, in the same relation to it as
        /// [`laguerre_function_i`](SpecialMath::laguerre_function_i) is to
        /// [`laguerre_function`](SpecialMath::laguerre_function). See there for what the integer
        /// form buys.
        #[skip_dispatch] fn laguerre_function_series_i_n[const N: usize][N](self: Self, alpha: i32, coeffs: &[Self::Element; N]) -> Self;

        /// [`laguerre_function_series_i_n`](SpecialMath::laguerre_function_series_i_n) over a
        /// runtime-length coefficient slice.
        ///
        /// The `_n` is the coefficient count and the `_i` is the integer weight, in that
        /// order because the length is the newer axis, and both mean what they do everywhere else.
        /// An empty coefficient slice is `0`.
        #[skip_dispatch] fn laguerre_function_series_i[][](self: Self, alpha: i32, coeffs: &[Self::Element]) -> Self;

        /// Evaluates a finite series of [Chebyshev polynomials](https://en.wikipedia.org/wiki/Chebyshev_polynomials)
        /// of the `K`-th kind at `x = self`:
        ///
        /// ```math
        /// \sum_{k=0}^{N-1} \mathrm{coeffs}[k] \cdot P_k(x)
        /// ```
        ///
        /// where `P_k` is `T_k`, `U_k`, `V_k`, or `W_k` depending on `K`. All four kinds share the
        /// recurrence `$P_{k+1}(x) = 2x \cdot P_k(x) - P_{k-1}(x)$` with `P_0(x) = 1`, and differ only in
        /// `P_1(x)`:
        ///
        /// | `K` | Kind   | `P_1(x)`   | Notes |
        /// |-----|--------|------------|-------|
        /// | `1` | First  (`T_k`) | `x`        | Most common, the minimax/approximation basis on `[-1, 1]`. |
        /// | `2` | Second (`U_k`) | `2x`       | Related to `$\sin((k+1)\theta)/\sin(\theta)$` under `$x = \cos\theta$`. |
        /// | `3` | Third  (`V_k`) | `2x - 1`   | "Airfoil" polynomials; `$\cos((k+\tfrac12)\theta)/\cos(\theta/2)$`. |
        /// | `4` | Fourth (`W_k`) | `2x + 1`   | `$\sin((k+\tfrac12)\theta)/\sin(\theta/2)$`. |
        ///
        /// Any other value of `K` is a compile-time error.
        ///
        /// There is deliberately no single-polynomial `T_n(x)` entry point beside this, unlike
        /// [`legendre`](SpecialMath::legendre) or [`hermite`](SpecialMath::hermite). Chebyshev
        /// polynomials are used almost exclusively as an approximation basis, i.e. as a series;
        /// their quadrature nodes and weights are closed-form, so nothing needs to iterate on a
        /// lone `$T_n$`; and the one genuine single-`$T_n$` application (Chebyshev filter response,
        /// Dolph-Chebyshev windows) needs `$|x| > 1$`, where the right evaluation is
        /// `$\cosh(n \cosh^{-1} x)$` and not this recurrence at all. A unit coefficient array
        /// recovers `$T_n$` if it is ever wanted.
        ///
        /// Evaluation is done via Clenshaw's backward recurrence with FMA, which is
        /// more numerically stable than a forward sum when the partial sums of
        /// `$\sum c_k P_k$` are much smaller than `$\max_k |c_k P_k|$` (e.g. fitted minimax series
        /// with alternating-sign coefficients). `N` is the *length* of the coefficient
        /// slice, so the highest polynomial term is `P_{N-1}`; `N = 0` is rejected,
        /// `N = 1` evaluates to `coeffs[0]`.
        ///
        /// `coeffs[0]` multiplies `P_0 = 1`, `coeffs[1]` multiplies `P_1(x)` (which depends on `K`),
        /// and so on. Because LLVM sees both `K` and `N` as constants, the recurrence loop and the
        /// `P_1` selection are fully unrolled and specialized at monomorphization time.
        ///
        /// # Accuracy near `$x = \pm 1$`
        ///
        /// The plain recurrence forms `$2x b_{k+1} - b_{k+2}$` with consecutive `$b_k$` of nearly
        /// equal magnitude as `x` approaches either endpoint, and cancels. This is a property of
        /// the *recurrence*, not of the series: measured against a 60-digit oracle at `N = 24`,
        /// it costs up to 37 ulp on sums whose own condition number is about 1, and up to 230 ulp
        /// on unstructured coefficients.
        ///
        /// Under a `Best`-or-better precision policy, real vectors instead take Reinsch's
        /// modification, which recurs on the differences (near `+1`) or sums (near `-1`) so the
        /// small quantity is never formed by subtraction. On the same grid that bounds the error
        /// envelope 2.5x to 17x tighter across all four kinds. It is an envelope improvement
        /// rather than a pointwise one (individual arguments can land worse), and costs
        /// roughly 2x on the recurrence's dependency chain, which is why it is gated.
        ///
        /// binary32 gains the same way, 2.6x to 13.5x on its own grid. Measuring it needs an
        /// f32-native one: `1 - 2^-j` rounds to exactly `1.0` for every `j >= 24`, so an f64
        /// grid piles two thirds of its points onto the endpoint itself, where the endpoint
        /// form degenerates into a plain running sum and the two policies agree, and never
        /// samples the f32 neighbourhood where the cancellation actually bites.
        ///
        /// Coefficients from a minimax or least-squares *fit* decay geometrically and barely
        /// notice either way (about 3 ulp to 1). The gap opens on slowly-decaying or
        /// non-decaying spectra: truncated expansions, near-singular functions, or coefficients
        /// that came from somewhere other than a fit.
        ///
        /// `Complex` and the composite arithmetics keep the plain recurrence at every policy,
        /// since Reinsch needs a real `copysign` and a meaningful nearest endpoint.
        #[skip_dispatch] fn chebyshev_n[const K: usize, const N: usize][K, N](self: Self, coeffs: &[Self::Element; N]) -> Self;

        /// [`chebyshev_n`](SpecialMath::chebyshev_n) over a runtime-length coefficient slice.
        ///
        /// `K` stays a const generic, since it selects *which* Chebyshev kind, not how many
        /// coefficients, and there are exactly four. Only the length becomes dynamic.
        ///
        /// Same recurrence and the same `Best`-precision Reinsch form near `$x = \pm 1$`; what
        /// the runtime length costs is the unrolling and the folded `coeffs` indices. An empty
        /// coefficient slice is `0`.
        #[skip_dispatch] fn chebyshev[const K: usize][K](self: Self, coeffs: &[Self::Element]) -> Self;

        /// Computes the Gaussian function with amplitude `a` and standard deviation `c`, defined as `$a\, e^{-\frac{1}{2}(x/c)^2}$`.
        ///
        /// The position `b` is assumed to be zero. For a non-zero position, use `self - b` as the input.
        fn gaussian[][](self: Self, a: Self, c: Self) -> Self;

        /// Computes the Planck shape factor `$\frac{x^3}{e^x - 1}$`, finite at `x = 0` where it
        /// vanishes like `$x^2$`.
        ///
        /// The dimensionless kernel of Planck's law: substituting `$x = h\nu/kT$` recovers the
        /// spectral radiance up to a scale factor, so this is the part worth computing carefully and
        /// the constants are left to the caller. Radiative transfer, climate radiation budgets, and
        /// stellar atmospheres.
        ///
        /// The denominator cancels for small `x` and the quotient is `$0/0$` at the origin.
        /// Evaluated here as `$x^2/\varphi_1(x)$` using
        /// `phi::<1>`, which is finite and equal to 1 there,
        /// so the singularity never forms rather than being patched after the fact.
        fn planck[][](self: Self) -> Self;

        /// Computes the m-th associated n-th degree Legendre polynomial,
        /// where m=0 signifies the regular n-th degree Legendre polynomial.
        ///
        /// If `m` is odd, the input is only valid between -1 and 1
        ///
        /// **NOTE**: Given constant `n` and/or `m`, LLVM will happily unroll and optimize inner loops.
        ///
        /// Internally, this is computed with [`jacobi`](SpecialMath::jacobi) when m > 0.
        fn legendre[][](self: Self, n: u32, m: u32) -> Self;

        /// Evaluates a finite [Legendre series](https://en.wikipedia.org/wiki/Legendre_polynomials)
        /// at `x = self`:
        ///
        /// ```math
        /// \sum_{k=0}^{N-1} \mathrm{coeffs}[k] \cdot P_k(x)
        /// ```
        ///
        /// The form a Legendre-moment expansion takes: Mie and Henyey-Greenstein scattering
        /// phase functions tabulated by their moments, multipole expansions in `$\cos\theta$`, and
        /// the polar factor of a spherical-harmonic expansion at fixed order.
        ///
        /// Evaluated by Clenshaw's backward recurrence on the Legendre three-term relation, which
        /// is more stable than building each `$P_k$` with [`legendre`](SpecialMath::legendre) and
        /// summing, and does `$O(N)$` work rather than `$O(N^2)$`. The recurrence ratios
        /// `$(2k+1)/(k+1)$` and `$k/(k+1)$` are literals under the unrolled loop, so the per-step
        /// cost matches [`chebyshev`](SpecialMath::chebyshev): one FMA on the critical path. `N`
        /// is the coefficient count; `N = 0` is rejected, `N = 1` evaluates to `coeffs[0]`.
        ///
        /// Plain Clenshaw at every policy: the endpoint cancellation that `chebyshev` treats
        /// under `Best` precision exists here too (`$P_n(1) = 1$` for every `n`), but its
        /// Reinsch-style rewrite for the Legendre ratios has not been derived or measured.
        #[skip_dispatch] fn legendre_series_n[const N: usize][N](self: Self, coeffs: &[Self::Element; N]) -> Self;

        /// [`legendre_series_n`](SpecialMath::legendre_series_n) over a runtime-length
        /// coefficient slice.
        ///
        /// Plain Clenshaw here too. The recurrence ratios `$(2k+1)/(k+1)$` and `$k/(k+1)$` are
        /// literals only when `N` is a constant, so this pays a division per step where the
        /// const form pays none, the widest const-versus-slice gap of the series family.
        /// An empty coefficient slice is `0`.
        #[skip_dispatch] fn legendre_series[][](self: Self, coeffs: &[Self::Element]) -> Self;

        /// Computes the [Zernike](https://en.wikipedia.org/wiki/Zernike_polynomials) radial
        /// polynomial `$R_n^m(\rho)$`, where `rho` is `self`.
        ///
        /// Returns zero unless `$m \le n$` with `$n - m$` even, the condition for the mode to
        /// exist. `m` is the *absolute* azimuthal frequency here. The sign only affects the
        /// angular factor, which lives in [`zernike`](SpecialMath::zernike).
        ///
        /// Evaluated through the shifted Jacobi identity
        ///
        /// ```math
        /// R_n^m(\rho) = \rho^m\, P_{(n-m)/2}^{(0,\,m)}\!\left(2\rho^2 - 1\right)
        /// ```
        ///
        /// rather than the textbook sum
        /// `$\sum_k (-1)^k \frac{(n-k)!}{k!\,((n+m)/2 - k)!\,((n-m)/2 - k)!} \rho^{n-2k}$`, which
        /// alternates factorials of size `$(n-k)!$` against an answer bounded by 1 and loses all
        /// precision somewhere around `n = 10-15`. That is well inside the range adaptive optics,
        /// ophthalmology and surface metrology actually use.
        ///
        /// The `$(-1)^{(n-m)/2}$` prefactor usually seen with this identity is absent because the
        /// argument is written `$2\rho^2 - 1$` rather than `$1 - 2\rho^2$`: reflecting a Jacobi
        /// polynomial swaps its two parameters and absorbs exactly that sign.
        ///
        /// The polynomial is only orthogonal on `$\rho \in [0, 1]$` and grows quickly outside it.
        /// Nothing clamps the argument, so an unnormalized pupil coordinate stays the caller's
        /// problem.
        fn zernike_r[][](self: Self, n: u32, m: u32) -> Self;

        /// Computes the Zernike polynomial `$Z_n^m(\rho, \theta)$` on the unit disc, with `rho`
        /// as `self`:
        ///
        /// ```math
        /// Z_n^m(\rho, \theta) = N_n^m\, R_n^{|m|}(\rho) \times
        ///   \begin{cases} \cos(m\theta) & m \ge 0 \\ \sin(|m|\theta) & m < 0 \end{cases}
        /// ```
        ///
        /// Returns zero unless `$|m| \le n$` with `$n - |m|$` even.
        ///
        /// `NORM` selects the normalization `$N_n^m$` and must be either
        /// [`ZERNIKE_UNIT_PEAK`] (`$N = 1$`, so `$R_n^m(1) = 1$` and coefficients read as peak
        /// amplitude) or [`ZERNIKE_ORTHONORMAL`]
        /// (`$N_n^m = \sqrt{2(n+1)/(1 + \delta_{m,0})}$`, the ANSI Z80.28 and Noll convention,
        /// under which coefficients read as RMS contributions). Any other value is a compile-time
        /// error. There is deliberately no default: the two differ by a factor of up to
        /// `$\sqrt{2(n+1)}$` per mode, and picking one silently is how coefficient sets get
        /// misinterpreted.
        ///
        /// `(n, m)` is a runtime pair rather than a const generic on purpose. The workload is a
        /// basis, not a function. A wavefront fit evaluates tens to hundreds of modes over
        /// thousands of pupil samples, with the mode list coming from a config or a sensor
        /// geometry, so the degree is loop-invariant across the vector axis and const-generic
        /// specialization would buy a jump table rather than an unrolled loop.
        ///
        /// The single-index conventions (ANSI Z80.28 / OSA, Noll, Fringe) and the conversions
        /// between them are in [`crate::zernike`]. They disagree from the second term
        /// onward, so convert at the boundary rather than assuming.
        fn zernike[const NORM: u8][NORM](self: Self, theta: Self, n: u32, m: i32) -> Self;

        /// Evaluates **all** Zernike modes through degree `L` at the Cartesian pupil point
        /// `(x, y)`, into `out[j]` for the ANSI Z80.28 / OSA index `$j = (n(n+2) + m)/2$`.
        ///
        /// `N` must equal `(L+1)(L+2)/2` (compile-time checked), and `NORM` is
        /// [`ZERNIKE_UNIT_PEAK`] or [`ZERNIKE_ORTHONORMAL`] as on
        /// [`zernike`](SpecialMath::zernike).
        ///
        /// This is the entry point a wavefront fit or reconstruction wants. It is not merely
        /// a loop over [`zernike`](SpecialMath::zernike). Substituting `$s = x^2+y^2$`
        /// splits every mode into a polynomial in `s` times `$\operatorname{Re}$` or
        /// `$\operatorname{Im}$` of `$(x+iy)^{|m|}$`, which is where the `$\rho^{|m|}$` and the
        /// `$\cos m\theta$` both come from at once. Evaluation is then **pure polynomial
        /// arithmetic**: no `atan2`, no `sqrt`, no trigonometry, no division, `$O(L^2)$` FMAs
        /// for the entire basis, and no singularity at the pupil centre. Calling the
        /// single-mode form per mode instead costs a `sin_cos` and a `powi` each and restarts
        /// the radial recurrence every time, for `$O(L^3)$` work.
        ///
        /// Cartesian input is part of that, not a convenience: pupil samples arrive as
        /// `(x, y)`, and a polar entry point would charge an `atan2` per sample for an angle
        /// this kernel immediately dissolves.
        ///
        /// Fully unrolled at compile time for each `L` up to
        /// [`MAX_ZERNIKE_DEGREE`](specialized::MAX_ZERNIKE_DEGREE); above that it takes a
        /// rolled path that is correct at any degree and substantially slower.
        ///
        /// Nothing normalizes `(x, y)` onto the unit disc. Outside it the polynomials still
        /// evaluate correctly and simply are not orthogonal.
        ///
        /// The layout is ANSI because it is the scheme whose index has a closed form *and*
        /// whose degree truncation is contiguous. Noll and Fringe callers gather through
        /// [`noll_to_ansi`](crate::zernike::noll_to_ansi) /
        /// [`fringe_to_ansi`](crate::zernike::fringe_to_ansi).
        ///
        /// ```
        /// use thermite::prelude::*;
        /// use thermite_special::{SpecialMath, ZERNIKE_ORTHONORMAL};
        /// use thermite_special::zernike::noll_to_ansi;
        ///
        /// type V = Vector<f64>;
        /// const L: usize = 4;
        /// const N: usize = 15; // (L+1)(L+2)/2
        ///
        /// let mut basis = [V::ZERO; N];
        /// V::zernike_basis::<L, ZERNIKE_ORTHONORMAL, N>(V::splat(0.3), V::splat(0.4), &mut basis);
        ///
        /// // Noll 4 is defocus, Z_2^0 = sqrt(3) (2 rho^2 - 1) orthonormal.
        /// let defocus = basis[noll_to_ansi(4) as usize].extract::<0>();
        /// assert!((defocus - 3f64.sqrt() * (2.0 * 0.25 - 1.0)).abs() < 1e-14);
        /// ```
        #[skip_dispatch] fn zernike_basis[const L: usize, const NORM: u8, const N: usize][L, NORM, N](x: Self, y: Self, out: &mut [Self; N]) -> ();

        /// Computes both branches of the Lambert W function simultaneously: (`$W_0(x)$`, `$W_{-1}(x)$`).
        ///
        /// The `$W_0$` result is valid for `x >= -1/e`; the `$W_{-1}$` result is valid for `-1/e <= x < 0`.
        /// Outside these domains, the respective result is NaN (when overflow checking is enabled).
        fn lambert_w[][](self: Self) -> (Self, Self);

        // TEMP(bessel_j): disabled until orders beyond J_0 exist. Only f32 `J_0` was
        // ever implemented, so every composite type (Dual, Complex, Compensated) could
        // do nothing but `todo!()`. Re-enable this line and the ones marked
        // TEMP(bessel_j) elsewhere together.
        //fn bessel_j[const N: usize][N](self: Self) -> Self;

        /// Computes the generalized exponential integral `E_n(x)` for integer order `n`.
        fn expint[const N: usize][N](self: Self) -> Self;

        /// Returns `$\varphi_N(x)$`, the `N`-th phi-function of exponential integrators.
        ///
        /// ```math
        /// \varphi_0(x) = e^x, \qquad
        /// \varphi_{k+1}(x) = \frac{\varphi_k(x) - 1/k!}{x}, \qquad
        /// \varphi_k(x) = \sum_{n \ge 0} \frac{x^n}{(n + k)!}, \qquad
        /// \varphi_k(0) = \frac{1}{k!}
        /// ```
        ///
        /// `phi::<0>` is `exp`. `phi::<1>` is `$(e^x - 1)/x$`, which written out
        /// directly is `$0/0$` at the origin and loses most of the mantissa near it, so it is
        /// evaluated as `$\mathrm{expm1}(x)/x$` with the removable singularity filled in (the
        /// value is 1), which is accurate across the whole line. Beyond that the recurrence is
        /// the wrong way to compute them: each step subtracts `1/k!` from a value that is barely
        /// larger while `|x|` is small, so `$\varphi_2 = (\mathrm{expm1}(x) - x)/x^2$` loses twice the bits
        /// `phi::<1>` would have, and gets worse with `N`. Below `|x| = N` this sums the series
        /// instead (its terms are monotone there, so nothing cancels), and above it runs the
        /// recurrence upward from `expm1`, where the amplification per step is bounded. Measured
        /// against mpmath, both arms sit within a few ulp for `N <= 8`.
        ///
        /// The series arm's length is bounded by the policy's `max_iterations`. The primitive
        /// float types know their precision statically and use a fixed count instead. Nothing
        /// caps `N`, though nothing needs it large: ETDRK4 wants `phi_1..phi_3`, and exponential
        /// Rosenbrock methods rarely go past `phi_4`.
        ///
        /// `phi::<1>` alone is the coefficient that keeps appearing wherever an exponential is
        /// integrated over a finite step:
        ///
        /// * The in-scattering integral through a homogeneous medium,
        ///   `$\int_0^t e^{-\sigma s}\,ds = t\,\varphi_1(-\sigma t)$`. The singular case is the empty
        ///   medium, which is not an edge case in practice.
        /// * Exact stepping of an Ornstein-Uhlenbeck process, and the Langevin thermostat's
        ///   mean-reversion factor.
        /// * Frame-rate-independent exponential smoothing, usually written `1 - exp(-k * dt)` and then
        ///   divided by `k`.
        ///
        /// The higher orders are the coefficients of exponential time differencing: integrating
        /// `y' = Ly + N(y)` exactly over a step gives `$y(h) = e^{hL} y_0 + h\,\varphi_1(hL)\,N$`, and
        /// expanding `N` in time along the step brings in `$\varphi_2, \varphi_3, \ldots$` as the
        /// weights of the higher-order terms.
        fn phi[const N: usize][N](self: Self) -> Self;

        @kinds {
            /// Carlson symmetric elliptic integral, selected by a [`CarlsonKind`] request struct
            /// with named fields. The arity (and which argument is the parameter / repeated one)
            /// is fixed per kind, so the wrong shape is a compile error.
            ///
            /// ```rust,ignore
            /// let rf = V::carlson(CarlsonRf { x, y, z });
            /// let rj = V::carlson_p::<Precision, _>(CarlsonRj { x, y, z, p });
            /// ```
            fn carlson: CarlsonKind;

            /// Legendre elliptic integral, selected by an [`EllipticKind`] request struct. Each
            /// form ([`EllintK`](elliptic::EllintK)/[`EllintF`](elliptic::EllintF)/[`EllintE`](elliptic::EllintE)/
            /// [`EllintEInc`](elliptic::EllintEInc)/[`EllintD`](elliptic::EllintD)/[`EllintDInc`](elliptic::EllintDInc)/
            /// [`EllintPi`](elliptic::EllintPi)/[`EllintPiInc`](elliptic::EllintPiInc)) carries exactly
            /// its own arguments, and completeness is encoded by whether the struct has a `phi` field.
            ///
            /// ```rust,ignore
            /// let k_int = V::ellint(EllintK { k });                       // K(k)
            /// let e_inc = V::ellint_p::<Precision, _>(EllintEInc { phi, k }); // E(phi, k)
            /// ```
            fn ellint: EllipticKind;
        }
    }

    /// Special math functions that are only defined for real-valued floating-point vectors.
    ///
    /// These functions either rely on ordering/sign information that has no complex analogue
    /// (e.g. `erfinv`, `probit`, `lgamma_r`), or use the real absolute value in a way that
    /// makes them non-holomorphic (e.g. `algebraic_sigmoid`).
    #[diagnostic::on_unimplemented(
        message = "`{Self}` does not provide real-valued special math (`erfinv`, `probit`, `lgamma_r`, ...)",
        note = "`RealSpecialMath` is only meaningful for real-valued float vectors. Complex number types deliberately do not implement it. A bare `f32`/`f64` does not qualify either. Wrap it in `Vector::<f32>::splat(x)`, or use `ScalarSpecialMath`."
    )]
    trait RealSpecial: SpecialMathWithPolicy {
        /// Computes the inverse error function.
        fn erfinv[][](self: Self) -> Self;

        /// Computes the Probit function, the inverse of the cumulative distribution function
        /// of the standard normal distribution.
        fn probit[][](self: Self) -> Self;

        /// Computes the Langevin function `$L(x) = \coth x - \frac{1}{x}$`.
        ///
        /// Odd, strictly increasing, `L(0) = 0`, `L'(0) = 1/3`, `L(x) -> 1` as `x -> ∞`.
        /// This is the mean resultant length `$A_3(\kappa)$` of a von Mises-Fisher
        /// distribution on the sphere, and the freely-jointed-chain force-extension law
        /// in polymer physics.
        ///
        /// Evaluated as an odd minimax polynomial for `|x| <= 2` (the direct form
        /// `coth x - 1/x` cancels catastrophically there, losing `3u/x^2`), and as
        /// `1 - 1/x + 2/(e^{2x} - 1)` beyond. Both branches are accurate to a few ulp
        /// at every precision policy. The policy mainly selects the `exp`.
        ///
        /// To also obtain the derivative `L'(x)`, use
        /// [`langevin_d`](crate::RealPrimalMath::langevin_d).
        fn langevin[][](self: Self) -> Self;

        /// Computes the inverse Langevin function `$L^{-1}(y)$` for `|y| < 1`.
        ///
        /// Odd, with a simple pole at `y = 1`: `L^-1(y) ~ 1/(1-y)`. `|y| = 1` returns
        /// `±∞`, and `|y| > 1` returns NaN under overflow checking (an unspecified
        /// value otherwise). Its condition number is `1/(1-y)`, so near the pole the
        /// result cannot be more accurate than that, however exact the arithmetic. A
        /// consumer that knows `1 - y` should form it before rounding.
        ///
        /// A rational seed (the same family as Cohen's Pade approximant, which the vMF
        /// literature knows as the Banerjee et al. concentration estimator) is refined by
        /// Newton (f32) or Halley (f64) steps whose count follows the precision policy:
        ///
        /// | precision | steps | relative error |
        /// |---|---|---|
        /// | `Worst` | 0 | ~2e-5 |
        /// | `Medium`, `Average`, `Best` | 1 | full (a few ulp) |
        /// | `Reference` | 2 | full |
        fn inv_langevin[][](self: Self) -> Self;

        /// Computes `1 - L(x)`, the complement of the [Langevin function](RealSpecialMath::langevin),
        /// accurately where `L(x)` is within rounding of 1.
        ///
        /// `1 - L(x) ~ 1/x`, so once `x > 1/u` (sharpness ~1e7 in f32, ~1e16 in f64)
        /// `langevin(x)` rounds to exactly 1 and its complement is gone. This returns it
        /// to full relative precision at any `x`, from the same intermediates. Same cost
        /// as `langevin`. Negative `x` gives `1 + L(|x|)`.
        ///
        /// Pairs with [`inv_langevin_1m`](RealSpecialMath::inv_langevin_1m): the vMF
        /// convolution `kappa' = L^-1(L(k1) L(k2))` should be formed as
        /// `inv_langevin_1m(a + b - a*b)` with `a = langevin_1m(k1)`, `b = langevin_1m(k2)`,
        /// which is cancellation-free at every sharpness.
        fn langevin_1m[][](self: Self) -> Self;

        /// Computes `L^-1(1 - t)` from the complement `t` directly.
        ///
        /// The [inverse Langevin function](RealSpecialMath::inv_langevin) has a pole at
        /// `y = 1` and a condition number of `1/(1-y)`, so a caller that knows `1 - y`
        /// (see [`langevin_1m`](RealSpecialMath::langevin_1m)) should pass it here rather
        /// than form `y` and lose its low digits: this entry point works in `t` throughout
        /// and is accurate to a few ulp at any sharpness. `t = 0` returns `+∞`, `t > 1`
        /// gives the negative branch, and `t < 0` is out of the domain (NaN under
        /// overflow checking). Same cost as `inv_langevin`.
        fn inv_langevin_1m[][](self: Self) -> Self;

        /// GELU activation function, defined as `$\tfrac{1}{2} x \left(1 + \operatorname{erf}\!\left(\frac{\alpha x}{\sqrt{2}}\right)\right)$`,
        /// where `alpha` helps control the shape of the curve. The standard GELU function
        /// is recovered when `alpha` is 1.
        ///
        /// For f32 vectors, this remains decently accurate even with the `Medium` and `Worst` precision policies,
        /// thanks to good `erf` implementations at the various precision levels. See `erf` for more details.
        ///
        /// To also obtain the derivative with respect to `x` (which shares most of the computation), use
        /// [`gelu_d`](crate::RealPrimalMath::gelu_d).
        fn gelu[][](self: Self, alpha: Self) -> Self;

        /// Swish activation function, defined as `$x\,\sigma(\beta x) = \frac{x}{1 + e^{-\beta x}}$`,
        /// where `beta` controls the sharpness of the gate. The standard Swish/SiLU function
        /// is recovered when `beta` is 1. As `beta -> 0`, the output approaches `x/2` (half-identity);
        /// as `beta -> inf`, Swish approaches ReLU.
        ///
        /// To also obtain the derivative with respect to `x`, use
        /// [`swish_d`](crate::RealPrimalMath::swish_d).
        fn swish[][](self: Self, beta: Self) -> Self;

        /// Computes the algebraic sigmoid function, defined as `$\frac{x}{(1 + |x|^N)^{1/N}}$`, where
        /// `N` is a positive integer parameter that controls the steepness of the curve.
        ///
        /// This also has the unique behavior where for `N=0`, the function is just the identity function,
        /// and for `N=1` it is the [softsign function](https://en.wikipedia.org/wiki/Activation_function#Softsign).
        ///
        /// **Note**: This function uses `$|x|^N$` (the real absolute value), so it is non-holomorphic
        /// and only meaningful for real-valued inputs.
        ///
        /// To also obtain the derivative with respect to `x`, use
        /// [`algebraic_sigmoid_d`](crate::RealPrimalMath::algebraic_sigmoid_d).
        fn algebraic_sigmoid[const N: usize][N](self: Self) -> Self;

        /// Algebraic analogue of the [Swish](https://en.wikipedia.org/wiki/Swish_function) activation,
        /// defined as `$x\left(\frac{1}{2} + \frac{x}{2\sqrt{1 + x^2}}\right)$`. Equivalent to gating `x` by
        /// `(1 + algebraic_sigmoid::<2>(x)) / 2`, the `[0, 1]`-rescaled `N=2` algebraic sigmoid.
        ///
        /// Like standard Swish/SiLU, this is smooth and non-monotonic (it dips slightly below zero
        /// for moderately negative `x` before rising) and shares the same asymptotes (`f(x) -> x` as
        /// `x -> ∞`, `f(x) -> 0` as `x -> -∞`). Unlike Swish, it requires no `exp` or `log`, which
        /// is substantially cheaper on hardware without fast transcendentals.
        ///
        /// To also obtain the derivative with respect to `x` (which shares most of the underlying
        /// computation, notably `$1/\sqrt{1 + x^2}$`), use
        /// [`algebraic_swish_d`](crate::RealPrimalMath::algebraic_swish_d).
        ///
        /// # Historical note
        ///
        /// Algebraic gating functions of this form are effectively unknown in modern deep learning,
        /// which standardized on `exp`-based activations (sigmoid, Swish/SiLU, GELU) once GPUs made
        /// `exp` essentially free, a single-cycle special-function-unit op on most modern hardware.
        /// On CPUs the calculus is different: a vectorized `exp` still costs ~20+ cycles even with
        /// good polynomial approximations, while `sqrt`/`rsqrt` are cheap hardware ops (often
        /// approximated in 4-7 cycles). For CPU-side inference, training on CPU, or embedded targets
        /// without a transcendental SFU, this remains a competitive Swish-shaped activation at a
        /// fraction of the cost.
        fn algebraic_swish[][](self: Self) -> Self;

        /// Computes the natural log of the Gamma function (`$\ln|\Gamma(x)|$`) for any real input, for each value in a vector,
        /// and returns the sign of the Gamma function from before the absolute value was taken.
        fn lgamma_r[][](self: Self) -> (Self, Self);

        /// Computes the definite integral of the Gaussian function from `x0` to `x1`, with amplitude `a` and standard deviation `c`.
        /// This is more efficient than evaluating the indefinite integral at both limits and subtracting.
        ///
        /// The position `b` is assumed to be zero, so offset the limits accordingly for a non-zero position.
        fn gaussian_integral[][](x0: Self, x1: Self, a: Self, c: Self) -> Self;

        /// The [Box-Cox transform](https://en.wikipedia.org/wiki/Power_transform) of `x = self`
        /// with parameter `lambda`.
        ///
        /// ```math
        /// \mathrm{boxcox}(x, \lambda) = \begin{cases} \dfrac{x^\lambda - 1}{\lambda} & \lambda \ne 0 \\[6pt] \ln x & \lambda = 0\end{cases}
        /// ```
        ///
        /// The variance-stabilizing power transform of applied statistics: `$\lambda$` is fitted
        /// to make skewed data as close to normal as possible before a model sees it, and the
        /// family interpolates the transforms people otherwise pick by hand: `$\lambda = 1$`
        /// leaves the data alone up to a shift, `$1/2$` is a square root, `$0$` a logarithm,
        /// `$-1$` a reciprocal. A fixture of statistical software since Box and Cox introduced
        /// it in 1964.
        ///
        /// The two cases are one function: `$\ln x$` is the limit as `$\lambda \to 0$`, not a
        /// separate rule. Written out, `$(x^\lambda - 1)/\lambda$` is `$0/0$` there, and the
        /// trouble is not confined to the point. Computing `$x^\lambda$` and subtracting one
        /// cancels, so the naive form is already wrong in the fifth digit at
        /// `$\lambda = 10^{-12}$` and returns a flat zero by `$10^{-300}$`. That matters because
        /// a fitting routine searches `$\lambda$` near zero, which is the usual answer for
        /// right-skewed data.
        ///
        /// Evaluated as [`powf_m1`](thermite::math::TranscendentalMath::powf_m1)`(x, lambda)/lambda`,
        /// which forms `$x^\lambda - 1$` without ever forming `$x^\lambda$`, so there is nothing to
        /// cancel and **no series or crossover is needed**. Measured against a 60-digit oracle,
        /// it holds a few ulp from `$\lambda = 10^{-300}$` to `$\lambda = \pm 8$`. Only the exact
        /// `$\lambda = 0$` is selected apart.
        ///
        /// Domain is `$x > 0$`, and a negative `x` gives NaN. At `$x = 0$` the limits are taken:
        /// `$-1/\lambda$` for `$\lambda > 0$` and `$-\infty$` otherwise, which is the
        /// conventional choice. That needs no special case: `powf_m1(0, lambda)` is `$-1$`
        /// above zero and `$+\infty$` below, and the division does the rest.
        fn boxcox[][](self: Self, lambda: Self) -> Self;

        /// The Box-Cox transform of `$1 + x$`, where `x = self`.
        ///
        /// ```math
        /// \mathrm{boxcox1p}(x, \lambda) = \begin{cases} \dfrac{(1 + x)^\lambda - 1}{\lambda} & \lambda \ne 0 \\[6pt] \ln (1 + x) & \lambda = 0\end{cases}
        /// ```
        ///
        /// The shifted form exists for the same reason [`ln_1p`](thermite::math::TranscendentalMath::ln_1p)
        /// does: when `x` is small, `$1 + x$` rounds it away, and every digit of the answer
        /// with it. Calling [`boxcox`](crate::RealSpecialMath::boxcox)`(1 + x, lambda)` loses `x` entirely once
        /// `$|x| < \varepsilon$`, where this returns `$\lambda x$` to full precision. Built on
        /// [`compound_m1`](thermite::math::TranscendentalMath::compound_m1), which forms
        /// `$(1 + x)^\lambda - 1$` without forming either `$1 + x$` or `$(1+x)^\lambda$`.
        ///
        /// This is also the kernel underneath [`yeo_johnson`](crate::RealSpecialMath::yeo_johnson), whose
        /// argument is data centered near zero by construction.
        ///
        /// Domain is `$x > -1$`; below that the result is NaN. At `$x = -1$` the limits are
        /// `$-1/\lambda$` for `$\lambda > 0$` and `$-\infty$` otherwise.
        fn boxcox_1p[][](self: Self, lambda: Self) -> Self;

        /// The inverse [Box-Cox transform](https://en.wikipedia.org/wiki/Power_transform) of
        /// `y = self` with parameter `lambda`, undoing [`boxcox`](crate::RealSpecialMath::boxcox).
        ///
        /// ```math
        /// \mathrm{boxcox}^{-1}(y, \lambda) = \begin{cases} (\lambda y + 1)^{1/\lambda} & \lambda \ne 0 \\[6pt] e^y & \lambda = 0\end{cases}
        /// ```
        ///
        /// Wanted by anyone who uses the forward transform: a model fitted on transformed
        /// data predicts in transformed units, and the prediction has to come back.
        ///
        /// Evaluated as `$\exp\!\left(\ln(1 + \lambda y)/\lambda\right)$` rather than as a
        /// literal power, which is not merely a rearrangement. The whole
        /// point of [`boxcox`](crate::RealSpecialMath::boxcox) is that it stays accurate as `$\lambda \to 0$`,
        /// and `$\lambda$` fitted near zero is the common case. There `$\lambda y$` is tiny,
        /// so forming `$\lambda y + 1$` and raising it to the power `$1/\lambda$` throws away
        /// exactly the digits the forward transform took care to keep. Through `ln_1p` the
        /// exponent tends smoothly to `y`, so the `$\lambda = 0$` case is the limit rather
        /// than a discontinuity, and only the exact zero is selected apart.
        ///
        /// The range of the forward transform is `$\lambda y + 1 > 0$`. Outside it the result
        /// is NaN, and on the boundary it is `$0$` for `$\lambda > 0$` and `$+\infty$` below.
        fn inv_boxcox[][](self: Self, lambda: Self) -> Self;

        /// The inverse of [`boxcox_1p`](crate::RealSpecialMath::boxcox_1p).
        ///
        /// ```math
        /// \mathrm{boxcox1p}^{-1}(y, \lambda) = \begin{cases} (\lambda y + 1)^{1/\lambda} - 1 & \lambda \ne 0 \\[6pt] e^y - 1 & \lambda = 0\end{cases}
        /// ```
        ///
        /// The same exponent as [`inv_boxcox`](crate::RealSpecialMath::inv_boxcox) with `expm1` outside it
        /// instead of `exp`, so a result near zero keeps its relative accuracy, which, this
        /// being the inverse of a transform applied to data centered near zero, is the
        /// ordinary case rather than an edge one. Also the kernel underneath
        /// [`inv_yeo_johnson`](crate::RealSpecialMath::inv_yeo_johnson).
        fn inv_boxcox_1p[][](self: Self, lambda: Self) -> Self;

        /// The [Yeo-Johnson transform](https://en.wikipedia.org/wiki/Power_transform) of
        /// `y = self` with parameter `lambda`.
        ///
        /// ```math
        /// \psi(y, \lambda) = \begin{cases}
        ///   \dfrac{(y + 1)^\lambda - 1}{\lambda} & y \ge 0,\ \lambda \ne 0 \\[6pt]
        ///   \ln(y + 1) & y \ge 0,\ \lambda = 0 \\[6pt]
        ///   -\dfrac{(1 - y)^{2 - \lambda} - 1}{2 - \lambda} & y < 0,\ \lambda \ne 2 \\[6pt]
        ///   -\ln(1 - y) & y < 0,\ \lambda = 2
        /// \end{cases}
        /// ```
        ///
        /// Box-Cox's sibling, and the one that gets used more, since it is defined on the whole
        /// real line rather than on `$x > 0$`. Same job (fit `$\lambda$` by maximum likelihood
        /// to make skewed data as close to normal as a power transform can) without the "add a
        /// constant to make everything positive first" step, which is an arbitrary choice that
        /// changes the fitted `$\lambda$`. Introduced by Yeo and Johnson in 2000.
        ///
        /// # One kernel, not four
        ///
        /// The four cases are one function seen twice. The `$y < 0$` branch is the `$y \ge 0$`
        /// branch applied to `$|y|$` with `$\lambda$` reflected to `$2 - \lambda$` and the
        /// result negated, which is what makes `$\psi$` smooth in `$\lambda$` across `$y = 0$`
        /// in the first place. Folding the sign out first therefore collapses the two
        /// logarithmic special cases (`$\lambda = 0$` above zero, `$\lambda = 2$` below) into
        /// the single seam that [`boxcox_1p`](crate::RealSpecialMath::boxcox_1p) already handles, and the whole
        /// transform is `$\pm\,\mathrm{boxcox1p}(|y|, \lambda\ \mathrm{or}\ 2 - \lambda)$`.
        ///
        /// That the kernel is the `1p` form and not [`boxcox`](crate::RealSpecialMath::boxcox) applied to
        /// `$1 + |y|$` matters here more than anywhere else. `$\psi(y, \lambda) \approx y$`
        /// near the origin for every `$\lambda$`, and the origin is where the data is: the
        /// transform's reason for existing is samples that straddle zero. Forming `$1 + |y|$`
        /// would round away everything below `$\varepsilon$` and return a flat zero there.
        ///
        /// The value is finite for every finite `y`, so there is nothing to guard: the two
        /// domain edges of the kernel are at `$|y| = -1$`, which the fold never reaches.
        fn yeo_johnson[][](self: Self, lambda: Self) -> Self;

        /// The inverse [Yeo-Johnson transform](https://en.wikipedia.org/wiki/Power_transform),
        /// undoing [`yeo_johnson`](crate::RealSpecialMath::yeo_johnson).
        ///
        /// ```math
        /// \psi^{-1}(z, \lambda) = \begin{cases}
        ///   (\lambda z + 1)^{1/\lambda} - 1 & z \ge 0,\ \lambda \ne 0 \\[6pt]
        ///   e^z - 1 & z \ge 0,\ \lambda = 0 \\[6pt]
        ///   1 - \left((\lambda - 2) z + 1\right)^{1/(2 - \lambda)} & z < 0,\ \lambda \ne 2 \\[6pt]
        ///   1 - e^{-z} & z < 0,\ \lambda = 2
        /// \end{cases}
        /// ```
        ///
        /// The same sign fold as the forward transform, over
        /// [`inv_boxcox_1p`](crate::RealSpecialMath::inv_boxcox_1p). `$\psi$` is increasing and fixes the origin,
        /// so the branch on the way back is the sign of the transformed value, which is the
        /// sign of `y`.
        ///
        /// Unlike the forward direction this one has a range to respect: for `$\lambda > 0$`
        /// the transform's image is bounded below by `$-1/\lambda$`, and a `z` past that came
        /// from no `y`. Such an input gives NaN rather than a plausible-looking number.
        fn inv_yeo_johnson[][](self: Self, lambda: Self) -> Self;

        /// Evaluates **all** real spherical harmonics through degree `L` at the unit
        /// direction `(x, y, z)`, into `out[l * (l + 1) + m]` for `m` in `-l..=l`.
        ///
        /// Orthonormal real harmonics. Evaluation is pure polynomial arithmetic:
        /// no trigonometry, no division, `O(L^2)` FMAs total, exact zeros for every
        /// `m != 0` harmonic at the poles, fully unrolled at compile time for each
        /// `L` up to [`MAX_SH_DEGREE`] (above that it takes the rolled general path,
        /// which is correct at any degree but roughly 10x slower).
        ///
        /// `CS` picks the phase convention. `false` gives the standard real-SH
        /// tables (`$Y_{11} = \sqrt{3/4\pi}\,x$`); `true` applies the Condon-Shortley
        /// `$(-1)^{|m|}$` phase, negating every odd-`|m|` harmonic to match Sloan's
        /// `SHEval` and the physics convention (`$Y_{11} = -\sqrt{3/4\pi}\,x$`). The
        /// choice is baked into a constant table, so neither costs an instruction,
        /// but mixing the two silently corrupts any projection/reconstruction
        /// round-trip, which is why it must be named.
        ///
        /// `N` must equal `(L + 1)^2` (compile-time checked). The direction is
        /// assumed unit-length, and nothing renormalizes. See
        /// [`sh_impl`](specialized::sh_impl) for the full convention, algorithm,
        /// and domain notes.
        ///
        /// ```
        /// use thermite::prelude::*;
        /// use thermite_special::RealSpecialMath;
        ///
        /// type V = Vector<f64>;
        /// let (x, y, z) = (V::splat(0.6), V::splat(0.0), V::splat(0.8));
        ///
        /// let mut sh = [V::ZERO; 9];
        /// V::spherical_harmonics::<2, 9, false>(x, y, z, &mut sh);
        /// // Y(1,1) = sqrt(3/4pi) * x
        /// assert!((sh[3].extract::<0>() - 0.48860251190292 * 0.6).abs() < 1e-14);
        ///
        /// // Condon-Shortley negates odd |m|, and agrees on even |m|.
        /// let mut cs = [V::ZERO; 9];
        /// V::spherical_harmonics::<2, 9, true>(x, y, z, &mut cs);
        /// assert_eq!(cs[3].extract::<0>(), -sh[3].extract::<0>());
        /// assert_eq!(cs[8].extract::<0>(), sh[8].extract::<0>());
        /// ```
        #[skip_dispatch] fn spherical_harmonics[const L: usize, const N: usize, const CS: bool][L, N, CS](x: Self, y: Self, z: Self, out: &mut [Self; N]) -> ();

        @scalar_sig {
        /// Builds the runtime coefficient table that [`spherical_harmonics_with`](RealSpecialMath::spherical_harmonics_with)
        /// and [`spherical_harmonics_d_with`](RealSpecialMath::spherical_harmonics_d_with) evaluate.
        ///
        /// The table depends only on `L` and `CS`, never on the direction, so a caller
        /// sweeping many directions should build it once rather than calling the
        /// one-shot [`spherical_harmonics`](RealSpecialMath::spherical_harmonics)
        /// per direction. The phase is baked in here, which is why the evaluators take
        /// no `CS`.
        ///
        /// The table is typed by `Self::Primal`, the unaugmented value type: the
        /// recurrence coefficients are constants, so a `Dual`'s derivative parts and a
        /// `Complex`'s imaginary part would only store zeros. For plain vectors and
        /// `Compensated` the primal is `Self` and nothing changes. For `Dual` the table
        /// is a fraction of the size and its entries multiply as reals.
        ///
        /// ```
        /// use thermite::prelude::*;
        /// use thermite_special::{RealSpecialMath, ShTable};
        ///
        /// type V = Vector<f64>;
        /// const L: usize = 3;
        /// const N: usize = (L + 1) * (L + 1);
        ///
        /// let mut table = ShTable::<V, N>::zeroed();
        /// V::spherical_harmonics_table::<L, N, false>(&mut table);
        ///
        /// let mut sh = [V::ZERO; N];
        /// for &(x, y, z) in &[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)] {
        ///     V::spherical_harmonics_with::<L, N>(
        ///         &table, V::splat(x), V::splat(y), V::splat(z), &mut sh,
        ///     );
        /// }
        /// assert!((sh[1].extract::<0>() - 0.48860251190292).abs() < 1e-14);
        /// ```
        #[skip_dispatch] fn spherical_harmonics_table[const L: usize, const N: usize, const CS: bool][L, N, CS](table: &mut ShTable<<Self as PrimalProjection>::Primal, N>) -> ()
            = scalar(table: &mut ShTable<Self, N>) -> ();

        /// Evaluates all harmonics through degree `L` from a prebuilt table.
        ///
        /// The table holds `Self::Primal` coefficients. See
        /// [`spherical_harmonics_table`](RealSpecialMath::spherical_harmonics_table)
        /// for how to build it and why, and
        /// [`spherical_harmonics`](RealSpecialMath::spherical_harmonics) for the
        /// conventions and layout.
        #[skip_dispatch] fn spherical_harmonics_with[const L: usize, const N: usize][L, N](table: &ShTable<<Self as PrimalProjection>::Primal, N>, x: Self, y: Self, z: Self, out: &mut [Self; N]) -> ()
            = scalar(table: &ShTable<Self, N>, x: Self, y: Self, z: Self, out: &mut [Self; N]) -> ();
        }
    }

    /// "Primal" special functions: the value-and-derivative (`_d`) forms of the activation
    /// functions, returning `(value, derivative)` together.
    ///
    /// These exist for *single-value* real numbers (`f32`, `f64`, `Compensated`, ...) where the
    /// analytic derivative is a useful, cheaply-shared byproduct of the value. They are **not**
    /// implemented for derivative-carrying numbers such as `Dual`: an automatic-differentiation
    /// type already produces the derivative from the plain value form (e.g. [`gelu`](RealSpecialMath::gelu)),
    /// so the bundled `_d` derivative would be redundant work at the wrong level of abstraction.
    ///
    /// Each `*_d` method mirrors the like-named value-only function in [`SpecialMath`] /
    /// [`RealSpecialMath`], returning that same value as the first tuple element.
    #[diagnostic::on_unimplemented(
        message = "`{Self}` does not provide value-and-derivative special math (`softplus_d`, `gelu_d`, `spherical_harmonics_d`, ...)",
        note = "`RealPrimalMath` builds on `RealSpecialMath` and is implemented only for primal real vectors (plain float vectors and `Compensated`), never for `Dual` or `Complex`, which get their derivatives from the value form instead. A bare `f32`/`f64` does not qualify either. Wrap it in `Vector::<f32>::splat(x)`, or use `ScalarSpecialMath`."
    )]
    trait RealPrimal: RealSpecialMathWithPolicy & PrimalMathWithPolicy {
        /// [`spherical_harmonics`](RealSpecialMath::spherical_harmonics) plus the
        /// ambient Cartesian gradient of every harmonic, into `ddx`/`ddy`/`ddz`.
        ///
        /// Lives on [`RealPrimalMath`] rather than [`RealSpecialMath`], so `Dual` does
        /// not get it, and should not want it. If you need `$\partial/\partial(x,y,z)$`,
        /// call this directly rather than evaluating
        /// [`spherical_harmonics`](RealSpecialMath::spherical_harmonics) on a
        /// `Dual<V, 3>` seeded with an identity Jacobian: this shares the recurrence
        /// between the value and all three gradients, whereas dual arithmetic carries a
        /// derivative through every operation and costs roughly twice as much.
        ///
        /// `Dual` earns its keep on the _value_ form instead, where `(x, y, z)` are
        /// themselves functions of upstream parameters and the chain rule has real work
        /// to do. Even there, going the other way (contracting these three gradients
        /// against an upstream Jacobian) loses: spherical harmonics cost about two
        /// operations per harmonic to evaluate but three per harmonic per parameter to
        /// contract, because one recurrence produces the whole basis.
        ///
        /// The derivatives are those of the polynomial form at the given (unit)
        /// input. Project out the radial component (`g - (g . n) n`) for the
        /// tangential gradient. Shares all recurrence work with the value pass, since
        /// the gradients come from tabulated norm ratios, not new recurrences.
        #[skip_dispatch] fn spherical_harmonics_d[const L: usize, const N: usize, const CS: bool][L, N, CS](
            x: Self,
            y: Self,
            z: Self,
            out: &mut [Self; N],
            ddx: &mut [Self; N],
            ddy: &mut [Self; N],
            ddz: &mut [Self; N],
        ) -> ();

        /// [`zernike_basis`](SpecialMath::zernike_basis) plus `$\partial Z_n^m/\partial x$`
        /// and `$\partial Z_n^m/\partial y$` for every mode, in the same ANSI layout.
        ///
        /// This is what a Shack-Hartmann wavefront reconstruction integrates against. The
        /// sensor measures local wavefront *slopes*, not the wavefront itself, so the fit
        /// matrix is built from the gradient basis and the value basis never appears in it.
        ///
        /// Lives on [`RealPrimalMath`] rather than [`SpecialMath`] for the same reason
        /// [`spherical_harmonics_d`](RealPrimalMath::spherical_harmonics_d) does: `Dual`
        /// should not get it and should not want it. Seeding a `Dual<V, 2>` and calling the
        /// value form carries two derivative components through every operation of the whole
        /// ladder, where this differentiates only the two factors that depend on the point
        /// and shares the radial recurrence between the value and both gradients.
        ///
        /// The gradient is finite everywhere, including the pupil centre. That is the
        /// practical dividend of the Cartesian formulation: the polar
        /// `$\partial_\theta Z/\rho$` is singular there, and hand-rolled polar
        /// implementations guard the origin with a special case.
        ///
        /// `N` must equal `(L+1)(L+2)/2`, and `NORM` is as on
        /// [`zernike_basis`](SpecialMath::zernike_basis). All three output buffers are
        /// written in full.
        #[skip_dispatch] fn zernike_basis_d[const L: usize, const NORM: u8, const N: usize][L, NORM, N](
            x: Self,
            y: Self,
            out: &mut [Self; N],
            ddx: &mut [Self; N],
            ddy: &mut [Self; N],
        ) -> ();

        /// [`spherical_harmonics_with`](RealSpecialMath::spherical_harmonics_with) plus
        /// the ambient Cartesian gradients, from a prebuilt table.
        #[skip_dispatch] fn spherical_harmonics_d_with[const L: usize, const N: usize][L, N](
            table: &ShTable<Self, N>,
            x: Self,
            y: Self,
            z: Self,
            out: &mut [Self; N],
            ddx: &mut [Self; N],
            ddy: &mut [Self; N],
            ddz: &mut [Self; N],
        ) -> ();

        /// [`softplus`](SpecialMath::softplus) together with its derivative w.r.t. `x`
        /// (the logistic sigmoid `$\sigma(kx)$`).
        fn softplus_d[][](self: Self, k: Self, rcp_k: Self) -> (Self, Self);

        /// [`gelu`](RealSpecialMath::gelu) together with its derivative w.r.t. `x`.
        fn gelu_d[][](self: Self, alpha: Self) -> (Self, Self);

        /// [`swish`](RealSpecialMath::swish) together with its derivative w.r.t. `x`.
        fn swish_d[][](self: Self, beta: Self) -> (Self, Self);

        /// [`algebraic_sigmoid`](RealSpecialMath::algebraic_sigmoid) together with its derivative w.r.t. `x`.
        fn algebraic_sigmoid_d[const N: usize][N](self: Self) -> (Self, Self);

        /// [`algebraic_swish`](RealSpecialMath::algebraic_swish) together with its derivative w.r.t. `x`.
        fn algebraic_swish_d[][](self: Self) -> (Self, Self);

        /// [`langevin`](RealSpecialMath::langevin) together with its derivative
        /// `$L'(x) = \frac{1}{x^2} - \operatorname{csch}^2 x$`.
        ///
        /// The derivative shares every intermediate with the value, so this costs a
        /// handful of arithmetic ops over `langevin` alone.
        fn langevin_d[][](self: Self) -> (Self, Self);
    }
}
