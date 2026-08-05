#![no_std]
#![allow(unused, clippy::needless_arbitrary_self_type, clippy::needless_range_loop)]

use thermite::{
    element::{Element, ElementExt, FloatElementWithBits},
    math::{
        FloatConsts, TranscendentalMathWithPolicy,
        policy::{DefaultPolicy, Policy},
        scalar::Unwrap,
    },
    vector::{FloatVector, FloatVectorWithBits},
};

pub mod specialized;
pub mod tables;

use crate::specialized::{CarlsonKind, EllipticKind, WrapTo};

/// Elliptic integral request structs and the traits they implement:
///
/// - Carlson symmetric integrals (for [`SpecialMath::carlson`]): [`CarlsonRf`](elliptic::CarlsonRf),
///   [`CarlsonRc`](elliptic::CarlsonRc), [`CarlsonRd`](elliptic::CarlsonRd), [`CarlsonRj`](elliptic::CarlsonRj),
///   [`CarlsonRg`](elliptic::CarlsonRg), implementing [`CarlsonKind`].
/// - Legendre integrals (for [`SpecialMath::ellint`]): [`EllintK`](elliptic::EllintK)/[`EllintF`](elliptic::EllintF),
///   [`EllintE`](elliptic::EllintE)/[`EllintEInc`](elliptic::EllintEInc),
///   [`EllintD`](elliptic::EllintD)/[`EllintDInc`](elliptic::EllintDInc),
///   [`EllintPi`](elliptic::EllintPi)/[`EllintPiInc`](elliptic::EllintPiInc), implementing
///   [`EllipticKind`]. Completeness is encoded by the struct - a complete integral has no `phi` field.
pub mod elliptic {
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
        // EllipticKind impl) does the work; this generates the full trait family (policy + default +
        // dispatched vector impl + scalar) around it. Because the struct's element backend
        // (EllipticEval) covers both `Vector<R>` and scalar floats, the same bound works at the
        // scalar layer - no Unwrap wrapping needed here.
        $(@kinds {$(
            $(#[$kmeta:meta])*
            fn $kname:ident : $ktrait:path;
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
        note = "The special-math traits are auto-implemented for every float vector (any `FloatVector` whose element is `f32`/`f64`) and for composite float types. A bare `f32`/`f64` does not qualify - wrap it in `Vector::<f32>::splat(x)`, or use `ScalarSpecialMath`'s `scalar_`-prefixed methods.",
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
        /// **NOTE**: The digamma function is not defined at zero or the negative integers; those inputs
        /// yield NaN when overflow checking is enabled.
        fn digamma[][](self: Self) -> Self;

        /// Computes the Beta function `$\mathrm{B}(x, y)$`
        fn beta[][](self: Self, y: Self) -> Self;

        /// Computes the m-th derivative of the n-th degree Jacobi polynomial
        ///
        /// A the special case where α and β are both zero, the Jacobi polynomial reduces to a
        /// Legendre polynomial.
        ///
        /// **NOTE**: Given constant α, β or `n`, LLVM will happily optimize those away and unroll loops.
        fn jacobi[][](self: Self, alpha: Self, beta: Self, n: u32, m: u32) -> Self;

        /// Computes the N-th degree physicists' [Hermite polynomial](https://en.wikipedia.org/wiki/Hermite_polynomials)
        /// `H_n(x)` where `x` is `self` and `N` is the polynomial degree.
        ///
        /// This uses the recurrence relation to compute the polynomial iteratively.
        fn hermite[const N: usize][N](self: Self) -> Self;

        /// Computes the n-th degree physicists' [Hermite polynomial](https://en.wikipedia.org/wiki/Hermite_polynomials)
        /// `H_n(x)` where `x` is `self` and `n` is a vector of unsigned integers representing the polynomial degree.
        ///
        /// The polynomial is calculated independently per-lane with the given degree in `n`.
        ///
        /// This uses the recurrence relation to compute the polynomial iteratively.
        fn hermitev[][](self: Self, n: Self::Unsigned) -> Self;

        /// Evaluates a finite series of [Chebyshev polynomials](https://en.wikipedia.org/wiki/Chebyshev_polynomials)
        /// of the `K`-th kind at `x = self`:
        ///
        /// ```math
        /// \sum_{k=0}^{N-1} \mathrm{coeffs}[k] \cdot P_k(x)
        /// ```
        ///
        /// where `P_k` is `T_k`, `U_k`, `V_k`, or `W_k` depending on `K`. All four kinds share the
        /// recurrence `$P_{k+1}(x) = 2x \cdot P_k(x) - P_{k-1}(x)$` with `P_0(x) = 1`; they differ only in
        /// `P_1(x)`:
        ///
        /// | `K` | Kind   | `P_1(x)`   | Notes |
        /// |-----|--------|------------|-------|
        /// | `1` | First  (`T_k`) | `x`        | Most common; minimax/approximation basis on `[-1, 1]`. |
        /// | `2` | Second (`U_k`) | `2x`       | Related to `$\sin((k+1)\theta)/\sin(\theta)$` under `$x = \cos\theta$`. |
        /// | `3` | Third  (`V_k`) | `2x - 1`   | "Airfoil" polynomials; `$\cos((k+\tfrac12)\theta)/\cos(\theta/2)$`. |
        /// | `4` | Fourth (`W_k`) | `2x + 1`   | `$\sin((k+\tfrac12)\theta)/\sin(\theta/2)$`. |
        ///
        /// Any other value of `K` is a compile-time error.
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
        #[skip_dispatch] fn chebyshev[const K: usize, const N: usize][K, N](self: Self, coeffs: &[Self::Element; N]) -> Self;

        /// Computes the Gaussian function with amplitude `a` and standard deviation `c`, defined as `$a\, e^{-\frac{1}{2}(x/c)^2}$`.
        ///
        /// The position `b` is assumed to be zero. For a non-zero position, use `self - b` as the input.
        fn gaussian[][](self: Self, a: Self, c: Self) -> Self;

        /// Computes the m-th associated n-th degree Legendre polynomial,
        /// where m=0 signifies the regular n-th degree Legendre polynomial.
        ///
        /// If `m` is odd, the input is only valid between -1 and 1
        ///
        /// **NOTE**: Given constant `n` and/or `m`, LLVM will happily unroll and optimize inner loops.
        ///
        /// Internally, this is computed with [`jacobi`](SpecialMath::jacobi) when m > 0.
        fn legendre[][](self: Self, n: u32, m: u32) -> Self;

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

        @kinds {
            /// Carlson symmetric elliptic integral, selected by a [`CarlsonKind`] request struct
            /// with named fields - the arity (and which argument is the parameter / repeated one)
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
            /// its own arguments; completeness is encoded by whether the struct has a `phi` field.
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
        note = "`RealSpecialMath` is only meaningful for real-valued float vectors - complex number types deliberately do not implement it. A bare `f32`/`f64` does not qualify - wrap it in `Vector::<f32>::splat(x)`, or use `ScalarSpecialMath`."
    )]
    trait RealSpecial: SpecialMathWithPolicy {
        /// Computes the inverse error function.
        fn erfinv[][](self: Self) -> Self;

        /// Computes the Probit function, the inverse of the cumulative distribution function
        /// of the standard normal distribution.
        fn probit[][](self: Self) -> Self;

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
        /// **Note**: This function uses `$|x|^N$` (the real absolute value), making it non-holomorphic
        /// and therefore only meaningful for real-valued inputs.
        ///
        /// To also obtain the derivative with respect to `x`, use
        /// [`algebraic_sigmoid_d`](crate::RealPrimalMath::algebraic_sigmoid_d).
        fn algebraic_sigmoid[const N: usize][N](self: Self) -> Self;

        /// Algebraic analogue of the [Swish](https://en.wikipedia.org/wiki/Swish_function) activation,
        /// defined as `$x\left(\frac{1}{2} + \frac{x}{2\sqrt{1 + x^2}}\right)$`. Equivalent to gating `x` by
        /// `(1 + algebraic_sigmoid::<2>(x)) / 2`, the `[0, 1]`-rescaled `N=2` algebraic sigmoid.
        ///
        /// Like standard Swish/SiLU, this is smooth and non-monotonic - it dips slightly below zero
        /// for moderately negative `x` before rising - and shares the same asymptotes (`f(x) -> x` as
        /// `x -> ∞`, `f(x) -> 0` as `x -> -∞`). Unlike Swish, it requires no `exp` or `log`, making
        /// it substantially cheaper on hardware without fast transcendentals.
        ///
        /// To also obtain the derivative with respect to `x` (which shares most of the underlying
        /// computation, notably `$1/\sqrt{1 + x^2}$`), use
        /// [`algebraic_swish_d`](crate::RealPrimalMath::algebraic_swish_d).
        ///
        /// # Historical note
        ///
        /// Algebraic gating functions of this form are effectively unknown in modern deep learning,
        /// which standardized on `exp`-based activations (sigmoid, Swish/SiLU, GELU) once GPUs made
        /// `exp` essentially free - a single-cycle special-function-unit op on most modern hardware.
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
        message = "`{Self}` does not provide value-and-derivative special math (`softplus_d`, `gelu_d`, ...)",
        note = "`RealPrimalMath` builds on `RealSpecialMath` and is only meaningful for real-valued float vectors. A bare `f32`/`f64` does not qualify - wrap it in `Vector::<f32>::splat(x)`, or use `ScalarSpecialMath`."
    )]
    trait RealPrimal: RealSpecialMathWithPolicy {
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
    }
}
