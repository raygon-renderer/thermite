// #![no_std]
#![allow(unused, clippy::needless_arbitrary_self_type)]

use thermite::{
    math::{
        FloatConsts, TranscendentalMathWithPolicy,
        policy::{DefaultPolicy, Policy},
    },
    vector::FloatVector,
};

pub mod specialized;

macro_rules! decl_math {
    (
        $(#[$trait_meta:meta])*
        trait $trait:ident $(: $($bound:ident)&+)? { $(
            $(#[$meta:meta])*
            fn $name:ident [ $($generics:tt)* ][$($generic_names:ident),*]( $($arg_name:ident :$arg_ty:ty),* $(,)?) -> $ret:ty
                $(where [ $($where_clause:tt)* ])?;
        )*}
    ) => {paste::paste! {
        #[doc = "" $trait " Math functions for floating-point vectors with customizable policies.\n\n"]
        #[doc = "Each method has a `_p`-suffixed variant in this trait that accepts a leading `P: Policy` generic.\n\n"]
        #[doc = "All floating-point vector types that implement [`Specialized" $trait "Math`](specialized::SpecializedSpecialMath) will\n"]
        #[doc = "automatically implement this trait, and [`" $trait "Math`] as well."]
        $(#[$trait_meta])*
        #[thermite_dispatch::dispatch(Self)]
        pub trait [<$trait MathWithPolicy>]: $($($bound +)+)? {$(
            $(#[$meta])* fn [<$name _p>]<P: Policy, $($generics)*>($($arg_name: $arg_ty),*) -> $ret
                $(where $($where_clause)*)?;
        )*}

        #[doc = "" $trait " Math functions for floating-point vectors using the default policy.\n\n"]
        #[doc = "Implementors of [`" $trait "MathWithPolicy`] automatically implement this trait.\n\n"]
        #[doc = "Each method here has a `_p`-suffixed counterpart in [`" $trait "MathWithPolicy`] that\n"]
        #[doc = "accepts a leading `P: Policy` generic for fine-grained precision/performance control."]
        $(#[$trait_meta])*
        #[thermite_dispatch::dispatch(Self)]
        pub trait [<$trait Math>]: [<$trait MathWithPolicy>] {$(
            $(#[$meta])* #[inline(always)] fn $name<$($generics)*>($($arg_name: $arg_ty),*) -> $ret
                $(where $($where_clause)*)?
            { [<$trait MathWithPolicy>]::[<$name _p>]::<DefaultPolicy, $($generic_names),*>($($arg_name),*) }
        )*}

        impl<M> [<$trait Math>] for M where M: [<$trait MathWithPolicy>] {}

        #[thermite_dispatch::dispatch(Self)]
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
        )*}
    }};
}

decl_math! {
    /// Special math functions that are valid for both real and complex floating-point vectors.
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

        /// Computes the Logistic sigmoid function, defined as `1 / (1 + exp(-x))`.
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

        /// Computes the softplus function, defined as `ln(1 + exp(k * x))`,
        /// as well as its derivative with respect to `x`.
        ///
        /// This is a smooth approximation to the ReLU function
        /// that is more numerically stable for large inputs.
        ///
        /// The parameter `k` controls the steepness of the curve, with larger values approaching ReLU more closely.
        /// However, computing softplus with a steepness value is a non-zero extra cost, and therefore passing `None`
        /// will be considered `k=1` and skip extra work.
        ///
        /// If the `Some`-ness of `k` is known at compile time, LLVM may optimize away the
        /// conditionals and extra computations when `k` is `None`.
        fn softplus[][](self: Self, k: Option<Self>) -> (Self, Self);

        /// Computes the Gamma function (`Γ(z)`) for any real input, for each value in a vector.
        ///
        /// This implementation uses a few different behaviors to ensure the greatest precision where possible.
        ///
        /// * For non-integer positive inputs, it uses the Lanczos approximation.
        /// * For small non-integer negative inputs, it uses the recursive identity `Γ(z)=Γ(z+1)/z` until `z` is positive.
        /// * For large non-integer negative inputs, it uses the reflection formula `-π/(Γ(z)sin(πz)z)`.
        /// * For positive integers, it simply computes the factorial in a tight loop to ensure precision. Lookup tables could not be used with SIMD.
        /// * At zero, the result will be positive or negative infinity based on the input sign (signed zero is a thing).
        ///
        /// **NOTE**: The Gamma function is not defined for negative integers.
        fn tgamma[][](self: Self) -> Self;

        /// Computes the natural log of the Gamma function (`ln(|Γ(x)|)`) for any real input, for each value in a vector.
        fn lgamma[][](self: Self) -> Self;

        /// Computes the Beta function `Β(x, y)`
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

        /// Computes the Gaussian function with amplitude `a` and standard deviation `c`, defined as `a * exp(-0.5 * (self / c)^2)`.
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

        /// Computes both branches of the Lambert W function simultaneously: (W₀(x), W₋₁(x)).
        ///
        /// The W₀ result is valid for x >= -1/e; the W₋₁ result is valid for -1/e <= x < 0.
        /// Outside these domains, the respective result is NaN (when overflow checking is enabled).
        fn lambert_w[][](self: Self) -> (Self, Self);

        fn bessel_j[const N: usize][N](self: Self) -> Self;

        /// Computes the generalized exponential integral `E_n(x)` for integer order `n`.
        fn expint[const N: usize][N](self: Self) -> Self;
    }
}

decl_math! {
    /// Special math functions that are only defined for real-valued floating-point vectors.
    ///
    /// These functions either rely on ordering/sign information that has no complex analogue
    /// (e.g. `erfinv`, `probit`, `lgamma_r`), or use the real absolute value in a way that
    /// makes them non-holomorphic (e.g. `algebraic_sigmoid`).
    trait RealSpecial: SpecialMathWithPolicy {
        /// Computes the inverse error function.
        fn erfinv[][](self: Self) -> Self;

        /// Computes the Probit function, the inverse of the cumulative distribution function
        /// of the standard normal distribution.
        fn probit[][](self: Self) -> Self;

        /// GELU activation function, defined as `0.5 * x * (1 + erf((alpha * x) / sqrt(2)))`,
        /// where `alpha` helps control the shape of the curve. The standard GELU function
        /// is recovered when `alpha` is 1.
        ///
        /// Returns both the GELU value and its derivative with respect to `x` simultaneously,
        /// as they share much of the same computation.
        ///
        /// For f32 vectors, this remains decently accurate even with the `Medium` and `Worst` precision policies,
        /// thanks to good `erf` implementations at the various precision levels. See `erf` for more details. Furthermore,
        /// on `Average` and above precision policies, or on GPUs with native `exp` support, the derivative
        /// is essentially free.
        fn gelu[][](self: Self, alpha: Self) -> (Self, Self);

        /// Swish activation function, defined as `x * sigmoid(beta * x) = x / (1 + exp(-beta * x))`,
        /// where `beta` controls the sharpness of the gate. The standard Swish/SiLU function
        /// is recovered when `beta` is 1. As `beta -> 0`, the output approaches `x/2` (half-identity);
        /// as `beta -> inf`, Swish approaches ReLU.
        ///
        /// Returns both the Swish value and its derivative with respect to `x` simultaneously.
        fn swish[][](self: Self, beta: Self) -> (Self, Self);

        /// Computes the algebraic sigmoid function, defined as `x / (1 + |x|^N)^(1/N)`, where
        /// `N` is a positive integer parameter that controls the steepness of the curve. It also
        /// returns the derivative with respect to `x` simultaneously, as it shares much of the same computation.
        ///
        /// This also has the unique behavior where for `N=0`, the function is just the identity function,
        /// and for `N=1` it is the [softsign function](https://en.wikipedia.org/wiki/Activation_function#Softsign).
        ///
        /// **Note**: This function uses `|x|^N` (the real absolute value), making it non-holomorphic
        /// and therefore only meaningful for real-valued inputs.
        fn algebraic_sigmoid[const N: usize][N](self: Self) -> (Self, Self);

        /// Computes the natural log of the Gamma function (`ln(|Γ(x)|)`) for any real input, for each value in a vector,
        /// and returns the sign of the Gamma function from before the absolute value was taken.
        fn lgamma_r[][](self: Self) -> (Self, Self);

        /// Computes the definite integral of the Gaussian function from `x0` to `x1`, with amplitude `a` and standard deviation `c`.
        /// This is more efficient than evaluating the indefinite integral at both limits and subtracting.
        ///
        /// The position `b` is assumed to be zero, so offset the limits accordingly for a non-zero position.
        fn gaussian_integral[][](x0: Self, x1: Self, a: Self, c: Self) -> Self;
    }
}
