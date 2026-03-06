/// Execution policy used for controlling performance/precision/size tradeoffs in mathematical functions.
pub trait Policy {
    /// The specific policy used. This is a constant to allow for dead-code elimination of branches.
    const POLICY: PolicyParameters;
}

/** Precision Policy, tradeoffs between precision and performance.

The precision policy modifies how functions are evaluated to provide extra precision
at the cost of performance, or sacrifice precision for extra performance.

For example, some functions have a generic solution that is technically correct but due to floating
point errors will not be very precise, and it's often better to fallback to another solution that
does not accrue such errors, at the cost of performance.
*/
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum PrecisionPolicy {
    /// Precision is not important, so prefer simpler or faster algorithms.
    Worst = 0,
    /// Precision is not that important, so prefer faster algorithms.
    Medium = 1,
    /// Precision is important, but not the focus, so avoid expensive fallbacks.
    Average = 2,
    /// Precision is very important, so do everything to improve it.
    Best = 3,
    /// Precision is the only factor, use infinite sums to compute reference solutions.
    Reference = 9,
}

impl PrecisionPolicy {
    pub const fn eq(self, other: PrecisionPolicy) -> bool {
        (self as u8) == (other as u8)
    }
    pub const fn gt(self, other: PrecisionPolicy) -> bool {
        (self as u8) > (other as u8)
    }
    pub const fn ge(self, other: PrecisionPolicy) -> bool {
        (self as u8) >= (other as u8)
    }
    pub const fn lt(self, other: PrecisionPolicy) -> bool {
        (self as u8) < (other as u8)
    }
    pub const fn le(self, other: PrecisionPolicy) -> bool {
        (self as u8) <= (other as u8)
    }

    /// Returns the multiple of `EPSILON` to use as the tolerance for this precision policy.
    #[inline(always)]
    pub const fn tolerance(self) -> i64 {
        match self {
            PrecisionPolicy::Worst => 100_000,
            PrecisionPolicy::Medium => 10_000,
            PrecisionPolicy::Average => 100,
            PrecisionPolicy::Best => 20,
            PrecisionPolicy::Reference => 8,
        }
    }
}

/// Denormal/Subnormal numbers cause performance hiccups even in
/// well-behaved code. They are a side-effect of IEEE-754 gracefully degrading
/// with very small numbers, rather than immediately going to zero on underflow.
///
/// However, due to how some processors handle this, even simple operations on
/// denormal numbers can be over 100x slower.
///
/// Most PrecisionPolicy's will default to `FlushToZero`, while the high performance oriented
/// policies will default to `Crush` for the faster happy path.
///
/// However, if the `preserve_denormal` crate feature is enabled, all will default to `Preserve`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DenormalBehavior {
    /// Use exact bitwise operations to flush denormals to zero. This has a non-zero performance
    /// cost, but is a good default since the cost is constant.
    FlushToZero,

    /// Uses a "crush denormals" trick of `(a - (a - x))` where `a` is a very small constant. This
    /// removes denormals and is very fast in the happy path where the number is NOT denormal,
    /// but will incur a heavy cost if the number is denormal.
    Crush,

    /// Do nothing to remove denormal values. This is useful when the processor handles it for you,
    /// so we can completely skip over trying to flush them manually. However, unless you know
    /// that's the case, it's typically a bad idea to preserve them.
    Preserve,
}

impl DenormalBehavior {
    const fn preserve_any(a: Self, b: Self) -> Self {
        match (a, b) {
            (DenormalBehavior::Preserve, _) | (_, DenormalBehavior::Preserve) => DenormalBehavior::Preserve,
            _ => a,
        }
    }

    const fn select_default(crush: bool) -> Self {
        match (cfg!(feature = "preserve_denormals"), crush) {
            (true, _) => DenormalBehavior::Preserve,
            (false, true) => DenormalBehavior::Crush,
            (false, false) => DenormalBehavior::FlushToZero,
        }
    }
}

/// Customizable Policy Parameters
pub struct PolicyParameters {
    /// If true, methods will check for infinity/NaN/invalid domain issues and give a well-formed standard result.
    ///
    /// If false, all of that work is avoided, and the result is undefined in those cases. Garbage in, garbage out.
    ///
    /// However, those checks can be expensive.
    pub check_overflow: bool,

    /// If true, unrolled and optimized versions of some algorithms will be used. These can be much faster than
    /// the linear variants. If code size is important, this will improve codegen when used with `opt-level=z`
    pub unroll_loops: bool,

    /// Controls if precision should be emphasized or de-emphasized.
    pub precision: PrecisionPolicy,

    /// If true, methods will not try to avoid extra work by branching. Some of the internal branches are expensive,
    /// but branchless may be desired in some cases, such as minimizing code size.
    pub avoid_branching: bool,

    /// Some special functions require many, many iterations of a function to converge on an accurate result.
    /// This parameter controls the maximum iterations allowed. Setting this too low may result in loss of precision.
    ///
    /// Note that this is the upper limit allowed for pathological cases, and many loops will
    /// terminate dynamically before this.
    pub max_iterations: usize,

    /// If true, use compensated algorithms where available (such as Kahan summation).
    ///
    /// This attribute will change depending on the precision policy selected, and selecting
    /// a new precision policy may overwrite this value. Apply combinators carefully.
    pub use_compensation: bool,

    /// Specifies how denormals are handled. See [`DenormalBehavior`] for more info.
    pub denormal_behavior: DenormalBehavior,
}

impl PolicyParameters {
    /// Returns true if the policy says to avoid branches at the cost of precision
    #[inline(always)]
    pub const fn avoid_precision_branches(self) -> bool {
        self.avoid_branching && self.precision.le(PrecisionPolicy::Worst)
    }
}

/** Execution Policies (precision, performance, etc.)

To define a custom policy:
```rust,ignore
pub struct MyPolicy;

impl Policy for MyPolicy {
    const POLICY: Parameters = Parameters {
        check_overflow: false,
        unroll_loops: false,
        precision: PrecisionPolicy::Average,
        avoid_branching: true,
        max_iterations: 10000,
        use_compensation: true,
        denormal_behavior: DenormalBehavior::FlushToZero,
    };
}

let y = x.cbrt_p::<MyPolicy>();
```
*/
pub mod policies {
    use core::marker::PhantomData;

    use super::{DenormalBehavior, Policy, PolicyParameters, PrecisionPolicy};

    /// Policy adapter that increases the precision requires by one level,
    /// e.g.: `Worst` -> `Medium`, `Medium` -> `Average`, `Average` -> `Best`, `Best` -> `Reference`
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct ExtraPrecision<P: Policy>(PhantomData<P>);

    /// Policy adapter that decreases the precision required by one level,
    /// e.g.: `Reference` -> `Best`, `Best` -> `Average`, `Average` -> `Medium`, `Medium` -> `Worst`
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct LessPrecision<P: Policy>(PhantomData<P>);

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct UseCompensation<P: Policy, const USE_COMPENSATION: bool>(PhantomData<P>);

    /// Policy adapter that modifies the base policy to change overflow checking.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct CheckOverflow<P: Policy, const CHECK_OVERFLOW: bool>(PhantomData<P>);

    /// Policy adapter that modifies the base policy to change loop unrolling.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct UnrollLoops<P: Policy, const UNROLL_LOOPS: bool>(PhantomData<P>);

    /// Policy adapter that modifies the base policy to change branching behavior.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct AvoidBranching<P: Policy, const AVOID_BRANCHING: bool>(PhantomData<P>);

    // /// Policy adapter that modifies the base policy to change denormal preserving behavior.
    // #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    // pub struct PreserveDenormals<P: Policy, const PRESERVE_DENORMALS: bool>(PhantomData<P>);

    /// Policy adapter that modifies the base policy to change the maximum number of iterations for numerical methods.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct MaxIterations<P: Policy, const MAX_ITERATIONS: usize>(PhantomData<P>);

    /// Policy for worst precision, which is the least precise and fastest.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct WorstPrecision<P: Policy>(PhantomData<P>);
    /// Policy for medium precision, which is a balance between performance and precision.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct MediumPrecision<P: Policy>(PhantomData<P>);
    /// Policy for average precision, which is more precise than medium but less than best.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct AveragePrecision<P: Policy>(PhantomData<P>);
    /// Policy for best precision, which is the most precise and may be slower.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct BestPrecision<P: Policy>(PhantomData<P>);
    /// Policy for reference precision, which is the most precise and may be very slow.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct ReferencePrecision<P: Policy>(PhantomData<P>);

    /// Takes the precision of the second policy only if it is less than the first policy,
    /// but otherwise uses the first policy's other parameters.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct CmpLessPrecision<A: Policy, B: Policy>(PhantomData<(A, B)>);

    /// Optimize for performance at the cost of precision and safety (doesn't handle special cases such as NaNs or overflow).
    ///
    /// On instruction sets with FMA, this usually doesn't hurt precision too much, but will still avoid overflow/underflow checking,
    /// which can result in undefined behavior.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct UltraPerformance;

    /// Optimize for performance at the cost of safety, but try to keep some precision.
    ///
    /// This avoids checking for special cases such as NaNs or overflow, but will still try to
    /// provide a reasonable result for most inputs.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct HighPerformance;

    /// Optimize for performance, ideally without losing precision.
    ///
    /// This is the default policy for the non-policy-specific math functions,
    /// and tries to provide as much precision and performance as possible.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Performance;

    /// Optimize for precision, at the cost of performance if necessary.
    ///
    /// On instruction sets with FMA, performance may not be hurt too much.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Precision;

    /// Optimize for code size, avoids hard-coded equations or loop unrolling.
    ///
    /// Performance is not a priority for this policy.
    ///
    /// Best used in conjunction with `opt-level=z`
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Size;

    /// Calculates a reference value for operations where possible, which can be very expensive.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Reference;

    const fn extra_precision(p: PrecisionPolicy) -> PrecisionPolicy {
        match p {
            PrecisionPolicy::Worst => PrecisionPolicy::Medium,
            PrecisionPolicy::Medium => PrecisionPolicy::Average,
            PrecisionPolicy::Average => PrecisionPolicy::Best,
            PrecisionPolicy::Best => PrecisionPolicy::Reference,
            PrecisionPolicy::Reference => PrecisionPolicy::Reference, // no change
        }
    }

    const fn less_precision(p: PrecisionPolicy) -> PrecisionPolicy {
        match p {
            PrecisionPolicy::Reference => PrecisionPolicy::Best,
            PrecisionPolicy::Best => PrecisionPolicy::Average,
            PrecisionPolicy::Average => PrecisionPolicy::Medium,
            PrecisionPolicy::Medium => PrecisionPolicy::Worst,
            PrecisionPolicy::Worst => PrecisionPolicy::Worst, // no change
        }
    }

    impl<P: Policy> Policy for ExtraPrecision<P> {
        const POLICY: PolicyParameters = PolicyParameters {
            check_overflow: P::POLICY.check_overflow,
            unroll_loops: P::POLICY.unroll_loops,
            precision: extra_precision(P::POLICY.precision),
            avoid_branching: P::POLICY.avoid_branching,
            max_iterations: P::POLICY.max_iterations,
            use_compensation: P::POLICY.precision.ge(PrecisionPolicy::Average),
            denormal_behavior: P::POLICY.denormal_behavior,
        };
    }

    impl<P: Policy> Policy for LessPrecision<P> {
        const POLICY: PolicyParameters = PolicyParameters {
            check_overflow: P::POLICY.check_overflow,
            unroll_loops: P::POLICY.unroll_loops,
            precision: less_precision(P::POLICY.precision),
            avoid_branching: P::POLICY.avoid_branching,
            max_iterations: P::POLICY.max_iterations,
            use_compensation: P::POLICY.precision.gt(PrecisionPolicy::Average),
            denormal_behavior: P::POLICY.denormal_behavior,
        };
    }

    impl<P: Policy, const USE_COMPENSATION: bool> Policy for UseCompensation<P, USE_COMPENSATION> {
        const POLICY: PolicyParameters = PolicyParameters {
            check_overflow: P::POLICY.check_overflow,
            unroll_loops: P::POLICY.unroll_loops,
            precision: P::POLICY.precision,
            avoid_branching: P::POLICY.avoid_branching,
            max_iterations: P::POLICY.max_iterations,
            use_compensation: USE_COMPENSATION,
            denormal_behavior: P::POLICY.denormal_behavior,
        };
    }

    impl<P: Policy, const CHECK_OVERFLOW: bool> Policy for CheckOverflow<P, CHECK_OVERFLOW> {
        const POLICY: PolicyParameters = PolicyParameters {
            check_overflow: CHECK_OVERFLOW,
            unroll_loops: P::POLICY.unroll_loops,
            precision: P::POLICY.precision,
            avoid_branching: P::POLICY.avoid_branching,
            max_iterations: P::POLICY.max_iterations,
            use_compensation: P::POLICY.use_compensation,
            denormal_behavior: P::POLICY.denormal_behavior,
        };
    }

    impl<P: Policy, const UNROLL_LOOPS: bool> Policy for UnrollLoops<P, UNROLL_LOOPS> {
        const POLICY: PolicyParameters = PolicyParameters {
            check_overflow: P::POLICY.check_overflow,
            unroll_loops: UNROLL_LOOPS,
            precision: P::POLICY.precision,
            avoid_branching: P::POLICY.avoid_branching,
            max_iterations: P::POLICY.max_iterations,
            use_compensation: P::POLICY.use_compensation,
            denormal_behavior: P::POLICY.denormal_behavior,
        };
    }

    impl<P: Policy, const AVOID_BRANCHING: bool> Policy for AvoidBranching<P, AVOID_BRANCHING> {
        const POLICY: PolicyParameters = PolicyParameters {
            check_overflow: P::POLICY.check_overflow,
            unroll_loops: P::POLICY.unroll_loops,
            precision: P::POLICY.precision,
            avoid_branching: AVOID_BRANCHING,
            max_iterations: P::POLICY.max_iterations,
            use_compensation: P::POLICY.use_compensation,
            denormal_behavior: P::POLICY.denormal_behavior,
        };
    }

    impl<P: Policy, const MAX_ITERATIONS: usize> Policy for MaxIterations<P, MAX_ITERATIONS> {
        const POLICY: PolicyParameters = PolicyParameters {
            check_overflow: P::POLICY.check_overflow,
            unroll_loops: P::POLICY.unroll_loops,
            precision: P::POLICY.precision,
            avoid_branching: P::POLICY.avoid_branching,
            max_iterations: MAX_ITERATIONS,
            use_compensation: P::POLICY.use_compensation,
            denormal_behavior: P::POLICY.denormal_behavior,
        };
    }

    impl<P: Policy> Policy for WorstPrecision<P> {
        const POLICY: PolicyParameters = PolicyParameters {
            check_overflow: P::POLICY.check_overflow,
            unroll_loops: P::POLICY.unroll_loops,
            precision: PrecisionPolicy::Worst,
            avoid_branching: P::POLICY.avoid_branching,
            max_iterations: P::POLICY.max_iterations,
            use_compensation: false,
            denormal_behavior: P::POLICY.denormal_behavior,
        };
    }

    impl<P: Policy> Policy for MediumPrecision<P> {
        const POLICY: PolicyParameters = PolicyParameters {
            check_overflow: P::POLICY.check_overflow,
            unroll_loops: P::POLICY.unroll_loops,
            precision: PrecisionPolicy::Medium,
            avoid_branching: P::POLICY.avoid_branching,
            max_iterations: P::POLICY.max_iterations,
            use_compensation: false,
            denormal_behavior: P::POLICY.denormal_behavior,
        };
    }

    impl<P: Policy> Policy for AveragePrecision<P> {
        const POLICY: PolicyParameters = PolicyParameters {
            check_overflow: P::POLICY.check_overflow,
            unroll_loops: P::POLICY.unroll_loops,
            precision: PrecisionPolicy::Average,
            avoid_branching: P::POLICY.avoid_branching,
            max_iterations: P::POLICY.max_iterations,
            use_compensation: false,
            denormal_behavior: P::POLICY.denormal_behavior,
        };
    }

    impl<P: Policy> Policy for BestPrecision<P> {
        const POLICY: PolicyParameters = PolicyParameters {
            check_overflow: P::POLICY.check_overflow,
            unroll_loops: P::POLICY.unroll_loops,
            precision: PrecisionPolicy::Best,
            avoid_branching: P::POLICY.avoid_branching,
            max_iterations: P::POLICY.max_iterations,
            use_compensation: true,
            denormal_behavior: P::POLICY.denormal_behavior,
        };
    }

    impl<P: Policy> Policy for ReferencePrecision<P> {
        const POLICY: PolicyParameters = PolicyParameters {
            check_overflow: P::POLICY.check_overflow,
            unroll_loops: P::POLICY.unroll_loops,
            precision: PrecisionPolicy::Reference,
            avoid_branching: P::POLICY.avoid_branching,
            max_iterations: P::POLICY.max_iterations,
            use_compensation: true,
            denormal_behavior: P::POLICY.denormal_behavior,
        };
    }

    impl<A: Policy, B: Policy> Policy for CmpLessPrecision<A, B> {
        const POLICY: PolicyParameters = PolicyParameters {
            check_overflow: A::POLICY.check_overflow,
            unroll_loops: A::POLICY.unroll_loops,
            precision: if B::POLICY.precision.lt(A::POLICY.precision) {
                B::POLICY.precision
            } else {
                A::POLICY.precision
            },
            avoid_branching: A::POLICY.avoid_branching,
            max_iterations: A::POLICY.max_iterations,
            use_compensation: A::POLICY.use_compensation && B::POLICY.use_compensation,
            denormal_behavior: DenormalBehavior::preserve_any(A::POLICY.denormal_behavior, B::POLICY.denormal_behavior),
        };
    }

    impl Policy for UltraPerformance {
        const POLICY: PolicyParameters = PolicyParameters {
            check_overflow: false,
            unroll_loops: true,
            precision: PrecisionPolicy::Worst,
            avoid_branching: true,
            max_iterations: 1000,
            use_compensation: false,
            denormal_behavior: DenormalBehavior::select_default(true),
        };
    }

    impl Policy for HighPerformance {
        const POLICY: PolicyParameters = PolicyParameters {
            check_overflow: false,
            unroll_loops: true,
            precision: PrecisionPolicy::Medium,
            avoid_branching: false,
            max_iterations: 10000,
            use_compensation: false,
            denormal_behavior: DenormalBehavior::select_default(true),
        };
    }

    impl Policy for Performance {
        const POLICY: PolicyParameters = PolicyParameters {
            check_overflow: true,
            unroll_loops: true,
            precision: PrecisionPolicy::Average,
            avoid_branching: false,
            max_iterations: 10000,
            use_compensation: false,
            denormal_behavior: DenormalBehavior::select_default(false),
        };
    }

    impl Policy for Precision {
        const POLICY: PolicyParameters = PolicyParameters {
            check_overflow: true,
            unroll_loops: true,
            precision: PrecisionPolicy::Best,
            avoid_branching: false,
            max_iterations: 50000,
            use_compensation: true,
            denormal_behavior: DenormalBehavior::select_default(false),
        };
    }

    impl Policy for Size {
        const POLICY: PolicyParameters = PolicyParameters {
            check_overflow: true,
            unroll_loops: false,
            precision: PrecisionPolicy::Average,

            // debatable, but for WASM it can't use
            // instruction-level parallelism anyway.
            avoid_branching: false,
            max_iterations: 10000,
            use_compensation: false,
            denormal_behavior: DenormalBehavior::select_default(true),
        };
    }

    impl Policy for Reference {
        const POLICY: PolicyParameters = PolicyParameters {
            check_overflow: true,
            unroll_loops: true,
            precision: PrecisionPolicy::Reference,
            avoid_branching: false,
            max_iterations: 100000,
            use_compensation: true,
            denormal_behavior: DenormalBehavior::select_default(false),
        };
    }
}

use policies::*;

#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
pub type DefaultPolicy = Size;

#[cfg(not(all(feature = "wasm", target_arch = "wasm32")))]
pub type DefaultPolicy = Performance;
