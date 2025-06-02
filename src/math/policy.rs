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
    pub max_series_iterations: usize,
}

impl PolicyParameters {
    /// Returns true if the policy says to avoid branches at the cost of precision
    #[inline(always)]
    pub const fn avoid_precision_branches(self) -> bool {
        // cheat the const-comparison here by casting to u8
        self.avoid_branching && self.precision as u8 == PrecisionPolicy::Worst as u8
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
        max_series_iterations: 10000,
    };
}

let y = x.cbrt_p::<MyPolicy>();
```
*/
pub mod policies {
    use core::marker::PhantomData;

    use super::{Policy, PolicyParameters, PrecisionPolicy};

    /// Policy adapter that increases the precision requires by one level,
    /// e.g.: `Worst` -> `Average`, `Average` -> `Best`
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct ExtraPrecision<P: Policy>(PhantomData<P>);

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
    /// This is the default policy for [`SimdVectorizedMath`](super::SimdVectorizedMath),
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
            max_series_iterations: P::POLICY.max_series_iterations,
        };
    }

    impl Policy for UltraPerformance {
        const POLICY: PolicyParameters = PolicyParameters {
            check_overflow: false,
            unroll_loops: true,
            precision: PrecisionPolicy::Worst,
            avoid_branching: true,
            max_series_iterations: 1000,
        };
    }

    impl Policy for HighPerformance {
        const POLICY: PolicyParameters = PolicyParameters {
            check_overflow: false,
            unroll_loops: true,
            precision: PrecisionPolicy::Medium,
            avoid_branching: false,
            max_series_iterations: 10000,
        };
    }

    impl Policy for Performance {
        const POLICY: PolicyParameters = PolicyParameters {
            check_overflow: true,
            unroll_loops: true,
            precision: PrecisionPolicy::Average,
            avoid_branching: false,
            max_series_iterations: 10000,
        };
    }

    impl Policy for Precision {
        const POLICY: PolicyParameters = PolicyParameters {
            check_overflow: true,
            unroll_loops: true,
            precision: PrecisionPolicy::Best,
            avoid_branching: false,
            max_series_iterations: 50000,
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
            max_series_iterations: 10000,
        };
    }

    impl Policy for Reference {
        const POLICY: PolicyParameters = PolicyParameters {
            check_overflow: true,
            unroll_loops: true,
            precision: PrecisionPolicy::Reference,
            avoid_branching: false,
            max_series_iterations: 100000,
        };
    }
}

use policies::*;

pub type DefaultPolicy = Performance;
