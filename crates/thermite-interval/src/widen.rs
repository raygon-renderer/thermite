//! The widening policy: how outward rounding is performed, carried on the
//! [`Interval`](crate::Interval) type itself rather than per call.
//!
//! This is deliberately NOT thermite's math [`Policy`](thermite::math::policy::Policy).
//! The math policy tunes _algorithms_ (polynomial degree, range-reduction
//! effort), while the widening policy tunes the _interval bookkeeping_. They
//! are orthogonal: a loose `sin` approximation over `Tightest` intervals gives
//! a cheap value that is still tightly enclosed, because the enclosure absorbs
//! the algorithm's documented error bound while the endpoint arithmetic stays
//! tight.
//!
//! Carrying the policy in the type also gives the policy-less surfaces
//! (`core::ops` operators, `FloatVector::sqrt`, `MulAddExt`) a widening
//! discipline with no `DefaultPolicy` guesswork, and makes accidentally
//! mixing tiers a type error.
//!
//! The three tiers are the measured rows of the policy matrix
//! (llvm-mca + width numbers in `bin/interval_probe`):
//!
//! | tier       | add / sub | mul / fma            | character |
//! |------------|-----------|----------------------|-----------|
//! | `Fastest`  | scale     | scale                | 2x the throughput of bump, ~3x looser |
//! | `Balanced` | bump      | residual (FMA), else bump | best serial latency, sound at overflow for free |
//! | `Tightest` | residual  | residual (Veltkamp on non-FMA) | tightest representable, exact ops do not widen |

/// Selects the outward-rounding strategy for every operation on an
/// [`Interval`](crate::Interval) carrying it. See the module docs.
pub trait WideningPolicy: 'static + Sized + Send + Sync + Copy + Default + core::fmt::Debug {
    const TIER: WideningTier;
}

/// The three measured widening tiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WideningTier {
    /// Multiplicative eps-widening everywhere. Highest throughput for
    /// data-parallel workloads, ~3x wider enclosures.
    Fastest,
    /// Unconditional ulp-stepping for add/sub, error-free-transform residual
    /// widening for multiplies where hardware FMA makes the residual one
    /// instruction. The default.
    Balanced,
    /// Residual widening everywhere: the tightest representable enclosure of
    /// every primitive, and exact operations do not widen at all (degenerate
    /// intervals stay degenerate through exact chains). Uses the
    /// Veltkamp-split product on non-FMA hardware, which is slower there, but
    /// tightness is this tier's contract.
    Tightest,
}

/// Scale-widening everywhere: the data-parallel fast tier.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Fastest;

/// Bump add + residual mul: the default tier.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Balanced;

/// Residual everywhere: the verification tier.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Tightest;

impl WideningPolicy for Fastest {
    const TIER: WideningTier = WideningTier::Fastest;
}
impl WideningPolicy for Balanced {
    const TIER: WideningTier = WideningTier::Balanced;
}
impl WideningPolicy for Tightest {
    const TIER: WideningTier = WideningTier::Tightest;
}
