//! Differential test harness for Thermite.
//!
//! The strategy is *differential testing*: every concrete SIMD backend
//! (`X86V2`, `X86V3`, ...) is run against the `Scalar` backend, which is the
//! reference implementation that operates element-by-element on primitive
//! types. Identical inputs (a mix of hand-picked edge cases and deterministic
//! pseudo-random values) are fed to both, and the resulting lanes are compared
//! with an op-appropriate tolerance.
//!
//! This is the same philosophy the `diff_swizzle` tests use (`scalar_*` as
//! ground truth), generalised to the whole register API and parametrised over
//! the backend × element-type × width matrix.
//!
//! Used by:
//!   - `diff_ops.rs`     - arithmetic / bitwise / shift / compare / rounding
//!   - `diff_math.rs`    - transcendental math vs `libm`
//!
//! Run everything with `cargo test -p thermite`. The differential suites are
//! `#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]` because the
//! reference vs. SIMD comparison only makes sense where the SIMD backends are
//! compiled in.

#![allow(dead_code)]

use generic_array::{GenericArray, sequence::GenericSequence, typenum::Unsigned};
use rand::{RngExt, rngs::SmallRng};

use thermite::register::{CoreRegister, MaskRegister, Register, Storage};

/// Comparison tolerance for a single op.
#[derive(Clone, Copy, Debug)]
pub enum Tol {
    /// Bit-exact. `NaN` compares equal to `NaN` (any payload); signed zeros
    /// must match exactly.
    Exact,
    /// Within `n` units in the last place. `NaN`/`Inf` must still match exactly.
    Ulp(i64),
    /// `|a - b| <= eps * max(1, |b|)` once cast to `f64`. For approximate ops
    /// (`rcp`, `rsqrt`) and order-sensitive reductions.
    Rel(f64),
    /// Bit-exact, but any lane where *either* side is `NaN` is ignored. Used
    /// for `min`/`max`, whose `NaN` propagation is explicitly *not* IEEE and is
    /// known to diverge between the scalar and x86 backends.
    ExactOrNan,
    /// Bit-exact, except that `+0.0` and `-0.0` compare equal.
    ///
    /// For the `_c` variants of the additive float ops. Where the mask register
    /// is the same width as the data register, `op_c` is lowered as
    /// `op(lhs, rhs & mask)` rather than a `blendv` of the result - an `and`
    /// plus the op instead of the op plus a select. `+0.0` is not quite the
    /// additive identity under round-to-nearest (`-0.0 + 0.0 == +0.0`), so a
    /// masked-off `-0.0` lane comes back as `+0.0`. The scalar oracle has no
    /// equal-size mask and keeps the `blendv` form, so it preserves the sign.
    /// That divergence is accepted; the magnitude is not.
    ExactOrZeroSign,
}

/// A scalar element type that can be differentially tested: it knows its own
/// interesting edge cases, how to draw a random value, and how to compare two
/// results under a [`Tol`].
pub trait Diff: Copy + core::fmt::Debug + 'static {
    /// Hand-picked edge cases (zeros, signs, infinities, limits, ...).
    fn edges() -> &'static [Self];
    /// A random value drawn from a wide, representative distribution.
    fn rand(rng: &mut SmallRng) -> Self;
    /// Does `got` match `want` under `tol`?
    fn close(got: Self, want: Self, tol: Tol) -> bool;
    /// Finite (not `NaN`/`Inf`). Always true for integers.
    fn finite(self) -> bool {
        true
    }
}

macro_rules! impl_diff_float {
    ($t:ty, $bits:ty, $ibits:ty) => {
        impl Diff for $t {
            fn edges() -> &'static [Self] {
                &[
                    0.0,
                    -0.0,
                    1.0,
                    -1.0,
                    2.0,
                    -2.0,
                    0.5,
                    -0.5,
                    <$t>::MIN,
                    <$t>::MAX,
                    <$t>::MIN_POSITIVE,
                    -<$t>::MIN_POSITIVE,
                    <$t>::EPSILON,
                    <$t>::INFINITY,
                    <$t>::NEG_INFINITY,
                    <$t>::NAN,
                    3.0,
                    -3.0,
                    0.1,
                    -0.1,
                    1e9,
                    -1e9,
                    1e-9,
                    -1e-9,
                    core::f64::consts::PI as $t,
                    -core::f64::consts::PI as $t,
                    core::f64::consts::E as $t,
                ]
            }
            fn finite(self) -> bool {
                <$t>::is_finite(self)
            }
            fn rand(rng: &mut SmallRng) -> Self {
                // Mix of "normal-ish" values and full-range bit patterns so we
                // hit denormals, huge magnitudes and odd exponents too.
                if rng.random::<bool>() {
                    let m: $t = rng.random_range(-1000.0..1000.0);
                    let e: i32 = rng.random_range(-30..30);
                    m * (2.0 as $t).powi(e)
                } else {
                    <$t>::from_bits(rng.random::<$bits>())
                }
            }
            fn close(got: Self, want: Self, tol: Tol) -> bool {
                if let Tol::ExactOrNan = tol {
                    if got.is_nan() || want.is_nan() {
                        return true;
                    }
                }
                if got.is_nan() || want.is_nan() {
                    return got.is_nan() && want.is_nan();
                }
                match tol {
                    Tol::Exact => got.to_bits() == want.to_bits(),
                    // Relaxed (non-strict) min/max may return +0.0 or -0.0 for an
                    // opposite-signed-zero input pair (impl-defined - e.g. wasm
                    // `f32x4_relaxed_min`); accept either, unless `strict_ieee754`
                    // pins the deterministic result. (`Exact`, used by copysign /
                    // signum, still distinguishes the sign of zero.)
                    Tol::ExactOrNan => {
                        (!cfg!(feature = "strict_ieee754") && got == 0.0 && want == 0.0)
                            || got.to_bits() == want.to_bits()
                    }
                    // `got == want` is already sign-of-zero-blind for floats.
                    Tol::ExactOrZeroSign => (got == 0.0 && want == 0.0) || got.to_bits() == want.to_bits(),
                    Tol::Ulp(n) => {
                        if got == want {
                            return true;
                        }
                        if got.is_infinite() || want.is_infinite() {
                            return got == want;
                        }
                        // Map to a monotonic ordering so ULP distance is well
                        // defined across the sign boundary.
                        let order = |x: $t| -> $ibits {
                            let b = x.to_bits() as $ibits;
                            if b < 0 { <$ibits>::MIN - b } else { b }
                        };
                        (order(got) - order(want)).unsigned_abs() as i64 <= n
                    }
                    Tol::Rel(eps) => {
                        if got == want {
                            return true;
                        }
                        if got.is_infinite() || want.is_infinite() {
                            return got == want;
                        }
                        let (g, w) = (got as f64, want as f64);
                        (g - w).abs() <= eps * w.abs().max(1.0)
                    }
                }
            }
        }
    };
}

impl_diff_float!(f32, u32, i32);
impl_diff_float!(f64, u64, i64);

macro_rules! impl_diff_int {
    ($t:ty) => {
        impl Diff for $t {
            fn edges() -> &'static [Self] {
                &[
                    0,
                    1,
                    2,
                    3,
                    7,
                    42,
                    <$t>::MAX,
                    <$t>::MIN,
                    <$t>::MAX - 1,
                    <$t>::MIN + 1,
                    <$t>::MAX / 2,
                ]
            }
            fn rand(rng: &mut SmallRng) -> Self {
                rng.random::<$t>()
            }
            fn close(got: Self, want: Self, tol: Tol) -> bool {
                match tol {
                    Tol::Exact => got == want,
                    // Integers only ever use Exact; treat the rest as exact too.
                    _ => got == want,
                }
            }
        }
    };
}

impl_diff_int!(i8);
impl_diff_int!(u8);
impl_diff_int!(i16);
impl_diff_int!(u16);
impl_diff_int!(i32);
impl_diff_int!(u32);
impl_diff_int!(i64);
impl_diff_int!(u64);

/// How many random trials each differential test runs (in addition to the
/// exhaustive edge-case sweep).
pub const TRIALS: usize = 4096;

/// Build a deterministic RNG. Mirrors the helper the swizzle test relies on.
pub fn rng() -> SmallRng {
    rand::make_rng()
}

/// One input array of exactly `LANES` elements.
pub fn make_array<R>(values: &[R::Element]) -> Storage<R>
where
    R: Register,
    R::Element: Copy,
{
    let lanes = <R::Lanes as Unsigned>::USIZE;
    debug_assert!(values.len() >= lanes);
    let arr: GenericArray<R::Element, R::Lanes> = GenericArray::generate(|i| values[i]);
    R::new(arr)
}

/// Read a register's lanes back as a `Vec`.
pub fn read<R>(storage: &Storage<R>) -> Vec<R::Element>
where
    R: Register,
    R::Element: Copy,
{
    R::as_slice(storage).to_vec()
}

/// Build a mask register from a known boolean pattern (`bools.len()` must be
/// at least `LANES`). Shared by the mask / comparison / predicate suites.
pub fn build_mask<R: Register>(bools: &[bool]) -> Storage<R::Mask> {
    let arr: GenericArray<bool, <R::Mask as CoreRegister>::Lanes> = GenericArray::generate(|i| bools[i]);
    <R::Mask as MaskRegister>::new_mask(arr)
}

/// Read a mask register back into a `Vec<bool>`, lane by lane.
pub fn read_mask<R: Register>(mask: Storage<R::Mask>, lanes: usize) -> Vec<bool> {
    (0..lanes).map(|i| <R::Mask as MaskRegister>::test(mask, i)).collect()
}

/// A corpus of boolean mask patterns: all-true, all-false, the two alternating
/// patterns, then deterministic random fills (one per `count` so each value
/// array in a differential loop can get its own mask).
pub fn mask_patterns(lanes: usize, count: usize, rng: &mut SmallRng) -> Vec<Vec<bool>> {
    let mut out: Vec<Vec<bool>> = vec![
        vec![true; lanes],
        vec![false; lanes],
        (0..lanes).map(|i| i % 2 == 0).collect(),
        (0..lanes).map(|i| i % 2 == 1).collect(),
    ];
    while out.len() < count {
        out.push((0..lanes).map(|_| rng.random::<bool>()).collect());
    }
    out
}

/// Generate the full corpus of input arrays for `lanes` lanes: an exhaustive
/// sweep over the cartesian-ish product of edge cases, followed by `TRIALS`
/// fully random arrays.
pub fn corpus<E: Diff>(lanes: usize, rng: &mut SmallRng) -> Vec<Vec<E>> {
    let edges = E::edges();
    let mut out: Vec<Vec<E>> = Vec::new();

    // Every edge value broadcast to all lanes.
    for &e in edges {
        out.push(vec![e; lanes]);
    }
    // Sliding window over the edge list so adjacent lanes differ - this
    // catches lane-routing / horizontal-op bugs the broadcasts miss.
    for start in 0..edges.len() {
        out.push((0..lanes).map(|i| edges[(start + i) % edges.len()]).collect());
    }
    // Random arrays, plus some arrays that splice an edge into a random one.
    for t in 0..TRIALS {
        let mut a: Vec<E> = (0..lanes).map(|_| E::rand(rng)).collect();
        if t % 4 == 0 {
            let idx = rng.random_range(0..lanes);
            a[idx] = edges[rng.random_range(0..edges.len())];
        }
        out.push(a);
    }
    out
}

/// Compare two lane slices, panicking with a precise diagnostic on the first
/// mismatch. `label` identifies the op + backend + type under test.
pub fn assert_lanes_eq<E: Diff>(label: &str, inputs: &[&[E]], got: &[E], want: &[E], tol: Tol) {
    assert_eq!(got.len(), want.len(), "{label}: lane count mismatch");
    for (lane, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        if !E::close(g, w, tol) {
            let mut ins = String::new();
            for (n, input) in inputs.iter().enumerate() {
                ins.push_str(&format!(
                    "\n  in{n}[{lane}] = {:?}\n  in{n} (full) = {input:?}",
                    input[lane]
                ));
            }
            panic!(
                "{label}: lane {lane} mismatch (tol = {tol:?})\n  got  = {g:?}\n  want = {w:?}{ins}\n  full got  = {got:?}\n  full want = {want:?}"
            );
        }
    }
}

/// Stamp a differential test for a **unary** register op.
///
/// `$ut` is the backend register under test, `$rf` the scalar reference
/// register (same `Element` and `Lanes`). `$method` is the trait method name
/// shared by both (e.g. `sqrt`, `neg`, `not`).
#[macro_export]
macro_rules! diff_unary {
    ($label:expr, $ut:ty, $rf:ty, $method:ident, $tol:expr) => {{
        let mut rng = $crate::harness::rng();
        type E = <$ut as ::thermite::register::Register>::Element;
        let lanes = <<$ut as ::thermite::register::CoreRegister>::Lanes as ::generic_array::typenum::Unsigned>::USIZE;
        for input in $crate::harness::corpus::<E>(lanes, &mut rng) {
            let a_ut = $crate::harness::make_array::<$ut>(&input);
            let a_rf = $crate::harness::make_array::<$rf>(&input);
            let got = $crate::harness::read::<$ut>(&<$ut>::$method(a_ut));
            let want = $crate::harness::read::<$rf>(&<$rf>::$method(a_rf));
            $crate::harness::assert_lanes_eq(
                concat!($label, " [", stringify!($method), "]"),
                &[input.as_slice()],
                &got,
                &want,
                $tol,
            );
        }
    }};
}

/// Stamp a differential test for a **binary** register op.
#[macro_export]
macro_rules! diff_binary {
    ($label:expr, $ut:ty, $rf:ty, $method:ident, $tol:expr) => {{
        let mut rng = $crate::harness::rng();
        type E = <$ut as ::thermite::register::Register>::Element;
        let lanes = <<$ut as ::thermite::register::CoreRegister>::Lanes as ::generic_array::typenum::Unsigned>::USIZE;
        let xs = $crate::harness::corpus::<E>(lanes, &mut rng);
        let ys = $crate::harness::corpus::<E>(lanes, &mut rng);
        for (x, y) in xs.iter().zip(ys.iter()) {
            let got = $crate::harness::read::<$ut>(&<$ut>::$method(
                $crate::harness::make_array::<$ut>(x),
                $crate::harness::make_array::<$ut>(y),
            ));
            let want = $crate::harness::read::<$rf>(&<$rf>::$method(
                $crate::harness::make_array::<$rf>(x),
                $crate::harness::make_array::<$rf>(y),
            ));
            $crate::harness::assert_lanes_eq(
                concat!($label, " [", stringify!($method), "]"),
                &[x.as_slice(), y.as_slice()],
                &got,
                &want,
                $tol,
            );
        }
    }};
}

/// Stamp a differential test for a **horizontal reduction** (`Storage -> Element`).
#[macro_export]
macro_rules! diff_reduce {
    ($label:expr, $ut:ty, $rf:ty, $method:ident, $tol:expr) => {{
        let mut rng = $crate::harness::rng();
        type E = <$ut as ::thermite::register::Register>::Element;
        let lanes = <<$ut as ::thermite::register::CoreRegister>::Lanes as ::generic_array::typenum::Unsigned>::USIZE;
        for input in $crate::harness::corpus::<E>(lanes, &mut rng) {
            let got = <$ut>::$method($crate::harness::make_array::<$ut>(&input));
            let want = <$rf>::$method($crate::harness::make_array::<$rf>(&input));
            $crate::harness::assert_lanes_eq(
                concat!($label, " [", stringify!($method), "]"),
                &[input.as_slice()],
                &[got],
                &[want],
                $tol,
            );
        }
    }};
}

/// Like [`diff_binary!`], but skips any input array containing a non-finite
/// element. Used for `min`/`max`, whose `NaN`/`Inf` behaviour is explicitly
/// non-IEEE and configurable via the `strict_ieee754` feature.
#[macro_export]
macro_rules! diff_binary_finite {
    ($label:expr, $ut:ty, $rf:ty, $method:ident, $tol:expr) => {{
        use $crate::harness::Diff as _;
        let mut rng = $crate::harness::rng();
        type E = <$ut as ::thermite::register::Register>::Element;
        let lanes = <<$ut as ::thermite::register::CoreRegister>::Lanes as ::generic_array::typenum::Unsigned>::USIZE;
        let xs = $crate::harness::corpus::<E>(lanes, &mut rng);
        let ys = $crate::harness::corpus::<E>(lanes, &mut rng);
        for (x, y) in xs.iter().zip(ys.iter()) {
            if x.iter().chain(y.iter()).any(|v| !v.finite()) {
                continue;
            }
            let got = $crate::harness::read::<$ut>(&<$ut>::$method(
                $crate::harness::make_array::<$ut>(x),
                $crate::harness::make_array::<$ut>(y),
            ));
            let want = $crate::harness::read::<$rf>(&<$rf>::$method(
                $crate::harness::make_array::<$rf>(x),
                $crate::harness::make_array::<$rf>(y),
            ));
            $crate::harness::assert_lanes_eq(
                concat!($label, " [", stringify!($method), "]"),
                &[x.as_slice(), y.as_slice()],
                &got,
                &want,
                $tol,
            );
        }
    }};
}

/// Like [`diff_reduce!`], but skips any input array containing a non-finite
/// element (see [`diff_binary_finite!`]).
#[macro_export]
macro_rules! diff_reduce_finite {
    ($label:expr, $ut:ty, $rf:ty, $method:ident, $tol:expr) => {{
        use $crate::harness::Diff as _;
        let mut rng = $crate::harness::rng();
        type E = <$ut as ::thermite::register::Register>::Element;
        let lanes = <<$ut as ::thermite::register::CoreRegister>::Lanes as ::generic_array::typenum::Unsigned>::USIZE;
        for input in $crate::harness::corpus::<E>(lanes, &mut rng) {
            if input.iter().any(|v| !v.finite()) {
                continue;
            }
            let got = <$ut>::$method($crate::harness::make_array::<$ut>(&input));
            let want = <$rf>::$method($crate::harness::make_array::<$rf>(&input));
            $crate::harness::assert_lanes_eq(
                concat!($label, " [", stringify!($method), "]"),
                &[input.as_slice()],
                &[got],
                &[want],
                $tol,
            );
        }
    }};
}

// ===========================================================================
// Independent-oracle macros. Instead of differencing against the `Scalar`
// backend (which may share a generic polyfill with the SIMD backend and so
// mask a shared bug), these compare against a pure-Rust closure. This is the
// strongest test for *polyfill-backed* register ops.
// ===========================================================================

/// Backend unary op vs. a `Fn($elem) -> $elem` Rust oracle.
#[macro_export]
macro_rules! oracle_unary {
    ($label:expr, $ut:ty, $elem:ty, $method:ident, $oracle:expr, $tol:expr) => {{
        let mut rng = $crate::harness::rng();
        let lanes = <<$ut as ::thermite::register::CoreRegister>::Lanes as ::generic_array::typenum::Unsigned>::USIZE;
        let oracle: fn($elem) -> $elem = $oracle;
        for input in $crate::harness::corpus::<$elem>(lanes, &mut rng) {
            let got = $crate::harness::read::<$ut>(&<$ut>::$method($crate::harness::make_array::<$ut>(&input)));
            let want: Vec<$elem> = input.iter().map(|&x| oracle(x)).collect();
            $crate::harness::assert_lanes_eq(
                concat!($label, " [", stringify!($method), " vs Rust]"),
                &[input.as_slice()],
                &got,
                &want,
                $tol,
            );
        }
    }};
}

/// Backend binary op vs. a `Fn($elem, $elem) -> $elem` Rust oracle.
#[macro_export]
macro_rules! oracle_binary {
    ($label:expr, $ut:ty, $elem:ty, $method:ident, $oracle:expr, $tol:expr) => {{
        let mut rng = $crate::harness::rng();
        let lanes = <<$ut as ::thermite::register::CoreRegister>::Lanes as ::generic_array::typenum::Unsigned>::USIZE;
        let oracle: fn($elem, $elem) -> $elem = $oracle;
        let xs = $crate::harness::corpus::<$elem>(lanes, &mut rng);
        let ys = $crate::harness::corpus::<$elem>(lanes, &mut rng);
        for (x, y) in xs.iter().zip(ys.iter()) {
            let got = $crate::harness::read::<$ut>(&<$ut>::$method(
                $crate::harness::make_array::<$ut>(x),
                $crate::harness::make_array::<$ut>(y),
            ));
            let want: Vec<$elem> = x.iter().zip(y.iter()).map(|(&a, &b)| oracle(a, b)).collect();
            $crate::harness::assert_lanes_eq(
                concat!($label, " [", stringify!($method), " vs Rust]"),
                &[x.as_slice(), y.as_slice()],
                &got,
                &want,
                $tol,
            );
        }
    }};
}

/// Backend `fn(Storage, u32) -> Storage` shift/rotate op vs. a
/// `Fn($elem, u32) -> $elem` Rust oracle, swept over every shift amount.
#[macro_export]
macro_rules! oracle_shift {
    ($label:expr, $ut:ty, $elem:ty, $method:ident, $oracle:expr) => {{
        let mut rng = $crate::harness::rng();
        let lanes = <<$ut as ::thermite::register::CoreRegister>::Lanes as ::generic_array::typenum::Unsigned>::USIZE;
        let bits = (core::mem::size_of::<$elem>() * 8) as u32;
        let oracle: fn($elem, u32) -> $elem = $oracle;
        for input in $crate::harness::corpus::<$elem>(lanes, &mut rng) {
            for sh in 0..bits {
                let got = $crate::harness::read::<$ut>(&<$ut>::$method($crate::harness::make_array::<$ut>(&input), sh));
                let want: Vec<$elem> = input.iter().map(|&x| oracle(x, sh)).collect();
                $crate::harness::assert_lanes_eq(
                    concat!($label, " [", stringify!($method), " vs Rust]"),
                    &[input.as_slice()],
                    &got,
                    &want,
                    $crate::harness::Tol::Exact,
                );
            }
        }
    }};
}

/// Differential test for a float -> int `saturating_cast`
/// (`<Dst as CastRegister<Src>>`) against the scalar backend, whose
/// float -> int saturating impl is literally `value as _`. Unlike `cast_diff!`
/// there is no domain prep: saturating casts are total (`as` semantics:
/// NaN -> 0, out-of-range clamps), so the raw corpus - NaN, infinities, and
/// out-of-range values included - is valid input and must be bit-exact.
#[macro_export]
macro_rules! sat_cast_diff {
    ($label:expr, $src_ut:ty, $dst_ut:ty, $src_rf:ty, $dst_rf:ty, $se:ty) => {{
        use ::thermite::register::CastRegister;
        let mut rng = $crate::harness::rng();
        let lanes =
            <<$src_ut as ::thermite::register::CoreRegister>::Lanes as ::generic_array::typenum::Unsigned>::USIZE;
        for input in $crate::harness::corpus::<$se>(lanes, &mut rng) {
            let got = $crate::harness::read::<$dst_ut>(&<$dst_ut as CastRegister<$src_ut>>::saturating_cast_from(
                $crate::harness::make_array::<$src_ut>(&input),
            ));
            let want = $crate::harness::read::<$dst_rf>(&<$dst_rf as CastRegister<$src_rf>>::saturating_cast_from(
                $crate::harness::make_array::<$src_rf>(&input),
            ));
            $crate::harness::assert_lanes_eq(
                concat!($label, " [saturating_cast vs scalar `as`]"),
                &[],
                &got,
                &want,
                Tol::Exact,
            );
        }
    }};
}

/// Differential test for `fast_cast` (`<Dst as CastRegister<Src>>::fast_cast_from`).
///
/// `fast_cast` is the deliberately-narrow-domain conversion: out of range it
/// returns unspecified values by contract, so `$prep` must map the corpus into
/// the domain where it *is* defined - and, for float sources, onto integral
/// values, because the magic-number lowerings round to nearest where `as`
/// truncates.
///
/// The scalar backend is the oracle and is a genuinely independent one here: it
/// has no `fast_cast` override at all, so it falls through to `cast_from`, which
/// is plain `as`. It cannot share a bug with the magic-number path under test.
#[macro_export]
macro_rules! fast_cast_diff {
    ($label:expr, $src_ut:ty, $dst_ut:ty, $src_rf:ty, $dst_rf:ty, $se:ty, $prep:expr) => {{
        use ::thermite::register::CastRegister;
        let mut rng = $crate::harness::rng();
        let lanes =
            <<$src_ut as ::thermite::register::CoreRegister>::Lanes as ::generic_array::typenum::Unsigned>::USIZE;
        let prep: fn($se) -> $se = $prep;
        for raw in $crate::harness::corpus::<$se>(lanes, &mut rng) {
            let input: Vec<$se> = raw.iter().map(|&x| prep(x)).collect();
            let got = $crate::harness::read::<$dst_ut>(&<$dst_ut as CastRegister<$src_ut>>::fast_cast_from(
                $crate::harness::make_array::<$src_ut>(&input),
            ));
            let want = $crate::harness::read::<$dst_rf>(&<$dst_rf as CastRegister<$src_rf>>::cast_from(
                $crate::harness::make_array::<$src_rf>(&input),
            ));
            $crate::harness::assert_lanes_eq(
                concat!($label, " [fast_cast vs scalar `as`]"),
                &[],
                &got,
                &want,
                Tol::Exact,
            );
        }
    }};
}

/// Differential test for a numeric `cast` (`<Dst as CastRegister<Src>>`).
///
/// The scalar backend's `cast_from` is literally `value as _`, so this is a
/// differential against Rust's built-in `as` (the documented "like `as`"
/// contract). `$prep` maps each source element before the cast (applied
/// identically to both backends) so float→int gates can stay in the
/// in-range domain where the contract is unambiguous; pass `|x| x` otherwise.
#[macro_export]
macro_rules! cast_diff {
    ($label:expr, $src_ut:ty, $dst_ut:ty, $src_rf:ty, $dst_rf:ty, $se:ty, $prep:expr, $tol:expr) => {{
        use ::thermite::register::CastRegister;
        let mut rng = $crate::harness::rng();
        let lanes =
            <<$src_ut as ::thermite::register::CoreRegister>::Lanes as ::generic_array::typenum::Unsigned>::USIZE;
        let prep: fn($se) -> $se = $prep;
        for raw in $crate::harness::corpus::<$se>(lanes, &mut rng) {
            let input: Vec<$se> = raw.iter().map(|&x| prep(x)).collect();
            let got = $crate::harness::read::<$dst_ut>(&<$dst_ut as CastRegister<$src_ut>>::cast_from(
                $crate::harness::make_array::<$src_ut>(&input),
            ));
            let want = $crate::harness::read::<$dst_rf>(&<$dst_rf as CastRegister<$src_rf>>::cast_from(
                $crate::harness::make_array::<$src_rf>(&input),
            ));
            $crate::harness::assert_lanes_eq(concat!($label, " [cast vs scalar `as`]"), &[], &got, &want, $tol);
        }
    }};
}

/// Stamp a differential test for a same-size `BitCastRegister::from_bits` (a byte
/// reinterpret, e.g. `i8 <-> u8`). The SIMD `from_bits` is compared against the scalar
/// backend's, which is the same byte reinterpret, so the result is bit-exact. `$se` is the
/// SOURCE element type the corpus is generated over.
#[macro_export]
macro_rules! bitcast_diff {
    ($label:expr, $src_ut:ty, $dst_ut:ty, $src_rf:ty, $dst_rf:ty, $se:ty) => {{
        use ::thermite::register::BitCastRegister;
        let mut rng = $crate::harness::rng();
        let lanes =
            <<$src_ut as ::thermite::register::CoreRegister>::Lanes as ::generic_array::typenum::Unsigned>::USIZE;
        for input in $crate::harness::corpus::<$se>(lanes, &mut rng) {
            let got = $crate::harness::read::<$dst_ut>(&<$dst_ut as BitCastRegister<$src_ut>>::from_bits(
                $crate::harness::make_array::<$src_ut>(&input),
            ));
            let want = $crate::harness::read::<$dst_rf>(&<$dst_rf as BitCastRegister<$src_rf>>::from_bits(
                $crate::harness::make_array::<$src_rf>(&input),
            ));
            $crate::harness::assert_lanes_eq(concat!($label, " [bitcast vs scalar]"), &[], &got, &want, Tol::Exact);
        }
    }};
}
