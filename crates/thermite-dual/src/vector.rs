//! Element and vector-trait integration for [`Dual`].
//!
//! - `Dual<E, N>` (where `E` is a scalar float element) implements
//!   [`Element`]/[`SignedElement`]/[`FloatElement`], so it can be the element
//!   type of a dual vector.
//! - `Dual<V, N>` (where `V` is a real [`FloatVector`]) implements the full
//!   [`GenericVector`] -> [`FloatVector`] stack, with `Element =
//!   Dual<V::Element, N>` and the same mask/lanes as `V`.
//!
//! ## Compile-time splats
//!
//! Building a `Dual` vector from a *compile-time constant* element (via
//! [`SplatConst`]/[`const_splat!`](thermite::const_splat) and the
//! [`NewVector`]/[`const_new!`](thermite::const_new) machinery) splats the
//! primal `re` correctly and sets the derivative parts to zero -- i.e. it treats
//! a compile-time constant as having zero derivative, which is the correct and
//! desired behaviour for every constant the library actually produces (`ZERO`,
//! `ONE`, `PI`, polynomial coefficients, ...). The runtime [`splat`] and [`new`]
//! constructors preserve derivative parts fully. (A non-array const path that
//! preserved arbitrary per-component derivatives would require const generic
//! recursion over `[V; N]`, which is unstable.)
//!
//! [`splat`]: GenericVector::splat
//! [`new`]: GenericVector::new

use core::marker::PhantomData;
use core::ops::{Add, Div, Mul, Rem, Sub};

use num_traits::Bounded;

use thermite::Swizzle;
use thermite::element::{Element, FloatElement, SignedElement};
use thermite::generic_array::{GenericArray, IntoArrayLength, typenum::Const};
use thermite::mask::{GenericMask, GenericSelectable};
use thermite::math::algorithms::reduce_in_place;
use thermite::register::SwizzleIndices;
use thermite::vector::ops::{AddSubExt, AddSubExtMasked, NegMasked, Square, SquareMasked};
use thermite::vector::{NewConst, NewVector, SplatConst, SplatVector, VectorValue, const_new, const_splat};
use thermite::{LargeInt, prelude::*};

use crate::{Dual, DualValue};

/// Build a `[_; $n]` array from a per-index expression without a closure, so it
/// always inlines under `#[target_feature]` -- unlike `core::array::from_fn` /
/// `array::map`, which are only `#[inline]` and can be left out-of-line (dropping
/// the target feature). The body is pasted directly into a `while` loop.
macro_rules! array_each {
    ([$init:expr; $n:expr], |$j:ident| $body:expr) => {{
        let mut out = [$init; $n];
        let mut $j = 0usize;
        while $j < $n {
            out[$j] = $body;
            $j += 1;
        }
        out
    }};
}

/// A real [`FloatVector`] usable as the inner storage of a [`Dual`] vector.
///
/// Requires the value type to support dual arithmetic ([`DualValue`]), to be a
/// real float vector whose element is itself a [`DualValue`], and to be castable
/// to itself.
pub trait DualFloatVector: DualValue + FloatVector<Element: DualValue> + CastVector<Self> + SwizzleVector {}
impl<V> DualFloatVector for V where V: DualValue + FloatVector<Element: DualValue> + CastVector<V> + SwizzleVector {}

// Lane swizzles apply to every component: the primal and each derivative move
// through the same permutation, so a swizzled dual is the dual of the
// swizzled inputs.
impl<V: DualFloatVector, const N: usize> Swizzle<V::Lanes> for Dual<V, N> {
    #[inline(always)]
    fn swizzle(self, other: Self, indices: GenericArray<u32, V::Lanes>) -> Self {
        let mut out = Self {
            re: self.re.swizzle(other.re, indices.clone()),
            dual: [V::ZERO; N],
        };
        let mut i = 0;
        while i < N {
            out.dual[i] = self.dual[i].swizzle(other.dual[i], indices.clone());
            i += 1;
        }
        out
    }

    #[inline(always)]
    fn permute(self, indices: GenericArray<u32, V::Lanes>) -> Self {
        let mut out = Self {
            re: self.re.permute(indices.clone()),
            dual: [V::ZERO; N],
        };
        let mut i = 0;
        while i < N {
            out.dual[i] = self.dual[i].permute(indices.clone());
            i += 1;
        }
        out
    }

    // The `_const` forms must forward per component rather than take the trait
    // defaults: the defaults route through the runtime-index methods, losing
    // the immediate-encoded shuffles the component vectors' own `_const`
    // overrides produce.
    #[inline(always)]
    fn swizzle_const<I: SwizzleIndices<V::Lanes>>(self, other: Self) -> Self {
        let mut out = Self {
            re: self.re.swizzle_const::<I>(other.re),
            dual: [V::ZERO; N],
        };
        let mut i = 0;
        while i < N {
            out.dual[i] = self.dual[i].swizzle_const::<I>(other.dual[i]);
            i += 1;
        }
        out
    }

    #[inline(always)]
    fn permute_const<I: SwizzleIndices<V::Lanes>>(self) -> Self {
        let mut out = Self {
            re: self.re.permute_const::<I>(),
            dual: [V::ZERO; N],
        };
        let mut i = 0;
        while i < N {
            out.dual[i] = self.dual[i].permute_const::<I>();
            i += 1;
        }
        out
    }
}

// =====================================================================================
// Element stack: Dual<E, N> as a scalar element
// =====================================================================================

#[rustfmt::skip]
impl<E: DualValue + Element, const N: usize> Element for Dual<E, N> {
    type Signed = <E as Element>::Signed;
    type Unsigned = <E as Element>::Unsigned;

    const ZERO: Self = Self::ZERO;
    const ONE: Self = Self::ONE;

    // Ordering is by the primal value, so the order extremes are constants
    // (zero derivative) at the primal's extremes, and unordered values (NaN)
    // exist exactly when the primal type has them.
    const ORDER_MAX: Self = Self::constant(E::ORDER_MAX);
    const ORDER_MIN: Self = Self::constant(E::ORDER_MIN);
    const HAS_UNORDERED: bool = E::HAS_UNORDERED;
    const IS_FLOAT: bool = E::IS_FLOAT;

    #[inline(always)] fn from_i8(value: i8) -> Self { Self::constant(E::from_i8(value)) }
    #[inline(always)] fn from_u8(value: u8) -> Self { Self::constant(E::from_u8(value)) }
    #[inline(always)] fn from_u16(value: u16) -> Self { Self::constant(E::from_u16(value)) }
}

impl<E: DualValue + SignedElement, const N: usize> SignedElement for Dual<E, N> {
    #[inline(always)]
    fn abs(self) -> Self {
        if self.re < E::ZERO { -self } else { self }
    }

    #[inline(always)]
    fn signum(self) -> Self {
        Self::constant(self.re.signum())
    }
}

/// Marker type splatting a compile-time integer constant as `Dual<E, DN>`
/// (with zero derivative).
pub struct DualIntConst<E, const DN: usize, const VAL: LargeInt>(PhantomData<E>);

/// Marker type splatting a compile-time rational constant `N/D` as `Dual<E, DN>`
/// (with zero derivative).
pub struct DualRatioConst<E, const DN: usize, const NUM: LargeInt, const DEN: LargeInt>(PhantomData<E>);

impl<E: DualValue + FloatElement, const DN: usize, const VAL: LargeInt> SplatConst<Dual<E, DN>>
    for DualIntConst<E, DN, VAL>
{
    const VALUE: Dual<E, DN> = Dual::constant(<E::ConstInt<VAL> as SplatConst<E>>::VALUE);
}

impl<E: DualValue + FloatElement, const DN: usize, const NUM: LargeInt, const DEN: LargeInt> SplatConst<Dual<E, DN>>
    for DualRatioConst<E, DN, NUM, DEN>
{
    const VALUE: Dual<E, DN> = Dual::constant(<E::ConstRatio<NUM, DEN> as SplatConst<E>>::VALUE);
}

#[rustfmt::skip]
impl<E: DualValue + FloatElement, const N: usize> FloatElement for Dual<E, N> {
    #[inline(always)]
    fn sqrt(this: Self) -> Self {
        let s = E::sqrt(this.re);
        // d/dx sqrt(x) = 1 / (2 sqrt(x))
        let factor = E::VAL_ONE / (s + s);
        this.chain(s, factor)
    }

    #[inline(always)] fn floor(this: Self) -> Self { Self::constant(E::floor(this.re)) }
    #[inline(always)] fn ceil(this: Self) -> Self { Self::constant(E::ceil(this.re)) }
    #[inline(always)] fn round(this: Self) -> Self { Self::constant(E::round(this.re)) }
    #[inline(always)] fn trunc(this: Self) -> Self { Self::constant(E::trunc(this.re)) }

    // next_up/next_down only perturb the representation, not the derivative.
    #[inline(always)] fn next_up(this: Self) -> Self { Self { re: E::next_up(this.re), dual: this.dual } }
    #[inline(always)] fn next_down(this: Self) -> Self { Self { re: E::next_down(this.re), dual: this.dual } }

    #[inline(always)]
    fn try_from_int(value: LargeInt) -> Option<Self> {
        E::try_from_int(value).map(Self::constant)
    }

    #[inline(always)]
    fn try_from_ratio(n: LargeInt, d: LargeInt) -> Option<Self> {
        E::try_from_ratio(n, d).map(Self::constant)
    }

    const HAS_INFINITY: bool = E::HAS_INFINITY;
    const HAS_SIGNED_ZERO: bool = E::HAS_SIGNED_ZERO;
    const HAS_SUBNORMALS: bool = E::HAS_SUBNORMALS;

    type ConstInt<const VAL: LargeInt> = DualIntConst<E, N, VAL>;
    type ConstRatio<const NUM: LargeInt, const DEN: LargeInt> = DualRatioConst<E, N, NUM, DEN>;
}

// =====================================================================================
// const_default / HasIsa / Selectable / Interleave
// =====================================================================================

impl<V: DualValue, const N: usize> thermite::const_default::ConstDefault for Dual<V, N> {
    const DEFAULT: Self = Self::ZERO;
}

macro_rules! impl_float_consts {
    ($($name:ident),* $(,)?) => {
        impl<V: DualValue + thermite::math::FloatConsts, const N: usize> thermite::math::FloatConsts for Dual<V, N> {
            $(const $name: Self = Self::constant(<V as thermite::math::FloatConsts>::$name);)*
        }
    };
}

thermite::for_each_float_const!(impl_float_consts);

impl<V: thermite::simd::HasIsa, const N: usize> thermite::simd::HasIsa for Dual<V, N> {
    type Native = V::Native;

    const ISA: thermite::isa::InstructionSet = V::ISA;
}

impl<V: DualFloatVector, const N: usize> GenericSelectable for Dual<V, N> {
    type SelectableMask = <V as GenericSelectable>::SelectableMask;

    #[inline(always)]
    fn select<M>(mask: M, t: Self, f: Self) -> Self
    where
        Self::SelectableMask: CastMask<M>,
    {
        let mask = <Self::SelectableMask as CastMask<M>>::mask_from(mask);

        // Per-component blend via a hand-rolled loop (this is the primitive behind min/max,
        // clamp and every masked `_c`/`_m`/`_z` op, so keep it allocation/closure-free).
        let mut dual = t.dual;
        let mut i = 0;
        while i < N {
            dual[i] = mask.select(t.dual[i], f.dual[i]);
            i += 1;
        }
        Self {
            re: mask.select(t.re, f.re),
            dual,
        }
    }
}

#[rustfmt::skip]
impl<V: DualFloatVector, const N: usize> Interleave for Dual<V, N> {
    #[inline(always)]
    fn interleave(self, other: Self) -> (Self, Self) {
        let (re_lo, re_hi) = self.re.interleave(other.re);
        let mut lo = Self { re: re_lo, dual: [V::ZERO; N] };
        let mut hi = Self { re: re_hi, dual: [V::ZERO; N] };
        for i in 0..N {
            let (d_lo, d_hi) = self.dual[i].interleave(other.dual[i]);
            lo.dual[i] = d_lo;
            hi.dual[i] = d_hi;
        }
        (lo, hi)
    }

    #[inline(always)]
    fn deinterleave(self, other: Self) -> (Self, Self) {
        let (re_lo, re_hi) = self.re.deinterleave(other.re);
        let mut lo = Self { re: re_lo, dual: [V::ZERO; N] };
        let mut hi = Self { re: re_hi, dual: [V::ZERO; N] };
        for i in 0..N {
            let (d_lo, d_hi) = self.dual[i].deinterleave(other.dual[i]);
            lo.dual[i] = d_lo;
            hi.dual[i] = d_hi;
        }
        (lo, hi)
    }
}

/// The lane-sort key: strictly-before by the primal alone. See
/// `thermite::sort::SortKey` for why this is a static trait method and not a
/// closure.
impl<V: DualFloatVector, const N: usize> thermite::sort::SortKey<Self> for Dual<V, N> {
    #[inline(always)]
    fn key_lt(a: Self, b: Self) -> V::Mask {
        a.re.cmp_lt(b.re)
    }
}

/// Scalar insertion walk over whole lanes, for widths past the network ladder.
/// Quadratic, like core's `sort_any`; compares composite elements through
/// `PartialOrd`.
#[inline(always)]
fn sort_lanes_scalar<V: NumericVector, O: thermite::sort::SortOrder>(v: V) -> V
where
    V::Element: PartialOrd,
{
    let mut out = v;
    let mut i = 1;
    while i < V::LANES {
        let key = out.extractv(i);
        let mut j = i;
        while j > 0 {
            let prev = out.extractv(j - 1);
            let misplaced = if O::IS_ASCENDING { prev > key } else { prev < key };
            if !misplaced {
                break;
            }
            out = out.insertv(j, prev);
            j -= 1;
        }
        out = out.insertv(j, key);
        i += 1;
    }
    out
}

// =====================================================================================
// Compile-time splat / new machinery (re correct, derivatives zeroed -- see module docs)
// =====================================================================================

/// `SplatConst<V::Element>` carrier extracting the `re` part of a `Dual` element constant.
struct DualReSplat<E, V, const N: usize>(PhantomData<(E, V)>);

impl<E, V: DualFloatVector, const N: usize> SplatConst<V::Element> for DualReSplat<E, V, N>
where
    E: SplatConst<Dual<V::Element, N>>,
{
    const VALUE: V::Element = <E as SplatConst<Dual<V::Element, N>>>::VALUE.re;
}

impl<V: DualFloatVector, const N: usize> SplatVector<Dual<V::Element, N>> for Dual<V, N> {
    type Splat<T: SplatConst<Dual<V::Element, N>>> = Self;
}

impl<V: DualFloatVector, const N: usize, E: SplatConst<Dual<V::Element, N>>> VectorValue<E, Dual<V, N>> for Dual<V, N> {
    const VALUE: Dual<V, N> = Dual {
        re: const_splat::<V, DualReSplat<E, V, N>>(),
        dual: [<V as NumericVector>::ZERO; N],
    };
}

/// `NewConst<V::Element, Lanes>` carrier extracting the per-lane `re` parts of a `Dual` element array.
struct DualReNew<C, V, const N: usize>(PhantomData<(C, V)>);

impl<C, V: DualFloatVector, const N: usize> NewConst<V::Element, V::Lanes> for DualReNew<C, V, N>
where
    C: NewConst<Dual<V::Element, N>, V::Lanes>,
{
    const VALUES: GenericArray<V::Element, V::Lanes> = const {
        let c_vals = C::VALUES;
        let src = c_vals.as_slice();
        let mut out: GenericArray<V::Element, V::Lanes> = unsafe { core::mem::zeroed() };
        let dst = out.as_mut_slice();
        let mut i = 0;
        while i < V::LANES {
            dst[i] = src[i].re;
            i += 1;
        }
        core::mem::forget(c_vals);
        out
    };
}

/// `VectorValue` implementor for per-lane (`new`) construction of `Dual` vectors.
pub struct DualNewImpl;

impl<T, V: DualFloatVector, const N: usize> VectorValue<T, Dual<V, N>> for DualNewImpl
where
    T: NewConst<Dual<V::Element, N>, V::Lanes>,
{
    const VALUE: Dual<V, N> = Dual {
        re: const_new::<V, V::Lanes, DualReNew<T, V, N>>(),
        dual: [<V as NumericVector>::ZERO; N],
    };
}

impl<V: DualFloatVector, const N: usize> NewVector<Dual<V::Element, N>, V::Lanes> for Dual<V, N> {
    type New<T: NewConst<Dual<V::Element, N>, V::Lanes>> = DualNewImpl;
}

// =====================================================================================
// CastVector
// =====================================================================================

impl<FROM, TO, const N: usize> CastVector<Dual<FROM, N>> for Dual<TO, N>
where
    FROM: DualFloatVector + CastVector<TO>,
    TO: DualFloatVector + CastVector<FROM>,
{
    #[inline(always)]
    fn cast_into(self) -> Dual<FROM, N> {
        Dual::<FROM, N>::cast_from(self)
    }

    #[inline(always)]
    fn cast_from(from: Dual<FROM, N>) -> Self {
        Self {
            re: TO::cast_from(from.re),
            dual: array_each!([TO::ZERO; N], |i| TO::cast_from(from.dual[i])),
        }
    }
}

// =====================================================================================
// GenericVector
// =====================================================================================

impl<V: DualFloatVector, const N: usize> GenericVector for Dual<V, N> {
    type Element = Dual<V::Element, N>;

    const EMPTY: Self = Self::ZERO;
    const LANES: usize = V::LANES;

    type Lanes = V::Lanes;

    type Unsigned = V::Unsigned;
    type Signed = V::Signed;
    type Mask = V::Mask;

    #[inline(always)]
    fn new<const M: usize>(value: [Self::Element; M]) -> Self
    where
        Const<M>: IntoArrayLength<ArrayLength = Self::Lanes>,
    {
        Self {
            re: V::new(array_each!([<V::Element as Element>::ZERO; M], |m| value[m].re)),
            dual: array_each!([V::ZERO; N], |j| V::new(array_each!(
                [<V::Element as Element>::ZERO; M],
                |m| value[m].dual[j]
            ))),
        }
    }

    #[inline(always)]
    fn into_array(self) -> GenericArray<Self::Element, Self::Lanes> {
        let mut arr = GenericArray::default();
        for i in 0..Self::LANES {
            arr[i] = Dual {
                re: self.re.extractv(i),
                dual: array_each!([<V::Element as Element>::ZERO; N], |j| self.dual[j].extractv(i)),
            };
        }
        arr
    }

    #[inline(always)]
    fn splat(value: Self::Element) -> Self {
        Self {
            re: V::splat(value.re),
            dual: array_each!([V::ZERO; N], |j| V::splat(value.dual[j])),
        }
    }

    #[inline(always)]
    fn single(value: Self::Element) -> Self {
        Self {
            re: V::single(value.re),
            dual: array_each!([V::ZERO; N], |j| V::single(value.dual[j])),
        }
    }

    // Composite element alignment only guarantees the alignment of a single
    // `V::Element`, so aligned load/store just forward to the unaligned path -
    // there is no separate "aligned" fast path to take.
    #[inline(always)]
    unsafe fn load(ptr: *const Self::Element) -> Self {
        unsafe { Self::load_unaligned(ptr) }
    }

    /// A `Dual` element is `#[repr(C)]` over `1 + N` floats, so a single
    /// element is exactly [`load_deinterleaved::<1>`](Self::load_deinterleaved)
    /// - which routes through the inner vector's tuned register engine, not a
    /// scalar lane-by-lane loop.
    #[inline(always)]
    unsafe fn load_unaligned(ptr: *const Self::Element) -> Self {
        let [out] = unsafe { Self::load_deinterleaved::<1>(ptr) };
        out
    }

    #[inline(always)]
    unsafe fn load_streaming(ptr: *const Self::Element) -> Self {
        unsafe { Self::load(ptr) }
    }

    /// A `Dual` element is `#[repr(C)]` over `1 + N` floats (the primal, then
    /// the `N` derivative parts), so `M` interleaved `Dual` streams are
    /// exactly `M * (N + 1)` interleaved float streams. That is precisely the
    /// factorization [`StreamGroup`] exists for: this hands `M` and `N`
    /// straight to the inner vector's [`GenericVector::load_deinterleaved_grouped`]
    /// (a NEON `LD2`/`LD3`/`LD4`, or a shuffle network on x86), for any `M`
    /// and `N` - no dispatch ladder, no scalar fallback.
    #[inline(always)]
    unsafe fn load_deinterleaved<const M: usize>(ptr: *const Self::Element) -> [Self; M] {
        let groups = unsafe { V::load_deinterleaved_grouped::<M, N>(ptr as *const V::Element) };

        let mut out = [Self::EMPTY; M];
        let mut j = 0;
        while j < M {
            out[j] = Dual {
                re: groups[j].head,
                dual: groups[j].tail,
            };
            j += 1;
        }
        out
    }

    /// The exact inverse of [`load_deinterleaved`](Self::load_deinterleaved).
    #[inline(always)]
    unsafe fn store_interleaved<const M: usize>(ptr: *mut Self::Element, values: [Self; M]) {
        let mut groups = [StreamGroup {
            head: V::ZERO,
            tail: [V::ZERO; N],
        }; M];
        let mut j = 0;
        while j < M {
            groups[j] = StreamGroup {
                head: values[j].re,
                tail: values[j].dual,
            };
            j += 1;
        }
        unsafe { V::store_interleaved_grouped::<M, N>(ptr as *mut V::Element, groups) }
    }

    #[inline(always)]
    unsafe fn load_m(src: Self, mask: Self::Mask, ptr: *const Self::Element) -> Self {
        let flags = mask.select(<V::Signed as NumericVector>::ONE, <V::Signed as NumericVector>::ZERO);
        let zero = <<V::Signed as GenericVector>::Element as Element>::ZERO;
        let mut out = src;
        for i in 0..Self::LANES {
            if flags.extractv(i) != zero {
                out = out.insertv(i, unsafe { ptr.add(i).read() });
            }
        }
        out
    }

    #[inline(always)]
    unsafe fn load_z(mask: Self::Mask, ptr: *const Self::Element) -> Self {
        unsafe { Self::load_m(Self::EMPTY, mask, ptr) }
    }

    // See the note on `load` above: aligned store forwards to unaligned.
    #[inline(always)]
    unsafe fn store(self, ptr: *mut Self::Element) {
        unsafe { self.store_unaligned(ptr) }
    }

    #[inline(always)]
    unsafe fn store_unaligned(self, ptr: *mut Self::Element) {
        unsafe { Self::store_interleaved::<1>(ptr, [self]) }
    }

    #[inline(always)]
    unsafe fn store_streaming(self, ptr: *mut Self::Element) {
        unsafe { self.store(ptr) }
    }

    #[inline(always)]
    fn interleave_by<const GROUP: usize>(self, other: Self) -> (Self, Self) {
        let (re_lo, re_hi) = self.re.interleave_by::<GROUP>(other.re);
        let mut lo = Self {
            re: re_lo,
            dual: [V::ZERO; N],
        };
        let mut hi = Self {
            re: re_hi,
            dual: [V::ZERO; N],
        };
        for i in 0..N {
            let (d_lo, d_hi) = self.dual[i].interleave_by::<GROUP>(other.dual[i]);
            lo.dual[i] = d_lo;
            hi.dual[i] = d_hi;
        }
        (lo, hi)
    }

    #[inline(always)]
    fn deinterleave_by<const GROUP: usize>(self, other: Self) -> (Self, Self) {
        let (re_lo, re_hi) = self.re.deinterleave_by::<GROUP>(other.re);
        let mut lo = Self {
            re: re_lo,
            dual: [V::ZERO; N],
        };
        let mut hi = Self {
            re: re_hi,
            dual: [V::ZERO; N],
        };
        for i in 0..N {
            let (d_lo, d_hi) = self.dual[i].deinterleave_by::<GROUP>(other.dual[i]);
            lo.dual[i] = d_lo;
            hi.dual[i] = d_hi;
        }
        (lo, hi)
    }

    // `M` is the radix (input count); `N` is the (fixed) derivative-part count.
    // Each component - `re` and each of the `N` duals - is radix-interleaved
    // independently across the `M` inputs.
    #[inline(always)]
    fn interleave_radix<const M: usize>(inputs: [Self; M]) -> [Self; M] {
        let mut re = [V::EMPTY; M];
        for i in 0..M {
            re[i] = inputs[i].re;
        }
        let re = V::interleave_radix::<M>(re);

        let mut out = [Self::EMPTY; M];
        for i in 0..M {
            out[i].re = re[i];
        }
        for d in 0..N {
            let mut comp = [V::EMPTY; M];
            for i in 0..M {
                comp[i] = inputs[i].dual[d];
            }
            let comp = V::interleave_radix::<M>(comp);
            for i in 0..M {
                out[i].dual[d] = comp[i];
            }
        }
        out
    }

    #[inline(always)]
    fn deinterleave_radix<const M: usize>(inputs: [Self; M]) -> [Self; M] {
        let mut re = [V::EMPTY; M];
        for i in 0..M {
            re[i] = inputs[i].re;
        }
        let re = V::deinterleave_radix::<M>(re);

        let mut out = [Self::EMPTY; M];
        for i in 0..M {
            out[i].re = re[i];
        }
        for d in 0..N {
            let mut comp = [V::EMPTY; M];
            for i in 0..M {
                comp[i] = inputs[i].dual[d];
            }
            let comp = V::deinterleave_radix::<M>(comp);
            for i in 0..M {
                out[i].dual[d] = comp[i];
            }
        }
        out
    }

    #[inline(always)]
    fn deinterleave_radix_by<const M: usize, const GROUP: usize>(inputs: [Self; M]) -> [Self; M] {
        let mut re = [V::EMPTY; M];
        for i in 0..M {
            re[i] = inputs[i].re;
        }
        let re = V::deinterleave_radix_by::<M, GROUP>(re);

        let mut out = [Self::EMPTY; M];
        for i in 0..M {
            out[i].re = re[i];
        }
        for d in 0..N {
            let mut comp = [V::EMPTY; M];
            for i in 0..M {
                comp[i] = inputs[i].dual[d];
            }
            let comp = V::deinterleave_radix_by::<M, GROUP>(comp);
            for i in 0..M {
                out[i].dual[d] = comp[i];
            }
        }
        out
    }

    #[inline(always)]
    fn interleave_radix_by<const M: usize, const GROUP: usize>(inputs: [Self; M]) -> [Self; M] {
        let mut re = [V::EMPTY; M];
        for i in 0..M {
            re[i] = inputs[i].re;
        }
        let re = V::interleave_radix_by::<M, GROUP>(re);

        let mut out = [Self::EMPTY; M];
        for i in 0..M {
            out[i].re = re[i];
        }
        for d in 0..N {
            let mut comp = [V::EMPTY; M];
            for i in 0..M {
                comp[i] = inputs[i].dual[d];
            }
            let comp = V::interleave_radix_by::<M, GROUP>(comp);
            for i in 0..M {
                out[i].dual[d] = comp[i];
            }
        }
        out
    }

    #[inline(always)]
    unsafe fn store_masked(self, mask: Self::Mask, ptr: *mut Self::Element) {
        let flags = mask.select(<V::Signed as NumericVector>::ONE, <V::Signed as NumericVector>::ZERO);
        let zero = <<V::Signed as GenericVector>::Element as Element>::ZERO;
        for i in 0..Self::LANES {
            if flags.extractv(i) != zero {
                unsafe { ptr.add(i).write(self.extractv(i)) };
            }
        }
    }

    #[inline(always)]
    unsafe fn lookup_unchecked(values: &[Self::Element], indices: Self::Unsigned) -> Self {
        let mut res = Self::EMPTY;
        for i in 0..Self::LANES {
            let Ok(idx) = indices.extractv(i).try_into() else {
                panic!("Index out of bounds for usize");
            };
            res = res.insertv(i, values[idx]);
        }
        res
    }

    #[inline(always)]
    fn broadcast<const I: usize>(self) -> Self {
        Self {
            re: V::broadcast::<I>(self.re),
            dual: array_each!([V::ZERO; N], |j| V::broadcast::<I>(self.dual[j])),
        }
    }

    #[inline(always)]
    fn broadcastv(self, idx: usize) -> Self {
        Self {
            re: self.re.broadcastv(idx),
            dual: array_each!([V::ZERO; N], |j| self.dual[j].broadcastv(idx)),
        }
    }

    #[inline(always)]
    fn extract<const I: usize>(self) -> Self::Element {
        Dual {
            re: V::extract::<I>(self.re),
            dual: array_each!([<V::Element as Element>::ZERO; N], |j| V::extract::<I>(self.dual[j])),
        }
    }

    #[inline(always)]
    fn extractv(self, idx: usize) -> Self::Element {
        Dual {
            re: self.re.extractv(idx),
            dual: array_each!([<V::Element as Element>::ZERO; N], |j| self.dual[j].extractv(idx)),
        }
    }

    #[inline(always)]
    fn insert<const I: usize>(self, value: Self::Element) -> Self {
        Self {
            re: V::insert::<I>(self.re, value.re),
            dual: array_each!([V::ZERO; N], |j| V::insert::<I>(self.dual[j], value.dual[j])),
        }
    }

    #[inline(always)]
    fn insertv(self, idx: usize, value: Self::Element) -> Self {
        Self {
            re: self.re.insertv(idx, value.re),
            dual: array_each!([V::ZERO; N], |j| self.dual[j].insertv(idx, value.dual[j])),
        }
    }

    #[inline(always)]
    fn reverse(self) -> Self {
        Self {
            re: self.re.reverse(),
            dual: array_each!([V::ZERO; N], |j| self.dual[j].reverse()),
        }
    }

    #[inline(always)]
    fn swap_bytes(self) -> Self {
        Self {
            re: self.re.swap_bytes(),
            dual: array_each!([V::ZERO; N], |j| self.dual[j].swap_bytes()),
        }
    }

    #[inline(always)]
    fn zz(self, mask: Self::Mask) -> Self {
        Self {
            re: self.re.zz(mask),
            dual: array_each!([V::ZERO; N], |j| self.dual[j].zz(mask)),
        }
    }

    #[inline(always)]
    fn nz(self, mask: Self::Mask) -> Self {
        Self {
            re: self.re.nz(mask),
            dual: array_each!([V::ZERO; N], |j| self.dual[j].nz(mask)),
        }
    }

    #[inline(always)]
    fn compress(self, mask: Self::Mask) -> Self {
        Self {
            re: self.re.compress(mask),
            dual: array_each!([V::ZERO; N], |j| self.dual[j].compress(mask)),
        }
    }

    #[inline(always)]
    fn compress_z(self, mask: Self::Mask) -> Self {
        Self {
            re: self.re.compress_z(mask),
            dual: array_each!([V::ZERO; N], |j| self.dual[j].compress_z(mask)),
        }
    }

    // The compaction family is pure lane movement driven by `mask` alone, so every
    // component takes the same permutation and the value/derivative pairing survives
    // it. That includes `compress_m`, whose keep-lanes are chosen by the population
    // count of the shared mask and so land identically in each component.
    #[inline(always)]
    fn compress_m(self, src: Self, mask: Self::Mask) -> Self {
        Self {
            re: self.re.compress_m(src.re, mask),
            dual: array_each!([V::ZERO; N], |j| self.dual[j].compress_m(src.dual[j], mask)),
        }
    }

    #[inline(always)]
    fn expand(self, mask: Self::Mask) -> Self {
        Self {
            re: self.re.expand(mask),
            dual: array_each!([V::ZERO; N], |j| self.dual[j].expand(mask)),
        }
    }

    #[inline(always)]
    fn expand_z(self, mask: Self::Mask) -> Self {
        Self {
            re: self.re.expand_z(mask),
            dual: array_each!([V::ZERO; N], |j| self.dual[j].expand_z(mask)),
        }
    }

    #[inline(always)]
    fn expand_m(self, src: Self, mask: Self::Mask) -> Self {
        Self {
            re: self.re.expand_m(src.re, mask),
            dual: array_each!([V::ZERO; N], |j| self.dual[j].expand_m(src.dual[j], mask)),
        }
    }

    #[inline(always)]
    fn align<const OFFSET: usize>(self, other: Self) -> Self {
        Self {
            re: self.re.align::<OFFSET>(other.re),
            dual: array_each!([V::ZERO; N], |j| self.dual[j].align::<OFFSET>(other.dual[j])),
        }
    }

    // Every component aligns through `V`, so this is only as native as `V` is.
    const HAS_NATIVE_ALIGN: bool = V::HAS_NATIVE_ALIGN;

    #[inline(always)]
    fn map<F>(mut self, f: F) -> Self
    where
        F: Fn(Self::Element) -> Self::Element,
    {
        for i in 0..Self::LANES {
            self = self.insertv(i, f(self.extractv(i)));
        }
        self
    }

    #[inline(always)]
    fn fold<F>(self, mut init: Self::Element, f: F) -> Self::Element
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element,
    {
        for i in 0..Self::LANES {
            init = f(init, self.extractv(i));
        }
        init
    }

    #[inline(always)]
    fn reduce<F>(self, f: F) -> Self::Element
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element,
    {
        let mut result = self.extractv(0);
        for i in 1..Self::LANES {
            result = f(result, self.extractv(i));
        }
        result
    }

    #[inline(always)]
    fn splat_m(src: Self, mask: Self::Mask, value: Self::Element) -> Self {
        mask.select(Self::splat(value), src)
    }
    #[inline(always)]
    fn splat_z(mask: Self::Mask, value: Self::Element) -> Self {
        mask.select(Self::splat(value), Self::EMPTY)
    }
    #[inline(always)]
    fn broadcast_c<const I: usize>(self, mask: Self::Mask) -> Self {
        mask.select(self.broadcast::<I>(), self)
    }
    #[inline(always)]
    fn broadcast_m<const I: usize>(self, src: Self, mask: Self::Mask) -> Self {
        mask.select(self.broadcast::<I>(), src)
    }
    #[inline(always)]
    fn broadcast_z<const I: usize>(self, mask: Self::Mask) -> Self {
        mask.select(self.broadcast::<I>(), Self::EMPTY)
    }
    #[inline(always)]
    fn broadcastv_c(self, mask: Self::Mask, idx: usize) -> Self {
        mask.select(self.broadcastv(idx), self)
    }
    #[inline(always)]
    fn broadcastv_m(self, src: Self, mask: Self::Mask, idx: usize) -> Self {
        mask.select(self.broadcastv(idx), src)
    }
    #[inline(always)]
    fn broadcastv_z(self, mask: Self::Mask, idx: usize) -> Self {
        mask.select(self.broadcastv(idx), Self::EMPTY)
    }
    #[inline(always)]
    fn reverse_c(self, mask: Self::Mask) -> Self {
        mask.select(self.reverse(), self)
    }
    #[inline(always)]
    fn reverse_m(self, src: Self, mask: Self::Mask) -> Self {
        mask.select(self.reverse(), src)
    }
    #[inline(always)]
    fn reverse_z(self, mask: Self::Mask) -> Self {
        mask.select(self.reverse(), Self::EMPTY)
    }
    #[inline(always)]
    fn swap_bytes_c(self, mask: Self::Mask) -> Self {
        mask.select(self.swap_bytes(), self)
    }
    #[inline(always)]
    fn swap_bytes_m(self, src: Self, mask: Self::Mask) -> Self {
        mask.select(self.swap_bytes(), src)
    }
    #[inline(always)]
    fn swap_bytes_z(self, mask: Self::Mask) -> Self {
        mask.select(self.swap_bytes(), Self::EMPTY)
    }
}

// =====================================================================================
// PartialOrdVector -- compare by primal value
// =====================================================================================

#[rustfmt::skip]
impl<V: DualFloatVector, const N: usize> PartialOrdVector for Dual<V, N> {
    #[inline(always)] fn cmp_eq(self, other: Self) -> Self::Mask { self.re.cmp_eq(other.re) }
    #[inline(always)] fn cmp_ne(self, other: Self) -> Self::Mask { self.re.cmp_ne(other.re) }
    #[inline(always)] fn cmp_lt(self, other: Self) -> Self::Mask { self.re.cmp_lt(other.re) }
    #[inline(always)] fn cmp_gt(self, other: Self) -> Self::Mask { self.re.cmp_gt(other.re) }
    #[inline(always)] fn cmp_le(self, other: Self) -> Self::Mask { self.re.cmp_le(other.re) }
    #[inline(always)] fn cmp_ge(self, other: Self) -> Self::Mask { self.re.cmp_ge(other.re) }
}

// =====================================================================================
// Iterator Sum/Product + Bounded
// =====================================================================================

impl<V: DualValue, const N: usize> core::iter::Sum for Dual<V, N> {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |a, b| a + b)
    }
}

impl<V: DualValue, const N: usize> core::iter::Product for Dual<V, N> {
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |a, b| a * b)
    }
}

#[rustfmt::skip]
impl<V: DualFloatVector, const N: usize> Bounded for Dual<V, N> {
    #[inline(always)] fn min_value() -> Self { Self::constant(V::MIN) }
    #[inline(always)] fn max_value() -> Self { Self::constant(V::MAX) }
}

// =====================================================================================
// Square (masked) -- needed by NumericVector
// =====================================================================================

impl<V: DualFloatVector, const N: usize> SquareMasked<V::Mask> for Dual<V, N> {
    #[inline(always)]
    fn square_c(self, mask: V::Mask) -> Self::Output {
        mask.select(self.square(), self)
    }

    #[inline(always)]
    fn square_m(self, src: Self, mask: V::Mask) -> Self::Output {
        mask.select(self.square(), src)
    }

    #[inline(always)]
    fn square_z(self, mask: V::Mask) -> Self::Output {
        mask.select(self.square(), Self::ZERO)
    }
}

// =====================================================================================
// Masked arithmetic ops (select-based, like thermite-compensated)
// =====================================================================================

macro_rules! impl_masked {
    (MUL_ADD: $($method:ident),*) => {paste::paste! {
        impl<V: DualFloatVector, const N: usize, A, B> thermite::vector::ops::MulAddExtMasked<V::Mask, A, B> for Dual<V, N>
        where
            Dual<V, N>: thermite::vector::ops::MulAddExt<A, B, Output = Self>,
        {
            $(
                #[inline(always)]
                fn [<$method _c>](self, mask: V::Mask, a: A, b: B) -> Self {
                    mask.select(self.$method(a, b), self)
                }
                #[inline(always)]
                fn [<$method _m>](self, src: Self, mask: V::Mask, a: A, b: B) -> Self {
                    mask.select(self.$method(a, b), src)
                }
                #[inline(always)]
                fn [<$method _z>](self, mask: V::Mask, a: A, b: B) -> Self {
                    mask.select(self.$method(a, b), Self::EMPTY)
                }
            )*
        }

        impl<V: DualFloatVector, const N: usize, A, B> thermite::vector::ops::MulAddAssignExtMasked<V::Mask, A, B> for Dual<V, N>
        where
            Dual<V, N>: thermite::vector::ops::MulAddExt<A, B, Output = Self>,
        {
            $(
                #[inline(always)]
                fn [<$method _assign_c>](&mut self, mask: V::Mask, a: A, b: B) {
                    *self = mask.select(self.$method(a, b), *self);
                }
                #[inline(always)]
                fn [<$method _assign_m>](&mut self, src: Self, mask: V::Mask, a: A, b: B) {
                    *self = mask.select(self.$method(a, b), src);
                }
                #[inline(always)]
                fn [<$method _assign_z>](&mut self, mask: V::Mask, a: A, b: B) {
                    *self = mask.select(self.$method(a, b), Self::EMPTY);
                }
            )*
        }
    }};

    ($trait:ident::$method:ident) => {paste::paste! {
        impl<V: DualFloatVector, const N: usize, Rhs> thermite::vector::ops::[<$trait Masked>]<V::Mask, Rhs> for Dual<V, N>
        where
            Dual<V, N>: core::ops::$trait<Rhs, Output = Self>,
        {
            #[inline(always)]
            fn [<$method _c>](self, mask: V::Mask, rhs: Rhs) -> Self {
                mask.select(self.$method(rhs), self)
            }
            #[inline(always)]
            fn [<$method _m>](self, src: Self, mask: V::Mask, rhs: Rhs) -> Self {
                mask.select(self.$method(rhs), src)
            }
            #[inline(always)]
            fn [<$method _z>](self, mask: V::Mask, rhs: Rhs) -> Self {
                mask.select(self.$method(rhs), Self::EMPTY)
            }
        }

        impl<V: DualFloatVector, const N: usize, Rhs> thermite::vector::ops::[<$trait AssignMasked>]<V::Mask, Rhs> for Dual<V, N>
        where
            Dual<V, N>: core::ops::$trait<Rhs, Output = Self>,
        {
            #[inline(always)]
            fn [<$method _assign_c>](&mut self, mask: V::Mask, rhs: Rhs) {
                *self = mask.select(self.$method(rhs), *self);
            }
            #[inline(always)]
            fn [<$method _assign_m>](&mut self, src: Self, mask: V::Mask, rhs: Rhs) {
                *self = mask.select(self.$method(rhs), src);
            }
            #[inline(always)]
            fn [<$method _assign_z>](&mut self, mask: V::Mask, rhs: Rhs) {
                *self = mask.select(self.$method(rhs), Self::EMPTY);
            }
        }
    }};
}

impl_masked!(MUL_ADD: mul_add, mul_sub, nmul_add, nmul_sub, mul_adde, mul_sube, nmul_adde, nmul_sube);
impl_masked!(Add::add);
impl_masked!(Sub::sub);
impl_masked!(Mul::mul);
impl_masked!(Div::div);
impl_masked!(Rem::rem);

// =====================================================================================
// Lane-alternating add/sub (`AddSubExt`). `addsub` is linear, so it differentiates
// exactly like an add: apply it component-wise. The building block is `neg_even`,
// which flips the sign of the even lanes of every stored component (exact) via the
// inner vector's `addsub(0, w) = [-w0, w1, -w2, ...]`. Then:
//   addsub(a, b)      = a + neg_even(b)
//   fmaddsub(a, b, c) = a*b + neg_even(c)   (via the inner product-rule mul_adde)
//   fmsubadd(a, b, c) = a*b - neg_even(c)
// =====================================================================================

#[inline(always)]
fn neg_even_dual<V: DualFloatVector, const N: usize>(x: Dual<V, N>) -> Dual<V, N> {
    let mut dual = x.dual;
    let mut i = 0;
    while i < N {
        dual[i] = V::ZERO.addsub(dual[i]);
        i += 1;
    }
    Dual {
        re: V::ZERO.addsub(x.re),
        dual,
    }
}

impl<V: DualFloatVector, const N: usize> AddSubExt for Dual<V, N> {
    type Output = Self;

    #[inline(always)]
    fn addsub(self, b: Self) -> Self {
        self + neg_even_dual(b)
    }
    #[inline(always)]
    fn fmaddsub(self, b: Self, c: Self) -> Self {
        self.mul_adde(b, neg_even_dual(c))
    }
    #[inline(always)]
    fn fmsubadd(self, b: Self, c: Self) -> Self {
        self.mul_sube(b, neg_even_dual(c))
    }
}

impl<V: DualFloatVector, const N: usize> AddSubExtMasked<V::Mask> for Dual<V, N> {
    #[inline(always)]
    fn addsub_c(self, mask: V::Mask, b: Self) -> Self {
        mask.select(self.addsub(b), self)
    }
    #[inline(always)]
    fn addsub_m(self, src: Self, mask: V::Mask, b: Self) -> Self {
        mask.select(self.addsub(b), src)
    }
    #[inline(always)]
    fn addsub_z(self, mask: V::Mask, b: Self) -> Self {
        mask.select(self.addsub(b), Self::EMPTY)
    }

    #[inline(always)]
    fn fmaddsub_c(self, mask: V::Mask, b: Self, c: Self) -> Self {
        mask.select(self.fmaddsub(b, c), self)
    }
    #[inline(always)]
    fn fmaddsub_m(self, src: Self, mask: V::Mask, b: Self, c: Self) -> Self {
        mask.select(self.fmaddsub(b, c), src)
    }
    #[inline(always)]
    fn fmaddsub_z(self, mask: V::Mask, b: Self, c: Self) -> Self {
        mask.select(self.fmaddsub(b, c), Self::EMPTY)
    }

    #[inline(always)]
    fn fmsubadd_c(self, mask: V::Mask, b: Self, c: Self) -> Self {
        mask.select(self.fmsubadd(b, c), self)
    }
    #[inline(always)]
    fn fmsubadd_m(self, src: Self, mask: V::Mask, b: Self, c: Self) -> Self {
        mask.select(self.fmsubadd(b, c), src)
    }
    #[inline(always)]
    fn fmsubadd_z(self, mask: V::Mask, b: Self, c: Self) -> Self {
        mask.select(self.fmsubadd(b, c), Self::EMPTY)
    }
}

// `_c`/`_m`/`_z` masked variants of the inherent unary (`fn m(self) -> Self`) and
// binary (`fn m(self, Self) -> Self`) vector ops, as plain blends -- the same
// select pattern `impl_masked!` uses for the `core::ops` methods above. Invoked
// inside the relevant trait impls below.
macro_rules! dual_masked {
    (unary: $($m:ident),* $(,)?) => { paste::paste! {
        $(
            #[inline(always)] fn [<$m _c>](self, mask: Self::Mask) -> Self { mask.select(self.$m(), self) }
            #[inline(always)] fn [<$m _m>](self, src: Self, mask: Self::Mask) -> Self { mask.select(self.$m(), src) }
            #[inline(always)] fn [<$m _z>](self, mask: Self::Mask) -> Self { mask.select(self.$m(), Self::ZERO) }
        )*
    }};
    (binary: $($m:ident),* $(,)?) => { paste::paste! {
        $(
            #[inline(always)] fn [<$m _c>](self, mask: Self::Mask, rhs: Self) -> Self { mask.select(self.$m(rhs), self) }
            #[inline(always)] fn [<$m _m>](self, src: Self, mask: Self::Mask, rhs: Self) -> Self { mask.select(self.$m(rhs), src) }
            #[inline(always)] fn [<$m _z>](self, mask: Self::Mask, rhs: Self) -> Self { mask.select(self.$m(rhs), Self::ZERO) }
        )*
    }};
}

// =====================================================================================
// NumericVector
// =====================================================================================

#[rustfmt::skip]
impl<V: DualFloatVector, const N: usize> NumericVector for Dual<V, N> {
    // The integer conversions are real/value-only in both directions: an integer has no
    // derivative, no imaginary part and no error term, so converting one in yields a
    // constant, and converting out is the value part alone.
    #[inline(always)]
    fn to_signed_integer(self) -> Self::Signed {
        self.re.to_signed_integer()
    }

    #[inline(always)]
    fn from_signed_integer(v: Self::Signed) -> Self {
        Self::constant(V::from_signed_integer(v))
    }

    #[inline(always)]
    fn to_unsigned_integer(self) -> Self::Unsigned {
        self.re.to_unsigned_integer()
    }

    #[inline(always)]
    fn from_unsigned_integer(v: Self::Unsigned) -> Self {
        Self::constant(V::from_unsigned_integer(v))
    }

    const ZERO: Self = <Self as crate::DualValue>::VAL_ZERO;
    const ONE: Self = <Self as crate::DualValue>::VAL_ONE;
    const TWO: Self = Self::constant(V::TWO);
    const MIN: Self = Self::constant(V::MIN);
    const MAX: Self = Self::constant(V::MAX);

    #[inline(always)] fn is_zero(self) -> Self::Mask { self.re.is_zero() }
    #[inline(always)] fn is_all_zero(self) -> bool { self.re.is_all_zero() }

    #[inline(always)] fn min(self, other: Self) -> Self { self.cmp_lt(other).select(self, other) }
    #[inline(always)] fn max(self, other: Self) -> Self { self.cmp_gt(other).select(self, other) }

    // Lane sorts are keyed on the PRIMAL alone: each compare-exchange derives its
    // routing mask from `re` and moves every derivative component through the same
    // permutation and select (`thermite::sort::sort_lanes_by_key`), so a sorted dual
    // is the dual of the sorted inputs. Widths past the network ladder take the
    // scalar walk (whose composite `PartialOrd` tie-breaks by derivatives - ties by
    // key are unspecified order either way).
    #[inline(always)]
    fn sort_by<O: thermite::sort::SortOrder>(self) -> Self {
        if const { Self::LANES <= 16 && Self::LANES.is_power_of_two() } {
            thermite::sort::sort_lanes_by_key::<Self, O, Self>(self)
        } else {
            sort_lanes_scalar::<Self, O>(self)
        }
    }

    #[inline(always)]
    fn bitonic_clean_by<O: thermite::sort::SortOrder>(self) -> Self {
        if const { Self::LANES <= 16 && Self::LANES.is_power_of_two() } {
            thermite::sort::bitonic_clean_lanes_by_key::<Self, O, Self>(self)
        } else {
            // A full sort trivially cleans a bitonic input.
            sort_lanes_scalar::<Self, O>(self)
        }
    }

    // Compare the primal against both bounds once, then select per component, rather than
    // `self.max(min).min(max)` which builds an intermediate `max` dual and recompares it.
    #[inline(always)]
    fn clamp(self, min: Self, max: Self) -> Self {
        let is_lt = self.re.cmp_lt(min.re);
        let is_gt = self.re.cmp_gt(max.re);

        let re = is_lt.select(min.re, is_gt.select(max.re, self.re));
        let mut dual = self.dual;
        let mut i = 0;
        while i < N {
            dual[i] = is_lt.select(min.dual[i], is_gt.select(max.dual[i], self.dual[i]));
            i += 1;
        }
        Self { re, dual }
    }

    // Ordering of a dual is by its primal, so let the inner vector's SIMD
    // arg_minmax locate the winning lanes, then extract that lane's primal and
    // derivative components -- never a scalar per-lane comparison.
    #[inline(always)]
    fn min_element(self) -> Self::Element {
        let (lo, _) = self.re.arg_minmax();
        self.extractv(lo)
    }

    #[inline(always)]
    fn max_element(self) -> Self::Element {
        let (_, hi) = self.re.arg_minmax();
        self.extractv(hi)
    }

    // One arg_minmax for both ends.
    #[inline(always)]
    fn min_max_element(self) -> (Self::Element, Self::Element) {
        let (lo, hi) = self.re.arg_minmax();
        (self.extractv(lo), self.extractv(hi))
    }

    // Sum is linear, so it commutes with the value/derivative split: reduce each
    // component with the inner vector's native horizontal sum rather than
    // extracting and folding `LANES` scalar duals.
    #[inline(always)]
    fn sum_elements(self) -> Self::Element {
        let mut dual = [<V::Element as Element>::ZERO; N];
        let mut j = 0;
        while j < N {
            dual[j] = self.dual[j].sum_elements();
            j += 1;
        }
        Dual { re: self.re.sum_elements(), dual }
    }

    // Same linearity as `sum_elements`: scanning each component with the inner
    // vector's own prefix sum is the scan of the duals.
    #[inline(always)]
    fn prefix_sum(self) -> Self {
        Self {
            re: self.re.prefix_sum(),
            dual: array_each!([V::ZERO; N], |j| self.dual[j].prefix_sum()),
        }
    }

    #[inline(always)]
    fn reverse_prefix_sum(self) -> Self {
        Self {
            re: self.re.reverse_prefix_sum(),
            dual: array_each!([V::ZERO; N], |j| self.dual[j].reverse_prefix_sum()),
        }
    }

    // min/max are *not* componentwise: a dual is ordered by its primal and the
    // derivative of the winner comes with it, so scanning `re` and `dual` separately
    // would pair a primal from one lane with a derivative from another. Run the
    // ladder over whole duals instead, on `Self::min`/`Self::max` above.
    #[inline(always)]
    fn prefix_min(self) -> Self {
        thermite::scan_ladder!(forward, self, self.broadcast::<0>(), Self::min)
    }

    #[inline(always)]
    fn prefix_max(self) -> Self {
        thermite::scan_ladder!(forward, self, self.broadcast::<0>(), Self::max)
    }

    #[inline(always)]
    fn reverse_prefix_min(self) -> Self {
        thermite::scan_ladder!(reverse, self, Self::splat(self.last_element()), Self::min)
    }

    #[inline(always)]
    fn reverse_prefix_max(self) -> Self {
        thermite::scan_ladder!(reverse, self, Self::splat(self.last_element()), Self::max)
    }

    // Product is *not* linear (the per-lane derivatives cross-multiply), so it
    // needs real dual multiplications across lanes. A log-depth tree reduction
    // shortens the dependency chain versus a sequential fold.
    #[inline(always)]
    fn prod_elements(self) -> Self::Element {
        let mut arr = self.into_array();
        reduce_in_place(&mut arr, |a, b| a * b);
        arr[0]
    }

    #[inline(always)] fn offset() -> Self { Self::constant(V::offset()) }
    #[inline(always)] fn indexed() -> Self { Self::constant(V::indexed()) }

    #[inline(always)] fn arg_minmax(self) -> (usize, usize) { self.re.arg_minmax() }

    // Product rule by a (possibly-dual) scalar, splatting the scalar components directly into
    // the inner ops rather than building an intermediate splatted `Dual` and going through `Mul`
    // (which a width-1 / GPU backend may not optimize away).
    #[inline(always)]
    fn scale(self, factor: Self::Element) -> Self {
        let fr = V::splat(factor.re);
        let mut dual = self.dual;
        let mut i = 0;
        while i < N {
            // re*factor.dual + self.dual*factor.re
            dual[i] = self.re.mul_adde(V::splat(factor.dual[i]), self.dual[i] * fr);
            i += 1;
        }
        Self { re: self.re * fr, dual }
    }

    #[inline(always)] fn scale_c(self, mask: Self::Mask, factor: Self::Element) -> Self { mask.select(self.scale(factor), self) }
    #[inline(always)] fn scale_m(self, src: Self, mask: Self::Mask, factor: Self::Element) -> Self { mask.select(self.scale(factor), src) }
    #[inline(always)] fn scale_z(self, mask: Self::Mask, factor: Self::Element) -> Self { mask.select(self.scale(factor), Self::ZERO) }
    dual_masked!(binary: min, max);

    // pairwise_sum is a linear rearrange-and-add, so the derivative is the
    // pairwise_sum of the corresponding component parts.
    #[inline(always)]
    fn pairwise_sum(lo: Self, hi: Self) -> Self {
        let mut dual = lo.dual;
        let mut i = 0;
        while i < N {
            dual[i] = V::pairwise_sum(lo.dual[i], hi.dual[i]);
            i += 1;
        }
        Self { re: V::pairwise_sum(lo.re, hi.re), dual }
    }

    #[inline(always)]
    fn relaxed_pairwise_sum(lo: Self, hi: Self) -> Self {
        let mut dual = lo.dual;
        let mut i = 0;
        while i < N {
            dual[i] = V::relaxed_pairwise_sum(lo.dual[i], hi.dual[i]);
            i += 1;
        }
        Self { re: V::relaxed_pairwise_sum(lo.re, hi.re), dual }
    }
}

// =====================================================================================
// SignedVector
// =====================================================================================

impl<V: DualFloatVector, const N: usize> NegMasked<V::Mask> for Dual<V, N> {
    // Blend the negation per component in a single loop, rather than `select(-self, src)` which
    // first builds the whole negated `Dual` (one loop) and then blends it (another loop).
    #[inline(always)]
    fn neg_c(self, mask: V::Mask) -> Self {
        let re = mask.select(-self.re, self.re);
        let mut dual = self.dual;
        let mut i = 0;
        while i < N {
            dual[i] = mask.select(-dual[i], dual[i]);
            i += 1;
        }
        Self { re, dual }
    }

    #[inline(always)]
    fn neg_m(self, src: Self, mask: V::Mask) -> Self {
        let re = mask.select(-self.re, src.re);
        let mut dual = self.dual;
        let mut i = 0;
        while i < N {
            dual[i] = mask.select(-dual[i], src.dual[i]);
            i += 1;
        }
        Self { re, dual }
    }

    #[inline(always)]
    fn neg_z(self, mask: V::Mask) -> Self {
        let re = mask.select(-self.re, V::ZERO);
        let mut dual = self.dual;
        let mut i = 0;
        while i < N {
            dual[i] = mask.select(-dual[i], V::ZERO);
            i += 1;
        }
        Self { re, dual }
    }
}

#[rustfmt::skip]
impl<V: DualFloatVector, const N: usize> SignedVector for Dual<V, N> {
    const NEG_ONE: Self = Self::constant(V::NEG_ONE);
    const MIN_POSITIVE: Self = Self::constant(V::MIN_POSITIVE);

    #[inline(always)]
    fn abs(self) -> Self {
        // |x|' = sign(x) * x'
        self.neg_c(self.re.cmp_lt(V::ZERO))
    }

    #[inline(always)] fn signum(self) -> Self { Self::constant(self.re.signum()) }
    #[inline(always)] fn is_positive(self) -> Self::Mask { self.re.is_positive() }
    #[inline(always)] fn is_negative(self) -> Self::Mask { self.re.is_negative() }
    #[inline(always)] fn select_negative(self, if_neg: Self, if_pos: Self) -> Self { self.is_negative().select(if_neg, if_pos) }

    #[inline(always)]
    fn copysign(self, sign: Self) -> Self {
        self.neg_c(self.is_negative() ^ sign.is_negative())
    }

    dual_masked!(unary: abs);
    dual_masked!(binary: copysign);
}

// =====================================================================================
// FloatVector
// =====================================================================================

#[rustfmt::skip]
impl<V: DualFloatVector, const N: usize> FloatVector for Dual<V, N> {
    const HALF: Self = Self::constant(<V as FloatVector>::HALF);
    const NEG_ZERO: Self = Self::constant(<V as FloatVector>::NEG_ZERO);
    const INFINITY: Self = Self::constant(<V as FloatVector>::INFINITY);
    const NEG_INFINITY: Self = Self::constant(<V as FloatVector>::NEG_INFINITY);
    const NAN: Self = Self::constant(<V as FloatVector>::NAN);
    const EPSILON: Self = Self::constant(<V as FloatVector>::EPSILON);

    type ExtendedPrecision = Self;

    // The dual `rcp`/`rsqrt` derivatives are built from the inner primal estimate,
    // so they're approximate exactly when the inner vector's are.
    const HAS_APPROX_RCP: bool = V::HAS_APPROX_RCP;
    const HAS_APPROX_RSQRT: bool = V::HAS_APPROX_RSQRT;

    #[inline(always)] fn is_infinite(self) -> Self::Mask { self.re.is_infinite() }
    #[inline(always)] fn is_finite(self) -> Self::Mask { self.re.is_finite() }
    #[inline(always)] fn is_nan(self) -> Self::Mask { self.re.is_nan() }
    #[inline(always)] fn is_zero_or_subnormal(self) -> Self::Mask { self.re.is_zero_or_subnormal() }
    #[inline(always)] fn is_normal(self) -> Self::Mask { self.re.is_normal() }
    #[inline(always)] fn is_subnormal(self) -> Self::Mask { self.re.is_subnormal() }

    #[inline(always)]
    fn sqrt(self) -> Self {
        let s = self.re.sqrt();
        // d/dx sqrt(x) = 1 / (2 sqrt(x))
        self.chain(s, V::HALF / s)
    }

    #[inline(always)]
    fn rcp(self) -> Self {
        let r = self.re.rcp();
        // d/dx (1/x) = -1/x^2
        self.chain(r, -(r * r))
    }

    #[inline(always)]
    fn rsqrt(self) -> Self {
        let r = self.re.rsqrt();
        // d/dx x^(-1/2) = -1/2 x^(-3/2) = -1/2 * rsqrt(x) / x
        self.chain(r, (V::HALF * r * r * r).neg())
    }

    #[inline(always)] fn floor(self) -> Self { Self::constant(self.re.floor()) }
    #[inline(always)] fn ceil(self) -> Self { Self::constant(self.re.ceil()) }
    #[inline(always)] fn round(self) -> Self { Self::constant(self.re.round()) }
    #[inline(always)] fn trunc(self) -> Self { Self::constant(self.re.trunc()) }
    // fract(x) = x - trunc(x); derivative 1, so the dual parts pass through unchanged.
    // Avoids the full dual subtract (N subtractions of zero) the `self - self.trunc()` form does.
    #[inline(always)] fn fract(self) -> Self { Self { re: self.re.fract(), dual: self.dual } }

    #[inline(always)]
    fn mul_sign(self, sign: Self) -> Self {
        Self {
            re: self.re.mul_sign(sign.re),
            dual: array_each!([V::ZERO; N], |i| self.dual[i].mul_sign(sign.re)),
        }
    }

    #[inline(always)] fn signed_zero(self) -> Self { Self::constant(self.re.signed_zero()) }

    #[inline(always)] fn next_up(self) -> Self { Self { re: self.re.next_up(), dual: self.dual } }
    #[inline(always)] fn next_down(self) -> Self { Self { re: self.re.next_down(), dual: self.dual } }

    #[inline(always)] unsafe fn block_autovectorization(&mut self) {
        unsafe {
            self.re.block_autovectorization();
            for i in 0..N {
                self.dual[i].block_autovectorization();
            }
        }
    }

    // mix(t) = a*(1 - t) + b*t = a + (b - a)*t, composed through dual arithmetic.
    #[inline(always)] fn mix(self, a: Self, b: Self) -> Self { a + (b - a) * self }

    dual_masked!(unary: sqrt, rsqrt, rcp, floor, ceil, round, trunc, fract, signed_zero, next_up, next_down);
    dual_masked!(binary: mul_sign);
}
