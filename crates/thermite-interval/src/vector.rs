//! The Thermite vector-trait tower for `Interval<V, W>`, closely following
//! thermite-compensated's delegation structure (the other two-field
//! composite).
//!
//! Semantics legend, per method class:
//!
//! - _Lane plumbing_ (swizzles, interleave, compress/expand, broadcast,
//!   extract/insert): both bounds move through the same permutation, so a
//!   permuted interval vector is still per-lane `[lo, hi]` pairs. Delegated
//!   component-wise, exactly like `Compensated`.
//! - _Comparisons_: **certainly** semantics, so `a.cmp_lt(b)` is true iff
//!   every point of `a` is below every point of `b` (`a.hi < b.lo`).
//!   Conservative for select-based generic code: a clamp that fails to
//!   trigger widens, it never lies. The possibly-forms are inherent methods
//!   on `Interval`.
//! - _Memory_ (`load`/`store` family, `lookup`): `todo!()` until the
//!   interleaved layout is settled. The slice-iteration story depends on it.

use core::marker::PhantomData;

use thermite::generic_array::GenericArray;
use thermite::mask::{CastMask, GenericMask, GenericSelectable};
use thermite::prelude::*;
use thermite::swizzle::{Swizzle, SwizzleIndices};
use thermite::vector::{
    AsFloatVectorWithBitsKernel, Interleave, NewConst, NewVector, SplatConst, SplatVector, VectorValue,
};

use crate::consts::BoundedFloatConsts;
use crate::element::IntervalElem;
use crate::widen::WideningPolicy;
use crate::{Interval, IntervalFloatVector};

// --- lane movement ------------------------------------------------------------

impl<V: IntervalFloatVector, W: WideningPolicy> Swizzle<V::Lanes> for Interval<V, W> {
    #[inline(always)]
    fn swizzle_const<I: SwizzleIndices<V::Lanes>>(self, other: Self) -> Self {
        Self::from_bounds_unchecked(
            self.lo.swizzle_const::<I>(other.lo),
            self.hi.swizzle_const::<I>(other.hi),
        )
    }

    #[inline(always)]
    fn permutev_const<I: SwizzleIndices<V::Lanes>>(self) -> Self {
        Self::from_bounds_unchecked(self.lo.permutev_const::<I>(), self.hi.permutev_const::<I>())
    }
}

impl<V: IntervalFloatVector, W: WideningPolicy> Interleave for Interval<V, W> {
    #[inline(always)]
    fn interleave(self, other: Self) -> (Self, Self) {
        let (lo_a, lo_b) = self.lo.interleave(other.lo);
        let (hi_a, hi_b) = self.hi.interleave(other.hi);
        (
            Self::from_bounds_unchecked(lo_a, hi_a),
            Self::from_bounds_unchecked(lo_b, hi_b),
        )
    }

    #[inline(always)]
    fn deinterleave(self, other: Self) -> (Self, Self) {
        let (lo_a, lo_b) = self.lo.deinterleave(other.lo);
        let (hi_a, hi_b) = self.hi.deinterleave(other.hi);
        (
            Self::from_bounds_unchecked(lo_a, hi_a),
            Self::from_bounds_unchecked(lo_b, hi_b),
        )
    }
}

impl<V: IntervalFloatVector, W: WideningPolicy> GenericSelectable for Interval<V, W> {
    type SelectableMask = <V as GenericSelectable>::SelectableMask;

    #[inline(always)]
    fn select<M>(mask: M, t: Self, f: Self) -> Self
    where
        Self::SelectableMask: CastMask<M>,
    {
        let mask = <Self::SelectableMask as CastMask<M>>::mask_from(mask);
        Self::from_bounds_unchecked(mask.select(t.lo, f.lo), mask.select(t.hi, f.hi))
    }
}

impl<V: thermite::simd::HasIsa, W: 'static> thermite::simd::HasIsa for Interval<V, W> {
    type Native = V::Native;

    const ISA: thermite::isa::InstructionSet = V::ISA;
}

// --- const splat / new machinery (transliterated from Compensated) ------------

#[doc(hidden)]
pub struct IntervalNewImpl;

struct IntervalLoConst<C, V, W>(PhantomData<(C, V, W)>);
struct IntervalHiConst<C, V, W>(PhantomData<(C, V, W)>);

impl<C, V: IntervalFloatVector, W: WideningPolicy> NewConst<V::Element, V::Lanes> for IntervalLoConst<C, V, W>
where
    C: NewConst<IntervalElem<V::Element>, V::Lanes>,
{
    const VALUES: GenericArray<V::Element, V::Lanes> = const {
        let c_vals = C::VALUES;
        let src = c_vals.as_slice();
        let mut out: GenericArray<V::Element, V::Lanes> = unsafe { core::mem::zeroed() };
        let dst = out.as_mut_slice();
        let mut i = 0;
        while i < V::LANES {
            dst[i] = src[i].lo;
            i += 1;
        }
        core::mem::forget(c_vals);
        out
    };
}

impl<C, V: IntervalFloatVector, W: WideningPolicy> NewConst<V::Element, V::Lanes> for IntervalHiConst<C, V, W>
where
    C: NewConst<IntervalElem<V::Element>, V::Lanes>,
{
    const VALUES: GenericArray<V::Element, V::Lanes> = const {
        let c_vals = C::VALUES;
        let src = c_vals.as_slice();
        let mut out: GenericArray<V::Element, V::Lanes> = unsafe { core::mem::zeroed() };
        let dst = out.as_mut_slice();
        let mut i = 0;
        while i < V::LANES {
            dst[i] = src[i].hi;
            i += 1;
        }
        core::mem::forget(c_vals);
        out
    };
}

impl<T, V: IntervalFloatVector, W: WideningPolicy> VectorValue<T, Interval<V, W>> for IntervalNewImpl
where
    T: NewConst<IntervalElem<V::Element>, V::Lanes>,
{
    const VALUE: Interval<V, W> = Interval {
        lo: <<V as NewVector<V::Element, V::Lanes>>::New<IntervalLoConst<T, V, W>> as VectorValue<
            IntervalLoConst<T, V, W>,
            V,
        >>::VALUE,
        hi: <<V as NewVector<V::Element, V::Lanes>>::New<IntervalHiConst<T, V, W>> as VectorValue<
            IntervalHiConst<T, V, W>,
            V,
        >>::VALUE,
        _widen: PhantomData,
    };
}

impl<V: IntervalFloatVector, W: WideningPolicy> NewVector<IntervalElem<V::Element>, V::Lanes> for Interval<V, W> {
    type New<T: NewConst<IntervalElem<V::Element>, V::Lanes>> = IntervalNewImpl;
}

#[rustfmt::skip]
impl<V: IntervalFloatVector, W: WideningPolicy, E: SplatConst<IntervalElem<V::Element>>> VectorValue<E, Interval<V, W>> for Interval<V, W> {
    const VALUE: Interval<V, W> = const {
        struct Lo<V: IntervalFloatVector, E: SplatConst<IntervalElem<V::Element>>>(PhantomData<(V, E)>);
        struct Hi<V: IntervalFloatVector, E: SplatConst<IntervalElem<V::Element>>>(PhantomData<(V, E)>);

        impl<V: IntervalFloatVector, E: SplatConst<IntervalElem<V::Element>>> SplatConst<V::Element> for Lo<V, E> {
            const VALUE: V::Element = <E as SplatConst<IntervalElem<V::Element>>>::VALUE.lo;
        }

        impl<V: IntervalFloatVector, E: SplatConst<IntervalElem<V::Element>>> SplatConst<V::Element> for Hi<V, E> {
            const VALUE: V::Element = <E as SplatConst<IntervalElem<V::Element>>>::VALUE.hi;
        }

        Interval {
            lo: thermite::vector::const_splat::<V, Lo<V, E>>(),
            hi: thermite::vector::const_splat::<V, Hi<V, E>>(),
            _widen: PhantomData,
        }
    };
}

/// Splat-carrier lifting an [`IntervalElem`] `SplatConst` to a full
/// `Interval` vector constant (the analogue of Compensated's vector-const
/// carrier, consumed by `const_splat!`-style machinery).
#[doc(hidden)]
#[allow(dead_code)]
pub struct IntervalVectorConst<Inner>(PhantomData<Inner>);

impl<V, W, Inner> SplatConst<Interval<V, W>> for IntervalVectorConst<Inner>
where
    V: IntervalFloatVector,
    W: WideningPolicy,
    Inner: SplatConst<IntervalElem<V::Element>>,
{
    const VALUE: Interval<V, W> = <Interval<V, W> as VectorValue<Inner, Interval<V, W>>>::VALUE;
}

impl<V: IntervalFloatVector, W: WideningPolicy> SplatVector<IntervalElem<V::Element>> for Interval<V, W> {
    type Splat<T: SplatConst<IntervalElem<V::Element>>> = Self;
}

// --- `_c`/`_m`/`_z` blends for inherent unary/binary ops ----------------------

macro_rules! interval_masked {
    (unary: $($m:ident),* $(,)?) => { paste::paste! {
        $(
            #[inline(always)] fn [<$m _c>](self, mask: Self::Mask) -> Self { mask.select(self.$m(), self) }
            #[inline(always)] fn [<$m _m>](self, src: Self, mask: Self::Mask) -> Self { mask.select(self.$m(), src) }
            #[inline(always)] fn [<$m _z>](self, mask: Self::Mask) -> Self { mask.select(self.$m(), Self::EMPTY) }
        )*
    }};
    (binary: $($m:ident),* $(,)?) => { paste::paste! {
        $(
            #[inline(always)] fn [<$m _c>](self, mask: Self::Mask, rhs: Self) -> Self { mask.select(self.$m(rhs), self) }
            #[inline(always)] fn [<$m _m>](self, src: Self, mask: Self::Mask, rhs: Self) -> Self { mask.select(self.$m(rhs), src) }
            #[inline(always)] fn [<$m _z>](self, mask: Self::Mask, rhs: Self) -> Self { mask.select(self.$m(rhs), Self::EMPTY) }
        )*
    }};
}

// --- masked op-trait families -------------------------------------------------

macro_rules! impl_masked {
    (MUL_ADD: $($method:ident),*) => {paste::paste! {
        impl<V: IntervalFloatVector, W: WideningPolicy, A, B> thermite::vector::ops::MulAddExtMasked<V::Mask, A, B> for Interval<V, W>
        where
            Interval<V, W>: thermite::vector::ops::MulAddExt<A, B, Output = Self>,
        {
            $(
                #[inline(always)]
                fn [<$method _c>](self, mask: V::Mask, a: A, b: B) -> Self {
                    mask.select(self.[<$method>](a, b), self)
                }

                #[inline(always)]
                fn [<$method _m>](self, src: Self, mask: V::Mask, a: A, b: B) -> Self {
                    mask.select(self.[<$method>](a, b), src)
                }

                #[inline(always)]
                fn [<$method _z>](self, mask: V::Mask, a: A, b: B) -> Self {
                    mask.select(self.[<$method>](a, b), Self::EMPTY)
                }
            )*
        }

        impl<V: IntervalFloatVector, W: WideningPolicy, A, B> thermite::vector::ops::MulAddAssignExtMasked<V::Mask, A, B> for Interval<V, W>
        where
            Interval<V, W>: thermite::vector::ops::MulAddExt<A, B, Output = Self>,
        {
            $(
                #[inline(always)]
                fn [<$method _assign_c>](&mut self, mask: V::Mask, a: A, b: B) {
                    *self = mask.select(self.[<$method>](a, b), *self);
                }

                #[inline(always)]
                fn [<$method _assign_m>](&mut self, src: Self, mask: V::Mask, a: A, b: B) {
                    *self = mask.select(self.[<$method>](a, b), src);
                }

                #[inline(always)]
                fn [<$method _assign_z>](&mut self, mask: V::Mask, a: A, b: B) {
                    *self = mask.select(self.[<$method>](a, b), Self::EMPTY);
                }
            )*
        }
    }};

    ($trait:ident::$method:ident) => {paste::paste! {
        impl<V: IntervalFloatVector, W: WideningPolicy, Rhs> thermite::vector::ops::[<$trait Masked>]<V::Mask, Rhs> for Interval<V, W>
        where
            Interval<V, W>: core::ops::$trait<Rhs, Output = Self>,
        {
            #[inline(always)]
            fn [<$method _c>](self, mask: V::Mask, rhs: Rhs) -> Self {
                mask.select(core::ops::$trait::$method(self, rhs), self)
            }

            #[inline(always)]
            fn [<$method _m>](self, src: Self, mask: V::Mask, rhs: Rhs) -> Self {
                mask.select(core::ops::$trait::$method(self, rhs), src)
            }

            #[inline(always)]
            fn [<$method _z>](self, mask: V::Mask, rhs: Rhs) -> Self {
                mask.select(core::ops::$trait::$method(self, rhs), Self::EMPTY)
            }
        }

        impl<V: IntervalFloatVector, W: WideningPolicy, Rhs> thermite::vector::ops::[<$trait AssignMasked>]<V::Mask, Rhs> for Interval<V, W>
        where
            Interval<V, W>: core::ops::$trait<Rhs, Output = Self>,
        {
            #[inline(always)]
            fn [<$method _assign_c>](&mut self, mask: V::Mask, rhs: Rhs) {
                *self = mask.select(core::ops::$trait::$method(*self, rhs), *self);
            }

            #[inline(always)]
            fn [<$method _assign_m>](&mut self, src: Self, mask: V::Mask, rhs: Rhs) {
                *self = mask.select(core::ops::$trait::$method(*self, rhs), src);
            }

            #[inline(always)]
            fn [<$method _assign_z>](&mut self, mask: V::Mask, rhs: Rhs) {
                *self = mask.select(core::ops::$trait::$method(*self, rhs), Self::EMPTY);
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

impl<V: IntervalFloatVector, W: WideningPolicy> thermite::vector::ops::SquareMasked<V::Mask> for Interval<V, W> {
    #[inline(always)]
    fn square_c(self, mask: V::Mask) -> Self {
        mask.select(self.square_interval(), self)
    }
    #[inline(always)]
    fn square_m(self, src: Self, mask: V::Mask) -> Self {
        mask.select(self.square_interval(), src)
    }
    #[inline(always)]
    fn square_z(self, mask: V::Mask) -> Self {
        mask.select(self.square_interval(), Self::EMPTY)
    }
}

impl<V: IntervalFloatVector, W: WideningPolicy> thermite::vector::ops::AddSubExtMasked<V::Mask> for Interval<V, W> {
    #[inline(always)]
    fn addsub_c(self, mask: V::Mask, b: Self) -> Self {
        mask.select(thermite::vector::ops::AddSubExt::addsub(self, b), self)
    }
    #[inline(always)]
    fn addsub_m(self, src: Self, mask: V::Mask, b: Self) -> Self {
        mask.select(thermite::vector::ops::AddSubExt::addsub(self, b), src)
    }
    #[inline(always)]
    fn addsub_z(self, mask: V::Mask, b: Self) -> Self {
        mask.select(thermite::vector::ops::AddSubExt::addsub(self, b), Self::EMPTY)
    }

    #[inline(always)]
    fn fmaddsub_c(self, mask: V::Mask, b: Self, c: Self) -> Self {
        mask.select(thermite::vector::ops::AddSubExt::fmaddsub(self, b, c), self)
    }
    #[inline(always)]
    fn fmaddsub_m(self, src: Self, mask: V::Mask, b: Self, c: Self) -> Self {
        mask.select(thermite::vector::ops::AddSubExt::fmaddsub(self, b, c), src)
    }
    #[inline(always)]
    fn fmaddsub_z(self, mask: V::Mask, b: Self, c: Self) -> Self {
        mask.select(thermite::vector::ops::AddSubExt::fmaddsub(self, b, c), Self::EMPTY)
    }

    #[inline(always)]
    fn fmsubadd_c(self, mask: V::Mask, b: Self, c: Self) -> Self {
        mask.select(thermite::vector::ops::AddSubExt::fmsubadd(self, b, c), self)
    }
    #[inline(always)]
    fn fmsubadd_m(self, src: Self, mask: V::Mask, b: Self, c: Self) -> Self {
        mask.select(thermite::vector::ops::AddSubExt::fmsubadd(self, b, c), src)
    }
    #[inline(always)]
    fn fmsubadd_z(self, mask: V::Mask, b: Self, c: Self) -> Self {
        mask.select(thermite::vector::ops::AddSubExt::fmsubadd(self, b, c), Self::EMPTY)
    }
}

#[rustfmt::skip]
impl<V: IntervalFloatVector, W: WideningPolicy, A, B> thermite::vector::ops::MulAddAssignExt<A, B> for Interval<V, W>
where
    Self: thermite::vector::ops::MulAddExt<A, B, Output = Self>,
{
    #[inline(always)] fn mul_add_assign(&mut self, a: A, b: B) { *self = self.mul_add(a, b); }
    #[inline(always)] fn mul_sub_assign(&mut self, a: A, b: B) { *self = self.mul_sub(a, b); }
    #[inline(always)] fn nmul_add_assign(&mut self, a: A, b: B) { *self = self.nmul_add(a, b); }
    #[inline(always)] fn nmul_sub_assign(&mut self, a: A, b: B) { *self = self.nmul_sub(a, b); }
    #[inline(always)] fn mul_adde_assign(&mut self, a: A, b: B) { *self = self.mul_adde(a, b); }
    #[inline(always)] fn mul_sube_assign(&mut self, a: A, b: B) { *self = self.mul_sube(a, b); }
    #[inline(always)] fn nmul_adde_assign(&mut self, a: A, b: B) { *self = self.nmul_adde(a, b); }
    #[inline(always)] fn nmul_sube_assign(&mut self, a: A, b: B) { *self = self.nmul_sube(a, b); }
}

impl<V: IntervalFloatVector, W: WideningPolicy> core::iter::Sum for Interval<V, W> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(<Self as NumericVector>::ZERO, |a, b| a + b)
    }
}

impl<V: IntervalFloatVector, W: WideningPolicy> core::iter::Product for Interval<V, W> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(<Self as NumericVector>::ONE, |a, b| a * b)
    }
}

// --- GenericVector ------------------------------------------------------------

#[rustfmt::skip]
impl<V: IntervalFloatVector, W: WideningPolicy> GenericVector for Interval<V, W> {
    type Element = IntervalElem<V::Element>;

    const EMPTY: Self = Interval { lo: V::ZERO, hi: V::ZERO, _widen: PhantomData };
    const LANES: usize = V::LANES;

    type Lanes = V::Lanes;

    type Unsigned = V::Unsigned;
    type Signed = V::Signed;

    type Mask = V::Mask;

    // Both bounds move through the same permutation, so a permuted interval
    // vector is still per-lane [lo, hi] pairs.
    #[inline(always)]
    fn permutev(self, indices: Self::Unsigned) -> Self {
        Self::from_bounds_unchecked(self.lo.permutev(indices), self.hi.permutev(indices))
    }

    #[inline(always)]
    fn swizzle(self, other: Self, indices: Self::Unsigned) -> Self {
        Self::from_bounds_unchecked(
            self.lo.swizzle(other.lo, indices),
            self.hi.swizzle(other.hi, indices),
        )
    }

    #[inline(always)]
    fn new<const N: usize>(value: [Self::Element; N]) -> Self
    where
        thermite::generic_array::typenum::Const<N>: thermite::generic_array::IntoArrayLength<ArrayLength = Self::Lanes>
    {
        Interval {
            lo: V::new(value.map(|c| c.lo)),
            hi: V::new(value.map(|c| c.hi)),
            _widen: PhantomData,
        }
    }

    #[inline(always)]
    fn into_array(self) -> GenericArray<Self::Element, Self::Lanes> {
        let mut arr = GenericArray::default();
        for i in 0..Self::LANES {
            arr[i] = IntervalElem { lo: self.lo.extractv(i), hi: self.hi.extractv(i) };
        }
        arr
    }

    #[inline(always)]
    fn splat(value: Self::Element) -> Self {
        Self::from_bounds_unchecked(V::splat(value.lo), V::splat(value.hi))
    }

    #[inline(always)]
    fn single(value: Self::Element) -> Self {
        Self::from_bounds_unchecked(V::single(value.lo), V::single(value.hi))
    }

    // Memory layout (interleaved [lo, hi] pairs vs split planes) is not
    // settled, so every raw-memory entry point is deferred with it.
    unsafe fn load(_ptr: *const Self::Element) -> Self { todo!("interval memory layout") }
    unsafe fn load_m(_src: Self, _mask: Self::Mask, _ptr: *const Self::Element) -> Self { todo!("interval memory layout") }
    unsafe fn load_z(_mask: Self::Mask, _ptr: *const Self::Element) -> Self { todo!("interval memory layout") }
    unsafe fn load_unaligned(_ptr: *const Self::Element) -> Self { todo!("interval memory layout") }
    unsafe fn load_streaming(_ptr: *const Self::Element) -> Self { todo!("interval memory layout") }
    unsafe fn store(self, _ptr: *mut Self::Element) { todo!("interval memory layout") }
    unsafe fn store_masked(self, _mask: Self::Mask, _ptr: *mut Self::Element) { todo!("interval memory layout") }
    unsafe fn store_unaligned(self, _ptr: *mut Self::Element) { todo!("interval memory layout") }
    unsafe fn store_streaming(self, _ptr: *mut Self::Element) { todo!("interval memory layout") }
    unsafe fn lookup_unchecked(_values: &[Self::Element], _indices: Self::Unsigned) -> Self { todo!("interval memory layout") }

    #[inline(always)]
    fn interleave_by<const GROUP: usize>(self, other: Self) -> (Self, Self) {
        let (lo_a, lo_b) = self.lo.interleave_by::<GROUP>(other.lo);
        let (hi_a, hi_b) = self.hi.interleave_by::<GROUP>(other.hi);
        (Self::from_bounds_unchecked(lo_a, hi_a), Self::from_bounds_unchecked(lo_b, hi_b))
    }

    #[inline(always)]
    fn deinterleave_by<const GROUP: usize>(self, other: Self) -> (Self, Self) {
        let (lo_a, lo_b) = self.lo.deinterleave_by::<GROUP>(other.lo);
        let (hi_a, hi_b) = self.hi.deinterleave_by::<GROUP>(other.hi);
        (Self::from_bounds_unchecked(lo_a, hi_a), Self::from_bounds_unchecked(lo_b, hi_b))
    }

    #[inline(always)]
    fn interleave_radix<const N: usize>(inputs: [Self; N]) -> [Self; N] {
        let (mut lo, mut hi) = ([V::EMPTY; N], [V::EMPTY; N]);
        for i in 0..N {
            lo[i] = inputs[i].lo;
            hi[i] = inputs[i].hi;
        }
        let lo = V::interleave_radix::<N>(lo);
        let hi = V::interleave_radix::<N>(hi);
        let mut out = [Self::EMPTY; N];
        for i in 0..N {
            out[i] = Self::from_bounds_unchecked(lo[i], hi[i]);
        }
        out
    }

    #[inline(always)]
    fn deinterleave_radix<const N: usize>(inputs: [Self; N]) -> [Self; N] {
        let (mut lo, mut hi) = ([V::EMPTY; N], [V::EMPTY; N]);
        for i in 0..N {
            lo[i] = inputs[i].lo;
            hi[i] = inputs[i].hi;
        }
        let lo = V::deinterleave_radix::<N>(lo);
        let hi = V::deinterleave_radix::<N>(hi);
        let mut out = [Self::EMPTY; N];
        for i in 0..N {
            out[i] = Self::from_bounds_unchecked(lo[i], hi[i]);
        }
        out
    }

    #[inline(always)]
    fn interleave_radix_by<const N: usize, const GROUP: usize>(inputs: [Self; N]) -> [Self; N] {
        let (mut lo, mut hi) = ([V::EMPTY; N], [V::EMPTY; N]);
        for i in 0..N {
            lo[i] = inputs[i].lo;
            hi[i] = inputs[i].hi;
        }
        let lo = V::interleave_radix_by::<N, GROUP>(lo);
        let hi = V::interleave_radix_by::<N, GROUP>(hi);
        let mut out = [Self::EMPTY; N];
        for i in 0..N {
            out[i] = Self::from_bounds_unchecked(lo[i], hi[i]);
        }
        out
    }

    #[inline(always)]
    fn deinterleave_radix_by<const N: usize, const GROUP: usize>(inputs: [Self; N]) -> [Self; N] {
        let (mut lo, mut hi) = ([V::EMPTY; N], [V::EMPTY; N]);
        for i in 0..N {
            lo[i] = inputs[i].lo;
            hi[i] = inputs[i].hi;
        }
        let lo = V::deinterleave_radix_by::<N, GROUP>(lo);
        let hi = V::deinterleave_radix_by::<N, GROUP>(hi);
        let mut out = [Self::EMPTY; N];
        for i in 0..N {
            out[i] = Self::from_bounds_unchecked(lo[i], hi[i]);
        }
        out
    }

    #[inline(always)]
    fn broadcast<const I: usize>(self) -> Self {
        Self::from_bounds_unchecked(self.lo.broadcast::<I>(), self.hi.broadcast::<I>())
    }

    #[inline(always)]
    fn broadcastv(self, idx: usize) -> Self {
        Self::from_bounds_unchecked(self.lo.broadcastv(idx), self.hi.broadcastv(idx))
    }

    #[inline(always)]
    fn extract<const I: usize>(self) -> Self::Element {
        IntervalElem { lo: self.lo.extract::<I>(), hi: self.hi.extract::<I>() }
    }

    #[inline(always)]
    fn extractv(self, idx: usize) -> Self::Element {
        IntervalElem { lo: self.lo.extractv(idx), hi: self.hi.extractv(idx) }
    }

    #[inline(always)]
    fn insert<const I: usize>(self, value: Self::Element) -> Self {
        Self::from_bounds_unchecked(self.lo.insert::<I>(value.lo), self.hi.insert::<I>(value.hi))
    }

    #[inline(always)]
    fn insertv(self, idx: usize, value: Self::Element) -> Self {
        Self::from_bounds_unchecked(self.lo.insertv(idx, value.lo), self.hi.insertv(idx, value.hi))
    }

    #[inline(always)]
    fn reverse(self) -> Self {
        Self::from_bounds_unchecked(self.lo.reverse(), self.hi.reverse())
    }

    #[inline(always)]
    fn swap_bytes(self) -> Self {
        Self::from_bounds_unchecked(self.lo.swap_bytes(), self.hi.swap_bytes())
    }

    #[inline(always)]
    fn zz(self, mask: Self::Mask) -> Self {
        Self::from_bounds_unchecked(self.lo.zz(mask), self.hi.zz(mask))
    }

    #[inline(always)]
    fn nz(self, mask: Self::Mask) -> Self {
        Self::from_bounds_unchecked(self.lo.nz(mask), self.hi.nz(mask))
    }

    #[inline(always)]
    fn compress(self, mask: Self::Mask) -> Self {
        Self::from_bounds_unchecked(self.lo.compress(mask), self.hi.compress(mask))
    }

    #[inline(always)]
    fn compress_z(self, mask: Self::Mask) -> Self {
        Self::from_bounds_unchecked(self.lo.compress_z(mask), self.hi.compress_z(mask))
    }

    #[inline(always)]
    fn compress_m(self, src: Self, mask: Self::Mask) -> Self {
        Self::from_bounds_unchecked(self.lo.compress_m(src.lo, mask), self.hi.compress_m(src.hi, mask))
    }

    #[inline(always)]
    fn expand(self, mask: Self::Mask) -> Self {
        Self::from_bounds_unchecked(self.lo.expand(mask), self.hi.expand(mask))
    }

    #[inline(always)]
    fn expand_z(self, mask: Self::Mask) -> Self {
        Self::from_bounds_unchecked(self.lo.expand_z(mask), self.hi.expand_z(mask))
    }

    #[inline(always)]
    fn expand_m(self, src: Self, mask: Self::Mask) -> Self {
        Self::from_bounds_unchecked(self.lo.expand_m(src.lo, mask), self.hi.expand_m(src.hi, mask))
    }

    #[inline(always)]
    fn align<const OFFSET: usize>(self, other: Self) -> Self {
        Self::from_bounds_unchecked(self.lo.align::<OFFSET>(other.lo), self.hi.align::<OFFSET>(other.hi))
    }

    const HAS_NATIVE_ALIGN: bool = V::HAS_NATIVE_ALIGN;

    fn map<F>(mut self, f: F) -> Self
    where
        F: Fn(Self::Element) -> Self::Element,
    {
        for i in 0..Self::LANES {
            self = self.insertv(i, f(self.extractv(i)));
        }
        self
    }

    fn fold<F>(self, mut init: Self::Element, f: F) -> Self::Element
    where
        F: Fn(Self::Element, Self::Element) -> Self::Element,
    {
        for i in 0..Self::LANES {
            init = f(init, self.extractv(i));
        }
        init
    }

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

    unsafe fn load_deinterleaved<const M: usize>(_ptr: *const Self::Element) -> [Self; M] { todo!("interval memory layout") }
    unsafe fn store_interleaved<const M: usize>(_ptr: *mut Self::Element, _values: [Self; M]) { todo!("interval memory layout") }

    #[inline(always)] fn splat_m(src: Self, mask: Self::Mask, value: Self::Element) -> Self { mask.select(Self::splat(value), src) }
    #[inline(always)] fn splat_z(mask: Self::Mask, value: Self::Element) -> Self { mask.select(Self::splat(value), Self::EMPTY) }
    #[inline(always)] fn broadcast_c<const I: usize>(self, mask: Self::Mask) -> Self { mask.select(self.broadcast::<I>(), self) }
    #[inline(always)] fn broadcast_m<const I: usize>(self, src: Self, mask: Self::Mask) -> Self { mask.select(self.broadcast::<I>(), src) }
    #[inline(always)] fn broadcast_z<const I: usize>(self, mask: Self::Mask) -> Self { mask.select(self.broadcast::<I>(), Self::EMPTY) }
    #[inline(always)] fn broadcastv_c(self, mask: Self::Mask, idx: usize) -> Self { mask.select(self.broadcastv(idx), self) }
    #[inline(always)] fn broadcastv_m(self, src: Self, mask: Self::Mask, idx: usize) -> Self { mask.select(self.broadcastv(idx), src) }
    #[inline(always)] fn broadcastv_z(self, mask: Self::Mask, idx: usize) -> Self { mask.select(self.broadcastv(idx), Self::EMPTY) }
    #[inline(always)] fn reverse_c(self, mask: Self::Mask) -> Self { mask.select(self.reverse(), self) }
    #[inline(always)] fn reverse_m(self, src: Self, mask: Self::Mask) -> Self { mask.select(self.reverse(), src) }
    #[inline(always)] fn reverse_z(self, mask: Self::Mask) -> Self { mask.select(self.reverse(), Self::EMPTY) }
    #[inline(always)] fn swap_bytes_c(self, mask: Self::Mask) -> Self { mask.select(self.swap_bytes(), self) }
    #[inline(always)] fn swap_bytes_m(self, src: Self, mask: Self::Mask) -> Self { mask.select(self.swap_bytes(), src) }
    #[inline(always)] fn swap_bytes_z(self, mask: Self::Mask) -> Self { mask.select(self.swap_bytes(), Self::EMPTY) }
}

// --- comparisons: certainly semantics -----------------------------------------

#[rustfmt::skip]
impl<V: IntervalFloatVector, W: WideningPolicy> PartialOrdVector for Interval<V, W> {
    /// Certainly equal: both sides are the same single point.
    #[inline(always)]
    fn cmp_eq(self, other: Self) -> Self::Mask {
        self.lo.cmp_eq(self.hi) & other.lo.cmp_eq(other.hi) & self.lo.cmp_eq(other.lo)
    }

    /// Certainly not equal: the intervals are disjoint.
    #[inline(always)]
    fn cmp_ne(self, other: Self) -> Self::Mask {
        self.hi.cmp_lt(other.lo) | other.hi.cmp_lt(self.lo)
    }

    /// Certainly less: every point of `self` is below every point of `other`.
    #[inline(always)]
    fn cmp_lt(self, other: Self) -> Self::Mask {
        self.hi.cmp_lt(other.lo)
    }

    #[inline(always)]
    fn cmp_gt(self, other: Self) -> Self::Mask {
        self.lo.cmp_gt(other.hi)
    }

    #[inline(always)]
    fn cmp_le(self, other: Self) -> Self::Mask {
        self.hi.cmp_le(other.lo)
    }

    #[inline(always)]
    fn cmp_ge(self, other: Self) -> Self::Mask {
        self.lo.cmp_ge(other.hi)
    }
}

impl<V: IntervalFloatVector, W: WideningPolicy> Interval<V, W> {
    /// Possibly less: some point of `self` is below some point of `other`.
    /// The negation of `other.cmp_le(self)` on non-empty lanes.
    #[inline(always)]
    pub fn possibly_lt(self, other: Self) -> V::Mask {
        self.lo.cmp_lt(other.hi)
    }

    /// Possibly equal: the intervals overlap.
    #[inline(always)]
    pub fn possibly_eq(self, other: Self) -> V::Mask {
        self.lo.cmp_le(other.hi) & other.lo.cmp_le(self.hi)
    }
}

// --- numeric tower ------------------------------------------------------------

#[rustfmt::skip]
impl<V: IntervalFloatVector, W: WideningPolicy> num_traits::Bounded for Interval<V, W> {
    #[inline(always)] fn min_value() -> Self { Self::from_bounds_unchecked(V::MIN, V::MIN) }
    #[inline(always)] fn max_value() -> Self { Self::from_bounds_unchecked(V::MAX, V::MAX) }
}

impl<V: IntervalFloatVector, W: WideningPolicy> NumericVector for Interval<V, W> {
    // Integer conversions: in = exact degenerate constant, out = midpoint.
    #[inline(always)]
    fn to_signed_integer(self) -> Self::Signed {
        self.midpoint().to_signed_integer()
    }

    #[inline(always)]
    fn from_signed_integer(v: Self::Signed) -> Self {
        Self::degenerate(V::from_signed_integer(v))
    }

    #[inline(always)]
    fn to_unsigned_integer(self) -> Self::Unsigned {
        self.midpoint().to_unsigned_integer()
    }

    #[inline(always)]
    fn from_unsigned_integer(v: Self::Unsigned) -> Self {
        Self::degenerate(V::from_unsigned_integer(v))
    }

    const ZERO: Self = Interval {
        lo: V::ZERO,
        hi: V::ZERO,
        _widen: PhantomData,
    };
    const ONE: Self = Interval {
        lo: V::ONE,
        hi: V::ONE,
        _widen: PhantomData,
    };
    const TWO: Self = Interval {
        lo: V::TWO,
        hi: V::TWO,
        _widen: PhantomData,
    };
    const MIN: Self = Interval {
        lo: V::MIN,
        hi: V::MIN,
        _widen: PhantomData,
    };
    const MAX: Self = Interval {
        lo: V::MAX,
        hi: V::MAX,
        _widen: PhantomData,
    };

    fn sort_by<O: thermite::sort::SortOrder>(self) -> Self {
        todo!("lane sort keyed on interval order is deferred with the SortKey story")
    }

    fn bitonic_clean_by<O: thermite::sort::SortOrder>(self) -> Self {
        todo!("lane sort keyed on interval order is deferred with the SortKey story")
    }

    /// Certainly zero: the degenerate interval `[0, 0]`.
    #[inline(always)]
    fn is_zero(self) -> Self::Mask {
        self.lo.is_zero() & self.hi.is_zero()
    }

    #[inline(always)]
    fn is_all_zero(self) -> bool {
        self.lo.is_all_zero() && self.hi.is_all_zero()
    }

    /// The set minimum (endpoint-wise), NOT a comparison-select: interval
    /// `min` is monotone in each endpoint and exact.
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        self.min_interval(other)
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        self.max_interval(other)
    }

    #[inline(always)]
    fn clamp(self, min: Self, max: Self) -> Self {
        self.max_interval(min).min_interval(max)
    }

    #[inline(always)]
    fn min_element(self) -> Self::Element {
        let mut best = self.extractv(0);
        for i in 1..Self::LANES {
            let e = self.extractv(i);
            if e.lo < best.lo {
                best = e;
            }
        }
        best
    }

    #[inline(always)]
    fn max_element(self) -> Self::Element {
        let mut best = self.extractv(0);
        for i in 1..Self::LANES {
            let e = self.extractv(i);
            if e.hi > best.hi {
                best = e;
            }
        }
        best
    }

    fn sum_elements(self) -> Self::Element {
        self.fold(<Self::Element as thermite::element::Element>::ZERO, |a, b| a + b)
    }

    fn prod_elements(self) -> Self::Element {
        self.fold(<Self::Element as thermite::element::Element>::ONE, |a, b| a * b)
    }

    // Scans run on Self's enclosing `+`, like Compensated's run on its
    // double-double `+`: component-wise scans of lo/hi would not widen.
    #[inline(always)]
    fn prefix_sum(self) -> Self {
        thermite::scan_ladder!(forward, self, Self::ZERO, core::ops::Add::add)
    }

    #[inline(always)]
    fn reverse_prefix_sum(self) -> Self {
        thermite::scan_ladder!(reverse, self, Self::ZERO, core::ops::Add::add)
    }

    #[inline(always)]
    fn prefix_min(self) -> Self {
        thermite::scan_ladder!(forward, self, self.broadcast::<0>(), NumericVector::min)
    }

    #[inline(always)]
    fn prefix_max(self) -> Self {
        thermite::scan_ladder!(forward, self, self.broadcast::<0>(), NumericVector::max)
    }

    #[inline(always)]
    fn reverse_prefix_min(self) -> Self {
        thermite::scan_ladder!(reverse, self, Self::splat(self.last_element()), NumericVector::min)
    }

    #[inline(always)]
    fn reverse_prefix_max(self) -> Self {
        thermite::scan_ladder!(reverse, self, Self::splat(self.last_element()), NumericVector::max)
    }

    #[inline(always)]
    fn offset() -> Self {
        Self::degenerate(V::offset())
    }

    #[inline(always)]
    fn indexed() -> Self {
        Self::degenerate(V::indexed())
    }

    interval_masked!(binary: min, max);

    #[inline(always)]
    fn scale(self, factor: Self::Element) -> Self {
        self * Self::splat(factor)
    }

    #[inline(always)]
    fn scale_c(self, mask: Self::Mask, factor: Self::Element) -> Self {
        mask.select(self.scale(factor), self)
    }

    #[inline(always)]
    fn scale_m(self, src: Self, mask: Self::Mask, factor: Self::Element) -> Self {
        mask.select(self.scale(factor), src)
    }

    #[inline(always)]
    fn scale_z(self, mask: Self::Mask, factor: Self::Element) -> Self {
        mask.select(self.scale(factor), Self::EMPTY)
    }

    #[inline(always)]
    fn pairwise_sum(lo: Self, hi: Self) -> Self {
        let (even, odd) = lo.deinterleave(hi);
        even + odd
    }

    #[inline(always)]
    fn relaxed_pairwise_sum(lo: Self, hi: Self) -> Self {
        Self::pairwise_sum(lo, hi)
    }

    fn min_max_element(self) -> (Self::Element, Self::Element) {
        (self.min_element(), self.max_element())
    }

    #[inline(always)]
    fn arg_minmax(self) -> (usize, usize) {
        // Heuristic ordering by midpoint, consistent enough for pivots.
        self.midpoint().arg_minmax()
    }
}

impl<V: IntervalFloatVector, W: WideningPolicy> thermite::vector::ops::NegMasked<V::Mask> for Interval<V, W> {
    // Interval negation swaps the bounds, so this cannot delegate `neg_c`
    // component-wise. It selects between the negated and original intervals.
    #[inline(always)]
    fn neg_c(self, mask: V::Mask) -> Self {
        mask.select(self.negate(), self)
    }

    #[inline(always)]
    fn neg_m(self, src: Self, mask: V::Mask) -> Self {
        mask.select(self.negate(), src)
    }

    #[inline(always)]
    fn neg_z(self, mask: V::Mask) -> Self {
        mask.select(self.negate(), Self::EMPTY)
    }
}

impl<V: IntervalFloatVector, W: WideningPolicy> SignedVector for Interval<V, W> {
    const NEG_ONE: Self = Interval {
        lo: V::NEG_ONE,
        hi: V::NEG_ONE,
        _widen: PhantomData,
    };
    const MIN_POSITIVE: Self = Interval {
        lo: V::MIN_POSITIVE,
        hi: V::MIN_POSITIVE,
        _widen: PhantomData,
    };

    #[inline(always)]
    fn abs(self) -> Self {
        self.abs_interval()
    }

    /// The set image of signum: `[signum(lo), signum(hi)]` (signum is
    /// monotone). A zero-straddling lane correctly yields `[-1, 1]`.
    #[inline(always)]
    fn signum(self) -> Self {
        Self::from_bounds_unchecked(self.lo.signum(), self.hi.signum())
    }

    /// Certainly positive (every point > 0... spelled >= +0 to match the
    /// sign-bit convention of the inner form as closely as an interval can).
    #[inline(always)]
    fn is_positive(self) -> Self::Mask {
        self.lo.is_positive()
    }

    /// Certainly negative.
    #[inline(always)]
    fn is_negative(self) -> Self::Mask {
        self.hi.is_negative()
    }

    #[inline(always)]
    fn select_negative(self, if_neg: Self, if_pos: Self) -> Self {
        self.is_negative().select(if_neg, if_pos)
    }

    /// Sign-uncertain lanes (the sign interval straddles zero) enclose both
    /// outcomes: `hull(|x|, -|x|)`.
    #[inline(always)]
    fn copysign(self, sign: Self) -> Self {
        let a = self.abs_interval();
        let neg = a.negate();

        let certainly_neg = sign.is_negative();
        let certainly_pos = sign.is_positive();

        let unsure = !(certainly_neg | certainly_pos);
        let signed = certainly_neg.select(neg, a);
        unsure.select(neg.hull(a), signed)
    }

    interval_masked!(unary: abs);

    #[inline(always)]
    fn copysign_c(self, mask: Self::Mask, sign: Self) -> Self {
        mask.select(self.copysign(sign), self)
    }

    #[inline(always)]
    fn copysign_m(self, src: Self, mask: Self::Mask, sign: Self) -> Self {
        mask.select(self.copysign(sign), src)
    }

    #[inline(always)]
    fn copysign_z(self, mask: Self::Mask, sign: Self) -> Self {
        mask.select(self.copysign(sign), Self::EMPTY)
    }
}

// One generic form, like Compensated: covers Self -> Self (required by
// GenericVector) and cross-width casts.
impl<FROM, TO, W> CastVector<Interval<FROM, W>> for Interval<TO, W>
where
    FROM: IntervalFloatVector + CastVector<TO>,
    TO: IntervalFloatVector + CastVector<FROM>,
    W: WideningPolicy,
{
    fn cast_into(self) -> Interval<FROM, W> {
        Interval::<FROM, W>::cast_from(self)
    }

    fn cast_from(from: Interval<FROM, W>) -> Self {
        let from_size = size_of::<FROM::Element>();
        let to_size = size_of::<TO::Element>();

        if from_size > to_size {
            // Narrowing rounds each bound to nearest: widen outward after.
            Interval::from_bounds_unchecked(TO::cast_from(from.lo).next_down(), TO::cast_from(from.hi).next_up())
        } else {
            // Widening (or same-width) casts are exact.
            Interval::from_bounds_unchecked(TO::cast_from(from.lo), TO::cast_from(from.hi))
        }
    }
}

#[rustfmt::skip]
impl<V: IntervalFloatVector + BoundedFloatConsts<V>, W: WideningPolicy> FloatVector for Interval<V, W> {
    const HALF: Self = Interval { lo: V::HALF, hi: V::HALF, _widen: PhantomData };
    const NEG_ZERO: Self = Interval { lo: <V as FloatVector>::NEG_ZERO, hi: <V as FloatVector>::NEG_ZERO, _widen: PhantomData };
    const INFINITY: Self = Interval { lo: V::INFINITY, hi: V::INFINITY, _widen: PhantomData };
    const NEG_INFINITY: Self = Interval { lo: V::NEG_INFINITY, hi: V::NEG_INFINITY, _widen: PhantomData };
    // NAN is the closest thing to a poison value this type has: it fails the
    // `lo <= hi` invariant check everywhere, i.e. every lane reads as empty.
    const NAN: Self = Interval { lo: V::NAN, hi: V::NAN, _widen: PhantomData };
    const EPSILON: Self = Interval { lo: <V as FloatVector>::EPSILON, hi: <V as FloatVector>::EPSILON, _widen: PhantomData };

    /// Extended precision changes the _inner_ type, so this stays `Self` (like
    /// `Compensated`). Use `CastVector` to a wider inner explicitly.
    type ExtendedPrecision = Self;

    // Value-class predicates, certainly-semantics where meaningful.
    #[inline(always)] fn is_infinite(self) -> Self::Mask { self.lo.is_infinite() & self.lo.cmp_eq(self.hi) }
    #[inline(always)] fn is_finite(self) -> Self::Mask { self.lo.is_finite() & self.hi.is_finite() }
    #[inline(always)] fn is_nan(self) -> Self::Mask { self.lo.is_nan() | self.hi.is_nan() }
    #[inline(always)] fn is_zero_or_subnormal(self) -> Self::Mask { self.magnitude().is_zero_or_subnormal() }
    #[inline(always)] fn is_normal(self) -> Self::Mask { self.lo.is_normal() & self.hi.is_normal() }
    #[inline(always)] fn is_subnormal(self) -> Self::Mask { self.lo.is_subnormal() | self.hi.is_subnormal() }

    // Approximate reciprocal has no per-lane error bound: banned for
    // intervals regardless of anything (ground rule 2 of the plan).
    const HAS_APPROX_RCP: bool = false;
    const HAS_APPROX_RSQRT: bool = false;

    #[inline(always)] fn sqrt(self) -> Self { self.sqrt_interval() }
    #[inline(always)] fn rsqrt(self) -> Self { self.sqrt_interval().recip_interval() }
    #[inline(always)] fn rcp(self) -> Self { self.recip_interval() }

    // Monotone step functions: per-endpoint, exact.
    #[inline(always)] fn floor(self) -> Self { Self::from_bounds_unchecked(self.lo.floor(), self.hi.floor()) }
    #[inline(always)] fn ceil(self) -> Self { Self::from_bounds_unchecked(self.lo.ceil(), self.hi.ceil()) }
    #[inline(always)] fn round(self) -> Self { Self::from_bounds_unchecked(self.lo.round(), self.hi.round()) }
    #[inline(always)] fn trunc(self) -> Self { Self::from_bounds_unchecked(self.lo.trunc(), self.hi.trunc()) }
    // Valid enclosure of the (discontinuous) set image, wide across integer
    // boundaries by nature.
    #[inline(always)] fn fract(self) -> Self { self - self.trunc() }

    /// Like `copysign`: sign-uncertain lanes enclose both `x` and `-x`.
    #[inline(always)]
    fn mul_sign(self, sign: Self) -> Self {
        let neg = self.negate();

        let certainly_neg = sign.is_negative();
        let certainly_pos = sign.is_positive();

        let unsure = !(certainly_neg | certainly_pos);
        let signed = certainly_neg.select(neg, self);
        unsure.select(neg.hull(self), signed)
    }

    #[inline(always)]
    fn signed_zero(self) -> Self {
        Self::from_bounds_unchecked(self.lo.signed_zero(), self.hi.signed_zero())
    }

    // Set-map semantics: both endpoints step (monotone, exact). This is NOT
    // outward widening, the rounding module does that.
    #[inline(always)] fn next_up(self) -> Self { Self::from_bounds_unchecked(self.lo.next_up(), self.hi.next_up()) }
    #[inline(always)] fn next_down(self) -> Self { Self::from_bounds_unchecked(self.lo.next_down(), self.hi.next_down()) }

    unsafe fn block_autovectorization(&mut self) {
        unsafe {
            self.lo.block_autovectorization();
            self.hi.block_autovectorization();
        }
    }

    interval_masked!(unary: sqrt, rsqrt, rcp, floor, ceil, round, trunc, fract, signed_zero, next_up, next_down);
    interval_masked!(binary: mul_sign);

    #[inline(always)]
    fn mix(self, a: Self, b: Self) -> Self {
        a + (b - a) * self
    }

    fn with_bits<const N: usize, K: AsFloatVectorWithBitsKernel<Self, N>>(
        _values: [Self; N],
        _kernel: K,
    ) -> Option<<K as AsFloatVectorWithBitsKernel<Self, N>>::Output> {
        // No bit-level view of an interval exists, so kernels must handle None.
        None
    }
}

// --- FloatConsts: rigorous enclosures ----------------------------------------

macro_rules! enclosing_consts {
    ($($name:ident),* $(,)?) => {
        impl<V: IntervalFloatVector + BoundedFloatConsts<V>, W: WideningPolicy> FloatConsts for Interval<V, W> {
            // Each is [next_down(fl(c)), next_up(fl(c))]: `fl(c)` is within
            // half an ulp of the true constant, so the pair encloses it.
            $(const $name: Self = Interval {
                lo: <V as BoundedFloatConsts<V>>::$name.0,
                hi: <V as BoundedFloatConsts<V>>::$name.1,
                _widen: PhantomData,
            };)*
        }
    };
}

thermite::for_each_float_const!(enclosing_consts);
