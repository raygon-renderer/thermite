//! Element and vector-trait integration for [`Complex`].
//!
//! `Complex<E>` over a scalar float element implements
//! [`Element`]/[`SignedElement`]/[`FloatElement`], so it can be the element of a
//! complex vector; `Complex<V>` over a real [`FloatVector`] implements the
//! [`GenericVector`] -> [`FloatVector`] stack, with `Element = Complex<V::Element>`
//! and the mask and lanes of `V`.
//!
//! The ordering, sign and rounding semantics are in the [crate docs](crate).

use core::marker::PhantomData;
use core::ops::{Add, Div, Mul, Rem, Sub};

use num_traits::Bounded;

use thermite::Swizzle;
use thermite::element::{Element, FloatElement, SignedElement};
use thermite::generic_array::{GenericArray, IntoArrayLength, typenum::Const};
use thermite::register::SwizzleIndices;
use thermite::mask::{GenericMask, GenericSelectable};
use thermite::math::RealMathWithPolicy;
use thermite::math::algorithms::reduce_in_place;
use thermite::math::policy::DefaultPolicy;
use thermite::vector::ops::{AddSubExt, AddSubExtMasked, NegMasked, Square, SquareMasked};
use thermite::vector::{NewConst, NewVector, SplatConst, SplatVector, VectorValue, const_new, const_splat};
use thermite::{LargeInt, prelude::*};

use crate::{Complex, ComplexValue};

/// A real [`FloatVector`] usable as the inner storage of a [`Complex`] vector.
///
/// The complex primitives (`sqrt`, `abs`, `signum`, `rcp`, ...) are built out of
/// the inner vector's `hypot`/`reciprocal`, so the policy math library is required
/// here. `Dual`/`Compensated` split theirs into a separate tier; there is no useful
/// math-free tier to split out of this one.
pub trait ComplexFloatVector:
    ComplexValue + FloatVector<Element: ComplexValue> + CastVector<Self> + RealMathWithPolicy + SwizzleVector
{
}

impl<V> ComplexFloatVector for V where
    V: ComplexValue + FloatVector<Element: ComplexValue> + CastVector<V> + RealMathWithPolicy + SwizzleVector
{
}

// Lane swizzles apply to both components: re and im move through the same
// permutation, so a swizzled complex vector is the complex of the swizzled
// inputs.
impl<V: ComplexFloatVector> Swizzle<V::Lanes> for Complex<V> {
    #[inline(always)]
    fn swizzle(self, other: Self, indices: GenericArray<u32, V::Lanes>) -> Self {
        Self {
            re: self.re.swizzle(other.re, indices.clone()),
            im: self.im.swizzle(other.im, indices),
        }
    }

    #[inline(always)]
    fn permute(self, indices: GenericArray<u32, V::Lanes>) -> Self {
        Self {
            re: self.re.permute(indices.clone()),
            im: self.im.permute(indices),
        }
    }

    // Forward the `_const` forms per component - the trait defaults route
    // through the runtime-index methods and lose the immediate-encoded
    // shuffles.
    #[inline(always)]
    fn swizzle_const<I: SwizzleIndices<V::Lanes>>(self, other: Self) -> Self {
        Self {
            re: self.re.swizzle_const::<I>(other.re),
            im: self.im.swizzle_const::<I>(other.im),
        }
    }

    #[inline(always)]
    fn permute_const<I: SwizzleIndices<V::Lanes>>(self) -> Self {
        Self {
            re: self.re.permute_const::<I>(),
            im: self.im.permute_const::<I>(),
        }
    }
}

// --- Element stack: Complex<E> as a scalar element ---

#[rustfmt::skip]
impl<E: ComplexValue + Element> Element for Complex<E> {
    type Signed = <E as Element>::Signed;
    type Unsigned = <E as Element>::Unsigned;

    const ZERO: Self = Self::ZERO;
    const ONE: Self = Self::ONE;

    // The (documented, if artificial) order on complex numbers here is
    // lexicographic (re, im) - see `PartialOrdVector for Complex` - so the
    // order extremes are extreme in both components, and unordered values
    // (NaN in either part) exist exactly when the component type has them.
    const ORDER_MAX: Self = Self { re: E::ORDER_MAX, im: E::ORDER_MAX };
    const ORDER_MIN: Self = Self { re: E::ORDER_MIN, im: E::ORDER_MIN };
    const HAS_UNORDERED: bool = E::HAS_UNORDERED;

    #[inline(always)] fn from_i8(value: i8) -> Self { Self::real(E::from_i8(value)) }
    #[inline(always)] fn from_u8(value: u8) -> Self { Self::real(E::from_u8(value)) }
    #[inline(always)] fn from_u16(value: u16) -> Self { Self::real(E::from_u16(value)) }
}

impl<E: ComplexValue + FloatElement> Complex<E> {
    /// The modulus `$|z|$` of a complex element, the vector math library not being
    /// available at the element level.
    #[inline(always)]
    fn elem_modulus(self) -> E {
        E::sqrt(self.re.mul_add(self.re, self.im * self.im))
    }
}

impl<E: ComplexValue + FloatElement> SignedElement for Complex<E> {
    /// The modulus `$|z|$`, as a real complex number.
    #[inline(always)]
    fn abs(self) -> Self {
        Self::real(self.elem_modulus())
    }

    /// `$z/|z|$`, the unit complex number along `z`, and zero at the origin.
    #[inline(always)]
    fn signum(self) -> Self {
        let m = self.elem_modulus();

        if m == E::ZERO {
            return Self::ZERO;
        }

        Self::new(self.re / m, self.im / m)
    }
}

/// Splats a compile-time integer constant as a real `Complex<E>`.
pub struct ComplexIntConst<E, const VAL: LargeInt>(PhantomData<E>);

/// Splats a compile-time rational constant `N/D` as a real `Complex<E>`.
pub struct ComplexRatioConst<E, const NUM: LargeInt, const DEN: LargeInt>(PhantomData<E>);

impl<E: ComplexValue + FloatElement, const VAL: LargeInt> SplatConst<Complex<E>> for ComplexIntConst<E, VAL> {
    const VALUE: Complex<E> = Complex::real(<E::ConstInt<VAL> as SplatConst<E>>::VALUE);
}

impl<E: ComplexValue + FloatElement, const NUM: LargeInt, const DEN: LargeInt> SplatConst<Complex<E>>
    for ComplexRatioConst<E, NUM, DEN>
{
    const VALUE: Complex<E> = Complex::real(<E::ConstRatio<NUM, DEN> as SplatConst<E>>::VALUE);
}

#[rustfmt::skip]
impl<E: ComplexValue + FloatElement> FloatElement for Complex<E> {
    /// The principal square root, in Kahan's form; see `FloatVector::sqrt` below for
    /// why the symmetric formula is unusable.
    #[inline(always)]
    fn sqrt(this: Self) -> Self {
        let half = E::from_ratio(1, 2);

        let t = E::sqrt((SignedElement::abs(this.re) + this.elem_modulus()) * half);

        if t == E::ZERO {
            return Self::ZERO;
        }

        let half_im = this.im * half;

        if this.re >= E::ZERO {
            Self::new(t, half_im / t)
        } else {
            let i = if this.im < E::ZERO { -t } else { t };

            Self::new(SignedElement::abs(half_im) / t, i)
        }
    }

    // Rounding is componentwise; see the crate docs.
    #[inline(always)] fn floor(this: Self) -> Self { Self::new(E::floor(this.re), E::floor(this.im)) }
    #[inline(always)] fn ceil(this: Self) -> Self { Self::new(E::ceil(this.re), E::ceil(this.im)) }
    #[inline(always)] fn round(this: Self) -> Self { Self::new(E::round(this.re), E::round(this.im)) }
    #[inline(always)] fn trunc(this: Self) -> Self { Self::new(E::trunc(this.re), E::trunc(this.im)) }

    #[inline(always)] fn next_up(this: Self) -> Self { Self::new(E::next_up(this.re), E::next_up(this.im)) }
    #[inline(always)] fn next_down(this: Self) -> Self { Self::new(E::next_down(this.re), E::next_down(this.im)) }

    #[inline(always)]
    fn try_from_int(value: LargeInt) -> Option<Self> {
        E::try_from_int(value).map(Self::real)
    }

    #[inline(always)]
    fn try_from_ratio(n: LargeInt, d: LargeInt) -> Option<Self> {
        E::try_from_ratio(n, d).map(Self::real)
    }

    const HAS_INFINITY: bool = E::HAS_INFINITY;
    const HAS_SIGNED_ZERO: bool = E::HAS_SIGNED_ZERO;
    const HAS_SUBNORMALS: bool = E::HAS_SUBNORMALS;

    type ConstInt<const VAL: LargeInt> = ComplexIntConst<E, VAL>;
    type ConstRatio<const NUM: LargeInt, const DEN: LargeInt> = ComplexRatioConst<E, NUM, DEN>;
}

// --- HasIsa / Selectable / Interleave ---

impl<V: thermite::simd::HasIsa> thermite::simd::HasIsa for Complex<V> {
    type Native = V::Native;

    const ISA: thermite::isa::InstructionSet = V::ISA;
}

impl<V: ComplexFloatVector> GenericSelectable for Complex<V> {
    type SelectableMask = <V as GenericSelectable>::SelectableMask;

    #[inline(always)]
    fn select<M>(mask: M, t: Self, f: Self) -> Self
    where
        Self::SelectableMask: CastMask<M>,
    {
        let mask = <Self::SelectableMask as CastMask<M>>::mask_from(mask);

        Self::new(mask.select(t.re, f.re), mask.select(t.im, f.im))
    }
}

#[rustfmt::skip]
impl<V: ComplexFloatVector> Interleave for Complex<V> {
    #[inline(always)]
    fn interleave(self, other: Self) -> (Self, Self) {
        let (re_lo, re_hi) = self.re.interleave(other.re);
        let (im_lo, im_hi) = self.im.interleave(other.im);

        (Self::new(re_lo, im_lo), Self::new(re_hi, im_hi))
    }

    #[inline(always)]
    fn deinterleave(self, other: Self) -> (Self, Self) {
        let (re_lo, re_hi) = self.re.deinterleave(other.re);
        let (im_lo, im_hi) = self.im.deinterleave(other.im);

        (Self::new(re_lo, im_lo), Self::new(re_hi, im_hi))
    }
}

// --- Compile-time splat / new machinery ---

/// Carriers extracting one component of a `Complex` element constant.
struct ComplexReSplat<E, V>(PhantomData<(E, V)>);
struct ComplexImSplat<E, V>(PhantomData<(E, V)>);

impl<E, V: ComplexFloatVector> SplatConst<V::Element> for ComplexReSplat<E, V>
where
    E: SplatConst<Complex<V::Element>>,
{
    const VALUE: V::Element = <E as SplatConst<Complex<V::Element>>>::VALUE.re;
}

impl<E, V: ComplexFloatVector> SplatConst<V::Element> for ComplexImSplat<E, V>
where
    E: SplatConst<Complex<V::Element>>,
{
    const VALUE: V::Element = <E as SplatConst<Complex<V::Element>>>::VALUE.im;
}

impl<V: ComplexFloatVector> SplatVector<Complex<V::Element>> for Complex<V> {
    type Splat<T: SplatConst<Complex<V::Element>>> = Self;
}

impl<V: ComplexFloatVector, E: SplatConst<Complex<V::Element>>> VectorValue<E, Complex<V>> for Complex<V> {
    const VALUE: Complex<V> = Complex {
        re: const_splat::<V, ComplexReSplat<E, V>>(),
        im: const_splat::<V, ComplexImSplat<E, V>>(),
    };
}

/// Carriers extracting the per-lane components of a `Complex` element array.
struct ComplexReNew<C, V>(PhantomData<(C, V)>);
struct ComplexImNew<C, V>(PhantomData<(C, V)>);

macro_rules! impl_new_const {
    ($($carrier:ident => $field:ident),* $(,)?) => {$(
        impl<C, V: ComplexFloatVector> NewConst<V::Element, V::Lanes> for $carrier<C, V>
        where
            C: NewConst<Complex<V::Element>, V::Lanes>,
        {
            const VALUES: GenericArray<V::Element, V::Lanes> = const {
                let c_vals = C::VALUES;
                let src = c_vals.as_slice();
                let mut out: GenericArray<V::Element, V::Lanes> = unsafe { core::mem::zeroed() };
                let dst = out.as_mut_slice();
                let mut i = 0;
                while i < V::LANES {
                    dst[i] = src[i].$field;
                    i += 1;
                }
                core::mem::forget(c_vals);
                out
            };
        }
    )*};
}

impl_new_const!(ComplexReNew => re, ComplexImNew => im);

/// `VectorValue` implementor for per-lane (`new`) construction of `Complex` vectors.
pub struct ComplexNewImpl;

impl<T, V: ComplexFloatVector> VectorValue<T, Complex<V>> for ComplexNewImpl
where
    T: NewConst<Complex<V::Element>, V::Lanes>,
{
    const VALUE: Complex<V> = Complex {
        re: const_new::<V, V::Lanes, ComplexReNew<T, V>>(),
        im: const_new::<V, V::Lanes, ComplexImNew<T, V>>(),
    };
}

impl<V: ComplexFloatVector> NewVector<Complex<V::Element>, V::Lanes> for Complex<V> {
    type New<T: NewConst<Complex<V::Element>, V::Lanes>> = ComplexNewImpl;
}

// --- CastVector ---

impl<FROM, TO> CastVector<Complex<FROM>> for Complex<TO>
where
    FROM: ComplexFloatVector + CastVector<TO>,
    TO: ComplexFloatVector + CastVector<FROM>,
{
    #[inline(always)]
    fn cast_into(self) -> Complex<FROM> {
        Complex::<FROM>::cast_from(self)
    }

    #[inline(always)]
    fn cast_from(from: Complex<FROM>) -> Self {
        Complex::new(TO::cast_from(from.re), TO::cast_from(from.im))
    }
}

// --- ComplexVector ---

#[rustfmt::skip]
impl<V: ComplexFloatVector> crate::specialized::ComplexVector for Complex<V> {
    type Real = V;

    #[inline(always)] fn re(self) -> V { self.re }
    #[inline(always)] fn im(self) -> V { self.im }
    #[inline(always)] fn from_parts(re: V, im: V) -> Self { Self::new(re, im) }

    #[inline(always)]
    unsafe fn store_streaming_block(self, ptr: *mut Self) {
        // Stream each half with the real per-vector NT store (`_mm256_stream_ps` on AVX2; a
        // plain store where the backend has no NT). `&raw mut (*ptr).re/.im` are the true field
        // addresses, so no `repr` assumption; a `[Self]` slot is `Self`-aligned, `re` sits at
        // offset 0 and `im` at `size_of::<V>()`, both aligned for the vector NT store. Preserves
        // the planar `[re | im]` block layout (NO interleave - see the trait doc).
        unsafe {
            self.re.store_streaming((&raw mut (*ptr).re).cast());
            self.im.store_streaming((&raw mut (*ptr).im).cast());
        }
    }

    // Policy-free, so these are inherent on Complex<V> as well, where they also
    // serve the element-level Complex<f32>. The trait methods forward.
    #[inline(always)] fn conj(self) -> Self { Complex::conj(self) }
    #[inline(always)] fn norm_sqr(self) -> V { Complex::norm_sqr(self) }
    #[inline(always)] fn inv(self) -> Self { Complex::inv(self) }

    #[inline(always)] fn norm_l1(self) -> V { self.re.abs() + self.im.abs() }
}

// --- GenericVector ---

impl<V: ComplexFloatVector> Complex<V> {
    /// Splat a real and imaginary part across every lane.
    #[inline(always)]
    pub fn splat_parts(re: V::Element, im: V::Element) -> Self {
        Self::new(V::splat(re), V::splat(im))
    }
}

impl<V: ComplexFloatVector> GenericVector for Complex<V> {
    type Element = Complex<V::Element>;

    const EMPTY: Self = Self::ZERO;
    const LANES: usize = V::LANES;

    type Lanes = V::Lanes;

    type Unsigned = V::Unsigned;
    type Signed = V::Signed;
    type Mask = V::Mask;

    #[inline(always)]
    fn new<const N: usize>(value: [Self::Element; N]) -> Self
    where
        Const<N>: IntoArrayLength<ArrayLength = Self::Lanes>,
    {
        let mut re = [<V::Element as Element>::ZERO; N];
        let mut im = [<V::Element as Element>::ZERO; N];

        let mut i = 0;
        while i < N {
            re[i] = value[i].re;
            im[i] = value[i].im;
            i += 1;
        }

        Complex::new(V::new(re), V::new(im))
    }

    #[inline(always)]
    fn into_array(self) -> GenericArray<Self::Element, Self::Lanes> {
        let mut arr = GenericArray::default();

        for i in 0..Self::LANES {
            arr[i] = Complex::new(self.re.extractv(i), self.im.extractv(i));
        }

        arr
    }

    #[inline(always)]
    fn splat(value: Self::Element) -> Self {
        Self::splat_parts(value.re, value.im)
    }

    #[inline(always)]
    fn single(value: Self::Element) -> Self {
        Complex::new(V::single(value.re), V::single(value.im))
    }

    // A Complex element only guarantees the alignment of one V::Element. The aligned
    // load/store have no fast path to take and forward to the unaligned one.
    #[inline(always)]
    unsafe fn load(ptr: *const Self::Element) -> Self {
        unsafe { Self::load_unaligned(ptr) }
    }

    /// A `Complex` element is `#[repr(C)]` over two floats, so one element is
    /// [`load_deinterleaved::<1>`](Self::load_deinterleaved), which routes through the
    /// inner vector's register engine and not a lane-by-lane loop.
    #[inline(always)]
    unsafe fn load_unaligned(ptr: *const Self::Element) -> Self {
        let [out] = unsafe { Self::load_deinterleaved::<1>(ptr) };
        out
    }

    #[inline(always)]
    unsafe fn load_streaming(ptr: *const Self::Element) -> Self {
        unsafe { Self::load(ptr) }
    }

    #[inline(always)]
    fn interleave_by<const GROUP: usize>(self, other: Self) -> (Self, Self) {
        let (re_lo, re_hi) = self.re.interleave_by::<GROUP>(other.re);
        let (im_lo, im_hi) = self.im.interleave_by::<GROUP>(other.im);
        (Self::new(re_lo, im_lo), Self::new(re_hi, im_hi))
    }

    #[inline(always)]
    fn deinterleave_by<const GROUP: usize>(self, other: Self) -> (Self, Self) {
        let (re_lo, re_hi) = self.re.deinterleave_by::<GROUP>(other.re);
        let (im_lo, im_hi) = self.im.deinterleave_by::<GROUP>(other.im);
        (Self::new(re_lo, im_lo), Self::new(re_hi, im_hi))
    }

    #[inline(always)]
    fn interleave_radix<const N: usize>(inputs: [Self; N]) -> [Self; N] {
        let (mut re, mut im) = ([V::EMPTY; N], [V::EMPTY; N]);
        for i in 0..N {
            re[i] = inputs[i].re;
            im[i] = inputs[i].im;
        }
        let re = V::interleave_radix::<N>(re);
        let im = V::interleave_radix::<N>(im);
        let mut out = [Self::EMPTY; N];
        for i in 0..N {
            out[i] = Self::new(re[i], im[i]);
        }
        out
    }

    #[inline(always)]
    fn deinterleave_radix<const N: usize>(inputs: [Self; N]) -> [Self; N] {
        let (mut re, mut im) = ([V::EMPTY; N], [V::EMPTY; N]);
        for i in 0..N {
            re[i] = inputs[i].re;
            im[i] = inputs[i].im;
        }
        let re = V::deinterleave_radix::<N>(re);
        let im = V::deinterleave_radix::<N>(im);
        let mut out = [Self::EMPTY; N];
        for i in 0..N {
            out[i] = Self::new(re[i], im[i]);
        }
        out
    }

    #[inline(always)]
    fn deinterleave_radix_by<const N: usize, const GROUP: usize>(inputs: [Self; N]) -> [Self; N] {
        let (mut re, mut im) = ([V::EMPTY; N], [V::EMPTY; N]);
        for i in 0..N {
            re[i] = inputs[i].re;
            im[i] = inputs[i].im;
        }
        let re = V::deinterleave_radix_by::<N, GROUP>(re);
        let im = V::deinterleave_radix_by::<N, GROUP>(im);
        let mut out = [Self::EMPTY; N];
        for i in 0..N {
            out[i] = Self::new(re[i], im[i]);
        }
        out
    }

    #[inline(always)]
    fn interleave_radix_by<const N: usize, const GROUP: usize>(inputs: [Self; N]) -> [Self; N] {
        let (mut re, mut im) = ([V::EMPTY; N], [V::EMPTY; N]);
        for i in 0..N {
            re[i] = inputs[i].re;
            im[i] = inputs[i].im;
        }
        let re = V::interleave_radix_by::<N, GROUP>(re);
        let im = V::interleave_radix_by::<N, GROUP>(im);
        let mut out = [Self::EMPTY; N];
        for i in 0..N {
            out[i] = Self::new(re[i], im[i]);
        }
        out
    }

    /// `M` interleaved `Complex` streams are `2 * M` interleaved float streams, i.e.
    /// a grouped problem with `TAIL = 1` (see [`StreamGroup`]). `M` goes straight to
    /// the inner vector's [`GenericVector::load_deinterleaved_grouped`] - a NEON
    /// `LD2`/`LD4`, or a shuffle network on x86 - for any `M`.
    #[inline(always)]
    unsafe fn load_deinterleaved<const M: usize>(ptr: *const Self::Element) -> [Self; M] {
        let groups = unsafe { V::load_deinterleaved_grouped::<M, 1>(ptr as *const V::Element) };

        let mut out = [Self::EMPTY; M];
        let mut j = 0;
        while j < M {
            out[j] = Complex::new(groups[j].head, groups[j].tail[0]);
            j += 1;
        }
        out
    }

    /// The exact inverse of [`load_deinterleaved`](Self::load_deinterleaved).
    #[inline(always)]
    unsafe fn store_interleaved<const M: usize>(ptr: *mut Self::Element, values: [Self; M]) {
        let mut groups = [StreamGroup {
            head: V::ZERO,
            tail: [V::ZERO; 1],
        }; M];

        let mut j = 0;
        while j < M {
            groups[j] = StreamGroup {
                head: values[j].re,
                tail: [values[j].im],
            };
            j += 1;
        }

        unsafe { V::store_interleaved_grouped::<M, 1>(ptr as *mut V::Element, groups) }
    }

    #[inline(always)]
    unsafe fn load_m(src: Self, mask: Self::Mask, ptr: *const Self::Element) -> Self {
        let ptr = ptr as *const V::Element;

        // A Complex lane is two consecutive floats, so lane i of the mask covers
        // positions 2i and 2i+1 of the interleaved layout.
        let (a_mask, b_mask) = mask.interleave(mask);
        let (src_a, src_b) = src.re.interleave(src.im);

        let a = unsafe { V::load_m(src_a, a_mask, ptr) };
        let b = unsafe { V::load_m(src_b, b_mask, ptr.add(V::LANES)) };

        let (re, im) = a.deinterleave(b);
        Complex::new(re, im)
    }

    #[inline(always)]
    unsafe fn load_z(mask: Self::Mask, ptr: *const Self::Element) -> Self {
        unsafe { Self::load_m(Self::EMPTY, mask, ptr) }
    }

    // See the note on `load` above: the aligned store forwards to unaligned.
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
    unsafe fn store_masked(self, mask: Self::Mask, ptr: *mut Self::Element) {
        let ptr = ptr as *mut V::Element;

        let (a_mask, b_mask) = mask.interleave(mask);
        let (a, b) = self.re.interleave(self.im);

        unsafe {
            a.store_masked(a_mask, ptr);
            b.store_masked(b_mask, ptr.add(V::LANES));
        }
    }

    #[inline(always)]
    unsafe fn lookup_unchecked(values: &[Self::Element], indices: Self::Unsigned) -> Self {
        // The table reinterpreted as its interleaved [re, im, re, im, ...] floats:
        // element i lives at float positions 2i and 2i+1, so each component is one
        // gather through the inner vector's engine.
        let floats = unsafe { core::slice::from_raw_parts(values.as_ptr() as *const V::Element, values.len() * 2) };

        let re_idx = indices << 1;
        let im_idx = re_idx + Self::Unsigned::ONE;

        let re = unsafe { V::lookup_unchecked(floats, re_idx) };
        let im = unsafe { V::lookup_unchecked(floats, im_idx) };

        Complex::new(re, im)
    }

    #[inline(always)]
    fn broadcast<const I: usize>(self) -> Self {
        Complex::new(V::broadcast::<I>(self.re), V::broadcast::<I>(self.im))
    }

    #[inline(always)]
    fn broadcastv(self, idx: usize) -> Self {
        Complex::new(self.re.broadcastv(idx), self.im.broadcastv(idx))
    }

    #[inline(always)]
    fn extract<const I: usize>(self) -> Self::Element {
        Complex::new(V::extract::<I>(self.re), V::extract::<I>(self.im))
    }

    #[inline(always)]
    fn extractv(self, idx: usize) -> Self::Element {
        Complex::new(self.re.extractv(idx), self.im.extractv(idx))
    }

    #[inline(always)]
    fn insert<const I: usize>(self, value: Self::Element) -> Self {
        Complex::new(V::insert::<I>(self.re, value.re), V::insert::<I>(self.im, value.im))
    }

    #[inline(always)]
    fn insertv(self, idx: usize, value: Self::Element) -> Self {
        Complex::new(self.re.insertv(idx, value.re), self.im.insertv(idx, value.im))
    }

    #[inline(always)]
    fn reverse(self) -> Self {
        Complex::new(self.re.reverse(), self.im.reverse())
    }

    #[inline(always)]
    fn swap_bytes(self) -> Self {
        Complex::new(self.re.swap_bytes(), self.im.swap_bytes())
    }

    #[inline(always)]
    fn zz(self, mask: Self::Mask) -> Self {
        Complex::new(self.re.zz(mask), self.im.zz(mask))
    }

    #[inline(always)]
    fn nz(self, mask: Self::Mask) -> Self {
        Complex::new(self.re.nz(mask), self.im.nz(mask))
    }

    #[inline(always)]
    fn compress(self, mask: Self::Mask) -> Self {
        Complex::new(self.re.compress(mask), self.im.compress(mask))
    }

    #[inline(always)]
    fn compress_z(self, mask: Self::Mask) -> Self {
        Complex::new(self.re.compress_z(mask), self.im.compress_z(mask))
    }

    // Pure lane movement driven by `mask` alone, so both parts take the same
    // permutation and no lane ends up with a re/im pair from different sources.
    // `compress_m` too: its keep-lanes come from the population count of the shared
    // mask, so they land at the same positions in each part.
    #[inline(always)]
    fn compress_m(self, src: Self, mask: Self::Mask) -> Self {
        Complex::new(self.re.compress_m(src.re, mask), self.im.compress_m(src.im, mask))
    }

    #[inline(always)]
    fn expand(self, mask: Self::Mask) -> Self {
        Complex::new(self.re.expand(mask), self.im.expand(mask))
    }

    #[inline(always)]
    fn expand_z(self, mask: Self::Mask) -> Self {
        Complex::new(self.re.expand_z(mask), self.im.expand_z(mask))
    }

    #[inline(always)]
    fn expand_m(self, src: Self, mask: Self::Mask) -> Self {
        Complex::new(self.re.expand_m(src.re, mask), self.im.expand_m(src.im, mask))
    }

    #[inline(always)]
    fn align<const OFFSET: usize>(self, other: Self) -> Self {
        Complex::new(self.re.align::<OFFSET>(other.re), self.im.align::<OFFSET>(other.im))
    }

    // Both parts align through `V`, so this is only as native as `V` is.
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

    #[rustfmt::skip]
    #[inline(always)]    fn splat_m(src: Self, mask: Self::Mask, value: Self::Element) -> Self { mask.select(Self::splat(value), src) }
    #[rustfmt::skip]
    #[inline(always)]    fn splat_z(mask: Self::Mask, value: Self::Element) -> Self { mask.select(Self::splat(value), Self::EMPTY) }
    #[rustfmt::skip]
    #[inline(always)]    fn broadcast_c<const I: usize>(self, mask: Self::Mask) -> Self { mask.select(self.broadcast::<I>(), self) }
    #[rustfmt::skip]
    #[inline(always)]    fn broadcast_m<const I: usize>(self, src: Self, mask: Self::Mask) -> Self { mask.select(self.broadcast::<I>(), src) }
    #[rustfmt::skip]
    #[inline(always)]    fn broadcast_z<const I: usize>(self, mask: Self::Mask) -> Self { mask.select(self.broadcast::<I>(), Self::EMPTY) }
    #[rustfmt::skip]
    #[inline(always)]    fn broadcastv_c(self, mask: Self::Mask, idx: usize) -> Self { mask.select(self.broadcastv(idx), self) }
    #[rustfmt::skip]
    #[inline(always)]    fn broadcastv_m(self, src: Self, mask: Self::Mask, idx: usize) -> Self { mask.select(self.broadcastv(idx), src) }
    #[rustfmt::skip]
    #[inline(always)]    fn broadcastv_z(self, mask: Self::Mask, idx: usize) -> Self { mask.select(self.broadcastv(idx), Self::EMPTY) }
    #[rustfmt::skip]
    #[inline(always)]    fn reverse_c(self, mask: Self::Mask) -> Self { mask.select(self.reverse(), self) }
    #[rustfmt::skip]
    #[inline(always)]    fn reverse_m(self, src: Self, mask: Self::Mask) -> Self { mask.select(self.reverse(), src) }
    #[rustfmt::skip]
    #[inline(always)]    fn reverse_z(self, mask: Self::Mask) -> Self { mask.select(self.reverse(), Self::EMPTY) }
    #[rustfmt::skip]
    #[inline(always)]    fn swap_bytes_c(self, mask: Self::Mask) -> Self { mask.select(self.swap_bytes(), self) }
    #[rustfmt::skip]
    #[inline(always)]    fn swap_bytes_m(self, src: Self, mask: Self::Mask) -> Self { mask.select(self.swap_bytes(), src) }
    #[rustfmt::skip]
    #[inline(always)]    fn swap_bytes_z(self, mask: Self::Mask) -> Self { mask.select(self.swap_bytes(), Self::EMPTY) }
}

// --- PartialOrdVector: lexicographic by (re, im) ---

#[rustfmt::skip]
impl<V: ComplexFloatVector> PartialOrdVector for Complex<V> {
    #[inline(always)]
    fn cmp_eq(self, other: Self) -> Self::Mask {
        self.re.cmp_eq(other.re) & self.im.cmp_eq(other.im)
    }

    #[inline(always)]
    fn cmp_ne(self, other: Self) -> Self::Mask {
        self.re.cmp_ne(other.re) | self.im.cmp_ne(other.im)
    }

    // (re < other.re) | (re == other.re & im <op> other.im), in one ternlog.
    #[inline(always)]
    fn cmp_lt(self, other: Self) -> Self::Mask {
        let re_lt = self.re.cmp_lt(other.re);
        let re_eq = self.re.cmp_eq(other.re);
        let im_lt = self.im.cmp_lt(other.im);

        GenericMask::ternlog::<{ thermite::ternlog_imm!(A | (B & C)) }>(re_lt, re_eq, im_lt)
    }

    #[inline(always)]
    fn cmp_gt(self, other: Self) -> Self::Mask {
        let re_gt = self.re.cmp_gt(other.re);
        let re_eq = self.re.cmp_eq(other.re);
        let im_gt = self.im.cmp_gt(other.im);

        GenericMask::ternlog::<{ thermite::ternlog_imm!(A | (B & C)) }>(re_gt, re_eq, im_gt)
    }

    #[inline(always)]
    fn cmp_le(self, other: Self) -> Self::Mask {
        let re_lt = self.re.cmp_lt(other.re);
        let re_eq = self.re.cmp_eq(other.re);
        let im_le = self.im.cmp_le(other.im);

        GenericMask::ternlog::<{ thermite::ternlog_imm!(A | (B & C)) }>(re_lt, re_eq, im_le)
    }

    #[inline(always)]
    fn cmp_ge(self, other: Self) -> Self::Mask {
        let re_gt = self.re.cmp_gt(other.re);
        let re_eq = self.re.cmp_eq(other.re);
        let im_ge = self.im.cmp_ge(other.im);

        GenericMask::ternlog::<{ thermite::ternlog_imm!(A | (B & C)) }>(re_gt, re_eq, im_ge)
    }
}

// --- Masked ops ---

macro_rules! impl_masked {
    (MUL_ADD: $($method:ident),*) => {paste::paste! {
        impl<V: ComplexFloatVector, A, B> thermite::vector::ops::MulAddExtMasked<V::Mask, A, B> for Complex<V>
        where
            Complex<V>: thermite::vector::ops::MulAddExt<A, B, Output = Self>,
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

        impl<V: ComplexFloatVector, A, B> thermite::vector::ops::MulAddAssignExtMasked<V::Mask, A, B> for Complex<V>
        where
            Complex<V>: thermite::vector::ops::MulAddExt<A, B, Output = Self>,
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
        impl<V: ComplexFloatVector, Rhs> thermite::vector::ops::[<$trait Masked>]<V::Mask, Rhs> for Complex<V>
        where
            Complex<V>: core::ops::$trait<Rhs, Output = Self>,
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

        impl<V: ComplexFloatVector, Rhs> thermite::vector::ops::[<$trait AssignMasked>]<V::Mask, Rhs> for Complex<V>
        where
            Complex<V>: core::ops::$trait<Rhs, Output = Self>,
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
// Lane-alternating add/sub (`AddSubExt`), over the *inner vector's* lanes - so the
// even/odd parity applies per complex number. `Add`/`Sub` are component-wise on
// `re`/`im`, so this is exact: `neg_even` flips the even-lane signs of both parts,
// then:
//   addsub(a, b)      = a + neg_even(b)
//   fmaddsub(a, b, c) = a*b + neg_even(c)   (via the complex product-rule mul_adde)
//   fmsubadd(a, b, c) = a*b - neg_even(c)
// =====================================================================================

#[inline(always)]
fn neg_even_complex<V: ComplexFloatVector>(x: Complex<V>) -> Complex<V> {
    // `addsub(0, w) = [-w0, w1, -w2, ...]` flips the even lanes exactly.
    Complex::new(V::ZERO.addsub(x.re), V::ZERO.addsub(x.im))
}

impl<V: ComplexFloatVector> AddSubExt for Complex<V> {
    type Output = Self;

    #[inline(always)] fn addsub(self, b: Self) -> Self { self + neg_even_complex(b) }
    #[inline(always)] fn fmaddsub(self, b: Self, c: Self) -> Self { self.mul_adde(b, neg_even_complex(c)) }
    #[inline(always)] fn fmsubadd(self, b: Self, c: Self) -> Self { self.mul_sube(b, neg_even_complex(c)) }
}

impl<V: ComplexFloatVector> AddSubExtMasked<V::Mask> for Complex<V> {
    #[inline(always)] fn addsub_c(self, mask: V::Mask, b: Self) -> Self { mask.select(self.addsub(b), self) }
    #[inline(always)] fn addsub_m(self, src: Self, mask: V::Mask, b: Self) -> Self { mask.select(self.addsub(b), src) }
    #[inline(always)] fn addsub_z(self, mask: V::Mask, b: Self) -> Self { mask.select(self.addsub(b), Self::EMPTY) }

    #[inline(always)] fn fmaddsub_c(self, mask: V::Mask, b: Self, c: Self) -> Self { mask.select(self.fmaddsub(b, c), self) }
    #[inline(always)] fn fmaddsub_m(self, src: Self, mask: V::Mask, b: Self, c: Self) -> Self { mask.select(self.fmaddsub(b, c), src) }
    #[inline(always)] fn fmaddsub_z(self, mask: V::Mask, b: Self, c: Self) -> Self { mask.select(self.fmaddsub(b, c), Self::EMPTY) }

    #[inline(always)] fn fmsubadd_c(self, mask: V::Mask, b: Self, c: Self) -> Self { mask.select(self.fmsubadd(b, c), self) }
    #[inline(always)] fn fmsubadd_m(self, src: Self, mask: V::Mask, b: Self, c: Self) -> Self { mask.select(self.fmsubadd(b, c), src) }
    #[inline(always)] fn fmsubadd_z(self, mask: V::Mask, b: Self, c: Self) -> Self { mask.select(self.fmsubadd(b, c), Self::EMPTY) }
}

impl<V: ComplexFloatVector> SquareMasked<V::Mask> for Complex<V> {
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

// The _c/_m/_z variants of the unary (fn m(self) -> Self) and binary
// (fn m(self, Self) -> Self) ops, as plain blends, like impl_masked! above.
macro_rules! complex_masked {
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

// --- NumericVector ---

#[rustfmt::skip]
impl<V: ComplexFloatVector> Bounded for Complex<V> {
    #[inline(always)] fn min_value() -> Self { Complex::new(V::MIN, V::MIN) }
    #[inline(always)] fn max_value() -> Self { Complex::new(V::MAX, V::MAX) }
}

#[rustfmt::skip]
/// The lane-sort key: strictly-before under the lexicographic (re, im)
/// order, i.e. `cmp_lt`. See `thermite::sort::SortKey` for why this is a
/// static trait method and not a closure.
impl<V: ComplexFloatVector> thermite::sort::SortKey<Self> for Complex<V> {
    #[inline(always)]
    fn key_lt(a: Self, b: Self) -> V::Mask {
        a.cmp_lt(b)
    }
}

/// Scalar insertion walk over whole lanes, for widths past the network ladder.
/// Quadratic, like core's `sort_any`; compares composite elements through
/// `PartialOrd` (lexicographic, matching the vector comparisons).
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

impl<V: ComplexFloatVector> NumericVector for Complex<V> {
    const ZERO: Self = Complex::new(V::ZERO, V::ZERO);
    const ONE: Self = Complex::new(V::ONE, V::ZERO);
    const TWO: Self = Self::real(V::TWO);

    // The lexicographic extremes, consistent with the ordering above.
    const MIN: Self = Complex::new(V::MIN, V::MIN);
    const MAX: Self = Complex::new(V::MAX, V::MAX);

    #[inline(always)] fn is_zero(self) -> Self::Mask { self.re.is_zero() & self.im.is_zero() }
    #[inline(always)] fn is_all_zero(self) -> bool { self.re.is_all_zero() && self.im.is_all_zero() }

    // Lane sorts are keyed on the lexicographic (re, im) order - which is exactly
    // `cmp_lt` here, so the key IS the comparison. Each compare-exchange derives one
    // routing mask from it and moves both components through the same permutation
    // and select (`thermite::sort::sort_lanes_by_key`).
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

    #[inline(always)] fn min(self, other: Self) -> Self { self.cmp_lt(other).select(self, other) }
    #[inline(always)] fn max(self, other: Self) -> Self { self.cmp_gt(other).select(self, other) }

    // Compare against both bounds once, then blend per component.
    #[inline(always)]
    fn clamp(self, min: Self, max: Self) -> Self {
        let is_lt = self.cmp_lt(min);
        let is_gt = self.cmp_gt(max);

        Complex::new(
            is_lt.select(min.re, is_gt.select(max.re, self.re)),
            is_lt.select(min.im, is_gt.select(max.im, self.im)),
        )
    }

    // Lexicographic order spans both components: there is no single inner vector to
    // hand to arg_minmax. Reduce the extracted elements in log2(LANES) steps with the
    // derived PartialOrd.
    #[inline(always)]
    fn min_element(self) -> Self::Element {
        let mut arr = self.into_array();
        reduce_in_place(&mut arr, |a, b| if b < a { b } else { a });
        arr[0]
    }

    #[inline(always)]
    fn max_element(self) -> Self::Element {
        let mut arr = self.into_array();
        reduce_in_place(&mut arr, |a, b| if b > a { b } else { a });
        arr[0]
    }

    #[inline(always)]
    fn min_max_element(self) -> (Self::Element, Self::Element) {
        (self.min_element(), self.max_element())
    }

    #[inline(always)]
    fn arg_minmax(self) -> (usize, usize) {
        let arr = self.into_array();

        let (mut lo, mut hi) = (0, 0);

        for i in 1..Self::LANES {
            if arr[i] < arr[lo] { lo = i; }
            if arr[i] > arr[hi] { hi = i; }
        }

        (lo, hi)
    }

    // Sum is linear and commutes with the re/im split: each component reduces through
    // the inner vector's horizontal sum. No extracting and folding LANES scalar
    // complex numbers.
    #[inline(always)]
    fn sum_elements(self) -> Self::Element {
        Complex::new(self.re.sum_elements(), self.im.sum_elements())
    }

    // Complex addition is componentwise, so the scan is too - the same argument as
    // `sum_elements`, one step at a time instead of all the way down.
    #[inline(always)]
    fn prefix_sum(self) -> Self {
        Complex::new(self.re.prefix_sum(), self.im.prefix_sum())
    }

    #[inline(always)]
    fn reverse_prefix_sum(self) -> Self {
        Complex::new(self.re.reverse_prefix_sum(), self.im.reverse_prefix_sum())
    }

    // min/max are lexicographic over both parts (see the ordering above), so there is
    // no per-component scan to delegate to: scanning `re` and `im` separately would
    // pair a real part from one lane with an imaginary part from another. The ladder
    // runs on whole complex values through `Self::min`/`Self::max`.
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
        thermite::scan_ladder!(reverse, self, self.reverse().broadcast::<0>(), Self::min)
    }

    #[inline(always)]
    fn reverse_prefix_max(self) -> Self {
        thermite::scan_ladder!(reverse, self, self.reverse().broadcast::<0>(), Self::max)
    }

    // Product is not componentwise (the parts cross-multiply) and needs complex
    // multiplies across lanes. The tree reduction keeps the dependency chain at log
    // depth.
    #[inline(always)]
    fn prod_elements(self) -> Self::Element {
        let mut arr = self.into_array();
        reduce_in_place(&mut arr, |a, b| a * b);
        arr[0]
    }

    #[inline(always)] fn offset() -> Self { Self::real(V::offset()) }
    #[inline(always)] fn indexed() -> Self { Self::real(V::indexed()) }

    #[inline(always)] fn scale(self, factor: Self::Element) -> Self { self * Self::splat(factor) }

    #[inline(always)] fn scale_c(self, mask: Self::Mask, factor: Self::Element) -> Self { mask.select(<Self as NumericVector>::scale(self, factor), self) }
    #[inline(always)] fn scale_m(self, src: Self, mask: Self::Mask, factor: Self::Element) -> Self { mask.select(<Self as NumericVector>::scale(self, factor), src) }
    #[inline(always)] fn scale_z(self, mask: Self::Mask, factor: Self::Element) -> Self { mask.select(<Self as NumericVector>::scale(self, factor), Self::ZERO) }
    complex_masked!(binary: min, max);

    // pairwise_sum is a lane rearrange-and-add, and a lane's components sit at the
    // same index in re and im, so it applies componentwise.
    #[inline(always)]
    fn pairwise_sum(lo: Self, hi: Self) -> Self {
        Complex::new(V::pairwise_sum(lo.re, hi.re), V::pairwise_sum(lo.im, hi.im))
    }

    #[inline(always)]
    fn relaxed_pairwise_sum(lo: Self, hi: Self) -> Self {
        Complex::new(
            V::relaxed_pairwise_sum(lo.re, hi.re),
            V::relaxed_pairwise_sum(lo.im, hi.im),
        )
    }
}

// --- SignedVector ---

impl<V: ComplexFloatVector> NegMasked<V::Mask> for Complex<V> {
    #[inline(always)]
    fn neg_c(self, mask: V::Mask) -> Self {
        Complex::new(self.re.neg_c(mask), self.im.neg_c(mask))
    }

    #[inline(always)]
    fn neg_m(self, src: Self, mask: V::Mask) -> Self {
        Complex::new(self.re.neg_m(src.re, mask), self.im.neg_m(src.im, mask))
    }

    #[inline(always)]
    fn neg_z(self, mask: V::Mask) -> Self {
        Complex::new(self.re.neg_z(mask), self.im.neg_z(mask))
    }
}

impl<V: ComplexFloatVector> Complex<V> {
    /// The modulus `$|z|$` under the default policy, for the vector-trait methods
    /// that take no policy. The policy-aware form is
    /// [`norm_p`](crate::ComplexMathWithPolicy::norm_p).
    #[inline(always)]
    pub(crate) fn modulus(self) -> V {
        self.re.hypot_p::<DefaultPolicy>(self.im)
    }
}

#[rustfmt::skip]
impl<V: ComplexFloatVector> SignedVector for Complex<V> {
    const NEG_ONE: Self = Self::real(V::NEG_ONE);
    const MIN_POSITIVE: Self = Self::real(V::MIN_POSITIVE);

    /// The modulus `$|z|$`, as a real complex number.
    #[inline(always)]
    fn abs(self) -> Self {
        Self::real(self.modulus())
    }

    /// `$z/|z|$`, the unit complex number along `z`, and zero at the origin.
    ///
    /// Preserves the real identity `abs(z) * signum(z) == z`.
    #[inline(always)]
    fn signum(self) -> Self {
        let m = self.modulus();
        let is_zero = m.is_zero();
        let inv = m.reciprocal_p::<DefaultPolicy>();

        // nz() zeroes where the mask is set, pinning signum(0) to 0; the division
        // there gives 0 * inf = NaN.
        Complex::new((self.re * inv).nz(is_zero), (self.im * inv).nz(is_zero))
    }

    // Sign-bit ops are componentwise. A mask has one bit per lane, and the sign
    // predicates report the sign of the real part. See the crate docs.
    #[inline(always)] fn is_positive(self) -> Self::Mask { self.re.is_positive() }
    #[inline(always)] fn is_negative(self) -> Self::Mask { self.re.is_negative() }
    #[inline(always)] fn select_negative(self, if_neg: Self, if_pos: Self) -> Self { self.is_negative().select(if_neg, if_pos) }

    #[inline(always)]
    fn copysign(self, sign: Self) -> Self {
        Complex::new(self.re.copysign(sign.re), self.im.copysign(sign.im))
    }

    complex_masked!(unary: abs);
    complex_masked!(binary: copysign);
}

// --- FloatVector ---

#[rustfmt::skip]
impl<V: ComplexFloatVector> FloatVector for Complex<V> {
    const HALF: Self = Self::real(<V as FloatVector>::HALF);
    const NEG_ZERO: Self = Self::real(<V as FloatVector>::NEG_ZERO);
    const EPSILON: Self = Self::real(<V as FloatVector>::EPSILON);

    // Directed along the real axis. A NaN in either component makes the whole
    // number NaN.
    const INFINITY: Self = Self::real(<V as FloatVector>::INFINITY);
    const NEG_INFINITY: Self = Self::real(<V as FloatVector>::NEG_INFINITY);
    const NAN: Self = Complex::new(<V as FloatVector>::NAN, <V as FloatVector>::NAN);

    type ExtendedPrecision = Self;

    // rcp/rsqrt go through the inner vector's reciprocal, and are approximate exactly
    // when it is.
    const HAS_APPROX_RCP: bool = V::HAS_APPROX_RCP;
    const HAS_APPROX_RSQRT: bool = V::HAS_APPROX_RCP;

    #[inline(always)] fn is_nan(self) -> Self::Mask { self.re.is_nan() | self.im.is_nan() }
    #[inline(always)] fn is_infinite(self) -> Self::Mask { self.re.is_infinite() | self.im.is_infinite() }
    #[inline(always)] fn is_finite(self) -> Self::Mask { self.re.is_finite() & self.im.is_finite() }

    // |z| is negligible only if both components are, and z is normal if either
    // component is and the other is finite.
    #[inline(always)] fn is_zero_or_subnormal(self) -> Self::Mask { self.re.is_zero_or_subnormal() & self.im.is_zero_or_subnormal() }
    #[inline(always)] fn is_normal(self) -> Self::Mask { (self.re.is_normal() | self.im.is_normal()) & self.is_finite() }
    #[inline(always)] fn is_subnormal(self) -> Self::Mask { (self.re.is_subnormal() | self.im.is_subnormal()) & self.is_zero_or_subnormal() }

    /// The principal square root, with a branch cut on the negative real axis.
    #[inline(always)]
    fn sqrt(self) -> Self {
        // Kahan's formulation. The symmetric form
        //
        //     sqrt((|z| + re)/2) + i*sign(im)*sqrt((|z| - re)/2)
        //
        // is only safe on one side: for a nearly-real z the smaller of |z| +- re is a
        // difference of nearly equal numbers, so it cancels to noise and the small
        // component has no correct digits left (asin/acos/asinh feed in arguments of
        // exactly that shape). Take the large component from the modulus, always
        // adding, and recover the small one from re*im_out = im/2. One division, no
        // cancellation.
        let half = <V as FloatVector>::HALF;

        let t = ((self.re.abs() + self.modulus()) * half).sqrt(); // sqrt((|re| + |z|)/2)
        let half_im = self.im * half;

        // re >= 0: (t, im/2t).   re < 0: (|im|/2t, sign(im)*t).
        let re_pos = self.re.cmp_ge(V::ZERO);

        let re = re_pos.select(t, half_im.abs() / t);
        let im = re_pos.select(half_im / t, t.mul_sign(self.im));

        // z == 0 makes t == 0, so both quotients are 0/0 = NaN; sqrt(0) is 0.
        let is_zero = t.is_zero();

        Complex::new(re.nz(is_zero), im.nz(is_zero))
    }

    /// `$1/z = \bar{z}/|z|^2$`
    #[inline(always)]
    fn rcp(self) -> Self {
        let inv = self.norm_sqr().rcp();

        Complex::new(self.re * inv, -(self.im * inv))
    }

    /// `$1/\sqrt{z} = \overline{\sqrt{z}}/|z|$`, since `$|\sqrt{z}|^2 = |z|$`.
    #[inline(always)]
    fn rsqrt(self) -> Self {
        let s = self.sqrt();
        let inv = self.modulus().rcp();

        Complex::new(s.re * inv, -(s.im * inv))
    }

    // Rounding is componentwise; see the crate docs.
    #[inline(always)] fn floor(self) -> Self { Complex::new(self.re.floor(), self.im.floor()) }
    #[inline(always)] fn ceil(self) -> Self { Complex::new(self.re.ceil(), self.im.ceil()) }
    #[inline(always)] fn round(self) -> Self { Complex::new(self.re.round(), self.im.round()) }
    #[inline(always)] fn trunc(self) -> Self { Complex::new(self.re.trunc(), self.im.trunc()) }
    #[inline(always)] fn fract(self) -> Self { Complex::new(self.re.fract(), self.im.fract()) }

    #[inline(always)] fn mul_sign(self, sign: Self) -> Self { Complex::new(self.re.mul_sign(sign.re), self.im.mul_sign(sign.im)) }
    #[inline(always)] fn signed_zero(self) -> Self { Complex::new(self.re.signed_zero(), self.im.signed_zero()) }

    #[inline(always)] fn next_up(self) -> Self { Complex::new(self.re.next_up(), self.im.next_up()) }
    #[inline(always)] fn next_down(self) -> Self { Complex::new(self.re.next_down(), self.im.next_down()) }

    #[inline(always)]
    unsafe fn block_autovectorization(&mut self) {
        unsafe {
            self.re.block_autovectorization();
            self.im.block_autovectorization();
        }
    }

    // mix(t) = a*(1 - t) + b*t = a + (b - a)*t, through complex arithmetic.
    #[inline(always)] fn mix(self, a: Self, b: Self) -> Self { a + (b - a) * self }

    complex_masked!(unary: sqrt, rsqrt, rcp, floor, ceil, round, trunc, fract, signed_zero, next_up, next_down);
    complex_masked!(binary: mul_sign);
}
