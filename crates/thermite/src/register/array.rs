use core::mem::MaybeUninit;
use core::ops::Mul;

use generic_array::ArrayLength;
use generic_array::typenum::{self, Const, Prod, ToUInt, Unsigned};

use crate::{Vector, math::policy::Policy};

use super::*;

// generates array_zip2, array_zip3, array_zip4, array_zip5, etc., to combine many arrays using a provided function
macro_rules! decl_array_zips {
    ($($count:literal => ($($part:ident,)+)),* $(,)?) => {paste::paste! {$(
        #[inline(always)] #[allow(dead_code)]
        fn [<array_zip $count>]<$($part: Copy),+, U, F, const N: usize>($([<$part:lower>]: [$part; N]),+, mut f: F) -> [U; N]
        where F: FnMut($($part),+) -> U {
            let mut result: [MaybeUninit<U>; N] = unsafe { MaybeUninit::uninit().assume_init() };
            for i in 0..N { result[i].write(f( $( [<$part:lower>][i] ),+)); }
            unsafe { MaybeUninit::assume_init(result.into()) }
        }
    )*}};
}

decl_array_zips! {
    2 => (A, B,),
    3 => (A, B, C,),
    4 => (A, B, C, D,),
    5 => (A, B, C, D, E,),
    6 => (A, B, C, D, E, G,), // skip F since the function is bound to F
}

#[inline(always)]
fn array_unzip2<T: Copy, A: Copy, B: Copy, F, const N: usize>(x: [T; N], mut f: F) -> ([A; N], [B; N])
where
    F: FnMut(T) -> (A, B),
{
    let mut a: [MaybeUninit<A>; N] = unsafe { MaybeUninit::uninit().assume_init() };
    let mut b: [MaybeUninit<B>; N] = unsafe { MaybeUninit::uninit().assume_init() };

    for i in 0..N {
        let (aa, bb) = f(x[i]);
        a[i].write(aa);
        b[i].write(bb);
    }

    unsafe { (MaybeUninit::assume_init(a.into()), MaybeUninit::assume_init(b.into())) }
}

#[repr(transparent)]
pub struct ArrayRegister<R: CoreRegister, const N: usize>(pub [Storage<R>; N]);

impl<R: CoreRegister, const N: usize> Clone for ArrayRegister<R, N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: CoreRegister, const N: usize> Copy for ArrayRegister<R, N> {}

impl<R: CoreRegister, const N: usize> core::fmt::Debug for ArrayRegister<R, N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let arr = self.0;

        f.debug_tuple("ArrayRegister").field(&arr).finish()
    }
}

impl<R: CoreRegister, const N: usize> ArrayRegister<R, N> {
    #[inline(always)]
    pub const fn idx(lane: usize) -> (usize, usize) {
        (lane / R::Lanes::USIZE, lane % R::Lanes::USIZE)
    }

    #[inline(always)]
    pub const fn into_array(self) -> [Storage<R>; N] {
        self.0
    }

    #[inline(always)]
    pub const fn from_array(arr: [Storage<R>; N]) -> Self {
        Self(arr)
    }
}

#[rustfmt::skip] #[thermite_macros::array_impl]
impl<R: CoreRegister, const N: usize> CoreRegister for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
{
    type NativeIsa = R::NativeIsa;
    type Lanes = Prod<typenum::U<N>, R::Lanes>;
    type Storage = Self;
    type Mask = ArrayRegister<R::Mask, N>;

    const IS_EMULATED: bool = true;
    const ISA: InstructionSet = R::ISA;

    const EMPTY: Storage<Self> = Self([R::EMPTY; N]);
    const HAS_EQUAL_SIZE_MASK: bool = R::HAS_EQUAL_SIZE_MASK;

    fn blendv(mask: Storage<Self::Mask>, on_false: Storage<Self>, on_true: Storage<Self>) -> Storage<Self> {}
    fn zz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {}
    fn nz(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {}

    fn zeroupper_z<Z: ZeroUpper>(_value: Storage<Self>) -> Storage<Self> {
        panic!("ArrayRegister does not support zeroupper operations");
    }

    fn from_mask(mask: Storage<Self::Mask>) -> Storage<Self> {}
}

#[rustfmt::skip] #[thermite_macros::array_impl]
impl<R: BitwiseRegister, const N: usize> BitwiseRegister for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
{
    const HAS_NATIVE_TERNLOG: bool = R::HAS_NATIVE_TERNLOG;

    #[conditional] fn bitxor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn bitand(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn bitor(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn not(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn bitandnot(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn ternlog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn bilog<const IMM: i32>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {}
}

#[rustfmt::skip] #[thermite_macros::array_impl]
impl<R: MaskRegister, const N: usize> MaskRegister for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
{
    const TRUTHY: Storage<Self> = Self([R::TRUTHY; N]);
    const FALSY: Storage<Self> = Self([R::FALSY; N]);

    fn set(mut mask: Storage<Self>, lane: usize, value: bool) -> Storage<Self> {
        let (idx, lane_in_reg) = Self::idx(lane);
        mask.0[idx] = R::set(mask.0[idx], lane_in_reg, value);
        mask
    }

    fn test(mask: Storage<Self>, lane: usize) -> bool {
        let (idx, lane_in_reg) = Self::idx(lane);
        R::test(mask.0[idx], lane_in_reg)
    }

    fn new_mask(value: GenericArray<bool, Self::Lanes>) -> Storage<Self> {
        let ptr = value.as_ptr() as *const GenericArray<bool, R::Lanes>;
        let mut res = [R::FALSY; N];
        for (i, r) in res.iter_mut().enumerate() {
            *r = R::new_mask(unsafe { ptr.add(i).read() });
        }
        Self(res)
    }

    fn all(mut value: Storage<Self>) -> bool {
        // O(log2(N))) reduction of bitwise AND across all registers
        crate::math::algorithms::reduce_in_place(&mut value.0, R::bitand);
        R::all(value.0[0])
    }

    fn any(mut value: Storage<Self>) -> bool {
        crate::math::algorithms::reduce_in_place(&mut value.0, R::bitor);
        R::any(value.0[0])
    }

    fn native_bitmask(value: Storage<Self>) -> Option<u64> {
        if const { Self::Lanes::USIZE <= 64 } {
            let mut bitmask = 0u64;
            for i in 0..N {
                if let Some(reg_bitmask) = R::native_bitmask(value.0[i]) {
                    bitmask |= reg_bitmask << (i * R::Lanes::USIZE);
                } else {
                    return None;
                }
            }
            Some(bitmask)
        } else {
            None
        }
    }

    #[cfg(feature = "bitvec")]
    fn fill_bitmask(value: Storage<Self>, view: &mut bitvec::slice::BitSlice<u32>) {
        for i in 0..N {
            R::fill_bitmask(value.0[i], &mut view[i * R::Lanes::USIZE..(i + 1) * R::Lanes::USIZE]);
        }
    }

    // Hand each sub-register its own window of the bitmask. `R` ignores bits at
    // or above its own lane count, so the shifted word needs no re-masking.
    // Sub-registers starting at or beyond bit 64 cannot be addressed by a `u64`
    // and stay `FALSY` (the documented > 64-lane truncation).
    fn from_native_bitmask(bitmask: u64) -> Storage<Self> {
        let mut result = [R::FALSY; N];

        for i in 0..N {
            let shift = i * R::Lanes::USIZE;

            if shift >= 64 {
                break;
            }

            result[i] = R::from_native_bitmask(bitmask >> shift);
        }

        Self(result)
    }

    #[cfg(feature = "bitvec")]
    fn from_bitmask(bits: &bitvec::slice::BitSlice<u32>) -> Storage<Self> {
        let mut result = [R::FALSY; N];

        for i in 0..N {
            let start = i * R::Lanes::USIZE;

            if start >= bits.len() {
                break;
            }

            let end = (start + R::Lanes::USIZE).min(bits.len());

            result[i] = R::from_bitmask(&bits[start..end]);
        }

        Self(result)
    }

    // Hand the whole sub-register array to `R` in one call rather than going
    // through the combined bitmask. That stays correct when the total lane
    // count exceeds 64 (where `native_bitmask` returns `None`), and it is what
    // lets a backend see all `N` masks at once - `R::count_set` in particular
    // merges them with a narrowing pack instead of extracting a bitmask per
    // register. `R`'s own default still short-circuits the scans.
    fn first_set_one(value: Storage<Self>) -> Option<usize> {
        R::first_set::<N>(value.0)
    }

    fn last_set_one(value: Storage<Self>) -> Option<usize> {
        R::last_set::<N>(value.0)
    }
    fn count_set_one(value: Storage<Self>) -> usize {
        R::count_set::<N>(value.0)
    }
}

impl<R: InterleaveRegister, const N: usize> InterleaveRegister for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
{
    #[inline(always)]
    fn interleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        let mut lo = [R::EMPTY; N];
        let mut hi = [R::EMPTY; N];

        for i in 0..N {
            // Generate the two sequential sub-registers for this chunk
            let (r_lo, r_hi) = R::interleave(a.0[i], b.0[i]);

            let idx1 = 2 * i;
            let idx2 = 2 * i + 1;

            // LLVM will unroll this loop and statically eliminate these branches
            if idx1 < N {
                lo[idx1] = r_lo;
            } else {
                hi[idx1 - N] = r_lo;
            }

            if idx2 < N {
                lo[idx2] = r_hi;
            } else {
                hi[idx2 - N] = r_hi;
            }
        }

        (Self(lo), Self(hi))
    }

    #[inline(always)]
    fn deinterleave(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        let mut out_a = [R::EMPTY; N];
        let mut out_b = [R::EMPTY; N];

        for i in 0..N {
            let idx1 = 2 * i;
            let idx2 = 2 * i + 1;

            // Treat `a` and `b` as a contiguous 2N slice
            let chunk1 = if idx1 < N { a.0[idx1] } else { b.0[idx1 - N] };
            let chunk2 = if idx2 < N { a.0[idx2] } else { b.0[idx2 - N] };

            let (de_a, de_b) = R::deinterleave(chunk1, chunk2);

            out_a[i] = de_a;
            out_b[i] = de_b;
        }

        (Self(out_a), Self(out_b))
    }
}

impl<R: Register, const N: usize> NewRegister<R::Element, Prod<typenum::U<N>, R::Lanes>, Self> for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
{
    type New<C: NewConst<R::Element, Prod<typenum::U<N>, R::Lanes>>> = ArrayNewConst<R, C, N>;
}

#[doc(hidden)]
pub struct ArrayNewConst<R, C, const N: usize>(PhantomData<[(R, C); N]>);

impl<C, R: Register, const N: usize> crate::vector::VectorValue<C, ArrayRegister<R, N>> for ArrayNewConst<R, C, N>
where
    C: NewConst<R::Element, <ArrayRegister<R, N> as CoreRegister>::Lanes>,
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
{
    const VALUE: Storage<ArrayRegister<R, N>> = {
        let arr = C::VALUES;
        let a = arr.as_slice();
        let mut res = [R::EMPTY; N];

        // NOTE: We can't assume much about the layout of Storage<R>,
        // other than it'll contain at R::Lanes of elements. In practice we could
        // just transmute the entire thing, but we should remain somewhat vigilant.
        let mut i = 0;
        while i < N {
            let mut j = 0;

            let ptr = &raw mut res[i] as *mut R::Element;

            while j < <R::Lanes as Unsigned>::USIZE {
                let k = i * <R::Lanes as Unsigned>::USIZE + j;

                unsafe { ptr.add(j).write(a[k]) };

                j += 1;
            }

            i += 1;
        }

        core::mem::forget(arr);

        ArrayRegister(res)
    };
}

// Dispatch a runtime within-chunk offset (`off < R::Lanes <= 16`) to a
// const-generic `R::align`. `OFFSET % L` is a const expr of the generic `OFFSET`,
// which stable rejects in const-generic position - the match turns it into a
// literal. Arms above the chunk's lane count are dead.
macro_rules! chunk_align {
    ($off:expr, $lo:expr, $hi:expr) => {
        match $off {
            0 => R::align::<0>($lo, $hi),
            1 => R::align::<1>($lo, $hi),
            2 => R::align::<2>($lo, $hi),
            3 => R::align::<3>($lo, $hi),
            4 => R::align::<4>($lo, $hi),
            5 => R::align::<5>($lo, $hi),
            6 => R::align::<6>($lo, $hi),
            7 => R::align::<7>($lo, $hi),
            8 => R::align::<8>($lo, $hi),
            9 => R::align::<9>($lo, $hi),
            10 => R::align::<10>($lo, $hi),
            11 => R::align::<11>($lo, $hi),
            12 => R::align::<12>($lo, $hi),
            13 => R::align::<13>($lo, $hi),
            14 => R::align::<14>($lo, $hi),
            _ => R::align::<15>($lo, $hi),
        }
    };
}

#[rustfmt::skip] #[thermite_macros::array_impl]
impl<R: Register, const N: usize> Register for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
{
    type Element = R::Element;
    type Signed = ArrayRegister<R::Signed, N>;
    type Unsigned = ArrayRegister<R::Unsigned, N>;

    // The per-chunk overrides below decompose into the inner register's memory
    // ops, so structural hardware shows through unchanged.
    const HAS_STRUCTURAL_MEMOPS: bool = R::HAS_STRUCTURAL_MEMOPS;

    fn into_mask(value: Storage<Self>) -> Storage<Self::Mask> {}
    fn into_mask_unchecked(value: Storage<Self>) -> Storage<Self::Mask> {}
    fn msb_to_mask(value: Storage<Self>) -> Storage<Self::Mask> {}

    fn new(value: GenericArray<Self::Element, Self::Lanes>) -> Storage<Self> {
        let ptr = value.as_ptr() as *const GenericArray<R::Element, R::Lanes>;

        let mut res = [R::EMPTY; N];

        for (i, r) in res.iter_mut().enumerate() {
            *r = unsafe { R::new(ptr.add(i).read()) };
        }

        Self(res)
    }

    fn single(value: Self::Element) -> Storage<Self> {
        let mut res = [R::EMPTY; N];
        res[0] = R::single(value);
        Self(res)
    }

    fn splat(value: Self::Element) -> Storage<Self> {
        Self([R::splat(value); N])
    }

    fn splat_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {}
    fn splat_z(mask: Storage<Self::Mask>, value: Self::Element) -> Storage<Self> {}

    fn broadcast<const I: usize>(mut value: Storage<Self>) -> Storage<Self> {
        let e = R::splat(Self::as_slice(&value)[I]);
        value.0.fill(e);
        value
    }

    fn broadcast_c<const I: usize>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let e = R::splat(Self::as_slice(&value)[I]);
        Self(core::array::from_fn(|j| R::blendv(mask.0[j], value.0[j], e)))
    }

    fn broadcast_m<const I: usize>(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let e = R::splat(Self::as_slice(&value)[I]);
        Self(core::array::from_fn(|j| R::blendv(mask.0[j], src.0[j], e)))
    }

    fn broadcast_z<const I: usize>(mask: Storage<Self::Mask>, value: Storage<Self>) -> Storage<Self> {
        let e = R::splat(Self::as_slice(&value)[I]);
        Self(mask.0.map(|mask_reg| R::blendv(mask_reg, R::EMPTY, e)))
    }

    fn last_element(value: Storage<Self>) -> Self::Element {
        R::last_element(value.0[N - 1])
    }

    fn broadcastv(mut value: Storage<Self>, idx: usize) -> Storage<Self> {
        let e = R::splat(Self::as_slice(&value)[idx]);
        value.0.fill(e);
        value
    }

    fn broadcastv_c(mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        let e = R::splat(Self::as_slice(&value)[idx]);
        Self(core::array::from_fn(|j| R::blendv(mask.0[j], value.0[j], e)))
    }

    fn broadcastv_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        let e = R::splat(Self::as_slice(&value)[idx]);
        Self(core::array::from_fn(|j| R::blendv(mask.0[j], src.0[j], e)))
    }

    fn broadcastv_z(mask: Storage<Self::Mask>, value: Storage<Self>, idx: usize) -> Storage<Self> {
        let e = R::splat(Self::as_slice(&value)[idx]);
        Self(mask.0.map(|mask_reg| R::blendv(mask_reg, R::EMPTY, e)))
    }

    unsafe fn load(ptr: *const Self::Element) -> Storage<Self> {
        let mut res = [R::EMPTY; N];

        for (i, r) in res.iter_mut().enumerate() {
            *r = unsafe { R::load(ptr.add(i * R::Lanes::USIZE)) };
        }

        Self(res)
    }

    unsafe fn load_m(src: Storage<Self>, mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        let mut res = [R::EMPTY; N];

        for (i, r) in res.iter_mut().enumerate() {
            *r = unsafe { R::load_m(src.0[i], mask.0[i], ptr.add(i * R::Lanes::USIZE)) };
        }

        Self(res)
    }

    unsafe fn load_z(mask: Storage<Self::Mask>, ptr: *const Self::Element) -> Storage<Self> {
        let mut res = [R::EMPTY; N];

        for (i, r) in res.iter_mut().enumerate() {
            *r = unsafe { R::load_z(mask.0[i], ptr.add(i * R::Lanes::USIZE)) };
        }

        Self(res)
    }

    unsafe fn load_unaligned(ptr: *const Self::Element) -> Storage<Self> {
        let mut res = [R::EMPTY; N];

        for (i, r) in res.iter_mut().enumerate() {
            *r = unsafe { R::load_unaligned(ptr.add(i * R::Lanes::USIZE)) };
        }

        Self(res)
    }

    unsafe fn load_stream(ptr: *const Self::Element) -> Storage<Self> {
        let mut res = [R::EMPTY; N];

        for (i, r) in res.iter_mut().enumerate() {
            *r = unsafe { R::load_stream(ptr.add(i * R::Lanes::USIZE)) };
        }

        Self(res)
    }

    unsafe fn store(ptr: *mut Self::Element, value: Storage<Self>) {
        for i in 0..N {
            unsafe { R::store(ptr.add(i * R::Lanes::USIZE), value.0[i]) };
        }
    }

    unsafe fn store_unaligned(ptr: *mut Self::Element, value: Storage<Self>) {
        for i in 0..N {
            unsafe { R::store_unaligned(ptr.add(i * R::Lanes::USIZE), value.0[i]) };
        }
    }

    unsafe fn store_stream(ptr: *mut Self::Element, value: Storage<Self>) {
        for i in 0..N {
            unsafe { R::store_stream(ptr.add(i * R::Lanes::USIZE), value.0[i]) };
        }
    }

    /// De-interleaving an array register decomposes into ONE inner
    /// de-interleave per chunk, which is what lets the inner register's native
    /// path do the work (on ARM, an `f32x8` `load_deinterleaved::<3>` becomes
    /// two `LD3`s rather than a cross-register gather over six registers).
    ///
    /// Chunk `k` of output stream `j` holds lanes `k * L .. (k+1) * L`, i.e.
    /// elements `ptr[(k * L + l) * S + j]` = `ptr[k * L * S + (l * S + j)]` -
    /// exactly the inner de-interleave of the sub-span starting at
    /// `k * L * S`.
    unsafe fn load_deinterleaved<const S: usize>(ptr: *const Self::Element) -> [Storage<Self>; S] {
        let l = R::Lanes::USIZE;

        let mut out = [Self::EMPTY; S];

        for k in 0..N {
            let inner = unsafe { R::load_deinterleaved::<S>(ptr.add(k * l * S)) };

            for (j, o) in out.iter_mut().enumerate() {
                o.0[k] = inner[j];
            }
        }

        out
    }

    /// The exact inverse; same per-chunk decomposition (`ST2`/`ST3`/`ST4` on ARM).
    unsafe fn store_interleaved<const S: usize>(ptr: *mut Self::Element, values: [Storage<Self>; S]) {
        let l = R::Lanes::USIZE;

        for k in 0..N {
            let mut inner = [R::EMPTY; S];
            for (j, v) in values.iter().enumerate() {
                inner[j] = v.0[k];
            }

            unsafe { R::store_interleaved::<S>(ptr.add(k * l * S), inner) };
        }
    }

    /// Radix-`S` interleave decomposes into ONE inner `interleave_radix::<S>` per
    /// chunk position - the same per-chunk delegation as
    /// [`interleave`](InterleaveRegister::interleave), lifted to any radix, so the
    /// inner register's native radix-`S` (e.g. a single `_mm_interleave3_ps`) does
    /// the work instead of the default's cross-chunk permute+blend gather over the
    /// full array width.
    ///
    /// The `S` inputs are `S * N` chunks; treated as one flat chunk span (flat
    /// chunk `c` is `inputs[c / N].0[c % N]`), the radix-`S` interleave sends
    /// output chunk `i` to flat chunks `S*i .. S*i + S` - and because a chunk is
    /// `L` lanes, `S*i*L` is always a chunk boundary, so each output chunk is
    /// exactly one inner `interleave_radix::<S>` of chunk `i` of the `S` inputs.
    fn interleave_radix<const S: usize>(inputs: [Storage<Self>; S]) -> [Storage<Self>; S] {
        let mut out = [Self::EMPTY; S];

        let mut i = 0;
        while i < N {
            // Interleave chunk `i` of each of the `S` input streams.
            let mut group = [R::EMPTY; S];
            let mut s = 0;
            while s < S {
                group[s] = inputs[s].0[i];
                s += 1;
            }

            let res = R::interleave_radix::<S>(group);

            // Scatter: inner result `k` is flat output chunk `S*i + k`.
            let mut k = 0;
            while k < S {
                let c = S * i + k;
                out[c / N].0[c % N] = res[k];
                k += 1;
            }

            i += 1;
        }

        out
    }

    /// The exact inverse of [`interleave_radix`](Self::interleave_radix): output
    /// chunk `i` of each stream is one inner `deinterleave_radix::<S>` of the `S`
    /// consecutive flat chunks `S*i .. S*i + S`.
    fn deinterleave_radix<const S: usize>(inputs: [Storage<Self>; S]) -> [Storage<Self>; S] {
        let mut out = [Self::EMPTY; S];

        let mut i = 0;
        while i < N {
            // Gather the `S` consecutive flat chunks for output-chunk position `i`.
            let mut group = [R::EMPTY; S];
            let mut k = 0;
            while k < S {
                let c = S * i + k;
                group[k] = inputs[c / N].0[c % N];
                k += 1;
            }

            let res = R::deinterleave_radix::<S>(group);

            // Inner result `s` is chunk `i` of output stream `s`.
            let mut s = 0;
            while s < S {
                out[s].0[i] = res[s];
                s += 1;
            }

            i += 1;
        }

        out
    }

    /// Group-granularity interleave is `interleave` on the register reinterpreted
    /// as `LANES / GROUP` elements of `GROUP *` the width - so when a group fits in
    /// a chunk (`GROUP` divides `L`) it is exactly the radix-2 chunk-chaining of
    /// [`interleave`](InterleaveRegister::interleave) with the inner op replaced by
    /// `R::interleave_by::<GROUP>`, and the inner register's native pair op (e.g.
    /// `unpcklo_pd`) does the work instead of a lane-wise spill. A group that spans
    /// chunks (`GROUP > L`) falls back to the lane-wise default.
    fn interleave_by<const GROUP: usize>(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        let l = R::Lanes::USIZE;
        if GROUP <= l && l % GROUP == 0 {
            let mut lo = [R::EMPTY; N];
            let mut hi = [R::EMPTY; N];
            let mut i = 0;
            while i < N {
                // Chunk `i` of `a`/`b` interleaves into flat output chunks `2i`, `2i+1`.
                let (r_lo, r_hi) = R::interleave_by::<GROUP>(a.0[i], b.0[i]);
                let (idx1, idx2) = (2 * i, 2 * i + 1);
                if idx1 < N {
                    lo[idx1] = r_lo;
                } else {
                    hi[idx1 - N] = r_lo;
                }
                if idx2 < N {
                    lo[idx2] = r_hi;
                } else {
                    hi[idx2 - N] = r_hi;
                }
                i += 1;
            }
            (Self(lo), Self(hi))
        } else {
            crate::backend::generic::polyfills::interleave_by_default::<Self, GROUP>(a, b)
        }
    }

    /// The exact inverse of [`interleave_by`](Self::interleave_by): the radix-2
    /// chunk de-interleave with the inner op `R::deinterleave_by::<GROUP>`.
    fn deinterleave_by<const GROUP: usize>(a: Storage<Self>, b: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        let l = R::Lanes::USIZE;
        if GROUP <= l && l % GROUP == 0 {
            let mut out_a = [R::EMPTY; N];
            let mut out_b = [R::EMPTY; N];
            let mut i = 0;
            while i < N {
                // Flat input chunks `2i`, `2i+1` de-interleave into chunk `i` of `a`/`b`.
                let (idx1, idx2) = (2 * i, 2 * i + 1);
                let chunk1 = if idx1 < N { a.0[idx1] } else { b.0[idx1 - N] };
                let chunk2 = if idx2 < N { a.0[idx2] } else { b.0[idx2 - N] };
                let (de_a, de_b) = R::deinterleave_by::<GROUP>(chunk1, chunk2);
                out_a[i] = de_a;
                out_b[i] = de_b;
                i += 1;
            }
            (Self(out_a), Self(out_b))
        } else {
            crate::backend::generic::polyfills::deinterleave_by_default::<Self, GROUP>(a, b)
        }
    }

    /// Both axes at once: the [`interleave_radix`](Self::interleave_radix) per-chunk
    /// delegation with the inner op carrying the group width, so the inner register's
    /// `interleave_radix_by` picks the strategy for its own width.
    ///
    /// The decomposition is unchanged from `interleave_radix` because a group that fits
    /// in a chunk (`GROUP` divides `L`) never straddles a chunk boundary: with `m = L /
    /// GROUP` groups per chunk, output group `q' = i*m + qc` draws flat group
    /// `q'*S + j = i*(m*S) + (qc*S + j)`, and `qc*S + j < m*S` is exactly `S` chunks'
    /// worth - so output chunk `i` of every stream is one inner
    /// `interleave_radix_by::<S, GROUP>` of the flat chunks `S*i .. S*i + S`, the same
    /// span `interleave_radix` uses. A group spanning chunks (`GROUP > L`) falls back.
    ///
    /// This is what carries the inner width's `radix_by` work up to the emulated
    /// widths - `f32x16 = ArrayRegister<F32x8V3, 2>` reaches `F32x8V3`'s native square
    /// transposes and its certified ladder plans through here, instead of running the
    /// portable engine at the full array width.
    fn interleave_radix_by<const S: usize, const GROUP: usize>(inputs: [Storage<Self>; S]) -> [Storage<Self>; S] {
        let l = R::Lanes::USIZE;
        if GROUP <= l && l % GROUP == 0 {
            let mut out = [Self::EMPTY; S];

            let mut i = 0;
            while i < N {
                // Interleave chunk `i` of each of the `S` input streams.
                let mut group = [R::EMPTY; S];
                let mut s = 0;
                while s < S {
                    group[s] = inputs[s].0[i];
                    s += 1;
                }

                let res = R::interleave_radix_by::<S, GROUP>(group);

                // Scatter: inner result `k` is flat output chunk `S*i + k`.
                let mut k = 0;
                while k < S {
                    let c = S * i + k;
                    out[c / N].0[c % N] = res[k];
                    k += 1;
                }

                i += 1;
            }

            out
        } else {
            crate::backend::generic::polyfills::interleave_radix_by_default::<Self, S, GROUP>(inputs)
        }
    }

    /// The exact inverse of [`interleave_radix_by`](Self::interleave_radix_by): output
    /// chunk `i` of each stream is one inner `deinterleave_radix_by::<S, GROUP>` of the
    /// `S` consecutive flat chunks `S*i .. S*i + S`.
    fn deinterleave_radix_by<const S: usize, const GROUP: usize>(inputs: [Storage<Self>; S]) -> [Storage<Self>; S] {
        let l = R::Lanes::USIZE;
        if GROUP <= l && l % GROUP == 0 {
            let mut out = [Self::EMPTY; S];

            let mut i = 0;
            while i < N {
                // Gather the `S` consecutive flat chunks for output-chunk position `i`.
                let mut group = [R::EMPTY; S];
                let mut k = 0;
                while k < S {
                    let c = S * i + k;
                    group[k] = inputs[c / N].0[c % N];
                    k += 1;
                }

                let res = R::deinterleave_radix_by::<S, GROUP>(group);

                // Inner result `s` is chunk `i` of output stream `s`.
                let mut s = 0;
                while s < S {
                    out[s].0[i] = res[s];
                    s += 1;
                }

                i += 1;
            }

            out
        } else {
            crate::backend::generic::polyfills::deinterleave_radix_by_default::<Self, S, GROUP>(inputs)
        }
    }

    /// The record ops decompose per chunk exactly like
    /// [`load_deinterleaved`](Self::load_deinterleaved) above, delegating each
    /// chunk to the INNER register's op so the inner width picks its own
    /// strategy (structural loads on ARM, the flat shuffle engine on x86).
    ///
    /// This matters even without structural hardware: running the flat engine at
    /// THIS register's width pairs up chunked stage ops, while per-chunk
    /// delegation folds each half into the tight inner-width network - an AVX2
    /// `Compensated<f32x16>` 3-record load measured 155 instructions flat versus
    /// 105 decomposed, identical to the plain 6-stream `load_deinterleaved` on
    /// the same data.
    ///
    /// The `TAIL`-shaped grouped ops need no override of their own: their default
    /// re-shapes into these, so they inherit the decomposition.
    unsafe fn load_deinterleaved_arrays<const M: usize, const C: usize>(
        ptr: *const Self::Element,
    ) -> [[Storage<Self>; C]; M] {
        let l = R::Lanes::USIZE;

        let mut out = [[Self::EMPTY; C]; M];

        for k in 0..N {
            let inner = unsafe { R::load_deinterleaved_arrays::<M, C>(ptr.add(k * l * M * C)) };

            for (j, o) in out.iter_mut().enumerate() {
                let mut c = 0;
                while c < C {
                    o[c].0[k] = inner[j][c];
                    c += 1;
                }
            }
        }

        out
    }

    /// The exact inverse; same per-chunk delegation to the inner array op.
    unsafe fn store_interleaved_arrays<const M: usize, const C: usize>(
        ptr: *mut Self::Element,
        values: [[Storage<Self>; C]; M],
    ) {
        let l = R::Lanes::USIZE;

        for k in 0..N {
            let mut inner = [[R::EMPTY; C]; M];

            for (j, v) in values.iter().enumerate() {
                let mut c = 0;
                while c < C {
                    inner[j][c] = v[c].0[k];
                    c += 1;
                }
            }

            unsafe { R::store_interleaved_arrays::<M, C>(ptr.add(k * l * M * C), inner) };
        }
    }

    unsafe fn store_masked(ptr: *mut Self::Element, mask: Storage<Self::Mask>, value: Storage<Self>) {
        for (i, r) in value.0.iter().enumerate() {
            unsafe { R::store_masked(ptr.add(i * R::Lanes::USIZE), mask.0[i], *r) };
        }
    }

    unsafe fn lookup(values: &[Self::Element], indices: Storage<Self::Unsigned>) -> Storage<Self> {
        let mut res = [R::EMPTY; N];

        for (i, r) in res.iter_mut().enumerate() {
            *r = unsafe { R::lookup(values, indices.0[i]) };
        }

        Self(res)
    }

    fn reverse(mut value: Storage<Self>) -> Storage<Self> {
        value.0.reverse();
        for r in &mut value.0 {
            *r = R::reverse(*r);
        }

        value
    }

    #[conditional] fn swap_bytes(value: Storage<Self>) -> Storage<Self> {}

    const HAS_PERMUTEV: bool = R::HAS_PERMUTEV;

    /// Left-pack via the shared 8-lane compaction table + one `permutev`
    /// ([`compress_permute8_raw`](crate::backend::generic::polyfills::compress_permute8_raw)),
    /// instead of the scalar stable-partition default. The emulated-wide
    /// registers (e.g. 2x128 `f32x8` on SSE/NEON/wasm) have branchless
    /// `native_bitmask` and `array_permutev`, so the whole compress stays
    /// branch-free: `movemask`s -> table row -> cross-chunk permute.
    ///
    /// A blanket impl cannot add the `Lanes: CompressTable` bound to a single
    /// method, so applicability is an `if const` guard on the raw polyfill.
    fn compress(value: Storage<Self>, mask: Storage<Self::Mask>) -> Storage<Self> {
        if const { Self::Lanes::USIZE <= 8 && Self::HAS_PERMUTEV } {
            // SAFETY: `Lanes <= 8` per the guard above.
            return unsafe { crate::backend::generic::polyfills::compress_permute8_raw::<Self>(value, mask) };
        }

        crate::backend::generic::polyfills::compress_default::<Self>(value, mask)
    }

    fn compress_z(value: Storage<Self>, mask: Storage<Self::Mask>) -> Storage<Self> {
        if const { Self::Lanes::USIZE <= 8 && Self::HAS_PERMUTEV } {
            // Zero the unselected lanes first; they carry into the tail.
            return Self::compress(Self::zz(mask, value), mask);
        }

        crate::backend::generic::polyfills::compress_z_default::<Self>(value, mask)
    }

    fn expand(value: Storage<Self>, mask: Storage<Self::Mask>) -> Storage<Self> {
        if const { Self::Lanes::USIZE <= 8 && Self::HAS_PERMUTEV } {
            // SAFETY: `Lanes <= 8` per the guard above.
            return unsafe { crate::backend::generic::polyfills::expand_permute8_raw::<Self>(value, mask) };
        }

        crate::backend::generic::polyfills::expand_default::<Self>(value, mask)
    }

    fn expand_z(value: Storage<Self>, mask: Storage<Self::Mask>) -> Storage<Self> {
        if const { Self::Lanes::USIZE <= 8 && Self::HAS_PERMUTEV } {
            // Zero the unselected lanes after routing the packed front out.
            return Self::zz(mask, Self::expand(value, mask));
        }

        crate::backend::generic::polyfills::expand_z_default::<Self>(value, mask)
    }

    fn permutev(value: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        if const { !Self::HAS_PERMUTEV } {
            return Self::scalar_permutev(value, idxs);
        }

        // Delegate the cross-chunk routing to the inner register, which can
        // override it with a faster per-register sequence.
        Self(R::array_permutev::<N>(value.0, idxs.as_slice()))
    }

    fn swizzle(a: Storage<Self>, b: Storage<Self>, idxs: GenericArray<u32, Self::Lanes>) -> Storage<Self> {
        if const { !Self::HAS_PERMUTEV } {
            return Self::scalar_swizzle(a, b, idxs);
        }

        Self(R::array_swizzle::<N>(a.0, b.0, idxs.as_slice()))
    }

    fn swizzle_const<I: SwizzleIndices<Self::Lanes>>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        if const { !Self::HAS_PERMUTEV } {
            // Forward the compile-time indices to the scalar fallback
            return Self::scalar_swizzle(a, b, I::INDICES);
        }

        // Same delegation as the runtime path, but with compile-time indices:
        // `array_swizzle` is `#[inline(always)]`, so the constant indices fold
        // the local/chunk split and blend selectors into immediates.
        Self(R::array_swizzle::<N>(a.0, b.0, I::INDICES.as_slice()))
    }

    fn permutev_const<I: SwizzleIndices<Self::Lanes>>(value: Storage<Self>) -> Storage<Self> {
        if const { !Self::HAS_PERMUTEV } {
            // Forward the compile-time indices to the scalar fallback
            return Self::scalar_permutev(value, I::INDICES);
        }

        Self(R::array_permutev::<N>(value.0, I::INDICES.as_slice()))
    }

    // The per-chunk `R::align` below is the whole implementation, so this width is
    // only as native as the register it is built from.
    const HAS_NATIVE_ALIGN: bool = R::HAS_NATIVE_ALIGN;

    // Cross-chunk element align: each output chunk is a window between two
    // adjacent source chunks of the concatenation [a.0 .., b.0 ..], so it reduces
    // to a per-chunk `R::align` (which itself uses the native fast path). For
    // `OFFSET = base*L + off`, output chunk `c` aligns source chunks `c+base` and
    // `c+base+1` by `off`. The second chunk is unused when `off == 0`.
    fn align<const OFFSET: usize>(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        let l = R::Lanes::USIZE;
        let base = OFFSET / l;
        let off = OFFSET % l;

        let src = |q: usize| -> Storage<R> {
            if q < N { a.0[q] } else if q < 2 * N { b.0[q - N] } else { R::EMPTY }
        };

        let mut result = [R::EMPTY; N];
        let mut c = 0;
        while c < N {
            result[c] = chunk_align!(off, src(c + base), src(c + base + 1));
            c += 1;
        }
        Self(result)
    }
}

/// Emulated widths take the portable widening: there is no single native
/// instruction spanning the chunks. Spelled out rather than defaulted on the
/// trait, so a native register can never silently land on this body.
impl<R: Register, const N: usize> WidenIndexRegister for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
    ArrayRegister<R, N>: Register,
{
    #[inline(always)]
    fn widen_indices(
        idxs: &GenericArray<u8, generic_array::typenum::U8>,
    ) -> GenericArray<u32, <Self as CoreRegister>::Lanes> {
        crate::backend::generic::polyfills::widen_row::<<Self as CoreRegister>::Lanes>(idxs)
    }
}

#[rustfmt::skip]
impl<R: PartialOrdRegister, const N: usize> PartialOrdRegister for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
{
    fn eq(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> { ArrayRegister(array_zip2(lhs.0, rhs.0, R::eq)) }
    fn gt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> { ArrayRegister(array_zip2(lhs.0, rhs.0, R::gt)) }
    fn ge(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> { ArrayRegister(array_zip2(lhs.0, rhs.0, R::ge)) }
    fn lt(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> { ArrayRegister(array_zip2(lhs.0, rhs.0, R::lt)) }
    fn le(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> { ArrayRegister(array_zip2(lhs.0, rhs.0, R::le)) }
    fn ne(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self::Mask> { ArrayRegister(array_zip2(lhs.0, rhs.0, R::ne)) }
}

/// One rung of the literal-`N` sort ladder: when `N` equals the literal, cast
/// the chunk array to its literal-size twin, run the network, and return.
///
/// The casts are identity reinterprets guarded by `if const { N == $n }`; the
/// dead arms of other instantiations still monomorphise (which is why this is
/// a cast and not a type equality - same pattern as `deinterleave_radix`'s
/// `N == 3` arm), but never execute.
macro_rules! sort_arm {
    ($n:literal, $f:ident, $value:ident, $order:ident) => {
        if const { N == $n } {
            // SAFETY: `N == $n` per the guard, so `[Storage<R>; N]` and
            // `[Storage<R>; $n]` are the same type.
            unsafe {
                let chunks = *(&$value.0 as *const [Storage<R>; N] as *const [Storage<R>; $n]);
                let sorted = crate::backend::generic::polyfills::sort::$f::<R, $order>(chunks);
                return Self(*(&sorted as *const [Storage<R>; $n] as *const [Storage<R>; N]));
            }
        }
    };
}

#[rustfmt::skip] #[thermite_macros::array_impl]
impl<R: NumericRegister, const N: usize> NumericRegister for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
{
    const ZERO: Storage<Self> = Self([R::ZERO; N]);
    const ONE: Storage<Self> = Self([R::ONE; N]);
    const TWO: Storage<Self> = Self([R::TWO; N]);

    const MIN: Storage<Self> = Self([R::MIN; N]);
    const MAX: Storage<Self> = Self([R::MAX; N]);

    fn is_all_zero(mut value: Storage<Self>) -> bool {
        crate::math::algorithms::reduce_in_place(&mut value.0, R::bitor);
        R::is_all_zero(value.0[0])
    }

    #[conditional] fn add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Self(array_zip2(lhs.0, rhs.0, R::add)) }
    #[conditional] fn sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Self(array_zip2(lhs.0, rhs.0, R::sub)) }
    #[conditional] fn mul(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Self(array_zip2(lhs.0, rhs.0, R::mul)) }
    #[conditional] fn div(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Self(array_zip2(lhs.0, rhs.0, R::div)) }
    #[conditional] fn rem(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Self(array_zip2(lhs.0, rhs.0, R::rem)) }
    #[conditional] fn min(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Self(array_zip2(lhs.0, rhs.0, R::min)) }
    #[conditional] fn max(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> { Self(array_zip2(lhs.0, rhs.0, R::max)) }
    #[conditional] fn square(lhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn scale(value: Storage<Self>, scalar: Self::Element) -> Storage<Self> {}

    // Block scans: per-chunk native scan plus a broadcast carry (the previous
    // chunk's last lane forward, the next chunk's first lane reverse).
    // ArrayRegister has no native align, so the trait default would walk every
    // lane sequentially at the composite width. The forward carry is
    // extract-then-splat: LLVM folds it to one broadcast (bin/lastlane_probe).
    fn prefix_sum(value: Storage<Self>) -> Storage<Self> {
        let mut out = value;
        out.0[0] = R::prefix_sum(out.0[0]);
        for i in 1..N {
            let carry = R::splat(R::last_element(out.0[i - 1]));
            out.0[i] = R::add(R::prefix_sum(out.0[i]), carry);
        }
        out
    }

    fn prefix_min(value: Storage<Self>) -> Storage<Self> {
        let mut out = value;
        out.0[0] = R::prefix_min(out.0[0]);
        for i in 1..N {
            let carry = R::splat(R::last_element(out.0[i - 1]));
            out.0[i] = R::min(R::prefix_min(out.0[i]), carry);
        }
        out
    }

    fn prefix_max(value: Storage<Self>) -> Storage<Self> {
        let mut out = value;
        out.0[0] = R::prefix_max(out.0[0]);
        for i in 1..N {
            let carry = R::splat(R::last_element(out.0[i - 1]));
            out.0[i] = R::max(R::prefix_max(out.0[i]), carry);
        }
        out
    }

    fn reverse_prefix_sum(value: Storage<Self>) -> Storage<Self> {
        let mut out = value;
        out.0[N - 1] = R::reverse_prefix_sum(out.0[N - 1]);
        for i in (0..N - 1).rev() {
            let carry = R::broadcast::<0>(out.0[i + 1]);
            out.0[i] = R::add(R::reverse_prefix_sum(out.0[i]), carry);
        }
        out
    }

    fn reverse_prefix_min(value: Storage<Self>) -> Storage<Self> {
        let mut out = value;
        out.0[N - 1] = R::reverse_prefix_min(out.0[N - 1]);
        for i in (0..N - 1).rev() {
            let carry = R::broadcast::<0>(out.0[i + 1]);
            out.0[i] = R::min(R::reverse_prefix_min(out.0[i]), carry);
        }
        out
    }

    fn reverse_prefix_max(value: Storage<Self>) -> Storage<Self> {
        let mut out = value;
        out.0[N - 1] = R::reverse_prefix_max(out.0[N - 1]);
        for i in (0..N - 1).rev() {
            let carry = R::broadcast::<0>(out.0[i + 1]);
            out.0[i] = R::max(R::reverse_prefix_max(out.0[i]), carry);
        }
        out
    }

    /// Sort all `N * R::LANES` elements into one ascending run (not N sorted
    /// chunks, not sorted columns).
    ///
    /// Bitonic decomposition by comparator distance: cross-chunk comparators
    /// are whole-register `R::min`/`R::max` (columnar, zero shuffles); every
    /// within-chunk stage routes through `R::sort`/`R::bitonic_clean`, so this
    /// body knows no lane count. Power-of-two `N` only, spelled as a literal
    /// ladder - a const-generic merge-tree loop is the documented shape that
    /// defeats the unroller. Other `N` (3 exists) keep the scalar fallback:
    /// bitonic needs a power of two, and padding with `NumericRegister::MAX`
    /// (finite!) is the +inf trap the prefix-scan ladder already hit once.
    fn sort_by<O: crate::sort::SortOrder>(value: Storage<Self>) -> Storage<Self> {
        sort_arm!(2, sort_array_2, value, O);
        sort_arm!(4, sort_array_4, value, O);
        sort_arm!(8, sort_array_8, value, O);
        sort_arm!(16, sort_array_16, value, O);
        crate::backend::generic::polyfills::sort::sort_any::<Self, O>(value)
    }

    /// Sort a **bitonic** `N * R::LANES`-element register: chunk-stride
    /// columnar min/max layers, then `R::bitonic_clean` per chunk. Emitted
    /// alongside [`sort`](Self::sort) under the same ladder (no-drift rule: a
    /// fast `sort` with a defaulted clean would make cross-register merges
    /// silently quadratic). Garbage in, garbage out on the network arms;
    /// the fallback happens to fully sort.
    fn bitonic_clean_by<O: crate::sort::SortOrder>(value: Storage<Self>) -> Storage<Self> {
        sort_arm!(2, bitonic_clean_array_2, value, O);
        sort_arm!(4, bitonic_clean_array_4, value, O);
        sort_arm!(8, bitonic_clean_array_8, value, O);
        sort_arm!(16, bitonic_clean_array_16, value, O);
        crate::backend::generic::polyfills::sort::sort_any::<Self, O>(value)
    }

    fn min_element(mut value: Storage<Self>) -> Self::Element {
        crate::math::algorithms::reduce_in_place(&mut value.0, R::min);
        R::min_element(value.0[0])
    }

    fn max_element(mut value: Storage<Self>) -> Self::Element {
        crate::math::algorithms::reduce_in_place(&mut value.0, R::max);
        R::max_element(value.0[0])
    }

    fn sum_elements(mut value: Storage<Self>) -> Self::Element {
        crate::math::algorithms::reduce_in_place(&mut value.0, R::add);
        R::sum_elements(value.0[0])
    }

    fn prod_elements(mut value: Storage<Self>) -> Self::Element {
        crate::math::algorithms::reduce_in_place(&mut value.0, R::mul);
        R::prod_elements(value.0[0])
    }

    fn pairwise_sum(lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self> {
        // Pairs adjacent inner registers within lo first, then within hi.
        // Works uniformly for 1-lane and multi-lane R.
        Self(core::array::from_fn(|i| {
            if i < const { N / 2 } {
                R::pairwise_sum(lo.0[2 * i], lo.0[2 * i + 1])
            } else {
                let j = i - const { N / 2 };
                R::pairwise_sum(hi.0[2 * j], hi.0[2 * j + 1])
            }
        }))
    }

    fn relaxed_pairwise_sum(lo: Storage<Self>, hi: Storage<Self>) -> Storage<Self> {
        Self(array_zip2(lo.0, hi.0, |lo, hi| R::relaxed_pairwise_sum(lo, hi)))
    }

    fn offset() -> Storage<Self> {
        let mut offset = R::offset();
        for _ in 1..N {
            offset = R::add(offset, R::offset());
        }
        Self([offset; N])
    }

    fn indexed() -> Storage<Self> {
        let mut result = [R::ZERO; N];
        result[0] = R::indexed();

        for i in 1..N {
            result[i] = R::add(result[i - 1], R::offset());
        }

        Self(result)
    }
}

#[rustfmt::skip] #[thermite_macros::array_impl]
impl<R: SignedRegister, const N: usize> SignedRegister for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
{
    const NEG_ONE: Storage<Self> = Self([R::NEG_ONE; N]);
    const MIN_POSITIVE: Storage<Self> = Self([R::MIN_POSITIVE; N]);

    #[conditional] fn neg(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn abs(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn copysign(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}

    fn signum(value: Storage<Self>) -> Storage<Self> {}
    fn is_negative(value: Storage<Self>) -> Storage<Self::Mask> {}
    fn is_positive(value: Storage<Self>) -> Storage<Self::Mask> {}
    fn select_negative(value: Storage<Self>, on_neg: Storage<Self>, on_pos: Storage<Self>) -> Storage<Self> {}
}

#[rustfmt::skip] #[thermite_macros::array_impl]
impl<R: BitshiftRegister, const N: usize> BitshiftRegister for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
{
    const HAS_WIDE_BYTE_SHIFTS: bool = false;
    const HAS_TRUE_SHIFTV: bool = R::HAS_TRUE_SHIFTV;

    #[conditional] fn shli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn shri<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn shr(value: Storage<Self>, shift: u32) -> Storage<Self> {}
    #[conditional] fn shl(value: Storage<Self>, shift: u32) -> Storage<Self> {}
    #[conditional] fn shrv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {}
    #[conditional] fn shlv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {}
    #[conditional] fn rol(value: Storage<Self>, shift: u32) -> Storage<Self> {}
    #[conditional] fn ror(value: Storage<Self>, shift: u32) -> Storage<Self> {}
    #[conditional] fn roli<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn rori<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn rorv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {}
    #[conditional] fn rolv(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {}
    #[conditional] fn reverse_bits(value: Storage<Self>) -> Storage<Self> {}
}

// Dispatch a runtime within-chunk offset (`off < R::Lanes <= 16`) to a
// const-generic `R::align`. The product `OFFSET % L` is a const expr of the
// generic `OFFSET`, which stable rejects in const-generic position - the match
// turns it into a literal. Arms above the chunk's lane count are dead.
#[rustfmt::skip] #[thermite_macros::array_impl]
impl<R: IntegerRegister, const N: usize> IntegerRegister for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
{
    #[conditional] fn mulhi(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn mullo(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn saturating_add(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn saturating_sub(lhs: Storage<Self>, rhs: Storage<Self>) -> Storage<Self> {}

    fn wrapping_product(mut value: Storage<Self>) -> Self::Element {
        crate::math::algorithms::reduce_in_place(&mut value.0, R::mullo);
        R::wrapping_product(value.0[0])
    }

    fn wrapping_sum(mut value: Storage<Self>) -> Self::Element {
        crate::math::algorithms::reduce_in_place(&mut value.0, R::add);
        R::wrapping_sum(value.0[0])
    }

    #[conditional] fn div_branched(value: Storage<Self>, divider: Divider<Self::Element>) -> Storage<Self> {}
    #[conditional] fn div_branchfree(value: Storage<Self>, divider: BranchfreeDivider<Self::Element>) -> Storage<Self> {}

    fn divv_branchfree(value: Storage<Self>, dividers: VectorDivider<Self>) -> Storage<Self> {
        let multipliers = dividers.multipliers.0;
        let shifts = dividers.shifts.0;

        Self(array_zip3(value.0, multipliers.0, shifts.0, |value, multiplier, shift| {
            R::divv_branchfree(value, VectorDivider { multipliers: Vector(multiplier), shifts: Vector(shift) })
        }))
    }

    fn divv_branchfree_c(mask: Storage<Self::Mask>, value: Storage<Self>, dividers: VectorDivider<Self>) -> Storage<Self> {
        let multipliers = dividers.multipliers.0;
        let shifts = dividers.shifts.0;

        Self(array_zip4(mask.0, value.0, multipliers.0, shifts.0, |mask_reg, value_reg, multiplier, shift| {
            R::divv_branchfree_c(mask_reg, value_reg, VectorDivider { multipliers: Vector(multiplier), shifts: Vector(shift) })
        }))
    }

    fn divv_branchfree_m(src: Storage<Self>, mask: Storage<Self::Mask>, value: Storage<Self>, dividers:VectorDivider<Self>) -> Storage<Self> {
        let multipliers = dividers.multipliers.0;
        let shifts = dividers.shifts.0;

        Self(array_zip5(src.0, mask.0, value.0, multipliers.0, shifts.0, |src_reg, mask_reg, value_reg, multiplier, shift| {
            R::divv_branchfree_m(src_reg, mask_reg, value_reg, VectorDivider { multipliers: Vector(multiplier), shifts: Vector(shift) })
        }))
    }

    fn divv_branchfree_z(mask: Storage<Self::Mask>, value: Storage<Self>, dividers: VectorDivider<Self>) -> Storage<Self> {
        let multipliers = dividers.multipliers.0;
        let shifts = dividers.shifts.0;

        Self(array_zip4(mask.0, value.0, multipliers.0, shifts.0, |mask_reg, value_reg, multiplier, shift| {
            R::divv_branchfree_z(mask_reg, value_reg, VectorDivider { multipliers: Vector(multiplier), shifts: Vector(shift) })
        }))
    }

    const HAS_HARDWARE_POPCNT: bool = R::HAS_HARDWARE_POPCNT;

    #[conditional] fn count_ones(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn count_zeros(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn leading_zeros(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn trailing_zeros(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn leading_ones(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn trailing_ones(value: Storage<Self>) -> Storage<Self> {}
}

#[rustfmt::skip] #[thermite_macros::array_impl]
impl<R: UnsignedIntegerRegister, const N: usize> UnsignedIntegerRegister for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
{
    #[conditional] fn ilog2p1(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn next_power_of_two_m1(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn parity(value: Storage<Self>) -> Storage<Self> {}
    fn avg(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {}
    fn abs_diff(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {}
    fn is_power_of_two(value: Storage<Self>) -> Storage<Self::Mask> {}

    // Hand-written: the [Storage<Self>; D] shapes are not splittable by
    // array_impl. Per-chunk delegation reaches the per-backend fast paths
    // (CLMUL/PDEP) the cascade default would bypass.
    fn morton<const D: usize>(values: [Storage<Self>; D]) -> Storage<Self> {
        let mut out = Self::ZERO;
        for i in 0..N {
            let mut chunk = [R::ZERO; D];
            for k in 0..D {
                chunk[k] = values[k].0[i];
            }
            out.0[i] = R::morton(chunk);
        }
        out
    }

    fn reverse_morton<const D: usize>(code: Storage<Self>) -> [Storage<Self>; D] {
        let mut out = [Self::ZERO; D];
        for i in 0..N {
            let coords = R::reverse_morton::<D>(code.0[i]);
            for k in 0..D {
                out[k].0[i] = coords[k];
            }
        }
        out
    }
}

#[rustfmt::skip] #[thermite_macros::array_impl]
impl<R: SignedIntegerRegister, const N: usize> SignedIntegerRegister for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
{
    #[conditional] fn sra(value: Storage<Self>, shift: u32) -> Storage<Self> {}
    #[conditional] fn srai<const IMM8: i32>(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn srav(value: Storage<Self>, shifts: Storage<Self::Unsigned>) -> Storage<Self> {}
    fn avg_floor(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {}
    fn avg_ceil(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {}
    fn mulhrs(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {}
}

#[rustfmt::skip] #[thermite_macros::array_impl]
impl<R: FloatRegister, const N: usize> FloatRegister for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
{
    const NAN: Storage<Self> = Self([R::NAN; N]);
    const INFINITY: Storage<Self> = Self([R::INFINITY; N]);
    const NEG_INFINITY: Storage<Self> = Self([R::NEG_INFINITY; N]);
    const EPSILON: Storage<Self> = Self([R::EPSILON; N]);
    const EXP_MASK: Storage<Self::Bits> = ArrayRegister([R::EXP_MASK; N]);
    const HALF: Storage<Self> = Self([R::HALF; N]);
    const NEG_ZERO: Storage<Self> = Self([R::NEG_ZERO; N]);

    type Bits = ArrayRegister<R::Bits, N>;
    type SignedBits = ArrayRegister<R::SignedBits, N>;
    type ExtendedPrecision = ArrayRegister<R::ExtendedPrecision, N>;

    const HAS_APPROX_RCP: bool = R::HAS_APPROX_RCP;
    const HAS_APPROX_RSQRT: bool = R::HAS_APPROX_RSQRT;
    const HAS_TRUE_FMA: bool = R::HAS_TRUE_FMA;

    const NATIVE_CAP: NativeCapability = R::NATIVE_CAP;

    unsafe fn block_autovectorization(value: &mut Storage<Self>) {
        unsafe { R::block_autovectorization(&mut value.0[0]) };
    }

    unsafe fn native_ldexp(value: Storage<Self>, exp: Storage<Self::SignedBits>) -> Storage<Self> {
        Self(array_zip2(value.0, exp.0, |v, e| unsafe { R::native_ldexp(v, e) }))
    }

    unsafe fn native_frexp(value: Storage<Self>) -> (Storage<Self>, Storage<Self::SignedBits>) {
        let (v, e) = array_unzip2(value.0, |v| unsafe { R::native_frexp(v) });

        (Self(v), ArrayRegister(e))
    }

    unsafe fn native_sin_cos<P: Policy>(value: Storage<Self>) -> (Storage<Self>, Storage<Self>) {
        let (s, c) = array_unzip2(value.0, |v| unsafe { R::native_sin_cos::<P>(v) });

        (Self(s), Self(c))
    }

    unsafe fn native_sin<P: Policy>(value: Storage<Self>) -> Storage<Self> {}
    unsafe fn native_cos<P: Policy>(value: Storage<Self>) -> Storage<Self> {}
    unsafe fn native_exp<P: Policy>(value: Storage<Self>) -> Storage<Self> {}
    unsafe fn native_exp2<P: Policy>(value: Storage<Self>) -> Storage<Self> {}
    unsafe fn native_ln<P: Policy>(value: Storage<Self>) -> Storage<Self> {}
    unsafe fn native_log2<P: Policy>(value: Storage<Self>) -> Storage<Self> {}
    unsafe fn native_powf<P: Policy>(base: Storage<Self>, exp: Storage<Self>) -> Storage<Self> {}
    unsafe fn native_tan<P: Policy>(value: Storage<Self>) -> Storage<Self> {}

    fn total_order(value: Storage<Self>) -> Storage<Self::SignedBits> {}
    fn is_nan(value: Storage<Self>) -> Storage<Self::Mask> {}
    fn is_finite(value: Storage<Self>) -> Storage<Self::Mask> {}
    fn is_infinite(value: Storage<Self>) -> Storage<Self::Mask> {}
    fn is_normal(value: Storage<Self>) -> Storage<Self::Mask> {}
    fn is_subnormal(value: Storage<Self>) -> Storage<Self::Mask> {}
    fn is_zero_or_subnormal(value: Storage<Self>) -> Storage<Self::Mask> {}

    #[conditional] fn mul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn mul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn nmul_add(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn nmul_sub(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn mul_adde(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn mul_sube(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn nmul_adde(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn nmul_sube(lhs: Storage<Self>, rhs: Storage<Self>, acc: Storage<Self>) -> Storage<Self> {}

    #[conditional] fn sqrt(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn rsqrt(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn rcp(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn floor(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn ceil(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn round(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn trunc(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn fract(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn signed_zero(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn mul_sign(value: Storage<Self>, sign: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn next_down(value: Storage<Self>) -> Storage<Self> {}
    #[conditional] fn next_up(value: Storage<Self>) -> Storage<Self> {}

    fn addsub(a: Storage<Self>, b: Storage<Self>) -> Storage<Self> {
        if const { <R::Lanes as Unsigned>::USIZE % 2 == 0 } {
            let mut out = a;
            for i in 0..N {
                out.0[i] = R::addsub(out.0[i], b.0[i]);
            }
            out
        } else {
            Self::add(a, Self::bitxor(b, Self::ALT_NEG))
        }
    }

    fn fmaddsub(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        if const { <R::Lanes as Unsigned>::USIZE % 2 == 0 } {
            let mut out = a;
            for i in 0..N {
                out.0[i] = R::fmaddsub(out.0[i], b.0[i], c.0[i]);
            }
            out
        } else {
            Self::mul_adde(a, b, Self::bitxor(c, Self::ALT_NEG))
        }
    }

    fn fmsubadd(a: Storage<Self>, b: Storage<Self>, c: Storage<Self>) -> Storage<Self> {
        if const { <R::Lanes as Unsigned>::USIZE % 2 == 0 } {
            let mut out = a;
            for i in 0..N {
                out.0[i] = R::fmsubadd(out.0[i], b.0[i], c.0[i]);
            }
            out
        } else {
            Self::mul_adde(a, b, Self::bitxor(c, Self::ALT_POS))
        }
    }

    fn mix(a: Storage<Self>, b: Storage<Self>, t: Storage<Self>) -> Storage<Self> {}
}

#[rustfmt::skip] #[thermite_macros::array_impl]
impl<IDX, R: IndexableRegister<IDX>, const N: usize> IndexableRegister<ArrayRegister<IDX, N>> for ArrayRegister<R, N>
where
    IDX: UnsignedIntegerRegister<Lanes = R::Lanes>,
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
{
    unsafe fn gather(ptr: *const Self::Element, indices: Storage<ArrayRegister<IDX, N>>) -> Storage<Self> {
        Self(indices.0.map(|idx| unsafe { R::gather(ptr, idx) }))
    }

    unsafe fn gather_m(
        src: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<ArrayRegister<IDX, N>>,
    ) -> Storage<Self> {
        Self(array_zip3(src.0, mask.0, indices.0, |src_reg, mask_reg, idx| unsafe {
            R::gather_m(src_reg, mask_reg, ptr, idx)
        }))
    }

    unsafe fn gather_z(
        mask: Storage<Self::Mask>,
        ptr: *const Self::Element,
        indices: Storage<ArrayRegister<IDX, N>>,
    ) -> Storage<Self> {
        Self(array_zip2(mask.0, indices.0, |mask_reg, idx| unsafe {
            R::gather_z(mask_reg, ptr, idx)
        }))
    }

    unsafe fn scatter(value: Storage<Self>, ptr: *mut Self::Element, indices: Storage<ArrayRegister<IDX, N>>) {
        for i in 0..N {
            unsafe { R::scatter(value.0[i], ptr, indices.0[i]) };
        }
    }

    unsafe fn scatter_m(
        value: Storage<Self>,
        mask: Storage<Self::Mask>,
        ptr: *mut Self::Element,
        indices: Storage<ArrayRegister<IDX, N>>,
    ) {
        for i in 0..N {
            unsafe { R::scatter_m(value.0[i], mask.0[i], ptr, indices.0[i]) };
        }
    }
}

/// ArrayRegister implements ExtendRegister and ConcatRegister in the most non-strict way possible, not requiring
/// either any specific relationship between N and M, or even that the total number of lanes is the same.
/// It just copies as many lanes as it can, and fills the rest with empty registers.
impl<R: CoreRegister, const N: usize, const M: usize> ExtendRegister<ArrayRegister<R, N>> for ArrayRegister<R, M>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
    Const<M>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
{
    #[inline(always)]
    fn extend(value: Storage<ArrayRegister<R, N>>) -> Storage<Self> {
        let mut result = [R::EMPTY; M];
        let min = const { if N < M { N } else { M } };

        // copy the low `min` registers; the rest stay `R::EMPTY`. The
        // destination subslice must match the source length - `copy_from_slice`
        // requires equal lengths (writing `result.copy_from_slice` panicked when
        // M > N).
        result[..min].copy_from_slice(&value.0[..min]);

        Self(result)
    }

    #[inline(always)]
    fn narrow(value: Storage<Self>) -> Storage<ArrayRegister<R, N>> {
        let mut result = [R::EMPTY; N];
        let min = const { if N < M { N } else { M } };

        result[..min].copy_from_slice(&value.0[..min]);

        ArrayRegister(result)
    }
}

impl<R: CoreRegister, const N: usize> ExtendRegister<R> for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
{
    #[inline(always)]
    fn extend(value: Storage<R>) -> Storage<Self> {
        let mut result = [R::EMPTY; N];
        result[0] = value;
        Self(result)
    }

    #[inline(always)]
    fn narrow(value: Storage<Self>) -> Storage<R> {
        value.0[0]
    }
}

/// ArrayRegister implements ExtendRegister and ConcatRegister in the most non-strict way possible, not requiring
/// either any specific relationship between N and M, or even that the total number of lanes is the same.
/// It just copies as many lanes as it can, and fills the rest with empty registers.
impl<R: CoreRegister, const N: usize, const M: usize> ConcatRegister<ArrayRegister<R, N>> for ArrayRegister<R, M>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
    Const<M>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
{
    #[inline(always)]
    fn concat(lo: Storage<ArrayRegister<R, N>>, hi: Storage<ArrayRegister<R, N>>) -> Storage<Self> {
        let mut result = [R::EMPTY; M];
        let min = const { if N < M { N } else { M } };

        result[..min].copy_from_slice(&lo.0[..min]);
        if min < M {
            result[min..M].copy_from_slice(&hi.0[..(M - min)]);
        }

        Self(result)
    }

    #[inline(always)]
    fn split(value: Storage<Self>) -> (Storage<ArrayRegister<R, N>>, Storage<ArrayRegister<R, N>>) {
        let mut lo = [R::EMPTY; N];
        let mut hi = [R::EMPTY; N];
        let min = const { if N < M { N } else { M } };

        lo.copy_from_slice(&value.0[..min]);
        if min < M {
            hi.copy_from_slice(&value.0[min..M]);
        }

        (ArrayRegister(lo), ArrayRegister(hi))
    }
}

impl<R: CoreRegister> ConcatRegister<R> for ArrayRegister<R, 2>
where
    typenum::U2: Mul<R::Lanes, Output: Lanes>,
{
    #[inline(always)]
    fn concat(lo: Storage<R>, hi: Storage<R>) -> Storage<Self> {
        Self([lo, hi])
    }

    #[inline(always)]
    fn split(value: Storage<Self>) -> (Storage<R>, Storage<R>) {
        (value.0[0], value.0[1])
    }
}

impl<R: FloatRegister, const N: usize> LinAlg4Register for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
    Self: LinAlg3Register + FloatRegister<Lanes = typenum::U4>,
{
    // default implementations are fine
}

impl<R: FloatRegister, const N: usize> LinAlg3Register for ArrayRegister<R, N>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<R::Lanes, Output: Lanes>>,
    Self: FloatRegister<Lanes: ValidLinAlg3Length<Self>, Storage = Self, Element = R::Element>,
{
    #[inline(always)]
    fn min_element3(value: Storage<Self>) -> Self::Element {
        let mut a = Self::extract::<0>(value);
        let b = Self::extract::<1>(value);
        let c = Self::extract::<2>(value);

        if b < a {
            a = b;
        }

        if c < a {
            a = c;
        }

        a
    }

    #[inline(always)]
    fn max_element3(value: Storage<Self>) -> Self::Element {
        let mut a = Self::extract::<0>(value);
        let b = Self::extract::<1>(value);
        let c = Self::extract::<2>(value);

        if b > a {
            a = b;
        }

        if c > a {
            a = c;
        }

        a
    }

    #[inline(always)]
    fn sum_elements3(value: Storage<Self>) -> Self::Element {
        let lo = if <R::Lanes as Unsigned>::USIZE == 2 {
            R::sum_elements(value.0[0])
        } else {
            let a = Self::extract::<0>(value);
            let b = Self::extract::<1>(value);

            a + b
        };

        let c = Self::extract::<2>(value);

        lo + c
    }

    #[inline(always)]
    fn prod_elements3(value: Storage<Self>) -> Self::Element {
        let lo = if <R::Lanes as Unsigned>::USIZE == 2 {
            R::prod_elements(value.0[0])
        } else {
            let a = Self::extract::<0>(value);
            let b = Self::extract::<1>(value);

            a * b
        };

        let c = Self::extract::<2>(value);

        lo * c
    }
}

impl<FROM: CoreRegister, INTO: CastMaskRegister<FROM, Lanes = FROM::Lanes>, const N: usize>
    CastMaskRegister<ArrayRegister<FROM, N>> for ArrayRegister<INTO, N>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<FROM::Lanes, Output: Lanes>>,
{
    #[inline(always)]
    fn mask_from(value: Storage<ArrayRegister<FROM, N>>) -> Storage<Self> {
        Self(value.0.map(INTO::mask_from))
    }
}

#[thermite_macros::inline_always]
impl<FROM: CoreRegister, INTO: CastRegister<FROM, Lanes = FROM::Lanes>, const N: usize>
    CastRegister<ArrayRegister<FROM, N>> for ArrayRegister<INTO, N>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<FROM::Lanes, Output: Lanes>>,
{
    fn cast_from(value: Storage<ArrayRegister<FROM, N>>) -> Storage<Self> {
        Self(value.0.map(INTO::cast_from))
    }

    fn fast_cast_from(value: Storage<ArrayRegister<FROM, N>>) -> Storage<Self> {
        Self(value.0.map(INTO::fast_cast_from))
    }

    fn saturating_cast_from(value: Storage<ArrayRegister<FROM, N>>) -> Storage<Self> {
        Self(value.0.map(INTO::saturating_cast_from))
    }
}

impl<FROM: CoreRegister, INTO: BitCastRegister<FROM, Lanes = FROM::Lanes>, const N: usize>
    BitCastRegister<ArrayRegister<FROM, N>> for ArrayRegister<INTO, N>
where
    Const<N>: ToUInt<Output: ArrayLength + Mul<FROM::Lanes, Output: Lanes>>,
{
    #[inline(always)]
    fn from_bits(value: Storage<ArrayRegister<FROM, N>>) -> Storage<Self> {
        Self(value.0.map(INTO::from_bits))
    }
}

macro_rules! impl_casts {
    ($a:literal $b:literal $c:literal => $trait:ident :: $($method:ident),+) => {paste::paste! {
        const _: () = {
            impl<FROM: CoreRegister, INTO: CoreRegister> $trait<ArrayRegister<FROM, $b>> for ArrayRegister<INTO, $c>
            where
                typenum::[<U $a>]: Mul<FROM::Lanes, Output: Lanes>,
                typenum::[<U $b>]: Mul<FROM::Lanes, Output: Lanes> + Mul<INTO::Lanes, Output: Lanes>,
                typenum::[<U $c>]: Mul<INTO::Lanes, Output: Lanes>,
                ArrayRegister<INTO, $b>: $trait<ArrayRegister<FROM, $a>> + CoreRegister<Storage = ArrayRegister<INTO, $b>>,
            {$(
                #[inline(always)] fn $method(value: Storage<ArrayRegister<FROM, $b>>) -> Storage<Self> {
                    let [lo, hi] = unsafe { generic_array::const_transmute(value.into_array()) };
                    let lo = ArrayRegister::<INTO, $b>::$method(ArrayRegister::<FROM, $a>::from_array(lo));
                    let hi = ArrayRegister::<INTO, $b>::$method(ArrayRegister::<FROM, $a>::from_array(hi));
                    Self::from_array(unsafe { generic_array::const_transmute([lo, hi]) })
                }
            )+}

            impl<FROM: CoreRegister, INTO: CoreRegister> $trait<ArrayRegister<FROM, $c>> for ArrayRegister<INTO, $b>
            where
                typenum::[<U $a>]: Mul<INTO::Lanes, Output: Lanes>,
                typenum::[<U $b>]: Mul<FROM::Lanes, Output: Lanes> + Mul<INTO::Lanes, Output: Lanes>,
                typenum::[<U $c>]: Mul<FROM::Lanes, Output: Lanes>,
                ArrayRegister<INTO, $a>: $trait<ArrayRegister<FROM, $b>>,
                ArrayRegister<FROM, $b>: CoreRegister<Storage = ArrayRegister<FROM, $b>>,
            {$(
                #[inline(always)] fn $method(value: Storage<ArrayRegister<FROM, $c>>) -> Storage<Self> {
                    let [lo, hi] = unsafe { generic_array::const_transmute(value.into_array()) };
                    let lo = ArrayRegister::<INTO, $a>::$method(ArrayRegister::<FROM, $b>::from_array(lo));
                    let hi = ArrayRegister::<INTO, $a>::$method(ArrayRegister::<FROM, $b>::from_array(hi));
                    Self::from_array(unsafe { generic_array::const_transmute([lo, hi]) })
                }
            )+}
        };
    }};

    ($trait:ident :: $($method:ident),+) => {
        // base cases
        const _: () = {
            impl<FROM: CoreRegister, INTO: CoreRegister> $trait<ArrayRegister<FROM, 2>> for ArrayRegister<INTO, 4>
            where
                typenum::U2: Mul<FROM::Lanes, Output: Lanes> + Mul<INTO::Lanes, Output: Lanes>,
                typenum::U4: Mul<INTO::Lanes, Output: Lanes>,
                ArrayRegister<INTO, 2>: $trait<FROM> + CoreRegister<Storage = ArrayRegister<INTO, 2>>,
            {$(
                #[inline(always)] fn $method(value: Storage<ArrayRegister<FROM, 2>>) -> Storage<Self> {
                    let [lo, hi] = unsafe { generic_array::const_transmute(value.into_array()) };
                    let lo = ArrayRegister::<INTO, 2>::$method(lo).into_array();
                    let hi = ArrayRegister::<INTO, 2>::$method(hi).into_array();
                    Self::from_array(unsafe { generic_array::const_transmute([lo, hi]) })
                }
            )+}

            impl<FROM: CoreRegister, INTO: CoreRegister> $trait<ArrayRegister<FROM, 4>> for ArrayRegister<INTO, 2>
            where
                typenum::U2: Mul<INTO::Lanes, Output: Lanes>,
                typenum::U4: Mul<FROM::Lanes, Output: Lanes>,
                INTO: $trait<ArrayRegister<FROM, 2>>,
                ArrayRegister<FROM, 2>: CoreRegister<Storage = ArrayRegister<FROM, 2>>,
            {$(
                #[inline(always)] fn $method(value: Storage<ArrayRegister<FROM, 4>>) -> Storage<Self> {
                    let [lo, hi] = unsafe { generic_array::const_transmute(value.into_array()) };
                    let lo = INTO::$method(ArrayRegister::<FROM, 2>::from_array(lo));
                    let hi = INTO::$method(ArrayRegister::<FROM, 2>::from_array(hi));
                    Self::from_array(unsafe { generic_array::const_transmute([lo, hi]) })
                }
            )+}
        };

        impl_casts!(2 4 8 => $trait::$($method),+);
        impl_casts!(4 8 16 => $trait::$($method),+);
        impl_casts!(8 16 32 => $trait::$($method),+);
    }
}

impl_casts!(CastMaskRegister::mask_from);
impl_casts!(BitCastRegister::from_bits);
impl_casts!(CastRegister::cast_from, fast_cast_from, saturating_cast_from);

macro_rules! impl_indexable {
    ($a:literal $b:literal $c:literal) => {paste::paste! {
        const _: () = {
            impl<IDX: UnsignedIntegerRegister, R: Register> IndexableRegister<ArrayRegister<IDX, $b>> for ArrayRegister<R, $c>
            where
                ArrayRegister<IDX, $a>: UnsignedIntegerRegister<Lanes = <ArrayRegister<R, $b> as CoreRegister>::Lanes>,
                ArrayRegister<R, $b>: IndexableRegister<ArrayRegister<IDX, $a>, Element = R::Element>,
                typenum::[<U $b>]: Mul<R::Lanes, Output: Lanes> + Mul<IDX::Lanes, Output: Lanes>
                    + Mul<<ArrayRegister<R, $b> as CoreRegister>::Lanes, Output: Lanes>,
                typenum::[<U $c>]: Mul<R::Lanes, Output: Lanes> + Mul<IDX::Lanes, Output: Lanes>
                    + Mul<<ArrayRegister<R, $b> as CoreRegister>::Lanes, Output: Lanes>
                    + Mul<R::Lanes, Output = <ArrayRegister<IDX, $b> as CoreRegister>::Lanes>,
            {
                unsafe fn gather(ptr: *const Self::Element, indices: Storage<ArrayRegister<IDX, $b>>) -> Storage<Self> {
                    let [lo_idx, hi_idx] = unsafe { generic_array::const_transmute(indices) };
                    let lo_val = unsafe { <ArrayRegister<R, $b> as IndexableRegister<ArrayRegister<IDX, $a>>>::gather(ptr, lo_idx) };
                    let hi_val = unsafe { <ArrayRegister<R, $b> as IndexableRegister<ArrayRegister<IDX, $a>>>::gather(ptr, hi_idx) };
                    ArrayRegister(unsafe { generic_array::const_transmute([lo_val, hi_val]) })
                }
            }

            impl<IDX: UnsignedIntegerRegister, R: Register> IndexableRegister<ArrayRegister<IDX, $c>> for ArrayRegister<R, $b>
            where
                ArrayRegister<IDX, $b>: UnsignedIntegerRegister<Lanes = <ArrayRegister<R, $a> as CoreRegister>::Lanes>,
                ArrayRegister<IDX, $c>: UnsignedIntegerRegister<Lanes = <ArrayRegister<R, $b> as CoreRegister>::Lanes>,
                ArrayRegister<R, $a>: IndexableRegister<ArrayRegister<IDX, $b>, Element = R::Element>,
                typenum::[<U $b>]: Mul<R::Lanes, Output: Lanes> + Mul<IDX::Lanes, Output: Lanes>,
                typenum::[<U $c>]: Mul<R::Lanes, Output: Lanes> + Mul<IDX::Lanes, Output: Lanes>,
            {
                unsafe fn gather(ptr: *const Self::Element, indices: Storage<ArrayRegister<IDX, $c>>) -> Storage<Self> {
                    let [lo_idx, hi_idx] = unsafe { generic_array::const_transmute(indices) };
                    let lo_val = unsafe { <ArrayRegister<R, $a> as IndexableRegister<ArrayRegister<IDX, $b>>>::gather(ptr, lo_idx) };
                    let hi_val = unsafe { <ArrayRegister<R, $a> as IndexableRegister<ArrayRegister<IDX, $b>>>::gather(ptr, hi_idx) };
                    ArrayRegister(unsafe { generic_array::const_transmute([lo_val, hi_val]) })
                }
            }
        };
    }};

    () => {
        // base case
        impl<IDX, R: Register> IndexableRegister<ArrayRegister<IDX, 2>> for ArrayRegister<R, 4>
        where
            IDX: UnsignedIntegerRegister<Lanes = <ArrayRegister<R, 2> as CoreRegister>::Lanes>,
            ArrayRegister<R, 2>: IndexableRegister<IDX, Element = R::Element>,
            typenum::U2: Mul<R::Lanes, Output: Lanes> + Mul<IDX::Lanes, Output: Lanes>
                + Mul<<ArrayRegister<R, 2> as CoreRegister>::Lanes, Output: Lanes>,
            typenum::U4: Mul<R::Lanes, Output: Lanes> + Mul<IDX::Lanes, Output: Lanes>
                + Mul<<ArrayRegister<R, 2> as CoreRegister>::Lanes, Output: Lanes>
                + Mul<R::Lanes, Output = <ArrayRegister<IDX, 2> as CoreRegister>::Lanes>,
        {
            unsafe fn gather(ptr: *const Self::Element, indices: Storage<ArrayRegister<IDX, 2>>) -> Storage<Self> {
                let [lo_idx, hi_idx] = unsafe { generic_array::const_transmute(indices) };
                let lo_val = unsafe { <ArrayRegister<R, 2> as IndexableRegister<IDX>>::gather(ptr, lo_idx) };
                let hi_val = unsafe { <ArrayRegister<R, 2> as IndexableRegister<IDX>>::gather(ptr, hi_idx) };
                ArrayRegister(unsafe { generic_array::const_transmute([lo_val, hi_val]) })
            }
        }

        impl<IDX: UnsignedIntegerRegister, R: Register> IndexableRegister<ArrayRegister<IDX, 4>> for ArrayRegister<R, 2>
        where
            ArrayRegister<IDX, 2>: UnsignedIntegerRegister<Lanes = R::Lanes>,
            ArrayRegister<IDX, 4>: UnsignedIntegerRegister<Lanes = <ArrayRegister<R, 2> as CoreRegister>::Lanes>,
            R: IndexableRegister<ArrayRegister<IDX, 2>>,
            typenum::U2: Mul<R::Lanes, Output: Lanes> + Mul<IDX::Lanes, Output: Lanes>,
            typenum::U4: Mul<R::Lanes, Output: Lanes> + Mul<IDX::Lanes, Output: Lanes>,
        {
            unsafe fn gather(ptr: *const Self::Element, indices: Storage<ArrayRegister<IDX, 4>>) -> Storage<Self> {
                let [lo_idx, hi_idx] = unsafe { generic_array::const_transmute(indices) };
                let lo_val = unsafe { <R as IndexableRegister<ArrayRegister<IDX, 2>>>::gather(ptr, lo_idx) };
                let hi_val = unsafe { <R as IndexableRegister<ArrayRegister<IDX, 2>>>::gather(ptr, hi_idx) };
                ArrayRegister(unsafe { generic_array::const_transmute([lo_val, hi_val]) })
            }
        }

        impl_indexable!(2 4 8);
        impl_indexable!(4 8 16);
        impl_indexable!(8 16 32);
        // impl_indexable!(16 32 64);
    }
}

impl_indexable!();
