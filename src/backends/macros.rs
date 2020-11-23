macro_rules! impl_ops {
    (@UNARY $name:ident $is:ident => $($op_trait:ident::$op:ident),*) => {paste::paste! {$(
        impl $op_trait for $name<$is> {
            type Output = Self;
            #[inline(always)] fn $op(self) -> Self { unsafe { self. [<_mm_ $op>]() } }
        }
    )*}};

    (@BINARY $name:ident $is:ident => $($op_trait:ident::$op:ident),*) => {paste::paste! {$(
        impl $op_trait<Self> for $name<$is> {
            type Output = Self;
            #[inline(always)] fn $op(self, rhs: Self) -> Self { unsafe { self. [<_mm_ $op>](rhs) } }
        }
        //impl $op_trait<<Self as SimdVectorBase<$is>>::Element> for $name<$is> {
        //    type Output = Self;
        //    #[inline(always)] fn $op(self, rhs: <Self as SimdVectorBase<$is>>::Element) -> Self {
        //        $op_trait::$op(self, Self::splat(rhs))
        //    }
        //}
        //impl $op_trait<$name<$is>> for <$name<$is> as SimdVectorBase<$is>>::Element {
        //    type Output = $name<$is>;
        //    #[inline(always)] fn $op(self, rhs: $name<$is>) -> $name<$is> {
        //        $op_trait::$op($name::<$is>::splat(self), rhs)
        //    }
        //}

        impl [<$op_trait Assign>]<Self> for $name<$is> {
            #[inline(always)] fn [<$op _assign>](&mut self, rhs: Self) { *self = $op_trait::$op(*self, rhs); }
        }
        impl [<$op_trait Assign>]<<Self as SimdVectorBase<$is>>::Element> for $name<$is> {
            #[inline(always)] fn [<$op _assign>](&mut self, rhs: <Self as SimdVectorBase<$is>>::Element) {
                *self = $op_trait::$op(*self, Self::splat(rhs));
            }
        }
    )*}};

    (@SHIFTS $name:ident $is:ident => $($op_trait:ident::$op:ident),*) => {paste::paste! {$(
        impl $op_trait<<$is as Simd>::Vu32> for $name<$is> {
            type Output = Self;
            #[inline(always)] fn $op(self, rhs: <$is as Simd>::Vu32) -> Self { unsafe { self. [<_mm_ $op>](rhs) } }
        }
        impl $op_trait<u32> for $name<$is> {
            type Output = Self;
            #[inline(always)] fn $op(self, rhs: u32) -> Self { unsafe { self.[<_mm_ $op i>](rhs) } }
        }

        impl [<$op_trait Assign>]<<$is as Simd>::Vu32> for $name<$is> {
            #[inline(always)] fn [<$op _assign>](&mut self, rhs: <$is as Simd>::Vu32) { *self = $op_trait::$op(*self, rhs); }
        }
        impl [<$op_trait Assign>]<u32> for $name<$is> {
            #[inline(always)] fn [<$op _assign>](&mut self, rhs: u32) { *self = $op_trait::$op(*self, rhs); }
        }
    )*}};
}

macro_rules! decl_base_common {
    (#[$meta:meta] $name:ident: $ety:ty => $ty:ty) => {
        #[inline]
        #[$meta]
        unsafe fn extract_unchecked(self, index: usize) -> Self::Element {
            *transmute::<&_, *const Self::Element>(&self).add(index)
        }

        #[inline]
        #[$meta]
        unsafe fn replace_unchecked(mut self, index: usize, value: Self::Element) -> Self {
            *transmute::<&mut _, *mut Self::Element>(&mut self).add(index) = value;
            self
        }

        #[inline]
        #[$meta]
        unsafe fn shuffle_unchecked<INDICES: SimdShuffleIndices>(self, b: Self, indices: INDICES) -> Self {
            let mut dst = Self::undefined();
            for i in 0..Self::NUM_ELEMENTS {
                let idx = *INDICES::INDICES.get_unchecked(i);
                dst = dst.replace_unchecked(
                    i,
                    if idx < Self::NUM_ELEMENTS {
                        self.extract_unchecked(idx)
                    } else {
                        b.extract_unchecked(idx - Self::NUM_ELEMENTS)
                    },
                );
            }
            dst
        }
    };
}

macro_rules! decl {
    ($($name:ident: $ety:ty => $ty:ty),*) => {$(
        #[derive(Clone, Copy)]
        #[repr(transparent)]
        pub struct $name<S: Simd> {
            pub(crate) value: $ty,
            _is: PhantomData<S>,
        }

        impl<S: Simd> $name<S> {
            #[inline(always)]
            pub(crate) fn new(value: $ty) -> Self {
                Self { value, _is: PhantomData }
            }
        }

        impl<S: Simd> $name<S> where Self: SimdVectorBase<S, Element = $ety> {
            #[inline(always)]
            pub(crate) unsafe fn map<F>(mut self, f: F) -> Self
            where F: Fn($ety) -> $ety {
                for i in 0..Self::NUM_ELEMENTS {
                    let ptr = transmute::<&mut _, *mut $ety>(&mut self).add(i);
                    *ptr = f(*ptr);
                }
                self
            }

            #[inline(always)]
            pub(crate) unsafe fn zip<F, V>(a: Self, b: V, f: F) -> Self
            where F: Fn($ety, <V as SimdVectorBase<S>>::Element) -> $ety,
                Self: SimdVectorBase<S>,
                  V: SimdVectorBase<S> {
                let mut out = Self::default();
                for i in 0..Self::NUM_ELEMENTS {
                    *transmute::<&mut _, *mut $ety>(&mut out).add(i) =
                        f(a.extract_unchecked(i), b.extract_unchecked(i));
                }
                out
            }

            #[inline(always)]
            pub(crate) unsafe fn reduce<F>(self, mut init: $ety, f: F) -> $ety
            where F: Fn($ety, $ety) -> $ety {
                for i in 0..Self::NUM_ELEMENTS {
                    init = f(init, self.extract_unchecked(i));
                }
                init
            }

            #[inline(always)]
            pub(crate) unsafe fn reduce2<F>(self, f: F) -> $ety
            where F: Fn($ety, $ety) -> $ety {
                let mut accum = self.extract_unchecked(0);
                for i in 1..Self::NUM_ELEMENTS {
                    accum = f(accum, self.extract_unchecked(i));
                }
                accum
            }
        }

        impl<S: Simd> fmt::Debug for $name<S> where Self: SimdVectorBase<S> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let mut t = f.debug_tuple(stringify!($name));
                for i in 0..Self::NUM_ELEMENTS {
                    t.field(unsafe { &*transmute::<&_, *const $ety>(self).add(i) });
                }
                t.finish()
            }
        }
    )*};
}

macro_rules! decl_brute_force_convert {
    (#[$meta:meta] $from:ty => $to:ty) => {
        paste::paste! {
            #[$meta]
            #[inline]
            unsafe fn do_convert(value: [<V $from>]) -> [<V $to>] {
                let mut res = mem::MaybeUninit::uninit();
                for i in 0..[<V $from>]::NUM_ELEMENTS {
                    *(res.as_mut_ptr() as *mut $to).add(i) = (*transmute::<&_, *const $from>(&value).add(i)) as $to;
                }
                res.assume_init()
            }
        }
    };
}
