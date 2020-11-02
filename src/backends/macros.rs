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

        impl [<$op_trait Assign>]<Self> for $name<$is> {
            #[inline(always)] fn [<$op _assign>](&mut self, rhs: Self) { *self = $op_trait::$op(*self, rhs); }
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

            pub(crate) unsafe fn zip<F>(a: Self, b: Self, f: F) -> Self
            where F: Fn($ety, $ety) -> $ety {
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

macro_rules! brute_force_convert {
    ($value:expr; $from:ty => $to:ty) => {
        unsafe {
            let mut res = mem::MaybeUninit::uninit();
            for i in 0..Self::NUM_ELEMENTS {
                *(res.as_mut_ptr() as *mut $to).add(i) = (*transmute::<&_, *const $from>($value).add(i)) as $to;
            }
            res.assume_init()
        }
    };
}
