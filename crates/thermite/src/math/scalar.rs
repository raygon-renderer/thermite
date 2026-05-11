pub trait Unwrap {
    type Unwrapped;

    fn wrap(value: Self::Unwrapped) -> Self;
    fn unwrap(self) -> Self::Unwrapped;
}

impl<R> Unwrap for crate::Vector<R>
where
    R: crate::register::Register<Storage = R>,
{
    type Unwrapped = R;

    #[inline(always)]
    fn wrap(value: Self::Unwrapped) -> Self {
        crate::Vector(value)
    }

    #[inline(always)]
    fn unwrap(self) -> Self::Unwrapped {
        self.0
    }
}

impl<T> Unwrap for &[T] {
    type Unwrapped = Self;

    #[inline(always)]
    fn wrap(value: Self::Unwrapped) -> Self {
        value
    }

    #[inline(always)]
    fn unwrap(self) -> Self::Unwrapped {
        self
    }
}

impl<T, const N: usize> Unwrap for &[T; N] {
    type Unwrapped = Self;

    #[inline(always)]
    fn wrap(value: Self::Unwrapped) -> Self {
        value
    }

    #[inline(always)]
    fn unwrap(self) -> Self::Unwrapped {
        self
    }
}

impl<T, const N: usize> Unwrap for [T; N]
where
    T: Unwrap,
{
    type Unwrapped = [<T as Unwrap>::Unwrapped; N];

    #[inline(always)]
    fn wrap(value: Self::Unwrapped) -> Self {
        value.map(Unwrap::wrap)
    }

    #[inline(always)]
    fn unwrap(self) -> Self::Unwrapped {
        self.map(Unwrap::unwrap)
    }
}

impl<T: Unwrap> Unwrap for Option<T> {
    type Unwrapped = Option<T::Unwrapped>;

    #[inline(always)]
    fn wrap(value: Self::Unwrapped) -> Self {
        value.map(Unwrap::wrap)
    }

    #[inline(always)]
    fn unwrap(self) -> Self::Unwrapped {
        self.map(Unwrap::unwrap)
    }
}

macro_rules! impl_identity_unwrap {
    ($($num:ty),*) => {
        $(
            impl Unwrap for $num {
                type Unwrapped = Self;

                #[inline(always)]
                fn wrap(value: Self::Unwrapped) -> Self {
                    value
                }

                #[inline(always)]
                fn unwrap(self) -> Self::Unwrapped {
                    self
                }
            }
        )*
    };
}

macro_rules! impl_tuple_unwrap {
    ($( ($($V:ident,)*) ),* $(,)?) => {$(
        #[allow(non_snake_case)]
        impl<$($V),*> Unwrap for ($($V,)*)
        where
            $($V: Unwrap,)*
        {
            type Unwrapped = ($($V::Unwrapped,)*);

            #[inline(always)]
            fn wrap(value: Self::Unwrapped) -> Self {
                let ($($V,)*) = value;
                ($($V::wrap($V),)*)
            }

            #[inline(always)]
            fn unwrap(self) -> Self::Unwrapped {
                let ($($V,)*) = self;
                ($($V.unwrap(),)*)
            }
        }
    )*};
}

impl_identity_unwrap!((), f32, f64, i32, i64, u32, u64);
impl_tuple_unwrap! {
    (A,),
    (A, B,),
    (A, B, C,),
    (A, B, C, D,),
    (A, B, C, D, E,),
}
