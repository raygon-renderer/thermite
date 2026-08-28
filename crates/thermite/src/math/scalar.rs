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

/// Out-parameter arrays of vectors: the scalar layer's `&mut [E; N]` reinterprets as
/// the vector layer's `&mut [Vector<E>; N]` in place.
impl<'a, R, const N: usize> Unwrap for &'a mut [crate::Vector<R>; N]
where
    R: crate::register::Register<Storage = R>,
{
    type Unwrapped = &'a mut [R; N];

    #[inline(always)]
    fn wrap(value: Self::Unwrapped) -> Self {
        // SAFETY: `Vector<R>` is `#[repr(transparent)]` over `Storage<R>`, and
        // `R: Register<Storage = R>` pins `Storage<R> = R`, so `[Vector<R>; N]`
        // and `[R; N]` have identical layout.
        unsafe { &mut *(value as *mut [R; N] as *mut [crate::Vector<R>; N]) }
    }

    #[inline(always)]
    fn unwrap(self) -> Self::Unwrapped {
        // SAFETY: as in `wrap`.
        unsafe { &mut *(self as *mut [crate::Vector<R>; N] as *mut [R; N]) }
    }
}

/// Runtime-length slices of vectors: the scalar layer's `&[E]` reinterprets as the vector
/// layer's `&[Vector<E>]` in place, exactly as the out-parameter array impl above does.
///
/// This is why the identity impl below is bounded on [`Element`](crate::element::Element)
/// rather than blanket over `T`: a blanket `&[T]` would overlap this one, and the two
/// cases are genuinely different. `&[Self::Element]` (a coefficient list, `poly`) is the
/// same type at both layers and passes through; `&[Self]` (a value list, `hypot_s`) is
/// `&[Vector<E>]` at the vector layer and `&[E]` at the scalar one.
impl<'a, R> Unwrap for &'a [crate::Vector<R>]
where
    R: crate::register::Register<Storage = R>,
{
    type Unwrapped = &'a [R];

    #[inline(always)]
    fn wrap(value: Self::Unwrapped) -> Self {
        // SAFETY: `Vector<R>` is `#[repr(transparent)]` over `Storage<R>`, and
        // `R: Register<Storage = R>` pins `Storage<R> = R`, so `[Vector<R>]` and `[R]`
        // have identical layout.
        unsafe { &*(value as *const [R] as *const [crate::Vector<R>]) }
    }

    #[inline(always)]
    fn unwrap(self) -> Self::Unwrapped {
        // SAFETY: as in `wrap`.
        unsafe { &*(self as *const [crate::Vector<R>] as *const [R]) }
    }
}

impl<T: crate::element::Element> Unwrap for &[T] {
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
    (A, B, C, D, E, F,),
    (A, B, C, D, E, F, G,),
    (A, B, C, D, E, F, G, H,),
}
