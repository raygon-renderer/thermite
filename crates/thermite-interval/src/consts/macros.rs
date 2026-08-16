//! Hand-written machinery behind the generated wiring in `mod.rs`.
//!
//! One invocation through `thermite::for_each_float_const!` declares the
//! `BoundedFloatConsts` trait, implements it for `f32`/`f64` off the generated
//! `consts_table`, and lifts it to `Vector<R>` through one splat-carrier pair
//! per constant name (the same shape as thermite-compensated's consts).

macro_rules! bounded_consts {
    ($($name:ident),* $(,)?) => {
        /// Enclosure pairs `(lower, upper)` bracketing the true mathematical value
        /// of each [`FloatConsts`](thermite::math::FloatConsts) constant.
        ///
        /// See the module docs. The pairs themselves are generated.
        pub trait BoundedFloatConsts<T = Self> {
            $(const $name: (T, T);)*
        }

        impl BoundedFloatConsts for f64 {
            $(const $name: (f64, f64) = crate::consts_table::f64_table::$name;)*
        }

        impl BoundedFloatConsts for f32 {
            $(const $name: (f32, f32) = crate::consts_table::f32_table::$name;)*
        }

        paste::paste! {$(
            #[allow(non_camel_case_types)]
            #[doc(hidden)]
            pub struct [<$name _Lower>]<E>(PhantomData<E>);

            impl<E: BoundedFloatConsts + Copy> SplatConst<E> for [<$name _Lower>]<E> {
                const VALUE: E = <E as BoundedFloatConsts>::$name.0;
            }

            #[allow(non_camel_case_types)]
            #[doc(hidden)]
            pub struct [<$name _Upper>]<E>(PhantomData<E>);

            impl<E: BoundedFloatConsts + Copy> SplatConst<E> for [<$name _Upper>]<E> {
                const VALUE: E = <E as BoundedFloatConsts>::$name.1;
            }
        )*}

        impl<R: FloatRegister> BoundedFloatConsts<Self> for Vector<R>
        where
            R::Element: BoundedFloatConsts<R::Element>,
        {
            $(const $name: (Self, Self) = paste::paste! {(
                const_splat::<Self, [<$name _Lower>]<R::Element>>(),
                const_splat::<Self, [<$name _Upper>]<R::Element>>(),
            )};)*
        }
    };
}
