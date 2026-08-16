macro_rules! impl_consts {
    (@ $ty:ty { $($name:ident = $value:expr),* $(,)? }) => {
        impl FloatConsts for $ty {
            $(const $name: Self = $value;)*
        }
    };

    ($($name:ident),*) => {
        impl<R: FloatRegister<Element: FloatConsts>> FloatConsts for Vector<R> {
            $(const $name: Self = const {
                struct FC<R: FloatRegister>(core::marker::PhantomData<R>);
                impl<R: FloatRegister> SplatConst<R::Element> for FC<R> { const VALUE: R::Element = <R::Element as FloatConsts>::$name; }
                <<Vector<R> as SplatVector<R::Element>>::Splat<FC<R>> as VectorValue<FC<R>, Vector<R>>>::VALUE
            };)*
        }
    };
}
