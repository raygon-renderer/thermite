use super::{Element, GenericVector, Vector};

#[inline(never)]
pub const fn splat<V: GenericVector, E: SplatConst<V::Element>>() -> V {
    <<V as SplatVector<V::Element>>::Splat<E> as SplatVectorValue<E, V>>::VALUE
}

/// Simple associated constant splat trait.
///
/// Used with `GenericVector::splat_const` to splat compile-time constant values into vectors.
///
/// This is effectively a workaround for the lack of `const generics` for generic types.
pub trait SplatConst<E> {
    const VALUE: E;
}

pub trait SplatVector<E>: Sized {
    type Splat<T: SplatConst<E>>: SplatVectorValue<T, Self>;
}

pub trait SplatVectorValue<C, V: Sized>: Sized {
    const VALUE: V;
}
