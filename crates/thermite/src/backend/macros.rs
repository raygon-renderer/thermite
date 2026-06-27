/// Attach the generic-default `PackedFloatRegister` impls for the two fp8 formats
/// ([`Fp8E4M3`](crate::element::float::spec::Fp8E4M3) /
/// [`Fp8E5M2`](crate::element::float::spec::Fp8E5M2)) on one or more `(u8 register, f32 register)`
/// pairs. No hardware transcodes fp8, so every backend just takes the branchless defaults; this
/// macro is the per-backend wiring (invoked from each `registers/mod.rs`).
macro_rules! impl_packed_fp8 {
    ($($u8:ty => $f32:ty),* $(,)?) => {$(
        impl $crate::register::PackedFloatRegister<$crate::element::float::spec::Fp8E4M3, $f32> for $u8 {}
        impl $crate::register::PackedFloatRegister<$crate::element::float::spec::Fp8E5M2, $f32> for $u8 {}
    )*};
}

macro_rules! impl_bit_casts {
    ($($from:ty as $to:ty => $conv:ident),* $(,)?) => {
        const _: () = {$(
            #[thermite_macros::inline_always]
            impl $crate::register::BitCastRegister<$from> for $to {
                fn from_bits(value: Storage<$from>) -> Storage<Self> {
                    unsafe { arch::$conv(value) }
                }
            }
        )*};
    };
}

macro_rules! impl_type_casts {
    ($($from:ty as $to:ty => $conv:ident $(| $fast_conv:ident)?),* $(,)?) => {
        const _: () = {$(
            #[thermite_macros::inline_always]
            impl $crate::register::CastRegister<$from> for $to {
                fn cast_from(value: Storage<$from>) -> Storage<Self> {
                    unsafe { arch::$conv(value) }
                }

                $(
                    fn fast_cast_from(value: Storage<$from>) -> Storage<Self> {
                        unsafe { arch::$fast_conv(value) }
                    }
                )?
            }
        )*};
    };
}

macro_rules! impl_mask_casts {
    ($($from:ty as $to:ty => $conv:ident),* $(,)?) => {
        const _: () = {$(
            #[thermite_macros::inline_always]
            impl $crate::register::CastMaskRegister<$from> for $to {
                fn mask_from(value: Storage<$from>) -> Storage<Self> {
                    unsafe { arch::$conv(value) }
                }
            }
        )*};
    };
}

macro_rules! impl_concat_bool_register2 {
    ($e:ty, $r:ty) => {
        const _: () = {
            use $crate::element::MaskElement;

            #[thermite_macros::inline_always]
            impl $crate::register::ConcatRegister<bool> for $r {
                fn concat(lo: Storage<bool>, hi: Storage<bool>) -> Storage<Self> {
                    <Self as $crate::register::ConcatRegister<$e>>::concat(<$e>::from_bool(lo), <$e>::from_bool(hi))
                }

                fn split(value: Storage<Self>) -> (Storage<bool>, Storage<bool>) {
                    let (lo, hi) = <Self as $crate::register::ConcatRegister<$e>>::split(value);
                    (lo.to_bool(), hi.to_bool())
                }
            }

            #[thermite_macros::inline_always]
            impl $crate::register::ExtendRegister<bool> for $r {
                fn extend(value: Storage<bool>) -> Storage<Self> {
                    <Self as $crate::register::ExtendRegister<$e>>::extend(<$e>::from_bool(value))
                }

                fn narrow(value: Storage<Self>) -> Storage<bool> {
                    <Self as $crate::register::ExtendRegister<$e>>::narrow(value).to_bool()
                }
            }
        };
    };
}

macro_rules! impl_newregister {
    ($($r:ty),*) => {$(
        impl $crate::register::NewRegister<
            <$r as $crate::register::Register>::Element,
            <$r as $crate::register::CoreRegister>::Lanes,
            <$r as $crate::register::CoreRegister>::Storage
        > for $r {
            type New<T: $crate::vector::splat::NewConst<
                <$r as $crate::register::Register>::Element,
                <$r as $crate::register::CoreRegister>::Lanes>
            > = Self;
        }

        impl<T> $crate::vector::splat::VectorValue<T, Storage<Self>> for $r
        where
            T: $crate::vector::splat::NewConst<
                <$r as $crate::register::Register>::Element,
                <$r as $crate::register::CoreRegister>::Lanes
            >,
        {
            const VALUE: Storage<Self> = const { unsafe { $crate::generic_array::const_transmute(T::VALUES) } };
        }
    )*};
}
