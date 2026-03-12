macro_rules! impl_bit_casts {
    ($($from:ty as $to:ty => $conv:ident),* $(,)?) => {
        const _: () = {$(
            impl $crate::register::BitCastRegister<$from> for $to {
                #[inline(always)]
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
            impl $crate::register::CastRegister<$from> for $to {
                #[inline(always)]
                fn cast_from(value: Storage<$from>) -> Storage<Self> {
                    unsafe { arch::$conv(value) }
                }

                $(
                    #[inline(always)]
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
            impl $crate::register::CastMaskRegister<$from> for $to {
                #[inline(always)]
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

            impl $crate::register::ConcatRegister<bool> for $r {
                #[inline(always)]
                fn concat(lo: Storage<bool>, hi: Storage<bool>) -> Storage<Self> {
                    <Self as $crate::register::ConcatRegister<$e>>::concat(<$e>::from_bool(lo), <$e>::from_bool(hi))
                }

                #[inline(always)]
                fn split(value: Storage<Self>) -> (Storage<bool>, Storage<bool>) {
                    let (lo, hi) = <Self as $crate::register::ConcatRegister<$e>>::split(value);
                    (lo.to_bool(), hi.to_bool())
                }
            }

            impl $crate::register::ExtendRegister<bool> for $r {
                #[inline(always)]
                fn extend(value: Storage<bool>) -> Storage<Self> {
                    <Self as $crate::register::ExtendRegister<$e>>::extend(<$e>::from_bool(value))
                }

                #[inline(always)]
                fn narrow(value: Storage<Self>) -> Storage<bool> {
                    <Self as $crate::register::ExtendRegister<$e>>::narrow(value).to_bool()
                }
            }
        };
    };
}
