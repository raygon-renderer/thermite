macro_rules! impl_bit_casts {
    ($($from:ty as $to:ty => $conv:ident),* $(,)?) => {
        const _: () = {$(
            impl $crate::register::BitsRegister<$from> for $to {
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
