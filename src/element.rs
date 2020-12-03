use crate::*;

/// Umbrella trait for SIMD vector element bounds
pub trait SimdElement: mask::Truthy + CastFromAll + Clone + Debug + Copy + Default + Send + Sync {}

impl<T> SimdElement for T where T: mask::Truthy + CastFromAll + Clone + Debug + Copy + Default + Send + Sync {}

macro_rules! decl_cast_from_all {
    ($($ty:ty),*) => {
        pub trait CastFromAll: $(CastFrom<$ty>+)* {}
        impl<T> CastFromAll for T where T: $(CastFrom<$ty>+)* {}
    }
}

pub trait CastFrom<T>: Sized {
    fn cast_from(value: T) -> Self;
}

macro_rules! impl_cast_from {
    (@INNER $ty:ty as $as:ty) => {
        impl CastFrom<$ty> for $as {
            #[inline(always)]
            fn cast_from(value: $ty) -> $as {
                value as $as
            }
        }
    };
    ($($ty:ty),*) => {
        $(
            impl_cast_from_bool!($ty);
            impl_cast_from!(@INNER $ty as i8);
            impl_cast_from!(@INNER $ty as i16);
            impl_cast_from!(@INNER $ty as i32);
            impl_cast_from!(@INNER $ty as i64);
            impl_cast_from!(@INNER $ty as isize);
            impl_cast_from!(@INNER $ty as u8);
            impl_cast_from!(@INNER $ty as u16);
            impl_cast_from!(@INNER $ty as u32);
            impl_cast_from!(@INNER $ty as u64);
            impl_cast_from!(@INNER $ty as usize);
            impl_cast_from!(@INNER $ty as f32);
            impl_cast_from!(@INNER $ty as f64);
        )*
    };
}

macro_rules! impl_cast_from_bool {
    ($ty:ty) => {
        impl CastFrom<bool> for $ty {
            #[inline(always)]
            fn cast_from(value: bool) -> Self {
                if value {
                    1 as $ty
                } else {
                    0 as $ty
                }
            }
        }

        impl CastFrom<$ty> for bool {
            #[inline(always)]
            fn cast_from(value: $ty) -> bool {
                value != (0 as $ty)
            }
        }
    };
}

decl_cast_from_all!(i8, i16, i32, i64, u8, u16, u32, u64, isize, usize, f32, f64, bool);
impl_cast_from!(i8, i16, i32, i64, u8, u16, u32, u64, isize, usize, f32, f64);

impl CastFrom<bool> for bool {
    #[inline(always)]
    fn cast_from(value: bool) -> bool {
        value
    }
}
