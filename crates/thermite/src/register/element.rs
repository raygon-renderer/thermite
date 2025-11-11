/// Common trait for types that can be used as elements in SIMD registers.
pub trait Element: Sized + Copy + Default + PartialEq + PartialOrd + core::fmt::Debug + 'static {
    /// Unsigned integer type to be used with operations that require unsigned counts, such as shifts.
    type UCOUNT: Element;
    /// Signed integer type to be used with operations that require signed counts, such as shifts.
    type SCOUNT: Element;

    /// When used as a mask, represents "true"
    const TRUTHY: Self;
    /// When used as a mask, represents "false"
    const FALSY: Self;

    /// Convert the element, as a mask, to a boolean value.
    fn to_bool(self) -> bool;

    /// Create the element, as a mask, from a boolean value.
    #[inline(always)]
    fn from_bool(value: bool) -> Self {
        if value { Self::TRUTHY } else { Self::FALSY }
    }
}

macro_rules! impl_element {
    ($(($t:ty, $u:ty, $s:ty)),+) => {$(
        impl Element for $t {
            type UCOUNT = $u;
            type SCOUNT = $s;

            const TRUTHY: Self = !0;
            const FALSY: Self = 0;

            #[inline(always)]
            fn to_bool(self) -> bool { self != 0 }
        }
    )+};
}

impl_element! {
    (u8, u8, i8),
    (u16, u16, i16),
    (u32, u32, i32),
    (u64, u64, i64),
    (i8, u8, i8),
    (i16, u16, i16),
    (i32, u32, i32),
    (i64, u64, i64)
    //(f32, u32, i32),
    //(f64, u64, i64)
}

impl Element for f32 {
    type UCOUNT = u32;
    type SCOUNT = i32;

    const TRUTHY: Self = f32::from_bits(!0);
    const FALSY: Self = f32::from_bits(0);

    #[inline(always)]
    fn to_bool(self) -> bool {
        self.to_bits() != 0
    }
}

impl Element for f64 {
    type UCOUNT = u64;
    type SCOUNT = i64;

    const TRUTHY: Self = f64::from_bits(!0);
    const FALSY: Self = f64::from_bits(0);

    #[inline(always)]
    fn to_bool(self) -> bool {
        self.to_bits() != 0
    }
}
