#![allow(unused)]

#[cfg(feature = "nightly")]
macro_rules! likely {
    ($e:expr) => {{
        #[allow(unused_unsafe)]
        unsafe {
            std::intrinsics::likely($e)
        }
    }};
}

#[cfg(feature = "nightly")]
macro_rules! unlikely {
    ($e:expr) => {{
        #[allow(unused_unsafe)]
        unsafe {
            std::intrinsics::unlikely($e)
        }
    }};
}

#[cfg(not(feature = "nightly"))]
macro_rules! likely {
    ($e:expr) => {
        $e
    };
}

#[cfg(not(feature = "nightly"))]
macro_rules! unlikely {
    ($e:expr) => {
        $e
    };
}
