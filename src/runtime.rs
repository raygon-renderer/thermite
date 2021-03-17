#[macro_export]
macro_rules! dispatch_dyn {
    ($code:block) => {
        dispatch_dyn!(S, $code)
    };
    ($t:ident, $code:block) => {{
        use $crate::{backends, Simd, SimdInstructionSet};

        match SimdInstructionSet::runtime_detect() {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SimdInstructionSet::AVX2 => {
                type $t = backends::avx2::AVX2;
                $code
            }
            _ => unsafe { $crate::unreachable_unchecked() },
        }
    }};
}
