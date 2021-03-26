/**
Detects processor architecture at runtime and generates a type definition for the current SIMD instruction-set to be passed into the given code-block.

The code block given is duplicated, manually monomorphised, to give the type definition to it.

```ignore
fn my_algorithm<S: Simd>(x: &mut [f32]) {
    assert!(x.len() >= Vf32::<S>::NUM_ELEMENTS);

    Vf32::<S>::load_unaligned(x).sin().store_unaligned(x);
}

let mut values = vec![0.5; 8];

dispatch_dyn!({ my_algorithm::<S>(&mut values) });

// or with a custom generic parameter name:

dispatch_dyn!(ISA, { my_algorithm::<ISA>(&mut values) });
```
*/
#[macro_export]
macro_rules! dispatch_dyn {
    ($code:block) => {
        dispatch_dyn!(S, $code)
    };
    ($s:ident, $code:block) => {{
        use $crate::{backends, Simd, SimdInstructionSet};

        match SimdInstructionSet::runtime_detect() {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            SimdInstructionSet::AVX2 => {
                type $s = backends::avx2::AVX2;
                $code
            }
            _ => unsafe { $crate::unreachable_unchecked() },
        }
    }};
}
