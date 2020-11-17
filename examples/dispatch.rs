use thermite::*;

/// Test doc
#[inline(always)]
#[dispatch(S)]
pub fn test<'a, S: Simd>(x: &'a S::Vf32) -> S::Vf32
where
    S: Simd,
{
    x.tgamma()
}

//struct A<S: Simd>(S::Vf32);

//impl<S: Simd> A<S> {
//    #[dispatch(S)]
//    fn test(self) -> S::Vf32 {
//        self.0.sin()
//    }
//}

fn main() {}
