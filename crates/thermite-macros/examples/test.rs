pub trait CoreRegister: Sized {
    type Mask: CoreRegister;
    type Storage;

    const EMPTY: Storage<Self>;

    fn blendv(mask: Storage<Self::Mask>, a: Storage<Self>, b: Storage<Self>) -> Storage<Self>;
}

pub type Storage<R> = <R as CoreRegister>::Storage;

#[thermite_macros::register_trait]
pub trait RegisterTest: CoreRegister {
    fn test<const IMM: i32>(value: Storage<Self>) -> Storage<Self>;
}

pub struct Vector<R: RegisterTest>(pub Storage<R>);
pub struct Mask<R: RegisterTest>(pub Storage<R::Mask>);

#[thermite_macros::vector_trait]
pub trait GenericVector {
    type Mask;

    fn test<const IMM: i32>(self) -> Self;
}

#[thermite_macros::vector_impl]
impl<R: RegisterTest> GenericVector for Vector<R> {
    type Mask = Mask<R>;

    fn test<const IMM: i32>(self) -> Self {}
}

fn main() {}
