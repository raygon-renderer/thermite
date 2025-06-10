use crate::backend::x86::sse42 as arch;

pub mod polyfills;
//pub mod registers;

use crate::{register::DoublePump, vector::Vector};

//decl_vectors! {
//    f32x4 = Vector<registers::f32x4::F32x4SSE41>,
//    f32x8 = DoublePump<f32x4>,
//    f32x16 = DoublePump<f32x8>,
//}
