(function() {var implementors = {};
implementors["thermite"] = [{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/ops/bit/trait.BitAndAssign.html\" title=\"trait core::ops::bit::BitAndAssign\">BitAndAssign</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.f32x8.html\" title=\"struct thermite::backends::avx2::f32x8\">f32x8</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.AVX2.html\" title=\"struct thermite::backends::avx2::AVX2\">AVX2</a>&gt;&gt; for <a class=\"struct\" href=\"thermite/backends/avx2/struct.f32x8.html\" title=\"struct thermite::backends::avx2::f32x8\">f32x8</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.AVX2.html\" title=\"struct thermite::backends::avx2::AVX2\">AVX2</a>&gt;","synthetic":false,"types":["thermite::backends::avx2::vf32::f32x8"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/ops/bit/trait.BitAndAssign.html\" title=\"trait core::ops::bit::BitAndAssign\">BitAndAssign</a>&lt;&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.f32x8.html\" title=\"struct thermite::backends::avx2::f32x8\">f32x8</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.AVX2.html\" title=\"struct thermite::backends::avx2::AVX2\">AVX2</a>&gt; as <a class=\"trait\" href=\"thermite/trait.SimdVectorBase.html\" title=\"trait thermite::SimdVectorBase\">SimdVectorBase</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.AVX2.html\" title=\"struct thermite::backends::avx2::AVX2\">AVX2</a>&gt;&gt;::<a class=\"type\" href=\"thermite/trait.SimdVectorBase.html#associatedtype.Element\" title=\"type thermite::SimdVectorBase::Element\">Element</a>&gt; for <a class=\"struct\" href=\"thermite/backends/avx2/struct.f32x8.html\" title=\"struct thermite::backends::avx2::f32x8\">f32x8</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.AVX2.html\" title=\"struct thermite::backends::avx2::AVX2\">AVX2</a>&gt;","synthetic":false,"types":["thermite::backends::avx2::vf32::f32x8"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/ops/bit/trait.BitAndAssign.html\" title=\"trait core::ops::bit::BitAndAssign\">BitAndAssign</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.f64x8.html\" title=\"struct thermite::backends::avx2::f64x8\">f64x8</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.AVX2.html\" title=\"struct thermite::backends::avx2::AVX2\">AVX2</a>&gt;&gt; for <a class=\"struct\" href=\"thermite/backends/avx2/struct.f64x8.html\" title=\"struct thermite::backends::avx2::f64x8\">f64x8</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.AVX2.html\" title=\"struct thermite::backends::avx2::AVX2\">AVX2</a>&gt;","synthetic":false,"types":["thermite::backends::avx2::vf64::f64x8"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/ops/bit/trait.BitAndAssign.html\" title=\"trait core::ops::bit::BitAndAssign\">BitAndAssign</a>&lt;&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.f64x8.html\" title=\"struct thermite::backends::avx2::f64x8\">f64x8</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.AVX2.html\" title=\"struct thermite::backends::avx2::AVX2\">AVX2</a>&gt; as <a class=\"trait\" href=\"thermite/trait.SimdVectorBase.html\" title=\"trait thermite::SimdVectorBase\">SimdVectorBase</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.AVX2.html\" title=\"struct thermite::backends::avx2::AVX2\">AVX2</a>&gt;&gt;::<a class=\"type\" href=\"thermite/trait.SimdVectorBase.html#associatedtype.Element\" title=\"type thermite::SimdVectorBase::Element\">Element</a>&gt; for <a class=\"struct\" href=\"thermite/backends/avx2/struct.f64x8.html\" title=\"struct thermite::backends::avx2::f64x8\">f64x8</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.AVX2.html\" title=\"struct thermite::backends::avx2::AVX2\">AVX2</a>&gt;","synthetic":false,"types":["thermite::backends::avx2::vf64::f64x8"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/ops/bit/trait.BitAndAssign.html\" title=\"trait core::ops::bit::BitAndAssign\">BitAndAssign</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.i32x8.html\" title=\"struct thermite::backends::avx2::i32x8\">i32x8</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.AVX2.html\" title=\"struct thermite::backends::avx2::AVX2\">AVX2</a>&gt;&gt; for <a class=\"struct\" href=\"thermite/backends/avx2/struct.i32x8.html\" title=\"struct thermite::backends::avx2::i32x8\">i32x8</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.AVX2.html\" title=\"struct thermite::backends::avx2::AVX2\">AVX2</a>&gt;","synthetic":false,"types":["thermite::backends::avx2::vi32::i32x8"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/ops/bit/trait.BitAndAssign.html\" title=\"trait core::ops::bit::BitAndAssign\">BitAndAssign</a>&lt;&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.i32x8.html\" title=\"struct thermite::backends::avx2::i32x8\">i32x8</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.AVX2.html\" title=\"struct thermite::backends::avx2::AVX2\">AVX2</a>&gt; as <a class=\"trait\" href=\"thermite/trait.SimdVectorBase.html\" title=\"trait thermite::SimdVectorBase\">SimdVectorBase</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.AVX2.html\" title=\"struct thermite::backends::avx2::AVX2\">AVX2</a>&gt;&gt;::<a class=\"type\" href=\"thermite/trait.SimdVectorBase.html#associatedtype.Element\" title=\"type thermite::SimdVectorBase::Element\">Element</a>&gt; for <a class=\"struct\" href=\"thermite/backends/avx2/struct.i32x8.html\" title=\"struct thermite::backends::avx2::i32x8\">i32x8</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.AVX2.html\" title=\"struct thermite::backends::avx2::AVX2\">AVX2</a>&gt;","synthetic":false,"types":["thermite::backends::avx2::vi32::i32x8"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/ops/bit/trait.BitAndAssign.html\" title=\"trait core::ops::bit::BitAndAssign\">BitAndAssign</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.i64x8.html\" title=\"struct thermite::backends::avx2::i64x8\">i64x8</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.AVX2.html\" title=\"struct thermite::backends::avx2::AVX2\">AVX2</a>&gt;&gt; for <a class=\"struct\" href=\"thermite/backends/avx2/struct.i64x8.html\" title=\"struct thermite::backends::avx2::i64x8\">i64x8</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.AVX2.html\" title=\"struct thermite::backends::avx2::AVX2\">AVX2</a>&gt;","synthetic":false,"types":["thermite::backends::avx2::vi64::i64x8"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/ops/bit/trait.BitAndAssign.html\" title=\"trait core::ops::bit::BitAndAssign\">BitAndAssign</a>&lt;&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.i64x8.html\" title=\"struct thermite::backends::avx2::i64x8\">i64x8</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.AVX2.html\" title=\"struct thermite::backends::avx2::AVX2\">AVX2</a>&gt; as <a class=\"trait\" href=\"thermite/trait.SimdVectorBase.html\" title=\"trait thermite::SimdVectorBase\">SimdVectorBase</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.AVX2.html\" title=\"struct thermite::backends::avx2::AVX2\">AVX2</a>&gt;&gt;::<a class=\"type\" href=\"thermite/trait.SimdVectorBase.html#associatedtype.Element\" title=\"type thermite::SimdVectorBase::Element\">Element</a>&gt; for <a class=\"struct\" href=\"thermite/backends/avx2/struct.i64x8.html\" title=\"struct thermite::backends::avx2::i64x8\">i64x8</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.AVX2.html\" title=\"struct thermite::backends::avx2::AVX2\">AVX2</a>&gt;","synthetic":false,"types":["thermite::backends::avx2::vi64::i64x8"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/ops/bit/trait.BitAndAssign.html\" title=\"trait core::ops::bit::BitAndAssign\">BitAndAssign</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.u32x8.html\" title=\"struct thermite::backends::avx2::u32x8\">u32x8</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.AVX2.html\" title=\"struct thermite::backends::avx2::AVX2\">AVX2</a>&gt;&gt; for <a class=\"struct\" href=\"thermite/backends/avx2/struct.u32x8.html\" title=\"struct thermite::backends::avx2::u32x8\">u32x8</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.AVX2.html\" title=\"struct thermite::backends::avx2::AVX2\">AVX2</a>&gt;","synthetic":false,"types":["thermite::backends::avx2::vu32::u32x8"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/ops/bit/trait.BitAndAssign.html\" title=\"trait core::ops::bit::BitAndAssign\">BitAndAssign</a>&lt;&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.u32x8.html\" title=\"struct thermite::backends::avx2::u32x8\">u32x8</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.AVX2.html\" title=\"struct thermite::backends::avx2::AVX2\">AVX2</a>&gt; as <a class=\"trait\" href=\"thermite/trait.SimdVectorBase.html\" title=\"trait thermite::SimdVectorBase\">SimdVectorBase</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.AVX2.html\" title=\"struct thermite::backends::avx2::AVX2\">AVX2</a>&gt;&gt;::<a class=\"type\" href=\"thermite/trait.SimdVectorBase.html#associatedtype.Element\" title=\"type thermite::SimdVectorBase::Element\">Element</a>&gt; for <a class=\"struct\" href=\"thermite/backends/avx2/struct.u32x8.html\" title=\"struct thermite::backends::avx2::u32x8\">u32x8</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.AVX2.html\" title=\"struct thermite::backends::avx2::AVX2\">AVX2</a>&gt;","synthetic":false,"types":["thermite::backends::avx2::vu32::u32x8"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/ops/bit/trait.BitAndAssign.html\" title=\"trait core::ops::bit::BitAndAssign\">BitAndAssign</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.u64x8.html\" title=\"struct thermite::backends::avx2::u64x8\">u64x8</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.AVX2.html\" title=\"struct thermite::backends::avx2::AVX2\">AVX2</a>&gt;&gt; for <a class=\"struct\" href=\"thermite/backends/avx2/struct.u64x8.html\" title=\"struct thermite::backends::avx2::u64x8\">u64x8</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.AVX2.html\" title=\"struct thermite::backends::avx2::AVX2\">AVX2</a>&gt;","synthetic":false,"types":["thermite::backends::avx2::vu64::u64x8"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/ops/bit/trait.BitAndAssign.html\" title=\"trait core::ops::bit::BitAndAssign\">BitAndAssign</a>&lt;&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.u64x8.html\" title=\"struct thermite::backends::avx2::u64x8\">u64x8</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.AVX2.html\" title=\"struct thermite::backends::avx2::AVX2\">AVX2</a>&gt; as <a class=\"trait\" href=\"thermite/trait.SimdVectorBase.html\" title=\"trait thermite::SimdVectorBase\">SimdVectorBase</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.AVX2.html\" title=\"struct thermite::backends::avx2::AVX2\">AVX2</a>&gt;&gt;::<a class=\"type\" href=\"thermite/trait.SimdVectorBase.html#associatedtype.Element\" title=\"type thermite::SimdVectorBase::Element\">Element</a>&gt; for <a class=\"struct\" href=\"thermite/backends/avx2/struct.u64x8.html\" title=\"struct thermite::backends::avx2::u64x8\">u64x8</a>&lt;<a class=\"struct\" href=\"thermite/backends/avx2/struct.AVX2.html\" title=\"struct thermite::backends::avx2::AVX2\">AVX2</a>&gt;","synthetic":false,"types":["thermite::backends::avx2::vu64::u64x8"]},{"text":"impl&lt;S:&nbsp;<a class=\"trait\" href=\"thermite/trait.Simd.html\" title=\"trait thermite::Simd\">Simd</a>, V&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/ops/bit/trait.BitAndAssign.html\" title=\"trait core::ops::bit::BitAndAssign\">BitAndAssign</a>&lt;<a class=\"struct\" href=\"thermite/struct.BitMask.html\" title=\"struct thermite::BitMask\">BitMask</a>&lt;S, V&gt;&gt; for <a class=\"struct\" href=\"thermite/struct.BitMask.html\" title=\"struct thermite::BitMask\">BitMask</a>&lt;S, V&gt;","synthetic":false,"types":["thermite::mask::BitMask"]},{"text":"impl&lt;S:&nbsp;<a class=\"trait\" href=\"thermite/trait.Simd.html\" title=\"trait thermite::Simd\">Simd</a>, V&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/ops/bit/trait.BitAndAssign.html\" title=\"trait core::ops::bit::BitAndAssign\">BitAndAssign</a>&lt;<a class=\"struct\" href=\"thermite/struct.Mask.html\" title=\"struct thermite::Mask\">Mask</a>&lt;S, V&gt;&gt; for <a class=\"struct\" href=\"thermite/struct.Mask.html\" title=\"struct thermite::Mask\">Mask</a>&lt;S, V&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;V: <a class=\"trait\" href=\"thermite/trait.SimdVector.html\" title=\"trait thermite::SimdVector\">SimdVector</a>&lt;S&gt;,&nbsp;</span>","synthetic":false,"types":["thermite::mask::Mask"]}];
if (window.register_implementors) {window.register_implementors(implementors);} else {window.pending_implementors = implementors;}})()