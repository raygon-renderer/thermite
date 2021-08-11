(function() {var implementors = {};
implementors["thermite"] = [{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite/backends/avx1/struct.AVX1.html\" title=\"struct thermite::backends::avx1::AVX1\">AVX1</a>","synthetic":true,"types":["thermite::backends::avx1::AVX1"]},{"text":"impl&lt;S&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite/backends/avx2/struct.f32x8.html\" title=\"struct thermite::backends::avx2::f32x8\">f32x8</a>&lt;S&gt;","synthetic":true,"types":["thermite::backends::avx2::vf32::f32x8"]},{"text":"impl&lt;S&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite/backends/avx2/struct.f64x8.html\" title=\"struct thermite::backends::avx2::f64x8\">f64x8</a>&lt;S&gt;","synthetic":true,"types":["thermite::backends::avx2::vf64::f64x8"]},{"text":"impl&lt;S&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite/backends/avx2/struct.i32x8.html\" title=\"struct thermite::backends::avx2::i32x8\">i32x8</a>&lt;S&gt;","synthetic":true,"types":["thermite::backends::avx2::vi32::i32x8"]},{"text":"impl&lt;S&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite/backends/avx2/struct.i64x8.html\" title=\"struct thermite::backends::avx2::i64x8\">i64x8</a>&lt;S&gt;","synthetic":true,"types":["thermite::backends::avx2::vi64::i64x8"]},{"text":"impl&lt;S&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite/backends/avx2/struct.u32x8.html\" title=\"struct thermite::backends::avx2::u32x8\">u32x8</a>&lt;S&gt;","synthetic":true,"types":["thermite::backends::avx2::vu32::u32x8"]},{"text":"impl&lt;S&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite/backends/avx2/struct.u64x8.html\" title=\"struct thermite::backends::avx2::u64x8\">u64x8</a>&lt;S&gt;","synthetic":true,"types":["thermite::backends::avx2::vu64::u64x8"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite/backends/avx2/struct.AVX2.html\" title=\"struct thermite::backends::avx2::AVX2\">AVX2</a>","synthetic":true,"types":["thermite::backends::avx2::AVX2"]},{"text":"impl&lt;T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite/struct.Divider.html\" title=\"struct thermite::Divider\">Divider</a>&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a>,&nbsp;</span>","synthetic":true,"types":["thermite::divider::Divider"]},{"text":"impl&lt;T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite/struct.BranchfreeDivider.html\" title=\"struct thermite::BranchfreeDivider\">BranchfreeDivider</a>&lt;T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a>,&nbsp;</span>","synthetic":true,"types":["thermite::divider::BranchfreeDivider"]},{"text":"impl&lt;S, T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite/struct.VPtr.html\" title=\"struct thermite::VPtr\">VPtr</a>&lt;S, T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a>,&nbsp;</span>","synthetic":true,"types":["thermite::pointer::VPtr"]},{"text":"impl&lt;S, V&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite/struct.Mask.html\" title=\"struct thermite::Mask\">Mask</a>&lt;S, V&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;V: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a>,&nbsp;</span>","synthetic":true,"types":["thermite::mask::Mask"]},{"text":"impl&lt;S, V&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite/struct.BitMask.html\" title=\"struct thermite::BitMask\">BitMask</a>&lt;S, V&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;V: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a>,&nbsp;</span>","synthetic":true,"types":["thermite::mask::BitMask"]},{"text":"impl&lt;S, V&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite/math/compensated/struct.Compensated.html\" title=\"struct thermite::math::compensated::Compensated\">Compensated</a>&lt;S, V&gt;","synthetic":true,"types":["thermite::math::compensated::Compensated"]},{"text":"impl&lt;P&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite/math/policies/struct.ExtraPrecision.html\" title=\"struct thermite::math::policies::ExtraPrecision\">ExtraPrecision</a>&lt;P&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;P: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a>,&nbsp;</span>","synthetic":true,"types":["thermite::math::policies::ExtraPrecision"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite/math/policies/struct.UltraPerformance.html\" title=\"struct thermite::math::policies::UltraPerformance\">UltraPerformance</a>","synthetic":true,"types":["thermite::math::policies::UltraPerformance"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite/math/policies/struct.Performance.html\" title=\"struct thermite::math::policies::Performance\">Performance</a>","synthetic":true,"types":["thermite::math::policies::Performance"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite/math/policies/struct.Precision.html\" title=\"struct thermite::math::policies::Precision\">Precision</a>","synthetic":true,"types":["thermite::math::policies::Precision"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite/math/policies/struct.Size.html\" title=\"struct thermite::math::policies::Size\">Size</a>","synthetic":true,"types":["thermite::math::policies::Size"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite/math/policies/struct.Reference.html\" title=\"struct thermite::math::policies::Reference\">Reference</a>","synthetic":true,"types":["thermite::math::policies::Reference"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"enum\" href=\"thermite/math/enum.PrecisionPolicy.html\" title=\"enum thermite::math::PrecisionPolicy\">PrecisionPolicy</a>","synthetic":true,"types":["thermite::math::PrecisionPolicy"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite/math/struct.PolicyParameters.html\" title=\"struct thermite::math::PolicyParameters\">PolicyParameters</a>","synthetic":true,"types":["thermite::math::PolicyParameters"]},{"text":"impl&lt;S&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite/rng/pcg32/struct.PCG32.html\" title=\"struct thermite::rng::pcg32::PCG32\">PCG32</a>&lt;S&gt;","synthetic":true,"types":["thermite::rng::pcg32::PCG32"]},{"text":"impl&lt;S&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite/rng/xoshiro/struct.SplitMix64.html\" title=\"struct thermite::rng::xoshiro::SplitMix64\">SplitMix64</a>&lt;S&gt;","synthetic":true,"types":["thermite::rng::xoshiro::SplitMix64"]},{"text":"impl&lt;S&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite/rng/xoshiro/struct.Xoshiro128Plus.html\" title=\"struct thermite::rng::xoshiro::Xoshiro128Plus\">Xoshiro128Plus</a>&lt;S&gt;","synthetic":true,"types":["thermite::rng::xoshiro::Xoshiro128Plus"]},{"text":"impl&lt;S&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite/rng/xoshiro/struct.Xoshiro256Plus.html\" title=\"struct thermite::rng::xoshiro::Xoshiro256Plus\">Xoshiro256Plus</a>&lt;S&gt;","synthetic":true,"types":["thermite::rng::xoshiro::Xoshiro256Plus"]},{"text":"impl&lt;'a, S, V&gt; !<a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite/iter/struct.AlignedMut.html\" title=\"struct thermite::iter::AlignedMut\">AlignedMut</a>&lt;'a, S, V&gt;","synthetic":true,"types":["thermite::iter::aligned::AlignedMut"]},{"text":"impl&lt;'a, S, V&gt; !<a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite/iter/struct.AlignedMutIter.html\" title=\"struct thermite::iter::AlignedMutIter\">AlignedMutIter</a>&lt;'a, S, V&gt;","synthetic":true,"types":["thermite::iter::aligned::AlignedMutIter"]},{"text":"impl&lt;'a, S, V&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite/iter/struct.SimdSliceIter.html\" title=\"struct thermite::iter::SimdSliceIter\">SimdSliceIter</a>&lt;'a, S, V&gt;","synthetic":true,"types":["thermite::iter::slice::SimdSliceIter"]},{"text":"impl&lt;S, I, V, U&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite/iter/struct.SimdCastIter.html\" title=\"struct thermite::iter::SimdCastIter\">SimdCastIter</a>&lt;S, I, V, U&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a>,<br>&nbsp;&nbsp;&nbsp;&nbsp;U: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a>,<br>&nbsp;&nbsp;&nbsp;&nbsp;V: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a>,&nbsp;</span>","synthetic":true,"types":["thermite::iter::SimdCastIter"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"enum\" href=\"thermite/enum.SimdInstructionSet.html\" title=\"enum thermite::SimdInstructionSet\">SimdInstructionSet</a>","synthetic":true,"types":["thermite::SimdInstructionSet"]},{"text":"impl&lt;S:&nbsp;<a class=\"trait\" href=\"thermite/trait.Simd.html\" title=\"trait thermite::Simd\">Simd</a>, V:&nbsp;<a class=\"trait\" href=\"thermite/trait.SimdVectorBase.html\" title=\"trait thermite::SimdVectorBase\">SimdVectorBase</a>&lt;S&gt;&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite/struct.VectorBuffer.html\" title=\"struct thermite::VectorBuffer\">VectorBuffer</a>&lt;S, V&gt;","synthetic":false,"types":["thermite::buffer::VectorBuffer"]}];
implementors["thermite2"] = [{"text":"impl&lt;R&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite2/backends/vector/struct.Vector.html\" title=\"struct thermite2::backends::vector::Vector\">Vector</a>&lt;R&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;&lt;R as <a class=\"trait\" href=\"thermite2/backends/register/trait.Register.html\" title=\"trait thermite2::backends::register::Register\">Register</a>&gt;::<a class=\"type\" href=\"thermite2/backends/register/trait.Register.html#associatedtype.Storage\" title=\"type thermite2::backends::register::Register::Storage\">Storage</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a>,&nbsp;</span>","synthetic":true,"types":["thermite2::backends::vector::Vector"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite2/backends/avx2/struct.AVX2.html\" title=\"struct thermite2::backends::avx2::AVX2\">AVX2</a>","synthetic":true,"types":["thermite2::backends::avx2::AVX2"]},{"text":"impl&lt;const N:&nbsp;usize&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite2/backends/avx2/struct.AVX2F32Register.html\" title=\"struct thermite2::backends::avx2::AVX2F32Register\">AVX2F32Register</a>&lt;N&gt;","synthetic":true,"types":["thermite2::backends::avx2::AVX2F32Register"]},{"text":"impl&lt;const N:&nbsp;usize&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite2/backends/avx2/struct.AVX2F64Register.html\" title=\"struct thermite2::backends::avx2::AVX2F64Register\">AVX2F64Register</a>&lt;N&gt;","synthetic":true,"types":["thermite2::backends::avx2::AVX2F64Register"]},{"text":"impl&lt;const N:&nbsp;usize&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite2/backends/avx2/struct.AVX2U32Register.html\" title=\"struct thermite2::backends::avx2::AVX2U32Register\">AVX2U32Register</a>&lt;N&gt;","synthetic":true,"types":["thermite2::backends::avx2::AVX2U32Register"]},{"text":"impl&lt;const N:&nbsp;usize&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite2/backends/avx2/struct.AVX2U64Register.html\" title=\"struct thermite2::backends::avx2::AVX2U64Register\">AVX2U64Register</a>&lt;N&gt;","synthetic":true,"types":["thermite2::backends::avx2::AVX2U64Register"]},{"text":"impl&lt;const N:&nbsp;usize&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite2/backends/avx2/struct.AVX2I32Register.html\" title=\"struct thermite2::backends::avx2::AVX2I32Register\">AVX2I32Register</a>&lt;N&gt;","synthetic":true,"types":["thermite2::backends::avx2::AVX2I32Register"]},{"text":"impl&lt;const N:&nbsp;usize&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite2/backends/avx2/struct.AVX2I64Register.html\" title=\"struct thermite2::backends::avx2::AVX2I64Register\">AVX2I64Register</a>&lt;N&gt;","synthetic":true,"types":["thermite2::backends::avx2::AVX2I64Register"]},{"text":"impl <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"enum\" href=\"thermite2/iset/enum.SimdInstructionSet.html\" title=\"enum thermite2::iset::SimdInstructionSet\">SimdInstructionSet</a>","synthetic":true,"types":["thermite2::iset::SimdInstructionSet"]},{"text":"impl&lt;S, V, const N:&nbsp;usize&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite2/widen/struct.Widen.html\" title=\"struct thermite2::widen::Widen\">Widen</a>&lt;S, V, N&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;V: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a>,&nbsp;</span>","synthetic":true,"types":["thermite2::widen::Widen"]}];
implementors["thermite_complex"] = [{"text":"impl&lt;S, V, P&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite_complex/struct.Complex.html\" title=\"struct thermite_complex::Complex\">Complex</a>&lt;S, V, P&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;P: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a>,&nbsp;</span>","synthetic":true,"types":["thermite_complex::Complex"]}];
implementors["thermite_hyperdual"] = [{"text":"impl&lt;S, V, P, const N:&nbsp;usize&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"thermite_hyperdual/struct.HyperdualP.html\" title=\"struct thermite_hyperdual::HyperdualP\">HyperdualP</a>&lt;S, V, P, N&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;P: <a class=\"trait\" href=\"https://doc.rust-lang.org/1.54.0/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a>,&nbsp;</span>","synthetic":true,"types":["thermite_hyperdual::HyperdualP"]}];
if (window.register_implementors) {window.register_implementors(implementors);} else {window.pending_implementors = implementors;}})()