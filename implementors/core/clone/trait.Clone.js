(function() {var implementors = {};
implementors["thermite"] = [{"text":"impl Clone for AVX1","synthetic":false,"types":[]},{"text":"impl&lt;S:&nbsp;Clone + Simd&gt; Clone for f32x8&lt;S&gt;","synthetic":false,"types":[]},{"text":"impl&lt;S:&nbsp;Clone + Simd&gt; Clone for f64x8&lt;S&gt;","synthetic":false,"types":[]},{"text":"impl&lt;S:&nbsp;Clone + Simd&gt; Clone for i32x8&lt;S&gt;","synthetic":false,"types":[]},{"text":"impl&lt;S:&nbsp;Clone + Simd&gt; Clone for i64x8&lt;S&gt;","synthetic":false,"types":[]},{"text":"impl&lt;S:&nbsp;Clone + Simd&gt; Clone for u32x8&lt;S&gt;","synthetic":false,"types":[]},{"text":"impl&lt;S:&nbsp;Clone + Simd&gt; Clone for u64x8&lt;S&gt;","synthetic":false,"types":[]},{"text":"impl Clone for AVX2","synthetic":false,"types":[]},{"text":"impl&lt;T:&nbsp;Copy&gt; Clone for BranchfreeDivider&lt;T&gt;","synthetic":false,"types":[]},{"text":"impl&lt;T:&nbsp;Copy&gt; Clone for Divider&lt;T&gt;","synthetic":false,"types":[]},{"text":"impl&lt;S:&nbsp;Clone + Simd, T:&nbsp;Clone&gt; Clone for VPtr&lt;S, T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;S::Vusize: Clone,&nbsp;</span>","synthetic":false,"types":[]},{"text":"impl&lt;S:&nbsp;Clone + Simd, V:&nbsp;Clone&gt; Clone for Mask&lt;S, V&gt;","synthetic":false,"types":[]},{"text":"impl&lt;S:&nbsp;Simd, V&gt; Clone for BitMask&lt;S, V&gt;","synthetic":false,"types":[]},{"text":"impl&lt;S:&nbsp;Clone + Simd, V:&nbsp;Clone + SimdFloatVector&lt;S&gt;&gt; Clone for Compensated&lt;S, V&gt;","synthetic":false,"types":[]},{"text":"impl&lt;S:&nbsp;Simd, V:&nbsp;SimdFloatVector&lt;S&gt;, P:&nbsp;Policy&gt; Clone for Complex&lt;S, V, P&gt;","synthetic":false,"types":[]},{"text":"impl Clone for PrecisionPolicy","synthetic":false,"types":[]},{"text":"impl Clone for UltraPerformance","synthetic":false,"types":[]},{"text":"impl Clone for Performance","synthetic":false,"types":[]},{"text":"impl Clone for Precision","synthetic":false,"types":[]},{"text":"impl Clone for Size","synthetic":false,"types":[]},{"text":"impl Clone for Reference","synthetic":false,"types":[]},{"text":"impl&lt;S:&nbsp;Clone + Simd&gt; Clone for PCG32&lt;S&gt;","synthetic":false,"types":[]},{"text":"impl&lt;S:&nbsp;Clone + Simd&gt; Clone for SplitMix64&lt;S&gt;","synthetic":false,"types":[]},{"text":"impl&lt;S:&nbsp;Clone + Simd&gt; Clone for Xoshiro128Plus&lt;S&gt;","synthetic":false,"types":[]},{"text":"impl&lt;S:&nbsp;Clone + Simd&gt; Clone for Xoshiro256Plus&lt;S&gt;","synthetic":false,"types":[]},{"text":"impl&lt;S:&nbsp;Simd, V:&nbsp;SimdVectorBase&lt;S&gt;&gt; Clone for SimdSliceIter&lt;'_, S, V&gt;","synthetic":false,"types":[]},{"text":"impl&lt;S:&nbsp;Simd, I, V, U&gt; Clone for SimdCastIter&lt;S, I, V, U&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;I: Clone,&nbsp;</span>","synthetic":false,"types":[]},{"text":"impl Clone for SimdInstructionSet","synthetic":false,"types":[]}];
if (window.register_implementors) {window.register_implementors(implementors);} else {window.pending_implementors = implementors;}})()