(function() {var implementors = {};
implementors["thermite"] = [{"text":"impl&lt;S, V&gt; Unpin for SimdBuffer&lt;S, V&gt;","synthetic":true,"types":[]},{"text":"impl&lt;S, T&gt; Unpin for VPtr&lt;S, T&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;T: Unpin,<br>&nbsp;&nbsp;&nbsp;&nbsp;&lt;S as Simd&gt;::Vusize: Unpin,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;S, V&gt; Unpin for BitMask&lt;S, V&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;S: Unpin,<br>&nbsp;&nbsp;&nbsp;&nbsp;V: Unpin,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;S, V&gt; Unpin for Mask&lt;S, V&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;S: Unpin,<br>&nbsp;&nbsp;&nbsp;&nbsp;V: Unpin,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl Unpin for AVX1","synthetic":true,"types":[]},{"text":"impl&lt;S&gt; Unpin for f32x8&lt;S&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;S: Unpin,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;S&gt; Unpin for f64x8&lt;S&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;S: Unpin,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;S&gt; Unpin for i32x8&lt;S&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;S: Unpin,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;S&gt; Unpin for i64x8&lt;S&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;S: Unpin,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;S&gt; Unpin for u32x8&lt;S&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;S: Unpin,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl&lt;S&gt; Unpin for u64x8&lt;S&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;S: Unpin,&nbsp;</span>","synthetic":true,"types":[]},{"text":"impl Unpin for AVX2","synthetic":true,"types":[]},{"text":"impl&lt;S&gt; Unpin for i16x8&lt;S&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;S: Unpin,&nbsp;</span>","synthetic":true,"types":[]}];
if (window.register_implementors) {window.register_implementors(implementors);} else {window.pending_implementors = implementors;}})()