; Synthetic fixture: one bad featureless x86_v3 kernel, one good trampoline,
; one out-of-line core_arch call, one featureless outer wrapper that must pass.

define internal <8 x float> @_ZN8thermite7backend6x86_v39registers5F32x83bad17h0000000000000001E(<8 x float> %a, <8 x float> %b) unnamed_addr #0 {
start:
  %0 = fadd <8 x float> %a, %b
  %1 = fmul <8 x float> %0, %a
  ret <8 x float> %1
}

define internal <8 x float> @_ZN8thermite7backend6x86_v39registers5F32x84good17h0000000000000002E(<8 x float> %a, <8 x float> %b) unnamed_addr #1 {
start:
  %0 = fadd <8 x float> %a, %b
  ret <8 x float> %0
}

define internal <8 x float> @_ZN6kernel17__dispatch_x86v3_17h0000000000000003E(<8 x float> %a) unnamed_addr #1 {
start:
  %0 = call <8 x float> @llvm.fma.v8f32(<8 x float> %a, <8 x float> %a, <8 x float> %a)
  ret <8 x float> %0
}

define <8 x float> @_ZN6kernel5outer5X86V317h0000000000000004E(<8 x float> %a) unnamed_addr #0 {
start:
  %0 = tail call <8 x float> @_ZN6kernel17__dispatch_x86v3_17h0000000000000003E(<8 x float> %a)
  ret <8 x float> %0
}

define internal <8 x float> @_ZN5oops6X86V3_17h0000000000000005E(<8 x float> %a) unnamed_addr #0 {
start:
  %0 = call <8 x float> @"_ZN4core9core_arch3x863avx13_mm256_add_ps17h0000000000000006E"(<8 x float> %a, <8 x float> %a)
  ret <8 x float> %0
}

define internal <8 x float> @"_ZN4core9core_arch3x863avx13_mm256_add_ps17h0000000000000006E"(<8 x float> %a, <8 x float> %b) unnamed_addr #1 {
start:
  %0 = fadd <8 x float> %a, %b
  ret <8 x float> %0
}

; Featureless x86_v3 kernel whose only SIMD is inline asm: must be flagged.
define internal <8 x float> @_ZN8thermite7backend6x86_v39registers5F32x87asm_bad17h0000000000000008E(<8 x float> %a, <8 x float> %b) unnamed_addr #0 {
start:
  %0 = tail call <8 x float> asm "vaddps $2, $1, $0", "=v,v,v,~{dirflag},~{fpsr},~{flags}"(<8 x float> %a, <8 x float> %b)
  ret <8 x float> %0
}

define internal i32 @_ZN8thermite3isa11detect_once4init17h0000000000000007E() unnamed_addr #0 {
start:
  ret i32 0
}

declare <8 x float> @llvm.fma.v8f32(<8 x float>, <8 x float>, <8 x float>)

attributes #0 = { nounwind "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" }
attributes #1 = { nounwind "target-cpu"="x86-64" "target-features"="+avx,+avx2,+fma,+popcnt,+sse,+sse2" }

; Rule C control: the round-to-odd emulation reached from an AVX2+FMA
; register type. Attribute group #1 carries avx2/fma, so this trips Rule C
; alone and neither Rule A nor Rule B.
define internal <8 x float> @_ZN9fma_gated4kern17h0000000000000009E(<8 x float> %a) unnamed_addr #1 {
start:
  %0 = call <8 x float> @"thermite::backend::generic::polyfills::math::fmadd_ro_rescue::<thermite::backend::x86_v3::registers::f64x4::F64x4V3>"(<8 x float> %a)
  ret <8 x float> %0
}

declare <8 x float> @"thermite::backend::generic::polyfills::math::fmadd_ro_rescue::<thermite::backend::x86_v3::registers::f64x4::F64x4V3>"(<8 x float>)
