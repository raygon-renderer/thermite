# The vector trait hierarchy

All in `thermite::vector` (prelude re-exports), defined in
`crates/thermite/src/vector/mod.rs`. Constrain on the weakest trait that supplies
what you call ([generic-programming.md](generic-programming.md)).

```
GenericVector                         (mod.rs:512)
  |- BitwiseVector                    (1092)
  |    |- BitshiftVector              (1182)
  |- PartialOrdVector                 (1356)
       |- NumericVector               (1401)
            |- SignedVector           (1523)
            |    |- FloatVector       (1797)
            |    |    |- FloatVectorWithBits   (2054)
            |    |- (SignedIntegerVector also here)
            |- IntegerVector          (1580)
                 |- SignedIntegerVector    (1683)
                 |- UnsignedIntegerVector  (1707)

LinAlg3Vector : FloatVector           (2383)   3D/quaternion/mat3 helpers
LinAlg4Vector : LinAlg3Vector         (2499)   4D/mat4 helpers
GenericVector2/3/4                    (2200+)  named-lane accessors x()/y()/z()/w()
```

(Line numbers drift; when one misses, grep `^pub trait <Name>` in `vector/mod.rs`.)

Exact supertrait declarations (from source):

- `GenericVector: 'static + Sized + Default + Copy + Debug + ConstDefault + SplatVector + NewVector + GenericSelectable + HasIsa + CastVector + Interleave`
- `BitwiseVector: GenericVector + <masked bitwise ops>`
- `BitshiftVector: BitwiseVector + <masked shift ops>`
- `PartialOrdVector: GenericVector + PartialEq`
- `NumericVector: PartialOrdVector<Element: NumOps> + <masked arith ops> + NumOps + NumAssignOps + Sum + Product`
- `SignedVector: NumericVector + NegMasked`
- `IntegerVector: NumericVector + BitshiftVector + <divider/saturating/wrapping>`
- `SignedIntegerVector: SignedVector + IntegerVector`
- `UnsignedIntegerVector: IntegerVector`
- `FloatVector: SignedVector<Element: FloatElement> + FloatConsts + CastVector + <fused mul-add ops>`
- `FloatVectorWithBits: BitwiseVector + FloatVector + FullyInteroperable<Bits, SignedBits>`

## Associated types

On `GenericVector`:

| Type / const | Meaning |
|---|---|
| `Element` | scalar element type |
| `LANES: usize` | lane count (const) |
| `lanes() -> usize` | lane count as a value; prefer in loop bounds (today always `LANES`) |
| `Lanes` | lane count as a typenum |
| `Unsigned` | unsigned int vector, same lanes & bit width |
| `Signed` | signed int vector, same lanes & bit width |
| `Mask` | mask type for comparisons (`GenericMask + CastMask`) |
| `EMPTY` | all-zero value (const) |

On `FloatVector`: `ExtendedPrecision` (wider float vector, e.g. `f64` for `f32`,
used internally for compensation).

On `FloatVectorWithBits`: `Bits` (unsigned int view of raw bits), `SignedBits`
(signed view), `NATIVE_CAP` (`NativeCapability` bitflag advertising native
transcendental support).

On `IntegerVector`: `Divider`, `BranchfreeDivider`, `VectorizedDivider` (divisor
representations used by `/`; see [vector-api.md](vector-api.md) and the `divider`
module).

## Capability constants (compile-time, branch with `if const`)

Resolve at compile time inside the dispatcher's `target_feature` context, so
`if const { V::HAS_... } { ... } else { ... }` emits different instruction
sequences per backend at zero runtime cost ([performance.md](performance.md) sec 3):

- `FloatVector::HAS_APPROX_RCP`, `HAS_APPROX_RSQRT` -- `rcp`/`rsqrt` are real
  approximate-reciprocal instructions (true for f32 on x86) vs `1.0/x` fallbacks.
- `MulAddExt::HAS_NATIVE_FMA` (on the FMA ops) -- a `tribool::Tribool`, not a bool:
  `True` = `mul_adde` lowers to a real fused instruction, `False` = definitely
  unfused, `Indeterminate` = runtime-decided (wasm relaxed madd).
- `BitshiftVector::HAS_TRUE_SHIFTV`, `HAS_WIDE_BYTE_SHIFTS` -- variable / byte
  shift hardware support.

## Method index (details in [vector-api.md](vector-api.md) by section)

- **GenericVector** (sec 1): construction, lane access, load/store, gather/scatter,
  lookup, widen/narrow (`extend`/`narrow`/`concat`/`split`),
  interleave/deinterleave, reverse/swap_bytes, `compress`/`compress_z`/`compress_m`
  and the inverse `expand`/`expand_z`/`expand_m`, `align::<OFFSET>`
  (two-vector lane window) + `HAS_NATIVE_ALIGN`, cast/into_bits, `zz`/`nz`,
  prefix/suffix mask, map/fold/reduce.
- **BitwiseVector / BitshiftVector** (sec 2): `&` `|` `^` `!`, `bitandnot`,
  `ternlog`/`bilog`; `shl`/`shr`/`shlv`/`shrv`/`shli`/`shri`, byte shifts,
  rotates, `reverse_bits`.
- **PartialOrdVector** (sec 3): `cmp_lt/le/gt/ge/eq/ne -> Mask`;
  `group_by_value(valid) -> ValueGroups` (equal-value lane groups).
- **NumericVector** (sec 4): `+ - * / %`, `square`, `min`/`max`/`clamp`,
  reductions (`sum_elements`, `prod_elements`, `min_element`, `max_element`,
  `min_max_element`, `arg_minmax`), inclusive scans (`prefix_sum`/`min`/`max` and
  the `reverse_prefix_*` forms), `is_zero`/`is_all_zero`, `pairwise_sum`,
  `scale`, `indexed`/`offset`, constants `ZERO/ONE/TWO/MIN/MAX`.
- **SignedVector** (sec 5): `abs`, `signum`, `copysign`, `neg`,
  `is_positive`/`is_negative`, `NEG_ONE`, `MIN_POSITIVE`.
- **IntegerVector family** (sec 6): `mulhi`/`mullo`, `saturating_add/sub`,
  `wrapping_sum/prod`, dividers, `count_ones/zeros`, `leading_ones/zeros`,
  `trailing_ones/zeros`, `count_conflicts`; signed
  `srai/sra/srav`, `avg_floor/ceil`, `mulhrs` (rounded Q-format multiply);
  unsigned `is_power_of_two`, `avg`, `parity`, `ilog2p1`, `abs_diff`, `in_range`,
  Morton (Z-order) interleave `morton::<N>([Self; N])` / `reverse_morton::<N>()`.
- **FloatVector** (sec 7): `sqrt`, `rcp`, `rsqrt`, `floor/ceil/round/trunc/fract`,
  `mix`, `next_up/down`, `mul_sign`, `signed_zero`, `one_minus_sq`,
  classification (`is_nan/finite/infinite/normal/subnormal`), constants
  `HALF/NEG_ZERO/INFINITY/NEG_INFINITY/NAN/EPSILON`, FMA family
  (`mul_add(e)`, `mul_sub(e)`, `nmul_add(e)`, `nmul_sub(e)`).
- **FloatVectorWithBits** (sec 8): `native_ldexp`/`native_frexp`,
  `native_sin_cos`/... (unsafe, gated by `NATIVE_CAP`),
  `total_order`/`linear_order`.
- **PackedFloatVector<S, F>** (sec 9): fp16/bf16/fp8 storage in u16/u8 vectors;
  `pack` (f32 -> packed, RTNE) / `unpack` (packed -> f32, exact). Formats
  `Fp16`/`Fp16Fast`/`Bf16`/`Fp8E4M3`/`Fp8E5M2` in `element::float::spec`.
- **Sad16/32/64Vector<W>** (sec 10): sum of absolute differences over groups of
  2/4/8 byte lanes of a `u8` vector into `u16`/`u32`/`u64` lanes (same total
  width). `sad32_accum`/`sad64_accum` for blocked loops; no `sad16_accum` (u16
  saturates). x86 `psadbw` / NEON `vpaddlq` / wasm `extadd_pairwise` natively.
- **LinAlg3Vector / LinAlg4Vector** (sec 11 + [geometry.md](geometry.md)):
  `dot3`/`dot4`, `cross3`, `refract`, mat3/mat4 transpose/product/det/inverse,
  quaternion ops.
