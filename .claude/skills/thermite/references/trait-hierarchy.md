# The vector trait hierarchy

All of these live in `thermite::vector` (re-exported by the prelude) and are
defined in `crates/thermite/src/vector/mod.rs`. Constrain generic code on the
weakest trait that supplies what you call ([generic-programming.md](generic-programming.md)).

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

(Line numbers drift as the file grows; when one misses, grep
`^pub trait <Name>` in `vector/mod.rs` -- the trait names are stable.)

Exact supertrait declarations (quoted from source):

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
| `Lanes` | lane count as a typenum |
| `Unsigned` | unsigned int vector, same lanes & bit width |
| `Signed` | signed int vector, same lanes & bit width |
| `Mask` | mask type for comparisons (`GenericMask + CastMask`) |
| `EMPTY` | all-zero value (const) |

On `FloatVector`: `ExtendedPrecision` (a wider float vector, e.g. `f64` for `f32`,
used internally for compensation).

On `FloatVectorWithBits`: `Bits` (unsigned int view of the raw bits), `SignedBits`
(signed int view), `NATIVE_CAP` (a `NativeCapability` bitflag advertising native
transcendental support).

On `IntegerVector`: `Divider`, `BranchfreeDivider`, `VectorizedDivider` (the
divisor representations used by `/`; see [vector-api.md](vector-api.md) and the
`divider` module).

## Capability constants (compile-time, branch with `if const`)

These resolve at compile time inside the `target_feature` context the dispatcher
establishes, so `if const { V::HAS_... } { ... } else { ... }` emits different
instruction sequences per backend at zero runtime cost
([performance.md](performance.md) section 3):

- `FloatVector::HAS_APPROX_RCP`, `HAS_APPROX_RSQRT` -- whether `rcp`/`rsqrt` are real
  approximate-reciprocal instructions (true for f32 on x86) vs `1.0/x` fallbacks.
- `MulAddExt::HAS_TRUE_FMA` (on the FMA ops) -- whether `mul_adde` lowers to a real
  fused instruction.
- `BitshiftVector::HAS_TRUE_SHIFTV`, `HAS_WIDE_BYTE_SHIFTS` -- variable / byte shift
  hardware support.

## Where the methods live (index)

- **GenericVector**: construction, lane access, load/store, gather/scatter, lookup,
  widen/narrow (`extend`/`narrow`/`concat`/`split`), interleave/deinterleave,
  reverse/swap_bytes/compress, `align::<OFFSET>` (two-vector lane window), cast/into_bits,
  `zz`/`nz`, prefix/suffix mask, map/fold/reduce. -> [vector-api.md](vector-api.md) section 1.
- **BitwiseVector / BitshiftVector**: `&` `|` `^` `!`, `bitandnot`, `ternlog`/`bilog`;
  `shl`/`shr`/`shlv`/`shrv`/`shli`/`shri`, byte shifts, rotates, `reverse_bits`.
  -> section 2.
- **PartialOrdVector**: `cmp_lt/le/gt/ge/eq/ne -> Mask`. -> section 3.
- **NumericVector**: `+ - * / %`, `square`, `min`/`max`/`clamp`, reductions
  (`sum_elements`, `prod_elements`, `min_element`, `max_element`, `min_max_element`,
  `arg_minmax`), `is_zero`/`is_all_zero`, `pairwise_sum`, `scale`, `indexed`/`offset`,
  constants `ZERO/ONE/TWO/MIN/MAX`. -> section 4.
- **SignedVector**: `abs`, `signum`, `copysign`, `neg`, `is_positive`/`is_negative`,
  `NEG_ONE`, `MIN_POSITIVE`. -> section 5.
- **IntegerVector family**: `mulhi`/`mullo`, `saturating_add/sub`, `wrapping_sum/prod`,
  dividers, `count_ones/zeros`, `leading_ones/zeros`; signed `srai/sra/srav`,
  `avg_floor/ceil`, `mulhrs` (rounded Q-format multiply); unsigned `is_power_of_two`,
  `avg`, `parity`, `ilog2p1`, `abs_diff`, `in_range`, and Morton-code
  (Z-order) interleave: `morton::<N>([Self; N])` / `reverse_morton::<N>()`. -> section 6.
- **FloatVector**: `sqrt`, `rcp`, `rsqrt`, `floor/ceil/round/trunc/fract`, `mix`,
  `next_up/down`, `mul_sign`, `signed_zero`, `one_minus_sq`, classification
  (`is_nan/finite/infinite/normal/subnormal`), constants
  `HALF/NEG_ZERO/INFINITY/NEG_INFINITY/NAN/EPSILON`, and the FMA family
  (`mul_add(e)`, `mul_sub(e)`, `nmul_add(e)`, `nmul_sub(e)`). -> section 7.
- **FloatVectorWithBits**: `native_ldexp`/`native_frexp`, `native_sin_cos`/... (unsafe,
  gated by `NATIVE_CAP`), `total_order`/`linear_order`. -> section 8.
- **LinAlg3Vector / LinAlg4Vector**: `dot3`/`dot4`, `cross3`, `refract`, mat3/mat4
  transpose/product/det/inverse, quaternion ops. -> [geometry.md](geometry.md) and
  [vector-api.md](vector-api.md) section 9.
