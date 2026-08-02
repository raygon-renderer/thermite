# Vector method reference

Methods on `Vector<R>`, grouped by providing trait. `V` = vector type,
`E = V::Element`, `M = V::Mask`. Defined in `crates/thermite/src/vector/mod.rs`
and `vector/ops.rs`.

> Every `[masked]` method gets `_c`/`_m`/`_z` siblings (last section). `_c`/`_z`:
> **mask is the first extra argument**; `_m` (merge): `src` first, then mask:
> `a.op_m(src, mask, rhs)`.

## 1. GenericVector

```rust
// Construction
V::new([e0, e1, ...])        V::splat(e)         V::single(e)   // single: lane 0 = e, rest 0
V::EMPTY                                              // all-zero (also ZERO/ONE/... on NumericVector)
const X: V = thermite::const_new!(f32: [1.0, 0.0, 0.0]);  // const vector value (usable in const fn / assoc consts)
v.into_array() -> GenericArray<E, V::Lanes>          V::from_slice(&[E])     v.copy_to_slice(&mut [E])
v.as_slice() -> &[E]         v.as_mut_slice() -> &mut [E]   // borrow lanes as a slice (VectorWithRegister
                                                            // trait; the prelude imports it)
V::LANES -> usize (const)    V::lanes() -> usize            // prefer lanes() in loop bounds/address math
                                                            // (forward-compatible with runtime-length backends)

// Lane access
v.extract::<I>() -> E        v.insert::<I>(e) -> V   v.broadcast::<I>() -> V     // compile-time index
v.extractv(i)    -> E        v.insertv(i, e) -> V    v.broadcastv(i) -> V        // runtime index
v.x() v.y() v.z() v.w()                                                          // GenericVector2/3/4
v.reverse()                  v.swap_bytes()

// Memory (unsafe load/store; ptr must satisfy alignment for the aligned forms).
// FOOTGUN: V::load is ALIGNED -- loading a table from a plain Box/Vec faults
// NONDETERMINISTICALLY (0xc0000005) since heap alloc may or may not be aligned
// enough. Use load_unaligned or an aligned container for heap-allocated tables.
V::load(ptr)  V::load_unaligned(ptr)  V::load_streaming(ptr)
v.store(ptr)  v.store_unaligned(ptr)  v.store_streaming(ptr)  v.store_masked(mask, ptr)
V::align_slice(&[E])     -> (&[E], &[V], &[E])      // head, aligned middle, tail
V::align_slice_mut(&mut [E])                          // (prefer the SimdSlice trait, see slices doc)

// Gather / scatter / lookup  (I: VectorIndices)
V::gather(slice, idx)   V::gather_or(slice, idx, or)   V::gather_or_zero(slice, idx)
V::gather_if(slice, mask, idx, or)
v.scatter(slice, idx)   v.scatter_if(slice, mask, idx)
V::lookup(table, unsigned_idx)    // small-table gather

// Widen / narrow
let wide:   W = narrow.extend()     let narrow: N = wide.narrow()        // zero-extend / drop upper
let full:   F = lo.concat(hi)       let (lo, hi)  = full.split()

// Interleave / deinterleave (AoS <-> SoA)
let (lo, hi) = a.interleave(b)      let (a, b)    = lo.deinterleave(hi)
a.interleave_by::<GROUP>(b)  a.deinterleave_by::<GROUP>(b)   // group-granularity (GROUP must divide LANES)
V::interleave_radix([a, b, ...])   V::deinterleave_radix([...])   // radix-N, N inferred from array len;
    // concat(out)[q*N + r] == inputs[r].extract(q). N==2 -> native interleave, N==3 -> native radix-3.
V::deinterleave_radix_by::<N, GROUP>([...])  V::interleave_radix_by::<N, GROUP>([...])
    // two-axis unification (GROUP==1 -> _radix, N==2 -> _by). Square case N == LANES/GROUP is a
    // register-array transpose of GROUP-wide elements: ::<4,2> on 8-lane f32 = 4x4 interleaved-complex
    // transpose (8 ops on AVX2); ::<4,1> on f64x4 = 4x4 f64 transpose (FFT codelet / small-matrix primitive).
V::load_deinterleaved(...)  v.store_interleaved(...)   // AoS<->SoA memory form, arbitrary N (NEON LD3/ST3)

// Compaction, and its inverse (AVX-512 vpcompress/vpexpand; table fallback <=8 lanes)
v.compress(mask)            // stable left-pack of true lanes
v.compress_z(mask)          // left-pack true lanes, zero the rest
v.compress_m(src, mask)     // left-pack; lanes at/beyond popcount keep src's own lanes
                            // (position-addressed, not mask-addressed) -- the accumulator
                            // step of a buffered stream compactor
v.expand(mask)              // scatter the packed low lanes back out to the true lanes;
                            // the EXACT inverse permutation of compress, so
                            // v.compress(m).expand(m) == v and v.expand(m).compress(m) == v
v.expand_z(mask)            // ...unselected lanes zeroed instead of reading the tail
v.expand_m(src, mask)       // ...unselected lanes take src

// Wavefront round trip: compact the active lanes, work on the packed front, scatter back.
let packed = v.compress_z(active);
let done   = kernel(packed);
let out    = done.expand_m(background, active);   // == active.select(kernel-per-lane, background)

// Two-vector element align (palignr family; any element type): the window of
// LANES lanes starting at lane OFFSET of [a, b]. OFFSET=0 -> a, OFFSET=LANES -> b.
a.align::<OFFSET>(b)   // sliding window across a load boundary; int backends use native byte aligns
V::HAS_NATIVE_ALIGN    // whether that's one instruction or a shuffle+blend fallback.
    // Same results either way, so it only picks a lowering - gate on it when building a
    // LADDER of aligns (the prefix-scan family), where the emulated form can lose to a
    // lane walk. Composites (Dual/Compensated/Complex) forward their inner vector's.

// Casting
v.cast::<W>()         // numeric cast, like `as`
v.fast_cast::<W>()    // faster, may skip edge cases
v.into_bits::<W>()    // zero-cost bit reinterpret
v.saturating_cast::<W>()

// Shuffles (swizzle.rs; `Swizzle` trait + `swizzle!` macro)
a.swizzle(b, indices)              // runtime-index two-vector shuffle
                                   // (indices: GenericArray<u32, Lanes>; 0..LANES = a, LANES..2*LANES = b)
a.permute(indices)                 // runtime-index single-vector permute (pshufb/vqtbl/i8x16.swizzle class)
a.swizzle_const::<I>(b)  a.permute_const::<I>()   // compile-time indices via SwizzleIndices
thermite::swizzle!(v, [1, 0, 3, 2])  // const shuffle macro; stay in-register in hot loops
// (register layer also has permutev / shuffle::<IMM8> -- backend impls only)

// Mask helpers
v.zz(mask)   // zero lanes where mask is FALSE   (keep where true)
v.nz(mask)   // zero lanes where mask is TRUE
V::prefix_mask(n)    V::suffix_mask(n)            // first / last n lanes true

// Scalar fallback (per-lane closures; see perf doc: avoid in hot target_feature code)
v.map(|x| ...)   v.fold(init, |acc, x| ...)   v.reduce(|a, b| ...)
```

## 2. BitwiseVector / BitshiftVector

```rust
a & b   a | b   a ^ b   !a              a.bitandnot(b)   // a & !b at the VECTOR/MASK layer.
    // NOTE the register layer is the x86 convention R::bitandnot(lhs, rhs) = !lhs & rhs;
    // the Vector/Mask impls swap operands when delegating ("exposed logic is reversed").
V::ternlog::<IMM>(a, b, c)              V::bilog::<IMM>(a, b)   // see ternlog_imm! macro
a << n  a >> n   (n: u32 or V::Unsigned)
a.shli::<I>()  a.shri::<I>()  a.shl(n)  a.shr(n)  a.shlv(unsigned)  a.shrv(unsigned)
a.bshli::<I>()  a.bshri::<I>()                              // byte shifts
a.rol(n)  a.ror(n)  a.roli::<I>()  a.rori::<I>()  a.rolv(u)  a.rorv(u)   a.reverse_bits()
```

## 3. PartialOrdVector

```rust
let m: M = a.cmp_lt(b);   // also cmp_le, cmp_gt, cmp_ge, cmp_eq, cmp_ne  -> Mask

// Divergence -> a short run of uniform sub-packets: partition the lanes selected by
// `valid` into groups of equal value, in order of first occurrence, each lane once.
let mut groups = ids.group_by_value(valid);          // -> ValueGroups<V>
while let Some((value, lanes)) = groups.next_group() {   // also impls Iterator
    do_uniform_work(value, lanes);
}
groups.remaining()  groups.is_empty()   // stop part-way and keep the rest
// Cost scales with the number of DISTINCT values, not LANES (~broadcast + cmp + 2 mask
// ops per group); a uniform packet is one iteration. Pass Mask::TRUTHY for all lanes.
```

(`PartialEq`/`PartialOrd` for `Vector` itself are whole-vector: `==` is "all lanes
equal". For per-lane results use `cmp_*`.)

## 4. NumericVector

```rust
a + b   a - b   a * b   a / b   a % b     a.square()
a.min(b)   a.max(b)   a.clamp(lo, hi)
v.sum_elements()  v.prod_elements()  v.min_element()  v.max_element()
v.min_max_element() -> (E, E)    v.arg_minmax() -> (usize, usize)

// Inclusive prefix scans: keep every partial in its own lane instead of collapsing
// to a scalar (bin offsets, compaction write indices, running extents).
v.prefix_sum()   v.prefix_min()   v.prefix_max()            // out[i] = op(v[0]..=v[i])
v.reverse_prefix_sum()  v.reverse_prefix_min()  v.reverse_prefix_max()  // out[i] = op(v[i]..)
// O(log2 LANES) align ladder where the backend has a native align, a sequential lane
// walk where it does not -- chosen at compile time. No masked variants: neutralise the
// lanes you want out first, e.g. `v.zz(mask).prefix_sum()`.
// After prefix_sum the LAST lane is the whole-register total, so the carry into the
// next chunk is `scanned.reverse().broadcast::<0>()` -- no horizontal reduction.
// min/max: exact including infinities; on NaN input, which operand wins is unspecified.
v.is_zero() -> M    v.is_all_zero() -> bool
V::pairwise_sum(lo, hi)    V::relaxed_pairwise_sum(lo, hi)
v.scale(e)                 // v * splat(e); lowers to OpVectorTimesScalar on SPIR-V
V::indexed()  // [0, 1, ..., LANES-1]      V::offset()  // splat(LANES)
// constants:
V::ZERO  V::ONE  V::TWO  V::MIN  V::MAX
```

## 5. SignedVector

```rust
v.abs()   v.signum()   v.copysign(sign)   -v
v.is_positive() -> M   v.is_negative() -> M   v.select_negative(if_neg, if_pos)
V::NEG_ONE   V::MIN_POSITIVE
```

## 6. Integer vectors

```rust
// IntegerVector
a.mulhi(b)  a.mullo(b)   a.saturating_add(b)  a.saturating_sub(b)
v.wrapping_sum()  v.wrapping_prod()   v.count_ones()  v.count_zeros()
v.leading_ones()  v.leading_zeros()   v.trailing_ones()  v.trailing_zeros()
v.count_conflicts()   // per lane, how many EARLIER lanes hold the same value
    // (AVX-512CD vpconflict + popcount; portable rotate ladder otherwise).
    // `.cmp_eq(V::ZERO)` is the first-occurrence mask, and the count IS the round
    // number for a conflicting read-modify-write: a lane of rank r is safe in round r.
    // That is what makes a vectorized histogram / bin increment correct where a plain
    // scatter silently drops duplicate writes.
// SignedIntegerVector
v.srai::<I>()  v.sra(n)  v.srav(unsigned)   a.avg_floor(b)  a.avg_ceil(b)
a.mulhrs(b)    // rounded Q(W-1) fixed-point multiply (i16: Q15, x86 PMULHRSW); rounds, not truncates
// UnsignedIntegerVector
v.is_power_of_two() -> M   a.avg(b)   v.parity()   v.ilog2p1()   v.next_power_of_two_m1()
a.abs_diff(b)              // |a - b| without overflow (saturating-sub form)
x.in_range(lo, hi) -> M    // mask of lo <= x <= hi, inclusive (branchless, one compare)

// Morton codes (Z-order curve): bit-interleave N coordinate vectors into one
// code and back. N = 2 or 3 typically (BVH/octree keys, grid binning).
// With `avx2-pclmul` (default) the 2D path uses CLMUL on u64 lanes.
let code = V::morton::<N>([x, y, ...]);   let [x, y, ...] = code.reverse_morton::<N>();

// Division by a precomputed divisor (constant-time, branchfree). See `divider` module.
use thermite::{BranchfreeDivider, Divider};
let d = BranchfreeDivider::u32(7);   let q = my_u32_vec / d;   // BranchfreeDivider::u32(1) unsupported
let d = denominators.to_divider();   let q = numerators / d;   // per-lane divisors
```

## 7. FloatVector

```rust
v.sqrt()  v.rcp()  v.rsqrt()                       // rcp/rsqrt approximate where HAS_APPROX_* (f32)
v.floor() v.ceil() v.round() v.trunc() v.fract()
v.abs()  v.signum()  v.copysign(s)  v.mul_sign(s)  v.signed_zero()
v.mix(a, b)            // a*(1-v) + b*v   (linear interp; v is the parameter)
v.one_minus_sq()       // accurate 1 - v*v (no cancellation; see perf doc)
v.next_up()  v.next_down()
v.is_nan() v.is_finite() v.is_infinite() v.is_normal() v.is_subnormal() v.is_zero_or_subnormal()
// constants (plus ~45 from FloatConsts: V::PI, V::E, V::LN_2, V::SQRT_2, ...):
V::HALF  V::NEG_ZERO  V::INFINITY  V::NEG_INFINITY  V::NAN  V::EPSILON

// Interleaved-complex arithmetic (x86 ADDSUBPS / VFMADDSUB semantics; masked
// variants exist; NEON/wasm use a sign-mask polyfill folded at compile time):
a.addsub(b)          // [a0-b0, a1+b1, a2-b2, ...]  even lanes subtract, odd add
a.fmaddsub(b, c)     // [a0*b0 - c0, a1*b1 + c1, ...]  fused multiply then addsub
a.fmsubadd(b, c)     // opposite parity: even lanes add, odd subtract
// complex mul over [re, im, ...] lanes: fmaddsub(a, wr_splat, a_swapped * wi_splat)

// FMA family. Sign conventions:
a.mul_adde(b, c)   // a*b + c   <-- PREFER the `e` (estimating) forms by default
a.mul_sube(b, c)   // a*b - c
a.nmul_adde(b, c)  // c - a*b
a.nmul_sube(b, c)  // -a*b - c
a.mul_add(b, c)    // a*b + c, always single-rounded: real FMA, else vectorized compensated
                   // emulation (scalar libm::fma only under disable_fast_fma). See math.md.
```

See [performance.md](performance.md) for which FMA variant to use -- this is the
single most common perf footgun.

## 8. FloatVectorWithBits

```rust
unsafe { v.native_ldexp(exp: V::SignedBits) }      unsafe { v.native_frexp() -> (V, V::SignedBits) }
unsafe { v.native_sin_cos::<P>() }  // and native_sin/cos/tan/exp/exp2/ln/log2/powf, gated by NATIVE_CAP
v.total_order() -> V::SignedBits    v.linear_order() -> V::SignedBits   // IEEE total order keys (NaN-safe sort)
```

Generic float code that needs bit access without requiring `FloatVectorWithBits`
can use `FloatVector::with_bits([...], kernel)`, which returns `Option` (None when
the concrete type lacks bit views, e.g. some composites).

## 9. Packed floats: fp16 / bf16 / fp8 (storage formats, not compute)

Sub-f32 formats live in **unsigned integer vectors** (u16 for the 16-bit
formats, u8 for fp8) and are transcoded to/from the same-lane-count `f32`
vector at boundaries -- compute stays in f32. Formats
(`thermite::element::float::spec`, all implementing `FloatSpec`):

| Format | Layout (s/e/m, bias) | Specials |
|---|---|---|
| `Fp16` | 1/5/10, 15 | IEEE binary16: inf + NaN |
| `Fp16Fast` | same layout | `Unchecked`: inf/NaN elided for speed -- unpack decodes all-ones-exponent as large normals, pack flushes non-finite/overflow to signed zero. Finite data only. |
| `Bf16` | 1/8/7, 127 | exactly the top 16 bits of an f32; inf + NaN |
| `Fp8E4M3` | 1/4/3, 7 | OCP: no inf, single NaN `S.1111.111`, max finite 448, out-of-range saturates |
| `Fp8E5M2` | 1/5/2, 15 | OCP, IEEE-style: inf + NaN |

```rust
use thermite::element::float::spec::Fp16;
use thermite::vector::PackedFloatVector;   // : GenericVector

// Blanket-implemented wherever the register implements PackedFloatRegister<S, FR>,
// e.g. u16x8<S>: PackedFloatVector<Fp16, f32x8<S>>.
fn widen<U, F>(halves: U) -> F where U: PackedFloatVector<Fp16, F> { halves.unpack() }

U::pack(f32_vec) -> U   // encode: round-to-nearest-ties-even; overflow per format
                        // (+-inf IEEE, saturate for no-inf schemes, flush for Unchecked)
u.unpack() -> F         // decode: EXACT (every sub-f32 value is representable in f32)
```

Backends override with hardware where it exists (F16C `vcvtph2ps` -- assumed
with AVX2 under the default `avx2-f16c` feature; AVX512-BF16 on that tier);
otherwise a generic branchless shift/mask/select fallback that is denormal-safe
(exact independent of the FPU's flush-to-zero mode).

## 10. Sum of absolute differences (`SadN`, u8 vectors)

Sums `|a - b|` over aligned groups of lanes into wider accumulator lanes. The
suffix is the **output element width**; the group size is
`output_bits / input_element_bits`, so the output is always a same-width view of
the input. Output lanes = `max(1, LANES / group)`; a register holding fewer lanes
than one group sums everything it has into a single lane.

**Every unsigned slot that can carry these does**, at every width, so generic code
never case-splits:

| input | group | `sad16` | `sad32` | `sad64` |
|---|---|---|---|---|
| `u8` | 2/4/8 | `u16`, half lanes | `u32`, quarter | `u64`, eighth |
| `u16` | 2/4 | -- | `u32`, half lanes | `u64`, quarter |
| `u32` | 2 | -- | -- | `u64`, half lanes |

Concretely: `u8x16 -> u16x8/u32x4/u64x2`, `u16x8 -> u32x4/u64x2`,
`u32x4 -> u64x2`, and likewise for `u8x8`/`u8x4`/`u8x2`, `u16x16`/`u16x4`/`u16x2`,
`u32x16`/`u32x8`/`u32x2`. Where a rung is narrower than one group the single
output lane sums the whole register (e.g. `u8x2.sad64()` sums both bytes).

The **`xN` native-width slots carry it too** (`u8xN`/`u16xN`/`u32xN`), and the lane
ratios line up on every backend: `u8xN.sad16()` is `u16xN`, `.sad32()` is `u32xN`,
`.sad64()` is `u64xN`. That is how AVX2's 256-bit byte register is reached -- there
is no fixed-width `u8x32` slot, but `u8xN` *is* `u8x32` on v3, so `_mm256_sad_epu8`
(and the 256-bit `pmaddubsw`/`pmaddwd`) are on that path.

Below 128 bits there is no SIMD win, so the sub-native ladder is lane-wise on
every backend; `ArrayRegister` composites (`u16x16 = [u16x8; 2]`, ...) delegate to
the inner register, since a group never spans two of them.

> **Inference wart:** on the scalar backend the 1-lane elements also implement SAD
> (they are its `xN` slots), which makes the composite blanket a second applicable
> impl for `ArrayRegister`-backed vectors. Generic code is unambiguous because the
> `Simd`/`SimdVectors` bounds pin the output, but a *concrete* `a.sad16(b)` on such
> a vector may need an explicit output type.

> `Sad16` exists only for `u8` inputs -- on `u16` its group would be one lane,
> i.e. just `abs_diff`.

```rust
use thermite::vector::{Sad16Vector, Sad32Vector, Sad64Vector};

a.sad16(b) -> u16x8    // groups of 2 bytes, each result <= 510
a.sad32(b) -> u32x4    // groups of 4 bytes, each result <= 1020
a.sad64(b) -> u64x2    // groups of 8 bytes, each result <= 2040 (x86 PSADBW semantics)

a.sad32_accum(acc, b)  // acc + a.sad32(b) -- ~4.2e6 accumulations of headroom
a.sad64_accum(acc, b)  // acc + a.sad64(b) -- effectively unbounded (~9e15)
```

`sad64`'s u64 lanes are deliberate accumulation headroom, so the intended shape of
a byte-buffer reduction is to accumulate through the loop and reduce horizontally
exactly once at the end:

```rust
let mut acc = u64x2::ZERO;
for (a, b) in blocks { acc = a.sad64_accum(acc, b); }
let total = acc.sum_elements();
```

**No `sad16_accum` exists** -- a u16 lane saturates after only ~128 accumulations,
so widen deliberately (`sad32`/`sad64`) rather than accumulating the narrow form.

Instruction count past the absolute difference, per backend:

u8 input:

| | `sad16` | `sad32` | `sad64` |
|---|---|---|---|
| x86 v1 (SSE2) | SWAR | SWAR | **`psadbw`** (1) |
| x86 v2/v3 | **`pmaddubsw`** (1) | +**`pmaddwd`** (2) | **`psadbw`** (1) |
| NEON | **`vpaddlq_u8`** (1) | +**`vpaddlq_u16`** (2) | +**`vpaddlq_u32`** (3) |
| wasm | **`extadd_pairwise`** (1) | x2 (2) | x2 + fold (5) |
| scalar | SWAR | SWAR | SWAR |

u16 / u32 input:

| | u16 `sad32` | u16 `sad64` | u32 `sad64` |
|---|---|---|---|
| x86 (all) | SWAR | SWAR | SWAR |
| NEON | **`vpaddlq_u16`** (1) | +**`vpaddlq_u32`** (2) | **`vpaddlq_u32`** (1) |
| wasm | **`extadd_pairwise`** (1) | + fold | SWAR |
| scalar | lane-wise | lane-wise | lane-wise |

**x86 has no native form for the wider inputs.** `pmaddwd` against ones looks like
the `pmaddubsw` trick one size up, but it reads its inputs as **signed i16**, and a
u16 absolute difference reaches 65535 -- anything over 32767 is treated as negative
and the result is wrong. (The u8 case is safe only because byte-pair sums cap at
510.) Same reasoning rules it out for u32.

`psadbw` is SSE2, so the u8 u64-grouping is one instruction on every x86 tier, and
NEON gets every grouping at every input width natively via `vabdq` + `vpaddlq`. The
portable fallback at full width is a SWAR cascade over a same-width reinterpret
(`sad_cascade_*` in `register/mod.rs`), not a scalar loop; the sub-native ladder
uses `sad_scalar_*`.

One correctness note in the cascades: the u32->u64 fold **must** mask before adding
(two u32 differences sum to 33 bits), unlike the narrower folds where the cheap
add-then-mask form is provably safe.

## 11. LinAlg3Vector / LinAlg4Vector (3D/4D, lanes ARE the components)

```rust
a.dot3(b) -> E     a.cross3::<DOP>(b) -> V    a.refract(n, eta)   v.zero4()  v.one4()
v.min_element3()  v.max_element3()  v.sum_elements3()  v.prod_elements3()
V::mat3_transpose(&[V;3])   v.mat3_vec3_product::<COL_MAJOR>(&m)   V::mat3_product::<COL_MAJOR>(&l,&r)
V::mat3_det(&m)   V::mat3_inverse_inplace(&mut m) -> E   // returns determinant
// 4D:
a.dot4(b)   a.quat4_product(b)   a.quat4_vec3_product::<DOP>(v)   a.quat_to_mat3::<COL>()  a.quat_to_mat4::<COL>()
V::mat4_transpose/_product/_det/_inverse_inplace, mat4_vec4_product, mat4_vec3_product
```

Note (known issue): `mat4_inverse` only catches exactly-singular
(zero-determinant) matrices, not ill-conditioned ones.

## Masked variants: `_c` / `_m` / `_z`

Generated for every `[masked]`/`[conditional]` method. **In `_c`/`_z` the mask is
the first extra argument; `_m` takes `src` first, then the mask.** For a binary
op `op(self, rhs)`:

```rust
a.op_c(mask, rhs)        // mask ? op(a, rhs) : a          (conditional: keep self where false)
a.op_m(src, mask, rhs)   // mask ? op(a, rhs) : src        (merge: src is FIRST, then mask)
a.op_z(mask, rhs)        // mask ? op(a, rhs) : 0          (zero where false)
```

For a unary op `op(self)` (e.g. `sqrt`, `abs`, `neg`):

```rust
v.op_c(mask)             v.op_m(src, mask)             v.op_z(mask)
```

Concrete examples:

```rust
a.add_c(mask, b)         // mask ? a+b : a
a.mul_m(src, mask, b)    // mask ? a*b : src
a.sub_z(mask, b)         // mask ? a-b : 0
v.sqrt_c(mask)           v.abs_c(mask)        v.neg_z(mask)
v.sqrt_m(fallback, mask) // mask ? sqrt(v) : fallback
```

Assignment forms exist too (`add_assign_c`, etc.). On AVX-512 these map to single
masked instructions; on pre-AVX512 some `_c` forms have optimized branchless
encodings, others lower to `select`.
