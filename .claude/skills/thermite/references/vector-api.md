# Vector method reference

Methods on `Vector<R>` via the trait hierarchy. Grouped by the trait that provides
them. `V` denotes the vector type, `E = V::Element`, `M = V::Mask`. Defined in
`crates/thermite/src/vector/mod.rs` and `vector/ops.rs`.

> Masked variants: every method tagged `[masked]` automatically gets `_c`/`_m`/`_z`
> siblings -- see the last section. The **mask is always the first extra argument**.

## 1. GenericVector

```rust
// Construction
V::new([e0, e1, ...])        V::splat(e)         V::single(e)   // single: lane 0 = e, rest 0
V::EMPTY                                              // all-zero (also ZERO/ONE/... on NumericVector)
const X: V = thermite::const_new!(f32: [1.0, 0.0, 0.0]);  // const vector value (usable in const fn / assoc consts)
v.into_array() -> GenericArray<E, V::Lanes>          V::from_slice(&[E])     v.copy_to_slice(&mut [E])

// Lane access
v.extract::<I>() -> E        v.insert::<I>(e) -> V   v.broadcast::<I>() -> V     // compile-time index
v.extractv(i)    -> E        v.insertv(i, e) -> V    v.broadcastv(i) -> V        // runtime index
v.x() v.y() v.z() v.w()                                                          // GenericVector2/3/4
v.reverse()                  v.swap_bytes()

// Memory (unsafe load/store; ptr must satisfy alignment for the aligned forms)
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

// Compaction
v.compress(mask)    // stable left-pack of true lanes (AVX-512 vpcompress; portable fallback)
v.compress_z(mask)  // left-pack true lanes, zero the rest

// Two-vector element align (palignr family; any element type): the window of
// LANES lanes starting at lane OFFSET of [a, b]. OFFSET=0 -> a, OFFSET=LANES -> b.
a.align::<OFFSET>(b)   // sliding window across a load boundary; int backends use native byte aligns

// Casting
v.cast::<W>()         // numeric cast, like `as`
v.fast_cast::<W>()    // faster, may skip edge cases
v.into_bits::<W>()    // zero-cost bit reinterpret
v.saturating_cast::<W>()

// Mask helpers
v.zz(mask)   // zero lanes where mask is FALSE   (keep where true)
v.nz(mask)   // zero lanes where mask is TRUE
V::prefix_mask(n)    V::suffix_mask(n)            // first / last n lanes true

// Scalar fallback (per-lane closures; see perf doc: avoid in hot target_feature code)
v.map(|x| ...)   v.fold(init, |acc, x| ...)   v.reduce(|a, b| ...)
```

## 2. BitwiseVector / BitshiftVector

```rust
a & b   a | b   a ^ b   !a              a.bitandnot(b)   // a & !b
V::ternlog::<IMM>(a, b, c)              V::bilog::<IMM>(a, b)   // see ternlog_imm! macro
a << n  a >> n   (n: u32 or V::Unsigned)
a.shli::<I>()  a.shri::<I>()  a.shl(n)  a.shr(n)  a.shlv(unsigned)  a.shrv(unsigned)
a.bshli::<I>()  a.bshri::<I>()                              // byte shifts
a.rol(n)  a.ror(n)  a.roli::<I>()  a.rori::<I>()  a.rolv(u)  a.rorv(u)   a.reverse_bits()
```

## 3. PartialOrdVector

```rust
let m: M = a.cmp_lt(b);   // also cmp_le, cmp_gt, cmp_ge, cmp_eq, cmp_ne  -> Mask
```

(`PartialEq`/`PartialOrd` for `Vector` itself are whole-vector: `==` is "all lanes
equal". For per-lane results use `cmp_*`.)

## 4. NumericVector

```rust
a + b   a - b   a * b   a / b   a % b     a.square()
a.min(b)   a.max(b)   a.clamp(lo, hi)
v.sum_elements()  v.prod_elements()  v.min_element()  v.max_element()
v.min_max_element() -> (E, E)    v.arg_minmax() -> (usize, usize)
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
v.leading_ones()  v.leading_zeros()
// SignedIntegerVector
v.srai::<I>()  v.sra(n)  v.srav(unsigned)   a.avg_floor(b)  a.avg_ceil(b)
a.mulhrs(b)    // rounded Q(W-1) fixed-point multiply (i16: Q15, x86 PMULHRSW); rounds, not truncates
// UnsignedIntegerVector
v.is_power_of_two() -> M   a.avg(b)   v.parity()   v.ilog2p1()   v.next_power_of_two_m1()
a.abs_diff(b)              // |a - b| without overflow (saturating-sub form)
x.in_range(lo, hi) -> M    // mask of lo <= x <= hi, inclusive (branchless, one compare)

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

// FMA family. Sign conventions:
a.mul_adde(b, c)   // a*b + c   <-- PREFER the `e` (estimating) forms by default
a.mul_sube(b, c)   // a*b - c
a.nmul_adde(b, c)  // c - a*b
a.nmul_sube(b, c)  // -a*b - c
a.mul_add(b, c)    // a*b + c, ALWAYS fused (libm::fma if no hardware -- slow). Only inside HAS_TRUE_FMA gate.
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

## 9. LinAlg3Vector / LinAlg4Vector (3D/4D, lanes ARE the components)

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

Generated for every `[masked]`/`[conditional]` method. **Mask is the first extra
argument.** For a binary op `op(self, rhs)`:

```rust
a.op_c(mask, rhs)        // mask ? op(a, rhs) : a          (conditional: keep self where false)
a.op_m(src, mask, rhs)   // mask ? op(a, rhs) : src        (merge: src is FIRST, then mask)
a.op_z(mask, rhs)        // mask ? op(a, rhs) : 0          (zero where false)
```

For a unary op `op(self)` (e.g. `sqrt`, `abs`, `neg`):

```rust
v.op_c(mask)             v.op_m(src, mask)             v.op_z(mask)
```

Concrete examples (verified in the codebase):

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
